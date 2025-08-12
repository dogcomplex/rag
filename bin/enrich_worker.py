import argparse, time
from kn.config import load_configs
from kn.jobs_sqlite import ensure_db, dequeue_batch, ack_job, iter_docs_for_jobs

def run_once(plugins, cfg, batch_size=16):
    ensure_db(cfg)
    jobs = dequeue_batch(cfg, wanted_plugins=plugins, limit=batch_size)
    if not jobs:
        return 0
    docs = iter_docs_for_jobs(jobs)
    import subprocess, json, pathlib, sys
    by_plugin = {}
    for j in jobs:
        by_plugin.setdefault(j["plugin"], []).append(j)
    for plugin, items in by_plugin.items():
        fs_name = plugin.replace('-', '_')
        pypath = pathlib.Path(f"plugins/attributes/{fs_name}.py")
        if not pypath.exists():
            print(f"[enrich] plugin not found: {plugin}")
            for j in items: ack_job(cfg, j["id"])
            continue
        inp_lines = []
        for j in items:
            doc = docs.get(j["doc_id"])
            if doc:
                inp_lines.append(json.dumps(doc, ensure_ascii=False))
        if not inp_lines:
            for j in items: ack_job(cfg, j["id"])
            continue
        proc = subprocess.Popen([sys.executable, str(pypath)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        out, _ = proc.communicate("\n".join(inp_lines))
        print(f"[enrich] {plugin}: {len(items)} docs processed")
        for j in items: ack_job(cfg, j["id"])
    return len(jobs)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--plugins", required=True, help="comma-separated plugin names")
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()
    cfg = load_configs()
    plugins = [p.strip() for p in args.plugins.split(",") if p.strip()]
    while True:
        n = run_once(plugins, cfg, batch_size=args.batch)
        if not args.watch:
            break
        if n == 0:
            time.sleep(2)