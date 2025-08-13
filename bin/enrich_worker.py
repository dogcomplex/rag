import argparse, time
from kn.config import load_configs
from kn.jobs_sqlite import ensure_db, dequeue_batch, ack_job, iter_docs_for_jobs, fail_and_requeue_job

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
            for j in items:
                fail_and_requeue_job(cfg, j["id"], error_message="plugin not found", back_to_pending=False)
            continue
        inp_lines = []
        id_order = []
        for j in items:
            doc = docs.get(j["doc_id"])
            if doc:
                payload = j.get("payload") or {}
                merged = dict(doc)
                merged["payload"] = payload
                inp_lines.append(json.dumps(merged, ensure_ascii=False))
                id_order.append(j["id"])
        if not inp_lines:
            for j in items:
                fail_and_requeue_job(cfg, j["id"], error_message="no input doc", back_to_pending=False)
            continue
        try:
            proc = subprocess.Popen([sys.executable, str(pypath)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
            out, _ = proc.communicate("\n".join(inp_lines), timeout=300)
            print(f"[enrich] {plugin}: {len(items)} docs processed")
            for j in items:
                ack_job(cfg, j["id"])
        except subprocess.TimeoutExpired as e:
            print(f"[enrich] {plugin}: timeout, requeueing batch")
            for j in items:
                fail_and_requeue_job(cfg, j["id"], error_message="timeout", back_to_pending=True)
        except Exception as e:
            msg = str(e)[:500]
            print(f"[enrich] {plugin}: error {msg}")
            for j in items:
                fail_and_requeue_job(cfg, j["id"], error_message=msg, back_to_pending=True)
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