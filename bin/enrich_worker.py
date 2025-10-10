import argparse, time, json
from kn.config import load_configs
from kn.jobs_sqlite import ensure_db, dequeue_batch, ack_job, iter_docs_for_jobs, fail_and_requeue_job, list_pending_plugins, try_acquire, release, set_limit

def run_once(plugins, cfg, batch_size=16):
    ensure_db(cfg)
    # If no jobs for requested plugins, peek pending list and suggest available
    jobs = dequeue_batch(cfg, wanted_plugins=plugins, limit=batch_size)
    if not jobs:
        # try to broaden to any pending plugins if requested plugins empty
        pend = list_pending_plugins(cfg)
        avail = [p for p in pend if p in plugins]
        if not avail:
            return 0
        jobs = dequeue_batch(cfg, wanted_plugins=avail, limit=batch_size)
        if not jobs:
            return 0
    docs = iter_docs_for_jobs(jobs)
    def _load_chunk_by_id(chunk_id: str):
        import json as _json, pathlib as _pathlib
        p = _pathlib.Path('.knowledge/indexes/chunks')/f"{chunk_id}.json"
        if p.exists():
            try:
                return _json.loads(p.read_text(encoding='utf-8'))
            except Exception:
                return None
        return None
    import subprocess, pathlib, sys
    by_plugin = {}
    for j in jobs:
        by_plugin.setdefault(j["plugin"], []).append(j)
    for plugin, items in by_plugin.items():
        # global concurrency cap
        if not try_acquire(cfg, 'llm_concurrency'):
            # can't acquire slot, skip this plugin batch this round
            continue
        fs_name = plugin.replace('-', '_')
        pypath = pathlib.Path(f"plugins/attributes/{fs_name}.py")
        if not pypath.exists():
            print(f"[enrich] plugin not found: {plugin}")
            for j in items:
                fail_and_requeue_job(cfg, j["id"], error_message="plugin not found", back_to_pending=False)
            continue
        inp_lines = []
        doc_ids = []
        for j in items:
            payload = j.get("payload") or {}
            doc = None
            if payload.get('chunk_id'):
                doc = _load_chunk_by_id(payload['chunk_id'])
            if doc is None:
                doc = docs.get(j["doc_id"])  # fallback
            if doc:
                merged = dict(doc)
                merged["payload"] = payload
                inp_lines.append(json.dumps(merged, ensure_ascii=False))
                doc_ids.append(j["doc_id"])
        if not inp_lines:
            for j in items:
                fail_and_requeue_job(cfg, j["id"], error_message="no input doc", back_to_pending=False)
            continue
        try:
            # per-plugin process timeout (seconds)
            pcfg = (cfg.get('plugins') or {}).get(plugin) or {}
            proc_timeout = pcfg.get('process_timeout') if isinstance(pcfg, dict) else None
            if not isinstance(proc_timeout, (int, float)):
                proc_timeout = 600 if plugin in ('multi-basic','doc-skeleton') else 300
            summary_docs = ",".join(doc_ids[:4])
            if len(doc_ids) > 4:
                summary_docs += ",…"
            if summary_docs:
                print(f"[worker-current] plugin={plugin} docs={summary_docs}")
            proc = subprocess.Popen([sys.executable, str(pypath)], stdin=subprocess.PIPE,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            out, _ = proc.communicate("\n".join(inp_lines), timeout=proc_timeout)
            if out:
                for line in out.splitlines():
                    print(f"[plugin:{plugin}] {line}")
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
        finally:
            release(cfg, 'llm_concurrency')
            print(f"[worker-current-clear] plugin={plugin}")
    return len(jobs)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--plugins", required=True, help="comma-separated plugin names or '*' for any pending")
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--any-pending", action="store_true", dest="any_pending")
    ap.add_argument("--max-inflight", type=int, default=2, help="global concurrent LLM calls across workers")
    args = ap.parse_args()
    cfg = load_configs()
    plugins = [p.strip() for p in args.plugins.split(",") if p.strip()]
    any_pending = args.any_pending or (len(plugins)==1 and plugins[0] in ("*","any"))
    # set concurrency limit at start
    set_limit(cfg, 'llm_concurrency', max(1, int(args.max_inflight)))
    while True:
        use_plugins = plugins
        if any_pending:
            dyn = list_pending_plugins(cfg)
            use_plugins = dyn if dyn else []
        n = run_once(use_plugins, cfg, batch_size=args.batch)
        if not args.watch:
            break
        if n == 0:
            time.sleep(2)