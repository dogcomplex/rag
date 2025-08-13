import argparse, pathlib, time
from kn.config import load_configs
from kn.jobs_sqlite import ensure_db, enqueue

ATTR_ROOT = pathlib.Path('.knowledge/indexes/attributes')
CHUNKS_DIR = pathlib.Path('.knowledge/indexes/chunks')

def already_has(plugin: str, doc_id: str) -> bool:
    if plugin == 'summaries':
        pdir = ATTR_ROOT / 'summaries'
        if not pdir.exists():
            return False
        return any(pp.name.startswith(f"{doc_id}_") for pp in pdir.glob(f"{doc_id}_*.json"))
    p = ATTR_ROOT / plugin / f"{doc_id}.json"
    return p.exists()

def iter_doc_ids():
    seen = set()
    for p in CHUNKS_DIR.glob('*.json'):
        doc_id = p.stem.split('-')[0]
        if doc_id not in seen:
            seen.add(doc_id)
            yield doc_id

def latest_mtime_for_doc(doc_id: str) -> float:
    mt = 0.0
    for p in CHUNKS_DIR.glob(f"{doc_id}-*.json"):
        mt = max(mt, p.stat().st_mtime)
    return mt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--plugins', required=False, default='glossary,requirements,todo-items,faq-pairs', help='comma-separated plugin names')
    ap.add_argument('--limit', type=int, default=0, help='max jobs to enqueue in total (0 = unlimited)')
    ap.add_argument('--only-missing', action='store_true', help='enqueue only if attribute output file missing')
    ap.add_argument('--changed-since-min', type=int, default=0, help='only enqueue docs whose chunks changed in last N minutes (0 = ignore)')
    ap.add_argument('--summaries-modes', default='short,medium', help='for plugin "summaries", comma list of modes (short,medium,long,outline)')
    ap.add_argument('--payload-json', default=None, help='optional JSON to attach as payload to each job (e.g., {"llm":{"model":"qwen2.5-7b-instruct","timeout":60}})')
    ap.add_argument('--map-reduce', action='store_true', help='enqueue chunk map (chunk-summary) then doc reduce (doc-reduce)')
    args = ap.parse_args()

    cfg = load_configs()
    ensure_db(cfg)
    plugins = [p.strip() for p in args.plugins.split(',') if p.strip()]
    sum_modes = [m.strip() for m in args.summaries_modes.split(',') if m.strip()] if 'summaries' in plugins else []
    try:
        base_payload = json.loads(args.payload_json) if args.payload_json else {}
    except Exception:
        base_payload = {}

    cutoff = None
    if args.changed_since_min and args.changed_since_min > 0:
        cutoff = time.time() - args.changed_since_min * 60

    count = 0
    for doc_id in iter_doc_ids():
        if cutoff is not None and latest_mtime_for_doc(doc_id) < cutoff:
            continue
        for plugin in plugins:
            if args.only_missing and already_has(plugin.replace('-', '_'), doc_id):
                continue
            if plugin == 'summaries' and sum_modes:
                for mode in sum_modes:
                    pl = dict(base_payload); pl['mode'] = mode
                    enqueue(cfg, plugin, doc_id, payload=pl)
                    count += 1
            else:
                enqueue(cfg, plugin, doc_id, payload=base_payload)
                count += 1
        if args.map_reduce:
            # enqueue chunk map jobs for all chunks of this doc, then doc reduce
            from pathlib import Path
            cdir = Path('.knowledge/indexes/chunks')
            chunks = sorted(cdir.glob(f"{doc_id}-*.json"))
            for cp in chunks:
                chk_id = cp.stem
                pl = dict(base_payload); pl['chunk_id'] = chk_id
                enqueue(cfg, 'chunk-summary', doc_id, payload=pl)
                count += 1
            enqueue(cfg, 'doc-reduce', doc_id, payload=base_payload)
            count += 1
            if args.limit and count >= args.limit:
                print(f"[plan] enqueued {count} jobs (limit reached)")
                return
    print(f"[plan] enqueued {count} jobs for plugins: {', '.join(plugins)}")

if __name__ == '__main__':
    main()

