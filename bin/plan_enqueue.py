import argparse, pathlib
from kn.config import load_configs
from kn.jobs_sqlite import ensure_db, enqueue

ATTR_ROOT = pathlib.Path('.knowledge/indexes/attributes')
CHUNKS_DIR = pathlib.Path('.knowledge/indexes/chunks')

def already_has(plugin: str, doc_id: str) -> bool:
    p = ATTR_ROOT / plugin / f"{doc_id}.json"
    return p.exists()

def iter_doc_ids():
    seen = set()
    for p in CHUNKS_DIR.glob('*.json'):
        doc_id = p.stem.split('-')[0]
        if doc_id not in seen:
            seen.add(doc_id)
            yield doc_id

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--plugins', required=False, default='glossary,requirements,todo-items,faq-pairs', help='comma-separated plugin names')
    ap.add_argument('--limit', type=int, default=0, help='max docs to enqueue per plugin (0 = all)')
    ap.add_argument('--only-missing', action='store_true', help='enqueue only if attribute output file missing')
    args = ap.parse_args()

    cfg = load_configs()
    ensure_db(cfg)
    plugins = [p.strip() for p in args.plugins.split(',') if p.strip()]
    count = 0
    for doc_id in iter_doc_ids():
        for plugin in plugins:
            if args.only_missing and already_has(plugin.replace('-', '_'), doc_id):
                continue
            enqueue(cfg, plugin, doc_id, payload={})
            count += 1
        if args.limit and count >= args.limit:
            break
    print(f"[plan] enqueued {count} jobs for plugins: {', '.join(plugins)}")

if __name__ == '__main__':
    main()

