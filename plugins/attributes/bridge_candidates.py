import sys, json, pathlib, re
from collections import defaultdict

OUTDIR = pathlib.Path('.knowledge/indexes/attributes/bridge-candidates')
OUTDIR.mkdir(parents=True, exist_ok=True)

CHUNKS_DIR = pathlib.Path('.knowledge/indexes/chunks')

def naive_entities(text: str):
    return list(set(re.findall(r"[A-Z][a-zA-Z0-9_]+|\b[A-Za-z0-9_]+\.\w+|#\w+", text)))[:200]

def build_entity_domains_map(limit_files: int|None=None):
    ent_to_domains: dict[str,set[str]] = defaultdict(set)
    count = 0
    for p in CHUNKS_DIR.glob('*.json'):
        rec = json.loads(p.read_text(encoding='utf-8'))
        dom = rec.get('meta',{}).get('domain','root')
        for e in naive_entities(rec.get('text','')):
            ent_to_domains[e].add(dom)
        count += 1
        if limit_files and count >= limit_files:
            break
    return ent_to_domains

ENT_DOMAINS = build_entity_domains_map()

for line in sys.stdin:
    job = json.loads(line)
    text = job.get('text','') or ''
    if not text.strip():
        continue
    dom = (job.get('meta') or {}).get('domain','root')
    ents = naive_entities(text)
    bridges = []
    for e in ents:
        doms = ENT_DOMAINS.get(e, set())
        other = [d for d in doms if d != dom]
        if other:
            bridges.append({'entity': e, 'other_domains': sorted(other)})
    # rank by number of other domains desc
    bridges.sort(key=lambda x: (-len(x['other_domains']), x['entity'].lower()))
    (OUTDIR / f"{job['doc_id']}.json").write_text(json.dumps({
        'doc_id': job['doc_id'], 'attribute': 'bridge-candidates', 'value': bridges[:25]
    }, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({'status':'ok','doc_id':job['doc_id']}))

