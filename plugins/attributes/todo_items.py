import sys, json, pathlib, re

OUTDIR = pathlib.Path('.knowledge/indexes/attributes/todo-items')
OUTDIR.mkdir(parents=True, exist_ok=True)

PAT = re.compile(r"(?im)^(?:#|//|\*|[-\d.]*)?\s*(TODO|FIXME|NOTE)[:\-\s]+(.{4,120})$")

for line in sys.stdin:
    job = json.loads(line)
    text = job.get('text','') or ''
    items = []
    for m in PAT.finditer(text):
        items.append({'kind': m.group(1).upper(), 'text': m.group(2).strip()})
    (OUTDIR / f"{job['doc_id']}.json").write_text(json.dumps({
        'doc_id': job['doc_id'], 'attribute': 'todo-items', 'value': items, 'confidence': 0.6
    }, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({'status':'ok','doc_id':job['doc_id']}))

