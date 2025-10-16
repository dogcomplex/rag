import sys, json, pathlib
from kn.utils.llm_client import chat

CHUNK_SUM_DIR = pathlib.Path('.knowledge/indexes/attributes/chunk-summary')
OUTDIR = pathlib.Path('.knowledge/indexes/attributes/doc-reduce')
OUTDIR.mkdir(parents=True, exist_ok=True)

PROMPT = (
    "You are given ordered chunk summaries for a single document.\n"
    "Produce a compact JSON with keys: outline (bulleted), summary (~150 words), tags (3-7 comma-separated).\n"
    "Return only JSON.\n\nSUMMARIES:\n{body}\n"
)

def load_chunk_summaries(doc_id: str):
    items = []
    for p in sorted(CHUNK_SUM_DIR.glob('*.json')):
        try:
            rec = json.loads(p.read_text(encoding='utf-8'))
            if rec.get('doc_id') == doc_id:
                items.append(rec)
        except Exception:
            continue
    # try to sort by chunk_id numeric tail if present in metadata; fallback original order
    def _key(r):
        cid = r.get('chunk_id','')
        try:
            return int(cid.split('-')[-1], 16)
        except Exception:
            return 0
    items.sort(key=_key)
    return items

for line in sys.stdin:
    job = json.loads(line)
    doc_id = job.get('doc_id')
    if not doc_id:
        continue
    chunks = load_chunk_summaries(doc_id)
    if not chunks:
        continue
    ordered = '\n- '.join([''] + [c.get('value','') for c in chunks])
    raw = chat(PROMPT.format(body=ordered), max_tokens=360, temperature=0.2, plugin_name='doc-reduce', overwrite=True)
    try:
        data = json.loads(raw)
    except Exception:
        data = {'summary': raw.strip()}
    (OUTDIR/f"{doc_id}.json").write_text(json.dumps({'doc_id':doc_id,'attribute':'doc-reduce','value':data}, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({'status':'ok','doc_id':doc_id}))

