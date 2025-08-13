import sys, json, pathlib
from kn.utils.llm_client import chat

OUTDIR = pathlib.Path('.knowledge/indexes/attributes/doc-skeleton')
OUTDIR.mkdir(parents=True, exist_ok=True)

PROMPT = (
    "Produce a compact JSON skeleton for the document with keys: outline (bulleted), summary100 (~100 words), keyphrases (comma-separated)."
    " Return only JSON.\n\nTEXT:\n{body}\n"
)

for line in sys.stdin:
    job = json.loads(line)
    text = job.get('text','') or ''
    if not text.strip():
        continue
    doc_id = job.get('doc_id')
    overrides = (job.get('payload') or {}).get('llm') or {}
    # Qwen 32B has ~12k context; keep outputs compact
    raw = chat(PROMPT.format(body=text), max_tokens=240, temperature=0.2, overrides=overrides, plugin_name='doc-skeleton')
    try:
        data = json.loads(raw)
    except Exception:
        data = {'summary100': raw.strip()}
    (OUTDIR/f"{doc_id}.json").write_text(json.dumps({'doc_id':doc_id,'attribute':'doc-skeleton','value':data}, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({'status':'ok','doc_id':doc_id}))

