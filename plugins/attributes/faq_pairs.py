import sys, json, pathlib
from kn.utils.llm_client import chat

OUTDIR = pathlib.Path('.knowledge/indexes/attributes/faq-pairs')
OUTDIR.mkdir(parents=True, exist_ok=True)

PROMPT = (
    "Generate 5-10 FAQ Q/A pairs strictly grounded in the provided text. "
    "Format as '- Q: ...\n  A: ...'. Avoid hallucinations; answer 'unknown' if not supported."
)

for line in sys.stdin:
    job = json.loads(line)
    text = job.get('text','') or ''
    if not text.strip():
        continue
    out = chat(PROMPT + "\n\n" + text, max_tokens=600, temperature=0.2)
    (OUTDIR / f"{job['doc_id']}.json").write_text(json.dumps({
        'doc_id': job['doc_id'], 'attribute': 'faq-pairs', 'value': out.strip(), 'confidence': 0.6
    }, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({'status':'ok','doc_id':job['doc_id']}))

