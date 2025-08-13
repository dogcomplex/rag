import sys, json, pathlib
from kn.utils.llm_client import chat

OUTDIR = pathlib.Path('.knowledge/indexes/attributes/summary-medium')
OUTDIR.mkdir(parents=True, exist_ok=True)

PROMPT = "Summarize the text in ~150 words, clear, objective.\n\n{body}\n"

for line in sys.stdin:
    job = json.loads(line)
    text = job.get('text','') or ''
    if not text.strip():
        continue
    out = chat(PROMPT.format(body=text), max_tokens=380, temperature=0.2)
    (OUTDIR / f"{job['doc_id']}.json").write_text(json.dumps({
        'doc_id': job['doc_id'], 'attribute': 'summary-medium', 'value': out.strip(), 'confidence': 0.7
    }, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({'status':'ok','doc_id':job['doc_id']}))

