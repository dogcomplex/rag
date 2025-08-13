import sys, json, pathlib
from kn.utils.llm_client import chat

OUTDIR = pathlib.Path('.knowledge/indexes/attributes/chunk-summary')
OUTDIR.mkdir(parents=True, exist_ok=True)

PROMPT = "Summarize this chunk in 2-3 sentences, terse and factual.\n\n{body}\n"

for line in sys.stdin:
    job = json.loads(line)
    text = job.get('text','') or ''
    if not text.strip():
        continue
    out = chat(PROMPT.format(body=text), max_tokens=120, temperature=0.2, plugin_name='chunk-summary')
    (OUTDIR / f"{job['chunk_id']}.json").write_text(json.dumps({
        'doc_id': job['doc_id'], 'chunk_id': job.get('chunk_id'), 'attribute': 'chunk-summary', 'value': out.strip(), 'confidence': 0.7
    }, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({'status':'ok','chunk_id': job.get('chunk_id')}))

