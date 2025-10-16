import sys, json, pathlib
from kn.utils.llm_client import chat

OUTDIR = pathlib.Path('.knowledge/indexes/attributes/recent-summary')
OUTDIR.mkdir(parents=True, exist_ok=True)

PROMPT = (
    "Summarize the most salient updates and current state. Be concise (<=120 words)."
    " If there are TODOs or open questions, list them as short bullets.\n\nTEXT:\n{body}\n"
)

for line in sys.stdin:
    job = json.loads(line)
    text = job.get('text','') or ''
    if not text.strip():
        continue
    out = chat(PROMPT.format(body=text), max_tokens=220, temperature=0.2, plugin_name='recent-summary', overwrite=True)
    (OUTDIR / f"{job['doc_id']}.json").write_text(json.dumps({
        'doc_id': job['doc_id'], 'attribute': 'recent-summary', 'value': out.strip(), 'confidence': 0.7
    }, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({'status':'ok','doc_id':job['doc_id']}))

