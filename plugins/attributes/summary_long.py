import sys, json, pathlib
from kn.utils.llm_client import chat
from kn.utils.skeleton import load_skeleton_text

OUTDIR = pathlib.Path('.knowledge/indexes/attributes/summary-long')
OUTDIR.mkdir(parents=True, exist_ok=True)

PROMPT = "Summarize the text in ~400 words, comprehensive but concise.\n\n{body}\n"

for line in sys.stdin:
    job = json.loads(line)
    text = load_skeleton_text(job.get('doc_id')) or job.get('text','') or ''
    if not text.strip():
        continue
    out = chat(PROMPT.format(body=text), max_tokens=900, temperature=0.2)
    (OUTDIR / f"{job['doc_id']}.json").write_text(json.dumps({
        'doc_id': job['doc_id'], 'attribute': 'summary-long', 'value': out.strip(), 'confidence': 0.7
    }, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({'status':'ok','doc_id':job['doc_id']}))

