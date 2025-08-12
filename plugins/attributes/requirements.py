import sys, json, pathlib
from kn.utils.llm_client import chat

OUTDIR = pathlib.Path('.knowledge/indexes/attributes/requirements')
OUTDIR.mkdir(parents=True, exist_ok=True)

PROMPT = (
    "Extract requirement-like statements (shall/should/must or implied constraints). "
    "Output 5-15 bullets, each concise and testable. If none, return an empty list."
)

for line in sys.stdin:
    job = json.loads(line)
    text = job.get('text','') or ''
    if not text.strip():
        continue
    out = chat(PROMPT + "\n\n" + text, max_tokens=350, temperature=0.2)
    (OUTDIR / f"{job['doc_id']}.json").write_text(json.dumps({
        'doc_id': job['doc_id'], 'attribute': 'requirements', 'value': out.strip(), 'confidence': 0.6
    }, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({'status':'ok','doc_id':job['doc_id']}))

