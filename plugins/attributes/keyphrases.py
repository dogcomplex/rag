import sys, json, pathlib
from kn.utils.llm_client import chat

OUTDIR = pathlib.Path('.knowledge/indexes/attributes/keyphrases')
OUTDIR.mkdir(parents=True, exist_ok=True)

PROMPT = (
    "Extract 8-20 keyphrases capturing entities, terms, and concepts.\n"
    "- Return a comma-separated list\n"
    "- Lowercase, hyphenate multi-word terms, no spaces around commas\n"
    "- Prefer domain-specific terminology\n\nTEXT:\n{body}\n"
)

for line in sys.stdin:
    job = json.loads(line)
    text = job.get('text','') or ''
    if not text.strip():
        continue
    out = chat(PROMPT.format(body=text), max_tokens=120, temperature=0.2)
    (OUTDIR / f"{job['doc_id']}.json").write_text(json.dumps({
        'doc_id': job['doc_id'], 'attribute': 'keyphrases', 'value': out.strip(), 'confidence': 0.7
    }, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({'status':'ok','doc_id':job['doc_id']}))

