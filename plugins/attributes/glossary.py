import sys, json, pathlib
from kn.utils.llm_client import chat

OUTDIR = pathlib.Path('.knowledge/indexes/attributes/glossary')
OUTDIR.mkdir(parents=True, exist_ok=True)

PROMPT = (
    "Extract 5-12 project-specific terms from the text with brief definitions. "
    "Return a bullet list 'term: definition' focusing on unique or overloaded terms."
)

for line in sys.stdin:
    job = json.loads(line)
    text = job.get('text','') or ''
    if not text.strip():
        continue
    out = chat(PROMPT + "\n\n" + text, max_tokens=300, temperature=0.2)
    (OUTDIR / f"{job['doc_id']}.json").write_text(json.dumps({
        'doc_id': job['doc_id'], 'attribute': 'glossary', 'value': out.strip(), 'confidence': 0.6
    }, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({'status':'ok','doc_id':job['doc_id']}))

