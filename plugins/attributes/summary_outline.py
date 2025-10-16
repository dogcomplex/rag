import sys, json, pathlib
from kn.utils.llm_client import chat
from kn.utils.skeleton import load_skeleton_text

OUTDIR = pathlib.Path('.knowledge/indexes/attributes/summary-outline')
OUTDIR.mkdir(parents=True, exist_ok=True)

PROMPT = "Produce a hierarchical outline (bulleted) of the main ideas. Keep concise.\n\n{body}\n"

for line in sys.stdin:
    job = json.loads(line)
    text = load_skeleton_text(job.get('doc_id')) or job.get('text','') or ''
    if not text.strip():
        continue
    out = chat(PROMPT.format(body=text), max_tokens=360, temperature=0.2, plugin_name='summary-outline', overwrite=True)
    (OUTDIR / f"{job['doc_id']}.json").write_text(json.dumps({
        'doc_id': job['doc_id'], 'attribute': 'summary-outline', 'value': out.strip(), 'confidence': 0.7
    }, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({'status':'ok','doc_id':job['doc_id']}))

