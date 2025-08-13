import sys, json, pathlib
from kn.utils.llm_client import chat
from kn.utils.skeleton import load_skeleton_text

OUTDIR = pathlib.Path('.knowledge/indexes/attributes/risk-scan')
OUTDIR.mkdir(parents=True, exist_ok=True)

PROMPT = (
    "Assess the following text for risks. Return JSON with fields:"
    " {{contradictions:boolean, speculation:boolean, outdated:boolean, pii_hint:boolean, notes:string}}."
    " Keep notes <= 40 words.\n\nTEXT:\n{body}\n"
)

for line in sys.stdin:
    job = json.loads(line)
    text = load_skeleton_text(job.get('doc_id')) or job.get('text','') or ''
    if not text.strip():
        continue
    raw = chat(PROMPT.format(body=text), max_tokens=160, temperature=0.1)
    # best-effort JSON parse; if fails, wrap as notes
    try:
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError()
    except Exception:
        data = {'contradictions': False, 'speculation': False, 'outdated': False, 'pii_hint': False, 'notes': raw.strip()[:240]}
    (OUTDIR / f"{job['doc_id']}.json").write_text(json.dumps({
        'doc_id': job['doc_id'], 'attribute': 'risk-scan', 'value': data
    }, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({'status':'ok','doc_id':job['doc_id']}))

