import sys
import json
import pathlib
from kn.llm_gateway.client import submit_chat_request

OUTDIR = pathlib.Path('.knowledge/indexes/attributes/summary-short')
OUTDIR.mkdir(parents=True, exist_ok=True)

for line in sys.stdin:
    job = json.loads(line)
    text = job.get('text', '')
    if not text:
        print(json.dumps({'status': 'skip', 'doc_id': job.get('doc_id')}, ensure_ascii=False))
        continue
    doc_id = job.get('doc_id')
    payload = job.get('payload') or {}
    prompt = payload.get('prompt') or "Summarize the text in ~120 words, clear and structured.\n\n" + text
    try:
        out = submit_chat_request(
            prompt,
            max_tokens=payload.get('max_tokens', job.get('max_tokens', 512)),
            temperature=payload.get('temperature', 0.2),
            overrides=payload.get('llm'),
            cache_key=f"summary-short|{doc_id}",
            plugin_name='summary-short',
            overwrite=True
        )
    except Exception as exc:
        print(json.dumps({'status': 'error', 'doc_id': doc_id, 'error': str(exc)}, ensure_ascii=False))
        continue
    rec = {
        'doc_id': doc_id,
        'attribute': 'summary-short',
        'value': out.strip(),
        'confidence': 0.75,
        'pass': payload.get('pass', 'default')
    }
    OUTDIR.joinpath(f"{doc_id}.json").write_text(json.dumps(rec, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({'status': 'ok', 'doc_id': doc_id, 'length': len(out)}, ensure_ascii=False))

