import sys, json, pathlib

from kn.llm_gateway.errors import RequestRejectedError
from kn.utils.llm_client import chat
from kn.utils.skeleton import load_skeleton_text


OUTDIR = pathlib.Path('.knowledge/indexes/attributes/summary-medium')
OUTDIR.mkdir(parents=True, exist_ok=True)

PROMPT = "Summarize the text in ~150 words, clear, objective.\n\n{body}\n"
_ATTEMPT_LIMITS = [10000, 6000, 3500]


def _safe_text(text: str) -> str:
    if not isinstance(text, str):
        text = str(text or '')
    return text.encode('utf-8', errors='replace').decode('utf-8')


def _trim_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    snippet = text[:limit]
    cut = snippet.rfind('\n\n')
    if cut >= int(limit * 0.6):
        snippet = snippet[:cut]
    return snippet + '\n\n[Truncated for length]'


def _write_attr(doc_id: str, value: str, confidence: float = 0.7, error: str | None = None):
    payload = {
        'doc_id': doc_id,
        'attribute': 'summary-medium',
        'value': value.strip(),
        'confidence': confidence,
    }
    if error:
        payload['error'] = error
    OUTDIR.joinpath(f"{doc_id}.json").write_text(json.dumps(payload, ensure_ascii=False), encoding='utf-8')


def _fallback(reason: str) -> str:
    return f"[summary unavailable: {reason}]"


for line in sys.stdin:
    job = json.loads(line)
    doc_id = job.get('doc_id')
    text = load_skeleton_text(doc_id) or job.get('text', '') or ''
    if not text.strip():
        print(json.dumps({'status': 'skip', 'doc_id': doc_id}, ensure_ascii=False))
        continue

    text = _safe_text(text)
    success = False
    last_err = None

    for limit in _ATTEMPT_LIMITS:
        trimmed = _trim_text(text, limit)
        prompt = PROMPT.format(body=trimmed)
        try:
            out = chat(prompt, max_tokens=380, temperature=0.2, plugin_name='summary-medium')
            out = _safe_text(out)
            _write_attr(doc_id, out)
            print(json.dumps({'status': 'ok', 'doc_id': doc_id, 'length': len(out)}, ensure_ascii=False))
            success = True
            break
        except RequestRejectedError as exc:
            last_err = str(exc)
            if 'Context too large' in last_err:
                continue
            _write_attr(doc_id, _fallback('LLM request error'), error=last_err)
            print(json.dumps({'status': 'error', 'doc_id': doc_id, 'error': last_err}, ensure_ascii=False))
            success = True
            break
        except Exception as exc:
            last_err = str(exc)
            _write_attr(doc_id, _fallback('unexpected error'), error=last_err)
            print(json.dumps({'status': 'error', 'doc_id': doc_id, 'error': last_err}, ensure_ascii=False))
            success = True
            break

    if not success:
        reason = last_err or 'context too large'
        _write_attr(doc_id, _fallback('context too large'), error=reason)
        print(json.dumps({'status': 'error', 'doc_id': doc_id, 'error': reason}, ensure_ascii=False))

