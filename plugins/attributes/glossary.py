import json
import pathlib
import sys

from kn.llm_gateway.errors import RequestRejectedError
from kn.utils.llm_client import chat

OUTDIR = pathlib.Path('.knowledge/indexes/attributes/glossary')
OUTDIR.mkdir(parents=True, exist_ok=True)

PROMPT = (
    "Extract 5-12 project-specific terms from the text with brief definitions. "
    "Return a bullet list 'term: definition' focusing on unique or overloaded terms."
)

_ATTEMPT_LIMITS = [6000, 3500, 2000]


def _safe_text(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    return text.encode('utf-8', errors='replace').decode('utf-8')


def _trim_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    snippet = text[:limit]
    cut = snippet.rfind('\n\n')
    if cut >= int(limit * 0.6):
        snippet = snippet[:cut]
    return snippet + '\n\n[Truncated for length]'


def _write_attr(doc_id: str, value: str, confidence: float = 0.6, error: str | None = None):
    payload = {
        'doc_id': doc_id,
        'attribute': 'glossary',
        'value': value.strip(),
        'confidence': confidence,
    }
    if error:
        payload['error'] = error
    OUTDIR.joinpath(f"{doc_id}.json").write_text(json.dumps(payload, ensure_ascii=False), encoding='utf-8')


for line in sys.stdin:
    job = json.loads(line)
    doc_id = job.get('doc_id')
    text = job.get('text', '') or ''
    if not text.strip():
        print(json.dumps({'status': 'skip', 'doc_id': doc_id}))
        continue

    text = _safe_text(text)
    success = False
    last_error = None

    for limit in _ATTEMPT_LIMITS:
        trimmed = _trim_text(text, limit)
        try:
            out = chat(PROMPT + "\n\n" + trimmed, max_tokens=300, temperature=0.2)
            out = _safe_text(out)
            _write_attr(doc_id, out)
            print(json.dumps({'status': 'ok', 'doc_id': doc_id}))
            success = True
            break
        except RequestRejectedError as exc:
            last_error = str(exc)
            if 'Context too large' in last_error or exc.args and exc.args[0].startswith('404'):
                continue
            _write_attr(doc_id, '[glossary unavailable: gateway error]', error=last_error)
            print(json.dumps({'status': 'error', 'doc_id': doc_id, 'error': last_error}))
            success = True
            break
        except Exception as exc:
            last_error = str(exc)
            _write_attr(doc_id, '[glossary unavailable: unexpected error]', error=last_error)
            print(json.dumps({'status': 'error', 'doc_id': doc_id, 'error': last_error}))
            success = True
            break

    if not success:
        msg = last_error or 'context too large or endpoint missing'
        _write_attr(doc_id, '[glossary unavailable: context too large]', error=msg)
        print(json.dumps({'status': 'error', 'doc_id': doc_id, 'error': msg}))

