import json
import pathlib
import sys

from kn.llm_gateway.errors import RequestRejectedError
from kn.utils.llm_client import chat

OUTDIR = pathlib.Path('.knowledge/indexes/attributes/summary-20w')
OUTDIR.mkdir(parents=True, exist_ok=True)

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


def _write_attr(doc_id: str, value: str, confidence: float = 0.8, notes: dict | None = None):
    payload = {
        "doc_id": doc_id,
        "attribute": "summary-20w",
        "value": value.strip(),
        "confidence": confidence,
        "pass": "cheap",
    }
    if notes:
        payload.update(notes)
    OUTDIR.joinpath(f"{doc_id}.json").write_text(json.dumps(payload, ensure_ascii=False), encoding='utf-8')


def _fallback_value(reason: str) -> str:
    return f"[summary unavailable: {reason}]"


for line in sys.stdin:
    job = json.loads(line)
    doc_id = job.get('doc_id')
    text = job.get('text', '')
    if not text:
        print(json.dumps({"status": "skip", "doc_id": doc_id}, ensure_ascii=False))
        continue

    text = _safe_text(text)
    success = False
    last_error = None

    for limit in _ATTEMPT_LIMITS:
        trimmed = _trim_text(text, limit)
        prompt = _safe_text(f"Summarize the following in ~20 words, terse and factual. If truncated, focus on the excerpt provided.\n\n{trimmed}")
        try:
            out = chat(prompt, max_tokens=100, temperature=0.1)
            out = _safe_text(out)
            _write_attr(doc_id, out)
            print(json.dumps({"status": "ok", "doc_id": doc_id, "length": len(out)}, ensure_ascii=False))
            success = True
            break
        except RequestRejectedError as exc:
            last_error = str(exc)
            if 'Context too large' in last_error:
                continue
            _write_attr(doc_id, _fallback_value('LLM request error'), notes={"error": last_error})
            print(json.dumps({"status": "error", "doc_id": doc_id, "error": last_error}, ensure_ascii=False))
            success = True
            break
        except Exception as exc:
            last_error = str(exc)
            _write_attr(doc_id, _fallback_value('unexpected error'), notes={"error": last_error})
            print(json.dumps({"status": "error", "doc_id": doc_id, "error": last_error}, ensure_ascii=False))
            success = True
            break

    if not success:
        msg = last_error or 'context too large even after truncation'
        _write_attr(doc_id, _fallback_value('context too large'), notes={"error": msg})
        print(json.dumps({"status": "error", "doc_id": doc_id, "error": msg}, ensure_ascii=False))