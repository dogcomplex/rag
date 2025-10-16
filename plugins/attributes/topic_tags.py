import json
import pathlib
import sys

from kn.llm_gateway.client import submit_chat_request
from kn.utils.skeleton import load_skeleton_text

OUTDIR = pathlib.Path('.knowledge/indexes/attributes/topic-tags')
OUTDIR.mkdir(parents=True, exist_ok=True)


def _safe_text(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    return text.encode('utf-8', errors='replace').decode('utf-8')


for line in sys.stdin:
    job = json.loads(line)
    text = load_skeleton_text(job.get('doc_id')) or job.get('text', '')
    if not text:
        continue
    text = _safe_text(text)
    prompt = _safe_text(
        "Extract 3-7 topical tags (comma-separated, lowercase, no spaces, use-hyphens).\n"
        "Prefer domain-relevant terms.\n\n"
        f"TEXT:\n{text}\n"
    )
    out = submit_chat_request(prompt, max_tokens=64, temperature=0.2, plugin_name='topic-tags')
    out = _safe_text(out)
    OUTDIR.joinpath(f"{job['doc_id']}.json").write_text(json.dumps({
        "doc_id": job['doc_id'],
        "attribute": "topic-tags",
        "value": out.strip(),
        "confidence": 0.7,
        "pass": "cheap"
    }, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({"status": "ok", "doc_id": job['doc_id']}, ensure_ascii=False))