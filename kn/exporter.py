import pathlib
from .retrieval import retrieve_context
import json, pathlib as _p

def _approx_trim_to_tokens(text: str, max_tokens: int) -> str:
    if not max_tokens:
        return text
    # ~4 chars/token heuristic
    max_chars = max(0, int(max_tokens * 4))
    return text[:max_chars]

def _pii_flagged(doc_id: str) -> bool:
    p = _p.Path('.knowledge/indexes/attributes/pii-scan') / f"{doc_id}.json"
    if not p.exists():
        return False
    try:
        data = json.loads(p.read_text(encoding='utf-8'))
        return (data.get('risk') or 0) >= 0.8
    except Exception:
        return False

def export_monofile(q: str, cfg, out: pathlib.Path, budget: int|None=None, scope=None, include_meta: bool=True):
    pref, chunks = retrieve_context(q, cfg, scope=scope, topk=64)
    is_md = cfg.get("export", {}).get("format","md") == "md"
    if cfg.get('export', {}).get('exclude_pii', False):
        chunks = [c for c in chunks if not _pii_flagged(c.get('doc_id'))]
    parts = []
    if is_md:
        parts.append(f"# Export: {q}\n")
    if pref:
        parts.append("\n".join(pref) + ("\n\n" if is_md else "\n\n"))
    for c in chunks:
        meta = c.get('meta', {})
        if include_meta and is_md:
            parts.append(f"## {meta.get('rel', meta.get('path',''))} — {c['chunk_id']}\n")
        parts.append(c.get('text',''))
        parts.append("\n\n")
    body = "".join(parts)
    body = _approx_trim_to_tokens(body, budget or cfg.get('export',{}).get('default_budget_tokens'))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(body, encoding='utf-8')