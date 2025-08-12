import pathlib
from .retrieval import answer_query
def export_monofile(q: str, cfg, out: pathlib.Path, budget: int|None=None, scope=None):
    body = answer_query(q, cfg, scope=scope, topk=64)
    if cfg.get("export", {}).get("format","md") == "md":
        header = f"# Export: {q}\n\n"
    else:
        header = ""
    text = header + body
    if budget:
        text = text[: int(budget*4)]
    out.write_text(text, encoding='utf-8')