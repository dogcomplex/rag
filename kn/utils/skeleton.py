import json, pathlib

SKELETON_DIR = pathlib.Path('.knowledge/indexes/attributes/doc-skeleton')

def load_skeleton_text(doc_id: str) -> str|None:
    p = SKELETON_DIR / f"{doc_id}.json"
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding='utf-8'))
        val = data.get('value') or {}
        parts = []
        if isinstance(val, dict):
            outline = val.get('outline')
            if outline:
                parts.append("Outline:\n" + outline)
            s100 = val.get('summary100') or val.get('summary')
            if s100:
                parts.append("Summary:\n" + s100)
            kp = val.get('keyphrases')
            if kp:
                parts.append("Keyphrases: " + kp)
        return "\n\n".join(parts) if parts else None
    except Exception:
        return None

