import pathlib, json
from .utils.io import ensure_dirs
from .hashing import chunk_hash

CHUNKS_DIR = pathlib.Path('.knowledge/indexes/chunks')

def chunk_text(text: str, max_chars=4000, overlap=400):
    out = []
    i = 0
    L = len(text)
    step = max_chars - overlap if max_chars > overlap else max_chars
    while i < L:
        out.append(text[i:i+max_chars])
        i += step
    return out

def detect_domain(rel_path: pathlib.Path):
    parts = rel_path.parts
    return parts[0] if parts else "root"

def chunk_repo(docs, cfg, repo_root: pathlib.Path):
    ensure_dirs()
    policy = cfg.get("chunking", {}).get("policies", {}).get("default", {"max_chars":4000,"overlap":400})
    chunks = []
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    for d in docs:
        path = pathlib.Path(d["path"]).resolve()
        try: rel = path.relative_to(repo_root)
        except Exception: rel = pathlib.Path(path.name)
        domain = detect_domain(rel)
        parts = chunk_text(d.get("text",""), **policy)
        for idx, t in enumerate(parts):
            chunk_id = chunk_hash(d["doc_id"], idx)
            rec = {
                "doc_id": d["doc_id"],
                "chunk_id": chunk_id,
                "text": t,
                "meta": {"path": str(path), "rel": str(rel), "domain": domain, "mimetype": "text/plain"}
            }
            chunks.append(rec)
            (CHUNKS_DIR / f"{chunk_id}.json").write_text(json.dumps(rec, ensure_ascii=False), encoding='utf-8')
    return chunks