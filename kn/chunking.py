import pathlib, json, re
from .utils.io import ensure_dirs
from .hashing import chunk_hash

CHUNKS_DIR = pathlib.Path('.knowledge/indexes/chunks')

def _split_paragraphs(text: str):
    # split on blank lines; keep headings by not stripping too aggressively
    paras = re.split(r"\n\s*\n+", text)
    # normalize paragraphs
    return [p.strip() for p in paras if p and p.strip()]

def chunk_text(text: str, max_chars=3000, overlap_ratio=0.1):
    # paragraph-first packing with small overlap (last paragraph of previous chunk)
    paras = _split_paragraphs(text)
    chunks = []
    current = []
    current_len = 0
    for para in paras:
        p_len = len(para) + 2  # account for join newlines
        if current_len + p_len <= max_chars or not current:
            current.append(para)
            current_len += p_len
        else:
            chunks.append("\n\n".join(current))
            # overlap: keep last paragraph from previous chunk
            overlap_n = 1 if current else 0
            tail = current[-overlap_n:] if overlap_n else []
            current = tail + [para]
            current_len = sum(len(t) + 2 for t in current)
    if current:
        chunks.append("\n\n".join(current))
    return chunks

def detect_domain(rel_path: pathlib.Path):
    parts = rel_path.parts
    return parts[0] if parts else "root"

def chunk_repo(docs, cfg, repo_root: pathlib.Path):
    ensure_dirs(cfg)
    policy = cfg.get("chunking", {}).get("policies", {}).get("default", {"max_chars":3000,"overlap":300})
    chunks = []
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    for d in docs:
        path = pathlib.Path(d["path"]).resolve()
        try: rel = path.relative_to(repo_root)
        except Exception: rel = pathlib.Path(path.name)
        domain = detect_domain(rel)
        parts = chunk_text(d.get("text",""), max_chars=policy.get("max_chars",3000), overlap_ratio=(policy.get("overlap",300)/max(policy.get("max_chars",3000),1)))
        total = len(parts)
        for idx, t in enumerate(parts):
            chunk_id = chunk_hash(d["doc_id"], idx)
            rec = {
                "doc_id": d["doc_id"],
                "chunk_id": chunk_id,
                "text": t,
                "meta": {
                    "path": str(path),
                    "rel": str(rel),
                    "domain": domain,
                    "mimetype": "text/plain",
                    "seq": idx,
                    "total": total,
                    "prev": chunk_hash(d["doc_id"], idx-1) if idx>0 else None,
                    "next": chunk_hash(d["doc_id"], idx+1) if idx<total-1 else None
                }
            }
            chunks.append(rec)
            (CHUNKS_DIR / f"{chunk_id}.json").write_text(json.dumps(rec, ensure_ascii=False), encoding='utf-8')
    return chunks