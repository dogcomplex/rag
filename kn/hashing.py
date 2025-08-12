import hashlib

def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()

def short_hash(h: str, n=8) -> str:
    return h[:n]

def chunk_hash(doc_id: str, chunk_index: int) -> str:
    return f"{doc_id}-{chunk_index:04d}"