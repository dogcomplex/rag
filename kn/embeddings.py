from sentence_transformers import SentenceTransformer
import numpy as np
from .vector_hnsw import HNSWIndex

_model = None
def _get_model(name: str, device: str="auto"):
    global _model
    if _model is None:
        _model = SentenceTransformer(name, device=(None if device=="auto" else device))
    return _model

def embed_chunks(chunks, cfg):
    emc = cfg.get("embeddings", {})
    m = _get_model(emc.get("name", "BAAI/bge-small-en-v1.5"), emc.get("device", "auto"))
    if not chunks:
        return
    texts = [c["text"] for c in chunks]
    vecs = m.encode(texts, normalize_embeddings=emc.get("normalize", True), show_progress_bar=True)
    idx = HNSWIndex.open(cfg, dim=len(vecs[0]))
    idx.add([c["chunk_id"] for c in chunks], np.asarray(vecs, dtype=np.float32))
    idx.save()