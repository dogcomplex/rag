import json, pathlib
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from .vector_hnsw import HNSWIndex
COMM_PATH = pathlib.Path('.knowledge/indexes/summaries/communities.jsonl')
CHUNK_DIR = pathlib.Path('.knowledge/indexes/chunks')
_embed_model = None
def _get_model(name):
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(name)
    return _embed_model
def _iter_chunks():
    for p in CHUNK_DIR.glob('*.json'):
        yield json.loads(p.read_text(encoding='utf-8'))
def _bm25_corpus():
    docs = list(_iter_chunks())
    corpus = [ (d["chunk_id"], (d["text"] or "").split()) for d in docs ]
    return docs, corpus
def answer_query(q: str, cfg, scope=None, topk=12):
    emc = cfg.get('embeddings', {})
    m = _get_model(emc.get('name','BAAI/bge-small-en-v1.5'))
    qv = m.encode([q], normalize_embeddings=emc.get('normalize', True))
    idx = HNSWIndex.open(cfg, dim=len(qv[0]))
    ids, dists = idx.search(np.asarray(qv, dtype=np.float32), k=topk)
    idset = set(ids[0]) if ids else set()
    chunks_by_id = { }
    for rec in _iter_chunks():
        chunks_by_id[rec['chunk_id']] = rec
    dense_hits = [chunks_by_id[i] for i in idset if i in chunks_by_id]
    bm25_hits = []
    try:
        docs, corpus = _bm25_corpus()
        bm25 = BM25Okapi([tokens for _, tokens in corpus])
        scores = bm25.get_scores(q.split())
        k = cfg.get("retrieval", {}).get("bm25_k", 8)
        ranked = np.argsort(scores)[::-1][:k]
        for idx_i in ranked:
            bm25_hits.append(docs[idx_i])
    except Exception:
        pass
    used = set()
    merged = []
    for h in dense_hits + bm25_hits:
        cid = h['chunk_id']
        if cid in used: continue
        used.add(cid)
        merged.append(h)
        if len(merged) >= topk: break
    pref = []
    if COMM_PATH.exists():
        for line in COMM_PATH.read_text(encoding='utf-8').splitlines()[:3]:
            c = json.loads(line)
            pref.append(f"[Community {c['id']} size={c['size']}] {c['summary']}")
    text = "\n\n".join(pref) + "\n\n" + "\n\n".join([h['text'] for h in merged])
    return text