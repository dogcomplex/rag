import pathlib, json
import numpy as np
try:
    import hnswlib  # optional on Windows
    _HNSW_AVAILABLE = True
except Exception:
    hnswlib = None
    _HNSW_AVAILABLE = False

class _BruteForceIndex:
    def __init__(self, space: str, dim: int, store_path: str | None = None):
        self.space = space
        self.dim = dim
        self.vectors = None  # np.ndarray shape (N, D)
        self.store_path = store_path
    def load_index(self, path: str):
        # For brute-force, read from an .npy file alongside meta
        npy = self.store_path or (path + ".npy")
        try:
            if pathlib.Path(npy).exists():
                self.vectors = np.load(npy).astype(np.float32)
        except Exception:
            self.vectors = None
    def set_ef(self, ef: int):
        pass
    def init_index(self, max_elements: int, ef_construction: int, M: int):
        pass
    def add_items(self, vecs: np.ndarray, labels: np.ndarray):
        if self.vectors is None:
            self.vectors = vecs.astype(np.float32)
        else:
            self.vectors = np.vstack([self.vectors, vecs.astype(np.float32)])
    def save_index(self, path: str):
        # Persist vectors so future processes can search
        npy = self.store_path or (path + ".npy")
        try:
            if self.vectors is not None:
                pathlib.Path(npy).parent.mkdir(parents=True, exist_ok=True)
                np.save(npy, self.vectors)
        except Exception:
            pass
    def knn_query(self, q: np.ndarray, k: int = 10):
        if self.vectors is None or len(self.vectors) == 0:
            return np.zeros((len(q), 0), dtype=int), np.zeros((len(q), 0), dtype=np.float32)
        # cosine or l2
        Q = q.astype(np.float32)
        X = self.vectors
        if self.space == 'cosine':
            def normalize(a):
                n = np.linalg.norm(a, axis=1, keepdims=True) + 1e-12
                return a / n
            Qn = normalize(Q)
            Xn = normalize(X)
            sims = Qn @ Xn.T
            # higher is better; convert to distances for compatibility
            dists = 1.0 - sims
        else:
            # l2 distance
            # ||Q - X||^2 = ||Q||^2 + ||X||^2 - 2 Q X^T
            Q2 = np.sum(Q * Q, axis=1, keepdims=True)
            X2 = np.sum(X * X, axis=1)[None, :]
            dists = Q2 + X2 - 2.0 * (Q @ X.T)
        idxs = np.argsort(dists, axis=1)[:, :k]
        rows = np.take_along_axis(dists, idxs, axis=1)
        return idxs, rows
class HNSWIndex:
    def __init__(self, path: pathlib.Path, dim=768, space='cosine'):
        self.path = path
        self.meta_path = path.with_suffix('.meta.json')
        self.dim = dim
        self.space = space
        if _HNSW_AVAILABLE:
            self.index = hnswlib.Index(space=space, dim=dim)
        else:
            self.index = _BruteForceIndex(space=space, dim=dim, store_path=str(self.path) + ".npy")
        self.inited = False
        self.ids = []
    @classmethod
    def open(cls, cfg, dim=768):
        path = pathlib.Path(cfg["stores"]["vector"]["path"])
        meta = path.with_suffix('.meta.json')
        if path.exists() and meta.exists():
            m = json.loads(meta.read_text())
            dim = m.get("dim", dim)
            obj = cls(path, dim=dim)
            # if brute-force, nothing to load
            try:
                obj.index.load_index(str(path))
                obj.index.set_ef(128)
            except Exception:
                pass
            obj.inited = True
            obj.ids = m.get("ids", [])
            return obj
        return cls(path, dim=dim)
    def _ensure_init(self, total=10000):
        if not self.inited:
            try:
                self.index.init_index(max_elements=total, ef_construction=200, M=16)
                self.index.set_ef(128)
            except Exception:
                pass
            self.inited = True
    def add(self, keys, vecs: np.ndarray):
        self._ensure_init(max(10000, len(self.ids) + len(keys) + 1000))
        labels = np.arange(len(self.ids), len(self.ids)+len(keys))
        self.index.add_items(vecs, labels)
        self.ids.extend(list(keys))
    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.index.save_index(str(self.path))
        except Exception:
            pass
        self.meta_path.write_text(json.dumps({"dim": self.dim, "ids": self.ids}))
    def search(self, vecs: np.ndarray, k=10):
        labels, dists = self.index.knn_query(vecs, k=k)
        inv = self.ids
        mapped = [[inv[i] for i in row] for row in labels]
        return mapped, dists