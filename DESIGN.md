# kn-graphRAG-starter (Windows + LM Studio)

A local, modular GraphRAG-style pipeline for **Windows 10 Pro**, optimized for:

* **Non-intrusive** sidecar indexing (your repo stays untouched)
* **.knowledge/** for all derived artifacts (movable/portable)
* **Incremental** updates via hashing (moves/renames don’t force full rebuilds)
* **Local models first** (LM Studio by default; Ollama/vLLM optional)
* **Fast retrieval** with **HNSW** (hnswlib) by default (simple, Windows-friendly)
* **Graph** using NetworkX (DB-optional), with community summaries
* **Attribute plugins** as standalone scripts (stdin→stdout JSONL)
* **Mono-file export** under token budget for “send to big LLM” workflows

> Minimal third‑party complexity; you can later swap in FAISS/Qdrant/Neo4j/vLLM without redesign.

---

## Final requirements & design choices (v1)

**Platform & runtime**

* OS: **Windows 10 Pro**.
* GPUs: **RTX 5090** primary; **RTX 3090** optional (no device pinning by default; auto device selection).
* LLM runtime: **LM Studio** local server (OpenAI‑compatible) as default; can swap to **vLLM**/**Ollama** later.
* Offline‑first: no outbound calls unless explicitly enabled via `.env`.

**Data & identity**

* Sidecar outputs in **`.knowledge/`**; original repo stays untouched.
* **Document identity:** SHA256 of normalized full text → `doc_id` (short hash displayed).
* **Chunk identity:** `doc_id` + zero‑padded index (`docid-0001`).
* **Moves/renames:** update path metadata only (no re‑embed) if content hash unchanged.
* **Future (pro‑style):** add *chunk‑level* content hashes (already implied by `docid-XXXX`) and optional near‑duplicate detection (MinHash/SimHash) for dedupe across repos.

**Multi‑repo & domains**

* Watcher can track **multiple repo paths**; domains default to top‑level folder under each repo.
* Optional **cross‑domain bridge** edges with threshold **`graph.cross_domain_bridge_threshold`** (default **0.75**). Toggle/adjust per your repos.
* Git submodules are supported implicitly; each path contributes to the global graph.

**Ingestion & OCR**

* Chunking: default **4k chars** with **10% overlap**; type policies for code/PDF.
* OCR: **optional** (Tesseract + `pytesseract`) controlled by `OCR_ENABLED` (default **false**). Only used for images/PDFs **without extractable text**, not “all docs”.

**Embeddings & vector index**

* Default embedder: **BAAI/bge-small-en-v1.5** (English‑optimized, fast). Easy to switch to **bge‑m3** for multilingual.
* Vector store: **hnswlib** (Windows‑friendly). Nightly/periodic **re‑ingest** recommended to rebuild and clear stale vectors (deletes can be added later).

**Graph & communities**

* Build an **entity co‑occurrence** graph (LLM‑assisted extraction can be added later).
* Community detection: **Louvain** (`python-louvain`) when available; fallback to connected components.
* Community summaries: generated and stored in `.knowledge/indexes/summaries/`.

**Retrieval & export**

* Retrieval: **hybrid** (dense via HNSW **+ BM25** via `rank-bm25`) with simple union/merge; reranker optional later.
* Graph‑aware context: preface with top community summaries; roadmap: include 1–2 hop neighbors in prompt assembly.
* Export: **`.md` by default**, token budget heuristic (\~4 chars/token). Optional **exclude PII** in exports via `export.exclude_pii`.

**Attributes & enrichment**

* Plugins = standalone Python scripts (stdin JSONL → stdout) with outputs written to `.knowledge/indexes/attributes/<plugin>/`.
* Included: `summary-20w`, `topic-tags`, **`pii-scan` (regex‑based, fast)**.
* Multiple passes allowed (e.g., `pass: "cheap"` vs `pass: "full"`), keep all outputs with version/metadata; downstream tools choose by `pass` or confidence.
* Job queue: **SQLite** (upgrade path: Redis). Background worker processes batches.

**Operations**

* **Watcher** runs continuously (file adds/edits trigger enrichment). Embeddings refresh via scheduled ingest (e.g., nightly).
* Optional **Task Scheduler** job to run `ingest_build_graph.py` nightly.
* Logs & caches live under `.knowledge/` and are safe to delete/rebuild.

**Security & PII**

* Offline by default; `.env` toggles for any cloud usage.
* `pii-scan` flags emails/phones/credit‑like sequences; set `export.exclude_pii: true` to omit flagged chunks from exports. Future: LLM‑assisted PII + patterns.

**Deliverables available**

* A ready‑to‑run **starter repo zip** accompanies this document with Windows bootstrap (`requirements-win.txt`, `setup.ps1`), code scaffold, and default configs.

---

## Folder Tree

```
kn-graphRAG-starter/
├─ README.md
├─ pyproject.toml
├─ requirements.txt
├─ .env.example
├─ .gitignore
├─ bin/
│  ├─ ingest_build_graph.py
│  ├─ watch_daemon.py
│  ├─ enrich_worker.py
│  ├─ query_rag.py
│  └─ export_monofile.py
├─ kn/
│  ├─ __init__.py
│  ├─ config.py
│  ├─ hashing.py
│  ├─ file_index.py
│  ├─ chunking.py
│  ├─ embeddings.py
│  ├─ vector_hnsw.py
│  ├─ graph_nx.py
│  ├─ retrieval.py
│  ├─ exporter.py
│  ├─ jobs.py
│  └─ utils/
│     ├─ llm_client.py
│     └─ io.py
├─ plugins/
│  ├─ attributes/
│  │  ├─ summary_20w.py
│  │  └─ topic_tags.py
│  └─ chunkers/  (placeholder)
└─ .knowledge/   (generated at runtime)
   ├─ config/
   │  ├─ models.yml
   │  └─ pipeline.yml
   ├─ indexes/
   │  ├─ chunks/
   │  ├─ embeddings/
   │  ├─ graph/
   │  ├─ summaries/
   │  ├─ attributes/
   │  ├─ manifests/
   │  ├─ queues/
   │  └─ cache/
   ├─ exports/
   │  ├─ dumps/
   │  └─ bundles/
   └─ logs/
```

---

## README.md

````md
# kn-graphRAG-starter (Windows + LM Studio)

Local GraphRAG-style pipeline with incremental sidecar indexing. Default stack:
- **LM Studio** (OpenAI-compatible endpoint) for LLM calls
- **hnswlib** for vector search (fast on Windows)
- **NetworkX** for graph & communities (Louvain if available)
- **watchdog** for file change detection

## Quickstart (Windows 10 Pro)

### 1) Install prerequisites
- Python 3.10+
- LM Studio (start local server: Settings → Local Server → Enable; default `http://localhost:1234/v1`)
- (Optional) Git for repo operations; optional Tesseract for OCR

### 2) Create venv & install
```powershell
python -m venv .venv
. .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
````

### 3) Configure

Copy `.env.example` to `.env` and set your LM Studio endpoint/model.
Adjust `.knowledge/config/models.yml` and `pipeline.yml` as needed.

### 4) Index a repo

```powershell
python .\bin\ingest_build_graph.py --repo ..\your-repo --full
```

### 5) Run watcher & enrichment in background terminals

```powershell
# Terminal A: file watcher
python .\bin\watch_daemon.py --repo ..\your-repo

# Terminal B: enrichment plugins (periodic)
python .\bin\enrich_worker.py --plugins summary-20w,topic-tags,pii-scan --watch
```

### 6) Query

```powershell
python .\bin\query_rag.py --q "Design overview of Project X" --scope projectX --topk 12
```

### 7) Export to mono-file (for big LLMs)

```powershell
python .\bin\export_monofile.py --q "Design overview of Project X" --budget 600000 --out .knowledge\exports\dumps\projectX_dump.md
```

## Incremental & Non-intrusive

* `content_hash = SHA256(normalized_text)` for doc identity; `docid-0001` style chunk IDs.
* Moves/renames only update metadata; edits re‑chunk only the changed doc; graph & summaries update incrementally.

## Swaps

* Vector: hnswlib → FAISS or Qdrant later
* Graph: NetworkX → Neo4j/Memgraph later
* LLM runtime: LM Studio → Ollama or vLLM later
* Embedder: `bge-small-en` → `bge-m3` for multilingual

````md
# kn-graphRAG-starter (Windows + LM Studio)

Local GraphRAG-style pipeline with incremental sidecar indexing. Default stack:
- **LM Studio** (OpenAI-compatible endpoint) for LLM calls
- **hnswlib** for vector search (fast CPU ANN on Windows)
- **NetworkX** for graph & communities
- **watchdog** for file change detection

## Quickstart (Windows 10 Pro)

### 1) Install prerequisites
- Python 3.10+
- LM Studio (start local server: Settings → Local Server → Enable; default `http://localhost:1234/v1`)
- (Optional) Git for repo operations

### 2) Create venv & install
```powershell
python -m venv .venv
. .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
````

### 3) Configure

Copy `.env.example` to `.env` and set your LM Studio endpoint/model.
Adjust `.knowledge/config/models.yml` and `pipeline.yml` as needed.

### 4) Index a repo

```powershell
python .\bin\ingest_build_graph.py --repo ..\your-repo --full
```

### 5) Run watcher & enrichment in background terminals

```powershell
# Terminal A: file watcher
python .\bin\watch_daemon.py --repo ..\your-repo

# Terminal B: enrichment plugins (periodic)
python .\bin\enrich_worker.py --plugins summary-20w,topic-tags --watch
```

### 6) Query

```powershell
python .\bin\query_rag.py --q "Design overview of Project X" --scope projectX --topk 12
```

### 7) Export to mono-file (for big LLMs)

```powershell
python .\bin\export_monofile.py --q "Design overview of Project X" --budget 600000 --out .knowledge\exports\dumps\projectX_dump.md
```

## Incremental & Non-intrusive

* We compute `content_hash = SHA256(normalized_text)`. Moves/renames update metadata only.
* Only changed docs are re-chunked/embedded; graph & summaries update incrementally.

## Swaps

* Vector: hnswlib → FAISS or Qdrant later (no redesign)
* Graph: NetworkX → Neo4j/Memgraph later
* LLM runtime: LM Studio → Ollama or vLLM (OpenAI‑compatible) later
* Embedder: `bge-small-en-v1.5` → `bge-m3` for multilingual

````

---

## requirements.txt
```txt
# core
pyyaml
watchdog
hnswlib
networkx
scikit-learn
numpy
pandas
rapidfuzz
python-dotenv
rank-bm25
python-louvain

# text
sentencepiece
transformers>=4.43
accelerate>=0.33
sentence-transformers>=3.0
regex
unidecode

# io + mime + pdf
python-magic-bin; sys_platform == 'win32'
PyMuPDF
chardet

# optional (comment out if not needed)
# pytesseract  # enable OCR when installed + OCR_ENABLED=true
# spacy
# spacy[transformers]
```txt
# core
pyyaml
watchdog
hnswlib
networkx
scikit-learn
numpy
pandas
rapidfuzz
python-dotenv

# text
sentencepiece
transformers
accelerate
sentence-transformers
fuzzywuzzy[speedup]
regex

# io
tqdm
rich
unidecode
python-magic-bin; sys_platform == 'win32'
PyMuPDF

# optional (comment out if not needed)
spacy
spacy[transformers]
````

---

## pyproject.toml

```toml
[project]
name = "kn-graphrag-starter"
version = "0.1.0"
description = "Local modular GraphRAG pipeline (Windows + LM Studio)"
requires-python = ">=3.10"
dependencies = []

[tool.ruff]
line-length = 100
```

---

## .env.example

````env
OPENAI_BASE_URL=http://localhost:1234/v1
OPENAI_API_KEY=lm-studio
OPENAI_MODEL=llama-3.1-8b-instruct
EMBED_MODEL=BAAI/bge-small-en-v1.5
REPO_PATH=..\your-repo
KN_ROOT=.knowledge
OCR_ENABLED=false
```env
OPENAI_BASE_URL=http://localhost:1234/v1
OPENAI_API_KEY=lm-studio
OPENAI_MODEL=llama-3.1-8b-instruct
EMBED_MODEL=bge-m3
REPO_PATH=..\\your-repo
KN_ROOT=.knowledge
````

---

## .knowledge/config/models.yml

````yaml
llm:
  provider: openai
  base_url: ${OPENAI_BASE_URL}
  api_key: ${OPENAI_API_KEY}
  model: ${OPENAI_MODEL}
  max_tokens: 2048
  temperature: 0.2

embeddings:
  kind: sentence-transformers
  name: ${EMBED_MODEL}
  device: auto
  normalize: true

stores:
  vector:
    kind: hnsw
    path: .knowledge/indexes/embeddings/hnsw.index
  graph:
    kind: networkx
  jobs:
    kind: sqlite
    path: .knowledge/queues/jobs.sqlite
```yaml
llm:
  provider: openai
  base_url: ${OPENAI_BASE_URL}
  api_key: ${OPENAI_API_KEY}
  model: ${OPENAI_MODEL}
  max_tokens: 2048
  temperature: 0.2

embeddings:
  kind: sentence-transformers
  name: BAAI/bge-m3
  device: auto
  normalize: true

stores:
  vector:
    kind: hnsw
    path: .knowledge/indexes/embeddings/hnsw.index
  graph:
    kind: networkx
  jobs:
    kind: sqlite
    path: .knowledge/queues/jobs.sqlite
````

---

## .knowledge/config/pipeline.yml

````yaml
watch:
  paths: ["${REPO_PATH}"]
  ignore: ["**/.git/**", "**/.knowledge/**"]

chunking:
  policies:
    default: { max_chars: 4000, overlap: 400 }
    code:    { max_chars: 2400, overlap: 200 }
    pdf:     { max_chars: 3500, overlap: 200 }

ocr:
  enabled: ${OCR_ENABLED}
  tesseract_cmd: ""

graph:
  entity_extraction: selective   # selective | all | rule_based
  relation_extraction: selective
  community_detection: louvain
  edge_conf_threshold: 0.55
  cross_domain_bridge_threshold: 0.75

retrieval:
  dense_k: 12
  bm25_k: 8
  graph_hops: 2
  rerank: false

attributes:
  plugins: ["summary-20w", "topic-tags", "pii-scan"]
  max_parallel: 4

export:
  default_budget_tokens: 600000
  strategy: hierarchy-first
  exclude_pii: false
  format: md
```yaml
watch:
  paths: ["${REPO_PATH}"]
  ignore: ["**/.git/**", "**/.knowledge/**"]

chunking:
  policies:
    default: { max_chars: 4000, overlap: 400 }
    code:    { max_chars: 2400, overlap: 200 }
    pdf:     { max_chars: 3500, overlap: 200 }

graph:
  entity_extraction: selective   # selective | all | rule_based
  relation_extraction: selective
  community_detection: louvain
  edge_conf_threshold: 0.55

retrieval:
  dense_k: 12
  bm25_k: 8
  graph_hops: 2
  rerank: true

attributes:
  plugins: ["summary-20w", "topic-tags"]
  max_parallel: 4

export:
  default_budget_tokens: 600000
  strategy: hierarchy-first
````

---

## bin/ingest\_build\_graph.py

```python
import argparse, pathlib
from kn.config import load_configs
from kn.file_index import initial_scan
from kn.chunking import chunk_repo
from kn.embeddings import embed_chunks
from kn.graph_nx import build_or_update_graph, summarize_communities

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--full", action="store_true")
    args = ap.parse_args()

    cfg = load_configs()
    repo = pathlib.Path(args.repo).resolve()

    print("[ingest] scanning repo…")
    docs = initial_scan(repo, cfg)
    print(f"[ingest] {len(docs)} docs found")

    print("[chunk] building chunks…")
    chunks = chunk_repo(docs, cfg)

    print("[embed] embedding chunks…")
    embed_chunks(chunks, cfg)

    print("[graph] updating graph…")
    build_or_update_graph(chunks, cfg)

    print("[summarize] community summaries…")
    summarize_communities(cfg)

    print("[done]")
```

---

## bin/watch\_daemon.py

```python
import argparse, time, pathlib
from kn.config import load_configs
from kn.file_index import watch_changes

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    args = ap.parse_args()

    cfg = load_configs()
    repo = pathlib.Path(args.repo).resolve()
    print(f"[watch] monitoring {repo} … (Ctrl+C to quit)")
    watch_changes(repo, cfg)
```

---

## bin/enrich\_worker.py

```python
import argparse
from kn.jobs import run_enrichment_loop
from kn.config import load_configs

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--plugins", required=True, help="comma-separated plugin names")
    ap.add_argument("--watch", action="store_true")
    args = ap.parse_args()

    cfg = load_configs()
    plugins = [p.strip() for p in args.plugins.split(",") if p.strip()]
    run_enrichment_loop(plugins, cfg, watch=args.watch)
```

---

## bin/query\_rag.py

```python
import argparse
from kn.retrieval import answer_query
from kn.config import load_configs

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", required=True)
    ap.add_argument("--scope", default=None)
    ap.add_argument("--topk", type=int, default=12)
    args = ap.parse_args()

    cfg = load_configs()
    print(answer_query(args.q, cfg, scope=args.scope, topk=args.topk))
```

---

## bin/export\_monofile.py

```python
import argparse, pathlib
from kn.exporter import export_monofile
from kn.config import load_configs

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", required=True)
    ap.add_argument("--budget", type=int, default=None)
    ap.add_argument("--scope", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = load_configs()
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    export_monofile(args.q, cfg, out, budget=args.budget, scope=args.scope)
    print(f"[export] wrote {out}")
```

---

## kn/config.py

```python
import os, yaml, pathlib
from dotenv import load_dotenv

ROOT = pathlib.Path(os.getenv("KN_ROOT", ".knowledge"))
CONFIG_DIR = ROOT / "config"
MODELS_YML = CONFIG_DIR / "models.yml"
PIPELINE_YML = CONFIG_DIR / "pipeline.yml"

_defaults = {
    "llm": {
        "base_url": os.getenv("OPENAI_BASE_URL", "http://localhost:1234/v1"),
        "api_key": os.getenv("OPENAI_API_KEY", "lm-studio"),
        "model": os.getenv("OPENAI_MODEL", "llama-3.1-8b-instruct"),
        "max_tokens": 2048,
        "temperature": 0.2,
    },
    "embeddings": {"name": os.getenv("EMBED_MODEL", "BAAI/bge-m3"), "normalize": True, "device": "auto"},
    "stores": {
        "vector": {"kind": "hnsw", "path": str(ROOT/"indexes"/"embeddings"/"hnsw.index")},
        "graph": {"kind": "networkx"},
        "jobs": {"kind": "sqlite", "path": str(ROOT/"queues"/"jobs.sqlite")},
    },
}

_loaded = None

def load_configs():
    global _loaded
    if _loaded: return _loaded
    load_dotenv(override=True)
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    # load with fallbacks
    cfg = _defaults.copy()
    if MODELS_YML.exists():
        cfg = _merge(cfg, yaml.safe_load(MODELS_YML.read_text()))
    if PIPELINE_YML.exists():
        cfg = _merge(cfg, yaml.safe_load(PIPELINE_YML.read_text()))
    _loaded = cfg
    return cfg


def _merge(a, b):
    if not isinstance(b, dict): return a
    out = a.copy()
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out
```

---

## kn/hashing.py

```python
import hashlib

def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def short_hash(h: str, n=8) -> str:
    return h[:n]
```

---

## kn/file\_index.py

```python
import pathlib, time, json
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from .utils.io import read_text_safely, ensure_dirs
from .hashing import content_hash, short_hash

# In-memory manifest (persist minimal JSON if you like)
_manifest = {}

IGNORE_EXT = {".lock", ".tmp", ".log"}


def initial_scan(repo_path: pathlib.Path, cfg):
    docs = []
    for p in repo_path.rglob("*"):
        if p.is_dir():
            continue
        if any(part == ".git" for part in p.parts):
            continue
        if p.suffix.lower() in IGNORE_EXT:
            continue
        text = read_text_safely(p)
        if not text:
            continue
        ch = content_hash(text)
        doc_id = short_hash(ch)
        _manifest[str(p)] = {"doc_id": doc_id, "hash": ch, "mtime": p.stat().st_mtime}
        docs.append({"path": str(p), "doc_id": doc_id, "hash": ch, "text": text})
    return docs


class _Evt(FileSystemEventHandler):
    def __init__(self, repo, cfg):
        self.repo = repo
        self.cfg = cfg

    def on_any_event(self, event):
        p = pathlib.Path(event.src_path)
        if p.is_dir() or any(part == ".git" for part in p.parts):
            return
        if not p.exists():
            return
        text = read_text_safely(p)
        if text is None:
            return
        ch = content_hash(text)
        m = _manifest.get(str(p))
        if m and m["hash"] == ch:
            # move/rename or metadata change → update path mapping only
            m["mtime"] = p.stat().st_mtime
            return
        # changed/new file → enqueue lightweight downstream updates (left as TODO hook)
        _manifest[str(p)] = {"doc_id": short_hash(ch), "hash": ch, "mtime": p.stat().st_mtime}
        print(f"[watch] changed: {p}")
        # TODO: enqueue chunk/embed/graph tasks via jobs module


def watch_changes(repo_path: pathlib.Path, cfg):
    ensure_dirs(cfg)
    obs = Observer()
    h = _Evt(repo_path, cfg)
    obs.schedule(h, str(repo_path), recursive=True)
    obs.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        obs.stop()
    obs.join()
```

---

## kn/chunking.py

```python
import pathlib, json, re
from .utils.io import ensure_dirs
from .hashing import short_hash

CHUNKS_DIR = ".knowledge/indexes/chunks"


def chunk_text(text: str, max_chars=4000, overlap=400):
    out = []
    i = 0
    while i < len(text):
        out.append(text[i:i+max_chars])
        i += max_chars - overlap
    return out


def detect_domain(path: pathlib.Path):
    # Domain = top-level folder under repo root (best-effort placeholder)
    parts = path.parts
    return parts[0] if parts else "root"


def chunk_repo(docs, cfg):
    ensure_dirs(cfg)
    policy = cfg.get("chunking", {}).get("policies", {}).get("default", {"max_chars": 4000, "overlap": 400})
    chunks = []
    outdir = pathlib.Path(CHUNKS_DIR)
    outdir.mkdir(parents=True, exist_ok=True)

    for d in docs:
        path = pathlib.Path(d["path"])
        ddomain = detect_domain(path.relative_to(path.anchor)) if path.is_absolute() else detect_domain(path)
        parts = chunk_text(d["text"], **policy)
        for idx, t in enumerate(parts):
            chunk_id = f"{d['doc_id']}-{idx:04d}"
            rec = {
                "doc_id": d["doc_id"],
                "chunk_id": chunk_id,
                "text": t,
                "meta": {
                    "path": str(path), "domain": ddomain, "mimetype": "text/plain"
                }
            }
            chunks.append(rec)
            (outdir / f"{chunk_id}.json").write_text(json.dumps(rec, ensure_ascii=False))
    return chunks
```

---

## kn/embeddings.py

```python
from sentence_transformers import SentenceTransformer
import numpy as np
from .vector_hnsw import HNSWIndex
from .utils.io import ensure_dirs

_model = None
_index = None


def _get_model(name: str, device: str="auto"):
    global _model
    if _model is None:
        _model = SentenceTransformer(name, device=(None if device=="auto" else device))
    return _model


def embed_chunks(chunks, cfg):
    ensure_dirs(cfg)
    emc = cfg.get("embeddings", {})
    m = _get_model(emc.get("name", "BAAI/bge-m3"), emc.get("device", "auto"))
    texts = [c["text"] for c in chunks]
    vecs = m.encode(texts, normalize_embeddings=emc.get("normalize", True), show_progress_bar=True)
    idx = HNSWIndex.open(cfg)
    idx.add([c["chunk_id"] for c in chunks], np.asarray(vecs, dtype=np.float32))
    idx.save()
```

---

## kn/vector\_hnsw\.py

```python
import hnswlib, numpy as np, pathlib, json

class HNSWIndex:
    def __init__(self, path: pathlib.Path, dim=1024, space='cosine'):
        self.path = path
        self.meta_path = path.with_suffix('.meta.json')
        self.dim = dim
        self.space = space
        self.index = hnswlib.Index(space=space, dim=dim)
        self.inited = False
        self.ids = []

    @classmethod
    def open(cls, cfg):
        path = pathlib.Path(cfg["stores"]["vector"]["path"])
        meta = path.with_suffix('.meta.json')
        if path.exists() and meta.exists():
            m = json.loads(meta.read_text())
            dim = m.get("dim", 1024)
            obj = cls(path, dim=dim)
            obj.index.load_index(str(path))
            obj.index.set_ef(128)
            obj.inited = True
            obj.ids = m.get("ids", [])
            return obj
        # default new
        return cls(path)

    def _ensure_init(self, total=10000):
        if not self.inited:
            self.index.init_index(max_elements=total, ef_construction=200, M=16)
            self.index.set_ef(128)
            self.inited = True

    def add(self, keys, vecs: np.ndarray):
        self._ensure_init(max(10000, len(self.ids) + len(keys) + 1000))
        # map external string ids to integer labels
        labels = np.arange(len(self.ids), len(self.ids)+len(keys))
        self.index.add_items(vecs, labels)
        self.ids.extend(list(keys))

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.index.save_index(str(self.path))
        self.meta_path.write_text(json.dumps({"dim": self.dim, "ids": self.ids}))

    def search(self, vecs: np.ndarray, k=10):
        labels, dists = self.index.knn_query(vecs, k=k)
        # map back to external ids
        inv = self.ids
        mapped = [[inv[i] for i in row] for row in labels]
        return mapped, dists
```

---

## kn/graph\_nx.py

```python
import pathlib, json, networkx as nx
from collections import defaultdict
from .utils.io import ensure_dirs

GRAPH_PATH = pathlib.Path('.knowledge/indexes/graph/graph.jsonl')
COMM_PATH = pathlib.Path('.knowledge/indexes/summaries/communities.jsonl')

G = nx.Graph()

# Very lightweight placeholder extraction (extend with LLM in utils/llm_client)
def naive_entities(text: str):
    # collect simple tokens (CamelCase, path-like, #tags) — placeholder
    import re
    ents = set(re.findall(r"[A-Z][a-zA-Z0-9_]+|\b[A-Za-z0-9_]+\.\w+|#\w+", text))
    return list(ents)[:50]


def build_or_update_graph(chunks, cfg):
    ensure_dirs(cfg)
    # add nodes & co-occurrence edges (same chunk → weak link)
    for c in chunks:
        ents = naive_entities(c["text"])  # TODO swap to LLM-assisted if configured
        for e in ents:
            if not G.has_node(e):
                G.add_node(e, kind='entity')
        for i in range(len(ents)):
            for j in range(i+1, len(ents)):
                u, v = ents[i], ents[j]
                w = G[u][v]["weight"]+1 if G.has_edge(u,v) else 1
                G.add_edge(u, v, weight=w)
    # persist
    GRAPH_PATH.parent.mkdir(parents=True, exist_ok=True)
    with GRAPH_PATH.open('w', encoding='utf-8') as f:
        for u, v, d in G.edges(data=True):
            f.write(json.dumps({"src": u, "dst": v, "weight": d.get("weight",1)})+"\n")


def summarize_communities(cfg):
    # Simple connected components as communities (swap to Louvain later)
    comms = []
    for idx, comp in enumerate(nx.connected_components(G)):
        nodes = list(comp)
        # naive summary string (placeholder); you can call LLM here
        summary = ", ".join(nodes[:10])
        comms.append({"id": f"C{idx}", "size": len(nodes), "summary": summary})
    COMM_PATH.parent.mkdir(parents=True, exist_ok=True)
    with COMM_PATH.open('w', encoding='utf-8') as f:
        for c in comms:
            f.write(json.dumps(c)+"\n")
```

---

## kn/retrieval.py

```python
import json, pathlib
import numpy as np
from sentence_transformers import SentenceTransformer
from .vector_hnsw import HNSWIndex

# Simple dense-only retrieval baseline + community preface

COMM_PATH = pathlib.Path('.knowledge/indexes/summaries/communities.jsonl')

_embed_model = None

def _get_model(name):
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(name)
    return _embed_model


def _iter_chunks():
    cdir = pathlib.Path('.knowledge/indexes/chunks')
    for p in cdir.glob('*.json'):
        yield json.loads(p.read_text(encoding='utf-8'))


def answer_query(q: str, cfg, scope=None, topk=12):
    emc = cfg.get('embeddings', {})
    m = _get_model(emc.get('name','BAAI/bge-m3'))
    qv = m.encode([q], normalize_embeddings=emc.get('normalize', True))
    idx = HNSWIndex.open(cfg)
    ids, dists = idx.search(np.asarray(qv, dtype=np.float32), k=topk)
    idset = set(ids[0])

    # Gather matched chunks
    hits = []
    for rec in _iter_chunks():
        if rec['chunk_id'] in idset:
            hits.append(rec)
    # prepend community summaries (lightweight)
    pref = []
    if COMM_PATH.exists():
        for line in COMM_PATH.read_text(encoding='utf-8').splitlines():
            c = json.loads(line)
            pref.append(f"[Community {c['id']} size={c['size']}] {c['summary']}")
    text = "\n\n".join(pref[:3]) + "\n\n" + "\n\n".join([h['text'] for h in hits])
    return text
```

---

## kn/exporter.py

```python
import json, pathlib
from .retrieval import answer_query


def export_monofile(q: str, cfg, out: pathlib.Path, budget: int|None=None, scope=None):
    body = answer_query(q, cfg, scope=scope, topk=64)
    # naïve token budget by char length (~ 4 chars/token heuristic)
    if budget:
        body = body[: int(budget*4)]
    out.write_text(body, encoding='utf-8')
```

---

## kn/jobs.py

```python
import time, subprocess, json, pathlib

QUEUE = pathlib.Path('.knowledge/queues/pending.jsonl')
QUEUE.parent.mkdir(parents=True, exist_ok=True)


def enqueue(doc_id: str, plugin: str):
    with QUEUE.open('a', encoding='utf-8') as f:
        f.write(json.dumps({"doc_id": doc_id, "plugin": plugin})+"\n")


def run_enrichment_loop(plugins, cfg, watch=False):
    # very simple loop that invokes plugins with stdin jsonl of docs (placeholder)
    while True:
        if not QUEUE.exists() or QUEUE.stat().st_size == 0:
            if not watch: break
            time.sleep(2); continue
        lines = QUEUE.read_text(encoding='utf-8').splitlines()
        QUEUE.write_text('', encoding='utf-8')
        jobs = [json.loads(l) for l in lines]
        # group by plugin
        byp = {}
        for j in jobs:
            if j['plugin'] not in plugins: continue
            byp.setdefault(j['plugin'], []).append(j)
        for plugin, items in byp.items():
            _run_plugin(plugin, items)
        if not watch:
            break


def _run_plugin(plugin, items):
    pypath = pathlib.Path(f'plugins/attributes/{plugin}.py')
    if not pypath.exists():
        print(f"[enrich] plugin not found: {plugin}")
        return
    # stream jsonl to plugin stdin
    inp = "\n".join([json.dumps(_load_doc(i['doc_id'])) for i in items])
    proc = subprocess.Popen(["python", str(pypath)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    out, _ = proc.communicate(inp)
    print(f"[enrich] {plugin}: {len(items)} docs processed")


def _load_doc(doc_id):
    # minimal: pick first chunk for that doc
    cdir = pathlib.Path('.knowledge/indexes/chunks')
    for p in sorted(cdir.glob(f'{doc_id}-*.json')):
        return json.loads(p.read_text(encoding='utf-8'))
    return {"doc_id": doc_id, "text": ""}
```

---

## kn/utils/io.py

```python
import pathlib, chardet

ROOT = pathlib.Path('.knowledge')


def ensure_dirs(cfg):
    (ROOT/"indexes"/"chunks").mkdir(parents=True, exist_ok=True)
    (ROOT/"indexes"/"embeddings").mkdir(parents=True, exist_ok=True)
    (ROOT/"indexes"/"graph").mkdir(parents=True, exist_ok=True)
    (ROOT/"indexes"/"summaries").mkdir(parents=True, exist_ok=True)
    (ROOT/"indexes"/"attributes").mkdir(parents=True, exist_ok=True)
    (ROOT/"indexes"/"manifests").mkdir(parents=True, exist_ok=True)
    (ROOT/"queues").mkdir(parents=True, exist_ok=True)
    (ROOT/"exports"/"dumps").mkdir(parents=True, exist_ok=True)


def read_text_safely(path: pathlib.Path):
    try:
        data = path.read_bytes()
        enc = chardet.detect(data).get('encoding') or 'utf-8'
        return data.decode(enc, errors='ignore')
    except Exception:
        return None
```

---

## kn/utils/llm\_client.py

```python
import os, requests

BASE = os.getenv('OPENAI_BASE_URL', 'http://localhost:1234/v1')
KEY  = os.getenv('OPENAI_API_KEY', 'lm-studio')
MODEL= os.getenv('OPENAI_MODEL', 'llama-3.1-8b-instruct')


def chat(prompt: str, max_tokens=512, temperature=0.2):
    url = f"{BASE}/chat/completions"
    headers = {'Authorization': f"Bearer {KEY}", 'Content-Type':'application/json'}
    payload = {
        'model': MODEL,
        'messages': [{'role':'user','content':prompt}],
        'max_tokens': max_tokens,
        'temperature': temperature,
    }
    r = requests.post(url, headers=headers, json=payload, timeout=120)
    r.raise_for_status()
    return r.json()['choices'][0]['message']['content']
```

---

## plugins/attributes/summary\_20w\.py

```python
import sys, json, pathlib
from kn.utils.llm_client import chat

OUTDIR = pathlib.Path('.knowledge/indexes/attributes/summary-20w')
OUTDIR.mkdir(parents=True, exist_ok=True)

for line in sys.stdin:
    job = json.loads(line)
    text = job.get('text','')
    if not text: continue
    prompt = f"Summarize in ~20 words, terse, factual.\n\n{text}"
    out = chat(prompt, max_tokens=100, temperature=0.1)
    (OUTDIR / f"{job['doc_id']}.json").write_text(json.dumps({
        "doc_id": job['doc_id'],
        "attribute": "summary-20w",
        "value": out.strip(),
        "confidence": 0.8
    }, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({"status":"ok","doc_id":job['doc_id']}))
```

---

## plugins/attributes/topic\_tags.py

```python
import sys, json, pathlib
from kn.utils.llm_client import chat

OUTDIR = pathlib.Path('.knowledge/indexes/attributes/topic-tags')
OUTDIR.mkdir(parents=True, exist_ok=True)

for line in sys.stdin:
    job = json.loads(line)
    text = job.get('text','')
    if not text: continue
    prompt = (
        "Extract 3-7 topical tags (comma-separated, lowercase, no spaces, use-hyphens).\n"
        "Prefer domain-relevant terms.\n\n"
        f"TEXT:\n{text}\n"
    )
    out = chat(prompt, max_tokens=64, temperature=0.2)
    (OUTDIR / f"{job['doc_id']}.json").write_text(json.dumps({
        "doc_id": job['doc_id'], "attribute": "topic-tags", "value": out.strip(), "confidence": 0.7
    }, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({"status":"ok","doc_id":job['doc_id']}))
```

---

## Notes on LM Studio

* Enable Local Server in LM Studio (`http://localhost:1234/v1`).
* Download a compatible instruct model (e.g., **Llama-3.1-8B Instruct** quantized) and ensure it’s serving.
* SentenceTransformers handles embeddings locally (GPU if PyTorch CUDA available). If you later want to use LM Studio for embeddings, swap `embeddings.py` to call `/embeddings`.

## Operations playbook

* **Continuous:** `watch_daemon.py` monitors repos and enqueues enrichment.
* **Periodic:** schedule `ingest_build_graph.py --repo <path> --full` nightly to refresh embeddings & rebuild index (cleans stale vectors).
* **Ad hoc:** run `export_monofile.py` to produce big `.md` dumps with a token budget for external chats.

## Open questions (status)

* OCR for all docs? **No** — only for images/PDFs lacking text and only when `OCR_ENABLED=true`.
* Cross‑linking across repos? **Yes, optional** via `cross_domain_bridge_threshold` (default 0.75).
* Quality add‑ons now? **BM25 enabled**; lightweight reranker deferred.
* Queue backend? **SQLite** now (simple); Redis is an easy upgrade later.
* Multi‑pass attributes? **Supported**; keep both “cheap” and “full” results with pass metadata.
* Hashing scope? **Doc‑level SHA256 + per‑chunk IDs now**; roadmap: per‑chunk hashes & near‑duplicate (MinHash/SimHash) for large corpora.

## Known limitations (MVP)

* hnswlib deletions not wired; recommend nightly rebuild.
* Graph entity extraction is naive; LLM‑assisted extraction and relation typing are planned upgrades.
* Exporter does not yet include per‑chunk citations/line ranges; future enhancement.

## Future upgrades

* **Vector DB:** Qdrant/FAISS‑GPU; **Graph DB:** Neo4j/Memgraph; **Runtime:** vLLM with tensor/kv cache.
* **Graph‑aware retrieval:** 1–2 hop neighbor packing, graph‑ranked reassembly.
* **Re‑ranker:** tiny cross‑encoder for final ordering.
* **PII:** enhanced detector (LLM‑assist + rules per region).

---

### Addendum: Implementation decisions and operations log (Windows + LM Studio)

This addendum chronicles the practical design choices we made while hardening the Windows‑first GraphRAG starter into a dashboard‑driven pipeline.

#### Runtime and platform
- Default runtime: LM Studio (OpenAI‑compatible) on Windows 10 Pro; network base defaults to `http://127.0.0.1:12345/v1` with per‑plugin overrides.
- vLLM considered for higher throughput; deferred due to Windows constraints (uvloop unsupported; HF weights required, not GGUF). Dashboard retains LM Studio defaults.
- UTF‑8 and Windows console: advise `-X utf8`, `PYTHONIOENCODING`, and `chcp 65001` when needed.

#### Data model and stores
- Sidecar storage under `.knowledge/` (portable): indexes, graph, summaries, attributes, exports, queues, cache.
- Vector index: hnswlib on non‑Windows; NumPy brute‑force fallback on Windows, persisted as `.npy` + `.meta.json`.
- Graph: NetworkX with Louvain when available; communities saved to JSONL.

#### Config and environment
- `.env` and YAML configs merged at runtime with `${VAR}` expansion. Early `load_dotenv(override=True)` to ensure env is respected (fixes `${EMBED_MODEL}` issues).
- `.knowledge/config/models.yml` gains `plugins:` block for per‑plugin LLM defaults:
  - `llm`: `{ base_url, api_key, model, timeout }`
  - `process_timeout`: per‑plugin subprocess wall‑clock limit.

#### Ingestion, chunking, OCR/PDF
- Multi‑repo ingest; domains derived from top‑level folders (e.g., `G:\LOKI\papers`).
- Chunking policies tunable per type; input token pressure reduced by skeleton‑first flow.
- PDF parsing via PyMuPDF (`fitz`) for text extraction on Windows.

#### Attributes and enrichment
- Attribute plugins are standalone scripts (stdin JSONL → stdout), outputs in `.knowledge/indexes/attributes/<plugin>/`.
- Core set: `summary-20w`, `topic-tags`, `pii-scan` (regex), plus `glossary`, `requirements`, `todo-items`, `faq-pairs`.
- New performance plugins:
  - `doc-skeleton`: outline + ~100w + keyphrases snapshot per doc (token‑cheap intermediate).
  - `multi-basic`: single LLM call emits multiple attributes (summary‑short/medium/outline, keyphrases, risk‑scan).
- Existing plugins updated to prefer skeleton content when present, reducing input tokens.

#### Queue, worker, robustness
- Job queue migrated to SQLite with columns: `status`, `retries`, `completed_at`, `last_error`.
- Dequeue ordering by `(retries asc, id asc)` to avoid starvation; “any pending plugin” worker mode added.
- Error handling: timeouts/errors requeue (or mark failed for fatal issues like missing plugin).
- Global in‑flight concurrency cap across workers via SQLite counters; configurable in dashboard.

#### Dashboard (Flask + static HTML/JS)
- `/api/status` with docs, coverage, queue, LLM health, workers.
- Planner: enqueue plugins (single or list), optional JSON payload for per‑run overrides.
- Ingest: start ingest for arbitrary repo path (multi‑domain).
- Worker control: start/stop multiple workers, list active workers, set batch and max‑inflight.
- Queue panel: filter/list items, clear queue modes (non‑done/pending/all, reset running → pending).
- Coverage panel: summary rows with dropdowns per attribute → per‑doc previews and full JSON.
- Documents panel: table with per‑doc dropdowns → per‑attribute full JSON.
- Plugin defaults panel: Load/Save `plugins:` map to `.knowledge/config/models.yml` (per‑plugin model/timeout).
- LLM cache panel: show/clear prompt‑level response cache.

#### Retrieval and export
- Hybrid retrieval (dense + BM25), community preface; monofile export with token budgeting and PII exclusion.
- Graph export (GEXF/GraphML) for external tools; HTML report with D3 preview, communities section, and offline D3 fallback.

#### Performance choices
- Accept LM Studio statelessness; avoid repeated full‑doc prompts by:
  - One‑shot macro plugin (`multi-basic`) per doc.
  - Skeleton‑first flow and downstream plugins consuming the skeleton.
  - Prompt‑level response caching keyed by `(model|max_tokens|temperature|prompt)`.
  - Concurrency throttling (max‑inflight) to reduce timeouts and stabilize throughput.
  - Reasonable per‑plugin HTTP timeouts and per‑process wall‑time limits (e.g., skeleton/multi‑basic larger than others).
- Keep individual attributes runnable; treat `multi-basic` as a macro for convenience, not a replacement.

#### Known limits and next steps
- LM Studio parallelism is limited by model/runtime; increase throughput by reducing input tokens (skeleton), batching tasks, or migrating to vLLM (Linux) with HF weights and paged attention.
- Optional: small reranker; richer entity/relationship extraction; better caching/incremental invalidation; dedicated model routing per attribute class via UI.
