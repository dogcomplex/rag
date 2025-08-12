import os, yaml, pathlib, re
from dotenv import load_dotenv

def _merge(a, b):
    if not isinstance(b, dict): return a
    out = a.copy()
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out

def load_configs():
    load_dotenv(override=True)
    root = pathlib.Path(os.getenv("KN_ROOT", ".knowledge"))
    cfg_dir = root / "config"
    models_yml = cfg_dir / "models.yml"
    pipeline_yml = cfg_dir / "pipeline.yml"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg = {
        "llm": {
            "base_url": os.getenv("OPENAI_BASE_URL", "http://localhost:1234/v1"),
            "api_key": os.getenv("OPENAI_API_KEY", "lm-studio"),
            "model": os.getenv("OPENAI_MODEL", "llama-3.1-8b-instruct"),
            "max_tokens": 2048,
            "temperature": 0.2,
        },
        "embeddings": {"name": os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5"), "normalize": True, "device": "auto"},
        "stores": {
            "vector": {"kind": "hnsw", "path": str(root/"indexes"/"embeddings"/"hnsw.index")},
            "graph": {"kind": "networkx"},
            "jobs": {"kind": "sqlite", "path": str(root/"queues"/"jobs.sqlite")},
        },
        "watch": {"paths": [os.getenv("REPO_PATH", "../your-repo")], "ignore": ["**/.git/**","**/.knowledge/**"]},
        "ocr": {"enabled": os.getenv("OCR_ENABLED","false").lower() == "true", "tesseract_cmd": ""},
        "export": {"default_budget_tokens": 600000, "strategy":"hierarchy-first", "exclude_pii": False, "format":"md"},
        "retrieval": {"dense_k":12, "bm25_k":8, "graph_hops":2, "rerank": False},
        "graph": {"community_detection":"louvain", "edge_conf_threshold":0.55, "cross_domain_bridge_threshold":0.75,
                  "entity_extraction":"selective", "relation_extraction":"selective"},
        "chunking": {"policies":{"default":{"max_chars":4000,"overlap":400},
                                 "code":{"max_chars":2400,"overlap":200},
                                 "pdf":{"max_chars":3500,"overlap":200}}}
    }
    if models_yml.exists():
        cfg = _merge(cfg, yaml.safe_load(models_yml.read_text(encoding="utf-8")))
    if pipeline_yml.exists():
        cfg = _merge(cfg, yaml.safe_load(pipeline_yml.read_text(encoding="utf-8")))
    cfg = _expand_env_vars(cfg)
    cfg["_root"] = str(root)
    return cfg

_ENV_VAR_PATTERN = re.compile(r"\$\{([A-Z0-9_]+)\}")

def _expand_env_vars(obj):
    if isinstance(obj, dict):
        return {k: _expand_env_vars(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_env_vars(v) for v in obj]
    if isinstance(obj, str):
        def repl(m):
            key = m.group(1)
            return os.getenv(key, m.group(0))
        return _ENV_VAR_PATTERN.sub(repl, obj)
    return obj