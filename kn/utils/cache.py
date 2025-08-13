import hashlib, json, pathlib

CACHE_DIR = pathlib.Path('.knowledge/cache/llm')

def _hash_key(s: str) -> str:
    return hashlib.sha1(s.encode('utf-8', errors='ignore')).hexdigest()

def get_cached_response(key: str) -> str|None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    p = CACHE_DIR / (_hash_key(key) + '.json')
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding='utf-8'))
        return data.get('content')
    except Exception:
        return None

def set_cached_response(key: str, content: str) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    p = CACHE_DIR / (_hash_key(key) + '.json')
    p.write_text(json.dumps({'content': content}, ensure_ascii=False), encoding='utf-8')

