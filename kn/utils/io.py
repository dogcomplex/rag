import pathlib, chardet
ROOT = pathlib.Path('.knowledge')
def ensure_dirs():
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