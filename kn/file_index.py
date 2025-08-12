import pathlib, time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from .utils.io import read_text_safely, ensure_dirs
from .hashing import content_hash, short_hash
from .jobs_sqlite import ensure_db as _ensure_jobs_db, enqueue as _enqueue_job

_manifest = {}  # path -> {doc_id, hash, mtime}
IGNORE_DIRS = {".git", ".knowledge"}
IGNORE_EXT = {".lock", ".tmp", ".log"}

def initial_scan(repo_path: pathlib.Path, cfg):
    docs = []
    for p in repo_path.rglob("*"):
        if p.is_dir():
            if p.name in IGNORE_DIRS: continue
            continue
        if any(part in IGNORE_DIRS for part in p.parts): continue
        if p.suffix.lower() in IGNORE_EXT: continue
        text = read_text_safely(p)
        if not text: continue
        ch = content_hash(text)
        doc_id = short_hash(ch)
        _manifest[str(p)] = {"doc_id": doc_id, "hash": ch, "mtime": p.stat().st_mtime}
        docs.append({"path": str(p), "doc_id": doc_id, "hash": ch, "text": text})
    return docs

class _Evt(FileSystemEventHandler):
    def __init__(self, cfg):
        self.cfg = cfg
    def on_any_event(self, event):
        p = pathlib.Path(getattr(event, 'dest_path', event.src_path))
        if p.is_dir() or any(part in IGNORE_DIRS for part in p.parts): return
        if not p.exists(): return
        text = read_text_safely(p)
        if text is None: return
        ch = content_hash(text)
        m = _manifest.get(str(p))
        if m and m["hash"] == ch:
            m["mtime"] = p.stat().st_mtime
            return
        record = {"doc_id": short_hash(ch), "hash": ch, "mtime": p.stat().st_mtime}
        _manifest[str(p)] = record
        print(f"[watch] change: {p}")
        # auto-enqueue attribute jobs (doc-level) if configured
        attrs = self.cfg.get("attributes", {})
        if attrs.get("auto_enqueue", False):
            plugins = attrs.get("plugins", [])
            for plugin in plugins:
                _enqueue_job(self.cfg, plugin, record["doc_id"], payload={"path": str(p)})

def watch_changes(repo_path: pathlib.Path, cfg):
    ensure_dirs()
    # ensure jobs db exists if auto-enqueue is enabled
    attrs = cfg.get("attributes", {})
    if attrs.get("auto_enqueue", False):
        _ensure_jobs_db(cfg)
    obs = Observer()
    obs.schedule(_Evt(cfg), str(repo_path), recursive=True)
    obs.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        obs.stop()
    obs.join()