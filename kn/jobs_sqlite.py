import sqlite3, json, pathlib
JOB_ATTR_PREFIX = "attribute:"
def _db_path(cfg):
    return pathlib.Path(cfg["stores"]["jobs"]["path"])
def ensure_db(cfg):
    p = _db_path(cfg); p.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(p)
    con.execute("""
    CREATE TABLE IF NOT EXISTS jobs(
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      plugin TEXT NOT NULL,
      doc_id TEXT NOT NULL,
      payload TEXT,
      status TEXT DEFAULT 'pending',
      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    con.commit(); con.close()
def enqueue(cfg, plugin: str, doc_id: str, payload: dict|None=None):
    con = sqlite3.connect(_db_path(cfg))
    con.execute("INSERT INTO jobs(plugin, doc_id, payload) VALUES (?,?,?)",
                (plugin, doc_id, json.dumps(payload or {})))
    con.commit(); con.close()
def dequeue_batch(cfg, wanted_plugins, limit=16):
    con = sqlite3.connect(_db_path(cfg)); cur = con.cursor()
    qmarks = ",".join(["?"]*len(wanted_plugins))
    cur.execute(f"""
      SELECT id, plugin, doc_id, payload FROM jobs
      WHERE status='pending' AND plugin IN ({qmarks})
      ORDER BY id ASC LIMIT ?
    """, (*wanted_plugins, limit))
    rows = cur.fetchall(); ids = [r[0] for r in rows]
    if ids:
      cur.execute(f"UPDATE jobs SET status='running' WHERE id IN ({','.join(['?']*len(ids))})", ids)
    con.commit(); con.close()
    return [{"id": r[0], "plugin": r[1], "doc_id": r[2], "payload": json.loads(r[3] or '{}')} for r in rows]
def ack_job(cfg, job_id: int):
    con = sqlite3.connect(_db_path(cfg)); con.execute("UPDATE jobs SET status='done' WHERE id=?", (job_id,))
    con.commit(); con.close()
def iter_docs_for_jobs(jobs):
    from pathlib import Path
    cdir = Path('.knowledge/indexes/chunks'); out = {}
    for j in jobs:
        for p in sorted(cdir.glob(f"{j['doc_id']}-*.json")):
            out[j['doc_id']] = json.loads(p.read_text(encoding='utf-8')); break
    return out