"""Persistence helpers for the LLM gateway SQLite queue."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

from .models import GatewayRequest, GatewayResponse

_SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id TEXT UNIQUE,
    service TEXT NOT NULL,
    model TEXT NOT NULL,
    payload TEXT NOT NULL,
    metadata TEXT,
    created_ts REAL NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending'
);

CREATE TABLE IF NOT EXISTS responses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id TEXT UNIQUE,
    success INTEGER NOT NULL,
    content TEXT,
    raw TEXT,
    error TEXT,
    latency_ms REAL,
    created_ts REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_requests_status ON requests(status);
CREATE INDEX IF NOT EXISTS idx_requests_service ON requests(service);
"""


class QueueStorage:
    """Thin SQLite wrapper for storing gateway requests and responses."""

    def __init__(self, db_path: Path, busy_timeout_ms: int = 5000):
        self.db_path = Path(db_path)
        self._lock = threading.Lock()
        self._busy_timeout_ms = busy_timeout_ms
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(str(self.db_path))
        con.execute(f"PRAGMA busy_timeout = {self._busy_timeout_ms}")
        return con

    def _ensure_schema(self):
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as con:
            con.executescript(_SCHEMA)
            con.commit()

    def enqueue(self, req: GatewayRequest):
        payload_json = _safe_json(req.payload)
        metadata_json = _safe_json(req.metadata or {})
        with self._connect() as con:
            con.execute(
                "INSERT OR REPLACE INTO requests(request_id, service, model, payload, metadata, created_ts, status)"
                " VALUES(?,?,?,?,?,?, 'pending')",
                (
                    req.request_id,
                    req.service,
                    req.model,
                    payload_json,
                    metadata_json,
                    req.created_ts,
                ),
            )
            con.commit()

    def dequeue_pending(self, service: str) -> Optional[GatewayRequest]:
        with self._lock:
            with self._connect() as con:
                cur = con.execute(
                    "SELECT request_id, service, model, payload, metadata, created_ts FROM requests"
                    " WHERE status='pending' AND service=? ORDER BY id LIMIT 1",
                    (service,),
                )
                row = cur.fetchone()
                if not row:
                    return None
                con.execute("UPDATE requests SET status='running' WHERE request_id=?", (row[0],))
                con.commit()
        payload = json.loads(row[3]) if row[3] else {}
        metadata = json.loads(row[4]) if row[4] else {}
        return GatewayRequest(
            request_id=row[0],
            service=row[1],
            model=row[2],
            payload=payload,
            metadata=metadata,
            created_ts=row[5],
        )

    def store_response(self, response: GatewayResponse):
        content = _safe_text(response.content)
        raw_json = _safe_json(response.raw) if response.raw is not None else None
        error_text = _safe_text(response.error)
        with self._connect() as con:
            con.execute(
                "INSERT OR REPLACE INTO responses(request_id, success, content, raw, error, latency_ms, created_ts)"
                " VALUES(?,?,?,?,?,?,?)",
                (
                    response.request_id,
                    1 if response.success else 0,
                    content,
                    raw_json,
                    error_text,
                    response.latency_ms,
                    time.time(),
                ),
            )
            con.execute("UPDATE requests SET status=? WHERE request_id=?", ("done" if response.success else "failed", response.request_id))
            con.commit()

    def fetch_response(self, request_id: str) -> Optional[GatewayResponse]:
        with self._connect() as con:
            cur = con.execute(
                "SELECT success, content, raw, error, latency_ms FROM responses WHERE request_id=?",
                (request_id,),
            )
            row = cur.fetchone()
            if not row:
                return None
        raw = json.loads(row[2]) if row[2] else None
        return GatewayResponse(
            request_id=request_id,
            success=bool(row[0]),
            content=row[1],
            raw=raw,
            error=row[3],
            latency_ms=row[4],
        )

    def mark_failed(self, request_id: str, error_message: str):
        resp = GatewayResponse(
            request_id=request_id,
            success=False,
            content=None,
            raw=None,
            error=error_message,
            latency_ms=None,
        )
        self.store_response(resp)

    def purge_stale(self, max_age_sec: float):
        cutoff = time.time() - max_age_sec
        with self._connect() as con:
            con.execute("DELETE FROM requests WHERE status IN ('done','failed') AND created_ts<?", (cutoff,))
            con.execute("DELETE FROM responses WHERE created_ts<?", (cutoff,))
            con.commit()

    def stats(self, service: Optional[str] = None) -> Dict[str, Any]:
        where = ""
        params = ()
        if service:
            where = " WHERE service=?"
            params = (service,)
        with self._connect() as con:
            pending = con.execute(f"SELECT COUNT(*) FROM requests WHERE status='pending'" + (" AND service=?" if service else ""), params).fetchone()[0]
            running = con.execute(f"SELECT COUNT(*) FROM requests WHERE status='running'" + (" AND service=?" if service else ""), params).fetchone()[0]
        return {"pending": pending, "running": running}


def _safe_json(value: Any) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False)
    except Exception:
        text = json.dumps(str(value), ensure_ascii=False)
    return _safe_text(text)


def _safe_text(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str):
        value = str(value)
    return value.encode('utf-8', errors='replace').decode('utf-8')






