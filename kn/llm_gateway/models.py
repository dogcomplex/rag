"""Data models for LLM gateway requests and responses."""

from __future__ import annotations

import dataclasses
import time
from typing import Any, Dict, Optional


@dataclasses.dataclass(slots=True)
class GatewayRequest:
    """A unit of work destined for an LLM service agent."""

    request_id: str
    service: str
    model: str
    payload: Dict[str, Any]
    metadata: Dict[str, Any]
    created_ts: float = dataclasses.field(default_factory=lambda: time.time())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "service": self.service,
            "model": self.model,
            "payload": json.loads(json.dumps(self.payload, ensure_ascii=False, default=str)),
            "metadata": json.loads(json.dumps(self.metadata, ensure_ascii=False, default=str)),
            "created_ts": self.created_ts,
        }

    @classmethod
    def from_row(cls, row: Dict[str, Any]) -> "GatewayRequest":
        return cls(
            request_id=row["request_id"],
            service=row["service"],
            model=row["model"],
            payload=row["payload"],
            metadata=row.get("metadata", {}) or {},
            created_ts=row.get("created_ts", time.time()),
        )


@dataclasses.dataclass(slots=True)
class GatewayResponse:
    """Response emitted by an LLM service agent."""

    request_id: str
    success: bool
    content: Optional[str]
    raw: Optional[Dict[str, Any]]
    error: Optional[str]
    latency_ms: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "success": self.success,
            "content": self.content,
            "raw": self.raw,
            "error": self.error,
            "latency_ms": self.latency_ms,
        }






