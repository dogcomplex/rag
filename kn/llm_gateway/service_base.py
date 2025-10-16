"""Base classes and utilities for LLM service agents."""

from __future__ import annotations

import abc
import logging
import threading
import time
from typing import Any, Dict, Optional


class ServiceAgent(abc.ABC):
    """Abstract base for a throttled, single-backend LLM agent."""

    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.logger = logging.getLogger(f"llm_service.{name}")
        self._lock = threading.Lock()
        self._last_request_ts: Optional[float] = None
        self._current_model: Optional[str] = None
        self._last_model_switch: Optional[float] = None

    # ------------------------------------------------------------------
    # Abstract API
    # ------------------------------------------------------------------
    @abc.abstractmethod
    def ensure_model_loaded(self, model: str):
        """Make sure the requested model is loaded and ready."""

    @abc.abstractmethod
    def invoke(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Perform the actual request against the backend."""

    @abc.abstractmethod
    def unload_current_model(self):
        """Attempt to unload the currently loaded model if supported."""

    @abc.abstractmethod
    def health(self) -> Dict[str, Any]:
        """Return backend-specific health information."""

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------
    def guard_rate_limits(self):
        spacing_ms = int(self.config.get("min_request_spacing_ms", 0))
        if spacing_ms <= 0:
            return
        with self._lock:
            if self._last_request_ts is None:
                self._last_request_ts = time.time()
                return
            elapsed_ms = (time.time() - self._last_request_ts) * 1000.0
            if elapsed_ms < spacing_ms:
                delay = (spacing_ms - elapsed_ms) / 1000.0
                if delay > 0:
                    time.sleep(delay)
            self._last_request_ts = time.time()

    def should_unload_idle(self) -> bool:
        idle_sec = int(self.config.get("idle_unload_sec", 0) or 0)
        if idle_sec <= 0:
            return False
        if self._last_request_ts is None:
            return False
        return (time.time() - self._last_request_ts) >= idle_sec

    def update_model_tracking(self, model: str):
        if model != self._current_model:
            self._current_model = model
            self._last_model_switch = time.time()

    def current_model(self) -> Optional[str]:
        return self._current_model

    def model_switch_info(self) -> Optional[float]:
        return self._last_model_switch






