"""LM Studio-specific service agent implementation."""

from __future__ import annotations

import logging
import time
from typing import Any, Dict

import requests

from .service_base import ServiceAgent


class LmStudioAgent(ServiceAgent):
    """Agent that serializes requests to a single LM Studio backend."""

    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        self.session = requests.Session()
        self.base_url = config.get("base_url", "http://127.0.0.1:12345/v1").rstrip("/")
        self.api_key = config.get("api_key", "lm-studio")
        self.timeout = int(config.get("timeout", 120))
        self.manage_models = bool(config.get("manage_models"))

    # ------------------------------------------------------------------
    def ensure_model_loaded(self, model: str):
        if not self.manage_models:
            if self.current_model() != model:
                self.logger.debug("Skipping explicit load for %s (manage_models disabled)", model)
                self.update_model_tracking(model)
            return
        force_reload = bool(self.config.get("force_model_reload"))
        if self.current_model() == model and not force_reload:
            return
        if self.config.get("unload_on_switch", True) and self.current_model() and self.current_model() != model:
            self.unload_current_model()
        payload = {"model": model}
        response = self.session.post(
            f"{self.base_url}/models/load",
            headers=self._headers(),
            json=payload,
            timeout=self.timeout,
        )
        if response.status_code != 200:
            response.raise_for_status()
        self.logger.info("Loaded model %s", model)
        self.update_model_tracking(model)

    def invoke(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        request_started = time.time()
        response = self.session.post(
            f"{self.base_url}/chat/completions",
            headers=self._headers(),
            json=payload,
            timeout=self.timeout,
        )
        latency_ms = (time.time() - request_started) * 1000.0
        response.raise_for_status()
        data = response.json()
        data.setdefault("_latency_ms", latency_ms)
        return data

    def unload_current_model(self):
        if not self.manage_models:
            self.logger.debug("Skipping explicit unload (manage_models disabled)")
            self.update_model_tracking(model="")
            return
        model = self.current_model()
        if not model:
            return
        try:
            response = self.session.post(
                f"{self.base_url}/models/unload",
                headers=self._headers(),
                json={"model": model},
                timeout=self.timeout,
            )
            if response.status_code == 200:
                self.logger.info("Unloaded model %s", model)
            else:
                self.logger.warning("Failed to unload model %s: %s", model, response.text)
        finally:
            self.update_model_tracking(model="")

    def health(self) -> Dict[str, Any]:
        try:
            response = self.session.get(
                f"{self.base_url}/models",
                headers=self._headers(),
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
            return {
                "ok": True,
                "available_models": [m.get("id") for m in data.get("data", [])],
            }
        except Exception as exc:  # pylint: disable=broad-except
            return {"ok": False, "error": str(exc)}

    # ------------------------------------------------------------------
    def _headers(self) -> Dict[str, str]:
        headers = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers






