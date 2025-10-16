"""Client-facing helpers to submit requests to the LLM gateway."""

from __future__ import annotations

import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from kn.config import load_configs
from kn.utils.cache import get_cached_response, set_cached_response

from .errors import QueueTimeoutError, RequestRejectedError
from .models import GatewayRequest, GatewayResponse
from .storage import QueueStorage


def _storage(cfg: Optional[Dict[str, Any]] = None) -> QueueStorage:
    cfg = cfg or load_configs()
    path = cfg.get("llm_gateway", {}).get("queue_path")
    if path:
        return QueueStorage(Path(path))
    return QueueStorage(Path(cfg.get("_root", ".knowledge")) / "queues" / "llm_gateway.sqlite")


def submit_chat_request(
    prompt: str,
    max_tokens: int = 512,
    temperature: float = 0.2,
    overrides: Optional[Dict[str, Any]] = None,
    cache_key: Optional[str] = None,
    plugin_name: Optional[str] = None,
    cfg: Optional[Dict[str, Any]] = None,
    wait: bool = True,
) -> str:
    cfg = cfg or load_configs()
    merged_overrides = _merge_overrides(cfg, overrides, plugin_name)
    service_name = merged_overrides.get("service") or cfg.get("llm", {}).get("service", "lmstudio")
    model_name = merged_overrides.get("model") or cfg.get("llm", {}).get("model")
    max_tokens = int(merged_overrides.get("max_tokens", max_tokens))
    temperature = float(merged_overrides.get("temperature", temperature))

    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if "top_p" in merged_overrides:
        payload["top_p"] = merged_overrides["top_p"]

    local_models = os.getenv("LOCAL_MODELS_DIR")
    if local_models:
        payload.setdefault("extra_body", {})["local_models_dir"] = local_models

    ck = cache_key or f"{service_name}|{model_name}|{max_tokens}|{temperature}|{prompt}"
    cached = get_cached_response(ck)
    if cached is not None:
        return cached

    req_id = str(uuid.uuid4())
    metadata = {
        "cache_key": ck,
        "plugin": plugin_name,
    }
    request = GatewayRequest(request_id=req_id, service=service_name, model=model_name, payload=payload, metadata=metadata)
    storage = _storage(cfg)
    storage.enqueue(request)

    if not wait:
        return ""

    timeout_sec = float(cfg.get("llm_gateway", {}).get("wait_timeout_sec", 600))
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        response = storage.fetch_response(request.request_id)
        if response:
            return _handle_response(response, ck)
        time.sleep(0.2)
    raise QueueTimeoutError(f"LLM gateway timed out waiting for request {req_id}")


def submit_generic_request(service: str, payload: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None, cfg: Optional[Dict[str, Any]] = None, wait: bool = True, timeout_override: Optional[float] = None) -> GatewayResponse:
    cfg = cfg or load_configs()
    req_id = str(uuid.uuid4())
    storage = _storage(cfg)
    request = GatewayRequest(request_id=req_id, service=service, model=payload.get("model", ""), payload=payload, metadata=metadata or {})
    storage.enqueue(request)
    if not wait:
        return GatewayResponse(request_id=req_id, success=True, content=None, raw=None, error=None, latency_ms=None)
    timeout_sec = float(timeout_override if timeout_override is not None else cfg.get("llm_gateway", {}).get("wait_timeout_sec", 600))
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        response = storage.fetch_response(req_id)
        if response:
            return response
        time.sleep(0.2)
    raise QueueTimeoutError(f"Timeout waiting for generic gateway request {req_id}")


def _handle_response(response: GatewayResponse, cache_key: Optional[str]) -> str:
    if not response.success:
        raise RequestRejectedError(response.error or "Gateway request failed")
    content = ""
    if response.content:
        content = response.content
    elif response.raw:
        choices = (response.raw or {}).get("choices", [])
        if choices:
            content = choices[0].get("message", {}).get("content", "")
    if cache_key and content:
        set_cached_response(cache_key, content)
    return content


def _merge_overrides(cfg: Dict[str, Any], overrides: Optional[Dict[str, Any]], plugin_name: Optional[str]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    plugin_block = (cfg.get("plugins") or {}).get(plugin_name or "") if plugin_name else None
    if plugin_block and plugin_block.get("llm"):
        merged.update(plugin_block["llm"])
    if overrides:
        merged.update(overrides)
    return merged

