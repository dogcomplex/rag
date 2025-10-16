"""Gateway daemon that drains the queue and forwards to service agents."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, Optional

from kn.config import load_configs

from .errors import GatewayError
from .models import GatewayRequest, GatewayResponse
from .services import load_service_agent
from .storage import QueueStorage

logger = logging.getLogger("llm_gateway.runner")


def run_gateway_service(service_name: str, loop_forever: bool = True, cfg: Optional[Dict] = None):
    cfg = cfg or load_configs()
    gateway_cfg = cfg.get("llm_gateway", {})
    poll_interval_ms = int(gateway_cfg.get("poll_interval_ms", 200))
    storage = _load_storage(gateway_cfg, cfg)
    agent = load_service_agent(service_name, cfg.get("llm_services", {}).get(service_name))

    logger.info("Gateway service '%s' starting (poll=%sms)", service_name, poll_interval_ms)
    while True:
        req = storage.dequeue_pending(service_name)
        if not req:
            if not loop_forever:
                break
            time.sleep(poll_interval_ms / 1000.0)
            if agent.should_unload_idle():
                logger.info("Idle timeout reached; unloading model")
                agent.unload_current_model()
            continue

        logger.info("Processing request %s model=%s", req.request_id, req.model)
        try:
            response = _handle_request(agent, req)
        except Exception as exc:  # pylint: disable=broad-except
            logger.exception("Gateway request failed: %s", exc)
            storage.mark_failed(req.request_id, str(exc))
        else:
            storage.store_response(response)

        if not loop_forever:
            break


def _handle_request(agent, req: GatewayRequest) -> GatewayResponse:
    payload = req.payload or {}
    action = payload.get("action")
    if action == "health":
        result = agent.health()
        ok = bool(result.get("ok", True)) if isinstance(result, dict) else True
        return GatewayResponse(
            request_id=req.request_id,
            success=ok,
            content=None,
            raw=result if isinstance(result, dict) else {"result": result},
            error=None if ok else (result.get("error") if isinstance(result, dict) else "health check failed"),
            latency_ms=None,
        )

    agent.guard_rate_limits()
    agent.ensure_model_loaded(req.model)
    payload = req.payload

    max_context = int(agent.config.get("max_context_tokens", 8192))
    if _estimate_prompt_tokens(payload) > max_context:
        raise GatewayError(f"Context too large for model {req.model}")

    result = agent.invoke(payload)
    latency_ms = result.pop("_latency_ms", None)
    content = None
    if result.get("choices"):
        content = result["choices"][0].get("message", {}).get("content")
    return GatewayResponse(
        request_id=req.request_id,
        success=True,
        content=content,
        raw=result,
        error=None,
        latency_ms=latency_ms,
    )


def _estimate_prompt_tokens(payload: Dict) -> int:
    messages = payload.get("messages", [])
    total_chars = sum(len(m.get("content", "")) for m in messages)
    return int(total_chars / 3.5) + 1


def _load_storage(gateway_cfg: Dict, cfg: Dict) -> QueueStorage:
    queue_path = gateway_cfg.get("queue_path")
    if queue_path:
        return QueueStorage(Path(queue_path))
    root = Path(cfg.get("_root", ".knowledge"))
    return QueueStorage(root / "queues" / "llm_gateway.sqlite")






