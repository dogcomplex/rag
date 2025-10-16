"""Service agent registry and configuration loader."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from kn.config import load_configs

from .service_base import ServiceAgent
from .services_lmstudio import LmStudioAgent
from .errors import ServiceNotConfiguredError

_REGISTRY = {
    "lmstudio": LmStudioAgent,
}


def load_service_agent(name: str, cfg: Optional[Dict[str, Any]] = None) -> ServiceAgent:
    config = cfg or load_configs().get("llm_services", {}).get(name)
    if not config:
        raise ServiceNotConfiguredError(f"LLM service '{name}' not configured")
    provider = config.get("provider") or name
    cls = _REGISTRY.get(provider)
    if not cls:
        raise ServiceNotConfiguredError(f"No service agent registered for provider '{provider}'")
    logging.getLogger("llm_service").debug("Loading service agent %s (provider=%s)", name, provider)
    return cls(name=name, config=config)






