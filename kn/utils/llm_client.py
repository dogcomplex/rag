"""Backward-compatible LLM client that delegates to the gateway."""

from __future__ import annotations

from typing import Any, Dict, Optional

from kn.llm_gateway.client import submit_chat_request


def chat(
    prompt: str,
    max_tokens: int = 512,
    temperature: float = 0.2,
    overrides: Optional[Dict[str, Any]] = None,
    cache_key: Optional[str] = None,
    plugin_name: Optional[str] = None,
    force: bool = False,
    overwrite: bool = False,
) -> str:
    """Proxy chat helper used by legacy code paths."""

    return submit_chat_request(
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        overrides=overrides,
        cache_key=cache_key,
        plugin_name=plugin_name,
        force=force,
        overwrite=overwrite,
    )






