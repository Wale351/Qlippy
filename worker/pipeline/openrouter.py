"""
Shared OpenRouter client for the clip pipeline agents.
Each agent has a primary model and a fallback — the fallback is tried
automatically when the primary fails. All agents return strict JSON.
"""

from __future__ import annotations
import json
import logging
import os
import re
from typing import Any

import httpx

log = logging.getLogger(__name__)

OPENROUTER_BASE = os.getenv("LLM_BASE_URL", "http://127.0.0.1:11434/v1")

# Per-agent model routing (spec §9) — env-overridable, OpenRouter slugs
# Budget shape: cheap model for the high-fanout scout, mid model for per-clip
# crafting, strongest model for the single comparative-ranking call.
AGENT_MODELS: dict[str, tuple[str, str]] = {
    "scout": ("qwen3:4b", "qwen3:4b"),
    "crafter": ("qwen3:4b", "qwen3:4b"),
    "moment_builder": ("qwen3:4b", "qwen3:4b"),
    "ranker": ("qwen3:4b", "qwen3:4b"),
    "reader": ("qwen3:4b", "qwen3:4b"),
    "tension": ("qwen3:4b", "qwen3:4b"),
    "knowledge_gap": ("qwen3:4b", "qwen3:4b"),
    "evaluator": ("qwen3:4b", "qwen3:4b"),
    "hook_writer": ("qwen3:4b", "qwen3:4b"),
}


def parse_json_response(raw: str) -> Any:
    """Parse LLM output as JSON, stripping markdown fences if present."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-z]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
    return json.loads(raw)


async def _call_model(
    model: str,
    system: str,
    user: str,
    api_key: str,
    temperature: float,
    max_tokens: int,
) -> str:
    body = {
        "model": model,
        "stream": False,
        "think": False,
        "format": "json",
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
            "num_ctx": 8192,
        },
    }

    async with httpx.AsyncClient(timeout=300.0) as client:
        resp = await client.post(
            "http://127.0.0.1:11434/api/chat",
            json=body,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]


async def call_agent(
    agent: str,
    system: str,
    user: str,
    api_key: str,
    temperature: float = 0.3,
    max_tokens: int = 4096,
) -> Any:
    """
    Call an agent's primary model; on any failure (HTTP, JSON), retry once
    on the primary, then try the fallback model. Returns parsed JSON.
    """
    primary, fallback = AGENT_MODELS[agent]
    last_error: Exception | None = None
    for model in (primary, primary, fallback):
        try:
            raw = await _call_model(model, system, user, api_key, temperature, max_tokens)
            return parse_json_response(raw)
        except Exception as e:
            last_error = e
            log.warning("Agent %s call failed on %s: %s", agent, model, e)
    raise RuntimeError(f"Agent {agent} failed on primary and fallback: {last_error}")
