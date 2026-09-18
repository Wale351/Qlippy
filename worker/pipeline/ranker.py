"""
Stage 3 — Comparative Ranker
One strong-model call that orders all crafted clips side by side — LLMs are
far better at relative judgments than at absolute scores. Code then enforces
what an LLM can't be trusted with: temporal-overlap dedup, an archetype
diversity cap, and the user's max_clips limit. Falls back to checklist-score
ordering if the LLM call fails.
"""

from __future__ import annotations
import logging
from typing import Any

from .openrouter import call_agent

log = logging.getLogger(__name__)

MAX_OVERLAP_FRACTION = 0.4    # source-time overlap above this ⇒ duplicate moment
TEXT_PREVIEW_CHARS   = 600

SYSTEM_PROMPT = """\
You are the final editor for a batch of short-form clips cut from one video. \
Order them best-first by expected performance on Reels/TikTok/Shorts.

Judge each clip by:
- the scroll-stopping power of its FIRST line (clips are shown in play order)
- payoff strength: does it land a laugh, an insight, or a feeling
- self-containment for a total stranger
- variety: when two clips are close, prefer the one whose archetype is \
under-represented among your top picks{tone_hint}

Output strict JSON only:
{{"order": [<clip ids, best first, every id exactly once>]}}\
"""


def _overlap_fraction(a: list[dict], b: list[dict]) -> float:
    """Source-time overlap between two cut lists ÷ shorter clip's duration."""
    inter = 0.0
    for ca in a:
        for cb in b:
            inter += max(
                0.0, min(ca["end"], cb["end"]) - max(ca["start"], cb["start"])
            )
    dur_a = sum(c["end"] - c["start"] for c in a)
    dur_b = sum(c["end"] - c["start"] for c in b)
    shorter = min(dur_a, dur_b)
    return inter / shorter if shorter > 0 else 1.0


def _fallback_order(clips: list[dict]) -> list[int]:
    return sorted(
        range(len(clips)),
        key=lambda i: (clips[i]["score_total"], clips[i].get("scout_score", 0)),
        reverse=True,
    )


def _sanitize_order(raw: Any, n: int, clips: list[dict]) -> list[int]:
    """Keep valid unique ids from the LLM; append anything it forgot."""
    order: list[int] = []
    seen: set[int] = set()
    ids = raw.get("order") if isinstance(raw, dict) else raw
    for v in ids if isinstance(ids, list) else []:
        try:
            i = int(v)
        except (TypeError, ValueError):
            continue
        if 0 <= i < n and i not in seen:
            order.append(i)
            seen.add(i)
    for i in _fallback_order(clips):
        if i not in seen:
            order.append(i)
    return order


def _apply_constraints(
    order: list[int], clips: list[dict], max_clips: int,
) -> list[dict]:
    """Walk the preference order enforcing overlap dedup + archetype cap."""
    arch_cap = max(2, (max_clips + 1) // 2)

    def non_duplicate(i: int, kept: list[int]) -> bool:
        return all(
            _overlap_fraction(clips[i]["cuts_raw"], clips[k]["cuts_raw"])
            <= MAX_OVERLAP_FRACTION
            for k in kept
        )

    kept: list[int] = []
    arch_counts: dict[str, int] = {}
    deferred: list[int] = []
    for i in order:
        if len(kept) >= max_clips:
            break
        if not non_duplicate(i, kept):
            continue
        arch = clips[i]["archetype"]
        if arch_counts.get(arch, 0) >= arch_cap:
            deferred.append(i)   # diversity cap — reconsider only to fill spare slots
            continue
        kept.append(i)
        arch_counts[arch] = arch_counts.get(arch, 0) + 1

    # Fill remaining slots from capped-out leftovers (still no duplicates)
    for i in deferred:
        if len(kept) >= max_clips:
            break
        if non_duplicate(i, kept):
            kept.append(i)

    return [clips[i] for i in kept]


async def rank_clips(
    clips: list[dict[str, Any]],
    api_key: str,
    target_tones: list[str] | None = None,
    max_clips: int = 10,
) -> list[dict[str, Any]]:
    """Returns the final ordered clip list with `rank` set, capped at max_clips."""
    if not clips:
        return []

    # Local mode: skip the extra LLM ranking call.
    # Crafter/scout scores are already available, and the deterministic
    # ranking still applies overlap removal and archetype diversity.
    order = _fallback_order(clips)

    ranked = _apply_constraints(order, clips, max_clips)
    for rank, clip in enumerate(ranked, start=1):
        clip["rank"] = rank

    log.info("Ranker: %d clips → %d final (max %d)", len(clips), len(ranked), max_clips)
    return ranked
