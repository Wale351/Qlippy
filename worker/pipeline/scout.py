"""
Stage 1 — Moment Scout
One LLM call per ~20-minute chunk (pipeline.chunking) over the annotated
transcript. Finds candidate clip moments across seven archetypes with a
0-100 score that is *relative to the chunk* — comparative judgment instead
of the old per-segment binary tension gate. Replaces Reader + Tension
Detector + Knowledge Gap Scorer.
"""

from __future__ import annotations
import asyncio
import logging
from typing import Any

from .openrouter import call_agent
from .chunking import chunk_sentence_ranges
from .features import format_sentence_line

log = logging.getLogger(__name__)

ARCHETYPES = {
    "insight", "story", "humor", "debate", "emotion", "hot_take", "practical",
}

MAX_PARALLEL_CHUNKS = 4
TOP_K               = 10   # candidates passed on to the crafter
MAX_SPAN_SENTENCES  = 80   # over-long spans are trimmed around the payoff

SYSTEM_PROMPT = """\
You scout a podcast/video transcript for clip-worthy moments — the spans a \
short-form editor would pull for Reels, TikTok, or Shorts.

You receive numbered sentences with speaker IDs and inline delivery annotations: \
[laughter] = the room laughed right after the sentence, [energy spike] = unusually \
energetic delivery, [long pause before] = the speaker paused before saying it, \
[rapid exchange] = fast back-and-forth between speakers. "host" marks the likely \
interviewer — guest speech is usually richer clip material than host questions.

Find every moment a scrolling stranger would stop for. Type each one:
- insight — a claim or explanation that contradicts assumptions or reframes something familiar
- story — a self-contained anecdote with stakes and a payoff
- humor — a joke, roast, or absurd moment; [laughter] right after a sentence is strong evidence
- debate — disagreement, pushback, a heated or rapid exchange
- emotion — vulnerability, anger, awe, grief; the feeling itself is the content
- hot_take — a bold, quotable, controversial claim or rant
- practical — concrete how-to, numbers, or steps a viewer can use immediately

For each moment output:
- "archetype": one of the types above
- "payoff": the single sentence index that IS the moment — the punchline, key claim, or peak
- "lo"/"hi": the span including minimal setup before and landing after the payoff
- "score": 0-100, how strongly this would perform RELATIVE TO THE OTHER MOMENTS IN \
THIS TRANSCRIPT. Use the full range and be honest — most content is filler.
- "why": one line naming what makes it work

Bias toward recall: include borderline moments — downstream stages cut hard. \
Moments may overlap when a span genuinely contains two different moments.
{tone_hint}
Output strict JSON only:
{{"moments": [{{"archetype": "...", "score": 0, "payoff": 0, "lo": 0, "hi": 0, "why": "..."}}]}}\
"""


def _tone_hint(context: dict[str, Any]) -> str:
    parts = []
    if context.get("niche"):
        parts.append(f"Creator niche: {context['niche']}.")
    if context.get("audience"):
        parts.append(f"Audience: {context['audience']}.")
    tones = context.get("target_tones") or []
    if tones:
        parts.append(
            f"The creator prefers {', '.join(tones)} clips — weight matching "
            "archetypes up, but never suppress an exceptional moment of another type."
        )
    return ("\n" + " ".join(parts) + "\n") if parts else ""


def _validate_moments(
    raw: Any, lo0: int, hi0: int,
) -> list[dict[str, Any]]:
    """Clamp/spec-check one chunk's moments; drop anything malformed."""
    moments = raw.get("moments", []) if isinstance(raw, dict) else raw
    out: list[dict[str, Any]] = []
    for m in moments or []:
        try:
            archetype = str(m["archetype"])
            score     = int(m["score"])
            lo, hi    = int(m["lo"]), int(m["hi"])
            payoff    = int(m.get("payoff", lo))
        except (KeyError, TypeError, ValueError):
            continue
        if archetype not in ARCHETYPES:
            continue
        lo, hi = max(lo, lo0), min(hi, hi0)
        if lo > hi:
            continue
        payoff = min(max(payoff, lo), hi)
        if hi - lo + 1 > MAX_SPAN_SENTENCES:
            # Model went wide — re-centre the span on the payoff
            half = MAX_SPAN_SENTENCES // 2
            lo = max(lo, payoff - half)
            hi = min(hi, payoff + half)
        out.append({
            "archetype": archetype,
            "score":     min(max(score, 0), 100),
            "payoff":    payoff,
            "lo":        lo,
            "hi":        hi,
            "why":       str(m.get("why", ""))[:300],
        })
    return out


async def scout_moments(
    sentences: list[dict[str, Any]],
    features: list[dict[str, Any]],
    roles: dict[str, str],
    api_key: str,
    context: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """
    Returns up to TOP_K candidates sorted by score desc:
    {"candidate_id", "archetype", "score", "payoff", "lo", "hi", "why",
     "start", "end"}
    """
    if not sentences:
        return []
    context = context or {}
    system = SYSTEM_PROMPT.format(tone_hint=_tone_hint(context))
    title = context.get("title") or ""
    chunks = chunk_sentence_ranges(sentences)
    sem = asyncio.Semaphore(MAX_PARALLEL_CHUNKS)

    async def _one(lo0: int, hi0: int) -> list[dict[str, Any]]:
        numbered = "\n".join(
            format_sentence_line(sentences[i], features[i], roles)
            for i in range(lo0, hi0 + 1)
        )
        user = (f"Video title: {title}\n\n" if title else "") + f"Sentences:\n{numbered}"
        async with sem:
            try:
                data = await call_agent(
                    "scout", system, user,
                    api_key=api_key, temperature=0.3, max_tokens=2048,
                )
            except Exception as e:
                # One bad chunk must not sink a long job — skip it and keep going
                log.warning("Scout chunk %d-%d failed: %s", lo0, hi0, e)
                return []
        return _validate_moments(data, lo0, hi0)

    results = await asyncio.gather(*[_one(lo0, hi0) for lo0, hi0 in chunks])

    seen: set[tuple[int, int, str]] = set()
    merged: list[dict[str, Any]] = []
    for chunk_moments in results:
        for m in chunk_moments:
            key = (m["lo"], m["hi"], m["archetype"])
            if key in seen:
                continue
            seen.add(key)
            merged.append(m)

    # Local quality filter:
    # interviewer questions are useful context, but usually poor clip payoffs.
    # Demote them instead of deleting them completely.
    for m in merged:
        payoff = m.get("payoff")
        if isinstance(payoff, int) and 0 <= payoff < len(sentences):
            feat = features[payoff] if payoff < len(features) else {}
            speaker = feat.get("speaker")
            payoff_text = sentences[payoff].get("text", "").strip()

            if roles.get(speaker) == "host" and payoff_text.endswith("?"):
                m["score"] = max(0, m["score"] - 35)

    merged.sort(key=lambda m: m["score"], reverse=True)
    del merged[TOP_K:]
    for cid, m in enumerate(merged):
        m["candidate_id"] = cid
        m["start"] = sentences[m["lo"]]["start"]
        m["end"]   = sentences[m["hi"]]["end"]

    by_type = {a: sum(1 for m in merged if m["archetype"] == a) for a in ARCHETYPES}
    log.info(
        "Scout: %d sentences → %d candidates %s",
        len(sentences), len(merged),
        {k: v for k, v in by_type.items() if v},
    )
    return merged
