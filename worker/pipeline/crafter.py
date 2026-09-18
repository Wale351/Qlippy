"""
Stage 2 — Clip Crafter
One LLM call per scouted candidate. Sees the candidate span plus surrounding
context sentences and crafts the final clip boundaries as sentence groups in
PLAY ORDER — including the hook transplant: opening the clip with the payoff
line and playing the setup after it (sentence_groups out of chronological
order; the whole export path concats cuts in the given order).

Applies the must-have/signal checklist. The 5th must-have is "payoff" (laugh,
learning, or feeling) rather than the old "learning", so humor and emotion
clips can reach hero.
"""

from __future__ import annotations
import asyncio
import logging
from typing import Any

from .openrouter import call_agent
from .evaluator import classify
from .features import format_sentence_line

log = logging.getLogger(__name__)

MUST_HAVES = ["hook", "cold_start", "single_point", "clean_ending", "payoff"]
SIGNALS    = ["emotion", "quotable", "contrast", "energy", "curiosity_gap", "low_filler"]

# Hard duration gate (spec §3.1)
GATE_MIN_S = 6.0
GATE_MAX_S = 120.0

MAX_PARALLEL      = 8
CONTEXT_SENTENCES = 10   # sentences shown before/after the candidate span
MAX_GROUPS        = 4

ARCHETYPE_GUIDE = {
    "humor":     "keep the full setup, cut immediately after the punchline or laughter peak",
    "insight":   "open on the claim itself, proof and explanation after",
    "hot_take":  "open on the boldest line; the rant follows",
    "story":     "start where the stakes start, not where time starts; land on the payoff",
    "debate":    "keep both sides; never cut mid-rebuttal",
    "emotion":   "do not trim the pauses around the peak line — they are the content",
    "practical": "numbers/steps early; cut the throat-clearing",
}

SYSTEM_PROMPT = """\
You are a short-form video editor crafting ONE clip from a flagged moment in a \
podcast/video transcript. Choose the exact sentences the clip plays, in play order.

## The cold-start test (hard gate)
A total stranger must understand the clip with zero episode context and feel \
something. If no boundary choice passes this test, output {{"keep": false}}.

## Crafting rules
- Open as late as possible — the first line must hook. Cut all warm-up.
- End where the thought lands. Never trail into the next topic.
- You may drop tangent sentences in the middle (output multiple groups).
- HOOK TRANSPLANT: if the payoff line is buried, you may OPEN the clip with the \
payoff/quotable line and play the setup after it — list that group first even \
though it comes later in the source. Only do this when the line works cold AND \
the setup still makes sense heard after it. Set "transplant": true.
- This moment's archetype is {archetype}: {archetype_guide}.

## Checklist — tick honestly, on the clip exactly as you built it
must_haves:
- hook: the first line (in play order) stops a scrolling stranger
- cold_start: fully understandable with zero context
- single_point: one idea, story, or exchange — not three half-finished ones
- clean_ending: the thought lands fully
- payoff: the viewer laughs, learns something, or feels something real
signals:
- emotion: triggers curiosity, surprise, relatability, awe, or laughter
- quotable: a line someone would screenshot or repeat
- contrast: a reversal or counterintuitive angle
- energy: delivery is engaged — use the [energy spike]/[laughter] annotations
- curiosity_gap: satisfies but leaves one question open
- low_filler: minimal ums/uhs/false starts

## Duration
The clip must total {min_dur}-{max_dur} seconds — per-sentence durations are given.

Output strict JSON only:
{{"keep": true, "transplant": false, "sentence_groups": [[lo, hi], ...], \
"must_haves": {{"hook": false, "cold_start": false, "single_point": false, "clean_ending": false, "payoff": false}}, \
"signals": {{"emotion": false, "quotable": false, "contrast": false, "energy": false, "curiosity_gap": false, "low_filler": false}}, \
"reasoning": "<one line: boundary choices, and why transplant if used>"}}\
"""


def _resolve_groups(
    groups: list, sentences: list[dict], transplant: bool,
) -> tuple[list[list[int]], list[dict[str, float]], str] | None:
    """
    Validate sentence-index groups and resolve to play-order cuts + text.
    Groups may be non-chronological only when transplant is set; otherwise
    they are sorted into source order. Overlapping groups are rejected.
    """
    max_idx = len(sentences) - 1
    norm: list[list[int]] = []
    for group in groups or []:
        if not isinstance(group, (list, tuple)) or len(group) != 2:
            return None
        try:
            lo, hi = int(group[0]), int(group[1])
        except (TypeError, ValueError):
            return None
        # Repair reversed boundaries from smaller local models.
        if lo > hi:
            lo, hi = hi, lo

        lo, hi = max(0, lo), min(hi, max_idx)
        if lo > hi:
            continue

        if [lo, hi] not in norm:
            norm.append([lo, hi])

    if not norm:
        return None

    if not transplant:
        # Sort and merge overlapping/adjacent ranges rather than rejecting them.
        spans = sorted(norm)
        merged: list[list[int]] = []

        for lo, hi in spans:
            if merged and lo <= merged[-1][1] + 1:
                merged[-1][1] = max(merged[-1][1], hi)
            else:
                merged.append([lo, hi])

        norm = merged
    else:
        # Preserve hook-transplant play order, but overlapping cuts are invalid.
        spans = sorted(norm)
        for (alo, ahi), (blo, bhi) in zip(spans, spans[1:]):
            if blo <= ahi:
                return None

    if len(norm) > MAX_GROUPS:
        return None

    cuts: list[dict[str, float]] = []
    text_parts: list[str] = []
    for lo, hi in norm:
        group_sents = sentences[lo : hi + 1]
        cuts.append({
            "start": float(group_sents[0]["start"]),
            "end":   float(group_sents[-1]["end"]),
        })
        text_parts.append(" ".join(s["text"] for s in group_sents))
    return norm, cuts, " ".join(text_parts)


def _expand_short_clip(
    groups: list[list[int]],
    candidate: dict,
    sentences: list[dict],
    min_duration: float,
) -> tuple[list[list[int]], list[dict[str, float]], str] | None:
    """Expand a too-short single-group clip with nearby sentences.

    Prefer extending AFTER the selected hook first so the opening line stays intact.
    Only search within the candidate plus the normal Crafter context window.
    """
    if len(groups) != 1:
        return None

    lo, hi = groups[0]
    allowed_lo = max(0, candidate["lo"] - CONTEXT_SENTENCES)
    allowed_hi = min(len(sentences) - 1, candidate["hi"] + CONTEXT_SENTENCES)

    def duration(a: int, b: int) -> float:
        return float(sentences[b]["end"]) - float(sentences[a]["start"])

    # First extend forward to preserve the existing hook.
    while duration(lo, hi) < min_duration and hi < allowed_hi:
        hi += 1

    # If still short, add setup from immediately before.
    while duration(lo, hi) < min_duration and lo > allowed_lo:
        lo -= 1

    if duration(lo, hi) < min_duration:
        return None

    return _resolve_groups([[lo, hi]], sentences, False)


def _dominant_speaker(groups: list[list[int]], features: list[dict]) -> str | None:
    spks = [
        features[i]["speaker"]
        for lo, hi in groups
        for i in range(lo, hi + 1)
        if i < len(features) and features[i]["speaker"]
    ]
    return max(set(spks), key=spks.count) if spks else None


def _format_context(
    candidate: dict, sentences: list[dict], features: list[dict], roles: dict[str, str],
) -> str:
    lo = max(0, candidate["lo"] - CONTEXT_SENTENCES)
    hi = min(len(sentences) - 1, candidate["hi"] + CONTEXT_SENTENCES)
    lines = []
    for i in range(lo, hi + 1):
        marker = ">>> " if candidate["lo"] <= i <= candidate["hi"] else "    "
        lines.append(format_sentence_line(sentences[i], features[i], roles, marker))
    return "\n".join(lines)


def _fallback_clip(
    candidate: dict, sentences: list[dict], features: list[dict], reason: str,
) -> dict[str, Any] | None:
    """
    A rejected candidate still becomes a clip — on the Scout's own [lo, hi]
    span, labelled "weak" with the rejection reason — so nothing the Scout
    flagged is silently lost. The user sees why the Crafter passed on it and
    decides. Returns None only if the span itself is unusable.
    """
    resolved = _resolve_groups([[candidate["lo"], candidate["hi"]]], sentences, False)
    if resolved is None:
        log.info("Crafter: candidate %s dropped — scout span unusable", candidate["candidate_id"])
        return None
    groups, cuts_raw, text = resolved
    duration = sum(c["end"] - c["start"] for c in cuts_raw)
    log.info("Crafter: candidate %s kept as weak — %s", candidate["candidate_id"], reason)
    return {
        "archetype":       candidate["archetype"],
        "scout_score":     candidate["score"],
        "why":             candidate["why"],
        "label":           "weak",
        "score_total":     0,
        "must_haves":      0,
        "signals":         0,
        "must_have_ticks": {k: False for k in MUST_HAVES},
        "signal_ticks":    {k: False for k in SIGNALS},
        "sentence_groups": groups,
        "stitched":        False,
        "transplant":      False,
        "cuts_raw":        cuts_raw,
        "duration_sec":    round(duration, 3),
        "src_start":       min(c["start"] for c in cuts_raw),
        "speaker":         _dominant_speaker(groups, features),
        "text":            text,
        "reasoning":       reason,
    }


async def _craft_one(
    candidate: dict,
    sentences: list[dict],
    features: list[dict],
    roles: dict[str, str],
    api_key: str,
    sem: asyncio.Semaphore,
    gate_min: float,
    gate_max: float,
) -> dict[str, Any] | None:
    system = SYSTEM_PROMPT.format(
        archetype=candidate["archetype"],
        archetype_guide=ARCHETYPE_GUIDE[candidate["archetype"]],
        min_dur=int(gate_min),
        max_dur=int(gate_max),
    )
    user = (
        f"Moment [{candidate['archetype']}, scout score {candidate['score']}/100]: "
        f"{candidate['why']}\n"
        f"Payoff sentence: {candidate['payoff']}\n\n"
        f"Sentences (>>> marks the flagged span; unmarked lines are context you may use):\n"
        f"{_format_context(candidate, sentences, features, roles)}"
    )
    async with sem:
        data = await call_agent(
            "crafter", system, user,
            api_key=api_key, temperature=0.2, max_tokens=2048,
        )
    if not isinstance(data, dict) or not data.get("keep"):
        why = str((data or {}).get("reasoning") or "") if isinstance(data, dict) else ""
        return _fallback_clip(candidate, sentences, features,
                              "Crafter: doesn't stand alone without context" + (f" — {why}" if why else ""))

    transplant = bool(data.get("transplant"))
    resolved = _resolve_groups(data.get("sentence_groups"), sentences, transplant)
    if resolved is None:
        return _fallback_clip(candidate, sentences, features, "Crafter returned unusable boundaries")
    groups, cuts_raw, text = resolved

    mh_ticks  = {k: bool((data.get("must_haves") or {}).get(k)) for k in MUST_HAVES}
    sig_ticks = {k: bool((data.get("signals") or {}).get(k)) for k in SIGNALS}
    mh_count, sig_count = sum(mh_ticks.values()), sum(sig_ticks.values())

    label = classify(mh_count, sig_count)

    # Local-model calibration:
    # Hero should be rare. Require both a perfect Crafter checklist
    # and a very strong Scout score.
    if label == "hero" and candidate.get("score", 0) < 85:
        label = "strong"

    if label is None:
        return _fallback_clip(candidate, sentences, features,
                              f"Crafter: only {mh_count} of {len(MUST_HAVES)} must-haves")

    duration = sum(c["end"] - c["start"] for c in cuts_raw)
    was_expanded = False

    # Small local models sometimes identify the right moment but trim it down
    # to only a few seconds. Expand with nearby context instead of immediately
    # demoting an otherwise-good clip.
    if duration < gate_min and not transplant:
        expanded = _expand_short_clip(
            groups, candidate, sentences, gate_min
        )
        if expanded is not None:
            groups, cuts_raw, text = expanded
            duration = sum(c["end"] - c["start"] for c in cuts_raw)
            was_expanded = True
            log.info(
                "Crafter: candidate %s expanded to %.1fs",
                candidate["candidate_id"], duration,
            )

    if not (gate_min <= duration <= gate_max):
        return _fallback_clip(candidate, sentences, features,
                              f"Crafter: {duration:.1f}s is outside your {int(gate_min)}–{int(gate_max)}s range")

    # Expansion changed the exact clip after the LLM checklist was produced,
    # so do not call the modified version Hero without re-evaluating it.
    if was_expanded and label == "hero":
        label = "strong"

    # Transplant only counts when the play order actually differs from source order
    transplant = transplant and groups != sorted(groups)

    # Inspect the ACTUAL first sentence of the finished clip.
    first_idx = groups[0][0] if groups else None
    starts_with_host_question = False

    if first_idx is not None and 0 <= first_idx < len(sentences):
        first_feat = features[first_idx] if first_idx < len(features) else {}
        first_speaker = first_feat.get("speaker")
        first_text = sentences[first_idx].get("text", "").strip()

        starts_with_host_question = (
            roles.get(first_speaker) == "host"
            and first_text.endswith("?")
        )

    # A host question may be useful setup, but it should not receive Hero status.
    if starts_with_host_question and label == "hero":
        label = "strong"

    return {
        "archetype":       candidate["archetype"],
        "scout_score":     candidate["score"],
        "why":             candidate["why"],
        "label":           label,
        "score_total":     mh_count + sig_count,
        "must_haves":      mh_count,
        "signals":         sig_count,
        "must_have_ticks": mh_ticks,
        "signal_ticks":    sig_ticks,
        "sentence_groups": groups,
        "stitched":        len(groups) > 1,
        "transplant":      transplant,
        "host_question_open": starts_with_host_question,
        "cuts_raw":        cuts_raw,
        "duration_sec":    round(duration, 3),
        "src_start":       min(c["start"] for c in cuts_raw),
        "speaker":         _dominant_speaker(groups, features),
        "text":            text,
        "reasoning":       str(data.get("reasoning", "")),
    }


async def craft_clips(
    candidates: list[dict[str, Any]],
    sentences: list[dict[str, Any]],
    features: list[dict[str, Any]],
    roles: dict[str, str],
    api_key: str,
    min_duration_s: float | None = None,
    max_duration_s: float | None = None,
) -> list[dict[str, Any]]:
    """Craft one clip per candidate. Rejections and failures become "weak" clips
    on the Scout's span rather than disappearing — see _fallback_clip."""
    if not candidates:
        return []
    gate_min = max(min_duration_s or GATE_MIN_S, GATE_MIN_S)
    gate_max = min(max_duration_s or GATE_MAX_S, GATE_MAX_S)
    sem = asyncio.Semaphore(MAX_PARALLEL)

    results = await asyncio.gather(
        *(
            _craft_one(c, sentences, features, roles, api_key, sem, gate_min, gate_max)
            for c in candidates
        ),
        return_exceptions=True,
    )

    clips: list[dict[str, Any]] = []
    for cand, result in zip(candidates, results):
        if isinstance(result, Exception):
            log.warning("Crafter failed for candidate %s: %s", cand["candidate_id"], result)
            result = _fallback_clip(cand, sentences, features, "Crafter call failed")
        if result is not None:
            clips.append(result)

    log.info("Crafter: %d candidates → %d clips (%d transplants)",
             len(candidates), len(clips), sum(1 for c in clips if c["transplant"]))
    return clips
