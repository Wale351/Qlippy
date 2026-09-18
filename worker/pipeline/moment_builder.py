"""
Moment Builder

Takes the short scenes produced by the Crafter and groups related scenes into
complete post-ready Moments.

A Moment should feel like one coherent topic/story rather than a collection
of isolated highlights.

Preferred duration: 45-90 seconds.
The number of Moments scales with source length, but quality always wins over
hitting a quota.
"""

from __future__ import annotations

import logging
import math
from typing import Any

from .openrouter import call_agent

log = logging.getLogger(__name__)

TARGET_MIN_S = 45.0
TARGET_MAX_S = 90.0
ABSOLUTE_MAX_S = 105.0


SYSTEM_PROMPT = """\
You are assembling short-form social videos from scenes already selected from
a podcast, interview, or streamer video.

Your job is NOT to find new scenes.
Your job is to GROUP the supplied scenes into complete, coherent MOMENTS.

A Moment should:
- focus on ONE clear topic, story, argument, or conversation
- strongly prefer scenes with matching or closely related Topic labels
- prefer high payoff_strength and low filler_risk
- prefer guest_led scenes when they contain the real substance
- use low-standalone scenes only when another selected scene supplies the missing context
- have a natural beginning, development, and payoff
- make sense to a viewer who has not watched the original video
- preferably total 45-90 seconds
- never include unrelated scenes just to reach a duration target
- keep scenes in chronological source order unless changing order is absolutely
  necessary for comprehension
- avoid creating multiple Moments that say basically the same thing
- prefer guest/personality material over generic interviewer setup
- preserve interviewer questions only when they are genuinely necessary context

The number of Moments depends on the material.
Do NOT force a quota.

For roughly:
- 10 minutes of source: often 1-3 strong Moments
- 20 minutes: often 2-5
- 30 minutes: often 3-6
but fewer is completely fine when the material does not support more.

Each supplied scene has an integer id.

Output strict JSON only:

{
  "moments": [
    {
      "scene_ids": [0, 2, 3],
      "topic": "short plain-English topic",
      "summary": "one sentence describing the complete Moment",
      "why": "why these scenes belong together"
    }
  ]
}
"""


def _moment_limit(video_duration_sec: float | None) -> int:
    """Loose upper bound only. The model is never required to fill it."""
    if not video_duration_sec:
        return 3

    minutes = video_duration_sec / 60.0

    # About 3 potential Moments per 10 minutes, capped for long videos.
    return max(1, min(10, math.ceil(minutes / 10.0 * 3)))


def _format_scenes(scenes: list[dict[str, Any]]) -> str:
    blocks: list[str] = []

    for i, scene in enumerate(scenes):
        blocks.append(
            "\n".join(
                [
                    f"Scene {i}",
                    f"Archetype: {scene.get('archetype', 'unknown')}",
                    f"Topic: {scene.get('topic', '')}",
                    f"Quality: {scene.get('label', 'unknown')}",
                    f"Scout score: {scene.get('scout_score', 0)}",
                    f"Standalone: {scene.get('standalone', 0)}/3",
                    f"Payoff strength: {scene.get('payoff_strength', 0)}/3",
                    f"Filler risk: {scene.get('filler_risk', 0)}/3",
                    f"Guest led: {scene.get('guest_led', False)}",
                    f"Duration: {float(scene.get('duration_sec') or 0):.1f}s",
                    f"Source start: {float(scene.get('src_start') or 0):.1f}s",
                    f"Text: {scene.get('text', '')}",
                ]
            )
        )

    return "\n\n".join(blocks)



def _normalise_groups(
    groups: list[list[int]],
    sentences: list[dict[str, Any]],
) -> list[list[int]]:
    if not sentences:
        return []

    max_idx = len(sentences) - 1
    norm: list[list[int]] = []

    for group in groups:
        if not isinstance(group, (list, tuple)) or len(group) != 2:
            continue

        try:
            lo, hi = int(group[0]), int(group[1])
        except (TypeError, ValueError):
            continue

        lo = max(0, min(lo, max_idx))
        hi = max(0, min(hi, max_idx))

        if lo > hi:
            lo, hi = hi, lo

        norm.append([lo, hi])

    if not norm:
        return []

    norm.sort()

    # Merge overlapping / directly adjacent groups.
    merged = [norm[0]]

    for lo, hi in norm[1:]:
        if lo <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])

    return merged


def _groups_duration(
    groups: list[list[int]],
    sentences: list[dict[str, Any]],
) -> float:
    total = 0.0

    for lo, hi in groups:
        total += (
            float(sentences[hi]["end"])
            - float(sentences[lo]["start"])
        )

    return total


def _materialize_groups(
    groups: list[list[int]],
    sentences: list[dict[str, Any]],
) -> tuple[list[dict[str, float]], str]:
    cuts: list[dict[str, float]] = []
    text_parts: list[str] = []

    for lo, hi in groups:
        group_sents = sentences[lo : hi + 1]

        cuts.append({
            "start": float(group_sents[0]["start"]),
            "end": float(group_sents[-1]["end"]),
        })

        text_parts.append(
            " ".join(str(s.get("text") or "") for s in group_sents)
        )

    return cuts, " ".join(text_parts).strip()


def _expand_groups_to_min(
    groups: list[list[int]],
    sentences: list[dict[str, Any]],
) -> list[list[int]]:
    """
    Extend a grouped Moment into nearby transcript context when it is too short.

    We preserve the detected Moment and prefer extending forward so the existing
    cold open remains intact. Backward expansion is used only when necessary.
    """

    groups = _normalise_groups(groups, sentences)

    if not groups:
        return groups

    duration = _groups_duration(groups, sentences)

    if duration >= TARGET_MIN_S:
        return groups

    # First bridge tiny conversational gaps between selected scenes.
    bridged: list[list[int]] = [groups[0][:]]

    for lo, hi in groups[1:]:
        previous = bridged[-1]

        gap_seconds = (
            float(sentences[lo]["start"])
            - float(sentences[previous[1]]["end"])
        )

        if gap_seconds <= 5.0:
            previous[1] = hi
        else:
            bridged.append([lo, hi])

    groups = bridged
    duration = _groups_duration(groups, sentences)

    # Prefer extending the ending. This preserves the existing hook.
    while duration < TARGET_MIN_S:
        last_lo, last_hi = groups[-1]

        if last_hi >= len(sentences) - 1:
            break

        candidate = [g[:] for g in groups]
        candidate[-1][1] += 1

        candidate_duration = _groups_duration(candidate, sentences)

        if candidate_duration > TARGET_MAX_S:
            break

        groups = candidate
        duration = candidate_duration

    # If the video/topic ends there, add limited context before the opening.
    while duration < TARGET_MIN_S:
        first_lo, first_hi = groups[0]

        if first_lo <= 0:
            break

        candidate = [g[:] for g in groups]
        candidate[0][0] -= 1

        candidate_duration = _groups_duration(candidate, sentences)

        if candidate_duration > TARGET_MAX_S:
            break

        groups = candidate
        duration = candidate_duration

    return groups


def _build_moment(
    scene_ids: list[int],
    topic: str,
    summary: str,
    why: str,
    scenes: list[dict[str, Any]],
    sentences: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    valid_ids = sorted(set(i for i in scene_ids if 0 <= i < len(scenes)))

    if not valid_ids:
        return None

    selected = [scenes[i] for i in valid_ids]

    # Preserve the source conversation flow.
    selected.sort(key=lambda s: float(s.get("src_start") or 0))

    cuts_raw: list[dict[str, float]] = []
    sentence_groups: list[list[int]] = []
    text_parts: list[str] = []

    for scene in selected:
        cuts_raw.extend(scene.get("cuts_raw") or [])
        sentence_groups.extend(scene.get("sentence_groups") or [])

        scene_text = (scene.get("text") or "").strip()
        if scene_text:
            text_parts.append(scene_text)

    if sentences and sentence_groups:
        original_duration = sum(
            float(c["end"]) - float(c["start"])
            for c in cuts_raw
        )

        sentence_groups = _expand_groups_to_min(
            sentence_groups,
            sentences,
        )

        cuts_raw, text = _materialize_groups(
            sentence_groups,
            sentences,
        )

        duration = sum(
            float(c["end"]) - float(c["start"])
            for c in cuts_raw
        )

        if duration > original_duration + 0.5:
            log.info(
                "Moment expanded: %.1fs → %.1fs",
                original_duration,
                duration,
            )
    else:
        text = " ".join(text_parts)

        duration = sum(
            float(c["end"]) - float(c["start"])
            for c in cuts_raw
        )


    if duration <= 0 or duration > ABSOLUTE_MAX_S:
        return None

    # Use the strongest source scene as the temporary quality basis.
    # Proper Moment-level scoring comes in a later stage.
    strongest = max(
        selected,
        key=lambda s: (
            int(s.get("score_total") or 0),
            int(s.get("scout_score") or 0),
        ),
    )

    first = selected[0]

    return {
        "archetype": strongest.get("archetype", "story"),
        "scout_score": max(int(s.get("scout_score") or 0) for s in selected),
        "why": why or summary,
        "label": strongest.get("label", "strong"),
        "score_total": int(strongest.get("score_total") or 0),
        "must_haves": int(strongest.get("must_haves") or 0),
        "signals": int(strongest.get("signals") or 0),
        "must_have_ticks": strongest.get("must_have_ticks", {}),
        "signal_ticks": strongest.get("signal_ticks", {}),
        "sentence_groups": sentence_groups,
        "stitched": len(selected) > 1,
        "transplant": False,
        "cuts_raw": cuts_raw,
        "duration_sec": round(duration, 3),
        "src_start": min(float(s.get("src_start") or 0) for s in selected),
        "speaker": strongest.get("speaker"),
        "text": text,
        "reasoning": why or summary,
        "host_question_open": bool(first.get("host_question_open")),
        "moment_topic": topic.strip(),
        "moment_summary": summary.strip(),
        "scene_ids": valid_ids,
    }


async def build_moments(
    scenes: list[dict[str, Any]],
    api_key: str,
    video_duration_sec: float | None = None,
    sentences: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    if not scenes:
        return []

    # One scene may already be a complete Moment.
    if len(scenes) == 1:
        one = _build_moment(
            [0],
            scenes[0].get("archetype", "Moment"),
            scenes[0].get("text", "")[:160],
            "Single strong scene.",
            scenes,
            sentences,
        )
        return [one] if one else []

    max_moments = _moment_limit(video_duration_sec)

    user = (
        f"Source video duration: "
        f"{(video_duration_sec or 0) / 60:.1f} minutes\n"
        f"Maximum useful Moments: {max_moments}\n\n"
        f"Scenes:\n\n{_format_scenes(scenes)}"
    )

    try:
        data = await call_agent(
            "moment_builder",
            SYSTEM_PROMPT,
            user,
            api_key=api_key,
            temperature=0.2,
            max_tokens=2048,
        )
    except Exception as exc:
        log.warning("Moment Builder failed: %s", exc)
        return scenes

    raw_moments = data.get("moments", []) if isinstance(data, dict) else []

    moments: list[dict[str, Any]] = []
    used_signatures: set[tuple[int, ...]] = set()

    for raw in raw_moments[:max_moments]:
        if not isinstance(raw, dict):
            continue

        try:
            scene_ids = [int(i) for i in raw.get("scene_ids", [])]
        except (TypeError, ValueError):
            continue

        signature = tuple(sorted(set(scene_ids)))

        if not signature or signature in used_signatures:
            continue

        moment = _build_moment(
            scene_ids,
            str(raw.get("topic") or "Moment"),
            str(raw.get("summary") or ""),
            str(raw.get("why") or ""),
            scenes,
            sentences,
        )

        if moment is None:
            continue

        used_signatures.add(signature)
        moments.append(moment)

    # Never let a flaky local-model response erase every valid scene.
    if not moments:
        log.warning("Moment Builder returned no usable Moments; keeping scenes")
        return scenes

    log.info(
        "Moment Builder: %d scenes → %d Moments (%s)",
        len(scenes),
        len(moments),
        ", ".join(f"{m['duration_sec']:.0f}s" for m in moments),
    )

    return moments
