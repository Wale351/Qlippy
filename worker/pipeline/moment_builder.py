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

The requested target range is important.
Aim to return enough genuinely distinct Moments to reach that range when the
source material supports it.

Do not collapse many different strong moments into only one or two outputs.
Prefer broad source-timeline coverage: early, middle, and later sections should
all contribute when they contain worthwhile material.

Return fewer than the target minimum only when there genuinely are not enough
coherent, non-duplicate moments in the supplied scenes.

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


def _moment_target_range(
    video_duration_sec: float | None,
    target_duration_sec: float | None = None,
) -> tuple[int, int]:
    """Return a useful target range for unique Moments based on source length."""
    if not video_duration_sec:
        return (2, 4)

    minutes = video_duration_sec / 60.0

    if minutes <= 5:
        low, high = 2, 3
    elif minutes <= 10:
        low, high = 3, 5
    elif minutes <= 20:
        low, high = 5, 7
    elif minutes <= 30:
        low, high = 7, 9
    elif minutes <= 45:
        low, high = 8, 10
    elif minutes <= 60:
        low, high = 9, 12
    elif minutes <= 90:
        low, high = 10, 14
    else:
        low, high = 12, 16

    # Short requested clips can support slightly more distinct Moments.
    # Long requested clips need more room for complete ideas.
    if target_duration_sec and target_duration_sec <= 15:
        high = min(16, high + 2)
    elif target_duration_sec and target_duration_sec <= 30:
        high = min(16, high + 1)
    elif target_duration_sec and target_duration_sec >= 90:
        low = max(2, low - 1)

    return low, high


def _moment_limit(
    video_duration_sec: float | None,
    target_duration_sec: float | None = None,
) -> int:
    return _moment_target_range(
        video_duration_sec,
        target_duration_sec,
    )[1]


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
    target_min_s: float = TARGET_MIN_S,
    target_max_s: float = TARGET_MAX_S,
) -> list[list[int]]:
    """
    Extend a grouped Moment into nearby transcript context when it is too short.

    We preserve the detected Moment and prefer extending forward so the existing
    cold open remains intact. Backward expansion is used only when necessary.
    """

    groups = _normalise_groups(groups, sentences)

    if not groups:
        return groups

    # First bridge only tiny conversational gaps between selected scenes.
    # Bridging means keeping the intervening conversation, not jump-cutting it.
    bridged: list[list[int]] = [groups[0][:]]

    for lo, hi in groups[1:]:
        previous = bridged[-1]

        gap_seconds = (
            float(sentences[lo]["start"])
            - float(sentences[previous[1]]["end"])
        )

        if gap_seconds <= 5.0:
            candidate = [g[:] for g in bridged]
            candidate[-1][1] = hi

            if _groups_duration(candidate, sentences) <= target_max_s:
                previous[1] = hi
            else:
                bridged.append([lo, hi])
        else:
            bridged.append([lo, hi])

    groups = bridged

    # Prefer one continuous section over stitching distant parts of the video.
    # Pick the section containing the most selected material, then expand
    # naturally around it to reach the target duration.
    if len(groups) > 1:
        anchor_index = max(
            range(len(groups)),
            key=lambda i: (
                _groups_duration([groups[i]], sentences),
                -i,
            ),
        )

        anchor = groups[anchor_index][:]
        dropped = len(groups) - 1

        log.info(
            "Moment continuity: using anchor %s; ignoring %d distant group(s)",
            anchor,
            dropped,
        )

        groups = [anchor]

    duration = _groups_duration(groups, sentences)

    if duration >= target_min_s:
        return groups

    # Prefer extending the ending. This preserves the existing hook.
    while duration < target_min_s:
        last_lo, last_hi = groups[-1]

        if last_hi >= len(sentences) - 1:
            break

        candidate = [g[:] for g in groups]
        candidate[-1][1] += 1

        candidate_duration = _groups_duration(candidate, sentences)

        if candidate_duration > target_max_s:
            break

        groups = candidate
        duration = candidate_duration

    # If the video/topic ends there, add limited context before the opening.
    while duration < target_min_s:
        first_lo, first_hi = groups[0]

        if first_lo <= 0:
            break

        candidate = [g[:] for g in groups]
        candidate[0][0] -= 1

        candidate_duration = _groups_duration(candidate, sentences)

        if candidate_duration > target_max_s:
            break

        groups = candidate
        duration = candidate_duration

    return groups



def _repair_awkward_opening(
    groups: list[list[int]],
    sentences: list[dict[str, Any]],
) -> list[list[int]]:
    """Add one sentence of prior context when a Moment clearly starts mid-thought."""
    if not groups:
        return groups

    first_lo = groups[0][0]
    if first_lo <= 0:
        return groups

    text = str(sentences[first_lo].get("text") or "").strip().lower()

    continuation_starts = (
        "and ",
        "but ",
        "so ",
        "now ",
        "then ",
        "because ",
        "which ",
        "also ",
        "i mean ",
        "you know ",
        "the thing is ",
    )

    if not text.startswith(continuation_starts):
        return groups

    prev = first_lo - 1
    gap = (
        float(sentences[first_lo]["start"])
        - float(sentences[prev]["end"])
    )

    # A large pause usually means this really is a new thought.
    if gap > 3.0:
        return groups

    candidate = [g[:] for g in groups]
    candidate[0][0] = prev

    if _groups_duration(candidate, sentences) > TARGET_MAX_S:
        return groups

    log.info(
        "Moment opening repaired: prepended sentence %d before %d",
        prev,
        first_lo,
    )
    return candidate


def _trim_groups_to_max(
    groups: list[list[int]],
    sentences: list[dict[str, Any]],
    target_max_s: float,
) -> list[list[int]]:
    """Trim a Moment at sentence boundaries so it cannot exceed its target max."""
    groups = _normalise_groups(groups, sentences)

    if not groups:
        return groups

    duration = _groups_duration(groups, sentences)

    # Preserve the opening/hook where possible and trim excess from the end.
    while duration > target_max_s:
        lo, hi = groups[-1]

        if hi <= lo:
            break

        candidate = [g[:] for g in groups]
        candidate[-1][1] -= 1

        candidate_duration = _groups_duration(candidate, sentences)

        if candidate_duration <= 0:
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
    target_duration_sec: float | None = None,
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

        if target_duration_sec:
            target_min_s = max(6.0, target_duration_sec * 0.8)
            target_max_s = min(ABSOLUTE_MAX_S, target_duration_sec * 1.2)
        else:
            target_min_s = TARGET_MIN_S
            target_max_s = TARGET_MAX_S

        sentence_groups = _expand_groups_to_min(
            sentence_groups,
            sentences,
            target_min_s=target_min_s,
            target_max_s=target_max_s,
        )

        sentence_groups = _repair_awkward_opening(
            sentence_groups,
            sentences,
        )

        # Opening repair or an already-long source scene can push the final
        # Moment beyond the selected duration. Enforce the final ceiling here.
        if target_duration_sec:
            before_trim = _groups_duration(sentence_groups, sentences)

            sentence_groups = _trim_groups_to_max(
                sentence_groups,
                sentences,
                target_max_s,
            )

            after_trim = _groups_duration(sentence_groups, sentences)

            if after_trim < before_trim - 0.1:
                log.info(
                    "Moment trimmed to target: %.1fs → %.1fs",
                    before_trim,
                    after_trim,
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


def _moment_timeline_bucket(
    item: dict[str, Any],
    video_duration_sec: float,
    bucket_count: int,
) -> int:
    start = float(item.get("src_start") or 0)

    cuts = item.get("cuts_raw") or []
    if cuts:
        try:
            end = max(float(c.get("end") or start) for c in cuts)
        except (TypeError, ValueError):
            end = start + float(item.get("duration_sec") or 0)
    else:
        end = start + float(item.get("duration_sec") or 0)

    midpoint = max(0.0, min(video_duration_sec, (start + end) / 2.0))
    width = video_duration_sec / bucket_count

    if width <= 0:
        return 0

    return min(bucket_count - 1, int(midpoint / width))


def _scene_quality_key(scene: dict[str, Any]) -> tuple[int, int]:
    return (
        int(scene.get("score_total") or 0),
        int(scene.get("scout_score") or 0),
    )


def _ensure_moment_coverage(
    moments: list[dict[str, Any]],
    scenes: list[dict[str, Any]],
    video_duration_sec: float | None,
    sentences: list[dict[str, Any]] | None,
    target_min_moments: int,
    target_max_moments: int,
    target_duration_sec: float | None = None,
) -> list[dict[str, Any]]:
    """Recover strong unused scenes when model output is too sparse or clustered."""
    if not video_duration_sec or video_duration_sec <= 0 or not scenes:
        return moments

    # Measure coverage across the whole source using equal timeline zones.
    # Ten zones gives a useful early → middle → late spread without becoming
    # impossible for long videos with a limited Moment budget.
    bucket_count = max(
        4,
        min(10, math.ceil(video_duration_sec / 120.0)),
    )
    minimum_buckets = max(1, math.ceil(bucket_count * 0.5))
    preferred_buckets = max(
        minimum_buckets,
        math.ceil(bucket_count * 0.75),
    )

    used_scene_ids = {
        int(scene_id)
        for moment in moments
        for scene_id in (moment.get("scene_ids") or [])
        if isinstance(scene_id, int)
    }

    covered_buckets = {
        _moment_timeline_bucket(moment, video_duration_sec, bucket_count)
        for moment in moments
    }

    # Coverage recovery should never manufacture weak Moments just to tick
    # another timeline zone. Only recover scenes that survived as useful.
    available: list[tuple[int, dict[str, Any]]] = [
        (i, scene)
        for i, scene in enumerate(scenes)
        if i not in used_scene_ids
        and str(scene.get("label") or "strong").lower() != "weak"
    ]

    # First priority: recover strong scenes from uncovered timeline sections.
    while (
        len(moments) < target_max_moments
        and len(covered_buckets) < preferred_buckets
        and available
    ):
        available_buckets = {
            _moment_timeline_bucket(
                scene,
                video_duration_sec,
                bucket_count,
            )
            for _, scene in available
        }

        uncovered = [
            bucket
            for bucket in available_buckets
            if bucket not in covered_buckets
        ]

        if not uncovered:
            break

        # Pick the uncovered zone furthest from what we already have.
        # This prevents five Moments from all landing in the first half.
        if covered_buckets:
            bucket = max(
                uncovered,
                key=lambda b: min(abs(b - c) for c in covered_buckets),
            )
        else:
            bucket = uncovered[len(uncovered) // 2]

        choices = [
            (i, scene)
            for i, scene in available
            if _moment_timeline_bucket(
                scene,
                video_duration_sec,
                bucket_count,
            ) == bucket
        ]

        if not choices:
            break

        i, scene = max(
            choices,
            key=lambda pair: _scene_quality_key(pair[1]),
        )

        extra = _build_moment(
            [i],
            str(scene.get("topic") or scene.get("archetype") or "Moment"),
            str(scene.get("text") or "")[:160],
            "Strong unused scene recovered to improve source timeline coverage.",
            scenes,
            sentences,
            target_duration_sec,
        )

        if extra:
            moments.append(extra)
            used_scene_ids.add(i)
            covered_buckets.add(bucket)

        available = [(j, s) for j, s in available if j != i]

    # Second priority: if the model still returned too few Moments,
    # promote the strongest remaining unique scenes.
    while len(moments) < target_min_moments and available:
        i, scene = max(
            available,
            key=lambda pair: _scene_quality_key(pair[1]),
        )

        available = [(j, s) for j, s in available if j != i]

        extra = _build_moment(
            [i],
            str(scene.get("topic") or scene.get("archetype") or "Moment"),
            str(scene.get("text") or "")[:160],
            "Strong unused scene recovered to reach the source-length Moment target.",
            scenes,
            sentences,
            target_duration_sec,
        )

        if extra:
            moments.append(extra)
            used_scene_ids.add(i)

    coverage_pct = (
        len(covered_buckets) / bucket_count * 100.0
        if bucket_count
        else 0.0
    )

    log.info(
        "Moment coverage: %d/%d timeline zones (%.0f%%), %d Moments — target 75%%, floor 50%%",
        len(covered_buckets),
        bucket_count,
        coverage_pct,
        len(moments),
    )

    return moments[:target_max_moments]


async def build_moments(
    scenes: list[dict[str, Any]],
    video_duration_sec: float | None = None,
    sentences: list[dict[str, Any]] | None = None,
    target_duration_sec: float | None = None,
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
            target_duration_sec,
        )
        return [one] if one else []

    target_min_moments, target_max_moments = _moment_target_range(
        video_duration_sec,
        target_duration_sec,
    )
    max_moments = target_max_moments

    duration_instruction = (
        f"Target finished Moment duration: about {target_duration_sec:.0f} seconds. "
        f"Aim to stay roughly within {target_duration_sec * 0.8:.0f}-"
        f"{target_duration_sec * 1.2:.0f} seconds when the material allows. "
        "Do not combine separate good moments merely to make a longer clip.\n"
        if target_duration_sec
        else ""
    )

    user = (
        f"Source video duration: "
        f"{(video_duration_sec or 0) / 60:.1f} minutes\n"
        f"Target useful Moments: {target_min_moments}-{target_max_moments}\n"
        f"Maximum useful Moments: {max_moments}\n"
        f"Spread the Moments across the source timeline instead of clustering them "
        f"inside one short section. Each Moment must cover a genuinely different "
        f"topic, story, argument, insight, joke, or emotional beat.\n"
        f"{duration_instruction}\n"
        f"Scenes:\n\n{_format_scenes(scenes)}"
    )

    try:
        data = await call_agent(
            "moment_builder",
            SYSTEM_PROMPT,
            user,
            temperature=0.2,
            max_tokens=4096,
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
            target_duration_sec,
        )

        if moment is None:
            continue

        used_signatures.add(signature)
        moments.append(moment)

    # Never let a flaky local-model response erase every valid scene.
    if not moments:
        log.warning("Moment Builder returned no usable Moments; keeping scenes")
        moments = scenes[:target_max_moments]

    moments = _ensure_moment_coverage(
        moments,
        scenes,
        video_duration_sec,
        sentences,
        target_min_moments,
        target_max_moments,
        target_duration_sec,
    )

    log.info(
        "Moment Builder: %d scenes → %d Moments (%s)",
        len(scenes),
        len(moments),
        ", ".join(f"{m['duration_sec']:.0f}s" for m in moments),
    )

    return moments
