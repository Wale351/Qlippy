"""
identify_clips job — the agent clip pipeline:
Feature Enrichment → Moment Scout ∥ → Clip Crafter ∥ → Comparative Ranker
→ Audio Cut Finder → Hook Writer ∥
"""

from __future__ import annotations
import asyncio
import logging
import tempfile
import time
import uuid
from datetime import datetime
from pathlib import Path

from sources import ensure_source

import httpx

from pipeline.sentencize    import sentencize
from pipeline.features      import enrich_sentences, infer_roles
from pipeline.scout         import scout_moments
from pipeline.crafter       import craft_clips
from pipeline.moment_builder import build_moments
from pipeline.ranker        import rank_clips
from pipeline.cut_finder    import find_cut_points
from pipeline.hook_writer   import write_hooks
from pipeline.energy_index  import build_energy_index
from pipeline.clip_thumbnail import extract_clip_thumbnail, extract_portrait_preview
from jobs.common import _build_diar_timeline
from storage.r2 import download_file, public_url, upload_bytes, delete_keys
from db.supabase import (
    update_job, finish_job, update_video, get_video, get_transcript,
    insert_clips, get_exports_for_video, delete_jobs_for_exports,
    get_clip_ids_for_video, delete_clips_by_ids, get_max_clip_generation,
    update_clip_thumbnail,
)

log = logging.getLogger(__name__)


def _summary(refined: list[dict], runtime: float) -> str:
    """One-line job result (spec §7). Every label the Crafter can emit is counted."""
    counts = {"hero": 0, "strong": 0, "decent": 0, "weak": 0}
    for c in refined:
        counts[c.get("label") or "weak"] += 1
    return (
        f"Found {len(refined)} clips — {counts['hero']} hero, {counts['strong']} strong, "
        f"{counts['decent']} decent, {counts['weak']} weak."
        + (f" Top pick: Clip 1 · {(refined[0].get('hook_line') or refined[0].get('text', '')[:90]).strip()}." if refined else "")
        + f" Runtime: {runtime:.0f}s."
    )


async def handle_identify_clips(job: dict) -> None:
    """
    Run the agent clip pipeline.
    payload: {"video_id", "preferences": {...}, "niche_profile"?: {...}}
    """
    payload       = job["payload"]
    video_id      = payload["video_id"]
    preferences   = payload.get("preferences", {})
    niche_profile = payload.get("niche_profile")  # Stage 2: audience config

    raw_clip_duration = payload.get("clip_duration_seconds")
    clip_duration_seconds = (
        int(raw_clip_duration)
        if raw_clip_duration in (15, 30, 60, 90)
        else None
    )

    keep_existing = bool(payload.get("keep_existing"))  # keep old clips as a previous generation
    job_id        = job["id"]
    t0            = time.monotonic()

    update_job(job_id, status="processing", started_at=datetime.utcnow().isoformat())

    try:
        transcript = get_transcript(video_id)
        if not transcript:
            raise RuntimeError("No transcript found — transcribe first")

        raw_words = transcript["raw_words"]
        video     = get_video(video_id)
        audio_url = public_url(video["audio_r2_key"]) if video.get("audio_r2_key") else None

        log.info(
            "Moment generation settings: video=%.1f min, clip_duration=%s",
            float(video.get("duration_sec") or 0) / 60.0,
            f"{clip_duration_seconds}s" if clip_duration_seconds else "auto",
        )

        # Merge niche profile into preferences if provided
        if niche_profile:
            preferences = {
                **preferences,
                "min_duration_seconds": niche_profile.get("clip_duration_min", preferences.get("min_duration_seconds", 30)),
                "max_duration_seconds": niche_profile.get("clip_duration_max", preferences.get("max_duration_seconds", 90)),
                "target_tones":         niche_profile.get("preferred_tones", preferences.get("target_tones", [])),
                "audience":             niche_profile.get("audience", ""),
                "niche":                niche_profile.get("niche", ""),
                "hook_style":           niche_profile.get("hook_style", ""),
            }

        # Step 1: Sentencize
        update_video(video_id, processing_stage="preparing_analysis")
        sentences = sentencize(raw_words)
        log.info("Sentencize: %d sentences from video %s", len(sentences), video_id)

        if not sentences:
            finish_job(job_id, status="done", finished_at=datetime.utcnow().isoformat(),
                       result={"clips": [], "warning": "empty_transcript"})
            return

        diar_timeline = _build_diar_timeline(transcript)
        log.info(
            "Speaker timeline: %d segments, %d speakers",
            len(diar_timeline),
            len({speaker for _, _, speaker in diar_timeline}),
        )

        # Audio Preprocessor — runs once, concurrent with the Reader LLM call.
        # Failure is non-fatal: cut finder falls back to exact word timestamps.
        # Streamed to disk and decoded incrementally so multi-hour audio never
        # sits in RAM (a 10h WAV is ~1.2 GB, ~2.3 GB once decoded to float32).
        async def _prefetch_and_index(url: str):
            tmp_path = tempfile.mktemp(suffix=".audio")
            try:
                async with httpx.AsyncClient(timeout=300.0) as c:
                    async with c.stream("GET", url) as r:
                        r.raise_for_status()
                        with open(tmp_path, "wb") as f:
                            async for chunk in r.aiter_bytes(1 << 20):
                                f.write(chunk)
                return await asyncio.to_thread(build_energy_index, tmp_path)
            except Exception as e:
                log.warning("Energy index build failed — word-timestamp fallback: %s", e)
                return None
            finally:
                Path(tmp_path).unlink(missing_ok=True)

        index_task = asyncio.create_task(_prefetch_and_index(audio_url)) if audio_url else None

        # The index is awaited BEFORE scouting (not just before cut finding):
        # scout/crafter prompts embed acoustic annotations ([laughter],
        # [energy spike], …) computed from it. Without audio the pipeline
        # still runs, only text/speaker features apply.
        energy_index = await index_task if index_task else None

        # Stage 0 — deterministic acoustic + speaker features (no LLM)
        feats = enrich_sentences(sentences, diar_timeline, energy_index)
        roles = infer_roles(sentences, feats)

        # Stage 1 — Moment Scout: archetype candidates per ~20-min chunk
        scout_context = {
            "title":        video.get("title") or "",
            "niche":        preferences.get("niche", ""),
            "audience":     preferences.get("audience", ""),
            "target_tones": preferences.get("target_tones") or [],
        }
        update_video(video_id, processing_stage="finding_moments")
        candidates = await scout_moments(
            sentences, feats, roles, scout_context,
        )

        # Stage 1.5 — deterministic candidate quality filter.
        # Keep strong Scout picks while demoting obvious weak conversational setup.
        filtered_candidates = []

        for c in candidates:
            score = int(c.get("score") or 0)
            lo = int(c.get("lo") or 0)
            hi = int(c.get("hi") or lo)

            first_text = ""
            payoff_text = ""

            if 0 <= lo < len(sentences):
                first_text = str(sentences[lo].get("text") or "").strip()

            payoff_idx = int(c.get("payoff") or lo)
            if 0 <= payoff_idx < len(sentences):
                payoff_text = str(sentences[payoff_idx].get("text") or "").strip()

            first_lower = first_text.lower()

            # Penalize generic interviewer/question setup.
            if first_text.endswith("?"):
                score -= 8

            if first_lower.startswith(
                (
                    "so ",
                    "and ",
                    "but ",
                    "yeah",
                    "yes",
                    "no ",
                    "okay",
                    "ok ",
                    "what about",
                    "how about",
                    "do you",
                    "did you",
                    "are you",
                    "have you",
                )
            ):
                score -= 5

            # Reward candidates with an actual concrete payoff line.
            payoff_words = payoff_text.split()

            if len(payoff_words) >= 6:
                score += 3

            if payoff_text and not payoff_text.endswith("?"):
                score += 3

            if any(
                token in payoff_text.lower()
                for token in (
                    "because",
                    "realized",
                    "realised",
                    "started",
                    "never",
                    "always",
                    "remember",
                    "love",
                    "hate",
                    "money",
                    "family",
                    "career",
                    "life",
                    "first time",
                )
            ):
                score += 2

            # Keep Scout's original candidate, but attach adjusted score for sorting.
            c = dict(c)
            c["quality_score"] = score
            filtered_candidates.append(c)

        # Scout already returns candidates in a coverage-aware timeline order.
        # Keep enough material for long videos instead of collapsing everything
        # back to a tiny global top-7 list.
        video_minutes = float(video.get("duration_sec") or 0) / 60.0

        if video_minutes <= 10:
            candidate_budget = 6
        elif video_minutes <= 20:
            candidate_budget = 10
        elif video_minutes <= 30:
            candidate_budget = 12
        elif video_minutes <= 60:
            candidate_budget = 18
        else:
            candidate_budget = 24

        candidates = filtered_candidates[:candidate_budget]

        log.info(
            "Quality filter: kept %d of %d Scout candidates across %.1f min",
            len(candidates),
            len(filtered_candidates),
            video_minutes,
        )

        # Stage 2 — Clip Crafter: exact boundaries in play order (hook
        # transplant may put the payoff cut first), checklist, duration gate
        update_video(video_id, processing_stage="crafting_scenes")
        if clip_duration_seconds:
            craft_min_duration = max(6, clip_duration_seconds * 0.8)
            craft_max_duration = min(120, clip_duration_seconds * 1.2)
        else:
            craft_min_duration = 6
            craft_max_duration = preferences.get("max_duration_seconds") or 90

        crafted = await craft_clips(
            candidates, sentences, feats, roles,
            min_duration_s=craft_min_duration,
            max_duration_s=craft_max_duration,
        )

        # Stage 3 — Moment Builder: group related scenes into complete Moments
        update_video(video_id, processing_stage="building_moments")
        moments = await build_moments(
            crafted,
            video_duration_sec=video.get("duration_sec"),
            sentences=sentences,
            target_duration_sec=clip_duration_seconds,
        )

        # Stage 4 — Rank the finished Moments
        max_clips = int(preferences.get("max_clips") or 10)
        update_video(video_id, processing_stage="ranking_moments")
        ranked = await rank_clips(
            moments, scout_context["target_tones"], max_clips,
        )

        # Audio Cut Finder: pure index lookups per chunk boundary
        update_video(video_id, processing_stage="finalizing")
        refined = [find_cut_points(c, energy_index, diar_timeline) for c in ranked]

        # Hook Writer disabled for local mode.
        # Crafter already provides hook_line, saving additional local LLM calls.

        # Persist clips to Supabase — ordered by rank, so row 0 is the top pick
        now = datetime.utcnow().isoformat()
        clip_rows = [
            {
                "id":                str(uuid.uuid4()),
                "video_id":          video_id,
                "user_id":           job["user_id"],
                "hook":              (c.get("hook_line") or c.get("text", "")[:120]).strip(),
                "text":              c.get("text", ""),
                "cuts":              c.get("cuts", c.get("cuts_raw", [])),
                "duration_sec":      c.get("duration_sec"),
                "score":             round(c["score_total"] / 11, 3),  # normalized for existing UI
                "tone":              c.get("archetype"),  # free-text column; UI has fallback styling
                "sentence_groups":   c.get("sentence_groups"),
                "reasoning":         c.get("reasoning"),
                "label":             c.get("label"),
                "must_haves":        c.get("must_haves"),
                "signals":           c.get("signals"),
                "score_total":       c.get("score_total"),
                "stitched":          c.get("stitched", False),
                "chunks":            c.get("chunks", []),
                "crosstalk_flagged": c.get("crosstalk_flagged", False),
                "source":            "ai",
                "created_at":        now,
                "updated_at":        now,
            }
            for c in refined
        ]

        if keep_existing:
            # Keep old clips + exports as a previous generation
            generation = get_max_clip_generation(video_id) + 1
            old_clip_ids: list[str] = []
            log.info("Keeping existing clips for video %s — new generation %d", video_id, generation)
        else:
            generation = 1
            old_clip_ids = get_clip_ids_for_video(video_id)

        for row in clip_rows:
            row["generation"] = generation
        saved_clips = insert_clips(clip_rows) if clip_rows else []

        # ── Delete replaced clips only AFTER the new ones are safely saved ────
        # so a failed insert never loses existing clips. Jobs have no FK to
        # exports, so orphaned rows must be removed manually; R2 files for old
        # exports are also deleted to reclaim storage (clip delete cascades to
        # export rows).
        if old_clip_ids:
            try:
                # New clips have no exports yet, so this is old-clip exports only
                old_exports = get_exports_for_video(video_id)
                if old_exports:
                    old_export_ids  = [e["id"]     for e in old_exports]
                    old_r2_keys     = [e["r2_key"] for e in old_exports if e.get("r2_key")]
                    delete_jobs_for_exports(old_export_ids)
                    delete_keys(old_r2_keys)
                    log.info(
                        "Re-run cleanup: deleted %d export jobs + %d R2 files for video %s",
                        len(old_export_ids), len(old_r2_keys), video_id,
                    )
                delete_clips_by_ids(old_clip_ids)
            except Exception as cleanup_err:
                log.warning("Re-run cleanup failed (non-fatal): %s", cleanup_err)

        # ── Face thumbnail extraction ─────────────────────────────────────────
        # Download source video once, scan each clip's first cut for the best
        # face frame, upload as a JPEG thumbnail, and update the clip row.
        source_r2_key = video.get("source_r2_key")
        if source_r2_key and saved_clips:
            source_local_thumb = tempfile.mktemp(suffix=".mp4")
            try:
                log.info("Loading cached source for thumbnail extraction")
                source_local_thumb, _ = await ensure_source(
                    video_id, video.get("youtube_url") or ""
                )
                for saved, row in zip(saved_clips, clip_rows):
                    try:
                        cuts = row.get("cuts") or []
                        if not cuts:
                            continue
                        thumb = extract_clip_thumbnail(
                            source_local_thumb,
                            float(cuts[0]["start"]),
                            float(cuts[0]["end"]),
                        )
                        if thumb:
                            r2_key  = f"thumbnails/{saved['id']}.jpg"
                            upload_bytes(thumb, r2_key, content_type="image/jpeg")
                            update_clip_thumbnail(saved["id"], public_url(r2_key))
                    except Exception as thumb_err:
                        log.warning("Thumbnail failed for clip %s: %s", saved.get("id"), thumb_err)
            except Exception as dl_err:
                log.warning("Source download for thumbnails failed: %s", dl_err)
            finally:
                # Persistent source cache is intentionally kept for future jobs.
                pass

        # Portrait preview generation skipped here in local mode.
        # A preview is already generated during transcription, avoiding
        # another full source-video download from R2.

        update_video(video_id, status="ready", processing_stage=None)

        summary = _summary(refined, time.monotonic() - t0)
        finish_job(
            job_id,
            status="done",
            finished_at=now,
            result={
                "clip_count":           len(saved_clips),
                "summary":              summary,
                "energy_index_applied": energy_index is not None,
            },
        )
        log.info("Clip identification done: %s", summary)

    except Exception as e:
        log.exception("Clip identification failed for job %s", job_id)
        finish_job(job_id, status="error", error_msg=str(e), finished_at=datetime.utcnow().isoformat())
        update_video(video_id, status="error", error_msg=str(e), processing_stage=None)
        raise
