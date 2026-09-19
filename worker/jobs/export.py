"""
Two-phase export jobs.

edit_clip  — expensive, cached: download + stitch + face-tracked portrait →
             clean portrait MP4 stored on the clip_edits row.
style_clip — cheap, re-runnable: composite styling (captions, sticker,
             watermark) on top of the cached portrait.

edit_clip chains style_clip by inserting a queued job row and starting it —
if the process dies in the chain window, the row is still 'queued' and the
poll loop recovers it (the old raw create_task version lost the job forever).
"""

from __future__ import annotations
import asyncio
import logging
import os
import subprocess
import json
import tempfile
from datetime import datetime
from pathlib import Path

from sources import download_and_stitch_cuts
from jobs.common import _build_diar_timeline, _remap_diar_timeline, speaker_faces_key
from pipeline.portrait import cut_and_portrait, bind_speakers_from_source, SPEAKER_FACES_VERSION
from pipeline.subtitle import burn_subtitles
from pipeline.clip_spec import build_clip_spec
from storage.r2 import upload_file, download_file, public_url, upload_bytes, key_exists
from db.supabase import (
    update_job, finish_job, update_export, update_clip_edit,
    get_video, get_transcript, get_clip, get_preset, get_clip_edit,
    insert_job, update_clip_thumbnail,
)

log = logging.getLogger(__name__)

# Bulk exports queue many jobs at once — cap concurrent ffmpeg renders so the
# instance isn't overwhelmed; excess jobs wait their turn inside the semaphore.
# The gate wraps only the CPU-heavy portrait/render step (see below) so that
# network-bound downloads can still overlap across jobs.
EXPORT_CONCURRENCY = max(1, int(os.getenv("EXPORT_CONCURRENCY", "2")))
_export_semaphore = asyncio.Semaphore(EXPORT_CONCURRENCY)


def _video_width(video_path: str, fallback: int = 1080) -> int:
    """Read the actual rendered canvas width from the video."""
    probe = subprocess.run(
        [
            "ffprobe", "-v", "quiet",
            "-select_streams", "v:0",
            "-show_entries", "stream=width",
            "-of", "csv=p=0",
            video_path,
        ],
        capture_output=True,
        text=True,
    )
    try:
        return int(probe.stdout.strip().split(",")[0])
    except (ValueError, IndexError):
        return fallback


async def _apply_text_sticker(video_path: str, cfg: dict) -> str | None:
    """Burn an Instagram-style text sticker onto the video. Returns new path or None."""
    from pipeline.text_sticker import render_sticker_png

    text = (cfg.get("text") or "").strip()[:160]
    if not text:
        return None

    size_pct = float(cfg.get("size_pct", 8))
    scale    = float(cfg.get("scale", 1.0))
    x_pct    = float(cfg.get("x_pct", 50))
    y_pct    = float(cfg.get("y_pct", 50))
    bg       = str(cfg.get("bg") or "#ffffff")
    fg       = str(cfg.get("fg") or "#000000")

    # Line breaks captured from the live browser preview (the source of truth the
    # user saw and dragged). Render them verbatim so the wrap matches exactly.
    raw_lines = cfg.get("lines")
    lines = (
        [str(ln) for ln in raw_lines if str(ln).strip()]
        if isinstance(raw_lines, list)
        else None
    )

    # Scale against the actual output canvas so sticker sizing matches every ratio.
    video_w = _video_width(video_path)
    font_px      = max(int(video_w * size_pct / 100), 8)
    max_width_px = int(video_w * 0.85)

    try:
        png_bytes = await asyncio.to_thread(
            render_sticker_png, text, font_px, max_width_px, lines, bg, fg
        )
    except Exception as e:
        log.warning("Text sticker render failed: %s", e)
        return None

    sticker_local = tempfile.mktemp(suffix=".png")
    Path(sticker_local).write_bytes(png_bytes)

    # Place the sticker *center* at (x_pct, y_pct), exactly like the preview: the
    # preview keeps the center on (x_pct, y_pct) and lets the box overflow the
    # frame symmetrically, so we do NOT clamp the box into frame here — clamping
    # shifts a large/off-centre sticker sideways and clips words, diverging from
    # what the user saw. ffmpeg overlay clips any overflow at the frame edge.
    x_expr = f"main_w*{x_pct / 100}-overlay_w/2"
    y_expr = f"main_h*{y_pct / 100}-overlay_h/2"

    # Scale the whole sticker as one object (no re-wrap) before compositing.
    scale_clamped = max(0.5, min(2.0, scale))
    filter_complex = (
        f"[1:v]scale=iw*{scale_clamped}:-1[st];"
        f"[0:v][st]overlay={x_expr}:{y_expr}"
    )

    out_path = tempfile.mktemp(suffix=".mp4")
    result = await asyncio.to_thread(
        subprocess.run,
        ["ffmpeg", "-y", "-i", video_path, "-i", sticker_local,
         "-filter_complex", filter_complex,
         "-codec:a", "copy", out_path],
        capture_output=True,
    )
    Path(sticker_local).unlink(missing_ok=True)

    if result.returncode == 0:
        return out_path

    log.warning("Text sticker overlay failed: %s", result.stderr.decode(errors="replace")[-500:])
    Path(out_path).unlink(missing_ok=True)
    return None


async def _apply_branding_watermark(video_path: str) -> str | None:
    """
    Burn the forced free-tier "Edited by Clippy" badge (bottom-centre).
    Returns new path or None on failure (non-fatal — export still ships).
    """
    from pipeline.branding import make_branding_badge

    probe = subprocess.run(
        ["ffprobe", "-v", "quiet", "-select_streams", "v:0",
         "-show_entries", "stream=width", "-of", "csv=p=0", video_path],
        capture_output=True, text=True,
    )
    try:
        video_w = int(probe.stdout.strip().split(",")[0])
    except (ValueError, IndexError):
        video_w = 1080

    badge_local = tempfile.mktemp(suffix=".png")
    out_path    = tempfile.mktemp(suffix=".mp4")
    try:
        Path(badge_local).write_bytes(make_branding_badge(int(video_w * 0.30)))
        result = await asyncio.to_thread(
            subprocess.run,
            ["ffmpeg", "-y", "-i", video_path, "-i", badge_local,
             "-filter_complex", "[0:v][1:v]overlay=(main_w-overlay_w)/2:main_h-overlay_h-36",
             "-codec:a", "copy", out_path],
            capture_output=True,
        )
        if result.returncode == 0:
            return out_path
        log.warning("Branding watermark failed: %s", result.stderr.decode(errors="replace")[-500:])
    except Exception as e:
        log.warning("Branding watermark failed: %s", e)
    Path(out_path).unlink(missing_ok=True)
    return None


async def _apply_image_sticker(video_path: str, cfg: dict) -> str | None:
    """Composite an uploaded PNG sticker (logo) onto the video. Returns new path or None."""
    from pipeline.image_sticker import build_overlay_filter

    r2_key = str(cfg.get("r2_key") or "")
    if not r2_key.startswith("stickers/"):
        return None

    png_local = tempfile.mktemp(suffix=".png")
    try:
        await asyncio.to_thread(download_file, r2_key, png_local)
    except Exception as e:
        log.warning("Image sticker download failed (%s): %s", r2_key, e)
        return None

    # Scale uploaded stickers against the actual output canvas width.
    video_w = _video_width(video_path)
    out_path = tempfile.mktemp(suffix=".mp4")
    result = await asyncio.to_thread(
        subprocess.run,
        ["ffmpeg", "-y", "-i", video_path, "-i", png_local,
         "-filter_complex", build_overlay_filter(cfg, canvas_w=video_w),
         "-codec:a", "copy", out_path],
        capture_output=True,
    )
    Path(png_local).unlink(missing_ok=True)

    if result.returncode == 0:
        return out_path
    log.warning("Image sticker composite failed: %s", result.stderr[-800:].decode(errors="replace"))
    Path(out_path).unlink(missing_ok=True)
    return None


async def _chain_style_clip(edit_job: dict, portrait_key: str) -> None:
    """Create + start a style_clip job once the cached portrait (edit) is ready."""
    p = edit_job["payload"]
    style_job = insert_job({
        "user_id":   edit_job["user_id"],
        "type":      "style_clip",
        "entity_id": p["export_id"],
        "status":    "queued",
        "payload": {
            "export_id":           p["export_id"],
            "edit_id":             p["edit_id"],
            "clip_id":             p["clip_id"],
            "video_id":            p["video_id"],
            "aspect_ratio":        p.get("aspect_ratio"),
            "preset_id":           p.get("preset_id"),
            "preset_config":       p.get("preset_config"),
            "text_sticker_config": p.get("text_sticker_config"),
            "image_sticker_config": p.get("image_sticker_config"),
            "branding_watermark":  p.get("branding_watermark"),
            "clip_hook":           p.get("clip_hook"),
            "preset_name":         p.get("preset_name"),
        },
    })
    # Late import (runner imports this module via the handler registry).
    import runner
    runner.start_job(style_job)
    log.info("Auto-chained style_clip job %s for export %s", style_job["id"], p["export_id"])



def _load_speaker_faces(video: dict, diar_timeline: list) -> dict[str, list[float]] | None:
    """
    The whole-video speaker→face map, generated at transcribe time. Videos
    transcribed before that existed get it built here once from the stored
    source and cached back, so nobody has to re-transcribe to get correct
    face tracking. Returns None when there's nothing to bind against.
    """
    if not diar_timeline:
        return None
    key = speaker_faces_key(video["id"])
    if key_exists(key):
        local = tempfile.mktemp(suffix=".json")
        try:
            download_file(key, local)
            with open(local) as f:
                stored = json.load(f)
        finally:
            Path(local).unlink(missing_ok=True)
        if stored.get("version") == SPEAKER_FACES_VERSION:
            return stored
        log.info("Stored speaker→face map for video %s is from an older method — rebuilding", video["id"])

    source_key = video.get("source_r2_key")
    if not source_key or not key_exists(source_key):
        log.warning("No speaker→face map and no stored source for video %s — binding per clip", video["id"])
        return None
    log.info("Building speaker→face map for video %s from the stored source (one-time)", video["id"])
    local = tempfile.mktemp(suffix=".mp4")
    try:
        download_file(source_key, local)
        faces = bind_speakers_from_source(local, diar_timeline)
    finally:
        Path(local).unlink(missing_ok=True)
    if faces.get("speakers"):
        upload_bytes(json.dumps(faces).encode(), key, "application/json")
        return faces
    return None


async def handle_edit_clip(job: dict) -> None:
    """
    Edit stage (expensive, cached): download + stitch + face-tracked portrait →
    clean portrait MP4 stored on the clip_edits row. On success, chains a
    style_clip job that composites styling on top of this cached portrait.
    payload: {"edit_id", "export_id", "clip_id", "video_id", "aspect_ratio", + style config}
    """
    payload   = job["payload"]
    edit_id   = payload["edit_id"]
    export_id = payload["export_id"]
    clip_id   = payload["clip_id"]
    video_id  = payload["video_id"]
    job_id    = job["id"]
    log.info("edit_clip job %s: edit_id=%s clip_id=%s", job_id, edit_id, clip_id)

    update_job(job_id, status="processing", started_at=datetime.utcnow().isoformat())
    update_clip_edit(edit_id, status="processing")
    update_export(export_id, status="processing")

    stitched_local = None
    cut_local      = None
    analysis_local = None
    try:
        clip, video, transcript = await asyncio.gather(
            asyncio.to_thread(get_clip, clip_id),
            asyncio.to_thread(get_video, video_id),
            asyncio.to_thread(get_transcript, video_id),
        )
        # Give the outer speech boundaries a tiny safety cushion.
        # Transcript timestamps can end a fraction early and clip the first/last
        # spoken sound during the physical video cut. Internal cut points remain exact.
        cuts = [dict(c) for c in clip["cuts"]]
        if cuts:
            cuts[0]["start"] = max(0.0, float(cuts[0]["start"]) - 0.15)
            cuts[-1]["end"] = float(cuts[-1]["end"]) + 0.30

        diar_timeline = _build_diar_timeline(transcript)

        # Download only the clip time ranges and stitch into one landscape file.
        # InsightFace then scans this short clip instead of the full source video.
        stitched_local = await download_and_stitch_cuts(video["youtube_url"], cuts)

        remapped_diar  = _remap_diar_timeline(diar_timeline, cuts)
        total_duration = sum(float(c["end"]) - float(c["start"]) for c in cuts)
        speaker_faces  = await asyncio.to_thread(_load_speaker_faces, video, diar_timeline)

        cut_local = tempfile.mktemp(suffix=".mp4")
        analysis_local = tempfile.mktemp(suffix=".json")
        async with _export_semaphore:
            await asyncio.to_thread(
                cut_and_portrait,
                stitched_local,
                [{"start": 0, "end": total_duration}],
                cut_local,
                diar_timeline=remapped_diar,
                analysis_out_path=analysis_local,
                speaker_faces=speaker_faces,
                aspect_ratio=payload.get("aspect_ratio") or "9:16",
                fit_mode=payload.get("fit_mode") or "fill",
            )

        portrait_key = f"clip_edits/{edit_id}/portrait.mp4"
        await asyncio.to_thread(upload_file, cut_local, portrait_key, "video/mp4")

        # Persist the per-frame face/active-speaker scan (analysis_ref) — feedstock
        # for the presence map, heatmap, auto-placement, and overlap lint.
        analysis_key = None
        if Path(analysis_local).exists() and Path(analysis_local).stat().st_size > 0:
            analysis_key = f"clip_edits/{edit_id}/analysis.json"
            await asyncio.to_thread(upload_file, analysis_local, analysis_key, "application/json")

        update_clip_edit(edit_id, status="done", r2_key=portrait_key,
                         **({"analysis_r2_key": analysis_key} if analysis_key else {}))
        finish_job(job_id, status="done", finished_at=datetime.utcnow().isoformat(),
                   result={"edit_id": edit_id, "r2_key": portrait_key})
        log.info("Edit done: %s → %s", edit_id, portrait_key)

        await _chain_style_clip(job, portrait_key)

    except Exception as e:
        log.exception("Edit failed for job %s", job_id)
        update_clip_edit(edit_id, status="error", error_msg=str(e))
        update_export(export_id, status="error", error_msg=str(e))
        finish_job(job_id, status="error", error_msg=str(e), finished_at=datetime.utcnow().isoformat())
        raise
    finally:
        for p in [stitched_local, cut_local, analysis_local]:
            if p:
                Path(p).unlink(missing_ok=True)
        if stitched_local:
            # Portrait analysis writes its timeline cache next to the source file
            Path(stitched_local + ".portrait_timeline.json").unlink(missing_ok=True)


async def handle_style_clip(job: dict) -> None:
    """
    Style stage (cheap, re-runnable): composite styling on top of the cached
    portrait from clip_edits — no face-tracking. Restyles reuse the same edit.
    payload: {"export_id", "edit_id", "clip_id", "video_id", "preset_id"?, + style config}
    """
    payload   = job["payload"]
    export_id = payload["export_id"]
    edit_id   = payload["edit_id"]
    clip_id   = payload["clip_id"]
    video_id  = payload["video_id"]
    preset_id     = payload.get("preset_id")
    inline_config = payload.get("preset_config")  # config passed directly, no DB lookup needed
    job_id        = job["id"]
    hf_comp = (inline_config or {}).get("hyperframes_component") if inline_config else None
    log.info(
        "style_clip job %s: edit_id=%s preset_id=%s inline_config_present=%s hyperframes_component=%s",
        job_id, edit_id, preset_id, bool(inline_config), hf_comp,
    )

    update_job(job_id, status="processing", started_at=datetime.utcnow().isoformat())
    update_export(export_id, status="processing")

    portrait_local = None
    rendered_local = None
    extra_tmp: list[str] = []

    try:
        edit, clip, preset, transcript = await asyncio.gather(
            asyncio.to_thread(get_clip_edit, edit_id),
            asyncio.to_thread(get_clip, clip_id),
            asyncio.to_thread(get_preset, preset_id) if preset_id else asyncio.sleep(0, result=None),
            asyncio.to_thread(get_transcript, video_id),
        )
        if not edit or not edit.get("r2_key"):
            raise RuntimeError(f"clip_edit {edit_id} has no cached portrait to style")
        cuts = clip["cuts"]
        raw_words     = transcript["raw_words"] if transcript else []

        # Resolve preset config: saved preset takes priority, then inline config from payload
        preset_config = None
        if preset:
            preset_config = preset["config"]
        elif inline_config:
            preset_config = inline_config

        # Pull the cached clean portrait instead of re-cutting/re-tracking.
        portrait_local = tempfile.mktemp(suffix=".mp4")
        await asyncio.to_thread(download_file, edit["r2_key"], portrait_local)

        working_path = portrait_local

        if preset_config:
            rendered_local = tempfile.mktemp(suffix=".mp4")
            await asyncio.to_thread(
                burn_subtitles,
                input_path=working_path,
                output_path=rendered_local,
                raw_words=raw_words,
                cuts=cuts,
                preset_config=preset_config,
            )
            working_path = rendered_local

        text_sticker_config = payload.get("text_sticker_config")
        if text_sticker_config:
            ts_result = await _apply_text_sticker(working_path, text_sticker_config)
            if ts_result:
                extra_tmp.append(ts_result)
                working_path = ts_result

        image_sticker_config = payload.get("image_sticker_config")
        if image_sticker_config:
            is_result = await _apply_image_sticker(working_path, image_sticker_config)
            if is_result:
                extra_tmp.append(is_result)
                working_path = is_result

        # Free-tier branding — set server-side in the export route, never by the client
        if payload.get("branding_watermark"):
            branded = await _apply_branding_watermark(working_path)
            if branded:
                extra_tmp.append(branded)
                working_path = branded

        r2_key = f"exports/{export_id}/export.mp4"
        export_url = public_url(r2_key)

        async def _upload_thumbnail():
            try:
                # Extract from the clean cached portrait, not the styled output, so the
                # clip thumbnail stays caption/sticker-free — it backs the re-clip design
                # preview, where burned-in captions would overlap the new layers.
                thumb_local = tempfile.mktemp(suffix=".jpg")
                thumb_result = await asyncio.to_thread(
                    subprocess.run,
                    ["ffmpeg", "-y", "-i", portrait_local, "-frames:v", "1", thumb_local],
                    capture_output=True,
                )
                if thumb_result.returncode == 0 and Path(thumb_local).exists() and Path(thumb_local).stat().st_size > 0:
                    with open(thumb_local, "rb") as f:
                        thumb_bytes = f.read()
                    thumb_r2_key = f"thumbnails/{clip_id}.jpg"
                    await asyncio.to_thread(upload_bytes, thumb_bytes, thumb_r2_key, content_type="image/jpeg")
                    update_clip_thumbnail(clip_id, public_url(thumb_r2_key))
                    log.info("Thumbnail updated for clip %s", clip_id)
                Path(thumb_local).unlink(missing_ok=True)
            except Exception as thumb_err:
                log.warning("Thumbnail extraction failed for clip %s: %s", clip_id, thumb_err)

        await asyncio.gather(
            asyncio.to_thread(upload_file, working_path, r2_key, "video/mp4"),
            _upload_thumbnail(),
        )

        # Canonical ClipSpec — record what this export was rendered from (cuts in
        # play order, normalized coords, layer manifest). Non-fatal: a build/store
        # failure must never block the finished export.
        try:
            clip_spec = build_clip_spec(
                video_id=video_id,
                clip=clip,
                edit=edit,
                aspect_ratio=payload.get("aspect_ratio"),
                preset_config=preset_config,
                text_sticker_config=text_sticker_config,
                branding_watermark=payload.get("branding_watermark"),
            )
        except Exception as spec_err:
            log.warning("ClipSpec build failed (non-fatal) for export %s: %s", export_id, spec_err)
            clip_spec = None

        update_export(export_id, status="done", r2_key=r2_key, r2_url=export_url,
                      updated_at=datetime.utcnow().isoformat(),
                      **({"clip_spec": clip_spec} if clip_spec else {}))
        finish_job(
            job_id, status="done",
            finished_at=datetime.utcnow().isoformat(),
            result={"r2_key": r2_key, "url": export_url},
        )
        log.info("Export done: %s → %s", export_id, export_url)

    except Exception as e:
        log.exception("Export failed for job %s", job_id)
        update_export(export_id, status="error", error_msg=str(e))
        finish_job(job_id, status="error", error_msg=str(e), finished_at=datetime.utcnow().isoformat())
        raise
    finally:
        for p in [portrait_local, rendered_local, *extra_tmp]:
            if p:
                Path(p).unlink(missing_ok=True)
