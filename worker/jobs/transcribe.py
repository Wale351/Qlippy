"""
transcribe job — yt-dlp download → WAV → Replicate WhisperX → transcript.
On success auto-chains an identify_clips job (queued row + immediate start,
so a crash between the two leaves a recoverable queued job, not a lost task).
"""

from __future__ import annotations
import asyncio
import json
import logging
import subprocess
import time
from datetime import datetime
from pathlib import Path

import httpx

from config import REPLICATE_API_TOKEN, HUGGINGFACE_TOKEN, WHISPERX_MODEL_SIZES
from sources import ensure_source, ensure_audio
from pipeline.clip_thumbnail import extract_face_from_image, extract_portrait_preview
from pipeline.portrait import bind_speakers_from_source
from pipeline.diarize import diarize_audio, attach_speakers_to_segments
from jobs.common import _build_diar_timeline, speaker_faces_key
from storage.r2 import upload_bytes, public_url, get_presigned_url
from db.supabase import (
    update_job, finish_job, update_video, get_video,
    insert_transcript, insert_job, get_user_profile, get_transcript,
)

log = logging.getLogger(__name__)


def _replicate_poll_timeout(duration_sec: float | None) -> float:
    """
    Transcription poll deadline scaled to content length: 30 min base
    (covers queueing + cold start), plus time proportional to the audio —
    WhisperX large runs roughly 10–30x realtime — capped at 3 hours.
    """
    base = 1800.0
    if not duration_sec or duration_sec <= 0:
        return base
    return min(max(base, duration_sec * 0.3), 3 * 3600.0)


def _transcribe_error_msg(e: Exception) -> str:
    msg = str(e)
    if "yt-dlp" in msg or "download" in msg.lower():
        return "Could not download the video. Check the URL is public and not age-restricted."
    if "empty" in msg.lower():
        return "Downloaded video was empty. The video may be unavailable or geo-restricted."
    if "audio" in msg.lower():
        return "Audio extraction failed. The video may have no audio track."
    if "replicate" in msg.lower() or "whisperx" in msg.lower():
        return f"Transcription service error: {msg}"
    return f"Transcription failed: {msg}"


def _flatten_segment_words(segments: list) -> list:
    words = []
    for seg in segments:
        words.extend(seg.get("words") or [])
    return words


async def _chain_identify_clips(transcribe_job: dict, video_id: str) -> None:
    """Create and start an identify_clips job immediately after transcription succeeds."""
    user_id = transcribe_job["user_id"]
    profile = get_user_profile(user_id)

    preferences: dict = {
        "min_duration_seconds": (profile or {}).get("clip_duration_min") or 30,
        "max_duration_seconds": (profile or {}).get("clip_duration_max") or 90,
        "max_clips": 10,
        "target_tones": (profile or {}).get("preferred_tones") or ["educational", "entertaining", "hook", "emotional", "cta"],
    }
    job_payload: dict = {"video_id": video_id, "preferences": preferences}

    if profile and (profile.get("niche") or profile.get("audience")):
        job_payload["niche_profile"] = {
            "niche":             profile.get("niche"),
            "audience":          profile.get("audience"),
            "hook_style":        profile.get("hook_style"),
            "clip_duration_min": profile.get("clip_duration_min"),
            "clip_duration_max": profile.get("clip_duration_max"),
            "preferred_tones":   profile.get("preferred_tones"),
        }

    identify_job = insert_job({
        "user_id":   user_id,
        "type":      "identify_clips",
        "entity_id": video_id,
        "status":    "queued",
        "payload":   job_payload,
    })

    update_video(video_id, status="identifying")
    # Late import (runner imports this module via the handler registry).
    # start_job claims the queued row and runs it here; if this process dies
    # first, the row is still 'queued' and the poll loop picks it up.
    import runner
    runner.start_job(identify_job)
    log.info("Auto-chained identify_clips job %s for video %s", identify_job["id"], video_id)


async def handle_transcribe(job: dict) -> None:
    """
    Full transcription flow:
      1. yt-dlp download → R2 (ensure_source)
      2. Extract 16kHz mono WAV → R2 (ensure_audio)
      3. Upload WAV to Replicate-accessible URL
      4. Start victor-upmeet/whisperx prediction
      5. Poll until done
      6. Store transcript in Supabase + R2
    payload: {"video_id", "youtube_url", "model_size"?, "reprocess"?}
    """
    payload      = job["payload"]
    video_id     = payload["video_id"]
    youtube_url  = payload["youtube_url"]
    model_size   = payload.get("model_size", "large")
    job_id       = job["id"]

    # Reuse an existing transcript unless a deliberate fresh transcription
    # was explicitly requested.
    if not payload.get("force_transcribe"):
        existing = get_transcript(video_id)

        if existing and existing.get("raw_words"):
            log.info(
                "Existing transcript found for video %s (%d words) — skipping transcription",
                video_id,
                len(existing.get("raw_words") or []),
            )

            finish_job(
                job_id,
                status="done",
                finished_at=datetime.utcnow().isoformat(),
                result={
                    "transcript_id": existing["id"],
                    "reused": True,
                },
            )

            await _chain_identify_clips(job, video_id)
            return

    update_job(job_id, status="processing", started_at=datetime.utcnow().isoformat())
    update_video(video_id, status="transcribing")

    last_error: Exception | None = None

    for attempt in range(1, 3):
        source_local = audio_local = None
        try:
            # Step 0: extract face from YouTube thumbnail (best-effort, once only)
            if attempt == 1:
                video_row = get_video(video_id)
                yt_thumb  = video_row.get("thumbnail_url")
                if yt_thumb:
                    try:
                        async with httpx.AsyncClient(timeout=20.0) as _c:
                            _resp = await _c.get(yt_thumb)
                            _resp.raise_for_status()
                        face_bytes = extract_face_from_image(_resp.content)
                        if face_bytes:
                            face_r2_key = f"faces/{video_id}.jpg"
                            upload_bytes(face_bytes, face_r2_key, content_type="image/jpeg")
                            update_video(video_id, face_url=public_url(face_r2_key))
                            log.info("Face extracted from thumbnail for video %s", video_id)
                    except Exception as _fe:
                        log.warning("Face extraction failed for video %s: %s", video_id, _fe)

            # Step 1: download source video
            source_local, source_r2_key = await ensure_source(video_id, youtube_url)

            # Step 1b: portrait preview — extract now so the Frame Designer
            # is usable before clip identification runs.
            # Use ffprobe to find a representative timestamp (~20% in, min 10 s).
            vid_dur = None  # also scales the Replicate poll deadline in Step 5
            try:
                dur_result = subprocess.run(
                    ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
                     "-of", "default=noprint_wrappers=1:nokey=1", source_local],
                    capture_output=True, text=True,
                )
                try:
                    vid_dur = float(dur_result.stdout.strip())
                except (ValueError, AttributeError):
                    vid_dur = 300.0

                update_video(video_id, duration_sec=int(vid_dur))

                preview_start = max(10.0, vid_dur * 0.2)
                preview_bytes = await asyncio.to_thread(
                    extract_portrait_preview, source_local, preview_start, preview_start + 30.0
                )
                if preview_bytes:
                    preview_r2_key = f"previews/{video_id}/portrait_preview.jpg"
                    upload_bytes(preview_bytes, preview_r2_key, content_type="image/jpeg")
                    update_video(video_id, portrait_preview_url=public_url(preview_r2_key))
                    log.info("Early portrait preview uploaded for video %s", video_id)
            except Exception as _pe:
                log.warning("Early portrait preview failed (non-fatal): %s", _pe)

            # Step 2: extract audio
            audio_local, audio_r2_key = await ensure_audio(video_id, source_local)

            # Step 3: local transcription with MLX Whisper
            import mlx_whisper

            local_model = "mlx-community/whisper-small-mlx"
            log.info("Starting local MLX Whisper transcription: %s", local_model)

            output = await asyncio.to_thread(
                mlx_whisper.transcribe,
                audio_local,
                path_or_hf_repo=local_model,
                word_timestamps=True,
                language="en",
                verbose=False,
            )

            segments = output.get("segments") or []

            # Local speaker diarization.
            # MLX Whisper provides timestamps but no speaker identities.
            try:
                log.info("Starting local speaker diarization")
                diar_timeline = await asyncio.to_thread(
                    diarize_audio,
                    audio_local,
                )
                segments = attach_speakers_to_segments(
                    segments,
                    diar_timeline,
                )
                speakers = sorted({speaker for _, _, speaker in diar_timeline})
                log.info(
                    "Speaker diarization complete: %d segments, %d speakers %s",
                    len(diar_timeline),
                    len(speakers),
                    speakers,
                )
            except Exception as diar_error:
                log.warning(
                    "Speaker diarization failed (non-fatal): %s",
                    diar_error,
                )

            raw_words = _flatten_segment_words(segments)
            language = output.get("language") or "en"
            full_text = (output.get("text") or "").strip()

            r2_key = f"transcripts/{video_id}/transcript.json"
            upload_bytes(
                json.dumps({
                    "language": language,
                    "raw_words": raw_words,
                    "segments": segments,
                }).encode(),
                r2_key,
                "application/json",
            )

            transcript = insert_transcript({
                "video_id": video_id,
                "user_id": job["user_id"],
                "language": language,
                "raw_words": raw_words,
                "segments": segments,
                "full_text": full_text,
                "r2_key": r2_key,
                "replicate_id": None,
                "model_size": "local-small",
            })

            log.info("Local transcription complete: %d words", len(raw_words))

            # Speaker → face map over the whole source, for the export's face
            # tracker. Learning it from a 30 s clip is fooled by every reaction
            # cutaway; over the full episode the real speaker wins. Source is
            # already local here, so this is the cheap moment to do it.
            try:
                diar_tl = _build_diar_timeline(transcript)
                if diar_tl:
                    faces = await asyncio.to_thread(bind_speakers_from_source, source_local, diar_tl)
                    if faces.get("speakers"):
                        await asyncio.to_thread(
                            upload_bytes, json.dumps(faces).encode(),
                            speaker_faces_key(video_id), "application/json",
                        )
            except Exception as _be:
                log.warning("Speaker→face binding failed (non-fatal, exports bind per clip): %s", _be)

            finish_job(
                job_id,
                status="done",
                finished_at=datetime.utcnow().isoformat(),
                result={"transcript_id": transcript["id"]},
            )
            log.info("Transcription done for video %s (%d words)", video_id, len(raw_words))
            last_error = None

        except Exception as e:
            last_error = e
            if attempt < 2:
                log.warning("Transcription attempt 1/2 failed, retrying in 5s: %s", e)
                await asyncio.sleep(5)
            else:
                log.exception("Transcription attempt 2/2 failed for job %s", job_id)
        finally:
            # Keep source_local in the persistent local cache.
            # Only temporary audio is removed.
            if audio_local:
                Path(audio_local).unlink(missing_ok=True)

        if last_error is None:
            break

    if last_error is not None:
        msg = _transcribe_error_msg(last_error)
        finish_job(job_id, status="error", error_msg=msg, finished_at=datetime.utcnow().isoformat())
        update_video(video_id, status="error", error_msg=msg)
        log.error("Transcription failed after 2 attempts for video %s: %s", video_id, msg)
        return

    # Auto-chain: fire identify_clips immediately — no user action required
    await _chain_identify_clips(job, video_id)
