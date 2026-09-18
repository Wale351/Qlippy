"""
Supabase client for the worker.
Uses service-role key (bypasses RLS) — worker has full access.
"""

from __future__ import annotations
import os
import time
from supabase import create_client, Client

_client: Client | None = None


def get_db() -> Client:
    global _client
    if _client is None:
        _client = create_client(
            os.environ["SUPABASE_URL"],
            os.environ["SUPABASE_SERVICE_ROLE_KEY"],  # service role — bypasses RLS
        )
    return _client


# ---- Convenience helpers ----

def update_job(job_id: str, **fields) -> None:
    get_db().table("jobs").update(fields).eq("id", job_id).execute()


def finish_job(job_id: str, **fields) -> None:
    """Terminal job update (done/error) that never overwrites a cancellation."""
    get_db().table("jobs").update(fields).eq("id", job_id).neq("status", "cancelled").execute()


def update_video(video_id: str, **fields) -> None:
    get_db().table("videos").update(fields).eq("id", video_id).execute()


def update_export(export_id: str, **fields) -> None:
    get_db().table("exports").update(fields).eq("id", export_id).execute()


def get_job(job_id: str) -> dict:
    res = get_db().table("jobs").select("*").eq("id", job_id).single().execute()
    return res.data


def get_video(video_id: str) -> dict:
    res = get_db().table("videos").select("*").eq("id", video_id).single().execute()
    return res.data


def get_transcript(video_id: str) -> dict | None:
    res = get_db().table("transcripts").select("*").eq("video_id", video_id).maybe_single().execute()
    return res.data if res else None


def get_clip(clip_id: str) -> dict:
    res = get_db().table("clips").select("*").eq("id", clip_id).single().execute()
    return res.data


def get_export(export_id: str) -> dict:
    res = get_db().table("exports").select("*").eq("id", export_id).single().execute()
    return res.data


def get_clip_edit(edit_id: str) -> dict | None:
    res = get_db().table("clip_edits").select("*").eq("id", edit_id).maybe_single().execute()
    return res.data if res else None


def update_clip_edit(edit_id: str, **fields) -> None:
    get_db().table("clip_edits").update(fields).eq("id", edit_id).execute()


def get_preset(preset_id: str) -> dict | None:
    if not preset_id:
        return None
    res = get_db().table("presets").select("*").eq("id", preset_id).maybe_single().execute()
    return res.data if res else None


def delete_clips_for_video(video_id: str) -> None:
    get_db().table("clips").delete().eq("video_id", video_id).execute()


def get_clip_ids_for_video(video_id: str) -> list[str]:
    res = get_db().table("clips").select("id").eq("video_id", video_id).execute()
    return [c["id"] for c in (res.data or [])]


def delete_clips_by_ids(clip_ids: list[str]) -> None:
    if clip_ids:
        get_db().table("clips").delete().in_("id", clip_ids).execute()


def get_max_clip_generation(video_id: str) -> int:
    res = (
        get_db().table("clips").select("generation")
        .eq("video_id", video_id)
        .order("generation", desc=True).limit(1).execute()
    )
    return res.data[0]["generation"] if res.data else 0


def get_exports_for_video(video_id: str) -> list[dict]:
    """Return all export rows (id, r2_key) for clips belonging to this video."""
    clips = get_db().table("clips").select("id").eq("video_id", video_id).execute().data
    if not clips:
        return []
    clip_ids = [c["id"] for c in clips]
    res = get_db().table("exports").select("id, r2_key").in_("clip_id", clip_ids).execute()
    return res.data or []


def delete_jobs_for_exports(export_ids: list[str]) -> None:
    """Delete jobs whose entity_id matches any of the given export IDs."""
    if not export_ids:
        return
    get_db().table("jobs").delete().in_("entity_id", export_ids).execute()


def insert_clips(clips: list[dict]) -> list[dict]:
    res = get_db().table("clips").insert(clips).execute()
    return res.data


def update_clip_thumbnail(clip_id: str, thumbnail_url: str) -> None:
    get_db().table("clips").update({"thumbnail_url": thumbnail_url}).eq("id", clip_id).execute()


def insert_transcript(data: dict) -> dict:
    """
    Save transcript safely.
    video_id is unique, so retries cannot create duplicate transcripts.
    """
    last_error = None

    for attempt in range(3):
        try:
            res = (
                get_db()
                .table("transcripts")
                .upsert(data, on_conflict="video_id")
                .execute()
            )
            if res.data:
                return res.data[0]
        except Exception as e:
            last_error = e
            if attempt < 2:
                time.sleep(2 * (attempt + 1))

    # The DB may have committed even if the response connection dropped.
    try:
        existing = get_transcript(data["video_id"])
        if existing:
            return existing
    except Exception:
        pass

    if last_error:
        raise last_error
    raise RuntimeError("Transcript save failed")


def insert_job(data: dict) -> dict:
    res = get_db().table("jobs").insert(data).execute()
    return res.data[0]


def claim_job(job_id: str) -> dict | None:
    """
    Atomically flip a queued/errored job to 'processing' and return the row.
    A single conditional UPDATE — if two workers race, exactly one gets the
    row back; the loser gets None. 'error' is claimable so the retry flow
    (re-trigger an errored job) keeps working.
    """
    from datetime import datetime
    res = (
        get_db().table("jobs")
        .update({"status": "processing", "started_at": datetime.utcnow().isoformat()})
        .eq("id", job_id)
        .in_("status", ["queued", "error"])
        .execute()
    )
    rows = res.data or []
    return rows[0] if rows else None


def get_queued_jobs(limit: int = 20) -> list[dict]:
    """Oldest queued jobs first — feed for the recovery poll loop."""
    res = (
        get_db().table("jobs").select("*")
        .eq("status", "queued")
        .order("queued_at")
        .limit(limit)
        .execute()
    )
    return res.data or []


def get_user_profile(user_id: str) -> dict | None:
    try:
        res = get_db().table("user_profiles").select("*").eq("user_id", user_id).maybe_single().execute()
        return res.data if res else None
    except Exception:
        return None


def get_reapable_jobs() -> list[dict]:
    # Orphans: jobs left mid-flight ('processing') by a dead worker. 'queued'
    # jobs are NOT orphans — the runner poll loop claims and runs them, which
    # covers restarts in a chain window and failed triggerJob HTTP calls.
    return get_db().table("jobs").select("*").eq("status", "processing").execute().data or []


def reap_orphaned_jobs() -> int:
    """
    Mark every orphaned 'processing' job as errored and surface the failure
    on its entity (video/export). Meant to run once at worker startup: jobs
    run as in-memory tasks, so a job still 'processing' at boot was owned by
    a dead worker and will never finish — without this, its video/export
    stays pinned at "Finding clips…"/"Transcribing…"/90% forever.
    Deliberately not auto-requeued: a crash loop must not re-spend LLM /
    Replicate money unattended; the user retries from the UI.

    Assumes a single worker process. Returns the number of jobs reaped.
    """
    from datetime import datetime

    orphans = get_reapable_jobs()
    now = datetime.utcnow().isoformat()
    msg = "Interrupted by a worker restart — please retry."
    for job in orphans:
        finish_job(job["id"], status="error", error_msg=msg, finished_at=now)
        jtype   = job.get("type")
        payload = job.get("payload") or {}
        if jtype in ("transcribe", "identify_clips"):
            vid = payload.get("video_id") or job.get("entity_id")
            if vid:
                update_video(vid, status="error", error_msg=msg)
        elif jtype in ("export_clip", "style_clip"):
            eid = payload.get("export_id") or job.get("entity_id")
            if eid:
                update_export(eid, status="error", error_msg=msg)
        elif jtype == "edit_clip":
            if payload.get("edit_id"):
                update_clip_edit(payload["edit_id"], status="error", error_msg=msg)
            eid = payload.get("export_id")
            if eid:
                update_export(eid, status="error", error_msg=msg)
    return len(orphans)
