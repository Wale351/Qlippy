"""
Job execution engine.

One path for every job regardless of origin — HTTP trigger, in-process
chain, or the recovery poll loop:

  start_job(row) → claim_job() atomically flips queued/error → processing
                 → exactly one worker wins → handler runs as a strongly-
                   referenced background task.

Because chained jobs are inserted as 'queued' rows before they run, a crash
at any point leaves either a 'queued' row (recovered by the poll loop) or a
'processing' row (marked errored by the startup reaper) — never a silently
lost in-memory task.
"""

from __future__ import annotations
import asyncio
import gc
import logging
import os

from db.supabase import claim_job, get_queued_jobs, finish_job
from pipeline.memlog import log_rss
from jobs import HANDLERS

log = logging.getLogger(__name__)

# 0 disables the recovery poll loop (then only HTTP triggers start jobs).
POLL_INTERVAL_S = float(os.getenv("JOB_POLL_INTERVAL", "10"))

# Strong references to running tasks. asyncio only keeps a weak reference to
# a bare create_task(), so a long-running job could be garbage-collected
# mid-await, leaving its video/export pinned mid-progress forever.
_tasks: set[asyncio.Task] = set()
_service_tasks: set[asyncio.Task] = set()
_active_keys: set[tuple[str, str]] = set()


def _spawn(coro) -> asyncio.Task:
    task = asyncio.create_task(coro)
    _tasks.add(task)
    task.add_done_callback(_tasks.discard)
    return task


def _spawn_service(coro) -> asyncio.Task:
    task = asyncio.create_task(coro)
    _service_tasks.add(task)
    task.add_done_callback(_service_tasks.discard)
    return task


def active_jobs() -> int:
    return len(_tasks)


async def execute(job: dict) -> None:
    """Run a claimed job's handler with RSS logging at the boundaries
    (separates per-job growth from climb-and-stay, and Python-object
    retention from native memory)."""
    handler = HANDLERS.get(job["type"])
    if handler is None:
        from datetime import datetime
        log.error("Unknown job type %r for job %s", job["type"], job["id"])
        finish_job(job["id"], status="error", error_msg=f"Unknown job type: {job['type']}",
                   finished_at=datetime.utcnow().isoformat())
        return
    tag = f"{job['type']}:{job['id']}"
    log_rss(f"{tag} start")
    key = (job["type"], str(job.get("entity_id") or ""))
    try:
        await handler(job)
    finally:
        _active_keys.discard(key)
        log_rss(f"{tag} end")
        gc.collect()
        log_rss(f"{tag} post-gc")


def start_job(job: dict) -> bool:
    """
    Atomically claim a queued/errored job and run it in the background.
    Returns True if this process won the claim, False if another worker
    (or the poll loop vs. an HTTP trigger) got there first.
    """
    key = (job["type"], str(job.get("entity_id") or ""))

    # Only one transcription / clip-analysis job per video at a time.
    if job["type"] in {"transcribe", "identify_clips"} and key in _active_keys:
        finish_job(
            job["id"],
            status="cancelled",
            error_msg="Duplicate job skipped because this video is already processing.",
        )
        log.info("Skipped duplicate job %s for %s", job["id"], key)
        return False

    claimed = claim_job(job["id"])
    if not claimed:
        return False

    _active_keys.add(key)
    _spawn(execute(claimed))
    return True


async def poll_queued_forever() -> None:
    """
    Recovery loop: pick up 'queued' jobs nobody is running — chained jobs
    orphaned by a restart, or trigger HTTP calls that never landed. Claiming
    is atomic, so racing the HTTP trigger (or another instance) is harmless.
    Only 'queued' rows are polled — errored jobs stay terminal until the
    user retries them.
    """
    if POLL_INTERVAL_S <= 0:
        log.info("Job poll loop disabled (JOB_POLL_INTERVAL=0)")
        return
    log.info("Job poll loop running every %.0fs", POLL_INTERVAL_S)
    while True:
        await asyncio.sleep(POLL_INTERVAL_S)
        try:
            for row in get_queued_jobs(limit=20):
                if start_job(row):
                    log.info("Poll loop picked up queued job %s (%s)", row["id"], row["type"])
        except Exception as e:
            log.warning("Job poll failed (non-fatal): %s", e)
