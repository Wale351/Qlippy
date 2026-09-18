"""
Clippy Worker — FastAPI shell.

Job handlers live in jobs/ (one module per type), source download helpers in
sources.py, and claim/dispatch/poll in runner.py. This module only owns the
HTTP surface and process lifecycle:
  POST /jobs/run — claim + start a job by id (called by the Next.js app)
  GET  /status   — active/queued job counts + RSS (authenticated)
  GET  /health   — liveness probe
"""

from __future__ import annotations
import logging
import os

from fastapi import FastAPI, HTTPException, Header, Request
from pydantic import BaseModel

from config import WORKER_SECRET, INSTALL_HINT, missing_binaries, missing_env
import runner
from cleanup import schedule_daily_cleanup
from security import is_blocked, register_auth_failure
from pipeline.memlog import rss_mb
from db.supabase import get_job, get_queued_jobs, reap_orphaned_jobs

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

app = FastAPI(title="Clippy Worker", version="2.0.0")


# ============================================================
# Auth middleware
# ============================================================

def verify_worker_secret(x_worker_secret: str = Header(...)) -> None:
    if x_worker_secret != WORKER_SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized")


@app.middleware("http")
async def auth_bruteforce_guard(request: Request, call_next):
    """Block IPs that keep failing worker-secret auth (bot/scanner traffic)."""
    # Prefer x-real-ip (single proxy-set value) over X-Forwarded-For, which is a
    # client-suppliable list that could be padded to forge a fresh IP per request
    # and evade the per-IP lockout. Fall back to the last XFF hop (appended by the
    # trusted proxy), then the socket peer.
    real = (request.headers.get("x-real-ip") or "").strip()
    if real:
        ip = real
    else:
        hops = [h.strip() for h in (request.headers.get("x-forwarded-for") or "").split(",") if h.strip()]
        ip = (hops[-1] if hops else "") or (request.client.host if request.client else "unknown")
    if is_blocked(ip):
        from fastapi.responses import JSONResponse
        return JSONResponse({"detail": "Too many failed attempts"}, status_code=429)
    response = await call_next(request)
    if response.status_code == 401:
        register_auth_failure(ip)
    return response


# ============================================================
# Endpoints
# ============================================================

class JobTrigger(BaseModel):
    job_id: str


@app.post("/jobs/run")
async def run_job(trigger: JobTrigger, x_worker_secret: str = Header(...)):
    verify_worker_secret(x_worker_secret)
    job = get_job(trigger.job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["type"] not in runner.HANDLERS:
        raise HTTPException(status_code=400, detail=f"Unknown job type: {job['type']}")
    # Atomic claim — if another instance (or the poll loop) already took it,
    # report the conflict instead of double-running.
    if not runner.start_job(job):
        raise HTTPException(status_code=409, detail=f"Job already {job['status']}")
    return {"status": "accepted", "job_id": job["id"]}


@app.get("/status")
async def status(x_worker_secret: str = Header(...)):
    verify_worker_secret(x_worker_secret)
    return {
        "active_jobs": runner.active_jobs(),
        "queued_jobs": len(get_queued_jobs(limit=100)),
        "rss_mb":      round(rss_mb()),
        "export_concurrency": int(os.getenv("EXPORT_CONCURRENCY", "2")),
    }


@app.get("/health")
async def health():
    return {"status": "ok", "version": "2.0.0"}


# ============================================================
# Lifecycle
# ============================================================

@app.on_event("startup")
async def _startup() -> None:
    missing = missing_env()
    if missing:
        raise RuntimeError(f"Missing required variables in worker/.env: {', '.join(missing)}. See docs/SETUP.md")
    missing = missing_binaries()
    if missing:
        raise RuntimeError(f"Required binaries not on PATH: {', '.join(missing)}. {INSTALL_HINT}")

    try:
        reaped = reap_orphaned_jobs()
        if reaped:
            log.warning("Startup reaper: marked %d orphaned 'processing' job(s) as error", reaped)
    except Exception as e:
        log.warning("Startup reaper failed (non-fatal): %s", e)

    runner._spawn_service(runner.poll_queued_forever())

    if os.getenv("DAILY_CLEANUP_ENABLED", "false").lower() == "true":
        runner._spawn_service(schedule_daily_cleanup())
        log.info("Daily cleanup scheduler enabled — runs at midnight UTC")
    else:
        log.info("Daily cleanup disabled (set DAILY_CLEANUP_ENABLED=true to enable)")
