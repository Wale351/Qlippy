"""
Tests for the job runner (claim → dispatch → poll recovery).

The DB is mocked at the module seam (runner.claim_job / runner.get_queued_jobs).
Run: .venv/bin/python test_runner.py
"""

from __future__ import annotations
import asyncio
import os
from unittest import mock

os.environ.setdefault("WORKER_SECRET", "test")

import runner


def test_start_job_runs_handler_when_claim_wins():
    """A won claim dispatches the claimed row (not the pre-claim row) to its handler."""
    ran: dict = {}

    async def fake_handler(job):
        ran["job"] = job

    claimed_row = {"id": "j1", "type": "transcribe", "status": "processing"}

    async def drive():
        runner._tasks.clear()
        with mock.patch.object(runner, "claim_job", lambda jid: claimed_row), \
             mock.patch.dict(runner.HANDLERS, {"transcribe": fake_handler}, clear=False), \
             mock.patch.object(runner, "log_rss", lambda *a, **k: 0.0):
            won = runner.start_job({"id": "j1", "type": "transcribe", "status": "queued"})
            assert won is True
            # drain the spawned task
            await asyncio.gather(*list(runner._tasks))

    asyncio.run(drive())
    assert ran["job"] is claimed_row, "handler must receive the claimed (processing) row"


def test_start_job_noop_when_claim_lost():
    """If another worker already claimed the job, start_job does nothing."""
    called = mock.MagicMock()

    with mock.patch.object(runner, "claim_job", lambda jid: None), \
         mock.patch.dict(runner.HANDLERS, {"transcribe": called}, clear=False):
        runner._tasks.clear()
        won = runner.start_job({"id": "j1", "type": "transcribe", "status": "queued"})

    assert won is False
    assert called.call_count == 0
    assert not runner._tasks, "no task should be spawned on a lost claim"


def test_execute_unknown_type_marks_error():
    """An unregistered job type is finished as error rather than raising into the loop."""
    finished: dict = {}

    with mock.patch.object(runner, "finish_job", lambda jid, **kw: finished.update({jid: kw})), \
         mock.patch.object(runner, "log_rss", lambda *a, **k: 0.0):
        asyncio.run(runner.execute({"id": "jX", "type": "nope"}))

    assert finished["jX"]["status"] == "error"
    assert "Unknown job type" in finished["jX"]["error_msg"]


def test_poll_picks_up_queued_jobs_once():
    """The poll loop claims each queued row via start_job exactly once per pass."""
    started: list = []

    # One pass: return two queued rows, then stop the loop by raising CancelledError
    # out of the sleep after the first iteration.
    rows = [{"id": "a", "type": "transcribe"}, {"id": "b", "type": "cut_clip"}]
    sleeps = {"n": 0}

    async def fake_sleep(_):
        sleeps["n"] += 1
        if sleeps["n"] >= 2:      # first sleep enters loop; second ends it
            raise asyncio.CancelledError

    async def drive():
        with mock.patch.object(runner, "POLL_INTERVAL_S", 10), \
             mock.patch.object(runner, "get_queued_jobs", lambda limit=20: rows), \
             mock.patch.object(runner, "start_job", lambda row: started.append(row["id"]) or True), \
             mock.patch.object(runner.asyncio, "sleep", fake_sleep):
            try:
                await runner.poll_queued_forever()
            except asyncio.CancelledError:
                pass

    asyncio.run(drive())
    assert started == ["a", "b"], started


def test_poll_disabled_when_interval_zero():
    """JOB_POLL_INTERVAL=0 disables the recovery loop entirely."""
    async def drive():
        with mock.patch.object(runner, "POLL_INTERVAL_S", 0):
            # returns immediately without ever touching the DB
            await runner.poll_queued_forever()

    asyncio.run(drive())


if __name__ == "__main__":
    test_start_job_runs_handler_when_claim_wins()
    test_start_job_noop_when_claim_lost()
    test_execute_unknown_type_marks_error()
    test_poll_picks_up_queued_jobs_once()
    test_poll_disabled_when_interval_zero()
    print("OK")
