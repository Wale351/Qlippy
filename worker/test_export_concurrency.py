"""
Regression test for the export-clip semaphore deadlock.

The old code acquired `_export_semaphore` twice per job (an outer wrapper plus an
inner gate around the portrait step). Being non-reentrant, two concurrent jobs —
or even one job under a gate of size 1 — deadlocked forever, leaving exports
pinned at the UI's 90% ceiling. This test runs jobs through the gate and asserts
they all reach "done" instead of hanging.

Run: .venv/bin/python test_export_concurrency.py
"""

from __future__ import annotations
import asyncio
import tempfile
import time
from pathlib import Path
from unittest import mock

import jobs.export as jx
import runner


def _run_two_concurrent_exports() -> dict:
    finished: dict[str, str] = {}

    async def fake_download(url, cuts, source_r2_key=None):
        return str(Path(tempfile.gettempdir()) / "_nonexistent_stitched.mp4")

    def fake_cut_and_portrait(stitched, ranges, out_path, **kw):
        # Simulate the CPU-heavy gated step; create the expected output file.
        time.sleep(0.05)
        Path(out_path).write_bytes(b"")

    def record_finish(job_id, **kw):
        finished[job_id] = kw.get("status")

    jobs = [
        {"id": f"job-{i}", "user_id": "u-1", "payload": {
            "edit_id": f"edit-{i}", "export_id": f"exp-{i}", "clip_id": f"clip-{i}", "video_id": "vid-1",
        }}
        for i in range(2)
    ]

    async def drive():
        # Gate of size 1 — under the old double-acquire even a single job deadlocks.
        # The CPU-heavy portrait step (and its semaphore) now live in handle_edit_clip.
        with mock.patch.object(jx, "_export_semaphore", asyncio.Semaphore(1)), \
             mock.patch.object(jx, "get_clip", lambda cid: {"cuts": [{"start": 0, "end": 5}]}), \
             mock.patch.object(jx, "get_video", lambda vid: {"youtube_url": "https://x"}), \
             mock.patch.object(jx, "get_transcript", lambda vid: None), \
             mock.patch.object(jx, "_build_diar_timeline", lambda t: []), \
             mock.patch.object(jx, "_remap_diar_timeline", lambda d, c: []), \
             mock.patch.object(jx, "download_and_stitch_cuts", fake_download), \
             mock.patch.object(jx, "cut_and_portrait", fake_cut_and_portrait), \
             mock.patch.object(jx, "upload_file", lambda *a, **k: None), \
             mock.patch.object(jx, "update_clip_edit", lambda *a, **k: None), \
             mock.patch.object(jx, "insert_job", lambda data: {"id": "style-job", **data}), \
             mock.patch.object(runner, "start_job", lambda job: True), \
             mock.patch.object(jx, "public_url", lambda key: f"https://r2/{key}"), \
             mock.patch.object(jx, "update_job", lambda *a, **k: None), \
             mock.patch.object(jx, "update_export", lambda *a, **k: None), \
             mock.patch.object(jx, "finish_job", record_finish):
            await asyncio.wait_for(
                asyncio.gather(*(jx.handle_edit_clip(j) for j in jobs)),
                timeout=10,
            )

    asyncio.run(drive())
    return finished


def test_concurrent_exports_do_not_deadlock():
    finished = _run_two_concurrent_exports()
    assert finished == {"job-0": "done", "job-1": "done"}, finished


if __name__ == "__main__":
    test_concurrent_exports_do_not_deadlock()
    print("OK")
