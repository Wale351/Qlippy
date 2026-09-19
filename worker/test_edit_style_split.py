"""
Tests for the edit/style split (two-phase render).

  handle_edit_clip  — expensive: download + stitch + face-tracked portrait,
                      caches the clean portrait on the clip_edits row, then
                      chains a style_clip job (queued row + runner.start_job).
  handle_style_clip — cheap: pulls the cached portrait from R2 and composites
                      styling. Must NOT re-download/re-track.

Run: .venv/bin/python test_edit_style_split.py
"""

from __future__ import annotations
import asyncio
import tempfile
from pathlib import Path
from unittest import mock

import jobs.export as jx
import runner


def test_edit_clip_caches_portrait_and_chains_style():
    calls: dict = {}

    async def fake_download(url, cuts, source_r2_key=None):
        calls["stitched"] = True
        return str(Path(tempfile.gettempdir()) / "_split_stitched.mp4")

    def fake_cut_and_portrait(stitched, ranges, out_path, **kw):
        Path(out_path).write_bytes(b"")

    def capture_start(job):
        calls["chained_style_job"] = job
        return True

    def record_upload(local, key, ctype):
        calls["upload_key"] = key

    def record_edit(edit_id, **kw):
        calls.setdefault("edit_updates", []).append((edit_id, kw))

    job = {"id": "job-1", "user_id": "u-1", "payload": {
        "edit_id": "edit-1", "export_id": "exp-1", "clip_id": "clip-1",
        "video_id": "vid-1", "aspect_ratio": "9:16", "preset_id": "preset-9",
    }}

    async def drive():
        with mock.patch.object(jx, "get_clip", lambda cid: {"cuts": [{"start": 0, "end": 5}]}), \
             mock.patch.object(jx, "get_video", lambda vid: {"youtube_url": "https://x"}), \
             mock.patch.object(jx, "get_transcript", lambda vid: None), \
             mock.patch.object(jx, "_build_diar_timeline", lambda t: []), \
             mock.patch.object(jx, "_remap_diar_timeline", lambda d, c: []), \
             mock.patch.object(jx, "download_and_stitch_cuts", fake_download), \
             mock.patch.object(jx, "cut_and_portrait", fake_cut_and_portrait), \
             mock.patch.object(jx, "upload_file", record_upload), \
             mock.patch.object(jx, "update_clip_edit", record_edit), \
             mock.patch.object(jx, "insert_job", lambda data: {"id": "style-job", **data}), \
             mock.patch.object(runner, "start_job", capture_start), \
             mock.patch.object(jx, "update_job", lambda *a, **k: None), \
             mock.patch.object(jx, "update_export", lambda *a, **k: None), \
             mock.patch.object(jx, "finish_job", lambda *a, **k: None):
            await jx.handle_edit_clip(job)

    asyncio.run(drive())

    # Expensive stage ran and cached the clean portrait under the edit id.
    assert calls.get("stitched") is True
    assert calls["upload_key"] == "clip_edits/edit-1/portrait.mp4"
    # clip_edits row was marked done with the portrait key.
    done = [u for u in calls["edit_updates"] if u[1].get("status") == "done"]
    assert done and done[0][1]["r2_key"] == "clip_edits/edit-1/portrait.mp4", calls["edit_updates"]
    # A style_clip job was chained, carrying the same export + edit + style config.
    chained = calls["chained_style_job"]
    assert chained["type"] == "style_clip"
    assert chained["status"] == "queued"     # restart-safe: row exists before it runs
    assert chained["payload"]["export_id"] == "exp-1"
    assert chained["payload"]["edit_id"] == "edit-1"
    assert chained["payload"]["preset_id"] == "preset-9"


def test_style_clip_reuses_cached_portrait_no_retrack():
    calls: dict = {}

    def fake_download_file(r2_key, local_path):
        calls["downloaded_key"] = r2_key
        Path(local_path).write_bytes(b"")

    not_called = mock.MagicMock()

    def record_export(export_id, **kw):
        calls.setdefault("export_updates", []).append(kw)

    # No preset / watermark / sticker / branding → style stage just re-wraps the
    # cached portrait, so we avoid the heavy render paths entirely.
    job = {"id": "job-2", "user_id": "u-1", "payload": {
        "export_id": "exp-2", "edit_id": "edit-1", "clip_id": "clip-1", "video_id": "vid-1",
    }}

    class _Proc:
        returncode = 1  # make the thumbnail ffmpeg step a graceful no-op

    async def drive():
        with mock.patch.object(jx, "get_clip_edit", lambda eid: {"r2_key": "clip_edits/edit-1/portrait.mp4"}), \
             mock.patch.object(jx, "get_clip", lambda cid: {"cuts": [{"start": 0, "end": 5}]}), \
             mock.patch.object(jx, "get_transcript", lambda vid: None), \
             mock.patch.object(jx, "download_file", fake_download_file), \
             mock.patch.object(jx, "download_and_stitch_cuts", not_called), \
             mock.patch.object(jx, "cut_and_portrait", not_called), \
             mock.patch.object(jx, "subprocess") as msub, \
             mock.patch.object(jx, "upload_file", lambda *a, **k: None), \
             mock.patch.object(jx, "upload_bytes", lambda *a, **k: None), \
             mock.patch.object(jx, "update_clip_thumbnail", lambda *a, **k: None), \
             mock.patch.object(jx, "public_url", lambda key: f"https://r2/{key}"), \
             mock.patch.object(jx, "update_job", lambda *a, **k: None), \
             mock.patch.object(jx, "update_export", record_export), \
             mock.patch.object(jx, "finish_job", lambda *a, **k: None):
            msub.run.return_value = _Proc()
            await jx.handle_style_clip(job)

    asyncio.run(drive())

    # The cached portrait was pulled; nothing was re-downloaded or re-tracked.
    assert calls["downloaded_key"] == "clip_edits/edit-1/portrait.mp4"
    assert not_called.call_count == 0, "style stage must not stitch or face-track"
    # Export reached done.
    assert any(u.get("status") == "done" for u in calls["export_updates"]), calls["export_updates"]


def test_style_clip_thumbnail_uses_clean_portrait_not_styled():
    """The clip thumbnail must come from the clean portrait, never the styled
    (captioned/stickered) output — it backs the re-clip design preview."""
    calls: dict = {}

    def fake_download_file(r2_key, local_path):
        calls["portrait_local"] = local_path  # this is the clean cached portrait
        Path(local_path).write_bytes(b"")

    def fake_burn_subtitles(input_path, output_path, **kw):
        calls["rendered_local"] = output_path  # the styled output
        Path(output_path).write_bytes(b"")

    class _Proc:
        returncode = 1  # graceful no-op upload, we only inspect the ffmpeg args

    def capture_run(args, **kw):
        if args[0] == "ffmpeg" and "-frames:v" in args:
            calls["thumb_input"] = args[args.index("-i") + 1]
        return _Proc()

    # A preset is present, so styling produces a distinct rendered file.
    job = {"id": "job-3", "user_id": "u-1", "payload": {
        "export_id": "exp-3", "edit_id": "edit-1", "clip_id": "clip-1", "video_id": "vid-1",
        "preset_config": {"hyperframes_component": "caption-highlight"},
    }}

    async def drive():
        with mock.patch.object(jx, "get_clip_edit", lambda eid: {"r2_key": "clip_edits/edit-1/portrait.mp4"}), \
             mock.patch.object(jx, "get_clip", lambda cid: {"cuts": [{"start": 0, "end": 5}]}), \
             mock.patch.object(jx, "get_transcript", lambda vid: {"raw_words": []}), \
             mock.patch.object(jx, "get_preset", lambda pid: None), \
             mock.patch.object(jx, "download_file", fake_download_file), \
             mock.patch.object(jx, "burn_subtitles", fake_burn_subtitles), \
             mock.patch.object(jx.subprocess, "run", capture_run), \
             mock.patch.object(jx, "upload_file", lambda *a, **k: None), \
             mock.patch.object(jx, "upload_bytes", lambda *a, **k: None), \
             mock.patch.object(jx, "update_clip_thumbnail", lambda *a, **k: None), \
             mock.patch.object(jx, "public_url", lambda key: f"https://r2/{key}"), \
             mock.patch.object(jx, "update_job", lambda *a, **k: None), \
             mock.patch.object(jx, "update_export", lambda *a, **k: None), \
             mock.patch.object(jx, "finish_job", lambda *a, **k: None):
            await jx.handle_style_clip(job)

    asyncio.run(drive())

    assert calls.get("rendered_local"), "styling should have produced a rendered output"
    assert calls["thumb_input"] == calls["portrait_local"], \
        f"thumbnail must come from the clean portrait, got {calls['thumb_input']}"
    assert calls["thumb_input"] != calls["rendered_local"], "thumbnail must not be the styled output"


if __name__ == "__main__":
    test_edit_clip_caches_portrait_and_chains_style()
    test_style_clip_reuses_cached_portrait_no_retrack()
    test_style_clip_thumbnail_uses_clean_portrait_not_styled()
    print("OK")
