"""
Source acquisition — yt-dlp downloads and audio extraction.

ensure_* helpers are idempotent: they cache into R2 and reuse the cached
object on subsequent calls, so retries and multi-job access never
re-download from YouTube.
"""

from __future__ import annotations
import asyncio
import base64
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from storage.r2 import upload_file, download_file, key_exists

# Run yt-dlp as a module of *this* interpreter, never a bare "yt-dlp" from
# PATH: that resolved to a stale system copy once and 403'd every download.
YTDLP_CMD = [sys.executable, "-m", "yt_dlp"]
from db.supabase import update_video

log = logging.getLogger(__name__)

LOCAL_CACHE_DIR = Path.home() / ".clippy-cache"

def cached_source_path(video_id: str) -> str:
    folder = LOCAL_CACHE_DIR / "sources" / video_id
    folder.mkdir(parents=True, exist_ok=True)
    return str(folder / "source.mp4")


def ytdlp_proxy_args() -> list[str]:
    """Route yt-dlp through a proxy (e.g. rotating residential) when YT_PROXY is set.
    Needed in production: YouTube blocks datacenter IPs with bot checks."""
    proxy = os.getenv("YT_PROXY")
    return ["--proxy", proxy] if proxy else []


_yt_cookies_path: str | None = None


def ytdlp_cookies_args() -> list[str]:
    """Pass a cookies file to yt-dlp when YT_COOKIES_B64 is set (base64-encoded
    cookies.txt from a logged-in YouTube account). Helps avoid YouTube's
    'Sign in to confirm you're not a bot' challenge on datacenter IPs."""
    global _yt_cookies_path
    b64 = os.getenv("YT_COOKIES_B64")
    if not b64:
        return []
    if _yt_cookies_path is None:
        _yt_cookies_path = tempfile.mktemp(suffix="_cookies.txt")
        Path(_yt_cookies_path).write_bytes(base64.b64decode(b64))
    return ["--cookies", _yt_cookies_path]


async def ensure_source(video_id: str, youtube_url: str) -> tuple[str, str]:
    """
    Download source video once, cache in R2.
    Returns (local_temp_path, r2_key).
    Idempotent — if already in R2 (and non-empty), downloads from there instead.
    """
    r2_key = f"sources/{video_id}/source.mp4"
    local_path = cached_source_path(video_id)

    if Path(local_path).exists() and Path(local_path).stat().st_size > 0:
        log.info("Source already cached locally: %s", local_path)
    elif await asyncio.to_thread(key_exists, r2_key) and await asyncio.to_thread(_r2_size, r2_key) > 0:
        log.info("Source already in R2, downloading once to local cache: %s", r2_key)
        await asyncio.to_thread(download_file, r2_key, local_path)
    else:
        log.info("Downloading source from YouTube: %s", youtube_url)
        result = await asyncio.to_thread(
            subprocess.run,
            [
                *YTDLP_CMD,
                "--format", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
                "--merge-output-format", "mp4",
                "--output", local_path,
                "--no-playlist",
                "--no-part",
                "--js-runtimes", "node",
                "--remote-components", "ejs:github",
                *ytdlp_proxy_args(),
                *ytdlp_cookies_args(),
                youtube_url,
            ],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"yt-dlp failed:\n{result.stderr[-2000:]}")

        size = Path(local_path).stat().st_size if Path(local_path).exists() else 0
        if size == 0:
            raise RuntimeError(f"yt-dlp produced an empty file for {youtube_url}")

        log.info("Downloaded %s (%.1f MB)", youtube_url, size / 1_000_000)
        await asyncio.to_thread(upload_file, local_path, r2_key, "video/mp4")
        update_video(video_id, source_r2_key=r2_key)

    return local_path, r2_key


async def download_and_stitch_cuts(youtube_url: str, cuts: list[dict]) -> str:
    """
    Download only the cut time ranges from YouTube and stitch into one landscape MP4.
    Returns a local temp path. Caller is responsible for cleanup.
    """
    segment_paths: list[str] = []
    try:
        for cut in cuts:
            start = float(cut["start"])
            end   = float(cut["end"])
            out   = tempfile.mktemp(suffix=".mp4")
            segment_paths.append(out)

            last_err: Exception | None = None
            for attempt in range(1, 4):
                Path(out).unlink(missing_ok=True)
                result = await asyncio.to_thread(
                    subprocess.run,
                    [
                        *YTDLP_CMD,
                        "--format", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
                        "--merge-output-format", "mp4",
                        "--download-sections", f"*{start}-{end}",
                        "--force-keyframes-at-cuts",
                        "--output", out,
                        "--no-playlist",
                        "--no-part",
                        "--socket-timeout", "30",
                        "--retries", "3",
                        "--js-runtimes", "node",
                        "--remote-components", "ejs:github",
                        *ytdlp_proxy_args(),
                        *ytdlp_cookies_args(),
                        youtube_url,
                    ],
                    capture_output=True, text=True,
                )
                size = Path(out).stat().st_size if Path(out).exists() else 0
                if result.returncode == 0 and size > 0:
                    last_err = None
                    break
                last_err = RuntimeError(
                    f"yt-dlp failed for section {start}-{end} (attempt {attempt}/3):\n{result.stderr[-2000:]}"
                )
                log.warning("Section download attempt %d/3 failed: %s", attempt, result.stderr[-300:])
                if attempt < 3:
                    await asyncio.sleep(5 * attempt)

            if last_err:
                raise last_err
            log.info("Downloaded section %.1fs-%.1fs (%.1f MB)", start, end, size / 1e6)

        if len(segment_paths) == 1:
            return segment_paths.pop()

        stitched = tempfile.mktemp(suffix=".mp4")
        list_file = tempfile.mktemp(suffix=".txt")
        try:
            with open(list_file, "w") as f:
                for p in segment_paths:
                    f.write(f"file '{p}'\n")
            # Audio is re-encoded (video stream-copied): the sections are
            # independently encoded files, and concat -c copy lets per-file AAC
            # priming/padding accumulate as audio drift at each joint.
            await asyncio.to_thread(
                subprocess.run,
                ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_file,
                 "-c:v", "copy",
                 "-af", "aresample=async=1000:first_pts=0", "-c:a", "aac",
                 stitched],
                capture_output=True, check=True,
            )
        finally:
            Path(list_file).unlink(missing_ok=True)
        return stitched
    finally:
        for p in segment_paths:
            Path(p).unlink(missing_ok=True)


def _r2_size(r2_key: str) -> int:
    try:
        from storage.r2 import _get_client, BUCKET
        head = _get_client().head_object(Bucket=BUCKET, Key=r2_key)
        return head["ContentLength"]
    except Exception:
        return 0


async def ensure_audio(video_id: str, source_local: str) -> tuple[str, str]:
    """
    Extract 16kHz mono WAV audio from source video, cache in R2.
    Returns (local_audio_path, audio_r2_key).
    """
    r2_key = f"sources/{video_id}/audio.wav"
    audio_local = tempfile.mktemp(suffix=".wav")

    if await asyncio.to_thread(key_exists, r2_key):
        await asyncio.to_thread(download_file, r2_key, audio_local)
    else:
        result = await asyncio.to_thread(
            subprocess.run,
            [
                "ffmpeg", "-y", "-i", source_local,
                "-ac", "1", "-ar", "16000",
                "-vn", audio_local,
            ],
            capture_output=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg audio extract failed (exit {result.returncode}):\n"
                + result.stderr.decode(errors="replace")[-2000:]
            )
        await asyncio.to_thread(upload_file, audio_local, r2_key, "audio/wav")
        update_video(video_id, audio_r2_key=r2_key)

    return audio_local, r2_key
