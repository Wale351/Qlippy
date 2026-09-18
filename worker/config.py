"""
Central env/config seam for the worker.

load_dotenv() runs at import time so every entry point — main.py, the job
modules, or a test importing one of them directly — sees worker/.env without
depending on import order.
"""

from __future__ import annotations
import os
import shutil

from dotenv import load_dotenv

load_dotenv()

# Read leniently here so tests can import this module; main.py refuses to
# start if any of REQUIRED_ENV is missing (see missing_env).
WORKER_SECRET       = os.getenv("WORKER_SECRET", "")
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN", "")
HUGGINGFACE_TOKEN   = os.getenv("HUGGINGFACE_TOKEN", "")  # required only for diarization

# Model size → WhisperX model version tag
WHISPERX_MODEL_SIZES = {
    "base":   "base",
    "medium": "medium",
    "large":  "large-v2",
}


REQUIRED_ENV = (
    "WORKER_SECRET",
    "SUPABASE_URL",
    "SUPABASE_SERVICE_ROLE_KEY",
    "R2_ENDPOINT",
    "R2_ACCESS_KEY_ID",
    "R2_SECRET_ACCESS_KEY",
)


def missing_env() -> list[str]:
    return [k for k in REQUIRED_ENV if not os.getenv(k)]


# Binaries every job shells out to. Checked once at startup so a missing
# ffmpeg fails the boot with an install hint instead of the first export.
REQUIRED_BINARIES = ("ffmpeg", "ffprobe")

INSTALL_HINT = "macOS: brew install ffmpeg · Windows: winget install ffmpeg · Debian/Ubuntu: apt install ffmpeg"


def missing_binaries() -> list[str]:
    return [b for b in REQUIRED_BINARIES if shutil.which(b) is None]
