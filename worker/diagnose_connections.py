"""
Connection diagnostics for every external service the pipeline touches.
Run: .venv/bin/python diagnose_connections.py

Each check is isolated and timed. PASS/FAIL + a one-line detail per service.
Read-only except R2, which does a tiny put/get/delete round-trip on a temp key.
"""

from __future__ import annotations
import os
import sys
import time
import json
import subprocess
import logging

logging.disable(logging.CRITICAL)  # silence app/httpx noise

from dotenv import load_dotenv
load_dotenv()

import httpx

RESULTS: list[tuple[str, bool, str]] = []


def check(name):
    def deco(fn):
        t = time.monotonic()
        try:
            detail = fn()
            ok = True
        except Exception as e:
            detail = f"{type(e).__name__}: {e}"
            ok = False
        ms = int((time.monotonic() - t) * 1000)
        RESULTS.append((name, ok, f"{detail}  [{ms}ms]"))
        return fn
    return deco


@check("Supabase (Postgres/REST)")
def _supabase():
    from db.supabase import get_db
    r = get_db().table("jobs").select("id", count="exact").limit(1).execute()
    return f"reachable, jobs count={r.count}"


@check("Cloudflare R2 (storage round-trip)")
def _r2():
    from storage.r2 import upload_bytes, download_file, delete_keys, BUCKET
    import tempfile
    key = f"_diag/conn_test_{int(time.time())}.txt"
    payload = b"clipfactory-conn-test"
    upload_bytes(payload, key, content_type="text/plain")
    local = tempfile.mktemp(suffix=".txt")
    download_file(key, local)
    got = open(local, "rb").read()
    delete_keys([key])
    os.unlink(local)
    assert got == payload, "round-trip mismatch"
    return f"put+get+delete ok on bucket '{BUCKET}'"


@check("Ollama local LLM")
def _ollama():
    r = httpx.get("http://127.0.0.1:11434/api/tags", timeout=10)
    r.raise_for_status()
    models = [
        m.get("name", "")
        for m in r.json().get("models", [])
    ]
    return "available models: " + ", ".join(models)


def _ytdlp_probe(url: str) -> str:
    proxy = os.getenv("YT_PROXY")
    cookies = "set" if os.getenv("YT_COOKIES_B64") else "none"
    args = [sys.executable, "-m", "yt_dlp", "--simulate", "--no-warnings", "--print", "id"]
    if proxy:
        args += ["--proxy", proxy]
    args.append(url)
    p = subprocess.run(args, capture_output=True, text=True, timeout=60)
    if p.returncode != 0:
        err = p.stderr.strip().splitlines()[-1] if p.stderr.strip() else "unknown"
        return f"FAILED (proxy={'set' if proxy else 'none'} cookies={cookies}): {err[:160]}"
    return f"ok id={p.stdout.strip()} (proxy={'set' if proxy else 'none'} cookies={cookies})"


@check("YouTube via yt-dlp (generic public video)")
def _ytdlp_generic():
    return _ytdlp_probe("https://www.youtube.com/watch?v=jNQXAC9V6Rk")


@check("YouTube via yt-dlp (a queued video's URL)")
def _ytdlp_real():
    url = os.environ.get("DIAG_YT_URL", "https://youtu.be/8e7_eg-MJLY")
    return _ytdlp_probe(url)


@check("HyperFrames registry (raw.githubusercontent.com)")
def _hf_registry():
    r = httpx.get("https://raw.githubusercontent.com/heygen-com/hyperframes/main/registry/registry.json",
                  timeout=20, follow_redirects=True)
    return f"HTTP {r.status_code} ({len(r.content)} bytes)"


@check("npx / node available (HyperFrames render dep)")
def _npx():
    p = subprocess.run(["npx", "--version"], capture_output=True, text=True, timeout=20)
    n = subprocess.run(["node", "--version"], capture_output=True, text=True, timeout=20)
    return f"npx {p.stdout.strip()} / node {n.stdout.strip()}"


@check("Chrome/Chromium binary for HyperFrames render (node/puppeteer)")
def _chromium():
    from pathlib import Path
    candidates = [
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Chromium.app/Contents/MacOS/Chromium",
    ]
    puppeteer_cache = Path.home() / ".cache" / "puppeteer"
    found = [c for c in candidates if Path(c).exists()]
    if puppeteer_cache.exists() and any(puppeteer_cache.rglob("Google Chrome for Testing")):
        found.append(str(puppeteer_cache))
    if not found:
        raise RuntimeError(
            "no Chrome found (system Chrome or ~/.cache/puppeteer). "
            "HyperFrames render downloads it on first run via npx."
        )
    return f"available: {found[0]}"


if __name__ == "__main__":
    print("\n=== ClipFactory connection diagnostics ===\n")
    width = max(len(n) for n, _, _ in RESULTS)
    for name, ok, detail in RESULTS:
        mark = "PASS" if ok else "FAIL"
        print(f"[{mark}] {name.ljust(width)}  {detail}")
    fails = [n for n, ok, _ in RESULTS if not ok]
    print(f"\n{len(RESULTS) - len(fails)}/{len(RESULTS)} passed.",
          f"FAILED: {', '.join(fails)}" if fails else "All green.")
