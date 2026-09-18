"""
Portrait Conversion — Two-pass pipeline.

Pass 1 (Analysis):
  InsightFace detection + WhisperX speaker diarisation → timeline JSON (cached next to source).

Pass 2 (Render):
  FFmpeg filter chains driven by the timeline. No ML.
"""

from __future__ import annotations
import json
import logging
import os
import shutil
import subprocess
import tempfile
from bisect import bisect_left
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from pipeline.memlog import log_rss

log = logging.getLogger(__name__)

OUT_W, OUT_H     = 1080, 1920
SCAN_FPS         = 4       # full/fallback analysis
FAST_SCAN_FPS    = 3       # stored speaker bindings available
SCAN_W           = 320     # resize width for detection

EMA_ALPHA        = 0.15    # position smoothing (lower = smoother, less responsive)
DEAD_ZONE_PX     = 20      # source px: movements smaller than this don't update EMA
KEYFRAME_DIST    = 15      # source px: emit new crop keyframe when EMA drifts this far
PAN_S            = 0.3     # crop pans to a new keyframe over this duration (sec)
SNAP_FRAC        = 1 / 3   # jump > crop_w/3 (or crop_h/3) = shot change → snap, don't pan
FACE_GRACE_S     = 1.5     # hold last single-face crop this long through missed detections

PORTRAIT_FILL    = 0.85
EYE_IN_FACE      = 0.25
EYE_TARGET_Y     = 0.33

SPEAKER_SWITCH_S = 0.8    # hysteresis: don't switch speaker unless dominant for 800ms
SPEAKER_MIN_DUR  = 1.5    # min clean-segment length (sec) to initialise speaker-face bind
OVERLAP_WINDOW_S = 1.0    # speakers within 1s of each other = transition → split / letterbox
MIN_SEGMENT_S    = 0.5    # merge segments shorter than 500ms into their predecessor
PANEL_MISMATCH_S = 0.5    # sustained panel identity mismatch = person change → break segment
SPLIT_FACE_RATIO = 0.4    # split faces must be within 40% size of each other, else letterbox
CAST_MIN_PRESENCE = 0.5   # identity must appear in ≥50% of frames to count as main cast

# Full-frame mode: instead of black bars, fill the canvas with a blurred,
# zoomed copy of the clip and overlay the sharp full frame centered on top.
# Background is blurred at 1/4 resolution then upscaled — same look, ~16x cheaper.
_BLUR_FILL = (
    "[0:v]split=2[bg][fg];"
    f"[bg]scale={OUT_W // 4}:{OUT_H // 4}:force_original_aspect_ratio=increase,"
    f"crop={OUT_W // 4}:{OUT_H // 4},gblur=sigma=12,eq=brightness=-0.06,"
    f"scale={OUT_W}:{OUT_H}[bgf];"
    f"[fg]scale={OUT_W}:{OUT_H}:force_original_aspect_ratio=decrease:flags=lanczos[fgf];"
    "[bgf][fgf]overlay=(W-w)/2:(H-h)/2,setsar=1:1,format=yuv420p[out]"
)


# ── Encoder ───────────────────────────────────────────────────────────────────

def _detect_encoder() -> list[str]:
    try:
        r = subprocess.run(["ffmpeg", "-encoders"], capture_output=True, text=True, timeout=5)
        if "h264_videotoolbox" in r.stdout:
            return ["-c:v", "h264_videotoolbox", "-q:v", "55"]
    except Exception:
        pass
    return ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "23"]

_venc_cached: list[str] | None = None

def _venc() -> list[str]:
    global _venc_cached
    if _venc_cached is None:
        _venc_cached = _detect_encoder()
    return _venc_cached


# ── InsightFace singleton ─────────────────────────────────────────────────────

_face_app = None
_face_app_fast = None

def _get_face_app():
    global _face_app
    if _face_app is None:
        from insightface.app import FaceAnalysis
        _face_app = FaceAnalysis(allowed_modules=["detection", "recognition", "landmark_2d_106"])
        _face_app.prepare(ctx_id=0, det_size=(320, 320))
    return _face_app


def _get_face_app_fast():
    global _face_app_fast
    if _face_app_fast is None:
        from insightface.app import FaceAnalysis
        _face_app_fast = FaceAnalysis(allowed_modules=["detection", "recognition"])
        _face_app_fast.prepare(ctx_id=0, det_size=(320, 320))
    return _face_app_fast


# ── Video meta ────────────────────────────────────────────────────────────────

def _video_meta(source: str) -> tuple[float, int, int]:
    cap   = cv2.VideoCapture(source)
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if src_w == 0 or src_h == 0:
        raise RuntimeError(f"Cannot read video dimensions: {source}")
    return fps, src_w, src_h


# ── Audio extraction + diarisation ───────────────────────────────────────────

def _extract_audio(source: str, out_path: str) -> None:
    _run([
        "ffmpeg", "-y", "-i", source,
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        out_path,
    ])


def _run_diarization(audio_path: str) -> list[tuple[float, float, str]]:
    """Run the local pyannote diarizer for portrait-analysis fallback."""
    from .diarize import diarize_audio

    timeline = diarize_audio(audio_path)
    return sorted(timeline, key=lambda x: x[0])


class _SpeakerSweep:
    """
    Amortised-linear speaker lookup over a start-sorted diarisation timeline.
    at(t) must be called with non-decreasing t (true for the scan loop).
    Returns (active_speaker | None, overlap) where overlap means two or more
    distinct speakers are active within OVERLAP_WINDOW_S of t.
    """

    def __init__(self, timeline: list):
        self._timeline = timeline
        self._i = 0
        self._active: list = []

    def at(self, t: float) -> tuple[str | None, bool]:
        # Admit segments entering the ±OVERLAP_WINDOW_S window, expire those leaving it.
        while self._i < len(self._timeline) and self._timeline[self._i][0] <= t + OVERLAP_WINDOW_S:
            self._active.append(self._timeline[self._i])
            self._i += 1
        self._active = [s for s in self._active if s[1] + OVERLAP_WINDOW_S > t]

        speaker = next((sp for start, end, sp in self._active if start <= t < end), None)
        nearby  = {sp for _, _, sp in self._active}
        return speaker, len(nearby) > 1


# ── Face utilities ────────────────────────────────────────────────────────────

def _cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na > 0 and nb > 0 else 0.0


# ── Pass 1: scan full source ──────────────────────────────────────────────────

# ── Mouth movement ────────────────────────────────────────────────────────────
# InsightFace's 106-point layout: the outer upper-lip arc and outer lower-lip
# arc. Openness is the vertical gap between them over face height; it's the
# CHANGE in openness over time that says "talking", so the closed-mouth
# baseline (lip thickness) doesn't matter and inner-lip indices aren't needed.
MOUTH_UPPER = [64, 63, 71, 67, 68]
MOUTH_LOWER = [55, 56, 53, 59, 58]
TALK_WINDOW_S = 1.0    # trailing window over which mouth activity is measured
TALK_MIN      = 0.006  # activity below this is a still mouth (measured: listener ≈0.002, speaker ≈0.02)
TRACK_IOU     = 0.3    # same face across consecutive scan frames


def _mouth_openness(face) -> float | None:
    lm = getattr(face, "landmark_2d_106", None)
    if lm is None:
        return None
    fh = float(face.bbox[3] - face.bbox[1])
    if fh <= 0:
        return None
    return float(lm[MOUTH_LOWER, 1].mean() - lm[MOUTH_UPPER, 1].mean()) / fh


def _iou(a: list, b: list) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _mouth_activity(records: list[dict], window_s: float = TALK_WINDOW_S) -> None:
    """
    Set face["talk"] on every scanned face: mean |Δ openness| over the trailing
    window along that face's track (faces matched frame-to-frame by IoU).
    None when the face has fewer than two samples in the window or no
    landmarks. Relative comparison between faces in the same frame is the
    robust use; TALK_MIN is only a noise floor.
    """
    tracks: list[dict] = []   # {"bbox": last bbox, "hist": [(t, openness)]}
    for r in records:
        t = r["t"]
        used: set[int] = set()
        for f in r["faces"]:
            best, best_iou = None, TRACK_IOU
            for i, tr in enumerate(tracks):
                if i in used:
                    continue
                v = _iou(f["bbox"], tr["bbox"])
                if v >= best_iou:
                    best, best_iou = i, v
            if best is None:
                tracks.append({"bbox": f["bbox"], "hist": []}); best = len(tracks) - 1
            used.add(best)
            tr = tracks[best]
            tr["bbox"] = f["bbox"]
            if f.get("mouth") is not None:
                tr["hist"].append((t, f["mouth"]))
            tr["hist"] = [(tt, m) for tt, m in tr["hist"] if t - tt <= window_s]
            vals = [m for _, m in tr["hist"]]
            f["talk"] = float(np.abs(np.diff(vals)).mean()) if len(vals) >= 2 else None
        # tracks unmatched this frame keep their history until it ages out
        tracks = [tr for i, tr in enumerate(tracks) if i in used or (tr["hist"] and t - tr["hist"][-1][0] <= window_s)]


def _scan_source(
    source: str,
    src_w: int,
    src_h: int,
    diar_timeline: list,
    fps: float = SCAN_FPS,
    start: float = 0.0,
    duration: float | None = None,
    use_fast: bool = False,
) -> list[dict]:
    """
    Pipe the source (or the window [start, start+duration]) at `fps` and run
    InsightFace on every frame. Returns per-frame records
    [{t, active_speaker, overlap, faces: [...]}] with t in source seconds and
    all coordinates in original source resolution. Mouth activity is
    attached to every face before returning (see _mouth_activity).
    """
    scale  = SCAN_W / src_w
    scan_h = int(src_h * scale)
    fsize  = SCAN_W * scan_h * 3

    face_app = _get_face_app_fast() if use_fast else _get_face_app()

    cmd = [
        "ffmpeg",
        *(["-ss", f"{start:.3f}"] if start > 0 else []),
        "-i", source,
        *(["-t", f"{duration:.3f}"] if duration is not None else []),
        "-vf", f"scale={SCAN_W}:{scan_h},fps={fps}",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-an", "pipe:1",
    ]
    proc  = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    sweep = _SpeakerSweep(diar_timeline)

    records: list[dict] = []
    frame_idx = 0
    try:
        while True:
            raw = proc.stdout.read(fsize)
            if len(raw) < fsize:
                break
            t = start + frame_idx / fps
            frame_idx += 1
            frame = np.frombuffer(raw, dtype=np.uint8).reshape(scan_h, SCAN_W, 3)

            # InsightFace expects BGR (cv2 convention) — frame is already bgr24
            detected = face_app.get(frame)

            faces_out = []
            for face in detected:
                x1, y1, x2, y2 = face.bbox.astype(int)
                fw_s = x2 - x1
                fh_s = y2 - y1
                faces_out.append({
                    "bbox":       [int(x1/scale), int(y1/scale), int(x2/scale), int(y2/scale)],
                    "embedding":  face.embedding.tolist() if face.embedding is not None else None,
                    "cx":         int((x1 + fw_s / 2) / scale),
                    "cy":         int((y1 + fh_s / 2) / scale),
                    "fw":         int(fw_s / scale),
                    "fh":         int(fh_s / scale),
                    "mouth":      _mouth_openness(face),
                })

            speaker, overlap = sweep.at(t)
            records.append({
                "t":              t,
                "active_speaker": speaker,
                "overlap":        overlap,
                "faces":          faces_out,
            })

        # EOF on stdout — a short read can also mean ffmpeg died mid-file.
        # Surface that instead of silently returning a truncated scan.
        rc = proc.wait(timeout=30)
        if rc != 0:
            raise RuntimeError(
                f"ffmpeg scan exited {rc} — analysis truncated at {frame_idx} frames: {source}"
            )
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()

    if not use_fast:
        _mouth_activity(records)
    log.debug("Scanned %d frames from %s", frame_idx, source)
    return records


# ── Speaker-face binding ──────────────────────────────────────────────────────

def _bind_speakers(
    records: list[dict],
    diar_timeline: list,
) -> dict[str, np.ndarray | None]:
    """
    Returns {speaker_id: mean_embedding | None}.

    Audio picks the time window (diarisation segments > SPEAKER_MIN_DUR, no
    overlap); video picks the face within it. In frames where mouth activity
    is measurable, each face's embedding is weighted by its share of the
    mouth movement in that frame — a listener's still face contributes ~0
    even in a solo reaction shot, and the speaker is found in two-shots
    where single-face frames never occur. Frames with no measurable
    movement fall back to the old rule: count the face only if it's alone.
    """
    times = [r["t"] for r in records]
    sums:    dict[str, np.ndarray] = {}
    weights: dict[str, float]      = {}
    bound:   dict[str, np.ndarray | None] = {}

    def add(speaker: str, emb: list, w: float) -> None:
        vec = np.array(emb) * w
        if speaker in sums:
            sums[speaker] += vec; weights[speaker] += w
        else:
            sums[speaker] = vec; weights[speaker] = w

    for start, end, speaker in diar_timeline:
        bound.setdefault(speaker, None)
        if (end - start) < SPEAKER_MIN_DUR:
            continue
        lo = bisect_left(times, start)
        hi = bisect_left(times, end)
        for r in records[lo:hi]:
            if r["overlap"]:
                continue
            faces = [f for f in r["faces"] if f.get("embedding") is not None]
            if not faces:
                continue
            talk = [f.get("talk") for f in faces]
            total = sum(v for v in talk if v is not None)
            if any(v is not None and v >= TALK_MIN for v in talk):
                for f, v in zip(faces, talk):
                    if v:
                        add(speaker, f["embedding"], v / total)
            elif len(faces) == 1:
                add(speaker, faces[0]["embedding"], 1.0)

    for speaker, total in sums.items():
        bound[speaker] = total / weights[speaker]
    return bound


SPEAKER_FACES_VERSION = 3   # bump when the binding method changes; stale stored maps are rebuilt
BIND_BURST_S        = 1.5   # seconds sampled per burst — enough for mouth activity at SCAN_FPS
BIND_BURSTS_PER_SPK = 30    # bursts per speaker, spread across their segments


def _burst_windows(diar_timeline: list, per_speaker: int = BIND_BURSTS_PER_SPK,
                   burst_s: float = BIND_BURST_S) -> list[tuple[float, float]]:
    """Up to `per_speaker` windows per speaker, evenly spread over that
    speaker's longer segments, each centred in its segment. Sorted by start."""
    by_spk: dict[str, list[tuple[float, float]]] = {}
    for start, end, spk in diar_timeline:
        if end - start >= max(SPEAKER_MIN_DUR, burst_s):
            by_spk.setdefault(spk, []).append((start, end))
    windows: list[tuple[float, float]] = []
    for segs in by_spk.values():
        segs.sort()
        step = max(1, len(segs) / per_speaker)
        picked = [segs[int(i * step)] for i in range(min(per_speaker, len(segs)))]
        for start, end in picked:
            mid = (start + end) / 2
            windows.append((max(0.0, mid - burst_s / 2), burst_s))
    return sorted(set(windows))


def bind_speakers_from_source(
    source_path: str, diar_timeline: list,
) -> dict[str, list[float]]:
    """
    Learn which face belongs to each diarised speaker from the WHOLE source.

    Binding from a single clip is fooled by the first reaction shot: the
    editor cuts to the listener while the speaker talks. Here we sample
    short bursts inside each speaker's own segments across the full episode
    and weight faces by mouth movement (see _bind_speakers), so the person
    actually talking wins even in two-shots and solo cutaways. Runs once per
    video (transcribe time, source already local); the result is stored
    beside the source and fed to every export via `speaker_faces`.

    Returns {"version": SPEAKER_FACES_VERSION, "speakers": {speaker_id:
    512-d embedding as a list}} for speakers that got a binding;
    JSON-serialisable.
    """
    _, src_w, src_h = _video_meta(source_path)
    records: list[dict] = []
    for start, dur in _burst_windows(diar_timeline):
        records.extend(_scan_source(source_path, src_w, src_h, diar_timeline, start=start, duration=dur))
    records.sort(key=lambda r: r["t"])
    bound = _bind_speakers(records, diar_timeline)
    out = {spk: vec.tolist() for spk, vec in bound.items() if vec is not None}

    # Two speakers bound to the same face is not a trustworthy binding.
    # Drop both so exports fall back to the safer local/full analysis path.
    ids = sorted(out)
    ambiguous: set[str] = set()
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            sim = _cos_sim(np.array(out[a]), np.array(out[b]))
            if sim > 0.8:
                log.warning(
                    "Speaker binding ambiguous: %s and %s map to the same face (cos %.2f)",
                    a, b, sim,
                )
                ambiguous.update((a, b))

    for spk in ambiguous:
        out.pop(spk, None)
    log.info("Speaker→face binding from %d scanned frames: %s", len(records),
             {k: "bound" for k in out} or "none")
    return {"version": SPEAKER_FACES_VERSION, "speakers": out}


def _has_complete_preloaded_bindings(
    preloaded: dict | None,
    diar_timeline: list,
) -> bool:
    """True when every diarised speaker already has a stored face embedding."""
    if not preloaded or not diar_timeline:
        return False

    speakers = preloaded.get("speakers", preloaded)
    if not speakers:
        return False

    diar_speakers = {spk for _, _, spk in diar_timeline if spk}
    return bool(diar_speakers) and diar_speakers.issubset(set(speakers))


def _merge_bindings(
    local: dict[str, np.ndarray | None], preloaded: dict | None,
) -> dict[str, np.ndarray | None]:
    """Whole-video bindings override clip-local ones; local fills any gaps.
    Accepts the stored map ({"version", "speakers"}) or a bare speaker dict."""
    speakers = (preloaded or {}).get("speakers", preloaded) if preloaded else None
    if not speakers:
        return local
    merged = dict(local)
    for spk, vec in speakers.items():
        merged[spk] = np.array(vec, dtype=float)
    return merged


# ── Crop geometry ─────────────────────────────────────────────────────────────

def _compute_crop_rect(
    cx: float, cy: float, fh: float,
    src_w: int, src_h: int,
    crop_w: int, crop_h: int,
) -> dict:
    half_w = crop_w // 2
    eye_y  = cy - fh * (0.5 - EYE_IN_FACE)
    crop_x = max(0, min(src_w - crop_w, int(cx) - half_w))
    crop_y = max(0, min(src_h - crop_h, int(eye_y - crop_h * EYE_TARGET_Y)))
    return {"x": crop_x, "y": crop_y, "w": crop_w, "h": crop_h}


def _ema_keyframes(
    timed_faces: list[tuple[float, dict]],
    seg_end: float,
    src_w: int,
    src_h: int,
    crop_w: int,
    crop_h: int,
) -> dict | None:
    """
    Convert per-frame (timestamp, face) pairs into ONE segment carrying
    EMA-smoothed crop keyframes. The renderer pans smoothly between
    keyframes (see _pan_expr) instead of hard-cutting.

    Dead zone: if the new detection is within DEAD_ZONE_PX of the current
    smoothed position, the EMA is not updated — head turns and small jitter
    don't shift the crop at all. The crop only advances when the EMA has
    drifted more than KEYFRAME_DIST from the last committed crop position.

    Shot change: a detection jumping more than SNAP_FRAC of the crop size
    away from the EMA is a camera cut, not movement — reset the EMA and emit
    a snap keyframe (renderer hard-cuts instead of panning across the frame).
    """
    if not timed_faces:
        return None

    t0, f0 = timed_faces[0]
    ema_cx = float(f0["cx"])
    ema_cy = float(f0["cy"])
    ema_fh = float(f0["fh"])

    kf_cx, kf_cy = ema_cx, ema_cy
    kf_crop = _compute_crop_rect(ema_cx, ema_cy, ema_fh, src_w, src_h, crop_w, crop_h)
    keyframes: list[tuple[float, dict, bool]] = [(t0, kf_crop, False)]

    snap_x = crop_w * SNAP_FRAC
    snap_y = crop_h * SNAP_FRAC
    prev_cx, prev_cy = float(f0["cx"]), float(f0["cy"])

    for t, face in timed_faces[1:]:
        raw_cx = float(face["cx"])
        raw_cy = float(face["cy"])
        raw_fh = float(face["fh"])

        # Shot change = discontinuity between CONSECUTIVE detections (the EMA
        # lags real motion, so comparing against it would mis-fire on a person
        # moving steadily across the frame).
        if abs(raw_cx - prev_cx) > snap_x or abs(raw_cy - prev_cy) > snap_y:
            prev_cx, prev_cy = raw_cx, raw_cy
            ema_cx, ema_cy, ema_fh = raw_cx, raw_cy, raw_fh
            crop = _compute_crop_rect(ema_cx, ema_cy, ema_fh, src_w, src_h, crop_w, crop_h)
            keyframes.append((t, crop, True))
            kf_cx, kf_cy = ema_cx, ema_cy
            continue
        prev_cx, prev_cy = raw_cx, raw_cy

        # Only pull EMA toward new position when movement exceeds dead zone
        if abs(raw_cx - ema_cx) > DEAD_ZONE_PX or abs(raw_cy - ema_cy) > DEAD_ZONE_PX:
            ema_cx = EMA_ALPHA * raw_cx + (1.0 - EMA_ALPHA) * ema_cx
            ema_cy = EMA_ALPHA * raw_cy + (1.0 - EMA_ALPHA) * ema_cy
        # Face size smoothed regardless (stable crop height)
        ema_fh = EMA_ALPHA * raw_fh + (1.0 - EMA_ALPHA) * ema_fh

        # Commit a new keyframe only when accumulated drift is significant
        if abs(ema_cx - kf_cx) > KEYFRAME_DIST or abs(ema_cy - kf_cy) > KEYFRAME_DIST:
            crop = _compute_crop_rect(ema_cx, ema_cy, ema_fh, src_w, src_h, crop_w, crop_h)
            keyframes.append((t, crop, False))
            kf_cx, kf_cy = ema_cx, ema_cy

    return {
        "start":     t0,
        "end":       seg_end,
        "mode":      "single",
        "crop":      keyframes[0][1],   # w/h are constant; x/y animate via keyframes
        "crops":     None,
        "keyframes": [{"t": kf_t, "x": c["x"], "y": c["y"], "snap": snap}
                      for kf_t, c, snap in keyframes],
    }


# ── Timeline builder ──────────────────────────────────────────────────────────

def _face_score(
    face: dict,
    active_speaker: str | None,
    bound: dict,
    src_w: int,
    src_h: int,
) -> float:
    bbox_area  = (face["fw"] * face["fh"]) / (src_w * src_h)
    centrality = 1.0 - abs(face["cx"] - src_w / 2) / (src_w / 2)
    speaking   = 0.0
    if active_speaker and bound.get(active_speaker) is not None and face["embedding"] is not None:
        sim      = _cos_sim(np.array(face["embedding"]), bound[active_speaker])
        speaking = max(0.0, sim)
    return 0.6 * speaking + 0.3 * bbox_area + 0.1 * centrality


def _pick_two_faces(
    faces: list[dict],
    active_speaker: str | None,
    bound: dict,
    src_w: int,
    src_h: int,
) -> list[dict] | None:
    """
    Identity-first: when two or more speakers have bound embeddings, the
    panels must show faces matching real speakers — a face matching nobody
    (a poster, a screen in the shot, audience) is never eligible, no matter
    how well it scores. Falls back to score order + horizontal-thirds +
    size-ratio checks when identity is inconclusive.
    Returns None when no acceptable pair exists (letterbox beats a junk panel).
    """
    bound_embs = {sp: ref for sp, ref in bound.items() if ref is not None}
    if len(bound_embs) >= 2:
        best: dict[str, tuple[float, int]] = {}   # speaker -> (sim, face index)
        for idx, f in enumerate(faces):
            if not f.get("embedding"):
                continue
            emb = np.array(f["embedding"])
            for sp, ref in bound_embs.items():
                sim = _cos_sim(emb, ref)
                if sim >= 0.4 and (sp not in best or sim > best[sp][0]):
                    best[sp] = (sim, idx)
        # Active speaker's face first, then strongest match; distinct faces only
        order = sorted(best.items(), key=lambda kv: (kv[0] != active_speaker, -kv[1][0]))
        chosen_idx: list[int] = []
        for _sp, (_sim, idx) in order:
            if idx not in chosen_idx:
                chosen_idx.append(idx)
            if len(chosen_idx) == 2:
                return [faces[i] for i in chosen_idx]

    scored   = sorted(faces, key=lambda f: _face_score(f, active_speaker, bound, src_w, src_h), reverse=True)
    thirds_w = src_w / 3
    chosen:      list[dict] = []
    used_thirds: set[int]   = set()
    for f in scored:
        third = int(f["cx"] / thirds_w)
        if third not in used_thirds:
            chosen.append(f)
            used_thirds.add(third)
        if len(chosen) == 2:
            break
    if len(chosen) < 2:
        return None
    small_fh, big_fh = sorted((chosen[0]["fh"], chosen[1]["fh"]))
    if small_fh < SPLIT_FACE_RATIO * big_fh:
        return None
    return chosen


def _main_cast(records: list[dict]) -> list[np.ndarray]:
    """
    Greedily cluster face embeddings across the whole scan and return the mean
    embeddings of identities present in ≥ CAST_MIN_PRESENCE of frames (up to 4).
    These are the people the clip is about — transient face-like detections
    (a face on a screen in the shot, posters, audience) never qualify.
    Returns [] when fewer than two identities qualify (no filtering possible).
    """
    if not records:
        return []
    clusters: list[list] = []   # [embedding_sum, frame_count]
    for rec in records:
        for f in rec["faces"]:
            if f["embedding"] is None:
                continue
            emb = np.array(f["embedding"])
            for c in clusters:
                if _cos_sim(c[0], emb) >= 0.45:
                    c[0] = c[0] + emb
                    c[1] += 1
                    break
            else:
                clusters.append([emb, 1])
    min_frames = CAST_MIN_PRESENCE * len(records)
    cast = sorted((c for c in clusters if c[1] >= min_frames),
                  key=lambda c: c[1], reverse=True)[:4]
    return [c[0] / c[1] for c in cast] if len(cast) >= 2 else []


IDENT_SIM = 0.45   # cosine similarity at which a face is confidently a bound speaker


def _identify(face: dict, bound: dict) -> tuple[str | None, float]:
    """
    Which bound speaker this face is, if any: (speaker, sim) when the best
    match clears IDENT_SIM, else (None, best_sim). A near-profile or blurred
    face typically matches nobody — that means "unknown", not "not them".
    """
    emb = face.get("embedding")
    sims = {sp: _cos_sim(np.array(emb), ref) for sp, ref in bound.items() if ref is not None} if emb else {}
    if not sims:
        return None, -1.0
    sp = max(sims, key=sims.get)
    return (sp if sims[sp] >= IDENT_SIM else None), sims[sp]


def _best_face_for_speaker(
    faces: list[dict],
    speaker: str | None,
    bound: dict,
) -> dict:
    """
    The face to follow for `speaker`. Identity is checked against EVERY bound
    speaker, not just the active one: a face that confidently belongs to
    someone else is ruled out even if it happens to resemble the speaker more
    than an unrecognisable one does. Preference order —
      1. confidently the speaker (highest sim first)
      2. unidentified — a profile view of the speaker looks like this; among
         these, the mouth that's moving, then the largest
      3. confidently another speaker (last resort, so a solo shot still tracks)
    With no binding for the speaker (or no diarisation at all), follow the
    mouth that's moving; if none is, the largest face as before.
    """
    ref = bound.get(speaker) if speaker else None
    if ref is None:
        talking = [f for f in faces if (f.get("talk") or 0.0) >= TALK_MIN]
        if talking:
            return max(talking, key=lambda f: f["talk"])
        return max(faces, key=lambda f: f["fw"] * f["fh"])

    def rank(f: dict) -> tuple[int, float, float]:
        who, sim = _identify(f, bound)
        if who == speaker:
            return (2, sim, 0.0)
        if who is None:
            talk = f.get("talk") or 0.0
            return (1, talk if talk >= TALK_MIN else 0.0, float(f["fw"] * f["fh"]))
        return (0, sim, 0.0)

    return max(faces, key=rank)


def _build_timeline(
    records: list[dict],
    diar_timeline: list,
    src_w: int,
    src_h: int,
    crop_w: int,
    crop_h: int,
    speaker_faces: dict[str, list[float]] | None = None,
) -> list[dict]:
    """
    Converts per-frame records into timeline segments [{start, end, mode, crop, crops}].
    Applies hysteresis and merges short segments.
    """
    if _has_complete_preloaded_bindings(speaker_faces, diar_timeline):
        bound = _merge_bindings({}, speaker_faces)
    else:
        bound = _merge_bindings(_bind_speakers(records, diar_timeline), speaker_faces)

    cast = _main_cast(records)

    # Panel dimensions for split mode: each face gets a 9:8 crop → scales to 1080×960
    panel_crop_h = (int(src_h * 0.6) // 2) * 2
    panel_crop_w = (int(panel_crop_h * 9 / 8) // 2) * 2

    current_speaker: str | None = None
    pending_speaker: str | None = None
    pending_since:   float       = 0.0
    last_face:       dict | None = None   # last real single-mode detection …
    last_face_t:     float       = 0.0    # … and when it happened (grace period)
    single_run:      int         = 0      # consecutive real single frames (grace needs ≥2)
    split_top_emb:   np.ndarray | None = None  # established top-panel identity
    split_swap_since: float | None     = None  # when a swapped ordering first appeared
    last_two:        list | None = None   # last real split pair …
    last_two_t:      float       = 0.0    # … and when it happened (split grace)
    split_run:       int         = 0      # consecutive real split frames (grace needs ≥2)

    per_frame: list[tuple[float, str, dict | None, list | None]] = []

    def _overlaps(a: list[dict], b: list[dict]) -> bool:
        return any(_iou(x["bbox"], y["bbox"]) >= TRACK_IOU for x in a for y in b)

    prev_faces: list[dict] = []   # last frame's raw detections, for shot-change detection
    for idx, rec in enumerate(records):
        t       = rec["t"]
        faces   = rec["faces"]
        active  = rec["active_speaker"]
        overlap = rec["overlap"]
        # Shot change: nothing in this frame overlaps the last frame, AND the
        # new layout persists into the next frame. A single-frame detection
        # blink fails the second test and is still treated as a blink.
        next_faces = records[idx + 1]["faces"] if idx + 1 < len(records) else faces
        shot_changed = (bool(prev_faces) and bool(faces)
                        and not _overlaps(faces, prev_faces)
                        and _overlaps(faces, next_faces))
        prev_faces = faces

        # Only main-cast faces drive framing: a transient face-like detection
        # must not occupy a panel or force split mode (n is counted after).
        if cast:
            faces = [
                f for f in faces
                if f.get("embedding")
                and max(_cos_sim(np.array(f["embedding"]), c) for c in cast) >= 0.4
            ]
        n = len(faces)

        # Brief detection misses hold the last crop (single or split pair)
        # instead of flashing to letterbox and back. Lone detections
        # (run < 2, likely false positives) get no grace.
        in_single_grace = (last_face is not None and single_run >= 2
                           and t - last_face_t <= FACE_GRACE_S)
        in_split_grace  = (last_two is not None and split_run >= 2
                           and t - last_two_t <= FACE_GRACE_S)

        if n == 0:
            split_top_emb, split_swap_since = None, None
            if in_split_grace:
                per_frame.append((t, "split", None, last_two))
            elif in_single_grace:
                per_frame.append((t, "single", last_face, None))
            else:
                per_frame.append((t, "letterbox", None, None))
                last_face,  single_run = None, 0
                last_two,   split_run  = None, 0
            continue

        if overlap or n >= 3:
            last_face  = None
            single_run = 0
            two = _pick_two_faces(faces, active, bound, src_w, src_h) if n >= 2 else None
            if two:
                # Active speaker → top panel (index 0); other face → bottom (index 1)
                ordering = "score"
                reason   = "no_active_speaker"
                if active and bound.get(active) is not None:
                    ref  = bound[active]
                    sims = [
                        _cos_sim(np.array(f["embedding"]), ref) if f["embedding"] else -1.0
                        for f in two
                    ]
                    top_sim = max(sims)
                    if top_sim >= 0.4:
                        if sims[1] > sims[0]:
                            two = [two[1], two[0]]
                        ordering = "diarisation"
                        reason   = f"speaker={active} sim={top_sim:.2f}"
                    else:
                        reason = f"sim_below_threshold ({top_sim:.2f})"
                # Ordering hysteresis: keep the established panel assignment
                # unless the swapped ordering wins for SPEAKER_SWITCH_S —
                # score noise must not make faces trade panels frame-to-frame.
                e0, e1 = two[0].get("embedding"), two[1].get("embedding")
                if e0 and e1:
                    if split_top_emb is not None and \
                            _cos_sim(np.array(e1), split_top_emb) > _cos_sim(np.array(e0), split_top_emb):
                        if split_swap_since is None:
                            split_swap_since = t
                        if t - split_swap_since < SPEAKER_SWITCH_S:
                            two = [two[1], two[0]]   # hold established ordering
                        else:
                            split_swap_since = None  # swap sustained → accept
                    else:
                        split_swap_since = None
                    split_top_emb = np.array(two[0]["embedding"])
                log.debug("[split] ts=%.1f ordering=%s reason=%s", t, ordering, reason)
                last_two, last_two_t = two, t
                split_run += 1
                per_frame.append((t, "split", None, two))
            elif in_split_grace:
                # One panel's face dropped out for a frame — hold the pair.
                per_frame.append((t, "split", None, last_two))
            else:
                split_top_emb, split_swap_since = None, None
                last_two, split_run = None, 0
                per_frame.append((t, "letterbox", None, None))
            continue

        # Single dominant face — apply hysteresis before committing to a speaker
        if active != current_speaker:
            if pending_speaker != active:
                pending_speaker = active
                pending_since   = t
            elif t - pending_since >= SPEAKER_SWITCH_S:
                current_speaker = active
                pending_speaker = None

        best = _best_face_for_speaker(faces, current_speaker or active, bound)
        split_top_emb, split_swap_since = None, None
        last_two, split_run = None, 0
        ref = bound.get(current_speaker or active)
        # If we were confidently on the speaker and this frame's best face
        # isn't them (their face dropped out and another got picked up), treat
        # it as a detection miss and hold — no 250ms cuts to the wrong person.
        # But never hold a crop that was itself someone else: that's how a
        # listener's cutaway pinned the whole clip on the listener while the
        # speaker sat in profile, unrecognised, one face over.
        # A detection blink leaves the newly picked face overlapping something
        # that was on screen a frame ago; a real shot change doesn't. Never
        # hold across a shot change — it paints the old crop over new footage.
        following = current_speaker or active
        leaving_speaker_for_other = (
            ref is not None and best.get("embedding") is not None
            and _identify(best, bound)[0] != following
            and last_face is not None and last_face.get("embedding") is not None
            and _identify(last_face, bound)[0] == following
            and not shot_changed
        )
        if (leaving_speaker_for_other
                and single_run >= 2
                and t - last_face_t <= FACE_GRACE_S):
            per_frame.append((t, "single", last_face, None))
            continue
        # Without diarisation the "biggest face" choice flip-flops on a few px
        # of detection noise — stick with the face we're already following
        # unless another is decisively bigger (a real framing change).
        if ref is None and last_face is not None and last_face.get("embedding"):
            last_emb = np.array(last_face["embedding"])
            sticky = [f for f in faces if f.get("embedding")
                      and _cos_sim(np.array(f["embedding"]), last_emb) >= 0.45]
            if sticky:
                followed = max(sticky, key=lambda f: f["fh"])
                # ...but a mouth that is clearly the one moving wins over
                # stickiness: that's the whole point of following the talker.
                best_talk, kept_talk = best.get("talk") or 0.0, followed.get("talk") or 0.0
                talk_decisive = best_talk >= TALK_MIN and best_talk > 2 * kept_talk
                if best is not followed and best["fh"] < 1.3 * followed["fh"] and not talk_decisive:
                    best = followed
        last_face, last_face_t = best, t
        single_run += 1
        per_frame.append((t, "single", best, None))

    if not per_frame:
        return []

    # Group consecutive frames with the same mode into segments. Identity
    # continuity is checked against RUNNING MEAN embeddings (a fixed first-
    # frame reference breaks spuriously when the same person turns their head).
    # Single mode breaks immediately on a speaker change; split panels skip
    # one-off outlier faces (the panel holds its crop) and only break the
    # segment when the mismatch is sustained — a genuine person change.
    segments: list[dict] = []
    i = 0
    while i < len(per_frame):
        seg_t, seg_mode, seg_face, seg_two = per_frame[i]
        j = i + 1

        if seg_mode == "single":
            group_tf: list[tuple[float, dict]] = [(seg_t, seg_face)]
            ref_sum = np.array(seg_face["embedding"]) if seg_face.get("embedding") else None
            while j < len(per_frame):
                tj, mj, fj, _ = per_frame[j]
                if mj != "single":
                    break
                if ref_sum is not None and fj.get("embedding"):
                    cand_emb = np.array(fj["embedding"])
                    if _cos_sim(ref_sum, cand_emb) < 0.45:
                        break
                    ref_sum = ref_sum + cand_emb
                group_tf.append((tj, fj))
                j += 1
            t_end = per_frame[j][0] if j < len(per_frame) else per_frame[-1][0] + 1.0 / SCAN_FPS
            seg = _ema_keyframes(group_tf, t_end, src_w, src_h, crop_w, crop_h)
            if seg:
                segments.append(seg)

        elif seg_mode == "split":
            # Each panel keeps its own identity reference and keyframe track —
            # an unverified "other face" must never steer a panel's crop.
            tracks:   list[list[tuple[float, dict]]] = [[(seg_t, seg_two[0])], [(seg_t, seg_two[1])]]
            ref_sums: list[np.ndarray | None] = [
                np.array(f["embedding"]) if f.get("embedding") else None for f in seg_two
            ]
            mismatch_since: list[float | None] = [None, None]

            while j < len(per_frame):
                tj, mj, _, twoj = per_frame[j]
                if mj != "split":
                    break
                appends: list[tuple[int, dict]] = []
                sustained = False
                for p in (0, 1):
                    emb = twoj[p].get("embedding")
                    if ref_sums[p] is not None and emb:
                        cand_emb = np.array(emb)
                        if _cos_sim(ref_sums[p], cand_emb) < 0.45:
                            # Outlier face: contributes nothing — the panel holds.
                            if mismatch_since[p] is None:
                                mismatch_since[p] = tj
                            elif tj - mismatch_since[p] >= PANEL_MISMATCH_S:
                                sustained = True
                            continue
                        mismatch_since[p] = None
                        ref_sums[p] = ref_sums[p] + cand_emb
                    appends.append((p, twoj[p]))
                if sustained:
                    break
                for p, f in appends:
                    tracks[p].append((tj, f))
                j += 1

            t_end  = per_frame[j][0] if j < len(per_frame) else per_frame[-1][0] + 1.0 / SCAN_FPS
            panels = [
                _ema_keyframes(tracks[p], t_end, src_w, src_h, panel_crop_w, panel_crop_h)
                for p in (0, 1)
            ]
            segments.append({
                "start": seg_t, "end": t_end, "mode": "split", "crop": None,
                "crops":           [p["crop"] for p in panels],
                "panel_keyframes": [p["keyframes"] for p in panels],
            })

        else:  # letterbox
            while j < len(per_frame) and per_frame[j][1] == "letterbox":
                j += 1
            t_end = per_frame[j][0] if j < len(per_frame) else per_frame[-1][0] + 1.0 / SCAN_FPS
            segments.append({"start": seg_t, "end": t_end, "mode": "letterbox", "crop": None, "crops": None})

        i = j

    # Merge short MODE-CHANGE segments into predecessor (suppresses flicker).
    # Extending a "single" segment is safe: the crop holds its last keyframe.
    merged: list[dict] = []
    for seg in segments:
        if (seg["end"] - seg["start"] < MIN_SEGMENT_S
                and merged
                and merged[-1]["mode"] != seg["mode"]):
            merged[-1]["end"] = seg["end"]
        else:
            merged.append(seg)

    return merged


# ── Pass 1 entry point (with cache) ───────────────────────────────────────────

_CACHE_VERSION = 7  # bump when algorithm changes to invalidate stale caches

def _cache_path(source: str) -> str:
    return source + ".portrait_timeline.json"


def _cache_params(
    crop_w: int,
    crop_h: int,
    scan_fps: float = SCAN_FPS,
    fast_scan: bool = False,
) -> dict:
    """Tuning constants baked into a timeline — a cache is only valid if these match."""
    return {
        "crop_w": crop_w, "crop_h": crop_h,
        "scan_fps": scan_fps, "fast_scan": fast_scan, "scan_w": SCAN_W,
        "ema_alpha": EMA_ALPHA, "dead_zone": DEAD_ZONE_PX,
        "keyframe_dist": KEYFRAME_DIST, "snap_frac": SNAP_FRAC,
        "fill": PORTRAIT_FILL, "eye_in_face": EYE_IN_FACE, "eye_target_y": EYE_TARGET_Y,
        "switch_s": SPEAKER_SWITCH_S, "min_dur": SPEAKER_MIN_DUR,
        "overlap_s": OVERLAP_WINDOW_S, "min_seg": MIN_SEGMENT_S,
        "grace_s": FACE_GRACE_S,
        "panel_mismatch_s": PANEL_MISMATCH_S, "split_face_ratio": SPLIT_FACE_RATIO,
        "cast_min_presence": CAST_MIN_PRESENCE,
    }


ANALYSIS_VERSION = 3  # bump when the persisted analysis artifact format changes


def _slim_records(records: list[dict]) -> list[dict]:
    """
    Drop the heavy 512-d face embeddings from the scan records, keeping just the
    geometry + speaker fields needed by the presence map / heatmap / overlap lint.
    """
    return [
        {
            "t":              round(r["t"], 3),
            "active_speaker": r["active_speaker"],
            "overlap":        r["overlap"],
            "faces":          [
                {"bbox": f["bbox"], "cx": f["cx"], "cy": f["cy"], "fw": f["fw"], "fh": f["fh"],
                 "mouth": f.get("mouth"), "talk": f.get("talk")}
                for f in r["faces"]
            ],
        }
        for r in records
    ]


def _write_analysis(
    path: str, slim_records: list[dict], reframe_timeline: list[dict],
    fps: float, src_w: int, src_h: int,
) -> None:
    """
    Persist the per-frame face/active-speaker scan as a queryable artifact.

    Coordinates are in the STITCHED-clip landscape frame (the concatenated cut
    ranges fed to the portrait pass), NOT the original source: bbox/cx/cy are
    pixels in the src_w×src_h stitched video, and `t` is clip-relative seconds.
    Source→output is two-hop: source →(cuts)→ stitched →(crop timeline)→ output;
    `reframe_timeline` carries that second hop (the crop rects per segment) so a
    consumer can map face boxes into output coords without re-running the scan.
    """
    try:
        with open(path, "w") as f:
            json.dump({
                "version":          ANALYSIS_VERSION,
                "coord_space":      "stitched_landscape_px",
                "dims":             [src_w, src_h],
                "fps":              fps,
                "scan_fps":         SCAN_FPS,
                "frames":           slim_records,
                "reframe_timeline": reframe_timeline,
            }, f)
        log.info("Wrote clip analysis artifact (%d frames, %d reframe segs) to %s",
                 len(slim_records), len(reframe_timeline), path)
    except Exception as e:
        log.warning("Could not write analysis artifact: %s", e)


def _analyze_source(
    source: str,
    fps: float,
    src_w: int,
    src_h: int,
    crop_w: int,
    crop_h: int,
    diar_timeline: list | None = None,
    analysis_out_path: str | None = None,
    speaker_faces: dict[str, list[float]] | None = None,
) -> list[dict]:
    """
    Run Pass 1 with JSON cache. Returns timeline segments.
    If diar_timeline is provided (from the stored transcript) it is used directly
    and no extra diarization pass is needed. Falls back to local pyannote only when
    the transcript has no speaker data (e.g. videos transcribed before diarisation
    was enabled).
    When analysis_out_path is set, also writes the per-frame scan (face boxes +
    active speaker) to that path as the persisted analysis artifact.
    """
    fast_scan = _has_complete_preloaded_bindings(
        speaker_faces, diar_timeline or []
    )
    scan_fps = FAST_SCAN_FPS if fast_scan else SCAN_FPS

    cache  = _cache_path(source)
    params = _cache_params(crop_w, crop_h, scan_fps, fast_scan)
    if os.path.exists(cache):
        try:
            with open(cache) as f:
                data = json.load(f)
            if data.get("version") == _CACHE_VERSION and data.get("params") == params:
                if analysis_out_path and data.get("records") is not None:
                    _write_analysis(analysis_out_path, data["records"], data["timeline"], fps, src_w, src_h)
                return data["timeline"]
            log.info("Portrait timeline cache is stale (version/params mismatch) — re-analysing")
        except Exception as e:
            log.warning("Timeline cache unreadable (%s) — re-analysing", e)

    if diar_timeline is not None:
        log.info("Using transcript speaker timeline (%d segments)", len(diar_timeline))
        diar = diar_timeline
    else:
        # Fallback: run local pyannote for transcripts created before diarisation was enabled
        audio_tmp = _mktemp(suffix=".wav")
        try:
            _extract_audio(source, audio_tmp)
            diar = _run_diarization(audio_tmp)
        except Exception as e:
            log.warning("Diarisation failed (%s) — proceeding without speaker info", e)
            diar = []
        finally:
            _cleanup([audio_tmp])

    if fast_scan:
        log.info(
            "Using fast portrait scan: stored speaker bindings are complete (%.1f FPS)",
            scan_fps,
        )

    log_rss("portrait pre-scan")
    records = _scan_source(
        source, src_w, src_h, diar,
        fps=scan_fps,
        use_fast=fast_scan,
    )
    log_rss(f"portrait scan done ({len(records)} records)")
    timeline = _build_timeline(records, diar, src_w, src_h, crop_w, crop_h, speaker_faces=speaker_faces)
    slim     = _slim_records(records)
    log_rss("portrait timeline+slim done")

    try:
        with open(cache, "w") as f:
            json.dump({"version": _CACHE_VERSION, "params": params, "source": source,
                       "fps": fps, "timeline": timeline, "records": slim}, f, indent=2)
        log.info("Wrote portrait timeline cache to %s", cache)
    except Exception as e:
        log.warning("Could not write timeline cache: %s", e)

    if analysis_out_path:
        _write_analysis(analysis_out_path, slim, timeline, fps, src_w, src_h)

    log_rss("portrait analysis done")
    return timeline


# ── Pass 2: render ────────────────────────────────────────────────────────────

def _pan_expr(keyframes: list[dict], axis: str, render_start: float, render_end: float) -> str:
    """
    Piecewise ffmpeg expression that pans between crop keyframes with
    smoothstep ease-in/out. Each keyframe ramps in over PAN_S (clamped to the
    gap to the next one), then holds. Snap keyframes (shot changes) hard-cut
    to their value instead of ramping. Keyframe times are absolute source
    seconds; the crop filter's `t` starts at 0 because -ss is an input option,
    so times are shifted.
    """
    times = [kf["t"] - render_start for kf in keyframes]
    vals  = [kf[axis] for kf in keyframes]
    snaps = [kf.get("snap", False) for kf in keyframes]
    n = len(vals)
    if n == 1:
        return str(vals[0])

    # expr = behaviour from times[i] onward, built from the last keyframe backwards
    expr = ""
    for i in range(n - 1, 0, -1):
        if snaps[i]:
            piece = str(vals[i])
        else:
            next_t = times[i + 1] if i + 1 < n else (render_end - render_start)
            d    = max(0.05, min(PAN_S, next_t - times[i]))
            # smoothstep p*p*(3-2p): p stored in st(0) to avoid repeating the min()
            prog = f"st(0\\,min((t-{times[i]:.3f})/{d:.3f}\\,1))*ld(0)*(3-2*ld(0))"
            piece = f"({vals[i-1]}+({vals[i] - vals[i-1]})*{prog})"
        expr = piece if not expr else f"if(lt(t\\,{times[i+1]:.3f})\\,{piece}\\,{expr})"
    return f"if(lt(t\\,{times[1]:.3f})\\,{vals[0]}\\,{expr})"


def _render_face_crop(
    source: str, start: float, dur: float, seg: dict, output: str,
) -> None:
    crop = seg["crop"]
    kfs  = seg.get("keyframes") or []
    if len(kfs) > 1:
        x_expr = _pan_expr(kfs, "x", start, start + dur)
        y_expr = _pan_expr(kfs, "y", start, start + dur)
    else:
        x_expr, y_expr = str(crop["x"]), str(crop["y"])
    vf = (
        f"crop=w={crop['w']}:h={crop['h']}:x={x_expr}:y={y_expr},"
        f"scale={OUT_W}:{OUT_H}:flags=lanczos,setsar=1:1,format=yuv420p"
    )
    _run([
        "ffmpeg", "-y",
        "-ss", str(start), "-i", source, "-t", str(dur),
        "-vf", vf, *_venc(), "-c:a", "aac", "-movflags", "+faststart",
        output,
    ])


def _render_split(
    source: str, start: float, dur: float, seg: dict, output: str,
) -> None:
    half_h = OUT_H // 2   # 960 — each panel is 1080×960, vstack → 1080×1920
    pkfs   = seg.get("panel_keyframes") or [None, None]

    def _panel_exprs(crop: dict, kfs: list | None) -> tuple[str, str]:
        if kfs and len(kfs) > 1:
            return (_pan_expr(kfs, "x", start, start + dur),
                    _pan_expr(kfs, "y", start, start + dur))
        return str(crop["x"]), str(crop["y"])

    c0, c1 = seg["crops"]
    x0, y0 = _panel_exprs(c0, pkfs[0])
    x1, y1 = _panel_exprs(c1, pkfs[1])
    vf = (
        f"[0:v]crop=w={c0['w']}:h={c0['h']}:x={x0}:y={y0},scale={OUT_W}:{half_h}:flags=lanczos[top];"
        f"[0:v]crop=w={c1['w']}:h={c1['h']}:x={x1}:y={y1},scale={OUT_W}:{half_h}:flags=lanczos[bottom];"
        "[top][bottom]vstack=inputs=2[out]"
    )
    _run([
        "ffmpeg", "-y",
        "-ss", str(start), "-i", source, "-t", str(dur),
        "-filter_complex", vf, "-map", "[out]", "-map", "0:a?",
        *_venc(), "-c:a", "aac", "-movflags", "+faststart",
        output,
    ])


def _render_letterbox(source: str, start: float, dur: float, output: str) -> None:
    _run([
        "ffmpeg", "-y",
        "-ss", str(start), "-i", source, "-t", str(dur),
        "-filter_complex", _BLUR_FILL, "-map", "[out]", "-map", "0:a?",
        *_venc(), "-c:a", "aac", "-movflags", "+faststart",
        output,
    ])


def _dispatch_render(source: str, start: float, dur: float, seg: dict, output: str) -> None:
    if seg["mode"] == "single" and seg.get("crop"):
        _render_face_crop(source, start, dur, seg, output)
    elif seg["mode"] == "split" and seg.get("crops"):
        _render_split(source, start, dur, seg, output)
    else:
        _render_letterbox(source, start, dur, output)


def _render_cut(
    source: str,
    cut: dict,
    timeline: list[dict],
    output: str,
) -> None:
    cut_start = float(cut["start"])
    cut_end   = float(cut["end"])

    relevant = [s for s in timeline if s["end"] > cut_start and s["start"] < cut_end]
    if not relevant:
        _render_letterbox(source, cut_start, cut_end - cut_start, output)
        return

    if len(relevant) == 1:
        _dispatch_render(source, cut_start, cut_end - cut_start, relevant[0], output)
        return

    tasks: list[tuple] = []
    tmp_files: list[str] = []
    try:
        for seg in relevant:
            seg_start = max(cut_start, seg["start"])
            seg_end   = min(cut_end,   seg["end"])
            dur = seg_end - seg_start
            if dur <= 0:
                continue
            tmp = _mktemp(suffix=".mp4")
            tmp_files.append(tmp)
            tasks.append((seg_start, dur, seg, tmp))

        if not tasks:
            _render_letterbox(source, cut_start, cut_end - cut_start, output)
        elif len(tasks) == 1:
            s, d, seg, tmp = tasks[0]
            _dispatch_render(source, s, d, seg, tmp)
            shutil.move(tmp_files.pop(), output)
            tmp_files.clear()
        else:
            with ThreadPoolExecutor() as pool:
                futs = [pool.submit(_dispatch_render, source, s, d, seg, tmp) for s, d, seg, tmp in tasks]
                for f in futs:
                    f.result()
            _concat(tmp_files, output)
    finally:
        _cleanup(tmp_files)


# ── Public API ────────────────────────────────────────────────────────────────

def cut_and_portrait(
    source_path: str,
    cuts: list[dict[str, float]],
    output_path: str,
    face_track: bool = True,
    diar_timeline: list | None = None,
    analysis_out_path: str | None = None,
    speaker_faces: dict[str, list[float]] | None = None,
) -> None:
    fps, src_w, src_h = _video_meta(source_path)
    crop_h = (int(src_h * PORTRAIT_FILL) // 2) * 2
    crop_w = (int(crop_h * 9 / 16) // 2) * 2

    # Face-tracked crops assume a landscape source; square/portrait sources
    # would need a crop wider than the frame — letterbox instead.
    if crop_w > src_w:
        log.warning("Source too narrow for face-tracked crop (%dx%d) — letterboxing", src_w, src_h)
        face_track = False

    if not face_track:
        if len(cuts) == 1:
            _render_letterbox(source_path, float(cuts[0]["start"]), float(cuts[0]["end"]) - float(cuts[0]["start"]), output_path)
        else:
            tmp_files = [_mktemp(suffix=".mp4") for _ in cuts]
            try:
                with ThreadPoolExecutor() as pool:
                    futs = [pool.submit(_render_letterbox, source_path, float(c["start"]), float(c["end"]) - float(c["start"]), tmp)
                            for c, tmp in zip(cuts, tmp_files)]
                    for f in futs:
                        f.result()
                _concat(tmp_files, output_path)
            finally:
                _cleanup(tmp_files)
        return

    timeline = _analyze_source(source_path, fps, src_w, src_h, crop_w, crop_h,
                               diar_timeline=diar_timeline, analysis_out_path=analysis_out_path,
                               speaker_faces=speaker_faces)

    if len(cuts) == 1:
        _render_cut(source_path, cuts[0], timeline, output_path)
    else:
        tmp_files = [_mktemp(suffix=".mp4") for _ in cuts]
        try:
            with ThreadPoolExecutor() as pool:
                futs = [pool.submit(_render_cut, source_path, cut, timeline, tmp)
                        for cut, tmp in zip(cuts, tmp_files)]
                for f in futs:
                    f.result()
            _concat(tmp_files, output_path)
        finally:
            _cleanup(tmp_files)
    log_rss("portrait render done")


# ── Utilities ─────────────────────────────────────────────────────────────────

def _concat(tmp_files: list[str], output: str) -> None:
    # Re-encode audio (video stays stream-copied): each segment's AAC carries
    # encoder priming samples and a duration quantised to 1024-sample frames,
    # so stream-copying through the concat demuxer drifts audio ~20-45 ms per
    # joint. aresample heals the gaps/overlaps back onto the video timeline.
    list_file = _mktemp(suffix=".txt")
    try:
        with open(list_file, "w") as f:
            for p in tmp_files:
                f.write(f"file '{p}'\n")
        _run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_file,
              "-c:v", "copy",
              "-af", "aresample=async=1000:first_pts=0", "-c:a", "aac",
              output])
    finally:
        try:
            os.unlink(list_file)
        except Exception:
            pass


def _mktemp(suffix: str) -> str:
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    return path


def _cleanup(files: list[str]) -> None:
    for f in files:
        try:
            os.unlink(f)
        except Exception:
            pass


def _run(cmd: list[str]) -> None:
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed (exit {result.returncode}):\n"
            + result.stderr[-3000:].decode(errors="replace")
        )
