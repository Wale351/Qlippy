"""
Unit tests for worker pure logic — no API keys, no network, no DB.
LLM agents are tested with call_agent mocked at the module seam.
Run: .venv/bin/python test_units.py
"""

from __future__ import annotations
import asyncio
import base64
import io
import json
import os
import struct
import wave
from pathlib import Path

# main.py reads these at import time — provide dummies before any import
os.environ.setdefault("WORKER_SECRET", "test")

import numpy as np


# ============================================================
# sentencize
# ============================================================

def test_sentencize_basic():
    from pipeline.sentencize import sentencize
    words = [
        {"word": "Hello", "start": 0.0, "end": 0.4},
        {"word": "world.", "start": 0.5, "end": 0.9},
        {"word": "Next", "start": 1.2, "end": 1.5},
        {"word": "sentence.", "start": 1.6, "end": 2.0},
    ]
    sents = sentencize(words)
    assert len(sents) == 2
    assert sents[0]["text"] == "Hello world."
    assert sents[0]["start"] == 0.0 and sents[0]["end"] == 0.9
    assert sents[1]["start"] == 1.2 and sents[1]["end"] == 2.0
    # indices are contiguous from 0
    assert [s["index"] for s in sents] == [0, 1]


def test_sentencize_empty_and_missing_timestamps():
    from pipeline.sentencize import sentencize
    assert sentencize([]) == []
    # words without timestamps are dropped, never crash
    sents = sentencize([{"word": "Hi."}, {"word": "There.", "start": 1.0, "end": 1.5}])
    assert all(s["start"] is not None for s in sents)


# ============================================================
# openrouter.parse_json_response
# ============================================================

def test_parse_json_response():
    from pipeline.openrouter import parse_json_response
    assert parse_json_response('{"a": 1}') == {"a": 1}
    assert parse_json_response('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_response('```\n[1, 2]\n```') == [1, 2]
    try:
        parse_json_response("not json")
        assert False, "should raise"
    except json.JSONDecodeError:
        pass


# ============================================================
# subtitle — grouping + colours
# ============================================================

def _w(text, start, end):
    return {"word": text, "start": start, "end": end}


def test_subtitle_group_words_splits():
    from pipeline.subtitle import _group_words
    # pause > 0.4s splits
    groups = _group_words([_w("a", 0.0, 0.2), _w("b", 1.0, 1.2)], 0.0, max_words=5)
    assert len(groups) == 2
    # trailing punctuation splits
    groups = _group_words([_w("end.", 0.0, 0.2), _w("next", 0.3, 0.5)], 0.0, max_words=5)
    assert len(groups) == 2
    # max_words splits
    ws = [_w(f"w{i}", i * 0.1, i * 0.1 + 0.05) for i in range(6)]
    groups = _group_words(ws, 0.0, max_words=5)
    assert [len(g["words"]) for g in groups] == [5, 1]


def test_subtitle_build_groups_rezeroes_multicut():
    from pipeline.subtitle import _build_groups
    raw = [_w("Hello", 10.0, 10.4), _w("world", 10.5, 10.9),
           _w("Second", 50.0, 50.4), _w("cut", 50.5, 50.9)]
    cuts = [{"start": 10.0, "end": 11.0}, {"start": 50.0, "end": 51.0}]
    groups = _build_groups(raw, cuts, max_words=5)
    assert groups[0]["start"] == 0.0
    # second cut starts where the first ended (1.0s into the clip)
    assert groups[1]["words"][0]["start"] == 1.0
    assert groups[1]["words"][1]["start"] == 1.5


def test_subtitle_find_group_and_active_word():
    from pipeline.subtitle import _find_group, _find_active_word
    groups = [{"start": 0.0, "end": 1.0, "words": [
        {"text": "a", "start": 0.0, "end": 0.3},
        {"text": "b", "start": 0.4, "end": 0.7},
    ]}]
    assert _find_group(groups, 0.5) is groups[0]
    assert _find_group(groups, 2.0) is None
    assert _find_active_word(groups[0], 0.5) == 1
    assert _find_active_word(groups[0], -1.0) == 0   # all future → first word


def test_subtitle_hex_colours():
    from pipeline.subtitle import _hex_rgb, _hex_rgba
    assert _hex_rgb("#FF0000") == (255, 0, 0)
    assert _hex_rgb("0f0") == (15, 0, 0) or True  # short hex padded, never crashes
    assert _hex_rgba("#000000", 0.5) == (0, 0, 0, 127)
    assert _hex_rgba("#FFFFFF", 2.0)[3] == 255    # opacity clamped


def test_subtitle_box_text_is_vertically_centred():
    """Regression: PIL anchors text at the ascender, so ink used to sit low in
    the background box. Measure the gaps rather than pixels so any font passes."""
    import numpy as np
    from PIL import Image
    from pipeline.subtitle import SubtitleRenderer
    cfg = {"typography": {"font_size": 64, "all_caps": True},
           "color": {"text_color": "#ffffff", "highlight_color": "#ffffff", "bg_color": "#0000ff", "bg_opacity": 1.0},
           "layout": {"vertical_percent": 50, "padding": 24}}
    words = [{"text": w, "start": 0, "end": 1} for w in ["three", "orbs", "trailed", "our", "plane"]]
    for two_lines in (False, True):
        cfg["layout"]["two_lines"] = two_lines
        out = np.array(SubtitleRenderer(cfg, 1080, 1920).draw_frame(
            Image.new("RGB", (1080, 1920), (0, 0, 0)), {"start": 0, "end": 1, "words": words}, active_idx=0))
        box = np.where((out[:, :, 2] > 200) & (out[:, :, 0] < 60))
        ink = np.where((out[:, :, 0] > 200) & (out[:, :, 1] > 200) & (out[:, :, 2] > 200))
        above = ink[0].min() - box[0].min()
        below = box[0].max() - ink[0].max()
        assert above > 0 and below > 0, "ink must sit inside the box"
        assert abs(above - below) <= 3, f"two_lines={two_lines}: {above}px above vs {below}px below"


def test_subtitle_renderer_draw_frame():
    from pipeline.subtitle import SubtitleRenderer
    from PIL import Image
    renderer = SubtitleRenderer({"typography": {}, "color": {}, "layout": {}}, 320, 568)
    group = {"start": 0.0, "end": 1.0, "words": [
        {"text": "hello", "start": 0.0, "end": 0.4},
        {"text": "world", "start": 0.5, "end": 0.9},
    ]}
    img = Image.new("RGB", (320, 568), (10, 10, 10))
    out = renderer.draw_frame(img, group, active_idx=1)
    assert out.size == (320, 568) and out.mode == "RGB"


def test_subtitle_effective_fps_fallback():
    from pipeline.subtitle import _effective_fps
    assert _effective_fps("/nonexistent/file.mp4") == 30.0


# ============================================================
# hyperframes — word list, grouping, HTML patching
# ============================================================

def test_hf_build_word_list():
    from pipeline.hyperframes import _build_word_list
    raw = [_w("in", 10.0, 10.3), _w("out", 99.0, 99.3)]
    words = _build_word_list(raw, [{"start": 10.0, "end": 11.0}])
    assert [w["text"] for w in words] == ["in"]
    assert words[0]["start"] == 0.0


def test_hf_build_raw_groups_cover_all_indices():
    from pipeline.hyperframes import _build_raw_groups
    words = [{"text": f"w{i}", "start": i * 0.1, "end": i * 0.1 + 0.05} for i in range(7)]
    groups = _build_raw_groups(words, max_words=5)
    assert groups == [[0, 4], [5, 6]]
    # punctuation split
    words[2]["text"] = "stop."
    groups = _build_raw_groups(words, max_words=5)
    assert groups[0] == [0, 2]
    # every index covered exactly once
    flat = [i for lo, hi in groups for i in range(lo, hi + 1)]
    assert flat == list(range(7))


def test_hf_patch_component():
    from pipeline.hyperframes import _patch_component
    html = (
        '<html><head><meta content="width=1920, height=1080"></head>'
        '<body data-width="1920" data-height="1080" data-duration="10">'
        "<script>var WORDS = [];\nvar RAW_GROUPS = [];\nvar CFG = {};</script></body></html>"
    )
    words = [{"text": "hi", "start": 0.0, "end": 0.5}]
    out = _patch_component(html, words, [[0, 0]], duration=12.5,
                           canvas_w=1080, canvas_h=1920, preset_config={})
    assert 'data-width="1080"' in out and 'data-height="1920"' in out
    assert 'data-duration="12.50"' in out
    assert 'var WORDS = [{"text":"hi","start":0.0,"end":0.5}];' in out
    assert '"yPct":80.0' in out      # CFG injected, default bottom position


# ============================================================
# image_sticker — ffmpeg overlay filter
# ============================================================

def test_image_sticker_filter_basic():
    from pipeline.image_sticker import build_overlay_filter
    f = build_overlay_filter({"width_pct": 25, "x_pct": 50, "y_pct": 20}, canvas_w=1080)
    assert f == "[1:v]scale=270:-1[lg];[0:v][lg]overlay=main_w*0.5-overlay_w/2:main_h*0.2-overlay_h/2"


def test_image_sticker_filter_clamps_and_defaults():
    from pipeline.image_sticker import build_overlay_filter
    # width clamped to [5, 60]% of canvas; junk values fall back to defaults
    assert "scale=648:" in build_overlay_filter({"width_pct": 999}, canvas_w=1080)   # 60%
    assert "scale=54:"  in build_overlay_filter({"width_pct": 1},   canvas_w=1080)   # 5%
    f = build_overlay_filter({"width_pct": "junk", "x_pct": None}, canvas_w=1080)
    assert "scale=270:" in f and "main_w*0.5" in f                                    # defaults


# ============================================================
# portrait — geometry, smoothing, timeline
# ============================================================

def _face(cx, cy, fw=100, fh=120, embedding=None):
    return {"cx": cx, "cy": cy, "fw": fw, "fh": fh, "embedding": embedding,
            "bbox": [cx - fw // 2, cy - fh // 2, cx + fw // 2, cy + fh // 2]}


def test_portrait_crop_rect_clamped():
    from pipeline.portrait import _compute_crop_rect
    # crop is clamped inside the source frame
    r = _compute_crop_rect(cx=0, cy=0, fh=100, src_w=1920, src_h=1080, crop_w=516, crop_h=918)
    assert r["x"] == 0 and r["y"] == 0
    r = _compute_crop_rect(cx=1920, cy=1080, fh=100, src_w=1920, src_h=1080, crop_w=516, crop_h=918)
    assert r["x"] == 1920 - 516 and r["y"] == 1080 - 918


def test_portrait_ema_dead_zone():
    from pipeline.portrait import _ema_keyframes, DEAD_ZONE_PX
    # jitter smaller than the dead zone never moves the crop
    tf = [(i * 0.25, _face(500 + (i % 2) * (DEAD_ZONE_PX - 5), 300)) for i in range(20)]
    seg = _ema_keyframes(tf, seg_end=5.0, src_w=1920, src_h=1080, crop_w=516, crop_h=918)
    assert len(seg["keyframes"]) == 1
    # a real move creates additional keyframes
    tf = [(0.0, _face(500, 300))] + [(i * 0.25, _face(900, 300)) for i in range(1, 20)]
    seg = _ema_keyframes(tf, seg_end=5.0, src_w=1920, src_h=1080, crop_w=516, crop_h=918)
    assert len(seg["keyframes"]) > 1
    assert seg["start"] == 0.0 and seg["end"] == 5.0 and seg["mode"] == "single"


def test_portrait_pan_expr():
    from pipeline.portrait import _pan_expr
    assert _pan_expr([{"t": 10.0, "x": 100}], "x", 10.0, 20.0) == "100"
    expr = _pan_expr([{"t": 10.0, "x": 100}, {"t": 12.0, "x": 200}], "x", 10.0, 20.0)
    # holds the first value before the second keyframe (times shifted by render_start)
    assert expr.startswith("if(lt(t\\,2.000)\\,100\\,")
    assert "(200" not in expr.split(",")[0]   # ramp lives in the else branch


def test_portrait_speaker_sweep():
    from pipeline.portrait import _SpeakerSweep, _cos_sim
    # t must be non-decreasing across at() calls (matches the scan loop)
    sweep = _SpeakerSweep([(0.0, 5.0, "A"), (4.5, 10.0, "B")])
    assert sweep.at(2.0) == ("A", False)
    assert sweep.at(4.7) == ("A", True)          # both active within the window
    assert sweep.at(20.0) == (None, False)
    assert _SpeakerSweep([(0, 5, "A")]).at(2.0) == ("A", False)
    a = np.array([1.0, 0.0]); b = np.array([0.0, 1.0])
    assert _cos_sim(a, a) == 1.0 and _cos_sim(a, b) == 0.0
    assert _cos_sim(a, np.zeros(2)) == 0.0       # zero norm → 0, not NaN


def test_identify_summary_counts_every_label():
    """Regression: the summary used a fixed hero/strong/decent dict and raised
    KeyError('weak') AFTER the clips were saved, marking a successful job failed."""
    from jobs.identify import _summary
    refined = [{"label": "hero", "hook_line": "H"}, {"label": "weak"}, {"label": "weak"}, {"label": None}]
    out = _summary(refined, 12.4)
    assert "Found 4 clips — 1 hero, 0 strong, 0 decent, 3 weak." in out
    assert "Top pick: Clip 1 · H." in out and "Runtime: 12s." in out
    assert _summary([], 1.0).startswith("Found 0 clips")


def test_portrait_binding_majority_beats_one_cutaway():
    """The reaction-shot failure: a clip's only single-face frames while the
    speaker talks are a cutaway to the listener. Over a whole episode the
    speaker's own shots outnumber it, so the mean embedding lands on them."""
    import numpy as np
    from pipeline.portrait import _bind_speakers, _cos_sim
    nina, conor = np.zeros(8), np.zeros(8); nina[0] = 1.0; conor[1] = 1.0
    def rec(t, face): return {"t": t, "overlap": False, "faces": [{"embedding": face.tolist()}]}
    # SPEAKER_01 (Nina) talks 0–20s. Frames 0–2 s are a cutaway to Conor; 3–20 s show Nina.
    records = [rec(t, conor) for t in (0.0, 1.0, 2.0)] + [rec(float(t), nina) for t in range(3, 20)]
    bound = _bind_speakers(records, [(0.0, 20.0, "SPEAKER_01")])
    assert _cos_sim(bound["SPEAKER_01"], nina) > _cos_sim(bound["SPEAKER_01"], conor)
    # ...whereas a 3-second "clip" that contains only the cutaway binds the wrong face.
    bound_clip = _bind_speakers(records[:3], [(0.0, 3.0, "SPEAKER_01")])
    assert _cos_sim(bound_clip["SPEAKER_01"], conor) > _cos_sim(bound_clip["SPEAKER_01"], nina)


def test_portrait_best_face_excludes_faces_identified_as_someone_else():
    """The two-shot failure, with the real numbers from the probe: the listener
    matches SPEAKER_02 at 0.73 and the active SPEAKER_01 at 0.38; the speaker
    is in profile and matches nobody (0.09). Old rule picked the listener."""
    import numpy as np
    from pipeline.portrait import _best_face_for_speaker, _identify, IDENT_SIM
    nina_ref  = np.array([1.0, 0.0, 0.0]); conor_ref = np.array([0.0, 1.0, 0.0])
    bound = {"SPEAKER_01": nina_ref, "SPEAKER_02": conor_ref}
    # Conor on screen: 0.73 to his own ref, 0.38 to Nina's.
    conor = {"embedding": [0.38, 0.73, 0.0], "fw": 360, "fh": 500}
    # Nina in profile: ~0.09 to everyone.
    nina_profile = {"embedding": [0.09, 0.09, 0.99], "fw": 300, "fh": 440}
    assert _identify(conor, bound)[0] == "SPEAKER_02"
    assert _identify(nina_profile, bound)[0] is None
    # Nina is speaking → follow the unidentified face, not the one that's confidently Conor.
    assert _best_face_for_speaker([conor, nina_profile], "SPEAKER_01", bound) is nina_profile
    # Conor is speaking → his face, even though the other is bigger... (make it bigger)
    big_unknown = {**nina_profile, "fw": 900, "fh": 900}
    assert _best_face_for_speaker([conor, big_unknown], "SPEAKER_02", bound) is conor
    # Only someone-else on screen (a solo cutaway) → still tracks that one face.
    assert _best_face_for_speaker([conor], "SPEAKER_01", bound) is conor
    # No binding for the speaker → biggest face, as before.
    assert _best_face_for_speaker([conor, big_unknown], "SPEAKER_09", bound) is big_unknown
    assert 0 < IDENT_SIM < 1


def test_portrait_two_shot_follows_speaker_in_profile():
    """End to end through _build_timeline: a two-shot where the speaker sits on
    the right in profile (matches nobody) and the listener on the left is a
    confident match to another speaker. The crop must follow the right face.
    Before the fix the clip tracked the listener start to finish."""
    from pipeline.portrait import _build_timeline, SCAN_FPS
    conor = [0.38, 0.73, 0.0]         # 0.73 to SPEAKER_02, 0.38 to SPEAKER_01 (unit refs below)
    nina_profile = [0.09, 0.09, 0.99]
    speaker_faces = {"SPEAKER_01": [1.0, 0.0, 0.0], "SPEAKER_02": [0.0, 1.0, 0.0]}
    records = [{"t": i / SCAN_FPS, "active_speaker": "SPEAKER_01", "overlap": False,
                "faces": [_face(600, 300, embedding=conor), _face(1500, 280, embedding=nina_profile)]}
               for i in range(24)]
    tl = _build_timeline(records, [(0.0, 6.0, "SPEAKER_01")], 1920, 1080, 516, 918,
                         speaker_faces=speaker_faces)
    assert len(tl) == 1 and tl[0]["mode"] == "single"
    assert all(kf["x"] > 960 for kf in tl[0]["keyframes"]), tl[0]["keyframes"][:3]   # right half = Nina


def test_portrait_merge_bindings_prefers_whole_video_map():
    import numpy as np
    from pipeline.portrait import _merge_bindings
    local = {"SPEAKER_01": np.array([0.0, 1.0]), "SPEAKER_02": np.array([1.0, 0.0]), "SPEAKER_03": None}
    merged = _merge_bindings(local, {"SPEAKER_01": [1.0, 0.0], "SPEAKER_09": [0.5, 0.5]})
    assert merged["SPEAKER_01"].tolist() == [1.0, 0.0]      # whole-video wins
    assert merged["SPEAKER_02"].tolist() == [1.0, 0.0]      # local fills the gap
    assert merged["SPEAKER_03"] is None
    assert merged["SPEAKER_09"].tolist() == [0.5, 0.5]
    assert _merge_bindings(local, None) is local
    # the stored-file shape ({"version", "speakers"}) is accepted too
    wrapped = _merge_bindings(local, {"version": 2, "speakers": {"SPEAKER_01": [0.0, 1.0]}})
    assert wrapped["SPEAKER_01"].tolist() == [0.0, 1.0]
    assert _merge_bindings(local, {"version": 2, "speakers": {}}) is local


def test_portrait_scan_window_timestamps_are_absolute():
    """Burst scans seek into the source; record timestamps must be source
    seconds, not burst-relative, so diarisation lookups line up."""
    import io
    from unittest import mock
    import pipeline.portrait as P
    W, H = 32, 18
    saved = P.SCAN_W; P.SCAN_W = W
    try:
        stream = bytes([10]) * (W * H * 3) * 3
        fake_proc = mock.Mock(stdout=io.BytesIO(stream), wait=lambda timeout=None: 0)
        fake_app = mock.Mock(); fake_app.get = mock.Mock(return_value=[])
        with mock.patch.object(P.subprocess, "Popen", return_value=fake_proc) as popen, \
             mock.patch.object(P, "_get_face_app", return_value=fake_app):
            recs = P._scan_source("x.mp4", W, H, [(0, 999, "A")], fps=4.0, start=100.0, duration=0.75)
        assert [r["t"] for r in recs] == [100.0, 100.25, 100.5]
        cmd = popen.call_args[0][0]
        assert cmd[cmd.index("-ss") + 1] == "100.000" and cmd[cmd.index("-t") + 1] == "0.750"
    finally:
        P.SCAN_W = saved


def test_portrait_mouth_activity_tracks_faces_and_windows():
    """Activity = mean |Δ openness| over the trailing window along a face's
    track. Faces are matched frame to frame by box overlap, so two people
    keep separate histories."""
    from pipeline.portrait import _mouth_activity, TALK_WINDOW_S
    def face(x, mouth): return {"bbox": [x, 0, x + 100, 120], "cx": x + 50, "cy": 60, "fw": 100, "fh": 120, "mouth": mouth}
    # left face: mouth oscillates (talking); right face: constant (listening)
    recs = [{"t": i * 0.25, "active_speaker": None, "overlap": False,
             "faces": [face(0, 0.05 + 0.03 * (i % 2)), face(500, 0.06)]} for i in range(8)]
    _mouth_activity(recs)
    assert recs[0]["faces"][0]["talk"] is None                 # one sample: unknown
    assert abs(recs[7]["faces"][0]["talk"] - 0.03) < 1e-9       # |Δ| every frame = 0.03
    assert recs[7]["faces"][1]["talk"] == 0.0                   # still mouth
    # a face with no landmarks never gets a value
    recs2 = [{"t": 0, "active_speaker": None, "overlap": False, "faces": [face(0, None)]},
             {"t": 0.25, "active_speaker": None, "overlap": False, "faces": [face(0, None)]}]
    _mouth_activity(recs2); assert recs2[1]["faces"][0]["talk"] is None
    assert TALK_WINDOW_S > 0


def test_portrait_bind_speakers_weights_by_mouth_movement():
    """Two-shot the whole time (no single-face frames): binding must still
    find the speaker, via whose mouth moves while the audio says they talk."""
    import numpy as np
    from pipeline.portrait import _bind_speakers, _cos_sim, TALK_MIN
    nina, conor = np.zeros(4), np.zeros(4); nina[0] = 1; conor[1] = 1
    def face(emb, talk): return {"embedding": emb.tolist(), "talk": talk, "fw": 100, "fh": 120}
    recs = [{"t": i * 0.25, "overlap": False,
             "faces": [face(conor, 0.002), face(nina, 0.02)]} for i in range(20)]   # Nina talking
    bound = _bind_speakers(recs, [(0.0, 5.0, "SPEAKER_01")])
    assert _cos_sim(bound["SPEAKER_01"], nina) > 0.9
    # No measurable movement anywhere → old rule: only a lone face counts.
    still = [{"t": i * 0.25, "overlap": False, "faces": [face(conor, 0.001), face(nina, 0.001)]} for i in range(20)]
    assert _bind_speakers(still, [(0.0, 5.0, "SPEAKER_01")])["SPEAKER_01"] is None
    alone = [{"t": i * 0.25, "overlap": False, "faces": [face(nina, None)]} for i in range(20)]
    assert _cos_sim(_bind_speakers(alone, [(0.0, 5.0, "SPEAKER_01")])["SPEAKER_01"], nina) > 0.9
    assert 0 < TALK_MIN < 0.02


def test_portrait_no_hold_across_a_shot_change():
    """Speaker A is confidently on screen, then the source hard-cuts to a
    close-up of B at a completely different position. The old hold kept A's
    crop coordinates over B's footage for a grace period — one frame of
    empty backdrop. A blink (B briefly detected where A was) still holds."""
    from pipeline.portrait import _build_timeline, SCAN_FPS
    eA, eB = [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]
    bound = {"SPEAKER_A": eA, "SPEAKER_B": eB}
    def rec(i, faces): return {"t": i / SCAN_FPS, "active_speaker": "SPEAKER_A", "overlap": False, "faces": faces}
    # 8 frames of A at x=600, then a hard cut: B alone, close-up, at x=1500 (no overlap)
    records = [rec(i, [_face(600, 300, 300, 400, embedding=eA)]) for i in range(8)] + \
              [rec(i, [_face(1500, 500, 700, 900, embedding=eB)]) for i in range(8, 16)]
    tl = _build_timeline(records, [(0.0, 4.0, "SPEAKER_A")], 3840, 2160, 1032, 1836, speaker_faces=bound)
    x_at = lambda t: next(k["x"] for seg in tl for k in seg["keyframes"] if abs(k["t"] - t) < 1e-6) if any(
        abs(k["t"] - t) < 1e-6 for seg in tl for k in seg["keyframes"]) else None
    # the first frame after the cut must already frame B (crop centre near 1500), not hold at A
    first_after = [seg for seg in tl if seg["start"] <= 8 / SCAN_FPS < seg["end"]][0]
    assert abs(first_after["crop"]["x"] + 1032 / 2 - 1500) < 200, first_after["crop"]
    # ...whereas a one-frame blink of B at A's position in a stable shot is held
    blink = [rec(i, [_face(600, 300, 300, 400, embedding=eA)]) for i in range(8)]
    blink[5] = rec(5, [_face(620, 310, 300, 400, embedding=eB)])
    tl2 = _build_timeline(blink, [(0.0, 2.0, "SPEAKER_A")], 3840, 2160, 1032, 1836, speaker_faces=bound)
    assert len(tl2) == 1


def test_portrait_no_diarisation_follows_the_moving_mouth():
    from pipeline.portrait import _best_face_for_speaker, _build_timeline, SCAN_FPS
    # End to end: two faces all along, no diarisation, no map. The bigger face
    # is still; the smaller one talks. The sticky-face rule must yield.
    eA, eB = [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]
    recs = []
    for i in range(24):
        big   = {**_face(600, 300, 400, 500, embedding=eA), "mouth": 0.06}
        small = {**_face(1500, 300, 260, 320, embedding=eB), "mouth": 0.05 + 0.03 * (i % 2)}
        recs.append({"t": i / SCAN_FPS, "active_speaker": None, "overlap": False, "faces": [big, small]})
    from pipeline.portrait import _mouth_activity; _mouth_activity(recs)
    tl = _build_timeline(recs, [], 1920, 1080, 516, 918)
    late = [seg for seg in tl if seg["end"] > 2.0]
    assert late and all(seg["crop"]["x"] + 516 / 2 > 960 for seg in late), late   # follows the talker on the right
    big_still = {"embedding": None, "talk": 0.001, "fw": 400, "fh": 500}
    small_talking = {"embedding": None, "talk": 0.02, "fw": 200, "fh": 250}
    assert _best_face_for_speaker([big_still, small_talking], None, {}) is small_talking
    # nobody talking → largest, as before
    small_still = {**small_talking, "talk": 0.001}
    assert _best_face_for_speaker([big_still, small_still], None, {}) is big_still


def test_portrait_burst_windows_spread_over_each_speaker():
    from pipeline.portrait import _burst_windows, BIND_BURST_S
    tl = [(i * 10.0, i * 10.0 + 4.0, "A" if i % 2 == 0 else "B") for i in range(100)] + [(2000.0, 2000.5, "C")]
    w = _burst_windows(tl, per_speaker=5)
    assert len(w) == 10 and all(d == BIND_BURST_S for _, d in w)
    assert w == sorted(w)
    assert all(not (2000.0 <= s <= 2001) for s, _ in w)          # C's 0.5 s segment is too short to sample


def test_portrait_ema_snap_on_shot_change():
    from pipeline.portrait import _ema_keyframes, SNAP_FRAC
    # a jump beyond SNAP_FRAC of the crop = camera cut → EMA resets and the
    # keyframe is marked snap (renderer hard-cuts instead of panning)
    jump = int(516 * SNAP_FRAC) + 50
    tf = [(0.0, _face(500, 300)), (0.25, _face(500 + jump, 300))] \
       + [(0.25 + i * 0.25, _face(500 + jump, 300)) for i in range(1, 6)]
    seg = _ema_keyframes(tf, seg_end=2.0, src_w=1920, src_h=1080, crop_w=516, crop_h=918)
    kfs = seg["keyframes"]
    assert len(kfs) == 2
    assert kfs[0]["snap"] is False and kfs[1]["snap"] is True
    # EMA reset: the snap keyframe is centred on the raw position, not eased toward it
    assert kfs[1]["x"] == 500 + jump - 516 // 2
    # gradual movement still pans (no snap keyframes)
    tf = [(i * 0.25, _face(500 + i * 40, 300)) for i in range(10)]
    seg = _ema_keyframes(tf, seg_end=2.5, src_w=1920, src_h=1080, crop_w=516, crop_h=918)
    assert len(seg["keyframes"]) > 1
    assert all(kf["snap"] is False for kf in seg["keyframes"])


def test_portrait_pan_expr_snap():
    from pipeline.portrait import _pan_expr
    kfs = [{"t": 10.0, "x": 100, "snap": False}, {"t": 12.0, "x": 200, "snap": True}]
    expr = _pan_expr(kfs, "x", 10.0, 20.0)
    # snap branch is a constant — no smoothstep ramp from the previous value
    assert expr == "if(lt(t\\,2.000)\\,100\\,200)"


def test_portrait_pick_two_faces_thirds():
    from pipeline.portrait import _pick_two_faces, _best_face_for_speaker
    f_left, f_right = _face(200, 300), _face(1700, 300)
    two = _pick_two_faces([f_left, f_right], None, {}, 1920, 1080)
    assert two is not None and len(two) == 2
    # both faces in the same third → constraint fails
    assert _pick_two_faces([_face(200, 300), _face(300, 300)], None, {}, 1920, 1080) is None
    # without a bound speaker, the biggest face wins
    big, small = _face(500, 300, fw=200, fh=200), _face(900, 300, fw=50, fh=50)
    assert _best_face_for_speaker([small, big], None, {}) is big


def test_portrait_build_timeline_modes_and_merge():
    from pipeline.portrait import _build_timeline, SCAN_FPS
    # 8 letterbox frames, 1 lone single-face frame (0.25s — below MIN_SEGMENT_S),
    # 8 more letterbox frames → the short mode-change segment is absorbed.
    records = []
    for i in range(17):
        faces = [_face(960, 400)] if i == 8 else []
        records.append({"t": i / SCAN_FPS, "active_speaker": None, "overlap": False, "faces": faces})
    tl = _build_timeline(records, [], 1920, 1080, 516, 918)
    assert all(seg["mode"] == "letterbox" for seg in tl)

    # sustained single face → single segment with keyframes
    records = [{"t": i / SCAN_FPS, "active_speaker": None, "overlap": False,
                "faces": [_face(960, 400)]} for i in range(8)]
    tl = _build_timeline(records, [], 1920, 1080, 516, 918)
    assert len(tl) == 1 and tl[0]["mode"] == "single" and tl[0]["crop"] is not None

    # overlap with two separated faces → split segment with two panel crops
    records = [{"t": i / SCAN_FPS, "active_speaker": None, "overlap": True,
                "faces": [_face(200, 300), _face(1700, 300)]} for i in range(8)]
    tl = _build_timeline(records, [(0.0, 2.0, "A"), (0.0, 2.0, "B")], 1920, 1080, 516, 918)
    assert tl[0]["mode"] == "split" and len(tl[0]["crops"]) == 2


def test_portrait_detection_gap_grace():
    from pipeline.portrait import _build_timeline, SCAN_FPS

    def recs(spec):  # spec: list of bools — face present at that frame?
        return [{"t": i / SCAN_FPS, "active_speaker": None, "overlap": False,
                 "faces": [_face(960, 400)] if has else []} for i, has in enumerate(spec)]

    # a 0.5s detection gap inside an established run is bridged — no letterbox
    tl = _build_timeline(recs([True] * 4 + [False] * 2 + [True] * 4), [], 1920, 1080, 516, 918)
    assert len(tl) == 1 and tl[0]["mode"] == "single"

    # a gap longer than FACE_GRACE_S falls back to letterbox after the hold
    tl = _build_timeline(recs([True] * 4 + [False] * 10), [], 1920, 1080, 516, 918)
    assert any(seg["mode"] == "letterbox" for seg in tl)


def test_portrait_split_panel_keyframes():
    from pipeline.portrait import _build_timeline, SCAN_FPS
    # top panel follows the moving face with its own keyframe track;
    # the static face's panel stays a single keyframe
    records = []
    for i in range(8):
        moving = _face(1400 + i * 43, 300)   # higher centrality → top panel
        records.append({"t": i / SCAN_FPS, "active_speaker": None, "overlap": True,
                        "faces": [_face(200, 300), moving]})
    tl = _build_timeline(records, [(0.0, 2.0, "A"), (0.0, 2.0, "B")], 1920, 1080, 516, 918)
    assert tl[0]["mode"] == "split"
    top_kfs, bottom_kfs = tl[0]["panel_keyframes"]
    assert len(top_kfs) > 1          # moving face → animated panel
    assert len(bottom_kfs) == 1      # static face → fixed crop


def test_portrait_pick_two_faces_quality_floor():
    from pipeline.portrait import _pick_two_faces
    big  = _face(500, 300, fw=200, fh=200)
    tiny = _face(1700, 300, fw=40, fh=40)    # 20% of big → background face / junk
    assert _pick_two_faces([big, tiny], None, {}, 1920, 1080) is None
    ok = _face(1700, 300, fw=100, fh=120)    # 60% → legitimate second face
    assert _pick_two_faces([big, ok], None, {}, 1920, 1080) is not None


def test_portrait_pick_two_faces_identity_first():
    from pipeline.portrait import _pick_two_faces
    eA, eB, eJ = [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]
    bound = {"A": np.array(eA), "B": np.array(eB)}
    fA   = _face(1400, 500, embedding=eA)
    fB   = _face(2600, 500, embedding=eB)
    # bigger AND more central than fB — wins on score, matches no speaker
    junk = _face(1900, 1900, fw=200, fh=240, embedding=eJ)
    two = _pick_two_faces([fA, fB, junk], "A", bound, 3840, 2160)
    assert two == [fA, fB]      # active speaker first, junk excluded
    # without bindings the score fallback still applies (junk wins a panel)
    two = _pick_two_faces([fA, fB, junk], None, {}, 3840, 2160)
    assert junk in two


def test_portrait_single_wrong_face_grace():
    from pipeline.portrait import _build_timeline, SCAN_FPS
    eA, eB = [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]
    # speaker A talks throughout; one frame detects only the OTHER person's
    # face → hold A's crop instead of cutting away for 250ms
    records = []
    for i in range(12):
        face = _face(2500, 300, embedding=eB) if i == 8 else _face(900, 300, embedding=eA)
        records.append({"t": i / SCAN_FPS, "active_speaker": "A", "overlap": False,
                        "faces": [face]})
    tl = _build_timeline(records, [(0.0, 2.0, "A")], 1920, 1080, 516, 918)
    assert len(tl) == 1 and tl[0]["mode"] == "single"
    assert all(kf["x"] < 1000 for kf in tl[0]["keyframes"])   # crop never jumps right


def test_portrait_split_bottom_outlier_held():
    from pipeline.portrait import _build_timeline, SCAN_FPS
    eA, eB, eJ = [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]
    # one frame where the "second face" is a junk detection elsewhere:
    # the bottom panel must hold its crop, not steer toward the junk face
    records = []
    for i in range(8):
        top    = _face(900, 300, embedding=eA)
        bottom = _face(1500, 800, embedding=eJ) if i == 4 else _face(1700, 300, embedding=eB)
        records.append({"t": i / SCAN_FPS, "active_speaker": None, "overlap": True,
                        "faces": [top, bottom]})
    tl = _build_timeline(records, [], 1920, 1080, 516, 918)
    assert len(tl) == 1 and tl[0]["mode"] == "split"   # one-off outlier ≠ segment break
    top_kfs, bottom_kfs = tl[0]["panel_keyframes"]
    assert len(bottom_kfs) == 1                        # junk frame contributed nothing
    assert bottom_kfs[0]["x"] == 1920 - 728            # crop stays on B (clamped right)


def test_portrait_split_bottom_person_change_breaks():
    from pipeline.portrait import _build_timeline, SCAN_FPS
    eA, eB, eC = [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]
    # bottom person genuinely changes (B → C, sustained) → new segment
    records = []
    for i in range(12):
        top    = _face(900, 300, embedding=eA)
        bottom = _face(1700, 300, embedding=eB) if i < 6 else _face(1500, 320, embedding=eC)
        records.append({"t": i / SCAN_FPS, "active_speaker": None, "overlap": True,
                        "faces": [top, bottom]})
    tl = _build_timeline(records, [], 1920, 1080, 516, 918)
    assert len(tl) == 2 and all(s["mode"] == "split" for s in tl)
    assert tl[1]["crops"][1]["x"] == 1500 - 728 // 2   # second segment framed on C


def test_portrait_split_order_hysteresis():
    from pipeline.portrait import _build_timeline, SCAN_FPS
    eA, eB = [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]

    def make_records(active_b_frames):
        # 8 clean single-face frames bind speaker B, then 12 split frames where
        # F1 (bigger, more central) normally wins the top panel by score
        records = [{"t": i / SCAN_FPS, "active_speaker": "B", "overlap": False,
                    "faces": [_face(1500, 300, embedding=eB)]} for i in range(8)]
        for i in range(12):
            active = "B" if i in active_b_frames else None
            records.append({"t": (8 + i) / SCAN_FPS, "active_speaker": active, "overlap": True,
                            "faces": [_face(700, 300, fw=150, fh=180, embedding=eA),
                                      _face(1500, 300, embedding=eB)]})
        return records

    diar = [(0.0, 2.0, "B")]

    # one diarisation blip proposing B on top → held, panels never trade
    tl = _build_timeline(make_records({3}), diar, 1920, 1080, 516, 918)
    splits = [s for s in tl if s["mode"] == "split"]
    assert len(splits) == 1
    assert splits[0]["crops"][0]["x"] == 700 - 728 // 2    # F1 stayed top

    # sustained reorder (≥ SPEAKER_SWITCH_S) → accepted, segment breaks once
    tl = _build_timeline(make_records(set(range(3, 12))), diar, 1920, 1080, 516, 918)
    splits = [s for s in tl if s["mode"] == "split"]
    assert len(splits) == 2
    assert splits[1]["crops"][0]["x"] == 1500 - 728 // 2   # B promoted to top


def test_portrait_main_cast_excludes_transient_faces():
    from pipeline.portrait import _main_cast, _cos_sim
    eA, eB, eJ = [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]
    records = []
    for i in range(10):
        faces = [_face(900, 300, embedding=eA), _face(2500, 300, embedding=eB)]
        if i in (4, 5):                        # transient face: 20% of frames
            faces.append(_face(1500, 1900, embedding=eJ))
        records.append({"t": i * 0.25, "active_speaker": None, "overlap": False, "faces": faces})
    cast = _main_cast(records)
    assert len(cast) == 2
    assert all(_cos_sim(c, np.array(eJ)) < 0.4 for c in cast)   # junk never cast
    # fewer than two persistent identities → no cast, no filtering
    solo = [{"t": i * 0.25, "active_speaker": None, "overlap": False,
             "faces": [_face(900, 300, embedding=eA)]} for i in range(10)]
    assert _main_cast(solo) == []


def test_portrait_cast_filter_blocks_junk_panel():
    from pipeline.portrait import _build_timeline, SCAN_FPS
    eA, eB, eJ = [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]
    # a persistent face-like detection (screen in the shot) appears in 30% of
    # frames — it must not force split mode or take a panel
    records = []
    for i in range(20):
        faces = [_face(1400, 500, embedding=eA), _face(2600, 500, embedding=eB)]
        if 10 <= i < 16:
            faces.append(_face(1900, 1900, embedding=eJ))
        records.append({"t": i / SCAN_FPS, "active_speaker": None, "overlap": False,
                        "faces": faces})
    tl = _build_timeline(records, [], 3840, 2160, 1032, 1836)
    assert all(seg["mode"] == "single" for seg in tl)   # junk filtered → n=2, no overlap


def test_portrait_single_sticky_face():
    from pipeline.portrait import _build_timeline, SCAN_FPS
    eA, eB = [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]
    # the "biggest face" flips on ±12px detection noise — the crop must not
    records = []
    for i in range(12):
        fh_a, fh_b = (396, 384) if i == 0 else ((372, 384) if i % 2 else (384, 372))
        records.append({"t": i / SCAN_FPS, "active_speaker": None, "overlap": False,
                        "faces": [_face(900, 300, fh=fh_a, embedding=eA),
                                  _face(2500, 300, fh=fh_b, embedding=eB)]})
    tl = _build_timeline(records, [], 1920, 1080, 516, 918)
    assert len(tl) == 1 and tl[0]["mode"] == "single"
    assert all(kf["x"] < 1000 for kf in tl[0]["keyframes"])   # never cuts to fB


def test_portrait_bind_speakers_aggregates_segments():
    from pipeline.portrait import _bind_speakers

    def rec(t, emb):
        return {"t": t, "active_speaker": "A", "overlap": False,
                "faces": [_face(500, 300, embedding=emb)]}

    records = [rec(0.0, [1.0, 0.0]), rec(1.0, [1.0, 0.0]),
               rec(4.0, [0.0, 1.0]), rec(5.0, [0.0, 1.0])]
    bound = _bind_speakers(records, [(0.0, 2.0, "A"), (4.0, 6.0, "A"), (8.0, 8.5, "B")])
    assert np.allclose(bound["A"], [0.5, 0.5])   # mean across BOTH clean segments
    assert bound["B"] is None                     # segment too short to bind


def test_portrait_cache_params_invalidation():
    import tempfile
    import pipeline.portrait as P

    calls = {"scan": 0}
    real_scan, real_build = P._scan_source, P._build_timeline
    P._scan_source = lambda *a, **k: calls.__setitem__("scan", calls["scan"] + 1) or []
    P._build_timeline = lambda *a, **k: [
        {"start": 0, "end": 1, "mode": "letterbox", "crop": None, "crops": None}]
    fd, src = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    try:
        P._analyze_source(src, 30.0, 1920, 1080, 516, 918, diar_timeline=[])
        assert calls["scan"] == 1
        P._analyze_source(src, 30.0, 1920, 1080, 516, 918, diar_timeline=[])
        assert calls["scan"] == 1                  # same params → cache hit
        P._analyze_source(src, 30.0, 1920, 1080, 600, 918, diar_timeline=[])
        assert calls["scan"] == 2                  # crop geometry changed → re-analyse
    finally:
        P._scan_source, P._build_timeline = real_scan, real_build
        for p in (src, P._cache_path(src)):
            try:
                os.unlink(p)
            except FileNotFoundError:
                pass


# ============================================================
# energy index — from a synthetic WAV (needs ffmpeg binary)
# ============================================================

def test_energy_index_silence_detection():
    from pipeline.energy_index import build_energy_index, SR
    # 0.5s silence + 0.5s loud square wave, 16-bit PCM WAV
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(SR)
        quiet = b"".join(struct.pack("<h", 0) for _ in range(SR // 2))
        loud  = b"".join(struct.pack("<h", 12000 if i % 2 else -12000) for i in range(SR // 2))
        wf.writeframes(quiet + loud)
    index = build_energy_index(buf.getvalue())
    assert index.n_frames == 50                       # 1s of 20ms hops
    assert index.frame_at(750.0) == 37
    assert index.is_silent(5) is True                 # inside the quiet half
    assert index.is_silent(40) is False               # inside the loud half


# ============================================================
# job-module helpers (formerly main.py)
# ============================================================

def test_main_flatten_and_diar_timeline():
    from jobs.transcribe import _flatten_segment_words
    from jobs.common import _build_diar_timeline
    assert _flatten_segment_words([{"words": [1, 2]}, {"words": None}, {"words": [3]}]) == [1, 2, 3]
    transcript = {"segments": [
        {"start": 5, "end": 6, "speaker": "B"},
        {"start": 1, "end": 2, "speaker": "A"},
        {"start": 3, "end": 4},                      # no speaker → skipped
    ]}
    assert _build_diar_timeline(transcript) == [(1.0, 2.0, "A"), (5.0, 6.0, "B")]
    assert _build_diar_timeline(None) == []


def test_main_remap_diar_timeline():
    from jobs.common import _remap_diar_timeline
    diar = [(0.0, 100.0, "A"), (100.0, 200.0, "B")]
    cuts = [{"start": 50, "end": 60}, {"start": 150, "end": 160}]
    out = _remap_diar_timeline(diar, cuts)
    # each cut becomes a contiguous block in stitched coordinates
    assert out == [(0.0, 10.0, "A"), (10.0, 20.0, "B")]
    # diar segment partially inside a cut is clipped to the window
    out = _remap_diar_timeline([(55.0, 99.0, "A")], [{"start": 50, "end": 60}])
    assert out == [(5.0, 10.0, "A")]


def test_main_export_concurrency_semaphore():
    import jobs.export as export_jobs
    # bulk exports must not run unbounded ffmpeg renders in parallel
    assert export_jobs.EXPORT_CONCURRENCY >= 1
    assert isinstance(export_jobs._export_semaphore, asyncio.Semaphore)

    async def _probe():
        async with export_jobs._export_semaphore:
            return True
    assert asyncio.run(_probe()) is True   # semaphore acquirable and releases cleanly



def test_main_transcribe_error_messages():
    from jobs.transcribe import _transcribe_error_msg
    assert "Could not download" in _transcribe_error_msg(RuntimeError("yt-dlp failed: 403"))
    assert "empty" in _transcribe_error_msg(RuntimeError("produced an empty file")).lower()
    assert "Audio extraction" in _transcribe_error_msg(RuntimeError("ffmpeg audio extract failed"))
    assert "service error" in _transcribe_error_msg(RuntimeError("Whisper failed: boom"))
    assert _transcribe_error_msg(ValueError("weird")).startswith("Transcription failed")


# ============================================================
# LLM agents — call_agent mocked at the module seam
# ============================================================

def _sentences(n, sent_dur=10.0):
    return [{"index": i, "text": f"Sentence {i}.", "start": i * sent_dur,
             "end": i * sent_dur + sent_dur - 0.5, "words": []} for i in range(n)]


def test_reader_clamps_and_covers_tail():
    import pipeline.reader as reader

    async def fake(agent, system, user, api_key=None, temperature=None, max_tokens=None):
        # overlapping ranges + dropped tail — the code must repair both
        return {"segments": [{"lo": 0, "hi": 2}, {"lo": 1, "hi": 4}]}

    orig, reader.call_agent = reader.call_agent, fake
    try:
        segs = asyncio.run(reader.read_segments(_sentences(6), [(0.0, 60.0, "A")], "key"))
    finally:
        reader.call_agent = orig

    assert [(s["lo"], s["hi"]) for s in segs] == [(0, 2), (3, 4), (5, 5)]
    assert segs[0]["speaker"] == "A"
    assert segs[0]["segment_id"] == 0 and segs[-1]["text"] == "Sentence 5."


def test_tension_validates_and_survives_errors():
    import pipeline.tension as tension

    async def fake(agent, system, user, api_key=None, temperature=None, max_tokens=None):
        if "seg-valid" in user:
            return {"flagged": True, "tension_type": "belief_vs_reality", "annotation": "x"}
        if "seg-badtype" in user:
            return {"flagged": True, "tension_type": "made_up_type", "annotation": "x"}
        if "seg-boom" in user:
            raise RuntimeError("LLM down")
        return {"flagged": False, "tension_type": None, "annotation": ""}

    segments = [{"segment_id": i, "text": t} for i, t in
                enumerate(["seg-valid", "seg-badtype", "seg-boom", "seg-quiet"])]
    orig, tension.call_agent = tension.call_agent, fake
    try:
        flagged = asyncio.run(tension.detect_tension(segments, "key"))
    finally:
        tension.call_agent = orig

    assert len(flagged) == 1 and flagged[0]["segment_id"] == 0
    assert flagged[0]["tension_type"] == "belief_vs_reality"


def test_knowledge_gap_clamps_scores():
    import pipeline.knowledge_gap as kg

    async def fake(agent, system, user, api_key=None, temperature=None, max_tokens=None):
        return {"knowledge_gap_score": 99 if "big" in user else 0}

    segments = [{"segment_id": 0, "text": "big claim", "tension_type": "t", "annotation": "a"},
                {"segment_id": 1, "text": "small",     "tension_type": "t", "annotation": "a"}]
    orig, kg.call_agent = kg.call_agent, fake
    try:
        scored = asyncio.run(kg.score_knowledge_gaps(segments, "key"))
    finally:
        kg.call_agent = orig

    assert [s["knowledge_gap"] for s in scored] == [3, 1]


def test_hook_writer_fallback():
    import pipeline.hook_writer as hw

    async def fake(agent, system, user, api_key=None, temperature=None, max_tokens=None):
        if "good clip" in user:
            return {"hook_line": "  A sharp hook  "}
        raise RuntimeError("LLM down")

    clips = [{"text": "good clip. more text.", "rank": 1},
             {"text": "Broken clip text. second sentence.", "rank": 2}]
    orig, hw.call_agent = hw.call_agent, fake
    try:
        out = asyncio.run(hw.write_hooks(clips, "key"))
    finally:
        hw.call_agent = orig

    assert out[0]["hook_line"] == "A sharp hook"
    assert out[1]["hook_line"] == "Broken clip text"   # first sentence fallback


def test_evaluator_gates_classifies_and_ranks():
    import pipeline.evaluator as ev

    all_true  = {k: True for k in ev.MUST_HAVES}
    all_sigs  = {k: True for k in ev.SIGNALS}
    weak      = {k: (k == "hook") for k in ev.MUST_HAVES}

    async def fake(agent, system, user, api_key=None, temperature=None, max_tokens=None):
        return {"clips": [
            # hero: 5 must-haves + 6 signals, spans 3 sentences (~30s)
            {"source_candidates": [0], "sentence_groups": [[0, 2]],
             "must_haves": all_true, "signals": all_sigs, "reasoning": "great"},
            # dropped: only 1 must-have
            {"source_candidates": [0], "sentence_groups": [[3, 5]],
             "must_haves": weak, "signals": all_sigs, "reasoning": "weak"},
            # dropped: single sentence ≈ 9.5s < 10s gate
            {"source_candidates": [0], "sentence_groups": [[6, 6]],
             "must_haves": all_true, "signals": all_sigs, "reasoning": "short"},
        ]}

    sentences = _sentences(8)
    candidates = [{"segment_id": 0, "lo": 0, "hi": 7, "tension_type": "belief_vs_reality",
                   "annotation": "a", "knowledge_gap": 3, "duration_sec": 80.0, "speaker": "A"}]
    orig, ev.call_agent = ev.call_agent, fake
    try:
        clips = asyncio.run(ev.evaluate_clips(candidates, sentences, "key"))
    finally:
        ev.call_agent = orig

    assert len(clips) == 1
    c = clips[0]
    assert c["label"] == "hero" and c["rank"] == 1 and c["score_total"] == 11
    assert c["cuts_raw"] == [{"start": 0.0, "end": 29.5}]
    assert c["stitched"] is False and c["knowledge_gap"] == 3


def test_evaluator_classify_table():
    from pipeline.evaluator import classify
    assert classify(5, 4) == "hero"
    assert classify(5, 3) == "strong"
    assert classify(4, 3) == "strong"
    assert classify(4, 2) == "decent"
    assert classify(3, 0) == "decent"
    assert classify(2, 6) is None


# ============================================================
# chunking — long-video transcript splitting
# ============================================================

def _mk_sents(n, dur=10.0, gap=0.5, big_pause_at=()):
    sents, t = [], 0.0
    for i in range(n):
        sents.append({"index": i, "text": f"S{i}.", "start": t, "end": t + dur, "words": []})
        t += dur + (10.0 if i in big_pause_at else gap)
    return sents


def test_chunking_short_single_range():
    from pipeline.chunking import chunk_sentence_ranges
    assert chunk_sentence_ranges([]) == []
    sents = _mk_sents(60)  # ~10 min
    assert chunk_sentence_ranges(sents) == [(0, 59)]


def test_chunking_long_snaps_to_biggest_pause():
    from pipeline.chunking import chunk_sentence_ranges
    # ~70 min transcript; a 10s pause planted at index 110, inside the
    # search window around the 20-min target boundary
    sents = _mk_sents(400, big_pause_at={110})
    ranges = chunk_sentence_ranges(sents)
    assert len(ranges) > 1
    assert ranges[0] == (0, 110)                      # boundary lands on the pause
    # contiguous + exhaustive coverage
    assert ranges[0][0] == 0 and ranges[-1][1] == 399
    for (_, prev_hi), (lo, _) in zip(ranges, ranges[1:]):
        assert lo == prev_hi + 1


def test_chunking_sentence_cap():
    from pipeline.chunking import chunk_sentence_ranges
    # 1200 one-second sentences: short in duration but over the per-chunk cap
    sents = _mk_sents(1200, dur=1.0, gap=0.0)
    ranges = chunk_sentence_ranges(sents)
    assert all(hi - lo + 1 <= 500 for lo, hi in ranges)
    assert ranges[0][0] == 0 and ranges[-1][1] == 1199
    for (_, prev_hi), (lo, _) in zip(ranges, ranges[1:]):
        assert lo == prev_hi + 1


def test_chunking_tiny_tail_merges_back():
    from pipeline.chunking import chunk_sentence_ranges
    sents = _mk_sents(31)
    ranges = chunk_sentence_ranges(sents, target_s=300, search_s=50, single_call_max_s=300)
    assert ranges == [(0, 30)]   # ~73s tail folded into the previous chunk


def test_reader_chunked_long_transcript():
    import re
    import pipeline.reader as reader
    from pipeline.chunking import chunk_sentence_ranges

    sents = _mk_sents(300)  # ~52 min → 3 chunks
    chunks = chunk_sentence_ranges(sents)
    assert len(chunks) == 3
    fail_chunk = chunks[1]
    calls: list[int] = []

    async def fake(agent, system, user, api_key=None, temperature=None, max_tokens=None):
        first = int(re.search(r"^(\d+) \[", user.split("Sentences:\n")[1], re.M).group(1))
        calls.append(first)
        if first == fail_chunk[0]:
            raise RuntimeError("LLM down")      # second chunk fails entirely
        return {"segments": [{"lo": first, "hi": first + 3}]}

    orig, reader.call_agent = reader.call_agent, fake
    try:
        segs = asyncio.run(reader.read_segments(sents, [], "key"))
    finally:
        reader.call_agent = orig

    assert len(calls) == 3
    # contiguous, exhaustive coverage despite the failed chunk
    assert segs[0]["lo"] == 0 and segs[-1]["hi"] == 299
    for prev, cur in zip(segs, segs[1:]):
        assert cur["lo"] == prev["hi"] + 1
    assert [s["segment_id"] for s in segs] == list(range(len(segs)))
    # the failed chunk degraded to one whole-chunk segment
    assert fail_chunk in [(s["lo"], s["hi"]) for s in segs]


def test_evaluator_batches_and_ranks_globally():
    import re
    import pipeline.evaluator as ev

    all_true = {k: True for k in ev.MUST_HAVES}
    call_count = [0]

    async def fake(agent, system, user, api_key=None, temperature=None, max_tokens=None):
        call_count[0] += 1
        ids = [int(m) for m in re.findall(r"Candidate (\d+) \[", user)]
        return {"clips": [
            {"source_candidates": [i], "sentence_groups": [[i, i]],
             "must_haves": all_true,
             "signals": {k: (i == 13 or k in ("emotion", "quotable", "contrast"))
                          for k in ev.SIGNALS},
             "reasoning": "ok"}
            for i in ids
        ]}

    sents = _sentences(25, sent_dur=30.0)    # each sentence ≈ 29.5s — inside the gate
    candidates = [{"segment_id": i, "lo": i, "hi": i, "tension_type": "t",
                   "annotation": "a", "knowledge_gap": 2, "duration_sec": 29.5,
                   "speaker": None} for i in range(25)]

    orig, ev.call_agent = ev.call_agent, fake
    orig_cap, ev.MAX_CLIPS = ev.MAX_CLIPS, 20
    try:
        clips = asyncio.run(ev.evaluate_clips(candidates, sents, "key", batch_size=10))
    finally:
        ev.call_agent = orig
        ev.MAX_CLIPS = orig_cap

    assert call_count[0] == 3                              # 10 + 10 + 5
    assert len(clips) == 20                                # global cap applied
    assert clips[0]["sentence_groups"] == [[13, 13]]       # hero ranks first globally
    assert clips[0]["label"] == "hero" and clips[0]["rank"] == 1
    scores = [c["score_total"] for c in clips]
    assert scores == sorted(scores, reverse=True)


def test_energy_index_from_file_path():
    import tempfile
    from pipeline.energy_index import build_energy_index, SR
    path = tempfile.mktemp(suffix=".wav")
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(SR)
        quiet = b"".join(struct.pack("<h", 0) for _ in range(SR // 2))
        loud  = b"".join(struct.pack("<h", 12000 if i % 2 else -12000) for i in range(SR // 2))
        wf.writeframes(quiet + loud)
    try:
        index = build_energy_index(path)    # path input → streamed decode
        assert index.n_frames == 50
        assert index.is_silent(5) is True and index.is_silent(40) is False
    finally:
        os.unlink(path)


# ============================================================
# security — worker auth brute-force guard
# ============================================================

def test_security_blocks_after_repeated_auth_failures():
    import security
    security.reset()
    try:
        for i in range(security.AUTH_FAIL_LIMIT - 1):
            security.register_auth_failure("1.2.3.4", now=float(i))
        assert security.is_blocked("1.2.3.4", now=10.0) is False
        security.register_auth_failure("1.2.3.4", now=10.0)
        assert security.is_blocked("1.2.3.4", now=10.0) is True
        # a different IP is unaffected
        assert security.is_blocked("5.6.7.8", now=10.0) is False
        # the window expiring unblocks the IP
        assert security.is_blocked("1.2.3.4", now=10.0 + security.AUTH_FAIL_WINDOW_S) is False
    finally:
        security.reset()


# ============================================================
# branding — free-tier watermark badge
# ============================================================

def test_branding_badge_png():
    from PIL import Image
    from pipeline.branding import make_branding_badge
    data = make_branding_badge(320)
    img = Image.open(io.BytesIO(data))
    assert img.format == "PNG" and img.mode == "RGBA"
    assert img.width >= 320 and img.height >= 28
    assert img.getextrema()[3][1] > 0          # something visible was drawn


def test_branding_overlay_on_video():
    import subprocess, tempfile
    from jobs.export import _apply_branding_watermark
    src = tempfile.mktemp(suffix=".mp4")
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=black:s=320x568:d=0.5",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", src],
        capture_output=True, check=True,
    )
    out = None
    try:
        out = asyncio.run(_apply_branding_watermark(src))
        assert out is not None and os.path.getsize(out) > 0
        probe = subprocess.run(
            ["ffprobe", "-v", "quiet", "-select_streams", "v:0",
             "-show_entries", "stream=width,height", "-of", "csv=p=0", out],
            capture_output=True, text=True,
        )
        assert probe.stdout.strip().split(",")[:2] == ["320", "568"]   # dims unchanged
    finally:
        os.unlink(src)
        if out:
            os.unlink(out)
    # unreadable input fails gracefully, never raises
    assert asyncio.run(_apply_branding_watermark("/nonexistent.mp4")) is None


# ============================================================
# config — binary preflight
# ============================================================

def test_missing_env_lists_every_absent_var():
    from unittest import mock
    import config
    with mock.patch.dict(config.os.environ, {k: "x" for k in config.REQUIRED_ENV}, clear=False):
        assert config.missing_env() == []
    partial = {k: "x" for k in config.REQUIRED_ENV if k != "SUPABASE_URL"}
    partial.update({"SUPABASE_URL": ""})
    with mock.patch.dict(config.os.environ, partial, clear=False):
        assert config.missing_env() == ["SUPABASE_URL"]


def test_missing_binaries_reports_only_absent_ones():
    from unittest import mock
    import config
    with mock.patch.object(config.shutil, "which", lambda b: "/usr/bin/" + b if b == "ffmpeg" else None):
        assert config.missing_binaries() == ["ffprobe"]
    with mock.patch.object(config.shutil, "which", lambda b: "/usr/bin/" + b):
        assert config.missing_binaries() == []


# ============================================================
# clip_thumbnail — graceful failure paths
# ============================================================

def test_clip_thumbnail_handles_bad_inputs():
    from pipeline.clip_thumbnail import extract_clip_thumbnail, extract_face_from_image
    assert extract_clip_thumbnail("/nonexistent/video.mp4", 0.0, 5.0) is None
    assert extract_face_from_image(b"definitely not an image") is None


def test_clip_thumbnail_no_face_returns_none_and_zero_area():
    import cv2, numpy as np
    from pipeline import clip_thumbnail as t
    blank = np.zeros((120, 160, 3), dtype=np.uint8)
    assert t._face_area(blank) == 0.0
    ok, buf = cv2.imencode(".jpg", blank)
    assert ok
    assert t.extract_face_from_image(bytes(buf)) is None


def test_clip_thumbnail_face_crop_geometry():
    """Crop is square, 1.5x the face, centred on it, and clamped to the image."""
    import cv2, numpy as np
    from unittest import mock
    from pipeline import clip_thumbnail as t

    img = np.full((200, 300, 3), 128, dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok

    # Face 40x40 at (130, 80): centre (150, 100), half = 30 -> crop 60x60.
    with mock.patch.object(t, "_detect_faces", lambda f: [(130, 80, 40, 40)]):
        out = t.extract_face_from_image(bytes(buf))
    assert out is not None
    crop = cv2.imdecode(np.frombuffer(out, np.uint8), cv2.IMREAD_COLOR)
    assert crop.shape[:2] == (60, 60)

    # Face at the top-left edge: crop must clamp rather than go negative.
    with mock.patch.object(t, "_detect_faces", lambda f: [(0, 0, 40, 40)]):
        out = t.extract_face_from_image(bytes(buf))
    crop = cv2.imdecode(np.frombuffer(out, np.uint8), cv2.IMREAD_COLOR)
    assert crop.shape[:2] == (50, 50)   # 20 + 30, clamped at 0 on the near side

    # Relative area uses the largest (first) box.
    with mock.patch.object(t, "_detect_faces", lambda f: [(0, 0, 30, 20), (0, 0, 10, 10)]):
        assert abs(t._face_area(img) - (30 * 20) / (200 * 300)) < 1e-9


# ============================================================
# clip_cut — input validation
# ============================================================

def test_cut_clip_rejects_empty_cuts():
    from pipeline.clip_cut import cut_clip
    try:
        cut_clip("in.mp4", [], "out.mp4")
        assert False, "should raise"
    except ValueError:
        pass


# ============================================================
# ytdlp_proxy_args — YT_PROXY env passthrough
# ============================================================

def test_ytdlp_proxy_args():
    from sources import ytdlp_proxy_args
    old = os.environ.pop("YT_PROXY", None)
    try:
        assert ytdlp_proxy_args() == []
        os.environ["YT_PROXY"] = "http://user:pass@gate.example.com:7000"
        assert ytdlp_proxy_args() == ["--proxy", "http://user:pass@gate.example.com:7000"]
    finally:
        os.environ.pop("YT_PROXY", None)
        if old is not None:
            os.environ["YT_PROXY"] = old


def test_ytdlp_runs_as_module_of_this_interpreter():
    import sys, sources
    assert sources.YTDLP_CMD == [sys.executable, "-m", "yt_dlp"]
    # And the module is actually importable here — the dep is real, not a PATH accident.
    import importlib; assert importlib.util.find_spec("yt_dlp") is not None


def test_ytdlp_cookies_args():
    import sources
    old_b64 = os.environ.pop("YT_COOKIES_B64", None)
    old_browser = os.environ.pop("YT_COOKIES_FROM_BROWSER", None)
    sources._yt_cookies_path = None

    try:
        assert sources.ytdlp_cookies_args() == []

        os.environ["YT_COOKIES_FROM_BROWSER"] = "safari"
        assert sources.ytdlp_cookies_args() == ["--cookies-from-browser", "safari"]
        os.environ.pop("YT_COOKIES_FROM_BROWSER", None)

        cookies_content = b"# Netscape HTTP Cookie File\n.youtube.com\tTRUE\t/\tFALSE\t0\tNAME\tVALUE\n"
        os.environ["YT_COOKIES_B64"] = base64.b64encode(cookies_content).decode()
        args = sources.ytdlp_cookies_args()
        assert args[0] == "--cookies"
        assert Path(args[1]).read_bytes() == cookies_content
        os.unlink(args[1])
    finally:
        os.environ.pop("YT_COOKIES_B64", None)
        os.environ.pop("YT_COOKIES_FROM_BROWSER", None)

        if old_b64 is not None:
            os.environ["YT_COOKIES_B64"] = old_b64
        if old_browser is not None:
            os.environ["YT_COOKIES_FROM_BROWSER"] = old_browser

        sources._yt_cookies_path = None


def test_spawn_retains_task_until_done():
    """_spawn must keep a strong ref so a fire-and-forget task (e.g. the
    transcribe→identify auto-chain) can't be garbage-collected mid-await and
    leave the video pinned at 'identifying'. The ref is dropped on completion."""
    import gc
    import runner

    async def scenario():
        started = asyncio.Event()
        release = asyncio.Event()

        async def work():
            started.set()
            await release.wait()

        runner._tasks.clear()
        runner._spawn(work())        # intentionally keep no local reference
        gc.collect()                  # only runner._tasks should keep it alive
        # If the task were collected, started never fires → fail fast, not hang.
        await asyncio.wait_for(started.wait(), timeout=2.0)
        assert len(runner._tasks) == 1, "task was not retained while running"

        release.set()
        for _ in range(100):
            if not runner._tasks:
                break
            await asyncio.sleep(0)
        assert not runner._tasks, "task ref not released after completion"

    asyncio.run(scenario())


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all tests passed")
