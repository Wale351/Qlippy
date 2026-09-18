"""
Offline tests for the Clip Crafter — LLM mocked at the call_agent seam.
Run: python test_crafter.py
"""

from __future__ import annotations
import asyncio

import pipeline.crafter as crafter
from pipeline.crafter import craft_clips, _resolve_groups


def _sentences(n: int, sent_dur: float = 5.0) -> list[dict]:
    out = []
    for i in range(n):
        start = i * (sent_dur + 0.5)
        out.append({
            "index": i, "text": f"S{i}.",
            "start": start, "end": start + sent_dur,
            "words": [{"word": "w"}] * 5,
        })
    return out


def _features(n: int, speaker: str = "A") -> list[dict]:
    return [{
        "speaker": speaker, "wps": 1.0, "lead_pause": 0.0, "trail_pause": 0.5,
        "energy_pct": 50, "laughter_after": False, "rapid_exchange": False,
    } for _ in range(n)]


def _candidate(lo: int, hi: int, payoff: int | None = None, archetype: str = "humor") -> dict:
    return {
        "candidate_id": 0, "archetype": archetype, "score": 80,
        "payoff": payoff if payoff is not None else hi, "lo": lo, "hi": hi,
        "why": "test", "start": 0.0, "end": 0.0,
    }


ALL_TRUE_MH  = {k: True for k in crafter.MUST_HAVES}
ALL_TRUE_SIG = {k: True for k in crafter.SIGNALS}


def _run(candidates, sentences, features, response, min_dur=None, max_dur=None):
    async def fake_call_agent(agent, system, user, **kw):
        assert agent == "crafter"
        fake_call_agent.last = (system, user)
        return response

    orig = crafter.call_agent
    crafter.call_agent = fake_call_agent
    try:
        clips = asyncio.run(craft_clips(
            candidates, sentences, features, {},
            min_duration_s=min_dur, max_duration_s=max_dur,
        ))
        return clips, fake_call_agent
    finally:
        crafter.call_agent = orig


def test_resolve_groups_validation():
    sents = _sentences(10)
    # Chronological order enforced when transplant is off
    norm, cuts, text = _resolve_groups([[6, 7], [1, 2]], sents, transplant=False)
    assert norm == [[1, 2], [6, 7]]
    assert cuts[0]["start"] < cuts[1]["start"]
    # Play order preserved when transplant is on
    norm, cuts, text = _resolve_groups([[6, 7], [1, 2]], sents, transplant=True)
    assert norm == [[6, 7], [1, 2]]
    assert cuts[0]["start"] > cuts[1]["start"]
    assert text == "S6. S7. S1. S2."
    # Overlapping groups are merged into one continuous range
    norm, cuts, text = _resolve_groups([[1, 4], [3, 6]], sents, transplant=False)
    assert norm == [[1, 6]]
    # Malformed
    assert _resolve_groups([[1]], sents, transplant=False) is None
    assert _resolve_groups([], sents, transplant=False) is None
    assert _resolve_groups([["a", "b"]], sents, transplant=False) is None
    # Too many groups
    too_many = [[i * 2, i * 2] for i in range(crafter.MAX_GROUPS + 1)]
    assert _resolve_groups(too_many, sents, transplant=False) is None
    # Out-of-range clamped
    norm, cuts, _ = _resolve_groups([[-3, 99]], sents, transplant=False)
    assert norm == [[0, 9]]


def test_craft_basic_clip():
    sents = _sentences(20)
    clips, fake = _run([_candidate(5, 9, payoff=8)], sents, _features(20), {
        "keep": True, "transplant": False,
        "sentence_groups": [[5, 9]],
        "must_haves": ALL_TRUE_MH, "signals": ALL_TRUE_SIG,
        "reasoning": "solid",
    })
    [clip] = clips
    assert clip["label"] == "strong" and clip["score_total"] == 11
    assert clip["cuts_raw"] == [{"start": sents[5]["start"], "end": sents[9]["end"]}]
    assert not clip["stitched"] and not clip["transplant"]
    assert clip["speaker"] == "A"
    assert clip["src_start"] == sents[5]["start"]
    # Prompt sanity: span marked, payoff named, archetype guidance present
    system, user = fake.last
    assert ">>> 5" in user and "Payoff sentence: 8" in user
    assert "punchline" in system  # humor guidance


def test_hook_transplant_play_order():
    sents = _sentences(20)
    clips, _ = _run([_candidate(2, 10)], sents, _features(20), {
        "keep": True, "transplant": True,
        "sentence_groups": [[9, 10], [2, 4]],   # payoff first, setup after
        "must_haves": ALL_TRUE_MH, "signals": ALL_TRUE_SIG,
        "reasoning": "payoff works cold",
    })
    [clip] = clips
    assert clip["transplant"] and clip["stitched"]
    assert clip["sentence_groups"] == [[9, 10], [2, 4]]
    assert clip["cuts_raw"][0]["start"] > clip["cuts_raw"][1]["start"]
    assert clip["text"] == "S9. S10. S2. S3. S4."
    assert clip["src_start"] == sents[2]["start"]   # earliest source second


def test_transplant_flag_cleared_when_order_is_chronological():
    sents = _sentences(20)
    clips, _ = _run([_candidate(2, 10)], sents, _features(20), {
        "keep": True, "transplant": True,           # claims transplant…
        "sentence_groups": [[2, 4], [9, 10]],       # …but order is chronological
        "must_haves": ALL_TRUE_MH, "signals": ALL_TRUE_SIG,
    })
    assert not clips[0]["transplant"]


def test_unflagged_out_of_order_gets_sorted():
    sents = _sentences(20)
    clips, _ = _run([_candidate(2, 10)], sents, _features(20), {
        "keep": True, "transplant": False,
        "sentence_groups": [[9, 10], [2, 4]],   # out of order without the flag
        "must_haves": ALL_TRUE_MH, "signals": ALL_TRUE_SIG,
    })
    assert clips[0]["sentence_groups"] == [[2, 4], [9, 10]]
    assert not clips[0]["transplant"]


def test_rejections_become_weak_clips_on_the_scout_span():
    """Nothing the Scout flagged disappears: every Crafter rejection falls back
    to a "weak" clip on the candidate's own [lo, hi] span, with the reason."""
    sents = _sentences(20)
    feats = _features(20)
    base = {
        "keep": True, "transplant": False, "sentence_groups": [[5, 9]],
        "must_haves": ALL_TRUE_MH, "signals": ALL_TRUE_SIG,
    }
    def weak_of(clips):
        assert len(clips) == 1, clips
        c = clips[0]
        assert c["label"] == "weak" and c["score_total"] == 0 and not c["transplant"]
        return c

    # keep: false — cold-start test; the Crafter's reasoning is carried through
    c = weak_of(_run([_candidate(5, 9)], sents, feats, {"keep": False, "reasoning": "needs the episode"})[0])
    assert c["sentence_groups"] == [[5, 9]] and "needs the episode" in c["reasoning"]
    # under 3 must-haves
    weak_mh = {**ALL_TRUE_MH, "hook": False, "cold_start": False, "payoff": False}
    c = weak_of(_run([_candidate(5, 9)], sents, feats, {**base, "must_haves": weak_mh})[0])
    assert "must-haves" in c["reasoning"]
    # Short cuts are expanded with nearby context instead of being rejected.
    c = _run([_candidate(5, 9)], sents, feats, {**base, "sentence_groups": [[5, 5]]})[0][0]
    assert c["label"] == "strong"
    assert c["duration_sec"] >= crafter.GATE_MIN_S
    assert c["sentence_groups"] != [[5, 5]]
    # Reversed boundaries from small local models are repaired automatically.
    c = _run([_candidate(5, 9)], sents, feats, {**base, "sentence_groups": [[9, 5]]})[0][0]
    assert c["label"] == "strong"
    assert c["sentence_groups"] == [[5, 9]]
    # LLM exception
    async def boom(*a, **kw):
        raise RuntimeError("down")
    orig = crafter.call_agent
    crafter.call_agent = boom
    try:
        c = weak_of(asyncio.run(craft_clips([_candidate(5, 9)], sents, feats, {})))
    finally:
        crafter.call_agent = orig
    assert "failed" in c["reasoning"]
    # A good candidate is untouched by all of this
    clips, _ = _run([_candidate(5, 9)], sents, feats, base)
    assert len(clips) == 1 and clips[0]["label"] != "weak"


def test_humor_can_reach_hero_without_learning():
    # The old 'learning' must-have is gone — 'payoff' covers laughs
    sents = _sentences(20)
    candidate = _candidate(5, 9, archetype="humor")
    candidate["score"] = 90
    clips, _ = _run([candidate], sents, _features(20), {
        "keep": True, "transplant": False, "sentence_groups": [[5, 9]],
        "must_haves": {"hook": True, "cold_start": True, "single_point": True,
                       "clean_ending": True, "payoff": True},
        "signals": {"emotion": True, "quotable": True, "contrast": False,
                    "energy": True, "curiosity_gap": True, "low_filler": False},
    })
    assert clips[0]["label"] == "hero"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all tests passed")
