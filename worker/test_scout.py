"""
Offline tests for the Moment Scout — LLM mocked at the call_agent seam.
Run: python test_scout.py
"""

from __future__ import annotations
import asyncio

import pipeline.scout as scout
from pipeline.scout import scout_moments, _validate_moments, _tone_hint


def _sentences(n: int, sent_dur: float = 4.0) -> list[dict]:
    out = []
    for i in range(n):
        start = i * (sent_dur + 0.5)
        out.append({
            "index": i, "text": f"Sentence {i}.",
            "start": start, "end": start + sent_dur,
            "words": [{"word": "w"}] * 5,
        })
    return out


def _features(n: int) -> list[dict]:
    return [{
        "speaker": "S0", "wps": 1.0, "lead_pause": 0.0, "trail_pause": 0.5,
        "energy_pct": 50, "laughter_after": False, "rapid_exchange": False,
    } for _ in range(n)]


def test_validate_moments():
    raw = {"moments": [
        {"archetype": "humor", "score": 150, "payoff": 5, "lo": 3, "hi": 8, "why": "lol"},
        {"archetype": "insight", "score": -5, "payoff": 0, "lo": 0, "hi": 2},
        {"archetype": "bogus", "score": 90, "payoff": 1, "lo": 0, "hi": 2},      # bad type
        {"archetype": "story", "score": 50, "lo": 9, "hi": 4},                   # lo > hi
        {"archetype": "story", "score": "x", "lo": 0, "hi": 1},                  # bad score
        {"archetype": "emotion", "score": 70, "payoff": 99, "lo": 0, "hi": 9},   # payoff clamped
    ]}
    out = _validate_moments(raw, 0, 9)
    assert len(out) == 3
    assert out[0]["score"] == 100 and out[1]["score"] == 0
    assert out[2]["payoff"] == 9
    # Span clamped to chunk bounds
    out = _validate_moments({"moments": [
        {"archetype": "humor", "score": 50, "payoff": 5, "lo": 0, "hi": 500},
    ]}, 3, 20)
    assert out[0]["lo"] == 3 and out[0]["hi"] == 20


def test_overlong_span_recentred_on_payoff():
    raw = {"moments": [
        {"archetype": "story", "score": 80, "payoff": 100, "lo": 0, "hi": 199},
    ]}
    [m] = _validate_moments(raw, 0, 199)
    assert m["hi"] - m["lo"] + 1 <= scout.MAX_SPAN_SENTENCES + 1
    assert m["lo"] <= 100 <= m["hi"]


def test_scout_merges_dedupes_and_ranks():
    sents = _sentences(20)
    calls = []

    async def fake_call_agent(agent, system, user, **kw):
        calls.append((agent, system, user))
        return {"moments": [
            {"archetype": "humor", "score": 90, "payoff": 5, "lo": 3, "hi": 7, "why": "a"},
            {"archetype": "humor", "score": 70, "payoff": 5, "lo": 3, "hi": 7, "why": "dup"},
            {"archetype": "insight", "score": 40, "payoff": 10, "lo": 9, "hi": 12, "why": "b"},
        ]}

    orig = scout.call_agent
    scout.call_agent = fake_call_agent
    try:
        out = asyncio.run(scout_moments(
            sents, _features(20), {},
            context={"title": "My Pod", "target_tones": ["entertaining"], "niche": "fitness"},
        ))
    finally:
        scout.call_agent = orig

    assert [m["archetype"] for m in out] == ["humor", "insight"]  # dup dropped, sorted
    assert out[0]["candidate_id"] == 0
    assert out[0]["start"] == sents[3]["start"] and out[0]["end"] == sents[7]["end"]
    # Context made it into the prompts
    agent, system, user = calls[0]
    assert agent == "scout"
    assert "fitness" in system and "entertaining" in system
    assert "Video title: My Pod" in user
    assert "0 (4.0s) [S0]: Sentence 0." in user


def test_scout_chunk_failure_is_nonfatal():
    async def boom(*a, **kw):
        raise RuntimeError("model down")

    orig = scout.call_agent
    scout.call_agent = boom
    try:
        out = asyncio.run(scout_moments(_sentences(5), _features(5), {}))
    finally:
        scout.call_agent = orig
    assert out == []
    assert asyncio.run(scout_moments([], [], {})) == []


def test_tone_hint_empty_without_context():
    assert _tone_hint({}) == ""
    assert "fitness" in _tone_hint({"niche": "fitness"})


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all tests passed")
