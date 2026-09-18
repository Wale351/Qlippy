"""Local speaker diarization using pyannote Community-1."""

from __future__ import annotations

import os
from functools import lru_cache


@lru_cache(maxsize=1)
def _get_pipeline():
    from pyannote.audio import Pipeline

    token = os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is not configured")

    return Pipeline.from_pretrained(
        "pyannote/speaker-diarization-community-1",
        token=token,
    )


def diarize_audio(audio_path: str) -> list[tuple[float, float, str]]:
    """Return [(start, end, speaker_id), ...] for an audio file."""
    pipeline = _get_pipeline()
    result = pipeline(audio_path)
    diar = result.speaker_diarization

    return [
        (float(turn.start), float(turn.end), str(speaker))
        for turn, _, speaker in diar.itertracks(yield_label=True)
    ]


def attach_speakers_to_segments(
    segments: list[dict],
    diar_timeline: list[tuple[float, float, str]],
) -> list[dict]:
    """Assign each Whisper segment to the speaker with the most overlap."""
    for segment in segments:
        start = float(segment.get("start", 0))
        end = float(segment.get("end", start))

        overlap_by_speaker: dict[str, float] = {}

        for diar_start, diar_end, speaker in diar_timeline:
            overlap = max(0.0, min(end, diar_end) - max(start, diar_start))
            if overlap > 0:
                overlap_by_speaker[speaker] = (
                    overlap_by_speaker.get(speaker, 0.0) + overlap
                )

        if overlap_by_speaker:
            segment["speaker"] = max(
                overlap_by_speaker,
                key=overlap_by_speaker.get,
            )

    return segments
