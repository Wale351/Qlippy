"""
ClipSpec v1 — the canonical, serializable description of a clip + its styling.

Source of truth per the Clip/Export ADRs: clip = write it, export = render it,
preview = composite it. Assembled here from data currently scattered across the
clips row, the cached clip_edit, the resolved preset config, and the export job
payload — normalized into one object with one coordinate convention.

Coordinate convention (R5): every position is normalized 0-1 against the OUTPUT
frame. Legacy 0-100 fields (vertical_percent, x_pct/y_pct, size_pct) are divided
by 100 at this boundary; watermark corner presets resolve to anchors. The
source->output transform is two-hop and lives in the cached edit + analysis_ref,
not in this object: source ->(cuts)-> stitched ->(crop timeline)-> output.
"""
from __future__ import annotations

SPEC_VERSION = 1

# aspect_ratio -> output pixel dims (the portrait render is the 9:16 default).
_ASPECT_DIMS = {
    "9:16": (1080, 1920),
    "16:9": (1920, 1080),
    "1:1":  (1080, 1080),
    "4:5":  (1080, 1350),
    "3:4":  (1080, 1440),
    "4:3":  (1440, 1080),
}


def _norm(v, default: float) -> float:
    """0-100 (legacy) -> 0-1, rounded to 4 dp."""
    try:
        return round(float(v) / 100, 4)
    except (TypeError, ValueError):
        return round(default / 100, 4)


def _output(aspect: str | None) -> dict:
    w, h = _ASPECT_DIMS.get(aspect or "9:16", _ASPECT_DIMS["9:16"])
    return {"w": w, "h": h, "aspect": aspect or "9:16"}


def _captions_layer(preset_config: dict) -> dict:
    layout = preset_config.get("layout") or {}
    position = {"anchor": layout.get("vertical_position", "bottom")}
    if layout.get("vertical_percent") is not None:
        position["y"] = _norm(layout["vertical_percent"], 80)
    return {
        "type":         "captions",
        "component_id":  preset_config.get("hyperframes_component") or "pil",
        "words_ref":     "transcript:raw_words",   # timing from stage-1 data (R6)
        "position":      position,
        "props":         preset_config.get("typography") or {},
    }


def _text_sticker_layer(cfg: dict) -> dict:
    return {
        "type":         "text_sticker",
        "component_id":  "sticker.basic.v1",
        "text":          (cfg.get("text") or "")[:30],
        "position":      {"x": _norm(cfg.get("x_pct"), 50), "y": _norm(cfg.get("y_pct"), 50)},
        "props":         {"size": _norm(cfg.get("size_pct"), 8), "scale": float(cfg.get("scale", 1.0))},
    }


def build_clip_spec(
    *,
    video_id: str,
    clip: dict,
    edit: dict | None,
    aspect_ratio: str | None,
    preset_config: dict | None,
    text_sticker_config: dict | None = None,
    branding_watermark: bool = False,
) -> dict:
    """Assemble the canonical ClipSpec for one export. Pure; no I/O."""
    edit          = edit or {}
    preset_config = preset_config or {}

    layers: list[dict] = []
    if preset_config:
        layers.append(_captions_layer(preset_config))
    if text_sticker_config and (text_sticker_config.get("text") or "").strip():
        layers.append(_text_sticker_layer(text_sticker_config))
    if branding_watermark:
        layers.append({
            "type": "watermark", "component_id": "brand.badge.v1",
            "anchor": "bottom-center", "props": {"forced": True},
        })

    style: dict = {"render": {"engine": "hyperframes"}, "layers": layers}

    return {
        "version":      SPEC_VERSION,
        "source":       {"id": video_id},
        "cuts":         clip.get("cuts") or [],   # play order preserved (decision #1)
        "output":       _output(aspect_ratio),
        "analysis_ref": edit.get("analysis_r2_key"),
        "edit":         {"reframe": {"mode": "auto"}, "cache_ref": edit.get("r2_key")},
        "style":        style,
    }
