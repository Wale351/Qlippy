"""
HyperFrames subtitle rendering.

Loads our custom-authored caption components from worker/captions/,
injects WhisperX word timings + a CFG customization object, and renders
to MP4 via Puppeteer.

Every component follows the same injection contract:
  var WORDS      = [...];   word timings   (patched)
  var RAW_GROUPS = [...];   group indices  (patched)
  var CFG        = {...};   customization  (patched — colors, fontScale, yPct…)
plus data-width / data-height / data-duration attributes on the root.

Falls back to the PIL renderer if rendering fails.
"""

from __future__ import annotations
import json
import logging
import re
import subprocess
import tempfile
from pathlib import Path

log = logging.getLogger(__name__)

# Our custom-authored caption components, bundled with the worker.
CAPTIONS_DIR = Path(__file__).resolve().parent.parent / "captions"

HF_COMPONENTS: dict[str, str] = {
    "pill":    "Highlight pill sweeps behind the active word (Montserrat)",
    "impact":  "Hormozi-style ALL-CAPS pop, thick stroke, accent active word (Anton)",
    "beast":   "Chunky comic slam-in with elastic word bounce (Luckiest Guy)",
    "karaoke": "Words fill to the accent color as spoken and stay lit (Montserrat)",
}

# Style names stored by older exports/presets → current component names.
LEGACY_COMPONENTS = {"caption-highlight": "pill"}

PAUSE_THRESHOLD = 0.4
SPLIT_PUNCT = set(".,?!:;")


# ── Word grouping ──────────────────────────────────────────────────────────────

def _build_word_list(raw_words: list[dict], cuts: list[dict]) -> list[dict]:
    """Convert WhisperX words + cuts to clip-relative [{text, start, end}]."""
    result: list[dict] = []
    cumulative = 0.0
    for cut in cuts:
        s, e = float(cut["start"]), float(cut["end"])
        offset = s - cumulative
        for w in raw_words:
            ws, we = w.get("start"), w.get("end")
            if ws is None or we is None:
                continue
            if not (s - 0.05 <= ws <= e + 0.05):
                continue
            txt = (w.get("word") or w.get("text") or "").strip()
            if txt:
                result.append({
                    "text":  txt,
                    "start": round(ws - offset, 3),
                    "end":   round(we - offset, 3),
                })
        cumulative += e - s
    return result


def _build_raw_groups(words: list[dict], max_words: int) -> list[list[int]]:
    """Build [[start_idx, end_idx], ...] groups from a flat word list."""
    groups: list[list[int]] = []
    group_start = 0
    count = 0

    for i, w in enumerate(words):
        # Split on silence gap
        if i > 0 and group_start < i:
            if w["start"] - words[i - 1]["end"] > PAUSE_THRESHOLD:
                groups.append([group_start, i - 1])
                group_start = i
                count = 0

        count += 1

        if count >= max_words:
            groups.append([group_start, i])
            group_start = i + 1
            count = 0
            continue

        # Split on sentence-ending punctuation
        if w["text"] and w["text"][-1] in SPLIT_PUNCT:
            groups.append([group_start, i])
            group_start = i + 1
            count = 0

    if group_start < len(words):
        groups.append([group_start, len(words) - 1])

    return groups


# ── Component loading ──────────────────────────────────────────────────────────

def _ensure_component(component_name: str) -> Path:
    """Resolve a caption style name to its bundled HTML file."""
    name = LEGACY_COMPONENTS.get(component_name, component_name)
    path = CAPTIONS_DIR / f"{name}.html"
    if not path.exists():
        raise RuntimeError(f"Unknown caption style: {component_name}")
    return path


# ── HTML patching ──────────────────────────────────────────────────────────────

def _build_cfg(preset_config: dict) -> dict:
    """Map PresetConfig fields to the CFG object every component reads."""
    hl  = preset_config.get("highlight") or {}
    lay = preset_config.get("layout")    or {}
    typ = preset_config.get("typography") or {}

    v_pct = lay.get("vertical_percent")
    if v_pct is None:
        v_pos_map = {"top": 15, "middle": 50, "bottom": 80}
        v_pct = v_pos_map.get(lay.get("vertical_position", "bottom"), 80)

    return {
        "accent":    hl.get("fill_color", "#ff1745"),
        "text":      hl.get("font_color", "#ffffff"),
        "stroke":    preset_config.get("color", {}).get("stroke_color", "#000000"),
        "fontScale": float(typ.get("font_size", 100)) / 100.0,
        "yPct":      float(v_pct),
        "radius":    int(hl.get("border_radius", 10)),
        "glow":      int(hl.get("shadow_blur", 30)),
        "allCaps":   bool(typ.get("all_caps", False)),
    }


def _patch_component(
    html: str,
    words: list[dict],
    raw_groups: list[list[int]],
    duration: float,
    canvas_w: int,
    canvas_h: int,
    preset_config: dict,
) -> str:
    """Inject clip data + customization into a bundled component template."""

    # Canvas dimensions + duration — components size themselves from these attrs.
    html = re.sub(
        r'content="width=\d+,\s*height=\d+"',
        f'content="width={canvas_w}, height={canvas_h}"',
        html,
    )
    html = re.sub(r'data-width="\d+"',  f'data-width="{canvas_w}"',  html)
    html = re.sub(r'data-height="\d+"', f'data-height="{canvas_h}"', html)
    html = re.sub(
        r'data-duration="\d+(?:\.\d+)?"',
        f'data-duration="{duration:.2f}"',
        html,
    )

    # Word timings, groups, and the customization object
    words_js = json.dumps(words, separators=(",", ":"))
    html = re.sub(r'var WORDS\s*=\s*\[[\s\S]*?\];', f'var WORDS = {words_js};', html)

    groups_js = json.dumps(raw_groups, separators=(",", ":"))
    html = re.sub(r'var RAW_GROUPS\s*=\s*\[[\s\S]*?\];', f'var RAW_GROUPS = {groups_js};', html)

    cfg_js = json.dumps(_build_cfg(preset_config), separators=(",", ":"))
    html = re.sub(r'var CFG\s*=\s*\{[\s\S]*?\};', f'var CFG = {cfg_js};', html)

    return html


# ── Render ─────────────────────────────────────────────────────────────────────

def burn_subtitles_hf(
    input_path: str,
    output_path: str,
    raw_words: list[dict],
    cuts: list[dict],
    preset_config: dict,
    canvas_w: int,
    canvas_h: int,
    fps: float = 30.0,
) -> None:
    """
    Render animated captions with HyperFrames (Puppeteer-based).

    component_name is read from preset_config['hyperframes_component'].
    Raises RuntimeError on render failure — caller should fall back to PIL.
    """
    component_name = preset_config.get("hyperframes_component") or "pill"
    two_lines = preset_config.get("layout", {}).get("two_lines", False)
    max_words = 8 if two_lines else 5

    words = _build_word_list(raw_words, cuts)
    if not words:
        log.warning("burn_subtitles_hf: no words — copying video unchanged")
        subprocess.run(["ffmpeg", "-y", "-i", input_path, "-c", "copy", output_path],
                       check=True, capture_output=True)
        return

    raw_groups = _build_raw_groups(words, max_words)
    duration = sum(float(c["end"]) - float(c["start"]) for c in cuts)

    log.info(
        "HyperFrames render: %s | %d words | %d groups | %.1fs | %dx%d",
        component_name, len(words), len(raw_groups), duration, canvas_w, canvas_h,
    )

    component_html = _ensure_component(component_name).read_text()

    patched = _patch_component(
        component_html, words, raw_groups,
        duration=duration,
        canvas_w=canvas_w, canvas_h=canvas_h,
        preset_config=preset_config,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        proj = Path(tmpdir) / "proj"
        (proj / "compositions" / "components").mkdir(parents=True)
        (proj / "assets").mkdir()

        # Minimal project config
        (proj / "hyperframes.json").write_text(json.dumps({
            "$schema": "https://hyperframes.heygen.com/schema/hyperframes.json",
            "registry": "https://raw.githubusercontent.com/heygen-com/hyperframes/main/registry",
            "paths": {
                "blocks": "compositions",
                "components": "compositions/components",
                "assets": "assets",
            },
        }))
        (proj / "meta.json").write_text(json.dumps({"id": "clipforge", "name": "clipforge"}))
        (proj / "index.html").write_text("<!doctype html><html><head></head><body></body></html>")

        (proj / "compositions" / "components" / "caption.html").write_text(patched)

        # Step 1: render captions-only as RGBA PNG sequence (transparent background)
        png_dir = proj / "frames"
        result = subprocess.run(
            [
                "npx", "--yes", "hyperframes", "render", str(proj),
                "-c", "compositions/components/caption.html",
                "-o", str(png_dir),
                "--format", "png-sequence",
                "--fps", str(round(fps)),
                "--quiet",
            ],
            capture_output=True,
            timeout=900,
        )

        if result.returncode != 0:
            err = result.stderr.decode(errors="replace")
            raise RuntimeError(
                f"HyperFrames render failed (exit {result.returncode}):\n{err[-4000:]}"
            )

        pngs = sorted(png_dir.glob("*.png"))
        if not pngs:
            raise RuntimeError("HyperFrames produced no PNG frames")

        stem = pngs[0].name.rsplit(".", 1)[0]
        prefix = stem.rstrip("0123456789")
        num_part = stem[len(prefix):]
        frame_pattern = str(png_dir / f"{prefix}%0{len(num_part)}d.png")
        log.info("HyperFrames PNG sequence: %d frames → %s", len(pngs), frame_pattern)

        # Step 2: composite RGBA PNG frames directly onto the source video
        composite = subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", input_path,
                "-framerate", str(round(fps)), "-start_number", num_part,
                "-i", frame_pattern,
                "-filter_complex",
                (
                    f"[0:v]scale={canvas_w}:{canvas_h}:flags=lanczos[base];"
                    f"[1:v]scale={canvas_w}:{canvas_h}:flags=lanczos,format=rgba[cap];"
                    "[base][cap]overlay=0:0:shortest=1,setsar=1[out]"
                ),
                "-map", "[out]", "-map", "0:a?",
                "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
                "-c:a", "copy",
                output_path,
            ],
            capture_output=True,
        )

    if composite.returncode != 0:
        err = composite.stderr.decode(errors="replace")
        raise RuntimeError(
            f"ffmpeg overlay composite failed (exit {composite.returncode}):\n{err[-2000:]}"
        )

    log.info("HyperFrames render + composite complete → %s", output_path)
