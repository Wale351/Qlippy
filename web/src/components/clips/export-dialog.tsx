"use client";

import { useState, useEffect, useRef } from "react";
import {
  Clip, PresetConfig, DEFAULT_PRESET_CONFIG, DEFAULT_HIGHLIGHT_STYLE,
  TextStickerConfig, DEFAULT_TEXT_STICKER_CONFIG, TEXT_STICKER_MAX_CHARS,
  STICKER_COLOR_PRESETS,
  ImageStickerConfig, IMAGE_STICKER_MIN_WPCT, IMAGE_STICKER_MAX_WPCT,
} from "@/lib/types";
import { validateStickerPng } from "@/lib/image-sticker";
import { contrastRatio, MIN_CAPTION_CONTRAST } from "@/lib/contrast";
import { POSITION_MIN, POSITION_MAX, CENTER, STICKER_SCALE_MIN, STICKER_SCALE_MAX } from "@/lib/editor-bounds";
import { History, initHistory, pushHistory, undo, redo } from "@/lib/editor-history";
import { Loader2, AlertCircle, AlertTriangle, Scissors, Trash2, ChevronUp, ChevronDown, ChevronLeft, ChevronRight, Check, Bookmark, X } from "lucide-react";
import { CaptionOverlay, ColorPicker } from "@/components/templates/template-controls";
import { CAPTION_STYLES, applyCaptionStyle, getCaptionStyle, normalizeCaptionConfig } from "@/lib/caption-styles";
import { TextStickerOverlay, ImageStickerOverlay, extractWrappedLines } from "@/components/clips/text-sticker";
import { Modal } from "@/components/ui/modal";
import { Button } from "@/components/ui/button";

type SavedPreset = { id: string; name: string; config: PresetConfig; is_default?: boolean };

interface Props {
  clip:                     Clip;
  posterUrl?:               string | null;   // fallback frame when the clip has no extracted thumbnail (old clips)
  initialConfig?:           PresetConfig | null;
  initialSubtitlePresetId?: string | null;
  restyle?:                 boolean;   // clip already exported → reuses the cached edit, free
  onClose:                  () => void;
  onQueued:                 (exportId: string, label: string) => void;
}

const STYLE_W  = 150;  // width of a style-grid tile
const LAST_KEY = "cf_last_export";

// Legacy configs may carry a "transparent" pill fill (from the era when the
// pill could be toggled off) — coerce back to a solid fill so it's visible.
const withSolidPill = (c: PresetConfig): PresetConfig => {
  const h = c.highlight ?? DEFAULT_HIGHLIGHT_STYLE;
  return h.fill_color === "transparent"
    ? { ...c, highlight: { ...h, fill_color: DEFAULT_HIGHLIGHT_STYLE.fill_color, shadow_blur: DEFAULT_HIGHLIGHT_STYLE.shadow_blur } }
    : c;
};

// Every stored config (presets, last export, restyle) gets a valid style
// identity + the solid-pill coercion before it drives the dialog.
const normalize = (c: PresetConfig): PresetConfig => withSolidPill(normalizeCaptionConfig(c));

type AspectRatio = "9:16" | "1:1" | "3:4" | "4:3";
type FitMode = "fill" | "fit";

const ASPECT_RATIO_OPTIONS: AspectRatio[] = ["9:16", "1:1", "3:4", "4:3"];

const ASPECT_RATIO_CSS: Record<AspectRatio, string> = {
  "9:16": "9 / 16",
  "1:1": "1 / 1",
  "3:4": "3 / 4",
  "4:3": "4 / 3",
};

type LastExport = {
  config?: PresetConfig;
  captionsOn?: boolean;
  aspectRatio?: AspectRatio;
  fitMode?: FitMode;
  sticker?: Partial<TextStickerConfig>;
  imageSticker?: ImageStickerConfig | null;
};
function readLastExport(): LastExport | null {
  if (typeof window === "undefined") return null;
  try { return JSON.parse(localStorage.getItem(LAST_KEY) ?? "null"); } catch { return null; }
}

// Boxed numeric input with stepper arrows — replaces the old range sliders.
// Free typing while focused; clamps/snaps to [min,max] on blur or Enter.
function NumberField({ value, min, max, step = 1, unit, onChange }: {
  value: number; min: number; max: number; step?: number; unit?: string;
  onChange: (n: number) => void;
}) {
  const [raw, setRaw] = useState(String(value));
  useEffect(() => { setRaw(String(value)); }, [value]);

  const clamp = (n: number) => Math.min(max, Math.max(min, n));
  const snap  = (n: number) => Math.round(n / step) * step;
  const commit = () => {
    const n = Number(raw);
    if (raw.trim() === "" || Number.isNaN(n)) { setRaw(String(value)); return; }
    onChange(clamp(snap(n)));
  };
  // Shift multiplies the step ×10 — the Figma-standard coarse nudge.
  const nudge = (dir: 1 | -1, mult = 1) => onChange(clamp(snap(value + dir * step * mult)));

  return (
    <div className="flex items-stretch rounded-lg overflow-hidden"
      style={{ background: "var(--yt-surface-2)", border: "1px solid rgba(255,255,255,0.08)" }}>
      <input
        type="text" inputMode="decimal"
        value={raw}
        onChange={e => setRaw(e.target.value)}
        onBlur={commit}
        onKeyDown={e => {
          if (e.key === "Enter") { (e.target as HTMLInputElement).blur(); return; }
          if (e.key === "ArrowUp")   { e.preventDefault(); nudge(1,  e.shiftKey ? 10 : 1); }
          if (e.key === "ArrowDown") { e.preventDefault(); nudge(-1, e.shiftKey ? 10 : 1); }
        }}
        className="w-12 bg-transparent pl-2.5 py-1.5 text-[12px] font-semibold tabular-nums outline-none text-right"
        style={{ color: "var(--yt-text)" }}
      />
      {unit && <span className="flex items-center pl-0.5 pr-1 text-[9px]" style={{ color: "var(--yt-text-2)" }}>{unit}</span>}
      <div className="flex flex-col border-l" style={{ borderColor: "rgba(255,255,255,0.08)" }}>
        <button type="button" aria-label="Increase" onClick={() => nudge(1)}
          className="flex-1 px-1.5 flex items-center transition-colors hover:bg-white/5">
          <ChevronUp className="w-3 h-3" style={{ color: "var(--yt-text-2)" }} />
        </button>
        <button type="button" aria-label="Decrease" onClick={() => nudge(-1)}
          className="flex-1 px-1.5 flex items-center border-t transition-colors hover:bg-white/5"
          style={{ borderColor: "rgba(255,255,255,0.08)" }}>
          <ChevronDown className="w-3 h-3" style={{ color: "var(--yt-text-2)" }} />
        </button>
      </div>
    </div>
  );
}

const Label = ({ children }: { children: React.ReactNode }) => (
  <span className="text-[10px] uppercase tracking-widest font-medium" style={{ color: "var(--yt-text-2)" }}>{children}</span>
);

export function ExportDialog({ clip, posterUrl, initialConfig, initialSubtitlePresetId, restyle, onClose, onQueued }: Props) {
  const [submitting, setSubmitting] = useState(false);
  const [errorMsg,   setErrorMsg]   = useState("");

  // Caption style (knobs live on these nested fields). Captions can be toggled
  // off entirely, in which case no caption layer is exported. Seed from the
  // last export so the dialog opens on the user's previous choices.
  const [captionsOn, setCaptionsOn] = useState<boolean>(() => readLastExport()?.captionsOn ?? true);
  const [config,     setConfig]     = useState<PresetConfig>(() => normalize(initialConfig ?? readLastExport()?.config ?? DEFAULT_PRESET_CONFIG));
  const [aspectRatio, setAspectRatio] = useState<AspectRatio>(() => readLastExport()?.aspectRatio ?? "9:16");
  const [fitMode, setFitMode] = useState<FitMode>(() => readLastExport()?.fitMode ?? "fill");

  // Saved caption styles now live behind the ⋯ menu.
  const [presets,   setPresets]   = useState<SavedPreset[]>([]);
  const [appliedId, setAppliedId] = useState<string | null>(initialSubtitlePresetId ?? null);
  const [dirty,     setDirty]     = useState(false);
  const [naming,    setNaming]    = useState(false);
  const [nameDraft, setNameDraft] = useState("");
  const [menuOpen,  setMenuOpen]  = useState(false);   // preset-picker dropdown
  const [saveOpen,  setSaveOpen]  = useState(false);   // footer "Save preset" inline panel
  const [armedId,   setArmedId]   = useState<string | null>(null);   // two-step preset delete
  const menuRef = useRef<HTMLDivElement>(null);

  // Properties panel tab — captions vs the text sticker.
  const [tab, setTab] = useState<"captions" | "sticker">("captions");

  // Hero width, sized once so the full-height video column fits the dialog.
  const [heroW] = useState(() => {
    if (typeof window === "undefined") return 320;
    const dialogH = Math.min(780, window.innerHeight * 0.92);
    const availH  = dialogH - 150;   // header + footer + canvas padding
    return Math.min(360, Math.max(220, Math.round(availH * 9 / 16)));
  });

  // Falls back to the neutral placeholder when the thumbnail is missing OR fails to load.
  const [thumbFailed, setThumbFailed] = useState(false);

  // Text sticker — per-clip content, never stored in a preset. Colors come from
  // the vetted presets and are remembered across clips.
  const [sticker, setSticker] = useState<TextStickerConfig>(() => {
    const s = readLastExport()?.sticker;
    return { ...DEFAULT_TEXT_STICKER_CONFIG, bg: s?.bg ?? DEFAULT_TEXT_STICKER_CONFIG.bg, fg: s?.fg ?? DEFAULT_TEXT_STICKER_CONFIG.fg, size_pct: s?.size_pct ?? DEFAULT_TEXT_STICKER_CONFIG.size_pct, scale: s?.scale ?? DEFAULT_TEXT_STICKER_CONFIG.scale };
  });
  const previewRef = useRef<HTMLDivElement>(null);

  // Image sticker (uploaded PNG — logo etc.). Remembered across clips so a
  // logo survives from one export to the next.
  const [imgSticker,   setImgSticker]   = useState<ImageStickerConfig | null>(() => readLastExport()?.imageSticker ?? null);
  const [imgUploading, setImgUploading] = useState(false);
  const [imgError,     setImgError]     = useState("");
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Persist image-sticker changes immediately (merge into the last-export blob).
  // Everything else is saved on Clip, but a removal must survive reopening the
  // dialog without requiring an export — otherwise a deleted logo comes back.
  useEffect(() => {
    try {
      const cur = readLastExport() ?? {};
      localStorage.setItem(LAST_KEY, JSON.stringify({ ...cur, imageSticker: imgSticker }));
    } catch {}
  }, [imgSticker]);

  const handleStickerFile = async (file: File) => {
    setImgError("");
    const bytes = new Uint8Array(await file.arrayBuffer());
    const err = validateStickerPng(bytes);
    if (err) { setImgError(err); return; }
    setImgUploading(true);
    try {
      const res = await fetch("/api/stickers/upload", {
        method: "POST", headers: { "Content-Type": "image/png" }, body: bytes,
      });
      const d = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(d.error ?? "Upload failed");
      // Keep the previous placement when replacing an existing image.
      setImgSticker(s => ({
        r2_key: d.r2_key, url: d.url,
        x_pct: s?.x_pct ?? 50, y_pct: s?.y_pct ?? 20, width_pct: s?.width_pct ?? 25,
      }));
    } catch (e) {
      setImgError(e instanceof Error ? e.message : "Upload failed");
    } finally {
      setImgUploading(false);
    }
  };

  // While an overlay is being dragged, show alignment guides in the hero.
  const [captionDrag, setCaptionDrag] = useState(false);
  const [stickerDrag, setStickerDrag] = useState(false);

  // Undo/redo history over the editable state. Kept in a ref (keyboard-driven,
  // no visible controls) with a debounce so drag frames collapse into one entry.
  type Snapshot = { config: PresetConfig; sticker: TextStickerConfig; imgSticker: ImageStickerConfig | null; captionsOn: boolean };
  const historyRef = useRef<History<Snapshot>>(initHistory({ config, sticker, imgSticker, captionsOn }));
  const applyingHistory = useRef(false);

  useEffect(() => {
    if (applyingHistory.current) { applyingHistory.current = false; return; }
    const t = setTimeout(() => {
      const snap: Snapshot = { config, sticker, imgSticker, captionsOn };
      if (JSON.stringify(snap) !== JSON.stringify(historyRef.current.present)) {
        historyRef.current = pushHistory(historyRef.current, snap);
      }
    }, 350);
    return () => clearTimeout(t);
  }, [config, sticker, imgSticker, captionsOn]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (!(e.metaKey || e.ctrlKey) || e.key.toLowerCase() !== "z") return;
      e.preventDefault();
      const h = historyRef.current;
      const next = e.shiftKey ? redo(h) : undo(h);
      if (next === h) return;
      historyRef.current = next;
      applyingHistory.current = true;
      setConfig(next.present.config);
      setSticker(next.present.sticker);
      setImgSticker(next.present.imgSticker);
      setCaptionsOn(next.present.captionsOn);
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, []);

  useEffect(() => {
    fetch("/api/presets").then(r => r.ok ? r.json() : []).then((loaded: SavedPreset[]) => {
      setPresets(loaded);
      // Auto-apply a default preset only when we have nothing else to open with —
      // an explicit preset id wins; a remembered last export suppresses the pick.
      const pick =
        (initialSubtitlePresetId && loaded.find(p => p.id === initialSubtitlePresetId)) ||
        (!initialConfig && !readLastExport()?.config && (loaded.find(p => p.is_default) || loaded[0]));
      if (pick) {
        setConfig(normalize(pick.config));
        setAppliedId(pick.id);
      }
    }).catch(() => {});
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Close the ⋯ menu on any click outside it.
  useEffect(() => {
    if (!menuOpen) return;
    const h = (e: MouseEvent) => { if (!menuRef.current?.contains(e.target as Node)) setMenuOpen(false); };
    document.addEventListener("mousedown", h);
    return () => document.removeEventListener("mousedown", h);
  }, [menuOpen]);

  // Escape closes the menu first, otherwise the dialog (same as the header X).
  useEffect(() => {
    const h = (e: KeyboardEvent) => {
      if (e.key !== "Escape" || submitting) return;
      if (menuOpen) setMenuOpen(false); else onClose();
    };
    document.addEventListener("keydown", h);
    return () => document.removeEventListener("keydown", h);
  }, [menuOpen, submitting, onClose]);

  const hl = config.highlight ?? DEFAULT_HIGHLIGHT_STYLE;
  const style = getCaptionStyle(config.style_id ?? "pill");

  // The two color knobs mean different things per style: HF styles read
  // highlight.*, PIL styles (clean/box) read color.*. PIL writes also mirror
  // into highlight.* so preset swatches stay meaningful.
  const textColor   = style.component ? hl.font_color : config.color.text_color;
  const accentColor = style.component
    ? hl.fill_color
    : style.id === "box" ? config.color.bg_color : config.color.highlight_color;
  const lowContrast = style.accentIsSurface && contrastRatio(textColor, accentColor) < MIN_CAPTION_CONTRAST;

  const editConfig = (next: PresetConfig) => { setConfig(next); setDirty(true); };
  const updateHl     = (patch: Partial<typeof hl>)            => editConfig({ ...config, highlight: { ...hl, ...patch } });
  const updateTypo   = (patch: Partial<PresetConfig["typography"]>) => editConfig({ ...config, typography: { ...config.typography, ...patch } });
  const updateLayout = (patch: Partial<PresetConfig["layout"]>)     => editConfig({ ...config, layout: { ...config.layout, ...patch } });

  const setTextColor = (v: string) => style.component
    ? updateHl({ font_color: v })
    : editConfig({ ...config, color: { ...config.color, text_color: v }, highlight: { ...hl, font_color: v } });
  const setAccentColor = (v: string) => style.component
    ? updateHl({ fill_color: v })
    : editConfig({
        ...config,
        color: { ...config.color, ...(style.id === "box" ? { bg_color: v } : { highlight_color: v }) },
        highlight: { ...hl, fill_color: v },
      });

  const applyPreset = (p: SavedPreset) => {
    setConfig(normalize(p.config)); setAppliedId(p.id); setDirty(false); setCaptionsOn(true);
  };

  // Carousel arrows step the selected style (wrapping); the row scrolls the
  // active tile into view.
  const styleRowRef = useRef<HTMLDivElement>(null);
  const stepStyle = (dir: 1 | -1) => {
    const i = CAPTION_STYLES.findIndex(s => s.id === style.id);
    const next = CAPTION_STYLES[(i + dir + CAPTION_STYLES.length) % CAPTION_STYLES.length];
    editConfig(applyCaptionStyle(config, next));
  };
  useEffect(() => {
    styleRowRef.current
      ?.querySelector(`[data-style-tile="${style.id}"]`)
      ?.scrollIntoView({ behavior: "smooth", block: "nearest", inline: "center" });
  }, [style.id]);

  const resetDefaults = () => {
    setConfig(normalize(DEFAULT_PRESET_CONFIG));
    setCaptionsOn(true);
    setSticker(s => ({ ...s, bg: DEFAULT_TEXT_STICKER_CONFIG.bg, fg: DEFAULT_TEXT_STICKER_CONFIG.fg, size_pct: DEFAULT_TEXT_STICKER_CONFIG.size_pct, scale: DEFAULT_TEXT_STICKER_CONFIG.scale }));
    setImgSticker(null);
    setAppliedId(null); setDirty(true);
  };

  const saveOverwrite = async () => {
    if (!appliedId) return;
    const res = await fetch(`/api/presets/${appliedId}`, {
      method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ config }),
    });
    if (res.ok) {
      setPresets(ps => ps.map(p => p.id === appliedId ? { ...p, config } : p));
      setDirty(false); setSaveOpen(false);
    } else {
      const d = await res.json().catch(() => ({}));
      setErrorMsg(d.error ?? "Couldn't save preset");
    }
  };

  const saveAsNew = async () => {
    const name = nameDraft.trim();
    if (!name) return;
    const res = await fetch("/api/presets", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ name, config }),
    });
    const d = await res.json().catch(() => ({}));
    if (res.ok) {
      setPresets(ps => [...ps, d]);
      setAppliedId(d.id); setDirty(false); setNaming(false); setNameDraft(""); setSaveOpen(false);
    } else {
      setErrorMsg(d.error ?? "Couldn't create preset");   // includes the 15-preset cap message
    }
  };

  const deletePreset = async (id: string) => {
    const res = await fetch(`/api/presets/${id}`, { method: "DELETE" });
    if (res.ok) {
      setPresets(ps => ps.filter(p => p.id !== id));
      if (appliedId === id) { setAppliedId(null); setDirty(true); }
    }
  };

  // Two-step delete: first click arms (3s), second confirms — matches VersionPicker.
  const armOrDelete = (id: string) => {
    if (armedId === id) { setArmedId(null); deletePreset(id); }
    else { setArmedId(id); setTimeout(() => setArmedId(c => (c === id ? null : c)), 3000); }
  };

  const handleClip = async () => {
    setErrorMsg("");

    setSubmitting(true);
    // Capture the sticker's actual line breaks from the live preview so the worker
    // reproduces them verbatim — the browser's layout is the source of truth.
    let stickerPayload: TextStickerConfig | null = null;
    if (sticker.text.trim()) {
      const span = previewRef.current?.querySelector<HTMLElement>("[data-sticker-span]");
      const lines = span ? extractWrappedLines(span) : undefined;
      stickerPayload = { ...sticker, lines };
    }
    try {
      const res = await fetch(`/api/clips/${clip.id}/export`, {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify({
          aspect_ratio:        aspectRatio,
          fit_mode:            fitMode,
          preset_config:       captionsOn ? config : null,
          text_sticker_config: stickerPayload,
          image_sticker_config: imgSticker
            ? { r2_key: imgSticker.r2_key, x_pct: imgSticker.x_pct, y_pct: imgSticker.y_pct, width_pct: imgSticker.width_pct }
            : null,
        }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error ?? "Failed to start export");
      // Remember these settings as the defaults for the next clip.
      try {
        localStorage.setItem(LAST_KEY, JSON.stringify({
          config, captionsOn, aspectRatio, fitMode,
          sticker: { bg: sticker.bg, fg: sticker.fg, size_pct: sticker.size_pct, scale: sticker.scale },
          imageSticker: imgSticker,
        }));
      } catch {}
      onQueued(data.export_id, `${clip.hook?.slice(0, 40) ?? "Clip"} · ${aspectRatio}`);
      onClose();
    } catch (err: unknown) {
      setErrorMsg(err instanceof Error ? err.message : "Something went wrong");
      setSubmitting(false);
    }
  };

  const verticalPct = config.layout.vertical_percent ?? 80;
  const thumbSrc = thumbFailed ? null : (clip.thumbnail_url ?? posterUrl);

  return (
    <Modal onClose={onClose} closeOnEsc={false} label="Export clip"
      className="flex flex-col rounded-2xl overflow-hidden shadow-2xl"
      style={{ width: "min(1100px, 96vw)", height: "min(780px, 92vh)", background: "var(--yt-bg)", border: "1px solid var(--yt-border)" }}>

        {/* ── Header ── */}
        <div className="flex items-center justify-between pl-5 pr-3 py-2.5 flex-shrink-0" style={{ borderBottom: "1px solid var(--yt-border)" }}>
          <div className="flex items-baseline gap-2">
            <p className="text-sm font-semibold" style={{ color: "var(--yt-text)" }}>{restyle ? "Restyle clip" : "Export clip"}</p>
            <span className="text-[11px]" style={{ color: "var(--yt-text-2)" }}>{aspectRatio} frame</span>
          </div>
          <button aria-label="Close" onClick={onClose} disabled={submitting}
            className="p-1.5 rounded-md transition-colors hover:bg-white/5 disabled:opacity-40">
            <X className="w-4 h-4" style={{ color: "var(--yt-text-2)" }} />
          </button>
        </div>

        <div
          className="flex flex-wrap items-center justify-center gap-3 px-4 py-2.5 flex-shrink-0"
          style={{ borderBottom: "1px solid var(--yt-border)", background: "var(--yt-bg)" }}
        >
          <div className="flex items-center gap-1.5 rounded-lg p-1" style={{ background: "rgba(255,255,255,0.06)" }}>
            {ASPECT_RATIO_OPTIONS.map((ratio) => {
              const active = aspectRatio === ratio;
              return (
                <button
                  key={ratio}
                  type="button"
                  onClick={() => {
                    setAspectRatio(ratio);
                    setErrorMsg("");
                  }}
                  className="px-3 py-1.5 rounded-md text-[11px] font-medium transition-colors"
                  style={{
                    background: active ? "#fff" : "transparent",
                    color: active ? "#0a0a0a" : "var(--yt-text-2)",
                  }}
                >
                  {ratio}
                </button>
              );
            })}
          </div>

          <div className="flex items-center gap-1.5 rounded-lg p-1" style={{ background: "rgba(255,255,255,0.06)" }}>
            {(["fill", "fit"] as FitMode[]).map((mode) => {
              const active = fitMode === mode;
              return (
                <button
                  key={mode}
                  type="button"
                  onClick={() => {
                    setFitMode(mode);
                    setErrorMsg("");
                  }}
                  className="px-4 py-1.5 rounded-md text-[11px] font-medium transition-colors"
                  style={{
                    background: active ? "#fff" : "transparent",
                    color: active ? "#0a0a0a" : "var(--yt-text-2)",
                  }}
                >
                  {mode === "fill" ? "Fill" : "Fit"}
                </button>
              );
            })}
          </div>
        </div>

        {/* ── Main: video | properties panel ── */}
        <div className="flex flex-col sm:flex-row flex-1 min-h-0 overflow-y-auto sm:overflow-hidden">

          {/* Video column — the hero preview owns the left side */}
          <div className="flex-none flex flex-col items-center justify-start sm:justify-center gap-4 p-5 sm:min-h-0" style={{ background: "#0f0f0f" }}>
              <div ref={previewRef} className="relative bg-black rounded-lg overflow-hidden flex-shrink-0 shadow-2xl transition-all duration-200"
                style={{ width: heroW, aspectRatio: ASPECT_RATIO_CSS[aspectRatio] }}>
                {thumbSrc ? (
                  fitMode === "fill" ? (
                    // eslint-disable-next-line @next/next/no-img-element
                    <img
                      src={thumbSrc}
                      alt=""
                      onError={() => setThumbFailed(true)}
                      className="absolute inset-0 w-full h-full object-cover"
                    />
                  ) : (
                    <>
                      {/* Blurred background preview */}
                      {/* eslint-disable-next-line @next/next/no-img-element */}
                      <img
                        src={thumbSrc}
                        alt=""
                        className="absolute inset-0 w-full h-full object-cover scale-110 blur-xl opacity-70"
                      />

                      {/* Full source frame */}
                      {/* eslint-disable-next-line @next/next/no-img-element */}
                      <img
                        src={thumbSrc}
                        alt=""
                        onError={() => setThumbFailed(true)}
                        className="absolute inset-0 w-full h-full object-contain"
                      />
                    </>
                  )
                ) : (
                  <div className="absolute inset-0" style={{ background: "#ffffff" }} />
                )}

                {/* Alignment guides — only while dragging, to avoid clutter. */}
                {(captionDrag || stickerDrag) && (
                  <>
                    {/* Safe zones: the TikTok/Reels top and bottom UI bands. */}
                    <div className="absolute inset-x-0 top-0 z-10 pointer-events-none" style={{ height: "14%", background: "rgba(255,255,255,0.06)", borderBottom: "1px dashed rgba(255,255,255,0.25)" }} />
                    <div className="absolute inset-x-0 bottom-0 z-10 pointer-events-none" style={{ height: "15%", background: "rgba(255,255,255,0.06)", borderTop: "1px dashed rgba(255,255,255,0.25)" }} />
                  </>
                )}
                {/* Center guide — appears when a sticker snaps to X=50. */}
                {stickerDrag && (sticker.x_pct === CENTER || imgSticker?.x_pct === CENTER) && (
                  <div className="absolute top-0 bottom-0 z-30 pointer-events-none" style={{ left: "50%", width: 1, background: "var(--yt-red)" }} />
                )}

                {captionsOn && (
                  <CaptionOverlay
                    config={config}
                    previewWidth={heroW}
                    text={clip.text}
                    editable
                    containerRef={previewRef}
                    onMoveY={pct => updateLayout({ vertical_percent: Math.round(pct) })}
                    onDragStateChange={setCaptionDrag}
                  />
                )}
                <TextStickerOverlay
                  config={sticker}
                  previewWidth={heroW}
                  editable
                  containerRef={previewRef}
                  onMove={(x_pct, y_pct) => setSticker(s => ({ ...s, x_pct, y_pct }))}
                  onScale={scale => setSticker(s => ({ ...s, scale: Math.round(scale * 100) / 100 }))}
                  onDragStateChange={setStickerDrag}
                />
                {imgSticker && (
                  <ImageStickerOverlay
                    config={imgSticker}
                    previewWidth={heroW}
                    editable
                    containerRef={previewRef}
                    onMove={(x_pct, y_pct) => setImgSticker(s => s && ({ ...s, x_pct, y_pct }))}
                    onResize={width_pct => setImgSticker(s => s && ({ ...s, width_pct }))}
                    onDragStateChange={setStickerDrag}
                  />
                )}
              </div>
          </div>

          {/* ── Properties panel — styles + controls fill the right side ── */}
          <div className="flex flex-col flex-1 min-w-0 sm:min-h-0 border-t sm:border-t-0 sm:border-l" style={{ borderColor: "var(--yt-border)" }}>
            {/* Tabs — left-aligned with the panel content (px-5) */}
            <div className="flex gap-6 px-5 flex-shrink-0" style={{ borderBottom: "1px solid var(--yt-border)" }}>
              {(["captions", "sticker"] as const).map(t => (
                <button key={t} onClick={() => setTab(t)}
                  className="py-2.5 text-[12px] font-medium transition-colors"
                  style={{
                    color: tab === t ? "var(--yt-text)" : "var(--yt-text-2)",
                    boxShadow: tab === t ? "inset 0 -2px 0 #fff" : "none",
                  }}>
                  {t === "captions" ? "Captions" : "Sticker"}
                  {t === "sticker" && (sticker.text.trim() || imgSticker) && (
                    <span className="inline-block w-1.5 h-1.5 rounded-full ml-1.5 align-middle" style={{ background: "var(--yt-red)" }} />
                  )}
                </button>
              ))}
            </div>

            <div className="flex-1 sm:min-h-0 sm:overflow-y-auto px-5 py-4">
              {tab === "captions" ? (
                <div className="space-y-5">
                  {/* Toggle sits next to its label — not across the panel. */}
                  <div className="flex items-center gap-3">
                    <Label>Captions</Label>
                    <button
                      role="switch" aria-checked={captionsOn}
                      onClick={() => setCaptionsOn(v => !v)}
                      className="relative w-9 h-5 rounded-full transition-colors"
                      style={{ background: captionsOn ? "#fff" : "var(--yt-surface-2)", border: "1px solid rgba(255,255,255,0.1)" }}
                    >
                      <span className="absolute top-0.5 w-4 h-4 rounded-full transition-all"
                        style={{ left: captionsOn ? 18 : 2, background: captionsOn ? "#0a0a0a" : "var(--yt-text-2)" }} />
                    </button>
                  </div>

                  {/* Everything below greys out + disables when captions are off. */}
                  <div className="space-y-5" style={{ opacity: captionsOn ? 1 : 0.4, pointerEvents: captionsOn ? "auto" : "none" }}>
                    {/* Style carousel — one row; arrows step the selection, tiles preview
                        each style on the clip's own thumbnail. */}
                    <div>
                      <div className="flex items-baseline gap-2 mb-2">
                        <Label>Style</Label>
                        <span className="text-[11px] font-medium" style={{ color: "var(--yt-text)" }}>{style.name}</span>
                      </div>
                      <div className="flex items-center gap-2">
                        <button
                          aria-label="Previous style"
                          onClick={() => stepStyle(-1)}
                          className="flex-shrink-0 p-1.5 rounded-full transition-colors hover:bg-white/10"
                          style={{ background: "var(--yt-surface-2)", border: "1px solid rgba(255,255,255,0.08)" }}
                        >
                          <ChevronLeft className="w-4 h-4" style={{ color: "var(--yt-text)" }} />
                        </button>
                        <div ref={styleRowRef} className="no-scrollbar flex-1 min-w-0 flex gap-2.5 overflow-x-auto py-0.5">
                          {CAPTION_STYLES.map(s => {
                            const active = s.id === style.id;
                            // Selected tile tracks the user's live colors; others show the style's defaults.
                            const tileCfg = active ? config : applyCaptionStyle(config, s);
                            return (
                              <button
                                key={s.id}
                                data-style-tile={s.id}
                                onClick={() => editConfig(applyCaptionStyle(config, s))}
                                className="relative block rounded-lg overflow-hidden bg-black flex-shrink-0 transition-transform hover:scale-[1.02]"
                                style={{ width: STYLE_W, aspectRatio: "9/16", outline: active ? "2px solid #fff" : "1px solid rgba(255,255,255,0.1)" }}
                                title={s.name}
                              >
                                {thumbSrc ? (
                                  // eslint-disable-next-line @next/next/no-img-element
                                  <img src={thumbSrc} alt="" onError={() => setThumbFailed(true)} className="absolute inset-0 w-full h-full object-cover" />
                                ) : (
                                  <div className="absolute inset-0" style={{ background: "#ffffff" }} />
                                )}
                                <CaptionOverlay config={tileCfg} previewWidth={STYLE_W} text={clip.text} />
                                <span
                                  className="absolute bottom-0 left-0 right-0 pt-4 pb-1.5 text-center text-[10px] font-semibold uppercase tracking-wider pointer-events-none"
                                  style={{ color: "#fff", background: "linear-gradient(transparent, rgba(0,0,0,0.75))" }}
                                >{s.name}</span>
                              </button>
                            );
                          })}
                        </div>
                        <button
                          aria-label="Next style"
                          onClick={() => stepStyle(1)}
                          className="flex-shrink-0 p-1.5 rounded-full transition-colors hover:bg-white/10"
                          style={{ background: "var(--yt-surface-2)", border: "1px solid rgba(255,255,255,0.08)" }}
                        >
                          <ChevronRight className="w-4 h-4" style={{ color: "var(--yt-text)" }} />
                        </button>
                      </div>
                    </div>

                    {/* Customize — one grid, every control labeled above, so columns align. */}
                    <div className="pt-4 border-t space-y-4" style={{ borderColor: "var(--yt-border)" }}>
                    <div className="relative max-w-[320px]" ref={menuRef}>
                      <span className="block mb-1.5"><Label>Preset</Label></span>
                      <button aria-label="Pick preset" onClick={() => setMenuOpen(o => !o)} disabled={submitting}
                        className="flex items-center justify-between gap-1.5 w-full px-3 py-2 rounded-lg text-[12px] disabled:opacity-40 transition-colors hover:bg-white/5"
                        style={{ background: "var(--yt-surface-2)", border: "1px solid rgba(255,255,255,0.08)", color: "var(--yt-text)" }}>
                        <span className="truncate">{presets.find(p => p.id === appliedId)?.name ?? "Pick preset"}</span>
                        <ChevronDown className="w-3.5 h-3.5 flex-shrink-0" style={{ color: "var(--yt-text-2)" }} />
                      </button>

                      {menuOpen && (
                        <div className="absolute left-0 right-0 top-full mt-1.5 z-40 rounded-xl p-2 shadow-2xl"
                          style={{ background: "var(--yt-bg)", border: "1px solid var(--yt-border)" }}>
                          <div className="flex flex-col gap-0.5 max-h-56 overflow-y-auto no-scrollbar">
                            {presets.length === 0 && (
                              <p className="text-[11px] px-2 py-1.5" style={{ color: "var(--yt-text-2)" }}>No saved presets yet.</p>
                            )}
                            {presets.map(p => (
                              <div key={p.id} className="flex items-center gap-1 group">
                                <button
                                  onClick={() => { applyPreset(p); setMenuOpen(false); }}
                                  className="flex-1 min-w-0 flex items-center gap-2 px-2 py-1.5 rounded-md text-[12px] text-left transition-colors hover:bg-white/5"
                                  style={{ color: "var(--yt-text)" }}
                                >
                                  <span className="w-3 h-3 rounded-[3px] flex-shrink-0" style={{ background: (p.config.highlight ?? DEFAULT_HIGHLIGHT_STYLE).fill_color, boxShadow: "0 0 0 1px rgba(255,255,255,0.12)" }} />
                                  <span className="truncate flex-1">{p.name}</span>
                                  {appliedId === p.id && <Check className="w-3.5 h-3.5 flex-shrink-0" style={{ color: "var(--yt-text-2)" }} />}
                                </button>
                                <button
                                  aria-label={armedId === p.id ? `Confirm delete ${p.name}` : `Delete ${p.name}`}
                                  title={armedId === p.id ? "Click again to delete" : "Delete preset"}
                                  onClick={() => armOrDelete(p.id)}
                                  className={`flex items-center gap-1 px-1.5 py-1 rounded-md flex-shrink-0 transition-opacity hover:bg-white/5 ${
                                    armedId === p.id ? "opacity-100" : "opacity-0 group-hover:opacity-100"
                                  }`}
                                >
                                  <Trash2 className="w-3.5 h-3.5" style={{ color: armedId === p.id ? "#f87171" : "var(--yt-text-2)" }} />
                                  {armedId === p.id && <span className="text-[10px] font-semibold" style={{ color: "#f87171" }}>Delete?</span>}
                                </button>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>

                    {/* Placement + size — labels above fields, matching the color pickers */}
                    <div className="grid grid-cols-2 gap-x-4 gap-y-4 max-w-[480px]">
                      <div className="flex flex-col gap-1.5 items-start">
                        <Label>Position Y</Label>
                        <NumberField value={verticalPct} min={POSITION_MIN} max={POSITION_MAX} step={1} unit="%"
                          onChange={v => updateLayout({ vertical_percent: v })} />
                        <span className="text-[9px]" style={{ color: "var(--yt-text-2)" }}>Or drag on the preview</span>
                      </div>
                      <div className="flex flex-col gap-1.5 items-start">
                        <Label>Font size</Label>
                        <NumberField value={config.typography.font_size} min={50} max={200} step={5} unit="%"
                          onChange={v => updateTypo({ font_size: v })} />
                      </div>
                      <ColorPicker label="Text color" value={textColor} onChange={setTextColor} />
                      <ColorPicker label={style.accentLabel} value={accentColor} onChange={setAccentColor} />
                    </div>
                    {lowContrast && (
                      <div className="flex items-center gap-2 text-[11px]" style={{ color: "#fbbf24" }}>
                        <AlertTriangle className="w-3.5 h-3.5 flex-shrink-0" />
                        Low contrast — the text may be hard to read.
                      </div>
                    )}
                    </div>
                  </div>
                </div>
              ) : (
                <div className="max-w-[440px]">
                  {/* Text sticker — independent of captions, stays active when they're off */}
                  <Label>Text sticker</Label>
                  <input
                    value={sticker.text}
                    maxLength={TEXT_STICKER_MAX_CHARS}
                    onChange={e => setSticker(s => ({ ...s, text: e.target.value }))}
                    placeholder="Add a text sticker — drag it on the preview"
                    className="w-full mt-2 px-3 py-2 rounded-lg text-[11px] outline-none"
                    style={{ background: "var(--yt-surface-2)", border: "1px solid rgba(255,255,255,0.06)", color: "var(--yt-text)" }}
                  />
                  <div className="flex items-center justify-between mt-1">
                    <span className="text-[10px]" style={{ color: "var(--yt-text-2)" }}>Pick a color combo below</span>
                    <span className="text-[10px] tabular-nums" style={{ color: "var(--yt-text-2)" }}>{sticker.text.length}/{TEXT_STICKER_MAX_CHARS}</span>
                  </div>

                  {/* Vetted high-contrast color presets */}
                  <div className="flex flex-wrap gap-2 mt-3">
                    {STICKER_COLOR_PRESETS.map(p => {
                      const active = sticker.bg.toLowerCase() === p.bg && sticker.fg.toLowerCase() === p.fg;
                      return (
                        <button
                          key={p.id}
                          onClick={() => setSticker(s => ({ ...s, bg: p.bg, fg: p.fg }))}
                          title={p.label}
                          className="flex items-center justify-center rounded-md text-[12px] font-bold transition-transform"
                          style={{
                            width: 40, height: 28, background: p.bg, color: p.fg,
                            outline: active ? "2px solid #fff" : "1px solid rgba(255,255,255,0.15)",
                            outlineOffset: active ? 1 : 0,
                          }}
                        >Aa</button>
                      );
                    })}
                  </div>

                  {sticker.text.trim() && (
                    <div className="grid grid-cols-2 gap-x-4 mt-4">
                      <div className="flex flex-col gap-1.5 items-start">
                        <Label>Size</Label>
                        <NumberField value={sticker.size_pct} min={4} max={16} step={0.5} unit="%"
                          onChange={v => setSticker(s => ({ ...s, size_pct: v }))} />
                      </div>
                      <div className="flex flex-col gap-1.5 items-start">
                        <Label>Scale</Label>
                        <NumberField value={sticker.scale} min={STICKER_SCALE_MIN} max={STICKER_SCALE_MAX} step={0.05} unit="×"
                          onChange={v => setSticker(s => ({ ...s, scale: v }))} />
                      </div>
                    </div>
                  )}

                  {/* Logo / PNG sticker — upload once, reused across clips. */}
                  <div className="mt-5 pt-4 border-t" style={{ borderColor: "var(--yt-border)" }}>
                    <Label>Logo / image</Label>
                    {!imgSticker ? (
                      <button
                        onClick={() => fileInputRef.current?.click()}
                        onDragOver={e => e.preventDefault()}
                        onDrop={e => { e.preventDefault(); const f = e.dataTransfer.files?.[0]; if (f) handleStickerFile(f); }}
                        disabled={imgUploading}
                        className="w-full mt-2 px-3 py-5 rounded-lg text-[11px] text-center border border-dashed transition-colors hover:bg-white/5 disabled:opacity-50"
                        style={{ borderColor: "rgba(255,255,255,0.18)", color: "var(--yt-text-2)", background: "transparent" }}
                      >
                        {imgUploading ? (
                          <Loader2 className="w-4 h-4 animate-spin mx-auto" />
                        ) : (
                          <>
                            Drop a PNG here or click to browse
                            <span className="block mt-1 text-[9px]">Transparent PNG · max 2 MB</span>
                          </>
                        )}
                      </button>
                    ) : (
                      <div className="flex items-center gap-2.5 mt-2 p-2 rounded-lg"
                        style={{ background: "var(--yt-surface-2)", border: "1px solid rgba(255,255,255,0.06)" }}>
                        {/* eslint-disable-next-line @next/next/no-img-element */}
                        <img src={imgSticker.url} alt="" className="w-10 h-10 object-contain rounded flex-shrink-0"
                          style={{ background: "rgba(255,255,255,0.08)" }} />
                        <div className="flex-1 min-w-0">
                          <p className="text-[11px]" style={{ color: "var(--yt-text)" }}>PNG sticker</p>
                          <p className="text-[9px]" style={{ color: "var(--yt-text-2)" }}>Drag it on the preview</p>
                        </div>
                        <button onClick={() => fileInputRef.current?.click()} disabled={imgUploading}
                          className="px-2 py-1 rounded-md text-[10px] flex-shrink-0 transition-colors hover:bg-white/5 disabled:opacity-50"
                          style={{ color: "var(--yt-text)", border: "1px solid rgba(255,255,255,0.08)" }}>
                          {imgUploading ? "…" : "Replace"}
                        </button>
                        <button aria-label="Remove image sticker" onClick={() => { setImgSticker(null); setImgError(""); }}
                          className="p-1.5 rounded-md flex-shrink-0 transition-colors hover:bg-white/5">
                          <Trash2 className="w-3.5 h-3.5" style={{ color: "var(--yt-text-2)" }} />
                        </button>
                      </div>
                    )}
                    <input
                      ref={fileInputRef} type="file" accept="image/png" className="hidden"
                      onChange={e => { const f = e.target.files?.[0]; if (f) handleStickerFile(f); e.target.value = ""; }}
                    />
                    {imgError && <p className="text-[10px] mt-1.5" style={{ color: "#f87171" }}>{imgError}</p>}
                    {imgSticker && (
                      <div className="flex flex-col gap-1.5 items-start mt-3">
                        <Label>Width</Label>
                        <NumberField value={imgSticker.width_pct} min={IMAGE_STICKER_MIN_WPCT} max={IMAGE_STICKER_MAX_WPCT} step={1} unit="%"
                          onChange={v => setImgSticker(s => s && ({ ...s, width_pct: v }))} />
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* ── Footer ── */}
        <div className="flex-shrink-0 px-5 py-3" style={{ borderTop: "1px solid var(--yt-border)" }}>
          {errorMsg && (
            <div className="flex items-center gap-2 text-xs text-red-400 mb-2">
              <AlertCircle className="w-3.5 h-3.5 flex-shrink-0" />
              {errorMsg}
            </div>
          )}
          {/* Inline save panel — opened by the Save preset button below. */}
          {saveOpen && (
            <div className="mb-2 p-2.5 rounded-lg" style={{ background: "var(--yt-surface-2)", border: "1px solid rgba(255,255,255,0.08)" }}>
              {appliedId && dirty && !naming ? (
                <div className="flex items-center gap-2">
                  <span className="text-[11px] flex-1 truncate" style={{ color: "var(--yt-text-2)" }}>
                    Update “{presets.find(p => p.id === appliedId)?.name}”?
                  </span>
                  <button onClick={saveOverwrite} className="px-2 py-1 rounded-md text-[11px] font-medium" style={{ background: "#fff", color: "#0a0a0a" }}>Update</button>
                  <button onClick={() => { setNameDraft(""); setNaming(true); }} className="px-2 py-1 rounded-md text-[11px]" style={{ background: "var(--yt-bg)", color: "var(--yt-text)", border: "1px solid rgba(255,255,255,0.08)" }}>Save as new</button>
                  <button onClick={() => setSaveOpen(false)} className="px-1.5 py-1 rounded-md text-[11px]" style={{ color: "var(--yt-text-2)" }}>Cancel</button>
                </div>
              ) : (
                <div className="flex items-center gap-1.5">
                  <input
                    autoFocus
                    value={nameDraft}
                    onChange={e => setNameDraft(e.target.value)}
                    onKeyDown={e => { if (e.key === "Enter") saveAsNew(); if (e.key === "Escape") { setNaming(false); setSaveOpen(false); } }}
                    placeholder="Preset name"
                    className="flex-1 min-w-0 px-2 py-1 rounded-md text-[11px] outline-none"
                    style={{ background: "var(--yt-bg)", border: "1px solid rgba(255,255,255,0.08)", color: "var(--yt-text)" }}
                  />
                  <button onClick={saveAsNew} className="px-2 py-1 rounded-md text-[11px] font-medium" style={{ background: "#fff", color: "#0a0a0a" }}>Save</button>
                  <button onClick={() => { setNaming(false); setSaveOpen(false); }} className="px-1.5 py-1 rounded-md text-[11px]" style={{ color: "var(--yt-text-2)" }}>Cancel</button>
                </div>
              )}
            </div>
          )}
          <div className="flex items-center gap-2">
            <p className="hidden sm:block text-xs flex-1 min-w-0 truncate" style={{ color: "var(--yt-text-2)" }}>
              {restyle
                ? "Reuses your cached edit — restyling won't cost a credit."
                : "Runs in the background — you'll get a notification when it's ready."}
            </p>
            <button onClick={resetDefaults} disabled={submitting}
              className="text-[11px] underline-offset-2 hover:underline flex-shrink-0 px-1 disabled:opacity-40"
              style={{ color: "var(--yt-text-2)" }}>
              Reset
            </button>
            {/* Secondary: save the current caption style as a preset (inline panel above). */}
            <button
              onClick={() => {
                const open = !saveOpen;
                setSaveOpen(open);
                if (open && !(appliedId && dirty)) { setNameDraft(""); setNaming(true); }
                else if (open) { setNaming(false); }
              }}
              disabled={submitting}
              className="flex items-center justify-center gap-1.5 px-3 py-2 flex-shrink-0 disabled:opacity-50 text-[12px] font-medium rounded-lg transition-colors hover:bg-white/5"
              style={{ background: "var(--yt-surface-2)", color: "var(--yt-text)", border: "1px solid rgba(255,255,255,0.08)" }}
              title="Save the current caption style as a preset"
            >
              <Bookmark className="w-3.5 h-3.5" /> Save preset
            </button>
            <Button
              variant="white"
              onClick={handleClip}
              disabled={submitting}
              className="flex-1 sm:flex-none sm:px-8 py-2 gap-2 active:scale-95"
            >
              {submitting ? <Loader2 className="w-4 h-4 animate-spin" /> : <Scissors className="w-4 h-4" />}
              {submitting ? "Queuing…" : restyle ? "Restyle" : "Clip"}
            </Button>
          </div>
        </div>
    </Modal>
  );
}
