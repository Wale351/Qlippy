// ============================================================
// Domain types — shared across the app
// ============================================================

export type VideoStatus = "pending" | "transcribing" | "identifying" | "ready" | "error";
export type JobStatus   = "queued" | "processing" | "done" | "error" | "cancelled";
export type JobType     = "transcribe" | "identify_clips" | "cut_clip" | "export_clip" | "edit_clip" | "style_clip";
export type AspectRatio = "9:16" | "16:9" | "1:1" | "4:5" | "3:4" | "4:3";
export type Resolution  = "1080p";
export type Tone        = "hook" | "educational" | "entertaining" | "emotional" | "cta" | "story" | "controversy";
export type ModelSize   = "base" | "medium" | "large";

export interface Collection {
  id: string;
  user_id: string;
  name: string;
  created_at: string;
}

export interface Video {
  id: string;
  user_id: string;
  youtube_url: string;
  youtube_id: string;
  title: string;
  thumbnail_url: string | null;
  duration_sec: number | null;
  channel_name: string | null;
  tags: string[];
  status: VideoStatus;
  processing_stage: string | null;
  error_msg: string | null;
  face_url: string | null;
  collection_id: string | null;
  source_r2_key: string | null;
  audio_r2_key: string | null;
  portrait_preview_url: string | null;
  has_transcript?: boolean;
  created_at: string;
  updated_at: string;
}

export interface Transcript {
  id: string;
  video_id: string;
  user_id: string;
  language: string | null;
  raw_words: WhisperWord[];
  segments: WhisperSegment[];
  full_text: string | null;
  r2_key: string | null;
  replicate_id: string | null;
  speaker_labels: Record<string, string> | null;  // { "SPEAKER_00": "John", ... }
  model_size: ModelSize | null;
  created_at: string;
}

export interface WhisperWord {
  word?: string;
  text?: string;
  start: number;
  end: number;
  score?: number;
  speaker?: string;
}

export interface WhisperSegment {
  text: string;
  start: number;
  end: number;
  speaker?: string;
}

export interface Cut {
  start: number;
  end: number;
}

export interface Clip {
  id: string;
  video_id: string;
  user_id: string;
  hook: string | null;
  text: string | null;
  cuts: Cut[];
  duration_sec: number | null;
  score: number | null;
  tone: Tone | null;
  sentence_groups: number[][] | null;
  stitched: boolean;
  chunks: {
    start_ms: number;
    end_ms: number;
    crosstalk_flagged?: boolean;
  }[];
  crosstalk_flagged: boolean;
  reasoning: string | null;
  source: "ai" | "manual";
  notes: string | null;
  social_title: string | null;       // selected/default social title
  social_description: string | null; // selected/default social description
  social_title_options: string[];
  social_description_options: string[];
  overlay_text_options: string[];
  overlay_text: string | null;
  approved: boolean | null;          // null = not reviewed, true = approved, false = rejected
  series_id: string | null;          // group clips into a series
  thumbnail_url: string | null;      // face-forward frame extracted by worker
  label?: "hero" | "strong" | "decent" | "weak" | null;  // checklist classification; "weak" = Crafter rejected it, see reasoning
  generation?: number;               // identify run number (keep-old-clips re-runs)
  created_at: string;
  updated_at: string;
}

export interface Export {
  id: string;
  clip_id: string;
  user_id: string;
  aspect_ratio: AspectRatio;
  preset_id: string | null;
  edit_id: string | null;            // cached portrait this styled export was built from
  r2_key: string | null;
  r2_url: string | null;
  status: "queued" | "processing" | "done" | "error";
  error_msg: string | null;
  created_at: string;
  updated_at: string;
}

// Cached face-tracked portrait — the expensive "edit" stage, reused across exports.
export interface ClipEdit {
  id: string;
  clip_id: string;
  user_id: string;
  aspect_ratio: AspectRatio;
  cuts_hash: string;
  r2_key: string | null;
  status: "queued" | "processing" | "done" | "error";
  error_msg: string | null;
  created_at: string;
  updated_at: string;
}

export interface Job {
  id: string;
  user_id: string;
  type: JobType;
  status: JobStatus;
  entity_id: string;
  payload: Record<string, unknown>;
  result: Record<string, unknown>;
  error_msg: string | null;
  attempts: number;
  queued_at: string;
  started_at: string | null;
  finished_at: string | null;
}

// ---- Preset types ----

export interface TypographyConfig {
  font_family: string;
  font_size: number;
  font_weight: string;
  all_caps: boolean;
  font_url: string;
  // Stage 2 additions
  letter_spacing?: number;
  line_height?: number;
  italic?: boolean;
}

export interface ColorConfig {
  text_color: string;
  highlight_color: string;
  stroke_color: string;
  stroke_width: number;
  bg_color: string;
  bg_opacity: number;
  // Stage 2 additions
  drop_shadow?: boolean;
  shadow_color?: string;
  shadow_blur?: number;
  shadow_x?: number;
  shadow_y?: number;
  text_opacity?: number;
  gradient?: boolean;
  gradient_color?: string;
}

export interface LayoutConfig {
  vertical_position: "top" | "middle" | "bottom";
  text_align: "left" | "center" | "right";
  two_lines: boolean;
  // Stage 2 additions
  vertical_percent?: number;  // 0–100, overrides vertical_position
  padding?: number;
}

export interface AnimationConfig {
  fade_in_duration: number;
  word_bounce: boolean;
  // Stage 2 additions
  time_scale?: number;        // GSAP timeScale multiplier
  slide_in?: boolean;
  slide_direction?: "up" | "down" | "left" | "right";
}

export interface HighlightStyle {
  fill_color: string;    // hex — pill background
  font_color: string;    // hex — word text color
  shadow_blur: number;   // px — 0 = no glow
  border_radius: number; // px
  letter_spacing: number;// em
}

export const DEFAULT_HIGHLIGHT_STYLE: HighlightStyle = {
  fill_color: "#ff1745",
  font_color: "#ffffff",
  shadow_blur: 30,
  border_radius: 10,
  letter_spacing: 0.02,
};

export interface PresetConfig {
  typography: TypographyConfig;
  color: ColorConfig;
  layout: LayoutConfig;
  animation: AnimationConfig;
  hyperframes_component?: string | null;   // null = classic PIL renderer
  highlight?: HighlightStyle;
  style_id?: string;                       // caption style id (see lib/caption-styles.ts)
}

export interface Preset {
  id: string;
  user_id: string;
  name: string;
  is_default: boolean;
  config: PresetConfig;
  created_at: string;
  updated_at: string;
}

// Per-account cap on saved caption-style presets. At the limit, creating a new
// one is blocked (the user must delete one) rather than silently dropping any.
export const MAX_PRESETS = 15;

// ---- Niche / audience profile (Stage 2) ----

export interface NicheProfile {
  niche: string;              // e.g. "tech entrepreneurship"
  audience: string;           // e.g. "founders and indie hackers"
  hook_style: string;         // e.g. "bold contrarian statements"
  clip_duration_min: number;
  clip_duration_max: number;
  preferred_tones: Tone[];
}

// ---- Default preset ----

export const DEFAULT_PRESET_CONFIG: PresetConfig = {
  typography: {
    font_family: "Inter",
    font_size: 100,
    font_weight: "700",
    all_caps: false,
    font_url: "https://fonts.googleapis.com/css2?family=Inter:wght@700&display=swap",
    letter_spacing: 0,
    line_height: 1.2,
    italic: false,
  },
  color: {
    text_color: "#FFFFFF",
    highlight_color: "#FFDD00",
    stroke_color: "#000000",
    stroke_width: 2,
    bg_color: "#000000",
    bg_opacity: 0,
    drop_shadow: false,
    shadow_color: "#000000",
    shadow_blur: 8,
    shadow_x: 0,
    shadow_y: 2,
    text_opacity: 1,
    gradient: false,
    gradient_color: "#FF6B6B",
  },
  layout: {
    vertical_position: "bottom",
    text_align: "center",
    two_lines: false,
    vertical_percent: undefined,
    padding: 16,
  },
  animation: {
    fade_in_duration: 0.15,
    word_bounce: false,
    time_scale: 1.0,
    slide_in: false,
    slide_direction: "up",
  },
  highlight: DEFAULT_HIGHLIGHT_STYLE,
  hyperframes_component: null,
  style_id: "clean",
};


// ---- Text sticker (Instagram-style: black text on solid-white pill) ----

export const TEXT_STICKER_MAX_CHARS = 30;

export interface TextStickerConfig {
  text: string;       // ≤ TEXT_STICKER_MAX_CHARS
  x_pct: number;      // 0–100, horizontal center of the sticker
  y_pct: number;      // 0–100, vertical center
  size_pct: number;   // font size as % of video width
  scale: number;      // overall object scale multiplier (0.5–2)
  bg: string;         // pill background hex
  fg: string;         // text hex
  lines?: string[];   // browser's actual wrap, captured at export so the render matches
}

// ---- Image sticker (user-uploaded PNG — a logo, watermark, badge…) ----

export const MAX_IMAGE_STICKER_BYTES = 2 * 1024 * 1024;  // 2 MB upload cap
export const IMAGE_STICKER_MIN_WPCT  = 5;    // width as % of video width
export const IMAGE_STICKER_MAX_WPCT  = 60;

export interface ImageStickerConfig {
  r2_key: string;     // stickers/<user>/<uuid>.png — the worker downloads this
  url: string;        // browser preview URL (presigned, may expire; export uses r2_key)
  x_pct: number;      // 0–100, horizontal center
  y_pct: number;      // 0–100, vertical center
  width_pct: number;  // rendered width as % of video width
}

// Vetted high-contrast sticker color pairs. Preset-only by design — guarantees the
// sticker stays legible on busy video frames (no white-on-white). Blue/red use deep
// shades; the bright mid versions drop below comfortable contrast over footage.
export interface StickerColorPreset { id: string; label: string; bg: string; fg: string }
export const STICKER_COLOR_PRESETS: StickerColorPreset[] = [
  { id: "classic", label: "Classic", bg: "#ffffff", fg: "#000000" },
  { id: "inverse", label: "Inverse", bg: "#000000", fg: "#ffffff" },
  { id: "caution", label: "Caution", bg: "#ffd400", fg: "#000000" },
  { id: "cobalt",  label: "Cobalt",  bg: "#1d4ed8", fg: "#ffffff" },
  { id: "alert",   label: "Alert",   bg: "#dc2626", fg: "#ffffff" },
  { id: "green",   label: "Green",   bg: "#16a34a", fg: "#000000" },
];

export const DEFAULT_TEXT_STICKER_CONFIG: TextStickerConfig = {
  text: "",
  x_pct: 50,
  y_pct: 50,
  size_pct: 8,
  scale: 1,
  bg: "#ffffff",
  fg: "#000000",
};
