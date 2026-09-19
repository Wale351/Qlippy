import { NextRequest, NextResponse } from "next/server";
import { auth } from "@/lib/auth";
import { createAdminClient } from "@/lib/supabase/server";
import { triggerJob } from "@/lib/worker";
import { cutsHash, FACE_TRACK_VERSION } from "@/lib/cuts-hash";
import { TEXT_STICKER_MAX_CHARS } from "@/lib/types";
import { clampImageSticker } from "@/lib/image-sticker";
// POST /api/clips/[id]/export
export async function POST(req: NextRequest, { params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  const { userId } = await auth();
  if (!userId) return NextResponse.json({ error: "Unauthorized" }, { status: 401 });

  const {
    aspect_ratio       = "9:16",
    fit_mode           = "fill",
    preset_id          = null,
    preset_config      = null,
    text_sticker_config = null,
    image_sticker_config = null,
  } = await req.json();

  // Cap sticker text server-side; drop it if empty. `lines` (the browser-captured
  // wrap) is sanitised to short string entries so it can't blow up the render canvas.
  const stickerLines = Array.isArray(text_sticker_config?.lines)
    ? text_sticker_config.lines
        .filter((l: unknown): l is string => typeof l === "string" && l.trim().length > 0)
        .slice(0, 8)
        .map((l: string) => l.slice(0, TEXT_STICKER_MAX_CHARS))
    : undefined;
  const sticker =
    text_sticker_config && typeof text_sticker_config.text === "string" && text_sticker_config.text.trim()
      ? { ...text_sticker_config, text: text_sticker_config.text.slice(0, TEXT_STICKER_MAX_CHARS), lines: stickerLines }
      : null;

  // Image sticker: only keys under the caller's own prefix, placement clamped.
  const imgSticker =
    image_sticker_config
    && typeof image_sticker_config.r2_key === "string"
    && image_sticker_config.r2_key.startsWith(`stickers/${userId}/`)
      ? clampImageSticker(image_sticker_config)
      : null;

  const db = createAdminClient();

  const { data: clip } = await db
    .from("clips")
    .select("id, video_id, hook, cuts")
    .eq("id", id)
    .eq("user_id", userId)
    .maybeSingle();

  if (!clip) return NextResponse.json({ error: "Clip not found" }, { status: 404 });

  // Resolve the cached face-tracked portrait ("edit") for this exact cut +
  // aspect ratio. A hit means we skip face-tracking and restyle for free.
  const cuts_hash = cutsHash(clip.cuts, FACE_TRACK_VERSION);
  const { data: existingEdit } = await db
    .from("clip_edits")
    .select("id, status, r2_key")
    .eq("clip_id", clip.id)
    .eq("aspect_ratio", aspect_ratio)
    .eq("fit_mode", fit_mode)
    .eq("cuts_hash", cuts_hash)
    .maybeSingle();

  let editId: string;
  let jobType: "edit_clip" | "style_clip";

  if (existingEdit?.status === "done" && existingEdit.r2_key) {
    // Cache hit — the portrait already exists. Style only.
    editId = existingEdit.id;
    jobType = "style_clip";
  } else if (existingEdit) {
    // A build is already in flight (queued/processing) or previously errored —
    // reuse the row; our edit_clip job (re)builds + chains our style.
    editId = existingEdit.id;
    jobType = "edit_clip";
  } else {
    // Cache miss — claim the edit atomically (on-conflict-do-nothing so two
    // simultaneous exports of the same cut produce exactly one edit row).
    const { data: claimed } = await db
      .from("clip_edits")
      .upsert(
        { clip_id: clip.id, user_id: userId, aspect_ratio, fit_mode, cuts_hash, status: "queued" },
        { onConflict: "clip_id,aspect_ratio,fit_mode,cuts_hash", ignoreDuplicates: true },
      )
      .select("id")
      .maybeSingle();
    if (claimed) {
      editId = claimed.id;
    } else {
      // Lost the insert race — reuse the row the concurrent request just created.
      const { data: raced } = await db
        .from("clip_edits")
        .select("id")
        .eq("clip_id", clip.id)
        .eq("aspect_ratio", aspect_ratio)
        .eq("cuts_hash", cuts_hash)
        .single();
      editId = raced!.id;
    }
    jobType = "edit_clip";
  }

  // Resolve preset name for the jobs page display
  let preset_name: string | null = null;
  if (preset_id) {
    const { data: preset } = await db
      .from("presets")
      .select("name")
      .eq("id", preset_id)
      .maybeSingle();
    preset_name = preset?.name ?? null;
  }

  const { data: exportRecord, error: exportErr } = await db
    .from("exports")
    .insert({
      clip_id:      clip.id,
      user_id:      userId,
      aspect_ratio,
      fit_mode,
      preset_id:    preset_id || null,
      edit_id:      editId,
      status:       "queued",
    })
    .select()
    .single();

  if (exportErr || !exportRecord) {
    return NextResponse.json({ error: exportErr?.message ?? "Failed to create export" }, { status: 500 });
  }

  const { data: job, error: jobErr } = await db
    .from("jobs")
    .insert({
      user_id:   userId,
      type:      jobType,
      entity_id: exportRecord.id,
      payload: {
        export_id:     exportRecord.id,
        edit_id:       editId,
        clip_id:       clip.id,
        video_id:      clip.video_id,
        aspect_ratio,
        fit_mode,
        preset_id:     preset_id     || null,
        preset_config:    preset_config    || null,
        text_sticker_config: sticker,
        image_sticker_config: imgSticker,
        branding_watermark: false,
        clip_hook:        clip.hook        || null,
        preset_name:      preset_name      || null,
      },
    })
    .select()
    .single();

  if (jobErr || !job) {
    return NextResponse.json({ error: jobErr?.message ?? "Failed to create job" }, { status: 500 });
  }

  triggerJob(job.id).catch(console.error);

  return NextResponse.json({ export_id: exportRecord.id, job_id: job.id });
}
