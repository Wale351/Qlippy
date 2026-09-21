import { NextRequest, NextResponse } from "next/server";
import { auth } from "@/lib/auth";
import { createAdminClient } from "@/lib/supabase/server";
import { triggerJob } from "@/lib/worker";
// POST /api/videos — add a supported video URL + kick off transcription
export async function POST(req: NextRequest) {
  const { userId } = await auth();
  if (!userId) return NextResponse.json({ error: "Unauthorized" }, { status: 401 });

  const body = await req.json();
  const {
    model_size = "large",
    collection_id = null,
    resolution = 720,
  } = body;

  const rawClipDuration = body.clip_duration_seconds;
  const clipDurationSeconds =
    rawClipDuration === null ||
    rawClipDuration === undefined ||
    rawClipDuration === ""
      ? null
      : Number(rawClipDuration);

  if (
    clipDurationSeconds !== null &&
    ![15, 30, 60, 90].includes(clipDurationSeconds)
  ) {
    return NextResponse.json(
      { error: "Invalid clip duration" },
      { status: 400 },
    );
  }

  // URL imports and direct file uploads share the same processing pipeline.
  const sourceUrl = String(
    body.video_url || body.source_url || body.youtube_url || ""
  ).trim();

  const sourceR2Key = String(body.source_r2_key || "").trim();
  const isUpload = Boolean(sourceR2Key);

  if (!sourceUrl && !isUpload) {
    return NextResponse.json(
      { error: "video_url or source_r2_key required" },
      { status: 400 },
    );
  }

  if (sourceUrl) {
    try {
      const parsed = new URL(sourceUrl);
      if (parsed.protocol !== "http:" && parsed.protocol !== "https:") {
        throw new Error();
      }
    } catch {
      return NextResponse.json({ error: "Invalid video URL" }, { status: 400 });
    }
  }

  // Never allow a client to attach another user's uploaded R2 object.
  if (
    isUpload &&
    !sourceR2Key.startsWith(`sources/uploads/${userId}/`)
  ) {
    return NextResponse.json(
      { error: "Invalid uploaded video key" },
      { status: 400 },
    );
  }

  const sourceType = isUpload
    ? "upload"
    : String(body.source_type || "other").toLowerCase();

  // yt-dlp normally provides an ID. Uploaded files use their unique R2 key.
  const sourceId = String(
    body.source_id || (isUpload ? sourceR2Key : sourceUrl)
  );

  const title =
    String(body.title || "").trim() ||
    (isUpload ? "Uploaded video" : `Imported video (${sourceType})`);

  const thumbnailUrl =
    body.thumbnail_url || body.thumbnail || null;

  const channelName =
    String(body.channel_name || "").trim() || null;

  const rawDuration = Number(body.duration);
  const durationSec =
    Number.isFinite(rawDuration) && rawDuration > 0
      ? Math.round(rawDuration)
      : null;

  const db = createAdminClient();

  // Check duplicate across all supported source types.
  const { data: existing } = await db
    .from("videos")
    .select("id")
    .eq("source_type", sourceType)
    .eq("source_id", sourceId)
    .eq("user_id", userId)
    .maybeSingle();

  if (existing) {
    return NextResponse.json(
      { error: "Video already in library" },
      { status: 409 }
    );
  }

  // Create video record.
  // youtube_url remains populated as a temporary compatibility alias because
  // the current worker still reads that legacy field for yt-dlp downloads.
  const { data: video, error: videoErr } = await db
    .from("videos")
    .insert({
      user_id:       userId,
      source_url:    sourceUrl || null,
      source_type:   sourceType,
      source_id:     sourceId,
      source_r2_key: sourceR2Key || null,

      // Legacy compatibility fields remain populated for URL imports only.
      youtube_url:   sourceUrl || null,
      youtube_id:    sourceType === "youtube" ? sourceId : null,

      title,
      thumbnail_url: thumbnailUrl,
      duration_sec:  durationSec,
      channel_name:  channelName,
      status:        "pending",
      collection_id: collection_id || null,
    })
    .select()
    .single();

  if (videoErr) return NextResponse.json({ error: videoErr.message }, { status: 500 });

  // Start Replicate WhisperX prediction
  // We use a webhook so Replicate calls back when done
  // Audio will be downloaded by worker from the source video
  // For now: create transcription job, worker downloads video + extracts audio + uploads to R2,
  // then starts Replicate with the R2 audio URL.
  const { data: job } = await db
    .from("jobs")
    .insert({
      user_id:   userId,
      type:      "transcribe",
      entity_id: video.id,
      payload:   {
        video_id:    video.id,

        // Legacy worker compatibility + new generic source fields.
        youtube_url: sourceUrl || null,
        source_url:  sourceUrl || null,
        source_r2_key: sourceR2Key || null,
        source_type: sourceType,
        source_id:   sourceId,

        model_size:  model_size,
        resolution:  Number(resolution) || 720,
        clip_duration_seconds: clipDurationSeconds,
      },
    })
    .select()
    .single();

  // Fire-and-forget trigger to worker
  triggerJob(job.id).catch(console.error);

  return NextResponse.json({ video_id: video.id }, { status: 201 });
}

// GET /api/videos — list user's videos
export async function GET() {
  const { userId } = await auth();
  if (!userId) return NextResponse.json({ error: "Unauthorized" }, { status: 401 });

  const db = createAdminClient();
  const [{ data, error }, { data: transcriptRows }] = await Promise.all([
    db.from("videos").select("*").eq("user_id", userId).order("created_at", { ascending: false }),
    db.from("transcripts").select("video_id").eq("user_id", userId),
  ]);

  if (error) return NextResponse.json({ error: error.message }, { status: 500 });

  const withTranscript = new Set((transcriptRows ?? []).map((r: { video_id: string }) => r.video_id));
  return NextResponse.json(
    (data ?? []).map((v: { id: string }) => ({ ...v, has_transcript: withTranscript.has(v.id) })),
  );
}
