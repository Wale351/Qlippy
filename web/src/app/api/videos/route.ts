import { NextRequest, NextResponse } from "next/server";
import { auth } from "@/lib/auth";
import { createAdminClient } from "@/lib/supabase/server";
import { triggerJob } from "@/lib/worker";
import { extractYouTubeId, fetchYouTubeMeta } from "@/lib/youtube";
// POST /api/videos — add a YouTube video + kick off transcription
export async function POST(req: NextRequest) {
  const { userId } = await auth();
  if (!userId) return NextResponse.json({ error: "Unauthorized" }, { status: 401 });

  const body = await req.json();
  const { youtube_url, model_size = "large", collection_id = null, resolution = 720 } = body;
  if (!youtube_url) return NextResponse.json({ error: "youtube_url required" }, { status: 400 });

  const youtubeId = extractYouTubeId(youtube_url);
  if (!youtubeId) return NextResponse.json({ error: "Invalid YouTube URL" }, { status: 400 });

  const db = createAdminClient();

  // Check duplicate
  const { data: existing } = await db
    .from("videos")
    .select("id")
    .eq("youtube_id", youtubeId)
    .eq("user_id", userId)
    .maybeSingle();

  if (existing) return NextResponse.json({ error: "Video already in library" }, { status: 409 });

  // Fetch metadata via oEmbed (no API key)
  const meta = await fetchYouTubeMeta(youtubeId).catch(() => ({
    title: `YouTube video ${youtubeId}`,
    thumbnail_url: `https://img.youtube.com/vi/${youtubeId}/maxresdefault.jpg`,
    channel_name: "",
  }));

  // Create video record
  const { data: video, error: videoErr } = await db
    .from("videos")
    .insert({
      user_id:       userId,
      youtube_url:   youtube_url.trim(),
      youtube_id:    youtubeId,
      title:         meta.title,
      thumbnail_url: meta.thumbnail_url,
      channel_name:  meta.channel_name,
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
        youtube_url: youtube_url.trim(),
        model_size:  model_size,
        resolution:  Number(resolution) || 720,
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
