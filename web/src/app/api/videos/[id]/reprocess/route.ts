import { NextRequest, NextResponse } from "next/server";
import { auth } from "@/lib/auth";
import { createAdminClient } from "@/lib/supabase/server";
import { triggerJob } from "@/lib/worker";

// POST /api/videos/[id]/reprocess — force re-transcribe
export async function POST(req: NextRequest, { params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  const { userId } = await auth();
  if (!userId) return NextResponse.json({ error: "Unauthorized" }, { status: 401 });

  const { model_size = "large" } = await req.json().catch(() => ({}));

  const db = createAdminClient();

  const { data: video } = await db
    .from("videos")
    .select("id, youtube_url")
    .eq("id", id)
    .eq("user_id", userId)
    .single();

  if (!video) return NextResponse.json({ error: "Video not found" }, { status: 404 });

  // Mark video as pending again
  await db
    .from("videos")
    .update({
      status: "pending",
      processing_stage: "preparing",
      error_msg: null,
    })
    .eq("id", id);

  // Delete existing transcript
  await db.from("transcripts").delete().eq("video_id", id);

  // Create new transcription job
  const { data: job } = await db
    .from("jobs")
    .insert({
      user_id:   userId,
      type:      "transcribe",
      entity_id: id,
      payload:   {
        video_id:    id,
        youtube_url: video.youtube_url,
        model_size,
        reprocess:   true,
      },
    })
    .select()
    .single();

  triggerJob(job.id).catch(console.error);

  return NextResponse.json({ job_id: job.id });
}
