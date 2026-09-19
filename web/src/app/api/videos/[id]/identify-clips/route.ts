import { NextRequest, NextResponse } from "next/server";
import { auth } from "@/lib/auth";
import { createAdminClient } from "@/lib/supabase/server";
import { triggerJob } from "@/lib/worker";

// POST /api/videos/[id]/identify-clips — queue clip identification job
export async function POST(req: NextRequest, { params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  const { userId } = await auth();
  if (!userId) return NextResponse.json({ error: "Unauthorized" }, { status: 401 });

  // keep_existing: keep current clips + exports as a previous generation
  const body = await req.json().catch(() => ({}));
  const keepExisting = body?.keep_existing === true;

  const db = createAdminClient();

  // Verify video ownership + transcript exists
  const [{ data: video }, { data: transcript }] = await Promise.all([
    db.from("videos").select("id, status").eq("id", id).eq("user_id", userId).single(),
    db.from("transcripts").select("id").eq("video_id", id).maybeSingle(),
  ]);

  if (!video) return NextResponse.json({ error: "Video not found" }, { status: 404 });
  if (!transcript) {
    return NextResponse.json({ error: "Video must be transcribed before identifying clips" }, { status: 422 });
  }

  // Fetch user's niche profile (Stage 2) — used to tune preferences
  const { data: userProfile } = await db
    .from("user_profiles")
    .select("*")
    .eq("user_id", userId)
    .maybeSingle();

  // Default to the full hard duration gate (10-120s) when no preference is saved
  const preferences = {
    min_duration_seconds: userProfile?.clip_duration_min ?? 10,
    max_duration_seconds: userProfile?.clip_duration_max ?? 120,
    max_clips: 10,
    target_tones: userProfile?.preferred_tones?.length
      ? userProfile.preferred_tones
      : ["educational", "entertaining", "hook", "emotional", "cta"],
  };

  const jobPayload: Record<string, unknown> = { video_id: id, preferences, keep_existing: keepExisting };

  // Pass niche profile to worker if configured
  if (userProfile?.niche || userProfile?.audience) {
    jobPayload.niche_profile = {
      niche:              userProfile.niche,
      audience:           userProfile.audience,
      hook_style:         userProfile.hook_style,
      clip_duration_min:  userProfile.clip_duration_min,
      clip_duration_max:  userProfile.clip_duration_max,
      preferred_tones:    userProfile.preferred_tones,
    };
  }

  const { data: job } = await db
    .from("jobs")
    .insert({
      user_id:   userId,
      type:      "identify_clips",
      entity_id: id,
      payload:   jobPayload,
    })
    .select()
    .single();

  // Immediately expose re-analysis progress in the library UI.
  await db
    .from("videos")
    .update({
      status: "identifying",
      processing_stage: "preparing_analysis",
      error_msg: null,
    })
    .eq("id", id);

  triggerJob(job.id).catch(console.error);

  return NextResponse.json({ job_id: job.id });
}
