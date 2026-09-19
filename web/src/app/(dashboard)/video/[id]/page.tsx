import { createAdminClient } from "@/lib/supabase/server";
import { auth } from "@/lib/auth";
import { notFound } from "next/navigation";
import { Clip } from "@/lib/types";
import { VideoChannelClient } from "@/components/video/video-channel-client";
import { freshenExports, freshenClipThumbnails, freshenPortraitPreview, freshUrl } from "@/lib/r2";

export default async function VideoPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  const { userId } = await auth();
  if (!userId) notFound();

  const db = createAdminClient();

  const { data: video } = await db
    .from("videos")
    .select("*")
    .eq("id", id)
    .eq("user_id", userId)
    .maybeSingle();

  if (!video) notFound();

  const [transcriptRes, clipsRes] = await Promise.all([
    db.from("transcripts")
      .select("id, language, full_text, model_size, created_at")
      .eq("video_id", id)
      .maybeSingle(),
    db.from("clips")
      .select("*")
      .eq("video_id", id)
      .order("generation", { ascending: false })
      .order("score", { ascending: false }),
  ]);

  const clipIds = (clipsRes.data ?? []).map((c: { id: string }) => c.id);

  const [doneExportsRes, pendingExportsRes] = clipIds.length > 0
    ? await Promise.all([
        db.from("exports")
          .select("id, clip_id, aspect_ratio, r2_key, r2_url, created_at")
          .in("clip_id", clipIds).eq("user_id", userId).eq("status", "done")
          .order("created_at", { ascending: false }),
        db.from("exports")
          .select("id, clip_id, updated_at")
          .in("clip_id", clipIds).eq("user_id", userId)
          .in("status", ["queued", "processing"]),
      ])
    : [{ data: [] }, { data: [] }];

  type ExportRow = { id: string; clip_id: string; aspect_ratio: string; r2_key: string | null; r2_url: string | null; created_at?: string };
  const exportsData = await freshenExports((doneExportsRes.data ?? []) as ExportRow[]);

  const clipExports: Record<string, { id: string; aspect_ratio: string; r2_url: string; created_at?: string }[]> = {};
  for (const exp of exportsData as ExportRow[]) {
    if (!clipExports[exp.clip_id]) clipExports[exp.clip_id] = [];
    clipExports[exp.clip_id].push({ id: exp.id, aspect_ratio: exp.aspect_ratio, r2_url: exp.r2_url ?? "", created_at: exp.created_at });
  }

  const pendingExports = (pendingExportsRes.data ?? []) as { id: string; clip_id: string; updated_at: string }[];

  const freshClips = await freshenClipThumbnails((clipsRes.data ?? []) as Clip[]);
  const freshVideo = await freshenPortraitPreview(video);

  // YouTube keeps its native embed. Other supported sources play from the
  // source copy Qlippy cached in R2 after import.
  const sourcePlaybackUrl =
    video.source_type !== "youtube" && video.source_r2_key
      ? await freshUrl(video.source_r2_key)
      : null;

  const playableVideo = {
    ...freshVideo,
    source_playback_url: sourcePlaybackUrl,
  };

  return (
    <VideoChannelClient
      video={playableVideo}
      transcript={transcriptRes.data}
      clips={freshClips}
      clipExports={clipExports}
      pendingExports={pendingExports}
    />
  );
}
