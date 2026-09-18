import { NextRequest, NextResponse } from "next/server";
import { auth } from "@/lib/auth";
import { createAdminClient } from "@/lib/supabase/server";

type CutInput = {
  start: number;
  end: number;
};

// PATCH /api/clips/[id] — manually edit an existing Moment
export async function PATCH(
  req: NextRequest,
  { params }: { params: Promise<{ id: string }> },
) {
  const { id } = await params;
  const { userId } = await auth();

  if (!userId) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const body = await req.json();
  const db = createAdminClient();

  const { data: existing } = await db
    .from("clips")
    .select("id, video_id, chunks")
    .eq("id", id)
    .eq("user_id", userId)
    .maybeSingle();

  if (!existing) {
    return NextResponse.json({ error: "Moment not found" }, { status: 404 });
  }

  const update: Record<string, unknown> = {};

  if ("hook" in body) {
    update.hook = typeof body.hook === "string" ? body.hook.trim() : null;
  }

  if ("text" in body) {
    update.text = typeof body.text === "string" ? body.text.trim() : null;
  }

  if ("notes" in body) {
    update.notes = typeof body.notes === "string" ? body.notes.trim() : null;
  }

  if ("cuts" in body) {
    if (!Array.isArray(body.cuts) || body.cuts.length === 0) {
      return NextResponse.json(
        { error: "At least one cut is required" },
        { status: 400 },
      );
    }

    const cuts: CutInput[] = body.cuts.map((cut: unknown) => {
      if (
        !cut ||
        typeof cut !== "object" ||
        !("start" in cut) ||
        !("end" in cut)
      ) {
        throw new Error("Invalid cut");
      }

      const start = Number((cut as { start: unknown }).start);
      const end = Number((cut as { end: unknown }).end);

      if (
        !Number.isFinite(start) ||
        !Number.isFinite(end) ||
        start < 0 ||
        end <= start
      ) {
        throw new Error("Invalid cut");
      }

      return { start, end };
    });

    update.cuts = cuts;
    update.duration_sec =
      Math.round(
        cuts.reduce((sum, cut) => sum + (cut.end - cut.start), 0) * 1000,
      ) / 1000;

    // Keep stored scene metadata aligned with the manually edited cuts.
    const oldChunks = Array.isArray(existing.chunks) ? existing.chunks : [];
    update.chunks = cuts.map((cut, index) => ({
      start_ms: Math.round(cut.start * 1000),
      end_ms: Math.round(cut.end * 1000),
      crosstalk_flagged:
        typeof oldChunks[index]?.crosstalk_flagged === "boolean"
          ? oldChunks[index].crosstalk_flagged
          : false,
    }));
    update.stitched = cuts.length > 1;

    // Rebuild the visible Moment transcript from the original timestamped
    // transcript so captions/text follow the edited scene boundaries.
    const { data: transcript } = await db
      .from("transcripts")
      .select("raw_words, segments")
      .eq("video_id", existing.video_id)
      .maybeSingle();

    if (transcript) {
      const rawWords = Array.isArray(transcript.raw_words)
        ? transcript.raw_words
        : [];

      const segments = Array.isArray(transcript.segments)
        ? transcript.segments
        : [];

      const parts: string[] = [];

      for (const cut of cuts) {
        const wordsInCut = rawWords.filter((word: any) => {
          const start = Number(word?.start);
          const end = Number(word?.end);

          if (!Number.isFinite(start) || !Number.isFinite(end)) return false;

          const midpoint = (start + end) / 2;
          return midpoint >= cut.start && midpoint <= cut.end;
        });

        if (wordsInCut.length > 0) {
          parts.push(
            wordsInCut
              .map((word: any) => String(word?.word ?? word?.text ?? ""))
              .join(" "),
          );
          continue;
        }

        const segmentsInCut = segments.filter((segment: any) => {
          const start = Number(segment?.start);
          const end = Number(segment?.end);

          if (!Number.isFinite(start) || !Number.isFinite(end)) return false;

          return end > cut.start && start < cut.end;
        });

        if (segmentsInCut.length > 0) {
          parts.push(
            segmentsInCut
              .map((segment: any) => String(segment?.text ?? ""))
              .join(" "),
          );
        }
      }

      const rebuiltText = parts
        .join(" ")
        .replace(/\s+/g, " ")
        .replace(/\s+([,.!?;:])/g, "$1")
        .trim();

      if (rebuiltText) {
        update.text = rebuiltText;
      }
    }
  }

  if (Object.keys(update).length === 0) {
    return NextResponse.json(
      { error: "No editable fields supplied" },
      { status: 400 },
    );
  }

  // Existing publishing copy may no longer describe the edited Moment.
  if ("cuts" in body || "hook" in body || "text" in body) {
    update.social_title = null;
    update.social_description = null;
    update.social_title_options = [];
    update.social_description_options = [];
    update.overlay_text = null;
    update.overlay_text_options = [];
  }

  const { data, error } = await db
    .from("clips")
    .update(update)
    .eq("id", id)
    .eq("user_id", userId)
    .select()
    .single();

  if (error) {
    return NextResponse.json({ error: error.message }, { status: 500 });
  }

  return NextResponse.json(data);
}
