import { NextRequest, NextResponse } from "next/server";
import { auth } from "@/lib/auth";
import { createAdminClient } from "@/lib/supabase/server";

const OLLAMA_BASE = process.env.OLLAMA_BASE_URL ?? "http://127.0.0.1:11434";
const MODEL = process.env.OLLAMA_MODEL ?? "qwen3:4b";

function cleanOptions(value: unknown): string[] {
  if (!Array.isArray(value)) return [];

  return value
    .filter((item): item is string => typeof item === "string")
    .map((item) => item.trim())
    .filter(Boolean)
    .slice(0, 3);
}

export async function POST(
  _req: NextRequest,
  { params }: { params: Promise<{ id: string }> },
) {
  const { id } = await params;
  const { userId } = await auth();

  if (!userId) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const db = createAdminClient();

  const { data: clip } = await db
    .from("clips")
    .select("id, video_id, hook, text, tone, duration_sec")
    .eq("id", id)
    .eq("user_id", userId)
    .single();

  if (!clip) {
    return NextResponse.json({ error: "Clip not found" }, { status: 404 });
  }

  const { data: video } = await db
    .from("videos")
    .select("title, channel_name")
    .eq("id", clip.video_id)
    .single();

  const prompt = `Create a PERSONALIZED publishing pack for this short-form video Moment.

SOURCE VIDEO:
Title: ${video?.title ?? "Unknown"}
Channel: ${video?.channel_name ?? "Unknown"}

MOMENT:
Hook: ${clip.hook ?? ""}

Transcript:
${(clip.text ?? "").slice(0, 3000)}

Tone: ${clip.tone ?? "conversational"}
Duration: ${Math.round(clip.duration_sec ?? 60)} seconds

Return strict JSON only:
{
  "titles": ["...", "...", "..."],
  "descriptions": ["...", "...", "..."],
  "overlays": ["...", "...", "..."]
}

IMPORTANT:
- Identify the actual person or people involved using the source video title and Moment transcript.
- If the person's name is clearly known, USE THEIR NAME.
- Make the copy feel specifically about THIS person and THIS Moment.
- Never replace a known person's name with generic phrases like "he", "she", "this rapper", "this celebrity", or "this person" when the name fits naturally.
- The SOURCE VIDEO title may help identify WHO the person is, but it must NOT be used to invent claims about the Moment.
- Every factual claim in the title, description, and overlay must be directly supported by the Moment transcript.
- Never invent ages, dates, numbers, relationships, feuds, sexuality, crimes, quotes, events, or facts that are not explicitly stated in the Moment.
- Do not turn a casual statement into a dramatic "confession" or "admission" unless the transcript genuinely supports that framing.
- Keep factual wording close to the transcript. Do not strengthen "growing food" into "farming", "rarely travelling" into "no travel", or similar stronger claims.
- The CLICKBAIT should come from curiosity, phrasing, and what is emphasized, not from exaggerating the underlying facts.
- Prefer truthful high-interest verbs such as "reveals", "opens up", "explains", "reacts", "shares", "talks about", or "gets real about".

TITLES:
- Exactly 3.
- Personal and clickbait-style, while remaining truthful.
- Prefer the person's name near the beginning.
- Tell the viewer what revelation, reaction, argument, confession, opinion, or story makes the Moment interesting.
- Maximum 70 characters.
- No hashtags.
Examples of STYLE:
"Central Cee Reveals What He Really Wants From Life"
"Central Cee Opens Up About Having Kids"
"Cardi B Gets Heated While Talking About Nicki Minaj"

DESCRIPTIONS:
- Exactly 3.
- Explain the CONTEXT of what is actually happening in the Moment.
- Say who is speaking and what they are discussing.
- 1-3 natural sentences.
- Do not write vague motivational summaries.
- Add 3-5 relevant hashtags after the context.

OVERLAYS:
- Exactly 3.
- Personal, punchy hook text designed to stop scrolling.
- Use the person's name when known.
- Can use curiosity/clickbait framing, but cannot make a false claim.
- Roughly 4-10 words.
Examples of STYLE:
"Central Cee Opens Up About Having Kids"
"Central Cee Gets Real About His Future"
"Cardi B's Heated Nicki Minaj Moment"

Make all three options in each category meaningfully different.`;

  try {
    const resp = await fetch(`${OLLAMA_BASE}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: MODEL,
        stream: false,
        think: false,
        format: "json",
        messages: [{ role: "user", content: prompt }],
        options: {
          temperature: 0.7,
          num_predict: 700,
          num_ctx: 8192,
        },
      }),
    });

    if (!resp.ok) {
      throw new Error(`Ollama error ${resp.status}`);
    }

    const data = await resp.json();
    const raw = data?.message?.content;

    if (!raw) {
      throw new Error("Ollama returned an empty response");
    }

    const parsed = JSON.parse(raw);

    const titles = cleanOptions(parsed.titles);
    const descriptions = cleanOptions(parsed.descriptions);
    const overlays = cleanOptions(parsed.overlays);

    if (!titles.length || !descriptions.length || !overlays.length) {
      throw new Error("Ollama returned an incomplete publishing pack");
    }

    const title = titles[0];
    const description = descriptions[0];
    const overlay = overlays[0];

    await db
      .from("clips")
      .update({
        social_title: title,
        social_description: description,
        social_title_options: titles,
        social_description_options: descriptions,
        overlay_text_options: overlays,
        overlay_text: overlay,
      })
      .eq("id", id)
      .eq("user_id", userId);

    return NextResponse.json({
      title,
      description,
      overlay,
      titles,
      descriptions,
      overlays,
    });
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : "Publishing pack generation failed";
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
