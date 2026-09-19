import { NextRequest, NextResponse } from "next/server";
import { auth } from "@/lib/auth";
import { createAdminClient } from "@/lib/supabase/server";

const OLLAMA_BASE =
  process.env.OLLAMA_BASE_URL ?? "http://127.0.0.1:11434";

const MODEL = process.env.OLLAMA_MODEL ?? "qwen3:4b";


function cleanOptions(value: unknown): string[] {
  if (!Array.isArray(value)) return [];

  return [...new Set(
    value
      .filter((item): item is string => typeof item === "string")
      .map((item) => item.trim())
      .filter(Boolean),
  )].slice(0, 3);
}

function parseJson(raw: string): Record<string, unknown> {
  let cleaned = raw
    .trim()
    .replace(/^```json\s*/i, "")
    .replace(/^```\s*/, "")
    .replace(/\s*```$/, "");

  const start = cleaned.indexOf("{");
  const end = cleaned.lastIndexOf("}");

  if (start < 0 || end <= start) {
    throw new Error("Ollama returned invalid JSON");
  }

  cleaned = cleaned.slice(start, end + 1);
  cleaned = cleaned.replace(/,\s*([}\]])/g, "$1");

  return JSON.parse(cleaned);
}

async function askOllama(
  prompt: string,
  temperature = 0.3,
  numPredict = 700,
): Promise<Record<string, unknown>> {
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
        temperature,
        num_predict: numPredict,
        num_ctx: 8192,
      },
    }),
  });

  if (!resp.ok) {
    throw new Error(`Ollama error ${resp.status}`);
  }

  const data = await resp.json();
  const raw = String(data?.message?.content ?? "");

  if (!raw.trim()) {
    throw new Error("Ollama returned an empty response");
  }

  return parseJson(raw);
}

async function extractParticipants(sourceTitle: string): Promise<string[]> {
  if (!sourceTitle.trim()) return [];

  try {
    const parsed = await askOllama(
      `Extract ONLY the actual hosts, guests, interviewees, or featured speakers
whose names are explicitly present in this video title.

VIDEO TITLE:
${sourceTitle}

Return JSON only:
{"participants":["Name 1","Name 2"]}

RULES:
- Copy names exactly from the title.
- Never invent a name.
- Do not include brands, shows, locations, topics, or ordinary words.
- Do not add people who are merely referenced as a topic.
- If uncertain, omit the name.`,
      0.1,
      150,
    );

    if (!Array.isArray(parsed.participants)) return [];

    return [...new Set(
      parsed.participants
        .filter((item): item is string => typeof item === "string")
        .map((item) => item.trim())
        .filter(Boolean),
    )].slice(0, 4);
  } catch {
    return [];
  }
}

function normalizeForMatch(value: string): string {
  return value
    .toLowerCase()
    .replace(/[’‘]/g, "'")
    .replace(/\s+/g, " ")
    .trim();
}

async function extractEvidence(
  transcript: string,
): Promise<string[]> {
  const parsed = await askOllama(
    `Select the strongest useful excerpts from this short-form clip transcript.

TRANSCRIPT:
${transcript}

Return JSON only:
{
  "quotes": [
    "exact words copied from the transcript",
    "another exact excerpt"
  ]
}

RULES:
- Return 3-6 excerpts.
- Every excerpt MUST be copied verbatim from the transcript.
- Do not paraphrase.
- Do not summarize.
- Do not identify speakers.
- Do not explain what the quote means.
- Prefer excerpts that capture the main conversation, opinion, joke, question, story, or reaction.
- Keep each excerpt reasonably short.`,
    0.1,
    650,
  );

  if (!Array.isArray(parsed.quotes)) return [];

  const normalizedTranscript = normalizeForMatch(transcript);

  return [
    ...new Set(
      parsed.quotes
        .filter((item): item is string => typeof item === "string")
        .map((item) => item.trim())
        .filter(Boolean)
        .filter((quote) =>
          normalizedTranscript.includes(
            normalizeForMatch(quote),
          ),
        ),
    ),
  ].slice(0, 6);
}

function startsWithMentionedPerson(
  value: string,
  mentionedPeople: string[],
): boolean {
  const lower = value.trim().toLowerCase();

  return mentionedPeople.some((name) => {
    const n = name.toLowerCase();

    return (
      lower.startsWith(n + " ") ||
      lower.startsWith(n + "'s ") ||
      lower === n
    );
  });
}

export async function POST(
  _req: NextRequest,
  { params }: { params: Promise<{ id: string }> },
) {
  const { id } = await params;
  const { userId } = await auth();

  if (!userId) {
    return NextResponse.json(
      { error: "Unauthorized" },
      { status: 401 },
    );
  }

  const db = createAdminClient();

  const { data: clip } = await db
    .from("clips")
    .select("id, video_id, text, tone, duration_sec")
    .eq("id", id)
    .eq("user_id", userId)
    .single();

  if (!clip) {
    return NextResponse.json(
      { error: "Clip not found" },
      { status: 404 },
    );
  }

  const { data: video } = await db
    .from("videos")
    .select("title")
    .eq("id", clip.video_id)
    .single();

  const transcript = (clip.text ?? "").slice(0, 6000);

  if (!transcript.trim()) {
    return NextResponse.json(
      { error: "This clip has no transcript" },
      { status: 422 },
    );
  }

  try {
    // The source title is used ONLY to identify likely participants.
    const participants = await extractParticipants(
      video?.title ?? "",
    );

    // Everything about WHAT HAPPENS comes from the transcript.
    const evidence = await extractEvidence(
      transcript,
    );


    if (evidence.length < 2) {
      throw new Error(
        "Not enough grounded evidence found",
      );
    }

    const subjectLabel =
      participants.length === 1
        ? participants[0]
        : participants.length > 1
          ? participants.slice(0, 2).join(" and ")
          : "";

    const evidenceText = evidence
      .map(
        (quote, index) =>
          `EVIDENCE ${index + 1}: "${quote}"`,
      )
      .join("\n");

    const prompt = `
CONFIRMED PARTICIPANTS:
${participants.length ? participants.join(", ") : "Unknown"}

SAFE SUBJECT LABEL:
${subjectLabel || "Unknown"}

VERIFIED TRANSCRIPT EVIDENCE:
${evidenceText}

Create a Publishing Pack using ONLY the verified evidence above.

Return JSON only:
{
  "titles": ["...", "...", "..."],
  "descriptions": ["...", "...", "..."],
  "overlays": ["...", "...", "..."]
}

GROUNDING:
- Every claim must be directly supported by VERIFIED TRANSCRIPT EVIDENCE.
- Do not add information from outside the evidence.
- Do not invent motives, emotions, causes, relationships, events, identities, or context.
- Do not identify a mentioned person as a speaker.
- Avoid dramatic words such as "frantic", "desperate", "chaotic", "demands", "reveals", "admits", "heated", "shocking", or "secret" unless the evidence literally supports them.
- Never invent a quote.

${
  participants.length > 1
    ? `There are multiple confirmed participants.
Use "${subjectLabel}" collectively when identifying the conversation.
Do NOT guess which individual participant said a line.
Prefer wording such as:
"${subjectLabel} talk about..."
"${subjectLabel} joke about..."
"${subjectLabel} discuss..."
"${subjectLabel} react to..."
"${subjectLabel} get into..."
Do not write "Drake says..." or "Ben says..." unless the evidence itself explicitly identifies that speaker.`
    : participants.length === 1
      ? `The confirmed participant is ${subjectLabel}.`
      : `Participant identity is uncertain. Use neutral wording.`
}

TITLES:
- Exactly 3.
- Maximum 70 characters.
- No hashtags.
- No emojis.
${
  subjectLabel
    ? `- Every title MUST begin with "${subjectLabel}".`
    : "- Use neutral wording."
}
- Describe the actual subject of the evidence.
- Each option must take a genuinely different angle.

DESCRIPTIONS:
- Exactly 3.
- 1-2 short natural sentences.
- End with 2-4 relevant hashtags.
${
  subjectLabel
    ? `- Every description MUST begin with "${subjectLabel}".`
    : "- Use neutral wording."
}
- Explain only what the verified evidence supports.
- Sound like natural social-media copy, not a report.

OVERLAYS:
- Exactly 3.
- 3-7 words.
- No hashtags.
- No emojis.
- Do not use participant names.
- Prefer wording close to the transcript.
- Make each option clearly different.
`;

    let parsed: Record<string, unknown> | null = null;

    for (let attempt = 0; attempt < 2; attempt++) {
      parsed = await askOllama(
        attempt === 0
          ? prompt
          : `${prompt}

RETRY:
The previous answer failed validation.
Stay strictly inside the evidence.
Return one valid JSON object only.`,
        attempt === 0 ? 0.35 : 0.1,
        900,
      );

      const titles = cleanOptions(parsed.titles);
      const descriptions = cleanOptions(
        parsed.descriptions,
      );
      const overlays = cleanOptions(parsed.overlays);

      const requiredOpening =
        subjectLabel.toLowerCase();

      const titleValidation =
        titles.length === 3 &&
        (
          !requiredOpening ||
          titles.every((value) =>
            value.toLowerCase().startsWith(requiredOpening),
          )
        );

      const descriptionValidation =
        descriptions.length === 3 &&
        (
          !requiredOpening ||
          descriptions.every((value) =>
            value.toLowerCase().startsWith(requiredOpening),
          )
        );

      const overlayValidation =
        overlays.length === 3 &&
        overlays.every((overlay) => {
          const lower = overlay.toLowerCase();

          return !participants.some((participant) =>
            lower.includes(participant.toLowerCase()),
          );
        });

      if (
        !titleValidation ||
        !descriptionValidation ||
        !overlayValidation
      ) {

        continue;
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
        participants,
        evidence,
      });
    }

    throw new Error(
      "Publishing Pack failed evidence validation",
    );
  } catch (err: unknown) {
    console.error("Publishing Pack:", err);

    return NextResponse.json(
      {
        error:
          "Publishing Pack could not produce grounded copy. Please regenerate.",
      },
      { status: 500 },
    );
  }
}
