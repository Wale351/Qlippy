import { NextRequest, NextResponse } from "next/server";
import { auth } from "@/lib/auth";
import { spawn } from "child_process";

type FormatRow = {
  format_id?: string;
  ext?: string;
  height?: number | null;
  filesize?: number | null;
  filesize_approx?: number | null;
  vcodec?: string;
  acodec?: string;
  tbr?: number | null;
};

function runYtDlp(url: string): Promise<any> {
  return new Promise((resolve, reject) => {
    const proc = spawn(process.env.CLIPPY_PYTHON || "../worker/.venv/bin/python", [
      "-m",
      "yt_dlp",
      "-J",
      "--no-playlist",
      "--no-warnings",
      "--js-runtimes",
      "node",
      "--remote-components",
      "ejs:github",
      ...(process.platform === "darwin"
        ? ["--cookies-from-browser", process.env.YT_COOKIES_FROM_BROWSER || "safari"]
        : []),
      url,
    ]);

    let stdout = "";
    let stderr = "";

    proc.stdout.on("data", (d) => (stdout += d.toString()));
    proc.stderr.on("data", (d) => (stderr += d.toString()));

    proc.on("close", (code) => {
      if (code !== 0) {
        reject(new Error(stderr || `yt-dlp exited with code ${code}`));
        return;
      }

      try {
        resolve(JSON.parse(stdout));
      } catch {
        reject(new Error("Could not parse yt-dlp metadata"));
      }
    });
  });
}

export async function POST(req: NextRequest) {
  const { userId } = await auth();
  if (!userId) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const body = await req.json();
  const video_url = String(
    body.video_url || body.source_url || body.youtube_url || ""
  ).trim();

  if (!video_url) {
    return NextResponse.json({ error: "video_url required" }, { status: 400 });
  }

  try {
    const info = await runYtDlp(video_url);
    const formats: FormatRow[] = Array.isArray(info.formats) ? info.formats : [];

    const heights = [360, 480, 720, 1080];

    const resolutions = heights
      .map((height) => {
        const candidates = formats.filter(
          (f) =>
            typeof f.height === "number" &&
            f.height <= height &&
            f.vcodec &&
            f.vcodec !== "none"
        );

        if (!candidates.length) return null;

        const best = candidates.sort(
          (a, b) => (b.height || 0) - (a.height || 0)
        )[0];

        const duration = Number(info.duration) || 0;
        const videoSize =
          best.filesize ??
          best.filesize_approx ??
          (best.tbr && duration
            ? Math.round((best.tbr * 1000 * duration) / 8)
            : null);

        const audioCandidates = formats.filter(
          (f) =>
            f.acodec &&
            f.acodec !== "none" &&
            (!f.vcodec || f.vcodec === "none")
        );

        const bestAudio = audioCandidates
          .filter((f) => (f.ext || "").includes("m4a") || f.ext === "mp4")
          .sort(
            (a, b) =>
              (b.filesize ?? b.filesize_approx ?? 0) -
              (a.filesize ?? a.filesize_approx ?? 0)
          )[0];

        const audioSize =
          bestAudio?.filesize ??
          bestAudio?.filesize_approx ??
          (bestAudio?.tbr && duration
            ? Math.round((bestAudio.tbr * 1000 * duration) / 8)
            : null);

        const estimated_size =
          typeof videoSize === "number"
            ? videoSize + (typeof audioSize === "number" ? audioSize : 0)
            : null;

        return {
          height: best.height,
          requested_height: height,
          estimated_size,
        };
      })
      .filter(Boolean);

    const extractor = String(
      info.extractor_key || info.extractor || ""
    ).toLowerCase();

    let source_type = "other";
    if (extractor.includes("youtube")) source_type = "youtube";
    else if (extractor.includes("twitter") || extractor === "x") source_type = "twitter";
    else if (extractor.includes("tiktok")) source_type = "tiktok";
    else if (extractor.includes("instagram")) source_type = "instagram";
    else if (extractor.includes("vimeo")) source_type = "vimeo";
    else if (extractor.includes("generic")) source_type = "direct";

    return NextResponse.json({
      title: info.title ?? "",
      duration: info.duration ?? null,
      thumbnail: info.thumbnail ?? null,
      channel_name: info.channel ?? info.uploader ?? info.creator ?? "",
      source_type,
      source_id: info.id ? String(info.id) : null,
      source_url: info.webpage_url ?? video_url,
      resolutions,
    });
  } catch (err: unknown) {
    return NextResponse.json(
      { error: err instanceof Error ? err.message : "Failed to inspect video" },
      { status: 500 }
    );
  }
}
