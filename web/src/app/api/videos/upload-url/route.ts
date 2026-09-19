import { NextRequest, NextResponse } from "next/server";
import { auth } from "@/lib/auth";
import { presignedPutUrl } from "@/lib/r2";

const MAX_UPLOAD_BYTES = 10 * 1024 * 1024 * 1024; // 10 GB

export async function POST(req: NextRequest) {
  const { userId } = await auth();

  if (!userId) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const body = await req.json();

  const filename = String(body.filename || "").trim();
  const contentType = String(body.content_type || "video/mp4").trim();
  const size = Number(body.size || 0);

  if (!filename) {
    return NextResponse.json({ error: "filename required" }, { status: 400 });
  }

  if (!contentType.startsWith("video/")) {
    return NextResponse.json(
      { error: "Please choose a video file" },
      { status: 400 },
    );
  }

  if (!Number.isFinite(size) || size <= 0) {
    return NextResponse.json({ error: "Invalid file size" }, { status: 400 });
  }

  if (size > MAX_UPLOAD_BYTES) {
    return NextResponse.json(
      { error: "Video is larger than the 10 GB upload limit" },
      { status: 413 },
    );
  }

  const ext =
    filename
      .split(".")
      .pop()
      ?.toLowerCase()
      .replace(/[^a-z0-9]/g, "") || "mp4";

  const r2Key = `sources/uploads/${userId}/${crypto.randomUUID()}.${ext}`;

  const uploadUrl = await presignedPutUrl(r2Key, contentType);

  return NextResponse.json({
    upload_url: uploadUrl,
    r2_key: r2Key,
  });
}
