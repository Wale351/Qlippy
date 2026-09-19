import { S3Client, GetObjectCommand, DeleteObjectCommand, PutObjectCommand } from "@aws-sdk/client-s3";
import { getSignedUrl } from "@aws-sdk/s3-request-presigner";

const PUBLIC_DOMAIN = process.env.R2_PUBLIC_DOMAIN;
const BUCKET        = process.env.R2_BUCKET!;

// Lazily initialised — only created when R2_PUBLIC_DOMAIN is absent
let _client: S3Client | null = null;
function getClient(): S3Client {
  if (!_client) {
    _client = new S3Client({
      endpoint:        process.env.R2_ENDPOINT!,
      credentials: {
        accessKeyId:     process.env.R2_ACCESS_KEY_ID!,
        secretAccessKey: process.env.R2_SECRET_ACCESS_KEY!,
      },
      region:          process.env.R2_REGION ?? "auto",
      forcePathStyle:  true,
    });
  }
  return _client;
}

/**
 * Return a permanent public URL (if R2_PUBLIC_DOMAIN is set) or a fresh
 * 7-day presigned URL.  Call this at read time — never store the result.
 */
export async function freshUrl(r2Key: string | null | undefined): Promise<string | null> {
  if (!r2Key) return null;
  if (PUBLIC_DOMAIN) return `https://${PUBLIC_DOMAIN}/${r2Key}`;
  return getSignedUrl(
    getClient(),
    new GetObjectCommand({ Bucket: BUCKET, Key: r2Key }),
    { expiresIn: 604800 }, // 7 days — maximum R2 allows
  );
}

/**
 * Re-sign each clip's thumbnail from its deterministic R2 key at read time.
 * The stored thumbnail_url is an expiring presigned URL — trust only its
 * presence (= the worker uploaded a thumb), never the stale value itself.
 */
export async function freshenClipThumbnails<T extends { id: string; thumbnail_url?: string | null }>(
  clips: T[],
): Promise<T[]> {
  return Promise.all(
    clips.map(async (c) =>
      c.thumbnail_url ? { ...c, thumbnail_url: await freshUrl(`thumbnails/${c.id}.jpg`) } : c,
    ),
  );
}

/** Re-sign a video's portrait preview from its deterministic R2 key at read time. */
export async function freshenPortraitPreview<T extends { id: string; portrait_preview_url?: string | null }>(
  video: T,
): Promise<T> {
  return video.portrait_preview_url
    ? { ...video, portrait_preview_url: await freshUrl(`previews/${video.id}/portrait_preview.jpg`) }
    : video;
}

export async function deleteObject(r2Key: string): Promise<void> {
  await getClient().send(new DeleteObjectCommand({ Bucket: BUCKET, Key: r2Key }));
}

export async function putObject(r2Key: string, body: Uint8Array, contentType: string): Promise<void> {
  await getClient().send(new PutObjectCommand({ Bucket: BUCKET, Key: r2Key, Body: body, ContentType: contentType }));
}

/**
 * Create a short-lived URL that lets the browser upload a file directly to R2.
 * The video bytes never pass through the Next.js server.
 */
export async function presignedPutUrl(
  r2Key: string,
  contentType: string,
): Promise<string> {
  return getSignedUrl(
    getClient(),
    new PutObjectCommand({
      Bucket: BUCKET,
      Key: r2Key,
      ContentType: contentType,
    }),
    { expiresIn: 3600 },
  );
}

/**
 * Apply freshUrl to every export row in a list.
 * Accepts rows that have r2_key (canonical) or fall back to stored r2_url.
 */
export async function freshenExports<T extends { r2_key?: string | null; r2_url?: string | null }>(
  rows: T[],
): Promise<T[]> {
  return Promise.all(
    rows.map(async (row) => {
      const url = await freshUrl(row.r2_key) ?? row.r2_url ?? null;
      return { ...row, r2_url: url };
    }),
  );
}
