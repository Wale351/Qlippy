"use client";

import { useState, useEffect, useRef } from "react";
import { Plus, Loader2, X, AlertCircle, Clock, Upload } from "lucide-react";
import { Collection } from "@/lib/types";
import { formatDuration } from "@/lib/youtube";
import { Modal } from "@/components/ui/modal";
import { Button } from "@/components/ui/button";

// ── Video URL validation ──────────────────────────────────────────────────────

const URL_MIN = 8;
const URL_MAX = 1000;

function validateUrl(raw: string): string | null {
  const value = raw.trim();

  if (!value) return "URL is required";
  if (value.includes(",")) return "Only one URL at a time — remove commas";
  if (value.includes("\n")) return "Only one URL at a time";
  if (value.length < URL_MIN) return "URL is too short";
  if (value.length > URL_MAX) return `URL too long (max ${URL_MAX} characters)`;

  try {
    const parsed = new URL(value);
    if (parsed.protocol !== "http:" && parsed.protocol !== "https:") {
      return "Enter a valid http or https video URL";
    }
  } catch {
    return "Enter a valid video URL";
  }

  return null;
}

function videoContentType(file: File): string | null {
  if (file.type.startsWith("video/")) return file.type;

  const ext = file.name.split(".").pop()?.toLowerCase();

  const types: Record<string, string> = {
    mp4: "video/mp4",
    mov: "video/quicktime",
    webm: "video/webm",
    mkv: "video/x-matroska",
    m4v: "video/x-m4v",
    avi: "video/x-msvideo",
  };

  return ext ? types[ext] ?? null : null;
}


// ── Component ─────────────────────────────────────────────────────────────────

interface Props { onAdded: () => void; }

export function AddVideoDialog({ onAdded }: Props) {
  const [open, setOpen]         = useState(false);
  const [mode, setMode]         = useState<"url" | "upload">("url");
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [url, setUrl]           = useState("");
  const [urlTouched, setUrlTouched] = useState(false);
  const [loading, setLoading]   = useState(false);
  const [error, setError]       = useState("");

  type InspectResolution = {
    height: number;
    requested_height: number;
    estimated_size: number | null;
  };

  type InspectData = {
    title: string;
    duration: number | null;
    thumbnail: string | null;
    channel_name: string;
    source_type: string;
    source_id: string | null;
    source_url: string;
    resolutions: InspectResolution[];
  };

  const [inspectData, setInspectData] = useState<InspectData | null>(null);
  const [inspecting, setInspecting] = useState(false);
  const [resolution, setResolution] = useState(720);
  const [clipDuration, setClipDuration] =
    useState<"auto" | 15 | 30 | 60 | 90>("auto");

  const [collections, setCollections]           = useState<Collection[]>([]);
  const [collectionId, setCollectionId]         = useState("");
  const [selectedCollName, setSelectedCollName] = useState("");
  const [collSearch, setCollSearch]             = useState("");
  const [collDropOpen, setCollDropOpen]         = useState(false);

  const urlRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (!open) return;
    fetch("/api/collections")
      .then((r) => r.ok ? r.json() : [])
      .then((d) => { if (Array.isArray(d)) setCollections(d); });
  }, [open]);

  const resetAndClose = () => {
    setOpen(false);
    setMode("url");
    setSelectedFile(null);
    setUrl("");
    setUrlTouched(false);
    setError("");
    setInspectData(null);
    setInspecting(false);
    setResolution(720);
    setClipDuration("auto");
    setCollectionId("");
    setSelectedCollName("");
    setCollSearch("");
    setCollDropOpen(false);
  };

  // Collection helpers
  const selectCollection = (id: string, name: string) => {
    setCollectionId(id); setSelectedCollName(name);
    setCollSearch(""); setCollDropOpen(false);
  };
  const clearCollection = () => {
    setCollectionId(""); setSelectedCollName(""); setCollSearch("");
  };
  const handleCreateAndSelect = async () => {
    const name = collSearch.trim();
    if (!name) return;
    const res  = await fetch("/api/collections", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name }),
    });
    const data = await res.json();
    if (res.ok) {
      setCollections((prev) => [...prev, data]);
      selectCollection(data.id, data.name);
    } else if (res.status === 409) {
      const existing = collections.find(
        (c) => c.name.toLowerCase() === name.toLowerCase()
      );
      if (existing) selectCollection(existing.id, existing.name);
    }
  };

  const filteredCollections = collections.filter((c) =>
    c.name.toLowerCase().includes(collSearch.toLowerCase())
  );
  const exactMatch = collections.some(
    (c) => c.name.toLowerCase() === collSearch.trim().toLowerCase()
  );
  const showCreate = collSearch.trim().length > 0 && !exactMatch;

  // Inline validation (only shown after field is touched)
  const urlError = urlTouched ? validateUrl(url) : null;

  const handleSubmit = async (e: { preventDefault(): void }) => {
    e.preventDefault();

    if (mode === "upload") {
      if (!selectedFile) {
        setError("Choose a video file first");
        return;
      }

      const contentType = videoContentType(selectedFile);

      if (!contentType) {
        setError("Please choose a supported video file");
        return;
      }

      setError("");
      setLoading(true);

      try {
        // 1. Ask Qlippy for a short-lived direct R2 upload URL.
        const initRes = await fetch("/api/videos/upload-url", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            filename: selectedFile.name,
            content_type: contentType,
            size: selectedFile.size,
          }),
        });

        const initData = await initRes.json();

        if (!initRes.ok) {
          throw new Error(initData.error || "Failed to prepare upload");
        }

        // 2. Send the actual video directly from the browser to R2.
        const uploadRes = await fetch(initData.upload_url, {
          method: "PUT",
          headers: {
            "Content-Type": contentType,
          },
          body: selectedFile,
        });

        if (!uploadRes.ok) {
          throw new Error(`Video upload failed (${uploadRes.status})`);
        }

        // 3. Create the Qlippy video record and start processing.
        const title =
          selectedFile.name.replace(/\.[^.]+$/, "").trim() ||
          selectedFile.name;

        const createRes = await fetch("/api/videos", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            source_r2_key: initData.r2_key,
            source_type: "upload",
            source_id: initData.r2_key,
            title,
            model_size: "large",
            clip_duration_seconds:
              clipDuration === "auto" ? null : clipDuration,
            collection_id: collectionId || null,
          }),
        });

        const createData = await createRes.json();

        if (!createRes.ok) {
          throw new Error(createData.error || "Failed to add uploaded video");
        }

        resetAndClose();
        onAdded();
      } catch (err: unknown) {
        setError(err instanceof Error ? err.message : "Upload failed");
      } finally {
        setLoading(false);
      }

      return;
    }

    setUrlTouched(true);

    const validationErr = validateUrl(url);
    if (validationErr) {
      setError(validationErr);
      return;
    }

    setError("");

    if (!inspectData) {
      setInspecting(true);
      try {
        const res = await fetch("/api/videos/inspect", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ video_url: url.trim() }),
        });

        const data = await res.json();
        if (!res.ok) throw new Error(data.error || "Failed to inspect video");

        setInspectData(data);

        const available = (data.resolutions ?? []).map(
          (r: InspectResolution) => r.requested_height
        );

        if (!available.includes(720)) {
          setResolution(available.includes(480) ? 480 : available[0] ?? 720);
        }
      } catch (err: unknown) {
        setError(err instanceof Error ? err.message : "Failed to inspect video");
      } finally {
        setInspecting(false);
      }
      return;
    }

    setLoading(true);

    try {
      const res = await fetch("/api/videos", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          video_url: url.trim(),
          source_url: inspectData.source_url,
          source_type: inspectData.source_type,
          source_id: inspectData.source_id,
          title: inspectData.title,
          thumbnail_url: inspectData.thumbnail,
          channel_name: inspectData.channel_name,
          duration: inspectData.duration,
          model_size: "large",
          clip_duration_seconds:
            clipDuration === "auto" ? null : clipDuration,
          collection_id: collectionId || null,
          resolution,
        }),
      });

      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Failed to add video");

      resetAndClose();
      onAdded();
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Something went wrong");
    } finally {
      setLoading(false);
    }
  };

  const formatBytes = (bytes: number | null) => {
    if (!bytes) return "Size unavailable";
    const mb = bytes / 1_000_000;
    return mb >= 1000 ? `${(mb / 1000).toFixed(1)} GB` : `${Math.round(mb)} MB`;
  };

  if (!open) {
    return (
      <Button variant="surface" pill onClick={() => setOpen(true)} className="gap-2">
        <Plus className="w-4 h-4" />
        Add video
      </Button>
    );
  }

  return (
    <Modal onClose={resetAndClose} initialFocus={urlRef} label="Add video"
      className="rounded-2xl p-6 w-full max-w-md shadow-2xl"
      style={{ background: "var(--yt-surface)", border: "1px solid var(--yt-border)" }}>
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-lg font-semibold text-white">Add video</h2>
          <button aria-label="Close"
            onClick={resetAndClose}
            className="w-9 h-9 flex items-center justify-center rounded-full transition-colors text-[var(--yt-text-2)] hover:bg-[var(--yt-surface-2)]"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Loading state — shown while import is in progress */}
        {loading ? (
          <div className="flex flex-col items-center gap-5 py-6">
            <Loader2 className="w-8 h-8 animate-spin" style={{ color: "var(--yt-text-2)" }} />
            <p className="text-sm font-medium text-[var(--yt-text)]">Importing video…</p>
          </div>
        ) : (
        <div>

        <form onSubmit={handleSubmit} className="space-y-4">

          <div
            className="grid grid-cols-2 gap-1 p-1 rounded-xl"
            style={{ background: "var(--yt-surface-2)" }}
          >
            <button
              type="button"
              onClick={() => {
                setMode("url");
                setSelectedFile(null);
                setError("");
              }}
              className="px-3 py-2 rounded-lg text-sm font-medium transition-colors"
              style={{
                background: mode === "url" ? "var(--yt-surface)" : "transparent",
                color: mode === "url" ? "white" : "var(--yt-text-2)",
              }}
            >
              Paste URL
            </button>

            <button
              type="button"
              onClick={() => {
                setMode("upload");
                setInspectData(null);
                setUrlTouched(false);
                setError("");
              }}
              className="px-3 py-2 rounded-lg text-sm font-medium transition-colors"
              style={{
                background: mode === "upload" ? "var(--yt-surface)" : "transparent",
                color: mode === "upload" ? "white" : "var(--yt-text-2)",
              }}
            >
              Upload file
            </button>
          </div>

          <div>
            <p
              className="text-sm mb-2"
              style={{ color: "var(--yt-text-2)" }}
            >
              Clip duration
            </p>

            <div className="grid grid-cols-5 gap-1.5">
              {(["auto", 15, 30, 60, 90] as const).map((duration) => (
                <button
                  key={duration}
                  type="button"
                  onClick={() => setClipDuration(duration)}
                  className="rounded-lg px-2 py-2 text-xs font-medium transition-colors"
                  style={{
                    background:
                      clipDuration === duration
                        ? "var(--yt-red)"
                        : "var(--yt-surface-2)",
                    color:
                      clipDuration === duration
                        ? "white"
                        : "var(--yt-text-2)",
                    border: "1px solid var(--yt-border)",
                  }}
                >
                  {duration === "auto" ? "Auto" : `${duration}s`}
                </button>
              ))}
            </div>

            <p
              className="mt-1.5 text-xs"
              style={{ color: "var(--yt-text-2)", opacity: 0.7 }}
            >
              Qlippy will aim for Moments around this length.
            </p>
          </div>

          {mode === "url" ? (
            <>
          {/* Video URL */}
          <div>
            <label className="block text-sm mb-1.5" style={{ color: "var(--yt-text-2)" }}>
              Video URL
              <span className="ml-2 text-xs" style={{ color: "var(--yt-text-2)", opacity: 0.6 }}>
                {url.length}/{URL_MAX}
              </span>
            </label>
            <input
              ref={urlRef}
              type="text"
              value={url}
              onChange={(e) => {
                setUrl(e.target.value);
                setInspectData(null);
                setResolution(720);
                if (error) setError("");
              }}
              onBlur={() => setUrlTouched(true)}
              onPaste={(e) => {
                // Strip leading/trailing whitespace and reject multi-line pastes
                const pasted = e.clipboardData.getData("text");
                if (pasted.includes("\n")) {
                  e.preventDefault();
                  setUrl(pasted.split("\n")[0].trim());
                  setUrlTouched(true);
                }
              }}
              placeholder="Paste a YouTube, TikTok, X, Instagram, Vimeo or video URL"
              maxLength={URL_MAX}
              className={`w-full px-3 py-2.5 rounded-lg text-sm text-white placeholder-[var(--yt-text-2)] focus:outline-none focus:ring-1 ${
                urlError ? "ring-1 ring-red-500" : "focus:ring-[var(--yt-text-2)]"
              }`}
              style={{ background: "var(--yt-surface-2)", border: `1px solid ${urlError ? "#ef4444" : "var(--yt-border)"}` }}
            />
            {urlError && (
              <p className="flex items-center gap-1.5 mt-1.5 text-xs text-red-400">
                <AlertCircle className="w-3.5 h-3.5 flex-shrink-0" />
                {urlError}
              </p>
            )}
            {!urlError && urlTouched && url.trim() && (
              <p className="mt-1.5 text-xs text-green-500">Valid video URL</p>
            )}
          </div>

          {inspectData && (
            <div
              className="rounded-xl p-3 space-y-3"
              style={{ background: "var(--yt-surface-2)", border: "1px solid var(--yt-border)" }}
            >
              <div>
                <p className="text-sm font-medium text-white line-clamp-2">{inspectData.title}</p>
                {inspectData.duration != null && (
                  <p className="mt-1 text-xs" style={{ color: "var(--yt-text-2)" }}>
                    Duration: {formatDuration(inspectData.duration)}
                  </p>
                )}
              </div>

              <div>
                <p className="text-xs mb-2" style={{ color: "var(--yt-text-2)" }}>
                  Download resolution
                </p>

                <div className="grid grid-cols-4 gap-2">
                  {inspectData.resolutions.map((r) => (
                    <button
                      key={r.requested_height}
                      type="button"
                      onClick={() => setResolution(r.requested_height)}
                      className="rounded-lg px-2 py-2 text-xs font-medium transition-colors"
                      style={{
                        background:
                          resolution === r.requested_height
                            ? "var(--yt-red)"
                            : "var(--yt-surface)",
                        color: "white",
                        border: "1px solid var(--yt-border)",
                      }}
                    >
                      {r.requested_height}p
                    </button>
                  ))}
                </div>

                {(() => {
                  const selected = inspectData.resolutions.find(
                    (r) => r.requested_height === resolution
                  );

                  return (
                    <p className="mt-2 text-xs" style={{ color: "var(--yt-text-2)" }}>
                      Estimated download: {formatBytes(selected?.estimated_size ?? null)}
                    </p>
                  );
                })()}
              </div>
            </div>
          )}

            </>
          ) : (
            <div>
              <label className="block text-sm mb-1.5" style={{ color: "var(--yt-text-2)" }}>
                Video file
              </label>

              <label
                className="flex flex-col items-center justify-center gap-3 rounded-xl px-5 py-8 cursor-pointer transition-colors hover:bg-[var(--yt-surface-2)]"
                style={{
                  border: "1px dashed var(--yt-border)",
                  background: "rgba(255,255,255,0.015)",
                }}
              >
                <Upload className="w-7 h-7" style={{ color: "var(--yt-text-2)" }} />

                {selectedFile ? (
                  <div className="text-center min-w-0">
                    <p className="text-sm font-medium text-white truncate max-w-[320px]">
                      {selectedFile.name}
                    </p>
                    <p className="text-xs mt-1" style={{ color: "var(--yt-text-2)" }}>
                      {formatBytes(selectedFile.size)}
                    </p>
                  </div>
                ) : (
                  <div className="text-center">
                    <p className="text-sm font-medium text-white">
                      Choose a video
                    </p>
                    <p className="text-xs mt-1" style={{ color: "var(--yt-text-2)" }}>
                      MP4, MOV, WebM, MKV and other video formats
                    </p>
                  </div>
                )}

                <input
                  type="file"
                  accept="video/*,.mkv,.m4v"
                  className="hidden"
                  onChange={(e) => {
                    const file = e.target.files?.[0] ?? null;
                    setSelectedFile(file);
                    setError("");
                  }}
                />
              </label>
            </div>
          )}

          {/* Collection */}
          <div>
            <label className="block text-sm mb-1.5" style={{ color: "var(--yt-text-2)" }}>
              Collection <span style={{ opacity: 0.6 }}>(optional)</span>
            </label>
            <div className="relative">
              <input
                value={collDropOpen ? collSearch : selectedCollName}
                onChange={(e) => { setCollSearch(e.target.value); setCollDropOpen(true); }}
                onFocus={() => { setCollSearch(""); setCollDropOpen(true); }}
                onKeyDown={(e) => {
                  if (e.key === "Escape") {
                    // Close only the dropdown; don't let the Modal's Esc close the whole dialog.
                    if (collDropOpen) e.nativeEvent.stopImmediatePropagation();
                    setCollDropOpen(false); setCollSearch("");
                  }
                  if (e.key === "Enter") {
                    e.preventDefault();
                    if (showCreate) handleCreateAndSelect();
                    else if (filteredCollections.length > 0)
                      selectCollection(filteredCollections[0].id, filteredCollections[0].name);
                  }
                }}
                placeholder="Search or create a collection…"
                className="w-full px-3 py-2 rounded-lg text-sm text-white placeholder-[var(--yt-text-2)] focus:outline-none focus:ring-1 focus:ring-[var(--yt-text-2)]"
                style={{ background: "var(--yt-surface-2)", border: "1px solid var(--yt-border)" }}
              />
              {selectedCollName && !collDropOpen && (
                <button aria-label="Clear collection" type="button" onClick={clearCollection}
                  className="absolute right-2 top-1/2 -translate-y-1/2 transition-colors"
                  style={{ color: "var(--yt-text-2)" }}>
                  <X className="w-3.5 h-3.5" />
                </button>
              )}
              {collDropOpen && (
                <>
                  <div className="fixed inset-0 z-10" onClick={() => { setCollDropOpen(false); setCollSearch(""); }} />
                  <div
                    className="absolute z-20 w-full mt-1 rounded-xl py-1 shadow-2xl max-h-48 overflow-y-auto"
                    style={{ background: "var(--yt-surface)", border: "1px solid var(--yt-border)" }}
                  >
                    <button type="button" onClick={() => { clearCollection(); setCollDropOpen(false); }}
                      className="w-full text-left px-3 py-2 text-sm transition-colors hover:bg-[var(--yt-surface-2)]"
                      style={{ color: "var(--yt-text-2)" }}>
                      — No collection —
                    </button>
                    {filteredCollections.map((c) => (
                      <button key={c.id} type="button" onClick={() => selectCollection(c.id, c.name)}
                        className="w-full text-left px-3 py-2 text-sm text-white transition-colors hover:bg-[var(--yt-surface-2)]">
                        {c.name}
                      </button>
                    ))}
                    {showCreate && (
                      <button type="button" onClick={handleCreateAndSelect}
                        className="w-full text-left px-3 py-2 text-sm text-white transition-colors hover:bg-[var(--yt-surface-2)]">
                        + Create &quot;{collSearch.trim()}&quot;
                      </button>
                    )}
                    {filteredCollections.length === 0 && !showCreate && (
                      <p className="px-3 py-2 text-xs" style={{ color: "var(--yt-text-2)" }}>No collections yet</p>
                    )}
                  </div>
                </>
              )}
            </div>
          </div>

          {/* API / server error */}
          {error && (
            <div className="flex items-center gap-2 text-xs text-red-400">
              <AlertCircle className="w-3.5 h-3.5 flex-shrink-0" />
              {error}
            </div>
          )}

          <div className="flex gap-3 justify-end pt-1">
            <button type="button" onClick={resetAndClose}
              className="px-4 py-2 text-sm transition-colors"
              style={{ color: "var(--yt-text-2)" }}>
              Cancel
            </button>
            <button
              type="submit"
              disabled={
                mode === "url"
                  ? !!validateUrl(url) || inspecting
                  : !selectedFile
              }
              className="flex items-center gap-2 px-4 py-2 disabled:opacity-50 text-white text-sm font-medium rounded-full transition-colors"
              style={{ background: "var(--yt-red)" }}
            >
              {mode === "upload"
                ? "Upload & process"
                : inspecting
                  ? "Checking video…"
                  : inspectData
                    ? "Start processing"
                    : "Check video"}
            </button>
          </div>
        </form>
        </div>
        )}
    </Modal>
  );
}
