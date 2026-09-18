"use client";

import { useState, useEffect, useRef } from "react";
import { Plus, Loader2, X, AlertCircle, Clock } from "lucide-react";
import { Collection } from "@/lib/types";
import { formatDuration } from "@/lib/youtube";
import { Modal } from "@/components/ui/modal";
import { Button } from "@/components/ui/button";

// ── YouTube URL validation ────────────────────────────────────────────────────

const URL_MIN = 20;
const URL_MAX = 200;

const YT_PATTERNS = [
  /^https?:\/\/(www\.|m\.)?youtube\.com\/watch\?.*v=[\w-]{11}/,
  /^https?:\/\/youtu\.be\/[\w-]{11}/,
  /^https?:\/\/(www\.)?youtube\.com\/shorts\/[\w-]{11}/,
  /^https?:\/\/(www\.)?youtube\.com\/live\/[\w-]{11}/,
  /^https?:\/\/(www\.)?youtube\.com\/embed\/[\w-]{11}/,
];

function validateUrl(raw: string): string | null {
  const url = raw.trim();
  if (!url)                    return "URL is required";
  if (url.includes(","))       return "Only one URL at a time — remove commas";
  if (url.includes("\n"))      return "Only one URL at a time";
  if (url.length < URL_MIN)    return `URL too short (min ${URL_MIN} characters)`;
  if (url.length > URL_MAX)    return `URL too long (max ${URL_MAX} characters)`;
  if (!YT_PATTERNS.some((p) => p.test(url)))
    return "Not a valid YouTube video URL";
  return null;
}

// ── Component ─────────────────────────────────────────────────────────────────

interface Props { onAdded: () => void; }

export function AddVideoDialog({ onAdded }: Props) {
  const [open, setOpen]         = useState(false);
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
    resolutions: InspectResolution[];
  };

  const [inspectData, setInspectData] = useState<InspectData | null>(null);
  const [inspecting, setInspecting] = useState(false);
  const [resolution, setResolution] = useState(720);

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
    setUrl("");
    setUrlTouched(false);
    setError("");
    setInspectData(null);
    setInspecting(false);
    setResolution(720);
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
          body: JSON.stringify({ youtube_url: url.trim() }),
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
          youtube_url: url.trim(),
          model_size: "large",
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

          {/* YouTube URL */}
          <div>
            <label className="block text-sm mb-1.5" style={{ color: "var(--yt-text-2)" }}>
              YouTube URL
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
              placeholder="https://www.youtube.com/watch?v=..."
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
              <p className="mt-1.5 text-xs text-green-500">Valid YouTube URL</p>
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
              disabled={!!validateUrl(url) || inspecting}
              className="flex items-center gap-2 px-4 py-2 disabled:opacity-50 text-white text-sm font-medium rounded-full transition-colors"
              style={{ background: "var(--yt-red)" }}
            >
              {inspecting
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
