"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { Clip, Video } from "@/lib/types";
import { formatDuration } from "@/lib/youtube";
import { ExportDialog } from "@/components/clips/export-dialog";
import { IdentifyClipsButton } from "@/components/clips/identify-clips-button";
import { DownloadDialog, shouldSkipDownloadDialog, defaultFilename, downloadHref } from "@/components/clips/download-dialog";
import { VersionPicker } from "@/components/clips/version-picker";
import { Modal } from "@/components/ui/modal";
import { Menu, MenuItem } from "@/components/ui/menu";
import { useExportTracker } from "@/lib/export-tracker";
import { exportProgressPct } from "@/lib/processing-progress";
import { downloadUrl } from "@/lib/save-file";
import { queueBulkExports } from "@/lib/bulk-export";
import {
  ArrowLeft, Scissors, Download, Clock,
  Film, Play, Pause, Loader2, RefreshCw, Sparkles, AlertCircle,
  Copy, CheckCheck, ArrowUpDown, ChevronDown, CheckSquare, X, History, FileText,
} from "lucide-react";
import type { PresetConfig } from "@/lib/types";

type ExportInfo = { id: string; aspect_ratio: string; r2_url: string; created_at?: string };

interface Transcript {
  id: string;
  language: string | null;
  full_text: string | null;
  model_size: string | null;
  created_at: string;
}

interface Props {
  video: Video;
  transcript: Transcript | null;
  clips: Clip[];
  clipExports: Record<string, { id: string; aspect_ratio: string; r2_url: string; created_at?: string }[]>;
  pendingExports: { id: string; clip_id: string; updated_at: string }[];
}

export function VideoChannelClient({ video, transcript, clips, clipExports, pendingExports }: Props) {
  const [exportClip, setExportClip] = useState<Clip | null>(null);
  const [localExports, setLocalExports] = useState<Record<string, ExportInfo[]>>(clipExports);
  const [exportingClips, setExportingClips] = useState<Map<string, string>>(
    () => new Map(pendingExports.map((e) => [e.clip_id, e.updated_at])),
  );
  const [frameConfig]             = useState<PresetConfig | null>(null);
  const [activeSubtitlePresetId] = useState<string | null>(null);
  const { trackExport } = useExportTracker();
  const router = useRouter();

  // Bulk export — selection mode + confirm-with-warning flow
  const [selectMode, setSelectMode]     = useState(false);
  const [selectedIds, setSelectedIds]   = useState<Set<string>>(new Set());
  const [bulkConfirmIds, setBulkConfirmIds] = useState<string[] | null>(null);
  const [bulkRunning, setBulkRunning]   = useState(false);
  const [bulkNotice, setBulkNotice]     = useState<string | null>(null);

  const toggleSelected = (clipId: string) => {
    setSelectedIds((prev) => {
      const next = new Set(prev);
      if (next.has(clipId)) next.delete(clipId);
      else next.add(clipId);
      return next;
    });
  };

  const iframeRef     = useRef<HTMLIFrameElement>(null);
  const topSectionRef = useRef<HTMLDivElement>(null);
  type SortKey = "sequence" | "duration" | "virality";
  const [sortBy, setSortBy]     = useState<SortKey>("sequence");

  const handleSeek = useCallback((seconds: number) => {
    // Use the IFrame Player API (postMessage) so the embed seeks in place
    // instead of reloading the whole player on every timestamp click.
    const win = iframeRef.current?.contentWindow;
    if (win && video.youtube_id) {
      const cmd = (func: string, args: unknown[]) =>
        win.postMessage(JSON.stringify({ event: "command", func, args }), "*");
      cmd("seekTo", [Math.floor(seconds), true]);
      cmd("playVideo", []);
    }
    topSectionRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
  }, [video.youtube_id]);

  // Auto-refresh while processing
  useEffect(() => {
    const active = video.status === "pending" || video.status === "transcribing" || video.status === "identifying";
    if (!active) return;
    const interval = video.status === "identifying" ? 3000 : 5000;
    const iv = setInterval(() => router.refresh(), interval);
    return () => clearInterval(iv);
  }, [video.status, router]);

  // Auto-retry on error — up to 2 times. The count is persisted per video in
  // sessionStorage so a page reload doesn't reset the cap and loop forever.
  const retryStorageKey = `cf_auto_retry_${video.id}`;
  const [retryCount, setRetryCount] = useState(0);
  const autoRetryFiredRef = useRef(false);

  useEffect(() => {
    try { setRetryCount(Number(sessionStorage.getItem(retryStorageKey)) || 0); } catch { /* SSR / blocked storage */ }
  }, [retryStorageKey]);

  useEffect(() => {
    if (video.status !== "error") { autoRetryFiredRef.current = false; return; }
    if (retryCount >= 2 || autoRetryFiredRef.current) return;
    autoRetryFiredRef.current = true;
    const t = setTimeout(async () => {
      try {
        const endpoint = transcript
          ? `/api/videos/${video.id}/identify-clips`
          : `/api/videos/${video.id}/reprocess`;

        await fetch(endpoint, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(
            transcript
              ? { keep_existing: false }
              : { model_size: "large" }
          ),
        });
        setRetryCount((n) => {
          const next = n + 1;
          try { sessionStorage.setItem(retryStorageKey, String(next)); } catch { /* ignore */ }
          return next;
        });
        router.refresh();
      } catch { /* ignore */ }
    }, 2000);
    return () => clearTimeout(t);
  }, [video.status, retryCount, video.id, retryStorageKey, router]);

  const stageProgress: Record<string, number> = {
    preparing: 5,
    downloading: 12,
    extracting_audio: 20,
    transcribing_audio: 35,
    diarizing_speakers: 48,
    binding_speakers: 58,
    preparing_analysis: 65,
    finding_moments: 72,
    crafting_scenes: 80,
    building_moments: 86,
    ranking_moments: 92,
    finalizing: 96,
  };

  const processingPct =
    video.status === "ready"
      ? 100
      : stageProgress[video.processing_stage ?? ""] ?? 2;

  // Stable export polling
  const pendingRef = useRef<Map<string, string>>(
    new Map(pendingExports.map((e) => [e.id, e.clip_id])),
  );

  useEffect(() => {
    const iv = setInterval(async () => {
      if (pendingRef.current.size === 0) return;
      for (const [exportId, clipId] of [...pendingRef.current]) {
        try {
          const res = await fetch(`/api/exports/${exportId}`);
          if (!res.ok) continue;
          const data = await res.json();
          if (data.status === "done" && data.r2_url) {
            setLocalExports((prev) => ({
              ...prev,
              [clipId]: [
                { id: exportId, aspect_ratio: data.aspect_ratio, r2_url: data.r2_url,
                  created_at: data.created_at ?? new Date().toISOString() },
                ...(prev[clipId] ?? []),
              ],
            }));
            setExportingClips((prev) => { const m = new Map(prev); m.delete(clipId); return m; });
            pendingRef.current.delete(exportId);
          } else if (data.status === "error") {
            setExportingClips((prev) => { const m = new Map(prev); m.delete(clipId); return m; });
            pendingRef.current.delete(exportId);
          }
        } catch { /* retry next tick */ }
      }
    }, 3000);
    return () => clearInterval(iv);
  }, []);

  const markQueued = (clipId: string, exportId: string, hook: string | null) => {
    trackExport(exportId, `${hook?.slice(0, 40) ?? "Clip"} · 9:16`);
    pendingRef.current.set(exportId, clipId);
    setExportingClips((prev) => new Map([...prev, [clipId, new Date().toISOString()]]));
  };

  const runBulkExport = async (clipIds: string[]) => {
    setBulkConfirmIds(null);
    setBulkRunning(true);
    setBulkNotice(null);
    const hooks = new Map(clips.map((c) => [c.id, c.hook]));
    const result = await queueBulkExports(
      clipIds,
      { aspect_ratio: "9:16", preset_config: frameConfig ?? null },
      (clipId, exportId) => markQueued(clipId, exportId, hooks.get(clipId) ?? null),
    );
    setBulkRunning(false);
    setSelectMode(false);
    setSelectedIds(new Set());
    if (result.failed > 0) {
      setBulkNotice(`Queued ${result.queued.length} exports · ${result.failed} failed to start.`);
    } else {
      setBulkNotice(`Queued ${result.queued.length} exports — they'll appear on each clip as they finish.`);
    }
    setTimeout(() => setBulkNotice(null), 5000);
  };

  const totalDuration = clips.reduce((sum, c) => sum + c.cuts.reduce((s, cut) => s + (cut.end - cut.start), 0), 0);
  const maxClipEnd = clips.length > 0 ? Math.max(...clips.flatMap((c) => c.cuts.map((cut) => cut.end))) : 0;
  const videoDuration = video.duration_sec ?? (maxClipEnd > 0 ? maxClipEnd : null);
  const coverage = videoDuration && totalDuration > 0 ? Math.min(Math.round((totalDuration / videoDuration) * 100), 100) : null;

  const sortedClips = [...clips].sort((a, b) => {
    if (sortBy === "duration") {
      const da = a.cuts.reduce((s, c) => s + (c.end - c.start), 0);
      const db = b.cuts.reduce((s, c) => s + (c.end - c.start), 0);
      return db - da;
    }
    if (sortBy === "virality") return (b.score ?? 0) - (a.score ?? 0);
    return (a.cuts[0]?.start ?? 0) - (b.cuts[0]?.start ?? 0); // sequence
  });

  const SORT_LABELS: Record<SortKey, string> = {
    sequence: "Video sequence",
    duration: "Clip length",
    virality: "Virality score",
  };

  return (
    <div className="flex flex-col min-h-full" style={{ background: "var(--yt-bg)" }}>

      {/* ── Top section: embed + designers (stacks below lg, side-by-side on desktop) ── */}
      <div ref={topSectionRef} className="flex-shrink-0 flex flex-col lg:flex-row" style={{ borderBottom: "1px solid rgba(255,255,255,0.05)" }}>

        {/* ── Left: YouTube embed + title below ── */}
        <div className="flex-shrink-0 flex flex-col w-full lg:w-[38%]">
          <div className="px-8 pt-5 pb-3">
            <div
              className="relative w-full overflow-hidden"
              style={{ aspectRatio: "16/9", borderRadius: 14, background: "var(--yt-surface)" }}
            >
              {video.youtube_id ? (
                <iframe
                  ref={iframeRef}
                  src={`https://www.youtube.com/embed/${video.youtube_id}?enablejsapi=1&playsinline=1`}
                  title={video.title}
                  allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                  allowFullScreen
                  className="absolute inset-0 w-full h-full border-0"
                  style={{ borderRadius: 14 }}
                />
              ) : video.thumbnail_url ? (
                // eslint-disable-next-line @next/next/no-img-element
                <img src={video.thumbnail_url} alt="" className="absolute inset-0 w-full h-full object-cover" />
              ) : (
                <div className="absolute inset-0 flex items-center justify-center" style={{ color: "var(--yt-text-2)" }}>
                  <Film className="w-8 h-8 opacity-30" />
                </div>
              )}
            </div>
          </div>
          {/* Title below video */}
          <div className="px-8 pb-5">
            <h1 className="text-sm font-bold leading-snug line-clamp-2" style={{ color: "var(--yt-text)" }}>
              {video.title}
            </h1>
            {video.channel_name && (
              <p className="text-xs mt-0.5" style={{ color: "var(--yt-text-2)" }}>{video.channel_name}</p>
            )}
          </div>
        </div>

        {/* ── Right: back link + stats + config button ── */}
        <div className="flex flex-col px-6 py-5 flex-1 min-w-0 border-t lg:border-t-0 lg:border-l border-[rgba(255,255,255,0.05)]">
          <div className="flex items-center justify-between gap-3 mb-4 flex-shrink-0">
            <Link
              href="/library"
              className="inline-flex items-center gap-1.5 text-xs transition-colors"
              style={{ color: "var(--yt-text-2)" }}
            >
              <ArrowLeft className="w-3.5 h-3.5" />
              Back to Library
            </Link>
            {transcript && (
              <Link
                href={`/video/${video.id}/transcript`}
                className="inline-flex items-center gap-1.5 text-xs transition-colors"
                style={{ color: "var(--yt-text-2)" }}
              >
                <FileText className="w-3.5 h-3.5" />
                Transcript
              </Link>
            )}
          </div>

          <ProcessingStatus video={video} processingPct={processingPct} retryCount={retryCount} />

          {/* Stats row */}
          <div className="flex flex-row gap-2 mt-4">
            {[
              { label: "Duration",  value: videoDuration ? formatDuration(videoDuration) : "—" },
              { label: "Clips",     value: String(clips.length) },
              { label: "Clip time", value: clips.length > 0 ? formatDuration(totalDuration) : "—" },
              { label: "Coverage",  value: coverage !== null ? `${coverage}%` : "—" },
            ].map(({ label, value }) => (
              <div key={label} className="flex-1 flex flex-col gap-0.5 px-2 py-2 rounded-lg" style={{ background: "var(--yt-surface-2)", border: "1px solid var(--yt-border)" }}>
                <span className="text-[10px] uppercase tracking-wider font-medium truncate" style={{ color: "var(--yt-text-2)" }}>{label}</span>
                <span className="text-xs font-semibold tabular-nums" style={{ color: "var(--yt-text)" }}>{value}</span>
              </div>
            ))}
          </div>

        </div>

      </div>

      {/* ── Clips section ── */}
      <div className="flex-1 px-8 py-5">
        {clips.length === 0 ? (
          <EmptyClips video={video} hasTranscript={!!transcript} />
        ) : (
          <>
            {/* Clips header */}
            <div className="flex items-center gap-2 mb-4">
              <Menu
                ariaLabel="Sort clips"
                align="start"
                width={176}
                buttonClassName="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-colors bg-[var(--yt-surface-2)] border border-[var(--yt-border)] text-[var(--yt-text-2)] hover:text-[var(--yt-text)]"
                className="bg-[var(--yt-surface-2)] border border-[var(--yt-border)]"
                button={<>
                  <ArrowUpDown className="w-3.5 h-3.5" />
                  {SORT_LABELS[sortBy]}
                  <ChevronDown className="w-3 h-3 opacity-50" />
                </>}
              >
                {(["sequence", "duration", "virality"] as SortKey[]).map(key => (
                  <MenuItem
                    key={key}
                    size="sm"
                    onClick={() => setSortBy(key)}
                    className={sortBy === key ? "text-[var(--yt-text)] bg-white/5" : "!text-[var(--yt-text-2)]"}
                  >
                    {SORT_LABELS[key]}
                  </MenuItem>
                ))}
              </Menu>

              {/* Bulk export controls */}
              <div className="ml-auto flex items-center gap-2">
                {bulkNotice && (
                  <span role="status" aria-live="polite" className="text-xs" style={{ color: "var(--yt-text-2)" }}>{bulkNotice}</span>
                )}
                {selectMode ? (
                  <>
                    <span className="text-xs font-medium" style={{ color: "var(--yt-text)" }}>
                      {selectedIds.size} selected
                    </span>
                    <button
                      onClick={() => setBulkConfirmIds([...selectedIds])}
                      disabled={selectedIds.size === 0 || bulkRunning}
                      className="px-3 py-1.5 rounded-lg text-xs font-semibold transition-colors disabled:opacity-40"
                      style={{ background: "white", color: "#0a0a0a" }}
                    >
                      Export selected
                    </button>
                    <button
                      onClick={() => { setSelectMode(false); setSelectedIds(new Set()); }}
                      className="p-1.5 rounded-lg transition-colors"
                      style={{ color: "var(--yt-text-2)" }}
                      title="Cancel selection"
                    >
                      <X className="w-4 h-4" />
                    </button>
                  </>
                ) : (
                  <>
                    <button
                      onClick={() => setSelectMode(true)}
                      className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-colors"
                      style={{ background: "var(--yt-surface-2)", border: "1px solid var(--yt-border)", color: "var(--yt-text-2)" }}
                    >
                      <CheckSquare className="w-3.5 h-3.5" />
                      Select
                    </button>
                    <button
                      onClick={() => setBulkConfirmIds(sortedClips.map((c) => c.id))}
                      disabled={bulkRunning || sortedClips.length === 0}
                      className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-colors disabled:opacity-40"
                      style={{ background: "var(--yt-surface-2)", border: "1px solid var(--yt-border)", color: "var(--yt-text-2)" }}
                    >
                      {bulkRunning ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Scissors className="w-3.5 h-3.5" />}
                      Export all ({sortedClips.length})
                    </button>
                  </>
                )}
              </div>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
              {sortedClips.map((clip) => (
                <ClipPreviewCard
                  key={clip.id}
                  clip={clip}
                  video={video}
                  exports={localExports[clip.id] ?? []}
                  exportStartedAt={exportingClips.get(clip.id) ?? null}
                  onExport={() => setExportClip(clip)}
                  onSeek={handleSeek}
                  selected={selectMode ? selectedIds.has(clip.id) : undefined}
                  onToggleSelect={selectMode ? () => toggleSelected(clip.id) : undefined}
                />
              ))}
            </div>
          </>
        )}
      </div>

      {bulkConfirmIds && (
        <BulkExportConfirm
          count={bulkConfirmIds.length}
          onConfirm={() => runBulkExport(bulkConfirmIds)}
          onClose={() => setBulkConfirmIds(null)}
        />
      )}

      {exportClip && (
        <ExportDialog
          clip={exportClip}
          posterUrl={video.portrait_preview_url ?? video.thumbnail_url ?? null}
          initialConfig={frameConfig}
          initialSubtitlePresetId={activeSubtitlePresetId}
          onClose={() => setExportClip(null)}
          onQueued={(id, label) => {
            trackExport(id, label);
            const clipId = exportClip!.id;
            pendingRef.current.set(id, clipId);
            setExportingClips((prev) => new Map([...prev, [clipId, new Date().toISOString()]]));
            setExportClip(null);
          }}
        />
      )}
    </div>
  );
}

// ── Bulk export confirmation ──────────────────────────────────────────────────

function BulkExportConfirm({ count, onConfirm, onClose }: {
  count: number;
  onConfirm: () => void;
  onClose: () => void;
}) {
  return (
    <Modal onClose={onClose} label="Export clips"
      className="bg-[var(--yt-surface)] border border-[var(--yt-border)] rounded-xl w-full max-w-md shadow-2xl">
        <div className="flex items-center justify-between p-4 border-b border-[var(--yt-border)]">
          <div className="flex items-center gap-2">
            <Scissors className="w-4 h-4 text-[var(--yt-text)]" />
            <span className="font-semibold text-white text-sm">
              Export {count} clip{count === 1 ? "" : "s"}?
            </span>
          </div>
          <button aria-label="Close" onClick={onClose} className="text-[var(--yt-text-2)] hover:text-[var(--yt-text)] transition-colors">
            <X className="w-4 h-4" />
          </button>
        </div>

        <div className="p-4 space-y-3">
          <p className="text-xs text-[var(--yt-text)] flex items-start gap-2">
            <Clock className="w-3.5 h-3.5 flex-shrink-0 mt-0.5 text-yellow-400" />
            Each clip takes a few minutes to render, and clips are processed a couple
            at a time — exporting {count === 1 ? "this clip" : `all ${count}`} may take a while.
            Exports run in the background, so you can keep working (or leave) and each
            finished video will appear on its clip.
          </p>
          <p className="text-xs text-[var(--yt-text-2)]">
            Uses {count} export credit{count === 1 ? "" : "s"} · current subtitle &amp; watermark
            settings are applied to every clip.
          </p>
        </div>

        <div className="flex items-center justify-end gap-2 p-4 border-t border-[var(--yt-border)]">
          <button
            onClick={onClose}
            className="px-3 py-1.5 text-xs text-[var(--yt-text-2)] hover:text-[var(--yt-text)] rounded-lg hover:bg-[var(--yt-hover)] transition-colors"
          >
            Cancel
          </button>
          <button
            onClick={onConfirm}
            className="flex items-center gap-1.5 px-4 py-1.5 bg-white hover:bg-zinc-200 text-zinc-950 text-xs font-medium rounded-lg transition-colors"
          >
            <Scissors className="w-3 h-3" />
            Export {count} clip{count === 1 ? "" : "s"}
          </button>
        </div>
    </Modal>
  );
}

// ── Processing status strip ───────────────────────────────────────────────────

function ProcessingStatus({ video, processingPct, retryCount }: { video: Video; processingPct: number; retryCount: number }) {
  const stageLabels: Record<string, string> = {
    preparing: "Preparing…",
    downloading: "Downloading video…",
    extracting_audio: "Extracting audio…",
    transcribing_audio: "Transcribing audio…",
    diarizing_speakers: "Detecting speakers…",
    binding_speakers: "Matching speakers…",
    preparing_analysis: "Preparing analysis…",
    finding_moments: "Finding moments…",
    crafting_scenes: "Crafting scenes…",
    building_moments: "Building moments…",
    ranking_moments: "Ranking moments…",
    finalizing: "Finalizing clips…",
  };

  const stageLabel =
    stageLabels[video.processing_stage ?? ""] ??
    (video.status === "identifying" ? "Finding moments…" : "Analysing…");
  if (video.status === "pending") return (
    <span className="flex items-center gap-1.5 text-xs" style={{ color: "var(--yt-text-2)" }}>
      <Clock className="w-3.5 h-3.5 animate-pulse" /> Queued…
    </span>
  );

  if (video.status === "transcribing") return (
    <div className="flex flex-col gap-1">
      <span className="flex items-center justify-between text-xs" style={{ color: "var(--yt-text-2)" }}>
        <span className="flex items-center gap-1.5"><Clock className="w-3.5 h-3.5 animate-pulse" />{stageLabel}</span>
        <span>{Math.round(processingPct)}%</span>
      </span>
      <div className="h-1 rounded-full overflow-hidden" style={{ background: "var(--yt-surface-2)" }}>
        <div className="h-full rounded-full transition-all duration-1000 ease-out" style={{ width: `${processingPct}%`, background: "var(--yt-red)" }} />
      </div>
    </div>
  );

  if (video.status === "identifying") return (
    <div className="flex flex-col gap-1">
      <span className="flex items-center justify-between text-xs font-medium" style={{ color: "var(--yt-text-2)" }}>
        <span className="flex items-center gap-1.5"><Loader2 className="w-3.5 h-3.5 animate-spin" />{stageLabel}</span>
        <span>{Math.round(processingPct)}%</span>
      </span>
      <div className="h-1 rounded-full overflow-hidden" style={{ background: "var(--yt-surface-2)" }}>
        <div className="h-full rounded-full transition-all duration-1000 ease-out" style={{ width: `${processingPct}%`, background: "var(--yt-red)" }} />
      </div>
    </div>
  );

  if (video.status === "error") {
    if (retryCount < 2) return (
      <span className="flex items-center gap-1.5 text-xs" style={{ color: "var(--yt-text-2)" }}>
        <Loader2 className="w-3.5 h-3.5 animate-spin" /> Retrying… ({retryCount + 1}/2)
      </span>
    );
    return (
      <div className="flex items-start gap-2">
        <AlertCircle className="w-4 h-4 text-red-400 flex-shrink-0 mt-0.5" />
        <div>
          <p className="text-xs font-medium text-red-400">Processing failed</p>
          {video.error_msg && <p className="text-xs mt-0.5" style={{ color: "var(--yt-text-2)" }}>{video.error_msg}</p>}
        </div>
      </div>
    );
  }

  return null;
}

// ── Empty state ───────────────────────────────────────────────────────────────

function EmptyClips({ video, hasTranscript }: { video: Video; hasTranscript: boolean }) {
  return (
    <div className="flex flex-col items-center justify-center py-32" style={{ color: "var(--yt-text-2)" }}>
      {video.status === "identifying" ? (
        <>
          <Sparkles className="w-12 h-12 mb-4 opacity-40" />
          <p className="text-lg font-medium mb-2">Finding clips…</p>
          <p className="text-xs" style={{ color: "var(--yt-text-2)" }}>
            Clips will appear automatically
          </p>
        </>
      ) : (
        <>
          <Scissors className="w-12 h-12 mb-4 opacity-20" />
          <p className="text-lg font-medium mb-1">No clips yet</p>
          <p className="text-sm mb-4">
            {hasTranscript ? "Find the best moments." : "Analyse the video first."}
          </p>
          {hasTranscript && <IdentifyClipsButton videoId={video.id} />}
        </>
      )}
    </div>
  );
}

// ── Clip preview card ─────────────────────────────────────────────────────────

function ClipPreviewCard({
  clip,
  video,
  exports,
  exportStartedAt,
  onExport,
  onSeek,
  selected,
  onToggleSelect,
}: {
  clip: Clip;
  video: Video;
  exports: ExportInfo[];
  exportStartedAt: string | null;
  onExport: () => void;
  onSeek: (seconds: number) => void;
  selected?: boolean;
  onToggleSelect?: () => void;
}) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [playing, setPlaying] = useState(false);
  const [hookCopied, setHookCopied] = useState(false);
  const [showScript, setShowScript] = useState(false);
  const [showPublishingPack, setShowPublishingPack] = useState(false);
  const [packLoading, setPackLoading] = useState(false);
  const [packError, setPackError] = useState<string | null>(null);
  const [packCopied, setPackCopied] = useState<string | null>(null);
  const [titleOptions, setTitleOptions] = useState<string[]>(
    clip.social_title_options?.length
      ? clip.social_title_options
      : clip.social_title
        ? [clip.social_title]
        : [],
  );
  const [descriptionOptions, setDescriptionOptions] = useState<string[]>(
    clip.social_description_options?.length
      ? clip.social_description_options
      : clip.social_description
        ? [clip.social_description]
        : [],
  );
  const [overlayOptions, setOverlayOptions] = useState<string[]>(
    clip.overlay_text_options?.length
      ? clip.overlay_text_options
      : clip.overlay_text
        ? [clip.overlay_text]
        : [],
  );
  const [thumbSrc, setThumbSrc] = useState<string | null>(clip.thumbnail_url || video.thumbnail_url || null);
  const isPending = exportStartedAt !== null;

  const [exportPct, setExportPct] = useState(0);
  useEffect(() => {
    if (!isPending) { setExportPct(0); return; }
    const update = () => setExportPct(exportProgressPct(exportStartedAt));
    update();
    const iv = setInterval(update, 1000);
    return () => clearInterval(iv);
  }, [isPending, exportStartedAt]);

  const duration = clip.cuts.reduce((s, c) => s + (c.end - c.start), 0);
  const latestExport = exports[0] ?? null;

  // Version-aware download: multiple exports → picker first, then save dialog
  const [showVersions, setShowVersions] = useState(false);
  const [downloadTarget, setDownloadTarget] =
    useState<{ exportId: string; aspectRatio: string; versionLabel?: string } | null>(null);

  const startDownload = (exp: ExportInfo, label?: string) => {
    if (!exp?.r2_url) return;
    if (shouldSkipDownloadDialog()) {
      const name = defaultFilename(clip.hook ?? "clip", exp.aspect_ratio);
      downloadUrl(downloadHref(exp.id, name), name).catch(() => {});
    } else {
      setDownloadTarget({ exportId: exp.id, aspectRatio: exp.aspect_ratio, versionLabel: label });
    }
  };

  const togglePlay = (e: React.MouseEvent) => {
    e.stopPropagation();
    const v = videoRef.current;
    if (!v) return;
    v.paused ? v.play().catch(() => {}) : v.pause();
  };

  const generatePublishingPack = async () => {
    setPackLoading(true);
    setPackError(null);

    try {
      const res = await fetch(`/api/clips/${clip.id}/social-copy`, {
        method: "POST",
      });

      const data = await res.json();

      if (!res.ok) {
        throw new Error(data.error || "Publishing Pack generation failed");
      }

      setTitleOptions(data.titles ?? []);
      setDescriptionOptions(data.descriptions ?? []);
      setOverlayOptions(data.overlays ?? []);
      setShowPublishingPack(true);
    } catch (err) {
      setPackError(err instanceof Error ? err.message : "Publishing Pack generation failed");
      setShowPublishingPack(true);
    } finally {
      setPackLoading(false);
    }
  };

  const togglePublishingPack = () => {
    const hasPack =
      titleOptions.length > 0 ||
      descriptionOptions.length > 0 ||
      overlayOptions.length > 0;

    if (!hasPack) {
      generatePublishingPack();
      return;
    }

    setShowPublishingPack((v) => !v);
  };

  const copyPackText = (value: string, key: string) => {
    navigator.clipboard.writeText(value);
    setPackCopied(key);
    setTimeout(() => setPackCopied(null), 1800);
  };

  return (
    <div className="flex flex-col select-none gap-2">
      <div
        className="relative rounded-2xl overflow-hidden transition-transform duration-200 hover:scale-[1.01]"
        style={{
          aspectRatio: "9/16", background: "#000",
          border: selected ? "2px solid white" : "2px solid rgba(255,255,255,0.12)",
        }}
      >
        {isPending && <div className="clip-shimmer absolute inset-0 z-30 pointer-events-none" />}

        {/* Selection overlay (bulk export mode) — click anywhere to toggle */}
        {onToggleSelect && (
          <button
            onClick={(e) => { e.stopPropagation(); onToggleSelect(); }}
            className="absolute inset-0 z-40 cursor-pointer transition-colors"
            style={{ background: selected ? "rgba(255,255,255,0.18)" : "transparent" }}
            title={selected ? "Deselect clip" : "Select clip for export"}
            aria-pressed={selected}
          >
            {selected && (
              <span
                className="absolute top-2 left-2 w-6 h-6 rounded-full flex items-center justify-center"
                style={{ background: "white" }}
              >
                <CheckCheck className="w-3.5 h-3.5" style={{ color: "#0a0a0a" }} />
              </span>
            )}
          </button>
        )}
        {latestExport?.r2_url ? (
          <>
            <video
              ref={videoRef}
              src={latestExport.r2_url}
              poster={thumbSrc ?? undefined}
              preload="none"
              controls
              playsInline
              controlsList="nodownload nofullscreen noremoteplayback noplaybackrate"
              disablePictureInPicture
              onPlay={() => setPlaying(true)}
              onPause={() => setPlaying(false)}
              className="absolute inset-0 w-full h-full object-cover"
            />
            {/* Actions */}
            <div className="absolute top-2 right-2 z-20 flex items-center gap-1.5">
              {/* Transcript toggle — reveal the spoken text over the video */}
              <button
                onClick={(e) => { e.stopPropagation(); setShowScript((s) => !s); }}
                title={showScript ? "Hide transcript" : "Show transcript"}
                aria-pressed={showScript}
                className="w-9 h-9 flex items-center justify-center rounded-full transition-opacity hover:opacity-80"
                style={{
                  background: showScript ? "white" : "rgba(0,0,0,0.55)",
                  backdropFilter: "blur(4px)",
                  color: showScript ? "#0a0a0a" : "white",
                }}
              >
                <FileText className="w-4 h-4" />
              </button>
              {exports.length > 1 && (
                <button
                  onClick={(e) => { e.stopPropagation(); setShowVersions(true); }}
                  title={`Previous versions (${exports.length})`}
                  className="w-9 h-9 flex items-center justify-center rounded-full transition-opacity hover:opacity-80"
                  style={{ background: "rgba(0,0,0,0.55)", backdropFilter: "blur(4px)", color: "white" }}
                >
                  <History className="w-4 h-4" />
                </button>
              )}
              <button
                onClick={(e) => { e.stopPropagation(); startDownload(latestExport); }}
                title="Download latest"
                className="w-9 h-9 flex items-center justify-center rounded-full transition-opacity hover:opacity-80"
                style={{ background: "rgba(0,0,0,0.55)", backdropFilter: "blur(4px)", color: "white" }}
              >
                <Download className="w-4 h-4" />
              </button>
              <button
                onClick={(e) => { e.stopPropagation(); if (!isPending) onExport(); }}
                disabled={isPending}
                title="Re-clip"
                className={`h-9 flex items-center justify-center rounded-full transition-opacity hover:opacity-80 ${isPending ? "gap-1 px-3" : "w-9"}`}
                style={{ background: "rgba(0,0,0,0.55)", backdropFilter: "blur(4px)", color: "white" }}
              >
                {isPending ? (
                  <><Loader2 className="w-3.5 h-3.5 animate-spin" /> <span className="text-xs font-semibold">{Math.round(exportPct)}%</span></>
                ) : (
                  <RefreshCw className="w-3.5 h-3.5" />
                )}
              </button>
            </div>
            {/* Hook */}
            {clip.hook && (
              <div
                className="absolute bottom-0 left-0 right-0 z-20 px-4 pb-14 pt-20 group/hook transition-opacity duration-300"
                style={{ background: "linear-gradient(to top, rgba(0,0,0,0.88) 0%, rgba(0,0,0,0.55) 50%, transparent 100%)", opacity: playing ? 0 : 1, pointerEvents: playing ? "none" : "auto" }}
              >
                <div className="flex items-end gap-1.5">
                  <p className="flex-1 text-white font-bold text-sm leading-snug line-clamp-3">{clip.hook}</p>
                  <button
                    onClick={(e) => { e.stopPropagation(); navigator.clipboard.writeText(clip.hook!); setHookCopied(true); setTimeout(() => setHookCopied(false), 2000); }}
                    className="flex-shrink-0 text-white/40 hover:text-white opacity-100 lg:opacity-0 lg:group-hover/hook:opacity-100 lg:group-focus-within/hook:opacity-100 focus-visible:opacity-100 transition-opacity"
                  >
                    {hookCopied ? <CheckCheck className="w-3 h-3 text-green-400" /> : <Copy className="w-3 h-3" />}
                  </button>
                </div>
              </div>
            )}
            {/* Play/pause */}
            <button
              onClick={togglePlay}
              className={`absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 z-10 w-14 h-14 rounded-full flex items-center justify-center transition-opacity hover:opacity-100 ${playing ? "opacity-0" : "opacity-100"}`}
              style={{ background: "rgba(0,0,0,0.45)", backdropFilter: "blur(4px)" }}
            >
              {playing ? <Pause className="w-6 h-6 text-white" /> : <Play className="w-6 h-6 text-white ml-1" />}
            </button>
            {/* Transcript overlay — the spoken text + hook, toggled by the document button */}
            {showScript && (
              <div
                className="absolute inset-0 z-30 flex flex-col p-4"
                style={{ background: "rgba(0,0,0,0.92)", backdropFilter: "blur(2px)" }}
                onClick={(e) => e.stopPropagation()}
              >
                <div className="flex items-start gap-1.5 flex-shrink-0">
                  <p className="flex-1 min-w-0 text-white font-bold text-[15px] leading-snug line-clamp-3">
                    {clip.hook || "Untitled clip"}
                  </p>
                  <button
                    onClick={(e) => { e.stopPropagation(); setShowScript(false); }}
                    className="flex-shrink-0 text-white/60 hover:text-white transition-colors"
                    title="Hide transcript"
                  >
                    <X className="w-4 h-4" />
                  </button>
                </div>
                <div className="flex-1 min-h-0 overflow-y-auto no-scrollbar mt-3">
                  {clip.text ? (
                    <p className="text-[11px] leading-[1.9]" style={{ color: "var(--yt-text-2)" }}>{clip.text}</p>
                  ) : (
                    <p className="text-[11px] italic" style={{ color: "var(--yt-text-2)" }}>No transcript available</p>
                  )}
                </div>
              </div>
            )}
          </>
        ) : (
          <>
            {/* Thumbnail background — slightly blurred */}
            {thumbSrc ? (
              // eslint-disable-next-line @next/next/no-img-element
              <img
                src={thumbSrc}
                alt=""
                loading="lazy"
                className="absolute inset-0 w-full h-full object-cover"
                style={{ filter: "blur(8px) brightness(0.45)", transform: "scale(1.08)" }}
                onError={() => {
                  if (thumbSrc !== video.thumbnail_url && video.thumbnail_url) setThumbSrc(video.thumbnail_url);
                  else setThumbSrc(null);
                }}
              />
            ) : (
              <div className="absolute inset-0" style={{ background: "var(--yt-surface)" }} />
            )}

            {/* Content */}
            <div className="absolute inset-0 flex flex-col p-4">

              {/* Hook — up to 3 lines */}
              <div className="flex items-start gap-1.5 group/hook flex-shrink-0">
                <p className="flex-1 min-w-0 text-white font-bold text-[17px] leading-snug line-clamp-3">
                  {clip.hook || "Untitled clip"}
                </p>
                {clip.hook && (
                  <button
                    onClick={(e) => { e.stopPropagation(); navigator.clipboard.writeText(clip.hook!); setHookCopied(true); setTimeout(() => setHookCopied(false), 2000); }}
                    className="flex-shrink-0 mt-0.5 opacity-100 lg:opacity-0 lg:group-hover/hook:opacity-100 lg:group-focus-within/hook:opacity-100 focus-visible:opacity-100 transition-opacity"
                    style={{ color: "var(--yt-text-2)" }}
                  >
                    {hookCopied ? <CheckCheck className="w-3 h-3 text-green-400" /> : <Copy className="w-3 h-3" />}
                  </button>
                )}
              </div>

              {/* Timestamps + duration — between hook and script */}
              <div className="flex items-center gap-1.5 flex-wrap mt-2.5 flex-shrink-0">
                {clip.cuts.map((cut, i) => (
                  <button
                    key={i}
                    onClick={(e) => { e.stopPropagation(); onSeek(cut.start); }}
                    className="text-[10px] font-mono px-2.5 py-1 rounded-full cursor-pointer transition-opacity hover:opacity-100"
                    style={{ background: "rgba(255,255,255,0.18)", color: "white", opacity: 0.85 }}
                  >
                    {formatDuration(cut.start)} – {formatDuration(cut.end)}
                  </button>
                ))}
                <span
                  className="text-[10px] font-mono px-2.5 py-1 rounded-full flex-shrink-0"
                  style={{ background: "rgba(255,255,255,0.18)", color: "white", opacity: 0.85, border: "0.5px solid rgba(255,255,255,0.7)" }}
                >
                  {Math.round(duration)}s
                </span>
              </div>

              {/* Script — scrollable, generous line-height */}
              <div className="flex-1 min-h-0 overflow-y-auto no-scrollbar mt-2.5">
                {clip.text ? (
                  <p className="text-[11px] leading-[1.9]" style={{ color: "var(--yt-text-2)" }}>
                    {clip.text}
                  </p>
                ) : (
                  <p className="text-[11px] italic" style={{ color: "var(--yt-text-2)" }}>No transcript available</p>
                )}
              </div>

              {/* Bottom actions */}
              <div className="mt-3 flex-shrink-0 flex items-center justify-end gap-2">
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    togglePublishingPack();
                  }}
                  disabled={packLoading}
                  className="flex-shrink-0 flex items-center gap-1 px-3 py-1.5 rounded-full text-xs font-semibold transition-all disabled:opacity-50 active:scale-95"
                  style={{ background: "rgba(255,255,255,0.14)", color: "white" }}
                >
                  {packLoading ? (
                    <>
                      <Loader2 className="w-3 h-3 animate-spin" />
                      Generating…
                    </>
                  ) : (
                    <>
                      <FileText className="w-3 h-3" />
                      Publishing pack
                    </>
                  )}
                </button>

                <button
                  onClick={(e) => { e.stopPropagation(); if (!isPending) onExport(); }}
                  disabled={isPending}
                  className="flex-shrink-0 flex items-center gap-1 px-3 py-1.5 rounded-full text-xs font-semibold transition-all disabled:opacity-50 active:scale-95"
                  style={{ background: "white", color: "#0a0a0a" }}
                >
                  {isPending ? (
                    <><Loader2 className="w-3 h-3 animate-spin" /> {Math.round(exportPct)}%</>
                  ) : (
                    <><Scissors className="w-3 h-3" /> Clip</>
                  )}
                </button>
              </div>
            </div>
          </>
        )}

        {/* Score — subtle, bottom-left */}
        {typeof clip.score === "number" && (
          <span
            className="absolute bottom-2 left-2 z-20 px-2 py-0.5 rounded-full text-[10px] font-mono pointer-events-none"
            style={{ background: "rgba(0,0,0,0.5)", backdropFilter: "blur(4px)", color: "rgba(255,255,255,0.7)" }}
            title={clip.label ?? undefined}
          >
            {Math.round(clip.score * 100)}
          </span>
        )}
      </div>

      {showPublishingPack && (
        <div
          className="rounded-xl p-3 space-y-4"
          style={{
            background: "var(--yt-surface-2)",
            border: "1px solid var(--yt-border)",
          }}
        >
          <div className="flex items-center justify-between gap-2">
            <p className="text-xs font-semibold text-white">Publishing Pack</p>

            <button
              onClick={generatePublishingPack}
              disabled={packLoading}
              className="text-[10px] transition-opacity hover:opacity-100 disabled:opacity-40"
              style={{ color: "var(--yt-text-2)", opacity: 0.8 }}
            >
              {packLoading ? "Regenerating…" : "Regenerate"}
            </button>
          </div>

          {packError && (
            <p className="text-[11px] text-red-400">{packError}</p>
          )}

          {titleOptions.length > 0 && (
            <div>
              <p
                className="text-[10px] uppercase tracking-wide mb-1.5"
                style={{ color: "var(--yt-text-2)" }}
              >
                Titles
              </p>

              <div className="space-y-1.5">
                {titleOptions.map((value, i) => (
                  <div
                    key={`title-${i}`}
                    className="flex items-start gap-2 rounded-lg p-2"
                    style={{ background: "rgba(0,0,0,0.2)" }}
                  >
                    <p className="flex-1 text-xs text-white">{value}</p>

                    <button
                      onClick={() => copyPackText(value, `title-${i}`)}
                      className="flex-shrink-0"
                      style={{ color: "var(--yt-text-2)" }}
                    >
                      {packCopied === `title-${i}` ? (
                        <CheckCheck className="w-3.5 h-3.5 text-green-400" />
                      ) : (
                        <Copy className="w-3.5 h-3.5" />
                      )}
                    </button>
                  </div>
                ))}
              </div>
            </div>
          )}

          {descriptionOptions.length > 0 && (
            <div>
              <p
                className="text-[10px] uppercase tracking-wide mb-1.5"
                style={{ color: "var(--yt-text-2)" }}
              >
                Descriptions
              </p>

              <div className="space-y-1.5">
                {descriptionOptions.map((value, i) => (
                  <div
                    key={`description-${i}`}
                    className="flex items-start gap-2 rounded-lg p-2"
                    style={{ background: "rgba(0,0,0,0.2)" }}
                  >
                    <p
                      className="flex-1 text-[11px] leading-relaxed"
                      style={{ color: "var(--yt-text)" }}
                    >
                      {value}
                    </p>

                    <button
                      onClick={() => copyPackText(value, `description-${i}`)}
                      className="flex-shrink-0"
                      style={{ color: "var(--yt-text-2)" }}
                    >
                      {packCopied === `description-${i}` ? (
                        <CheckCheck className="w-3.5 h-3.5 text-green-400" />
                      ) : (
                        <Copy className="w-3.5 h-3.5" />
                      )}
                    </button>
                  </div>
                ))}
              </div>
            </div>
          )}

          {overlayOptions.length > 0 && (
            <div>
              <p
                className="text-[10px] uppercase tracking-wide mb-1.5"
                style={{ color: "var(--yt-text-2)" }}
              >
                Overlay Text
              </p>

              <div className="space-y-1.5">
                {overlayOptions.map((value, i) => (
                  <div
                    key={`overlay-${i}`}
                    className="flex items-center gap-2 rounded-lg p-2"
                    style={{ background: "rgba(0,0,0,0.2)" }}
                  >
                    <p className="flex-1 text-xs text-white">{value}</p>

                    <button
                      onClick={() => copyPackText(value, `overlay-${i}`)}
                      className="flex-shrink-0"
                      style={{ color: "var(--yt-text-2)" }}
                    >
                      {packCopied === `overlay-${i}` ? (
                        <CheckCheck className="w-3.5 h-3.5 text-green-400" />
                      ) : (
                        <Copy className="w-3.5 h-3.5" />
                      )}
                    </button>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {showVersions && (
        <VersionPicker
          versions={exports}
          showRatio
          onSelect={(exp, label) => {
            setShowVersions(false);
            startDownload(exp as ExportInfo, label);
          }}
          onClose={() => setShowVersions(false)}
        />
      )}

      {downloadTarget && (
        <DownloadDialog
          exportId={downloadTarget.exportId}
          hook={clip.hook ?? "clip"}
          aspectRatio={downloadTarget.aspectRatio}
          versionLabel={downloadTarget.versionLabel}
          onClose={() => setDownloadTarget(null)}
        />
      )}
    </div>
  );
}
