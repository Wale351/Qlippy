"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import { Clip, Export } from "@/lib/types";
import { formatSeconds } from "@/lib/utils";
import { ExportDialog } from "./export-dialog";
import { DownloadDialog, shouldSkipDownloadDialog, defaultFilename, downloadHref } from "./download-dialog";
import { VersionPicker } from "./version-picker";
import { useExportTracker } from "@/lib/export-tracker";
import { exportProgressPct } from "@/lib/processing-progress";
import { groupExportsByRatio } from "@/lib/export-versions";
import { downloadUrl } from "@/lib/save-file";
import {
  Scissors, ChevronDown, ChevronUp, Lightbulb, Download,
  ThumbsUp, ThumbsDown, Sparkles, Copy, CheckCheck, Loader2, X, History, FileText,
} from "lucide-react";

const TONE_COLORS: Record<string, string> = {
  hook:          "bg-yellow-950 text-yellow-400",
  educational:   "bg-blue-950 text-blue-400",
  entertaining:  "bg-pink-950 text-pink-400",
  emotional:     "bg-emerald-950 text-emerald-400",
  cta:           "bg-orange-950 text-orange-400",
  story:         "bg-teal-950 text-teal-400",
  controversy:   "bg-red-950 text-red-400",
  myth_bust:     "bg-rose-950 text-rose-400",
  fact:          "bg-cyan-950 text-cyan-400",
  qa:            "bg-sky-950 text-sky-400",
};

const SCORE_COLOR = (score: number) => {
  if (score >= 0.8) return "bg-green-500";
  if (score >= 0.6) return "bg-yellow-500";
  return "bg-[#606060]";
};

interface Props {
  clip: Clip;
  videoId: string;
  selected?: boolean;
  onToggleSelect?: (id: string) => void;
}

type DoneExport = Pick<Export, "id" | "aspect_ratio" | "r2_url" | "status" | "created_at">;

export function ClipCard({ clip, videoId, selected, onToggleSelect }: Props) {
  const { trackExport }               = useExportTracker();
  const [expanded, setExpanded]       = useState(false);
  const [exportOpen, setExportOpen]   = useState(false);
  const [exports, setExports]         = useState<DoneExport[]>([]);
  const [approved, setApproved]       = useState<boolean | null>(clip.approved ?? null);
  const [socialTitle, setSocialTitle] = useState(clip.social_title ?? "");
  const [socialDesc, setSocialDesc]   = useState(clip.social_description ?? "");
  const [overlayText, setOverlayText] = useState(clip.overlay_text ?? "");
  const [titleOptions, setTitleOptions] = useState<string[]>(clip.social_title_options ?? []);
  const [descriptionOptions, setDescriptionOptions] = useState<string[]>(clip.social_description_options ?? []);
  const [overlayOptions, setOverlayOptions] = useState<string[]>(clip.overlay_text_options ?? []);
  const [genLoading, setGenLoading]   = useState(false);
  const [copied, setCopied]           = useState<string | null>(null);
  const [showSocial, setShowSocial]   = useState(false);
  const [showTranscript, setShowTranscript] = useState(false);
  const [downloadTarget, setDownloadTarget] = useState<{ exportId: string; aspectRatio: string; versionLabel?: string } | null>(null);
  const [versionPickerRatio, setVersionPickerRatio] = useState<string | null>(null);
  const [showAllVersions, setShowAllVersions] = useState(false);
  const [armedDeleteId, setArmedDeleteId] = useState<string | null>(null);

  const resumedRef = useRef(false);

  const fetchExports = useCallback(async () => {
    const res = await fetch(`/api/clips/${clip.id}/exports`);
    if (!res.ok) return;
    const data: DoneExport[] = await res.json();
    setExports(data.filter((e) => e.status === "done"));

    // On first fetch, resume tracking any active export that survived navigation
    if (!resumedRef.current) {
      resumedRef.current = true;
      const active = data.find((e) => e.status === "queued" || e.status === "processing");
      if (active) {
        setInProgressExportId(active.id);
        setExportStartedAt((active as { updated_at?: string }).updated_at ?? new Date().toISOString());
      }
    }
  }, [clip.id]);

  useEffect(() => { fetchExports(); }, [fetchExports]);

  // In-card export progress — startedAt is the source of truth (shared formula)
  const [inProgressExportId, setInProgressExportId] = useState<string | null>(null);
  const [exportStartedAt,    setExportStartedAt]    = useState<string | null>(null);
  // Start at 0 to match SSR; useEffect sets the real value after hydration
  const [exportPct,          setExportPct]          = useState(0);

  useEffect(() => {
    if (!inProgressExportId || !exportStartedAt) { setExportPct(0); return; }
    const update = () => setExportPct(exportProgressPct(exportStartedAt));
    update();
    const tickIv = setInterval(update, 1000);
    const pollIv = setInterval(async () => {
      try {
        const res = await fetch(`/api/exports/${inProgressExportId}`);
        if (!res.ok) return;
        const data = await res.json();
        if (data.status === "done" || data.status === "error") {
          setInProgressExportId(null);
          setExportStartedAt(null);
          if (data.status === "done") fetchExports();
        }
      } catch {}
    }, 3000);
    return () => { clearInterval(tickIv); clearInterval(pollIv); };
  }, [inProgressExportId, exportStartedAt, fetchExports]);

  const handleDeleteExport = async (exportId: string) => {
    await fetch(`/api/exports/${exportId}`, { method: "DELETE" });
    setExports((prev) => prev.filter((e) => e.id !== exportId));
  };

  // Two-step delete: first click arms (3s window), second click deletes.
  // Deleting an export is permanent and re-rendering may cost a credit.
  const armOrDeleteExport = (exportId: string) => {
    if (armedDeleteId === exportId) {
      setArmedDeleteId(null);
      handleDeleteExport(exportId);
    } else {
      setArmedDeleteId(exportId);
      setTimeout(() => setArmedDeleteId((curr) => (curr === exportId ? null : curr)), 3000);
    }
  };

  // Download flow: version chosen (or only one exists) → filename dialog,
  // unless the user opted into instant downloads with auto-generated names.
  const startDownload = (exp: DoneExport, versionLabel?: string) => {
    if (!exp.r2_url) return;
    if (shouldSkipDownloadDialog()) {
      const name = defaultFilename(clip.hook ?? "clip", exp.aspect_ratio);
      downloadUrl(downloadHref(exp.id, name), name).catch(() => {});
    } else {
      setDownloadTarget({ exportId: exp.id, aspectRatio: exp.aspect_ratio, versionLabel });
    }
  };

  const handleExportQueued = (exportId: string, label: string) => {
    trackExport(exportId, label);
    setInProgressExportId(exportId);
    setExportStartedAt(new Date().toISOString());
    setExportOpen(false);
  };

  const handleApprove = async (val: boolean) => {
    const next = approved === val ? null : val;
    setApproved(next);
    await fetch(`/api/clips/bulk-approve`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ clip_ids: [clip.id], approved: next }),
    });
  };

  const handleGenerateSocial = async () => {
    setGenLoading(true);
    try {
      const res = await fetch(`/api/clips/${clip.id}/social-copy`, { method: "POST" });
      if (res.ok) {
        const data = await res.json();
        setSocialTitle(data.title ?? "");
        setSocialDesc(data.description ?? "");
        setOverlayText(data.overlay ?? "");
        setTitleOptions(data.titles ?? []);
        setDescriptionOptions(data.descriptions ?? []);
        setOverlayOptions(data.overlays ?? []);
        setShowSocial(true);
      }
    } finally {
      setGenLoading(false);
    }
  };

  const copyText = (text: string, which: string) => {
    navigator.clipboard.writeText(text);
    setCopied(which);
    setTimeout(() => setCopied(null), 2000);
  };

  const totalDuration = clip.cuts.reduce((sum, c) => sum + (c.end - c.start), 0);
  const score = clip.score ?? 0;

  const displayTitles = titleOptions.length ? titleOptions : (socialTitle ? [socialTitle] : []);
  const displayDescriptions = descriptionOptions.length ? descriptionOptions : (socialDesc ? [socialDesc] : []);
  const displayOverlays = overlayOptions.length ? overlayOptions : (overlayText ? [overlayText] : []);

  return (
    <div className={`bg-[var(--yt-surface)] border rounded-xl p-5 hover:border-[var(--yt-hover)] transition-colors ${
      approved === true  ? "border-green-800" :
      approved === false ? "border-red-900"   :
      selected           ? "border-white" : "border-[var(--yt-border)]"
    }`}>
      <div className="flex items-start gap-4">
        {/* Select checkbox (shown when bulk mode is active) */}
        {onToggleSelect && (
          <button
            onClick={() => onToggleSelect(clip.id)}
            className={`flex-shrink-0 w-5 h-5 rounded border-2 transition-colors mt-0.5 ${
              selected
                ? "bg-white border-white"
                : "border-[var(--yt-border)] hover:border-[#606060]"
            }`}
          >
            {selected && <CheckCheck className="w-3 h-3 text-zinc-950 m-auto" />}
          </button>
        )}

        {/* Score ring + bar */}
        <div className="flex-shrink-0 flex flex-col items-center gap-1">
          <div className="w-12 h-12 rounded-full border-2 border-[var(--yt-border)] flex items-center justify-center">
            <span className="text-sm font-bold text-white">
              {clip.score ? Math.round(clip.score * 100) : "?"}
            </span>
          </div>
          <div className="w-12 h-1 bg-[var(--yt-surface-2)] rounded-full overflow-hidden">
            <div
              className={`h-full rounded-full transition-all ${SCORE_COLOR(score)}`}
              style={{ width: `${score * 100}%` }}
            />
          </div>
        </div>

        {/* Content */}
        <div className="flex-1 min-w-0">
          <div className="flex items-start justify-between gap-3">
            <div className="flex items-start gap-1.5 group/hook">
              <h3 className="font-semibold text-white line-clamp-2">
                {clip.hook || "Untitled clip"}
              </h3>
              {clip.hook && (
                <button
                  onClick={() => copyText(clip.hook!, "hook")}
                  className="flex-shrink-0 mt-0.5 text-[var(--yt-text-2)] hover:text-[var(--yt-text)] opacity-100 lg:opacity-0 lg:group-hover/hook:opacity-100 lg:group-focus-within/hook:opacity-100 focus-visible:opacity-100 transition-opacity"
                  title="Copy hook"
                >
                  {copied === "hook"
                    ? <CheckCheck className="w-3.5 h-3.5 text-green-400" />
                    : <Copy className="w-3.5 h-3.5" />}
                </button>
              )}
            </div>
            <div className="flex items-center gap-2 flex-shrink-0">
              {/* Transcript toggle — round on/off switch revealing the spoken text */}
              <button
                onClick={() => setShowTranscript((v) => !v)}
                aria-pressed={showTranscript}
                className={`flex-shrink-0 w-7 h-7 rounded-full border flex items-center justify-center transition-colors ${
                  showTranscript
                    ? "bg-white border-white text-zinc-950"
                    : "border-[var(--yt-border)] text-[var(--yt-text-2)] hover:text-[var(--yt-text)] hover:border-[#606060]"
                }`}
                title={showTranscript ? "Hide transcript" : "Show transcript"}
              >
                <FileText className="w-3.5 h-3.5" />
              </button>
              {exports.length > 1 && (
                <button
                  onClick={() => setShowAllVersions(true)}
                  className="text-[var(--yt-text-2)] hover:text-white transition-colors"
                  title={`Previous versions (${exports.length})`}
                >
                  <History className="w-3.5 h-3.5" />
                </button>
              )}
              {clip.label === "weak" && (
                // The Crafter passed on this one; it's kept on the Scout's span so the
                // user can judge. The reason is one hover away.
                <span
                  className="text-xs px-2 py-0.5 rounded-full font-medium bg-orange-950 text-orange-400 cursor-help"
                  title={clip.reasoning ?? "The clip crafter passed on this moment"}
                >
                  weak
                </span>
              )}
              {clip.tone && (
                <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${TONE_COLORS[clip.tone] ?? "bg-[var(--yt-surface-2)] text-[var(--yt-text-2)]"}`}>
                  {clip.tone}
                </span>
              )}
              <span className="text-xs text-[var(--yt-text-2)] bg-[var(--yt-surface-2)] px-2 py-0.5 rounded-full">
                {formatSeconds(totalDuration)}
              </span>
            </div>
          </div>

          {/* Cut timestamps */}
          <div className="flex flex-wrap gap-2 mt-2">
            {clip.cuts.map((cut, i) => (
              <span key={i} className="text-xs text-[var(--yt-text-2)] bg-[var(--yt-surface-2)] px-2 py-1 rounded-md font-mono">
                {formatSeconds(cut.start)} → {formatSeconds(cut.end)}
              </span>
            ))}
            {clip.cuts.length > 1 && (
              <span className="text-xs text-[var(--yt-text)] px-2 py-1">
                {clip.cuts.length} cuts stitched
              </span>
            )}
          </div>

          {/* Transcript panel — spoken text, toggled by the round document button */}
          {showTranscript && (
            <div className="mt-3 bg-[var(--yt-surface-2)]/50 border border-[var(--yt-border)] rounded-lg p-3">
              <p className="text-[10px] uppercase tracking-widest font-medium text-[var(--yt-text-2)] mb-1.5">
                {clip.hook || "Transcript"}
              </p>
              <p className="text-sm text-[var(--yt-text)] whitespace-pre-wrap leading-relaxed">
                {clip.text || "No transcript text available for this clip."}
              </p>
            </div>
          )}

          {/* Reasoning toggle */}
          {clip.reasoning && (
            <button
              onClick={() => setExpanded(!expanded)}
              className="flex items-center gap-1 mt-2 text-xs text-[var(--yt-text-2)] hover:text-[var(--yt-text)] transition-colors"
            >
              <Lightbulb className="w-3 h-3" />
              {expanded ? "Hide reasoning" : "Why this clip?"}
              {expanded ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
            </button>
          )}
          {expanded && clip.reasoning && (
            <p className="mt-2 text-sm text-[var(--yt-text-2)] italic border-l-2 border-[var(--yt-border)] pl-3">
              {clip.reasoning}
            </p>
          )}

          {/* Publishing Pack */}
          {showSocial && (displayTitles.length > 0 || displayDescriptions.length > 0 || displayOverlays.length > 0) && (
            <div className="mt-3 bg-[var(--yt-surface-2)]/50 border border-[var(--yt-border)] rounded-lg p-3 space-y-4">
              <div className="flex items-center justify-between">
                <p className="text-xs font-semibold text-white flex items-center gap-1.5">
                  <Sparkles className="w-3.5 h-3.5 text-yellow-400" />
                  Publishing Pack
                </p>
                <button
                  onClick={handleGenerateSocial}
                  disabled={genLoading}
                  className="text-[10px] text-[var(--yt-text-2)] hover:text-white disabled:opacity-50"
                >
                  {genLoading ? "Regenerating…" : "Regenerate"}
                </button>
              </div>

              {displayTitles.length > 0 && (
                <div>
                  <p className="text-[10px] uppercase tracking-wide text-[var(--yt-text-2)] mb-2">
                    Titles
                  </p>
                  <div className="space-y-1.5">
                    {displayTitles.map((title, i) => (
                      <div
                        key={`title-${i}`}
                        className="flex items-start justify-between gap-2 rounded-md bg-black/20 p-2"
                      >
                        <p className="text-xs text-white font-medium">{title}</p>
                        <button
                          onClick={() => copyText(title, `title-${i}`)}
                          className="text-[var(--yt-text-2)] hover:text-white flex-shrink-0"
                        >
                          {copied === `title-${i}` ? (
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

              {displayDescriptions.length > 0 && (
                <div>
                  <p className="text-[10px] uppercase tracking-wide text-[var(--yt-text-2)] mb-2">
                    Descriptions
                  </p>
                  <div className="space-y-1.5">
                    {displayDescriptions.map((description, i) => (
                      <div
                        key={`desc-${i}`}
                        className="flex items-start justify-between gap-2 rounded-md bg-black/20 p-2"
                      >
                        <p className="text-xs text-[var(--yt-text)] leading-relaxed">
                          {description}
                        </p>
                        <button
                          onClick={() => copyText(description, `desc-${i}`)}
                          className="text-[var(--yt-text-2)] hover:text-white flex-shrink-0"
                        >
                          {copied === `desc-${i}` ? (
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

              {displayOverlays.length > 0 && (
                <div>
                  <p className="text-[10px] uppercase tracking-wide text-[var(--yt-text-2)] mb-2">
                    Overlay Text
                  </p>
                  <div className="space-y-1.5">
                    {displayOverlays.map((overlay, i) => (
                      <div
                        key={`overlay-${i}`}
                        className="flex items-center justify-between gap-2 rounded-md bg-black/20 p-2"
                      >
                        <p className="text-xs text-white">{overlay}</p>
                        <button
                          onClick={() => copyText(overlay, `overlay-${i}`)}
                          className="text-[var(--yt-text-2)] hover:text-white flex-shrink-0"
                        >
                          {copied === `overlay-${i}` ? (
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

          {/* In-card export progress */}
          {inProgressExportId && (
            <div className="mt-3 flex flex-col gap-1.5">
              <div className="flex items-center justify-between text-xs text-[var(--yt-text-2)]">
                <span className="flex items-center gap-1.5">
                  <Loader2 className="w-3 h-3 animate-spin" />
                  Exporting…
                </span>
                <span className="font-semibold text-white">{Math.round(exportPct)}%</span>
              </div>
              <div className="h-1.5 rounded-full overflow-hidden bg-[var(--yt-surface-2)]">
                <div
                  className="h-full rounded-full transition-all duration-1000 ease-out"
                  style={{ width: `${exportPct}%`, background: "var(--yt-red)" }}
                />
              </div>
            </div>
          )}

          {/* Actions */}
          <div className="flex items-center gap-2 mt-4 flex-wrap">
            <button
              onClick={() => setExportOpen(true)}
              disabled={!!inProgressExportId}
              className="flex items-center gap-1.5 px-3 py-1.5 bg-white hover:bg-zinc-200 text-zinc-950 text-xs font-medium rounded-lg transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
            >
              <Scissors className="w-3 h-3" />
              {exports.length > 0 ? "Restyle" : "Clip"}
            </button>

            {/* Social copy */}
            <button
              onClick={socialTitle ? () => setShowSocial(!showSocial) : handleGenerateSocial}
              disabled={genLoading}
              className="flex items-center gap-1.5 px-3 py-1.5 bg-[var(--yt-surface-2)] hover:bg-[var(--yt-hover)] text-[var(--yt-text)] text-xs font-medium rounded-lg transition-colors disabled:opacity-50"
            >
              {genLoading ? <Loader2 className="w-3 h-3 animate-spin" /> : <Sparkles className="w-3 h-3 text-yellow-400" />}
              {genLoading ? "Generating…" : socialTitle ? (showSocial ? "Hide pack" : "Publishing pack") : "Publishing pack"}
            </button>

            {/* Approve / Reject */}
            <div className="flex items-center gap-1 ml-auto">
              <button
                onClick={() => handleApprove(true)}
                className={`p-1.5 rounded-lg transition-colors ${
                  approved === true
                    ? "bg-green-900 text-green-400"
                    : "text-[var(--yt-text-2)] hover:text-green-400 hover:bg-[var(--yt-hover)]"
                }`}
                title="Approve clip"
              >
                <ThumbsUp className="w-3.5 h-3.5" />
              </button>
              <button
                onClick={() => handleApprove(false)}
                className={`p-1.5 rounded-lg transition-colors ${
                  approved === false
                    ? "bg-red-900 text-red-400"
                    : "text-[var(--yt-text-2)] hover:text-red-400 hover:bg-[var(--yt-hover)]"
                }`}
                title="Reject clip"
              >
                <ThumbsDown className="w-3.5 h-3.5" />
              </button>
            </div>
          </div>

          {/* Completed exports — one button per aspect ratio; multiple
              versions of the same ratio open a version picker first */}
          {exports.length > 0 && (
            <div className="flex flex-wrap gap-2 mt-3 pt-3 border-t border-[var(--yt-border)]">
              {Object.entries(groupExportsByRatio(exports)).map(([ratio, versions]) => (
                <div key={ratio} className="flex items-center rounded-lg overflow-hidden bg-[var(--yt-surface-2)]">
                  <button
                    onClick={() =>
                      versions.length > 1
                        ? setVersionPickerRatio(ratio)
                        : startDownload(versions[0])
                    }
                    className="flex items-center gap-1.5 px-3 py-1.5 hover:bg-[var(--yt-hover)] text-[var(--yt-text)] text-xs font-medium transition-colors"
                  >
                    <Download className="w-3 h-3 text-green-400" />
                    {ratio}
                    {versions.length > 1 && (
                      <span className="text-[10px] text-[var(--yt-text)] font-semibold">×{versions.length}</span>
                    )}
                  </button>
                  {versions.length === 1 && (
                    <button
                      onClick={() => armOrDeleteExport(versions[0].id)}
                      className={`px-2 py-1.5 transition-colors border-l border-[var(--yt-border)] ${
                        armedDeleteId === versions[0].id
                          ? "bg-red-900/60 text-red-300"
                          : "hover:bg-red-900/50 text-[var(--yt-text-2)] hover:text-red-400"
                      }`}
                      title={armedDeleteId === versions[0].id ? "Click again to delete" : "Delete this export"}
                      aria-label={armedDeleteId === versions[0].id ? "Confirm delete export" : "Delete this export"}
                    >
                      {armedDeleteId === versions[0].id
                        ? <span className="text-[10px] font-semibold px-0.5">Delete?</span>
                        : <X className="w-3 h-3" />}
                    </button>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {exportOpen && (
        <ExportDialog
          clip={clip}
          initialConfig={null}
          restyle={exports.length > 0}
          onClose={() => setExportOpen(false)}
          onQueued={handleExportQueued}
        />
      )}

      {versionPickerRatio && (() => {
        const versions = groupExportsByRatio(exports)[versionPickerRatio] ?? [];
        if (versions.length === 0) return null;   // all versions deleted from the picker
        return (
          <VersionPicker
            versions={versions}
            aspectRatio={versionPickerRatio}
            onSelect={(exp, label) => {
              setVersionPickerRatio(null);
              startDownload(exp as DoneExport, label);
            }}
            onDelete={handleDeleteExport}
            onClose={() => setVersionPickerRatio(null)}
          />
        );
      })()}

      {showAllVersions && exports.length > 0 && (
        <VersionPicker
          versions={exports}
          showRatio
          onSelect={(exp, label) => {
            setShowAllVersions(false);
            startDownload(exp as DoneExport, label);
          }}
          onDelete={handleDeleteExport}
          onClose={() => setShowAllVersions(false)}
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
