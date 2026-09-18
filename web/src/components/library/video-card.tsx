"use client";

import Link from "next/link";
import Image from "next/image";
import { Video, VideoStatus } from "@/lib/types";
import { formatDuration } from "@/lib/youtube";
import { Trash2, Loader2, RefreshCw, MoreVertical, FolderInput, Check } from "lucide-react";
import { useState, useRef, useEffect } from "react";
import { Collection } from "@/lib/types";
import { transcribingPct, identifyingPct } from "@/lib/processing-progress";
import { Modal } from "@/components/ui/modal";
import { Menu, MenuItem } from "@/components/ui/menu";

const STATUS_DOT: Record<VideoStatus, string> = {
  pending:      "bg-[var(--yt-text-2)]",
  transcribing: "bg-[var(--yt-red)] animate-pulse",
  identifying:  "bg-[var(--yt-red)] animate-pulse",
  ready:        "bg-green-500",
  error:        "bg-red-500",
};

const STATUS_LABELS: Record<VideoStatus, string> = {
  pending:      "Queued",
  transcribing: "Analysing…",
  identifying:  "Finding clips…",
  ready:        "Ready",
  error:        "Error",
};

interface Props {
  video:       Video;
  collections: Collection[];
  onDeleted:   () => void;
  onMoved:     (videoId: string, collectionId: string | null) => void;
}

export function VideoCard({ video, collections, onDeleted, onMoved }: Props) {
  const [showMove,             setShowMove]             = useState(false);
  const [reprocessing,         setReprocessing]         = useState(false);
  const [showReprocessConfirm, setShowReprocessConfirm] = useState(false);
  const [showDeleteConfirm,    setShowDeleteConfirm]    = useState(false);

  // Progress tracking
  const identifyStartRef = useRef<number | null>(null);
  const prevStatusRef    = useRef<VideoStatus>(video.status);
  const [, setProgressTick] = useState(0);

  if (video.status === "identifying" && prevStatusRef.current !== "identifying") {
    identifyStartRef.current = Date.now();
  }
  prevStatusRef.current = video.status;

  useEffect(() => {
    if (video.status !== "pending" && video.status !== "transcribing" && video.status !== "identifying") return;
    const iv = setInterval(() => setProgressTick((n) => n + 1), 1000);
    return () => clearInterval(iv);
  }, [video.status]);

  const isProcessing = video.status === "pending" || video.status === "transcribing" || video.status === "identifying";
  const progressPct = (() => {
    if (video.status === "pending") return 3;
    if (video.status === "transcribing") return transcribingPct(video.created_at);
    if (video.status === "identifying")  return identifyingPct(identifyStartRef.current);
    return 0;
  })();

  const handleDelete = async () => {
    setShowDeleteConfirm(false);
    await fetch(`/api/videos/${video.id}`, { method: "DELETE" });
    onDeleted();
  };

  const handleReprocess = async () => {
    setShowReprocessConfirm(false);
    setReprocessing(true);
    try {
      const endpoint = video.has_transcript
        ? `/api/videos/${video.id}/identify-clips`
        : `/api/videos/${video.id}/reprocess`;

      await fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(
          video.has_transcript
            ? { keep_existing: false }
            : { model_size: "large" }
        ),
      });
      onDeleted();
    } finally {
      setReprocessing(false);
    }
  };

  const cardBody = (
    <>
      {/* Thumbnail */}
      <div className="relative aspect-video rounded-xl overflow-hidden bg-[var(--yt-surface)]">
        {video.thumbnail_url ? (
          <Image
            src={video.thumbnail_url}
            alt={video.title}
            fill
            className="object-cover group-hover:scale-105 transition-transform duration-200"
            sizes="(max-width: 640px) 100vw, (max-width: 1024px) 50vw, 33vw"
          />
        ) : (
          <div className="absolute inset-0 flex items-center justify-center text-[var(--yt-text-2)] text-xs">
            No thumbnail
          </div>
        )}

        {/* Duration badge */}
        {video.duration_sec && (
          <span className="absolute bottom-1.5 right-1.5 text-[11px] font-medium bg-black/90 text-white px-1.5 py-0.5 rounded-sm">
            {formatDuration(video.duration_sec)}
          </span>
        )}

        {/* Status badge top-left — hide only when ready */}
        {video.status !== "ready" && (
          <span className="absolute top-1.5 left-1.5 flex items-center gap-1 text-[10px] font-medium bg-black/80 text-white px-2 py-0.5 rounded-sm">
            <span className={`w-1.5 h-1.5 rounded-full ${STATUS_DOT[video.status]}`} />
            {STATUS_LABELS[video.status]}{isProcessing ? ` · ${Math.round(progressPct)}%` : ""}
          </span>
        )}

        {/* Progress bar — bottom of thumbnail */}
        {isProcessing && (
          <div className="absolute bottom-0 left-0 right-0 h-1 bg-black/30">
            <div
              className="h-full transition-all duration-1000 ease-out"
              style={{
                width: `${progressPct}%`,
                background: "var(--yt-red)",
              }}
            />
          </div>
        )}
      </div>

      {/* Info row below thumbnail */}
      <div className="flex gap-2 mt-3">
        {/* Text */}
        <div className="flex-1 min-w-0">
          <p className="text-sm font-medium text-[var(--yt-text)] line-clamp-2 leading-snug">
            {video.title}
          </p>
          {video.channel_name && (
            <p className="text-xs text-[var(--yt-text-2)] mt-0.5 hover:text-[var(--yt-text)] transition-colors">
              {video.channel_name}
            </p>
          )}
          {(video.tags ?? []).length > 0 && (
            <div className="flex flex-wrap gap-1 mt-1">
              {(video.tags as string[]).slice(0, 3).map((tag) => (
                <span key={tag} className="text-[10px] text-[var(--yt-text-2)] bg-[var(--yt-surface)] px-1.5 py-0.5 rounded">
                  {tag}
                </span>
              ))}
            </div>
          )}
        </div>

        {/* 3-dot menu */}
        <div className="flex-shrink-0">
          <Menu
            ariaLabel="Video options"
            width={208}
            className="bg-[var(--yt-surface)] border border-[var(--yt-border)]"
            onClose={() => setShowMove(false)}
            buttonClassName="w-9 h-9 flex items-center justify-center rounded-full hover:bg-[var(--yt-surface-2)] opacity-100 lg:opacity-0 lg:group-hover:opacity-100 lg:group-focus-within:opacity-100 focus-visible:opacity-100 transition-all text-[var(--yt-text-2)]"
            button={<MoreVertical className="w-4 h-4" />}
          >
            {(video.status === "error" || video.has_transcript) && (
              <MenuItem onClick={() => setShowReprocessConfirm(true)} disabled={reprocessing}>
                {reprocessing ? <Loader2 className="w-4 h-4 animate-spin" /> : <RefreshCw className="w-4 h-4" />}
                Re-analyse
              </MenuItem>
            )}

            {/* Move to collection */}
            {collections.length > 0 && (
              <>
                <MenuItem keepOpen onClick={() => setShowMove((s) => !s)}>
                  <FolderInput className="w-4 h-4" />
                  Move to…
                </MenuItem>
                {showMove && (
                  <div className="px-2 pb-1">
                    <MenuItem size="sm" className="rounded-lg !text-[var(--yt-text-2)]"
                      onClick={() => onMoved(video.id, null)}>
                      {!video.collection_id && <Check className="w-3 h-3 text-green-400" />}
                      <span className={!video.collection_id ? "ml-0" : "ml-5"}>No collection</span>
                    </MenuItem>
                    {collections.map((c) => (
                      <MenuItem key={c.id} size="sm" className="rounded-lg"
                        onClick={() => onMoved(video.id, c.id)}>
                        {video.collection_id === c.id && <Check className="w-3 h-3 text-green-400" />}
                        <span className={video.collection_id === c.id ? "" : "ml-5"}>{c.name}</span>
                      </MenuItem>
                    ))}
                  </div>
                )}
                <div className="my-1 border-t" style={{ borderColor: "var(--yt-border)" }} />
              </>
            )}

            <MenuItem destructive onClick={() => setShowDeleteConfirm(true)}>
              <Trash2 className="w-4 h-4" />
              Delete
            </MenuItem>
          </Menu>
        </div>
      </div>
    </>
  );

  return (
    <>
    {isProcessing ? (
      // While the video is still being processed there are no clips to open,
      // so the card shows progress but isn't clickable.
      <div className="group flex flex-col">{cardBody}</div>
    ) : (
      <Link href={`/video/${video.id}`} className="group flex flex-col">{cardBody}</Link>
    )}

    {/* Delete confirmation modal */}
    {showDeleteConfirm && (
      <Modal onClose={() => setShowDeleteConfirm(false)} label="Delete video?"
        className="w-80 rounded-2xl p-6 shadow-2xl flex flex-col gap-4"
        style={{ background: "var(--yt-surface)", border: "1px solid var(--yt-border)" }}>
          <h3 className="text-base font-semibold" style={{ color: "var(--yt-text)" }}>
            Delete video?
          </h3>
          <p className="text-sm" style={{ color: "var(--yt-text-2)" }}>
            This will permanently delete <span className="font-medium" style={{ color: "var(--yt-text)" }}>{video.title}</span> and all its clips.
          </p>
          <div className="flex gap-3 justify-end">
            <button
              onClick={(e) => { e.preventDefault(); e.stopPropagation(); setShowDeleteConfirm(false); }}
              className="px-4 py-2 text-sm transition-colors"
              style={{ color: "var(--yt-text-2)" }}
            >
              Cancel
            </button>
            <button
              onClick={(e) => { e.preventDefault(); e.stopPropagation(); handleDelete(); }}
              className="px-4 py-2 text-sm font-medium rounded-full text-white"
              style={{ background: "#dc2626" }}
            >
              Delete
            </button>
          </div>
      </Modal>
    )}

    {/* Re-analyse confirmation modal */}
    {showReprocessConfirm && (
      <Modal onClose={() => setShowReprocessConfirm(false)} label="Re-analyse video?"
        className="w-80 rounded-2xl p-6 shadow-2xl flex flex-col gap-4"
        style={{ background: "var(--yt-surface)", border: "1px solid var(--yt-border)" }}>
          <h3 className="text-base font-semibold" style={{ color: "var(--yt-text)" }}>
            Re-analyse video?
          </h3>
          <p className="text-sm" style={{ color: "var(--yt-text-2)" }}>
            The existing transcript and all clips will be deleted and analysis will re-run from scratch.
          </p>
          <div className="flex gap-3 justify-end">
            <button
              onClick={(e) => { e.preventDefault(); e.stopPropagation(); setShowReprocessConfirm(false); }}
              className="px-4 py-2 text-sm transition-colors"
              style={{ color: "var(--yt-text-2)" }}
            >
              Cancel
            </button>
            <button
              onClick={(e) => { e.preventDefault(); e.stopPropagation(); handleReprocess(); }}
              className="px-4 py-2 text-sm font-medium rounded-full text-white"
              style={{ background: "var(--yt-red)" }}
            >
              Re-analyse
            </button>
          </div>
      </Modal>
    )}
    </>
  );
}
