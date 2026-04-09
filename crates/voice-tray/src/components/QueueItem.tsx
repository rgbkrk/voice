import React from "react";
import { ChevronRight, Play, X, Loader2 } from "lucide-react";
import type { QueueItem as QueueItemType } from "../types";
import { useAppStore } from "../store";
import { cn } from "../lib/utils";

interface Props {
  item: QueueItemType;
}

export default function QueueItem({ item }: Props) {
  const expandedItems = useAppStore((s) => s.expandedItems);
  const toggleExpanded = useAppStore((s) => s.toggleExpanded);
  const playQuestion = useAppStore((s) => s.playQuestion);
  const playAnswer = useAppStore((s) => s.playAnswer);
  const cancelItem = useAppStore((s) => s.cancelItem);
  const playingAudio = useAppStore((s) => s.playingAudio);

  const isExpanded = expandedItems.has(item.id);
  const isPlaying = playingAudio?.queueId === item.id;

  const formatTimestamp = (timestamp: number) => {
    const date = new Date(timestamp * 1000);
    const now = new Date();
    const diff = now.getTime() - date.getTime();
    const minutes = Math.floor(diff / 60000);

    if (minutes < 1) return "just now";
    if (minutes < 60) return `${minutes}m ago`;
    const hours = Math.floor(minutes / 60);
    if (hours < 24) return `${hours}h ago`;
    return date.toLocaleDateString();
  };

  const getStatusColor = () => {
    switch (item.status) {
      case "Processing": return "bg-blue-100 text-blue-700 border-blue-200";
      case "Completed": return "bg-green-100 text-green-700 border-green-200";
      case "Failed": return "bg-red-100 text-red-700 border-red-200";
      default: return "bg-gray-100 text-gray-700 border-gray-200";
    }
  };

  return (
    <div className="queue-item px-4 py-3" onClick={() => toggleExpanded(item.id)}>
      {/* Header */}
      <div className="flex items-start justify-between gap-2">
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1.5 flex-wrap">
            <span className={cn("inline-flex items-center px-2 py-0.5 rounded text-xs font-medium border", getStatusColor())}>
              {item.status}
            </span>
            {item.repo && (
              <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-purple-100 text-purple-700 border border-purple-200">
                {item.repo}
              </span>
            )}
            <span className="text-xs text-gray-500">
              {formatTimestamp(item.created_at)}
            </span>
          </div>

          <p className={cn(
            "text-sm text-gray-700 leading-relaxed",
            !isExpanded && "truncate"
          )}>
            {item.text_preview || item.method}
          </p>
        </div>

        <ChevronRight className={cn(
          "w-4 h-4 text-gray-400 transition-transform flex-shrink-0 mt-0.5",
          isExpanded && "rotate-90"
        )} />
      </div>

      {/* Expanded details */}
      {isExpanded && (
        <div className="mt-3 pt-3 border-t border-gray-100">
          {/* Result */}
          {item.result && (
            <div className="mb-3 p-3 bg-gray-50 rounded-md border border-gray-200">
              <p className="text-xs font-medium text-gray-600 mb-1">Result</p>
              <p className="text-sm text-gray-700 leading-relaxed">
                {item.result}
              </p>
            </div>
          )}

          {/* Audio controls */}
          <div className="flex gap-2 flex-wrap">
            <button
              onClick={(e) => {
                e.stopPropagation();
                playQuestion(item.id);
              }}
              disabled={isPlaying && playingAudio?.part === "question"}
              className={cn(
                "inline-flex items-center gap-1.5 px-3 py-1.5 text-sm font-medium rounded-md transition-colors",
                isPlaying && playingAudio?.part === "question"
                  ? "bg-blue-100 text-blue-700"
                  : "bg-blue-600 text-white hover:bg-blue-700",
                "disabled:opacity-50 disabled:cursor-not-allowed"
              )}
            >
              {isPlaying && playingAudio?.part === "question" ? (
                <>
                  <Loader2 className="w-3.5 h-3.5 animate-spin" />
                  <span>Playing...</span>
                </>
              ) : (
                <>
                  <Play className="w-3.5 h-3.5" />
                  <span>Question</span>
                </>
              )}
            </button>

            <button
              onClick={(e) => {
                e.stopPropagation();
                playAnswer(item.id);
              }}
              disabled={isPlaying && playingAudio?.part === "answer"}
              className={cn(
                "inline-flex items-center gap-1.5 px-3 py-1.5 text-sm font-medium rounded-md transition-colors",
                isPlaying && playingAudio?.part === "answer"
                  ? "bg-green-100 text-green-700"
                  : "bg-green-600 text-white hover:bg-green-700",
                "disabled:opacity-50 disabled:cursor-not-allowed"
              )}
            >
              {isPlaying && playingAudio?.part === "answer" ? (
                <>
                  <Loader2 className="w-3.5 h-3.5 animate-spin" />
                  <span>Playing...</span>
                </>
              ) : (
                <>
                  <Play className="w-3.5 h-3.5" />
                  <span>Answer</span>
                </>
              )}
            </button>

            {item.status === "Queued" && (
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  if (confirm(`Cancel "${item.text_preview || item.method}"?`)) {
                    cancelItem(item.id);
                  }
                }}
                className="inline-flex items-center gap-1.5 px-3 py-1.5 text-sm font-medium bg-red-600 text-white hover:bg-red-700 rounded-md transition-colors"
              >
                <X className="w-3.5 h-3.5" />
                <span>Cancel</span>
              </button>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
