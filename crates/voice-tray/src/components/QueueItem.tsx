import React from "react";
import type { QueueItem as QueueItemType } from "../types";
import { useAppStore } from "../store";

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
      case "Processing": return "bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200";
      case "Completed": return "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200";
      case "Failed": return "bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200";
      default: return "bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-200";
    }
  };

  return (
    <div className="queue-item">
      {/* Header - always visible */}
      <div
        className="flex items-center justify-between cursor-pointer"
        onClick={() => toggleExpanded(item.id)}
      >
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1">
            <span className={`badge ${getStatusColor()}`}>
              {item.status}
            </span>
            {item.repo && (
              <span className="badge bg-purple-100 text-purple-800 dark:bg-purple-900 dark:text-purple-200">
                {item.repo}
              </span>
            )}
            <span className="text-xs text-gray-500 dark:text-gray-400">
              {formatTimestamp(item.created_at)}
            </span>
          </div>

          <p className="text-sm truncate text-gray-700 dark:text-gray-300">
            {item.text_preview || item.method}
          </p>
        </div>

        <div className="ml-2 text-gray-400">
          {isExpanded ? "▼" : "▶"}
        </div>
      </div>

      {/* Expanded details */}
      {isExpanded && (
        <div className="mt-3 pt-3 border-t border-gray-200 dark:border-gray-700">
          {/* Full text */}
          {item.text_preview && (
            <div className="mb-3">
              <p className="text-sm text-gray-700 dark:text-gray-300 whitespace-pre-wrap">
                {item.text_preview}
              </p>
            </div>
          )}

          {/* Result/transcript */}
          {item.result && (
            <div className="mb-3 p-2 bg-gray-50 dark:bg-gray-750 rounded">
              <p className="text-xs font-semibold text-gray-600 dark:text-gray-400 mb-1">
                Result:
              </p>
              <p className="text-sm text-gray-700 dark:text-gray-300">
                {item.result}
              </p>
            </div>
          )}

          {/* Audio controls */}
          <div className="flex gap-2">
            <button
              onClick={(e) => {
                e.stopPropagation();
                playQuestion(item.id);
              }}
              disabled={isPlaying && playingAudio?.part === "question"}
              className="px-3 py-1.5 text-sm font-medium text-white bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400 rounded transition-colors"
            >
              {isPlaying && playingAudio?.part === "question" ? "Playing..." : "▶ Question"}
            </button>

            <button
              onClick={(e) => {
                e.stopPropagation();
                playAnswer(item.id);
              }}
              disabled={isPlaying && playingAudio?.part === "answer"}
              className="px-3 py-1.5 text-sm font-medium text-white bg-green-600 hover:bg-green-700 disabled:bg-gray-400 rounded transition-colors"
            >
              {isPlaying && playingAudio?.part === "answer" ? "Playing..." : "▶ Answer"}
            </button>

            {item.status === "Queued" && (
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  if (confirm(`Cancel "${item.text_preview || item.method}"?`)) {
                    cancelItem(item.id);
                  }
                }}
                className="px-3 py-1.5 text-sm font-medium text-white bg-red-600 hover:bg-red-700 rounded transition-colors"
              >
                Cancel
              </button>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
