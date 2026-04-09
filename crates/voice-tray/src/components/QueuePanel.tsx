import React from "react";
import { X, Loader2 } from "lucide-react";
import QueueItem from "./QueueItem";
import { useAppStore } from "../store";
import { invoke } from "@tauri-apps/api/core";

export default function QueuePanel() {
  const queueState = useAppStore((s) => s.queueState);

  const handleClose = async () => {
    await invoke("hide_window");
  };

  if (!queueState) {
    return (
      <div className="flex items-center justify-center p-8">
        <Loader2 className="w-5 h-5 animate-spin text-gray-400" />
      </div>
    );
  }

  const { current, pending, recent } = queueState;
  const hasPending = pending.length > 0;
  const hasRecent = recent.length > 0;

  return (
    <div className="queue-panel bg-white" style={{ minHeight: '400px', maxHeight: '600px', overflow: 'auto' }}>
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-gray-200">
        <div>
          <h1 className="text-sm font-semibold text-gray-900">Voice Queue</h1>
          <p className="text-xs text-gray-500 mt-0.5">
            {pending.length} pending · {recent.length} recent
          </p>
        </div>
        <button
          onClick={handleClose}
          className="p-1.5 text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded transition-colors"
          title="Close"
        >
          <X className="w-4 h-4" />
        </button>
      </div>

      {/* Current item */}
      {current && (
        <div>
          <div className="px-4 py-2 bg-blue-50 border-b border-blue-100">
            <p className="text-xs font-medium text-blue-700">Current</p>
          </div>
          <QueueItem item={current} />
        </div>
      )}

      {/* Pending items */}
      {hasPending && (
        <div>
          <div className="px-4 py-2 bg-gray-50 border-b border-gray-100">
            <p className="text-xs font-medium text-gray-600">Pending ({pending.length})</p>
          </div>
          {pending.map((item) => (
            <QueueItem key={item.id} item={item} />
          ))}
        </div>
      )}

      {/* Recent items */}
      {hasRecent && (
        <div>
          <div className="px-4 py-2 bg-gray-50 border-b border-gray-100">
            <p className="text-xs font-medium text-gray-600">Recent ({recent.length})</p>
          </div>
          {recent.map((item) => (
            <QueueItem key={item.id} item={item} />
          ))}
        </div>
      )}

      {/* Empty state */}
      {!current && !hasPending && !hasRecent && (
        <div className="flex flex-col items-center justify-center p-12 text-center">
          <div className="w-12 h-12 rounded-full bg-gray-100 flex items-center justify-center mb-3">
            <span className="text-xl text-gray-400">○</span>
          </div>
          <p className="text-sm font-medium text-gray-600">No queue items</p>
          <p className="text-xs text-gray-400 mt-1">
            Items will appear when agents make requests
          </p>
        </div>
      )}
    </div>
  );
}
