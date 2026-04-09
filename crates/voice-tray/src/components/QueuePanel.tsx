import React from "react";
import QueueItem from "./QueueItem";
import { useAppStore } from "../store";

export default function QueuePanel() {
  const queueState = useAppStore((s) => s.queueState);

  if (!queueState) {
    return (
      <div className="queue-panel p-4">
        <p className="text-center text-gray-500 dark:text-gray-400">
          Loading queue state...
        </p>
      </div>
    );
  }

  const { current, pending, recent } = queueState;
  const hasPending = pending.length > 0;
  const hasRecent = recent.length > 0;

  return (
    <div className="queue-panel">
      {/* Header */}
      <div className="p-4 border-b border-gray-200 dark:border-gray-700">
        <h2 className="text-lg font-semibold">Voice Queue</h2>
        <p className="text-sm text-gray-600 dark:text-gray-400">
          {pending.length} pending · {recent.length} recent
        </p>
      </div>

      {/* Current item */}
      {current && (
        <div className="border-b-2 border-blue-200 dark:border-blue-800">
          <div className="px-4 py-2 bg-blue-50 dark:bg-blue-950">
            <p className="text-xs font-semibold text-blue-600 dark:text-blue-400 uppercase">
              Current
            </p>
          </div>
          <QueueItem item={current} />
        </div>
      )}

      {/* Pending items */}
      {hasPending && (
        <div className="border-b-2 border-gray-200 dark:border-gray-700">
          <div className="px-4 py-2 bg-gray-50 dark:bg-gray-800">
            <p className="text-xs font-semibold text-gray-600 dark:text-gray-400 uppercase">
              Pending ({pending.length})
            </p>
          </div>
          {pending.map((item) => (
            <QueueItem key={item.id} item={item} />
          ))}
        </div>
      )}

      {/* Recent items */}
      {hasRecent && (
        <div>
          <div className="px-4 py-2 bg-gray-50 dark:bg-gray-800">
            <p className="text-xs font-semibold text-gray-600 dark:text-gray-400 uppercase">
              Recent ({recent.length})
            </p>
          </div>
          {recent.map((item) => (
            <QueueItem key={item.id} item={item} />
          ))}
        </div>
      )}

      {/* Empty state */}
      {!current && !hasPending && !hasRecent && (
        <div className="p-8 text-center">
          <p className="text-gray-500 dark:text-gray-400">
            No queue items yet
          </p>
          <p className="text-sm text-gray-400 dark:text-gray-500 mt-1">
            Items will appear here when agents make requests
          </p>
        </div>
      )}
    </div>
  );
}
