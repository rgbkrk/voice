import React from "react";
import type { QueueItem as QueueItemType } from "../types";
import { useAppStore } from "../store";

interface Props {
  item: QueueItemType;
  delay?: number;
}

export default function QueueItem({ item, delay = 0 }: Props) {
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

    if (minutes < 1) return "< 1M";
    if (minutes < 60) return `${minutes}M`;
    const hours = Math.floor(minutes / 60);
    if (hours < 24) return `${hours}H`;
    return date.toLocaleDateString();
  };

  const getStatusStyle = () => {
    switch (item.status) {
      case "Processing":
        return { color: 'var(--status-processing)', borderColor: 'var(--status-processing)' };
      case "Completed":
        return { color: 'var(--status-completed)', borderColor: 'var(--status-completed)' };
      case "Failed":
        return { color: 'var(--status-failed)', borderColor: 'var(--status-failed)' };
      default:
        return { color: 'var(--status-queued)', borderColor: 'var(--status-queued)' };
    }
  };

  return (
    <div
      className="queue-item"
      style={{
        animationDelay: `${delay}ms`,
        cursor: 'pointer'
      }}
      onClick={() => toggleExpanded(item.id)}
    >
      {/* Header - always visible */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <div style={{ flex: 1, minWidth: 0 }}>
          {/* Status and metadata row */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px', flexWrap: 'wrap' }}>
            <span className="badge" style={getStatusStyle()}>
              {item.status}
            </span>
            {item.repo && (
              <span className="badge" style={{ color: 'var(--purple)', borderColor: 'var(--purple)' }}>
                {item.repo}
              </span>
            )}
            <span className="metadata">
              {formatTimestamp(item.created_at)}
            </span>
          </div>

          {/* Preview text */}
          <p style={{
            fontSize: '12px',
            color: 'var(--text-primary)',
            lineHeight: '1.5',
            overflow: 'hidden',
            textOverflow: 'ellipsis',
            whiteSpace: isExpanded ? 'normal' : 'nowrap'
          }}>
            {item.text_preview || item.method}
          </p>
        </div>

        <div className={`expand-indicator ${isExpanded ? 'expanded' : ''}`} style={{ marginLeft: '12px', flexShrink: 0 }}>
          ▶
        </div>
      </div>

      {/* Expanded details */}
      {isExpanded && (
        <div style={{
          marginTop: '16px',
          paddingTop: '16px',
          borderTop: '1px solid var(--bg-tertiary)',
          animation: 'slide-in 0.2s ease-out'
        }}>
          {/* Full text */}
          {item.text_preview && (
            <div style={{ marginBottom: '16px' }}>
              <p style={{
                fontSize: '12px',
                color: 'var(--text-secondary)',
                lineHeight: '1.6',
                whiteSpace: 'pre-wrap'
              }}>
                {item.text_preview}
              </p>
            </div>
          )}

          {/* Result/transcript */}
          {item.result && (
            <div style={{
              marginBottom: '16px',
              padding: '12px',
              background: 'var(--bg-elevated)',
              borderRadius: '4px',
              border: '1px solid var(--cyan)',
              boxShadow: '0 0 10px var(--cyan-glow)'
            }}>
              <p className="metadata" style={{ marginBottom: '8px', color: 'var(--cyan)' }}>
                → RESULT
              </p>
              <p style={{
                fontSize: '12px',
                color: 'var(--text-primary)',
                lineHeight: '1.6'
              }}>
                {item.result}
              </p>
            </div>
          )}

          {/* Audio controls */}
          <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
            <button
              onClick={(e) => {
                e.stopPropagation();
                playQuestion(item.id);
              }}
              disabled={isPlaying && playingAudio?.part === "question"}
              className={`btn btn-play ${isPlaying && playingAudio?.part === "question" ? 'vu-active' : ''}`}
            >
              {isPlaying && playingAudio?.part === "question" ? (
                <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                  <span>▶</span>
                  <span className="audio-loader" style={{ height: '12px' }}>
                    <span style={{ height: '4px' }}></span>
                    <span style={{ height: '8px' }}></span>
                    <span style={{ height: '6px' }}></span>
                  </span>
                </span>
              ) : (
                "▶ QUESTION"
              )}
            </button>

            <button
              onClick={(e) => {
                e.stopPropagation();
                playAnswer(item.id);
              }}
              disabled={isPlaying && playingAudio?.part === "answer"}
              className={`btn btn-stop ${isPlaying && playingAudio?.part === "answer" ? 'vu-active' : ''}`}
            >
              {isPlaying && playingAudio?.part === "answer" ? (
                <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                  <span>▶</span>
                  <span className="audio-loader" style={{ height: '12px' }}>
                    <span style={{ height: '4px' }}></span>
                    <span style={{ height: '8px' }}></span>
                    <span style={{ height: '6px' }}></span>
                  </span>
                </span>
              ) : (
                "▶ ANSWER"
              )}
            </button>

            {item.status === "Queued" && (
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  if (confirm(`CANCEL "${item.text_preview || item.method}"?`)) {
                    cancelItem(item.id);
                  }
                }}
                className="btn btn-cancel"
              >
                ✕ CANCEL
              </button>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
