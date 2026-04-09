import React from "react";
import QueueItem from "./QueueItem";
import { useAppStore } from "../store";
import { invoke } from "@tauri-apps/api/core";

export default function QueuePanel() {
  const queueState = useAppStore((s) => s.queueState);

  const handleQuit = async () => {
    await invoke("quit_app");
  };

  if (!queueState) {
    return (
      <div className="waveform-bg" style={{ padding: '40px 20px', textAlign: 'center' }}>
        <div className="audio-loader">
          <span></span>
          <span></span>
          <span></span>
          <span></span>
          <span></span>
        </div>
        <p className="metadata" style={{ marginTop: '16px' }}>
          INITIALIZING QUEUE...
        </p>
      </div>
    );
  }

  const { current, pending, recent } = queueState;
  const hasPending = pending.length > 0;
  const hasRecent = recent.length > 0;

  return (
    <div className="queue-panel waveform-bg" style={{ minHeight: '400px', maxHeight: '600px', overflow: 'auto' }}>
      {/* Header */}
      <div style={{ padding: '20px 16px', borderBottom: '2px solid var(--bg-tertiary)', position: 'relative' }}>
        <h1 className="audio-header">VOICE QUEUE</h1>
        <div className="metadata" style={{ marginTop: '8px' }}>
          <span style={{ color: 'var(--status-queued)' }}>{pending.length} PENDING</span>
          <span style={{ margin: '0 8px', color: 'var(--text-tertiary)' }}>•</span>
          <span style={{ color: 'var(--status-completed)' }}>{recent.length} RECENT</span>
        </div>

        {/* Quit button */}
        <button
          onClick={handleQuit}
          className="btn btn-cancel"
          style={{
            position: 'absolute',
            top: '16px',
            right: '16px',
            padding: '6px 12px',
            fontSize: '10px',
            minWidth: 'auto'
          }}
          title="Quit Application"
        >
          ✕
        </button>
      </div>

      {/* Current item */}
      {current && (
        <div>
          <div className="section-header current">
            <span>▶</span>
            <span>CURRENT</span>
          </div>
          <QueueItem item={current} />
        </div>
      )}

      {/* Pending items */}
      {hasPending && (
        <div>
          <div className="section-header pending">
            <span>⏸</span>
            <span>PENDING ({pending.length})</span>
          </div>
          {pending.map((item, index) => (
            <QueueItem key={item.id} item={item} delay={index * 50} />
          ))}
        </div>
      )}

      {/* Recent items */}
      {hasRecent && (
        <div>
          <div className="section-header recent">
            <span>✓</span>
            <span>RECENT ({recent.length})</span>
          </div>
          {recent.map((item, index) => (
            <QueueItem key={item.id} item={item} delay={index * 50} />
          ))}
        </div>
      )}

      {/* Empty state */}
      {!current && !hasPending && !hasRecent && (
        <div style={{ padding: '80px 20px', textAlign: 'center' }}>
          <div style={{
            width: '60px',
            height: '60px',
            margin: '0 auto 20px',
            border: '2px solid var(--cyan)',
            borderRadius: '50%',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            fontSize: '24px',
            color: 'var(--cyan)',
            opacity: 0.3
          }}>
            ○
          </div>
          <p className="metadata" style={{ color: 'var(--text-secondary)', marginBottom: '8px' }}>
            NO QUEUE ITEMS
          </p>
          <p className="metadata" style={{ color: 'var(--text-tertiary)' }}>
            WAITING FOR AGENT REQUESTS...
          </p>
        </div>
      )}
    </div>
  );
}
