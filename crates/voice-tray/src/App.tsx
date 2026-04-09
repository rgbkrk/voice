import React, { useEffect, useState } from "react";
import QueuePanel from "./components/QueuePanel";
import { useAppStore } from "./store";

function App() {
  const loadInitialState = useAppStore((s) => s.loadInitialState);
  const subscribeToUpdates = useAppStore((s) => s.subscribeToUpdates);
  const checkDaemon = useAppStore((s) => s.checkDaemon);
  const daemonRunning = useAppStore((s) => s.daemonRunning);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let subscription: { unsubscribe: () => void } | null = null;

    // Initialize app
    (async () => {
      // Check if daemon is running
      await checkDaemon();

      // Load initial state
      await loadInitialState();

      // Subscribe to updates (now returns RxJS Subscription)
      subscription = subscribeToUpdates();

      setLoading(false);
    })();

    // Cleanup function
    return () => {
      if (subscription) {
        subscription.unsubscribe();
      }
    };
  }, []);

  if (loading) {
    return (
      <div className="waveform-bg" style={{ minHeight: '400px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <div style={{ textAlign: 'center' }}>
          <div className="audio-loader" style={{ justifyContent: 'center', marginBottom: '16px' }}>
            <span></span>
            <span></span>
            <span></span>
            <span></span>
            <span></span>
          </div>
          <p className="metadata">INITIALIZING...</p>
        </div>
      </div>
    );
  }

  if (!daemonRunning) {
    return (
      <div className="waveform-bg" style={{ minHeight: '400px', display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '20px' }}>
        <div style={{ textAlign: 'center', maxWidth: '320px' }}>
          <div style={{
            width: '60px',
            height: '60px',
            margin: '0 auto 20px',
            border: '2px solid var(--status-failed)',
            borderRadius: '50%',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            fontSize: '24px',
            color: 'var(--status-failed)',
            animation: 'pulse 2s ease-in-out infinite'
          }}>
            !
          </div>
          <h1 className="audio-header" style={{ fontSize: '14px', marginBottom: '12px', color: 'var(--status-failed)' }}>
            DAEMON OFFLINE
          </h1>
          <p className="metadata" style={{ color: 'var(--text-secondary)', marginBottom: '16px', lineHeight: '1.6' }}>
            THE VOICE DAEMON IS NOT RUNNING
            <br />
            START IT WITH:
          </p>
          <code style={{
            display: 'block',
            padding: '12px',
            background: 'var(--bg-elevated)',
            borderRadius: '4px',
            fontSize: '11px',
            fontFamily: 'JetBrains Mono, monospace',
            color: 'var(--cyan)',
            marginBottom: '16px',
            border: '1px solid var(--bg-tertiary)'
          }}>
            voice mcp
          </code>
          <button
            onClick={async () => {
              await checkDaemon();
              if (daemonRunning) {
                await loadInitialState();
              }
            }}
            className="btn btn-play"
            style={{ width: '100%' }}
          >
            ↻ RETRY CONNECTION
          </button>
        </div>
      </div>
    );
  }

  return (
    <QueuePanel />
  );
}

export default App;
