import React, { useEffect, useState } from "react";
import { Loader2, AlertCircle, RotateCw } from "lucide-react";
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

    (async () => {
      await checkDaemon();
      await loadInitialState();
      subscription = subscribeToUpdates();
      setLoading(false);
    })();

    return () => {
      if (subscription) {
        subscription.unsubscribe();
      }
    };
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center p-8">
        <Loader2 className="w-5 h-5 animate-spin text-gray-400" />
      </div>
    );
  }

  if (!daemonRunning) {
    return (
      <div className="flex items-center justify-center p-8">
        <div className="text-center max-w-sm">
          <div className="inline-flex items-center justify-center w-12 h-12 rounded-full bg-red-100 mb-4">
            <AlertCircle className="w-6 h-6 text-red-600" />
          </div>
          <h2 className="text-sm font-semibold text-gray-900 mb-2">
            Daemon not running
          </h2>
          <p className="text-xs text-gray-500 mb-4">
            The voice daemon is not running. Start it with:
          </p>
          <code className="block px-3 py-2 bg-gray-100 rounded text-xs text-gray-700 font-mono mb-4">
            voice mcp
          </code>
          <button
            onClick={async () => {
              await checkDaemon();
              if (daemonRunning) {
                await loadInitialState();
              }
            }}
            className="inline-flex items-center gap-2 px-4 py-2 text-sm font-medium bg-blue-600 text-white hover:bg-blue-700 rounded-md transition-colors"
          >
            <RotateCw className="w-4 h-4" />
            Retry Connection
          </button>
        </div>
      </div>
    );
  }

  return <QueuePanel />;
}

export default App;
