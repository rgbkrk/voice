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
    // Initialize app
    (async () => {
      // Check if daemon is running
      await checkDaemon();

      // Load initial state
      await loadInitialState();

      // Subscribe to updates
      const unlisten = await subscribeToUpdates();

      setLoading(false);

      // Cleanup
      return () => {
        unlisten();
      };
    })();
  }, []);

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto mb-2"></div>
          <p className="text-gray-600 dark:text-gray-400">Loading...</p>
        </div>
      </div>
    );
  }

  if (!daemonRunning) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <h1 className="text-xl font-bold text-red-600 mb-2">
            Daemon Not Running
          </h1>
          <p className="text-gray-600 dark:text-gray-400 mb-4">
            The voice daemon is not running. Please start it first:
          </p>
          <code className="px-3 py-1 bg-gray-100 dark:bg-gray-800 rounded text-sm">
            voiced
          </code>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen flex items-center justify-center p-4">
      <QueuePanel />
    </div>
  );
}

export default App;
