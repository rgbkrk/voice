import { create } from "zustand";
import { invoke } from "@tauri-apps/api/core";
import { listen, UnlistenFn } from "@tauri-apps/api/event";
import { Observable, from, fromEvent, Subscription } from "rxjs";
import { map, catchError, tap } from "rxjs/operators";
import type { VoiceState, QueueItem } from "../types";

// Create observable from Tauri event listener
function createTauriEventObservable<T>(eventName: string): Observable<T> {
  return new Observable<T>((subscriber) => {
    let unlisten: UnlistenFn | null = null;

    listen<T>(eventName, (event) => {
      subscriber.next(event.payload);
    })
      .then((fn) => {
        unlisten = fn;
      })
      .catch((err) => {
        subscriber.error(err);
      });

    // Cleanup
    return () => {
      if (unlisten) {
        unlisten();
      }
    };
  });
}

interface AppStore {
  // State
  queueState: VoiceState | null;
  expandedItems: Set<string>;
  playingAudio: { queueId: string; part: "question" | "answer" } | null;
  daemonRunning: boolean;

  // Actions
  setQueueState: (state: VoiceState) => void;
  toggleExpanded: (queueId: string) => void;
  playQuestion: (queueId: string) => Promise<void>;
  playAnswer: (queueId: string) => Promise<void>;
  cancelItem: (queueId: string) => Promise<void>;
  checkDaemon: () => Promise<void>;
  loadInitialState: () => Promise<void>;
  subscribeToUpdates: () => Subscription;
}

export const useAppStore = create<AppStore>((set, get) => ({
  // Initial state
  queueState: null,
  expandedItems: new Set(),
  playingAudio: null,
  daemonRunning: false,

  // Set queue state from Tauri event
  setQueueState: (state: VoiceState) => {
    set({ queueState: state });
  },

  // Toggle item expansion
  toggleExpanded: (queueId: string) => {
    set((state) => {
      const newExpanded = new Set(state.expandedItems);
      if (newExpanded.has(queueId)) {
        newExpanded.delete(queueId);
      } else {
        newExpanded.add(queueId);
      }
      return { expandedItems: newExpanded };
    });
  },

  // Play question audio
  playQuestion: async (queueId: string) => {
    set({ playingAudio: { queueId, part: "question" } });
    try {
      await invoke("play_question", { queueId });
    } catch (error) {
      console.error("Failed to play question:", error);
      alert(`Failed to play question: ${error}`);
    } finally {
      set({ playingAudio: null });
    }
  },

  // Play answer audio
  playAnswer: async (queueId: string) => {
    set({ playingAudio: { queueId, part: "answer" } });
    try {
      await invoke("play_answer", { queueId });
    } catch (error) {
      console.error("Failed to play answer:", error);
      alert(`Failed to play answer: ${error}`);
    } finally {
      set({ playingAudio: null });
    }
  },

  // Cancel queue item
  cancelItem: async (queueId: string) => {
    try {
      const cancelled = await invoke<boolean>("cancel_item", { queueId });
      if (!cancelled) {
        alert("Item not found or already completed");
      }
    } catch (error) {
      console.error("Failed to cancel item:", error);
      alert(`Failed to cancel: ${error}`);
    }
  },

  // Check if daemon is running
  checkDaemon: async () => {
    try {
      const running = await invoke<boolean>("is_daemon_running");
      set({ daemonRunning: running });
    } catch (error) {
      console.error("Failed to check daemon:", error);
      set({ daemonRunning: false });
    }
  },

  // Load initial queue state
  loadInitialState: async () => {
    try {
      const state = await invoke<VoiceState>("get_queue_state");
      set({ queueState: state });
    } catch (error) {
      console.error("Failed to load initial state:", error);
    }
  },

  // Subscribe to queue-updated events using RxJS
  subscribeToUpdates: () => {
    const queueUpdates$ = createTauriEventObservable<VoiceState>("queue-updated");

    const subscription = queueUpdates$
      .pipe(
        tap((state) => {
          console.log("Queue updated:", state.status, "pending:", state.pending.length);
        })
      )
      .subscribe({
        next: (state) => get().setQueueState(state),
        error: (err) => console.error("Queue update stream error:", err),
      });

    return subscription;
  },
}));
