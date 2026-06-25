import { voiceStore } from "./voice-store";
import type { UiCommandResult, UiEvent, UiSnapshot } from "@/types";

const SNAPSHOT_ENDPOINT = "/api/ui/snapshot";
const EVENTS_ENDPOINT = "/api/ui/events";
const COMMAND_ENDPOINT = "/api/ui/commands";

class VoiceBridge {
  private eventSource: EventSource | null = null;
  private generation = 0;
  private readonly audio = new Audio();
  private audioTrackId: string | null = null;

  constructor() {
    this.audio.addEventListener("loadedmetadata", () => {
      if (this.audioTrackId) {
        voiceStore.setDuration(this.audioTrackId, this.audio.duration);
      }
    });
    this.audio.addEventListener("timeupdate", () => {
      voiceStore.setPosition(this.audio.currentTime);
    });
    this.audio.addEventListener("ended", () => {
      const state = voiceStore.snapshot();
      const current = state.messages.find((message) => message.id === state.currentId);
      if (current) {
        voiceStore.setPosition(current.durationSeconds);
      }
      if (current?.intent === "converse") {
        voiceStore.setPlaybackPaused(true);
        void this.command("pause");
      } else {
        void this.next();
      }
    });
  }

  start() {
    const generation = ++this.generation;
    void this.bootstrap(generation);
  }

  stop() {
    this.generation += 1;
    this.eventSource?.close();
    this.eventSource = null;
    this.audio.pause();
  }

  async play(trackId?: string) {
    const result = await this.command("play", trackId);
    if (result.ok) {
      await this.playCurrentPrompt(result.trackId ?? trackId);
    }
    return result;
  }

  async respond(trackId?: string) {
    return this.command("respond", trackId);
  }

  async pause() {
    this.audio.pause();
    return this.command("pause");
  }

  async next() {
    const result = await this.command("next");
    if (result.ok) {
      await this.playCurrentPrompt(result.trackId);
    }
    return result;
  }

  async previous() {
    const result = await this.command("previous");
    if (result.ok) {
      await this.playCurrentPrompt(result.trackId);
    }
    return result;
  }

  async cancel(trackId?: string) {
    return this.command("cancel", trackId);
  }

  async clearRecent(trackId: string) {
    return this.command("clear-recent", trackId);
  }

  seek(seconds: number) {
    this.audio.currentTime = Math.max(0, seconds);
    voiceStore.setPosition(this.audio.currentTime);
  }

  private async bootstrap(generation: number) {
    try {
      const response = await fetch(SNAPSHOT_ENDPOINT, {
        headers: { accept: "application/json" },
      });
      if (!response.ok) {
        throw new Error(`snapshot failed: ${response.status}`);
      }
      const snapshot = (await response.json()) as UiSnapshot;
      if (generation !== this.generation) {
        return;
      }
      voiceStore.applySnapshot(snapshot);
      this.connectEvents(generation);
    } catch (error) {
      if (generation === this.generation) {
        voiceStore.setConnection(false, error instanceof Error ? error.message : String(error));
        window.setTimeout(() => {
          if (generation === this.generation) {
            void this.bootstrap(generation);
          }
        }, 1_500);
      }
    }
  }

  private connectEvents(generation: number) {
    this.eventSource?.close();
    const source = new EventSource(EVENTS_ENDPOINT);
    this.eventSource = source;

    source.onopen = () => {
      if (generation === this.generation) {
        voiceStore.setConnection(true);
      }
    };

    source.onerror = () => {
      source.close();
      if (generation === this.generation) {
        voiceStore.setConnection(false, "event stream disconnected");
        window.setTimeout(() => {
          if (generation === this.generation) {
            void this.bootstrap(generation);
          }
        }, 1_500);
      }
    };

    for (const eventName of [
      "snapshot",
      "track_upserted",
      "track_removed",
      "active_changed",
      "transport_changed",
      "command_result",
      "error",
    ]) {
      source.addEventListener(eventName, (event) => {
        if (generation !== this.generation) {
          return;
        }
        try {
          voiceStore.applyEvent(JSON.parse((event as MessageEvent).data) as UiEvent);
        } catch (error) {
          voiceStore.setConnection(true, error instanceof Error ? error.message : String(error));
        }
      });
    }
  }

  private async command(command: string, trackId?: string): Promise<UiCommandResult> {
    const response = await fetch(`${COMMAND_ENDPOINT}/${command}`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ trackId }),
    });
    if (!response.ok) {
      return {
        command,
        ok: false,
        trackId,
        message: `command failed: ${response.status}`,
      };
    }
    const result = (await response.json()) as UiCommandResult;
    if (!result.ok && result.message) {
      voiceStore.applyEvent({ type: "error", payload: { message: result.message } });
    }
    return result;
  }

  private async playCurrentPrompt(trackIdOverride?: string | null) {
    const state = voiceStore.snapshot();
    const trackId = trackIdOverride ?? state.currentId;
    const track = trackId ? state.snapshot.tracks[trackId] : undefined;
    const promptUrl = track?.audio.promptUrl;
    if (!promptUrl) {
      return;
    }

    if (!this.audio.src.endsWith(promptUrl)) {
      this.audio.src = promptUrl;
      this.audio.currentTime = 0;
      this.audioTrackId = trackId ?? null;
    }
    voiceStore.setPlaybackPaused(false);
    await this.audio.play();
  }
}

export const voiceBridge = new VoiceBridge();

export const voiceActions = {
  play: (trackId?: string) => voiceBridge.play(trackId),
  respond: (trackId?: string) => voiceBridge.respond(trackId),
  pause: () => voiceBridge.pause(),
  next: () => voiceBridge.next(),
  previous: () => voiceBridge.previous(),
  cancel: (trackId?: string) => voiceBridge.cancel(trackId),
  clearRecent: (trackId: string) => voiceBridge.clearRecent(trackId),
  seek: (seconds: number) => voiceBridge.seek(seconds),
};
