import { BehaviorSubject, Observable } from "rxjs";
import { distinctUntilChanged, map } from "rxjs/operators";
import { voiceMessages } from "@/data";
import type {
  TranscriptionSegment,
  UiEvent,
  UiSnapshot,
  UiTrack,
  UiTransport,
  VoiceMessage,
  VoiceState,
  VoiceUiState,
} from "@/types";

const EMPTY_SNAPSHOT: UiSnapshot = {
  connected: false,
  ready: false,
  daemonStatus: "disconnected",
  transport: {
    state: "idle",
    paused: true,
    positionSeconds: 0,
  },
  activeTrackId: null,
  queueIds: [],
  recentIds: [],
  tracks: {},
};

const initialState: VoiceUiState = {
  connected: false,
  ready: false,
  snapshot: EMPTY_SNAPSHOT,
  messages: voiceMessages,
  currentId: voiceMessages[0]?.id,
  respondingTrackId: undefined,
  positionSeconds: voiceMessages[0]?.positionSeconds ?? 0,
  paused: true,
  durationOverrides: {},
};

export class VoiceStore {
  private readonly stateSubject = new BehaviorSubject<VoiceUiState>(initialState);

  readonly state$ = this.stateSubject.asObservable();

  snapshot() {
    return this.stateSubject.value;
  }

  select<T>(selector: (state: VoiceUiState) => T): Observable<T> {
    return this.state$.pipe(map(selector), distinctUntilChanged());
  }

  applySnapshot(snapshot: UiSnapshot) {
    this.publish(fromSnapshot(snapshot, this.snapshot()));
  }

  applyEvent(event: UiEvent) {
    const state = this.snapshot();
    switch (event.type) {
      case "snapshot":
        this.applySnapshot(event.payload);
        break;
      case "track_upserted": {
        const snapshot = {
          ...state.snapshot,
          tracks: {
            ...state.snapshot.tracks,
            [event.payload.id]: event.payload,
          },
        };
        this.applySnapshot(snapshot);
        break;
      }
      case "track_removed": {
        const { [event.payload.id]: _removed, ...tracks } = state.snapshot.tracks;
        this.applySnapshot({
          ...state.snapshot,
          tracks,
          queueIds: state.snapshot.queueIds.filter((id) => id !== event.payload.id),
          recentIds: state.snapshot.recentIds.filter((id) => id !== event.payload.id),
          activeTrackId:
            state.snapshot.activeTrackId === event.payload.id
              ? null
              : state.snapshot.activeTrackId,
        });
        break;
      }
      case "active_changed":
        this.applySnapshot({
          ...state.snapshot,
          activeTrackId: event.payload.activeTrackId ?? null,
        });
        break;
      case "transport_changed":
        this.applySnapshot({
          ...state.snapshot,
          transport: event.payload,
        });
        break;
      case "command_result":
        if (event.payload.ok && event.payload.command === "respond" && event.payload.trackId) {
          this.publish({
            ...state,
            respondingTrackId: event.payload.trackId,
          });
        } else if (!event.payload.ok) {
          this.publish({ ...state, error: event.payload.message ?? "Command failed" });
        }
        break;
      case "error":
        this.publish({ ...state, error: event.payload.message });
        break;
    }
  }

  setConnection(connected: boolean, error?: string) {
    const state = this.snapshot();
    this.publish({
      ...state,
      connected,
      ready: connected && state.ready,
      error,
      snapshot: {
        ...state.snapshot,
        connected,
        ready: connected && state.snapshot.ready,
      },
    });
  }

  setPosition(seconds: number) {
    const state = this.snapshot();
    this.publish({
      ...state,
      positionSeconds: Math.max(0, seconds),
    });
  }

  setPlaybackPaused(paused: boolean) {
    const state = this.snapshot();
    this.publish({
      ...state,
      paused,
    });
  }

  setDuration(trackId: string, seconds: number) {
    if (!Number.isFinite(seconds) || seconds <= 0) {
      return;
    }
    const rounded = Math.max(1, Math.round(seconds));
    const state = this.snapshot();
    if (state.durationOverrides[trackId] === rounded) {
      return;
    }
    this.publish({
      ...state,
      durationOverrides: {
        ...state.durationOverrides,
        [trackId]: rounded,
      },
      messages: state.messages.map((message) =>
        message.id === trackId
          ? withDuration(message, rounded)
          : message,
      ),
    });
  }

  private publish(state: VoiceUiState) {
    this.stateSubject.next(state);
  }
}

export const voiceStore = new VoiceStore();

function fromSnapshot(snapshot: UiSnapshot, previous: VoiceUiState): VoiceUiState {
  const messages = messagesFromSnapshot(snapshot);
  const currentId =
    snapshot.activeTrackId && snapshot.tracks[snapshot.activeTrackId]
      ? snapshot.activeTrackId
      : messages[0]?.id;
  const current = messages.find((message) => message.id === currentId) ?? messages[0];
  const sameTrack = previous.currentId === currentId;

  return {
    connected: snapshot.connected,
    ready: snapshot.ready,
    snapshot,
    messages: messages.map((message) => {
      const duration = previous.durationOverrides[message.id];
      const withActualDuration = duration ? withDuration(message, duration) : message;
      return previous.respondingTrackId === message.id
        ? { ...withActualDuration, state: "listening" }
        : withActualDuration;
    }),
    currentId,
    respondingTrackId:
      previous.respondingTrackId && snapshot.tracks[previous.respondingTrackId]
        ? previous.respondingTrackId
        : undefined,
    positionSeconds: sameTrack
      ? previous.positionSeconds
      : current?.positionSeconds ?? snapshot.transport.positionSeconds,
    paused: snapshot.transport.paused,
    durationOverrides: previous.durationOverrides,
  };
}

function messagesFromSnapshot(snapshot: UiSnapshot): VoiceMessage[] {
  const ids = [
    ...(snapshot.activeTrackId ? [snapshot.activeTrackId] : []),
    ...snapshot.queueIds,
    ...snapshot.recentIds,
  ];
  const seen = new Set<string>();

  return ids
    .filter((id) => {
      if (seen.has(id)) {
        return false;
      }
      seen.add(id);
      return true;
    })
    .map((id) => snapshot.tracks[id])
    .filter((track): track is UiTrack => Boolean(track))
    .map(trackToMessage);
}

function trackToMessage(track: UiTrack): VoiceMessage {
  const text = track.prompt || track.answer || "Voice message";
  const durationSeconds = estimateDurationSeconds(text);

  return {
    id: track.id,
    agentName: track.agent.name,
    initial: track.agent.initial,
    color: track.agent.color,
    repo: track.agent.repo,
    branch: track.agent.branch,
    model: track.agent.model,
    voice: track.agent.session || "daemon",
    intent: track.intent === "respond" ? "converse" : "play",
    state: voiceStateForTrack(track),
    message: text,
    transcript: transcriptForText(text, durationSeconds),
    history: track.answer
      ? [
          {
            id: `${track.id}-answer`,
            speaker: "user",
            label: "You",
            at: track.completedAt ? new Date(track.completedAt * 1000).toISOString() : "",
            text: track.answer,
          },
        ]
      : [],
    durationSeconds,
    positionSeconds: 0,
  };
}

function withDuration(message: VoiceMessage, durationSeconds: number): VoiceMessage {
  return {
    ...message,
    durationSeconds,
    transcript: transcriptForText(message.message, durationSeconds),
  };
}

function voiceStateForTrack(track: UiTrack): VoiceState {
  switch (track.lifecycle) {
    case "failed":
    case "skipped":
      return "skipped";
    case "active":
    case "listening":
    case "completed":
      return "picked-up";
    default:
      return track.intent === "respond" ? "awaiting-user" : "queued";
  }
}

function transcriptForText(text: string, durationSeconds: number): TranscriptionSegment[] {
  const sentences = splitSentences(text);
  const segmentDuration = durationSeconds / sentences.length;

  return sentences.map((sentence, index) => ({
    startSecond: Math.round(index * segmentDuration),
    endSecond:
      index === sentences.length - 1
        ? durationSeconds
        : Math.max(Math.round((index + 1) * segmentDuration), index + 1),
    text: sentence,
  }));
}

function splitSentences(text: string): string[] {
  const matches = text.match(/[^.!?]+[.!?]?/g)?.map((item) => item.trim()).filter(Boolean);
  return matches && matches.length > 0 ? matches : [text || "Voice message"];
}

function estimateDurationSeconds(text: string): number {
  const words = text.trim().split(/\s+/).filter(Boolean).length;
  return Math.min(Math.max(Math.ceil(words * 0.55), 6), 45);
}

export function transportPosition(transport: UiTransport) {
  return transport.positionSeconds;
}
