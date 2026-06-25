export type VoiceIntent = "converse" | "play";
export type VoiceState = "queued" | "awaiting-user" | "picked-up" | "skipped";

export interface TranscriptionSegment {
  startSecond: number;
  endSecond: number;
  text: string;
}

export interface ConversationHistoryItem {
  id: string;
  speaker: "agent" | "user";
  label: string;
  at: string;
  text: string;
}

export interface VoiceMessage {
  id: string;
  agentName: string;
  initial: string;
  color: string;
  repo: string;
  branch: string;
  model: string;
  voice: string;
  intent: VoiceIntent;
  state: VoiceState;
  message: string;
  transcript: TranscriptionSegment[];
  history: ConversationHistoryItem[];
  durationSeconds: number;
  positionSeconds: number;
}

export type UiIntent = "play" | "respond";
export type UiLifecycle =
  | "queued"
  | "preparing"
  | "ready"
  | "active"
  | "listening"
  | "completed"
  | "failed"
  | "skipped";
export type UiTransportState = "idle" | "playing" | "paused" | "listening";

export interface UiAgent {
  name: string;
  initial: string;
  color: string;
  repo: string;
  branch: string;
  model: string;
  session: string;
}

export interface UiAudio {
  promptUrl?: string | null;
  answerUrl?: string | null;
}

export interface UiTrack {
  id: string;
  agent: UiAgent;
  intent: UiIntent;
  lifecycle: UiLifecycle;
  prompt: string;
  answer?: string | null;
  audio: UiAudio;
  createdAt: number;
  completedAt?: number | null;
}

export interface UiTransport {
  state: UiTransportState;
  paused: boolean;
  positionSeconds: number;
}

export interface UiSnapshot {
  connected: boolean;
  ready: boolean;
  daemonStatus: string;
  transport: UiTransport;
  activeTrackId?: string | null;
  queueIds: string[];
  recentIds: string[];
  tracks: Record<string, UiTrack>;
}

export type UiEvent =
  | { type: "snapshot"; payload: UiSnapshot }
  | { type: "track_upserted"; payload: UiTrack }
  | { type: "track_removed"; payload: { id: string } }
  | { type: "active_changed"; payload: { activeTrackId?: string | null } }
  | { type: "transport_changed"; payload: UiTransport }
  | { type: "command_result"; payload: UiCommandResult }
  | { type: "error"; payload: { message: string } };

export interface UiCommandResult {
  command: string;
  ok: boolean;
  trackId?: string | null;
  message?: string | null;
}

export interface VoiceUiState {
  connected: boolean;
  ready: boolean;
  error?: string;
  snapshot: UiSnapshot;
  messages: VoiceMessage[];
  currentId?: string;
  positionSeconds: number;
  paused: boolean;
}

export type DaemonItemStatus = "queued" | "processing" | "completed" | "failed";

export interface DaemonQueueItem {
  id: string;
  client_id: string;
  method: string;
  status: DaemonItemStatus;
  held_for_ui?: boolean;
  created_at: number;
  text_preview?: string;
  result?: string;
  repo?: string;
  completed_at?: number;
  auto_clear_at?: number;
}

export interface DaemonState {
  status: string;
  current?: DaemonQueueItem | null;
  pending: DaemonQueueItem[];
  recent: DaemonQueueItem[];
}
