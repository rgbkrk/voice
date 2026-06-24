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
