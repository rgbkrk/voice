import type {
  DaemonQueueItem,
  DaemonState,
  TranscriptionSegment,
  VoiceIntent,
  VoiceMessage,
  VoiceState,
} from "@/types";

const DAEMON_STATUS_ENDPOINT = "/api/daemon/status";
const DEFAULT_COLORS = ["#82aaff", "#f0b878", "#e89a6a", "#9ad7c2", "#b58cff"];

export async function fetchDaemonMessages(
  endpoint = DAEMON_STATUS_ENDPOINT,
): Promise<VoiceMessage[] | null> {
  try {
    const response = await fetch(endpoint, {
      headers: { accept: "application/json" },
    });

    if (!response.ok) {
      return null;
    }

    const state = (await response.json()) as DaemonState;
    const messages = daemonStateToMessages(state);
    return messages.length > 0 ? messages : null;
  } catch {
    return null;
  }
}

export function daemonStateToMessages(state: DaemonState): VoiceMessage[] {
  const items = [
    ...(state.current ? [state.current] : []),
    ...state.pending,
    ...state.recent,
  ];

  return items.map((item, index) => daemonItemToMessage(item, index));
}

function daemonItemToMessage(item: DaemonQueueItem, index: number): VoiceMessage {
  const intent = intentForMethod(item.method);
  const text = item.text_preview || item.result || labelForMethod(item.method);
  const durationSeconds = estimateDurationSeconds(text);

  return {
    id: item.id,
    agentName: agentName(item.client_id),
    initial: agentInitial(item.client_id),
    color: DEFAULT_COLORS[index % DEFAULT_COLORS.length],
    repo: item.repo || "voice daemon",
    branch: item.method,
    model: item.client_id,
    voice: "daemon",
    intent,
    state: stateForItem(item, intent),
    message: text,
    transcript: transcriptForText(text, durationSeconds),
    history: [],
    durationSeconds,
    positionSeconds: item.status === "processing" ? Math.min(1, durationSeconds) : 0,
  };
}

function intentForMethod(method: string): VoiceIntent {
  return method === "converse" || method === "listen" ? "converse" : "play";
}

function stateForItem(item: DaemonQueueItem, intent: VoiceIntent): VoiceState {
  if (item.status === "failed") {
    return "skipped";
  }

  if (item.status === "processing" || item.status === "completed") {
    return "picked-up";
  }

  return intent === "converse" ? "awaiting-user" : "queued";
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
  return matches && matches.length > 0 ? matches : [text || "Daemon message"];
}

function estimateDurationSeconds(text: string): number {
  const words = text.trim().split(/\s+/).filter(Boolean).length;
  return Math.min(Math.max(Math.ceil(words * 0.55), 6), 45);
}

function labelForMethod(method: string): string {
  return method.replace(/_/g, " ");
}

function agentName(clientId: string): string {
  const first = clientId.split(/[._:-]/).find(Boolean) || clientId || "Daemon";
  return first.charAt(0).toUpperCase() + first.slice(1);
}

function agentInitial(clientId: string): string {
  return (clientId.trim().charAt(0) || "D").toUpperCase();
}
