// Types that mirror Rust backend types

export interface VoiceState {
  status: string;
  current: QueueItem | null;
  pending: QueueItem[];
  recent: QueueItem[];
  audio: Record<string, AudioInfo>;
}

export interface QueueItem {
  id: string;
  client_id: string;
  method: string;
  status: string;
  created_at: number;
  text_preview: string | null;
  result: string | null;
  repo: string | null;
  completed_at: number | null;
  auto_clear_at: number | null;
}

export interface AudioInfo {
  question_path: string | null;
  answer_path: string | null;
  duration_ms: number;
}
