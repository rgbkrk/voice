import type { VoiceMessage } from "@/types";
import { QueueItem } from "./queue-item";

export function VoiceQueue({
  messages,
  onSelect,
}: {
  messages: VoiceMessage[];
  onSelect: (message: VoiceMessage) => void;
}) {
  return (
    <section>
      <div className="mb-3 flex items-end justify-between gap-4">
        <div>
          <p className="mb-1 text-xs font-bold uppercase tracking-[0.12em] text-muted-foreground">
            Up next
          </p>
          <h2 className="text-lg font-semibold">{messages.length} messages</h2>
        </div>
        <span className="font-mono text-xs text-muted-foreground">
          ready when you are
        </span>
      </div>
      <div className="grid gap-2">
        {messages.map((message) => (
          <QueueItem key={message.id} message={message} onSelect={() => onSelect(message)} />
        ))}
      </div>
    </section>
  );
}
