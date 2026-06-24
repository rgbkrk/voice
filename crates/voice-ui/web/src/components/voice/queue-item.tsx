import type { CSSProperties, KeyboardEvent } from "react";
import { MessageCircle, Play } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import type { VoiceMessage } from "@/types";
import { AgentAvatar } from "./agent-avatar";
import { MessageIdentity } from "./message-identity";

const stateLabels = {
  skipped: "skipped",
} as const;

export function QueueItem({
  message,
  onSelect,
}: {
  message: VoiceMessage;
  onSelect: () => void;
}) {
  const isConverse = message.intent === "converse";
  const handleKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      onSelect();
    }
  };

  return (
    <div
      className={cn(
        "grid min-w-0 cursor-pointer grid-cols-[auto_minmax(0,1fr)_auto] items-center gap-3 rounded-lg border border-border bg-card p-3 text-left transition-colors hover:bg-secondary/70 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
        isConverse && "bg-[linear-gradient(90deg,rgba(230,164,95,0.08),transparent_36%),hsl(var(--card))]",
        message.state === "skipped" && "opacity-50",
      )}
      data-message-id={message.id}
      role="button"
      style={{ "--agent-color": message.color } as CSSProperties}
      tabIndex={0}
      onClick={onSelect}
      onKeyDown={handleKeyDown}
    >
      <AgentAvatar initial={message.initial} color={message.color} />
      <span className="min-w-0">
        <span className="flex min-w-0 items-center gap-2">
          <strong className={cn("text-sm", message.state === "skipped" && "line-through")}>
            {message.agentName}
          </strong>
        </span>
        <MessageIdentity message={message} />
        <span
          className={cn(
            "mt-1 block truncate text-sm text-muted-foreground",
            message.state === "skipped" && "line-through",
          )}
        >
          {message.message}
        </span>
      </span>
      <span className="flex flex-col items-end gap-2">
        {message.state in stateLabels ? (
          <span className="font-mono text-xs text-muted-foreground">
            {stateLabels[message.state as keyof typeof stateLabels]}
          </span>
        ) : null}
        <Button
          className={cn(
            "h-8 min-w-24 text-xs text-muted-foreground",
            isConverse &&
              "border-[hsl(var(--voice-converse)_/_0.45)] bg-[hsl(var(--voice-converse)_/_0.10)] text-[hsl(var(--voice-converse))] hover:bg-[hsl(var(--voice-converse)_/_0.16)] hover:text-[hsl(var(--voice-converse))]",
          )}
          size="sm"
          type="button"
          variant="outline"
          onClick={(event) => {
            event.stopPropagation();
            onSelect();
          }}
        >
          {isConverse ? <MessageCircle size={14} /> : <Play size={14} />}
          {isConverse ? "Respond" : "Play"}
        </Button>
      </span>
    </div>
  );
}
