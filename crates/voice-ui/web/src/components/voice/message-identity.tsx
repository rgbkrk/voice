import type { VoiceMessage } from "@/types";

export function MessageIdentity({ message }: { message: VoiceMessage }) {
  return (
    <div className="flex min-w-0 flex-wrap items-center gap-x-2 gap-y-1 font-mono text-xs text-muted-foreground">
      <span className="text-foreground">{message.repo}</span>
      <span className="text-muted-foreground/60">/</span>
      <span className="text-blue-300">{message.branch}</span>
      <span className="text-muted-foreground/60">.</span>
      <span>{message.model}</span>
    </div>
  );
}
