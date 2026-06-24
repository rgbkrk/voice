import { AudioLines } from "lucide-react";

export function VoiceHeader({
  waitingCount,
}: {
  waitingCount: number;
}) {
  return (
    <header className="mx-auto flex w-[calc(100%_-_2rem)] max-w-[45rem] items-center justify-between gap-4 py-6">
      <div className="flex items-center gap-3">
        <div className="grid h-9 w-9 place-items-center rounded-md border border-border bg-secondary shadow-sm">
          <AudioLines size={18} />
        </div>
        <div>
          <h1 className="text-lg font-semibold leading-none">Voice</h1>
          <p className="mt-1 font-mono text-xs text-muted-foreground">{waitingCount} waiting</p>
        </div>
      </div>
    </header>
  );
}
