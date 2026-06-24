import type { CSSProperties } from "react";
import { Pause, Play, SkipBack, SkipForward } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { formatTime } from "@/lib/time";
import type { VoiceMessage } from "@/types";
import { AgentAvatar } from "./agent-avatar";

export function MiniTransport({
  current,
  paused,
  position,
  queueCount,
  onNext,
  onPositionChange,
  onPrevious,
  onTogglePause,
}: {
  current: VoiceMessage;
  paused: boolean;
  position: number;
  queueCount: number;
  onNext: () => void;
  onPositionChange: (seconds: number) => void;
  onPrevious: () => void;
  onTogglePause: () => void;
}) {
  return (
    <footer
      className="fixed bottom-0 left-1/2 z-20 grid w-[calc(100%_-_2rem)] max-w-[45rem] -translate-x-1/2 grid-cols-[minmax(0,1fr)_minmax(16rem,21rem)_minmax(0,1fr)] items-center gap-4 rounded-t-lg border border-b-0 border-border bg-background/85 px-4 py-3 shadow-2xl backdrop-blur supports-[backdrop-filter]:bg-background/75 max-sm:grid-cols-1"
      style={{ "--agent-color": current.color } as CSSProperties}
    >
      <div className="pause-stripes pointer-events-none absolute inset-0 hidden rounded-t-lg opacity-25 [[data-paused=true]_&]:block" />
      <div className="relative flex min-w-0 items-center gap-3">
        <AgentAvatar initial={current.initial} color={current.color} />
        <div className="min-w-0">
          <strong className="block truncate text-sm">{current.agentName}</strong>
          <span className="block truncate font-mono text-xs text-muted-foreground">
            {current.repo} / {current.branch} / {current.model}
          </span>
        </div>
      </div>
      <div className="relative grid min-w-0 gap-2">
        <div className="flex items-center justify-center gap-2">
          <Button aria-label="Previous message" size="icon" type="button" variant="outline" onClick={onPrevious}>
            <SkipBack size={16} />
          </Button>
          <Button
            aria-label={paused ? "Resume" : "Pause"}
            className="bg-foreground text-background hover:bg-foreground/90"
            size="player"
            type="button"
            onClick={onTogglePause}
          >
            {paused ? <Play size={20} /> : <Pause size={20} />}
          </Button>
          <Button aria-label="Next message" size="icon" type="button" variant="outline" onClick={onNext}>
            <SkipForward size={16} />
          </Button>
        </div>
        <div className="grid grid-cols-[2.3rem_minmax(0,1fr)_2.3rem] items-center gap-2 font-mono text-xs text-muted-foreground">
          <span>{formatTime(position)}</span>
          <Slider
            aria-label="Transport timeline"
            max={current.durationSeconds}
            min={0}
            step={1}
            style={{ "--agent-color": current.color } as CSSProperties}
            value={[position]}
            onValueChange={(value) => onPositionChange(value[0] ?? 0)}
          />
          <span className="text-right">{formatTime(current.durationSeconds)}</span>
        </div>
      </div>
      <div className="relative justify-self-end text-right max-sm:hidden">
        <strong className="block text-sm text-foreground">
          {paused
            ? "paused"
            : current.intent === "converse"
              ? "response needed"
              : "playing"}
        </strong>
        <span className="font-mono text-xs text-muted-foreground">{queueCount} waiting</span>
      </div>
    </footer>
  );
}
