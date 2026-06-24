import { cn } from "@/lib/utils";
import type { TranscriptionSegment as TranscriptionSegmentData } from "@/types";

export function Transcription({
  className,
  currentTime,
  onSeek,
  segments,
}: {
  className?: string;
  currentTime: number;
  onSeek?: (time: number) => void;
  segments: TranscriptionSegmentData[];
}) {
  const visibleSegments = segments.filter((segment) => segment.text.trim().length > 0);

  return (
    <div
      className={cn(
        "grid gap-2 rounded-md border border-border bg-secondary/20 px-4 py-3",
        className,
      )}
      data-slot="transcription"
    >
      {visibleSegments.map((segment, index) => (
        <TranscriptionSegment
          currentTime={currentTime}
          index={index}
          key={`${segment.startSecond}-${segment.endSecond}-${segment.text}`}
          segment={segment}
          onSeek={onSeek}
        />
      ))}
    </div>
  );
}

function TranscriptionSegment({
  currentTime,
  index,
  onSeek,
  segment,
}: {
  currentTime: number;
  index: number;
  onSeek?: (time: number) => void;
  segment: TranscriptionSegmentData;
}) {
  const isActive = currentTime >= segment.startSecond && currentTime < segment.endSecond;
  const isPast = currentTime >= segment.endSecond;
  const state = isActive ? "active" : isPast ? "past" : "future";

  return (
    <button
      aria-current={isActive ? "true" : undefined}
      className={cn(
        "min-w-0 rounded-sm py-0.5 text-left text-base leading-relaxed transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
        isActive && "font-semibold text-foreground",
        isPast && "text-muted-foreground",
        state === "future" && "text-muted-foreground/45",
        onSeek && "cursor-pointer hover:text-foreground",
      )}
      data-active={isActive ? "" : undefined}
      data-index={index}
      data-slot="transcription-segment"
      data-state={state}
      type="button"
      onClick={() => onSeek?.(segment.startSecond)}
    >
      {segment.text}
    </button>
  );
}
