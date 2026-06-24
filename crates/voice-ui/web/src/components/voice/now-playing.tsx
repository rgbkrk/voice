import type { CSSProperties } from "react";
import { Mic, Play } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Slider } from "@/components/ui/slider";
import { formatTime } from "@/lib/time";
import type { VoiceMessage } from "@/types";
import { AgentAvatar } from "./agent-avatar";
import { MessageIdentity } from "./message-identity";
import { Transcription } from "./transcription";

export function NowPlaying({
  message,
  position,
  onPositionChange,
  onPrimaryAction,
}: {
  message: VoiceMessage;
  position: number;
  onPositionChange: (seconds: number) => void;
  onPrimaryAction: () => void;
}) {
  const isConverse = message.intent === "converse";

  return (
    <section style={{ "--agent-color": message.color } as CSSProperties}>
      <div className="mb-3 flex items-end justify-between gap-4">
        <div>
          <p className="mb-1 text-xs font-bold uppercase tracking-[0.12em] text-muted-foreground">
            Now playing
          </p>
          <h2 className="text-lg font-semibold">Current message</h2>
        </div>
      </div>
      <Card className="relative overflow-hidden bg-card/95 shadow-[0_1px_0_rgba(255,255,255,0.04)_inset,0_18px_40px_-24px_rgba(0,0,0,0.8)]">
        <div className="absolute inset-x-0 top-0 h-0.5 bg-gradient-to-r from-transparent via-[var(--agent-color)] to-transparent opacity-70" />
        <CardHeader className="flex-row items-center gap-3 space-y-0 p-5">
          <AgentAvatar initial={message.initial} color={message.color} size="lg" />
          <div className="min-w-0 flex-1">
            <div className="flex min-w-0 items-center gap-2">
              <h3 className="truncate text-lg font-semibold">{message.agentName}</h3>
            </div>
            <MessageIdentity message={message} />
          </div>
          <Button className="hidden rounded-full font-mono text-xs sm:inline-flex" size="sm" variant="outline">
            <Play size={12} />
            {message.voice}
          </Button>
        </CardHeader>
        <CardContent className="space-y-4 p-5 pt-0">
          {message.transcript.length > 0 ? (
            <Transcription
              currentTime={position}
              segments={message.transcript}
              onSeek={onPositionChange}
            />
          ) : (
            <p className="text-[17px] leading-relaxed text-foreground">{message.message}</p>
          )}
          <Timeline
            color={message.color}
            duration={message.durationSeconds}
            position={position}
            onPositionChange={onPositionChange}
          />
          <Button
            className="h-12 w-full text-base text-neutral-950"
            style={{ backgroundColor: isConverse ? "hsl(var(--voice-converse))" : "hsl(var(--voice-play))" }}
            type="button"
            onClick={onPrimaryAction}
          >
            {isConverse ? <Mic size={17} /> : <Play size={17} />}
            {isConverse ? `Respond to ${message.agentName}` : `Play ${message.agentName}`}
          </Button>
        </CardContent>
      </Card>
    </section>
  );
}

export function Timeline({
  color,
  duration,
  position,
  onPositionChange,
}: {
  color: string;
  duration: number;
  position: number;
  onPositionChange: (seconds: number) => void;
}) {
  return (
    <div className="grid grid-cols-[2.5rem_minmax(0,1fr)_2.5rem] items-center gap-3 font-mono text-xs text-muted-foreground">
      <span>{formatTime(position)}</span>
      <Slider
        aria-label="Message timeline"
        max={duration}
        min={0}
        step={1}
        style={{ "--agent-color": color } as CSSProperties}
        value={[position]}
        onValueChange={(value) => onPositionChange(value[0] ?? 0)}
      />
      <span className="text-right">{formatTime(duration)}</span>
    </div>
  );
}
