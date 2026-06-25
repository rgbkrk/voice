import { useEffect } from "react";
import { MiniTransport } from "@/components/voice/mini-transport";
import { NowPlaying } from "@/components/voice/now-playing";
import { VoiceHeader } from "@/components/voice/voice-header";
import { VoiceQueue } from "@/components/voice/voice-queue";
import { voiceActions, voiceBridge } from "@/state/voice-bridge";
import { useVoiceSelector } from "@/state/react";
import type { VoiceMessage } from "@/types";

export default function App() {
  useEffect(() => {
    voiceBridge.start();
    return () => voiceBridge.stop();
  }, []);

  const messages = useVoiceSelector((state) => state.messages);
  const currentId = useVoiceSelector((state) => state.currentId);
  const position = useVoiceSelector((state) => state.positionSeconds);
  const paused = useVoiceSelector((state) => state.paused);
  const connected = useVoiceSelector((state) => state.connected);
  const ready = useVoiceSelector((state) => state.ready);
  const error = useVoiceSelector((state) => state.error);

  const current = messages.find((message) => message.id === currentId) ?? messages[0];
  const queuedMessages = current
    ? messages.filter((message) => message.id !== current.id)
    : [];

  if (!current) {
    return (
      <div className="min-h-screen text-foreground">
        <VoiceHeader waitingCount={0} />
        <main className="mx-auto grid w-[calc(100%_-_2rem)] max-w-[45rem] gap-8">
          <section className="rounded-lg border border-border bg-card p-5 text-muted-foreground">
            {connected && ready
              ? "No voice messages."
              : error ?? "Waiting for daemon state."}
          </section>
        </main>
      </div>
    );
  }

  return (
    <div className="min-h-screen pb-36 text-foreground" data-paused={paused}>
      <VoiceHeader waitingCount={queuedMessages.length} />
      <main className="mx-auto grid w-[calc(100%_-_2rem)] max-w-[45rem] gap-8">
        <NowPlaying
          message={current}
          position={position}
          onPositionChange={voiceActions.seek}
          onPrimaryAction={() => void activate(current)}
        />
        <VoiceQueue messages={queuedMessages} onSelect={(message) => void activate(message)} />
      </main>
      <MiniTransport
        current={current}
        paused={paused}
        position={position}
        queueCount={queuedMessages.length}
        onNext={() => void voiceActions.next()}
        onPositionChange={voiceActions.seek}
        onPrevious={() => void voiceActions.previous()}
        onTogglePause={() => {
          if (paused) {
            void voiceActions.play(current.id);
          } else {
            void voiceActions.pause();
          }
        }}
      />
    </div>
  );
}

function activate(message: VoiceMessage) {
  if (message.state === "listening") {
    return voiceActions.cancel(message.id);
  }

  return message.intent === "converse"
    ? voiceActions.respond(message.id)
    : voiceActions.play(message.id);
}
