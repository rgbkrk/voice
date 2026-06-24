import { useEffect, useMemo, useState } from "react";
import { voiceMessages } from "@/data";
import { fetchDaemonMessages } from "@/lib/daemon";
import type { VoiceMessage } from "@/types";
import { MiniTransport } from "@/components/voice/mini-transport";
import { NowPlaying } from "@/components/voice/now-playing";
import { VoiceHeader } from "@/components/voice/voice-header";
import { VoiceQueue } from "@/components/voice/voice-queue";

export default function App() {
  const [messages, setMessages] = useState<VoiceMessage[]>(voiceMessages);
  const [currentId, setCurrentId] = useState(voiceMessages[0].id);
  const [paused, setPaused] = useState(false);
  const [position, setPosition] = useState(voiceMessages[0].positionSeconds);

  useEffect(() => {
    let active = true;
    let timer: number | undefined;

    async function refreshDaemonState() {
      const daemonMessages = await fetchDaemonMessages();
      if (!active || !daemonMessages) {
        return false;
      }

      setMessages(daemonMessages);
      setCurrentId((value) => {
        const currentMessage = daemonMessages.find((message) => message.id === value);
        const nextMessage = currentMessage ?? daemonMessages[0];
        if (!currentMessage) {
          setPosition(nextMessage.positionSeconds);
        }
        return nextMessage.id;
      });
      return true;
    }

    void refreshDaemonState().then((connected) => {
      if (active && connected) {
        timer = window.setInterval(refreshDaemonState, 2_000);
      }
    });

    return () => {
      active = false;
      if (timer) {
        window.clearInterval(timer);
      }
    };
  }, []);

  const current = useMemo(
    () => messages.find((message) => message.id === currentId) ?? messages[0],
    [currentId, messages],
  );

  const queuedMessages = useMemo(
    () => messages.filter((message) => message.id !== current.id),
    [current.id, messages],
  );

  useEffect(() => {
    if (paused) {
      return;
    }

    const timer = window.setInterval(() => {
      setPosition((value) => {
        const nextValue = Math.min(value + 0.5, current.durationSeconds);
        if (nextValue >= current.durationSeconds) {
          setPaused(true);
        }
        return nextValue;
      });
    }, 500);

    return () => window.clearInterval(timer);
  }, [current.durationSeconds, current.id, paused]);

  function selectMessage(message: VoiceMessage) {
    setCurrentId(message.id);
    setPosition(message.positionSeconds);
    setPaused(false);
    setMessages((items) =>
      items.map((item) =>
        item.id === message.id && item.state !== "skipped"
          ? { ...item, state: "picked-up" }
          : item,
      ),
    );
  }

  function step(direction: 1 | -1) {
    const index = messages.findIndex((message) => message.id === current.id);
    const nextIndex = Math.min(Math.max(index + direction, 0), messages.length - 1);
    selectMessage(messages[nextIndex]);
  }

  return (
    <div className="min-h-screen pb-36 text-foreground" data-paused={paused}>
      <VoiceHeader
        waitingCount={queuedMessages.length}
      />
      <main className="mx-auto grid w-[calc(100%_-_2rem)] max-w-[45rem] gap-8">
        <NowPlaying
          message={current}
          position={position}
          onPositionChange={setPosition}
          onPrimaryAction={() => setPaused(false)}
        />
        <VoiceQueue messages={queuedMessages} onSelect={selectMessage} />
      </main>
      <MiniTransport
        current={current}
        paused={paused}
        position={position}
        queueCount={queuedMessages.length}
        onNext={() => step(1)}
        onPositionChange={setPosition}
        onPrevious={() => step(-1)}
        onTogglePause={() => setPaused((value) => !value)}
      />
    </div>
  );
}
