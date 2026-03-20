#!/usr/bin/env python3
"""Voice conversation loop using the JSON-RPC server.

Alternates between listening (mic → STT) and speaking (TTS → speaker).
The agent responds with a simple echo/commentary for now — swap in an
LLM API call to make it actually intelligent.

Usage:
    python3 examples/conversation.py
    python3 examples/conversation.py --voice am_michael
    python3 examples/conversation.py --max-turns 5
"""

import argparse
import json
import subprocess
import sys
import time


class VoiceConversation:
    def __init__(self, voice: str = "af_heart", speed: float = 1.0):
        self.voice = voice
        self.speed = speed
        self._id = 0
        self._proc = None

    def start(self):
        self._proc = subprocess.Popen(
            ["voice", "-v", self.voice, "-s", str(self.speed), "--jsonrpc", "-q"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        resp = self.rpc("ping")
        if resp.get("result") != "pong":
            print(f"Server didn't respond to ping: {resp}", file=sys.stderr)
            sys.exit(1)

    def rpc(self, method: str, params: dict | None = None) -> dict:
        self._id += 1
        msg = {"jsonrpc": "2.0", "method": method, "id": self._id}
        if params:
            msg["params"] = params
        self._proc.stdin.write(json.dumps(msg) + "\n")
        self._proc.stdin.flush()
        # Read lines until we get our response (skip notifications)
        while True:
            line = self._proc.stdout.readline()
            if not line:
                return {"error": {"message": "Server closed"}}
            obj = json.loads(line)
            if "id" in obj and obj["id"] is not None:
                return obj

    def speak(self, text: str):
        print(f"\033[36m  🔊 {text}\033[0m")
        self.rpc("speak", {"text": text})

    def listen(
        self, max_duration_ms: int = 10000, silence_timeout_ms: int = 2000
    ) -> str:
        resp = self.rpc(
            "listen",
            {
                "max_duration_ms": max_duration_ms,
                "silence_timeout_ms": silence_timeout_ms,
            },
        )
        if "error" in resp:
            print(f"\033[31m  Error: {resp['error']['message']}\033[0m")
            return ""
        result = resp.get("result", {})
        text = result.get("text", "").strip()
        tokens = result.get("tokens", 0)
        duration = result.get("duration_ms", 0)
        if text:
            print(
                f"\033[33m  🎤 {text}\033[0m  \033[2m({tokens} tokens, {duration}ms)\033[0m"
            )
        else:
            print(f"\033[2m  (no speech detected)\033[0m")
        return text

    def shutdown(self):
        if self._proc:
            try:
                self._proc.stdin.close()
            except OSError:
                pass
            self._proc.wait(timeout=5)


def generate_response(user_text: str, turn: int) -> str:
    """Generate a response to the user's speech.

    This is a simple placeholder — replace with an LLM API call
    (OpenAI, Anthropic, local model, etc.) to make it intelligent.
    """
    text = user_text.lower()

    if not user_text:
        return "I didn't catch that. Could you say it again?"

    if any(w in text for w in ["hello", "hi", "hey"]):
        return f"Hey there! You said: {user_text}. What else is on your mind?"

    if any(w in text for w in ["bye", "goodbye", "quit", "exit", "stop"]):
        return "Goodbye! It was nice talking to you."

    if "?" in user_text:
        return (
            f"That's a great question. You asked: {user_text}. Let me think about that."
        )

    if len(user_text.split()) < 5:
        return f"You said {user_text}. Tell me more!"

    return f"Interesting! I heard you say: {user_text}. What would you like to talk about next?"


def main():
    parser = argparse.ArgumentParser(description="Voice conversation loop")
    parser.add_argument("-v", "--voice", default="af_heart", help="TTS voice")
    parser.add_argument("-s", "--speed", type=float, default=1.0, help="TTS speed")
    parser.add_argument(
        "--max-turns", type=int, default=10, help="Max conversation turns"
    )
    parser.add_argument(
        "--max-listen",
        type=int,
        default=10000,
        help="Max listen duration in ms",
    )
    parser.add_argument(
        "--silence",
        type=int,
        default=2000,
        help="Silence timeout in ms (auto-stop after this much quiet)",
    )
    args = parser.parse_args()

    conv = VoiceConversation(voice=args.voice, speed=args.speed)
    conv.start()

    print(f"\033[1mVoice Conversation\033[0m — {args.voice} ×{args.speed}")
    print(
        f"\033[2mMax {args.max_turns} turns, {args.max_listen}ms listen, {args.silence}ms silence timeout\033[0m"
    )
    print()

    # Opening greeting
    conv.speak("Hello! I'm listening. Go ahead and speak after the ding.")

    should_exit = False

    for turn in range(args.max_turns):
        print(f"\n\033[2m── Turn {turn + 1}/{args.max_turns} ──\033[0m")

        # Listen
        user_text = conv.listen(
            max_duration_ms=args.max_listen,
            silence_timeout_ms=args.silence,
        )

        if not user_text:
            conv.speak("I didn't hear anything. Let's try again.")
            continue

        # Check for exit
        if any(
            w in user_text.lower() for w in ["goodbye", "bye", "quit", "exit", "stop"]
        ):
            should_exit = True

        # Generate and speak response
        response = generate_response(user_text, turn)
        conv.speak(response)

        if should_exit:
            break

        # Brief pause between turns
        time.sleep(0.3)

    if not should_exit:
        conv.speak("That's all the turns we have. Thanks for chatting!")

    print()
    conv.shutdown()


if __name__ == "__main__":
    main()
