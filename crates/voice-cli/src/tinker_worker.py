"""Warm JSONL boundary around Tinker's Python-only public audio surface."""

from __future__ import annotations

import asyncio
import json
import sys
import time
import wave
from pathlib import Path

import tinker
from tinker_cookbook import model_info
from tinker_cookbook.renderers import get_renderer, get_text_content
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tml_renderers import chat


def emit(message: dict[str, object]) -> None:
    print(json.dumps(message, separators=(",", ":")), flush=True)


async def main() -> None:
    model = sys.argv[-1]
    started = time.perf_counter()
    renderer = get_renderer(
        model_info.get_recommended_renderer_name(model), get_tokenizer(model)
    )
    service = tinker.ServiceClient()
    sampler = await service.create_sampling_client_async(base_model=model)
    emit({"type": "ready", "startup_ms": (time.perf_counter() - started) * 1000})

    user = chat.Author(chat.AuthorKind.User)
    for line in sys.stdin:
        request: dict[str, object] = json.loads(line)
        request_id = int(request["id"])
        try:
            audio_path = Path(str(request["audio_path"]))
            with wave.open(str(audio_path), "rb") as wav:
                frames = wav.getnframes()
                sample_rate = wav.getframerate()
            messages = chat.MessageList(
                [
                    chat.Message(
                        content=chat.Text(str(request["instruction"])), author=user
                    ),
                    chat.Message(
                        content=chat.AudioPointer(
                            location=str(audio_path),
                            format=chat.AudioFormat.Wav,
                            num_frames=frames,
                            sample_rate=sample_rate,
                        ),
                        author=user,
                    ),
                ]
            )
            render_started = time.perf_counter()
            prompt = renderer.build_generation_prompt(messages)
            render_ms = (time.perf_counter() - render_started) * 1000
            sample_started = time.perf_counter()
            response = await sampler.sample_async(
                prompt=prompt,
                num_samples=1,
                sampling_params=tinker.SamplingParams(
                    max_tokens=int(request["max_tokens"]),
                    temperature=float(request["temperature"]),
                    stop=renderer.get_stop_sequences(),
                ),
            )
            sample_ms = (time.perf_counter() - sample_started) * 1000
            message, termination = renderer.parse_response(response.sequences[0].tokens)
            emit(
                {
                    "type": "result",
                    "id": request_id,
                    "text": get_text_content(message),
                    "termination": termination.value,
                    "audio_ms": frames / sample_rate * 1000,
                    "render_ms": render_ms,
                    "sample_ms": sample_ms,
                }
            )
        except Exception as error:
            emit({"type": "error", "id": request_id, "error": str(error)})


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as error:
        emit({"type": "error", "error": str(error)})
