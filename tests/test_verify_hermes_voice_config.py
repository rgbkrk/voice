#!/usr/bin/env python3

from pathlib import Path
import subprocess
import tempfile
import textwrap
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "verify_hermes_voice_config.py"


def hermes_config(
    command: str = (
        "voice say --format ogg-opus --input-file {input_path} "
        "--output {output_path} --voice {voice} --speed {speed}"
    ),
    stt_command: str = "voice stream-transcribe --quiet {input_path}",
) -> str:
    return textwrap.dedent(
        f"""
        tts:
          provider: kokoro
          providers:
            kokoro:
              type: command
              command: {command}
              output_format: ogg
              voice_compatible: true
              voice: af_heart
              speed: 1.0
              timeout: 30
        stt:
          enabled: true
          provider: voice
          providers:
            voice:
              type: command
              command: {stt_command}
              format: txt
              timeout: 30
        """
    )


def write_fake_voice(directory: Path) -> Path:
    voice_path = directory / "voice"
    log_path = directory / "voice-args.txt"
    voice_path.write_text(
        textwrap.dedent(
            f"""\
            #!/usr/bin/env bash
            set -euo pipefail
            printf '%s\\n' "$@" > {str(log_path)!r}
            out=''
            while [[ $# -gt 0 ]]; do
              if [[ "$1" == "--output" ]]; then
                out="$2"
                shift 2
              else
                shift
              fi
            done
            [[ -n "$out" ]] || exit 64
            printf 'OggSfake-opus-payload' > "$out"
            """
        ),
        encoding="utf-8",
    )
    voice_path.chmod(0o755)
    return voice_path


def write_fake_ffprobe(directory: Path) -> Path:
    ffprobe_path = directory / "ffprobe"
    ffprobe_path.write_text(
        textwrap.dedent(
            """\
            #!/usr/bin/env bash
            set -euo pipefail
            cat <<'JSON'
            {"streams":[{"codec_name":"opus","sample_rate":"48000","channels":1}]}
            JSON
            """
        ),
        encoding="utf-8",
    )
    ffprobe_path.chmod(0o755)
    return ffprobe_path


class HermesVoiceConfigVerifierTests(unittest.TestCase):
    def test_verifier_accepts_voice_native_config_without_smoke(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "config.yaml"
            config.write_text(hermes_config(), encoding="utf-8")

            result = subprocess.run(
                [
                    str(SCRIPT_PATH),
                    "--config",
                    str(config),
                    "--skip-tts-smoke",
                ],
                check=True,
                capture_output=True,
                text=True,
            )

        self.assertIn("ok: Hermes voice config verifier passed", result.stdout)
        self.assertIn("tts.provider=kokoro", result.stdout)
        self.assertIn("tts.command=voice say --format ogg-opus", result.stdout)
        self.assertIn("stt.command=voice stream-transcribe --quiet", result.stdout)

    def test_verifier_executes_configured_tts_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            bin_dir = tmp_path / "bin"
            bin_dir.mkdir()
            voice_bin = write_fake_voice(bin_dir)
            write_fake_ffprobe(bin_dir)
            config = tmp_path / "config.yaml"
            config.write_text(hermes_config(), encoding="utf-8")

            result = subprocess.run(
                [
                    str(SCRIPT_PATH),
                    "--config",
                    str(config),
                    "--voice-bin",
                    str(voice_bin),
                ],
                check=True,
                capture_output=True,
                text=True,
                env={"PATH": f"{bin_dir}:{os_path()}"},
            )

            voice_args = (bin_dir / "voice-args.txt").read_text(encoding="utf-8")

        self.assertIn("tts_smoke=checked", result.stdout)
        self.assertIn("tts_probe=codec=opus,sample_rate=48000,channels=1", result.stdout)
        self.assertIn("say\n", voice_args)
        self.assertIn("--format\nogg-opus\n", voice_args)
        self.assertIn("--voice\naf_heart\n", voice_args)

    def test_verifier_rejects_wav_output_format(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "config.yaml"
            config.write_text(
                hermes_config().replace("output_format: ogg", "output_format: wav"),
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    str(SCRIPT_PATH),
                    "--config",
                    str(config),
                    "--skip-tts-smoke",
                ],
                check=False,
                capture_output=True,
                text=True,
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("must use output_format: ogg", result.stderr)

    def test_verifier_rejects_non_voice_compatible_provider(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "config.yaml"
            config.write_text(
                hermes_config().replace("voice_compatible: true", "voice_compatible: false"),
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    str(SCRIPT_PATH),
                    "--config",
                    str(config),
                    "--skip-tts-smoke",
                ],
                check=False,
                capture_output=True,
                text=True,
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("must set voice_compatible: true", result.stderr)

    def test_verifier_rejects_stt_without_quiet(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "config.yaml"
            config.write_text(
                hermes_config().replace(
                    "voice stream-transcribe --quiet {input_path}",
                    "voice stream-transcribe {input_path}",
                ),
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    str(SCRIPT_PATH),
                    "--config",
                    str(config),
                    "--skip-tts-smoke",
                ],
                check=False,
                capture_output=True,
                text=True,
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("stt command must pass --quiet", result.stderr)

    def test_verifier_accepts_voice_owned_command_shims_without_smoke(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "config.yaml"
            config.write_text(
                hermes_config(
                    command=(
                        f"{REPO_ROOT}/examples/hermes-command-tts.sh "
                        "{input_path} {output_path} {voice} {speed}"
                    ),
                    stt_command=f"{REPO_ROOT}/examples/hermes-command-stt.sh {{input_path}}",
                ),
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    str(SCRIPT_PATH),
                    "--config",
                    str(config),
                    "--skip-tts-smoke",
                ],
                check=True,
                capture_output=True,
                text=True,
            )

        self.assertIn("tts.command=hermes-command-tts.sh", result.stdout)
        self.assertIn("stt.command=hermes-command-stt.sh", result.stdout)

    def test_verifier_executes_tts_command_shim_with_voice_bin_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            bin_dir = tmp_path / "bin"
            bin_dir.mkdir()
            voice_bin = write_fake_voice(bin_dir)
            write_fake_ffprobe(bin_dir)
            config = tmp_path / "config.yaml"
            config.write_text(
                hermes_config(
                    command=(
                        f"{REPO_ROOT}/examples/hermes-command-tts.sh "
                        "{input_path} {output_path} {voice} {speed}"
                    ),
                    stt_command=f"{REPO_ROOT}/examples/hermes-command-stt.sh {{input_path}}",
                ),
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    str(SCRIPT_PATH),
                    "--config",
                    str(config),
                    "--voice-bin",
                    str(voice_bin),
                ],
                check=True,
                capture_output=True,
                text=True,
                env={"PATH": f"{bin_dir}:{os_path()}"},
            )

            voice_args = (bin_dir / "voice-args.txt").read_text(encoding="utf-8")

        self.assertIn("tts_smoke=checked", result.stdout)
        self.assertIn("tts.command=hermes-command-tts.sh", result.stdout)
        self.assertIn("say\n", voice_args)
        self.assertIn("--format\nogg-opus\n", voice_args)

    def test_verifier_rejects_arbitrary_command_wrapper(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "config.yaml"
            config.write_text(
                hermes_config(command="/tmp/custom-wrapper {input_path} {output_path}"),
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    str(SCRIPT_PATH),
                    "--config",
                    str(config),
                    "--skip-tts-smoke",
                ],
                check=False,
                capture_output=True,
                text=True,
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("voice say", result.stderr)

    def test_verifier_rejects_stt_shim_with_extra_arguments(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "config.yaml"
            config.write_text(
                hermes_config(
                    stt_command=(
                        f"{REPO_ROOT}/examples/hermes-command-stt.sh "
                        "{input_path} --json"
                    )
                ),
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    str(SCRIPT_PATH),
                    "--config",
                    str(config),
                    "--skip-tts-smoke",
                ],
                check=False,
                capture_output=True,
                text=True,
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("stt command shim must pass {input_path}", result.stderr)


def os_path() -> str:
    import os

    return os.environ.get("PATH", "")


if __name__ == "__main__":
    unittest.main()
