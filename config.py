"""
Design principles
-----------------
* dataclasses with frozen=True — instances are immutable after creation,
  preventing accidental mutation from any module at runtime.
* os.getenv() for every field — a .env file (loaded via python-dotenv in
  main.py) can override any default without touching source code. Useful
  for switching Ollama models or adjusting VAD sensitivity mid-project.
* Path objects, never raw strings — callers never need to do Path(cfg.x).
* All AMD/Vulkan parameters live in a dedicated sub-dataclass so they are
  visually grouped and easy to hand off to subprocess calls together.

Usage
-----
    from config import cfg          # Import the pre-built singleton
    print(cfg.whisper.bin_path)     # Path to whisper.cpp binary
    print(cfg.hardware.vulkan_device_id)

.env override example
---------------------
    WHISPER_BIN=./bin/whisper-custom
    OLLAMA_MODEL=llama3
    VAD_THRESHOLD=0.6
    WHISPER_THREADS=2
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Project root — single source of truth for path resolution
# ---------------------------------------------------------------------------
# Anchored to this file's own location, not the current working directory,
# so every relative path below resolves the same way regardless of where
# the assistant is launched from (an IDE run config, a terminal opened in
# a different folder, a scheduled task, etc.).
PROJECT_ROOT: Path = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Helper — typed getenv wrappers to keep field definitions readable
# ---------------------------------------------------------------------------
def _env_str(key: str, default: str) -> str:
    return os.getenv(key, default)


def _env_int(key: str, default: int) -> int:
    raw = os.getenv(key)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        # Malformed env var: fall back to default rather than crashing.
        return default


def _env_float(key: str, default: float) -> float:
    raw = os.getenv(key)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_path(key: str, default: str) -> Path:
    """
    A relative default or override resolves against PROJECT_ROOT, not the
    process's current working directory — this is what actually makes
    `AUDIO_DIR=./audio` (or its default) point at the same folder no matter
    where `python main.py` is launched from. An absolute override (e.g.
    ``WHISPER_BIN=C:\\tools\\whisper-cli.exe``) passes through unchanged:
    joining PROJECT_ROOT with an absolute path just yields that absolute
    path, which is standard pathlib behavior.
    """
    return (PROJECT_ROOT / os.getenv(key, default)).resolve()


# ---------------------------------------------------------------------------
# Sub-config: AMD / Vulkan hardware parameters
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class HardwareConfig:
    """
    AMD RX 6700 XT (12 GB VRAM) + Intel i5-12400F tuning.

    vulkan_device_id
        Passed to whisper.cpp as ``-vd <id>``. 0 is almost always correct
        for a single-GPU system; set to 1 if you have an iGPU at index 0.

    ollama_num_gpu
        Layers of the LLM to offload to the GPU. -1 = offload everything
        that fits; Ollama will respect the 12 GB ceiling automatically.
        Lower this (e.g. 33) if you see VRAM OOM errors with large models.

    whisper_threads
        CPU threads for whisper.cpp non-Vulkan work (tokenisation, beam
        search bookkeeping). The i5-12400F has 6 P-cores / 12 threads.
        Default 2 leaves ≥10 threads for the game. Never set above 4
        while gaming or you will feel frame-time spikes.

    sample_rate
        whisper.cpp requires 16 000 Hz mono PCM. Do not change.
    """

    vulkan_device_id: int = field(
        default_factory=lambda: _env_int("VULKAN_DEVICE_ID", 0)
    )
    ollama_num_gpu: int = field(
        default_factory=lambda: _env_int("OLLAMA_NUM_GPU", -1)
    )
    whisper_threads: int = field(
        default_factory=lambda: _env_int("WHISPER_THREADS", 2)
    )
    sample_rate: int = field(
        default_factory=lambda: _env_int("SAMPLE_RATE", 16_000)
    )


# ---------------------------------------------------------------------------
# Sub-config: whisper.cpp (STT)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class WhisperConfig:
    """
    Paths and flags for the whisper.cpp Vulkan subprocess.

    bin_path
        Absolute or project-relative path to the compiled Vulkan binary.
        Build from source with: ``cmake -DGGML_VULKAN=1 ..``

    model_path
        GGUF model file. ``ggml-base.en`` is the sweet spot for gaming:
        ~150 MB VRAM, ~300 ms transcription latency on RX 6700 XT.
        Upgrade to ``ggml-small.en`` if accuracy is more important than
        latency (uses ~500 MB VRAM).

    language
        ISO 639-1 code. Hard-coded to "en" avoids the auto-detect step
        which wastes ~100 ms of GPU time per utterance.

    timeout_seconds
        subprocess.run timeout. 30 s is generous; typical transcription
        is 0.3–1.5 s. Guards against hangs if the GPU stalls.
    """

    bin_path: Path = field(
        default_factory=lambda: _env_path("WHISPER_BIN", "./bin/whisper-vulkan")
    )
    model_path: Path = field(
        default_factory=lambda: _env_path("WHISPER_MODEL", "./models/ggml-base.en.bin")
    )
    language: str = field(
        default_factory=lambda: _env_str("WHISPER_LANGUAGE", "en")
    )
    timeout_seconds: int = field(
        default_factory=lambda: _env_int("WHISPER_TIMEOUT", 30)
    )


# ---------------------------------------------------------------------------
# Sub-config: Ollama (LLM)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OllamaConfig:
    """
    Connection and model settings for the local Ollama instance.

    base_url
        Ollama's default listen address. Change only if you've edited
        Ollama's OLLAMA_HOST env var.

    model
        Any model name pulled via ``ollama pull <model>``.
        Recommended for 12 GB VRAM while gaming:
          - mistral   (7B, ~5 GB)  ← default, fast
          - llama3    (8B, ~6 GB)  ← better reasoning
          - phi3:mini (3.8B, ~3GB) ← lowest VRAM, lowest latency

    request_timeout
        httpx timeout in seconds for the full streaming response.
        Long for slow models; 60 s is safe for mistral on RX 6700 XT.

    system_prompt
        Injected as the system role on every request to keep the
        assistant terse and gameplay-appropriate.
    """

    base_url: str = field(
        default_factory=lambda: _env_str("OLLAMA_BASE_URL", "http://localhost:11434")
    )
    model: str = field(
        default_factory=lambda: _env_str("OLLAMA_MODEL", "mistral")
    )
    request_timeout: int = field(
        default_factory=lambda: _env_int("OLLAMA_TIMEOUT", 60)
    )
    system_prompt: str = field(
        default_factory=lambda: _env_str(
            "OLLAMA_SYSTEM_PROMPT",
            "You are a concise voice assistant running during PC gaming. "
            "Keep all responses under 2 sentences. No markdown.",
        )
    )


# ---------------------------------------------------------------------------
# Sub-config: VAD (Voice Activity Detection)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class VADConfig:
    """
    Silero VAD / OpenWakeWord tuning parameters.

    threshold
        Probability threshold (0.0–1.0) above which a frame is classified
        as speech. 0.5 is Silero's recommended default. Raise to 0.65–0.75
        in noisy environments (mechanical keyboards, game audio bleed).
        Lower to 0.35 if the assistant misses quiet speech.

    silence_duration_ms
        Milliseconds of consecutive silence required to mark speech end
        and trigger the STT pipeline. 600 ms avoids cutting off mid-word
        pauses while not adding perceptible latency.

    min_speech_duration_ms
        Segments shorter than this are discarded as noise (keyboard clicks,
        mouse clicks, brief game audio spikes). 200 ms is a safe floor.

    wake_word_model_path
        Optional ONNX model for OpenWakeWord. Set to "" to disable wake
        word gating and process all detected speech immediately.
    """

    threshold: float = field(
        default_factory=lambda: _env_float("VAD_THRESHOLD", 0.15)
    )
    silence_duration_ms: int = field(
        default_factory=lambda: _env_int("VAD_SILENCE_MS", 600)
    )
    min_speech_duration_ms: int = field(
        default_factory=lambda: _env_int("VAD_MIN_SPEECH_MS", 200)
    )
    wake_word_model_path: Path = field(
        default_factory=lambda: _env_path("WAKE_WORD_MODEL", "./models/hey_jarvis_v0.1.onnx")
    )


# ---------------------------------------------------------------------------
# Sub-config: TTS
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TTSConfig:
    """
    Text-to-speech engine settings.

    speed
        Playback speed multiplier for Kokoro-82M output.
        1.0 = natural pace. 1.15–1.25 is faster without sounding robotic,
        useful to keep gaming interruptions brief.

    voice
        Kokoro voice ID string. "af_sky" is a clear, neutral English voice.
        See Kokoro docs for the full voice list.

    device
        "cpu" keeps TTS off the GPU entirely, preserving all 12 GB VRAM
        for whisper.cpp and Ollama. Kokoro-82M is fast enough on CPU
        for short gaming responses (<1 s for a 2-sentence reply).
    """

    speed: float = field(
        default_factory=lambda: _env_float("TTS_SPEED", 1.15)
    )
    voice: str = field(
        default_factory=lambda: _env_str("TTS_VOICE", "af_sky")
    )
    device: str = field(
        default_factory=lambda: _env_str("TTS_DEVICE", "cpu")
    )


# ---------------------------------------------------------------------------
# Sub-config: SFX (wake-word confirmation chirp)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SFXConfig:
    """
    Wake-word confirmation chirp settings.

    enabled
        If False, sfx.initialise() is a no-op and play_ping() never
        plays anything. Lets the feature be turned off entirely from
        .env without touching code.

    chirp_path
        Path to a short (<300ms recommended) WAV file played the instant
        the wake word fires. A missing file degrades to "chirp disabled,
        logged once" rather than a fail-fast crash — unlike Silero's or
        whisper.cpp's model files, this is cosmetic UX, not a functional
        dependency of the pipeline.
    """

    enabled: bool = field(
        default_factory=lambda: _env_str("WAKE_CHIRP_ENABLED", "true").lower() == "true"
    )
    chirp_path: Path = field(
        default_factory=lambda: _env_path("WAKE_CHIRP_PATH", "./assets/wake_chirp.wav")
    )


# ---------------------------------------------------------------------------
# Sub-config: Paths
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PathConfig:
    """
    Top-level filesystem layout for the project.

    All paths are relative to the project root by default and can be
    overridden to absolute paths via .env for non-standard installs.
    """

    audio_dir: Path = field(
        default_factory=lambda: _env_path("AUDIO_DIR", "./audio")
    )
    models_dir: Path = field(
        default_factory=lambda: _env_path("MODELS_DIR", "./models")
    )
    logs_dir: Path = field(
        default_factory=lambda: _env_path("LOGS_DIR", "./logs")
    )


# ---------------------------------------------------------------------------
# Root config — composes all sub-configs into one import
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AssistantConfig:
    """
    Root configuration object. Import `cfg` from this module everywhere.

    All sub-configs are instantiated once at module load time (after
    python-dotenv has populated os.environ in main.py). Because the
    dataclass is frozen, no module can mutate settings at runtime.
    """

    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    whisper: WhisperConfig = field(default_factory=WhisperConfig)
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    vad: VADConfig = field(default_factory=VADConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    sfx: SFXConfig = field(default_factory=SFXConfig)
    paths: PathConfig = field(default_factory=PathConfig)

    def validate(self) -> None:
        """
        Ensure every project-managed directory exists, then sanity-check
        critical paths at startup. Logs warnings rather than raising for
        the file checks, so the assistant can still boot in a degraded
        state (e.g. missing wake word model is non-fatal) — but directory
        creation itself is unconditional and silent on success, since an
        empty directory is never a reason to degrade anything.

        Directory creation runs first and is required for a "clone and
        run" experience: git does not track empty directories, so
        audio/, models/, and logs/ do not exist on a fresh clone until
        something creates them. Every field in PathConfig is created here,
        via mkdir(parents=True, exist_ok=True) — safe to call even if a
        directory already exists, and safe to call every startup, every
        time, unconditionally.

        This intentionally does NOT create parent directories for
        whisper.bin_path or whisper.model_path: those point at files the
        user must supply themselves (a compiled binary, a downloaded
        model), and creating an empty folder there wouldn't make either
        one exist — the existence checks below still correctly report
        them as missing either way.

        Call once from main.py after init_logging().
        """
        # Import here to avoid circular dependency at module load time.
        from src.logger import get_logger
        log = get_logger(__name__)

        for directory in (self.paths.audio_dir, self.paths.models_dir, self.paths.logs_dir):
            directory.mkdir(parents=True, exist_ok=True)

        if not self.whisper.bin_path.exists():
            log.error(
                "CONFIG — whisper.cpp binary not found at %s. "
                "Build with Vulkan support and set WHISPER_BIN in .env.",
                self.whisper.bin_path,
            )
        if not self.whisper.model_path.exists():
            log.error(
                "CONFIG — Whisper model not found at %s. "
                "Download from https://huggingface.co/ggerganov/whisper.cpp",
                self.whisper.model_path,
            )
        if not self.vad.wake_word_model_path.exists():
            log.warning(
                "CONFIG — Wake word model not found at %s. "
                "Wake word gating is disabled; all speech will be processed.",
                self.vad.wake_word_model_path,
            )
        if not (1 <= self.hardware.whisper_threads <= 4):
            log.warning(
                "CONFIG — WHISPER_THREADS=%d is outside the safe gaming range "
                "[1–4]. CPU contention with the game process may cause "
                "frame-time spikes.",
                self.hardware.whisper_threads,
            )

        log.info(
            "CONFIG — loaded: model=%s | vad_threshold=%.2f | "
            "whisper_threads=%d | vulkan_device=%d",
            self.ollama.model,
            self.vad.threshold,
            self.hardware.whisper_threads,
            self.hardware.vulkan_device_id,
        )

# ---------------------------------------------------------------------------
# Singleton — the one instance every module imports
# ---------------------------------------------------------------------------
cfg: AssistantConfig = AssistantConfig()