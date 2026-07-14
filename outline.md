# Voice Assistant — Project Structure

```
voice_assistant/
│
├── src/
│   ├── __init__.py
│   ├── logger.py           # Centralised logging (QueueHandler, RotatingFileHandler)
│   ├── stt.py              # Whisper.cpp subprocess wrapper (GPU via Vulkan)
│   ├── llm.py              # Ollama async HTTP client
│   ├── tts.py              # TTS engine wrapper (Kokoro / pyttsx3)
│   └── vad.py              # Voice Activity Detection (Silero VAD / OpenWakeWord)
│
├── audio/
│   ├── .gitkeep            # Temp WAV files land here; auto-deleted post-processing
│   └── wake_word/
│       └── hey_assistant.onnx  # Optional custom wake word model
│
├── logs/
│   ├── assistant.log       # Active log file (5 MB cap)
│   ├── assistant.log.1     # Rotated backup 1
│   ├── assistant.log.2     # Rotated backup 2
│   └── assistant.log.3     # Rotated backup 3 (oldest; auto-deleted on next rotation)
│
├── models/
│   └── .gitkeep            # Local ONNX / GGUF model files if needed
│
├── config.py               # ★ Frozen dataclasses: HardwareConfig, WhisperConfig,
│                           #   OllamaConfig, VADConfig, TTSConfig, PathConfig
├── main.py                 # Async entry point & orchestration loop
├── requirements.txt
├── .env.example            # Template for all os.getenv() overrides (see below)
├── .gitignore
└── README.md
```

## Module Responsibilities

| File | GPU Load | CPU Load | Notes |
|---|---|---|---|
| `config.py` | None | None | Frozen dataclasses; read-only singleton `cfg` |
| `logger.py` | None | ~0 (µs per call) | QueueHandler; all I/O on daemon thread |
| `stt.py` | ✅ Heavy (Vulkan) | Minimal | Spawns whisper.cpp subprocess |
| `llm.py` | ✅ Heavy (Ollama) | Minimal | Async HTTP; non-blocking |
| `tts.py` | None (CPU) | Low–Med | Kokoro on CPU; keeps GPU free |
| `vad.py` | Low (ONNX) | Low | Silero via ONNX Runtime |
| `main.py` | — | Minimal | Event loop only; no heavy work |

## Config Dataclass Hierarchy

```
AssistantConfig  (cfg)
├── HardwareConfig       vulkan_device_id, ollama_num_gpu, whisper_threads, sample_rate
├── WhisperConfig        bin_path, model_path, language, timeout_seconds
├── OllamaConfig         base_url, model, request_timeout, system_prompt
├── VADConfig            threshold, silence_duration_ms, min_speech_duration_ms, wake_word_model_path
├── TTSConfig            speed, voice, device
└── PathConfig           audio_dir, models_dir, logs_dir
```

## .env Override Reference

```dotenv
# Hardware
VULKAN_DEVICE_ID=0          # GPU index (0 for single-GPU systems)
OLLAMA_NUM_GPU=-1           # LLM layers on GPU (-1 = all that fit in VRAM)
WHISPER_THREADS=2           # CPU threads for whisper.cpp (safe gaming range: 1–4)
SAMPLE_RATE=16000           # Do not change — whisper.cpp requirement

# Whisper / STT
WHISPER_BIN=./bin/whisper-vulkan
WHISPER_MODEL=./models/ggml-base.en.bin
WHISPER_LANGUAGE=en
WHISPER_TIMEOUT=30

# Ollama / LLM
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=mistral
OLLAMA_TIMEOUT=60
OLLAMA_SYSTEM_PROMPT=You are a concise voice assistant. Keep responses under 2 sentences.

# VAD
VAD_THRESHOLD=0.5           # Raise to 0.65–0.75 for noisy environments
VAD_SILENCE_MS=600
VAD_MIN_SPEECH_MS=200
WAKE_WORD_MODEL=./audio/wake_word/hey_assistant.onnx

# TTS
TTS_SPEED=1.15
TTS_VOICE=af_sky
TTS_DEVICE=cpu

# Paths
AUDIO_DIR=./audio
MODELS_DIR=./models
LOGS_DIR=./logs
```

## Logging Architecture

```
Any module (asyncio coroutine or thread)
        │
        │  log.info(...)  ← ~1 µs, never blocks
        ▼
  QueueHandler  ──────────────────────────────────────────────┐
  (in-process queue, unbounded)                               │
                                                    Daemon thread (QueueListener)
                                                              │
                                          ┌───────────────────┴──────────────────┐
                                          ▼                                       ▼
                                  StreamHandler                       RotatingFileHandler
                                  (stdout, INFO+)                     (logs/assistant.log)
                                  Clean console                       DEBUG+, 5 MB × 3 backups
                                  during gameplay                     Full diagnostics on disk
```

## Standard Module Header

```python
# Top of stt.py, llm.py, tts.py, vad.py — three lines, nothing else needed:
from src.logger import get_logger
from config import cfg

log = get_logger(__name__)   # e.g. "assistant.src.stt"
```