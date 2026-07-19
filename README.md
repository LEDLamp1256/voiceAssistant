# Hey Jarvis — A Local Voice Assistant

A fully local, wake-word-triggered voice assistant designed to run in the
background. No API, just speech recognition, language model inference, 
and text-to-speech all run on your own hardware.

Say "Hey Jarvis," ask a question, get a spoken answer back.

Previous commits are on deprecated repo due to bad git management. 

---

## How It Works

The assistant is built around a two stage gating and `asyncio`, with a 
light weight wake word detector followed by the more resource intensive
voice activity detection, transcription, and language model processing:

```
Mic Capture → Wake Word ("Hey Jarvis") → VAD (speech end detected)
    → STT (whisper.cpp / Vulkan) → LLM (Ollama) → TTS (Kokoro) → Playback
                                        ↑
                          VAD barge-in watchdog (runs concurrently,
                          can interrupt at any point after STT)
```

**Design principles:**

- **GPU offloading.** transcription (whisper.cpp) and language model
  inference (Ollama) are the only stages that touch the GPU, via
  Vulkan — no CUDA anywhere in this stack. Wake-word detection and VAD
  stay on the CPU on purpose: they're cheap, and keeping them off the GPU
  leaves that headroom for the two stages that actually need it.
- **Non-blocking architecture.** High computer/blocking calls (ONNX inference,
  TTS synthesis, subprocess I/O) are routed through a *dedicated*
  `ThreadPoolExecutor` for its own module rather than `asyncio`'s shared
  default executor. Decoupling prevents bottlenecks from affecting input
  sensitivity under load.
- **Two stage VAD gating.** Wake-word detection gates entry into
  "recording" state; Silero VAD then decides when you've stopped talking,
  using a longer silence threshold before your command is confirmed.
- **Startup validation.** `config.py` checks every critical path at boot:
  a missing Silero VAD model is a hard failure, while a missing whisper.cpp
  binary/model or wake-word model is logged as an error or warning and the
  assistant still starts — the latter can run in a reduced mode (e.g. no
  wake-word gating) rather than refusing to boot over something non-essential.

---

## Prerequisites

| Requirement | Notes |
|---|---|
| **Python 3.10+** | |
| **A Vulkan-capable GPU** | AMD, NVIDIA, or Intel all work. Built and tested on an AMD RX 6700 XT specifically to avoid a CUDA dependency, but whisper.cpp's Vulkan backend runs on any Vulkan 1.2+ device. |
| **GPU driver with Vulkan support** | Most current AMD/NVIDIA/Intel drivers ship this already. |
| **[Vulkan SDK](https://vulkan.lunarg.com/sdk/home)** | Needed to *build* whisper.cpp's Vulkan backend (provides the shader compiler). Not required after the binary is built. |
| **CMake + a C++ toolchain** | On Windows: Visual Studio 2022 Build Tools ("Desktop development with C++") + CMake. |
| **[Ollama](https://ollama.com)** | Installed and running locally, with at least one model pulled (e.g. `ollama pull llama3.1:8b`). |
| **espeak-ng** | System-level dependency Kokoro uses for out-of-dictionary word fallback. [Windows installer here](https://github.com/espeak-ng/espeak-ng/releases); `apt-get install espeak-ng` on Linux. |
| **A working microphone** | |

> **A note on OS support:** this has been developed and tested on Windows
> 10. The Python side has POSIX-aware path handling and should run on
> Linux (PortAudio and whisper.cpp's Vulkan backend both support it), but
> it hasn't been exercised end-to-end there — issues/PRs welcome.

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

### 2. Create a virtual environment and install dependencies

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
```

**About `torch`:** Kokoro depends on PyTorch. The default `pip install
torch` can pull several gigabytes of CUDA runtime binaries you don't need
on non-NVIDIA hardware. Install the CPU-only build instead:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

(Kokoro's 82M parameters run comfortably on CPU — the GPU stays reserved
for whisper.cpp and Ollama.)

### 3. Install espeak-ng

Download and run the installer from the [espeak-ng releases
page](https://github.com/espeak-ng/espeak-ng/releases) (Windows), or
`sudo apt-get install espeak-ng` (Linux). Kokoro uses this for words
outside its training dictionary — without it, synthesis will error on
some inputs.

### 4. Build whisper.cpp with Vulkan support

There are currently no official prebuilt Vulkan binaries for Windows from
the upstream project, so build from source:

```bash
git clone https://github.com/ggml-org/whisper.cpp.git
cd whisper.cpp

cmake -B build -DGGML_VULKAN=1
cmake --build build --config Release
```

The compiled binary will be at `build/bin/Release/whisper-cli.exe`
(Windows) or `build/bin/whisper-cli` (Linux). You'll point `WHISPER_BIN`
at this file in step 7.

Next, download a GGML Whisper model (`.bin` file) — `ggml-base.en.bin` is
a good starting point for English — from the [whisper.cpp model
downloads](https://github.com/ggml-org/whisper.cpp#quick-start) and note
its path; you'll need it to run `main.py`.

### 5. Pull an Ollama model

```bash
ollama pull llama3.1:8b
```

Any model Ollama can run locally will work — pick one sized for your
available VRAM.

### 6. Download the model files

Create a `models/` folder in the project root (if it doesn't already
exist) and place these files in it:

| File | Source | Config default |
|---|---|---|
| `silero_vad.onnx` | [snakers4/silero-vad](https://github.com/snakers4/silero-vad/raw/master/files/silero_vad.onnx) | `models/silero_vad.onnx` |
| `hey_jarvis_v0.1.onnx` | [dscripka/openWakeWord releases](https://github.com/dscripka/openWakeWord/releases) | `models/hey_jarvis_v0.1.onnx` |
| A GGML whisper model, e.g. `ggml-base.en.bin` | [whisper.cpp model downloads](https://github.com/ggml-org/whisper.cpp#quick-start) | `models/ggml-base.en.bin` |

```
models/
├── silero_vad.onnx
├── hey_jarvis_v0.1.onnx
└── ggml-base.en.bin
```

The "Config default" column is where `config.py` looks by default. Save
each file under that exact name and you don't need to touch `.env` at
all for these three; if you'd rather use a different whisper model
(e.g. `ggml-small.en.bin` for better accuracy), save it wherever you like
and point `WHISPER_MODEL` at it in `.env` instead.

These paths all resolve relative to the project root regardless of where
you launch `main.py` from — `config.py` anchors them to its own file
location rather than the current working directory.

Kokoro's own weights are handled separately — the `kokoro` package
downloads them automatically from Hugging Face the first time it runs
(cached locally afterward). This is the one step in setup that needs
internet access; everything else runs fully offline once configured.

### 7. Configure your machine-specific paths

```bash
# Windows
copy .env.example .env
# Linux/macOS
cp .env.example .env
```

Open `.env` and set `WHISPER_BIN` to the binary you built in step 4:

```
WHISPER_BIN=C:\whisper.cpp\build\bin\Release\whisper-cli.exe
```

`config.py` documents every other override this project supports (Ollama
model/host, VAD sensitivity, TTS voice/speed, and more) at the top of the
file and in each dataclass's docstring — `.env.example` only ships the
ones that are actually machine-specific by default, but any of them can
be added to your own `.env` the same way.

### 8. Run it

```bash
python main.py
```

Say "Hey Jarvis," wait for the prompt tone/log line, then ask your
question.

---

## Project Structure

```
.
├── main.py             # asyncio orchestrator — entry point
├── config.py             # central, frozen-dataclass configuration
├── src/
│   ├── vad.py              # VAD state machine + wake-word gating (Silero + OpenWakeWord)
│   ├── stt.py                # whisper.cpp subprocess manager
│   ├── tts.py                  # Kokoro/pyttsx3 synthesis + persistent playback stream
│   ├── llm.py                   # Ollama HTTP streaming client
│   └── logger.py                  # QueueHandler/QueueListener logging setup
├── requirements.txt
├── .env.example                    # copy to .env and fill in your local paths
├── .env                              # gitignored — your machine-specific values
├── .gitignore
├── models/                            # gitignored contents — see Setup step 6
│   ├── silero_vad.onnx
│   ├── hey_jarvis_v0.1.onnx
│   └── ggml-base.en.bin
├── audio/                              # gitignored — scratch dir for temp audio
└── logs/                                # gitignored — application logs
```

> If `import src.vad` fails with `ModuleNotFoundError` on a fresh clone,
> add an empty `src/__init__.py` — depending on your Python version and
> how it's invoked, an implicit namespace package isn't always picked up
> the same way in every environment.

---

## Troubleshooting

- **A `CONFIG —` error or warning logged at startup about a missing
  path** — `config.py`'s `validate()` checks the whisper binary, whisper
  model, and wake-word model on every boot and logs (rather than crashes
  on) whichever one is missing, so the assistant can still start in a
  degraded state. Check the log line for the exact path it looked for —
  for the whisper binary specifically, that means `.env` either doesn't
  exist yet (did you `cp .env.example .env`?) or `WHISPER_BIN` inside it
  doesn't point at an actual file.
- **Kokoro fails to load and falls back to pyttsx3** — usually means
  espeak-ng isn't installed, or `torch` failed to install correctly. Check
  the startup logs; Kokoro logs its failure reason before falling back.
- **No GPU acceleration in whisper.cpp** — confirm the build log showed
  the Vulkan backend registering, and that your GPU driver actually
  exposes a Vulkan device (`vulkaninfo` on the command line is a quick
  way to check).
