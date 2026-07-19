"""
Public Interface
-----------------
    await vad.initialise()                       # once, at startup
    await vad.is_speech(chunk: bytes) -> bool
    await vad.listen_for_speech(stream) -> bytes
    vad.shutdown_executor()                       # once, at shutdown

Pipeline Architecture
----------------------
Two-stage gating pipeline:

    Stage 1 — OpenWakeWord (optional gating layer). If
    cfg.vad.wake_word_model_path points at an existing .onnx file, wake-word
    detection gates entry into the RECORDING_COMMAND state before Silero is
    consulted at all.

    Stage 2 — Silero VAD (always active). Runs on CPU via ONNX Runtime
    only — no PyTorch dependency anywhere in this module. Produces a
    per-frame speech probability and drives the silence-close logic that
    ends a recording segment.

Fail-Fast Initialization
-------------------------
initialise() checks for the Silero ONNX model file on disk at
cfg.paths.models_dir / "silero_vad.onnx" and raises FileNotFoundError, with
a manual-download URL in the log message, if it is missing. No network
calls, no implicit model materialization.

Path Resolution
-----------------
cfg.paths.models_dir is the single authoritative source for model file
locations, resolved to an absolute path once at module import time into
_SILERO_MODEL_PATH.

Dedicated Inference Executor
------------------------------
onnxruntime's session.run() releases the GIL during the underlying C++
work, but the call still occupies a Python thread for setup/teardown.
Inference is routed through a VAD-dedicated ThreadPoolExecutor
(max_workers=1) rather than asyncio's shared default executor, so a
stalled TTS synthesis call can never starve VAD responsiveness. A single
worker is a correctness requirement, not an arbitrary choice: Silero's
recurrent hidden state must be updated in strict frame order.

State Machine (listen_for_speech)
------------------------------------
    Wake-word gated mode (a wake-word model is loaded):
        IDLE --wake word--> RECORDING_COMMAND
            (buffering starts on this exact frame, unconditionally — does
            not wait for or require Silero's independent confirmation)
          --silence >= _PRE_COMMAND_GRACE_MS, command speech never
            confirmed--> segment returned as captured
          --command speech independently confirmed by Silero--
              --silence >= _POST_COMMAND_SILENCE_MS--> segment returned
          --_MAX_RECORDING_MS absolute ceiling reached, any time--> segment
            force-closed and returned

    Open mode (no wake-word model loaded):
        IDLE --speech detected by Silero--> RECORDING_COMMAND
          --silence >= _POST_COMMAND_SILENCE_MS--> segment returned

Two-Tier Silence Thresholds
------------------------------
A single fixed silence threshold cannot serve both pauses well: the pause
between finishing the wake word and starting the command (long, natural —
e.g. a breath before speaking) versus the pause after the command ends
(short, for a responsive turn-taking feel). _PRE_COMMAND_GRACE_MS
(2000 ms) covers the former; _POST_COMMAND_SILENCE_MS (800 ms, floored
against cfg.vad.silence_duration_ms) covers the latter. Which threshold
applies is selected dynamically by whether Silero has independently
confirmed real speech since entering RECORDING_COMMAND — that flag never
determines whether a wake-word-triggered segment is kept, only which
silence duration is used to decide when it closes.

Recording Ceiling
--------------------
_MAX_RECORDING_MS (30 s) is a memory-leak backstop only, for the
pathological case where confirmed speech never stops being detected (e.g.
background game audio keeps Silero's probability pinned above
cfg.vad.threshold indefinitely). It should not fire in normal use; regular
segment closes are handled by the two silence thresholds above.

Wake Word Debounce
----------------------
OpenWakeWord retains several seconds of raw-audio, melspectrogram, and
embedding buffer state internally, with no public reset for those buffers
(Model.reset() only clears the prediction-history deque used for its
optional hysteresis feature). Immediately after a segment closes, that
buffered audio still contains the wake word itself and can re-trigger
detection instantly. _OWW_DEBOUNCE_MS keeps feeding OpenWakeWord real
frames after every close — so its buffer naturally slides the old audio
out — while ignoring whatever it reports until the debounce deadline
passes.

Silero Input Contract Detection
-----------------------------------
Silero VAD's ONNX export contract has not been stable across versions.
Classic exports accept a bare [1, 512] frame; v5+ exports expect
[1, 576] — 512 frame samples plus 64 samples of context (the tail of the
previous chunk), concatenated by the caller rather than tracked internally
by the model. initialise() inspects the loaded model's declared input
shape and configures context concatenation accordingly rather than
assuming either contract.
"""

from __future__ import annotations

import asyncio
import enum
import time
import functools
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from config import cfg
from src.logger import get_logger

log = get_logger(__name__)

_SILERO_MODEL_PATH: Path = Path(cfg.paths.models_dir).resolve() / "silero_vad.onnx"

_SILERO_DOWNLOAD_URL: str = (
    "https://github.com/snakers4/silero-vad/raw/master/files/silero_vad.onnx"
)

try:
    import onnxruntime as ort
    _ORT_AVAILABLE = True
except ImportError:
    _ORT_AVAILABLE = False

try:
    from openwakeword.model import Model as OWWModel
    _OWW_AVAILABLE = True
except ImportError:
    _OWW_AVAILABLE = False

_FRAME_SAMPLES: int = 512
_FRAME_MS: float = (_FRAME_SAMPLES / cfg.hardware.sample_rate) * 1000
_OWW_SCORE_THRESHOLD: float = 0.5


class _State(enum.Enum):
    """
    Purpose:
        Enumerates the two states of the listen_for_speech() coordination
        state machine.

    Internal Mechanism:
        RECORDING_COMMAND covers both the wake-word-gated and open-mode
        recording paths — once entered, the only ways out are (a) a
        sustained post-speech silence appropriate to the current
        confirmation state, or (b) the _MAX_RECORDING_MS hard ceiling.
        Nothing closes a segment early merely because the frame immediately
        after the wake word happened to be quiet.
    """

    IDLE = enum.auto()
    RECORDING_COMMAND = enum.auto()


_PRE_COMMAND_GRACE_MS: float = 2_000.0
_POST_COMMAND_SILENCE_MS: float = 800.0
_MAX_RECORDING_MS: float = 30_000.0
_OWW_DEBOUNCE_MS: float = 3_000.0
_oww_debounce_until: float = 0.0

_silero_session: Optional["ort.InferenceSession"] = None
_silero_state: Optional[np.ndarray] = None
_silero_sr: Optional[np.ndarray] = None

_silero_context_samples: int = 0
_silero_context: Optional[np.ndarray] = None

_oww_model: Optional["OWWModel"] = None

_vad_executor: ThreadPoolExecutor = ThreadPoolExecutor(
    max_workers=1,
    thread_name_prefix="vad_worker",
)


async def _run_vad(fn, *args, **kwargs):
    """
    Purpose:
        Route a blocking inference call through the VAD-dedicated thread
        pool instead of asyncio's shared default executor.

    Internal Mechanism:
        Binds fn and its arguments into a single callable via
        functools.partial and awaits it on _vad_executor. Isolating VAD
        inference onto its own single-worker pool guarantees Silero's
        frame-ordered hidden state is never interleaved with unrelated work
        (e.g. TTS synthesis) and is never delayed by it.

    Args:
        fn: A blocking, synchronous callable to execute off the event loop.
        *args: Positional arguments passed to fn.
        **kwargs: Keyword arguments passed to fn.

    Returns:
        The return value of fn(*args, **kwargs).
    """
    loop = asyncio.get_running_loop()
    func_with_args = functools.partial(fn, *args, **kwargs)
    return await loop.run_in_executor(_vad_executor, func_with_args)


def shutdown_executor() -> None:
    """
    Purpose:
        Cleanly shut down the VAD-dedicated thread pool.

    Internal Mechanism:
        Blocks until any in-flight inference call on _vad_executor
        completes, then releases the pool's worker thread. Intended to be
        called exactly once, from main.py's shutdown sequence.

    Args:
        None.

    Returns:
        None.
    """
    log.info("VAD — shutting down dedicated executor")
    _vad_executor.shutdown(wait=True)


async def initialise() -> None:
    """
    Purpose:
        Load Silero VAD (and, if configured, OpenWakeWord) into memory.
        Must be awaited exactly once from main.py before any other public
        function in this module is called.

    Internal Mechanism:
        Loads the Silero ONNX session on the VAD-dedicated executor via
        _run_vad(), after first verifying the model file exists on disk
        (fail-fast: no download, no implicit materialization). Resets
        Silero's recurrent state, then inspects the session's declared
        'input' shape to detect whether the loaded model requires
        context-sample concatenation (the Silero v5+ convention) versus a
        bare 512-sample frame (the classic contract), configuring
        _silero_context_samples accordingly. If cfg.vad.wake_word_model_path
        points at an existing .onnx file and the openwakeword package is
        installed, also loads the wake-word model; otherwise wake-word
        gating is left disabled and listen_for_speech() falls back to
        ungated speech detection.

    Args:
        None.

    Returns:
        None.

    Raises:
        RuntimeError: onnxruntime is not installed (hard dependency).
        FileNotFoundError: the Silero ONNX model file is missing on disk.
    """
    global _silero_session, _silero_state, _silero_sr, _oww_model

    if not _ORT_AVAILABLE:
        raise RuntimeError(
            "VAD — onnxruntime is not installed. "
            "Run: pip install onnxruntime"
        )

    if not _SILERO_MODEL_PATH.exists():
        log.critical(
            "VAD — Silero ONNX model not found at %s. "
            "This file must be provided manually; no auto-download is "
            "performed. Download it from: %s and place it at the path "
            "above (create the directory first if needed: %s).",
            _SILERO_MODEL_PATH,
            _SILERO_DOWNLOAD_URL,
            _SILERO_MODEL_PATH.parent,
        )
        raise FileNotFoundError(
            f"VAD — required model file missing: {_SILERO_MODEL_PATH}\n"
            f"  Expected: {_SILERO_MODEL_PATH}\n"
            f"  Directory exists: {_SILERO_MODEL_PATH.parent.exists()}\n"
            f"  Download from: {_SILERO_DOWNLOAD_URL}"
        )

    try:
        _silero_session = await _run_vad(
            ort.InferenceSession,
            str(_SILERO_MODEL_PATH),
            providers=["CPUExecutionProvider"],
        )
    except Exception as exc:
        log.critical(
            "VAD — failed to load Silero ONNX session from %s. "
            "The file may be corrupt or incompatible with the installed "
            "onnxruntime version. Error: %s",
            _SILERO_MODEL_PATH,
            exc,
        )
        raise

    _reset_silero_state()

    global _silero_context_samples
    _silero_context_samples = 0
    try:
        _input_meta = next(
            i for i in _silero_session.get_inputs() if i.name == "input"
        )
        _declared_dim1 = _input_meta.shape[1] if len(_input_meta.shape) > 1 else None
        if isinstance(_declared_dim1, int) and _declared_dim1 > _FRAME_SAMPLES:
            _silero_context_samples = _declared_dim1 - _FRAME_SAMPLES
            log.info(
                "VAD — Silero 'input' declares a FIXED size of %d "
                "(%d frame + %d context samples) — context-concatenation "
                "mode enabled",
                _declared_dim1,
                _FRAME_SAMPLES,
                _silero_context_samples,
            )
        elif isinstance(_declared_dim1, int) and _declared_dim1 == _FRAME_SAMPLES:
            log.info(
                "VAD — Silero 'input' declares a FIXED size of %d, matching "
                "_FRAME_SAMPLES exactly — no context concatenation needed",
                _declared_dim1,
            )
        else:
            _silero_context_samples = 64
            log.warning(
                "VAD — Silero 'input' has a DYNAMIC shape (declared=%r) — "
                "cannot determine the exact contract from the shape alone. "
                "Defaulting to context-concatenation mode with 64 context "
                "samples (the current Silero v5+ convention). If VAD "
                "sensitivity is still wrong after this, that assumption "
                "may be incorrect for this specific model file.",
                _input_meta.shape,
            )
    except StopIteration:
        log.warning(
            "VAD — could not find an 'input' node in the Silero ONNX "
            "session to inspect — proceeding without context concatenation"
        )

    _reset_silero_state()

    log.info(
        "VAD — Silero VAD loaded from %s (CPUExecutionProvider)",
        _SILERO_MODEL_PATH,
    )

    wake_path: Path = Path(cfg.vad.wake_word_model_path).resolve()
    if wake_path.exists() and wake_path.suffix == ".onnx":
        if not _OWW_AVAILABLE:
            log.warning(
                "VAD — Wake word model found at %s but 'openwakeword' is not "
                "installed. Wake word gating disabled. "
                "Run: pip install openwakeword",
                wake_path,
            )
        else:
            try:
                _oww_model = await _run_vad(
                    OWWModel,
                    wakeword_models=[str(wake_path)],
                    inference_framework="onnx",
                )
                log.info(
                    "VAD — OpenWakeWord loaded from %s. "
                    "Pipeline: wake word → speech → STT",
                    wake_path,
                )
            except Exception:
                log.exception(
                    "VAD — failed to load OpenWakeWord model from %s. "
                    "Wake word gating disabled; falling back to ungated speech detection.",
                    wake_path,
                )
                _oww_model = None
    else:
        log.info(
            "VAD — No wake word model at %s. "
            "Pipeline: speech detected → STT (ungated)",
            wake_path,
        )


def _reset_silero_state() -> None:
    """
    Purpose:
        Reset Silero's recurrent hidden state (and context buffer, if the
        loaded model requires one) to a clean, zeroed starting point.

    Internal Mechanism:
        Called at initialise() time, at the top of every
        listen_for_speech() call, and after every completed or discarded
        segment within it. Silero v4+ uses a single combined `state` tensor
        of shape (2, 1, 128), replacing the separate (h, c) LSTM tensors
        used in v3 and earlier. If _silero_context_samples > 0 (the loaded
        model requires the tail of the previous chunk to be concatenated
        onto each new frame), a fresh zeroed context buffer of that size is
        also allocated. Both the recurrent state and the context buffer
        must never leak across two unrelated listening sessions — they may
        only accumulate within one.

    Args:
        None.

    Returns:
        None.
    """
    global _silero_state, _silero_sr, _silero_context
    _silero_state = np.zeros((2, 1, 128), dtype=np.float32)
    _silero_sr = np.array([cfg.hardware.sample_rate], dtype=np.int64)
    if _silero_context_samples > 0:
        _silero_context = np.zeros((1, _silero_context_samples), dtype=np.float32)
    else:
        _silero_context = None


def _silero_infer(frame_f32: np.ndarray) -> float:
    """
    Purpose:
        Run one audio frame through the Silero ONNX session and return the
        model's speech probability for that frame.

    Internal Mechanism:
        Uses the v4+ input contract: {"input": <audio>, "state": <(2,1,128)
        f32>, "sr": <int64 tensor>}. If the loaded model requires context
        concatenation (_silero_context_samples > 0, detected in
        initialise()), the tensor actually fed to the model is
        [context + frame] — e.g. 64 context samples + 512 frame samples =
        576 — matching the v5+ convention where the caller, not the model,
        is responsible for carrying the tail of the previous chunk forward
        as a rolling window. The next call's context is set to the tail of
        THIS chunk (pre-concatenation), matching the reference
        implementation's rolling-window behavior exactly. The recurrent
        `state` tensor returned by the session is stored back into module
        state for the next call.

    Args:
        frame_f32: New audio only, shape (1, _FRAME_SAMPLES) float32 in
                   [-1, 1]. Context (if any) is prepended internally and is
                   transparent to the caller.

    Returns:
        Speech probability in [0.0, 1.0].
    """
    global _silero_state, _silero_context

    if _silero_context_samples > 0:
        model_input = np.concatenate([_silero_context, frame_f32], axis=1)
        _silero_context = frame_f32[:, -_silero_context_samples:]
    else:
        model_input = frame_f32

    outputs = _silero_session.run(
        ["output", "stateN"],
        {
            "input": model_input,
            "state": _silero_state,
            "sr": _silero_sr,
        },
    )
    speech_prob: float = float(outputs[0].squeeze())
    _silero_state = outputs[1]
    return speech_prob


def _oww_infer(frame_pcm16: bytes) -> bool:
    """
    Purpose:
        Run one audio frame through OpenWakeWord and report whether the
        configured wake word was detected.

    Internal Mechanism:
        Converts the raw PCM16 frame to an int16 numpy array and scores it
        against every loaded wake-word model via _oww_model.predict(). The
        highest confidence score across all loaded models is compared
        against _OWW_SCORE_THRESHOLD.

    Args:
        frame_pcm16: Raw 16-bit PCM bytes — the same frame used for Silero
                     VAD scoring.

    Returns:
        True if the best wake-word confidence score meets or exceeds
        _OWW_SCORE_THRESHOLD, False otherwise.
    """
    audio_int16 = np.frombuffer(frame_pcm16, dtype=np.int16)
    scores = _oww_model.predict(audio_int16)
    best: float = max(scores.values(), default=0.0)
    return best >= _OWW_SCORE_THRESHOLD


def _pcm16_to_f32(pcm_bytes: bytes) -> np.ndarray:
    """
    Purpose:
        Convert a raw 16-bit PCM audio buffer into the float32 format
        Silero's ONNX session expects.

    Internal Mechanism:
        Interprets the byte buffer as int16 samples, casts to float32, and
        normalizes into the [-1, 1] range expected by the model.

    Args:
        pcm_bytes: Raw 16-bit PCM audio bytes.

    Returns:
        float32 numpy array of shape (1, N), values in [-1, 1].
    """
    audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
    audio /= 32768.0
    return audio.reshape(1, -1)


async def is_speech(audio_chunk: bytes) -> bool:
    """
    Purpose:
        Determine whether a single audio frame contains speech, using
        Silero alone with no wake-word gating.

    Internal Mechanism:
        Converts the frame to float32, pads or truncates it to exactly
        _FRAME_SAMPLES samples, and runs it through _silero_infer() on the
        VAD-dedicated executor. Wake-word gating is not applied here — that
        state machine lives entirely in listen_for_speech().

    Args:
        audio_chunk: Raw 16-bit PCM bytes. Padded with zeros if shorter
                     than _FRAME_SAMPLES, truncated if longer.

    Returns:
        True if Silero's speech probability for this frame is >=
        cfg.vad.threshold.

    Raises:
        RuntimeError: initialise() has not been awaited yet.
    """
    if _silero_session is None:
        raise RuntimeError("VAD — call await vad.initialise() before is_speech()")

    frame_f32 = _pcm16_to_f32(audio_chunk)

    n = frame_f32.shape[1]
    if n < _FRAME_SAMPLES:
        frame_f32 = np.pad(frame_f32, ((0, 0), (0, _FRAME_SAMPLES - n)))
    elif n > _FRAME_SAMPLES:
        frame_f32 = frame_f32[:, :_FRAME_SAMPLES]

    prob: float = await _run_vad(_silero_infer, frame_f32)
    return prob >= cfg.vad.threshold


async def listen_for_speech(
    stream: "asyncio.Queue[bytes]",
    on_trigger: Optional["Callable[[], None]"] = None,
) -> bytes:
    """
    Purpose:
        Asynchronously block until a complete speech segment has been
        captured, applying wake-word gating first if a wake-word model is
        loaded, then Silero-driven silence detection to determine when the
        segment ends.

    Barge-in Callback Timing (on_trigger)
    -----------------------------------------
        FIX [Barge-in latency]: this function previously offered no signal
        of ANY kind until it fully RETURNS — which only happens once the
        entire utterance closes (silence timeout or the max-recording
        ceiling). The caller's only hook was the return value, so anything
        gated on "the user started talking" (e.g. main.py calling
        tts.stop() to interrupt playback) was necessarily delayed by the
        caller's own recording+silence window, letting old TTS audio keep
        playing through the wake word AND the entire new command before
        being cut off — the opposite of an instant barge-in.

        on_trigger, if provided, is called synchronously the instant this
        function's internal state transitions IDLE -> RECORDING_COMMAND —
        i.e. the same frame the wake word fires (gated mode) or the same
        frame Silero first confirms speech (open mode) — not when the
        segment finishes being captured. It is a plain synchronous
        callable (not a coroutine): main.py passes tts.stop, which is
        itself documented as thread-safe / callable from any context and
        does the actual audio-silencing work (clears the playback queue
        and the in-flight chunk so the persistent output stream goes
        silent within one blocksize, ~32 ms). Any exception raised by
        on_trigger is caught and logged here rather than allowed to
        propagate — a broken interrupt hook must never take down VAD's
        own state machine.

    Internal Mechanism:
        Implements the module's state machine:

            Wake-word gated mode (a wake-word model is loaded):
                IDLE --wake word detected--> RECORDING_COMMAND
                    (buffering starts on this exact frame, unconditionally;
                    does not wait for or require Silero's independent
                    confirmation)
                  --silence >= _PRE_COMMAND_GRACE_MS, command speech never
                    confirmed--> segment returned as captured
                  --command speech independently confirmed by Silero--
                      --silence >= _POST_COMMAND_SILENCE_MS--> segment
                        returned
                  --_MAX_RECORDING_MS absolute ceiling reached, any time--
                    --> segment force-closed and returned

            Open mode (no wake-word model loaded):
                IDLE --speech detected by Silero--> RECORDING_COMMAND
                  --silence >= _POST_COMMAND_SILENCE_MS--> segment returned

        OpenWakeWord detecting the wake word is sufficient on its own to
        both start and keep a segment; Silero never gates either decision
        once in wake-word-gated mode. `_command_confirmed` only ever
        selects which of the two silence thresholds applies (generous
        pre-command grace vs. snappy post-command silence) — it never
        determines whether the captured segment is kept. In the worst case
        (wake word fires, nobody speaks), `_PRE_COMMAND_GRACE_MS` of
        near-silence is returned to STT, which whisper.cpp simply
        transcribes as empty output.

        Before entering the loop, Silero's recurrent state is
        unconditionally reset — this function may be cancelled externally
        mid-segment (e.g. by a barge-in watchdog), which would otherwise
        skip the reset and leave contaminated hidden state for the next
        call — and any frames already sitting in `stream` are drained
        non-blockingly so a fresh call never scores stale, previously
        buffered audio as if it just arrived.

        When a segment closes (by any of the three paths above),
        OpenWakeWord's internal audio buffers still contain the
        just-spoken audio and can re-trigger instantly on stale content. A
        module-level debounce deadline (`_oww_debounce_until`) is armed on
        every close; Stage 1 keeps feeding OpenWakeWord real frames during
        the debounce window (so its buffer slides the old audio out
        naturally) but ignores whatever it reports until the deadline
        passes.

    Args:
        stream: asyncio.Queue fed by the microphone capture loop in
                main.py. Each item must be exactly _FRAME_SAMPLES * 2 raw
                16-bit PCM bytes.
        on_trigger: Optional synchronous callable invoked exactly once,
                    the instant IDLE -> RECORDING_COMMAND fires (see
                    "Barge-in Callback Timing" above). None (the default)
                    preserves the prior no-callback behavior exactly.

    Returns:
        Raw 16-bit PCM bytes of the captured speech segment.

    Raises:
        RuntimeError: initialise() has not been awaited yet.
    """
    if _silero_session is None:
        raise RuntimeError("VAD — call await vad.initialise() before listen_for_speech()")

    global _oww_debounce_until

    _reset_silero_state()

    use_wake_word: bool = (_oww_model is not None)

    _flushed: int = 0
    while True:
        try:
            stream.get_nowait()
            _flushed += 1
        except asyncio.QueueEmpty:
            break
    if _flushed:
        log.debug("VAD — flushed %d stale frame(s) before listening", _flushed)

    _state: _State = _State.IDLE
    _silence_start: Optional[float] = None
    _speech_start: Optional[float] = None
    _pcm_buffer: list[bytes] = []
    _command_confirmed: bool = False
    _last_prob_trace_ms: float = -1000.0

    log.debug(
        "VAD — listening (threshold=%.2f, mode=%s)",
        cfg.vad.threshold,
        "wake-word gated" if use_wake_word else "open",
    )

    while True:
        chunk: bytes = await stream.get()
        now = time.monotonic()

        if _state is _State.RECORDING_COMMAND and _speech_start is not None:
            recording_ms = (now - _speech_start) * 1000
            if recording_ms >= _MAX_RECORDING_MS:
                _pcm_buffer.append(chunk)
                segment = b"".join(_pcm_buffer)
                log.warning(
                    "VAD — max recording ceiling hit (%.0f ms >= %.0f ms), "
                    "force-closing segment (%d bytes) regardless of VAD state",
                    recording_ms,
                    _MAX_RECORDING_MS,
                    len(segment),
                )
                _state = _State.IDLE
                _command_confirmed = False
                _pcm_buffer.clear()
                _silence_start = None
                _speech_start = None
                _reset_silero_state()
                _oww_debounce_until = now + (_OWW_DEBOUNCE_MS / 1000)
                return segment

        if use_wake_word and _state is _State.IDLE:
            detected: bool = await _run_vad(_oww_infer, chunk)

            if now < _oww_debounce_until:
                if detected:
                    log.debug(
                        "VAD — wake word trigger suppressed (debounce active, "
                        "%.0f ms remaining)",
                        (_oww_debounce_until - now) * 1000,
                    )
                continue

            if detected:
                _state = _State.RECORDING_COMMAND
                _speech_start = now
                _silence_start = None
                _command_confirmed = False
                _pcm_buffer.append(chunk)
                log.info("VAD — wake word triggered, instantly recording...")
                if on_trigger is not None:
                    try:
                        on_trigger()
                    except Exception:
                        log.exception(
                            "VAD — on_trigger callback raised; continuing "
                            "(barge-in interrupt hook must never break VAD)"
                        )
            continue

        frame_f32 = _pcm16_to_f32(chunk)
        if frame_f32.shape[1] < _FRAME_SAMPLES:
            frame_f32 = np.pad(
                frame_f32, ((0, 0), (0, _FRAME_SAMPLES - frame_f32.shape[1]))
            )

        prob: float = await _run_vad(_silero_infer, frame_f32)
        is_speech_frame = prob >= cfg.vad.threshold

        if _state is _State.RECORDING_COMMAND and _speech_start is not None:
            _elapsed_ms = (now - _speech_start) * 1000
            if _elapsed_ms - _last_prob_trace_ms >= 300:
                log.info(
                    "VAD_PROB_TRACE — elapsed=%.0fms prob=%.3f threshold=%.2f "
                    "is_speech=%s command_confirmed=%s",
                    _elapsed_ms,
                    prob,
                    cfg.vad.threshold,
                    is_speech_frame,
                    _command_confirmed,
                )
                _last_prob_trace_ms = _elapsed_ms

        if is_speech_frame:
            if _state is _State.IDLE:
                _state = _State.RECORDING_COMMAND
                _speech_start = now
                _silence_start = None
                log.info(
                    "VAD — speech detected (prob=%.2f threshold=%.2f)",
                    prob,
                    cfg.vad.threshold,
                )
                if on_trigger is not None:
                    try:
                        on_trigger()
                    except Exception:
                        log.exception(
                            "VAD — on_trigger callback raised; continuing "
                            "(barge-in interrupt hook must never break VAD)"
                        )
            if not _command_confirmed:
                log.debug("VAD — command speech confirmed, silence-close threshold now snappy (800ms)")
            _command_confirmed = True
            _pcm_buffer.append(chunk)
            _silence_start = None

        else:
            if _state is _State.RECORDING_COMMAND:
                _pcm_buffer.append(chunk)

                if _silence_start is None:
                    _silence_start = now

                silence_ms = (now - _silence_start) * 1000

                if _command_confirmed:
                    effective_close_ms = max(
                        cfg.vad.silence_duration_ms, _POST_COMMAND_SILENCE_MS
                    )
                else:
                    effective_close_ms = _PRE_COMMAND_GRACE_MS

                if silence_ms >= effective_close_ms:
                    speech_ms = (now - _speech_start) * 1000 if _speech_start else 0

                    if speech_ms >= cfg.vad.min_speech_duration_ms:
                        segment = b"".join(_pcm_buffer)
                        log.info(
                            "VAD — segment complete (%.0f ms, %d bytes, "
                            "command_confirmed=%s)",
                            speech_ms,
                            len(segment),
                            _command_confirmed,
                        )
                        _state = _State.IDLE
                        _command_confirmed = False
                        _pcm_buffer.clear()
                        _silence_start = None
                        _speech_start = None
                        _reset_silero_state()
                        _oww_debounce_until = now + (_OWW_DEBOUNCE_MS / 1000)
                        return segment

                    log.debug(
                        "VAD — segment too short (%.0f ms < %d ms), discarding",
                        speech_ms,
                        cfg.vad.min_speech_duration_ms,
                    )
                    _state = _State.IDLE
                    _command_confirmed = False
                    _pcm_buffer.clear()
                    _silence_start = None
                    _speech_start = None
                    _reset_silero_state()
                    _oww_debounce_until = now + (_OWW_DEBOUNCE_MS / 1000)