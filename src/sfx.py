"""
src/sfx.py — Wake-Word Confirmation Chirp
============================================
Public interface:
    await sfx.initialise()   # once, at startup
    sfx.play_ping()          # thread-safe, synchronous, fire-and-forget
    sfx.shutdown_executor()  # once, at shutdown

Why this exists
----------------
vad.py's listen_for_speech() calls on_trigger() synchronously the exact
frame its internal state flips IDLE -> RECORDING_COMMAND — the same frame
the wake word fires. main.py already uses this hook to fire tts.stop()
for barge-in. This module adds a second on_trigger consumer: a short
audio "ping" so the user gets audible confirmation the wake word landed,
without needing to look at a screen while gaming.

Dedicated executor, not asyncio.to_thread
-------------------------------------------
Same rationale as vad.py's _vad_executor and tts.py's _tts_executor:
on_trigger runs synchronously on the event loop thread (inside
listen_for_speech, itself an awaited coroutine on the loop). Calling
sd.play() directly there risks blocking the loop on PortAudio's device
negotiation. play_ping() instead schedules the actual playback onto a
chirp-dedicated single-worker ThreadPoolExecutor via
loop.run_in_executor() as a fire-and-forget call — never awaited, since
on_trigger's contract (per vad.py) is a plain synchronous callable, not
a coroutine. A THIRD dedicated pool (distinct from vad.py's and tts.py's)
keeps this purely cosmetic feature from ever contending with, or being
starved by, either of the two latency-critical pipelines — and vice
versa, a stalled chirp device can never stall VAD or TTS.

Playback method: sd.play(), not winsound / pygame
------------------------------------------------------
sounddevice is already a hard dependency (tts.py, main.py), so this adds
no new package. winsound + SND_ASYNC would also work and is technically
"free" (OS-level async, no thread needed) but is Windows-only and would
mean two different audio APIs for two different sound sources in the
same codebase. sd.play()'s per-call device-open cost is the exact jitter
source tts.py's module docstring flags as a reason to avoid it for
speech — but that reasoning doesn't transfer here: a chirp has no
sentence-to-sentence continuity to protect, so the one-time open cost of
sd.play() is irrelevant, and reusing tts.py's persistent stream instead
would mean resampling the chirp to match whichever engine (Kokoro 24kHz
vs pyttsx3 22.05kHz) initialise() happened to select — added complexity
for no real benefit.

Known limitation: no mic-gating (see chat explanation)
-----------------------------------------------------------
vad.py's on_trigger fires on the SAME frame buffering begins. The chirp
can bleed back into the mic as leading noise in the captured segment,
exactly like the undocumented TTS self-audio risk already called out in
tts.py. Low-risk in practice (short + non-speech), but if it ever causes
bad transcriptions, the fix belongs in vad.py (a short frame-skip after
on_trigger fires), not here.
"""

from __future__ import annotations

import asyncio
import functools
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import numpy as np
import sounddevice as sd

from config import cfg
from src.logger import get_logger

log = get_logger(__name__)

try:
    import soundfile as sf
    _SOUNDFILE_AVAILABLE = True
except ImportError:
    _SOUNDFILE_AVAILABLE = False

_sfx_executor: ThreadPoolExecutor = ThreadPoolExecutor(
    max_workers=1,
    thread_name_prefix="sfx_worker",
)

_chirp_audio: Optional[np.ndarray] = None
_chirp_sample_rate: int = 0
_chirp_ready: bool = False


def shutdown_executor() -> None:
    """
    Purpose:
        Cleanly shut down the chirp-dedicated thread pool.

    Internal Mechanism:
        Mirrors vad.shutdown_executor() / tts.shutdown_executor(). Blocks
        until any in-flight playback call completes. Call once from
        main.py's shutdown sequence.

    Args:
        None.

    Returns:
        None.
    """
    log.info("SFX — shutting down dedicated executor")
    _sfx_executor.shutdown(wait=True)


async def initialise() -> None:
    """
    Purpose:
        Pre-load the wake-word confirmation chirp into memory so
        play_ping() never touches disk on the hot path.

    Internal Mechanism:
        Reads cfg.sfx.chirp_path once via soundfile on the dedicated
        executor and caches the resulting float32 array + its native
        sample rate at module scope. Deliberately NOT fail-fast: unlike
        Silero/whisper's model files (core pipeline dependencies), a
        missing or unloadable chirp file degrades to "feature disabled,
        logged once" rather than raising — the assistant's actual job
        (listening and responding) works fine without it. Also a no-op
        if cfg.sfx.enabled is False, so the feature can be turned off
        from .env without touching this module.

    Args:
        None.

    Returns:
        None.
    """
    global _chirp_audio, _chirp_sample_rate, _chirp_ready

    if not cfg.sfx.enabled:
        log.info("SFX — wake chirp disabled via config (WAKE_CHIRP_ENABLED=false)")
        return

    if not _SOUNDFILE_AVAILABLE:
        log.warning(
            "SFX — soundfile not installed; wake chirp disabled. "
            "Run: pip install soundfile"
        )
        return

    if not cfg.sfx.chirp_path.exists():
        log.warning(
            "SFX — chirp file not found at %s; wake chirp disabled "
            "(non-fatal — this is cosmetic, not a pipeline dependency). "
            "Place a short (<300ms recommended) WAV there to enable it.",
            cfg.sfx.chirp_path,
        )
        return

    try:
        loop = asyncio.get_running_loop()
        audio, sample_rate = await loop.run_in_executor(
            _sfx_executor,
            functools.partial(sf.read, str(cfg.sfx.chirp_path), dtype="float32"),
        )
        # Collapse to mono — sd.play() handles stereo fine, but mono keeps
        # this consistent with the rest of the codebase's audio path
        # (mic capture, Silero, Kokoro/pyttsx3 are all mono throughout).
        if audio.ndim > 1:
            audio = audio.mean(axis=1).astype(np.float32)

        _chirp_audio = audio
        _chirp_sample_rate = int(sample_rate)
        _chirp_ready = True
        log.info(
            "SFX — wake chirp loaded from %s (%d samples @ %d Hz, %.0f ms)",
            cfg.sfx.chirp_path,
            len(audio),
            sample_rate,
            (len(audio) / sample_rate) * 1000,
        )
    except Exception:
        log.exception(
            "SFX — failed to load chirp file at %s; wake chirp disabled",
            cfg.sfx.chirp_path,
        )
        _chirp_ready = False


def _play_sync() -> None:
    """
    Purpose:
        Blocking playback call. Runs on _sfx_executor only — never on the
        event loop thread.

    Internal Mechanism:
        sd.play() returns as soon as playback is handed to PortAudio (it
        does not block for the chirp's full duration), but the call still
        does device negotiation synchronously on whichever thread invokes
        it — the same jitter source tts.py's module docstring documents
        for per-utterance sd.play() vs. a persistent stream. Routing
        through the dedicated executor keeps that cost off the event
        loop regardless of how small it is. Wrapped in try/except so a
        transient audio-device error (e.g. exclusive-mode contention from
        the game — the same failure mode tts.py documents for its own
        stream) degrades to a logged warning, never a crash.

    Args:
        None.

    Returns:
        None.
    """
    try:
        sd.play(_chirp_audio, samplerate=_chirp_sample_rate)
    except Exception:
        log.warning("SFX — chirp playback failed (non-fatal)", exc_info=True)


def play_ping() -> None:
    """
    Purpose:
        Fire the wake-word confirmation chirp. Designed to be passed
        directly as vad.listen_for_speech()'s on_trigger argument.

    Internal Mechanism:
        Matches on_trigger's documented contract exactly (see vad.py): a
        plain synchronous callable, invoked on the event loop thread the
        instant IDLE -> RECORDING_COMMAND fires. Schedules the actual
        blocking sd.play() call onto _sfx_executor via
        loop.run_in_executor() and does NOT await the resulting future —
        intentionally fire-and-forget, since on_trigger cannot itself be
        a coroutine and nothing downstream needs to know when the chirp
        finishes playing. If called before initialise() has successfully
        loaded a chirp (disabled, missing file, or load failure), this is
        a silent no-op — vad.py already wraps on_trigger in its own
        try/except, but "chirp not configured" isn't exceptional, so it's
        handled quietly here rather than logged on every single trigger.

    Args:
        None.

    Returns:
        None.
    """
    if not _chirp_ready:
        return

    try:
        loop = asyncio.get_running_loop()
        loop.run_in_executor(_sfx_executor, _play_sync)
    except RuntimeError:
        log.warning("SFX — play_ping() called with no running event loop")