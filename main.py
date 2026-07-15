"""
main.py — Voice Assistant Orchestrator (Production Build)
=========================================================
Entry point and async event-loop conductor. Wires together:
    vad  →  stt  →  llm  →  tts

Architecture: three concurrent asyncio Tasks
--------------------------------------------

  Task A  mic_capture_loop()
      sounddevice InputStream callback pushes 512-sample float32 frames
      onto _audio_queue via call_soon_threadsafe. The callback runs on a
      PortAudio C thread; this coroutine holds the stream open by awaiting
      _shutdown_event. Zero thread-pool slots consumed while capturing.

  Task B  pipeline_loop()
      Owns the request/response state machine. Each iteration races two
      sub-tasks against each other with asyncio.wait(FIRST_COMPLETED):

        Task P  _run_pipeline(pcm_segment)
            STT → LLM → TTS, fully wrapped in CancelledError handlers so
            GPU/audio resources are released cleanly on interruption.

        Task C  _barge_in_watchdog()
            Calls vad.listen_for_speech() concurrently with Task P.
            If the user speaks before the response is finished, Task C
            returns first; pipeline_loop() cancels Task P, calls tts.stop(),
            and immediately feeds the new PCM segment back into Task P.

  Executor (ThreadPoolExecutor, max_workers=1)
      Registered as the event loop's default executor. vad.py and tts.py
      each own a SEPARATE, dedicated ThreadPoolExecutor of their own
      (loop.run_in_executor(_vad_executor, ...) / (_tts_executor, ...))
      for their GPU/CPU-bound inference and synthesis work — they do NOT
      use asyncio.to_thread() or this default executor. This default
      executor is used only for main.py's own _write_temp_wav() and
      stt.py's transcript-file read, both via asyncio.to_thread(); both
      are quick, sequential, non-blocking file I/O, not inference.
      whisper.cpp inference runs in its own subprocess
      (asyncio.create_subprocess_exec) and does NOT use any thread pool.

Subprocess strategy for whisper.cpp
-------------------------------------
whisper.cpp is launched via asyncio.create_subprocess_exec in stt.py.
This is correct and must NOT be replaced with run_in_executor(subprocess.run):

    asyncio.create_subprocess_exec:
        Event loop drives OS pipe I/O (epoll/kqueue). Zero threads held
        during GPU inference. Cancellable at every await point.

    run_in_executor(subprocess.run):
        Holds a ThreadPoolExecutor thread for the full GPU inference
        duration (300–1500 ms on RX 6700 XT). With max_workers=2 a
        single whisper call would block all TTS synthesis.

Conversation history
--------------------
A bounded collections.deque (maxlen=HISTORY_TURNS * 2) accumulates
user/assistant message dicts. The deque is passed to llm.stream_response()
so the model can reference prior turns. Capped to prevent the Ollama
context window from growing unbounded during long gaming sessions.

Signal handling
---------------
SIGINT / SIGTERM → _handle_shutdown() → _shutdown_event.set()
    • mic_capture_loop exits via _shutdown_event.wait() returning
    • sounddevice callback raises CallbackAbort on next frame
    • pipeline_loop exits its while-not-shutdown loop
    • tts.stop() kills PortAudio immediately
    • shutdown_logging() flushes the QueueListener thread

All five of these happen in the correct order in async_main()'s finally.
"""

from __future__ import annotations

import asyncio
import os
import signal
import sys
import tempfile
import wave
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import AsyncGenerator, Deque, Optional

import numpy as np
import sounddevice as sd

# ---------------------------------------------------------------------------
# .env loading — must precede all config imports so env overrides are visible.
# ---------------------------------------------------------------------------
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv is optional; raw os.environ values are used instead.

from config import cfg
from src.logger import get_logger, init_logging, shutdown_logging
import src.vad as vad
import src.stt as stt
import src.llm as llm
import src.tts as tts

# ---------------------------------------------------------------------------
# Logging — initialise before any module-level log calls.
# QueueHandler pushes records onto an in-process queue (~1 µs per call).
# A background thread drains to RotatingFileHandler → logs/assistant.log.
# Zero disk I/O on the event loop thread.
# ---------------------------------------------------------------------------
init_logging()
log = get_logger(__name__)  # "assistant.main"

log.debug("MAIN — current working directory: %s", os.getcwd())

# ---------------------------------------------------------------------------
# Tuning constants
# ---------------------------------------------------------------------------

# Hard ceiling on the event loop's DEFAULT executor.
# vad.py and tts.py each run their own dedicated ThreadPoolExecutor now
# (see their module docstrings) and do NOT draw from this one.
# This default executor is used only by main.py's _write_temp_wav() and
# stt.py's transcript-file read (both asyncio.to_thread, both quick,
# sequential file I/O within a single pipeline pass) — 1 worker is
# sufficient for that; raise this only if a future asyncio.to_thread()
# call needs to run concurrently with one of those two.
_EXECUTOR_MAX_WORKERS: int = 1

# Rolling conversation history depth.
# Each turn = 2 messages (user + assistant). deque maxlen = turns * 2.
# 5 turns ≈ 400–600 tokens of context — enough for multi-step queries
# without risking Ollama context overflow on smaller models (phi3:mini 4K ctx).
_HISTORY_TURNS: int = 5

# ---------------------------------------------------------------------------
# Global coordination primitives
# ---------------------------------------------------------------------------

# Mic capture → VAD queue.
# Unbounded (maxsize=0): VAD consumes at the same rate frames are produced
# (~32 ms/frame). A bounded queue would drop frames on momentary event-loop
# stalls (GC, syscall jitter), corrupting Silero's GRU hidden state.
_audio_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=0)

# Set by signal handlers; checked by the main loop and mic callback.
_shutdown_event: asyncio.Event = asyncio.Event()

# Reference to the running event loop — needed for call_soon_threadsafe
# in the sounddevice callback (which runs on a PortAudio C thread).
_loop: Optional[asyncio.AbstractEventLoop] = None

# ---------------------------------------------------------------------------
# Task A — Microphone capture
# ---------------------------------------------------------------------------

def _sd_callback(
    indata: np.ndarray,
    frames: int,
    time_info: object,
    status: sd.CallbackFlags,
) -> None:
    """
    sounddevice InputStream callback. Runs on a PortAudio C thread.

    Responsibility: convert a float32 audio frame to int16 PCM bytes and
    schedule it onto _audio_queue via call_soon_threadsafe.

    call_soon_threadsafe is mandatory here. asyncio.Queue is not thread-safe;
    touching it directly from this C thread would cause data races. The method
    schedules put_nowait as a callback on the event loop thread where the Queue
    was created, making the write atomic from the loop's perspective.

    Args:
        indata:    float32 ndarray, shape (frames, 1) — mono capture.
        frames:    number of samples in this block (always 512 with our blocksize).
        time_info: PortAudio timing struct (unused; VAD uses wall-clock via time.monotonic).
        status:    xrun / overflow / underflow flags.

    Raises:
        sd.CallbackAbort: when _shutdown_event is set, terminating the PortAudio
                          stream from inside the callback without a Python-level join.
    """
    if status:
        # xruns are non-fatal during gaming (momentary CPU spike). Log at DEBUG
        # to avoid flooding the console; escalate to WARNING if they are frequent.
        log.debug("MIC — PortAudio status: %s", status)

    if _shutdown_event.is_set():
        raise sd.CallbackAbort  # PortAudio-native clean stop.

    # float32 [-1.0, 1.0] → int16 PCM bytes.
    # Silero VAD ONNX expects int16; the conversion here avoids it in every
    # VAD call (done once at capture time, not repeatedly in the hot path).
    #
    # FIX [WAV format review]: np.clip guards against samples outside the
    # nominal [-1.0, 1.0] range (mic gain, driver quirks, momentary DC
    # offset). Without it, .astype(np.int16) on an out-of-range float does
    # NOT clamp — it wraps around (e.g. a value corresponding to +40000
    # silently becomes a large negative number), injecting a sharp digital
    # pop into otherwise clean audio rather than a soft ceiling.
    clipped: np.ndarray = np.clip(indata[:, 0], -1.0, 1.0)
    pcm_bytes: bytes = (clipped * 32_767).astype(np.int16).tobytes()

    if _loop is not None and not _loop.is_closed():
        _loop.call_soon_threadsafe(_audio_queue.put_nowait, pcm_bytes)


async def mic_capture_loop() -> None:
    """
    Task A: Hold the sounddevice InputStream open for the session lifetime.

    blocksize=512 matches Silero VAD's _FRAME_SAMPLES exactly. There is no
    intermediate buffer or resampling — each callback invocation produces
    exactly one Silero-ready frame.

    The coroutine itself does nothing after opening the stream: the PortAudio
    C thread invokes _sd_callback independently. We simply await _shutdown_event
    so the async context manager keeps the stream alive.
    """
    log.info(
        "MIC — opening InputStream | %d Hz mono blocksize=512",
        cfg.hardware.sample_rate,
    )

    try:
        with sd.InputStream(
            samplerate=cfg.hardware.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=512,      # Hard requirement: matches Silero _FRAME_SAMPLES.
            callback=_sd_callback,
        ):
            log.info("MIC — stream open; capture running")
            await _shutdown_event.wait()  # Yield; PortAudio C thread does the work.

    except sd.PortAudioError as exc:
        log.error(
            "MIC — PortAudio error: %s | "
            "Verify microphone is connected and not locked by another process "
            "(game push-to-talk overlays often claim exclusive device access).",
            exc,
        )
        _shutdown_event.set()  # Propagate to all other tasks — no point continuing.

    except asyncio.CancelledError:
        log.debug("MIC — cancelled during shutdown")
        raise  # Re-raise so asyncio.gather() sees clean cancellation.

    finally:
        log.info("MIC — stream closed")


# ---------------------------------------------------------------------------
# WAV file helper
# ---------------------------------------------------------------------------

def _write_temp_wav_sync(pcm_bytes: bytes) -> Path:
    """
    Synchronous worker for _write_temp_wav() — see that function for the
    asyncio.to_thread rationale. All actual file I/O happens here, on a
    thread-pool thread, never on the event loop.

    Serialise raw 16-bit 16 kHz mono PCM to a temp WAV file on disk.

    whisper.cpp requires a file path; it cannot accept a byte buffer on
    stdin. We write to cfg.paths.audio_dir (not /tmp) so the file lives
    on the same drive as the project, avoiding cross-device rename issues.

    The returned Path's lifecycle:
        Created here → passed to stt.transcribe() → deleted in stt.py's
        finally block, even if transcription fails or is cancelled.

    CancelledError safety:
        If the enclosing coroutine (_run_pipeline) is cancelled between
        _write_temp_wav() returning and stt.transcribe() taking ownership,
        _run_pipeline's except CancelledError block calls unlink() explicitly
        to prevent the file from being orphaned. See _run_pipeline below.

    Args:
        pcm_bytes: Raw 16-bit PCM from vad.listen_for_speech().

    Returns:
        Path to the written WAV file (delete=False NamedTemporaryFile).
    """
    cfg.paths.audio_dir.mkdir(parents=True, exist_ok=True)

    # FIX [WAV format review]: 16-bit mono PCM requires an even byte count
    # (2 bytes/sample). mic_capture_loop's fixed blocksize=512 and vad.py's
    # matching _FRAME_SAMPLES=512 mean every chunk is exactly 1024 bytes
    # today, so an odd-length buffer shouldn't reach here — but if it ever
    # does, Python's wave module will NOT raise: writeframes() writes every
    # byte handed to it and _patchheader() silently reconciles the header's
    # declared size to match (verified against the installed wave module),
    # leaving a dangling half-sample with no error surfaced anywhere. A
    # silent truncation is a worse failure mode than a loud one.
    if len(pcm_bytes) % 2 != 0:
        log.warning(
            "STT_WAV — pcm_bytes has an ODD length (%d bytes), not a whole "
            "number of 16-bit samples. Dropping the dangling final byte. "
            "This should not happen given the current capture pipeline — "
            "check mic_capture_loop/vad.py chunk sizing if this recurs.",
            len(pcm_bytes),
        )
        pcm_bytes = pcm_bytes[:-1]

    sample_width = 2   # bytes/sample — 16-bit PCM, whisper.cpp requirement
    n_channels = 1
    sample_rate = cfg.hardware.sample_rate

    # FIX [Robust verification]: Log enough to tell a genuinely malformed/
    # truncated capture apart from a legitimately near-silent segment —
    # which vad.py's grace-period/debounce design can deliberately return
    # (see vad.py's Instant Record Override notes: a segment with no
    # confirmed command speech still gets returned rather than discarded,
    # specifically to avoid a ghost-audio re-trigger loop). Peak amplitude
    # is a cheap way to distinguish "quiet room, real audio" from "actual
    # silence" without a full RMS/dB computation.
    n_samples = len(pcm_bytes) // sample_width
    duration_ms = (n_samples / sample_rate) * 1000
    peak: int = int(np.abs(np.frombuffer(pcm_bytes, dtype=np.int16)).max()) if n_samples else 0

    log.info(
        "STT_WAV — %d bytes | %d samples | %.0f ms @ %d Hz | peak_amplitude=%d/32767",
        len(pcm_bytes),
        n_samples,
        duration_ms,
        sample_rate,
        peak,
    )
    if peak < 50:
        log.debug(
            "STT_WAV — near-silent segment (peak=%d) — an empty whisper.cpp "
            "transcript for this file is an expected outcome of vad.py's "
            "grace-period design, not necessarily a format bug",
            peak,
        )

    tmp = tempfile.NamedTemporaryFile(
        suffix=".wav",
        dir=cfg.paths.audio_dir,
        delete=False,   # whisper.cpp subprocess must open by path; cannot auto-delete.
    )

    try:
        with wave.open(tmp, "wb") as wf:
            wf.setnchannels(n_channels)
            wf.setsampwidth(sample_width)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm_bytes)

        # FIX [Robust verification]: wave.Wave_write.close() (invoked by the
        # `with` block above) already calls self._file.flush() unconditionally
        # — verified against the installed module, independent of whether
        # wave opened the file itself. fsync() goes one step further: cheap
        # for a file this size, and guarantees the bytes are durable on disk
        # (not just sitting in the OS page cache) before whisper.cpp is
        # asked to open this exact path.
        os.fsync(tmp.fileno())
    except Exception:
        tmp.close()
        Path(tmp.name).unlink(missing_ok=True)
        raise

    tmp.close()
    path = Path(tmp.name)
    log.debug("MIC — wrote temp WAV: %s (%d PCM bytes)", path.name, len(pcm_bytes))
    return path


async def _write_temp_wav(pcm_bytes: bytes) -> Path:
    """
    Serialise raw 16-bit 16 kHz mono PCM to a temp WAV file on disk.

    FIX [Non-blocking file I/O]: The actual write now runs via
    asyncio.to_thread rather than directly on the event loop. Same
    rationale stt.py used to justify create_subprocess_exec over
    run_in_executor(subprocess.run): _write_temp_wav_sync has no internal
    await points, so once called directly it blocks the entire event loop
    — including the VAD barge-in watchdog — for its full duration. The
    write is normally small (tens to a few hundred KB) and fast, but
    there's no reason to accept even an occasional stall from disk
    contention (e.g. the game streaming assets from the same drive) when
    moving it off the loop thread costs one thread-pool slot for a few
    milliseconds.

    See _write_temp_wav_sync() for the full docstring covering file
    lifecycle, CancelledError safety, and format details.

    Args:
        pcm_bytes: Raw 16-bit PCM from vad.listen_for_speech().

    Returns:
        Path to the written WAV file (delete=False NamedTemporaryFile).
    """
    return await asyncio.to_thread(_write_temp_wav_sync, pcm_bytes)


# ---------------------------------------------------------------------------
# Task P — Single pipeline pass: STT → LLM → TTS
# ---------------------------------------------------------------------------

async def _run_pipeline(
    pcm_segment: bytes,
    history: Deque[dict[str, str]],
) -> None:
    """
    Execute one complete request/response cycle for a captured speech segment.

    This coroutine is launched as a cancellable asyncio.Task. If barge-in
    occurs, asyncio.Task.cancel() propagates CancelledError at the next
    await point. Every stage has an explicit CancelledError handler to
    ensure hardware resources are released before re-raising.

    GPU resource release on cancellation
    --------------------------------------
    STT stage:
        whisper.cpp runs as a subprocess (asyncio.create_subprocess_exec).
        On CancelledError the subprocess is still running; proc.kill() sends
        SIGKILL and proc.wait() reaps it, releasing GPU/VRAM immediately.
        The temp WAV file is unlinked in the CancelledError handler because
        stt.transcribe()'s own finally block may not have run yet if we're
        cancelled between _write_temp_wav() and stt.transcribe() taking the path.

    LLM stage:
        httpx.AsyncClient uses the event loop's selector for network I/O.
        Cancellation closes the HTTP connection automatically — no explicit
        cleanup needed. Ollama will detect the closed socket and stop inference,
        freeing VRAM for the new pipeline cycle.

    TTS stage:
        tts.stop() is called by pipeline_loop() before cancelling this task,
        so the PortAudio stream is already dead by the time CancelledError
        propagates here. No additional audio cleanup is needed.

    Args:
        pcm_segment: Raw 16-bit PCM bytes from vad.listen_for_speech().
        history:     Shared conversation deque. Appended to by this function
                     on successful round-trips. Passed by reference; the deque's
                     maxlen enforces the rolling window automatically.
    """
    audio_path: Optional[Path] = None

    # ── Stage 1: STT ──────────────────────────────────────────────────────
    try:
        log.info("PIPELINE — STT start")
        audio_path = await _write_temp_wav(pcm_segment)

        # stt.transcribe() drives whisper.cpp via asyncio.create_subprocess_exec:
        #   • Non-blocking: event loop drives pipe I/O, no thread held.
        #   • Vulkan flags (-vd, -t) are built in stt._build_whisper_cmd().
        #   • Deletes audio_path in its own finally block.
        transcript: str = await stt.transcribe(audio_path)
        audio_path = None  # stt.transcribe() now owns cleanup.

    except asyncio.CancelledError:
        log.info("PIPELINE — cancelled during STT; cleaning up")
        # stt.transcribe() may not have taken ownership of audio_path yet.
        if audio_path is not None:
            audio_path.unlink(missing_ok=True)
            log.debug("PIPELINE — orphaned WAV removed: %s", audio_path.name)
        raise  # Must re-raise CancelledError — swallowing it breaks asyncio.

    except Exception:
        log.exception("PIPELINE — STT error")
        if audio_path is not None:
            audio_path.unlink(missing_ok=True)
        return  # Non-fatal; recover and return to idle.

    if not transcript:
        log.info("PIPELINE — empty transcript; returning to idle")
        return

    log.info("PIPELINE — transcript: %r", transcript)
    history.append({"role": "user", "content": transcript})

    # ── Stage 2: LLM (streaming) ──────────────────────────────────────────
    try:
        log.info(
            "PIPELINE — LLM start | model=%s history_turns=%d",
            cfg.ollama.model,
            len(history) // 2,
        )

        # Pass full history so the model has conversational context.
        # llm.stream_response() is an AsyncGenerator; we do NOT await it here.
        # It is passed directly to tts.speak_stream() so TTS synthesis begins
        # on the first sentence token — before the full LLM response is complete.
        token_stream = llm.stream_response(list(history))

        # ── Stage 3: TTS (sentence-buffered) ────────────────────────────
        log.info("PIPELINE — TTS start")

        # Accumulate the full text for history during streaming playback.
        # tts.speak_stream() consumes token_stream internally; we intercept
        # tokens by wrapping the generator in _accumulate_and_stream().
        full_reply_parts: list[str] = []
        await tts.speak_stream(_accumulate_stream(token_stream, full_reply_parts))

        # Commit assistant reply to history only after full playback.
        if full_reply_parts:
            reply_text = "".join(full_reply_parts)
            history.append({"role": "assistant", "content": reply_text})
            log.debug("PIPELINE — history updated (%d messages)", len(history))

        log.info("PIPELINE — cycle complete")

    except asyncio.CancelledError:
        # LLM: httpx connection closed by CancelledError propagation — Ollama
        # detects the broken socket and stops generation automatically.
        # TTS: already stopped by pipeline_loop() calling tts.stop() before
        # issuing the cancel.
        log.info("PIPELINE — cancelled during LLM/TTS; GPU inference stopped")
        # Do NOT append a partial reply to history — it would corrupt context.
        raise

    except Exception:
        log.exception("PIPELINE — LLM/TTS error")
        # Non-fatal; discard partial history entry and return to idle.


async def _accumulate_stream(
    token_stream: "AsyncGenerator[str, None]",
    accumulator: list[str],
) -> "AsyncGenerator[str, None]":
    """
    Transparent pass-through async generator that accumulates tokens.

    Wraps llm.stream_response()'s output so tts.speak_stream() receives
    tokens normally while we build the full reply string for history.

    Args:
        token_stream: AsyncGenerator from llm.stream_response().
        accumulator:  list[str] that receives each token. Caller reads it
                      after the generator is exhausted.

    Yields:
        str: Each token delta, unmodified.
    """
    async for token in token_stream:
        accumulator.append(token)
        yield token


# ---------------------------------------------------------------------------
# Task C — Barge-in watchdog
# ---------------------------------------------------------------------------

async def _barge_in_watchdog() -> bytes:
    """
    Task C: Monitor for new speech while the pipeline is active.

    Calls vad.listen_for_speech(_audio_queue), which blocks asynchronously
    on the queue awaiting Silero-confirmed speech (with optional wake word
    gating). Returns the raw PCM segment of the new utterance.

    Cancellation (when pipeline finishes first) is received at the
    `await stream.get()` inside vad.listen_for_speech() — a clean await
    point with no resource leaks.

    Returns:
        Raw 16-bit PCM bytes of the detected utterance.
    """
    log.debug("WATCHDOG — armed")
    segment: bytes = await vad.listen_for_speech(_audio_queue)
    log.info("WATCHDOG — barge-in detected (%d bytes)", len(segment))
    return segment


# ---------------------------------------------------------------------------
# Task B — Pipeline orchestration loop
# ---------------------------------------------------------------------------

async def pipeline_loop() -> None:
    """
    Task B: Main state machine. Runs for the session lifetime.

    Per-iteration flow:
      1. Await vad.listen_for_speech() for a confirmed speech segment.
         (Skipped if a barge-in segment is already queued from last cycle.)
      2. Launch Task P (_run_pipeline) and Task C (_barge_in_watchdog).
      3. asyncio.wait(FIRST_COMPLETED) — race P vs C.
         a. P wins (normal):   cancel C, loop to step 1.
         b. C wins (barge-in): call tts.stop(), cancel P, use C's segment.
      4. On shutdown: exit while-loop → log → return.

    Exception policy:
        CancelledError is re-raised immediately (propagates shutdown).
        All other exceptions are caught, logged, and the loop continues
        — the assistant must never crash on a single bad utterance.
    """
    log.info("PIPELINE — loop started")

    # Rolling conversation history. deque.maxlen enforces the cap automatically;
    # older messages are evicted from the left when the cap is reached.
    history: Deque[dict[str, str]] = deque(maxlen=_HISTORY_TURNS * 2)

    pending_segment: Optional[bytes] = None  # Barge-in carry-over

    while not _shutdown_event.is_set():
        try:
            # ── Step 1: Acquire speech segment ────────────────────────────
            if pending_segment is None:
                # FIX [Log-level visibility]: was log.debug() — same class of
                # bug as the watchdog-cancellation lines two turns ago. If
                # your logger is at INFO threshold, this line (and hence any
                # confirmation the loop actually reached this point) was
                # invisible the whole time, independent of whether the code
                # was really executing here or not.
                log.info("PIPELINE — idle; waiting for speech...")
                log.info("PIPELINE — Re-entering listen_for_speech...")
                try:
                    pcm_segment: bytes = await vad.listen_for_speech(_audio_queue)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    log.exception("PIPELINE — VAD error; re-entering listen")
                    await asyncio.sleep(0.1)  # Back-off to avoid tight spin on persistent errors.
                    continue
            else:
                pcm_segment = pending_segment
                pending_segment = None

            log.info(
                "PIPELINE — segment ready (%d bytes); starting pipeline",
                len(pcm_segment),
            )

            # ── Step 2: Launch pipeline and watchdog concurrently ─────────
            pipeline_task: asyncio.Task[None] = asyncio.create_task(
                _run_pipeline(pcm_segment, history),
                name="pipeline",
            )
            watchdog_task: asyncio.Task[bytes] = asyncio.create_task(
                _barge_in_watchdog(),
                name="watchdog",
            )

            # ── Step 3: Race ──────────────────────────────────────────────
            done, _ = await asyncio.wait(
                {pipeline_task, watchdog_task},
                return_when=asyncio.FIRST_COMPLETED,
            )

            if watchdog_task in done:
                # ── Barge-in path ─────────────────────────────────────────
                log.info("PIPELINE — barge-in: stopping TTS and cancelling pipeline")

                tts.stop()  # Kill PortAudio immediately — before cancel so
                            # _play_audio()'s sd.stop() and our sd.stop() don't race.

                log.info("PIPELINE — cancelling pipeline task (barge-in)...")
                pipeline_task.cancel()
                try:
                    await pipeline_task  # Wait for CancelledError to propagate and
                                         # all except/finally blocks in _run_pipeline to run.
                except (asyncio.CancelledError, Exception):
                    pass  # CancelledError is expected; other exceptions already logged.
                log.info("PIPELINE — pipeline task cancellation complete")

                try:
                    new_segment: bytes = watchdog_task.result()
                    pending_segment = new_segment
                    log.info(
                        "PIPELINE — barge-in segment queued (%d bytes); restarting",
                        len(new_segment),
                    )
                except Exception:
                    log.exception("PIPELINE — watchdog result retrieval failed")
                    pending_segment = None

            else:
                # ── Normal completion path ────────────────────────────────
                log.info("PIPELINE — cancelling watchdog task...")
                watchdog_task.cancel()
                try:
                    await watchdog_task  # Allow CancelledError to propagate inside watchdog.
                except (asyncio.CancelledError, Exception):
                    pass
                # DEBUG: if this line is missing from the logs after a hang,
                # the watchdog's await never returned — meaning it's stuck
                # inside vad.listen_for_speech(), most likely at a
                # run_in_executor() call that can't be force-cancelled
                # while genuinely in-flight on a worker thread.
                log.info("PIPELINE — watchdog task cancellation complete")

                # Surface pipeline exceptions (they were already logged inside
                # _run_pipeline, but we call .result() to prevent "Task exception
                # was never retrieved" warnings in the asyncio debug logs).
                try:
                    pipeline_task.result()
                except Exception:
                    pass  # Already logged inside _run_pipeline.

        except asyncio.CancelledError:
            log.info("PIPELINE — loop cancelled; shutting down")
            raise

        except Exception:
            # FIX [Silent loop death]: CRITICAL, not ERROR — this is the
            # top-level orchestration loop; anything reaching here means a
            # full cycle failed to complete cleanly. exc_info=True forces
            # the full traceback even if root-logger config would otherwise
            # trim it. This mechanism already existed (as log.exception,
            # functionally identical) — bumped to CRITICAL per request, but
            # note it only ever catches EXCEPTIONS. A genuine hang (stuck at
            # an await that never resolves) will not land here — see the
            # watchdog-cancellation logging above for that failure mode.
            log.critical(
                "PIPELINE — unhandled exception in orchestration loop; recovering",
                exc_info=True,
            )
            await asyncio.sleep(0.2)  # Brief back-off before retrying.

    log.info("PIPELINE — shutdown event set; loop exiting")


# ---------------------------------------------------------------------------
# Signal handling
# ---------------------------------------------------------------------------

def _register_signal_handlers(loop: asyncio.AbstractEventLoop) -> None:
    """
    Register SIGINT and SIGTERM handlers on the asyncio event loop.

    loop.add_signal_handler() (POSIX) fires as a scheduled event-loop
    callback, not as a raw UNIX signal handler. This means it can safely
    call asyncio primitives (Event.set()) and logging without risking
    re-entrancy issues that plague signal.signal() in async code.

    Windows fallback: loop.add_signal_handler is not implemented on Windows.
    We fall back to signal.signal(), which is slightly less clean but
    functionally correct for a single-threaded asyncio program.

    Args:
        loop: The running event loop to attach handlers to.
    """
    def _handle_shutdown(signum: int) -> None:
        sig_name = signal.Signals(signum).name
        log.info("SIGNAL — %s received; initiating graceful shutdown", sig_name)
        tts.stop()          # Abort PortAudio immediately before the loop exits.
        _shutdown_event.set()  # Unblocks _shutdown_event.wait() in mic_capture_loop
                               # and exits pipeline_loop()'s while condition.

    try:
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, _handle_shutdown, sig)
        log.debug("SIGNAL — POSIX handlers registered (SIGINT, SIGTERM)")

    except (NotImplementedError, AttributeError):
        # Windows
        def _win_handler(signum: int, frame: object) -> None:
            _handle_shutdown(signum)

        signal.signal(signal.SIGINT, _win_handler)
        signal.signal(signal.SIGTERM, _win_handler)
        log.debug("SIGNAL — Windows handlers registered (SIGINT, SIGTERM)")


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

async def _startup() -> None:
    """
    Initialise all subsystems before entering the main loop.

    Initialisation order is mandatory:
      1. cfg.validate() — creates audio/models/logs directories if
         missing, then runs non-fatal path checks (logs errors only).
      2. vad.initialise() — loads Silero ONNX (~1.8 MB) + optional OWW model.
      3. tts.initialise() — loads Kokoro-82M on CPU or pyttsx3 fallback.

    stt (whisper.cpp) and llm (Ollama) are subprocess/HTTP-based and
    require no Python-side init — they are invoked fresh per request.
    """
    log.info("STARTUP — voice assistant initialising")
    log.info(
        "STARTUP — AMD RX 6700 XT | vulkan_device=%d | "
        "ollama_num_gpu=%d | whisper_threads=%d | ollama_model=%s",
        cfg.hardware.vulkan_device_id,
        cfg.hardware.ollama_num_gpu,
        cfg.hardware.whisper_threads,
        cfg.ollama.model,
    )

    cfg.validate()                  # Non-fatal path checks.

    log.info("STARTUP — loading VAD (Silero ONNX + optional OpenWakeWord)...")
    await vad.initialise()          # CPU only; ~200 ms first load.

    log.info("STARTUP — loading TTS (Kokoro-82M / pyttsx3)...")
    await tts.initialise()          # CPU only; ~200 ms first load.

    log.info("STARTUP — all subsystems ready")


# ---------------------------------------------------------------------------
# Top-level async entry point
# ---------------------------------------------------------------------------

async def async_main() -> None:
    """
    Top-level coroutine: configure executor, run startup, launch tasks,
    orchestrate shutdown.

    Executor configuration
    ----------------------
    We install a ThreadPoolExecutor with max_workers=1 as the event loop's
    default executor before startup. vad.py and tts.py do not draw from
    this pool — each owns its own dedicated ThreadPoolExecutor, called via
    loop.run_in_executor() explicitly rather than asyncio.to_thread(). This
    default executor exists only for the two remaining asyncio.to_thread()
    call sites in the codebase:

        main.py's _write_temp_wav()   — writes the captured PCM to a WAV
        stt.py's transcript-file read — reads whisper.cpp's -otxt sidecar

    Both are quick, sequential file I/O within a single pipeline pass, not
    inference — 1 worker is sufficient.

    whisper.cpp: asyncio.create_subprocess_exec → 0 workers, any pool.
    Ollama:      httpx.AsyncClient → 0 workers, any pool.
    """
    global _loop
    _loop = asyncio.get_running_loop()

    # Install bounded executor before any asyncio.to_thread calls.
    executor = ThreadPoolExecutor(
        max_workers=_EXECUTOR_MAX_WORKERS,
        thread_name_prefix="assistant_worker",
    )
    _loop.set_default_executor(executor)
    log.info("MAIN — ThreadPoolExecutor installed (max_workers=%d)", _EXECUTOR_MAX_WORKERS)

    _register_signal_handlers(_loop)

    try:
        await _startup()
    except Exception:
        log.exception("STARTUP — fatal initialisation error; aborting")
        executor.shutdown(wait=False)
        
        tts.shutdown_executor()
        vad.shutdown_executor()
        
        log.info("MAIN — shutdown complete")
        return

    log.info("MAIN — launching tasks")

    capture_task: asyncio.Task[None] = asyncio.create_task(
        mic_capture_loop(), name="mic_capture"
    )
    pipeline_task_outer: asyncio.Task[None] = asyncio.create_task(
        pipeline_loop(), name="pipeline_loop"
    )

    # FIX [Silent task death]: previously only `await _shutdown_event.wait()`
    # was here. If EITHER background task died from an exception that
    # escapes its own top-level handlers during normal operation — e.g.
    # asyncio.CancelledError arriving somewhere unexpected; it's been a
    # BaseException subclass (not Exception) since Python 3.8, so a bare
    # `except Exception:` anywhere in the call chain will NOT catch a stray
    # one — nothing would notice. The other task (mic_capture_loop) keeps
    # running, so the process never exits either; it just sits there
    # indefinitely with a dead pipeline and no signal anything is wrong.
    # The ONLY place that exception would ever have surfaced was the
    # asyncio.gather() in the finally block below — which only runs at
    # shutdown. This is the most likely explanation for "hangs after cycle
    # complete, nothing further happens, no crash visible anywhere."
    #
    # Now races _shutdown_event.wait() against both background tasks
    # directly, so an unexpected death is detected and logged the moment
    # it happens, not discovered eventually (if ever) at shutdown.
    shutdown_wait_task: asyncio.Task[bool] = asyncio.create_task(
        _shutdown_event.wait(), name="shutdown_wait"
    )

    try:
        done, _ = await asyncio.wait(
            {capture_task, pipeline_task_outer, shutdown_wait_task},
            return_when=asyncio.FIRST_COMPLETED,
        )

        if not _shutdown_event.is_set():
            # A background task ended on its own — shutdown was never
            # requested. This is unconditionally a bug; surface it loudly.
            for task in (capture_task, pipeline_task_outer):
                if task in done:
                    exc = task.exception()
                    log.critical(
                        "MAIN — task '%s' terminated unexpectedly during "
                        "normal operation (shutdown was NOT requested) — "
                        "this is why the assistant appears to hang: %r",
                        task.get_name(),
                        exc,
                        exc_info=exc,
                    )
            _shutdown_event.set()  # Route through the same clean shutdown path below.

    finally:
        log.info("MAIN — shutdown: cancelling tasks")

        for task in (capture_task, pipeline_task_outer, shutdown_wait_task):
            if not task.done():
                task.cancel()

        results = await asyncio.gather(
            capture_task,
            pipeline_task_outer,
            shutdown_wait_task,
            return_exceptions=True,
        )

        for task, result in zip(
            (capture_task, pipeline_task_outer, shutdown_wait_task), results
        ):
            if isinstance(result, Exception) and not isinstance(
                result, asyncio.CancelledError
            ):
                log.error(
                    "MAIN — task '%s' raised on shutdown: %s",
                    task.get_name(),
                    result,
                )

        tts.stop()  # Belt-and-suspenders: kill PortAudio if still active.

        log.info("MAIN — shutting down executor")
        executor.shutdown(wait=True)  # wait=True: drain in-flight to_thread calls.

        log.info("MAIN — shutdown complete")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        # asyncio.run() installs its own SIGINT handler before our custom one
        # is registered. This catches the narrow window during startup where
        # Ctrl+C would otherwise print a bare traceback.
        pass
    finally:
        # Always runs — even on unhandled exceptions or hard exits.
        # Flushes the QueueListener background thread so no log records
        # are lost. This is the last line of Python that executes.
        shutdown_logging()
        sys.exit(0)