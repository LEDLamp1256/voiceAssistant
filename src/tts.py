"""
Public Interface
------------------
    await tts.initialise()                # once, at startup
    await tts.speak_stream(token_stream)   # consume an LLM token AsyncGenerator
    tts.stop()                             # thread-safe barge-in interrupt
    tts.shutdown_executor()                # once, at shutdown

Pipeline Architecture
------------------------
Three-layer pipeline:

    Layer 1 — Sentence Buffering (asyncio coroutine). speak_stream()
    consumes the LLM token generator character by character, accumulating
    text until it detects sentence-ending punctuation (. ? !). Each
    complete sentence is immediately handed to Layer 2 while the coroutine
    continues buffering the next one.

    Layer 2 — Audio Synthesis (dedicated ThreadPoolExecutor). _synthesise()
    converts a sentence string to a float32 PCM numpy array via Kokoro
    (primary) or pyttsx3 (fallback). Synthesis runs on a TTS-dedicated
    thread pool — not the asyncio default executor — so a slow synthesis
    call can never starve vad.py's wake-word/VAD inference, which lives on
    its own dedicated pool.

    Layer 3 — Playback (persistent PortAudio OutputStream, shared mode).
    Rather than opening a fresh PortAudio stream per sentence — a real
    jitter source on Windows, since each sd.play() negotiates the device
    and allocates buffers from scratch — a single OutputStream is opened
    at initialise() time and fed via a callback that pulls from an
    internal queue. Synthesised sentences are pushed onto that queue; the
    callback drains it continuously, removing per-sentence stream-open
    latency. Shared mode is used exclusively — see below.

Barge-in / Interruption Flow
--------------------------------
    VAD detects speech while TTS is playing
        -> main.py calls tts.stop()
            -> _stop_event.set()
                -> playback queue flushed, in-flight chunk cleared
                -> speak_stream() sees the flag, discards pending sentences
                -> "TTS — interrupted" is logged

Dedicated Executor vs. asyncio.to_thread
-------------------------------------------
asyncio.to_thread() always targets the event loop's default executor. In
main.py that pool is shared with whatever else calls to_thread() —
historically including vad.py's Silero/OpenWakeWord inference. If TTS
synthesis or a playback-queue put stalls (e.g. a GC pause or disk-backed
model load), VAD calls queued behind it stall too, making the assistant
deaf to wake words during gameplay. A dedicated pool, addressed via
loop.run_in_executor(_tts_executor, ...), structurally cannot be
contended by VAD or anything else. It uses two workers: one for synthesis
(_synthesise_kokoro / _synthesise_pyttsx3) and one for playback-queue
puts (which rarely block, since the queue has headroom); this also
enables future pipelining, where sentence N+1 can synthesize while
sentence N is still draining into the output stream.

Persistent OutputStream vs. WASAPI Exclusive Mode
-------------------------------------------------------
On Windows, sd.play() opens a brand-new PortAudio stream on every call —
device negotiation and buffer allocation cost real, variable time and are
themselves a jitter source independent of OS scheduling pressure. A
single long-lived OutputStream removes that per-utterance cost.

WASAPI exclusive mode was evaluated for its real benefits (bypassing the
Windows audio mixer, MMCSS thread priority) but is wrong for this project
specifically: exclusive mode gives the requesting stream sole hardware
access, which is fundamentally incompatible with the assistant running
alongside a game that also needs audio output. Worse than simply blocking
the game, it produced a silent failure mode — the stream opens
successfully (no exception, clean logs) and then dies later, with the
callback ceasing to fire and no error surfaced to Python, once the game
also needed the device. Without an explicit stream.active check,
speak_stream()'s drain-wait had no way to distinguish "still playing"
from "callback silently died," which previously manifested as a hang all
the way to the drain-wait safety ceiling. Shared mode is slightly less
performant in isolation but is the only mode compatible with this
project's actual requirement.

Playback-Completion Drain-Wait
-----------------------------------
speak_stream() must not return until audio has actually finished playing,
not merely been enqueued — _play_audio() is only a queue.put(); real
playback happens asynchronously on _output_callback's own PortAudio
thread. Without waiting for it, speak_stream() would return (re-arming
listen_for_speech()) while the response is still audibly playing. With no
mic-mute or echo cancellation anywhere in this codebase, that self-audio
could reach the microphone and false-trigger OpenWakeWord, arming vad.py's
ghost-audio debounce and silently swallowing a real wake-word attempt
that lands in the window right after the response ends. The drain-wait
therefore polls _playback_queue and _current_chunk (mutated on
PortAudio's own thread, not this one) rather than blocking on them
directly, and is bounded by _MAX_DRAIN_WAIT_S as a safety ceiling —
consistent with every other hard timeout in this project (stt.py's
subprocess timeout, vad.py's recording ceiling) — so a genuine bug in the
drain-detection logic degrades to a bounded delay instead of a true hang.
It also checks _output_stream.active on every poll tick: a dead callback
(stream crashed, device lost, or the exact exclusive-mode failure mode
described above) leaves the queue/chunk permanently non-empty with
nothing left to consume them, which would otherwise be indistinguishable
from genuinely still playing until the full ceiling expired.

Configuration Constants
---------------------------
_SENTENCE_END_RE matches sentence-terminating punctuation (. ? !),
optionally followed by closing quotes/brackets, then whitespace or
end-of-string — used to detect complete sentences in the token buffer.
_MIN_SENTENCE_CHARS (6) is a floor below which a "sentence" match is
treated as too short to be worth a synthesis call. _KOKORO_SAMPLE_RATE
(24 kHz) and _PYTTSX3_SAMPLE_RATE (22.05 kHz) are each engine's native
output rate, used to configure the persistent OutputStream to match
whichever engine initialise() selects. _MAX_DRAIN_WAIT_S (30 s) and
_DRAIN_POLL_INTERVAL_S (50 ms) bound and pace the playback-completion
drain-wait described above.
"""

from __future__ import annotations

import asyncio
import queue as _queue
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import AsyncGenerator, Optional, Any
import functools

import numpy as np
import sounddevice as sd

from config import cfg
from src.logger import get_logger

log = get_logger(__name__)

try:
    from kokoro import KPipeline
    _KOKORO_AVAILABLE = True
except ImportError:
    _KOKORO_AVAILABLE = False

try:
    import pyttsx3
    _PYTTSX3_AVAILABLE = True
except ImportError:
    _PYTTSX3_AVAILABLE = False

_SENTENCE_END_RE: re.Pattern[str] = re.compile(
    r'[.?!]+["\')}\]]*(?:\s|$)'
)

_MIN_SENTENCE_CHARS: int = 6

_KOKORO_SAMPLE_RATE: int = 24_000
_PYTTSX3_SAMPLE_RATE: int = 22_050

_MAX_DRAIN_WAIT_S: float = 30.0
_DRAIN_POLL_INTERVAL_S: float = 0.05

_tts_executor: ThreadPoolExecutor = ThreadPoolExecutor(
    max_workers=2,
    thread_name_prefix="tts_worker",
)

async def _run_tts(fn, *args, **kwargs) -> Any:
    """
    Purpose:
        Route a blocking TTS call (synthesis or a playback-queue put)
        through the TTS-dedicated thread pool instead of asyncio's shared
        default executor.

    Internal Mechanism:
        Binds fn and its arguments into a single callable via
        functools.partial and awaits it on _tts_executor. Isolating TTS
        work onto its own pool guarantees it can never be delayed by, or
        delay, vad.py's Silero/OpenWakeWord inference on its separate
        dedicated pool.

    Args:
        fn: A blocking, synchronous (or constructor) callable to execute
            off the event loop.
        *args: Positional arguments passed to fn.
        **kwargs: Keyword arguments passed to fn.

    Returns:
        The return value of fn(*args, **kwargs).
    """
    loop = asyncio.get_running_loop()
    func_with_args = functools.partial(fn, *args, **kwargs)
    return await loop.run_in_executor(_tts_executor, func_with_args)


def shutdown_executor() -> None:
    """
    Purpose:
        Cleanly shut down the TTS-dedicated thread pool.

    Internal Mechanism:
        Intended to be called once from main.py's shutdown sequence, after
        the output stream has been stopped. wait=True blocks until any
        in-flight synthesis call completes before the process exits.

    Args:
        None.

    Returns:
        None.
    """
    log.info("TTS — shutting down dedicated executor")
    _tts_executor.shutdown(wait=True)


_kokoro_pipeline: Optional["KPipeline"] = None
_pyttsx3_engine: Optional["pyttsx3.Engine"] = None
_stop_event: threading.Event = threading.Event()
_engine_mode: str = "none"

_output_stream: Optional[sd.OutputStream] = None
_output_sample_rate: int = _KOKORO_SAMPLE_RATE
_playback_queue: "_queue.Queue[np.ndarray]" = _queue.Queue(maxsize=8)
_current_chunk: Optional[np.ndarray] = None
_current_pos: int = 0


async def initialise() -> None:
    """
    Purpose:
        Load the TTS engine and open the persistent playback stream. Must
        be awaited exactly once from main.py before speak_stream() is
        used.

    Internal Mechanism:
        Attempts Kokoro first (neural, higher quality); falls back to
        pyttsx3 if Kokoro fails to load. Sets _engine_mode and
        _output_sample_rate to match whichever engine succeeds, then opens
        the persistent PortAudio output stream at that sample rate via
        _open_output_stream() — kept open for the assistant's entire
        lifetime rather than reopened per utterance.

    Args:
        None.

    Returns:
        None.

    Raises:
        RuntimeError: neither Kokoro nor pyttsx3 is available/initializable.
    """
    global _kokoro_pipeline, _pyttsx3_engine, _engine_mode, _output_sample_rate

    if _KOKORO_AVAILABLE:
        try:
            _kokoro_pipeline = await _run_tts(KPipeline, lang_code="a")
            _engine_mode = "kokoro"
            _output_sample_rate = _KOKORO_SAMPLE_RATE
            log.info(
                "TTS — Kokoro loaded | voice=%s speed=%.2f device=%s",
                cfg.tts.voice,
                cfg.tts.speed,
                cfg.tts.device,
            )
        except Exception:
            log.warning(
                "TTS — Kokoro failed to load, falling back to pyttsx3.",
                exc_info=True,
            )

    if _engine_mode == "none" and _PYTTSX3_AVAILABLE:
        try:
            _pyttsx3_engine = await _run_tts(pyttsx3.init)
            await _run_tts(
                _pyttsx3_engine.setProperty, "rate",
                int(200 * cfg.tts.speed),
            )
            _engine_mode = "pyttsx3"
            _output_sample_rate = _PYTTSX3_SAMPLE_RATE
            log.info("TTS — pyttsx3 fallback active (system voice)")
        except Exception:
            log.exception("TTS — pyttsx3 also failed to initialise.")

    if _engine_mode == "none":
        raise RuntimeError(
            "TTS — no engine available. "
            "Install kokoro (`pip install kokoro`) or pyttsx3 (`pip install pyttsx3`)."
        )

    await _open_output_stream(_output_sample_rate)


async def _open_output_stream(sample_rate: int) -> None:
    """
    Purpose:
        Open the single persistent PortAudio OutputStream used for the
        entire session.

    Internal Mechanism:
        Idempotent — returns immediately if a stream is already open.
        Otherwise builds and starts an sd.OutputStream in shared mode on
        the TTS-dedicated executor, with a small blocksize and low-latency
        setting to minimize time-to-first-audio. Shared mode is the only
        mode used here; WASAPI exclusive mode was evaluated and rejected
        for this project (see the module docstring) — an earlier design
        that tried exclusive mode first and dropped to shared mode only on
        an outright sd.PortAudioError at open time was insufficient, since
        exclusive mode's failure in practice was silent (the stream opened
        successfully and died later, once the game also needed the
        device) rather than an open-time exception.

    Args:
        sample_rate: The output sample rate to configure the stream at,
                     matching whichever TTS engine initialise() selected.

    Returns:
        None.
    """
    global _output_stream

    if _output_stream is not None:
        return

    def _build_and_start() -> sd.OutputStream:
        stream = sd.OutputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
            blocksize=512,
            latency="low",
            callback=_output_callback,
        )
        stream.start()
        return stream

    _output_stream = await _run_tts(_build_and_start)
    log.info(
        "TTS — persistent output stream open (WASAPI shared mode, sr=%d)",
        sample_rate,
    )


def _output_callback(outdata: np.ndarray, frames: int, time_info: object, status: sd.CallbackFlags) -> None:
    """
    Purpose:
        PortAudio's real-time output callback. Fills the requested audio
        frame buffer for each hardware playback tick.

    Internal Mechanism:
        Runs on PortAudio's own real-time thread — not the asyncio loop or
        the TTS executor. Pulls chunks from _playback_queue, copying
        samples into `outdata` until `frames` samples have been supplied;
        pads the remainder with silence on underrun rather than glitching
        or raising. On queue exhaustion, _current_chunk is explicitly
        reset to None — this is speak_stream()'s drain-wait completion
        signal from the other side of the C-thread boundary, so leaving a
        stale non-None reference here would make that signal unable to go
        False even after playback genuinely finished. The callback
        intentionally never raises sd.CallbackStop: the stream is designed
        to stay open and alive between TTS calls (the entire point of the
        persistent-OutputStream architecture), padding silence and
        returning normally whenever idle. The whole body is wrapped in a
        try/except because PortAudio's C layer silently swallows
        exceptions raised inside a Python callback — without this, a bug
        here would produce an invisible dead stream indistinguishable from
        one still working, rather than a logged, diagnosable error.

    Args:
        outdata: Output buffer to fill, shape (frames, 1) float32 —
                 supplied by PortAudio.
        frames: Number of audio frames PortAudio is requesting for this
                tick.
        time_info: PortAudio timing metadata (unused).
        status: PortAudio callback status flags, non-zero on an underrun/
                overrun or other stream condition.

    Returns:
        None.
    """
    global _current_chunk, _current_pos

    try:
        if status:
            log.debug("TTS — PortAudio output status: %s", status)

        filled = 0
        while filled < frames:
            if _current_chunk is None or _current_pos >= len(_current_chunk):
                try:
                    _current_chunk = _playback_queue.get_nowait()
                    _current_pos = 0
                    log.debug(
                        "TTS_CB — pulled new chunk from queue: %d samples, "
                        "qsize now=%d",
                        len(_current_chunk),
                        _playback_queue.qsize(),
                    )
                except _queue.Empty:
                    log.debug("TTS_CB — queue empty, no chunk to pull")
                    _current_chunk = None
                    _current_pos = 0
                    log.debug("TTS_CB — _current_chunk explicitly set to None")
                    outdata[filled:, 0] = 0.0
                    return

            remaining = len(_current_chunk) - _current_pos
            take = min(remaining, frames - filled)
            outdata[filled:filled + take, 0] = _current_chunk[_current_pos:_current_pos + take]
            _current_pos += take
            filled += take

    except Exception as e:
        log.exception(
            "TTS — exception inside _output_callback (PortAudio would "
            "otherwise swallow this silently — stream may now be dead)"
        )
        print(f"TTS_CB_FATAL — exception inside _output_callback: {e!r}", flush=True)
        outdata[:, 0] = 0.0
        _current_chunk = None
        _current_pos = 0


def stop() -> None:
    """
    Purpose:
        Signal an immediate playback halt for barge-in / interruption.

    Internal Mechanism:
        Thread-safe. Sets _stop_event, then flushes the playback queue and
        clears the in-flight chunk so the persistent output stream goes
        silent on its next callback tick (within one blocksize, ~32 ms)
        without needing to stop or restart the stream itself.

    Args:
        None.

    Returns:
        None.
    """
    global _current_chunk, _current_pos

    if not _stop_event.is_set():
        log.info("TTS — interrupted (barge-in)")
        _stop_event.set()

        while not _playback_queue.empty():
            try:
                _playback_queue.get_nowait()
            except _queue.Empty:
                break
        _current_chunk = None
        _current_pos = 0


def _synthesise_kokoro(text: str) -> Optional[np.ndarray]:
    """
    Purpose:
        Synthesise `text` to float32 PCM audio via the Kokoro neural TTS
        engine.

    Internal Mechanism:
        Runs on _tts_executor (called via _run_tts, not directly on the
        event loop). Iterates Kokoro's pipeline output, collecting each
        non-empty audio chunk it yields, then concatenates them into a
        single float32 array.

    Args:
        text: The sentence (or remainder fragment) to synthesise.

    Returns:
        A float32 numpy array of PCM audio samples, or None if synthesis
        produced no audio or raised an exception.
    """
    try:
        audio_chunks: list[np.ndarray] = []
        for _, _, audio in _kokoro_pipeline(
            text,
            voice=cfg.tts.voice,
            speed=cfg.tts.speed,
        ):
            if audio is not None and len(audio) > 0:
                audio_chunks.append(audio)

        if not audio_chunks:
            log.warning("TTS — Kokoro returned no audio for: %r", text[:80])
            return None

        return np.concatenate(audio_chunks).astype(np.float32)

    except Exception:
        log.exception("TTS — Kokoro synthesis error for text: %r", text[:80])
        return None


def _synthesise_pyttsx3(text: str) -> Optional[np.ndarray]:
    """
    Purpose:
        Synthesise `text` to float32 PCM audio via the pyttsx3 fallback
        engine.

    Internal Mechanism:
        Runs on _tts_executor (called via _run_tts). pyttsx3 has no
        in-memory synthesis API, so this writes to a temporary WAV file in
        cfg.paths.audio_dir via save_to_file()/runAndWait(), reads it back
        with soundfile, and deletes the temp file before returning.

    Args:
        text: The sentence (or remainder fragment) to synthesise.

    Returns:
        A float32 numpy array of PCM audio samples, or None if synthesis
        raised an exception.
    """
    import os
    import tempfile
    import soundfile as sf

    try:
        cfg.paths.audio_dir.mkdir(exist_ok=True)
        with tempfile.NamedTemporaryFile(
            suffix=".wav", dir=cfg.paths.audio_dir, delete=False
        ) as tmp:
            tmp_path = tmp.name

        _pyttsx3_engine.save_to_file(text, tmp_path)
        _pyttsx3_engine.runAndWait()

        audio, _ = sf.read(tmp_path, dtype="float32")
        os.unlink(tmp_path)
        return audio

    except Exception:
        log.exception("TTS — pyttsx3 synthesis error for text: %r", text[:80])
        return None


async def _synthesise(text: str) -> Optional[np.ndarray]:
    """
    Purpose:
        Dispatch synthesis to whichever TTS engine is currently active.

    Internal Mechanism:
        Routes to _synthesise_kokoro or _synthesise_pyttsx3 (via _run_tts,
        on the dedicated executor) based on _engine_mode, set by
        initialise().

    Args:
        text: The sentence (or remainder fragment) to synthesise.

    Returns:
        A float32 numpy array of PCM audio samples, or None if no engine
        is initialised or synthesis failed.
    """
    if _engine_mode == "kokoro":
        return await _run_tts(_synthesise_kokoro, text)
    if _engine_mode == "pyttsx3":
        return await _run_tts(_synthesise_pyttsx3, text)
    log.error("TTS — no engine initialised; call await tts.initialise() first")
    return None


def _play_audio(audio: np.ndarray, sample_rate: int) -> None:
    """
    Purpose:
        Hand synthesised audio off to the persistent output stream for
        playback.

    Internal Mechanism:
        Non-blocking handoff to _playback_queue — actual playback happens
        asynchronously on PortAudio's callback thread via
        _output_callback(). Runs in _tts_executor (called via _run_tts),
        since the queue.put() call can briefly block if the queue is full,
        which is why this stays off the event loop thread. If a barge-in
        has already been signalled, the call is a no-op. A sample_rate
        mismatch against the stream's configured rate indicates the engine
        was changed without reopening the stream — a configuration error
        logged loudly rather than allowed to silently mis-play audio.

    Args:
        audio: float32 PCM samples to enqueue for playback.
        sample_rate: The sample rate `audio` was synthesised at.

    Returns:
        None.
    """
    if _stop_event.is_set():
        return

    if sample_rate != _output_sample_rate:
        log.error(
            "TTS — sample rate mismatch: chunk=%d stream=%d. "
            "Engine/stream were not reopened together; audio will be wrong speed.",
            sample_rate, _output_sample_rate,
        )

    try:
        _playback_queue.put(audio, timeout=5.0)
        log.info("TTS — queued for playback (%.2f s)", len(audio) / sample_rate)
    except _queue.Full:
        log.warning("TTS — playback queue full, dropping sentence to avoid backlog")


async def speak_stream(token_stream: AsyncGenerator[str, None]) -> None:
    """
    Purpose:
        Consume an LLM token stream end-to-end: buffer it into sentences,
        synthesise each one, enqueue it for playback on the persistent
        output stream, and block until playback has genuinely finished.

    Internal Mechanism:
        Sentences are detected via _SENTENCE_END_RE as tokens arrive; each
        complete sentence (or, at stream end, a sufficiently long
        remainder) is synthesised and handed to _play_audio() in turn.
        Synthesis is awaited sequentially per sentence, not concurrently,
        to preserve playback order — the playback queue itself provides
        the pipelining, since sentence N+1 can synthesize while sentence N
        is still draining from the queue into the callback. If
        _stop_event is set at any point (barge-in), the remainder of
        token_stream is drained without further synthesis and the
        function returns immediately.

        Once every sentence has been enqueued, this function does not
        return immediately — audio has only been synthesised and queued
        at that point, not necessarily played, since _play_audio() is a
        queue.put() and real playback happens asynchronously on
        _output_callback's own thread. A poll loop waits for
        _playback_queue to empty and _current_chunk to become None (both
        mutated on PortAudio's thread, not this one), checking on every
        tick whether _output_stream is still active and bailing
        immediately if it is not, and enforcing _MAX_DRAIN_WAIT_S as an
        absolute ceiling either way. See the module docstring's
        "Playback-Completion Drain-Wait" section for why this wait exists.

    Args:
        token_stream: AsyncGenerator[str, None] from llm.stream_response().

    Returns:
        None.
    """
    if _engine_mode == "none":
        log.error("TTS — not initialised. Call await tts.initialise() first.")
        return

    _stop_event.clear()

    buffer: str = ""
    total_sentences: int = 0

    async for token in token_stream:
        if _stop_event.is_set():
            log.info("TTS — token stream discarded after interruption")
            async for _ in token_stream:
                pass
            return

        buffer += token

        match = _SENTENCE_END_RE.search(buffer)
        if match and len(buffer[:match.end()].strip()) >= _MIN_SENTENCE_CHARS:
            sentence = buffer[:match.end()].strip()
            buffer = buffer[match.end():]

            total_sentences += 1
            log.info("TTS — synthesising sentence %d: %r", total_sentences, sentence)

            audio = await _synthesise(sentence)
            if audio is None:
                log.warning("TTS — synthesis returned no audio, skipping sentence")
                continue

            if _stop_event.is_set():
                return

            await _run_tts(_play_audio, audio, _output_sample_rate)

    remainder = buffer.strip()
    if remainder and not _stop_event.is_set() and len(remainder) >= _MIN_SENTENCE_CHARS:
        total_sentences += 1
        log.info(
            "TTS — synthesising remainder (sentence %d): %r",
            total_sentences, remainder,
        )
        audio = await _synthesise(remainder)
        if audio is not None:
            await _run_tts(_play_audio, audio, _output_sample_rate)

    if not _stop_event.is_set():
        elapsed = 0.0
        _last_tracker_log = 0.0
        while (not _playback_queue.empty() or _current_chunk is not None):
            if _stop_event.is_set():
                break

            if elapsed - _last_tracker_log >= 1.0:
                _chunk_snapshot = _current_chunk
                log.info(
                    "TTS_DRAIN_TRACK — elapsed=%.1fs qsize=%d "
                    "current_chunk_is_none=%s stream_active=%s",
                    elapsed,
                    _playback_queue.qsize(),
                    _chunk_snapshot is None,
                    _output_stream.active if _output_stream is not None else "N/A",
                )
                _last_tracker_log = elapsed

            if _output_stream is not None and not _output_stream.active:
                log.critical(
                    "TTS — output stream is not active (dead callback / "
                    "device lost) — aborting drain-wait immediately instead "
                    "of hanging until the %.0fs safety ceiling",
                    _MAX_DRAIN_WAIT_S,
                )
                break

            if elapsed >= _MAX_DRAIN_WAIT_S:
                log.warning(
                    "TTS — drain wait exceeded %.0fs; returning anyway rather "
                    "than hanging the pipeline (possible stream/callback issue)",
                    _MAX_DRAIN_WAIT_S,
                )
                break
            await asyncio.sleep(_DRAIN_POLL_INTERVAL_S)
            elapsed += _DRAIN_POLL_INTERVAL_S

        log.info("TTS — response complete (%d sentence(s))", total_sentences)