"""
stt.py — Whisper.cpp Subprocess Manager
========================================
Owns the entire lifecycle of a single transcription request:

    1.  Receives a Path to a temporary WAV file written by the VAD consumer.
    2.  Spawns whisper.cpp as a child process via asyncio.create_subprocess_exec.
        The event loop drives all OS pipe I/O via epoll/kqueue — zero thread-pool
        slots consumed during GPU inference on the RX 6700 XT.
    3.  Enforces a hard wall-clock timeout (cfg.whisper.timeout_seconds,
        default 30s) via asyncio.wait_for.  If whisper.cpp hangs because
        the game has exhausted VRAM, the process is killed, VRAM is
        released, and an empty string is returned so the pipeline recovers.
    4.  On success, reads the transcript from the sidecar {audio_path}.txt file
        (see "File-based output" below) rather than parsing stdout.
    5.  Deletes BOTH the temporary WAV and the .txt transcript file in a
        try-finally block that runs even on timeout, CancelledError, or any
        other exception.

Why asyncio.create_subprocess_exec and NOT run_in_executor(subprocess.run)
---------------------------------------------------------------------------
    asyncio.create_subprocess_exec:
        • Event loop drives pipe I/O (epoll on Linux, kqueue on macOS, IOCP on
          Windows).  The await returns immediately once the OS delivers data.
        • Zero threads are held during the 300–1500 ms Vulkan inference window.
        • Fully cancellable at every await point — CancelledError propagates
          cleanly and proc.kill() + proc.wait() release VRAM immediately.

    run_in_executor(subprocess.run):
        • Holds a ThreadPoolExecutor worker for the full GPU inference duration.
        • With max_workers=2 (TTS + VAD), one whisper call blocks ALL synthesis.
        • Not cancellable — the blocked thread cannot receive CancelledError.

File-based output (not stdout)
-------------------------------
    -otxt        write the transcript to {audio_path}.txt instead of relying
                 on stdout. Verified against whisper.cpp's actual CLI/source:
                 with no -of/--output-file override, the default output path
                 is exactly the input filename with ".txt" appended — e.g.
                 "audio123.wav" -> "audio123.wav.txt" (append, not replace).
                 stdout can still carry Vulkan init banners or get tangled
                 with stderr depending on the build; reading a real file
                 sidesteps all of that instead of parsing around it.
    --no-prints  (-np) suppresses everything except the transcript itself —
                 kept for clean logs even though the .txt file is now the
                 authoritative source, not stdout.

Vulkan / GPU flags
------------------
    (none required) GPU offload is on by default (whisper_params.use_gpu =
                 true) — modern whisper.cpp auto-detects and uses Vulkan
                 natively, same as the --vulkan flag becoming automatic.
                 -ngl is a llama.cpp convention, NOT a whisper.cpp flag —
                 it does not appear anywhere in whisper.cpp's argument
                 parser (verified against examples/cli/cli.cpp) and was
                 being silently rejected. Explicit opt-out is -ng/--no-gpu;
                 device selection is -dev/--device (see cfg.hardware.
                 vulkan_device_id, currently unused — wire it up via -dev
                 if you ever need a non-default GPU index).
    -t   <n>      CPU threads used for non-GPU work (tokenizer, sampling).
                  Kept low (cfg.hardware.whisper_threads, default 2) to avoid
                  competing with the game on the i5-12400F's P-cores.
"""

from __future__ import annotations

import asyncio
import re
import shlex
from pathlib import Path
from typing import Final, Optional

from config import cfg
from src.logger import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants derived from config
# ---------------------------------------------------------------------------

# Regex that strips whisper.cpp's timestamp prefix "[HH:MM:SS.mmm --> ...]"
# when -nt is absent (we pass -nt, but belt-and-suspenders cleanup is free).
_TIMESTAMP_RE: Final[re.Pattern[str]] = re.compile(
    r"\[\d{2}:\d{2}:\d{2}\.\d{3}\s*-->\s*\d{2}:\d{2}:\d{2}\.\d{3}\]\s*"
)

# whisper.cpp emits a bracketed/parenthesised placeholder instead of an
# empty string when a segment is processed but judged to contain no
# speech — the exact wording varies by model/version ("[BLANK AUDIO]",
# "[SILENCE]", "(silence)", "[NO SPEECH]", etc.), so this matches the
# general shape (entirely-bracketed or entirely-parenthesised, letters/
# spaces only) rather than a single hardcoded literal. Applied to the
# WHOLE cleaned transcript, not as a substring strip mid-sentence, so a
# real utterance that happens to end with e.g. "(laughs)" is untouched —
# only a transcript that is ENTIRELY one of these placeholders collapses
# to "".
_NON_SPEECH_PLACEHOLDER_RE: Final[re.Pattern[str]] = re.compile(
    r"^[\[\(][A-Za-z_ ]+[\]\)]$"
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_whisper_cmd(audio_path: Path) -> list[str]:
    """
    Construct the whisper.cpp command-line argument list.

    Kept as a pure function (no I/O) so it is trivially testable and so
    logging of the exact command precedes the subprocess launch.

    Args:
        audio_path: Absolute path to the temporary 16-bit 16 kHz mono WAV.

    Returns:
        List of strings ready for asyncio.create_subprocess_exec(*cmd).
    """
    cmd: list[str] = [
        str(cfg.whisper.bin_path),           # e.g. ./bin/whisper-vulkan
        "--model",    str(cfg.whisper.model_path),  # .bin weights file
        "--file",     str(audio_path),
        "--language", cfg.whisper.language,  # e.g. "en"
        "--threads",  str(cfg.hardware.whisper_threads),  # CPU thread cap
        "--no-timestamps",                   # clean output — no [hh:mm] lines
        "--no-prints",                       # suppress banner/progress noise on stdout
        "-otxt",                             # write transcript to {audio_path}.txt
    ]
    log.debug("STT_CMD | %s", shlex.join(cmd))
    return cmd


def _parse_transcript(raw: str) -> str:
    """
    Sanitise whisper.cpp's transcript text into a plain string.

    Strips timestamp lines (belt-and-suspenders in case --no-timestamps is
    ignored by an older whisper.cpp build), collapses repeated whitespace,
    and strips leading/trailing whitespace. Used on the contents of the
    -otxt sidecar file (see transcribe()), not stdout.

    FIX [Blank-audio placeholder leak]: whisper.cpp does not always return
    an empty string for a no-speech segment (e.g. a wake-word trigger
    followed by silence within the pre-command grace period). It can
    instead emit a literal placeholder token such as "[BLANK AUDIO]" —
    non-empty, so it previously survived this function, then survived
    main.py's `if not transcript:` guard (a truthy string), and was fed
    straight into conversation history and on to the LLM/TTS GPU stages.
    Collapsing it to "" HERE, at the single choke point every transcript
    passes through, means that existing downstream guard now catches it
    for free — no second guard clause needed in the orchestrator, and no
    placeholder text ever reaches history or a GPU-bound stage.

    Args:
        raw: Raw text content, already decoded to str.

    Returns:
        Clean transcript, or empty string if nothing remains (including
        when the only content was a non-speech placeholder token).
    """
    cleaned: str = _TIMESTAMP_RE.sub("", raw)
    cleaned = " ".join(cleaned.split())

    if _NON_SPEECH_PLACEHOLDER_RE.match(cleaned):
        log.debug(
            "STT_PARSE — non-speech placeholder token %r collapsed to "
            "empty transcript",
            cleaned,
        )
        return ""

    return cleaned


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def transcribe(audio_path: Path) -> str:
    """
    Transcribe a WAV file via whisper.cpp on the AMD RX 6700 XT (Vulkan).

    This is the ONLY public symbol exported by stt.py.  It is called by the
    STT consumer task in main.py after dequeueing an audio path.

    Lifecycle contract
    ------------------
    The caller (main.py's _stt_consumer) is responsible for creating the WAV
    file.  THIS function is responsible for DELETING both it AND the -otxt
    sidecar .txt file — always, in the finally block — even on timeout or
    CancelledError.  This prevents temp-file leakage regardless of the error
    path taken.

    GPU deadlock prevention
    -----------------------
    asyncio.wait_for wraps proc.communicate() with a wall-clock timeout
    (cfg.whisper.timeout_seconds, default 30s).  On expiry:
        1.  TimeoutError is raised at the await point.
        2.  proc.kill() sends SIGKILL (Linux) or TerminateProcess (Windows).
        3.  proc.wait() reaps the zombie and releases all GPU/VRAM handles.
        4.  An empty string is returned so the pipeline loop recovers cleanly.

    CancelledError safety
    ---------------------
    If pipeline_loop() cancels the STT task (barge-in), CancelledError arrives
    at the await inside wait_for.  The finally block still runs:
    proc.kill() + proc.wait() drain the subprocess, and the WAV is unlinked.
    CancelledError is then re-raised so asyncio.gather() sees clean cancellation.

    Args:
        audio_path: Path to a 16-bit 16 kHz mono WAV file.  WILL BE DELETED.

    Returns:
        Transcript string, or "" on error / timeout / empty output.

    Raises:
        asyncio.CancelledError: Re-raised after cleanup if the task is cancelled.
    """
    log.info(
        "STT_STARTED | file=%s size_bytes=%d",
        audio_path.name,
        audio_path.stat().st_size if audio_path.exists() else -1,
    )

    proc: Optional[asyncio.subprocess.Process] = None

    # FIX [Stdout trap]: computed once, up front, so it's available in the
    # finally block for cleanup regardless of which exit path is taken
    # (timeout, non-zero returncode, exception, or success).
    txt_path: Path = Path(f"{audio_path}.txt")

    try:
        cmd: list[str] = _build_whisper_cmd(audio_path)

        # asyncio.create_subprocess_exec:
        #   • Registers stdout/stderr pipe fds with the event loop's selector.
        #   • Returns immediately; no thread is held.
        #   • The await below yields control back to the event loop, which can
        #     continue running TTS / VAD / the barge-in watchdog concurrently.
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        log.debug("STT_PROC_STARTED | pid=%d", proc.pid)

        # ── GPU-deadlock timeout ──────────────────────────────────────────
        try:
            stdout_bytes: bytes
            stderr_bytes: bytes
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(),
                timeout=cfg.whisper.timeout_seconds,
            )
        except asyncio.TimeoutError:
            log.error(
                "STT_TIMEOUT | pid=%d exceeded %.0fs — "
                "likely VRAM exhaustion from game; killing process",
                proc.pid,
                cfg.whisper.timeout_seconds,
            )
            proc.kill()
            await proc.wait()  # Reap zombie; releases VRAM and file handles.
            log.debug("STT_TIMEOUT_CLEANUP | pid=%d reaped", proc.pid)
            return ""

        # FIX [Silent stderr]: decoded unconditionally, not just on non-zero
        # returncode. whisper.cpp's argument parser calls exit(0) — a
        # "successful" code — even on a fatal "unknown argument" error (see
        # cli.cpp's whisper_params_parse), so returncode==0 is NOT sufficient
        # proof that anything actually ran. stderr is the only place that
        # error message exists; discarding it on the "success" path is how
        # a real failure went silent here before.
        stderr_text: str = stderr_bytes.decode(errors="replace").strip()

        # ── Return-code check ────────────────────────────────────────────
        if proc.returncode != 0:
            log.error(
                "STT_PROC_ERROR | pid=%d returncode=%d stderr=%r",
                proc.pid,
                proc.returncode,
                stderr_text[:400],  # Truncate to avoid flooding logs.
            )
            return ""

        # ── Parse and return ─────────────────────────────────────────────
        # FIX [Stdout trap]: -otxt writes the transcript to a real file
        # (verified default naming: {audio_path}.txt — input path with
        # ".txt" appended, since no -of/--output-file override is passed).
        # Reading it sidesteps stdout entirely — no risk of Vulkan init
        # banners, mixed stdout/stderr interleaving, or pipe-buffering
        # quirks contaminating what we treat as the transcript. Off-thread
        # for the same non-blocking reason as _write_temp_wav's own I/O.
        try:
            raw_text: str = await asyncio.to_thread(txt_path.read_text, encoding="utf-8")
        except FileNotFoundError:
            log.error(
                "STT_TXT_NOT_FOUND | pid=%d returncode=0 expected=%s — "
                "whisper.cpp exited 0 but did not write the -otxt sidecar "
                "file. This usually means it never reached transcription "
                "at all (e.g. an argument it silently rejected via exit(0) "
                "rather than a non-zero code). stderr=%r",
                proc.pid,
                txt_path.name,
                stderr_text[:400],
            )
            return ""

        transcript: str = _parse_transcript(raw_text)

        if transcript:
            log.info(
                "STT_COMPLETED | pid=%d chars=%d transcript=%r",
                proc.pid,
                len(transcript),
                transcript[:120],
            )
        else:
            log.info(
                "STT_COMPLETED | pid=%d — empty transcript "
                "(silence, noise, or model confidence too low)",
                proc.pid,
            )

        return transcript

    except asyncio.CancelledError:
        # Barge-in or shutdown cancelled this task mid-await.
        if proc is not None:
            log.info(
                "STT_CANCELLED | pid=%d — killing subprocess and releasing GPU",
                proc.pid,
            )
            proc.kill()
            await proc.wait()
        raise  # Re-raise — swallowing CancelledError breaks asyncio task machinery.

    except FileNotFoundError:
        log.error(
            "STT_BINARY_NOT_FOUND | path=%s — "
            "verify cfg.whisper.bin_path and that the Vulkan build is installed",
            cfg.whisper.bin_path,
        )
        return ""

    except Exception:
        log.exception("STT_UNEXPECTED_ERROR | audio_path=%s", audio_path)
        if proc is not None:
            try:
                proc.kill()
                await proc.wait()
            except Exception:
                pass  # Best-effort cleanup; original exception takes precedence.
        return ""

    finally:
        # ── Temp-file cleanup (ALWAYS runs) ──────────────────────────────
        # Runs on: success, timeout, CancelledError, FileNotFoundError, Exception.
        # The caller (main.py) must NOT attempt to delete either file after
        # calling transcribe(); this function owns cleanup of BOTH the
        # input .wav and the -otxt sidecar .txt file. Each delete is
        # independent — a failure removing one must not skip the other.
        try:
            audio_path.unlink(missing_ok=True)
            log.debug("STT_FILE_CLEANUP | removed %s", audio_path.name)
        except OSError:
            log.warning(
                "STT_FILE_CLEANUP_FAILED | could not remove %s — "
                "manual cleanup may be required",
                audio_path.name,
            )

        try:
            txt_path.unlink(missing_ok=True)
            log.debug("STT_FILE_CLEANUP | removed %s", txt_path.name)
        except OSError:
            log.warning(
                "STT_FILE_CLEANUP_FAILED | could not remove %s — "
                "manual cleanup may be required",
                txt_path.name,
            )