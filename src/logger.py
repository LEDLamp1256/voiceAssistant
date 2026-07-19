"""
Architecture notes
------------------
* QueueHandler + QueueListener  — the critical non-blocking design.
  Log calls from asyncio coroutines and threads push a LogRecord onto
  an in-process queue and return immediately (~1 µs). A dedicated
  background thread drains the queue and performs the actual I/O
  (file writes, console output). This means a slow disk write or a
  RotatingFileHandler rollover can NEVER stall the audio pipeline.

* RotatingFileHandler — caps disk usage at MAX_BYTES × BACKUP_COUNT.
  At defaults that's 5 MB × 3 = 15 MB maximum, safe for long gaming
  sessions.

* Child loggers — every module calls get_logger(__name__). Because
  they are all children of the "assistant" root logger, they inherit
  its level and handlers automatically; no per-module setup needed.

Usage in any module
-------------------
    from src.logger import get_logger
    log = get_logger(__name__)
    log.info("Module ready")
"""

from __future__ import annotations

import logging
import logging.handlers
import queue
import sys
from pathlib import Path
from typing import Optional

from config import cfg

# ---------------------------------------------------------------------------
# Constants — edit here rather than hunting through code
# ---------------------------------------------------------------------------
_ROOT_LOGGER_NAME: str = "assistant"
_LOG_DIR: Path = cfg.paths.logs_dir
_LOG_FILE: Path = _LOG_DIR / "assistant.log"

_MAX_BYTES: int = 5 * 1024 * 1024   # 5 MB per file
_BACKUP_COUNT: int = 3              # assistant.log, .log.1, .log.2, .log.3

# Console shows INFO and above to keep the terminal readable during gameplay.
# File captures DEBUG and above for post-session diagnostics.
_CONSOLE_LEVEL: int = logging.INFO
_FILE_LEVEL: int = logging.DEBUG

_FMT: str = "[{asctime}] {levelname:<8} {name} — {message}"
_DATE_FMT: str = "%Y-%m-%d %H:%M:%S"

# Shared queue; size=0 means unbounded — safe here because log volume
# is low. For very high-throughput scenarios cap at e.g. maxsize=10_000.
_log_queue: queue.Queue[logging.LogRecord] = queue.Queue(maxsize=0)

# Module-level reference so callers can stop the listener on shutdown.
_listener: Optional[logging.handlers.QueueListener] = None

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _build_formatter() -> logging.Formatter:
    return logging.Formatter(fmt=_FMT, datefmt=_DATE_FMT, style="{")


def _build_console_handler() -> logging.StreamHandler:
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(_CONSOLE_LEVEL)
    handler.setFormatter(_build_formatter())
    return handler


def _build_file_handler() -> logging.handlers.RotatingFileHandler:
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    handler = logging.handlers.RotatingFileHandler(
        filename=_LOG_FILE,
        maxBytes=_MAX_BYTES,
        backupCount=_BACKUP_COUNT,
        encoding="utf-8",
    )
    handler.setLevel(_FILE_LEVEL)
    handler.setFormatter(_build_formatter())
    return handler


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def init_logging(console_level: int = _CONSOLE_LEVEL) -> None:
    """
    Initialise the logging system. Call ONCE from main.py at startup.

    Sets up the QueueListener (background I/O thread) and attaches a
    QueueHandler to the root "assistant" logger so all child loggers
    route through the non-blocking queue automatically.

    Args:
        console_level: Override the console verbosity at runtime.
                       Pass logging.DEBUG for verbose output during
                       development, logging.WARNING for quiet gameplay.
    """
    global _listener

    if _listener is not None:
        # Guard: calling init_logging() twice would duplicate handlers.
        logging.getLogger(_ROOT_LOGGER_NAME).warning(
            "init_logging() called more than once — ignoring."
        )
        return

    console_handler = _build_console_handler()
    console_handler.setLevel(console_level)   # Respect runtime override
    file_handler = _build_file_handler()

    # QueueListener runs on a dedicated daemon thread — it drains the
    # queue and forwards records to the real handlers (file + console).
    # respect_handler_level=True means each handler's own level is
    # honoured even though records are queued at DEBUG level.
    _listener = logging.handlers.QueueListener(
        _log_queue,
        console_handler,
        file_handler,
        respect_handler_level=True,
    )
    _listener.start()

    # Attach the non-blocking QueueHandler to the hierarchy root.
    root = logging.getLogger(_ROOT_LOGGER_NAME)
    root.setLevel(logging.DEBUG)          # Let handlers decide what to show
    root.addHandler(logging.handlers.QueueHandler(_log_queue))
    root.propagate = False                # Never bubble up to the stdlib root


def shutdown_logging() -> None:
    """
    Flush the queue and stop the background listener thread.
    Call from main.py's finally block to ensure all records are
    written before the process exits.
    """
    global _listener
    if _listener is not None:
        _listener.stop()
        _listener = None


def get_logger(name: str) -> logging.Logger:
    """
    Return a child logger under the "assistant" namespace.

    Args:
        name: Typically ``__name__`` of the calling module, e.g.
              "src.stt" or "src.vad". This appears in every log line
              so you can filter by module in the log file.

    Returns:
        logging.Logger: Configured child logger.

    Example::

        from src.logger import get_logger
        log = get_logger(__name__)
        log.info("STT module loaded")
    """
    # Prefix with root name so the hierarchy is always assistant.src.stt etc.
    if not name.startswith(_ROOT_LOGGER_NAME):
        name = f"{_ROOT_LOGGER_NAME}.{name}"
    return logging.getLogger(name)