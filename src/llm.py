"""
src/llm.py — Async Streaming LLM Connector (Ollama)
=====================================================
Public interface:
    async def stream_response(
        messages: list[dict[str, str]]
    ) -> AsyncGenerator[str, None]

Architecture
------------
Ollama exposes a local HTTP API. We target /api/chat (the multi-turn
messages endpoint) rather than /api/generate (single-prompt) so the
caller can pass a full conversation history for context, and because
the messages format matches OpenAI's schema — making future provider
swaps trivial.

Streaming works as newline-delimited JSON (NDJSON): Ollama writes one
JSON object per token to the response body and flushes immediately.
httpx.AsyncClient.stream() reads these chunks as they arrive via
async iteration over response.aiter_lines(), so tokens reach the TTS
pipeline as fast as the GPU produces them — no waiting for the full
response before speaking begins.

Why NOT asyncio.to_thread here
-------------------------------
httpx.AsyncClient is natively async — its network I/O is driven
directly by the event loop's selector, consuming zero thread-pool
slots while waiting for GPU tokens. This is the correct choice for
long-lived streaming requests where each token may arrive 50–200 ms
apart. Using to_thread would unnecessarily hold a thread for the full
inference duration.

Timeout strategy
----------------
httpx.Timeout separates two failure modes:
  connect=5s   — Ollama process not running / port not open.
                 Fails fast so the user gets a clear error immediately.
  read=<cfg>   — Time between token chunks during active streaming.
                 Set to cfg.ollama.request_timeout (default 60 s).
                 Generous for slow models / large context; a mid-stream
                 silence longer than this almost always means the Ollama
                 process crashed or was killed by the OS OOM killer.

Ollama /api/chat NDJSON chunk format
--------------------------------------
Each line is a JSON object. Terminal chunk has "done": true.

  {"model":"mistral","message":{"role":"assistant","content":"Hello"},"done":false}
  {"model":"mistral","message":{"role":"assistant","content":"!"},"done":false}
  {"model":"mistral","message":{"role":"assistant","content":""},"done":true,
   "done_reason":"stop","total_duration":834000000,...}

We extract chunk["message"]["content"] and yield it. The final chunk's
content is always "" so yielding it is harmless, but we break on
"done": true to avoid processing the metadata-heavy tail object.
"""

from __future__ import annotations

import json
from typing import AsyncGenerator

import httpx

from config import cfg
from src.logger import get_logger

log = get_logger(__name__)   # "assistant.src.llm"

# ---------------------------------------------------------------------------
# Endpoint & timeout constants derived from cfg
# ---------------------------------------------------------------------------
_CHAT_ENDPOINT: str = "/api/chat"

# Separate connect vs. read timeout so a crashed Ollama fails fast (5 s)
# but an active stream is allowed cfg.ollama.request_timeout before we
# assume it has stalled.
def _build_timeout() -> httpx.Timeout:
    return httpx.Timeout(
        connect=5.0,
        read=float(cfg.ollama.request_timeout),
        write=5.0,
        pool=5.0,
    )


# ---------------------------------------------------------------------------
# Payload builder
# ---------------------------------------------------------------------------
def _build_payload(messages: list[dict[str, str]]) -> dict:
    """
    Construct the /api/chat request body.

    The system prompt from cfg is prepended as the first message only if
    the caller hasn't already included a system role, preventing duplicate
    injection when the caller manages its own conversation history.

    Args:
        messages: Conversation history in role/content dicts, e.g.
                  [{"role": "user", "content": "What's my ping?"}]

    Returns:
        JSON-serialisable dict ready to POST to Ollama.
    """
    has_system = any(m.get("role") == "system" for m in messages)

    full_messages: list[dict[str, str]] = []
    if not has_system and cfg.ollama.system_prompt:
        full_messages.append({"role": "system", "content": cfg.ollama.system_prompt})
    full_messages.extend(messages)

    return {
        "model": cfg.ollama.model,
        "messages": full_messages,
        "stream": True,                 # NDJSON streaming; must be True
        "options": {
            "num_gpu": cfg.hardware.ollama_num_gpu,  # layers on RX 6700 XT
        },
    }


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------
async def stream_response(
    messages: list[dict[str, str]],
) -> AsyncGenerator[str, None]:
    """
    Stream token deltas from Ollama's /api/chat endpoint.

    Yields individual content strings as they arrive from the model.
    Callers should accumulate these for logging/TTS, but can also pipe
    them directly to a streaming TTS engine for minimum latency.

    Args:
        messages: Conversation turns as list of {"role": ..., "content": ...}
                  dicts. "user" and "assistant" roles are valid; a "system"
                  role is prepended automatically from cfg if not present.

    Yields:
        str: Token delta text. May be a single character, a word, or a
             short phrase depending on the model's tokeniser. Never empty
             (empty deltas from the terminal chunk are dropped).

    Example::

        full_reply = ""
        async for token in llm.stream_response([{"role": "user", "content": text}]):
            full_reply += token
            # optionally: pipe token to streaming TTS here

    Raises:
        Nothing — all exceptions are caught and logged. The generator
        terminates cleanly on error so the caller's async-for loop exits
        without raising, and the pipeline can continue to the next utterance.
    """
    url = f"{cfg.ollama.base_url}{_CHAT_ENDPOINT}"
    payload = _build_payload(messages)

    log.debug(
        "LLM — POST %s | model=%s num_gpu=%d timeout=%.0fs",
        url,
        cfg.ollama.model,
        cfg.hardware.ollama_num_gpu,
        cfg.ollama.request_timeout,
    )

    try:
        async with httpx.AsyncClient(timeout=_build_timeout()) as client:
            async with client.stream("POST", url, json=payload) as response:

                # Non-2xx before any tokens arrived — surface clearly.
                if response.status_code != 200:
                    body = await response.aread()
                    log.error(
                        "LLM — Ollama returned HTTP %d. Body: %s",
                        response.status_code,
                        body.decode(errors="replace")[:300],
                    )
                    return

                token_count = 0

                async for raw_line in response.aiter_lines():
                    raw_line = raw_line.strip()
                    if not raw_line:
                        continue   # blank keep-alive line; skip silently

                    # ── Parse NDJSON chunk ───────────────────────────────
                    try:
                        chunk = json.loads(raw_line)
                    except json.JSONDecodeError:
                        log.warning(
                            "LLM — non-JSON line from Ollama (ignored): %r",
                            raw_line[:200],
                        )
                        continue

                    # ── Extract delta content ────────────────────────────
                    # The "message" key is absent on the terminal done=true chunk
                    # that carries only metadata (total_duration, eval_count…).
                    delta: str = chunk.get("message", {}).get("content", "")
                    if delta:
                        token_count += 1
                        log.debug("LLM — token #%d: %r", token_count, delta)
                        yield delta

                    # ── Stream termination ───────────────────────────────
                    if chunk.get("done", False):
                        log.info(
                            "LLM — stream complete: %d token(s) | "
                            "model=%s eval_duration=%.2fs",
                            token_count,
                            chunk.get("model", cfg.ollama.model),
                            chunk.get("eval_duration", 0) / 1e9,
                        )
                        break

    # ── Connection-level failures ────────────────────────────────────────
    except httpx.ConnectError:
        log.error(
            "LLM — cannot connect to Ollama at %s. "
            "Is Ollama running? Start it with: ollama serve",
            cfg.ollama.base_url,
        )

    except httpx.ConnectTimeout:
        log.error(
            "LLM — connection to Ollama timed out after 5s (%s). "
            "The service may be starting up — retry in a moment.",
            cfg.ollama.base_url,
        )

    except httpx.ReadTimeout:
        log.error(
            "LLM — read timeout after %.0fs waiting for a token. "
            "Ollama may have been killed by the OS OOM killer — "
            "check available VRAM with: rocm-smi or radeontop. "
            "Consider switching to a smaller model (e.g. phi3:mini).",
            cfg.ollama.request_timeout,
        )

    except httpx.RemoteProtocolError as exc:
        log.error(
            "LLM — Ollama closed the connection unexpectedly: %s. "
            "This can occur if the model is swapped mid-request.",
            exc,
        )

    except Exception:
        log.exception("LLM — unexpected error during stream_response()")