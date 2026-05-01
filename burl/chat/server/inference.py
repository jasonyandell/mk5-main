"""Streaming bridge from in-process MLX-LM to async SSE.

Wraps ``burl.modal.gemma_local.GemmaLocalNative`` (which already exposes an
``on_chunk`` callback) with an asyncio.Queue so FastAPI can yield tokens as
they arrive without blocking the event loop.

MLX's default GPU stream has thread affinity: the thread that created the
stream is the only thread allowed to enqueue ops on it. We therefore pin the
model load AND every generate call to one dedicated worker thread via a
single-slot ThreadPoolExecutor.
"""
from __future__ import annotations

import asyncio
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import AsyncIterator

log = logging.getLogger(__name__)

DEFAULT_MODEL_REPO = os.environ.get(
    "BURL_CHAT_MODEL_REPO", "mlx-community/gemma-4-e2b-it-bf16"
)
DEFAULT_ADAPTER_PATH = os.environ.get("BURL_CHAT_ADAPTER_PATH") or None


class InferenceEngine:
    """Singleton wrapper around one loaded model + adapter, pinned to a
    single worker thread so MLX's default GPU stream stays valid.
    """

    def __init__(
        self,
        model_repo: str = DEFAULT_MODEL_REPO,
        adapter_path: str | None = DEFAULT_ADAPTER_PATH,
    ) -> None:
        self.model_repo = model_repo
        self.adapter_path = adapter_path
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="mlx",
        )
        self._native = self._executor.submit(self._load).result()

    def _load(self):
        # Imported lazily so the import itself runs on the MLX worker thread.
        from burl.modal.gemma_local import GemmaLocalNative

        return GemmaLocalNative(
            model_repo=self.model_repo, adapter_path=self.adapter_path,
        )

    @property
    def info(self) -> dict:
        return {
            "model_repo": self.model_repo,
            "adapter_path": self.adapter_path,
        }

    async def stream(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.6,
        enable_thinking: bool = False,
    ) -> AsyncIterator[dict]:
        """Yield ``{"type": "token"|"done", ...}`` events.

        The synchronous ``stream_generate`` runs in a thread; ``on_chunk``
        pushes each token text into an asyncio.Queue this coroutine drains.
        """
        queue: asyncio.Queue[dict | None] = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def on_chunk(text: str) -> None:
            if text:
                loop.call_soon_threadsafe(
                    queue.put_nowait, {"type": "token", "text": text},
                )

        def producer() -> dict:
            try:
                result = self._native.generate_native(
                    messages,
                    tools=tools,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    enable_thinking=enable_thinking,
                    on_chunk=on_chunk,
                )
                loop.call_soon_threadsafe(
                    queue.put_nowait,
                    {"type": "done", "n_tokens": result["n_tokens"]},
                )
                return result
            except Exception as exc:  # noqa: BLE001 — propagate to client
                log.exception("[burl-chat] generate failed")
                loop.call_soon_threadsafe(
                    queue.put_nowait,
                    {"type": "error", "message": str(exc)},
                )
                return {}
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        self._executor.submit(producer)

        while (event := await queue.get()) is not None:
            yield event
