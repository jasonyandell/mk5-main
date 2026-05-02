"""MLX-LM engine for burl.lab. Async-iterator surface.

The engine yields a strict event sequence per `step()` call:

    EngineStart -> EngineToken* -> EngineToolCall* -> [EngineError] -> EngineDone

It does NOT dispatch tool calls — when the model emits a complete
`<|tool_call>...<tool_call|>` envelope, the engine yields one
`EngineToolCall` move, then `EngineDone(reason="tool_dispatch")`, and stops
the underlying generation. The runtime owns dispatch.

The engine NEVER yields `EngineCommit`. Commits are emitted by the harness
(drive loop or phase post-processor) after `step()` returns, when the phase
decides the trailing region is a commit. See SPEC.md.

If MLX raises mid-decode, the engine yields `EngineError(message, traceback,
during=None)` followed by `EngineDone(reason="aborted")`. Lossless — full
traceback is journaled.

MLX's default GPU stream is bound to the thread that touched it. We pin
both the model load and every generate call to a single-slot
`ThreadPoolExecutor` so the stream stays valid across calls.

Gemma 4's bundled `chat_template.jinja` silently drops `role="tool"`
messages. For Gemma we pack tool responses onto the previous assistant
turn as `tool_responses=[{name, response}]`. Other tokenizers (Qwen,
OpenAI-style) keep `role="tool"`.

Stamp clock: anchored at the moment `EngineStart` is yielded. `t_wall_ms`
and `t_mono_ns` are zero on EngineStart and grow monotonically through
the step. The runtime is free to re-anchor stamps to the session clock
before journaling — these stamps are only required to be internally
consistent within one step.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, AsyncIterator, Protocol

from .tool import ToolSpec
from .transcript import (
    EngineDone,
    EngineError,
    EngineStart,
    EngineToken,
    EngineToolCall,
    Stamp,
)

log = logging.getLogger(__name__)

DEFAULT_MODEL_REPO = os.environ.get(
    "BURL_LAB_MODEL_REPO", "mlx-community/gemma-4-e2b-it-bf16"
)
DEFAULT_ADAPTER_PATH = os.environ.get("BURL_LAB_ADAPTER_PATH") or None

# Streaming cadence: emit an EngineToken at least this often.
_FLUSH_MS = 50.0
_FLUSH_TOKENS = 16

# Tool-call envelope markers (Gemma 4 native shape).
_OPEN_TOOL = "<|tool_call>"
_CLOSE_TOOL = "<tool_call|>"
_ENVELOPE_RE = re.compile(r"<\|tool_call>(.*?)<tool_call\|>", re.DOTALL)
_CALL_SIG_RE = re.compile(r"call\s*:\s*(\w+)\s*[({](.*?)[)}]", re.DOTALL)
_STRDELIM_RE = re.compile(r"<\|\"\|>")


# --------------------------------------------------------------------------- #
# Engine protocol                                                              #
# --------------------------------------------------------------------------- #


class Engine(Protocol):
    async def step(
        self,
        messages: list[dict],
        tools: list[ToolSpec],
        max_tokens: int = 2048,
    ) -> AsyncIterator[Any]: ...


# --------------------------------------------------------------------------- #
# Helpers — message rewriting, tool-spec rendering, args parsing               #
# --------------------------------------------------------------------------- #


def _is_gemma_tokenizer(tokenizer: Any) -> bool:
    name = getattr(tokenizer, "name_or_path", "") or ""
    return "gemma" in name.lower()


def _pack_tool_responses_for_gemma(messages: list[dict]) -> list[dict]:
    """Move every `role="tool"` message onto the previous assistant turn as
    `tool_responses=[{"name": ..., "response": ...}]`. Workaround for Gemma
    4's bundled chat template silently dropping `role="tool"`.
    """
    out: list[dict] = []
    for msg in messages:
        if msg.get("role") == "tool":
            if not out or out[-1].get("role") != "assistant":
                out.append({"role": "assistant", "content": ""})
            assistant = dict(out[-1])
            tool_responses = list(assistant.get("tool_responses") or [])
            tool_responses.append(
                {
                    "name": msg.get("name") or msg.get("tool_name") or "unknown",
                    "response": msg.get("content", ""),
                }
            )
            assistant["tool_responses"] = tool_responses
            out[-1] = assistant
        else:
            out.append(dict(msg))
    return out


def _render_tools(tools: list[ToolSpec]) -> list[dict]:
    """Render ToolSpec objects into the OpenAI-function shape that HF chat
    templates (Gemma, Qwen) accept via the `tools=[...]` kwarg.

    ToolSpec.params is JSON-Schema (draft 7), passed through as-is.
    """
    rendered: list[dict] = []
    for spec in tools:
        rendered.append(
            {
                "type": "function",
                "function": {
                    "name": spec.name,
                    "description": spec.description,
                    "parameters": spec.params or {
                        "type": "object",
                        "properties": {},
                    },
                },
            }
        )
    return rendered


def _hash_messages(messages: list[dict]) -> str:
    blob = json.dumps(messages, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _split_top_level(text: str, sep: str) -> list[str]:
    parts: list[str] = []
    depth = 0
    buf: list[str] = []
    for ch in text:
        if ch in "({[":
            depth += 1
        elif ch in ")}]":
            depth -= 1
        if ch == sep and depth == 0:
            parts.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
    if buf:
        parts.append("".join(buf))
    return [p.strip() for p in parts if p.strip()]


def _parse_args_blob(blob: str) -> dict[str, Any]:
    import ast

    blob = _STRDELIM_RE.sub('"', blob).strip()
    if not blob:
        return {}
    for attempt in (blob, "{" + blob + "}"):
        try:
            parsed = json.loads(attempt)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
    try:
        parsed = ast.literal_eval(blob)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    result: dict[str, Any] = {}
    for part in _split_top_level(blob, ","):
        if "=" not in part and ":" not in part:
            continue
        sep = "=" if "=" in part else ":"
        k, v = part.split(sep, 1)
        k, v = k.strip(), v.strip()
        if not k:
            continue
        try:
            result[k] = ast.literal_eval(v)
        except Exception:
            try:
                result[k] = json.loads(v)
            except Exception:
                result[k] = v.strip('"').strip("'")
    return result


def _parse_envelope(envelope_body: str) -> tuple[str, dict] | None:
    body = envelope_body.strip()
    try:
        parsed = json.loads(body)
        if isinstance(parsed, dict) and "name" in parsed:
            return (
                str(parsed["name"]),
                parsed.get("arguments") or parsed.get("args") or {},
            )
    except Exception:
        pass
    m = _CALL_SIG_RE.search(body)
    if m:
        return m.group(1), _parse_args_blob(m.group(2))
    return None


# --------------------------------------------------------------------------- #
# MLX engine                                                                   #
# --------------------------------------------------------------------------- #


class MlxEngine:
    """In-process MLX-LM engine. One model + optional adapter, pinned to a
    single worker thread.
    """

    def __init__(
        self,
        model_repo: str = DEFAULT_MODEL_REPO,
        adapter_path: str | None = DEFAULT_ADAPTER_PATH,
    ) -> None:
        self.model_repo = model_repo
        self.adapter_path = adapter_path
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="mlx-engine",
        )
        self.model, self.tokenizer = self._executor.submit(self._load).result()

    @property
    def info(self) -> dict:
        """JSON-serializable status blob. Read by `/api/health`."""
        return {"model": self.model_repo, "adapter": self.adapter_path}

    def _load(self):
        # Runs on the single-slot executor thread. We do TWO things here that
        # must happen on this thread:
        #   1. load() — model weights end up on the GPU stream that's bound
        #      to this thread.
        #   2. Rebind `mlx_lm.generate.generation_stream` to a fresh stream
        #      created on THIS thread. mlx_lm's generate.py creates that
        #      module-level stream at import time, so whichever thread first
        #      imported `mlx_lm` (often pytest collection or FastAPI lifespan)
        #      owns the stream — and `stream_generate` will then explode with
        #      "no Stream(gpu, 0) in current thread" when it tries to use it
        #      from our executor thread. The rebind makes the executor the
        #      sole owner of the generation stream, matching how the model
        #      and tokenizer are loaded.
        #
        # DO NOT "simplify" the sys.modules lookup below to
        # `mlx_lm.generate.generation_stream = ...` or `from mlx_lm import
        # generate; generate.generation_stream = ...`. Both silently no-op:
        # mlx_lm/__init__.py does `from .generate import ..., generate, ...`,
        # so the *function* `generate` shadows the submodule name in the
        # package namespace. `mlx_lm.generate` resolves to the function, and
        # setting an attribute on it does nothing — the real
        # `generation_stream` symbol still lives on the submodule, which
        # `stream_generate` reads at call time. The only correct handle is
        # `sys.modules["mlx_lm.generate"]`. If this rebind silently no-ops,
        # the symptom is `RuntimeError: There is no Stream(gpu, 0) in current
        # thread` from `mx.synchronize` inside `wired_limit`.
        import sys

        import mlx.core as mx
        from mlx_lm import load

        t0 = time.time()
        log.info(
            "[engine] loading %s (adapter=%s)", self.model_repo, self.adapter_path,
        )
        model, tokenizer = load(self.model_repo, adapter_path=self.adapter_path)
        mlx_generate_mod = sys.modules["mlx_lm.generate"]
        mlx_generate_mod.generation_stream = mx.new_stream(mx.default_device())
        log.info("[engine] ready in %.1fs", time.time() - t0)
        return model, tokenizer

    async def step(
        self,
        messages: list[dict],
        tools: list[ToolSpec],
        max_tokens: int = 2048,
    ) -> AsyncIterator[Any]:
        # Prepare messages/tools for the chat template.
        if _is_gemma_tokenizer(self.tokenizer):
            templated_messages = _pack_tool_responses_for_gemma(messages)
        else:
            templated_messages = [dict(m) for m in messages]
        rendered_tools = _render_tools(tools)

        template_kwargs: dict[str, Any] = dict(
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        if rendered_tools:
            template_kwargs["tools"] = rendered_tools
        prompt_text = self.tokenizer.apply_chat_template(
            templated_messages, **template_kwargs,
        )
        assert isinstance(prompt_text, str)
        tok_in = len(self.tokenizer.encode(prompt_text))

        # Anchor stamps at EngineStart.
        t0_mono_ns = time.monotonic_ns()
        t0_wall_ns = time.time_ns()

        def stamp(
            *,
            tok_in_delta: int = 0,
            tok_out_delta: int = 0,
            cum_in: int,
            cum_out: int,
            ms_ttft: int | None = None,
            ms_decode: int | None = None,
            tok_per_s: float | None = None,
        ) -> Stamp:
            now_mono_ns = time.monotonic_ns()
            now_wall_ns = time.time_ns()
            return Stamp(
                t_wall_ms=int((now_wall_ns - t0_wall_ns) // 1_000_000),
                t_mono_ns=int(now_mono_ns - t0_mono_ns),
                tok_in=tok_in_delta,
                tok_out=tok_out_delta,
                tok_cum_in=cum_in,
                tok_cum_out=cum_out,
                ms_ttft=ms_ttft,
                ms_decode=ms_decode,
                tok_per_s=tok_per_s,
            )

        cum_in = tok_in
        cum_out = 0
        yield EngineStart(
            stamp=stamp(tok_in_delta=tok_in, cum_in=cum_in, cum_out=cum_out),
            messages_hash=_hash_messages(templated_messages),
            n_messages=len(templated_messages),
            n_tools=len(tools),
        )

        # Producer thread streams chunks into queue; consumer batches them.
        queue: asyncio.Queue[dict | None] = asyncio.Queue()
        loop = asyncio.get_running_loop()

        class _ToolCallStop(Exception):
            pass

        producer_state: dict[str, Any] = {
            "look_buf": "",
            "tool_call": None,
            "full_text": "",
        }
        _BUF_CAP = 4096

        def on_chunk(text: str) -> None:
            if not text:
                return
            producer_state["full_text"] += text
            loop.call_soon_threadsafe(queue.put_nowait, {"text": text})
            buf = (producer_state["look_buf"] + text)[-_BUF_CAP:]
            producer_state["look_buf"] = buf
            if _CLOSE_TOOL in buf:
                full = producer_state["full_text"]
                m = None
                for match in _ENVELOPE_RE.finditer(full):
                    m = match
                if m is not None:
                    parsed = _parse_envelope(m.group(1))
                    if parsed is not None:
                        producer_state["tool_call"] = parsed
                        raise _ToolCallStop()

        def producer() -> None:
            import traceback as _tb

            from mlx_lm import stream_generate
            from mlx_lm.sample_utils import make_sampler

            sampler = make_sampler(temp=0.6)
            n_tokens = 0
            stopped_for_tool = False
            err: str | None = None
            err_tb: str | None = None
            try:
                try:
                    for response in stream_generate(
                        self.model,
                        self.tokenizer,
                        prompt=prompt_text,
                        max_tokens=max_tokens,
                        sampler=sampler,
                    ):
                        if response.text:
                            on_chunk(response.text)
                        n_tokens = response.generation_tokens
                except _ToolCallStop:
                    stopped_for_tool = True
            except Exception as exc:  # noqa: BLE001
                err = str(exc)
                err_tb = _tb.format_exc()
                log.exception("[engine] generate failed")
            finally:
                loop.call_soon_threadsafe(
                    queue.put_nowait,
                    {
                        "done": True,
                        "n_tokens": n_tokens,
                        "stopped_for_tool": stopped_for_tool,
                        "error": err,
                        "traceback": err_tb,
                    },
                )

        self._executor.submit(producer)

        # Consumer: batch tokens into EngineToken events at ~50ms / 16 tokens.
        # Per-flush tok_out delta is computed against the running n_tokens
        # reported by stream_generate (we capture it lazily — see flush).
        pending_text: list[str] = []
        pending_chunks = 0
        last_flush_mono = time.monotonic()
        ms_ttft: int | None = None
        ttft_set = False
        n_tokens_running = 0
        flushed_tok_out = 0

        def flush_tokens() -> EngineToken | None:
            nonlocal pending_text, pending_chunks, last_flush_mono
            nonlocal flushed_tok_out, cum_out
            if not pending_text:
                return None
            # Per-flush tok_out delta from the producer's running counter.
            tok_out_delta = max(0, n_tokens_running - flushed_tok_out)
            flushed_tok_out = n_tokens_running
            cum_out += tok_out_delta
            ev = EngineToken(
                stamp=stamp(
                    tok_out_delta=tok_out_delta,
                    cum_in=cum_in,
                    cum_out=cum_out,
                    ms_ttft=ms_ttft,
                ),
                text="".join(pending_text),
            )
            pending_text = []
            pending_chunks = 0
            last_flush_mono = time.monotonic()
            return ev

        producer_result: dict | None = None
        while True:
            try:
                event = await asyncio.wait_for(
                    queue.get(), timeout=_FLUSH_MS / 1000.0,
                )
            except asyncio.TimeoutError:
                ev = flush_tokens()
                if ev is not None:
                    yield ev
                continue

            if event is None:
                continue
            if "done" in event:
                producer_result = event
                # Take the producer's final n_tokens before final flush.
                n_tokens_running = int(event.get("n_tokens", 0) or 0)
                break

            text = event["text"]
            if not ttft_set:
                ms_ttft = int((time.monotonic_ns() - t0_mono_ns) // 1_000_000)
                ttft_set = True
            pending_text.append(text)
            pending_chunks += 1
            # Producer also updates n_tokens_running indirectly via on_chunk;
            # but we need a fresh value here. The producer updates
            # n_tokens AFTER on_chunk returns, so n_tokens_running may lag by
            # one chunk. That's fine — final flush at EngineDone reconciles.
            now = time.monotonic()
            if (
                pending_chunks >= _FLUSH_TOKENS
                or (now - last_flush_mono) * 1000.0 >= _FLUSH_MS
            ):
                ev = flush_tokens()
                if ev is not None:
                    yield ev

        # Final flush — reconciles cum_out to producer's authoritative count.
        ev = flush_tokens()
        if ev is not None:
            yield ev

        assert producer_result is not None
        elapsed_ns = time.monotonic_ns() - t0_mono_ns
        elapsed_s = elapsed_ns / 1e9
        n_tokens_final = int(producer_result.get("n_tokens", 0) or 0)
        ms_decode = int(elapsed_ns // 1_000_000)
        tok_per_s = (n_tokens_final / elapsed_s) if elapsed_s > 0 else 0.0

        # Determine terminal reason.
        if producer_result.get("error"):
            reason: Any = "aborted"
        elif producer_result.get("stopped_for_tool") and producer_state["tool_call"]:
            reason = "tool_dispatch"
        elif n_tokens_final >= max_tokens:
            reason = "budget"
        else:
            reason = "done"

        # Emit the optional inner Move (EngineToolCall or EngineError) in the
        # right slot before the terminal EngineDone.
        if reason == "tool_dispatch":
            name, args = producer_state["tool_call"]
            yield EngineToolCall(
                stamp=stamp(cum_in=cum_in, cum_out=cum_out),
                name=name,
                args=args,
                call_id=uuid.uuid4().hex,
            )
        elif reason == "aborted":
            yield EngineError(
                stamp=stamp(cum_in=cum_in, cum_out=cum_out),
                message=str(producer_result.get("error") or ""),
                traceback=producer_result.get("traceback"),
                during=None,
            )

        yield EngineDone(
            stamp=stamp(
                cum_in=cum_in,
                cum_out=cum_out,
                ms_ttft=ms_ttft,
                ms_decode=ms_decode,
                tok_per_s=tok_per_s,
            ),
            reason=reason,
        )
