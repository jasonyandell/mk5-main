"""Single-file live runner for the candlewax spike.

Runs one decision through Opus (plain or multimodal) and writes every event —
system prompt, user turn, thinking blocks, assistant text, tool calls, tool
results, commit, final result — as a JSONL stream to
``scratch/candlewax_spike/live/{run_id}/events.jsonl``. Each line is flushed
immediately so the browser viewer can read events as they land.

Also copies the candlewax PNG into the run dir as ``image.png`` when mode is
multimodal, so the viewer can display it without absolute-path games.

Run:
    PYTHONPATH=. .venv/bin/python -u -m burl.candlewax_spike.live_runner \
        --idx 2 --mode multimodal

Then open ``scratch/candlewax_spike/viewer.html?run=002_multimodal`` in a
browser (serve the directory with ``python -m http.server`` if file:// fetch
is blocked, or use the --serve flag to auto-start a local server).
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import http.server
import json
import shutil
import socketserver
import sys
import threading
import time
from pathlib import Path
from typing import Any

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
    query,
)

from burl.candlewax_spike.agent import (
    _CTX,
    _MCP_TOOL_NAMES,
    _build_mcp_server,
    _render_prompt,
    _user_content_for_mode,
    MCP_SERVER_NAME,
    DEFAULT_MODEL,
)
from burl.candlewax_spike.render import _domino_label
from burl.eval.decision_dataset import _replay_state


MANIFEST = Path("scratch/candlewax_spike/snapshots/index.jsonl")
LIVE_ROOT = Path("scratch/candlewax_spike/live")   # default; overridable via --live-dir
VIEWER_SRC = Path(__file__).resolve().parent / "viewer.html"
VIEWER_DST = Path("scratch/candlewax_spike/viewer.html")

_DISALLOWED_BUILTINS = [
    "Bash", "Read", "Edit", "Write", "Glob", "Grep", "Task", "WebFetch",
    "WebSearch", "NotebookEdit", "TodoWrite", "SlashCommand", "MultiEdit",
    "KillShell", "BashOutput",
]


# --------------------------------------------------------------------------- #
# Event writer — one JSON object per line, flushed immediately.                #
# --------------------------------------------------------------------------- #


class EventLog:
    """Append-only JSONL writer with per-event flush and wall-clock stamps."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = path.open("w", buffering=1)  # line-buffered
        self._t0 = time.perf_counter()
        self.path = path

    def emit(self, kind: str, **fields: Any) -> None:
        rec = {
            "kind": kind,
            "ts_rel": round(time.perf_counter() - self._t0, 3),
            **fields,
        }
        self._fh.write(json.dumps(rec, default=_json_default) + "\n")
        self._fh.flush()
        # Also mirror to stderr so terminal-watchers see something.
        short = _short_repr(rec)
        print(f"[live] t+{rec['ts_rel']:>6.2f}s {kind:<16} {short}",
              file=sys.stderr, flush=True)

    def close(self) -> None:
        self._fh.close()


def _json_default(x: Any) -> Any:
    try:
        return str(x)
    except Exception:
        return repr(x)


def _short_repr(rec: dict) -> str:
    kind = rec["kind"]
    if kind == "thinking":
        return rec.get("content", "")[:90].replace("\n", " ") + "…"
    if kind == "assistant_text":
        return rec.get("content", "")[:90].replace("\n", " ")
    if kind == "tool_call":
        return f"{rec.get('tool')}({json.dumps(rec.get('args', {}))})"
    if kind == "tool_result":
        c = rec.get("content")
        if isinstance(c, list):
            parts = [str(x.get("text", x))[:60] for x in c if isinstance(x, dict)]
            return " | ".join(parts)[:90]
        return str(c)[:90]
    if kind == "commit":
        return f"play={rec.get('final_play')} ({rec.get('label')})"
    if kind == "result":
        return (f"cost=${rec.get('cost_usd', 0):.3f} "
                f"turns={rec.get('num_turns', 0)} "
                f"is_error={rec.get('is_error', False)}")
    if kind == "prompt_system":
        return f"{rec.get('n_chars', 0)} chars"
    if kind == "prompt_user":
        imgs = rec.get('n_images', 0)
        return f"{rec.get('n_chars', 0)} chars, {imgs} image(s)"
    if kind == "error":
        return rec.get("message", "")
    if kind == "meta":
        return f"idx={rec.get('idx')} mode={rec.get('mode')} model={rec.get('model')}"
    return ""


# --------------------------------------------------------------------------- #
# Runner.                                                                      #
# --------------------------------------------------------------------------- #


def _load_manifest() -> list[dict]:
    with MANIFEST.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _state_from_entry(entry: dict):
    history = [(int(p), int(d)) for p, d in entry["play_history"]]
    return _replay_state(
        seed=int(entry["seed"]),
        decl_id=int(entry["declaration"]),
        play_history=history,
        bidder=int(entry.get("bidder", 0)),
    )


async def run_live(
    *,
    idx: int,
    mode: str,
    model: str,
    max_turns: int,
    max_budget_usd: float,
    thinking_tokens: int | None,
    live_root: Path | None = None,
) -> dict:
    manifest = _load_manifest()
    entry = next((e for e in manifest if int(e["idx"]) == idx), None)
    if entry is None:
        raise SystemExit(f"idx {idx} not in manifest {MANIFEST}")

    root = Path(live_root) if live_root is not None else LIVE_ROOT
    run_id = f"{idx:03d}_{mode}"
    run_dir = root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Copy image into the run dir (browser serves it).
    image_path = Path(entry["png"]) if mode == "multimodal" else None
    if image_path is not None:
        shutil.copy(image_path, run_dir / "image.png")

    # Install the viewer alongside the live dir if not already there.
    if VIEWER_SRC.exists():
        shutil.copy(VIEWER_SRC, VIEWER_DST)

    log = EventLog(run_dir / "events.jsonl")
    state = _state_from_entry(entry)

    log.emit(
        "meta",
        idx=idx,
        mode=mode,
        model=model,
        seed=entry["seed"],
        declaration=entry["declaration"],
        narrator_seat=entry["narrator_seat"],
        legal_plays=entry["legal_plays"],
        legal_labels=entry.get("legal_labels", [_domino_label(int(p)) for p in entry["legal_plays"]]),
        bot_play=entry["bot_play"],
        bot_label=_domino_label(int(entry["bot_play"])),
        bot_eq=entry["bot_eq"],
        eq_gap=entry["eq_gap"],
        per_play_eq=entry["per_play_eq"],
        per_play_summary=entry.get("per_play_summary"),
        image="image.png" if image_path else None,
    )

    _CTX.bind(state)
    system_content, user_text = _render_prompt(state, [int(p) for p in entry["legal_plays"]])
    image_bytes = image_path.read_bytes() if image_path else None
    user_content = _user_content_for_mode(user_text, image_bytes=image_bytes)

    log.emit("prompt_system", content=system_content, n_chars=len(system_content))
    if isinstance(user_content, list):
        text_block = next((b["text"] for b in user_content if b.get("type") == "text"), "")
        n_images = sum(1 for b in user_content if b.get("type") == "image")
        log.emit(
            "prompt_user",
            content=text_block,
            n_chars=len(text_block),
            n_images=n_images,
        )
    else:
        log.emit(
            "prompt_user",
            content=user_content,
            n_chars=len(user_content),
            n_images=0,
        )

    mcp = _build_mcp_server()
    options = ClaudeAgentOptions(
        model=model,
        system_prompt=system_content,
        mcp_servers={MCP_SERVER_NAME: mcp},
        allowed_tools=list(_MCP_TOOL_NAMES),
        disallowed_tools=list(_DISALLOWED_BUILTINS),
        permission_mode="bypassPermissions",
        max_turns=max_turns,
        max_budget_usd=max_budget_usd,
        setting_sources=[],
        max_thinking_tokens=thinking_tokens,
    )

    async def _prompt_stream():
        yield {"type": "user", "message": {"role": "user", "content": user_content}}

    result_payload: dict[str, Any] = {
        "cost_usd": 0.0, "num_turns": 0, "duration_ms": 0,
        "usage": {}, "is_error": False,
    }

    try:
        async for msg in query(prompt=_prompt_stream(), options=options):
            cls = type(msg).__name__
            if isinstance(msg, AssistantMessage):
                for blk in msg.content:
                    if isinstance(blk, TextBlock):
                        if blk.text.strip():
                            log.emit("assistant_text", content=blk.text)
                    elif isinstance(blk, ThinkingBlock):
                        log.emit("thinking", content=blk.thinking)
                    elif isinstance(blk, ToolUseBlock):
                        log.emit("tool_call",
                                 tool=blk.name, args=blk.input,
                                 tool_use_id=blk.id)
            elif isinstance(msg, UserMessage):
                content = msg.content
                if isinstance(content, list):
                    for blk in content:
                        if isinstance(blk, ToolResultBlock):
                            log.emit("tool_result",
                                     tool_use_id=blk.tool_use_id,
                                     content=blk.content,
                                     is_error=bool(blk.is_error))
            elif isinstance(msg, ResultMessage):
                result_payload = {
                    "cost_usd": float(msg.total_cost_usd or 0.0),
                    "num_turns": int(msg.num_turns or 0),
                    "duration_ms": int(msg.duration_ms or 0),
                    "usage": dict(msg.usage) if msg.usage else {},
                    "is_error": bool(msg.is_error),
                }
            else:
                log.emit("other_message", cls=cls)
    except Exception as e:
        log.emit("error", message=f"{type(e).__name__}: {e}")
        raise
    finally:
        if _CTX.final_play is not None:
            log.emit("commit",
                     final_play=int(_CTX.final_play),
                     label=_domino_label(int(_CTX.final_play)))
        log.emit("result", **result_payload)
        log.close()

    summary = {
        "idx": idx, "mode": mode, "run_id": run_id,
        "final_play": int(_CTX.final_play) if _CTX.final_play is not None else -1,
        "events_path": str(log.path),
    }
    return summary


# --------------------------------------------------------------------------- #
# Optional built-in static server.                                             #
# --------------------------------------------------------------------------- #


def _serve_in_thread(directory: Path, port: int) -> threading.Thread:
    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, directory=str(directory), **kwargs)

        def end_headers(self) -> None:
            self.send_header("Cache-Control", "no-store")
            super().end_headers()

        def log_message(self, fmt: str, *args: Any) -> None:
            pass

    def _run() -> None:
        with socketserver.ThreadingTCPServer(("127.0.0.1", port), Handler) as srv:
            print(f"[live] serving {directory} at http://127.0.0.1:{port}/viewer.html",
                  file=sys.stderr, flush=True)
            srv.serve_forever()

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return t


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--idx", type=int, default=2)
    parser.add_argument("--mode", choices=["plain", "multimodal"], default="multimodal")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--thinking", type=int, default=4000,
                        help="Extended thinking token budget; 0 disables.")
    parser.add_argument("--max-turns", type=int, default=6)
    parser.add_argument("--budget", type=float, default=0.50)
    parser.add_argument("--serve", action="store_true",
                        help="Launch a local HTTP server rooted at scratch/candlewax_spike/.")
    parser.add_argument("--port", type=int, default=8769)
    parser.add_argument("--live-dir", type=Path, default=None,
                        help="Override live/ output dir (e.g. scratch/candlewax_spike/live_haiku).")
    args = parser.parse_args()

    if args.serve:
        _serve_in_thread(Path("scratch/candlewax_spike"), args.port)
        # Give server a beat to bind.
        time.sleep(0.2)
        print(f"[live] open: http://127.0.0.1:{args.port}/viewer.html?run={args.idx:03d}_{args.mode}",
              file=sys.stderr, flush=True)

    summary = asyncio.run(run_live(
        idx=args.idx, mode=args.mode, model=args.model,
        max_turns=args.max_turns, max_budget_usd=args.budget,
        thinking_tokens=args.thinking if args.thinking > 0 else None,
        live_root=args.live_dir,
    ))
    print(f"\n[live] DONE: {summary}", file=sys.stderr, flush=True)
    if args.serve:
        print("[live] server still running — Ctrl-C to stop.", file=sys.stderr, flush=True)
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
