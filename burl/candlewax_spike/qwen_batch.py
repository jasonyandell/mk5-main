"""Batch runner for local Qwen3.6 on the candlewax spike.

Mirrors ``batch_live.py`` but calls ``qwen_local.qwen_decide`` instead of the
Claude Agent SDK. Events are written in the same JSONL schema so the same
``index.html`` dashboard renders Qwen runs via ``?root=live_qwen36``.

Run:
    PYTHONPATH=. .venv/bin/python -u -m burl.candlewax_spike.qwen_batch \
        --decisions 0-3 --modes plain multimodal
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
from pathlib import Path

from burl.candlewax_spike.live_runner import (
    VIEWER_SRC, VIEWER_DST, _serve_in_thread,
)
from burl.candlewax_spike.batch_live import INDEX_SRC, INDEX_DST
from burl.candlewax_spike.qwen_local import (
    DEFAULT_MODEL, qwen_decide, load_qwen,
)
from burl.candlewax_spike.post_commit_sim import simulate_post_commit
from burl.candlewax_spike.render import _domino_label
from burl.candlewax_spike.agent import _render_prompt
from burl.eval.decision_dataset import _replay_state
from burl.tools import engine as engine_tools
from burl.tools.eq_distribution import load_eq_oracle


DEFAULT_LIVE_DIR = Path("scratch/candlewax_spike/live_qwen36")
DEFAULT_SNAPSHOTS_DIR = Path("scratch/candlewax_spike/snapshots")
MANIFEST = DEFAULT_SNAPSHOTS_DIR / "index.jsonl"

# One-line prediction instruction — tighter than v1 which caused plain mode
# to overshoot max_tokens before committing. Single sentence, inline JSON.
_PREDICTION_INSTRUCTIONS = (
    "\nBefore `commit_play`, emit ONE inline prediction on its own line:\n"
    "<prediction>{\"winner_seat\":<0-3>,\"mine\":<int>,\"theirs\":<int>}</prediction>\n"
    "where winner_seat is who wins this trick, mine/theirs are count points "
    "this trick's winner captures for your team / opponents. Engine checks.\n"
)


# Match either the verbose v1 keys or the tightened v2 keys, whichever the
# model emits.
_PREDICTION_KEY_ALIASES = {
    "predicted_winner_seat": "winner_seat",
    "predicted_count_to_my_team": "mine",
    "predicted_count_to_opponents": "theirs",
}


def _normalize_prediction(raw: dict) -> dict:
    """Map v1 verbose keys -> v2 tight keys so the rest of the pipeline only
    has to know one shape. Preserves unknown keys untouched."""
    out = {}
    for k, v in raw.items():
        out[_PREDICTION_KEY_ALIASES.get(k, k)] = v
    return out


# Regex for extracting <prediction>{...}</prediction> blocks from Qwen output.
_PREDICTION_RE = re.compile(r"<prediction>\s*(\{.*?\})\s*</prediction>", re.DOTALL)

# Fallback for Qwen's informal "<commit_play>N</commit_play>" shortcut — when
# the model skips the Hermes nested-XML tool_call wrapper and just writes the
# commit directly. Pull out the integer domino_id. Only used if no proper
# tool_call block was found in the output.
_COMMIT_SHORTCUT_RE = re.compile(
    r"<commit_play>\s*(?:domino_id\s*=\s*)?(\d+)\s*</commit_play>",
    re.IGNORECASE,
)

# Qwen Hermes chat-template expects tools in OpenAI-style function-calling shape.
_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "commit_play",
            "description": (
                "Commit to a domino play. domino_id must be one of the legal "
                "plays listed in the user turn."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "domino_id": {
                        "type": "integer",
                        "description": "0..27, must be in legal_plays",
                    },
                },
                "required": ["domino_id"],
            },
        },
    },
]


def _parse_range(s: str) -> list[int]:
    if "-" in s:
        a, b = s.split("-", 1)
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(",")]


def _install_pages(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    if VIEWER_SRC.exists():
        shutil.copy(VIEWER_SRC, VIEWER_DST)
    if INDEX_SRC.exists():
        shutil.copy(INDEX_SRC, INDEX_DST)


def _load_manifest(manifest_path: Path | None = None) -> list[dict]:
    """Load the decision-snapshot manifest.

    ``manifest_path`` defaults to the canonical ``snapshots/index.jsonl``.
    Pass a different path when evaluating against an alternate image variant
    (e.g. ``snapshots_minimal/index.jsonl``).
    """
    path = manifest_path if manifest_path is not None else MANIFEST
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


class EventLog:
    """Minimal re-implementation of live_runner.EventLog for this batch."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = path.open("w", buffering=1)
        self._t0 = time.perf_counter()

    def emit(self, kind: str, **fields):
        rec = {
            "kind": kind,
            "ts_rel": round(time.perf_counter() - self._t0, 3),
            **fields,
        }
        self._fh.write(json.dumps(rec, default=str) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


def _run_one(
    *,
    idx: int,
    mode: str,
    root: Path,
    model_path: str,
    max_tokens: int,
    enable_thinking: bool = False,
    adapter_path: str | Path | None = None,
    manifest_path: Path | None = None,
) -> dict:
    """Run a single decision through Qwen and write an events.jsonl.

    ``adapter_path`` (optional): LoRA adapter directory; threaded into
    ``qwen_decide`` so the cache picks the right variant.

    ``manifest_path`` (optional): alternate snapshot manifest, e.g. for a
    ``snapshots_minimal`` variant. Each entry must have the same schema
    (``idx``, ``png``, ``legal_plays``, etc.) as ``snapshots/index.jsonl``.
    """
    manifest = _load_manifest(manifest_path)
    matches = [e for e in manifest if int(e["idx"]) == idx]
    if not matches:
        raise KeyError(
            f"decision idx={idx} not found in manifest "
            f"{manifest_path or MANIFEST}"
        )
    entry = matches[0]
    run_id = f"{idx:03d}_{mode}"
    run_dir = root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Copy the candlewax PNG into the run dir so the viewer can show it.
    image_path: Path | None = None
    if mode == "multimodal":
        src = Path(entry["png"])
        dst = run_dir / "image.png"
        if src.exists():
            shutil.copy(src, dst)
            image_path = dst  # local path for Qwen

    log = EventLog(run_dir / "events.jsonl")
    log.emit(
        "meta",
        idx=idx, mode=mode, model=model_path,
        adapter=(str(adapter_path) if adapter_path else None),
        seed=entry["seed"], declaration=entry["declaration"],
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

    state = _replay_state(
        seed=int(entry["seed"]),
        decl_id=int(entry["declaration"]),
        play_history=[(int(p), int(d)) for p, d in entry["play_history"]],
        bidder=int(entry.get("bidder", 0)),
    )

    legal_plays = [int(p) for p in entry["legal_plays"]]
    system_content, user_text = _render_prompt(state, legal_plays)
    system_content = system_content + _PREDICTION_INSTRUCTIONS

    log.emit("prompt_system", content=system_content, n_chars=len(system_content))
    log.emit(
        "prompt_user",
        content=user_text, n_chars=len(user_text),
        n_images=(1 if image_path else 0),
    )

    final_play = -1
    try:
        result = qwen_decide(
            system_content=system_content,
            user_content=user_text,
            image_path=image_path,
            tools=_TOOLS,
            max_tokens=max_tokens,
            enable_thinking=enable_thinking,
            model_path=model_path,
            adapter_path=adapter_path,
        )
    except Exception as e:
        log.emit("error", message=f"{type(e).__name__}: {e}")
        log.emit("result", cost_usd=0.0, num_turns=0,
                 duration_ms=0, usage={}, is_error=True)
        log.close()
        return {"run_id": run_id, "final_play": -1, "ok": False,
                "error": f"{type(e).__name__}: {e}"}

    # Extract predictions from the raw text BEFORE streaming blocks so we can
    # tag the prediction event in-order with the rest.
    predictions: list[dict] = []
    for m in _PREDICTION_RE.finditer(result.raw):
        try:
            p = json.loads(m.group(1))
            if isinstance(p, dict):
                predictions.append(_normalize_prediction(p))
        except json.JSONDecodeError:
            predictions.append({"_parse_error": True, "_raw": m.group(1)})
    if predictions:
        # Log the first (Qwen should emit exactly one; extras are surplus).
        log.emit("prediction", content=predictions[0],
                 extras=(predictions[1:] if len(predictions) > 1 else None))

    # Detect Qwen's informal shortcut "<commit_play>N</commit_play>" when no
    # proper tool_call block was emitted. Synthesize a tool_call block at the
    # tail so the downstream engine-execute path runs normally.
    has_tool_call = any(b.kind == "tool_call" for b in result.blocks)
    if not has_tool_call:
        m = _COMMIT_SHORTCUT_RE.search(result.raw)
        if m:
            from burl.candlewax_spike.qwen_local import QwenBlock
            dom = int(m.group(1))
            result.blocks.append(QwenBlock(
                kind="tool_call",
                content={"name": "commit_play", "arguments": {"domino_id": dom}},
                raw=m.group(0),
            ))
            log.emit("assistant_text",
                     content=f"[harness] recovered commit via <commit_play>{dom}</commit_play> shortcut")

    # Stream the parsed blocks out as events in source order.
    for b in result.blocks:
        if b.kind == "thinking":
            log.emit("thinking", content=b.content)
        elif b.kind == "tool_call":
            c = b.content
            name = c.get("name") if isinstance(c, dict) else None
            args = c.get("arguments") if isinstance(c, dict) else None
            log.emit("tool_call", tool=name, args=args, tool_use_id="qwen-local")
            # Validate + execute locally against the engine.
            if isinstance(args, dict) and name == "commit_play":
                dom = args.get("domino_id")
                try:
                    dom = int(dom)
                except (TypeError, ValueError):
                    log.emit("tool_result",
                             tool_use_id="qwen-local",
                             content=[{"type": "text", "text": json.dumps({
                                 "ok": False,
                                 "reason": f"domino_id not int: {dom!r}",
                             })}],
                             is_error=True)
                    continue
                legal, reason = engine_tools.is_legal(state, dom)
                if legal:
                    final_play = dom
                    log.emit("tool_result",
                             tool_use_id="qwen-local",
                             content=[{"type": "text", "text": json.dumps({
                                 "ok": True, "committed": dom,
                                 "note": "Decision recorded.",
                             })}],
                             is_error=False)
                else:
                    log.emit("tool_result",
                             tool_use_id="qwen-local",
                             content=[{"type": "text", "text": json.dumps({
                                 "ok": False, "reason": reason, "committed": False,
                             })}],
                             is_error=True)
        elif b.kind == "text":
            log.emit("assistant_text", content=b.content)

    if final_play >= 0:
        log.emit("commit",
                 final_play=int(final_play),
                 label=_domino_label(int(final_play)))
        # Engine-as-fact-checker: simulate the rest of the trick so we can
        # compare Qwen's reasoning ("my team gets 10 count") against what
        # actually happens. No LLM judge — the engine is the rules.
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            oracle = load_eq_oracle(device=device)
            sim = simulate_post_commit(
                state, int(final_play),
                narrator_seat=int(entry["narrator_seat"]),
                oracle=oracle, device=device,
            )
            log.emit("post_commit_sim", **sim)

            # Diff the model's prediction block against engine truth.
            if predictions and isinstance(predictions[0], dict) and "_parse_error" not in predictions[0]:
                pred = predictions[0]
                try:
                    p_win = int(pred.get("winner_seat", pred.get("predicted_winner_seat", -1)))
                    p_mine = int(pred.get("mine", pred.get("predicted_count_to_my_team", -1)))
                    p_theirs = int(pred.get("theirs", pred.get("predicted_count_to_opponents", -1)))
                except (TypeError, ValueError):
                    p_win = p_mine = p_theirs = -1
                actual_win = int(sim.get("trick_winner_seat", -99))
                actual_mine = int(sim.get("count_to_my_team", -99))
                actual_theirs = int(sim.get("count_to_opponents", -99))
                winner_ok = (p_win == actual_win)
                mine_ok = (p_mine == actual_mine)
                theirs_ok = (p_theirs == actual_theirs)
                all_ok = winner_ok and mine_ok and theirs_ok
                log.emit(
                    "prediction_check",
                    all_correct=all_ok,
                    winner_correct=winner_ok,
                    mine_correct=mine_ok,
                    theirs_correct=theirs_ok,
                    predicted={"winner_seat": p_win, "mine": p_mine, "theirs": p_theirs},
                    actual={"winner_seat": actual_win, "mine": actual_mine, "theirs": actual_theirs},
                )
            elif not predictions:
                log.emit("prediction_check", all_correct=None,
                         reason="no prediction block emitted")
        except Exception as e:
            log.emit("post_commit_sim", error=f"{type(e).__name__}: {e}")

    # Tokens/s ≈ gen_tokens / elapsed
    tps = (result.generation_tokens / result.elapsed_s) if result.elapsed_s > 0 else 0.0
    log.emit(
        "result",
        cost_usd=0.0,                    # local run
        num_turns=1,
        duration_ms=int(result.elapsed_s * 1000),
        usage={
            "prompt_tokens": result.prompt_tokens,
            "generation_tokens": result.generation_tokens,
            "tokens_per_sec": round(tps, 1),
            "peak_memory_gb": result.peak_memory_gb,
        },
        is_error=False,
    )
    log.close()

    return {"run_id": run_id, "final_play": final_play,
            "ok": True, "error": None,
            "elapsed_s": round(result.elapsed_s, 1)}


def _append_summary(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decisions", type=_parse_range, default="0-3")
    parser.add_argument("--modes", nargs="+",
                        default=["plain", "multimodal"],
                        choices=["plain", "multimodal"])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--thinking", action="store_true",
                        help="Enable Qwen extended thinking (default off — the "
                             "forced <think> prefix can blow max-tokens).")
    parser.add_argument("--live-dir", type=Path, default=DEFAULT_LIVE_DIR)
    parser.add_argument("--adapter", type=Path, default=None,
                        help="Optional LoRA adapter directory (mlx-vlm format).")
    parser.add_argument("--snapshots-dir", type=Path,
                        default=DEFAULT_SNAPSHOTS_DIR,
                        help="Snapshot directory; index.jsonl inside is the manifest.")
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--port", type=int, default=8769)
    args = parser.parse_args()

    manifest_path = args.snapshots_dir / "index.jsonl"

    args.live_dir.mkdir(parents=True, exist_ok=True)
    _install_pages(args.live_dir)

    if args.serve:
        _serve_in_thread(Path("scratch/candlewax_spike"), args.port)
        time.sleep(0.2)
        print(f"[qwen-batch] dashboard: "
              f"http://127.0.0.1:{args.port}/index.html?root={args.live_dir.name}",
              file=sys.stderr, flush=True)

    # Preload the model once; the cache in qwen_local keeps it hot.
    print(
        f"[qwen-batch] preloading {args.model} (adapter={args.adapter}) ...",
        file=sys.stderr, flush=True,
    )
    load_qwen(args.model, adapter_path=args.adapter)

    summary_path = args.live_dir / "batch_summary.jsonl"
    if summary_path.exists():
        summary_path.unlink()

    total = len(args.decisions) * len(args.modes)
    print(f"[qwen-batch] starting {total} runs", file=sys.stderr, flush=True)
    completed = 0
    for idx in args.decisions:
        for mode in args.modes:
            print(f"\n[qwen-batch] ▶ d{idx:03d} mode={mode}",
                  file=sys.stderr, flush=True)
            t0 = time.time()
            row = _run_one(
                idx=idx, mode=mode, root=args.live_dir,
                model_path=args.model, max_tokens=args.max_tokens,
                enable_thinking=args.thinking,
                adapter_path=args.adapter,
                manifest_path=manifest_path,
            )
            row["ts"] = time.strftime("%H:%M:%S")
            row["idx"] = idx
            row["mode"] = mode
            _append_summary(summary_path, row)
            elapsed = time.time() - t0
            print(f"[qwen-batch] ✓ d{idx:03d} mode={mode} "
                  f"play={row['final_play']} in {elapsed:.1f}s ok={row['ok']}",
                  file=sys.stderr, flush=True)
            completed += 1

    print(f"\n[qwen-batch] DONE — {completed}/{total}",
          file=sys.stderr, flush=True)

    if args.serve:
        print("[qwen-batch] server still running — Ctrl-C to stop.",
              file=sys.stderr, flush=True)
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
