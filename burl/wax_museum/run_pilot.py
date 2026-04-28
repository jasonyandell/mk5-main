"""N=5 pilot — hard-gated HATEOAS tool surface on trick-6 decisions.

Run:
    # Stub model (no Modal) — end-to-end plumbing check.
    source .venv/bin/activate
    python -u -m burl.wax_museum.run_pilot --model stub --n 2

    # Real Modal L4 base Gemma 4 E2B, 32k context.
    modal deploy burl/wax_museum/modal_serve.py   # first time only
    python -u -m burl.wax_museum.run_pilot --model modal --n 5

Tail the run:
    tail -f burl/wax_museum/logs/<run_id>/tail.log
    cat burl/wax_museum/logs/<run_id>/thoughts/d0_t1.md   # full turn-1 thought

Bail behavior:
    - Per-decision: if turn 1 produces no tool call AND <200 chars of thought,
      the decision is marked bailed and we skip to the next one.
    - Run-level: if decisions 1 AND 2 both bail, stop the run — the hypothesis
      is that the hard gate itself is alien to the base model.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from burl.eval.decision_dataset import BurlDecision, load_dataset
from burl.wax_museum.harness import WaxResult, run_decision_waxed


DEFAULT_DATASET = Path("burl/eval/data/move4_decisions_n50.jsonl")
LOG_ROOT = Path("burl/wax_museum/logs")

# Gemma 4 E2B's max_position_embeddings per its HF config. "Don't cut it off."
# We trust EOS (<turn|>, <tool_response|>) to stop generation naturally.
MODEL_MAX_TOKENS = 131072


# --------------------------------------------------------------------------- #
# Logging                                                                      #
# --------------------------------------------------------------------------- #


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _wall_time() -> str:
    return datetime.now().strftime("%H:%M:%S")


class PilotLogger:
    """Writes events.jsonl (machine-readable) + tail.log (human-tail-able) +
    thoughts/d<n>_t<t>.md (one per turn, full completion text) + live.log
    (token-by-token stream, also mirrored to stdout)."""

    def __init__(self, run_dir: Path, live_to_stdout: bool = True):
        self.run_dir = run_dir
        self.thoughts_dir = run_dir / "thoughts"
        self.thoughts_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = run_dir / "events.jsonl"
        self.tail_path = run_dir / "tail.log"
        self.live_path = run_dir / "live.log"
        self.events_fp = self.events_path.open("w", buffering=1)
        self.tail_fp = self.tail_path.open("w", buffering=1)
        self.live_fp = self.live_path.open("w", buffering=1)
        self.live_to_stdout = live_to_stdout
        self.current_idx: int = -1

    def close(self) -> None:
        self.events_fp.close()
        self.tail_fp.close()
        self.live_fp.close()

    def live_write(self, text: str) -> None:
        """Write a chunk to live.log + stdout. No newlines added — streamer
        feeds raw token text so the file reads naturally."""
        self.live_fp.write(text)
        if self.live_to_stdout:
            sys.stdout.write(text)
            sys.stdout.flush()

    def live_header(self, title: str) -> None:
        """Legible section divider for live.log."""
        header = f"\n\n{'=' * 78}\n=== {title}\n{'=' * 78}\n"
        self.live_write(header)

    def live_subheader(self, title: str) -> None:
        self.live_write(f"\n\n--- {title} ---\n")

    def live_input(self, messages: list[dict], tool_schemas: list[dict]) -> None:
        """Write the full prompt going into the model in a legible form.

        Renders structured assistant messages (tool_calls + tool_responses) in
        the same native shape the chat template emits, so what we read here
        matches what Gemma actually sees in the prompt.
        """
        self.live_subheader("INPUT messages")
        for m in messages:
            role = m.get("role", "?")
            name = m.get("name", "")
            content = m.get("content", "") or ""
            name_str = f" name={name}" if name else ""
            self.live_write(f"\n[{role}{name_str}]\n{content}\n")
            # Render structured tool_calls / tool_responses so what the model
            # sees via the chat template is legible here too.
            tcs = m.get("tool_calls") or []
            for tc in tcs:
                fn = tc.get("function", tc)
                self.live_write(
                    f"  <|tool_call>{fn.get('name')}({fn.get('arguments')})<tool_call|>\n"
                )
            trs = m.get("tool_responses") or []
            for tr in trs:
                resp = tr.get("response", "")
                self.live_write(
                    f"  <|tool_response name={tr.get('name')}>\n{resp}\n  <tool_response|>\n"
                )
        if tool_schemas:
            self.live_subheader(f"INPUT tools ({len(tool_schemas)} available)")
            names = [s["function"]["name"] for s in tool_schemas]
            self.live_write(" | ".join(names) + "\n")
        self.live_subheader("OUTPUT (streaming)")

    def _emit_event(self, event: dict) -> None:
        event = {"t": _now_iso(), **event}
        self.events_fp.write(json.dumps(event, default=str) + "\n")

    def _tail(self, line: str) -> None:
        self.tail_fp.write(f"[{_wall_time()}] {line}\n")

    # Pilot-level events --------------------------------------------------- #

    def run_start(self, n: int, cfg: dict) -> None:
        self._emit_event({"evt": "run_start", "n": n, "config": cfg})
        self._tail(f"[run_start] n={n} config={cfg}")

    def run_end(self, summary: dict) -> None:
        self._emit_event({"evt": "run_end", "summary": summary})
        self._tail(f"[run_end] {summary}")

    def decision_start(self, idx: int, decision: BurlDecision) -> None:
        self.current_idx = idx
        self._emit_event({
            "evt": "decision_start", "idx": idx,
            "seed": decision.seed, "decl": decision.declaration,
            "narrator": decision.narrator_seat,
            "legal": decision.legal_plays, "bot_play": decision.bot_play,
            "bot_eq": decision.bot_eq, "eq_gap": decision.eq_gap,
        })
        self._tail(
            f"[d{idx}] start seed={decision.seed} decl={decision.declaration} "
            f"seat={decision.narrator_seat} legal={decision.legal_plays} "
            f"bot_play={decision.bot_play} gap={decision.eq_gap:.2f}"
        )

    def decision_end(self, idx: int, result: WaxResult, grade: dict) -> None:
        summary = {
            "final_play": result.trace.final_play,
            "bailed": result.bailed,
            "bail_reason": result.bail_reason,
            "n_turns": result.n_turns,
            "probed": result.probed,
            "tool_seq": result.tool_call_sequence,
            "gated_commit_leaks": result.gated_commit_leaks,
            "tool_cache_stats": result.tool_cache_stats,
            **grade,
        }
        self._emit_event({"evt": "decision_end", "idx": idx, "summary": summary})
        if result.bailed:
            self._tail(f"[d{idx}] BAILED — {result.bail_reason}")
        else:
            match = "T" if grade.get("matches_bot") else "F"
            eq_delta = grade.get("eq_delta_vs_bot")
            delta_s = f"{eq_delta:+.2f}" if isinstance(eq_delta, (int, float)) else "—"
            self._tail(
                f"[d{idx}] COMMIT {result.trace.final_play} "
                f"(bot={grade.get('bot_play')} match={match} Δ={delta_s} "
                f"turns={result.n_turns} probed={'Y' if result.probed else 'N'} "
                f"seq={result.tool_call_sequence})"
            )

    # Turn-level events forwarded from the harness ------------------------- #

    def on_harness_event(self, idx: int, event: dict) -> None:
        event_with_idx = {"idx": idx, **event}
        self._emit_event(event_with_idx)
        evt = event["evt"]
        if evt == "turn_start":
            self._tail(f"[d{idx} t{event['turn']}] menu={event['menu']}")
        elif evt == "completion":
            # Persist full completion to thoughts/d<idx>_t<turn>.md.
            turn = event["turn"]
            path = self.thoughts_dir / f"d{idx}_t{turn}.md"
            path.write_text(
                f"# d{idx} turn {turn} — {event['n_chars']} chars\n\n"
                f"## Parsed tool calls\n\n"
                f"{event.get('tool_calls_parsed')}\n\n"
                f"## Native commit (if any)\n\n"
                f"{event.get('native_commit')}\n\n"
                f"## Raw completion\n\n"
                f"```\n{event['full_completion']}\n```\n"
            )
            preview = event["thought_preview"].replace("\n", " ")[:140]
            self._tail(
                f"[d{idx} t{turn}] {event['n_chars']} chars → "
                f"thoughts/d{idx}_t{turn}.md  preview: {preview!r}"
            )
        elif evt == "tool_call":
            marker = "ok" if event["ok"] else f"ERR: {event.get('error')}"
            self._tail(
                f"[d{idx} t{event['turn']}] tool_call: "
                f"{event['tool']}({event.get('args')}) → {marker}"
            )
        elif evt == "tool_cache_precompute":
            self._tail(f"[d{idx}] tool_cache_precompute: {event['stats']}")
        elif evt == "menu_change":
            added = sorted(set(event["after_menu"]) - set(event["before_menu"]))
            removed = sorted(set(event["before_menu"]) - set(event["after_menu"]))
            deltas = []
            if added:
                deltas.append("+" + ",+".join(added))
            if removed:
                deltas.append("-" + ",-".join(removed))
            self._tail(
                f"[d{idx} t{event['turn']}] menu_change: {event['before']} → "
                f"{event['after']} ({' '.join(deltas)})"
            )
        elif evt == "gate_reject":
            self._tail(
                f"[d{idx} t{event['turn']}] gate_reject: {event['tool']} — {event['reason']}"
            )
        elif evt == "commit_attempt":
            legal = "legal" if event["legal"] else f"ILLEGAL ({event.get('reason')})"
            self._tail(
                f"[d{idx} t{event['turn']}] commit_attempt: dom={event['domino_id']} → {legal}"
            )
        elif evt == "commit_ok":
            self._tail(f"[d{idx} t{event['turn']}] commit_ok: dom={event['domino_id']}")
        elif evt == "retry_exhausted":
            self._tail(f"[d{idx} t{event['turn']}] retry_exhausted")
        elif evt == "bail":
            self._tail(f"[d{idx}] bail: {event['reason']}")
        elif evt == "empty_nudge":
            self._tail(f"[d{idx} t{event['turn']}] empty_nudge")
        elif evt == "turn_end":
            self._tail(f"[d{idx} t{event['turn']}] ---")


# --------------------------------------------------------------------------- #
# Model adapters                                                               #
# --------------------------------------------------------------------------- #


def make_stub_model(decision: BurlDecision) -> Callable:
    """Plumbing check: emit explore_game then probe then commit, all legal."""
    target = int(decision.legal_plays[0])
    script = iter([
        "I'll look at my top candidate.\n"
        + "x" * 240
        + f"\n<|tool_call>call:explore_game{{play:{target}}}<tool_call|>",
        f"Probe downside.\n<|tool_call>call:probe_worst_case{{play:{target}}}<tool_call|>",
        f"Commit.\n<|tool_call>call:commit_play{{domino_id:{target}}}<tool_call|>",
    ])

    def call(messages: list[dict], tool_schemas: list[dict]) -> str:
        try:
            return next(script)
        except StopIteration:
            return f"<|tool_call>call:commit_play{{domino_id:{target}}}<tool_call|>"

    return call


def make_modal_model(server: Any, max_tokens: int = 8192) -> Callable:
    """Wrap the WaxMuseumServer remote."""
    def call(messages: list[dict], tool_schemas: list[dict]) -> str:
        result = server.generate.remote(
            messages, tools=tool_schemas, max_tokens=max_tokens,
            temperature=0.6, enable_thinking=False,
        )
        return result["text"]
    return call


def make_local_model(
    max_tokens: int = MODEL_MAX_TOKENS,
    on_chunk: Callable[[str], None] | None = None,
    on_turn_start: Callable[[list[dict], list[dict]], None] | None = None,
) -> Callable:
    """Wrap GemmaLocalNative — MLX-LM on M5 Max, ~83 tok/s single-stream.

    ``on_chunk`` fires once per streaming output chunk (token-ish). ``on_turn_start``
    fires with (messages, tool_schemas) right before generation kicks off, so the
    pilot can print/log the full input prompt alongside the streamed output.
    """
    from burl.modal.gemma_local import GemmaLocalNative
    server = GemmaLocalNative()

    def call(messages: list[dict], tool_schemas: list[dict]) -> str:
        if on_turn_start is not None:
            on_turn_start(messages, tool_schemas)
        result = server.generate_native(
            messages, tools=tool_schemas, max_tokens=max_tokens,
            temperature=0.6, enable_thinking=False,
            on_chunk=on_chunk,
        )
        return result["text"]
    return call


DEFAULT_QWEN_REPO = "mlx-community/Qwen3.6-35B-A3B-4bit"


def make_local_qwen_model(
    max_tokens: int = MODEL_MAX_TOKENS,
    on_chunk: Callable[[str], None] | None = None,
    on_turn_start: Callable[[list[dict], list[dict]], None] | None = None,
    model_repo: str = DEFAULT_QWEN_REPO,
) -> Callable:
    """Wrap Qwen3.6-35B-A3B-4bit via mlx-lm.

    Qwen's chat template reads tool responses from ``role='tool'`` messages
    (OpenAI-style); the harness needs ``tool_response_style='role_tool'``.
    Parser is ``burl.wax_museum.qwen_parser.parse_qwen_completion``.
    """
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler

    model, tokenizer = load(model_repo)

    def call(messages: list[dict], tool_schemas: list[dict]) -> str:
        if on_turn_start is not None:
            on_turn_start(messages, tool_schemas)
        prompt_text = tokenizer.apply_chat_template(
            messages, tools=tool_schemas,
            tokenize=False, add_generation_prompt=True,
        )
        sampler = make_sampler(temp=0.6)
        parts: list[str] = []
        for response in stream_generate(
            model, tokenizer, prompt=prompt_text,
            max_tokens=max_tokens, sampler=sampler,
        ):
            parts.append(response.text)
            if on_chunk is not None and response.text:
                on_chunk(response.text)
        return "".join(parts)
    return call


# --------------------------------------------------------------------------- #
# Grading                                                                      #
# --------------------------------------------------------------------------- #


def grade(result: WaxResult, decision: BurlDecision) -> dict:
    legal = set(decision.legal_plays)
    final = result.trace.final_play
    if final is None:
        return {
            "final_play": None, "legal_final": False, "matches_bot": False,
            "bot_play": decision.bot_play, "burl_eq": None,
            "eq_delta_vs_bot": None, "k1_pass": False,
        }
    legal_final = int(final) in legal
    burl_eq = decision.per_play_eq.get(int(final)) if legal_final else None
    delta = (burl_eq - decision.bot_eq) if burl_eq is not None else None
    return {
        "final_play": int(final), "legal_final": legal_final,
        "matches_bot": int(final) == int(decision.bot_play),
        "bot_play": int(decision.bot_play),
        "burl_eq": burl_eq, "eq_delta_vs_bot": delta,
        "k1_pass": (delta is not None) and (delta >= -1e-6),
    }


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--model", choices=["stub", "local", "qwen", "modal"], default="stub")
    ap.add_argument("--max-tokens", type=int, default=MODEL_MAX_TOKENS,
                    help=(
                        f"Per-turn generation cap. Default {MODEL_MAX_TOKENS} "
                        "(Gemma 4 E2B's max_position_embeddings). We trust EOS "
                        "to stop naturally — 'don't cut it off.'"
                    ))
    ap.add_argument("--no-live-stdout", action="store_true",
                    help="Write live stream only to live.log; suppress stdout mirror.")
    ap.add_argument("--max-turns", type=int, default=8)
    ap.add_argument("--max-retries", type=int, default=3)
    ap.add_argument("--eager-tool-cache", action="store_true",
                    help="Precompute the decision-local wax_museum tool lattice.")
    ap.add_argument("--run-tag", type=str, default="",
                    help="Optional suffix appended to the run_id.")
    ap.add_argument("--run-id", type=str, default="",
                    help=(
                        "Explicit run_id (creates logs/<run-id>/). Overrides "
                        "the timestamp scheme. Useful for background launches "
                        "where you want the log path up front."
                    ))
    args = ap.parse_args()

    dataset = load_dataset(args.dataset)
    sl = dataset[args.start : args.start + args.n]
    if not sl:
        print(f"[pilot] empty slice: start={args.start} n={args.n} dataset_size={len(dataset)}",
              file=sys.stderr)
        return 2

    if args.run_id:
        run_id = args.run_id
    else:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        if args.run_tag:
            run_id = f"{run_id}_{args.run_tag}"
    run_dir = LOG_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = PilotLogger(run_dir, live_to_stdout=not args.no_live_stdout)

    config = {
        "model": args.model,
        "dataset": str(args.dataset),
        "n": len(sl),
        "max_tokens": args.max_tokens,
        "max_turns": args.max_turns,
        "max_retries": args.max_retries,
        "eager_tool_cache": args.eager_tool_cache,
    }
    logger.run_start(len(sl), config)
    print(f"[pilot] run_id={run_id}", file=sys.stderr)
    print(f"[pilot] tail -f {run_dir}/tail.log", file=sys.stderr)

    oracle = None
    try:
        from burl.tools.eq_distribution import load_eq_oracle
        oracle = load_eq_oracle()
    except Exception as e:
        print(f"[pilot] oracle load failed: {e}", file=sys.stderr)
        return 3

    summary_rows: list[dict] = []
    consecutive_bails = 0
    run_start_s = time.time()

    modal_app_ctx = None
    server = None
    model_factory: Callable[[BurlDecision], Callable]

    # Mutable holder so the live callbacks can re-address the current decision.
    current = {"idx": -1, "turn": 0}

    # Harness plugins — per-model parser + tool-response shape.
    parse_completion = None
    tool_response_style = "gemma_native"

    if args.model == "stub":
        model_factory = make_stub_model
    elif args.model in ("local", "qwen"):
        def on_turn_start(messages: list[dict], tool_schemas: list[dict]) -> None:
            current["turn"] += 1
            logger.live_header(
                f"decision {current['idx']}  turn {current['turn']}  "
                f"menu: {[s['function']['name'] for s in tool_schemas]}"
            )
            logger.live_input(messages, tool_schemas)

        def on_chunk(text: str) -> None:
            logger.live_write(text)

        if args.model == "local":
            shared_local = make_local_model(
                max_tokens=args.max_tokens,
                on_chunk=on_chunk,
                on_turn_start=on_turn_start,
            )
        else:  # qwen
            from burl.wax_museum.qwen_parser import parse_qwen_completion
            parse_completion = parse_qwen_completion
            tool_response_style = "role_tool"
            shared_local = make_local_qwen_model(
                max_tokens=args.max_tokens,
                on_chunk=on_chunk,
                on_turn_start=on_turn_start,
            )
        model_factory = lambda _d: shared_local
    else:
        import modal as modal_lib  # type: ignore
        from burl.wax_museum.modal_serve import app as modal_app, WaxMuseumServer
        modal_app_ctx = modal_app.run()
        modal_app_ctx.__enter__()
        server = WaxMuseumServer()
        model_factory = lambda _d: make_modal_model(server, max_tokens=args.max_tokens)

    try:
        for i, decision in enumerate(sl):
            current["idx"] = i
            current["turn"] = 0
            logger.decision_start(i, decision)
            model = model_factory(decision)

            try:
                result = run_decision_waxed(
                    decision.game_state,
                    model,
                    max_turns=args.max_turns,
                    max_retries=args.max_retries,
                    on_event=lambda e, i=i: logger.on_harness_event(i, e),
                    oracle=oracle,
                    parse_completion=parse_completion,
                    tool_response_style=tool_response_style,
                    eager_tool_cache=args.eager_tool_cache,
                )
            except Exception as e:
                print(f"[pilot] d{i} EXCEPTION: {e}", file=sys.stderr)
                logger._emit_event({"evt": "exception", "idx": i, "error": str(e)})
                logger._tail(f"[d{i}] EXCEPTION: {e}")
                consecutive_bails += 1
                if consecutive_bails >= 2 and i >= 1:
                    logger._tail(f"[run] 2 consecutive exceptions — stopping.")
                    break
                continue

            g = grade(result, decision)
            logger.decision_end(i, result, g)
            summary_rows.append({"idx": i, **asdict(decision) | {"game_state": None}, **g,
                                 "bailed": result.bailed, "n_turns": result.n_turns,
                                 "probed": result.probed,
                                 "tool_seq": result.tool_call_sequence,
                                 "tool_cache_stats": result.tool_cache_stats})

            if result.bailed:
                consecutive_bails += 1
                if consecutive_bails >= 2:
                    logger._tail(
                        f"[run] {consecutive_bails} consecutive bails — "
                        f"model not engaging with the hard gate. Stopping."
                    )
                    break
            else:
                consecutive_bails = 0

        wall = time.time() - run_start_s
        n_committed = sum(1 for r in summary_rows if r.get("final_play") is not None and not r.get("bailed"))
        n_probed = sum(1 for r in summary_rows if r.get("probed"))
        n_bot_match = sum(1 for r in summary_rows if r.get("matches_bot"))
        n_bailed = sum(1 for r in summary_rows if r.get("bailed"))
        deltas = [r["eq_delta_vs_bot"] for r in summary_rows
                  if r.get("eq_delta_vs_bot") is not None]
        mean_delta = sum(deltas) / len(deltas) if deltas else None
        summary = {
            "n_total": len(summary_rows),
            "n_committed": n_committed,
            "n_probed": n_probed,
            "n_bot_match": n_bot_match,
            "n_bailed": n_bailed,
            "mean_eq_delta": round(mean_delta, 2) if mean_delta is not None else None,
            "wall_s": round(wall, 1),
            "probe_rate": round(n_probed / len(summary_rows), 2) if summary_rows else 0.0,
        }
        if args.eager_tool_cache:
            cache_rows = [r.get("tool_cache_stats") or {} for r in summary_rows]
            summary["tool_cache"] = {
                "precomputed_count": sum(int(r.get("precomputed_count", 0)) for r in cache_rows),
                "hit_count": sum(int(r.get("hit_count", 0)) for r in cache_rows),
                "wasted_count": sum(int(r.get("wasted_count", 0)) for r in cache_rows),
                "precompute_wall_s": round(
                    sum(float(r.get("precompute_wall_s", 0.0)) for r in cache_rows), 3
                ),
                "saved_tool_wall_s": round(
                    sum(float(r.get("saved_tool_wall_s", 0.0)) for r in cache_rows), 3
                ),
            }
        logger.run_end(summary)
        print(f"[pilot] done: {summary}", file=sys.stderr)

        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        with (run_dir / "decisions.jsonl").open("w") as f:
            for r in summary_rows:
                f.write(json.dumps(r, default=str) + "\n")

    finally:
        logger.close()
        if modal_app_ctx is not None:
            modal_app_ctx.__exit__(None, None, None)

    return 0


if __name__ == "__main__":
    sys.exit(main())
