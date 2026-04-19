"""Move 4 spike grader — Gemma 4 native tool-use path.

Mirror of ``run_move3.py`` but routes through ``agent_runner_native`` +
``gemma_serve_native``. Writes results to
``burl/eval/results/move4_spike_<timestamp>/``.

Usage:
    python -u -m burl.eval.run_move4_spike \\
        --dataset burl/eval/data/move3_decisions.jsonl \\
        --model-source stub --n 2                  # plumbing check

    python -u -m burl.eval.run_move4_spike \\
        --dataset burl/eval/data/move3_decisions.jsonl \\
        --model-source modal --n 10                # real endpoint
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from burl.eval.decision_dataset import BurlDecision, load_dataset
from burl.eval.run_move3 import (
    L4_USD_PER_HOUR,
    Move3Report,
    _build_record,
    _compute_metrics,
    _completed_keys,
    _decision_key,
    _format_table,
    _pick_traces,
    _pretty_trace,
    _write_summary,
)
from burl.harness.agent_runner import _state_key
from burl.harness.agent_runner_native import run_decision_native
from burl.harness.retry import RetryExhausted
from burl.harness.tool_loop_native import NativeModelCallable
from burl.harness.trace import BurlTrace


# --------------------------------------------------------------------------- #
# Model adapters                                                               #
# --------------------------------------------------------------------------- #


def _make_stub_native_model(decision: BurlDecision) -> NativeModelCallable:
    """Scripted stub: one native tool call, then commit first legal play via
    the native ``commit_play`` tool."""
    target = int(decision.legal_plays[0])
    commit_call = f'<|tool_call>call:commit_play{{domino_id:{target}}}<tool_call|>'
    script = iter([
        '<|tool_call>{"name":"trump_declared","arguments":{}}<tool_call|>',
        commit_call,
    ])

    def call(_messages: list[dict], _tools: list[dict]) -> str:
        try:
            return next(script)
        except StopIteration:
            return commit_call

    return call


def _make_modal_native_model(
    server: Any,
    max_tokens: int = 2048,
    adapter_name: str | None = None,
) -> NativeModelCallable:
    """Remote-Gemma native adapter. Caller holds ``server`` inside ``app.run()``."""
    def call(messages: list[dict], tools: list[dict]) -> str:
        result = server.generate_native.remote(
            messages, tools=tools, max_tokens=max_tokens,
            temperature=0.6, enable_thinking=False,
            adapter_name=adapter_name,
        )
        # The server returns {"text", "prompt_text", "n_tokens"}; we only need text.
        return result["text"]
    return call


# --------------------------------------------------------------------------- #
# Loop                                                                         #
# --------------------------------------------------------------------------- #


def _loop(
    dataset: list[BurlDecision],
    model_source: str,
    model_fn_factory: Callable[[BurlDecision], NativeModelCallable],
    out_dir: Path,
    max_turns: int,
    max_retries: int,
    cost_cap_usd: float | None,
) -> list[dict[str, Any]]:
    traces_path = out_dir / "traces.jsonl"
    already_done = _completed_keys(traces_path)
    records: list[dict[str, Any]] = []
    wall_start = time.time()

    mode = "a" if already_done else "w"
    with traces_path.open(mode) as trace_out:
        for i, decision in enumerate(dataset):
            key = _decision_key(decision)
            if key in already_done:
                print(f"[skip] ({i+1}/{len(dataset)}) seed={key[0]} decl={key[1]} seat={key[2]} already done")
                continue

            print(
                f"[run ] ({i+1}/{len(dataset)}) seed={key[0]} decl={key[1]} "
                f"seat={key[2]} bot_play={decision.bot_play} eq_gap={decision.eq_gap:.2f}",
                flush=True,
            )
            model_fn = model_fn_factory(decision)
            t0 = time.time()
            retry_exhausted = False
            try:
                trace = run_decision_native(
                    decision.game_state,
                    model_fn,
                    max_turns=max_turns,
                    max_retries=max_retries,
                )
            except RetryExhausted as exc:
                retry_exhausted = True
                trace = exc.trace if exc.trace is not None else BurlTrace(
                    game_state_key=_state_key(decision.game_state),
                    decision_prompt="",
                )
                trace.metadata["retry_exhausted"] = True
                trace.metadata["error"] = str(exc)
                print(f"       RETRY-EXHAUSTED: {exc}", flush=True)
            except Exception as e:  # pragma: no cover - defensive
                retry_exhausted = True
                trace = BurlTrace(
                    game_state_key=_state_key(decision.game_state),
                    decision_prompt="",
                )
                trace.metadata["error"] = f"{type(e).__name__}: {e}"
                print(f"       ERROR: {type(e).__name__}: {e}", flush=True)

            elapsed = time.time() - t0
            record = _build_record(decision, trace, retry_exhausted, elapsed)
            records.append(record)

            trace_out.write(trace.to_json() + "\n")
            trace_out.flush()

            burl_eq_s = (
                f"{record['burl_eq']:.2f}" if record["burl_eq"] is not None else "N/A"
            )
            print(
                f"       final={record['final_play']} legal={record['final_play_legal']} "
                f"retries={record['n_retries']} tools={record['n_tool_calls']} "
                f"burl_eq={burl_eq_s} bot_eq={record['bot_eq']:.2f} "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )

            if cost_cap_usd is not None and model_source == "modal":
                wall_so_far = time.time() - wall_start
                est_so_far = wall_so_far / 3600.0 * L4_USD_PER_HOUR
                if est_so_far > cost_cap_usd:
                    print(
                        f"[STOP] estimated cost ${est_so_far:.2f} "
                        f"exceeded cap ${cost_cap_usd:.2f} after {i+1} decisions",
                        flush=True,
                    )
                    break

    return records


# --------------------------------------------------------------------------- #
# Public entrypoint                                                            #
# --------------------------------------------------------------------------- #


def run_move4_spike(
    dataset_path: Path,
    model_source: str = "modal",
    out_dir: Path | None = None,
    n_decisions: int | None = None,
    seed_offset: int = 0,
    max_turns: int = 8,
    max_retries: int = 3,
    cost_cap_usd: float | None = 0.50,
    adapter_name: str | None = None,
) -> Move3Report:
    dataset_path = Path(dataset_path)
    if out_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path(f"burl/eval/results/move4_spike_{ts}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(dataset_path)
    if seed_offset:
        dataset = dataset[seed_offset:]
    if n_decisions is not None:
        dataset = dataset[:n_decisions]

    print(f"[move4-spike] dataset={dataset_path} n={len(dataset)} source={model_source}")
    print(f"[move4-spike] out_dir={out_dir}")
    if adapter_name:
        print(f"[move4-spike] adapter_name={adapter_name}")

    wall_start = time.time()
    if model_source == "stub":
        records = _loop(
            dataset, model_source, _make_stub_native_model,
            out_dir, max_turns, max_retries, cost_cap_usd=None,
        )
    elif model_source == "modal":
        from burl.modal.gemma_serve_native import GemmaServerNative, app as gemma_app
        print("[move4-spike] opening modal app context (ephemeral)...")
        with gemma_app.run():
            server = GemmaServerNative()
            factory = lambda _dec: _make_modal_native_model(  # noqa: E731
                server, adapter_name=adapter_name,
            )
            records = _loop(
                dataset, model_source, factory,
                out_dir, max_turns, max_retries, cost_cap_usd=cost_cap_usd,
            )
    else:
        raise ValueError(f"unknown model_source: {model_source!r}")
    wall_time = time.time() - wall_start

    report = _compute_metrics(
        records,
        wall_time=wall_time,
        model_source=model_source,
        dataset_path=str(dataset_path),
        out_dir=str(out_dir),
    )

    _write_summary(out_dir, report)
    _write_markdown(out_dir, report, records)
    _print_table(report)
    return report


# --------------------------------------------------------------------------- #
# Output writers — mirror run_move3 but title the report "Move 4 Spike".       #
# --------------------------------------------------------------------------- #


def _write_markdown(
    out_dir: Path, report: Move3Report, records: list[dict[str, Any]],
) -> None:
    sampled = _pick_traces(records)
    tool_lines = "\n".join(
        f"- `{n}`: {c}"
        for n, c in sorted(report.tool_histogram.items(), key=lambda x: -x[1])
    ) or "- (none)"

    content = [
        "# Move 4 Spike — Native Gemma 4 Tool-Use",
        "",
        f"- dataset: `{report.dataset_path}`",
        f"- model_source: `{report.model_source}`",
        f"- out_dir: `{report.out_dir}`",
        f"- generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Grading Table",
        "",
        *_format_table(report),
        "",
        "## Tool-Use Histogram",
        "",
        tool_lines,
        "",
        "## Sampled Traces",
        "",
    ]
    if not sampled:
        content.append("_No traces to sample._")
    else:
        for label, trace in sampled:
            content.append(_pretty_trace(trace, label))
    (out_dir / "report.md").write_text("\n".join(content) + "\n")


def _print_table(report: Move3Report) -> None:
    print()
    print("=" * 60)
    print("Move 4 Spike Grading (native Gemma 4 tool-use)")
    print("=" * 60)
    for line in _format_table(report):
        print(line)
    print(f"tool histogram: {report.tool_histogram}")
    print("=" * 60)


# --------------------------------------------------------------------------- #
# Debug-one: single decision, full raw dump.                                   #
# --------------------------------------------------------------------------- #


def debug_one(
    dataset_path: Path,
    model_source: str = "modal",
    seed_offset: int = 0,
    max_turns: int = 8,
    max_retries: int = 3,
) -> None:
    dataset_path = Path(dataset_path)
    dataset = load_dataset(dataset_path)
    if seed_offset >= len(dataset):
        raise IndexError(
            f"seed_offset={seed_offset} out of range for dataset of {len(dataset)} decisions"
        )
    decision = dataset[seed_offset]

    print(f"[debug-one-native] dataset={dataset_path} source={model_source}")
    print(
        f"[debug-one-native] decision[{seed_offset}]: seed={decision.seed} "
        f"decl={decision.declaration} narrator={decision.narrator_seat} "
        f"trick={decision.trick_idx}"
    )
    print(f"[debug-one-native] legal_plays={decision.legal_plays}")
    print(f"[debug-one-native] per_play_eq={decision.per_play_eq}")
    print(f"[debug-one-native] bot_play={decision.bot_play} bot_eq={decision.bot_eq:.3f}")
    print("=" * 72)

    def _run(model_fn: NativeModelCallable) -> tuple[BurlTrace, bool, str | None]:
        t0 = time.time()
        retry_exhausted = False
        err: str | None = None
        try:
            trace = run_decision_native(
                decision.game_state, model_fn,
                max_turns=max_turns, max_retries=max_retries,
            )
        except RetryExhausted as exc:
            retry_exhausted = True
            err = str(exc)
            trace = exc.trace if exc.trace is not None else BurlTrace(
                game_state_key=_state_key(decision.game_state), decision_prompt="",
            )
        elapsed = time.time() - t0
        print(f"[debug-one-native] elapsed={elapsed:.1f}s retry_exhausted={retry_exhausted}")
        return trace, retry_exhausted, err

    if model_source == "stub":
        trace, retry_exhausted, err = _run(_make_stub_native_model(decision))
    elif model_source == "modal":
        from burl.modal.gemma_serve_native import GemmaServerNative, app as gemma_app
        print("[debug-one-native] opening modal app context (ephemeral)...")
        with gemma_app.run():
            server = GemmaServerNative()
            trace, retry_exhausted, err = _run(_make_modal_native_model(server))
    else:
        raise ValueError(f"unknown model_source: {model_source!r}")

    print()
    print("=" * 72)
    print("DECISION PROMPT (sent to model on turn 0)")
    print("=" * 72)
    print(trace.decision_prompt)
    print()
    print("=" * 72)
    print(f"TURNS ({len(trace.turns)})")
    print("=" * 72)
    for i, turn in enumerate(trace.turns):
        print(f"\n----- TURN {i} -----")
        print(f"committed_play: {turn.committed_play}")
        print(f"engine_rejection: {turn.engine_rejection}")
        print(f"parsed thought: {turn.thought!r}")
        print(f"tool_calls ({len(turn.tool_calls)}):")
        for tc in turn.tool_calls:
            flag = "OK" if tc.ok else f"ERR: {tc.error}"
            result_s = json.dumps(tc.result, default=str)
            if len(result_s) > 300:
                result_s = result_s[:297] + "..."
            print(f"  - {tc.tool_name}({tc.args}) [{flag}] -> {result_s}")
        print("raw_completion:")
        print("-" * 40)
        print(turn.raw_completion)
        print("-" * 40)
    print()
    print("=" * 72)
    print(
        f"SUMMARY: final_play={trace.final_play} n_retries={trace.n_retries} "
        f"chars_in={trace.tokens_in} chars_out={trace.tokens_out} "
        f"retry_exhausted={retry_exhausted}"
    )
    if err:
        print(f"error: {err}")
    print("=" * 72)


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #


def _main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset", type=Path,
        default=Path("burl/eval/data/move3_decisions.jsonl"),
    )
    parser.add_argument(
        "--model-source", choices=["stub", "modal"], default="modal",
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--n", type=int, default=None, dest="n_decisions")
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--max-turns", type=int, default=8)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--cost-cap-usd", type=float, default=0.50)
    parser.add_argument(
        "--adapter", type=str, default=None,
        help="LoRA adapter alias (e.g. 'burl-iter0'); resolves to "
             "jasonyandell/gemma-4-e2b-texas42-<adapter>",
    )
    parser.add_argument("--debug-one", action="store_true")
    args = parser.parse_args()

    if args.debug_one:
        debug_one(
            dataset_path=args.dataset,
            model_source=args.model_source,
            seed_offset=args.seed_offset,
            max_turns=args.max_turns,
            max_retries=args.max_retries,
        )
        return

    run_move4_spike(
        dataset_path=args.dataset,
        model_source=args.model_source,
        out_dir=args.out_dir,
        n_decisions=args.n_decisions,
        seed_offset=args.seed_offset,
        max_turns=args.max_turns,
        max_retries=args.max_retries,
        cost_cap_usd=args.cost_cap_usd,
        adapter_name=args.adapter,
    )


if __name__ == "__main__":
    _main()
