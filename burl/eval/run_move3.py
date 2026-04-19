"""Move 3 grader — orchestrate Burl rollouts on the held-out dataset.

Consumes the three Move 3 dep artifacts:
  * ``burl/eval/decision_dataset.py``  -> held-out ``BurlDecision`` list
  * ``burl/harness/agent_runner.py``   -> ``run_decision(state, model_callable)``
  * ``burl/modal/gemma_serve.py``      -> remote ``GemmaServer`` endpoint

Usage:
    python -u -m burl.eval.run_move3 \\
        --dataset burl/eval/data/move3_decisions.jsonl \\
        --model-source stub --n 3                   # plumbing check

    python -u -m burl.eval.run_move3 \\
        --dataset burl/eval/data/move3_decisions.jsonl \\
        --model-source modal --n 10                 # real endpoint

Produces ``burl/eval/results/move3_<timestamp>/{traces.jsonl,summary.json,report.md}``.

Resumable: if interrupted, re-running with the same ``--out-dir`` reads any
existing ``traces.jsonl`` and skips decisions already completed (keyed by
``(seed, declaration, narrator_seat)``).
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from burl.eval.decision_dataset import BurlDecision, load_dataset
from burl.harness.agent_runner import _state_key, run_decision
from burl.harness.retry import RetryExhausted
from burl.harness.trace import BurlTrace, TurnStep
from burl.tools import engine as engine_tools

ModelCallable = Callable[[str], str]

# L4 on-demand pricing (approximate, matches the gemma-server estimate).
L4_USD_PER_HOUR = 0.80


# --------------------------------------------------------------------------- #
# Report struct                                                                #
# --------------------------------------------------------------------------- #


@dataclass
class Move3Report:
    n_attempted: int
    n_completed: int          # produced a trace with final_play != -1
    n_retry_exhausted: int

    legal_rate: float
    first_legal_rate: float
    bot_match_rate: float
    mean_eq_delta: float
    p_eq_geq_bot: float
    mean_retry_count: float
    empty_tool_rollout_rate: float

    tool_histogram: dict[str, int]
    mean_tokens_in: float     # chars-based; see note on BurlTrace fields
    mean_tokens_out: float

    wall_time_seconds: float
    estimated_usd: float      # wall_time * L4_USD_PER_HOUR

    model_source: str
    dataset_path: str
    out_dir: str

    def to_json_obj(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Model adapters                                                               #
# --------------------------------------------------------------------------- #


def _make_stub_model(decision: BurlDecision) -> ModelCallable:
    """Scripted stub: one tool probe, then commit the first legal play.

    Proves the plumbing end-to-end without any network or GPU. The first turn
    exercises the tool-dispatch path; the second turn exercises the commit +
    legality path.
    """
    target = int(decision.legal_plays[0])
    completions = iter([
        '<think>Stub: probe declaration first.</think>'
        '<tool>{"name":"trump_declared","args":{}}</tool>',
        f'<think>Commit legal target {target}.</think><commit>{target}</commit>',
    ])

    def call(prompt: str) -> str:  # noqa: ARG001 -- stub ignores prompt
        try:
            return next(completions)
        except StopIteration:
            # Safety net: if the harness asks for more turns (shouldn't happen
            # on the happy path), keep re-committing the legal play.
            return f'<commit>{target}</commit>'

    return call


def _with_think_prefix(inner: ModelCallable) -> ModelCallable:
    """Prefix-prime: have the model decode starting inside `<think>`.

    Base Gemma 4 E2B ignores the XML protocol described in the system prompt
    and emits a markdown reasoning section ("thought\\n**Current State...**").
    Appending `<think>` to the prompt forces the decoder to continue inside
    the tag; prepending it to the completion lets `<think>(.*?)</think>` in
    `tool_loop.parse_completion` pick up the emitted reasoning and any
    trailing `</think><tool>...</tool><commit>...</commit>` run.

    Does NOT mutate the trace's `decision_prompt` — the wrapper sits at the
    model-call boundary, so the training-corpus view of the prompt stays
    clean.
    """
    def call(prompt: str) -> str:
        primed = prompt + "\n<think>"
        completion = inner(primed)
        return "<think>" + completion
    return call


def _make_modal_model(server: Any, max_tokens: int = 2048) -> ModelCallable:
    """Remote-Gemma adapter. ``server`` must be a ``GemmaServer`` instance live
    inside an open ``app.run()`` context. Wraps the raw call with
    ``_with_think_prefix`` so Gemma's output lands inside the XML protocol.

    ``max_tokens=2048`` is deliberate, not the ``gemma_serve.py`` default of 512:
    per ``burl/GEMMA_4_ERGONOMICS.md``, Gemma 4 always emits a 200-500 token
    thinking channel, so 512 leaves nothing for the answer channel where our
    ``<tool>`` / ``<commit>`` tags live.
    """
    def raw_call(prompt: str) -> str:
        return server.generate.remote(
            prompt, max_tokens=max_tokens, stop=None, temperature=0.6,
        )
    return _with_think_prefix(raw_call)


# --------------------------------------------------------------------------- #
# Grading                                                                      #
# --------------------------------------------------------------------------- #


def _legal_on_engine(state: Any, dom: int) -> bool:
    if dom is None or dom < 0:
        return False
    ok, _reason = engine_tools.is_legal(state, int(dom))
    return bool(ok)


def _trace_tool_calls(trace: BurlTrace) -> list[Any]:
    return [tc for turn in trace.turns for tc in turn.tool_calls]


def _compute_metrics(
    records: list[dict[str, Any]],
    wall_time: float,
    model_source: str,
    dataset_path: str,
    out_dir: str,
) -> Move3Report:
    n_attempted = len(records)
    completed = [r for r in records if r["final_play"] >= 0 and not r["retry_exhausted"]]
    retry_exhausted = [r for r in records if r["retry_exhausted"]]

    n_completed = len(completed)
    n_retry = len(retry_exhausted)

    legal_rate = (
        sum(1 for r in completed if r["final_play_legal"]) / n_completed
        if n_completed else 0.0
    )
    first_legal_rate = (
        sum(1 for r in completed if r["n_retries"] == 0 and r["final_play_legal"]) / n_attempted
        if n_attempted else 0.0
    )
    bot_match_rate = (
        sum(1 for r in completed if r["final_play"] == r["bot_play"]) / n_completed
        if n_completed else 0.0
    )

    # E[Q] metrics only over completed rollouts with a legal play whose E[Q]
    # was measured in the dataset.
    eq_rows = [r for r in completed if r["burl_eq"] is not None]
    mean_eq_delta = (
        statistics.fmean(r["burl_eq"] - r["bot_eq"] for r in eq_rows)
        if eq_rows else 0.0
    )
    p_eq_geq_bot = (
        sum(1 for r in eq_rows if r["burl_eq"] >= r["bot_eq"]) / len(eq_rows)
        if eq_rows else 0.0
    )
    mean_retry_count = (
        statistics.fmean(r["n_retries"] for r in records) if records else 0.0
    )
    empty_tool_rate = (
        sum(1 for r in records if r["n_tool_calls"] == 0) / n_attempted
        if n_attempted else 0.0
    )

    tool_hist: Counter[str] = Counter()
    for r in records:
        for name in r["tool_names"]:
            tool_hist[name] += 1

    mean_tokens_in = (
        statistics.fmean(r["tokens_in"] for r in records) if records else 0.0
    )
    mean_tokens_out = (
        statistics.fmean(r["tokens_out"] for r in records) if records else 0.0
    )

    est_cost = wall_time / 3600.0 * L4_USD_PER_HOUR if model_source == "modal" else 0.0

    return Move3Report(
        n_attempted=n_attempted,
        n_completed=n_completed,
        n_retry_exhausted=n_retry,
        legal_rate=legal_rate,
        first_legal_rate=first_legal_rate,
        bot_match_rate=bot_match_rate,
        mean_eq_delta=mean_eq_delta,
        p_eq_geq_bot=p_eq_geq_bot,
        mean_retry_count=mean_retry_count,
        empty_tool_rollout_rate=empty_tool_rate,
        tool_histogram=dict(tool_hist),
        mean_tokens_in=mean_tokens_in,
        mean_tokens_out=mean_tokens_out,
        wall_time_seconds=wall_time,
        estimated_usd=est_cost,
        model_source=model_source,
        dataset_path=dataset_path,
        out_dir=out_dir,
    )


# --------------------------------------------------------------------------- #
# Resumability                                                                 #
# --------------------------------------------------------------------------- #


def _decision_key(d: BurlDecision) -> tuple[int, int, int]:
    return (int(d.seed), int(d.declaration), int(d.narrator_seat))


def _completed_keys(traces_path: Path) -> set[tuple[int, int, int]]:
    if not traces_path.exists():
        return set()
    out: set[tuple[int, int, int]] = set()
    with traces_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            md = entry.get("metadata", {})
            key = (int(md.get("seed", -1)), int(md.get("declaration", -1)), int(md.get("narrator_seat", -1)))
            if -1 not in key:
                out.add(key)
    return out


# --------------------------------------------------------------------------- #
# Trace sampling for the markdown report                                       #
# --------------------------------------------------------------------------- #


def _pretty_trace(trace_dict: dict[str, Any], label: str) -> str:
    md = trace_dict.get("metadata", {})
    lines = [
        f"### {label}",
        "",
        f"- seed={md.get('seed')} decl={md.get('declaration')} narrator={md.get('narrator_seat')}",
        f"- bot_play={md.get('bot_play')} bot_eq={md.get('bot_eq'):.3f}"
        if md.get('bot_eq') is not None else f"- bot_play={md.get('bot_play')}",
        f"- burl_final_play={trace_dict.get('final_play')} "
        f"burl_eq={md.get('burl_eq')}  eq_delta={md.get('eq_delta')}",
        f"- retries={trace_dict.get('n_retries')} tool_calls={sum(len(t.get('tool_calls',[])) for t in trace_dict.get('turns',[]))}",
        f"- tokens_in(chars)={trace_dict.get('tokens_in')}  tokens_out(chars)={trace_dict.get('tokens_out')}",
        "",
    ]
    for i, turn in enumerate(trace_dict.get("turns", [])):
        thought = (turn.get("thought") or "").strip()
        if thought:
            lines.append(f"    [turn {i}] <think>{thought[:500]}</think>")
        for tc in turn.get("tool_calls", []):
            result = tc.get("result")
            r_str = json.dumps(result, default=str)
            if len(r_str) > 140:
                r_str = r_str[:137] + "..."
            flag = "ok" if tc.get("ok") else f"ERR: {tc.get('error')}"
            lines.append(f"    [turn {i}] <tool {tc.get('tool_name')}({tc.get('args')}) -> {r_str}> [{flag}]")
        if turn.get("committed_play") is not None:
            rej = turn.get("engine_rejection")
            if rej:
                lines.append(f"    [turn {i}] <commit>{turn['committed_play']}</commit>  REJECTED: {rej}")
            else:
                lines.append(f"    [turn {i}] <commit>{turn['committed_play']}</commit>")
    lines.append("")
    return "\n".join(lines)


def _pick_traces(records: list[dict[str, Any]]) -> list[tuple[str, dict[str, Any]]]:
    picks: list[tuple[str, dict[str, Any]]] = []
    used_idx: set[int] = set()

    def _take(label: str, pred: Callable[[dict[str, Any]], bool]) -> None:
        for i, r in enumerate(records):
            if i in used_idx:
                continue
            if pred(r):
                picks.append((label, r["trace_dict"]))
                used_idx.add(i)
                return

    _take("Burl beats bot", lambda r: r["burl_eq"] is not None and r["burl_eq"] > r["bot_eq"])
    _take("Burl matches bot", lambda r: r["final_play"] == r["bot_play"] and not r["retry_exhausted"])
    _take("Burl loses to bot", lambda r: r["burl_eq"] is not None and r["burl_eq"] < r["bot_eq"])
    _take("Hit retry cap", lambda r: r["retry_exhausted"])
    _take("Random sample", lambda r: True)
    return picks


# --------------------------------------------------------------------------- #
# Core loop                                                                    #
# --------------------------------------------------------------------------- #


def _build_record(
    decision: BurlDecision,
    trace: BurlTrace,
    retry_exhausted: bool,
    wall_time: float,
) -> dict[str, Any]:
    final_play = int(trace.final_play)
    legal = _legal_on_engine(decision.game_state, final_play) if final_play >= 0 else False
    burl_eq: float | None = None
    eq_delta: float | None = None
    if final_play in decision.per_play_eq:
        burl_eq = float(decision.per_play_eq[final_play])
        eq_delta = burl_eq - float(decision.bot_eq)

    tool_calls = _trace_tool_calls(trace)
    tool_names = [tc.tool_name for tc in tool_calls]

    # Enrich metadata for downstream analysis.
    trace.metadata.update({
        "seed": int(decision.seed),
        "declaration": int(decision.declaration),
        "narrator_seat": int(decision.narrator_seat),
        "trick_idx": int(decision.trick_idx),
        "legal_plays": [int(x) for x in decision.legal_plays],
        "per_play_eq": {str(k): float(v) for k, v in decision.per_play_eq.items()},
        "bot_play": int(decision.bot_play),
        "bot_eq": float(decision.bot_eq),
        "eq_gap": float(decision.eq_gap),
        "burl_eq": burl_eq,
        "eq_delta": eq_delta,
        "retry_exhausted": bool(retry_exhausted),
        "wall_time_seconds": float(wall_time),
    })

    return {
        "trace": trace,
        "trace_dict": json.loads(trace.to_json()),
        "final_play": final_play,
        "final_play_legal": legal,
        "n_retries": int(trace.n_retries),
        "n_tool_calls": len(tool_calls),
        "tool_names": tool_names,
        "tokens_in": int(trace.tokens_in),
        "tokens_out": int(trace.tokens_out),
        "retry_exhausted": bool(retry_exhausted),
        "bot_play": int(decision.bot_play),
        "bot_eq": float(decision.bot_eq),
        "burl_eq": burl_eq,
        "eq_delta": eq_delta,
        "wall_time": wall_time,
    }


def _loop(
    dataset: list[BurlDecision],
    model_source: str,
    model_fn_factory: Callable[[BurlDecision], ModelCallable],
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
                trace = run_decision(
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

            burl_eq_s = f"{record['burl_eq']:.2f}" if record["burl_eq"] is not None else "N/A"
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


def run_move3(
    dataset_path: Path,
    model_source: str = "modal",
    out_dir: Path | None = None,
    n_decisions: int | None = None,
    seed_offset: int = 0,
    max_turns: int = 8,
    max_retries: int = 3,
    cost_cap_usd: float | None = 0.50,
) -> Move3Report:
    dataset_path = Path(dataset_path)
    if out_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path(f"burl/eval/results/move3_{ts}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(dataset_path)
    if seed_offset:
        dataset = dataset[seed_offset:]
    if n_decisions is not None:
        dataset = dataset[:n_decisions]

    print(f"[move3] dataset={dataset_path} n={len(dataset)} source={model_source}")
    print(f"[move3] out_dir={out_dir}")

    wall_start = time.time()
    if model_source == "stub":
        records = _loop(
            dataset,
            model_source,
            _make_stub_model,
            out_dir,
            max_turns,
            max_retries,
            cost_cap_usd=None,  # stub has no $ cost
        )
    elif model_source == "modal":
        from burl.modal.gemma_serve import GemmaServer, app as gemma_app
        print("[move3] opening modal app context (ephemeral)...")
        with gemma_app.run():
            server = GemmaServer()
            factory = lambda _dec: _make_modal_model(server)  # noqa: E731
            records = _loop(
                dataset,
                model_source,
                factory,
                out_dir,
                max_turns,
                max_retries,
                cost_cap_usd=cost_cap_usd,
            )
    elif model_source == "local-gguf":
        raise NotImplementedError(
            "local-gguf adapter not implemented; use 'stub' or 'modal'."
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
# Output writers                                                               #
# --------------------------------------------------------------------------- #


def _write_summary(out_dir: Path, report: Move3Report) -> None:
    (out_dir / "summary.json").write_text(json.dumps(report.to_json_obj(), indent=2) + "\n")


def _format_table(report: Move3Report) -> list[str]:
    pct = lambda x: f"{100*x:5.1f}%"  # noqa: E731
    lines = [
        "| metric | value |",
        "|---|---|",
        f"| n_attempted | {report.n_attempted} |",
        f"| n_completed | {report.n_completed} |",
        f"| n_retry_exhausted | {report.n_retry_exhausted} |",
        f"| legal_rate | {pct(report.legal_rate)} |",
        f"| first_legal_rate | {pct(report.first_legal_rate)} |",
        f"| bot_match_rate | {pct(report.bot_match_rate)} |",
        f"| mean_eq_delta | {report.mean_eq_delta:+.3f} |",
        f"| p_eq_geq_bot | {pct(report.p_eq_geq_bot)} |",
        f"| mean_retry_count | {report.mean_retry_count:.2f} |",
        f"| empty_tool_rollout_rate | {pct(report.empty_tool_rollout_rate)} |",
        f"| mean_tokens_in (chars) | {report.mean_tokens_in:.0f} |",
        f"| mean_tokens_out (chars) | {report.mean_tokens_out:.0f} |",
        f"| wall_time_seconds | {report.wall_time_seconds:.1f} |",
        f"| estimated_usd | ${report.estimated_usd:.2f} |",
    ]
    return lines


def _write_markdown(
    out_dir: Path, report: Move3Report, records: list[dict[str, Any]],
) -> None:
    sampled = _pick_traces(records)
    tool_lines = "\n".join(
        f"- `{n}`: {c}" for n, c in sorted(report.tool_histogram.items(), key=lambda x: -x[1])
    ) or "- (none)"

    content = [
        "# Move 3 Grading Report",
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
    print("Move 3 Grading")
    print("=" * 60)
    for line in _format_table(report):
        print(line)
    print(f"tool histogram: {report.tool_histogram}")
    print("=" * 60)


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #


def debug_one(
    dataset_path: Path,
    model_source: str = "modal",
    seed_offset: int = 0,
    max_turns: int = 8,
    max_retries: int = 3,
) -> None:
    """Run exactly one decision and dump every raw model completion verbatim.

    No grading, no jsonl, no report — pure evidence for diagnosing why the
    model fails on Move 3 rollouts. Use this before a full N-decision run
    whenever the failure mode is unclear.
    """
    dataset_path = Path(dataset_path)
    dataset = load_dataset(dataset_path)
    if seed_offset >= len(dataset):
        raise IndexError(
            f"seed_offset={seed_offset} out of range for dataset of {len(dataset)} decisions"
        )
    decision = dataset[seed_offset]

    print(f"[debug-one] dataset={dataset_path} source={model_source}")
    print(
        f"[debug-one] decision[{seed_offset}]: seed={decision.seed} "
        f"decl={decision.declaration} narrator={decision.narrator_seat} "
        f"trick={decision.trick_idx}"
    )
    print(f"[debug-one] legal_plays={decision.legal_plays}")
    print(f"[debug-one] per_play_eq={decision.per_play_eq}")
    print(f"[debug-one] bot_play={decision.bot_play} bot_eq={decision.bot_eq:.3f}")
    print(f"[debug-one] eq_gap={decision.eq_gap:.3f}")
    print("=" * 72)

    def _run(model_fn: ModelCallable) -> tuple[BurlTrace, bool, str | None]:
        t0 = time.time()
        retry_exhausted = False
        err: str | None = None
        try:
            trace = run_decision(
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
        print(f"[debug-one] elapsed={elapsed:.1f}s  retry_exhausted={retry_exhausted}")
        return trace, retry_exhausted, err

    if model_source == "stub":
        trace, retry_exhausted, err = _run(_make_stub_model(decision))
    elif model_source == "modal":
        from burl.modal.gemma_serve import GemmaServer, app as gemma_app
        print("[debug-one] opening modal app context (ephemeral)...")
        with gemma_app.run():
            server = GemmaServer()
            trace, retry_exhausted, err = _run(_make_modal_model(server))
    elif model_source == "local-gguf":
        raise NotImplementedError("local-gguf adapter not implemented")
    else:
        raise ValueError(f"unknown model_source: {model_source!r}")

    # Dump everything.
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


def _main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset", type=Path,
        default=Path("burl/eval/data/move3_decisions.jsonl"),
    )
    parser.add_argument(
        "--model-source", choices=["stub", "modal", "local-gguf"], default="modal",
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--n", type=int, default=None, dest="n_decisions",
                        help="Only run the first N decisions (after --seed-offset).")
    parser.add_argument("--seed-offset", type=int, default=0,
                        help="Skip the first K decisions of the dataset.")
    parser.add_argument("--max-turns", type=int, default=8)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--cost-cap-usd", type=float, default=0.50,
                        help="Abort modal runs when projected spend exceeds this.")
    parser.add_argument("--debug-one", action="store_true",
                        help="Run only decision[seed_offset] and print full trace + raw completions. "
                             "No grading, no artifacts. Diagnostic mode.")
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

    run_move3(
        dataset_path=args.dataset,
        model_source=args.model_source,
        out_dir=args.out_dir,
        n_decisions=args.n_decisions,
        seed_offset=args.seed_offset,
        max_turns=args.max_turns,
        max_retries=args.max_retries,
        cost_cap_usd=args.cost_cap_usd,
    )


if __name__ == "__main__":
    _main()
