"""Batched STaR rollout driver — N decisions through the tool loop in lockstep.

Sibling to ``run_move4_star_rollout.py``. The non-batched rollout is
async-sequential — ``asyncio.gather`` with ``concurrency`` per-decision
threads against a Modal endpoint. This driver is single-process but
runs the whole active set through one fused ``batch_generate`` call per
turn, which (per ``wiki/experiments/batch-throughput-bench.md``) wins
14-16x aggregate throughput on M5 Max over single-stream at the
recommended batch=64.

The tool-loop semantics come from ``burl/harness/tool_loop_native.py``
(``NativeHarness.run``) — that file is the source of truth. We do NOT
re-derive the parser; we import ``parse_native_completion`` directly.
Everything this file does per turn is a parallel replay of one iteration
of ``NativeHarness.run``'s inner step:

  1. Assemble messages (system + user + prior assistant/tool turns,
     plus any rejection "engine.commit" tool message).
  2. ``step_batch(active)`` fires one batched generate for all non-done
     decisions.
  3. Per-decision: parse completion -> append assistant msg -> dispatch
     each tool call (per-decision, CPU-bound) and append role="tool"
     messages -> check commit legality -> set done / rejection / final_play
     accordingly.
  4. Repeat until all decisions done or max_turns hit.

Output format mirrors ``run_move4_star_rollout.py``:

  * ``<out_dir>/rollout_traces.jsonl`` — one ``BurlTrace`` JSON per line.
  * ``<corpus>_stats.json`` — summary stats (fields match the sibling).

We deliberately skip the EQ-gate / corpus composition phases — those are
additive and belong in a follow-up if we want corpus-building parity.
This file's job is rollout throughput.

Usage::

    PYTHONPATH=. python -u -m burl.eval.run_move4_star_rollout_batched \\
        --dataset burl/eval/data/move4_decisions_n50.jsonl \\
        --out-dir burl/eval/results/move4_batched \\
        --corpus burl/data/batched_smoke.jsonl \\
        --n 16 --batch-size 16
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from burl.eval.decision_dataset import BurlDecision, load_dataset
from burl.eval.run_move3 import _build_record, _state_key
from burl.harness.agent_runner import (
    _current_player,
    _visible_history,
    build_tool_registry,
)
from burl.harness.agent_runner_native import (
    _COMMIT_INSTRUCTION,
    build_tool_schemas,
    render_native_messages,
)
from burl.harness.tool_loop import ToolProtocol
from burl.harness.tool_loop_native import parse_native_completion
from burl.harness.trace import BurlTrace, ToolCall, TurnStep
from burl.tools import engine as engine_tools

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Per-decision state                                                           #
# --------------------------------------------------------------------------- #


@dataclass
class _DecisionState:
    """Mutable per-decision scratchpad driven by the lockstep loop.

    One of these per dataset row. ``run_batch_decisions`` keeps a list of
    them; each outer iteration does one ``step_batch`` across the not-done
    entries and mutates these in place.
    """

    decision: BurlDecision
    messages: list[dict]
    tools: dict[str, ToolProtocol]
    tool_schemas: list[dict]
    trace: BurlTrace
    done: bool = False
    retry_exhausted: bool = False
    turns_used: int = 0
    pending_rejection: str | None = None
    # Incremental retries count is lifted into the trace; pending_rejection
    # is just the next-turn's input signal.


# --------------------------------------------------------------------------- #
# Helpers — mirrored from ``NativeHarness`` + ``run_star_rollout``             #
# --------------------------------------------------------------------------- #


def _json_default(obj: Any) -> Any:
    if isinstance(obj, set):
        return sorted(obj)
    if isinstance(obj, tuple):
        return list(obj)
    raise TypeError(f"not JSON-serializable: {type(obj).__name__}")


def _messages_char_len(messages: list[dict]) -> int:
    """Char count of serialized messages, same shape NativeHarness records.

    Matches ``tokens_in`` in the single-stream harness (chars, not tokens).
    Preserves comparability between batched and sequential trace JSONs.
    """
    return len(json.dumps(messages, default=str))


def _init_decision_state(
    decision: BurlDecision,
    *,
    enable_rules_tools: bool,
    enable_primer: bool,
) -> _DecisionState:
    state = decision.game_state
    me_abs = _current_player(state)
    hand = [d for d in state.hands[me_abs] if d not in state.played]
    history = _visible_history(state)
    system, user = render_native_messages(
        state, hand, history,
        enable_rules_tools=enable_rules_tools,
        enable_primer=enable_primer,
    )
    tools = build_tool_registry(
        game_state_provider=lambda s=state: s,
        enable_rules_tools=enable_rules_tools,
    )
    tool_schemas = build_tool_schemas(enable_rules_tools=enable_rules_tools)
    trace = BurlTrace(
        game_state_key=_state_key(state),
        decision_prompt=f"[SYSTEM]\n{system}\n\n[USER]\n{user}",
    )
    messages: list[dict] = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    return _DecisionState(
        decision=decision,
        messages=messages,
        tools=tools,
        tool_schemas=tool_schemas,
        trace=trace,
    )


def _execute_tool(
    game_state: Any,
    tools: dict[str, ToolProtocol],
    name: str,
    args: dict,
) -> ToolCall:
    """Mirrors ``NativeHarness._execute``."""
    tool = tools.get(name)
    if tool is None:
        return ToolCall(
            tool_name=name, args=args, result=None, ok=False,
            error=f"unknown tool: {name}",
        )
    try:
        result = tool(game_state, **args)
        return ToolCall(tool_name=name, args=args, result=result, ok=True)
    except Exception as e:  # pragma: no cover - defensive
        return ToolCall(
            tool_name=name, args=args, result=None, ok=False, error=str(e),
        )


def _apply_step(
    st: _DecisionState,
    completion: str,
    *,
    max_turns: int,
    max_retries: int,
    commit_instruction: str,
) -> None:
    """One iteration of ``NativeHarness.run``'s inner step, applied to ``st``.

    Mutates ``st.messages``, ``st.trace``, ``st.done`` and
    ``st.pending_rejection``. This is the CPU-bound half of the loop
    (parse + tool dispatch + legality); the GPU half (generation) happens
    outside in ``step_batch``.
    """
    # Prepend any pending engine-rejection message BEFORE the assistant
    # appended its latest reply? No — the NativeHarness order is: inject
    # rejection as a tool message AT THE START of the next step (so it
    # appears before the next assistant turn). But we already generated
    # the assistant turn with messages that had the rejection injected by
    # ``_prepare_step_messages`` below. So here we just consume the
    # completion.
    thought, tool_specs, commit = parse_native_completion(completion)
    st.trace.tokens_out += len(completion)

    assistant_msg: dict[str, Any] = {
        "role": "assistant",
        "content": completion,
    }
    st.messages.append(assistant_msg)

    executed: list[ToolCall] = []
    for name, args in tool_specs:
        tc = _execute_tool(st.decision.game_state, st.tools, name, args)
        executed.append(tc)
        tool_body = (
            {"error": tc.error} if not tc.ok else {"result": tc.result}
        )
        st.messages.append({
            "role": "tool",
            "name": tc.tool_name,
            "content": json.dumps(
                tool_body, default=_json_default,
                sort_keys=True, separators=(",", ":"),
            ),
        })

    # Nudge toward commit_play if the turn produced neither real tools
    # nor a commit (rare — malformed envelope or empty completion).
    if commit is None and not tool_specs:
        st.messages.append({
            "role": "user",
            "content": commit_instruction,
        })

    turn = TurnStep(
        thought=thought,
        tool_calls=executed,
        committed_play=commit,
        raw_completion=completion,
    )
    st.trace.turns.append(turn)
    st.turns_used += 1

    # Commit legality check — mirrors ``retry_on_illegal`` but inlined so
    # we don't interleave threads with the shared step loop.
    if commit is not None:
        ok, reason = engine_tools.is_legal(
            st.decision.game_state, int(commit),
        )
        if ok:
            st.trace.final_play = int(commit)
            st.done = True
            return
        # Illegal: record rejection, queue it for next turn's prompt.
        turn.engine_rejection = reason
        st.trace.n_retries += 1
        st.pending_rejection = reason
        if st.trace.n_retries > max_retries:
            # Matches RetryExhausted semantics from the single-stream
            # path: trace is retained, final_play stays -1, metadata flags.
            st.done = True
            st.retry_exhausted = True
            st.trace.metadata["retry_exhausted"] = True
            st.trace.metadata["error"] = (
                f"no legal play after {max_retries} retries "
                f"(state={st.trace.game_state_key}, "
                f"turns={len(st.trace.turns)})"
            )
            return
    else:
        # No commit this turn — rejection cleared.
        st.pending_rejection = None

    # Out of turns?
    if st.turns_used >= max_turns:
        st.done = True
        st.retry_exhausted = True
        st.trace.metadata["retry_exhausted"] = True
        st.trace.metadata["error"] = (
            f"max_turns={max_turns} exceeded "
            f"(state={st.trace.game_state_key}, "
            f"turns={len(st.trace.turns)})"
        )


def _prepare_step_messages(st: _DecisionState) -> list[dict]:
    """Return the messages list to feed this step's batch_generate.

    If a rejection is pending from the previous turn's illegal commit,
    append a synthetic ``role="tool"`` message named ``engine.commit``
    before we ask for the next assistant turn. This matches the wire
    format ``NativeHarness.run`` produces for the single-stream path.
    """
    if st.pending_rejection is not None:
        st.messages.append({
            "role": "tool",
            "name": "engine.commit",
            "content": json.dumps(
                {"error": f"illegal: {st.pending_rejection}"},
                separators=(",", ":"),
            ),
        })
        st.pending_rejection = None
    return st.messages


# --------------------------------------------------------------------------- #
# Batched driver                                                               #
# --------------------------------------------------------------------------- #


def run_batch_decisions(
    dataset: list[BurlDecision],
    model: Any,  # GemmaLocalNativeBatched — duck-typed so tests can stub
    *,
    max_turns: int,
    max_retries: int,
    batch_size: int,
    enable_rules_tools: bool,
    enable_primer: bool,
    commit_instruction: str = _COMMIT_INSTRUCTION,
    progress: bool = True,
) -> list[tuple[BurlTrace, bool, float]]:
    """Drive N decisions through the tool loop in lockstep.

    ``batch_size`` is the maximum fused-generation width per step; when
    fewer decisions are active, we shrink the batch to what's live. The
    order of returned traces matches the input dataset order.

    Returns ``[(trace, retry_exhausted, wall_time_seconds), ...]``.

    ``batch_size`` caveat: if the dataset has more decisions than
    ``batch_size``, the loop processes them in waves of at most
    ``batch_size`` simultaneously active; each wave runs its own
    lockstep loop to completion before the next wave starts. This matches
    the per-step KV budget we measured the M5 Max at (batch=64 is the
    recommended default; batch=128 is the aggregate peak with higher per-
    decision latency).
    """
    t0_total = time.time()
    results: list[tuple[BurlTrace, bool, float]] = [None] * len(dataset)  # type: ignore[list-item]

    # Initialize all per-decision state up front so the wave loop is the
    # only place mutation happens. Cheap (~ms per decision).
    states: list[_DecisionState] = [
        _init_decision_state(
            d,
            enable_rules_tools=enable_rules_tools,
            enable_primer=enable_primer,
        )
        for d in dataset
    ]

    for wave_start in range(0, len(states), batch_size):
        wave = states[wave_start : wave_start + batch_size]
        wave_t0 = time.time()
        if progress:
            print(
                f"[batch] wave {wave_start}..{wave_start + len(wave) - 1} "
                f"(size={len(wave)}) starting",
                flush=True,
            )
        step_idx = 0
        while any(not st.done for st in wave):
            step_idx += 1

            # Build per-decision message lists to pass into the batched
            # generator. This handles pending rejections (appends the
            # engine.commit tool message before the next assistant turn).
            active_payload: list[dict] = []
            for st in wave:
                if st.done:
                    continue
                msgs = _prepare_step_messages(st)
                # Mirror tokens_in accounting from NativeHarness.run.
                st.trace.tokens_in += _messages_char_len(msgs)
                active_payload.append({
                    "messages": msgs,
                    "tools": st.tool_schemas,
                    "done": False,
                })

            step_t0 = time.time()
            completions = model.step_batch(active_payload)
            step_wall = time.time() - step_t0

            if progress:
                n_active = len(active_payload)
                total_chars = sum(len(c) for c in completions)
                print(
                    f"[batch]   step={step_idx} active={n_active} "
                    f"wall={step_wall:.1f}s gen_chars={total_chars}",
                    flush=True,
                )

            # Pair completions back onto the still-active states.
            active_states = [st for st in wave if not st.done]
            assert len(active_states) == len(completions), (
                f"active={len(active_states)} completions={len(completions)}"
            )
            for st, completion in zip(active_states, completions):
                _apply_step(
                    st,
                    completion,
                    max_turns=max_turns,
                    max_retries=max_retries,
                    commit_instruction=commit_instruction,
                )

        wave_wall = time.time() - wave_t0
        if progress:
            n_legal = sum(
                1 for st in wave
                if st.trace.final_play >= 0 and not st.retry_exhausted
            )
            print(
                f"[batch] wave done wall={wave_wall:.1f}s "
                f"final_plays={n_legal}/{len(wave)}",
                flush=True,
            )

        # Fill in results for this wave. Per-decision wall is approximate
        # (same wave wall for all members) — single-stream comparison uses
        # the total anyway; this keeps the record shape intact.
        per_decision_wall = wave_wall / max(1, len(wave))
        for i, st in enumerate(wave):
            results[wave_start + i] = (st.trace, st.retry_exhausted, per_decision_wall)

    assert all(r is not None for r in results), "missed a decision slot"
    if progress:
        total_wall = time.time() - t0_total
        n_final = sum(
            1 for r in results
            if r[0].final_play >= 0 and not r[1]
        )
        print(
            f"[batch] ALL DONE wall={total_wall:.1f}s "
            f"final_plays={n_final}/{len(results)}",
            flush=True,
        )
    return results


# --------------------------------------------------------------------------- #
# Classification + stats — mirrors run_move4_star_rollout                      #
# --------------------------------------------------------------------------- #


def _classify(record: dict[str, Any]) -> str:
    """Same taxonomy as ``run_move4_star_rollout._classify``."""
    if record["retry_exhausted"]:
        return "exhausted"
    if record["final_play"] < 0 or not record["final_play_legal"]:
        return "illegal"
    if record["burl_eq"] is None:
        return "illegal"
    if record["burl_eq"] >= record["bot_eq"]:
        return "win"
    return "legal_loss"


# --------------------------------------------------------------------------- #
# Orchestrator                                                                 #
# --------------------------------------------------------------------------- #


def run_batched_rollout(
    *,
    dataset_path: Path,
    out_dir: Path,
    corpus_path: Path,
    n_decisions: int | None,
    max_turns: int,
    max_retries: int,
    batch_size: int,
    enable_rules_tools: bool,
    enable_primer: bool,
    adapter_path: str | None,
    model_repo: str,
    max_tokens: int,
    temperature: float,
) -> dict[str, Any]:
    from burl.modal.gemma_local_batched import GemmaLocalNativeBatched

    out_dir.mkdir(parents=True, exist_ok=True)
    corpus_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(dataset_path)
    if n_decisions is not None:
        dataset = dataset[:n_decisions]
    print(f"[batched] dataset={dataset_path} n={len(dataset)}")
    print(f"[batched] out_dir={out_dir}")
    print(f"[batched] corpus={corpus_path}")
    print(
        f"[batched] batch_size={batch_size} max_turns={max_turns} "
        f"max_retries={max_retries} enable_rules_tools={enable_rules_tools} "
        f"enable_primer={enable_primer} adapter_path={adapter_path}"
    )

    wall_start = time.time()
    print(f"[batched] loading model {model_repo} ...")
    model = GemmaLocalNativeBatched(
        model_repo=model_repo,
        adapter_path=adapter_path,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    load_wall = time.time() - wall_start

    rollout_start = time.time()
    results = run_batch_decisions(
        dataset,
        model,
        max_turns=max_turns,
        max_retries=max_retries,
        batch_size=batch_size,
        enable_rules_tools=enable_rules_tools,
        enable_primer=enable_primer,
    )
    rollout_wall = time.time() - rollout_start

    records: list[dict[str, Any]] = []
    rollout_traces_path = out_dir / "rollout_traces.jsonl"
    with rollout_traces_path.open("w") as f:
        for decision, (trace, exhausted, per_decision_wall) in zip(dataset, results):
            record = _build_record(decision, trace, exhausted, per_decision_wall)
            record["category"] = _classify(record)
            records.append(record)
            f.write(trace.to_json() + "\n")

    # -------------------- stats -------------------- #
    cat_hist = Counter(r["category"] for r in records)
    tool_hist: Counter[str] = Counter()
    for r in records:
        for name in r["tool_names"]:
            tool_hist[name] += 1
    decl_hist = Counter(r["trace"].metadata["declaration"] for r in records)

    rules_tool_names = {
        "count_dominoes_remaining",
        "trick_winner_if",
        "what_beats_what",
        "contract_progress",
    }
    rules_tool_hist = {
        n: c for n, c in tool_hist.items() if n in rules_tool_names
    }

    total_wall = time.time() - wall_start

    stats = {
        "dataset_path": str(dataset_path),
        "out_dir": str(out_dir),
        "corpus_path": str(corpus_path),
        "generated": datetime.now().isoformat(timespec="seconds"),
        "backend": "mlx-lm-batched",
        "model_repo": model_repo,
        "adapter_path": adapter_path,
        "batch_size": batch_size,
        "max_turns": max_turns,
        "max_retries": max_retries,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "enable_rules_tools": bool(enable_rules_tools),
        "enable_primer": bool(enable_primer),
        "n_decisions_attempted": len(records),
        "n_wins": cat_hist.get("win", 0),
        "n_legal_losses": cat_hist.get("legal_loss", 0),
        "n_illegal": cat_hist.get("illegal", 0),
        "n_exhausted": cat_hist.get("exhausted", 0),
        "tool_histogram_rollouts": dict(tool_hist),
        "rules_tool_histogram_rollouts": rules_tool_hist,
        "declaration_coverage_rollouts": {
            str(k): v for k, v in sorted(decl_hist.items())
        },
        "wall_time_seconds": total_wall,
        "model_load_seconds": load_wall,
        "rollout_wall_seconds": rollout_wall,
    }

    stats_path = corpus_path.with_name(corpus_path.stem + "_stats.json")
    stats_path.write_text(
        json.dumps(stats, indent=2, default=_json_default) + "\n"
    )
    print(f"[batched] stats -> {stats_path}")

    # Write a minimal corpus file so the --corpus path is always populated
    # (we don't EQ-gate here; only wins go in, mirroring the sibling's
    # rollout-win source). Keeping this round-tripped with the sibling's
    # composer keeps downstream consumers unsurprised.
    from burl.eval.run_move4_star_rollout import compose_sft_record
    corpus_entries: list[dict[str, Any]] = []
    for rec in records:
        if rec["category"] != "win":
            continue
        decision = next(
            d for d in dataset
            if int(d.seed) == rec["trace"].metadata["seed"]
            and int(d.declaration) == rec["trace"].metadata["declaration"]
            and int(d.narrator_seat) == rec["trace"].metadata["narrator_seat"]
        )
        corpus_entries.append(
            compose_sft_record(
                decision, rec["trace"], source="rollout_win",
                enable_rules_tools=enable_rules_tools,
                enable_primer=enable_primer,
            )
        )
    with corpus_path.open("w") as f:
        for entry in corpus_entries:
            f.write(json.dumps(entry, default=_json_default) + "\n")

    print()
    print("=" * 68)
    print(f"Batched rollout — N={len(records)} batch={batch_size}")
    print("=" * 68)
    print(f"  wins (K1 pass)              : {stats['n_wins']}")
    print(f"  legal losses                : {stats['n_legal_losses']}")
    print(f"  illegal / missing E[Q]      : {stats['n_illegal']}")
    print(f"  retry-exhausted             : {stats['n_exhausted']}")
    print(f"  corpus size (wins only)     : {len(corpus_entries)}")
    print(f"  model load                  : {load_wall:.1f}s")
    print(f"  rollout wall                : {rollout_wall:.1f}s")
    print(f"  total wall                  : {total_wall:.1f}s")
    print(f"  tool histogram              : {dict(tool_hist)}")
    if enable_rules_tools:
        print(f"  rules-tool histogram        : {rules_tool_hist}")
    print("=" * 68)

    return stats


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #


def _main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset", type=Path,
        default=Path("burl/eval/data/move4_decisions_n50.jsonl"),
    )
    parser.add_argument(
        "--out-dir", type=Path,
        default=Path("burl/eval/results/move4_batched"),
    )
    parser.add_argument(
        "--corpus", type=Path,
        default=Path("burl/data/batched_rollout_corpus.jsonl"),
    )
    parser.add_argument("--n", type=int, default=None, dest="n_decisions")
    parser.add_argument("--max-turns", type=int, default=8)
    parser.add_argument("--max-retries", type=int, default=7)
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help=(
            "Maximum fused-generation width per step. Default 64 "
            "(90%% of peak throughput on M5 Max with half the wall-per-cycle "
            "of batch=128; see wiki/experiments/batch-throughput-bench.md)."
        ),
    )
    parser.add_argument(
        "--enable-rules-tools",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "iter-3-rules prompt shape: include the four rules-as-tools "
            "schemas and the compact preamble. Default True (matches the "
            "shape the bench validated and current winning adapter)."
        ),
    )
    parser.add_argument(
        "--no-primer", action="store_true",
        help=(
            "Drop the trimmed primer (spike-v2 shape). Incompatible with "
            "--enable-rules-tools (rules-as-tools IS a primer)."
        ),
    )
    parser.add_argument(
        "--adapter-path", type=str, default=None,
        help="Optional local MLX adapter directory.",
    )
    parser.add_argument(
        "--model-repo", type=str,
        default="mlx-community/gemma-4-e2b-it-bf16",
    )
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.6)
    args = parser.parse_args()

    if args.no_primer and args.enable_rules_tools:
        parser.error(
            "--no-primer and --enable-rules-tools are incompatible "
            "(rules-as-tools IS a primer). Pass --no-primer with "
            "--no-enable-rules-tools."
        )

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    run_batched_rollout(
        dataset_path=args.dataset,
        out_dir=args.out_dir,
        corpus_path=args.corpus,
        n_decisions=args.n_decisions,
        max_turns=args.max_turns,
        max_retries=args.max_retries,
        batch_size=args.batch_size,
        enable_rules_tools=args.enable_rules_tools,
        enable_primer=not args.no_primer,
        adapter_path=args.adapter_path,
        model_repo=args.model_repo,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )


if __name__ == "__main__":
    _main()
