"""Move 4 STaR corpus builder — rollout N=50 + EQ-gate the K1 losses.

Phase B of Move 4. Produces an SFT corpus for iter-2+. The flow:

  1. Rollout each decision via the native endpoint.
  2. K1 filter: ``burl_eq >= bot_eq`` -> keep the trace as ``rollout_win``.
  3. EQ-gate the losses (iter-2 replacement for the iter-0/iter-1
     "reveal-answer-and-rationalize" step which hit 100% yes-bias):
       * ``check_commit`` gates on legal-but-sub-optimal commits.
       * ``gate_feedback_prompt`` issues a nudge that never names the
         bot play or any per-play E[Q] number (see
         ``burl/harness/eq_gate.py`` + design doc).
       * Re-run the harness with the nudge appended as a ``role="user"``
         message (option (b) — extra_user_messages kwarg on
         ``NativeHarness.run``; see design doc §Integration).
       * Classify the attempt chain: ``self_corrected`` / ``forced_flip``
         / ``stubborn`` / ``exhausted``.
     Only ``self_corrected`` traces enter the corpus (tagged
     ``source="eq_gate_self_correct"``).
  4. Emit the corpus as HF chat-format pairs:
     ``{"messages": [{"role":"user", "content":<prompt>},
                     {"role":"assistant","content":<trace>}]}``.

Usage:
    python -u -m burl.eval.run_move4_star_rollout \\
        --dataset burl/eval/data/move4_decisions_n50.jsonl \\
        --out-dir burl/eval/results/move4_star_rollout \\
        --corpus burl/data/star_iter2_corpus.jsonl \\
        --gate-variant tool-nudge --max-gate-retries 1 --eq-epsilon 0.25
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from burl.eval.decision_dataset import BurlDecision, load_dataset
from burl.eval.run_move3 import L4_USD_PER_HOUR, _build_record, _state_key
from burl.harness.agent_runner import (
    _current_player,
    _visible_history,
    build_tool_registry,
)
from burl.harness.agent_runner_native import (
    _COMMIT_INSTRUCTION,
    build_tool_schemas,
    render_native_messages,
    run_decision_native,
)
from burl.harness.eq_gate import (
    AttemptSummary,
    check_commit,
    classify_rationalization,
    gate_feedback_prompt,
)
from burl.harness.retry import RetryExhausted
from burl.harness.tool_loop_native import NativeHarness, NativeModelCallable
from burl.harness.trace import BurlTrace
from burl.tools import engine as engine_tools


# --------------------------------------------------------------------------- #
# Model adapter                                                                #
# --------------------------------------------------------------------------- #


def _make_modal_native_model(server: Any, max_tokens: int = 2048) -> NativeModelCallable:
    def call(messages: list[dict], tools: list[dict]) -> str:
        result = server.generate_native.remote(
            messages,
            tools=tools,
            max_tokens=max_tokens,
            temperature=0.6,
            enable_thinking=False,
        )
        return result["text"]
    return call


# --------------------------------------------------------------------------- #
# Rollout                                                                      #
# --------------------------------------------------------------------------- #


def _run_one_rollout(
    decision: BurlDecision,
    model_fn: NativeModelCallable,
    max_turns: int,
    max_retries: int,
    enable_rules_tools: bool = False,
    enable_primer: bool = True,
) -> tuple[BurlTrace, bool]:
    """Wrap ``run_decision_native`` with the same defensive try/except the
    spike grader uses. Returns ``(trace, retry_exhausted)``."""
    retry_exhausted = False
    try:
        trace = run_decision_native(
            decision.game_state,
            model_fn,
            max_turns=max_turns,
            max_retries=max_retries,
            enable_rules_tools=enable_rules_tools,
            enable_primer=enable_primer,
        )
    except RetryExhausted as exc:
        retry_exhausted = True
        trace = exc.trace if exc.trace is not None else BurlTrace(
            game_state_key=_state_key(decision.game_state),
            decision_prompt="",
        )
        trace.metadata["retry_exhausted"] = True
        trace.metadata["error"] = str(exc)
    except Exception as exc:  # pragma: no cover - defensive
        retry_exhausted = True
        trace = BurlTrace(
            game_state_key=_state_key(decision.game_state),
            decision_prompt="",
        )
        trace.metadata["error"] = f"{type(exc).__name__}: {exc}"
    return trace, retry_exhausted


# --------------------------------------------------------------------------- #
# EQ-gate re-run                                                               #
# --------------------------------------------------------------------------- #
#
# Design note — option (b) chosen (see design doc §Integration). Instead of
# mutating ``system_content`` to reveal the ground-truth play (iter-0/iter-1
# yes-bias trap), we re-run the harness with the decision prompt UNCHANGED
# and append the gate's nudge as an additional ``role="user"`` message via
# ``NativeHarness.run(..., extra_user_messages=[nudge])``. That keeps
# ``trace.decision_prompt`` clean for STaR corpus use (the nudge is
# training-time scaffolding stripped at inference) and is a 3-line additive
# change to the harness.


def _gate_one(
    decision: BurlDecision,
    model_fn: NativeModelCallable,
    nudge: str,
    max_turns: int,
    max_retries: int,
    enable_rules_tools: bool = False,
    enable_primer: bool = True,
) -> tuple[BurlTrace, bool]:
    """Re-run Gemma with an EQ-gate nudge appended as a user message.

    The nudge never names ``bot_play`` or any per-play E[Q]. Returns the
    resulting trace plus a ``retry_exhausted`` flag.
    """
    me_abs = _current_player(decision.game_state)
    hand_remaining = [
        d for d in decision.game_state.hands[me_abs] if d not in decision.game_state.played
    ]
    history = _visible_history(decision.game_state)
    system_content, user_content = render_native_messages(
        decision.game_state, hand_remaining, history,
        enable_rules_tools=enable_rules_tools,
        enable_primer=enable_primer,
    )

    tools = build_tool_registry(
        game_state_provider=lambda: decision.game_state,
        enable_rules_tools=enable_rules_tools,
    )
    harness = NativeHarness(
        model_callable=model_fn,
        tools=tools,
        tool_schemas=build_tool_schemas(enable_rules_tools=enable_rules_tools),
        is_legal_fn=lambda s, d: engine_tools.is_legal(s, int(d)),
        commit_instruction=_COMMIT_INSTRUCTION,
        max_turns=max_turns,
        max_retries=max_retries,
    )

    retry_exhausted = False
    try:
        trace = harness.run(
            game_state=decision.game_state,
            system_content=system_content,
            user_content=user_content,
            state_key=_state_key(decision.game_state),
            extra_user_messages=[nudge],
        )
    except RetryExhausted as exc:
        retry_exhausted = True
        trace = exc.trace if exc.trace is not None else BurlTrace(
            game_state_key=_state_key(decision.game_state),
            decision_prompt="",
        )
        trace.metadata["retry_exhausted"] = True
        trace.metadata["error"] = str(exc)
    except Exception as exc:  # pragma: no cover
        retry_exhausted = True
        trace = BurlTrace(
            game_state_key=_state_key(decision.game_state),
            decision_prompt="",
        )
        trace.metadata["error"] = f"{type(exc).__name__}: {exc}"

    trace.metadata["eq_gate"] = True
    return trace, retry_exhausted


# --------------------------------------------------------------------------- #
# Corpus composition                                                           #
# --------------------------------------------------------------------------- #


def compose_assistant_content(trace: BurlTrace) -> str:
    """Flatten a BurlTrace into one assistant-content string.

    Per team-lead spec, iter-0 collapses the multi-turn trace into a single
    assistant message. For each turn we emit the raw completion (which
    already carries Gemma's native ``<|tool_call>…<tool_call|>`` envelopes)
    followed by a synthesized ``<|tool_response>…<tool_response|>`` line per
    executed tool call. This shape matches how Gemma renders tool chatter
    during generation, so later SFT can parse it back without surgery.
    """
    parts: list[str] = []
    for turn in trace.turns:
        parts.append(turn.raw_completion)
        for tc in turn.tool_calls:
            body = {"result": tc.result} if tc.ok else {"error": tc.error}
            parts.append(
                f"<|tool_response>"
                f"{json.dumps(body, default=_json_default, separators=(',', ':'))}"
                f"<tool_response|>"
            )
    return "\n".join(parts).strip()


def _json_default(obj: Any) -> Any:
    if isinstance(obj, set):
        return sorted(obj)
    if isinstance(obj, tuple):
        return list(obj)
    raise TypeError(f"not JSON-serializable: {type(obj).__name__}")


def compose_sft_record(
    decision: BurlDecision,
    trace: BurlTrace,
    source: str,
    *,
    enable_rules_tools: bool = False,
    enable_primer: bool = True,
) -> dict[str, Any]:
    """Build an HF-chat-format record plus provenance metadata.

    The corpus prompt shape MUST match the rollout prompt shape — if the
    rollout ran with ``enable_primer=False`` (spike-v2 / iter-3-v2), the
    record's ``user_text`` needs to be regenerated with the same flag so
    training prompts match inference prompts at eval time.
    """
    me_abs = _current_player(decision.game_state)
    hand_remaining = [
        d for d in decision.game_state.hands[me_abs] if d not in decision.game_state.played
    ]
    history = _visible_history(decision.game_state)
    system_content, user_content = render_native_messages(
        decision.game_state, hand_remaining, history,
        enable_rules_tools=enable_rules_tools,
        enable_primer=enable_primer,
    )
    user_text = f"[SYSTEM]\n{system_content}\n\n[USER]\n{user_content}"
    assistant_text = compose_assistant_content(trace)
    return {
        "messages": [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": assistant_text},
        ],
        "source": source,
        "seed": int(decision.seed),
        "declaration": int(decision.declaration),
        "narrator_seat": int(decision.narrator_seat),
        "bot_play": int(decision.bot_play),
        "burl_play": int(trace.final_play),
        "bot_eq": float(decision.bot_eq),
        "burl_eq": float(decision.per_play_eq.get(int(trace.final_play), float("nan"))),
        "eq_gap": float(decision.eq_gap),
    }


# --------------------------------------------------------------------------- #
# Orchestrator                                                                 #
# --------------------------------------------------------------------------- #


def _classify(record: dict[str, Any]) -> str:
    """Return one of: 'win', 'legal_loss', 'illegal', 'exhausted'."""
    if record["retry_exhausted"]:
        return "exhausted"
    if record["final_play"] < 0 or not record["final_play_legal"]:
        return "illegal"
    if record["burl_eq"] is None:
        return "illegal"
    if record["burl_eq"] >= record["bot_eq"]:
        return "win"
    return "legal_loss"


def _lookup_decision(
    dataset: list[BurlDecision], trace: BurlTrace,
) -> BurlDecision:
    md = trace.metadata
    return next(
        d for d in dataset
        if int(d.seed) == md["seed"]
        and int(d.declaration) == md["declaration"]
        and int(d.narrator_seat) == md["narrator_seat"]
    )


def run_star_rollout(
    dataset_path: Path,
    out_dir: Path,
    corpus_path: Path,
    n_decisions: int | None,
    max_turns: int,
    max_retries: int,
    cost_cap_usd: float,
    *,
    gate_variant: str = "tool-nudge",
    max_gate_retries: int = 1,
    eq_epsilon: float = 0.25,
    enable_rules_tools: bool = False,
    enable_primer: bool = True,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    corpus_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(dataset_path)
    if n_decisions is not None:
        dataset = dataset[:n_decisions]
    print(f"[star] dataset={dataset_path} n={len(dataset)}")
    print(f"[star] out_dir={out_dir}")
    print(f"[star] corpus={corpus_path}")
    print(
        f"[star] gate_variant={gate_variant} max_gate_retries={max_gate_retries} "
        f"eq_epsilon={eq_epsilon} enable_rules_tools={enable_rules_tools} "
        f"enable_primer={enable_primer}"
    )

    rollout_traces_path = out_dir / "rollout_traces.jsonl"
    gate_traces_path = out_dir / "gate_traces.jsonl"

    from burl.modal.gemma_serve_native import GemmaServerNative, app as gemma_app

    wall_start = time.time()
    records: list[dict[str, Any]] = []
    gate_records: list[dict[str, Any]] = []

    print("[star] opening modal app context (ephemeral)...")
    with gemma_app.run():
        server = GemmaServerNative()
        model_fn = _make_modal_native_model(server)

        # ------------------------------------------------------------------ #
        # Phase A: rollout                                                   #
        # ------------------------------------------------------------------ #
        with rollout_traces_path.open("w") as trace_out:
            for i, decision in enumerate(dataset):
                print(
                    f"[rollout] ({i+1}/{len(dataset)}) seed={decision.seed} "
                    f"decl={decision.declaration} seat={decision.narrator_seat} "
                    f"bot_play={decision.bot_play} eq_gap={decision.eq_gap:.2f}",
                    flush=True,
                )
                t0 = time.time()
                trace, exhausted = _run_one_rollout(
                    decision, model_fn, max_turns, max_retries,
                    enable_rules_tools=enable_rules_tools,
                    enable_primer=enable_primer,
                )
                elapsed = time.time() - t0
                record = _build_record(decision, trace, exhausted, elapsed)
                record["category"] = _classify(record)
                records.append(record)

                trace_out.write(trace.to_json() + "\n")
                trace_out.flush()

                burl_eq_s = (
                    f"{record['burl_eq']:.2f}" if record["burl_eq"] is not None else "N/A"
                )
                print(
                    f"         final={record['final_play']} legal={record['final_play_legal']} "
                    f"cat={record['category']} retries={record['n_retries']} "
                    f"tools={record['n_tool_calls']} burl_eq={burl_eq_s} "
                    f"bot_eq={record['bot_eq']:.2f} elapsed={elapsed:.1f}s",
                    flush=True,
                )

                wall_so_far = time.time() - wall_start
                est_so_far = wall_so_far / 3600.0 * L4_USD_PER_HOUR
                if est_so_far > cost_cap_usd:
                    print(
                        f"[STOP] est ${est_so_far:.2f} > cap ${cost_cap_usd:.2f} "
                        f"after {i+1} rollouts",
                        flush=True,
                    )
                    break

        rollout_wall = time.time() - wall_start
        print(f"[rollout] done. wall={rollout_wall:.1f}s")

        # ------------------------------------------------------------------ #
        # Phase B: EQ-gate the legal losses                                  #
        # ------------------------------------------------------------------ #
        losses = [r for r in records if r["category"] == "legal_loss"]
        print(f"[gate] losses_to_consider={len(losses)}")

        with gate_traces_path.open("w") as trace_out:
            for j, rec in enumerate(losses):
                decision = _lookup_decision(dataset, rec["trace"])

                gate_decision = check_commit(
                    committed_play=rec["final_play"],
                    legal=rec["final_play_legal"],
                    per_play_eq=decision.per_play_eq,
                    bot_play=int(decision.bot_play),
                    bot_eq=float(decision.bot_eq),
                    eq_epsilon=eq_epsilon,
                )
                if not gate_decision.fire:
                    print(
                        f"[gate] skip seed={decision.seed}/{decision.declaration}"
                        f"/{decision.narrator_seat}: {gate_decision.reason} "
                        f"eq_delta={gate_decision.eq_delta:.2f}",
                        flush=True,
                    )
                    continue

                attempts: list[AttemptSummary] = [AttemptSummary(
                    committed_play=rec["final_play"],
                    legal=rec["final_play_legal"],
                    eq=rec["burl_eq"],
                    retry_exhausted=bool(rec.get("retry_exhausted", False)),
                )]
                last_trace: BurlTrace | None = None
                last_record: dict[str, Any] | None = None

                for gi in range(1, max_gate_retries + 1):
                    nudge = gate_feedback_prompt(gate_variant, attempt_idx=gi)
                    print(
                        f"[gate] ({j+1}/{len(losses)}) seed={decision.seed} "
                        f"decl={decision.declaration} seat={decision.narrator_seat} "
                        f"attempt={gi}/{max_gate_retries} variant={gate_variant} "
                        f"eq_delta={gate_decision.eq_delta:.2f}",
                        flush=True,
                    )
                    t0 = time.time()
                    g_trace, g_exhausted = _gate_one(
                        decision, model_fn, nudge, max_turns, max_retries,
                        enable_rules_tools=enable_rules_tools,
                        enable_primer=enable_primer,
                    )
                    elapsed = time.time() - t0
                    g_record = _build_record(decision, g_trace, g_exhausted, elapsed)
                    g_record["category"] = _classify(g_record)
                    g_record["gate_attempt"] = gi
                    g_record["gate_variant"] = gate_variant
                    last_trace = g_trace
                    last_record = g_record

                    attempts.append(AttemptSummary(
                        committed_play=g_record["final_play"],
                        legal=g_record["final_play_legal"],
                        eq=g_record["burl_eq"],
                        retry_exhausted=bool(g_record.get("retry_exhausted", False)),
                    ))

                    trace_out.write(g_trace.to_json() + "\n")
                    trace_out.flush()

                    print(
                        f"       final={g_record['final_play']} legal={g_record['final_play_legal']} "
                        f"retries={g_record['n_retries']} tools={g_record['n_tool_calls']} "
                        f"elapsed={elapsed:.1f}s",
                        flush=True,
                    )

                    if attempts[-1].committed_play == int(decision.bot_play):
                        break
                    if attempts[-1].retry_exhausted:
                        break

                verdict = classify_rationalization(attempts, int(decision.bot_play))
                print(f"[gate] verdict={verdict}", flush=True)
                gate_records.append({
                    "decision_meta": {
                        "seed": int(decision.seed),
                        "declaration": int(decision.declaration),
                        "narrator_seat": int(decision.narrator_seat),
                        "bot_play": int(decision.bot_play),
                        "bot_eq": float(decision.bot_eq),
                    },
                    "attempts": [asdict(a) for a in attempts],
                    "verdict": verdict,
                    "trace": last_trace,
                    "record": last_record,
                    "eq_delta": gate_decision.eq_delta,
                })

                wall_so_far = time.time() - wall_start
                est_so_far = wall_so_far / 3600.0 * L4_USD_PER_HOUR
                if est_so_far > cost_cap_usd:
                    print(
                        f"[STOP] est ${est_so_far:.2f} > cap ${cost_cap_usd:.2f} "
                        f"after {j+1} gated decisions",
                        flush=True,
                    )
                    break

    total_wall = time.time() - wall_start
    est_cost = total_wall / 3600.0 * L4_USD_PER_HOUR

    # ---------------------------------------------------------------------- #
    # Phase C: emit SFT corpus                                               #
    # ---------------------------------------------------------------------- #
    corpus_entries: list[dict[str, Any]] = []
    for rec in records:
        if rec["category"] != "win":
            continue
        decision = _lookup_decision(dataset, rec["trace"])
        corpus_entries.append(
            compose_sft_record(
                decision, rec["trace"], source="rollout_win",
                enable_rules_tools=enable_rules_tools,
                enable_primer=enable_primer,
            )
        )

    for g in gate_records:
        if g["verdict"] != "self_corrected":
            continue
        trace = g["trace"]
        if trace is None:
            continue
        decision = _lookup_decision(dataset, trace)
        corpus_entries.append(
            compose_sft_record(
                decision, trace, source="eq_gate_self_correct",
                enable_rules_tools=enable_rules_tools,
                enable_primer=enable_primer,
            )
        )

    with corpus_path.open("w") as f:
        for entry in corpus_entries:
            f.write(json.dumps(entry, default=_json_default) + "\n")

    # ---------------------------------------------------------------------- #
    # Stats                                                                  #
    # ---------------------------------------------------------------------- #
    cat_hist = Counter(r["category"] for r in records)
    tool_hist: Counter[str] = Counter()
    for r in records:
        for name in r["tool_names"]:
            tool_hist[name] += 1
    decl_hist = Counter(r["trace"].metadata["declaration"] for r in records)

    gate_tool_hist: Counter[str] = Counter()
    for g in gate_records:
        trace = g.get("trace")
        if trace is None:
            continue
        for turn in trace.turns:
            for tc in turn.tool_calls:
                gate_tool_hist[tc.tool_name] += 1

    verdict_hist = Counter(g["verdict"] for g in gate_records)

    rules_tool_names = {
        "count_dominoes_remaining",
        "trick_winner_if",
        "what_beats_what",
        "contract_progress",
    }
    rollout_rules_tool_hist = {
        n: c for n, c in tool_hist.items() if n in rules_tool_names
    }
    gate_rules_tool_hist = {
        n: c for n, c in gate_tool_hist.items() if n in rules_tool_names
    }

    stats = {
        "dataset_path": str(dataset_path),
        "out_dir": str(out_dir),
        "corpus_path": str(corpus_path),
        "generated": datetime.now().isoformat(timespec="seconds"),
        "gate_config": {
            "gate_variant": gate_variant,
            "max_gate_retries": max_gate_retries,
            "eq_epsilon": eq_epsilon,
        },
        "enable_rules_tools": bool(enable_rules_tools),
        "enable_primer": bool(enable_primer),
        "n_decisions_attempted": len(records),
        "n_wins": cat_hist.get("win", 0),
        "n_legal_losses": cat_hist.get("legal_loss", 0),
        "n_illegal": cat_hist.get("illegal", 0),
        "n_exhausted": cat_hist.get("exhausted", 0),
        "n_gate_fired": len(gate_records),
        "n_gate_converged": verdict_hist.get("self_corrected", 0),
        "n_gate_stubborn": verdict_hist.get("stubborn", 0),
        "n_gate_forced_flip": verdict_hist.get("forced_flip", 0),
        "n_gate_exhausted": verdict_hist.get("exhausted", 0),
        "n_corpus_entries": len(corpus_entries),
        "corpus_entries_by_source": dict(
            Counter(e["source"] for e in corpus_entries)
        ),
        "tool_histogram_rollouts": dict(tool_hist),
        "tool_histogram_gate": dict(gate_tool_hist),
        "rules_tool_histogram_rollouts": rollout_rules_tool_hist,
        "rules_tool_histogram_gate": gate_rules_tool_hist,
        "declaration_coverage_rollouts": {str(k): v for k, v in sorted(decl_hist.items())},
        "wall_time_seconds": total_wall,
        "estimated_usd": est_cost,
        "sample_corpus_entries": [
            {
                "source": e["source"],
                "seed": e["seed"],
                "declaration": e["declaration"],
                "narrator_seat": e["narrator_seat"],
                "burl_play": e["burl_play"],
                "bot_play": e["bot_play"],
                "user_preview": e["messages"][0]["content"][:400],
                "assistant_preview": e["messages"][1]["content"][:600],
            }
            for e in corpus_entries[: min(3, len(corpus_entries))]
        ],
    }

    stats_path = corpus_path.with_name(corpus_path.stem + "_stats.json")
    stats_path.write_text(json.dumps(stats, indent=2, default=_json_default) + "\n")
    print(f"[star] stats -> {stats_path}")

    # Print the headline.
    print()
    print("=" * 68)
    print(f"Move 4 — STaR corpus build (N={len(records)})")
    print("=" * 68)
    print(f"  wins (K1 pass)              : {stats['n_wins']}")
    print(f"  legal losses                : {stats['n_legal_losses']}")
    print(f"  illegal / missing E[Q]      : {stats['n_illegal']}")
    print(f"  retry-exhausted (rollouts)  : {stats['n_exhausted']}")
    print(f"  gate fired                  : {stats['n_gate_fired']}")
    print(f"    self_corrected            : {stats['n_gate_converged']}")
    print(f"    forced_flip               : {stats['n_gate_forced_flip']}")
    print(f"    stubborn                  : {stats['n_gate_stubborn']}")
    print(f"    exhausted                 : {stats['n_gate_exhausted']}")
    print(f"  corpus size                 : {stats['n_corpus_entries']}")
    print(f"  wall time                   : {total_wall:.1f}s")
    print(f"  estimated spend             : ${est_cost:.3f}")
    print(f"  tool histogram (rollouts)   : {dict(tool_hist)}")
    print(f"  tool histogram (gate)       : {dict(gate_tool_hist)}")
    if enable_rules_tools:
        print(f"  rules-tool hist (rollouts)  : {rollout_rules_tool_hist}")
        print(f"  rules-tool hist (gate)      : {gate_rules_tool_hist}")
    print("=" * 68)


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #


def _main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset", type=Path,
        default=Path("burl/eval/data/move4_decisions_n50.jsonl"),
    )
    parser.add_argument(
        "--out-dir", type=Path,
        default=Path("burl/eval/results/move4_star_rollout"),
    )
    parser.add_argument(
        "--corpus", type=Path,
        default=Path("burl/data/star_iter0_corpus.jsonl"),
    )
    parser.add_argument("--n", type=int, default=None, dest="n_decisions")
    parser.add_argument("--max-turns", type=int, default=8)
    parser.add_argument("--max-retries", type=int, default=7)
    parser.add_argument("--cost-cap-usd", type=float, default=0.60)
    parser.add_argument(
        "--gate-variant",
        choices=["minimal", "tool-nudge", "social"],
        default="tool-nudge",
        help="EQ-gate feedback nudge variant (iter-2 default: tool-nudge).",
    )
    parser.add_argument(
        "--max-gate-retries", type=int, default=1,
        help="Number of EQ-gate feedback rounds per legal loss (iter-2 default: 1).",
    )
    parser.add_argument(
        "--eq-epsilon", type=float, default=0.25,
        help="Dead-band (E[Q] points) below bot before the gate fires.",
    )
    parser.add_argument(
        "--enable-rules-tools", action="store_true",
        help=(
            "iter-3-rules lever: swap the trimmed primer for the compact "
            "rules-as-tools preamble and register/advertise "
            "count_dominoes_remaining, trick_winner_if, what_beats_what, "
            "contract_progress alongside the eight default tools."
        ),
    )
    parser.add_argument(
        "--no-primer", action="store_true",
        help=(
            "iter-3-v2 lever: drop the trimmed Texas-42 primer entirely "
            "(spike-v2 prompt shape — system = preamble + 42-framing only, "
            "no rules-as-tools). Mutually exclusive with --enable-rules-tools."
        ),
    )
    args = parser.parse_args()

    run_star_rollout(
        dataset_path=args.dataset,
        out_dir=args.out_dir,
        corpus_path=args.corpus,
        n_decisions=args.n_decisions,
        max_turns=args.max_turns,
        max_retries=args.max_retries,
        cost_cap_usd=args.cost_cap_usd,
        gate_variant=args.gate_variant,
        max_gate_retries=args.max_gate_retries,
        eq_epsilon=args.eq_epsilon,
        enable_rules_tools=args.enable_rules_tools,
        enable_primer=not args.no_primer,
    )


if __name__ == "__main__":
    _main()
