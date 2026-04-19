"""Move 4 STaR corpus builder — rollout N=50 + rationalize the K1 losses.

Phase 2 of Move 4. Produces an SFT corpus for training ``burl-iter0``. The
flow mirrors ``lem/gemma_star/star_loop.py`` conceptually:

  1. Rollout each decision via the post-Layer-1 native endpoint.
  2. K1 filter: ``burl_eq >= bot_eq`` -> keep the trace as a positive exemplar.
  3. Rationalize losses: re-run Gemma with the ground-truth play added to the
     system prompt, collect a new trace; keep only if the rationalization
     commits to ``bot_play`` legally. This is self-taught STaR — same model
     teaching itself how to arrive at the known-good play.
  4. Emit ``burl/data/star_iter0_corpus.jsonl`` as HF chat-format pairs:
     ``{"messages": [{"role":"user", "content":<prompt>},
                     {"role":"assistant","content":<trace>}]}``.

Usage:
    python -u -m burl.eval.run_move4_star_rollout \\
        --dataset burl/eval/data/move4_decisions_n50.jsonl \\
        --out-dir burl/eval/results/move4_star_rollout \\
        --corpus burl/data/star_iter0_corpus.jsonl
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
    _DOMINO_LABELS,
    _current_player,
    _visible_history,
    build_tool_registry,
)
from burl.harness.agent_runner_native import (
    TOOL_SCHEMAS,
    _COMMIT_INSTRUCTION,
    render_native_messages,
    run_decision_native,
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
# Rationalization                                                              #
# --------------------------------------------------------------------------- #


_RATIONALIZE_SUFFIX_TEMPLATE = (
    "\n\n# Rationalization hint (training signal, drop at inference)\n\n"
    "The correct play here is domino_id={dom_id} ({label}). "
    "Using the tools available (is_legal, is_trump, unseen, void_audit, "
    "trump_declared, eq_outcome_distribution, conditional_outcome), show "
    "how you would reason your way to this play. When your reasoning is "
    "complete, commit to it with commit_play(domino_id={dom_id})."
)


def _rationalize_one(
    decision: BurlDecision,
    model_fn: NativeModelCallable,
    max_turns: int,
    max_retries: int,
) -> tuple[BurlTrace, bool]:
    """Re-roll Gemma with the ground-truth play appended to the system prompt.

    Returns the resulting trace plus a ``retry_exhausted`` flag. Uses
    ``NativeHarness`` directly so we can substitute the augmented system
    message without disturbing ``render_native_messages``.
    """
    me_abs = _current_player(decision.game_state)
    hand_remaining = [
        d for d in decision.game_state.hands[me_abs] if d not in decision.game_state.played
    ]
    history = _visible_history(decision.game_state)
    system_content, user_content = render_native_messages(
        decision.game_state, hand_remaining, history,
    )
    label = _DOMINO_LABELS[int(decision.bot_play)]
    augmented_system = system_content + _RATIONALIZE_SUFFIX_TEMPLATE.format(
        dom_id=int(decision.bot_play), label=label,
    )

    tools = build_tool_registry(game_state_provider=lambda: decision.game_state)
    harness = NativeHarness(
        model_callable=model_fn,
        tools=tools,
        tool_schemas=TOOL_SCHEMAS,
        is_legal_fn=lambda s, d: engine_tools.is_legal(s, int(d)),
        commit_instruction=_COMMIT_INSTRUCTION,
        max_turns=max_turns,
        max_retries=max_retries,
    )

    retry_exhausted = False
    try:
        trace = harness.run(
            game_state=decision.game_state,
            system_content=augmented_system,
            user_content=user_content,
            state_key=_state_key(decision.game_state),
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

    trace.metadata["rationalization"] = True
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
) -> dict[str, Any]:
    """Build an HF-chat-format record plus provenance metadata."""
    me_abs = _current_player(decision.game_state)
    hand_remaining = [
        d for d in decision.game_state.hands[me_abs] if d not in decision.game_state.played
    ]
    history = _visible_history(decision.game_state)
    system_content, user_content = render_native_messages(
        decision.game_state, hand_remaining, history,
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


def run_star_rollout(
    dataset_path: Path,
    out_dir: Path,
    corpus_path: Path,
    n_decisions: int | None,
    max_turns: int,
    max_retries: int,
    cost_cap_usd: float,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    corpus_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(dataset_path)
    if n_decisions is not None:
        dataset = dataset[:n_decisions]
    print(f"[star] dataset={dataset_path} n={len(dataset)}")
    print(f"[star] out_dir={out_dir}")
    print(f"[star] corpus={corpus_path}")

    rollout_traces_path = out_dir / "rollout_traces.jsonl"
    rationalize_traces_path = out_dir / "rationalize_traces.jsonl"

    from burl.modal.gemma_serve_native import GemmaServerNative, app as gemma_app

    wall_start = time.time()
    records: list[dict[str, Any]] = []
    rationalization_records: list[dict[str, Any]] = []

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
        # Phase B: rationalize the legal losses                              #
        # ------------------------------------------------------------------ #
        losses = [r for r in records if r["category"] == "legal_loss"]
        print(f"[rationalize] losses={len(losses)}")

        with rationalize_traces_path.open("w") as trace_out:
            for j, rec in enumerate(losses):
                decision = next(
                    d for d in dataset
                    if int(d.seed) == rec["trace"].metadata["seed"]
                    and int(d.declaration) == rec["trace"].metadata["declaration"]
                    and int(d.narrator_seat) == rec["trace"].metadata["narrator_seat"]
                )
                print(
                    f"[rationalize] ({j+1}/{len(losses)}) seed={decision.seed} "
                    f"decl={decision.declaration} seat={decision.narrator_seat} "
                    f"target_bot_play={decision.bot_play}",
                    flush=True,
                )
                t0 = time.time()
                r_trace, r_exhausted = _rationalize_one(
                    decision, model_fn, max_turns, max_retries,
                )
                elapsed = time.time() - t0
                r_record = _build_record(decision, r_trace, r_exhausted, elapsed)
                r_record["category"] = _classify(r_record)
                r_record["rationalization_converged"] = (
                    not r_exhausted
                    and r_record["final_play_legal"]
                    and r_record["final_play"] == int(decision.bot_play)
                )
                rationalization_records.append(r_record)

                trace_out.write(r_trace.to_json() + "\n")
                trace_out.flush()

                conv = "OK" if r_record["rationalization_converged"] else "FAIL"
                print(
                    f"           rationalize={conv} final={r_record['final_play']} "
                    f"legal={r_record['final_play_legal']} retries={r_record['n_retries']} "
                    f"tools={r_record['n_tool_calls']} elapsed={elapsed:.1f}s",
                    flush=True,
                )

                wall_so_far = time.time() - wall_start
                est_so_far = wall_so_far / 3600.0 * L4_USD_PER_HOUR
                if est_so_far > cost_cap_usd:
                    print(
                        f"[STOP] est ${est_so_far:.2f} > cap ${cost_cap_usd:.2f} "
                        f"after {j+1} rationalizations",
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
        decision = next(
            d for d in dataset
            if int(d.seed) == rec["trace"].metadata["seed"]
            and int(d.declaration) == rec["trace"].metadata["declaration"]
            and int(d.narrator_seat) == rec["trace"].metadata["narrator_seat"]
        )
        corpus_entries.append(
            compose_sft_record(decision, rec["trace"], source="rollout_win")
        )

    rationalizations_kept: list[dict[str, Any]] = []
    for r_rec in rationalization_records:
        if not r_rec.get("rationalization_converged"):
            continue
        decision = next(
            d for d in dataset
            if int(d.seed) == r_rec["trace"].metadata["seed"]
            and int(d.declaration) == r_rec["trace"].metadata["declaration"]
            and int(d.narrator_seat) == r_rec["trace"].metadata["narrator_seat"]
        )
        entry = compose_sft_record(decision, r_rec["trace"], source="rationalization")
        corpus_entries.append(entry)
        rationalizations_kept.append(r_rec)

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
    decl_hist = Counter(
        r["trace"].metadata["declaration"] for r in records
    )
    ration_cat_hist = Counter(r["category"] for r in rationalization_records)

    stats = {
        "dataset_path": str(dataset_path),
        "out_dir": str(out_dir),
        "corpus_path": str(corpus_path),
        "generated": datetime.now().isoformat(timespec="seconds"),
        "n_decisions_attempted": len(records),
        "n_wins": cat_hist.get("win", 0),
        "n_legal_losses": cat_hist.get("legal_loss", 0),
        "n_illegal": cat_hist.get("illegal", 0),
        "n_exhausted": cat_hist.get("exhausted", 0),
        "n_rationalizations_attempted": len(rationalization_records),
        "n_rationalizations_converged": sum(
            1 for r in rationalization_records if r.get("rationalization_converged")
        ),
        "n_rationalizations_legal_wrong_play": sum(
            1 for r in rationalization_records
            if r.get("category") == "legal_loss"
            and not r.get("rationalization_converged")
        ),
        "n_rationalizations_exhausted": ration_cat_hist.get("exhausted", 0),
        "n_rationalizations_illegal": ration_cat_hist.get("illegal", 0),
        "n_corpus_entries": len(corpus_entries),
        "corpus_entries_by_source": dict(
            Counter(e["source"] for e in corpus_entries)
        ),
        "tool_histogram_rollouts": dict(tool_hist),
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
    print(f"  retry-exhausted             : {stats['n_exhausted']}")
    print(f"  rationalizations attempted  : {stats['n_rationalizations_attempted']}")
    print(f"  rationalizations converged  : {stats['n_rationalizations_converged']}")
    print(f"  corpus size                 : {stats['n_corpus_entries']}")
    print(f"  wall time                   : {total_wall:.1f}s")
    print(f"  estimated spend             : ${est_cost:.3f}")
    print(f"  tool histogram (rollouts)   : {dict(tool_hist)}")
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
    args = parser.parse_args()

    run_star_rollout(
        dataset_path=args.dataset,
        out_dir=args.out_dir,
        corpus_path=args.corpus,
        n_decisions=args.n_decisions,
        max_turns=args.max_turns,
        max_retries=args.max_retries,
        cost_cap_usd=args.cost_cap_usd,
    )


if __name__ == "__main__":
    _main()
