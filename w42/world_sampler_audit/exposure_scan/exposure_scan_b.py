"""Stage 0 exposure Scan B: decision-level harm on nonzero-mass states.

DO NOT RUN under the Stage 0 CPU battery -- this loads the Stage-1 oracle and is
intended to run later on MPS (or CUDA). It takes --device and --limit.

Scan A (exposure_scan_a.py) identifies decision states where the *legacy*
WorldSamplerMRV had nonzero malformed mass (dead_end_mass or invalid_mass > 0).
Scan A measures only distributional exposure -- it says nothing about whether the
malformed mass would have changed a decision. Scan B answers that: for each
nonzero-mass state it reconstructs the state, evaluates exact per-action E[Q] via
the audit's oracle path, and compares the *legacy-MRV-distribution-weighted*
action ranking against the *exact-uniform* ranking. Per state it reports:

  * argmax_flipped            -- did the legacy weighting change the top action?
  * exact_regret_of_mrv_choice_q -- E[Q] the legacy choice gives up vs the
                                 uniform-optimal action, scored under uniform.
  * max_abs_action_shift_q    -- largest per-action |E[Q]_mrv - E[Q]_uniform|.

These come directly from audit.rank_actions, reused unchanged so Scan B is a
faithful decision-harm reading of the same legacy emulator.

The oracle Q path here is analytic-weighted only (no empirical sampling): we build
an empirical_distribution equal to the analytic one so rank_actions' required
mass is satisfied without running the live tensor sampler. The argmax flip,
regret, and shift fields depend only on uniform vs analytic-MRV weights, so this
substitution does not affect the reported harm metrics.

Input:  exposure_scan_a_rows.jsonl (nonzero-mass rows are selected).
Output: exposure_scan_b_rows.jsonl + exposure_scan_b.md summary.

Usage (LATER, on MPS):
    python -u scratch/stage0/exposure_scan_b.py --device mps --limit 200
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from w42.world_sampler_audit.audit import (
    REPO_ROOT,
    analytic_mrv_distribution,
    candidate_masks,
    enumerate_exact_worlds,
    query_world_q_by_encoding,
    rank_actions,
    sampler_inputs,
)
from forge.zeb.game import apply_action, legal_actions, new_game

HERE = Path(__file__).resolve().parent
ROWS_IN = HERE / "exposure_scan_a_rows.jsonl"
ROWS_OUT = HERE / "exposure_scan_b_rows.jsonl"
MD_OUT = HERE / "exposure_scan_b.md"
LOG_PATH = HERE / "exposure_scan_b.log"

# Same checkpoint the audit manifest uses.
DEFAULT_CHECKPOINT = REPO_ROOT / "forge/models/domino-qval-3.3M-shuffle-qgap0.074-qmae0.96.ckpt"


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def reconstruct_state(deal_seed: int, rollout_seed: int, depth: int):
    """Replay the exact same rollout Scan A used to reach this decision state."""
    import random

    state = new_game(seed=deal_seed, skip_bidding=True)
    rng = random.Random(rollout_seed)
    for _ in range(depth):
        actions = legal_actions(state)
        if not actions:
            raise RuntimeError(f"seed {deal_seed}: no legal action before depth {depth}")
        state = apply_action(state, rng.choice(actions))
    if len(state.play_history) != depth:
        raise RuntimeError(f"seed {deal_seed}: reached depth {len(state.play_history)} != {depth}")
    return state


def evaluate_state(row: dict[str, Any], checkpoint: Path, device: str) -> dict[str, Any]:
    """Compute decision-level harm for one nonzero-mass Scan A row."""
    deal_seed = row["deal_seed"]
    rollout_seed = row.get("rollout_seed", deal_seed)
    depth = row["depth"]

    state = reconstruct_state(deal_seed, rollout_seed, depth)
    inputs = sampler_inputs(state)
    exact_worlds = enumerate_exact_worlds(inputs)
    exact_distribution = {world: 1.0 / len(exact_worlds) for world in exact_worlds}

    analytic_all, dead_end_mass = analytic_mrv_distribution(
        inputs.pool,
        inputs.hand_sizes,
        candidate_masks(inputs),
    )

    # Query Q for every world either the exact set or legacy MRV can produce
    # (including malformed ones), matching audit.run_audit's outputs_to_query.
    outputs_to_query = tuple(sorted(set(exact_worlds) | set(analytic_all)))
    q_by_encoding = query_world_q_by_encoding(
        state,
        outputs_to_query,
        checkpoint=checkpoint,
        device=device,
    )

    # rank_actions needs an empirical distribution too; the harm metrics we
    # report depend only on uniform vs analytic-MRV weights, so pass analytic
    # as the empirical stand-in (no live sampler run required).
    out: dict[str, Any] = {
        "deal_seed": deal_seed,
        "rollout_seed": rollout_seed,
        "depth": depth,
        "exact_world_count": len(exact_worlds),
        "dead_end_mass": float(dead_end_mass),
        "invalid_mass": float(row.get("invalid_mass", 0.0)),
        "by_encoding": {},
    }
    for encoding, q_rows in q_by_encoding.items():
        ranking = rank_actions(
            state,
            exact_worlds,
            q_rows,
            exact_distribution,
            analytic_all,       # legacy-MRV weighting (malformed mass retained)
            analytic_all,       # empirical stand-in; see docstring
        )
        out["by_encoding"][encoding] = {
            "argmax_flipped": ranking["argmax_flipped"],
            "valid_conditional_argmax_flipped": ranking["valid_conditional_argmax_flipped"],
            "exact_regret_of_mrv_choice_q": ranking["exact_regret_of_mrv_choice_q"],
            "max_abs_action_shift_q": ranking["max_abs_action_shift_q"],
            "analytic_valid_mass": ranking["analytic_valid_mass"],
            "uniform_best_slot": ranking["uniform_best_slot"],
            "analytic_mrv_best_slot": ranking["analytic_mrv_best_slot"],
            "legal_action_count": ranking["legal_action_count"],
        }
    primary = out["by_encoding"]["remaining_only_canonical"]
    out["argmax_flipped"] = primary["argmax_flipped"]
    out["exact_regret_of_mrv_choice_q"] = primary["exact_regret_of_mrv_choice_q"]
    out["max_abs_action_shift_q"] = primary["max_abs_action_shift_q"]
    return out


def write_summary(results: list[dict[str, Any]], n_candidates: int, device: str) -> None:
    evaluated = [r for r in results if "error" not in r]
    flips = [r for r in evaluated if r.get("argmax_flipped")]
    regrets = sorted(
        evaluated, key=lambda r: -abs(r.get("exact_regret_of_mrv_choice_q", 0.0))
    )
    shifts = sorted(
        evaluated, key=lambda r: -abs(r.get("max_abs_action_shift_q", 0.0))
    )
    lines = [
        "# Stage 0 exposure Scan B: decision-level harm",
        "",
        f"- device: `{device}`",
        f"- nonzero-mass candidate states (from Scan A): {n_candidates}",
        f"- states evaluated: {len(evaluated)}",
        f"- argmax flips (production encoding): {len(flips)}",
        "",
        "## Worst 10 by exact regret of legacy choice (Q)",
        "",
        "| seed | depth | regret Q | max shift Q | argmax flip |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in regrets[:10]:
        lines.append(
            f"| {r['deal_seed']} | {r['depth']} | "
            f"{r.get('exact_regret_of_mrv_choice_q', 0.0):.3f} | "
            f"{r.get('max_abs_action_shift_q', 0.0):.3f} | "
            f"{'yes' if r.get('argmax_flipped') else 'no'} |"
        )
    lines += ["", "## Worst 10 by max abs action shift (Q)", "",
              "| seed | depth | max shift Q | regret Q | argmax flip |",
              "|---|---:|---:|---:|---:|"]
    for r in shifts[:10]:
        lines.append(
            f"| {r['deal_seed']} | {r['depth']} | "
            f"{r.get('max_abs_action_shift_q', 0.0):.3f} | "
            f"{r.get('exact_regret_of_mrv_choice_q', 0.0):.3f} | "
            f"{'yes' if r.get('argmax_flipped') else 'no'} |"
        )
    MD_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="mps")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max nonzero-mass states to evaluate.")
    parser.add_argument("--rows-in", type=Path, default=ROWS_IN)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    args = parser.parse_args()

    if not args.rows_in.exists():
        raise FileNotFoundError(f"Scan A rows not found: {args.rows_in}. Run Scan A first.")

    candidates = []
    for line in args.rows_in.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("skipped"):
            continue
        if row.get("dead_end_mass", 0.0) > 0.0 or row.get("invalid_mass", 0.0) > 0.0:
            candidates.append(row)

    # Worst-first so a truncated --limit run still sees the biggest exposure.
    candidates.sort(key=lambda r: -(r.get("invalid_mass", 0.0) + r.get("dead_end_mass", 0.0)))
    if args.limit is not None:
        candidates = candidates[: args.limit]

    log(f"Scan B start: device={args.device} candidates={len(candidates)} "
        f"checkpoint={args.checkpoint}")

    ROWS_OUT.write_text("", encoding="utf-8")
    handle = ROWS_OUT.open("a", encoding="utf-8")
    results: list[dict[str, Any]] = []
    start = time.time()
    for i, row in enumerate(candidates):
        try:
            out = evaluate_state(row, args.checkpoint, args.device)
        except Exception as exc:  # pragma: no cover - defensive
            out = {
                "deal_seed": row["deal_seed"],
                "depth": row["depth"],
                "error": f"{type(exc).__name__}: {exc}",
            }
            log(f"seed {row['deal_seed']} depth {row['depth']}: ERROR {out['error']}")
        handle.write(json.dumps(out) + "\n")
        results.append(out)
        if (i + 1) % 10 == 0 or (time.time() - start) > 20:
            done = i + 1
            log(f"{done}/{len(candidates)} evaluated "
                f"({done/max(time.time()-start,1e-9):.2f} states/s)")
    handle.close()

    write_summary(results, len(candidates), args.device)
    log(f"Scan B done: evaluated={len(results)} elapsed={time.time()-start:.1f}s")
    log(f"Rows -> {ROWS_OUT}; summary -> {MD_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
