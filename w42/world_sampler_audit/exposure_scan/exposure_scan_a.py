"""Stage 0 exposure Scan A: state-level legacy-MRV malformed-mass scan.

Reconstructs a proxy population of late-game Texas 42 decision states via random
legal playouts (exactly the audit's ``advance_fixture`` construction, generalized
to snapshot at several play depths), and for each state computes the *analytic*
legacy WorldSamplerMRV output distribution -- no sampling, no oracle -- to measure:

  * dead_end_mass    -- probability the legacy greedy-MRV recursion hit a
                        no-candidate seat and injected domino 0 (malformed).
  * invalid_mass     -- total analytic MRV mass on worlds outside the exact
                        public-consistent world set (malformed / impossible).
  * exact_world_count-- size of the exact uniform world set.

This is CPU-only and analytic: it reuses ``analytic_mrv_distribution`` from the
audit, which is the only faithful reproduction of the retired legacy sampler.

Population caveat: this is a RECONSTRUCTED PROXY (random uniform legal playouts),
not the literal historical decision stream. See exposure_scan_a.md.

Usage:
    python -u scratch/stage0/exposure_scan_a.py --seeds 500 --smoke
    python -u scratch/stage0/exposure_scan_a.py --seeds 2000
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

# Faithful reuse of the audit's analytic legacy-MRV emulator and state plumbing.
from w42.world_sampler_audit.audit import (
    analytic_mrv_distribution,
    candidate_masks,
    enumerate_exact_worlds,
    sampler_inputs,
)
from forge.eq.enumeration_gpu import estimate_world_count, should_enumerate
from forge.zeb.game import apply_action, legal_actions, new_game

HERE = Path(__file__).resolve().parent
ROWS_PATH = HERE / "exposure_scan_a_rows.jsonl"
LOG_PATH = HERE / "exposure_scan_a.log"

# Play depths to snapshot at. Trick 4-6 territory: pool small enough that both
# exact enumeration and the analytic MRV recursion are cheap.
DEFAULT_DEPTHS = (16, 18, 20, 22)

# Tractability gate: skip states whose estimated exact world count exceeds this.
# should_enumerate uses estimate_world_count(history_len); we also apply an exact
# post-enumeration guard so a single blown-up state cannot dominate the run.
TRACTABILITY_THRESHOLD = 100_000


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def states_for_seed(deal_seed: int, depths: tuple[int, ...]):
    """Replay one random legal playout, yielding (depth, state) at each depth.

    Mirrors audit.advance_fixture: new_game(seed, skip_bidding=True) then an
    independent random.Random(rollout_seed) choosing legal slots. We use one
    rollout per deal seed and snapshot at each requested depth as we pass it.
    """
    state = new_game(seed=deal_seed, skip_bidding=True)
    rng = random.Random(deal_seed)  # rollout_seed == deal_seed, as in the panel
    target = set(depths)
    max_depth = max(depths)
    for step in range(max_depth + 1):
        depth = len(state.play_history)
        if depth in target:
            yield depth, state
        if depth >= max_depth:
            break
        actions = legal_actions(state)
        if not actions:
            break
        state = apply_action(state, rng.choice(actions))


def scan_state(deal_seed: int, depth: int, state) -> dict[str, Any]:
    """Compute analytic legacy-MRV malformed mass for one decision state.

    Returns a row dict. On tractability skip, sets skipped=True with a reason and
    no mass fields (exposure-unknown).
    """
    row: dict[str, Any] = {
        "deal_seed": deal_seed,
        "rollout_seed": deal_seed,
        "depth": depth,
        "estimated_world_count": estimate_world_count(depth),
    }

    if not should_enumerate(depth, threshold=TRACTABILITY_THRESHOLD):
        row["skipped"] = True
        row["skip_reason"] = "estimate_world_count_exceeds_threshold"
        return row

    inputs = sampler_inputs(state)
    exact_worlds = enumerate_exact_worlds(inputs)
    exact_count = len(exact_worlds)

    if exact_count > TRACTABILITY_THRESHOLD:
        row["skipped"] = True
        row["skip_reason"] = "exact_world_count_exceeds_threshold"
        row["exact_world_count"] = exact_count
        return row

    analytic_all, dead_end_mass = analytic_mrv_distribution(
        inputs.pool,
        inputs.hand_sizes,
        candidate_masks(inputs),
    )
    exact_set = set(exact_worlds)
    invalid_mass = sum(
        probability
        for world, probability in analytic_all.items()
        if world not in exact_set
    )

    row["skipped"] = False
    row["current_player"] = inputs.current_player
    row["decl_id"] = inputs.decl_id
    row["pool_size"] = len(inputs.pool)
    row["hand_sizes"] = list(inputs.hand_sizes)
    row["void_counts"] = [len(suits) for suits in inputs.voids]
    row["exact_world_count"] = exact_count
    row["analytic_world_count"] = len(analytic_all)
    row["dead_end_mass"] = float(dead_end_mass)
    row["invalid_mass"] = float(invalid_mass)
    row["legal_action_count"] = len(legal_actions(state))
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=2000,
                        help="Number of deal seeds to scan (each yields up to len(depths) states).")
    parser.add_argument("--seed-start", type=int, default=900000,
                        help="First deal seed (default 900000, near the audit panel seeds).")
    parser.add_argument("--depths", type=int, nargs="+", default=list(DEFAULT_DEPTHS))
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke mode: cap total states near 200 and report timing.")
    parser.add_argument("--max-seconds", type=float, default=None,
                        help="Wall-clock budget; stop launching new seeds past this.")
    parser.add_argument("--rows-path", type=Path, default=ROWS_PATH)
    args = parser.parse_args()

    depths = tuple(sorted(set(args.depths)))
    n_seeds = args.seeds
    if args.smoke:
        n_seeds = max(1, 200 // len(depths))

    log(f"Scan A start: seeds={n_seeds} seed_start={args.seed_start} "
        f"depths={depths} smoke={args.smoke} max_seconds={args.max_seconds}")

    args.rows_path.write_text("", encoding="utf-8")  # truncate
    handle = args.rows_path.open("a", encoding="utf-8")

    start = time.time()
    n_states = 0
    n_scanned = 0
    n_skipped = 0
    n_errors = 0
    last_log = start

    for i in range(n_seeds):
        deal_seed = args.seed_start + i
        if args.max_seconds is not None and (time.time() - start) > args.max_seconds:
            log(f"Hit max_seconds budget after {i} seeds; stopping.")
            break
        try:
            captured = list(states_for_seed(deal_seed, depths))
        except Exception as exc:  # pragma: no cover - defensive
            n_errors += 1
            log(f"seed {deal_seed}: rollout error: {exc!r}")
            continue
        for depth, state in captured:
            try:
                row = scan_state(deal_seed, depth, state)
            except Exception as exc:  # pragma: no cover - defensive
                n_errors += 1
                row = {
                    "deal_seed": deal_seed,
                    "depth": depth,
                    "skipped": True,
                    "skip_reason": f"error:{type(exc).__name__}",
                    "error": str(exc),
                }
            handle.write(json.dumps(row) + "\n")
            n_states += 1
            if row.get("skipped"):
                n_skipped += 1
            else:
                n_scanned += 1

        now = time.time()
        if now - last_log > 20 or args.smoke:
            rate = n_states / max(now - start, 1e-9)
            log(f"seed {deal_seed} ({i+1}/{n_seeds}): states={n_states} "
                f"scanned={n_scanned} skipped={n_skipped} errors={n_errors} "
                f"rate={rate:.1f} states/s")
            last_log = now

    handle.close()
    elapsed = time.time() - start
    log(f"Scan A done: states={n_states} scanned={n_scanned} skipped={n_skipped} "
        f"errors={n_errors} elapsed={elapsed:.1f}s "
        f"rate={n_states/max(elapsed,1e-9):.1f} states/s")
    log(f"Rows written to {args.rows_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
