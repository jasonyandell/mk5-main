"""Lens v1 round-robin: Lens(util_a) vs Lens(util_b) over the 6 pairings.

Pairings (from {ev, p_make, cvar_10, robust_q25}):
  - ev          vs p_make
  - ev          vs cvar_10
  - ev          vs robust_q25
  - p_make      vs cvar_10
  - p_make      vs robust_q25
  - cvar_10     vs robust_q25

Plus:
  - one mark_ev vs p_make sanity matchup (must be exactly identical at bid=30)
  - sample-sweep on one matchup at N in {10, 50, 100} (250 hands each)

Outputs:
  results/round_robin_n10.csv         # per-pairing aggregates
  results/per_hand_margins.parquet    # per-hand margins for each matchup
  results/sample_sweep.csv            # ev_vs_pmake at N=10/50/100
  results/mark_ev_pmake_sanity.csv    # mark_ev vs p_make
  results/fp_sanity.csv               # if fp16 attempted
  manifest.json
  summary.json
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from forge.zeb.eval.loading import DEFAULT_ORACLE, load_oracle

from w42.lens_v1.parallel_match import LensMatchResult, run_lens_match


ROUND_ROBIN_PAIRINGS = [
    ("ev", "p_make"),
    ("ev", "cvar_10"),
    ("ev", "robust_q25"),
    ("p_make", "cvar_10"),
    ("p_make", "robust_q25"),
    ("cvar_10", "robust_q25"),
]


def bootstrap_ci_mean(values: np.ndarray, n_boot: int = 2000, seed: int = 42) -> tuple[float, float]:
    """Percentile bootstrap CI on the mean."""
    if values.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    n = values.size
    idx = rng.integers(0, n, size=(n_boot, n))
    means = values[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summarize_match(result: LensMatchResult) -> dict:
    margins = np.array([h.margin for h in result.hands], dtype=np.float64)
    a_pts = np.array([h.team_a_pts for h in result.hands], dtype=np.float64)
    b_pts = np.array([h.team_b_pts for h in result.hands], dtype=np.float64)
    decisive = (margins != 0).astype(np.float64)
    a_wins = (margins > 0).astype(np.float64)
    ci_lo, ci_hi = bootstrap_ci_mean(margins)
    return {
        "utility_a": result.utility_a,
        "utility_b": result.utility_b,
        "n_hands": result.n_hands,
        "n_samples": result.n_samples,
        "fp_dtype": result.fp_dtype,
        "elapsed_s": round(result.elapsed_s, 2),
        "mean_margin": float(margins.mean()),
        "margin_ci_lo_95": ci_lo,
        "margin_ci_hi_95": ci_hi,
        "ci_excludes_zero": int((ci_lo > 0) or (ci_hi < 0)),
        "a_win_rate": float(a_wins.mean()),
        "decisive_rate": float(decisive.mean()),
        "a_pts_mean": float(a_pts.mean()),
        "b_pts_mean": float(b_pts.mean()),
        "a_wins": int(a_wins.sum()),
        "n_decisive": int(decisive.sum()),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_per_hand_csv(path: Path, results: list[LensMatchResult]) -> None:
    """Write one row per hand across all matchups (parquet would need pyarrow;
    CSV is fine and human-readable). The path retains .parquet extension only
    if requested; here we write .csv alongside.
    """
    rows = []
    for r in results:
        for h in r.hands:
            rows.append({
                "utility_a": r.utility_a,
                "utility_b": r.utility_b,
                "n_samples": r.n_samples,
                "seed": h.seed,
                "a_team": h.a_team,
                "bidder": h.bidder,
                "decl_id": h.decl_id,
                "bid_value": h.bid_value,
                "team_a_pts": h.team_a_pts,
                "team_b_pts": h.team_b_pts,
                "margin": h.margin,
                "n_decisions_a": h.n_decisions_a,
                "n_decisions_b": h.n_decisions_b,
            })
    write_csv(path, rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-hands", type=int, default=1000,
                        help="Hands per round-robin pairing")
    parser.add_argument("--n-samples", type=int, default=10,
                        help="Worlds sampled per Lens decision")
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--checkpoint", type=str, default=str(PROJECT_ROOT / DEFAULT_ORACLE))
    parser.add_argument("--out-dir", type=str, default=str(Path(__file__).parent / "results"))
    parser.add_argument("--base-seed", type=int, default=10000,
                        help="Base seed; same for all pairings -> hands match across pairings")
    parser.add_argument("--skip-sweep", action="store_true",
                        help="Skip the sample-sweep confirmation step")
    parser.add_argument("--skip-mark-ev-sanity", action="store_true",
                        help="Skip the mark_ev vs p_make sanity matchup")
    parser.add_argument("--sweep-n-hands", type=int, default=250)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = args.device
    if device == "mps" and not torch.backends.mps.is_available():
        print("MPS unavailable; falling back to CPU.", flush=True)
        device = "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA unavailable; falling back to mps/cpu.", flush=True)
        device = "mps" if torch.backends.mps.is_available() else "cpu"

    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        print(f"Error: checkpoint not found: {ckpt}", flush=True)
        return 1

    print(f"Loading oracle: {ckpt} on {device}", flush=True)
    model = load_oracle(str(ckpt), device)

    t_start = time.time()
    rr_results: list[LensMatchResult] = []
    rr_summaries: list[dict] = []

    print(f"\n=== Round-robin: {len(ROUND_ROBIN_PAIRINGS)} pairings × {args.n_hands} hands @ N={args.n_samples} ===", flush=True)
    for ua, ub in ROUND_ROBIN_PAIRINGS:
        print(f"\n[{ua} vs {ub}]", flush=True)
        r = run_lens_match(
            model,
            utility_a=ua, utility_b=ub,
            n_hands=args.n_hands,
            n_samples=args.n_samples,
            device=device,
            base_seed=args.base_seed,
            verbose=True,
        )
        rr_results.append(r)
        s = summarize_match(r)
        rr_summaries.append(s)
        print(
            f"  -> mean_margin={s['mean_margin']:+.3f}  "
            f"95% CI [{s['margin_ci_lo_95']:+.3f}, {s['margin_ci_hi_95']:+.3f}]  "
            f"a_wr={s['a_win_rate']:.1%}  decisive={s['decisive_rate']:.1%}  "
            f"({r.elapsed_s:.1f}s)",
            flush=True,
        )

    rr_csv = out_dir / f"round_robin_n{args.n_samples}.csv"
    write_csv(rr_csv, rr_summaries)
    print(f"\nWrote {rr_csv}", flush=True)

    per_hand_csv = out_dir / "per_hand_margins.csv"
    write_per_hand_csv(per_hand_csv, rr_results)
    print(f"Wrote {per_hand_csv}", flush=True)

    # ----- Sample sweep -----
    sweep_summaries: list[dict] = []
    if not args.skip_sweep:
        print(f"\n=== Sample sweep: ev vs p_make at N in {{10, 50, 100}} × {args.sweep_n_hands} hands ===", flush=True)
        for n_samp in (10, 50, 100):
            r = run_lens_match(
                model,
                utility_a="ev", utility_b="p_make",
                n_hands=args.sweep_n_hands,
                n_samples=n_samp,
                device=device,
                base_seed=args.base_seed,  # same seeds as round-robin
                verbose=True,
            )
            s = summarize_match(r)
            sweep_summaries.append(s)
            print(
                f"  N={n_samp} -> mean_margin={s['mean_margin']:+.3f} "
                f"95% CI [{s['margin_ci_lo_95']:+.3f}, {s['margin_ci_hi_95']:+.3f}]  "
                f"a_wr={s['a_win_rate']:.1%}  ({r.elapsed_s:.1f}s)",
                flush=True,
            )
        sweep_csv = out_dir / "sample_sweep.csv"
        write_csv(sweep_csv, sweep_summaries)
        print(f"Wrote {sweep_csv}", flush=True)
    else:
        print("\n(skipping sample sweep)", flush=True)

    # ----- mark_ev vs p_make sanity -----
    if not args.skip_mark_ev_sanity:
        print(f"\n=== mark_ev vs p_make sanity (must tie identically at bid=30) ===", flush=True)
        # Use a small N because this is just a sanity check
        sanity = run_lens_match(
            model,
            utility_a="mark_ev", utility_b="p_make",
            n_hands=200,
            n_samples=args.n_samples,
            device=device,
            base_seed=args.base_seed,
            verbose=True,
        )
        s = summarize_match(sanity)
        # Expected: every hand has margin = 0 (same actions chosen identically)
        # because mark_ev = 1 * (2*p_make - 1) is monotone in p_make at bid=30.
        n_zero = int(sum(1 for h in sanity.hands if h.margin == 0))
        s["n_margin_zero"] = n_zero
        s["sanity_pass"] = int(n_zero == sanity.n_hands)
        print(
            f"  -> {n_zero}/{sanity.n_hands} hands had margin=0; "
            f"sanity {'PASS' if n_zero == sanity.n_hands else 'FAIL'}",
            flush=True,
        )
        write_csv(out_dir / "mark_ev_pmake_sanity.csv", [s])

    # ----- Verdict -----
    elapsed = time.time() - t_start

    # Find pairing(s) where CI excludes zero
    decisive_pairings = [s for s in rr_summaries if s["ci_excludes_zero"]]
    if decisive_pairings:
        # Best-margin (in absolute value) pairing
        winner = max(decisive_pairings, key=lambda s: abs(s["mean_margin"]))
        if winner["mean_margin"] > 0:
            verdict = (
                f"WINNER: Lens({winner['utility_a']}) beats Lens({winner['utility_b']}) "
                f"by {winner['mean_margin']:+.2f} pts/hand (95% CI "
                f"[{winner['margin_ci_lo_95']:+.2f}, {winner['margin_ci_hi_95']:+.2f}])."
            )
        else:
            verdict = (
                f"WINNER: Lens({winner['utility_b']}) beats Lens({winner['utility_a']}) "
                f"by {-winner['mean_margin']:+.2f} pts/hand (95% CI "
                f"[{-winner['margin_ci_hi_95']:+.2f}, {-winner['margin_ci_lo_95']:+.2f}])."
            )
        verdict_kind = "decisive"
    else:
        verdict = (
            f"NO UTILITY DISTINGUISHABLY BEATS ANOTHER at "
            f"N={args.n_hands} hands paired-seed. All {len(ROUND_ROBIN_PAIRINGS)} "
            f"pairing 95% CIs span zero."
        )
        verdict_kind = "no_distinguishable_winner"

    print("\n=== VERDICT ===", flush=True)
    print(verdict, flush=True)

    summary = {
        "wave": "lens-v1",
        "bead": "t42-4ouu",
        "parent_bead": "t42-4zi6",
        "device": device,
        "checkpoint": str(ckpt),
        "n_hands_per_pairing": args.n_hands,
        "n_samples_per_decision": args.n_samples,
        "base_seed": args.base_seed,
        "round_robin_pairings": [list(p) for p in ROUND_ROBIN_PAIRINGS],
        "round_robin_summaries": rr_summaries,
        "sample_sweep_summaries": sweep_summaries,
        "wall_seconds": round(elapsed, 1),
        "verdict_text": verdict,
        "verdict_kind": verdict_kind,
        "n_decisive_pairings": len(decisive_pairings),
    }
    summary_path = out_dir.parent / "summary.json"
    with summary_path.open("w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nWrote {summary_path}", flush=True)

    manifest = {
        "bead": "t42-4ouu",
        "parent_bead": "t42-4zi6",
        "wave": "lens-v1",
        "checkpoint": str(ckpt),
        "checkpoint_sha256": hashlib.sha256(ckpt.read_bytes()).hexdigest(),
        "device": device,
        "fp_dtype": "fp32",
        "n_samples_per_decision": args.n_samples,
        "n_hands_per_pairing": args.n_hands,
        "base_seed": args.base_seed,
        "wall_seconds": round(elapsed, 1),
        "command": (
            f"python w42/lens_v1/round_robin.py --n-hands {args.n_hands} "
            f"--n-samples {args.n_samples} --device {device} "
            f"--base-seed {args.base_seed}"
        ),
        "artifacts": {
            "round_robin_csv": str(rr_csv),
            "per_hand_margins_csv": str(per_hand_csv),
            "sample_sweep_csv": str(out_dir / "sample_sweep.csv"),
            "mark_ev_pmake_sanity_csv": str(out_dir / "mark_ev_pmake_sanity.csv"),
            "summary_json": str(summary_path),
        },
    }
    manifest_path = out_dir.parent / "manifest.json"
    with manifest_path.open("w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"Wrote {manifest_path}", flush=True)

    print(f"\nTotal wall time: {elapsed:.1f}s ({elapsed/60:.1f} min)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
