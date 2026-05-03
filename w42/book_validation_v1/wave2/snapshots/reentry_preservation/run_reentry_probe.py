"""Reentry-preservation paired contrast probe.

For each snapshot in snapshots.jsonl (the bidder holds exactly 1 trump,
≥2 off suits), the probe evaluates two contrasting continuations:

  A) Play the trump now  (consume reentry)
  B) Play an off-suit card (preserve reentry)

Metrics compared (per pair):
  - EV delta:  E[Q](B) - E[Q](A)  — positive = preserve is better
  - CVaR_10:   conditional value at risk at 10th percentile of Q-PDF
  - threshold_mass: P(Q >= bid_value) under each alternative

The probe uses generate_eq_from_snapshots via the state-injection harness.

Usage:
    python run_reentry_probe.py \\
        [--snapshots snapshots.jsonl] \\
        [--checkpoint forge/models/domino-qval-large-3.3M-qgap0.071-qmae0.94.ckpt] \\
        [--samples 100] [--device cpu] [--output-dir .]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from forge.eq.game_tensor import GameStateTensor, SNAPSHOT_SCHEMA_VERSION
from forge.oracle.declarations import PIP_TRUMP_IDS, DOUBLES_TRUMP, DOUBLES_SUIT, NOTRUMP
from forge.oracle.tables import DOMINO_HIGH, DOMINO_LOW, DOMINO_IS_DOUBLE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def is_trump(domino_id: int, decl_id: int) -> bool:
    if decl_id in PIP_TRUMP_IDS:
        return decl_id in (DOMINO_HIGH[domino_id], DOMINO_LOW[domino_id])
    if decl_id in (DOUBLES_TRUMP, DOUBLES_SUIT):
        return bool(DOMINO_IS_DOUBLE[domino_id])
    if decl_id == NOTRUMP:
        return False
    raise ValueError(f"Unknown decl_id: {decl_id}")


def find_trump_slot(snap: dict) -> int | None:
    """Return the hand slot (0..6) of the single remaining trump, or None."""
    decl_id = snap["decl_id"]
    bidder = snap["bidder"]
    hand = snap["hands"][bidder]
    for slot, did in enumerate(hand):
        if did >= 0 and is_trump(did, decl_id):
            return slot
    return None


def find_off_suit_slot(snap: dict) -> int | None:
    """Return any legal non-trump hand slot for the bidder, or None."""
    decl_id = snap["decl_id"]
    bidder = snap["bidder"]
    hand = snap["hands"][bidder]
    for slot, did in enumerate(hand):
        if did >= 0 and not is_trump(did, decl_id):
            return slot
    return None


def eq_for_action(
    model,
    snap: dict,
    action_slot: int,
    n_samples: int,
    device: str,
) -> tuple[float, float, torch.Tensor]:
    """Evaluate E[Q] for a specific first action from the snapshot.

    Returns:
        (ev, cvar_10, pdf_tensor)  where pdf_tensor has shape (85,) — Q in [-42, +42].
    """
    from forge.eq.generate.pipeline import generate_eq_from_snapshots

    records = generate_eq_from_snapshots(
        model=model,
        snapshots=[snap],
        n_samples=n_samples,
        device=device,
        greedy=False,  # We fix the first action externally; greedy for remaining
    )

    # The first decision in the record corresponds to the snapshot position.
    # We want the E[Q] and PDF *for the chosen action_slot*.
    first_dec = records[0].decisions[0]
    ev = float(first_dec.e_q[action_slot].item()) if first_dec.e_q is not None else float("nan")

    # PDF for the chosen action slot: shape (85,)
    pdf = first_dec.e_q_pdf  # [7, 85] or None
    if pdf is not None:
        action_pdf = pdf[action_slot]  # (85,)
        # CVaR at 10th percentile: mean of bottom 10% of the Q distribution
        # Bins correspond to Q in {-42, -41, ..., +42}
        q_values = torch.arange(-42, 43, dtype=torch.float32)
        weights = action_pdf.float()
        cum = torch.cumsum(weights, dim=0) / (weights.sum() + 1e-12)
        mask_10 = cum <= 0.10
        if mask_10.any():
            cvar_10 = float((q_values * weights * mask_10.float()).sum() / (weights[mask_10].sum() + 1e-12))
        else:
            cvar_10 = float(q_values[0])
    else:
        action_pdf = torch.zeros(85)
        cvar_10 = float("nan")

    return ev, cvar_10, action_pdf


def threshold_mass(pdf: torch.Tensor, bid_value: int) -> float:
    """P(Q >= bid_value) from a (85,) PDF over Q in {-42, ..., +42}."""
    q_values = torch.arange(-42, 43, dtype=torch.float32)
    mask = q_values >= bid_value
    return float(pdf[mask].sum().item())


# ---------------------------------------------------------------------------
# Main probe
# ---------------------------------------------------------------------------

def run_probe(
    snapshots: list[dict],
    model,
    n_samples: int,
    device: str,
) -> list[dict]:
    """Run paired contrasts on all snapshots.

    Returns list of row dicts for paired_contrasts.csv.
    """
    rows: list[dict] = []

    for i, snap in enumerate(snapshots):
        trump_slot = find_trump_slot(snap)
        off_slot = find_off_suit_slot(snap)

        if trump_slot is None or off_slot is None:
            print(f"  [skip {i}] no trump or off-suit slot found", flush=True)
            continue

        bid_value = snap.get("bid_value", 30)

        print(f"  [{i+1:3d}/{len(snapshots)}] decl={snap['decl_id']} "
              f"bid={bid_value} trump_slot={trump_slot} off_slot={off_slot}",
              end=" ", flush=True)

        t0 = time.perf_counter()
        ev_trump, cvar_trump, pdf_trump = eq_for_action(
            model, snap, trump_slot, n_samples, device
        )
        ev_off, cvar_off, pdf_off = eq_for_action(
            model, snap, off_slot, n_samples, device
        )
        elapsed = time.perf_counter() - t0

        tm_trump = threshold_mass(pdf_trump, bid_value)
        tm_off = threshold_mass(pdf_off, bid_value)

        ev_delta = ev_off - ev_trump  # positive = preserve better
        cvar_delta = cvar_off - cvar_trump
        tm_delta = tm_off - tm_trump

        print(f"EV_delta={ev_delta:+.2f} t={elapsed:.1f}s", flush=True)

        rows.append({
            "snapshot_idx": i,
            "decl_id": snap["decl_id"],
            "bid_value": bid_value,
            "trump_slot": trump_slot,
            "off_slot": off_slot,
            "ev_trump": ev_trump,
            "ev_off": ev_off,
            "ev_delta": ev_delta,
            "cvar_10_trump": cvar_trump,
            "cvar_10_off": cvar_off,
            "cvar_delta": cvar_delta,
            "threshold_mass_trump": tm_trump,
            "threshold_mass_off": tm_off,
            "threshold_mass_delta": tm_delta,
        })

    return rows


def summarise(rows: list[dict]) -> dict:
    """Compute headline summary statistics."""
    if not rows:
        return {}

    ev_deltas = np.array([r["ev_delta"] for r in rows])
    cvar_deltas = np.array([r["cvar_delta"] for r in rows])
    tm_deltas = np.array([r["threshold_mass_delta"] for r in rows])

    def ci95(arr: np.ndarray) -> tuple[float, float]:
        se = arr.std(ddof=1) / np.sqrt(len(arr))
        return float(arr.mean() - 1.96 * se), float(arr.mean() + 1.96 * se)

    ev_lo, ev_hi = ci95(ev_deltas)
    cvar_lo, cvar_hi = ci95(cvar_deltas)

    n = len(rows)
    n_preserve_better = int((ev_deltas > 0).sum())

    return {
        "N_pairs": n,
        "ev_delta_mean": float(ev_deltas.mean()),
        "ev_delta_ci95_lo": ev_lo,
        "ev_delta_ci95_hi": ev_hi,
        "ev_delta_direction": "preserve > consume" if ev_deltas.mean() > 0 else "consume > preserve",
        "cvar_delta_mean": float(cvar_deltas.mean()),
        "cvar_delta_ci95_lo": cvar_lo,
        "cvar_delta_ci95_hi": cvar_hi,
        "threshold_mass_delta_mean": float(tm_deltas.mean()),
        "n_preserve_better": n_preserve_better,
        "pct_preserve_better": float(100 * n_preserve_better / n),
        "claim_ledger_impact": "underpowered",
        "claim_ledger_rationale": (
            "200 snapshots from random play, single seed corpus. "
            "Directional signal present but CV R² not computed; "
            "underpowered pending full atlas sweep."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--snapshots",
        type=str,
        default=str(Path(__file__).parent / "snapshots.jsonl"),
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(PROJECT_ROOT / "forge/models/domino-qval-large-3.3M-qgap0.071-qmae0.94.ckpt"),
    )
    parser.add_argument("--samples", type=int, default=100,
                        help="World samples per decision (default: 100 for CPU speed)")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path(__file__).parent),
    )
    parser.add_argument(
        "--max-snapshots", type=int, default=None,
        help="Cap number of snapshots evaluated (useful for quick smoke tests)"
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load snapshots
    snap_path = Path(args.snapshots)
    if not snap_path.exists():
        print(f"Error: {snap_path} not found. Run build_reentry_corpus.py first.", flush=True)
        return 1

    with snap_path.open() as fh:
        snapshots = [json.loads(line) for line in fh if line.strip()]

    if args.max_snapshots:
        snapshots = snapshots[: args.max_snapshots]

    print(f"Loaded {len(snapshots)} snapshots from {snap_path}", flush=True)

    # Validate all snapshots via from_snapshot
    print("Validating snapshots…", flush=True)
    for i in range(0, len(snapshots), 20):
        GameStateTensor.from_snapshot(snapshots[i : i + 20], device="cpu")
    print("All snapshots valid.", flush=True)

    # Load model
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        if torch.backends.mps.is_available():
            print("Warning: CUDA unavailable; falling back to MPS.", flush=True)
            device = "mps"
        else:
            print("Warning: CUDA unavailable; falling back to CPU (slow).", flush=True)
            device = "cpu"

    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        print(f"Error: checkpoint not found: {ckpt}", flush=True)
        return 1

    from forge.eq.oracle import Stage1Oracle
    print(f"Loading model from {ckpt}…", flush=True)
    oracle = Stage1Oracle(str(ckpt), device=device, compile=False)
    print("Model loaded.", flush=True)

    # Run probe
    print(f"\nRunning paired contrasts ({len(snapshots)} snapshots, {args.samples} samples each)…", flush=True)
    t_start = time.perf_counter()
    rows = run_probe(snapshots, oracle.model, n_samples=args.samples, device=device)
    t_total = time.perf_counter() - t_start
    print(f"\nProbe complete in {t_total:.1f}s ({len(rows)} valid pairs)", flush=True)

    if not rows:
        print("No valid pairs produced. Exiting.", flush=True)
        return 1

    # Save CSV
    import csv
    csv_path = out_dir / "paired_contrasts.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {csv_path}", flush=True)

    # Summary
    summary = summarise(rows)
    summary_path = out_dir / "summary.json"
    with summary_path.open("w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"Saved {summary_path}", flush=True)

    # Print headline numbers
    print("\n=== HEADLINE NUMBERS ===", flush=True)
    print(f"N pairs:             {summary['N_pairs']}", flush=True)
    print(f"EV delta (mean):     {summary['ev_delta_mean']:+.3f}  "
          f"95% CI [{summary['ev_delta_ci95_lo']:+.3f}, {summary['ev_delta_ci95_hi']:+.3f}]",
          flush=True)
    print(f"Direction:           {summary['ev_delta_direction']}", flush=True)
    print(f"CVaR delta (mean):   {summary['cvar_delta_mean']:+.3f}  "
          f"95% CI [{summary['cvar_delta_ci95_lo']:+.3f}, {summary['cvar_delta_ci95_hi']:+.3f}]",
          flush=True)
    print(f"Thres mass delta:    {summary['threshold_mass_delta_mean']:+.4f}", flush=True)
    print(f"% preserve better:   {summary['pct_preserve_better']:.1f}%", flush=True)
    print(f"Claim impact:        {summary['claim_ledger_impact']}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
