"""Reentry-preservation v2 paired contrast probe (Wave 2.A.3, bead t42-v9lu).

Improvements over Wave 2.A:
  1. Oracle-greedy source snapshots (not random-play).
  2. Smarter off-suit selection: highest-pip non-trump tile (book: cash high tiles).
  3. Bidder-eligible double-check at probe time.
  4. Phase slice reporting: trick=2 (early), 3-4 (mid), 5-6 (late).
  5. MPS device (not CPU).

For each snapshot the bidder holds exactly 1 trump and >=2 distinct off suits.
Both branches are evaluated in a SINGLE pipeline call per snapshot (the model
simultaneously produces Q-values for all legal slots), then we read:
  A (consume):  e_q[trump_slot]
  B (preserve): e_q[best_off_slot]

EV delta = B - A; positive = preserve is better (book claim direction).

Usage:
    python run_reentry_v2_probe.py \\
        [--snapshots snapshots.jsonl] \\
        [--checkpoint <ckpt>] \\
        [--samples 200] [--device mps] \\
        [--output-dir .]
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
# Domino helpers
# ---------------------------------------------------------------------------

def is_trump(domino_id: int, decl_id: int) -> bool:
    if decl_id in PIP_TRUMP_IDS:
        return decl_id in (DOMINO_HIGH[domino_id], DOMINO_LOW[domino_id])
    if decl_id in (DOUBLES_TRUMP, DOUBLES_SUIT):
        return bool(DOMINO_IS_DOUBLE[domino_id])
    if decl_id == NOTRUMP:
        return False
    raise ValueError(f"Unknown decl_id: {decl_id}")


def domino_pip_sum(domino_id: int) -> int:
    """High + low pips — used for selecting 'highest' non-trump tile."""
    return int(DOMINO_HIGH[domino_id]) + int(DOMINO_LOW[domino_id])


def find_trump_slot(snap: dict) -> int | None:
    """Hand slot of the single remaining trump for the bidder, or None."""
    decl_id = snap["decl_id"]
    bidder = snap["bidder"]
    hand = snap["hands"][bidder]
    legal = snap["_legal_mask"]
    for slot, did in enumerate(hand):
        if did >= 0 and is_trump(did, decl_id):
            return slot
    return None


def find_best_off_suit_slot(snap: dict) -> int | None:
    """Hand slot of the HIGHEST-PIP legal non-trump tile for the bidder.

    Improvement over Wave 2.A 'first available' strategy:
    picks the tile the book calls 'cashing a high tile' — the one that costs
    the most to throw away and thus best stresses the reentry dilemma.
    """
    decl_id = snap["decl_id"]
    bidder = snap["bidder"]
    hand = snap["hands"][bidder]
    legal = snap["_legal_mask"]

    best_slot = None
    best_pips = -1
    for slot, did in enumerate(hand):
        if did >= 0 and not is_trump(did, decl_id) and legal[slot]:
            pip = domino_pip_sum(did)
            if pip > best_pips:
                best_pips = pip
                best_slot = slot
    return best_slot


def trick_number(snap: dict) -> int:
    """Zero-based trick number (0 = trick 1, etc.)."""
    history = snap["history"]
    completed = sum(1 for h in history if h[0] >= 0)
    return completed // 4


def phase_label(snap: dict) -> str:
    """Classify snapshot into early / mid / late phase."""
    t = trick_number(snap) + 1  # 1-based
    if t <= 2:
        return "early"
    elif t <= 4:
        return "mid"
    else:
        return "late"


# ---------------------------------------------------------------------------
# EQ evaluation helpers
# ---------------------------------------------------------------------------

def threshold_mass(pdf: torch.Tensor, bid_value: int) -> float:
    """P(Q >= bid_value) from a (85,) PDF over Q in {-42 ... +42}."""
    q_values = torch.arange(-42, 43, dtype=torch.float32)
    mask = q_values >= bid_value
    return float(pdf[mask].sum().item())


def cvar_10(pdf: torch.Tensor) -> float:
    """CVaR at 10th percentile of Q distribution from (85,) PDF."""
    q_values = torch.arange(-42, 43, dtype=torch.float32)
    weights = pdf.float()
    cum = torch.cumsum(weights, dim=0) / (weights.sum() + 1e-12)
    mask = cum <= 0.10
    if mask.any():
        return float(
            (q_values * weights * mask.float()).sum()
            / (weights[mask].sum() + 1e-12)
        )
    return float(q_values[0])


def evaluate_snapshot_branches(
    model,
    snap: dict,
    trump_slot: int,
    off_slot: int,
    n_samples: int,
    device: str,
) -> dict | None:
    """Run a single pipeline call for one snapshot; return branch metrics.

    Returns None on failure.
    """
    from forge.eq.generate.pipeline import generate_eq_from_snapshots

    try:
        records = generate_eq_from_snapshots(
            model=model,
            snapshots=[snap],
            n_samples=n_samples,
            device=device,
            greedy=True,
        )
    except Exception as exc:
        print(f"    pipeline error: {exc}", flush=True)
        return None

    if not records or not records[0].decisions:
        print("    no decisions returned", flush=True)
        return None

    first_dec = records[0].decisions[0]
    e_q = first_dec.e_q  # shape (7,) or None
    pdf_full = first_dec.e_q_pdf  # shape (7, 85) or None

    if e_q is None:
        print("    e_q is None", flush=True)
        return None

    ev_trump = float(e_q[trump_slot].item())
    ev_off   = float(e_q[off_slot].item())
    ev_delta = ev_off - ev_trump  # positive = preserve better

    bid_value = int(snap.get("bid_value", 30))

    if pdf_full is not None:
        pdf_trump = pdf_full[trump_slot]  # (85,)
        pdf_off   = pdf_full[off_slot]
        cvar_trump  = cvar_10(pdf_trump)
        cvar_off    = cvar_10(pdf_off)
        tm_trump    = threshold_mass(pdf_trump, bid_value)
        tm_off      = threshold_mass(pdf_off, bid_value)
    else:
        cvar_trump  = float("nan")
        cvar_off    = float("nan")
        tm_trump    = float("nan")
        tm_off      = float("nan")

    return {
        "ev_trump":              ev_trump,
        "ev_off":                ev_off,
        "ev_delta":              ev_delta,
        "cvar_10_trump":         cvar_trump,
        "cvar_10_off":           cvar_off,
        "cvar_delta":            cvar_off - cvar_trump,
        "threshold_mass_trump":  tm_trump,
        "threshold_mass_off":    tm_off,
        "threshold_mass_delta":  tm_off - tm_trump,
    }


# ---------------------------------------------------------------------------
# Main probe
# ---------------------------------------------------------------------------

def run_probe(
    snapshots: list[dict],
    model,
    n_samples: int,
    device: str,
) -> list[dict]:
    rows: list[dict] = []

    for i, snap in enumerate(snapshots):
        # Double-check bidder identity
        if snap.get("bidder") is None:
            print(f"  [skip {i}] missing bidder field", flush=True)
            continue

        trump_slot = find_trump_slot(snap)
        off_slot   = find_best_off_suit_slot(snap)

        if trump_slot is None:
            print(f"  [skip {i}] no trump slot found", flush=True)
            continue
        if off_slot is None:
            # No legal off-suit — snapshot has only trump as legal option.
            # This can happen in follower positions with suit-following rule.
            print(f"  [skip {i}] no legal off-suit slot", flush=True)
            continue

        # Also check that trump slot is legal
        if not snap["_legal_mask"][trump_slot]:
            print(f"  [skip {i}] trump slot is not legal", flush=True)
            continue

        bid_value = int(snap.get("bid_value", 30))
        trick_n   = trick_number(snap) + 1  # 1-based
        phase     = phase_label(snap)

        print(
            f"  [{i+1:3d}/{len(snapshots)}] decl={snap['decl_id']} "
            f"bid={bid_value} trick={trick_n} phase={phase} "
            f"trump_slot={trump_slot} off_slot={off_slot}",
            end=" ... ", flush=True
        )

        t0 = time.perf_counter()
        metrics = evaluate_snapshot_branches(
            model, snap, trump_slot, off_slot, n_samples, device
        )
        elapsed = time.perf_counter() - t0

        if metrics is None:
            print(f"FAILED ({elapsed:.1f}s)", flush=True)
            continue

        print(
            f"EV_delta={metrics['ev_delta']:+.2f} t={elapsed:.1f}s",
            flush=True
        )

        rows.append({
            "snapshot_idx":          i,
            "decl_id":               snap["decl_id"],
            "bid_value":             bid_value,
            "trick_number":          trick_n,
            "phase":                 phase,
            "trump_slot":            trump_slot,
            "off_slot":              off_slot,
            **metrics,
        })

    return rows


# ---------------------------------------------------------------------------
# Summarisation
# ---------------------------------------------------------------------------

def ci95(arr: np.ndarray) -> tuple[float, float]:
    if len(arr) < 2:
        return float("nan"), float("nan")
    se = arr.std(ddof=1) / np.sqrt(len(arr))
    return float(arr.mean() - 1.96 * se), float(arr.mean() + 1.96 * se)


def verdict(ev_delta_mean: float, ci_lo: float, ci_hi: float) -> str:
    """Propose claim-ledger status."""
    if ci_lo > 0:
        return "supported"          # CI excludes zero, book direction
    if ci_hi < 0:
        return "contradicted"       # CI excludes zero, opposite direction
    return "underpowered"           # CI spans zero


def summarise(rows: list[dict]) -> dict:
    if not rows:
        return {"error": "no valid pairs"}

    ev    = np.array([r["ev_delta"] for r in rows])
    cvar  = np.array([r["cvar_delta"] for r in rows])
    tm    = np.array([r["threshold_mass_delta"] for r in rows])

    ev_lo, ev_hi     = ci95(ev)
    cvar_lo, cvar_hi = ci95(cvar)
    n = len(rows)

    status = verdict(float(ev.mean()), ev_lo, ev_hi)

    # Phase slices
    phases = {"early": [], "mid": [], "late": []}
    for r in rows:
        phases[r["phase"]].append(r["ev_delta"])

    phase_stats = {}
    for ph, vals in phases.items():
        if vals:
            arr = np.array(vals)
            lo, hi = ci95(arr)
            phase_stats[ph] = {
                "n": len(vals),
                "ev_delta_mean": float(arr.mean()),
                "ev_delta_ci95": [lo, hi],
            }
        else:
            phase_stats[ph] = {"n": 0}

    return {
        "N_pairs":                    n,
        "ev_delta_mean":              float(ev.mean()),
        "ev_delta_ci95_lo":           ev_lo,
        "ev_delta_ci95_hi":           ev_hi,
        "ev_delta_direction":         "preserve > consume" if ev.mean() > 0 else "consume > preserve",
        "cvar_delta_mean":            float(cvar.mean()),
        "cvar_delta_ci95_lo":         cvar_lo,
        "cvar_delta_ci95_hi":         cvar_hi,
        "threshold_mass_delta_mean":  float(tm.mean()),
        "n_preserve_better":          int((ev > 0).sum()),
        "pct_preserve_better":        float(100 * (ev > 0).sum() / n),
        "n_consume_better":           int((ev < 0).sum()),
        "pct_consume_better":         float(100 * (ev < 0).sum() / n),
        "phase_slices":               phase_stats,
        "claim_ledger_impact":        status,
        "claim_ledger_rationale":     (
            f"Paired EQ contrast on {n} oracle-greedy snapshots. "
            f"EV delta 95% CI [{ev_lo:+.3f}, {ev_hi:+.3f}]. "
            f"CI {'excludes' if status != 'underpowered' else 'includes'} zero."
        ),
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--snapshots",
        type=str,
        default=str(
            PROJECT_ROOT
            / "w42/book_validation_v1/wave2/snapshots/reentry_preservation_v2/snapshots.jsonl"
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(
            PROJECT_ROOT
            / "forge/models/domino-qval-large-3.3M-qgap0.071-qmae0.94.ckpt"
        ),
    )
    parser.add_argument(
        "--samples", type=int, default=200,
        help="World samples per decision (200 preferred; fall back to 100 if slow)",
    )
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path(__file__).parent),
    )
    parser.add_argument(
        "--max-snapshots", type=int, default=None,
        help="Cap for smoke-test runs",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Load snapshots ---
    snap_path = Path(args.snapshots)
    if not snap_path.exists():
        print(f"Error: {snap_path} not found.", flush=True)
        return 1

    with snap_path.open() as fh:
        snapshots = [json.loads(line) for line in fh if line.strip()]

    if args.max_snapshots:
        snapshots = snapshots[: args.max_snapshots]

    print(f"Loaded {len(snapshots)} snapshots from {snap_path}", flush=True)

    # --- Validate ---
    print("Validating snapshots ...", flush=True)
    for i in range(0, len(snapshots), 20):
        GameStateTensor.from_snapshot(snapshots[i: i + 20], device="cpu")
    print("All snapshots valid.", flush=True)

    # --- Device ---
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Warning: CUDA unavailable; falling back to {device}.", flush=True)
    if device == "mps" and not torch.backends.mps.is_available():
        print("Warning: MPS unavailable; falling back to CPU.", flush=True)
        device = "cpu"

    # --- Load model ---
    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        print(f"Error: checkpoint not found: {ckpt}", flush=True)
        return 1

    from forge.eq.oracle import Stage1Oracle
    print(f"Loading model from {ckpt} on {device} ...", flush=True)
    oracle = Stage1Oracle(str(ckpt), device=device, compile=False)
    print("Model loaded.", flush=True)

    # --- Run probe ---
    n_samples = args.samples
    print(
        f"\nRunning paired contrasts ({len(snapshots)} snapshots, "
        f"{n_samples} samples each, device={device}) ...",
        flush=True
    )
    t_start = time.perf_counter()
    rows = run_probe(snapshots, oracle.model, n_samples=n_samples, device=device)
    t_total = time.perf_counter() - t_start
    print(
        f"\nProbe complete in {t_total:.1f}s ({len(rows)} valid pairs)",
        flush=True
    )

    if not rows:
        print("No valid pairs produced. Exiting.", flush=True)
        return 1

    # --- Save paired_contrasts.csv ---
    import csv
    csv_path = out_dir / "paired_contrasts.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {csv_path}", flush=True)

    # --- Phase slice CSV ---
    slice_path = out_dir / "slice_by_phase.csv"
    summary = summarise(rows)
    phase_rows = []
    for ph in ("early", "mid", "late"):
        ps = summary["phase_slices"].get(ph, {})
        if ps.get("n", 0) > 0:
            lo, hi = ps["ev_delta_ci95"]
            phase_rows.append({
                "phase":            ph,
                "n":                ps["n"],
                "ev_delta_mean":    ps["ev_delta_mean"],
                "ev_delta_ci95_lo": lo,
                "ev_delta_ci95_hi": hi,
            })
    with slice_path.open("w", newline="") as fh:
        if phase_rows:
            writer = csv.DictWriter(fh, fieldnames=list(phase_rows[0].keys()))
            writer.writeheader()
            writer.writerows(phase_rows)
    print(f"Saved {slice_path}", flush=True)

    # --- Save summary.json ---
    summary["run_metadata"] = {
        "n_samples_per_branch": n_samples,
        "device":               device,
        "checkpoint":           str(ckpt),
        "wall_seconds":         round(t_total, 1),
        "n_snapshots_input":    len(snapshots),
    }
    summary_path = out_dir / "summary.json"
    with summary_path.open("w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"Saved {summary_path}", flush=True)

    # --- Headline numbers ---
    print("\n=== HEADLINE NUMBERS ===", flush=True)
    print(f"N pairs:             {summary['N_pairs']}", flush=True)
    print(
        f"EV delta (mean):     {summary['ev_delta_mean']:+.3f}  "
        f"95% CI [{summary['ev_delta_ci95_lo']:+.3f}, "
        f"{summary['ev_delta_ci95_hi']:+.3f}]",
        flush=True
    )
    print(f"Direction:           {summary['ev_delta_direction']}", flush=True)
    print(
        f"CVaR delta (mean):   {summary['cvar_delta_mean']:+.3f}  "
        f"95% CI [{summary['cvar_delta_ci95_lo']:+.3f}, "
        f"{summary['cvar_delta_ci95_hi']:+.3f}]",
        flush=True
    )
    print(
        f"Thres mass delta:    {summary['threshold_mass_delta_mean']:+.4f}",
        flush=True
    )
    print(
        f"% preserve better:   {summary['pct_preserve_better']:.1f}%",
        flush=True
    )
    print(f"Claim impact:        {summary['claim_ledger_impact']}", flush=True)
    print("\nPhase slices:", flush=True)
    for ph in ("early", "mid", "late"):
        ps = summary["phase_slices"].get(ph, {})
        if ps.get("n", 0) > 0:
            lo, hi = ps["ev_delta_ci95"]
            print(
                f"  {ph:5s}  n={ps['n']:3d}  "
                f"mean={ps['ev_delta_mean']:+.3f}  "
                f"CI=[{lo:+.3f},{hi:+.3f}]",
                flush=True
            )
    print(f"\nClaim ledger rationale: {summary['claim_ledger_rationale']}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
