"""Wave 4.0 — utility-argmax divergence on ch05-void-creation-follow snapshots.

For each of 500 snapshots:
  1. Reconstruct GameStateTensor.from_snapshot.
  2. Determine to-act player and enumerate ALL legal actions (engine helper).
  3. Run forge pipeline once -> e_q[7], e_q_pdf[7,85] for the to-act player.
  4. Compute 5 utilities per legal action:
       EV          = e_q[a]
       p_make      = sum of pdf bins above bid-aware threshold (forge convention)
       mark_ev     = mark_multiplier * (2 * p_make - 1); mm = max(1, bid//42)
       CVaR_10     = mean of bottom 10% of Q distribution
       robust_q25  = 25th percentile of Q distribution
  5. Argmax over legal actions per utility.
  6. Compute per-utility-pair disagreement rates (5x5) with bootstrap CIs.
  7. Restrict to original (void, preserve) action pair -> confusion table.

Usage:
    python analyze.py [--snapshots PATH] [--checkpoint PATH] [--samples 100] \\
                      [--device mps] [--output-dir .] [--max-snapshots N] \\
                      [--batch-size 50]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from forge.eq.game_tensor import GameStateTensor
from forge.eq.generate.actions import contract_threshold_bins
from forge.oracle.tables import (
    DOMINO_HIGH,
    DOMINO_LOW,
    DOMINO_COUNT_POINTS,
    led_suit_for_lead_domino,
)

# Reuse void/preserve identification from the reference probe so we get the
# *exact* action pair Wave 3.0 measured EV=+0.77 / p_make≈0 on.
REF_PROBE = (
    PROJECT_ROOT
    / "w42/book_validation_v1/wave2/probes/t42-z31l_void_creation_follow"
    / "run_void_creation_follow_probe.py"
)
sys.path.insert(0, str(REF_PROBE.parent))
from run_void_creation_follow_probe import (  # noqa: E402
    find_void_and_preserve_slots,
    infer_current_player,
)

UTILITIES = ["ev", "p_make", "mark_ev", "cvar_10", "robust_q25"]
EQ_BIN_COUNT = 85
Q_VALUES = np.arange(-42, 43, dtype=np.float32)  # length 85


# ---------------------------------------------------------------------------
# Per-action utilities from a single Q-distribution
# ---------------------------------------------------------------------------

def utilities_from_pdf(
    pdf: np.ndarray,           # [85]  P(Q=q | action)
    e_q: float,
    bid_value: int,
    is_offense: bool,
) -> dict[str, float]:
    """Compute the 5 utility scalars from a single action's Q-pdf.

    Conventions (matched to forge's bid-aware logic):
      offense_bin = 2 * bid     -> Q >= 2*bid - 42 (e.g. bid=30 -> Q >= 18)
      defense_bin = 85 - 2*bid  -> Q >= 43 - 2*bid (e.g. bid=30 -> Q >= -17)
    PDF is expressed from the model's "current player perspective"; offense
    chooses higher Q, defense chooses lower Q (for them) which translates to
    higher P(Q >= defense_threshold) under the forge convention.
    """
    contract = 42 if bid_value == 84 else bid_value
    if is_offense:
        threshold_bin = max(0, min(EQ_BIN_COUNT - 1, 2 * contract))
    else:
        threshold_bin = max(0, min(EQ_BIN_COUNT - 1, EQ_BIN_COUNT - 2 * contract))

    pdf_norm = pdf / max(pdf.sum(), 1e-12)

    p_make = float(pdf_norm[threshold_bin:].sum())

    mm = max(1, bid_value // 42)
    mark_ev = mm * (2 * p_make - 1)

    # CVaR_10 = mean of the lower-tail (cumulative <= 10%) Q values.
    cum = np.cumsum(pdf_norm)
    lower_mask = cum <= 0.10
    if lower_mask.any():
        w = pdf_norm * lower_mask
        wsum = w.sum()
        cvar_10 = float((Q_VALUES * w).sum() / wsum) if wsum > 1e-12 else float(Q_VALUES[0])
    else:
        # Smallest bin already exceeds 10% -- use the smallest bin as CVaR.
        # (defines worst-case tail at the lowest mass-bearing bin)
        first_idx = int(np.argmax(pdf_norm > 0)) if (pdf_norm > 0).any() else 0
        cvar_10 = float(Q_VALUES[first_idx])

    # robust_q25 = 25th percentile of Q (smallest q such that cum(q) >= 0.25).
    q25_idx = int(np.searchsorted(cum, 0.25))
    q25_idx = min(q25_idx, EQ_BIN_COUNT - 1)
    robust_q25 = float(Q_VALUES[q25_idx])

    return {
        "ev": float(e_q),
        "p_make": p_make,
        "mark_ev": float(mark_ev),
        "cvar_10": cvar_10,
        "robust_q25": robust_q25,
    }


def argmax_over_legal(values: np.ndarray, legal: np.ndarray, is_offense: bool) -> int:
    """Pick best legal slot under a utility. Tie-break: lowest slot index.

    For all five utilities here, "better" = larger value:
      - EV: bidder wants high Q; setter (Q is from "their perspective" via
        forge's defense threshold) also wants high p_make from defense bin -- so
        we just pick the per-utility max in scalar form (the utilities already
        encode role).
    """
    masked = np.where(legal, values, -np.inf)
    if not np.isfinite(masked).any():
        return -1
    return int(np.argmax(masked))


# ---------------------------------------------------------------------------
# Bootstrap CI on a binary disagreement vector
# ---------------------------------------------------------------------------

def bootstrap_ci(disagree: np.ndarray, n_boot: int = 2000, seed: int = 42) -> tuple[float, float]:
    if disagree.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    n = disagree.size
    samples = rng.choice(disagree, size=(n_boot, n), replace=True)
    rates = samples.mean(axis=1)
    return float(np.percentile(rates, 2.5)), float(np.percentile(rates, 97.5))


# ---------------------------------------------------------------------------
# Main run
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> int:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    snap_path = Path(args.snapshots)
    ckpt_path = Path(args.checkpoint)

    if not snap_path.exists():
        print(f"Error: snapshots not found: {snap_path}", flush=True)
        return 1
    if not ckpt_path.exists():
        print(f"Error: checkpoint not found: {ckpt_path}", flush=True)
        return 1

    snapshots_raw = []
    with snap_path.open() as fh:
        for line in fh:
            if line.strip():
                snapshots_raw.append(json.loads(line))
    if args.max_snapshots:
        snapshots_raw = snapshots_raw[: args.max_snapshots]
    print(f"Loaded {len(snapshots_raw)} snapshots from {snap_path}", flush=True)

    # Validate snapshots can be reconstructed
    print("Validating snapshots can be reconstructed...", flush=True)
    valid: list[dict] = []
    bad = 0
    for snap in snapshots_raw:
        snap_clean = {k: v for k, v in snap.items() if not k.startswith("_")}
        try:
            GameStateTensor.from_snapshot([snap_clean], device="cpu")
            valid.append(snap)
        except Exception as e:
            print(f"  skip invalid: {e}", flush=True)
            bad += 1
    print(f"Valid: {len(valid)}  Bad: {bad}", flush=True)

    # Device selection
    device = args.device
    if device == "mps" and not torch.backends.mps.is_available():
        print("MPS unavailable; falling back to CPU.", flush=True)
        device = "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA unavailable; falling back to mps/cpu.", flush=True)
        device = "mps" if torch.backends.mps.is_available() else "cpu"

    from forge.eq.oracle import Stage1Oracle
    print(f"Loading model from {ckpt_path} on {device}...", flush=True)
    oracle = Stage1Oracle(str(ckpt_path), device=device, compile=False)
    print("Model loaded.", flush=True)

    from forge.eq.generate.pipeline import generate_eq_from_snapshots

    # ------------------------------------------------------------------
    # Pipeline call (batched in chunks) -> per-snapshot e_q & e_q_pdf
    # ------------------------------------------------------------------
    print(
        f"Running forge pipeline on {len(valid)} snapshots, "
        f"samples={args.samples}, batch={args.batch_size}...",
        flush=True,
    )
    t_start = time.perf_counter()

    per_snap_eq: list[np.ndarray] = []        # list of [7] arrays
    per_snap_pdf: list[np.ndarray] = []       # list of [7, 85] arrays
    per_snap_legal: list[np.ndarray] = []     # list of [7] bool arrays
    per_snap_to_act: list[int] = []           # current player per snapshot
    per_snap_skipped: list[bool] = [False] * len(valid)

    for chunk_start in range(0, len(valid), args.batch_size):
        chunk = valid[chunk_start : chunk_start + args.batch_size]
        chunk_clean = [
            {k: v for k, v in s.items() if not k.startswith("_")} for s in chunk
        ]
        try:
            records = generate_eq_from_snapshots(
                model=oracle.model,
                snapshots=chunk_clean,
                n_samples=args.samples,
                device=device,
                greedy=True,
            )
        except Exception as e:
            print(f"  chunk {chunk_start} failed ({e}); falling back to per-snapshot.", flush=True)
            records = []
            for snap_clean in chunk_clean:
                try:
                    rec = generate_eq_from_snapshots(
                        model=oracle.model,
                        snapshots=[snap_clean],
                        n_samples=args.samples,
                        device=device,
                        greedy=True,
                    )[0]
                    records.append(rec)
                except Exception as e2:
                    print(f"    single snapshot failed: {e2}", flush=True)
                    records.append(None)

        for i, rec in enumerate(records):
            global_idx = chunk_start + i
            if rec is None or not rec.decisions:
                per_snap_skipped[global_idx] = True
                per_snap_eq.append(np.full(7, np.nan, dtype=np.float32))
                per_snap_pdf.append(np.zeros((7, 85), dtype=np.float32))
                per_snap_legal.append(np.zeros(7, dtype=bool))
                per_snap_to_act.append(-1)
                continue
            d0 = rec.decisions[0]
            eq = d0.e_q.detach().cpu().float().numpy()
            pdf = (
                d0.e_q_pdf.detach().cpu().float().numpy()
                if d0.e_q_pdf is not None
                else np.zeros((7, 85), dtype=np.float32)
            )
            legal = d0.legal_mask.detach().cpu().bool().numpy()
            per_snap_eq.append(eq)
            per_snap_pdf.append(pdf)
            per_snap_legal.append(legal)
            per_snap_to_act.append(int(d0.player))
        print(
            f"  chunk {chunk_start:4d}-{chunk_start + len(chunk):4d} done "
            f"({time.perf_counter() - t_start:.1f}s elapsed)",
            flush=True,
        )

    runtime_seconds = time.perf_counter() - t_start
    print(f"Pipeline done in {runtime_seconds:.1f}s", flush=True)

    # ------------------------------------------------------------------
    # Per-snapshot utility computation + argmax
    # ------------------------------------------------------------------
    rows: list[dict] = []
    n_skipped = 0
    n_no_legal = 0

    # Track per-utility argmax slot per snapshot for disagreement matrix later.
    util_argmax = {u: [] for u in UTILITIES}
    util_top_q = {u: [] for u in UTILITIES}

    void_subset_rows: list[dict] = []  # for void/preserve confusion

    for i, snap in enumerate(valid):
        if per_snap_skipped[i]:
            n_skipped += 1
            continue
        legal = per_snap_legal[i]
        if not legal.any():
            n_no_legal += 1
            continue

        bidder = snap["bidder"]
        bid_value = int(snap.get("bid_value", 30))
        decl_id = int(snap["decl_id"])
        to_act = per_snap_to_act[i]
        is_offense = (to_act % 2) == (bidder % 2)
        setter_player = bidder + 1  # canonical setter slot for record-keeping
        # The probe corpus has setter on team 1; setter_player here is just one
        # of the two team-1 seats (the to-act setter).
        setter_player = to_act if not is_offense else (bidder + 1) % 4

        eq = per_snap_eq[i]
        pdf = per_snap_pdf[i]

        # Compute per-action utility values for ALL 7 slots; mask later.
        per_action: dict[int, dict[str, float]] = {}
        for slot in range(7):
            if not legal[slot]:
                continue
            per_action[slot] = utilities_from_pdf(
                pdf[slot], eq[slot], bid_value, is_offense
            )

        n_legal = int(legal.sum())
        # Per-utility argmax slot
        argmax_slots: dict[str, int] = {}
        top_q: dict[str, float] = {}
        for u in UTILITIES:
            vals = np.full(7, -np.inf, dtype=np.float64)
            for slot, ud in per_action.items():
                vals[slot] = ud[u]
            best = argmax_over_legal(vals, legal, is_offense)
            argmax_slots[u] = best
            top_q[u] = float(vals[best]) if best >= 0 else float("nan")

        for u in UTILITIES:
            util_argmax[u].append(argmax_slots[u])
            util_top_q[u].append(top_q[u])

        rows.append({
            "snapshot_idx": i,
            "n_legal_actions": n_legal,
            "decl_id": decl_id,
            "bid_value": bid_value,
            "setter_player": setter_player,
            "to_act_player": to_act,
            "is_offense": int(is_offense),
            "action_argmax_ev": argmax_slots["ev"],
            "action_argmax_p_make": argmax_slots["p_make"],
            "action_argmax_mark_ev": argmax_slots["mark_ev"],
            "action_argmax_cvar_10": argmax_slots["cvar_10"],
            "action_argmax_robust_q25": argmax_slots["robust_q25"],
            "ev_top_q": top_q["ev"],
            "p_make_top_q": top_q["p_make"],
            "mark_ev_top_q": top_q["mark_ev"],
            "cvar_10_top_q": top_q["cvar_10"],
            "robust_q25_top_q": top_q["robust_q25"],
        })

        # Void / preserve subset (only when the original probe pair exists)
        void_slot, preserve_slot = find_void_and_preserve_slots(snap)
        if void_slot is not None and preserve_slot is not None:
            void_subset_rows.append({
                "snapshot_idx": i,
                "void_slot": void_slot,
                "preserve_slot": preserve_slot,
                **{f"argmax_{u}": argmax_slots[u] for u in UTILITIES},
            })

    n_kept = len(rows)
    print(
        f"Built per-snapshot table: kept={n_kept} skipped={n_skipped} no_legal={n_no_legal}",
        flush=True,
    )
    if not rows:
        print("No usable snapshots. Aborting.", flush=True)
        return 1

    # ------------------------------------------------------------------
    # Write per_snapshot_argmax.csv
    # ------------------------------------------------------------------
    per_snap_csv = out_dir / "per_snapshot_argmax.csv"
    with per_snap_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {per_snap_csv}", flush=True)

    # ------------------------------------------------------------------
    # Disagreement matrix (5x5)
    # ------------------------------------------------------------------
    arrs = {u: np.array(util_argmax[u], dtype=np.int64) for u in UTILITIES}
    n = len(rows)
    matrix_rows: list[dict] = []
    for ua in UTILITIES:
        for ub in UTILITIES:
            if ua == ub:
                disagree = np.zeros(n, dtype=np.int8)
            else:
                disagree = (arrs[ua] != arrs[ub]).astype(np.int8)
            count = int(disagree.sum())
            rate = float(disagree.mean())
            ci_lo, ci_hi = bootstrap_ci(disagree, n_boot=2000, seed=42)
            matrix_rows.append({
                "utility_a": ua,
                "utility_b": ub,
                "disagree_count": count,
                "n": n,
                "rate": rate,
                "ci_lo_95": ci_lo,
                "ci_hi_95": ci_hi,
            })
    dis_csv = out_dir / "disagreement_matrix.csv"
    with dis_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(matrix_rows[0].keys()))
        writer.writeheader()
        writer.writerows(matrix_rows)
    print(f"Wrote {dis_csv}", flush=True)

    # ------------------------------------------------------------------
    # Void / preserve confusion
    # ------------------------------------------------------------------
    confusion_rows: list[dict] = []
    n_subset = len(void_subset_rows)
    for u in UTILITIES:
        n_void = n_preserve = n_neither = 0
        for r in void_subset_rows:
            am = r[f"argmax_{u}"]
            if am == r["void_slot"]:
                n_void += 1
            elif am == r["preserve_slot"]:
                n_preserve += 1
            else:
                n_neither += 1
        confusion_rows.append({
            "utility": u,
            "n_void_subset": n_subset,
            "argmax_eq_void": n_void,
            "argmax_eq_preserve": n_preserve,
            "argmax_eq_neither": n_neither,
            "pct_void": (100.0 * n_void / n_subset) if n_subset else float("nan"),
            "pct_preserve": (100.0 * n_preserve / n_subset) if n_subset else float("nan"),
            "pct_neither": (100.0 * n_neither / n_subset) if n_subset else float("nan"),
        })
    conf_csv = out_dir / "void_subset_confusion.csv"
    with conf_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(confusion_rows[0].keys()))
        writer.writeheader()
        writer.writerows(confusion_rows)
    print(f"Wrote {conf_csv}", flush=True)

    # ------------------------------------------------------------------
    # Headline numbers + verdict gate
    # ------------------------------------------------------------------
    ev_pmake_disagree = (arrs["ev"] != arrs["p_make"]).astype(np.int8)
    ev_pmake_rate = float(ev_pmake_disagree.mean())
    ev_pmake_ci_lo, ev_pmake_ci_hi = bootstrap_ci(ev_pmake_disagree, 2000, 42)

    # Subset ev->void / pmake->preserve pattern
    ev_void_pmake_pres = 0
    for r in void_subset_rows:
        if (
            r["argmax_ev"] == r["void_slot"]
            and r["argmax_p_make"] == r["preserve_slot"]
        ):
            ev_void_pmake_pres += 1
    pattern_pct = (100.0 * ev_void_pmake_pres / n_subset) if n_subset else 0.0

    if ev_pmake_rate < 0.05:
        gate_decision = (
            "AGREE >= 95%. Wave 3.0's split is contrast-magnitude only, not "
            "behavior-level. Multi-objective architecture not justified by this "
            "campaign. Recommend Wave 4 pivots away from utility-conditioned models."
        )
        gate_verdict = "agree_ge_95"
    elif ev_pmake_rate >= 0.05 and pattern_pct >= 1.0:
        # We treat "meaningful" as ≥1% of the void-subset showing the
        # canonical EV->void / p_make->preserve split when overall divergence
        # already crosses the 5% threshold. The wiki page documents this rule.
        gate_decision = (
            "DISAGREE >= 5%. Divergence is real at the policy-action level. "
            "Recommend scoping rung-2 (utility-tunable searcher) as the next build."
        )
        gate_verdict = "disagree_ge_5_meaningful"
    elif 0.0 < ev_pmake_rate < 0.05:
        gate_decision = (
            "0% < disagreement < 5%. Pattern weak. Recommend running on the "
            "larger 10K-snapshot mixed corpus before deciding."
        )
        gate_verdict = "between_0_and_5"
    else:
        gate_decision = (
            "DISAGREE >= 5% but EV->void / p_make->preserve pattern not coherent. "
            "Recommend running on the larger 10K-snapshot mixed corpus before deciding."
        )
        gate_verdict = "disagree_ge_5_incoherent"

    summary = {
        "wave": "wave4",
        "bead": "t42-hmjr",
        "claim": "ch05-void-creation-follow",
        "question": (
            "Does an EV-greedy policy and a p_make-greedy policy disagree at the "
            "argmax-action level on these snapshots, or only at the contrast-"
            "magnitude level?"
        ),
        "n_snapshots_input": len(snapshots_raw),
        "n_snapshots_valid": len(valid),
        "n_snapshots_with_argmax": n_kept,
        "n_void_subset": n_subset,
        "samples_per_decision": args.samples,
        "device": device,
        "runtime_seconds": round(runtime_seconds, 1),
        "headline": {
            "ev_vs_p_make": {
                "disagree_count": int(ev_pmake_disagree.sum()),
                "n": n_kept,
                "rate": ev_pmake_rate,
                "ci_lo_95": ev_pmake_ci_lo,
                "ci_hi_95": ev_pmake_ci_hi,
            },
            "ev_void_pmake_preserve_pattern_count": ev_void_pmake_pres,
            "ev_void_pmake_preserve_pattern_pct": pattern_pct,
        },
        "gate_verdict": gate_verdict,
        "gate_decision_text": gate_decision,
        "claim_ledger_impact": "not-applicable (architecture-decision measurement)",
    }
    summary_path = out_dir / "summary.json"
    with summary_path.open("w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"Wrote {summary_path}", flush=True)

    # ------------------------------------------------------------------
    # Manifest
    # ------------------------------------------------------------------
    manifest = {
        "bead": "t42-hmjr",
        "wave": "wave4",
        "task": "utility_argmax_divergence",
        "snapshots_path": str(snap_path),
        "snapshots_sha256": hashlib.sha256(snap_path.read_bytes()).hexdigest(),
        "checkpoint": str(ckpt_path),
        "checkpoint_sha256": hashlib.sha256(ckpt_path.read_bytes()).hexdigest(),
        "device": device,
        "samples_per_decision": args.samples,
        "batch_size": args.batch_size,
        "n_snapshots_input": len(snapshots_raw),
        "n_snapshots_valid": len(valid),
        "n_snapshots_with_argmax": n_kept,
        "n_void_subset": n_subset,
        "n_pipeline_skipped": n_skipped,
        "n_no_legal": n_no_legal,
        "runtime_seconds": round(runtime_seconds, 1),
        "command": (
            f"python analyze.py --snapshots {snap_path} --checkpoint {ckpt_path} "
            f"--samples {args.samples} --device {device} "
            f"--batch-size {args.batch_size} --output-dir {out_dir}"
        ),
        "artifacts": {
            "per_snapshot_argmax_csv": str(per_snap_csv),
            "disagreement_matrix_csv": str(dis_csv),
            "void_subset_confusion_csv": str(conf_csv),
            "summary_json": str(summary_path),
        },
        "utility_definitions": {
            "ev": "mean(Q) per action -- forge e_q[slot]",
            "p_make": (
                "P(Q >= make_threshold) where threshold uses forge bid-aware "
                "convention: offense bin = 2*B (Q >= 2B-42); defense bin = "
                "85 - 2*B (Q >= 43-2B). For bid=30: offense Q>=18, defense Q>=-17."
            ),
            "mark_ev": "mm * (2*p_make - 1) where mm = max(1, bid // 42)",
            "cvar_10": "mean of pdf bins with cumulative mass <= 0.10",
            "robust_q25": "smallest Q with cumulative pdf mass >= 0.25",
        },
    }
    man_path = out_dir / "manifest.json"
    with man_path.open("w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"Wrote {man_path}", flush=True)

    # Stdout headline
    print("\n=== HEADLINE ===", flush=True)
    print(f"N argmax snapshots: {n_kept}", flush=True)
    print(
        f"EV vs p_make disagreement rate: {ev_pmake_rate*100:.2f}%  "
        f"95% CI [{ev_pmake_ci_lo*100:.2f}%, {ev_pmake_ci_hi*100:.2f}%]",
        flush=True,
    )
    print(f"Void subset N: {n_subset}", flush=True)
    print(f"  EV->void / p_make->preserve pattern: {ev_void_pmake_pres} ({pattern_pct:.1f}%)", flush=True)
    print(f"\nGate verdict: {gate_verdict}", flush=True)
    print(f"Decision: {gate_decision}", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    default_snaps = str(
        PROJECT_ROOT
        / "w42/book_validation_v1/wave2/snapshots/void_creation_follow/snapshots.jsonl"
    )
    default_ckpt = str(
        PROJECT_ROOT
        / "forge/models/domino-qval-large-3.3M-qgap0.071-qmae0.94.ckpt"
    )
    default_out = str(Path(__file__).parent)
    parser.add_argument("--snapshots", type=str, default=default_snaps)
    parser.add_argument("--checkpoint", type=str, default=default_ckpt)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--output-dir", type=str, default=default_out)
    parser.add_argument("--max-snapshots", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=50)
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
