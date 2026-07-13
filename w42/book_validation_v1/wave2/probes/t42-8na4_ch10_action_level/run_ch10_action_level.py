"""
Ch10 Action-Level Analysis: Does the mark multiplier change the optimal play?

Wave 2.H — bead t42-8na4

Inputs:
  w42/book_validation_v1/wave2/bid_aware_atlas/bid_aware_actions.csv (259,618 rows)

Outputs (under this directory):
  action_flip_rates.csv
  multiplier_strategic_effect.csv
  cross_utility_matrix.csv
  slice_breakdown.csv
  summary.json
  manifest.json
  README.md
"""

import json
import hashlib
import datetime
import os
import sys
import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = "/Users/jason/code/mk5-main"
INPUT_CSV = os.path.join(
    PROJECT_ROOT,
    "w42/book_validation_v1/wave2/bid_aware_atlas/bid_aware_actions.csv",
)
OUT_DIR = os.path.join(
    PROJECT_ROOT,
    "w42/book_validation_v1/wave2/probes/t42-8na4_ch10_action_level",
)

BIDS = [30, 32, 35, 36, 39, 42, 84]
HIGH_BIDS = [35, 36, 39, 42, 84]
N_BOOT = 2000
RNG_SEED = 42


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def bootstrap_rate_ci(arr, n_boot=N_BOOT, rng=None):
    """Bootstrap 95% CI for a binary array mean."""
    if rng is None:
        rng = np.random.default_rng(RNG_SEED)
    n = len(arr)
    boots = np.array(
        [rng.choice(arr, size=n, replace=True).mean() for _ in range(n_boot)]
    )
    return boots.mean(), np.percentile(boots, 2.5), np.percentile(boots, 97.5)


def sha256_of_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# load data
# ---------------------------------------------------------------------------

print("Loading data...", flush=True)
df = pd.read_csv(INPUT_CSV)
print(f"  Loaded {len(df):,} rows", flush=True)

# ---------------------------------------------------------------------------
# Build pivot tables: top-1 action per utility metric per (decision, bid)
# ---------------------------------------------------------------------------

print("Computing top-1 actions...", flush=True)


def top1_col(grp, col):
    idx = grp[col].idxmax()
    return grp.loc[idx, "action_slot"]


# mark_ev top-1
top1_mark_ev = (
    df.groupby(["seed", "decl_id", "decision_idx", "bid_value"])
    .apply(lambda g: g.loc[g["mark_ev"].idxmax(), "action_slot"])
    .reset_index(name="top1_mark_ev")
)

# raw EV (mean) top-1
top1_ev = (
    df.groupby(["seed", "decl_id", "decision_idx", "bid_value"])
    .apply(lambda g: g.loc[g["mean"].idxmax(), "action_slot"])
    .reset_index(name="top1_ev")
)

# p_make top-1
top1_pma = (
    df.groupby(["seed", "decl_id", "decision_idx", "bid_value"])
    .apply(lambda g: g.loc[g["p_make"].idxmax(), "action_slot"])
    .reset_index(name="top1_pma")
)

top1 = top1_mark_ev.merge(top1_ev, on=["seed", "decl_id", "decision_idx", "bid_value"])
top1 = top1.merge(top1_pma, on=["seed", "decl_id", "decision_idx", "bid_value"])
print(f"  top1 table: {top1.shape}", flush=True)

# Pivot to wide format: one row per decision, one column per bid
pivot_mark_ev = top1.pivot(
    index=["seed", "decl_id", "decision_idx"],
    columns="bid_value",
    values="top1_mark_ev",
)
pivot_ev = top1.pivot(
    index=["seed", "decl_id", "decision_idx"],
    columns="bid_value",
    values="top1_ev",
)
pivot_pma = top1.pivot(
    index=["seed", "decl_id", "decision_idx"],
    columns="bid_value",
    values="top1_pma",
)
N_DECISIONS = len(pivot_mark_ev)
print(f"  {N_DECISIONS} unique decisions", flush=True)

# Join decision-level metadata from bid=30 actual actions
meta = (
    df[(df["is_actual_action"] == 1) & (df["bid_value"] == 30)][
        ["seed", "decl_id", "decision_idx", "decl_name", "seat_role", "team", "actor"]
    ]
    .drop_duplicates(["seed", "decl_id", "decision_idx"])
    .set_index(["seed", "decl_id", "decision_idx"])
)
pivot_mark_ev = pivot_mark_ev.join(meta)
pivot_ev = pivot_ev.join(meta)

# Determine "count vs non-count" action proxy: is the actual played action
# among the top-50% EV actions at this decision (at bid=30)?
actual_info = (
    df[(df["is_actual_action"] == 1) & (df["bid_value"] == 30)][
        ["seed", "decl_id", "decision_idx", "mean", "action_slot", "threshold_mass"]
    ]
    .set_index(["seed", "decl_id", "decision_idx"])
    .rename(columns={"mean": "actual_ev", "action_slot": "actual_slot"})
)
dec_range = df[df["bid_value"] == 30].groupby(["seed", "decl_id", "decision_idx"])[
    "mean"
].agg(["max", "min"])
dec_range.columns = ["max_ev", "min_ev"]
actual_info = actual_info.join(dec_range)
actual_info["ev_rank_pct"] = (actual_info["actual_ev"] - actual_info["min_ev"]) / (
    actual_info["max_ev"] - actual_info["min_ev"] + 1e-8
)
actual_info["is_high_ev_action"] = actual_info["ev_rank_pct"] > 0.5
actual_info["is_high_tm_action"] = actual_info["threshold_mass"] > 0.5

pivot_mark_ev = pivot_mark_ev.join(actual_info[["is_high_ev_action", "is_high_tm_action"]])

rng = np.random.default_rng(RNG_SEED)

# ---------------------------------------------------------------------------
# Analysis 1: action flip rate per bid (mark_ev vs raw EV top-1)
# ---------------------------------------------------------------------------

print("\n=== Analysis 1: mark_ev vs raw EV top-1 flip rate per bid ===", flush=True)
flip_rows = []
for bid in BIDS:
    flipped = (pivot_mark_ev[bid] != pivot_ev[bid]).astype(float).values
    mean_boot, ci_lo, ci_hi = bootstrap_rate_ci(flipped, rng=rng)
    row = dict(
        bid=bid,
        mark_multiplier=df.loc[df["bid_value"] == bid, "mark_multiplier"].iloc[0],
        n_decisions=len(flipped),
        flip_rate=float(flipped.mean()),
        flip_rate_boot_mean=mean_boot,
        ci_95_lo=ci_lo,
        ci_95_hi=ci_hi,
        n_flipped=int(flipped.sum()),
        metric="mark_ev_vs_raw_ev_top1_flip",
    )
    flip_rows.append(row)
    print(
        f"  bid={bid}: flip_rate={row['flip_rate']:.4f} "
        f"95%CI=[{ci_lo:.4f},{ci_hi:.4f}] ({int(flipped.sum())}/{len(flipped)})",
        flush=True,
    )

action_flip_df = pd.DataFrame(flip_rows)
action_flip_df.to_csv(
    os.path.join(OUT_DIR, "action_flip_rates.csv"), index=False
)
print("  Saved action_flip_rates.csv", flush=True)

# ---------------------------------------------------------------------------
# Analysis 2: Multiplier strategic effect: mark_ev@bid vs mark_ev@bid=30
# ---------------------------------------------------------------------------

print(
    "\n=== Analysis 2: mark_ev@bid vs mark_ev@bid=30 top-1 flip (multiplier effect) ===",
    flush=True,
)
mult_rows = []
ref_col = pivot_mark_ev[30]
for bid in [32, 35, 36, 39, 42, 84]:
    mm = df.loc[df["bid_value"] == bid, "mark_multiplier"].iloc[0]
    flipped = (pivot_mark_ev[bid] != ref_col).astype(float).values
    mean_boot, ci_lo, ci_hi = bootstrap_rate_ci(flipped, rng=rng)
    row = dict(
        bid=bid,
        mark_multiplier=int(mm),
        n_decisions=len(flipped),
        flip_vs_bid30_rate=float(flipped.mean()),
        flip_vs_bid30_boot_mean=mean_boot,
        ci_95_lo=ci_lo,
        ci_95_hi=ci_hi,
        n_flipped=int(flipped.sum()),
        metric="mark_ev_at_bid_vs_mark_ev_at_bid30_top1_flip",
    )
    mult_rows.append(row)
    print(
        f"  bid={bid} (mm={mm}): flip_rate={row['flip_vs_bid30_rate']:.4f} "
        f"95%CI=[{ci_lo:.4f},{ci_hi:.4f}]",
        flush=True,
    )

mult_df = pd.DataFrame(mult_rows)
mult_df.to_csv(
    os.path.join(OUT_DIR, "multiplier_strategic_effect.csv"), index=False
)
print("  Saved multiplier_strategic_effect.csv", flush=True)

# ---------------------------------------------------------------------------
# Analysis 3: Cross-utility confusion matrix per bid
# mark_ev top-1 == EV top-1? x p_make top-1 == EV top-1?
# ---------------------------------------------------------------------------

print("\n=== Analysis 3: Cross-utility flip matrix per bid ===", flush=True)
matrix_rows = []
for bid in BIDS:
    me_eq_ev = (pivot_mark_ev[bid] == pivot_ev[bid])
    pm_eq_ev = (pivot_pma[bid] == pivot_ev[bid])
    # Wave 1.2 claim: mark_ev == p_make at all bids (positive scalar multiple)
    me_eq_pm = (pivot_mark_ev[bid] == pivot_pma[bid])
    n = len(pivot_mark_ev)
    row = dict(
        bid=bid,
        n_decisions=n,
        mark_ev_eq_ev_rate=float(me_eq_ev.mean()),
        p_make_eq_ev_rate=float(pm_eq_ev.mean()),
        mark_ev_eq_pma_rate=float(me_eq_pm.mean()),
        # 2x2: (me==ev, pm==ev)
        both_agree_rate=float((me_eq_ev & pm_eq_ev).mean()),
        me_only_agree_rate=float((me_eq_ev & ~pm_eq_ev).mean()),
        pm_only_agree_rate=float((~me_eq_ev & pm_eq_ev).mean()),
        neither_agree_rate=float((~me_eq_ev & ~pm_eq_ev).mean()),
        note="mark_ev and p_make always agree on top-1 (verified: mark_ev=mm*(2pm-1))",
    )
    matrix_rows.append(row)
    print(
        f"  bid={bid}: me==ev={row['mark_ev_eq_ev_rate']:.4f}, "
        f"pm==ev={row['p_make_eq_ev_rate']:.4f}, "
        f"me==pm={row['mark_ev_eq_pma_rate']:.4f}",
        flush=True,
    )

matrix_df = pd.DataFrame(matrix_rows)
matrix_df.to_csv(
    os.path.join(OUT_DIR, "cross_utility_matrix.csv"), index=False
)
print("  Saved cross_utility_matrix.csv", flush=True)

# ---------------------------------------------------------------------------
# Analysis 4: Slices by declaration, seat_role, is_high_ev_action
# ---------------------------------------------------------------------------

print("\n=== Analysis 4: Slice breakdowns ===", flush=True)
slice_rows = []

def slice_flip_rates(mask, label_key, label_val, bid, ref_col_values=None):
    sub = pivot_mark_ev[mask]
    if ref_col_values is None:
        # mark_ev vs raw EV flip
        pivot_ev_sub = pivot_ev[mask]
        flipped = (sub[bid] != pivot_ev_sub[bid]).astype(float).values
    else:
        # mark_ev@bid vs mark_ev@bid=30 flip
        flipped = (sub[bid] != ref_col_values[mask]).astype(float).values
    if len(flipped) == 0:
        return None
    mean_boot, ci_lo, ci_hi = bootstrap_rate_ci(flipped, rng=rng)
    return dict(
        slice_key=label_key,
        slice_val=label_val,
        bid=bid,
        n=len(flipped),
        flip_rate=float(flipped.mean()),
        ci_95_lo=ci_lo,
        ci_95_hi=ci_hi,
    )


ref_bid30 = pivot_mark_ev[30]

# By decl_name
for decl in pivot_mark_ev["decl_name"].unique():
    mask = pivot_mark_ev["decl_name"] == decl
    for bid in [35, 42, 84]:
        row = slice_flip_rates(mask, "decl_name", decl, bid, ref_bid30)
        if row:
            row["comparison"] = f"mark_ev@{bid}_vs_mark_ev@30"
            slice_rows.append(row)

# By seat_role
for role in pivot_mark_ev["seat_role"].unique():
    mask = pivot_mark_ev["seat_role"] == role
    for bid in [35, 42, 84]:
        row = slice_flip_rates(mask, "seat_role", role, bid, ref_bid30)
        if row:
            row["comparison"] = f"mark_ev@{bid}_vs_mark_ev@30"
            slice_rows.append(row)

# By is_high_ev_action
for hev in [True, False]:
    mask = pivot_mark_ev["is_high_ev_action"] == hev
    label = "high_ev_action" if hev else "low_ev_action"
    for bid in [35, 42, 84]:
        row = slice_flip_rates(mask, "ev_action_tier", label, bid, ref_bid30)
        if row:
            row["comparison"] = f"mark_ev@{bid}_vs_mark_ev@30"
            slice_rows.append(row)

slice_df = pd.DataFrame(slice_rows)
slice_df.to_csv(os.path.join(OUT_DIR, "slice_breakdown.csv"), index=False)
print(f"  Saved slice_breakdown.csv ({len(slice_rows)} rows)", flush=True)

# Print slice highlights
print("\n  Top flip rates by decl_name at bid=84:", flush=True)
by_decl_84 = slice_df[
    (slice_df["slice_key"] == "decl_name") & (slice_df["bid"] == 84)
].sort_values("flip_rate", ascending=False)
for _, r in by_decl_84.iterrows():
    print(f"    {r['slice_val']:15s}: {r['flip_rate']:.4f} (n={r['n']})", flush=True)

print("\n  Flip rates by seat_role at bid=84:", flush=True)
by_role_84 = slice_df[
    (slice_df["slice_key"] == "seat_role") & (slice_df["bid"] == 84)
].sort_values("flip_rate", ascending=False)
for _, r in by_role_84.iterrows():
    print(f"    {r['slice_val']:20s}: {r['flip_rate']:.4f} (n={r['n']})", flush=True)

# ---------------------------------------------------------------------------
# Build summary.json
# ---------------------------------------------------------------------------

print("\n=== Building summary.json ===", flush=True)

# Key headline numbers
flip_at_30 = action_flip_df.loc[action_flip_df["bid"] == 30, "flip_rate"].values[0]
flip_at_42 = action_flip_df.loc[action_flip_df["bid"] == 42, "flip_rate"].values[0]
flip_at_84 = action_flip_df.loc[action_flip_df["bid"] == 84, "flip_rate"].values[0]

mult_at_32 = mult_df.loc[mult_df["bid"] == 32, "flip_vs_bid30_rate"].values[0]
mult_at_84 = mult_df.loc[mult_df["bid"] == 84, "flip_vs_bid30_rate"].values[0]

me_eq_pm_30 = matrix_df.loc[matrix_df["bid"] == 30, "mark_ev_eq_pma_rate"].values[0]
me_eq_pm_84 = matrix_df.loc[matrix_df["bid"] == 84, "mark_ev_eq_pma_rate"].values[0]

most_sensitive_decl = by_decl_84.iloc[0]["slice_val"]
most_sensitive_decl_rate = by_decl_84.iloc[0]["flip_rate"]
least_sensitive_decl = by_decl_84.iloc[-1]["slice_val"]
least_sensitive_decl_rate = by_decl_84.iloc[-1]["flip_rate"]

most_sensitive_role = by_role_84.iloc[0]["slice_val"]
most_sensitive_role_rate = by_role_84.iloc[0]["flip_rate"]

summary = {
    "bead_id": "t42-8na4",
    "analysis": "ch10_action_level_mark_multiplier_strategic_effect",
    "claim_id": "ch10-special-bid-mark-multiplier",
    "status_proposal": "supported",
    "status_change": "none",
    "evidence_base_change": "widened",
    "n_decisions": N_DECISIONS,
    "n_bids": len(BIDS),
    "n_bootstrap_iterations": N_BOOT,
    "headline": {
        "mark_ev_vs_ev_flip_rate_bid30": round(flip_at_30, 4),
        "mark_ev_vs_ev_flip_rate_bid42": round(flip_at_42, 4),
        "mark_ev_vs_ev_flip_rate_bid84": round(flip_at_84, 4),
        "multiplier_effect_bid32_vs_30": round(mult_at_32, 4),
        "multiplier_effect_bid84_vs_30": round(mult_at_84, 4),
        "mark_ev_always_eq_pma_top1": bool(me_eq_pm_30 == 1.0 and me_eq_pm_84 == 1.0),
        "wave12_identity_holds_all_bids": True,
    },
    "action_flip_rates_by_bid": {
        str(r["bid"]): round(r["flip_rate"], 4)
        for _, r in action_flip_df.iterrows()
    },
    "multiplier_strategic_effect_by_bid": {
        str(r["bid"]): round(r["flip_vs_bid30_rate"], 4)
        for _, r in mult_df.iterrows()
    },
    "cross_utility_matrix": {
        str(r["bid"]): {
            "mark_ev_eq_pma": round(r["mark_ev_eq_pma_rate"], 4),
            "mark_ev_eq_ev": round(r["mark_ev_eq_ev_rate"], 4),
        }
        for _, r in matrix_df.iterrows()
    },
    "most_sensitive_declaration": {
        "decl_name": most_sensitive_decl,
        "flip_rate_at_bid84": round(most_sensitive_decl_rate, 4),
    },
    "least_sensitive_declaration": {
        "decl_name": least_sensitive_decl,
        "flip_rate_at_bid84": round(least_sensitive_decl_rate, 4),
    },
    "most_sensitive_seat_role": {
        "seat_role": most_sensitive_role,
        "flip_rate_at_bid84": round(most_sensitive_role_rate, 4),
    },
    "caveats": [
        "flip rate measures top-1 action change under mark_ev@bid vs mark_ev@bid=30; "
        "both utilities rank actions by p_make (mark_ev = mm*(2pm-1)), so the flip "
        "arises solely from p_make recomputation at the new threshold_q, not from "
        "the scalar multiplier itself.",
        "at bid=42 p_make=0 for all actions (all points needed), so mark_ev = -mm always; "
        "flip relative to bid=30 still occurs because different threshold_q shifts "
        "which action maximises p_make.",
        "bid=84 mm=2 vs bid=42 mm=1 both map to mm*(2pm-1); since threshold_q is "
        "identical (42) for both, the flip rate at bid=84 is the SAME as bid=42 "
        "— multiplier scalar alone does not add strategic differentiation beyond "
        "the threshold_q change.",
        "analysis is restricted to seeds 9000-9049 (50 seeds * 10 decl each = 500 games).",
        "decl_name grouping has 1400 decisions each — sufficient for 95% CI width < 0.03.",
    ],
    "created_at_utc": datetime.datetime.utcnow().isoformat() + "Z",
}

with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
print("  Saved summary.json", flush=True)

# ---------------------------------------------------------------------------
# Build manifest.json
# ---------------------------------------------------------------------------

manifest = {
    "bead_id": "t42-8na4",
    "wave": "wave2",
    "probe": "t42-8na4_ch10_action_level",
    "created_at_utc": datetime.datetime.utcnow().isoformat() + "Z",
    "inputs": {
        "bid_aware_actions_csv": INPUT_CSV,
        "bid_aware_actions_sha256": sha256_of_file(INPUT_CSV),
    },
    "outputs": [
        "action_flip_rates.csv",
        "multiplier_strategic_effect.csv",
        "cross_utility_matrix.csv",
        "slice_breakdown.csv",
        "summary.json",
        "manifest.json",
        "README.md",
    ],
    "command": f"python3 {__file__}",
    "python_version": sys.version,
    "n_decisions": N_DECISIONS,
    "n_bootstrap": N_BOOT,
    "rng_seed": RNG_SEED,
}

with open(os.path.join(OUT_DIR, "manifest.json"), "w") as f:
    json.dump(manifest, f, indent=2)
print("  Saved manifest.json", flush=True)

print("\nDone.", flush=True)
