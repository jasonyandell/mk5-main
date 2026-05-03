"""
Distribution Lens Reranker — Wave 1.1, bead t42-ybo6

Computes per-action utility scalars from per-world Q tensors (branch_atlas_scaled_v0
primary, branch_atlas_v1 supplemental), ranks actions by each utility lens, detects
EV-vs-alternative disagreements, and cross-tabs against claim-family detector tags.

Usage:
    python -u w42/book_validation_v1/wave1/t42-ybo6_distribution_lens_reranker/run_distribution_lens_reranker.py

Outputs (under w42/book_validation_v1/wave1/t42-ybo6_distribution_lens_reranker/):
    action_utility_scalars.csv
    utility_disagreement_matrix.csv
    detector_explained_disagreements.csv
    top_disagreements.csv
    summary.json
    manifest.json
    README.md
"""

from __future__ import annotations

import hashlib
import json
import sys
import os
from datetime import datetime, timezone
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import torch

# ── Project root ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path("/Users/jason/code/mk5-main")
sys.path.insert(0, str(PROJECT_ROOT))

OUT_DIR = PROJECT_ROOT / "w42/book_validation_v1/wave1/t42-ybo6_distribution_lens_reranker"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants ─────────────────────────────────────────────────────────────────
MAKE_THRESHOLD = 18          # Q >= 18 => "made" in ordinary 30-bid context
LOWER_TAIL_THRESHOLD = -18   # Q <= -18 => "set" / danger zone

UTILITY_COLS = [
    "ev",
    "p_make",
    "p_set",
    "threshold_mass_low",
    "threshold_mass_high",
    "cvar_10",
    "mark_ev",
    "robust_q25",
]

# ── Inputs ────────────────────────────────────────────────────────────────────
V0_PT   = PROJECT_ROOT / "w42/branch_atlas_scaled_v0/eq_pdf_s9430_d10_bid30_joint_1000s_v2.pt"
V0_ACT  = PROJECT_ROOT / "w42/branch_atlas_scaled_v0/decision_actions.csv"
V0_STAT = PROJECT_ROOT / "w42/branch_atlas_scaled_v0/decision_states.csv"

V1_PT   = PROJECT_ROOT / "w42/branch_atlas_v1/eq_pdf_s9420-9421_joint_1000s_v2.pt"
V1_ACT  = PROJECT_ROOT / "w42/branch_atlas_v1/decision_actions.csv"


# ── Helper: SHA256 ────────────────────────────────────────────────────────────
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ── Mark EV transform (Ch10 deterministic scoring) ────────────────────────────
# For each world, bidder captures (42 + Q)/2 count points (since Q is relative
# to Team 0 perspective, and Team 0 is always the bidder in this corpus).
# For ordinary bids (30-41): made if bidder_points >= bid; winner gets 1 mark.
# mark_ev = E[winner_marks_for_team0] across worlds.
#
# Q is in Team 0 count-point-swing units: Q = (bidder_pts - 21) * 2.
# So bidder_pts = Q/2 + 21.  For bid=30: made if bidder_pts >= 30 => Q >= 18.
# That is MAKE_THRESHOLD.

def mark_ev_from_qpw(q_per_world: np.ndarray, bid: int) -> float:
    """Compute mark EV for Team 0 over sampled worlds.

    q_per_world: shape [n_worlds] — Q values for one action.
    bid: declared bid (30-41 ordinary, 42 = all_42, 84+ = mark_ladder).
    Returns E[marks_for_team0] across worlds.
    """
    # bidder points from Q: Q = 2*(bidder_pts - 21), so bidder_pts = Q/2 + 21
    bidder_pts = q_per_world / 2.0 + 21.0
    if bid < 42:
        made = bidder_pts >= bid
    else:
        # bid==42 requires exactly 42 points
        made = bidder_pts >= 42
    # mark_multiplier: 1 for ordinary/42, bid//42 for 84+
    if bid >= 84:
        multiplier = bid // 42
    else:
        multiplier = 1
    # Team 0 is bidder; wins multiplier marks if made, 0 otherwise
    team0_marks = np.where(made, float(multiplier), 0.0)
    return float(team0_marks.mean())


# ── Core utility computation ──────────────────────────────────────────────────
def compute_utilities_for_record(
    decisions: list,
    bid: int,
    game_key: str,
) -> list[dict]:
    """Extract per-action utility scalars for every decision in one game record.

    Returns list of dicts, one per (decision, action).
    """
    rows = []
    for d_idx, dec in enumerate(decisions):
        legal_mask = dec.legal_mask.numpy().astype(bool)  # [7]
        qpw = dec.q_per_world.numpy()                     # [1000, 7]
        n_worlds = qpw.shape[0]

        for a_idx in range(len(legal_mask)):
            if not legal_mask[a_idx]:
                continue
            q = qpw[:, a_idx]  # [n_worlds]

            ev_val       = float(q.mean())
            p_make_val   = float((q >= MAKE_THRESHOLD).mean())
            p_set_val    = float((q < MAKE_THRESHOLD).mean())
            th_low_val   = float((q <= LOWER_TAIL_THRESHOLD).mean())
            th_high_val  = float((q >= MAKE_THRESHOLD).mean())  # same as p_make
            n10          = max(1, int(np.floor(n_worlds * 0.10)))
            cvar10_val   = float(np.sort(q)[:n10].mean())
            mark_ev_val  = mark_ev_from_qpw(q, bid)
            q25_val      = float(np.quantile(q, 0.25))

            rows.append({
                "game_key":             game_key,
                "decision_seq_idx":     d_idx,
                "action_slot":          a_idx,
                "n_worlds":             n_worlds,
                "bid":                  bid,
                "player":               int(dec.player),
                "action_taken":         int(dec.action_taken),
                "ev":                   ev_val,
                "p_make":               p_make_val,
                "p_set":                p_set_val,
                "threshold_mass_low":   th_low_val,
                "threshold_mass_high":  th_high_val,
                "cvar_10":              cvar10_val,
                "mark_ev":              mark_ev_val,
                "robust_q25":           q25_val,
            })
    return rows


# ── Load tensor file ──────────────────────────────────────────────────────────
def load_tensor_file(pt_path: Path, source_label: str) -> list[dict]:
    print(f"[load] {pt_path.name} ...", flush=True)
    data = torch.load(str(pt_path), map_location="cpu", weights_only=False)
    results = data["results"]
    all_rows = []
    for rec_idx, rec in enumerate(results):
        bid = int(rec.decisions[0].bid_value) if rec.decisions else 30
        game_key = f"{source_label}:r{rec_idx}"
        rows = compute_utilities_for_record(rec.decisions, bid, game_key)
        all_rows.extend(rows)
    print(f"  -> {len(results)} records, {len(all_rows)} action rows", flush=True)
    return all_rows


# ── Build decision-level metadata from branch atlas CSVs ─────────────────────
def build_decision_meta(act_csv: Path, source_label: str) -> pd.DataFrame:
    """Return per-action metadata with claim-family tags."""
    df = pd.read_csv(act_csv)
    df["source_label"] = source_label
    # Compute sequential decision index within each game record
    # game_idx is the record index within the file; decision_idx is already there
    df["decision_seq_idx"] = df["decision_idx"]
    df["action_slot"] = df["candidate_slot"]
    df["game_key"] = source_label + ":r" + df["game_idx"].astype(str)
    return df


# ── Rank disagreement ─────────────────────────────────────────────────────────
def top1_disagreement_rate(df: pd.DataFrame, u1: str, u2: str) -> tuple[float, int]:
    """Fraction of decisions where u1's top action != u2's top action."""
    diffs = 0
    total = 0
    for key, grp in df.groupby("decision_id"):
        if len(grp) < 2:
            continue
        top1_u1 = grp.loc[grp[u1].idxmax(), "action_slot"]
        top1_u2 = grp.loc[grp[u2].idxmax(), "action_slot"]
        if top1_u1 != top1_u2:
            diffs += 1
        total += 1
    return (diffs / total if total > 0 else 0.0), total


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    print("=== Distribution Lens Reranker — Wave 1.1 / t42-ybo6 ===", flush=True)

    # 1. Load per-world Q tensors
    rows_v0 = load_tensor_file(V0_PT, "v0")
    rows_v1 = load_tensor_file(V1_PT, "v1")
    all_rows = rows_v0 + rows_v1
    df_util = pd.DataFrame(all_rows)
    print(f"\nTotal utility rows: {len(df_util)}", flush=True)

    # 2. Load decision metadata
    meta_v0 = build_decision_meta(V0_ACT, "v0")
    meta_v1 = build_decision_meta(V1_ACT, "v1")
    meta = pd.concat([meta_v0, meta_v1], ignore_index=True)

    # 3. Join utilities with metadata
    # Key: game_key + decision_seq_idx + action_slot
    meta_slim = meta[[
        "game_key", "decision_seq_idx", "action_slot",
        "seed", "game_idx", "decl_id", "decl_name", "bid_value",
        "actor", "seat_role", "team", "trick_idx", "trick_position",
        "candidate_domino", "candidate_count_points",
        "candidate_is_double", "candidate_is_called_suit",
        "is_actual_action",
        "distribution_shape_tags", "strategy_context_tags",
        "matched_position_detectors",
        "mean", "std", "q10", "q25", "q50", "threshold_mass",
        "lower_tail_mass_le_neg18", "cvar_low_10",
    ]].copy()
    # meta already has mean/threshold_mass from branch atlas builder
    # rename to avoid conflict with our computed values
    meta_slim = meta_slim.rename(columns={
        "mean": "ba_ev",
        "std": "ba_std",
        "q10": "ba_q10",
        "q25": "ba_q25",
        "q50": "ba_q50",
        "threshold_mass": "ba_threshold_mass",
        "lower_tail_mass_le_neg18": "ba_lower_tail_mass",
        "cvar_low_10": "ba_cvar_low_10",
    })

    df_merged = df_util.merge(
        meta_slim,
        on=["game_key", "decision_seq_idx", "action_slot"],
        how="left",
    )
    n_unmatched = df_merged["seed"].isna().sum()
    print(f"Unmatched utility rows (no metadata): {n_unmatched}", flush=True)

    # Create stable decision_id
    df_merged["decision_id"] = (
        df_merged["game_key"] + ":d" + df_merged["decision_seq_idx"].astype(str)
    )

    print(f"Merged rows: {len(df_merged)}", flush=True)
    print(f"Unique decisions: {df_merged['decision_id'].nunique()}", flush=True)

    # 4. Per-decision ranking: rank each action by each utility (desc for good, asc for risk)
    # For ev, p_make, threshold_mass_high, mark_ev, robust_q25: higher = better (rank desc)
    # For p_set, threshold_mass_low, cvar_10: lower = better (rank asc for safety)
    # But for consistency in "top-1" comparison, we want:
    #   EV top-1 = argmax(ev)
    #   CVaR top-1 = argmax(cvar_10) (least-bad worst case)
    #   p_set top-1 = argmin(p_set) = argmax(-p_set) => argmax(1 - p_set)
    #   threshold_mass_low top-1 = argmin(threshold_mass_low)
    # We store ranks (1=best) per decision for each utility.

    RANK_HIGHER_BETTER = ["ev", "p_make", "threshold_mass_high", "mark_ev", "robust_q25"]
    RANK_LOWER_BETTER  = ["p_set", "threshold_mass_low", "cvar_10"]
    # cvar_10 is negative; argmax is least-bad (safest), so "higher = better" for cvar_10 too
    RANK_HIGHER_BETTER.append("cvar_10")
    RANK_LOWER_BETTER = ["p_set", "threshold_mass_low"]

    for col in UTILITY_COLS:
        ascending = col in RANK_LOWER_BETTER
        df_merged[f"rank_{col}"] = df_merged.groupby("decision_id")[col].rank(
            method="min", ascending=ascending
        )

    # Top-1 indicator per utility
    for col in UTILITY_COLS:
        df_merged[f"top1_{col}"] = df_merged[f"rank_{col}"] == 1

    # 5. Disagreement matrix: for each pair of utilities, fraction of decisions where top-1 differs
    print("\nComputing pairwise disagreement matrix...", flush=True)
    disagree_data = {}
    for u1 in UTILITY_COLS:
        row_data = {}
        for u2 in UTILITY_COLS:
            if u1 == u2:
                row_data[u2] = 0.0
                continue
            # Count decisions where top-1 action differs
            grp = df_merged.groupby("decision_id")
            n_decisions = 0
            n_disagree = 0
            for dec_id, g in grp:
                if len(g) < 2:
                    continue
                idx_u1 = g[f"rank_{u1}"].idxmin()
                idx_u2 = g[f"rank_{u2}"].idxmin()
                slot_u1 = g.loc[idx_u1, "action_slot"]
                slot_u2 = g.loc[idx_u2, "action_slot"]
                n_decisions += 1
                if slot_u1 != slot_u2:
                    n_disagree += 1
            rate = n_disagree / n_decisions if n_decisions > 0 else 0.0
            row_data[u2] = round(rate, 4)
        disagree_data[u1] = row_data

    df_disagree = pd.DataFrame(disagree_data).T
    df_disagree.index.name = "utility_row"
    print(df_disagree.to_string(), flush=True)

    # 6. Per-decision disagreement score: max rank discrepancy across utilities
    # For each decision, compute how often EV top-1 != alternative top-1
    decision_disagree_rows = []
    for dec_id, g in df_merged.groupby("decision_id"):
        if len(g) < 2:
            continue
        # Get top-1 action slot for each utility
        top1_slots = {}
        for col in UTILITY_COLS:
            idx = g[f"rank_{col}"].idxmin()
            top1_slots[col] = g.loc[idx, "action_slot"]

        ev_slot = top1_slots["ev"]
        n_disagree_with_ev = sum(1 for u in UTILITY_COLS if u != "ev" and top1_slots[u] != ev_slot)

        # EV gap: when EV's top-1 is not the best by another utility, what is the EV cost?
        ev_gaps = {}
        for col in UTILITY_COLS:
            if col == "ev" or top1_slots[col] == ev_slot:
                ev_gaps[col] = 0.0
            else:
                ev_of_ev_top   = g.loc[g["action_slot"] == ev_slot, "ev"].values
                ev_of_alt_top  = g.loc[g["action_slot"] == top1_slots[col], "ev"].values
                if len(ev_of_ev_top) > 0 and len(ev_of_alt_top) > 0:
                    # Positive = EV top-1 is better in EV; negative = alt top-1 dominates even in EV
                    ev_gaps[col] = round(float(ev_of_ev_top[0] - ev_of_alt_top[0]), 3)
                else:
                    ev_gaps[col] = float("nan")

        # Get metadata from first row of this decision
        meta_row = g.iloc[0]
        decision_disagree_rows.append({
            "decision_id":          dec_id,
            "game_key":             meta_row["game_key"],
            "seed":                 meta_row.get("seed", np.nan),
            "decl_name":            meta_row.get("decl_name", ""),
            "trick_idx":            meta_row.get("trick_idx", np.nan),
            "seat_role":            meta_row.get("seat_role", ""),
            "team":                 meta_row.get("team", ""),
            "legal_action_count":   len(g),
            "n_disagree_with_ev":   n_disagree_with_ev,
            "ev_top1_slot":         ev_slot,
            **{f"top1_{u}": top1_slots[u] for u in UTILITY_COLS},
            **{f"ev_gap_vs_{u}": ev_gaps[u] for u in UTILITY_COLS if u != "ev"},
            "strategy_context_tags":    meta_row.get("strategy_context_tags", ""),
            "distribution_shape_tags":  meta_row.get("distribution_shape_tags", ""),
            "matched_position_detectors": meta_row.get("matched_position_detectors", ""),
        })

    df_decisions = pd.DataFrame(decision_disagree_rows)
    n_total_decisions = len(df_decisions)
    print(f"\nTotal decisions analyzed: {n_total_decisions}", flush=True)

    # 7. Top 200 most-disagreed decisions
    df_decisions_sorted = df_decisions.sort_values("n_disagree_with_ev", ascending=False)
    top200 = df_decisions_sorted.head(200).copy()

    # 8. Detector-family disagree analysis
    # For each claim-family tag (from strategy_context_tags + matched_position_detectors),
    # compute the rate at which EV's top-1 is overridden by each alternative utility.
    print("\nComputing per-claim-family disagreement rates...", flush=True)

    # Collect all tags from decisions
    tag_disagree_rows = []

    def iter_tags(row):
        tags = set()
        for col in ["strategy_context_tags", "matched_position_detectors", "distribution_shape_tags"]:
            val = row.get(col, "")
            if isinstance(val, str) and val.strip():
                for t in val.split("|"):
                    t = t.strip()
                    if t:
                        tags.add(t)
        return tags

    # Build per-tag aggregates
    tag_stats = defaultdict(lambda: {
        "n_decisions": 0,
        "n_any_disagree": 0,
        **{f"n_disagree_{u}": 0 for u in UTILITY_COLS if u != "ev"},
    })

    for _, row in df_decisions.iterrows():
        tags = iter_tags(row)
        if not tags:
            tags = {"_no_tag"}
        for tag in tags:
            tag_stats[tag]["n_decisions"] += 1
            if row["n_disagree_with_ev"] > 0:
                tag_stats[tag]["n_any_disagree"] += 1
            for u in UTILITY_COLS:
                if u == "ev":
                    continue
                if row[f"top1_{u}"] != row["ev_top1_slot"]:
                    tag_stats[tag][f"n_disagree_{u}"] += 1

    detector_rows = []
    for tag, stats in sorted(tag_stats.items(), key=lambda x: -x[1]["n_decisions"]):
        n = stats["n_decisions"]
        detector_rows.append({
            "detector_tag": tag,
            "n_decisions": n,
            "n_any_disagree": stats["n_any_disagree"],
            "ev_lying_rate": round(stats["n_any_disagree"] / n, 4) if n > 0 else 0.0,
            **{
                f"disagree_rate_{u}": round(stats[f"n_disagree_{u}"] / n, 4)
                for u in UTILITY_COLS if u != "ev"
            },
        })

    df_detectors = pd.DataFrame(detector_rows)

    # 9. Save outputs
    print("\nSaving outputs...", flush=True)

    # action_utility_scalars.csv
    scalar_cols = [
        "decision_id", "game_key", "decision_seq_idx", "action_slot",
        "n_worlds", "bid", "player", "action_taken",
        "seed", "decl_name", "trick_idx", "seat_role", "team",
        "candidate_domino", "candidate_count_points",
        "candidate_is_double", "is_actual_action",
        "ev", "p_make", "p_set", "threshold_mass_low", "threshold_mass_high",
        "cvar_10", "mark_ev", "robust_q25",
        "rank_ev", "rank_p_make", "rank_p_set", "rank_threshold_mass_low",
        "rank_threshold_mass_high", "rank_cvar_10", "rank_mark_ev", "rank_robust_q25",
        "top1_ev", "top1_p_make", "top1_p_set", "top1_threshold_mass_low",
        "top1_threshold_mass_high", "top1_cvar_10", "top1_mark_ev", "top1_robust_q25",
        "strategy_context_tags", "distribution_shape_tags", "matched_position_detectors",
    ]
    scalar_out_cols = [c for c in scalar_cols if c in df_merged.columns]
    df_merged[scalar_out_cols].to_csv(OUT_DIR / "action_utility_scalars.csv", index=False)
    print(f"  action_utility_scalars.csv: {len(df_merged)} rows", flush=True)

    # utility_disagreement_matrix.csv
    df_disagree.to_csv(OUT_DIR / "utility_disagreement_matrix.csv")
    print(f"  utility_disagreement_matrix.csv", flush=True)

    # detector_explained_disagreements.csv
    df_detectors.to_csv(OUT_DIR / "detector_explained_disagreements.csv", index=False)
    print(f"  detector_explained_disagreements.csv: {len(df_detectors)} tag rows", flush=True)

    # top_disagreements.csv
    top200_cols = [c for c in top200.columns if c in df_decisions.columns]
    top200[top200_cols].to_csv(OUT_DIR / "top_disagreements.csv", index=False)
    print(f"  top_disagreements.csv: {len(top200)} rows", flush=True)

    # 10. Headline metrics
    n_decisions_with_any_disagree = int((df_decisions["n_disagree_with_ev"] > 0).sum())
    ev_lying_rate_overall = n_decisions_with_any_disagree / n_total_decisions if n_total_decisions > 0 else 0.0

    per_util_disagree = {}
    for u in UTILITY_COLS:
        if u == "ev":
            continue
        n_d = int((df_decisions[f"top1_{u}"] != df_decisions["ev_top1_slot"]).sum())
        per_util_disagree[u] = round(n_d / n_total_decisions, 4) if n_total_decisions > 0 else 0.0

    # Worst EV gaps (when alt utility prefers a different action)
    ev_gap_stats = {}
    for u in UTILITY_COLS:
        if u == "ev":
            continue
        col = f"ev_gap_vs_{u}"
        disagree_mask = df_decisions[f"top1_{u}"] != df_decisions["ev_top1_slot"]
        gaps = df_decisions.loc[disagree_mask, col].dropna()
        ev_gap_stats[u] = {
            "mean_ev_gap": round(float(gaps.mean()), 3) if len(gaps) > 0 else 0.0,
            "median_ev_gap": round(float(gaps.median()), 3) if len(gaps) > 0 else 0.0,
            "n_alt_dominates_ev": int((gaps < 0).sum()),
        }

    summary = {
        "question": "Where does scalar EV rank relative to p_make, p_set, threshold_mass_low/high, CVaR_10, mark_ev, robust_q25?",
        "slice": "branch_atlas_scaled_v0 (seed=9430, 10 decl, bid=30, 1000 worlds) + branch_atlas_v1 (seeds 9420-9421, 2 decl, bid=30, 1000 worlds)",
        "N_decisions": n_total_decisions,
        "N_action_rows": len(df_merged),
        "N_decisions_with_any_ev_disagree": n_decisions_with_any_disagree,
        "ev_lying_rate_any_utility": round(ev_lying_rate_overall, 4),
        "per_utility_ev_disagree_rate": per_util_disagree,
        "ev_gap_when_alternative_prefers_different": ev_gap_stats,
        "make_threshold_used": MAKE_THRESHOLD,
        "lower_tail_threshold_used": LOWER_TAIL_THRESHOLD,
        "note_on_mark_ev": "mark_ev uses team0-is-bidder assumption (all records in this corpus are bidder=team0, bid=30); mark_multiplier=1 for ordinary bids",
        "note_on_join": "branch_atlas seed 9430/9420-9421 is disjoint from joined_claim_row_model_table (seeds 0-99); claim-family tags come from decision_actions.csv distribution_shape_tags, strategy_context_tags, matched_position_detectors columns",
        "paired": "paired — same decision, same world sample, different utility lens",
        "metric": "top-1 action rank disagreement rate; EV gap in count-point units",
        "claim_ledger_impact": "underpowered",
        "caveats": [
            "Slice is 1 seed (9430) x 10 declarations x bid=30 only; no variation in bid value or seed.",
            "mark_ev transform assumes team0=bidder, ordinary bid (multiplier=1). Multi-mark bids (84+) not present in this corpus.",
            "p_make == threshold_mass_high by construction (both use Q >= 18); they are identical lenses.",
            "q_per_world encodes oracle-softmax-weighted world draws, not uniformly sampled hands.",
            "No direct row-join with joined_claim_row_model_table (different seeds); claim-family attribution uses branch_atlas internal tags only.",
        ],
        "status": "complete",
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  summary.json written", flush=True)

    # manifest.json
    input_shas = {
        "v0_pt":   sha256_file(V0_PT),
        "v0_act":  sha256_file(V0_ACT),
        "v0_stat": sha256_file(V0_STAT),
        "v1_pt":   sha256_file(V1_PT),
        "v1_act":  sha256_file(V1_ACT),
    }
    manifest = {
        "bead_id": "t42-ybo6",
        "wave": "wave1",
        "parent_epic": "t42-4zi6",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": "python -u w42/book_validation_v1/wave1/t42-ybo6_distribution_lens_reranker/run_distribution_lens_reranker.py",
        "inputs": {
            "v0_tensor":        str(V0_PT.relative_to(PROJECT_ROOT)),
            "v0_decision_actions": str(V0_ACT.relative_to(PROJECT_ROOT)),
            "v0_decision_states":  str(V0_STAT.relative_to(PROJECT_ROOT)),
            "v1_tensor":        str(V1_PT.relative_to(PROJECT_ROOT)),
            "v1_decision_actions": str(V1_ACT.relative_to(PROJECT_ROOT)),
        },
        "input_shas": input_shas,
        "outputs": {
            "action_utility_scalars":         "action_utility_scalars.csv",
            "utility_disagreement_matrix":    "utility_disagreement_matrix.csv",
            "detector_explained_disagreements": "detector_explained_disagreements.csv",
            "top_disagreements":              "top_disagreements.csv",
            "summary":                        "summary.json",
            "readme":                         "README.md",
        },
        "schema_version": "w42.bookval.v1.wave1.t42-ybo6.v1",
    }

    with open(OUT_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  manifest.json written", flush=True)

    # Print final summary
    print("\n=== HEADLINE RESULTS ===", flush=True)
    print(f"Decisions analyzed: {n_total_decisions}", flush=True)
    print(f"EV disagrees with ANY alternative: {ev_lying_rate_overall:.1%} ({n_decisions_with_any_disagree}/{n_total_decisions})", flush=True)
    print("\nPer-utility EV top-1 disagreement rates:", flush=True)
    for u, rate in sorted(per_util_disagree.items(), key=lambda x: -x[1]):
        gap = ev_gap_stats[u]
        print(f"  {u:22s}: {rate:.1%}  mean_ev_gap={gap['mean_ev_gap']:+.1f}  n_alt_dominates_ev={gap['n_alt_dominates_ev']}", flush=True)

    print("\nTop 5 most disagreed decisions:", flush=True)
    for _, row in df_decisions_sorted.head(5).iterrows():
        print(f"  {row['decision_id']} | n_disagree={row['n_disagree_with_ev']} | "
              f"seat={row.get('seat_role','?')} | ctx={str(row.get('strategy_context_tags',''))[:50]}", flush=True)

    print(f"\nAll outputs written to: {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
