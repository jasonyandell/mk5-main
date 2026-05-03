#!/usr/bin/env python3
"""W42 Book Validation v1 Wave 1 — Hidden Threat Impact Ranker.

Bead: t42-c2y9
Parent epic: t42-4zi6

For each decision in branch_atlas_v1 + branch_atlas_scaled_v0, rank unseen tiles
by impact magnitude and identify load-bearing tiles. Cross-tab against detector
tags from the joined-claim-row table. Output offline labels only.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "w42/book_validation_v1/wave1/t42-c2y9_hidden_threat_impact_ranker"
DATA_V1 = ROOT / "w42/branch_atlas_v1/hidden_threat_rows.csv"
DATA_SCALED = ROOT / "w42/branch_atlas_scaled_v0/hidden_threat_rows.csv"
JOINED_CLAIMS = ROOT / "w42/joined_claim_row_model_table/joined_claim_action_rows.csv"
LEGACY_DRIVER_ROLLUP = ROOT / "w42/hidden_threat_legacy_mining/hidden_driver_rollup.csv"
LEGACY_CLAIM_EVIDENCE = ROOT / "w42/hidden_threat_legacy_mining/claim_branch_impact_evidence.csv"
LEGACY_GROUP_METRICS = ROOT / "w42/hidden_threat_legacy_mining/group_metrics.csv"

TOP_K = 5
SINGLE_TILE_DOMINANT_N = 100
INFERABLE_N = 100

# Impact score threshold for "large" impact (from legacy mining: score >= 5.0)
LARGE_IMPACT_THRESH = 5.0

# Concentration threshold: fraction of total impact on top tile = "one tile decides"
CONCENTRATION_THRESH_PCT = 0.60


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()[:16]


def load_combined_hidden_threat() -> pd.DataFrame:
    """Load and combine both hidden-threat corpora."""
    df_v1 = pd.read_csv(DATA_V1)
    df_s = pd.read_csv(DATA_SCALED)
    df_v1["source_corpus"] = "branch_atlas_v1"
    df_s["source_corpus"] = "branch_atlas_scaled_v0"
    combined = pd.concat([df_v1, df_s], ignore_index=True)

    # Build canonical decision key
    combined["decision_key"] = (
        combined["source_corpus"].astype(str)
        + ":s"
        + combined["seed"].astype(str)
        + ":g"
        + combined["game_idx"].astype(str)
        + ":d"
        + combined["decl_id"].astype(str)
        + ":di"
        + combined["decision_idx"].astype(str)
    )

    # Build canonical decision-action key
    combined["da_key"] = combined["decision_key"] + ":a" + combined["action_slot"].astype(str)

    return combined


def build_decision_meta(combined: pd.DataFrame) -> pd.DataFrame:
    """One row per decision with stable metadata."""
    meta_cols = [
        "decision_key",
        "source_corpus",
        "seed",
        "game_idx",
        "decl_id",
        "decl_name",
        "decision_idx",
        "actor",
        "seat_role",
        "team",
    ]
    meta = combined[meta_cols].drop_duplicates("decision_key").set_index("decision_key")
    return meta


def rank_tiles_per_decision(combined: pd.DataFrame, top_k: int = TOP_K) -> pd.DataFrame:
    """For each decision, rank hidden tiles by max impact across all action slots.

    Returns a DataFrame with one row per (decision, rank) where rank <= top_k.
    """
    # For each (decision, hidden_domino, absolute_holder): max impact across actions
    grp = (
        combined.groupby(
            [
                "decision_key",
                "hidden_domino_id",
                "hidden_domino",
                "absolute_holder",
                "absolute_holder_role",
                "relative_holder",
            ]
        )
        .agg(
            max_impact_score=("impact_score", "max"),
            max_downside_score=("downside_score", "max"),
            max_upside_score=("upside_score", "max"),
            mean_q_delta=("mean_q_delta", "mean"),
            tail_low_mass_delta=("tail_low_mass_delta", "mean"),
            shelf_high_mass_delta=("shelf_high_mass_delta", "mean"),
            conditioned_mass=("conditioned_mass", "mean"),
            n_actions=("action_slot", "nunique"),
        )
        .reset_index()
    )

    # Rank within each decision by max_impact_score
    grp["rank"] = (
        grp.groupby("decision_key")["max_impact_score"]
        .rank(method="first", ascending=False)
        .astype(int)
    )

    top_k_df = grp[grp["rank"] <= top_k].copy()

    # Add decision-level total impact for concentration metric
    decision_total = grp.groupby("decision_key")["max_impact_score"].sum().rename("decision_total_impact")
    top_k_df = top_k_df.merge(decision_total, left_on="decision_key", right_index=True, how="left")
    top_k_df["impact_fraction_of_total"] = (
        top_k_df["max_impact_score"] / top_k_df["decision_total_impact"].clip(lower=1e-6)
    )

    # Direction: positive mean_q_delta = tile is good for actor, negative = bad
    top_k_df["direction"] = top_k_df["mean_q_delta"].apply(
        lambda v: "helpful" if v > 0.5 else ("harmful" if v < -0.5 else "neutral")
    )

    return top_k_df


def compute_single_tile_dominance(
    combined: pd.DataFrame, top_k_df: pd.DataFrame
) -> pd.DataFrame:
    """Identify decisions where one tile dominates (>= CONCENTRATION_THRESH_PCT of total impact).

    Returns top-100 by dominance concentration, sorted by concentration_pct descending.
    """
    # For each decision, get the top-1 tile's fraction
    top1 = top_k_df[top_k_df["rank"] == 1][
        [
            "decision_key",
            "hidden_domino",
            "absolute_holder_role",
            "max_impact_score",
            "max_downside_score",
            "mean_q_delta",
            "tail_low_mass_delta",
            "conditioned_mass",
            "direction",
            "impact_fraction_of_total",
        ]
    ].copy()
    top1 = top1.rename(columns={"impact_fraction_of_total": "concentration_pct"})

    # Add decision meta
    meta = build_decision_meta(combined)
    top1 = top1.merge(meta, left_on="decision_key", right_index=True, how="left")

    # Sort by concentration
    top1 = top1.sort_values("concentration_pct", ascending=False)

    # Return top N
    result = top1.head(SINGLE_TILE_DOMINANT_N).copy()
    result["is_single_tile_dominant"] = result["concentration_pct"] >= CONCENTRATION_THRESH_PCT
    return result


def compute_inferability(combined: pd.DataFrame, top_k_df: pd.DataFrame) -> pd.DataFrame:
    """Classify top-1 load-bearing tile as inferable vs not.

    Inferability heuristic:
    - A tile is 'easily inferable from public state' if conditioned_mass >= 0.5
      (i.e., >= 50% of worlds have this tile in this position) — the tile's location
      is highly constrained by prior plays.
    - A tile is 'hard to infer' if conditioned_mass < 0.2 — could be anywhere.

    Returns top-100 by impact, split by inferability label.
    """
    top1 = top_k_df[top_k_df["rank"] == 1][
        [
            "decision_key",
            "hidden_domino",
            "absolute_holder_role",
            "max_impact_score",
            "max_downside_score",
            "mean_q_delta",
            "conditioned_mass",
            "direction",
            "impact_fraction_of_total",
        ]
    ].copy()

    # Inferability bucket
    def inferability_label(mass: float) -> str:
        if mass >= 0.50:
            return "easily_inferable"
        elif mass >= 0.30:
            return "moderately_inferable"
        else:
            return "hard_to_infer"

    top1["inferability"] = top1["conditioned_mass"].apply(inferability_label)

    # Add decision meta
    meta = build_decision_meta(combined)
    top1 = top1.merge(meta, left_on="decision_key", right_index=True, how="left")

    # Sort by impact and take top N
    top1 = top1.sort_values("max_impact_score", ascending=False)
    result = top1.head(INFERABLE_N).copy()
    return result


def classify_tile_type(hidden_domino: str, decl_name: str) -> dict[str, Any]:
    """Classify a tile's strategic type given declaration.

    Returns a dict with tile_is_trump, tile_is_double, tile_is_count, tile_category.
    """
    try:
        high, low = [int(x) for x in hidden_domino.split("-")]
    except Exception:
        return {
            "tile_is_trump": False,
            "tile_is_double": False,
            "tile_is_count": False,
            "tile_category": "unknown",
        }

    # Doubles-suit maps to all doubles being trump
    trump_suit: int | None = None
    if decl_name not in ("no-trump", "doubles", "doubles-suit", "unknown"):
        suit_map = {
            "blanks": 0, "ones": 1, "twos": 2, "threes": 3,
            "fours": 4, "fives": 5, "sixes": 6,
        }
        trump_suit = suit_map.get(decl_name)

    is_double = high == low
    is_trump = False
    if decl_name == "no-trump":
        is_trump = False
    elif decl_name == "doubles":
        is_trump = is_double
    elif decl_name == "doubles-suit":
        is_trump = is_double  # doubles-suit: all doubles are trump
    elif trump_suit is not None:
        is_trump = (high == trump_suit) or (low == trump_suit)

    # Count points: 5 (5-0, 5-5) and 10 (6-4)
    count_pts = 0
    if high == 5 and low == 0:
        count_pts = 5
    elif high == 5 and low == 5:
        count_pts = 10
    elif high == 6 and low == 4:
        count_pts = 10
    is_count = count_pts > 0

    if is_trump and is_double:
        category = "trump_double"
    elif is_trump and is_count:
        category = "trump_count"
    elif is_trump:
        category = "trump_plain"
    elif is_double:
        category = "offsuit_double"
    elif is_count:
        category = "count_tile"
    else:
        category = "plain_tile"

    return {
        "tile_is_trump": is_trump,
        "tile_is_double": is_double,
        "tile_is_count": is_count,
        "tile_category": category,
        "tile_count_pts": count_pts,
    }


def compute_detector_correlation(
    combined: pd.DataFrame,
    top_k_df: pd.DataFrame,
    joined_df: pd.DataFrame,
) -> pd.DataFrame:
    """Cross-tab load-bearing tile patterns against detector tags.

    Since the datasets share no seeds, we cross-tab by:
    (decl_name, seat_role) as categorical context bridge.

    For each (decl_name, seat_role, tile_category), we compute:
    - mean_impact_score from hidden_threat corpus
    - feature_label co-occurrence rates from joined_claim corpus
    """
    rows = []

    # --- Side A: hidden-threat tile categories by (decl_name, seat_role) ---
    top1 = top_k_df[top_k_df["rank"] == 1].copy()
    meta = build_decision_meta(combined)
    top1 = top1.merge(meta[["decl_name", "seat_role"]], left_on="decision_key", right_index=True, how="left")

    # Add tile classification
    tile_class = top1.apply(
        lambda r: classify_tile_type(str(r["hidden_domino"]), str(r["decl_name"])), axis=1
    )
    top1 = pd.concat([top1, pd.DataFrame(list(tile_class))], axis=1)

    # Group by (decl_name, seat_role, tile_category)
    threat_grp = (
        top1.groupby(["decl_name", "seat_role", "tile_category"])
        .agg(
            n_decisions=("decision_key", "nunique"),
            mean_impact=("max_impact_score", "mean"),
            mean_downside=("max_downside_score", "mean"),
            mean_concentration=("impact_fraction_of_total", "mean"),
            pct_harmful=("direction", lambda s: (s == "harmful").mean()),
            pct_helpful=("direction", lambda s: (s == "helpful").mean()),
        )
        .reset_index()
    )

    # --- Side B: detector tags by (decl_name, seat_role) from joined table ---
    # Explode feature_labels
    joined_exploded = joined_df.copy()
    joined_exploded["feature_labels"] = joined_exploded["feature_labels"].fillna("")
    label_rows = []
    for _, row in joined_exploded.iterrows():
        labels = [l for l in str(row["feature_labels"]).split("|") if l]
        for label in labels:
            label_rows.append({
                "decl_name": row["decl_name"],
                "seat_role": row["seat_role"],
                "feature_label": label,
            })
    label_df = pd.DataFrame(label_rows)

    # Compute label frequencies per (decl_name, seat_role)
    label_counts = (
        label_df.groupby(["decl_name", "seat_role", "feature_label"])
        .size()
        .reset_index(name="label_count")
    )
    group_totals = (
        joined_exploded.groupby(["decl_name", "seat_role"])
        .size()
        .reset_index(name="group_total")
    )
    label_counts = label_counts.merge(group_totals, on=["decl_name", "seat_role"], how="left")
    label_counts["label_rate"] = label_counts["label_count"] / label_counts["group_total"].clip(lower=1)

    # Key detector tags for hidden-related claims
    KEY_DETECTORS = [
        "hidden_proxy_early_high_uncertainty",
        "pounce_window",
        "setter_count_window",
        "partner_forcedness",
        "bidder_lead_plan",
        "ch04_low_trump_trap",
        "ch05_count_protection",
        "ch08_one_to_four_last_trick",
    ]

    # For each (decl_name, seat_role, tile_category) combo, add detector rates
    detector_pivot = label_counts[label_counts["feature_label"].str.startswith(
        ("hidden_proxy", "pounce", "setter_count", "partner_force", "bidder_lead", "ch04", "ch05", "ch08")
    )].copy()
    detector_rates_grp = (
        detector_pivot.groupby(["decl_name", "seat_role", "feature_label"])["label_rate"].first()
    )

    for _, row in threat_grp.iterrows():
        dn = row["decl_name"]
        sr = row["seat_role"]
        tc = row["tile_category"]

        # Get detector rates for this (decl_name, seat_role) slice
        try:
            slice_rates = detector_rates_grp.xs((dn, sr), level=("decl_name", "seat_role"))
        except KeyError:
            slice_rates = pd.Series(dtype=float)

        out_row: dict[str, Any] = {
            "decl_name": dn,
            "seat_role": sr,
            "tile_category": tc,
            "n_decisions": int(row["n_decisions"]),
            "mean_impact_score": round(float(row["mean_impact"]), 4),
            "mean_downside_score": round(float(row["mean_downside"]), 4),
            "mean_concentration_pct": round(float(row["mean_concentration"]), 4),
            "pct_harmful": round(float(row["pct_harmful"]), 4),
            "pct_helpful": round(float(row["pct_helpful"]), 4),
        }
        # Add selected detector rates
        for det in ["hidden_proxy_early_high_uncertainty", "pounce_window",
                    "setter_count_before_certainty", "partner_forcedness_and_safety",
                    "bidder_lead_plan", "bidder_first_lead_plan",
                    "late_trick_threshold_closure", "doubles_regime_plan"]:
            out_row[f"det_rate_{det}"] = round(float(slice_rates.get(det, 0.0)), 4)
        rows.append(out_row)

    return pd.DataFrame(rows)


def compute_tile_type_enrichment(
    combined: pd.DataFrame, top_k_df: pd.DataFrame
) -> list[dict[str, Any]]:
    """Compute enrichment of tile categories among load-bearing top-1 tiles vs baseline."""
    meta = build_decision_meta(combined)
    top1 = top_k_df[top_k_df["rank"] == 1].copy()
    top1 = top1.merge(meta[["decl_name"]], left_on="decision_key", right_index=True, how="left")

    tile_class = top1.apply(
        lambda r: classify_tile_type(str(r["hidden_domino"]), str(r["decl_name"])), axis=1
    )
    top1 = pd.concat([top1, pd.DataFrame(list(tile_class))], axis=1)

    # Baseline: all tiles in the corpus (any rank)
    all_tiles = top_k_df.copy()
    all_tiles = all_tiles.merge(meta[["decl_name"]], left_on="decision_key", right_index=True, how="left")
    all_class = all_tiles.apply(
        lambda r: classify_tile_type(str(r["hidden_domino"]), str(r["decl_name"])), axis=1
    )
    all_tiles = pd.concat([all_tiles, pd.DataFrame(list(all_class))], axis=1)

    categories = ["trump_double", "trump_count", "trump_plain", "offsuit_double", "count_tile", "plain_tile"]
    total_top1 = len(top1)
    total_all = len(all_tiles)
    rows = []
    for cat in categories:
        n_top1 = (top1["tile_category"] == cat).sum()
        n_all = (all_tiles["tile_category"] == cat).sum()
        rate_top1 = n_top1 / max(total_top1, 1)
        rate_all = n_all / max(total_all, 1)
        enrichment = rate_top1 / max(rate_all, 1e-6)
        rows.append({
            "tile_category": cat,
            "n_top1": int(n_top1),
            "n_all_topk": int(n_all),
            "rate_as_top1": round(rate_top1, 4),
            "rate_in_topk": round(rate_all, 4),
            "enrichment_ratio": round(enrichment, 3),
        })
    return rows


def load_joined_claims() -> pd.DataFrame:
    return pd.read_csv(JOINED_CLAIMS)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    started = datetime.now(UTC).isoformat()
    print(f"[{started}] Loading hidden-threat corpora...")
    combined = load_combined_hidden_threat()
    print(f"  Combined: {len(combined):,} rows, {combined['decision_key'].nunique()} decisions")

    print("Loading joined claims table...")
    joined_df = load_joined_claims()
    print(f"  Joined: {len(joined_df):,} rows")

    # =========================================================================
    # STEP 1: Per-decision top-K tile ranking
    # =========================================================================
    print("Computing per-decision top-K tile ranking...")
    top_k_df = rank_tiles_per_decision(combined, top_k=TOP_K)
    print(f"  Top-K rows: {len(top_k_df):,}")

    meta = build_decision_meta(combined)
    # Add full meta to top_k_df for output
    top_k_out = top_k_df.merge(
        meta[["decl_name", "seat_role", "team", "actor", "decl_id", "decision_idx", "seed", "game_idx", "source_corpus"]],
        left_on="decision_key",
        right_index=True,
        how="left",
    )

    # Add tile type classification
    tile_class = top_k_out.apply(
        lambda r: classify_tile_type(str(r["hidden_domino"]), str(r["decl_name"])), axis=1
    )
    top_k_out = pd.concat([top_k_out.reset_index(drop=True), pd.DataFrame(list(tile_class))], axis=1)

    # =========================================================================
    # STEP 2: Single-tile dominant decisions (top-100 concentration)
    # =========================================================================
    print("Computing single-tile dominant decisions...")
    single_tile_df = compute_single_tile_dominance(combined, top_k_df)
    print(f"  Single-tile dominant rows: {len(single_tile_df)}")

    # Add tile type
    tile_class2 = single_tile_df.apply(
        lambda r: classify_tile_type(str(r["hidden_domino"]), str(r["decl_name"])), axis=1
    )
    single_tile_df = pd.concat(
        [single_tile_df.reset_index(drop=True), pd.DataFrame(list(tile_class2))], axis=1
    )

    # =========================================================================
    # STEP 3: Inferability split
    # =========================================================================
    print("Computing inferability split...")
    inferable_df = compute_inferability(combined, top_k_df)
    print(f"  Inferable rows: {len(inferable_df)}")

    tile_class3 = inferable_df.apply(
        lambda r: classify_tile_type(str(r["hidden_domino"]), str(r["decl_name"])), axis=1
    )
    inferable_df = pd.concat(
        [inferable_df.reset_index(drop=True), pd.DataFrame(list(tile_class3))], axis=1
    )

    # =========================================================================
    # STEP 4: Detector correlation cross-tab
    # =========================================================================
    print("Computing detector correlation cross-tab...")
    det_corr_df = compute_detector_correlation(combined, top_k_df, joined_df)
    print(f"  Detector correlation rows: {len(det_corr_df)}")

    # =========================================================================
    # STEP 5: Tile category enrichment
    # =========================================================================
    print("Computing tile category enrichment...")
    enrichment = compute_tile_type_enrichment(combined, top_k_df)

    # =========================================================================
    # STEP 6: Headline statistics for summary.json
    # =========================================================================
    print("Computing headline statistics...")

    n_decisions = combined["decision_key"].nunique()
    n_da_pairs = combined["da_key"].nunique()

    top1 = top_k_df[top_k_df["rank"] == 1].copy()
    top1_meta = top1.merge(meta[["decl_name", "seat_role"]], left_on="decision_key", right_index=True, how="left")
    tile_class4 = top1_meta.apply(
        lambda r: classify_tile_type(str(r["hidden_domino"]), str(r["decl_name"])), axis=1
    )
    top1_meta = pd.concat([top1_meta.reset_index(drop=True), pd.DataFrame(list(tile_class4))], axis=1)

    # Concentration stats
    concentration_vals = top1["impact_fraction_of_total"].dropna()
    pct_single_dominant = (concentration_vals >= CONCENTRATION_THRESH_PCT).mean()

    # Inferability stats
    inferable_all = inferable_df["inferability"].value_counts(normalize=True)

    # Tile category distribution at top-1
    tile_cat_dist = top1_meta["tile_category"].value_counts(normalize=True).to_dict()

    # Most common load-bearing tile overall
    top_tile_overall = (
        top1.groupby(["hidden_domino", "absolute_holder_role"])["max_impact_score"]
        .agg(["count", "mean"])
        .sort_values("mean", ascending=False)
        .head(10)
        .reset_index()
        .to_dict(orient="records")
    )

    # Direction distribution
    direction_dist = top1["direction"].value_counts(normalize=True).to_dict()

    # Legacy mining alignment: load legacy driver rollup for comparison
    legacy_top5 = []
    try:
        legacy_dr = pd.read_csv(LEGACY_DRIVER_ROLLUP)
        legacy_impact_top5 = (
            legacy_dr[legacy_dr["driver_kind"] == "impact"]
            .sort_values("mean_score", ascending=False)
            .head(5)[["hidden_domino", "holder_role", "decl_name", "seat_role", "mean_score"]]
            .to_dict(orient="records")
        )
        legacy_top5 = legacy_impact_top5
    except Exception as e:
        legacy_top5 = [{"error": str(e)}]

    # =========================================================================
    # WRITE OUTPUTS
    # =========================================================================
    print("Writing outputs...")

    # 1. per_decision_top_k_tiles.csv
    top_k_fields = [
        "decision_key", "source_corpus", "seed", "game_idx", "decl_id", "decl_name",
        "decision_idx", "actor", "seat_role", "team",
        "rank",
        "hidden_domino_id", "hidden_domino",
        "absolute_holder", "absolute_holder_role", "relative_holder",
        "max_impact_score", "max_downside_score", "max_upside_score",
        "mean_q_delta", "tail_low_mass_delta", "shelf_high_mass_delta",
        "conditioned_mass", "direction",
        "impact_fraction_of_total", "decision_total_impact",
        "n_actions",
        "tile_is_trump", "tile_is_double", "tile_is_count", "tile_category", "tile_count_pts",
    ]
    write_csv(
        OUT_DIR / "per_decision_top_k_tiles.csv",
        top_k_out.to_dict(orient="records"),
        top_k_fields,
    )
    print(f"  Wrote per_decision_top_k_tiles.csv ({len(top_k_out)} rows)")

    # 2. single_tile_dominant_decisions.csv
    single_fields = [
        "decision_key", "source_corpus", "seed", "game_idx", "decl_id", "decl_name",
        "decision_idx", "actor", "seat_role", "team",
        "hidden_domino", "absolute_holder_role",
        "max_impact_score", "max_downside_score",
        "mean_q_delta", "tail_low_mass_delta", "conditioned_mass",
        "direction", "concentration_pct", "is_single_tile_dominant",
        "tile_is_trump", "tile_is_double", "tile_is_count", "tile_category", "tile_count_pts",
    ]
    write_csv(
        OUT_DIR / "single_tile_dominant_decisions.csv",
        single_tile_df.to_dict(orient="records"),
        single_fields,
    )
    print(f"  Wrote single_tile_dominant_decisions.csv ({len(single_tile_df)} rows)")

    # 3. inferable_vs_not.csv
    inferable_fields = [
        "decision_key", "source_corpus", "seed", "game_idx", "decl_id", "decl_name",
        "decision_idx", "actor", "seat_role", "team",
        "hidden_domino", "absolute_holder_role",
        "max_impact_score", "max_downside_score",
        "mean_q_delta", "conditioned_mass",
        "direction", "inferability", "impact_fraction_of_total",
        "tile_is_trump", "tile_is_double", "tile_is_count", "tile_category", "tile_count_pts",
    ]
    write_csv(
        OUT_DIR / "inferable_vs_not.csv",
        inferable_df.to_dict(orient="records"),
        inferable_fields,
    )
    print(f"  Wrote inferable_vs_not.csv ({len(inferable_df)} rows)")

    # 4. detector_correlation.csv
    det_fields = list(det_corr_df.columns)
    write_csv(
        OUT_DIR / "detector_correlation.csv",
        det_corr_df.to_dict(orient="records"),
        det_fields,
    )
    print(f"  Wrote detector_correlation.csv ({len(det_corr_df)} rows)")

    # 5. summary.json
    summary = {
        "schema_version": "w42.book_validation_v1.wave1.hidden_threat_impact_ranker.v0",
        "bead_id": "t42-c2y9",
        "parent_epic": "t42-4zi6",
        "created_at_utc": started,
        "git_commit": git_sha(),
        "question": (
            "For each decision where hidden-holder labels exist, which specific unseen tile "
            "(in which seat) drives the largest swing in the outcome distribution, and is that "
            "'load-bearing tile' predictable from book-detector vocabulary?"
        ),
        "slice": {
            "corpus_v1": str(DATA_V1),
            "corpus_scaled": str(DATA_SCALED),
            "joined_claims": str(JOINED_CLAIMS),
            "note": (
                "hidden_threat corpora use seeds 9420/9421/9430 (not in joined_claim seeds 0-99); "
                "cross-tab bridges via (decl_name, seat_role) categorical context, not row-level join."
            ),
        },
        "N": {
            "total_hidden_threat_rows": int(len(combined)),
            "unique_decisions": int(n_decisions),
            "unique_decision_action_pairs": int(n_da_pairs),
            "top_k_rows": int(len(top_k_df)),
            "single_tile_dominant_rows": int(len(single_tile_df)),
            "inferable_rows": int(len(inferable_df)),
            "detector_correlation_rows": int(len(det_corr_df)),
        },
        "paired_or_unpaired": "unpaired within-decision tile ranking; cross-tab by shared categorical context",
        "metric": {
            "primary": "impact_score = |mean_q_delta| + 10*(|tail_low_mass_delta| + |shelf_high_mass_delta|)",
            "concentration": "impact_fraction_of_total = top1_impact / sum(all_tile_impacts_in_decision)",
            "inferability": "conditioned_mass (fraction of worlds with tile at that position)",
        },
        "claim_ledger_impact": "underpowered",
        "claim_ledger_impact_note": (
            "Wave 1.3 provides offline tile-ranking labels and categorical cross-tab evidence. "
            "No row-level join to book-chapter detectors is possible (seed mismatch). "
            "Evidence is a diagnostic marker only; no claim status promoted to 'supported'."
        ),
        "headline": {
            "pct_decisions_single_tile_dominant": round(float(pct_single_dominant), 4),
            "concentration_threshold_used": CONCENTRATION_THRESH_PCT,
            "tile_category_distribution_top1": {k: round(v, 4) for k, v in tile_cat_dist.items()},
            "direction_distribution": {k: round(v, 4) for k, v in direction_dist.items()},
            "inferability_distribution_top100": inferable_all.to_dict(),
            "top_load_bearing_tile_overall": top_tile_overall[:5],
            "legacy_mining_top5_impact_drivers": legacy_top5,
            "tile_category_enrichment": enrichment,
        },
        "caveats": [
            "Only 324 decisions across two small generated game slices (seeds 9420/9421/9430).",
            "No row-level join to joined_claim_action_rows (seeds 0-99); cross-tab is categorical only.",
            "Inferability heuristic uses conditioned_mass only — does not use actual game history.",
            "impact_score formula reused verbatim from hidden_threat_legacy_mining; no modification.",
            "Hidden-holder labels are offline eval only; not proposed as live features.",
        ],
        "artifacts": {
            "per_decision_top_k_tiles": str(OUT_DIR / "per_decision_top_k_tiles.csv"),
            "single_tile_dominant_decisions": str(OUT_DIR / "single_tile_dominant_decisions.csv"),
            "inferable_vs_not": str(OUT_DIR / "inferable_vs_not.csv"),
            "detector_correlation": str(OUT_DIR / "detector_correlation.csv"),
            "manifest": str(OUT_DIR / "manifest.json"),
        },
    }
    write_json(OUT_DIR / "summary.json", summary)
    print("  Wrote summary.json")

    # 6. manifest.json
    manifest = {
        "schema_version": "w42.book_validation_v1.wave1.manifest.v0",
        "bead_id": "t42-c2y9",
        "parent_epic": "t42-4zi6",
        "created_at_utc": started,
        "git_commit": git_sha(),
        "command": f"python {__file__}",
        "inputs": {
            "branch_atlas_v1_hidden_threat_rows": {
                "path": str(DATA_V1),
                "sha256_prefix": sha256_file(DATA_V1),
                "rows": int(len(pd.read_csv(DATA_V1))),
            },
            "branch_atlas_scaled_v0_hidden_threat_rows": {
                "path": str(DATA_SCALED),
                "sha256_prefix": sha256_file(DATA_SCALED),
                "rows": int(len(pd.read_csv(DATA_SCALED))),
            },
            "joined_claim_action_rows": {
                "path": str(JOINED_CLAIMS),
                "sha256_prefix": sha256_file(JOINED_CLAIMS),
                "rows": int(len(joined_df)),
            },
            "legacy_driver_rollup": {
                "path": str(LEGACY_DRIVER_ROLLUP),
                "sha256_prefix": sha256_file(LEGACY_DRIVER_ROLLUP),
            },
        },
        "outputs": {
            "per_decision_top_k_tiles": str(OUT_DIR / "per_decision_top_k_tiles.csv"),
            "single_tile_dominant_decisions": str(OUT_DIR / "single_tile_dominant_decisions.csv"),
            "inferable_vs_not": str(OUT_DIR / "inferable_vs_not.csv"),
            "detector_correlation": str(OUT_DIR / "detector_correlation.csv"),
            "summary": str(OUT_DIR / "summary.json"),
            "manifest": str(OUT_DIR / "manifest.json"),
        },
        "leakage_boundary": (
            "q_per_world, world_hands, E[Q|tile=t] are offline diagnostic labels only. "
            "Not proposed as live features. Inferability uses only public conditioned_mass proxy."
        ),
        "do_not_edit_central_ledger": True,
        "do_not_touch_dirs": [
            "w42/phase4_claim_completion_board/",
            "w42/phase4_final_claim_audit/",
        ],
    }
    write_json(OUT_DIR / "manifest.json", manifest)
    print("  Wrote manifest.json")

    print(f"\nDone. Outputs in: {OUT_DIR}")
    print(f"  Decisions analysed: {n_decisions}")
    print(f"  Pct single-tile dominant: {pct_single_dominant:.1%}")


if __name__ == "__main__":
    main()
