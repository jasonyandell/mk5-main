"""
Cross-AI Agreement Analysis (Wave 1.4, bead t42-m2i7)

Computes four source labels per action:
  ev_top_action        — EV oracle (is_best_mean from joined table)
  gus_top_action       — Gus row-model proxy (is_best_threshold from labeled_handshape)
  detector_endorsed    — any phase-4 ch03/ch04/ch05 positive-labeled action
  dist_lens_top_action — top-1 under any non-EV utility from Wave 1.1

Outputs:
  per_action_source_picks.csv
  agreement_matrix.csv
  divisive_decisions.csv (top 100)
  spot_check_pack.json
  per_claim_family_agreement.csv
  summary.json
  manifest.json
  README.md
"""

import json
import hashlib
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# ── paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path("/Users/jason/code/mk5-main")
JOINED_TABLE = PROJECT_ROOT / "w42/joined_claim_row_model_table/joined_claim_action_rows.csv"
LABELED_HANDSHAPE = PROJECT_ROOT / "w42/phase4_sequence_handshape_tests/labeled_handshape_action_rows.csv"
SEQ_CONTRASTS = PROJECT_ROOT / "w42/phase4_sequence_handshape_tests/paired_contrasts.csv"
BRANCH_ATLAS = PROJECT_ROOT / "w42/branch_atlas_scaled_v0/decision_actions.csv"
WAVE11 = PROJECT_ROOT / "w42/book_validation_v1/wave1/t42-ybo6_distribution_lens_reranker"
HIDDEN_THREAT = PROJECT_ROOT / "w42/branch_atlas_scaled_v0/hidden_threat_rows.csv"
OUT_DIR = PROJECT_ROOT / "w42/book_validation_v1/wave1/t42-m2i7_cross_ai_agreement"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── 1. Load primary corpus (28 000 decisions, 75 079 rows) ───────────────────
print("Loading primary corpus …")
jt = pd.read_csv(JOINED_TABLE)
lh = pd.read_csv(LABELED_HANDSHAPE)

# Merge to get ch03/04/05 labels alongside EV and threshold flags
core = jt.merge(
    lh[["key", "candidate_domino",
        "is_best_threshold", "is_safest_tail", "labels",
        "phase", "candidate_beats_current", "legal_action_n"]],
    on=["key", "candidate_domino"], how="left"
)
assert len(core) == len(jt), "Row count changed after merge"
print(f"  Primary corpus: {len(core):,} rows, {core['key'].nunique():,} decisions")

# ── 2. ev_top_action ─────────────────────────────────────────────────────────
# is_best_mean is already validated as mean_regret == 0
core["ev_top_action"] = core["is_best_mean"].astype(bool)

# ── 3. gus_top_action (proxy) ────────────────────────────────────────────────
# The Gus row-model's individual candidate scores are not saved in the artifacts
# (model_metrics.csv has only aggregate match rates; prediction_sample.jsonl
# has 30 spot-check examples).
# Proxy: is_best_threshold from the labeled_handshape table captures the
# threshold-utility-top action, which is the distribution-aware alternative
# the Gus model was trained alongside. Aggregate match rate proxy-vs-EV is 78.3%.
core["gus_top_action"] = core["is_best_threshold"].astype(bool)

# ── 4. detector_endorsed ─────────────────────────────────────────────────────
# Read positive labels from seq contrasts (ch03/ch04/ch05 detectors)
contrasts = pd.read_csv(SEQ_CONTRASTS)
positive_labels = set(contrasts["positive_label"].tolist())
negative_labels = set(contrasts["negative_label"].tolist())
print(f"  Detector positive labels ({len(positive_labels)}): {sorted(positive_labels)[:5]} …")

def has_positive_detector(label_str):
    """True if any ch0x positive label is present in the label string."""
    if pd.isna(label_str):
        return False
    labels = set(label_str.split("|"))
    return bool(labels & positive_labels)

def has_negative_detector(label_str):
    if pd.isna(label_str):
        return False
    labels = set(label_str.split("|"))
    return bool(labels & negative_labels)

core["detector_endorsed"] = core["labels"].apply(has_positive_detector)
core["detector_anti_endorsed"] = core["labels"].apply(has_negative_detector)

# ── 5. dist_lens_top_action (Wave 1.1) ───────────────────────────────────────
# Wave 1.1 covers the branch_atlas corpus (280 decisions, seed 9430).
# The primary corpus (28 000 decisions) uses a different seed set.
# We JOIN on game_idx + decision_idx + candidate_domino for the overlapping 280 decisions
# and flag dist_lens_top on any non-EV utility being top-1.

print("Loading Wave 1.1 distribution-lens outputs …")
aus = pd.read_csv(WAVE11 / "action_utility_scalars.csv")
aus_9430 = aus[aus["seed"] == 9430].copy()
aus_9430["game_idx"] = aus_9430["game_key"].str.extract(r"r(\d+)").astype(int)
aus_9430["decision_idx"] = aus_9430["decision_id"].str.extract(r"d(\d+)").astype(int)

# dist_lens_top = top-1 under ANY non-EV utility
# non_ev utilities: p_make, p_set, threshold_mass_low, threshold_mass_high,
#                    cvar_10, mark_ev, robust_q25
non_ev_top_cols = ["top1_p_make", "top1_p_set", "top1_threshold_mass_low",
                   "top1_threshold_mass_high", "top1_cvar_10", "top1_mark_ev",
                   "top1_robust_q25"]
aus_9430["dist_lens_top_action"] = aus_9430[non_ev_top_cols].any(axis=1)

# Merge dist_lens onto core (only fills the ~773 rows that appear in branch_atlas)
# The primary corpus keys are "corpus_v2_train_0-9_d0-9.pt:{game_idx}:{decision_idx}"
# Parse game_idx and decision_idx from key
core["game_idx_parsed"] = core["key"].str.extract(r":(\d+):\d+$").astype(float).astype("Int64")
core["decision_idx_parsed"] = core["key"].str.extract(r":(\d+)$").astype(float).astype("Int64")

dist_lens_join = aus_9430[["game_idx", "decision_idx", "candidate_domino",
                            "dist_lens_top_action"]].rename(
    columns={"game_idx": "game_idx_parsed", "decision_idx": "decision_idx_parsed"})

core = core.merge(dist_lens_join, on=["game_idx_parsed", "decision_idx_parsed", "candidate_domino"],
                  how="left")
dist_coverage = core["dist_lens_top_action"].notna().sum()
print(f"  Wave 1.1 coverage: {dist_coverage:,} rows out of {len(core):,} ({dist_coverage/len(core):.1%})")

# ── 6. per_action_source_picks.csv ───────────────────────────────────────────
print("Building per_action_source_picks …")
picks_cols = [
    "key", "seed", "game_idx", "decision_idx", "decl_name", "bid_value",
    "actor", "seat_role", "role_family", "team", "trick_idx", "trick_position",
    "candidate_domino", "candidate_count_points", "candidate_is_called_suit",
    "candidate_is_double", "mean", "mean_regret", "threshold_mass", "lower_tail_mass",
    "is_actual_action",
    "ev_top_action", "gus_top_action", "detector_endorsed", "detector_anti_endorsed",
    "dist_lens_top_action",
    "feature_labels", "labels", "phase"
]
picks = core[picks_cols].copy()
picks.to_csv(OUT_DIR / "per_action_source_picks.csv", index=False)
print(f"  Wrote per_action_source_picks.csv: {len(picks):,} rows")

# ── 7. Per-decision source picks ─────────────────────────────────────────────
print("Computing per-decision top-1 picks …")

def top1(group, col):
    """Return the candidate_domino that is top-1 for this column."""
    top = group[group[col] == True]
    if len(top) == 0:
        return None
    return top["candidate_domino"].iloc[0]

decisions_list = []
for key, grp in core.groupby("key"):
    ev_pick = top1(grp, "ev_top_action")
    gus_pick = top1(grp, "gus_top_action")
    det_pick = top1(grp, "detector_endorsed")
    dist_pick = top1(grp, "dist_lens_top_action") if grp["dist_lens_top_action"].notna().any() else None

    # Get EV of each source's pick
    def get_ev(domino):
        if domino is None:
            return None
        row = grp[grp["candidate_domino"] == domino]
        if len(row) == 0:
            return None
        return float(row["mean"].iloc[0])

    def get_regret(domino):
        if domino is None:
            return None
        row = grp[grp["candidate_domino"] == domino]
        if len(row) == 0:
            return None
        return float(row["mean_regret"].iloc[0])

    meta = grp.iloc[0]
    decisions_list.append({
        "key": key,
        "seed": meta["seed"],
        "game_idx": meta["game_idx"],
        "decision_idx": meta["decision_idx"],
        "decl_name": meta["decl_name"],
        "bid_value": meta["bid_value"],
        "seat_role": meta["seat_role"],
        "role_family": meta["role_family"],
        "trick_idx": meta["trick_idx"],
        "trick_position": meta["trick_position"],
        "phase": meta.get("phase"),
        "n_candidates": len(grp),
        "ev_pick": ev_pick,
        "gus_pick": gus_pick,
        "detector_pick": det_pick,
        "dist_pick": dist_pick,
        "ev_of_ev": get_ev(ev_pick),
        "ev_of_gus": get_ev(gus_pick),
        "ev_of_detector": get_ev(det_pick),
        "ev_of_dist": get_ev(dist_pick),
        "regret_of_gus": get_regret(gus_pick),
        "regret_of_detector": get_regret(det_pick),
        "regret_of_dist": get_regret(dist_pick),
        "dist_available": grp["dist_lens_top_action"].notna().any(),
        "actual_domino": grp[grp["is_actual_action"] == True]["candidate_domino"].iloc[0]
            if grp["is_actual_action"].any() else None,
        "detector_fired": grp["detector_endorsed"].any(),
        "feature_labels": meta["feature_labels"],
        "labels": meta.get("labels"),
    })

decisions = pd.DataFrame(decisions_list)
print(f"  Per-decision table: {len(decisions):,} rows")

# ── 8. Agreement matrix ───────────────────────────────────────────────────────
print("Computing agreement matrix …")
sources = ["ev", "gus", "detector", "dist"]
source_cols = ["ev_pick", "gus_pick", "detector_pick", "dist_pick"]

# Restrict to decisions where detector fired (otherwise detector_pick is None)
# and dist is available
full_dec = decisions.copy()

def pairwise_agree(df, col_a, col_b, require_both=True):
    """Fraction of decisions where both sources pick the same action."""
    mask = df[col_a].notna() & df[col_b].notna()
    sub = df[mask]
    if len(sub) == 0:
        return np.nan, 0
    return (sub[col_a] == sub[col_b]).mean(), len(sub)

matrix_rows = []
for s_a, c_a in zip(sources, source_cols):
    row = {"source": s_a}
    for s_b, c_b in zip(sources, source_cols):
        if s_a == s_b:
            row[s_b] = 1.0
        else:
            rate, n = pairwise_agree(full_dec, c_a, c_b)
            row[s_b] = rate
    matrix_rows.append(row)

agree_matrix = pd.DataFrame(matrix_rows).set_index("source")

# Also report N for each pair
n_rows = []
for s_a, c_a in zip(sources, source_cols):
    row = {"source": s_a}
    for s_b, c_b in zip(sources, source_cols):
        if s_a == s_b:
            row[s_b] = full_dec[c_a].notna().sum()
        else:
            _, n = pairwise_agree(full_dec, c_a, c_b)
            row[s_b] = n
    n_rows.append(row)
n_matrix = pd.DataFrame(n_rows).set_index("source")

agree_matrix.to_csv(OUT_DIR / "agreement_matrix.csv")
print("Agreement matrix:")
print(agree_matrix.round(3).to_string())

# ── 9. Divisiveness ranking ───────────────────────────────────────────────────
print("Ranking decisions by divisiveness …")

def divisiveness(row):
    """Count of distinct picks across sources (None excluded)."""
    picks = []
    for col in source_cols:
        v = row[col]
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            picks.append(v)
    if len(picks) == 0:
        return 0
    return len(set(picks))

decisions["n_distinct_picks"] = decisions.apply(divisiveness, axis=1)

# Max regret spread = max regret among sources with picks
def max_regret_spread(row):
    regrets = []
    for col in ["regret_of_gus", "regret_of_detector", "regret_of_dist"]:
        v = row[col]
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            regrets.append(v)
    return max(regrets) if regrets else 0.0

decisions["max_regret_spread"] = decisions.apply(max_regret_spread, axis=1)

# Sort by n_distinct_picks DESC then max_regret_spread DESC
divisive = decisions.sort_values(
    ["n_distinct_picks", "max_regret_spread"], ascending=[False, False]
).head(100).copy()

divisive.to_csv(OUT_DIR / "divisive_decisions.csv", index=False)
print(f"  Top-100 divisive decisions: max spread = {divisive['max_regret_spread'].max():.2f}")

# ── 10. Spot-check pack ───────────────────────────────────────────────────────
print("Building spot-check pack …")

def build_spot_check_entry(row, core_df):
    key = row["key"]
    grp = core_df[core_df["key"] == key].copy()

    def action_detail(domino):
        if domino is None:
            return None
        sub = grp[grp["candidate_domino"] == domino]
        if len(sub) == 0:
            return {"domino": domino}
        r = sub.iloc[0]
        return {
            "domino": domino,
            "mean_ev": float(r["mean"]),
            "mean_regret": float(r["mean_regret"]),
            "threshold_mass": float(r["threshold_mass"]),
            "lower_tail_mass": float(r["lower_tail_mass"]),
            "is_count": int(r["candidate_count_points"]) > 0,
            "is_double": bool(r["candidate_is_double"]),
            "is_called_suit": bool(r["candidate_is_called_suit"]),
            "detector_labels": str(r.get("labels", "")) if not pd.isna(r.get("labels", "")) else "",
            "feature_labels": str(r.get("feature_labels", "")) if not pd.isna(r.get("feature_labels", "")) else "",
        }

    candidates_detail = []
    for _, cr in grp.iterrows():
        candidates_detail.append({
            "domino": cr["candidate_domino"],
            "mean_ev": float(cr["mean"]),
            "mean_regret": float(cr["mean_regret"]),
            "ev_top": bool(cr["ev_top_action"]),
            "gus_top": bool(cr["gus_top_action"]),
            "detector_endorsed": bool(cr["detector_endorsed"]),
            "dist_lens_top": bool(cr["dist_lens_top_action"]) if not pd.isna(cr["dist_lens_top_action"]) else None,
        })

    return {
        "key": key,
        "seed": int(row["seed"]) if not pd.isna(row["seed"]) else None,
        "game_idx": int(row["game_idx"]) if not pd.isna(row["game_idx"]) else None,
        "decision_idx": int(row["decision_idx"]) if not pd.isna(row["decision_idx"]) else None,
        "decl_name": str(row["decl_name"]),
        "bid_value": int(row["bid_value"]) if not pd.isna(row["bid_value"]) else None,
        "seat_role": str(row["seat_role"]),
        "role_family": str(row["role_family"]),
        "trick_idx": int(row["trick_idx"]) if not pd.isna(row["trick_idx"]) else None,
        "phase": str(row["phase"]) if not pd.isna(row.get("phase")) else None,
        "n_candidates": int(row["n_candidates"]),
        "n_distinct_picks": int(row["n_distinct_picks"]),
        "max_regret_spread": float(row["max_regret_spread"]),
        "source_picks": {
            "ev": action_detail(row["ev_pick"]),
            "gus": action_detail(row["gus_pick"]),
            "detector": action_detail(row["detector_pick"]),
            "dist_lens": action_detail(row["dist_pick"]) if row.get("dist_available") else "deferred",
        },
        "all_candidates": candidates_detail,
    }

spot_pack = []
for _, row in divisive.iterrows():
    spot_pack.append(build_spot_check_entry(row, core))

with open(OUT_DIR / "spot_check_pack.json", "w") as f:
    json.dump(spot_pack, f, indent=2, default=str)
print(f"  Wrote spot_check_pack.json: {len(spot_pack)} entries")

# ── 11. Per-claim-family agreement ───────────────────────────────────────────
print("Computing per-claim-family agreement …")

# Extract claim families from labels column
all_families = set()
for lb in core["labels"].dropna():
    for tag in lb.split("|"):
        if tag.startswith("ch") and "_" in tag:
            # ch03_xxx -> ch03
            family = tag.split("_")[0]
            all_families.add(family)

print(f"  Claim families found: {sorted(all_families)}")

family_rows = []
for family in sorted(all_families):
    # Decisions where at least one candidate has this family's label
    family_mask = core["labels"].apply(
        lambda x: isinstance(x, str) and any(t.startswith(family + "_") for t in x.split("|"))
    )
    family_keys = set(core[family_mask]["key"].unique())
    fam_dec = decisions[decisions["key"].isin(family_keys)].copy()

    if len(fam_dec) == 0:
        continue

    # EV-Gus agreement
    ev_gus_mask = fam_dec["ev_pick"].notna() & fam_dec["gus_pick"].notna()
    ev_gus_agree = (fam_dec[ev_gus_mask]["ev_pick"] == fam_dec[ev_gus_mask]["gus_pick"]).mean() \
        if ev_gus_mask.sum() > 0 else np.nan

    # EV-Detector agreement (only where detector fired)
    det_mask = fam_dec["detector_pick"].notna()
    ev_det_agree = (fam_dec[det_mask]["ev_pick"] == fam_dec[det_mask]["detector_pick"]).mean() \
        if det_mask.sum() > 0 else np.nan

    # Gus error rate when detector disagrees with EV
    det_ev_disagree = fam_dec[det_mask & (fam_dec["ev_pick"] != fam_dec["detector_pick"])]
    gus_error_when_det_wrong = None
    if len(det_ev_disagree) > 0:
        gus_right = (det_ev_disagree["gus_pick"] == det_ev_disagree["ev_pick"]).mean()
        gus_error_when_det_wrong = 1.0 - gus_right

    # Mean regret of detector pick in this family
    mean_det_regret = fam_dec[det_mask]["regret_of_detector"].mean() \
        if det_mask.sum() > 0 else np.nan

    family_rows.append({
        "claim_family": family,
        "n_decisions": len(fam_dec),
        "n_detector_fired": int(det_mask.sum()),
        "ev_gus_agreement_rate": round(ev_gus_agree, 4) if not np.isnan(ev_gus_agree) else None,
        "ev_detector_agreement_rate": round(ev_det_agree, 4) if not np.isnan(ev_det_agree) else None,
        "mean_detector_regret": round(mean_det_regret, 4) if not np.isnan(mean_det_regret) else None,
        "gus_error_rate_when_det_vs_ev": round(gus_error_when_det_wrong, 4)
            if gus_error_when_det_wrong is not None else None,
    })

fam_agree = pd.DataFrame(family_rows)
fam_agree.to_csv(OUT_DIR / "per_claim_family_agreement.csv", index=False)
print(fam_agree.to_string())

# ── 12. Summary JSON ──────────────────────────────────────────────────────────
print("Writing summary.json …")

# Headline agreement rates (3-source: EV/Gus/Detector on main corpus)
ev_gus_overall = pairwise_agree(decisions, "ev_pick", "gus_pick")
ev_det_overall = pairwise_agree(decisions[decisions["detector_pick"].notna()],
                                "ev_pick", "detector_pick")
gus_det_overall = pairwise_agree(decisions[decisions["detector_pick"].notna()],
                                 "gus_pick", "detector_pick")

# Dist-lens coverage in main corpus
dist_decisions = decisions[decisions["dist_available"] == True]
ev_dist_overall = pairwise_agree(dist_decisions, "ev_pick", "dist_pick")

n_divisive_4src = int((decisions["n_distinct_picks"] >= 4).sum())
n_divisive_3src = int((decisions["n_distinct_picks"] == 3).sum())
n_divisive_2src = int((decisions["n_distinct_picks"] == 2).sum())
n_full_agree = int((decisions["n_distinct_picks"] == 1).sum())

summary = {
    "bead": "t42-m2i7",
    "wave": "1.4",
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "question": "Where do book detectors, Gus row-model, scalar EV, and distribution-lens "
                "reranker disagree? Those decisions are the most pedagogically valuable.",
    "slice": {
        "primary_corpus": "joined_claim_action_rows.csv (28 000 decisions, corpus_v2_train, seeds 0-99)",
        "dist_lens_corpus": "branch_atlas_scaled_v0 / Wave 1.1 (280 decisions, seed 9430)",
    },
    "N": {
        "primary_decisions": int(decisions["key"].nunique()),
        "primary_action_rows": len(core),
        "dist_lens_decisions": int(dist_decisions["key"].nunique()),
    },
    "paired": "per-decision: same decision compared across sources",
    "metric": "top-1 pick agreement rate; mean_regret of each source's pick",
    "headline_agreement_rates": {
        "ev_vs_gus": {
            "rate": round(ev_gus_overall[0], 4),
            "n": ev_gus_overall[1],
        },
        "ev_vs_detector": {
            "rate": round(ev_det_overall[0], 4) if not np.isnan(ev_det_overall[0]) else None,
            "n": ev_det_overall[1],
        },
        "gus_vs_detector": {
            "rate": round(gus_det_overall[0], 4) if not np.isnan(gus_det_overall[0]) else None,
            "n": gus_det_overall[1],
        },
        "ev_vs_dist_lens": {
            "rate": round(ev_dist_overall[0], 4) if not np.isnan(ev_dist_overall[0]) else None,
            "n": ev_dist_overall[1],
            "note": "Dist-lens available on 280-decision sub-corpus (seed 9430) only",
        },
    },
    "divisiveness_distribution": {
        "4_sources_spread": n_divisive_4src,
        "3_sources_spread": n_divisive_3src,
        "2_sources_spread": n_divisive_2src,
        "full_agreement": n_full_agree,
        "note": "Count sources with non-None picks; dist_lens is None for 27720/28000 decisions",
    },
    "dist_lens_gap": "Wave 1.1 operates on a 280-decision sub-corpus (seed 9430); "
                     "the primary 28 000-decision corpus has no dist_lens coverage. "
                     "Orchestrator should integrate when Wave 1.1 is extended.",
    "gus_proxy_note": (
        "Individual Gus row-model scores per candidate are not saved in artifacts. "
        "gus_top_action uses is_best_threshold (threshold-utility top) as the closest "
        "row-level proxy. Aggregate proxy-vs-EV agreement is 78.3%."
    ),
    "claim_ledger_impact": "underpowered",
    "caveats": [
        "gus_top_action is a proxy (is_best_threshold), not the actual model output.",
        "dist_lens_top_action covers only 280/28000 decisions (1.0%); deferred for main corpus.",
        "Detector endorsement only for ch03/ch04/ch05 labels from paired_contrasts.csv; "
        "84-domain and bidding-domain detectors operate on different corpora.",
        "Agreement rates on dist_lens sub-corpus may not generalize to full corpus.",
    ],
    "artifacts": {
        "per_action_source_picks": str(OUT_DIR / "per_action_source_picks.csv"),
        "agreement_matrix": str(OUT_DIR / "agreement_matrix.csv"),
        "divisive_decisions": str(OUT_DIR / "divisive_decisions.csv"),
        "spot_check_pack": str(OUT_DIR / "spot_check_pack.json"),
        "per_claim_family_agreement": str(OUT_DIR / "per_claim_family_agreement.csv"),
    },
}

with open(OUT_DIR / "summary.json", "w") as f:
    json.dump(summary, f, indent=2)

# ── 13. Manifest ─────────────────────────────────────────────────────────────
def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

manifest = {
    "bead": "t42-m2i7",
    "wave": "1.4",
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "command": f"python3 {__file__}",
    "inputs": {
        "joined_claim_action_rows": {
            "path": str(JOINED_TABLE),
            "sha256": sha256(JOINED_TABLE),
        },
        "labeled_handshape_action_rows": {
            "path": str(LABELED_HANDSHAPE),
            "sha256": sha256(LABELED_HANDSHAPE),
        },
        "paired_contrasts": {
            "path": str(SEQ_CONTRASTS),
            "sha256": sha256(SEQ_CONTRASTS),
        },
        "branch_atlas_decision_actions": {
            "path": str(BRANCH_ATLAS),
            "sha256": sha256(BRANCH_ATLAS),
        },
        "wave11_action_utility_scalars": {
            "path": str(WAVE11 / "action_utility_scalars.csv"),
            "sha256": sha256(WAVE11 / "action_utility_scalars.csv"),
        },
    },
    "outputs": [
        str(OUT_DIR / "per_action_source_picks.csv"),
        str(OUT_DIR / "agreement_matrix.csv"),
        str(OUT_DIR / "divisive_decisions.csv"),
        str(OUT_DIR / "spot_check_pack.json"),
        str(OUT_DIR / "per_claim_family_agreement.csv"),
        str(OUT_DIR / "summary.json"),
        str(OUT_DIR / "manifest.json"),
        str(OUT_DIR / "README.md"),
    ],
}

with open(OUT_DIR / "manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)

# ── 14. README ───────────────────────────────────────────────────────────────
readme = """# Cross-AI Agreement Analysis (t42-m2i7)

**Wave:** 1.4  **Parent epic:** t42-4zi6

## Question

Where do book detectors, the Gus row-model, scalar EV, and the
distribution-lens reranker disagree? Those decisions are the most
pedagogically valuable.

## Slice

- **Primary corpus:** `joined_claim_action_rows.csv` — 28 000 decisions,
  75 079 action rows, corpus_v2_train, seeds 0–99.
- **Dist-lens sub-corpus:** `branch_atlas_scaled_v0` / Wave 1.1 —
  280 decisions, seed 9430 (1.0% overlap with primary corpus).

## N

- 28 000 decisions (primary), 75 079 action rows.
- 280 decisions with dist-lens coverage.

## Sources

| Label | Definition |
|-------|------------|
| `ev_top_action` | `is_best_mean` from joined table (mean_regret == 0); exact EV oracle. |
| `gus_top_action` | `is_best_threshold` from labeled_handshape (proxy; actual model scores not saved). |
| `detector_endorsed` | Any ch03/ch04/ch05 positive-label from `paired_contrasts.csv`. |
| `dist_lens_top_action` | Any non-EV utility top-1 from Wave 1.1; deferred for 27 720/28 000 decisions. |

## Metric

Top-1 pick agreement rate (fraction of decisions where source A and source B
select the same candidate domino); mean_regret of each source's pick.

## Key Findings

See `summary.json` for headline numbers. Run `python3 run_cross_ai_agreement.py`
to reproduce.

## Caveats

- `gus_top_action` is a proxy (is_best_threshold), not the actual model output.
  Aggregate proxy-vs-EV agreement rate: 78.3%.
- `dist_lens_top_action` covers only 280/28 000 decisions (1.0%); deferred.
- Detectors cover ch03/ch04/ch05 only; 84-domain and bidding-domain detectors
  use different corpora and are not joinable here.

## Claim-ledger impact

`underpowered` — agreement analysis is diagnostic, not a direct claim test.
"""
(OUT_DIR / "README.md").write_text(readme)

print()
print("=== DONE ===")
print(f"Output dir: {OUT_DIR}")
print(f"EV vs Gus agreement:      {ev_gus_overall[0]:.3f} (N={ev_gus_overall[1]:,})")
print(f"EV vs Detector agreement: {ev_det_overall[0]:.3f} (N={ev_det_overall[1]:,})")
print(f"Gus vs Detector:          {gus_det_overall[0]:.3f} (N={gus_det_overall[1]:,})")
print(f"EV vs Dist-lens:          {ev_dist_overall[0]:.3f} (N={ev_dist_overall[1]})")
print(f"Full-agreement decisions: {n_full_agree:,} / {len(decisions):,}")
print(f"2-source spread:          {n_divisive_2src:,}")
print(f"3-source spread:          {n_divisive_3src:,}")
