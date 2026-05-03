#!/usr/bin/env python3
"""Wave 1.2 — Mark Utility Transform on branch_atlas_scaled_v0 Q PDFs.

Bead: t42-c6sa
Question: How often does the preferred action change when scoring Q values
          under Chapter 10 mark/match utility instead of raw point EV?

Method:
  1. Load q_per_world tensors from branch_atlas_scaled_v0.
  2. For each world w and action a, convert q_per_world[w,a] (point EV,
     Team 0 perspective, remaining differential) into a mark utility using
     the deterministic scoring transform from phase4_scoring_objective_tests.
  3. Compute mark_EV = mean over worlds of mark utility per action.
  4. Compare top-1 action under point EV vs mark EV.
  5. Slice flip rate by declaration, seat/role, bid, detector family, score state.
  6. Join with joined_claim_action_rows.csv for detector tags.
  7. Report top detector-endorsed flips and per-Ch10-claim correlations.

Scoring transform (reused from run_phase4_scoring_objective_tests.py):
  - q_per_world[w,a] ≈ team0_remaining_points - team1_remaining_points
  - At game start (no points yet captured), this equals the final differential.
  - At mid-game decision with pre-captured points [off_score, def_score]:
      team0_final = off_score_t0 + team0_remaining_from_q
    But since Q is remaining differential, we reconstruct:
      remaining_t0 = (q + remaining_total) / 2
    where remaining_total = 42 - pre_captured_total.
  - Final team0_points = pre_captured_t0 + remaining_t0
  - Apply score_hand_marks(bid, bidder_team, (team0_final, team1_final))
  - mark_utility[w,a] = mark_gained_by_team0 - mark_gained_by_team1
    (net mark delta, positive = Team 0 gains a mark this hand)

Per-world hook: fully maintained — mark_utility is computed per (world, action).
"""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import numpy as np
import pandas as pd

OUT_DIR = Path(__file__).parent
BEAD_ID = "t42-c6sa"
PT_PATH = ROOT / "w42/branch_atlas_scaled_v0/eq_pdf_s9430_d10_bid30_joint_1000s_v2.pt"
ACTIONS_CSV = ROOT / "w42/branch_atlas_scaled_v0/decision_actions.csv"
STATES_CSV = ROOT / "w42/branch_atlas_scaled_v0/decision_states.csv"
JOINED_CSV = ROOT / "w42/joined_claim_row_model_table/joined_claim_action_rows.csv"
COMPLETION_CSV = ROOT / "w42/phase4_claim_completion_board/completion_board.csv"

PHASE4_TRANSFORM_FILE = ROOT / "w42/phase4_scoring_objective_tests/run_phase4_scoring_objective_tests.py"

# Ch10 claim IDs we are correlating flips against
CH10_CLAIM_IDS = [
    "ch10-early-terminal-under-marks",
    "ch10-set-severity-compression",
    "ch10-special-bid-mark-multiplier",
    "ch10-low-bid-score-distortion",
    "ch10-tournament-speed-tradeoff",
    "ch10-score-mode-objective",
    "ch10-nonbidder-partial-points-erased",
    "ch10-point-system-skill-signal",
    "ch10-timed-marks-advancement-objective",
]

# Detector family prefixes from joined_claim_action_rows.csv
DETECTOR_FAMILIES = {
    "sequence": ["sequence"],
    "bidrisk": ["bidrisk"],
    "e84": ["e84"],
    "dnt": ["dnt"],
    "hidden": ["hidden"],
    "bidder_lead": ["bidder"],
    "setter": ["setter"],
    "partner": ["partner"],
    "closure": ["closure"],
    "follower": ["follower"],
    "ch04": ["ch04"],
    "ch05": ["ch05"],
    "seat": ["seat"],
    "phase": ["phase"],
    "role": ["role"],
}


# ──────────────────────────────────────────────────────────────────────────────
# Scoring transform functions (reused verbatim from phase4_scoring_objective_tests)
# ──────────────────────────────────────────────────────────────────────────────

def mark_multiplier(bid: int) -> int:
    if bid >= 84:
        return bid // 42
    return 1


def made_contract(bid: int, bidder_points: int) -> bool:
    if bid < 42:
        return bidder_points >= bid
    return bidder_points == 42


def score_hand_marks(bid: int, bidder_team: int, team_points: tuple[int, int]) -> tuple[int, int]:
    scores = [0, 0]
    winner_team = bidder_team if made_contract(bid, team_points[bidder_team]) else 1 - bidder_team
    scores[winner_team] = mark_multiplier(bid)
    return scores[0], scores[1]


def score_hand_points(bid: int, bidder_team: int, team_points: tuple[int, int]) -> tuple[int, int]:
    bidder_points = team_points[bidder_team]
    defender_team = 1 - bidder_team
    defender_points = team_points[defender_team]
    made = made_contract(bid, bidder_points)
    scores = [0, 0]
    if bid < 42:
        if made:
            scores[bidder_team] = bidder_points
            scores[defender_team] = defender_points
        else:
            scores[bidder_team] = 0
            scores[defender_team] = defender_points + bid
    else:
        if made:
            scores[bidder_team] = bid
            scores[defender_team] = 0
        else:
            scores[bidder_team] = 0
            scores[defender_team] = bid
    return scores[0], scores[1]


# ──────────────────────────────────────────────────────────────────────────────
# Per-world mark utility computation
# ──────────────────────────────────────────────────────────────────────────────

def q_to_mark_utility_per_world(
    q_per_world: torch.Tensor,  # shape (n_worlds, n_actions)
    legal_mask: torch.Tensor,   # shape (n_actions,)
    bid: int,
    bidder_team: int,
    pre_t0_points: int,         # points Team 0 has already captured before this decision
    pre_t1_points: int,         # points Team 1 has already captured before this decision
) -> torch.Tensor:
    """Convert per-world Q values to per-world mark utility (net marks for Team 0).

    V/Q semantics: q_per_world[w,a] = remaining (team0 - team1) point differential
    from this action onward, in world w. Points from Team 0's perspective.

    Steps:
      1. Remaining total available = 42 - pre_t0 - pre_t1
      2. remaining_t0[w,a] = (q[w,a] + remaining_total) / 2
      3. final_t0[w,a] = pre_t0 + remaining_t0[w,a]  (clamp 0..42)
      4. final_t1[w,a] = 42 - final_t0[w,a]
      5. Apply score_hand_marks(bid, bidder_team, (final_t0, final_t1))
      6. mark_utility[w,a] = mark_team0 - mark_team1

    Returns:
        mark_utility: shape (n_worlds, n_actions), illegal actions = 0.0
    """
    n_worlds, n_actions = q_per_world.shape
    remaining_total = 42 - pre_t0_points - pre_t1_points
    # remaining_t0[w,a] = (q[w,a] + remaining_total) / 2
    remaining_t0 = (q_per_world + remaining_total) / 2.0
    final_t0 = (torch.tensor(pre_t0_points, dtype=torch.float32) + remaining_t0).clamp(0, 42)
    final_t1 = 42.0 - final_t0  # always sums to 42

    mark_utility = torch.zeros_like(q_per_world)
    for a in range(n_actions):
        if not legal_mask[a].item():
            continue
        ft0 = final_t0[:, a]  # (n_worlds,)
        ft1 = final_t1[:, a]  # (n_worlds,)
        # Vectorized mark scoring
        bidder_points_w = ft0 if bidder_team == 0 else ft1
        if bid < 42:
            made_w = bidder_points_w >= bid
        else:
            made_w = bidder_points_w == 42.0
        multiplier = mark_multiplier(bid)
        # winner team 0 if (bidder_team==0 and made) or (bidder_team==1 and not made)
        if bidder_team == 0:
            mark_t0_w = torch.where(made_w, torch.tensor(float(multiplier)), torch.tensor(0.0))
            mark_t1_w = torch.where(made_w, torch.tensor(0.0), torch.tensor(float(multiplier)))
        else:
            mark_t0_w = torch.where(made_w, torch.tensor(0.0), torch.tensor(float(multiplier)))
            mark_t1_w = torch.where(made_w, torch.tensor(float(multiplier)), torch.tensor(0.0))
        mark_utility[:, a] = mark_t0_w - mark_t1_w

    return mark_utility


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def score_bucket(off_score: int, def_score: int, bid: int) -> str:
    """Classify score state into strategic buckets."""
    total = off_score + def_score
    # Close to mark territory: either team within striking distance
    marks_to_win = 7
    if off_score >= bid - 10 or def_score >= bid - 10:
        return "near_threshold"
    if off_score >= 20 or def_score >= 20:
        return "blowout"
    if total <= 10:
        return "early_hand"
    return "mid_hand"


def detector_family(feature_labels: str) -> list[str]:
    """Map detector labels to phase-4 family buckets."""
    if not feature_labels or feature_labels == "nan":
        return ["none"]
    labels = [l.strip() for l in str(feature_labels).split("|") if l.strip()]
    families_found = set()
    for label in labels:
        for fam, prefixes in DETECTOR_FAMILIES.items():
            for pfx in prefixes:
                if label.startswith(pfx):
                    families_found.add(fam)
    return sorted(families_found) if families_found else ["none"]


def ch10_flip_correlations(flip_flag: bool, bid: int, made_w_majority: bool,
                            early_terminal: bool, set_bid: bool) -> dict[str, bool]:
    """Tag which Ch10 mechanisms are plausibly active for a given decision."""
    return {
        "ch10-early-terminal-under-marks": early_terminal,
        "ch10-set-severity-compression": not made_w_majority and bid < 42,
        "ch10-special-bid-mark-multiplier": bid >= 84,
        "ch10-low-bid-score-distortion": bid <= 31,
        "ch10-tournament-speed-tradeoff": early_terminal,
        "ch10-score-mode-objective": True,  # always active (objective always differs)
        "ch10-nonbidder-partial-points-erased": made_w_majority and bid < 42,
        "ch10-point-system-skill-signal": True,
        "ch10-timed-marks-advancement-objective": early_terminal,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main analysis
# ──────────────────────────────────────────────────────────────────────────────

def main() -> int:
    print(f"[mark_utility_transform] ROOT={ROOT}")
    print(f"[mark_utility_transform] Loading PT file: {PT_PATH}")

    # Load the per-world Q tensors
    data = torch.load(PT_PATH, map_location="cpu", weights_only=False)
    print(f"[mark_utility_transform] Loaded {len(data['results'])} game records")

    # Load decision_actions.csv for score state + detector tags
    da = pd.read_csv(ACTIONS_CSV)
    print(f"[mark_utility_transform] decision_actions.csv: {len(da)} rows")

    # Note: joined_claim_action_rows.csv uses a different corpus (seeds 0-9 from
    # corpus_v2_train_*) while branch_atlas_scaled_v0 uses seed 9430.
    # The two datasets do not overlap, so direct join is not possible.
    # Instead, we use the detector tags already embedded in decision_actions.csv:
    #   - matched_position_detectors (book position detectors)
    #   - distribution_shape_tags (distribution shape tags)
    #   - strategy_context_tags (strategy context)
    # These are the canonical source of book-endorsement for this atlas.
    print("[mark_utility_transform] Using decision_actions.csv detector tags (joined table is a different corpus)")

    # Build an index from decision_actions.csv: (game_idx, decision_idx) -> row slice
    # da has multiple rows per decision (one per candidate action)
    da_grouped = da.groupby(["game_idx", "decision_idx"])

    # Collect all results
    action_rows: list[dict[str, Any]] = []
    flip_decisions: list[dict[str, Any]] = []

    total_decisions = 0
    total_flips = 0

    for gi, game_record in enumerate(data["results"]):
        seed = data["seeds"][gi]
        decl_id = data["decl_ids"][gi]
        bid = game_record.bid_value
        decl_name = da[da["game_idx"] == gi]["decl_name"].iloc[0] if len(da[da["game_idx"] == gi]) > 0 else "unknown"

        # All decisions for this game
        for di, decision in enumerate(game_record.decisions):
            player = decision.player
            bidder_team = 0  # From decision_actions.csv: the bidder is always team 0 offense
            # Get pre-captured score state from decision_actions.csv
            da_rows = da[(da["game_idx"] == gi) & (da["decision_idx"] == di)]
            if len(da_rows) == 0:
                continue

            da_row0 = da_rows.iloc[0]
            seat_role = da_row0["seat_role"]
            team = da_row0["team"]
            # Determine bidder_team: bidder is always offense=team0 in this dataset
            # offense_score_before = points already captured by the offense (bidder's) team
            # defense_score_before = points already captured by the defense (setter) team
            off_score = int(da_row0["offense_score_before"])
            def_score = int(da_row0["defense_score_before"])

            # The bidder's team is team 0 (offense). So pre_t0 = off_score, pre_t1 = def_score.
            # NOTE: This is consistent for the scaled_v0 atlas where bidder=team0.
            pre_t0 = off_score
            pre_t1 = def_score

            # Per-world Q for this decision
            qpw = decision.q_per_world  # (n_worlds, 7)
            legal = decision.legal_mask  # (7,)
            n_legal = legal.sum().item()
            if n_legal == 0:
                continue

            # Compute per-world mark utility
            mark_util = q_to_mark_utility_per_world(
                qpw, legal, bid, bidder_team, pre_t0, pre_t1
            )  # (n_worlds, 7)

            # Compute point EV and mark EV per action
            e_q_point = qpw.mean(dim=0)   # (7,) - already e_q from decision
            e_q_mark = mark_util.mean(dim=0)  # (7,)

            # Mask illegal actions with -inf
            neg_inf = torch.tensor(float("-inf"))
            e_q_point_masked = torch.where(legal, e_q_point, neg_inf)
            e_q_mark_masked = torch.where(legal, e_q_mark, neg_inf)

            # Top-1 action under each objective
            top1_point = int(e_q_point_masked.argmax().item())
            top1_mark = int(e_q_mark_masked.argmax().item())
            flip = top1_point != top1_mark

            # EV cost when flipping: how much point EV do we lose by switching to mark-preferred?
            if flip:
                ev_cost = float(e_q_point[top1_point].item() - e_q_point[top1_mark].item())
                mark_gain = float(e_q_mark[top1_mark].item() - e_q_mark[top1_point].item())
            else:
                ev_cost = 0.0
                mark_gain = 0.0

            total_decisions += 1
            if flip:
                total_flips += 1

            # Score bucket
            sc_bucket = score_bucket(off_score, def_score, bid)

            # Majority-world outcome under top1_mark action
            made_arr = mark_util[:, top1_mark] > 0  # Team 0 gains mark
            made_w_majority = float(made_arr.float().mean().item()) > 0.5

            # Early terminal: did any world reach terminal before trick 7?
            # We use the tricks_saved proxy: if remaining_total at this decision < 42
            # (some points already captured), we're not at trick 0
            remaining_total = 42 - pre_t0 - pre_t1
            early_terminal = remaining_total < 30  # at least 1-2 tricks played early

            # Detector tags from decision_actions
            dist_tags = str(da_row0.get("distribution_shape_tags", ""))
            strategy_tags = str(da_row0.get("strategy_context_tags", ""))
            position_tags = str(da_row0.get("matched_position_detectors", ""))

            # Book detector tags from decision_actions.csv (the joined table uses a different corpus)
            # Use all three detector tag columns from the atlas CSV
            book_detector_labels = "|".join(filter(None, [
                str(da_row0.get("matched_position_detectors", "")),
                str(da_row0.get("strategy_context_tags", "")),
                str(da_row0.get("distribution_shape_tags", "")),
            ]))

            # Detector families active
            fams = detector_family(book_detector_labels or position_tags)

            # Ch10 mechanism tags
            ch10_tags = ch10_flip_correlations(
                flip, bid, made_w_majority, early_terminal, not made_w_majority
            )

            # Build action rows (one per legal action)
            for a in range(7):
                if not legal[a].item():
                    continue
                a_rows = da_rows[da_rows["candidate_slot"] == a]
                candidate_domino = a_rows["candidate_domino"].iloc[0] if len(a_rows) > 0 else ""
                is_actual = bool(a_rows["is_actual_action"].iloc[0]) if len(a_rows) > 0 else False

                action_rows.append({
                    "game_idx": gi,
                    "seed": seed,
                    "decl_id": decl_id,
                    "decl_name": decl_name,
                    "bid": bid,
                    "decision_idx": di,
                    "player": player,
                    "seat_role": seat_role,
                    "team": team,
                    "offense_score_before": off_score,
                    "defense_score_before": def_score,
                    "score_bucket": sc_bucket,
                    "action_slot": a,
                    "candidate_domino": candidate_domino,
                    "is_actual_action": is_actual,
                    "point_ev": round(float(e_q_point[a].item()), 4),
                    "mark_ev": round(float(e_q_mark[a].item()), 4),
                    "point_rank": int((e_q_point_masked >= e_q_point[a]).sum().item()),
                    "mark_rank": int((e_q_mark_masked >= e_q_mark[a]).sum().item()),
                    "is_top1_point": a == top1_point,
                    "is_top1_mark": a == top1_mark,
                    "top1_flips": flip,
                    "ev_cost_if_flip": round(ev_cost, 4),
                    "mark_gain_if_flip": round(mark_gain, 4),
                    "book_detector_labels": book_detector_labels,
                    "detector_families": "|".join(fams),
                    **{f"ch10_{k}": v for k, v in ch10_tags.items()},
                })

            # Collect decision-level summary
            flip_decisions.append({
                "game_idx": gi,
                "seed": seed,
                "decl_id": decl_id,
                "decl_name": decl_name,
                "bid": bid,
                "decision_idx": di,
                "player": player,
                "seat_role": seat_role,
                "team": team,
                "offense_score_before": off_score,
                "defense_score_before": def_score,
                "score_bucket": sc_bucket,
                "n_legal": int(n_legal),
                "top1_point_action": top1_point,
                "top1_mark_action": top1_mark,
                "top1_point_ev": round(float(e_q_point[top1_point].item()), 4),
                "top1_mark_ev": round(float(e_q_mark[top1_mark].item()), 4),
                "flip": flip,
                "ev_cost": round(ev_cost, 4),
                "mark_gain": round(mark_gain, 4),
                "remaining_count_total": remaining_total,
                "book_detector_labels": book_detector_labels,
                "detector_families": "|".join(fams),
                **{f"ch10_{k}": v for k, v in ch10_tags.items()},
            })

    print(f"[mark_utility_transform] Processed {total_decisions} decisions, {total_flips} flips")
    flip_rate = total_flips / total_decisions if total_decisions > 0 else 0.0
    print(f"[mark_utility_transform] Flip rate: {flip_rate:.4f} ({total_flips}/{total_decisions})")

    flip_df = pd.DataFrame(flip_decisions)
    action_df = pd.DataFrame(action_rows)

    # ── Slice analysis ────────────────────────────────────────────────────────

    def flip_rate_slice(df: pd.DataFrame, col: str) -> pd.DataFrame:
        grp = df.groupby(col).agg(
            decisions=("flip", "count"),
            flips=("flip", "sum"),
            flip_rate=("flip", "mean"),
            mean_ev_cost=("ev_cost", "mean"),
            mean_mark_gain=("mark_gain", "mean"),
        ).reset_index()
        return grp

    # Slice by declaration
    by_decl = flip_rate_slice(flip_df, "decl_name")
    # Slice by seat/role
    by_role = flip_rate_slice(flip_df, "seat_role")
    # Slice by bid (only one bid=30 here, but include for schema completeness)
    by_bid = flip_rate_slice(flip_df, "bid")
    # Slice by score bucket
    by_score = flip_rate_slice(flip_df, "score_bucket")

    # Slice by detector family - explode the pipe-separated families
    family_rows = []
    for _, row in flip_df.iterrows():
        fams = str(row["detector_families"]).split("|")
        for fam in fams:
            fam = fam.strip()
            if fam:
                family_rows.append({"detector_family": fam, "flip": row["flip"],
                                     "ev_cost": row["ev_cost"], "mark_gain": row["mark_gain"]})
    if family_rows:
        fam_df = pd.DataFrame(family_rows)
        by_family = flip_rate_slice(fam_df, "detector_family")
    else:
        by_family = pd.DataFrame()

    # Combine slice results
    slice_rows: list[dict[str, Any]] = []
    for _, r in by_decl.iterrows():
        slice_rows.append({"slice_type": "declaration", "slice_value": r["decl_name"],
                           "decisions": int(r["decisions"]), "flips": int(r["flips"]),
                           "flip_rate": round(float(r["flip_rate"]), 4),
                           "mean_ev_cost": round(float(r["mean_ev_cost"]), 4),
                           "mean_mark_gain": round(float(r["mean_mark_gain"]), 4)})
    for _, r in by_role.iterrows():
        slice_rows.append({"slice_type": "seat_role", "slice_value": r["seat_role"],
                           "decisions": int(r["decisions"]), "flips": int(r["flips"]),
                           "flip_rate": round(float(r["flip_rate"]), 4),
                           "mean_ev_cost": round(float(r["mean_ev_cost"]), 4),
                           "mean_mark_gain": round(float(r["mean_mark_gain"]), 4)})
    for _, r in by_bid.iterrows():
        slice_rows.append({"slice_type": "bid", "slice_value": str(r["bid"]),
                           "decisions": int(r["decisions"]), "flips": int(r["flips"]),
                           "flip_rate": round(float(r["flip_rate"]), 4),
                           "mean_ev_cost": round(float(r["mean_ev_cost"]), 4),
                           "mean_mark_gain": round(float(r["mean_mark_gain"]), 4)})
    for _, r in by_score.iterrows():
        slice_rows.append({"slice_type": "score_bucket", "slice_value": r["score_bucket"],
                           "decisions": int(r["decisions"]), "flips": int(r["flips"]),
                           "flip_rate": round(float(r["flip_rate"]), 4),
                           "mean_ev_cost": round(float(r["mean_ev_cost"]), 4),
                           "mean_mark_gain": round(float(r["mean_mark_gain"]), 4)})
    if not by_family.empty:
        for _, r in by_family.iterrows():
            slice_rows.append({"slice_type": "detector_family", "slice_value": r["detector_family"],
                               "decisions": int(r["decisions"]), "flips": int(r["flips"]),
                               "flip_rate": round(float(r["flip_rate"]), 4),
                               "mean_ev_cost": round(float(r["mean_ev_cost"]), 4),
                               "mean_mark_gain": round(float(r["mean_mark_gain"]), 4)})

    # ── Top 100 detector-endorsed flips ──────────────────────────────────────
    flipped = flip_df[flip_df["flip"]].copy()
    flipped_sorted = flipped.sort_values("ev_cost", ascending=False)
    # "Detector endorsed" = flip has at least one named book-position detector
    # (from matched_position_detectors column, not just shape tags)
    BOOK_POSITION_DETECTORS = {
        "bidder_first_lead_plan", "first_setter_pounce_window", "first_setter_response",
        "partner_third_seat_safe_donation", "partner_third_seat_support",
        "last_to_act_closure_policy", "setter_count_before_certainty",
        "late_trick_threshold_closure", "no_trump_lead_control_and_support",
        "doubles_regime_plan", "defender_damage_lead_class",
        "partner_count_donation", "partner_forcedness_and_safety",
        "setter_lead_pressure", "setter_count_pressure",
        "called_suit_pressure",
    }

    def has_book_detector(labels_str: str) -> bool:
        s = str(labels_str).strip()
        if not s or s in ("nan", "none", "None"):
            return False
        tokens = {t.strip() for t in s.split("|") if t.strip()}
        return bool(tokens & BOOK_POSITION_DETECTORS)

    endorsed = flipped_sorted[flipped_sorted["book_detector_labels"].apply(has_book_detector)]
    # If fewer than 100 endorsed, also include any flips with position detector tags
    top100 = endorsed.head(100) if len(endorsed) >= 5 else flipped_sorted.head(100)
    top100_rows = top100.to_dict("records")

    # ── Per-Ch10-claim correlation ────────────────────────────────────────────
    ch10_corr_rows: list[dict[str, Any]] = []
    for claim_id in CH10_CLAIM_IDS:
        col = f"ch10_{claim_id}"
        if col not in flip_df.columns:
            continue
        sub = flip_df[flip_df[col] == True]
        all_sub = len(sub)
        flips_sub = int(sub["flip"].sum())
        flip_rate_sub = flips_sub / all_sub if all_sub > 0 else 0.0
        overall_flip_rate = flip_rate
        enrichment = (flip_rate_sub / overall_flip_rate) if overall_flip_rate > 0 else float("nan")
        ch10_corr_rows.append({
            "claim_id": claim_id,
            "mechanism_active_decisions": all_sub,
            "flips_when_active": flips_sub,
            "flip_rate_when_active": round(flip_rate_sub, 4),
            "overall_flip_rate": round(overall_flip_rate, 4),
            "enrichment_vs_baseline": round(enrichment, 4) if not np.isnan(enrichment) else "nan",
            "interpretation": (
                "enriched" if enrichment > 1.1
                else "depleted" if enrichment < 0.9
                else "neutral"
            ) if not np.isnan(enrichment) else "insufficient_data",
        })

    # ── Action-level mark EV scalars ─────────────────────────────────────────
    # action_mark_ev_scalars.csv: one row per legal action
    action_ev_rows: list[dict[str, Any]] = []
    for row in action_rows:
        action_ev_rows.append({
            "game_idx": row["game_idx"],
            "seed": row["seed"],
            "decl_name": row["decl_name"],
            "bid": row["bid"],
            "decision_idx": row["decision_idx"],
            "seat_role": row["seat_role"],
            "action_slot": row["action_slot"],
            "candidate_domino": row["candidate_domino"],
            "point_ev": row["point_ev"],
            "mark_ev": row["mark_ev"],
            "point_rank": row["point_rank"],
            "mark_rank": row["mark_rank"],
            "is_top1_point": row["is_top1_point"],
            "is_top1_mark": row["is_top1_mark"],
            "top1_flips": row["top1_flips"],
            "is_actual_action": row["is_actual_action"],
            "ev_cost_if_flip": row["ev_cost_if_flip"],
            "mark_gain_if_flip": row["mark_gain_if_flip"],
            "score_bucket": row["score_bucket"],
            "book_detector_labels": row["book_detector_labels"],
            "detector_families": row["detector_families"],
        })

    # ── Write outputs ─────────────────────────────────────────────────────────
    write_csv(OUT_DIR / "action_mark_ev_scalars.csv", action_ev_rows)
    write_csv(OUT_DIR / "flip_rate_by_slice.csv", slice_rows)
    write_csv(OUT_DIR / "detector_endorsed_flips.csv", top100_rows)
    write_csv(OUT_DIR / "per_ch10_claim_correlation.csv", ch10_corr_rows)

    # ── Summary ───────────────────────────────────────────────────────────────
    ev_cost_mean = float(flipped["ev_cost"].mean()) if len(flipped) > 0 else 0.0
    mark_gain_mean = float(flipped["mark_gain"].mean()) if len(flipped) > 0 else 0.0
    top_flip_by_ev = flipped_sorted.head(5).to_dict("records") if len(flipped_sorted) > 0 else []

    summary = {
        "schema": "w42.bookval.wave1.mark_utility_transform.v1",
        "bead": BEAD_ID,
        "wave": "wave1",
        "repo_commit": git_sha(),
        "inputs": {
            "pt_file": str(PT_PATH.relative_to(ROOT)),
            "pt_sha256": sha256_file(PT_PATH),
            "transform_source": str(PHASE4_TRANSFORM_FILE.relative_to(ROOT)),
            "transform_sha256": sha256_file(PHASE4_TRANSFORM_FILE),
            "bid_values": [int(b) for b in data.get("bid_values", [30] * 10)],
            "n_games": len(data["results"]),
            "n_worlds_per_decision": int(data["results"][0].decisions[0].q_per_world.shape[0]),
        },
        "coverage": {
            "total_decisions": total_decisions,
            "total_legal_actions": len(action_rows),
            "flips": total_flips,
            "endorsed_flips": len(endorsed),
        },
        "headline": {
            "top1_flip_rate": round(flip_rate, 4),
            "mean_ev_cost_on_flips": round(ev_cost_mean, 4),
            "mean_mark_gain_on_flips": round(mark_gain_mean, 4),
        },
        "top5_flips_by_ev_cost": [
            {
                "game_idx": int(r["game_idx"]),
                "decision_idx": int(r["decision_idx"]),
                "decl_name": r["decl_name"],
                "seat_role": r["seat_role"],
                "score_bucket": r["score_bucket"],
                "ev_cost": round(float(r["ev_cost"]), 4),
                "mark_gain": round(float(r["mark_gain"]), 4),
            }
            for r in top_flip_by_ev
        ],
        "caveats": [
            "Dataset is a single seed (9430), 10 declarations, bid=30 only.",
            "Score state (offense/defense before) used to reconstruct final team points per world.",
            "Q semantics: remaining (team0-team1) point differential, not raw capture totals.",
            "Mark utility computed per-world from q_per_world — the per-world hook is maintained.",
            "Bidder team assumed = team 0 (offense) per branch_atlas_scaled_v0 convention.",
            "Book detector endorsement uses matched_position_detectors from decision_actions.csv; "
            "joined_claim_action_rows.csv is a different corpus (seeds 0-9 from corpus_v2_train) "
            "and does not overlap with branch_atlas_scaled_v0 (seed 9430).",
            "bid=30 only; mark multiplier=1; early-terminal and special-bid transforms "
            "have limited activation at bid=30.",
        ],
    }
    write_json(OUT_DIR / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
