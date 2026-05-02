"""Scoring-objective drift validation for bead t42-csw6.23.

This is a deterministic report script, not a policy trainer. It validates the
terminal scoring algebra and simulates how point, mark, and tournament-style
objectives change the reward surface before any forge/Gus/Burl rollout exists.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scratch.w42.wandb_utils import add_wandb_args, init_wandb

OUT_DIR = Path("scratch/w42/scoring_objective_drift_claim_validation")
ORDINARY_BIDS = list(range(30, 42))
HIGH_BIDS = [42, 84, 126, 168]
REPORT_BIDS = [30, 35, 41, 42, 84, 126, 168]


def git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def mark_multiplier(bid: int) -> int:
    if bid >= 84:
        return bid // 42
    return 1


def point_scores(bid: int, bidder_capture_points: int, *, made: bool | None = None) -> tuple[int, int]:
    """Return bidder/opponent points under the Winning 42 chapter scoring model.

    For ordinary bids, captured points determine both make and point score. For
    42/84/higher contracts, the terminal contract outcome determines whether the
    bid amount is awarded. The `bidder_capture_points` field is still carried in
    tables to expose how marks erase partial point information.
    """

    opponent_capture_points = 42 - bidder_capture_points
    if bid < 42:
        made = bidder_capture_points >= bid
        if made:
            return bidder_capture_points, opponent_capture_points
        return 0, opponent_capture_points + bid

    if made is None:
        made = bidder_capture_points >= 42
    if made:
        return bid, 0
    return 0, bid


def mark_scores(bid: int, made: bool) -> tuple[int, int]:
    multiplier = mark_multiplier(bid)
    if made:
        return multiplier, 0
    return 0, multiplier


def tournament_points(mark_bidder: int, mark_opponent: int, total_marks_tiebreaker: bool) -> tuple[int, int]:
    if total_marks_tiebreaker:
        return mark_bidder, mark_opponent
    return (1, 0) if mark_bidder > mark_opponent else (0, 1)


def iter_terminal_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for bid in ORDINARY_BIDS:
        for bidder_capture in range(43):
            made = bidder_capture >= bid
            point_bidder, point_opponent = point_scores(bid, bidder_capture)
            mark_bidder, mark_opponent = mark_scores(bid, made)
            rows.append(
                {
                    "bid": bid,
                    "bid_class": "ordinary",
                    "bidder_capture_points": bidder_capture,
                    "opponent_capture_points": 42 - bidder_capture,
                    "made": made,
                    "point_bidder_score": point_bidder,
                    "point_opponent_score": point_opponent,
                    "point_net": point_bidder - point_opponent,
                    "mark_multiplier": mark_multiplier(bid),
                    "mark_bidder_score": mark_bidder,
                    "mark_opponent_score": mark_opponent,
                    "mark_net": mark_bidder - mark_opponent,
                    "partial_points_erased": made and (42 - bidder_capture) > 0,
                    "set_severity_points": 0 if made else bid + (42 - bidder_capture),
                }
            )

    for bid in HIGH_BIDS:
        for made in [False, True]:
            for bidder_capture in range(43):
                point_bidder, point_opponent = point_scores(bid, bidder_capture, made=made)
                mark_bidder, mark_opponent = mark_scores(bid, made)
                rows.append(
                    {
                        "bid": bid,
                        "bid_class": "high" if bid == 42 else "84_ladder",
                        "bidder_capture_points": bidder_capture,
                        "opponent_capture_points": 42 - bidder_capture,
                        "made": made,
                        "point_bidder_score": point_bidder,
                        "point_opponent_score": point_opponent,
                        "point_net": point_bidder - point_opponent,
                        "mark_multiplier": mark_multiplier(bid),
                        "mark_bidder_score": mark_bidder,
                        "mark_opponent_score": mark_opponent,
                        "mark_net": mark_bidder - mark_opponent,
                        "partial_points_erased": made and (42 - bidder_capture) > 0,
                        "set_severity_points": 0 if made else bid,
                    }
                )
    return rows


def early_terminal_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for bid in [30, 35, 41]:
        for trick_index in range(1, 8):
            for bidder_so_far in range(43):
                for remaining_count in range(43 - bidder_so_far):
                    made_now = bidder_so_far >= bid
                    set_now = bidder_so_far + remaining_count < bid
                    if made_now or set_now:
                        rows.append(
                            {
                                "bid": bid,
                                "trick_index": trick_index,
                                "bidder_points_so_far": bidder_so_far,
                                "remaining_count_points": remaining_count,
                                "mark_terminal_status": "made" if made_now else "set",
                                "point_objective_still_has_value": trick_index < 7,
                                "tricks_potentially_saved": 7 - trick_index,
                            }
                        )
    for bid in [84, 126, 168]:
        for trick_index in range(1, 8):
            rows.append(
                {
                    "bid": bid,
                    "trick_index": trick_index,
                    "bidder_points_so_far": "not applicable",
                    "remaining_count_points": "not applicable",
                    "mark_terminal_status": "set if opponents have won any trick",
                    "point_objective_still_has_value": False,
                    "tricks_potentially_saved": 7 - trick_index,
                }
            )
    return rows


def objective_threshold_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for bid in REPORT_BIDS:
        multiplier = mark_multiplier(bid)
        point_make, point_set = point_scores(bid, min(42, max(bid, 30)), made=True)
        point_fail, point_fail_opp = point_scores(bid, 0, made=False)
        point_reward = point_make - point_set
        point_loss = point_fail_opp - point_fail
        point_threshold = point_loss / (point_reward + point_loss)
        mark_threshold = multiplier / (multiplier + multiplier)
        rows.append(
            {
                "bid": bid,
                "mark_multiplier": multiplier,
                "point_success_net": point_reward,
                "point_failure_loss": point_loss,
                "point_break_even_make_probability": round(point_threshold, 6),
                "mark_break_even_make_probability": round(mark_threshold, 6),
                "threshold_delta_mark_minus_point": round(mark_threshold - point_threshold, 6),
            }
        )
    return rows


def scoreboard_rows() -> list[dict[str, Any]]:
    scenarios = [
        ("three_barely_made_30s", [(30, 30, True), (30, 30, True), (30, 30, True)]),
        ("one_clean_84", [(84, 42, True)]),
        ("one_set_84", [(84, 0, False)]),
        ("three_big_sets_ordinary", [(30, 0, False), (30, 0, False), (30, 0, False)]),
        ("mixed_point_close_mark_lopsided", [(30, 30, True), (30, 30, True), (41, 41, True)]),
    ]
    rows: list[dict[str, Any]] = []
    for name, hands in scenarios:
        point_a = point_b = mark_a = mark_b = tournament_a = tournament_b = 0
        for bid, bidder_capture, made in hands:
            pb, po = point_scores(bid, bidder_capture, made=made)
            mb, mo = mark_scores(bid, made)
            tb, to = tournament_points(mb, mo, total_marks_tiebreaker=False)
            point_a += pb
            point_b += po
            mark_a += mb
            mark_b += mo
            tournament_a += tb
            tournament_b += to
        rows.append(
            {
                "scenario": name,
                "hands": len(hands),
                "point_score_bidder_team": point_a,
                "point_score_opponent_team": point_b,
                "mark_score_bidder_team": mark_a,
                "mark_score_opponent_team": mark_b,
                "tournament_hand_wins_bidder_team": tournament_a,
                "tournament_hand_wins_opponent_team": tournament_b,
                "mark_lead_minus_point_lead_scaled": round((mark_a - mark_b) - ((point_a - point_b) / 42), 6),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, Any]], thresholds: list[dict[str, Any]], early_rows: list[dict[str, Any]]) -> dict[str, Any]:
    ordinary_rows = [row for row in rows if row["bid_class"] == "ordinary"]
    high_rows = [row for row in rows if row["bid_class"] != "ordinary"]
    made_ordinary = [row for row in ordinary_rows if row["made"]]
    set_ordinary = [row for row in ordinary_rows if not row["made"]]
    return {
        "schema": "w42.scoring_objective_drift.v1",
        "repo_commit": git_sha(),
        "terminal_rows": len(rows),
        "ordinary_terminal_rows": len(ordinary_rows),
        "high_terminal_rows": len(high_rows),
        "partial_point_erasure_rows": sum(1 for row in rows if row["partial_points_erased"]),
        "ordinary_made_rows": len(made_ordinary),
        "ordinary_set_rows": len(set_ordinary),
        "ordinary_set_severity_min": min(row["set_severity_points"] for row in set_ordinary),
        "ordinary_set_severity_max": max(row["set_severity_points"] for row in set_ordinary),
        "ordinary_set_severity_compressed_to_one_mark": len({row["set_severity_points"] for row in set_ordinary}),
        "mark_multiplier_by_bid": {str(bid): mark_multiplier(bid) for bid in REPORT_BIDS},
        "early_terminal_rows": len(early_rows),
        "max_tricks_saved_by_mark_terminal": max(row["tricks_potentially_saved"] for row in early_rows if isinstance(row["tricks_potentially_saved"], int)),
        "thresholds": thresholds,
        "claim_statuses": {
            "ch10-score-mode-objective": "supported",
            "ch10-early-terminal-under-marks": "supported",
            "ch10-nonbidder-partial-points-erased": "supported",
            "ch10-set-severity-compression": "supported",
            "ch10-special-bid-mark-multiplier": "supported",
            "ch10-low-bid-score-distortion": "supported",
            "ch10-point-system-skill-signal": "underpowered",
            "ch10-tournament-speed-tradeoff": "context-limited",
            "ch10-timed-marks-advancement-objective": "not-yet-tested",
        },
        "caveats": [
            "Deterministic terminal scoring and synthetic transforms only; no forge E[Q] or model policy rollout.",
            "Tournament objective is represented as marks/hand wins and tiebreak hooks, not a full bracket simulator.",
            "84/126/168 rows validate multipliers and objective scale, not make probability or optimal bidding.",
        ],
    }


def claim_ledger_entries(summary: dict[str, Any], command: str, wandb_status: Any) -> dict[str, Any]:
    provenance = {
        "commands": [command],
        "configs": "not applicable",
        "data_inputs": "wiki/experiments/winning42-ch10-tournament-scoring.md; wiki/experiments/winning42-ch07-taking-every-trick-84.md; wiki/experiments/winning42-ch08-setting-84.md; wiki/experiments/winning42-ch13-optional-variations.md; wiki/experiments/winning42-ch14-history-tournaments.md",
        "commit_sha": summary["repo_commit"],
        "random_seeds": "not applicable; exhaustive deterministic transform",
        "wandb_links": json.dumps(wandb_status) if isinstance(wandb_status, dict) else str(wandb_status),
        "hf_links": "not applicable",
    }
    claims = [
        (
            "ch10-score-mode-objective",
            "Marks change the scoring objective from captured hand points to hand-level make/set marks.",
            "score_mode_objective",
            "Terminal utility transform and point-vs-mark break-even comparison.",
            "supported",
            "Deterministic scoring algebra supports objective drift; policy optimality still needs oracle rollouts.",
        ),
        (
            "ch10-early-terminal-under-marks",
            "Marks create early terminal states once an ordinary contract is already made or impossible.",
            "mark_terminal_state",
            "Exhaustive arithmetic over captured points and remaining count.",
            "supported",
            "Terminal detector ignores legal trick paths; it only proves the arithmetic condition.",
        ),
        (
            "ch10-nonbidder-partial-points-erased",
            "Marks erase defender partial-point rewards when the bidder makes an ordinary contract.",
            "partial_points_erased",
            "Terminal utility rows where defenders capture points but receive zero marks.",
            "supported",
            "Supports terminal accounting, not whether defenders should chase points in a live state.",
        ),
        (
            "ch10-set-severity-compression",
            "Ordinary mark scoring compresses many point-set severities into the same one-mark result.",
            "set_severity_compressed",
            "Distinct ordinary point penalties collapsed to one mark on failed ordinary bids.",
            "supported",
            "High-bid multipliers remain distinct; live-state pursuit of extra count is untested.",
        ),
        (
            "ch10-special-bid-mark-multiplier",
            "84/126/168 scale to two/three/four marks while ordinary bids collapse to one mark.",
            "special_bid_mark_multiplier",
            "Multiplier table over report bids.",
            "supported",
            "Validates objective scale only, not bid selection or make probability.",
        ),
        (
            "ch10-low-bid-score-distortion",
            "Low ordinary bids can create mark-score leads that overstate point-score separation.",
            "scoreboard_distortion",
            "Synthetic scoreboard replay including three barely made 30 bids.",
            "supported",
            "Synthetic examples validate the transform; frequency in generated or human games is unmeasured.",
        ),
        (
            "ch10-point-system-skill-signal",
            "Point scoring is a richer skill signal than marks.",
            "point_system_skill_signal",
            "Requires policy population separation under both objectives.",
            "underpowered",
            "No policy population, oracle arena, or human/tournament data was run.",
        ),
        (
            "ch10-tournament-speed-tradeoff",
            "Marks speed tournaments through early termination and shorter matches.",
            "tournament_speed_tradeoff",
            "Early-terminal arithmetic plus saved-trick upper-bound table.",
            "context-limited",
            "Arithmetic supports possible saved tricks; real speed requires play-time or tournament simulation.",
        ),
        (
            "ch10-timed-marks-advancement-objective",
            "Timed competitions can make advancement probability the real objective.",
            "timed_marks_advancement_objective",
            "Tournament bracket or timed-round simulation.",
            "not-yet-tested",
            "This report did not simulate brackets, clocks, or tiebreak populations.",
        ),
    ]
    entries = []
    for claim_id, claim, detector, metric_test, status, caveats in claims:
        entries.append(
            {
                "claim_id": claim_id,
                "chapter_source": {
                    "wiki_page": "wiki/experiments/winning42-ch10-tournament-scoring.md",
                    "source_slice": "scratch/winning42/winning42.with_figures.md lines 4243-4331",
                    "chapter": 10,
                },
                "claim": claim,
                "detector": detector,
                "metric_test": metric_test,
                "data_source": "Deterministic scoring-objective transform tables generated by t42-csw6.23.",
                "readiness": ["enumeration", "ruleset", "report-only"],
                "evidence_artifact": "scratch/w42/scoring_objective_drift_claim_validation/summary.json",
                "status": status,
                "caveats": caveats,
                "provenance": provenance,
            }
        )
    return {
        "schema_version": "1.0.0",
        "ledger_impact": "9 Chapter 10 scoring-objective claims updated or preserved by deterministic transform evidence in t42-csw6.23.",
        "entries": entries,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--bead-id", default="t42-csw6.23")
    parser.add_argument("--seed", type=int, default=0)
    add_wandb_args(
        parser,
        default_group="w42-csw6-scoring-objective-drift",
        default_enabled=True,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    command = "python " + " ".join(__import__("sys").argv)
    config = {
        "bead_id": args.bead_id,
        "git_sha": git_sha(),
        "data_manifest": "not applicable",
        "source_corpus": "not applicable",
        "dataset_name": "w42-scoring-objective-drift-transform",
        "dataset_version": "v1",
        "ruleset": "straight42-plus-mark-scoring-transform",
        "label_source": "deterministic scoring algebra",
        "decision_slice": "terminal scoring and synthetic scoreboard transforms",
        "feature_set": "score_mode_objective",
        "concept_buckets": ["scoring", "84", "tournament"],
        "model_family": "not applicable",
        "model_params": "not applicable",
        "random_seed": "not applicable",
        "train_seed": "not applicable",
        "split_seed": "not applicable",
        "eval_seed": "not applicable",
        "baseline_policy": "not applicable",
        "metrics": ["terminal_rows", "partial_point_erasure_rows", "mark_multiplier_by_bid"],
        "claim_ledger_status_before": "chapter-harvest statuses",
        "local_artifact_path": str(args.output_dir),
        "hf_repo_id": "not applicable",
        "wandb_group": args.wandb_group,
        "seed": args.seed,
    }
    wandb_run = init_wandb(
        args,
        config=config,
        output_dir=args.output_dir,
        tags=["w42", args.bead_id, "strategy-validation", "scoring", "eighty-four", "tournament", "scratch"],
    )

    terminal_rows = iter_terminal_rows()
    early_rows = early_terminal_rows()
    threshold_rows = objective_threshold_rows()
    score_rows = scoreboard_rows()
    summary = summarize(terminal_rows, threshold_rows, early_rows)

    write_csv(args.output_dir / "terminal_objective_transform.csv", terminal_rows)
    write_csv(args.output_dir / "early_mark_terminal_states.csv", early_rows)
    write_csv(args.output_dir / "objective_thresholds.csv", threshold_rows)
    write_csv(args.output_dir / "scoreboard_distortion_examples.csv", score_rows)

    wandb_status = wandb_run.status()
    summary["wandb"] = wandb_status
    summary["hf_links"] = "not applicable"
    ledger = claim_ledger_entries(summary, command, wandb_status)

    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output_dir / "claim_ledger_delta.json").write_text(json.dumps(ledger, indent=2) + "\n")

    wandb_run.log(
        {
            "summary/terminal_rows": summary["terminal_rows"],
            "summary/partial_point_erasure_rows": summary["partial_point_erasure_rows"],
            "summary/ordinary_set_severity_compressed_to_one_mark": summary[
                "ordinary_set_severity_compressed_to_one_mark"
            ],
            "summary/early_terminal_rows": summary["early_terminal_rows"],
        }
    )
    wandb_run.update_summary(
        {
            "status": "completed",
            "terminal_rows": summary["terminal_rows"],
            "claim_ledger_delta": str(args.output_dir / "claim_ledger_delta.json"),
            "hf_links": "not applicable",
        }
    )
    wandb_run.log_artifact_files(
        name=f"w42-scoring-objective-drift-{git_sha()[:8]}",
        artifact_type="w42-report-artifacts",
        paths=[
            args.output_dir / "summary.json",
            args.output_dir / "claim_ledger_delta.json",
            args.output_dir / "terminal_objective_transform.csv",
            args.output_dir / "early_mark_terminal_states.csv",
            args.output_dir / "objective_thresholds.csv",
            args.output_dir / "scoreboard_distortion_examples.csv",
        ],
    )
    wandb_run.finish()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
