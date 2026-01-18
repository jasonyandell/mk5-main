#!/usr/bin/env python3
"""Analyze E[Q] predictions vs actual game outcomes.

Compares predicted E[Q] values at each decision to actual game outcomes
to assess prediction quality and calibration.

Usage:
    python scripts/analyze_eq_predictions.py [dataset_path]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

# Allow running as a script without needing `PYTHONPATH=.`.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from forge.eq.game import GameState
from forge.oracle.rng import deal_from_seed
from forge.oracle.tables import DOMINO_COUNT_POINTS, resolve_trick


def domino_name(d_id: int) -> str:
    high = 0
    while (high + 1) * (high + 2) // 2 <= d_id:
        high += 1
    low = d_id - high * (high + 1) // 2
    return f"[{high}:{low}]"


def compute_game_score(play_history, decl_id):
    """Compute final score from play history. Returns [team0, team1]."""
    scores = [0, 0]
    tricks = []
    current_trick = []

    for player, domino_id, lead_domino_id in play_history:
        current_trick.append((player, domino_id, lead_domino_id))
        if len(current_trick) == 4:
            tricks.append(current_trick)
            current_trick = []

    for trick_idx, trick in enumerate(tricks):
        lead_domino = trick[0][2]
        domino_ids = tuple(t[1] for t in trick)
        players = tuple(t[0] for t in trick)

        result = resolve_trick(lead_domino, domino_ids, decl_id)
        winner_player = players[result.winner_offset]
        winner_team = winner_player % 2

        trick_points = sum(DOMINO_COUNT_POINTS[d] for d in domino_ids)
        if trick_idx == 6:  # Last trick bonus
            trick_points += 1

        scores[winner_team] += trick_points

    return scores


def analyze_dataset(dataset_path: str):
    dataset = torch.load(dataset_path, map_location="cpu", weights_only=False)
    meta = dataset["metadata"]
    seed = meta["seed"]
    n_games = meta["n_games"]

    print(f"Dataset: {dataset_path}")
    print(f"Games: {n_games}, Seed: {seed}")

    # Check schema version and warn about semantics
    schema = meta.get("schema", {})
    if schema:
        print(f"Schema: version={meta.get('version', '?')}, units={schema.get('q_units', '?')}")
        if schema.get("warning"):
            print(f"  NOTE: {schema['warning']}")
    print()

    # Reconstruct game seeds (must match generator RNG consumption).
    # Generator shuffles a train/val assignment list, which advances RNG state.
    # Infer n_val_games from train_mask (robust to non-default val_fraction).
    train_mask = dataset.get("train_mask")
    n_val_games = 0
    if train_mask is not None and "game_idx" in dataset:
        for game_idx in range(n_games):
            game_mask = dataset["game_idx"] == game_idx
            if not game_mask.any():
                continue
            # All decisions in a game share the same train_mask.
            if not bool(train_mask[game_mask][0].item()):
                n_val_games += 1
    else:
        # Fallback: historical default.
        n_val_games = int(n_games * 0.1)

    rng = np.random.default_rng(seed)
    game_is_val = [False] * (n_games - n_val_games) + [True] * n_val_games
    rng.shuffle(game_is_val)  # Important even if we don't use the result.

    game_data = []
    for _ in range(n_games):
        game_data.append({
            "seed": int(rng.integers(0, 2**31)),
            "decl": int(rng.integers(0, 10)),
        })

    # Load E[Q] fields (t42-d6y1: Q-values in points, NOT logits)
    e_q_mean_all = dataset["e_q_mean"]
    e_q_var_all = dataset.get("e_q_var")
    if e_q_var_all is None:
        e_q_var_all = torch.zeros_like(e_q_mean_all)

    print("=" * 70)
    print("GAME-BY-GAME: E[Q] Predictions vs Actual Outcomes")
    print("=" * 70)

    all_results = []
    all_decision_preds = []  # (decision_idx, signed_eq_team0, outcome_team0)
    opening_sigma_covered_1 = 0
    opening_sigma_covered_2 = 0
    opening_sigma_total = 0

    for game_idx in range(n_games):
        gd = game_data[game_idx]
        hands = deal_from_seed(gd["seed"])
        decl_id = gd["decl"]

        game_mask = dataset["game_idx"] == game_idx
        game_indices = game_mask.nonzero().squeeze(-1).tolist()
        if isinstance(game_indices, int):
            game_indices = [game_indices]

        if not game_indices:
            continue

        e_q_mean = e_q_mean_all[game_mask]
        e_q_var = e_q_var_all[game_mask]
        actions = dataset["action_taken"][game_mask].tolist()
        legal_masks = dataset["legal_mask"][game_mask]

        game = GameState.from_hands(hands, decl_id, leader=0)
        decision_records = []

        for dec_idx, action_idx in enumerate(actions):
            player = game.current_player()
            remaining = sorted(game.hands[player])

            if action_idx >= len(remaining):
                break

            action_domino = remaining[action_idx]
            e_q = e_q_mean[dec_idx][action_idx].item()
            e_std = np.sqrt(max(0, e_q_var[dec_idx][action_idx].item()))

            masked = e_q_mean[dec_idx].clone()
            masked[~legal_masks[dec_idx]] = float("-inf")
            best_idx = masked.argmax().item()

            decision_records.append({
                "player": player,
                "action": domino_name(action_domino),
                "e_q": e_q,
                "e_std": e_std,
                "was_greedy": action_idx == best_idx,
            })

            try:
                game = game.apply_action(action_domino)
            except Exception as e:
                print(f"Game {game_idx} dec {dec_idx}: {e}")
                break

        if game.is_complete():
            scores = compute_game_score(game.play_history, decl_id)
            outcome = scores[0] - scores[1]
        else:
            outcome = 0
            scores = [0, 0]

        if decision_records:
            opening = decision_records[0]
            print(f"\nGame {game_idx}: Decl={decl_id}, Score={scores[0]}-{scores[1]} (Δ={outcome:+d})")
            print(f"  Opening: {opening['action']} E[Q]={opening['e_q']:+.1f}±{opening['e_std']:.1f}")

            # Opening uncertainty coverage (team0 perspective; opening is always P0/team0).
            opening_sigma_total += 1
            if abs(outcome - opening["e_q"]) <= opening["e_std"]:
                opening_sigma_covered_1 += 1
            if abs(outcome - opening["e_q"]) <= 2.0 * opening["e_std"]:
                opening_sigma_covered_2 += 1

            team0_dec = [d for d in decision_records if d["player"] in [0, 2]]
            avg_eq = np.mean([d["e_q"] for d in team0_dec]) if team0_dec else 0
            greedy_pct = np.mean([d["was_greedy"] for d in team0_dec]) if team0_dec else 0
            print(f"  Team0 avg E[Q]={avg_eq:+.1f}, greedy={greedy_pct:.0%}")

            # Decision-level tracking (convert per-player E[Q] to team0 perspective).
            for dec_idx, drec in enumerate(decision_records):
                signed_eq = drec["e_q"] if (drec["player"] % 2 == 0) else -drec["e_q"]
                all_decision_preds.append((dec_idx, signed_eq, outcome))

            all_results.append({
                "game": game_idx,
                "outcome": outcome,
                "opening_eq": opening["e_q"],
                "opening_std": opening["e_std"],
                "avg_eq": avg_eq,
            })

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if all_results:
        outcomes = [r["outcome"] for r in all_results]
        opening_eqs = [r["opening_eq"] for r in all_results]
        avg_eqs = [r["avg_eq"] for r in all_results]

        print(f"\nOutcomes: mean={np.mean(outcomes):+.1f}, std={np.std(outcomes):.1f}")
        print(f"Opening E[Q]: mean={np.mean(opening_eqs):+.1f}, std={np.std(opening_eqs):.1f}")
        print(f"Avg E[Q]: mean={np.mean(avg_eqs):+.1f}, std={np.std(avg_eqs):.1f}")

        if len(outcomes) > 2:
            corr_open = np.corrcoef(opening_eqs, outcomes)[0, 1]
            corr_avg = np.corrcoef(avg_eqs, outcomes)[0, 1]
            print(f"\nCorrelation with outcome:")
            print(f"  Opening E[Q]: r={corr_open:+.3f}")
            print(f"  Avg E[Q]:     r={corr_avg:+.3f} {'✓' if corr_avg > 0.5 else ''}")

        errors = [r["outcome"] - r["opening_eq"] for r in all_results]
        print(f"\nCalibration:")
        print(f"  Mean error: {np.mean(errors):+.1f}")
        print(f"  RMSE: {np.sqrt(np.mean([e**2 for e in errors])):.1f}")

        if opening_sigma_total:
            print("\nOpening uncertainty coverage:")
            print(
                f"  |Δ - E[Q]| ≤ 1σ: {opening_sigma_covered_1}/{opening_sigma_total} "
                f"({opening_sigma_covered_1 / opening_sigma_total:.0%})"
            )
            print(
                f"  |Δ - E[Q]| ≤ 2σ: {opening_sigma_covered_2}/{opening_sigma_total} "
                f"({opening_sigma_covered_2 / opening_sigma_total:.0%})"
            )

    if all_decision_preds:
        dec_idxs = np.array([d[0] for d in all_decision_preds], dtype=int)
        preds = np.array([d[1] for d in all_decision_preds], dtype=float)
        outcomes = np.array([d[2] for d in all_decision_preds], dtype=float)
        err = outcomes - preds

        print("\nDecision-level (all players, team0 perspective):")
        if len(preds) > 2:
            corr = np.corrcoef(preds, outcomes)[0, 1]
            print(f"  Corr(pred, outcome): {corr:+.3f}")
        print(f"  Mean error: {err.mean():+.1f} pts")
        print(f"  MAE: {np.mean(np.abs(err)):.1f} pts")
        print(f"  RMSE: {np.sqrt(np.mean(err**2)):.1f} pts")

        # Per-decision index summary (useful sanity check; small-N per bucket for tiny datasets).
        if len(np.unique(dec_idxs)) > 1:
            print("\nBy decision index:")
            for i in range(int(dec_idxs.max()) + 1):
                mask = dec_idxs == i
                if mask.sum() < 3:
                    continue
                sub_pred = preds[mask]
                sub_out = outcomes[mask]
                sub_err = sub_out - sub_pred
                sub_corr = np.corrcoef(sub_pred, sub_out)[0, 1] if mask.sum() > 2 else float("nan")
                print(f"  d{i:02d}: r={sub_corr:+.3f} MAE={np.mean(np.abs(sub_err)):.1f}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
    else:
        data_dir = Path("forge/data")
        datasets = sorted(data_dir.glob("eq_v2_*.pt"))
        if not datasets:
            print("No eq_v2 datasets found")
            sys.exit(1)
        dataset_path = str(datasets[-1])

    analyze_dataset(dataset_path)
