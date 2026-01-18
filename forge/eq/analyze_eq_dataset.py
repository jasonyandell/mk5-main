#!/usr/bin/env python3
"""Analyze E[Q] predictions from eq_473.pt dataset.

Validates E[Q] predictions by replaying games and comparing predictions
to actual outcomes (points-from-here-to-end).

Metrics computed:
- Correlation between E[Q] and actual outcome
- MAE (mean absolute error)
- Calibration (is mean error near zero?)
- Trends by decision_idx (does accuracy improve with game progress?)
- Uncertainty calibration (does actual fall within ±1σ ~68% of the time?)

Usage:
    python -m forge.eq.analyze_eq_dataset forge/data/eq_473.pt
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy import stats

from forge.eq.game import GameState
from forge.oracle.tables import resolve_trick, DOMINO_COUNT_POINTS


def random_deal(seed: int) -> list[list[int]]:
    """Generate a random deal with given seed (matches generate_local.py)."""
    rng = np.random.default_rng(seed)
    all_dominoes = list(range(28))
    rng.shuffle(all_dominoes)
    return [sorted(all_dominoes[i * 7 : (i + 1) * 7]) for i in range(4)]


@dataclass
class DecisionOutcome:
    """Outcome data for a single decision point."""

    game_idx: int
    decision_idx: int  # 0-27 within game
    player: int        # Who made this decision
    e_q_mean: float    # Predicted E[Q] for action taken
    e_q_var: float     # Variance of E[Q] prediction
    actual: float      # Actual outcome (points from here to end, absolute)
    actual_margin: float  # Actual outcome (my points - opp points from here)
    error: float       # e_q_mean - actual (absolute interpretation)
    error_margin: float  # e_q_mean - actual_margin (margin interpretation)
    hand_size: int     # Cards remaining when decision made (7=first, 1=last)


def compute_trick_points(trick_dominoes: tuple[int, ...]) -> int:
    """Compute points for a completed trick."""
    return 1 + sum(DOMINO_COUNT_POINTS[d] for d in trick_dominoes)


def replay_game_for_outcomes(
    hands: list[list[int]],
    decl_id: int,
    actions: list[int],  # action_taken indices per decision
    e_q_means: list[np.ndarray],  # E[Q] predictions per decision
    e_q_vars: list[np.ndarray],   # Variance per decision
    game_idx: int,
) -> list[DecisionOutcome]:
    """Replay a game and compute actual outcomes for each decision.

    Args:
        hands: Initial hands [hand0, hand1, hand2, hand3]
        decl_id: Declaration ID
        actions: List of action_taken indices (0-6 in remaining hand order)
        e_q_means: E[Q] predictions for each decision
        e_q_vars: Variance predictions for each decision
        game_idx: Game index for tracking

    Returns:
        List of DecisionOutcome for each decision point
    """
    game = GameState.from_hands(hands, decl_id, leader=0)
    outcomes = []

    # Track team scores as we replay
    team_scores = [0, 0]  # team 0 (players 0,2) and team 1 (players 1,3)

    # First pass: replay game, record plays and trick results
    plays = []  # (player, domino_id, team_score_at_this_point)
    trick_results = []  # (winner, points) for each trick

    current_trick_dominoes = []
    for decision_idx, action_idx in enumerate(actions):
        player = game.current_player()
        hand = list(game.hands[player])
        domino = hand[action_idx]  # action_idx is index into remaining hand

        # Record score before this play
        team = player % 2
        plays.append({
            'player': player,
            'domino': domino,
            'team': team,
            'team_scores_before': list(team_scores),
            'hand_size': len(hand),
        })

        # Track trick
        current_trick_dominoes.append(domino)

        # Apply action
        game = game.apply_action(domino)

        # Check if trick completed
        if len(current_trick_dominoes) == 4:
            # Find lead domino (first in this trick)
            lead_domino = current_trick_dominoes[0]
            outcome = resolve_trick(
                lead_domino,
                tuple(current_trick_dominoes),
                decl_id
            )

            # Compute winner player (not offset)
            # The trick started with leader = (plays[-4]['player'])
            trick_start_player = plays[-4]['player']
            winner = (trick_start_player + outcome.winner_offset) % 4
            winner_team = winner % 2

            # Update scores
            team_scores[winner_team] += outcome.points
            trick_results.append((winner_team, outcome.points))

            current_trick_dominoes = []

    # Final scores
    final_scores = list(team_scores)

    # Second pass: compute actual outcome for each decision
    # We compute TWO interpretations:
    # 1. actual_points = my team's points from here to end (absolute)
    # 2. actual_margin = (my team - opponent team) from here to end (relative)
    for decision_idx, play_info in enumerate(plays):
        team = play_info['team']
        opp_team = 1 - team
        my_score_before = play_info['team_scores_before'][team]
        opp_score_before = play_info['team_scores_before'][opp_team]

        # Absolute: my team's points from here to end
        actual_points = final_scores[team] - my_score_before

        # Margin: (my points - opponent points) from here to end
        actual_margin = (final_scores[team] - my_score_before) - (final_scores[opp_team] - opp_score_before)

        # Get E[Q] prediction for the action that was taken
        action_idx = actions[decision_idx]
        e_q = e_q_means[decision_idx][action_idx]
        e_q_v = e_q_vars[decision_idx][action_idx]

        outcomes.append(DecisionOutcome(
            game_idx=game_idx,
            decision_idx=decision_idx,
            player=play_info['player'],
            e_q_mean=float(e_q),
            e_q_var=float(e_q_v),
            actual=float(actual_points),  # Use points by default
            actual_margin=float(actual_margin),  # Store margin too
            error=float(e_q - actual_points),
            error_margin=float(e_q - actual_margin),
            hand_size=play_info['hand_size'],
        ))

    return outcomes


def analyze_dataset(dataset_path: Path) -> dict:
    """Load dataset and analyze E[Q] predictions.

    Returns dict with analysis results.
    """
    print(f"Loading dataset: {dataset_path}")
    data = torch.load(dataset_path, weights_only=False)

    n_games = data['n_games']
    n_decisions_total = data['n_decisions']

    print(f"  {n_games} games, {n_decisions_total} decisions")

    # Extract metadata
    metadata = data['metadata']
    base_seed = metadata['base_seed']
    game_metadata = metadata['game_metadata']

    # Tensors
    e_q_mean = data['e_q_mean'].numpy()  # (N, 7)
    e_q_var = data['e_q_var'].numpy()    # (N, 7)
    action_taken = data['action_taken'].numpy()  # (N,)
    game_idx_arr = data['game_idx'].numpy()  # (N,)

    print(f"\nReplaying {n_games} games...")
    all_outcomes: list[DecisionOutcome] = []

    for gm in game_metadata:
        game_idx = gm['game_idx']
        seed = gm['seed']
        decl_id = gm['decl_id']

        # Reconstruct hands using same RNG as generate_local.py
        hands = random_deal(seed)

        # Get decisions for this game
        mask = game_idx_arr == game_idx
        game_decisions = np.where(mask)[0]

        if len(game_decisions) == 0:
            continue

        # Extract per-decision data for this game
        actions = action_taken[game_decisions].tolist()
        eq_means = [e_q_mean[i] for i in game_decisions]
        eq_vars = [e_q_var[i] for i in game_decisions]

        # Replay and compute outcomes
        outcomes = replay_game_for_outcomes(
            hands=hands,
            decl_id=decl_id,
            actions=actions,
            e_q_means=eq_means,
            e_q_vars=eq_vars,
            game_idx=game_idx,
        )
        all_outcomes.extend(outcomes)

    print(f"  Analyzed {len(all_outcomes)} decision outcomes")

    # Compute metrics for BOTH interpretations
    errors_abs = np.array([o.error for o in all_outcomes])  # vs absolute points
    errors_margin = np.array([o.error_margin for o in all_outcomes])  # vs margin
    e_q_preds = np.array([o.e_q_mean for o in all_outcomes])
    actuals_abs = np.array([o.actual for o in all_outcomes])
    actuals_margin = np.array([o.actual_margin for o in all_outcomes])
    variances = np.array([o.e_q_var for o in all_outcomes])
    hand_sizes = np.array([o.hand_size for o in all_outcomes])
    decision_idxs = np.array([o.decision_idx for o in all_outcomes])

    stds = np.sqrt(variances)

    # Metrics for absolute interpretation
    mae_abs = np.abs(errors_abs).mean()
    rmse_abs = np.sqrt((errors_abs ** 2).mean())
    mean_error_abs = errors_abs.mean()
    corr_abs, pval_abs = stats.pearsonr(e_q_preds, actuals_abs)
    coverage_1sigma_abs = (np.abs(errors_abs) <= stds).mean()
    coverage_2sigma_abs = (np.abs(errors_abs) <= 2 * stds).mean()

    # Metrics for margin interpretation
    mae_margin = np.abs(errors_margin).mean()
    rmse_margin = np.sqrt((errors_margin ** 2).mean())
    mean_error_margin = errors_margin.mean()
    corr_margin, pval_margin = stats.pearsonr(e_q_preds, actuals_margin)
    coverage_1sigma_margin = (np.abs(errors_margin) <= stds).mean()
    coverage_2sigma_margin = (np.abs(errors_margin) <= 2 * stds).mean()

    results = {
        'n_games': n_games,
        'n_decisions': len(all_outcomes),
        # Absolute interpretation
        'mae_abs': mae_abs,
        'rmse_abs': rmse_abs,
        'mean_error_abs': mean_error_abs,
        'correlation_abs': corr_abs,
        'coverage_1sigma_abs': coverage_1sigma_abs,
        'coverage_2sigma_abs': coverage_2sigma_abs,
        # Margin interpretation
        'mae_margin': mae_margin,
        'rmse_margin': rmse_margin,
        'mean_error_margin': mean_error_margin,
        'correlation_margin': corr_margin,
        'coverage_1sigma_margin': coverage_1sigma_margin,
        'coverage_2sigma_margin': coverage_2sigma_margin,
        # Common
        'mean_uncertainty': stds.mean(),
        'outcomes': all_outcomes,
        'hand_sizes': hand_sizes,
        'decision_idxs': decision_idxs,
        'errors_abs': errors_abs,
        'errors_margin': errors_margin,
        'actuals_abs': actuals_abs,
        'actuals_margin': actuals_margin,
        'predictions': e_q_preds,
    }

    return results


def print_results(results: dict) -> None:
    """Print analysis results in a readable format."""
    print("\n" + "=" * 60)
    print("E[Q] PREDICTION ANALYSIS")
    print("=" * 60)

    print(f"\nDataset: {results['n_games']} games, {results['n_decisions']} decisions")

    # Compare two interpretations
    print("\n" + "=" * 60)
    print("COMPARING TARGET INTERPRETATIONS")
    print("=" * 60)
    print("\nThe oracle may predict either:")
    print("  1. ABSOLUTE: My team's points from here to end")
    print("  2. MARGIN:   (My team - Opponent) points from here to end")

    print("\n--- Interpretation 1: Absolute Points ---")
    print(f"  MAE:              {results['mae_abs']:.2f} points")
    print(f"  Mean Error:       {results['mean_error_abs']:+.2f} points (calibration)")
    print(f"  Correlation:      {results['correlation_abs']:.3f}")
    print(f"  Coverage ±1σ:     {results['coverage_1sigma_abs']:.1%}")
    print(f"  Mean pred:        {results['predictions'].mean():.1f}")
    print(f"  Mean actual:      {results['actuals_abs'].mean():.1f}")

    print("\n--- Interpretation 2: Margin (My - Opponent) ---")
    print(f"  MAE:              {results['mae_margin']:.2f} points")
    print(f"  Mean Error:       {results['mean_error_margin']:+.2f} points (calibration)")
    print(f"  Correlation:      {results['correlation_margin']:.3f}")
    print(f"  Coverage ±1σ:     {results['coverage_1sigma_margin']:.1%}")
    print(f"  Mean pred:        {results['predictions'].mean():.1f}")
    print(f"  Mean actual:      {results['actuals_margin'].mean():.1f}")

    # Determine which interpretation fits better
    better = "MARGIN" if results['mae_margin'] < results['mae_abs'] else "ABSOLUTE"
    print(f"\n  >>> Better fit: {better} (lower MAE)")

    # Use the better interpretation for detailed analysis
    if better == "MARGIN":
        errors = results['errors_margin']
        actuals = results['actuals_margin']
        mae = results['mae_margin']
        mean_error = results['mean_error_margin']
        corr = results['correlation_margin']
        cov1 = results['coverage_1sigma_margin']
        cov2 = results['coverage_2sigma_margin']
    else:
        errors = results['errors_abs']
        actuals = results['actuals_abs']
        mae = results['mae_abs']
        mean_error = results['mean_error_abs']
        corr = results['correlation_abs']
        cov1 = results['coverage_1sigma_abs']
        cov2 = results['coverage_2sigma_abs']

    print("\n" + "=" * 60)
    print(f"DETAILED ANALYSIS (using {better} interpretation)")
    print("=" * 60)

    hand_sizes = results['hand_sizes']
    predictions = results['predictions']

    print("\n--- Accuracy by Hand Size (Game Phase) ---")
    for hs in sorted(set(hand_sizes), reverse=True):
        mask = hand_sizes == hs
        hs_errors = errors[mask]
        hs_preds = predictions[mask]
        hs_actuals = actuals[mask]
        hs_mae = np.abs(hs_errors).mean()
        hs_mean_err = hs_errors.mean()
        n = mask.sum()
        phase = "early" if hs >= 5 else ("mid" if hs >= 3 else "late")
        print(f"  Hand size {hs} ({phase:5s}): MAE={hs_mae:5.2f}, bias={hs_mean_err:+6.2f}, "
              f"pred={hs_preds.mean():5.1f}, actual={hs_actuals.mean():5.1f} (n={n})")

    # Metrics by player position
    print("\n--- Accuracy by Player ---")
    outcomes = results['outcomes']
    for player in range(4):
        player_outcomes = [o for o in outcomes if o.player == player]
        if player_outcomes:
            if better == "MARGIN":
                player_errors = np.array([o.error_margin for o in player_outcomes])
            else:
                player_errors = np.array([o.error for o in player_outcomes])
            print(f"  Player {player}: MAE={np.abs(player_errors).mean():.2f}, "
                  f"mean_err={player_errors.mean():+.2f} (n={len(player_outcomes)})")

    print("\n--- Uncertainty Calibration ---")
    print(f"  Mean σ:           {results['mean_uncertainty']:.2f} points")
    print(f"  Coverage ±1σ:     {cov1:.1%} (expect ~68%)")
    print(f"  Coverage ±2σ:     {cov2:.1%} (expect ~95%)")

    # Bias diagnosis
    print("\n--- Bias Diagnosis ---")
    print(f"  E[Q] systematically {'under' if mean_error < 0 else 'over'}estimates by {abs(mean_error):.1f} points")

    # Check if bias is additive or proportional
    pred_quartiles = np.percentile(predictions, [25, 50, 75])
    print(f"\n  Bias by prediction quartile:")
    ranges = [
        ("Low (< Q1)", predictions < pred_quartiles[0]),
        ("Med-Low (Q1-Q2)", (predictions >= pred_quartiles[0]) & (predictions < pred_quartiles[1])),
        ("Med-High (Q2-Q3)", (predictions >= pred_quartiles[1]) & (predictions < pred_quartiles[2])),
        ("High (> Q3)", predictions >= pred_quartiles[2]),
    ]
    for name, mask in ranges:
        if mask.sum() > 0:
            print(f"    {name:18s}: bias={errors[mask].mean():+.1f}, n={mask.sum()}")


def main():
    parser = argparse.ArgumentParser(description="Analyze E[Q] dataset predictions")
    parser.add_argument("dataset", type=str, help="Path to dataset .pt file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show per-game details")

    args = parser.parse_args()

    results = analyze_dataset(Path(args.dataset))
    print_results(results)

    if args.verbose:
        print("\n--- Sample Decisions ---")
        for o in results['outcomes'][:10]:
            print(f"  Game {o.game_idx}, dec {o.decision_idx}: "
                  f"E[Q]={o.e_q_mean:.1f}, actual={o.actual:.1f}, "
                  f"error={o.error:+.1f}")


if __name__ == "__main__":
    main()
