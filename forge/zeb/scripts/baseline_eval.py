#!/usr/bin/env python3
"""Initial Zeb evaluation: random games baseline.

Validates:
1. Random vs random games work (~50% win rate expected)
2. Trajectory collection works correctly
3. Forward pass with small model
4. Rule-based player baseline

Run: python -m forge.zeb.scripts.baseline_eval
"""
import time
import torch
from statistics import mean, stdev

from forge.zeb.game import new_game, apply_action, is_terminal, legal_actions, current_player
from forge.zeb.evaluate import RandomPlayer, RuleBasedPlayer, play_match
from forge.zeb.self_play import play_games_batched, trajectories_to_batch
from forge.zeb.model import ZebModel, get_model_config
from forge.zeb.observation import observe, get_legal_mask


def test_random_vs_random(n_games: int = 100) -> dict:
    """Play random vs random games and verify ~50% win rate."""
    print(f"\n{'='*60}")
    print("1. RANDOM vs RANDOM BASELINE")
    print(f"{'='*60}")

    random_player = RandomPlayer()
    players = (random_player, random_player, random_player, random_player)

    start = time.time()
    results = play_match(players, n_games=n_games, base_seed=42)
    elapsed = time.time() - start

    win_rate = results['team0_win_rate']
    margin = results['avg_margin']

    print(f"Games played: {n_games}")
    print(f"Time: {elapsed:.2f}s ({n_games/elapsed:.1f} games/sec)")
    print(f"Team 0 wins: {results['team0_wins']}")
    print(f"Team 1 wins: {results['team1_wins']}")
    print(f"Team 0 win rate: {win_rate:.1%}")
    print(f"Avg margin: {margin:+.1f} points")

    # Sanity check: should be close to 50%
    expected = 0.50
    tolerance = 0.15  # Allow 35-65%
    ok = abs(win_rate - expected) < tolerance
    status = "✓ PASS" if ok else "✗ FAIL"
    print(f"\nSanity check (win rate ~50%): {status}")

    return {'win_rate': win_rate, 'margin': margin, 'games_per_sec': n_games/elapsed, 'ok': ok}


def test_trajectory_collection(n_games: int = 10) -> dict:
    """Verify trajectory collection works correctly."""
    print(f"\n{'='*60}")
    print("2. TRAJECTORY COLLECTION")
    print(f"{'='*60}")

    # Create small model for testing
    config = get_model_config('small')
    model = ZebModel(**config)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Device: {device}")
    print(f"Model config: {config}")

    start = time.time()
    trajectories = play_games_batched(
        model,
        n_games=n_games,
        temperature=1.0,
        device=device,
        base_seed=0,
    )
    elapsed = time.time() - start

    print(f"\nTrajectories collected: {len(trajectories)}")
    print(f"Time: {elapsed:.2f}s ({n_games/elapsed:.1f} games/sec)")

    # Validate trajectory structure
    total_steps = sum(len(t.steps) for t in trajectories)
    steps_per_game = [len(t.steps) for t in trajectories]

    print(f"Total steps: {total_steps}")
    print(f"Steps per game: min={min(steps_per_game)}, max={max(steps_per_game)}, avg={mean(steps_per_game):.1f}")

    # Each game has 28 plays (7 tricks x 4 players)
    expected_steps = 28
    ok = all(s == expected_steps for s in steps_per_game)
    print(f"\nExpected {expected_steps} steps per game: {'✓ PASS' if ok else '✗ FAIL'}")

    # Check trajectory data integrity
    sample = trajectories[0]
    step = sample.steps[0]

    print(f"\nSample trajectory:")
    print(f"  Final scores: {sample.final_scores}")
    print(f"  Winner: Team {sample.winner}")
    print(f"  Step 0 tokens shape: {step.tokens.shape}")
    print(f"  Step 0 mask shape: {step.mask.shape}")
    print(f"  Step 0 action: {step.action}")
    print(f"  Step 0 outcome: {step.outcome:.3f}")

    # Convert to batch and verify shapes
    batch = trajectories_to_batch(trajectories)
    tokens, masks, hand_indices, hand_masks, actions, outcomes = batch

    print(f"\nBatch tensors:")
    print(f"  tokens: {tokens.shape}")
    print(f"  masks: {masks.shape}")
    print(f"  hand_indices: {hand_indices.shape}")
    print(f"  hand_masks: {hand_masks.shape}")
    print(f"  actions: {actions.shape}")
    print(f"  outcomes: {outcomes.shape}")

    return {'n_trajectories': len(trajectories), 'total_steps': total_steps, 'ok': ok}


def test_forward_pass() -> dict:
    """Test forward pass with small model."""
    print(f"\n{'='*60}")
    print("3. FORWARD PASS TEST")
    print(f"{'='*60}")

    config = get_model_config('small')
    model = ZebModel(**config)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.eval()

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model size: {n_params:,} parameters ({n_params/1e6:.2f}M)")

    # Create a game and get observation
    state = new_game(seed=42, dealer=0, skip_bidding=True)
    player = current_player(state)

    tokens, mask, hand_indices = observe(state, player)
    legal = get_legal_mask(state, player)

    # Add batch dimension
    tokens = tokens.unsqueeze(0).to(device)
    mask = mask.unsqueeze(0).to(device)
    hand_indices = hand_indices.unsqueeze(0).to(device)
    legal = legal.unsqueeze(0).to(device)

    print(f"\nInput shapes:")
    print(f"  tokens: {tokens.shape}")
    print(f"  mask: {mask.shape}")
    print(f"  hand_indices: {hand_indices.shape}")
    print(f"  legal: {legal.shape}")

    with torch.no_grad():
        policy, value, _belief = model(tokens, mask, hand_indices, legal)
        action, log_prob, value2 = model.get_action(tokens, mask, hand_indices, legal)

    print(f"\nOutputs:")
    print(f"  policy: {policy.shape}, values: {policy[0].tolist()}")
    print(f"  value: {value.item():.3f}")
    print(f"  sampled action: {action.item()}")
    print(f"  log_prob: {log_prob.item():.3f}")

    # Verify action is legal
    legal_slots = [i for i in range(7) if legal[0, i].item()]
    ok = action.item() in legal_slots
    print(f"\nAction legality check: {'✓ PASS' if ok else '✗ FAIL'}")

    return {'n_params': n_params, 'value': value.item(), 'ok': ok}


def test_heuristic_baseline(n_games: int = 100) -> dict:
    """Test rule-based player baseline."""
    print(f"\n{'='*60}")
    print("4. HEURISTIC vs RANDOM BASELINE")
    print(f"{'='*60}")

    heuristic = RuleBasedPlayer()
    random_player = RandomPlayer()

    # Heuristic on team 0, random on team 1
    players = (heuristic, random_player, heuristic, random_player)

    start = time.time()
    results = play_match(players, n_games=n_games, base_seed=42)
    elapsed = time.time() - start

    win_rate = results['team0_win_rate']
    margin = results['avg_margin']

    print(f"Games played: {n_games}")
    print(f"Time: {elapsed:.2f}s")
    print(f"Heuristic win rate: {win_rate:.1%}")
    print(f"Avg margin: {margin:+.1f} points")

    # Heuristic should beat random
    ok = win_rate > 0.50
    print(f"\nHeuristic > random check: {'✓ PASS' if ok else '✗ FAIL'}")

    return {'win_rate': win_rate, 'margin': margin, 'ok': ok}


def test_neural_vs_random(n_games: int = 50) -> dict:
    """Test untrained neural network vs random."""
    print(f"\n{'='*60}")
    print("5. UNTRAINED NEURAL vs RANDOM")
    print(f"{'='*60}")

    from forge.zeb.evaluate import NeuralPlayer, evaluate_vs_random

    config = get_model_config('small')
    model = ZebModel(**config)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    start = time.time()
    results = evaluate_vs_random(model, n_games=n_games, device=device)
    elapsed = time.time() - start

    win_rate = results['team0_win_rate']
    margin = results['avg_margin']

    print(f"Games played: {n_games}")
    print(f"Time: {elapsed:.2f}s")
    print(f"Untrained neural win rate: {win_rate:.1%}")
    print(f"Avg margin: {margin:+.1f} points")

    # Untrained should be close to random (~50%)
    expected = 0.50
    tolerance = 0.20  # Allow 30-70%
    ok = abs(win_rate - expected) < tolerance
    print(f"\nUntrained ~random check: {'✓ PASS' if ok else '⚠ WARN (unusual but possible)'}")

    return {'win_rate': win_rate, 'margin': margin, 'ok': ok}


def main():
    print("="*60)
    print("ZEB BASELINE EVALUATION")
    print("="*60)

    results = {}

    # Run all tests
    results['random_vs_random'] = test_random_vs_random(n_games=200)
    results['trajectory'] = test_trajectory_collection(n_games=20)
    results['forward_pass'] = test_forward_pass()
    results['heuristic'] = test_heuristic_baseline(n_games=200)
    results['neural_vs_random'] = test_neural_vs_random(n_games=100)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    all_ok = all(r.get('ok', True) for r in results.values())

    print(f"\n1. Random vs Random: {results['random_vs_random']['win_rate']:.1%} win rate")
    print(f"2. Trajectory collection: {results['trajectory']['total_steps']} steps from {results['trajectory']['n_trajectories']} games")
    print(f"3. Forward pass: {results['forward_pass']['n_params']:,} params, value={results['forward_pass']['value']:.3f}")
    print(f"4. Heuristic vs Random: {results['heuristic']['win_rate']:.1%} win rate")
    print(f"5. Untrained Neural vs Random: {results['neural_vs_random']['win_rate']:.1%} win rate")

    print(f"\nAll checks passed: {'✓ YES' if all_ok else '✗ NO'}")

    # Key baselines for future comparison
    print(f"\n{'='*60}")
    print("BASELINE METRICS (for training comparison)")
    print(f"{'='*60}")
    print(f"Random win rate: ~50% (sanity check)")
    print(f"Heuristic vs Random: {results['heuristic']['win_rate']:.1%} (target to beat)")
    print(f"Untrained neural: {results['neural_vs_random']['win_rate']:.1%} (starting point)")
    print(f"Training target: >55% vs random, >50% vs heuristic")

    return all_ok


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
