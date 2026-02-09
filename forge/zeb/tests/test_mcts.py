"""Quick test of MCTS implementation."""
import time

from forge.eq.game import GameState
from forge.oracle.rng import deal_from_seed

from forge.zeb.mcts import MCTS, select_action_mcts
from forge.zeb.mcts_self_play import play_game_with_mcts, play_games_with_mcts


def test_basic_mcts():
    """Test basic MCTS on a single state."""
    print("=== Test Basic MCTS ===")

    # Create a game state
    hands = deal_from_seed(42)
    state = GameState.from_hands(hands, decl_id=0, leader=0)

    print(f"Initial state: player {state.current_player()} to move")
    print(f"Legal actions: {state.legal_actions()}")

    # Run MCTS
    mcts = MCTS(n_simulations=100)
    t0 = time.time()
    visits = mcts.search(state, player=0)
    elapsed = time.time() - t0

    print(f"\nMCTS results ({elapsed:.2f}s for 100 sims):")
    for action, count in sorted(visits.items(), key=lambda x: -x[1]):
        print(f"  Domino {action}: {count} visits")

    # Best action
    best = max(visits.keys(), key=lambda a: visits[a])
    print(f"\nBest action: domino {best}")


def test_mcts_game():
    """Test playing a full game with MCTS."""
    print("\n=== Test MCTS Full Game ===")

    t0 = time.time()
    game = play_game_with_mcts(seed=42, n_simulations=50, temperature=1.0)
    elapsed = time.time() - t0

    print(f"Game completed in {elapsed:.2f}s")
    print(f"Final points: Team 0 = {game.final_points[0]}, Team 1 = {game.final_points[1]}")
    print(f"Winner: Team {game.winner}")
    print(f"Training examples collected: {len(game.examples)}")

    # Show first example
    ex = game.examples[0]
    print(f"\nFirst example:")
    print(f"  Player: {ex.player}")
    print(f"  Action probs: {dict(sorted(ex.action_probs.items(), key=lambda x: -x[1])[:3])}...")
    print(f"  Outcome: {ex.outcome}")


def test_mcts_speed():
    """Benchmark MCTS speed."""
    print("\n=== MCTS Speed Benchmark ===")

    hands = deal_from_seed(42)
    state = GameState.from_hands(hands, decl_id=0, leader=0)

    for n_sims in [10, 50, 100, 200]:
        mcts = MCTS(n_simulations=n_sims)

        # Warm up
        mcts.search(state, player=0)

        # Time it
        t0 = time.time()
        n_trials = 10
        for i in range(n_trials):
            mcts.search(state, player=0)
        elapsed = time.time() - t0

        print(f"  {n_sims} sims: {elapsed/n_trials*1000:.1f}ms per search")


def test_multiple_games():
    """Test generating multiple games."""
    print("\n=== Multiple Games ===")

    t0 = time.time()
    games = play_games_with_mcts(
        n_games=10,
        n_simulations=50,
        temperature=1.0,
        base_seed=0,
    )
    elapsed = time.time() - t0

    print(f"Generated {len(games)} games in {elapsed:.1f}s")
    print(f"Total examples: {sum(len(g.examples) for g in games)}")

    # Win rate
    team0_wins = sum(1 for g in games if g.winner == 0)
    print(f"Team 0 wins: {team0_wins}/{len(games)}")


def test_oracle_value_fn():
    """Test oracle-based value function for MCTS."""
    print("\n=== Oracle Value Function Test ===")

    import torch
    if not torch.cuda.is_available():
        print("CUDA not available, skipping oracle test")
        return

    from forge.zeb.oracle_value import create_oracle_value_fn

    # Create oracle value function
    print("Loading oracle...")
    t0 = time.time()
    value_fn = create_oracle_value_fn(device="cuda", compile=True)
    print(f"Oracle loaded in {time.time() - t0:.1f}s")

    # Create a game state
    hands = deal_from_seed(42)
    state = GameState.from_hands(hands, decl_id=0, leader=0)

    print(f"\nState: player {state.current_player()} to move")
    print(f"Legal actions: {state.legal_actions()}")

    # Evaluate with oracle
    t0 = time.time()
    value = value_fn(state, player=0)
    elapsed = time.time() - t0

    print(f"\nOracle value (player 0 perspective): {value:.3f}")
    print(f"Query time: {elapsed*1000:.1f}ms")

    # Evaluate from player 1's perspective
    value_p1 = value_fn(state, player=1)
    print(f"Oracle value (player 1 perspective): {value_p1:.3f}")
    print(f"(Should be opposite sign: {-value:.3f})")

    # Test with MCTS
    print("\n--- MCTS with Oracle ---")
    mcts = MCTS(n_simulations=50, value_fn=value_fn)

    t0 = time.time()
    visits = mcts.search(state, player=0)
    elapsed = time.time() - t0

    print(f"MCTS with oracle ({elapsed:.2f}s for 50 sims):")
    for action, count in sorted(visits.items(), key=lambda x: -x[1]):
        print(f"  Domino {action}: {count} visits")

    print(f"\nTotal oracle queries: {value_fn.query_count}")


def test_mcts_oracle_vs_random():
    """Compare MCTS with oracle vs random rollout."""
    print("\n=== Oracle vs Random Rollout ===")

    import torch
    if not torch.cuda.is_available():
        print("CUDA not available, skipping")
        return

    from forge.zeb.oracle_value import create_oracle_value_fn

    # Load oracle
    value_fn = create_oracle_value_fn(device="cuda", compile=True)

    hands = deal_from_seed(42)
    state = GameState.from_hands(hands, decl_id=0, leader=0)

    # Random rollout MCTS
    mcts_random = MCTS(n_simulations=100, value_fn=None)
    t0 = time.time()
    visits_random = mcts_random.search(state, player=0)
    time_random = time.time() - t0

    # Oracle MCTS
    mcts_oracle = MCTS(n_simulations=100, value_fn=value_fn)
    t0 = time.time()
    visits_oracle = mcts_oracle.search(state, player=0)
    time_oracle = time.time() - t0

    print(f"\nRandom rollout ({time_random*1000:.0f}ms):")
    for action, count in sorted(visits_random.items(), key=lambda x: -x[1])[:3]:
        print(f"  Domino {action}: {count} visits")

    print(f"\nOracle ({time_oracle*1000:.0f}ms, {value_fn.query_count} queries):")
    for action, count in sorted(visits_oracle.items(), key=lambda x: -x[1])[:3]:
        print(f"  Domino {action}: {count} visits")

    # Compare distributions
    print("\nVisit distribution comparison:")
    all_actions = sorted(set(visits_random.keys()) | set(visits_oracle.keys()))
    for a in all_actions:
        r = visits_random.get(a, 0)
        o = visits_oracle.get(a, 0)
        diff = "***" if abs(r - o) > 20 else ""
        print(f"  Domino {a}: random={r:3d}, oracle={o:3d} {diff}")


def test_batched_mcts():
    """Test batched MCTS with oracle value function."""
    print("\n=== Batched MCTS Test ===")

    import torch
    if not torch.cuda.is_available():
        print("CUDA not available, skipping")
        return

    from forge.zeb.oracle_value import create_oracle_value_fn

    # Load oracle
    print("Loading oracle...")
    value_fn = create_oracle_value_fn(device="cuda", compile=True)

    hands = deal_from_seed(42)
    state = GameState.from_hands(hands, decl_id=0, leader=0)

    # Test batched search (default when value_fn has batch_evaluate)
    mcts = MCTS(n_simulations=100, value_fn=value_fn, wave_size=32)
    t0 = time.time()
    visits = mcts.search(state, player=0)
    elapsed = time.time() - t0

    print(f"Batched MCTS ({elapsed*1000:.0f}ms, {value_fn.query_count} queries):")
    for action, count in sorted(visits.items(), key=lambda x: -x[1])[:3]:
        print(f"  Domino {action}: {count} visits")

    # Verify reasonable distribution
    total_visits = sum(visits.values())
    assert total_visits == 100, f"Expected 100 visits, got {total_visits}"
    print(f"\nTotal visits: {total_visits} (correct)")


def test_batched_vs_sequential_benchmark():
    """Benchmark batched vs sequential oracle MCTS."""
    print("\n=== Batched vs Sequential Benchmark ===")

    import torch
    if not torch.cuda.is_available():
        print("CUDA not available, skipping")
        return

    from forge.zeb.oracle_value import create_oracle_value_fn

    # Load oracle
    value_fn_batched = create_oracle_value_fn(device="cuda", compile=True)

    # For sequential, we need a value function without batch_evaluate
    # We'll just wrap the single-call version
    class SequentialValueFn:
        def __init__(self, oracle_fn):
            self._fn = oracle_fn
            self.query_count = 0

        def __call__(self, state, player):
            self.query_count += 1
            return self._fn(state, player)

    value_fn_sequential = SequentialValueFn(value_fn_batched)

    hands = deal_from_seed(42)
    state = GameState.from_hands(hands, decl_id=0, leader=0)

    n_sims = 100
    n_trials = 3

    # Warm up both
    MCTS(n_simulations=10, value_fn=value_fn_batched).search(state, player=0)
    MCTS(n_simulations=10, value_fn=value_fn_sequential).search(state, player=0)

    # Benchmark sequential
    value_fn_sequential.query_count = 0
    t0 = time.time()
    for _ in range(n_trials):
        MCTS(n_simulations=n_sims, value_fn=value_fn_sequential).search(state, player=0)
    time_sequential = (time.time() - t0) / n_trials
    queries_sequential = value_fn_sequential.query_count // n_trials

    # Benchmark batched
    value_fn_batched.query_count = 0
    t0 = time.time()
    for _ in range(n_trials):
        MCTS(n_simulations=n_sims, value_fn=value_fn_batched, wave_size=32).search(state, player=0)
    time_batched = (time.time() - t0) / n_trials
    queries_batched = value_fn_batched.query_count // n_trials

    print(f"\nSequential: {time_sequential*1000:.0f}ms, {queries_sequential} queries")
    print(f"Batched:    {time_batched*1000:.0f}ms, {queries_batched} queries")
    print(f"Speedup:    {time_sequential/time_batched:.1f}x")


if __name__ == "__main__":
    test_basic_mcts()
    test_mcts_game()
    test_mcts_speed()
    test_multiple_games()
    test_oracle_value_fn()
    test_mcts_oracle_vs_random()
    test_batched_mcts()
    test_batched_vs_sequential_benchmark()
