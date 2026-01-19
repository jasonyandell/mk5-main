"""Tests for Phase 3 multi-game batching architecture (t42-tg2r).

This test file verifies that the multi-game batching implementation correctly:
1. Processes games in lockstep (all games at same decision round)
2. Batches oracle calls by decl_id
3. Concatenates worlds across games for efficient GPU usage
4. Reduces kernel launches from N_games × 28 to 28 × N_decl_types

Run with: python -m pytest forge/eq/test_batching_phase3.py -xvs
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from forge.eq.generate_batched import generate_eq_games_batched
from forge.eq.generate_game import generate_eq_game
from forge.oracle.rng import deal_from_seed


class InstrumentedOracle:
    """Oracle that tracks batching behavior for testing Phase 3."""

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.query_count = 0
        self.multi_state_calls = []  # Track each call: (n_worlds, decl_id, n_actors)

    def query_batch(self, worlds, game_state_info, current_player):
        """Single-game query (fallback)."""
        self.query_count += 1
        n = len(worlds)
        return torch.randn(n, 7, device=self.device)

    def query_batch_multi_state(
        self,
        *,
        worlds,
        decl_id,
        actors,
        leaders,
        trick_plays_list,
        remaining,
    ):
        """Multi-state batched query (Phase 3 optimization)."""
        n_samples = len(worlds)
        n_actors = len(set(actors))
        self.multi_state_calls.append({
            "n_worlds": n_samples,
            "decl_id": decl_id,
            "n_unique_actors": n_actors,
        })
        self.query_count += 1
        return torch.randn(n_samples, 7, device=self.device)


def test_phase3_batching_reduces_oracle_calls():
    """Verify Phase 3 batching reduces oracle calls vs per-game generation.

    Key insight: With N games and K unique decl_ids, we should make:
    - Per-game: N × 28 oracle calls
    - Batched: 28 × K oracle calls (where K ≤ N, typically K ≤ 10)

    This is the core optimization of Phase 3.
    """
    n_games = 10
    n_samples = 10  # Small for speed

    # Generate test data with diverse decl_ids
    rng = np.random.default_rng(42)
    hands_list = [deal_from_seed(int(rng.integers(0, 2**31))) for _ in range(n_games)]
    decl_ids = [int(rng.integers(0, 10)) for _ in range(n_games)]
    world_rngs = [np.random.default_rng(int(rng.integers(0, 2**31))) for _ in range(n_games)]

    # Count unique decl_ids
    n_unique_decls = len(set(decl_ids))

    # Test batched generation
    oracle_batched = InstrumentedOracle()
    records_batched = generate_eq_games_batched(
        oracle_batched, hands_list, decl_ids,
        n_samples=n_samples,
        world_rngs=world_rngs
    )

    # Verify batching efficiency
    assert len(records_batched) == n_games
    assert all(len(r.decisions) == 28 for r in records_batched)

    # Phase 3 should make ~28 × n_unique_decls calls
    expected_min = 28  # Best case: all same decl_id
    expected_max = 28 * n_unique_decls  # Worst case: all different decl_ids
    actual_calls = oracle_batched.query_count

    print(f"\nPhase 3 Batching Results:")
    print(f"  Games: {n_games}")
    print(f"  Unique decl_ids: {n_unique_decls}")
    print(f"  Oracle calls: {actual_calls}")
    print(f"  Expected range: [{expected_min}, {expected_max}]")
    print(f"  Per-game would be: {n_games * 28}")
    print(f"  Reduction: {(1 - actual_calls / (n_games * 28)) * 100:.1f}%")

    assert expected_min <= actual_calls <= expected_max, (
        f"Oracle calls {actual_calls} outside expected range [{expected_min}, {expected_max}]"
    )

    # Verify we're using batched multi-state calls
    assert len(oracle_batched.multi_state_calls) == actual_calls, (
        "All calls should use query_batch_multi_state (Phase 3 optimization)"
    )


def test_phase3_concatenates_worlds_across_games():
    """Verify Phase 3 concatenates worlds from multiple games into single batches.

    When games share the same decl_id, their worlds should be concatenated
    into a single oracle call for better GPU utilization.
    """
    n_games = 6
    n_samples = 10

    # All games use SAME decl_id to maximize batching
    rng = np.random.default_rng(100)
    hands_list = [deal_from_seed(int(rng.integers(0, 2**31))) for _ in range(n_games)]
    decl_ids = [3] * n_games  # All same decl_id
    world_rngs = [np.random.default_rng(int(rng.integers(0, 2**31))) for _ in range(n_games)]

    oracle = InstrumentedOracle()
    records = generate_eq_games_batched(
        oracle, hands_list, decl_ids,
        n_samples=n_samples,
        world_rngs=world_rngs
    )

    assert len(records) == n_games

    # With all same decl_id, we should make exactly 28 calls (one per decision round)
    assert oracle.query_count == 28, (
        f"Expected 28 oracle calls (one per decision round), got {oracle.query_count}"
    )

    # Each call should have worlds from ALL active games
    # Early calls should have ~n_games × n_samples worlds
    early_calls = [c for c in oracle.multi_state_calls if c["n_worlds"] >= n_games * n_samples * 0.8]
    assert len(early_calls) >= 10, (
        f"Expected at least 10 calls with large batches, got {len(early_calls)}. "
        f"This suggests games aren't being batched together properly."
    )

    print(f"\nWorld Concatenation Test:")
    print(f"  Games: {n_games}, Samples: {n_samples}")
    print(f"  Oracle calls: {oracle.query_count}")
    print(f"  Max batch size: {max(c['n_worlds'] for c in oracle.multi_state_calls)}")
    print(f"  Min batch size: {min(c['n_worlds'] for c in oracle.multi_state_calls)}")
    print(f"  Expected max: ~{n_games * n_samples}")


def test_phase3_games_processed_in_lockstep():
    """Verify games are processed in lockstep (all at same decision round).

    All games should progress through decisions together:
    - Decision 0 for all games
    - Then decision 1 for all games
    - Etc.

    This is key for effective batching.
    """
    n_games = 4
    n_samples = 5

    rng = np.random.default_rng(200)
    hands_list = [deal_from_seed(int(rng.integers(0, 2**31))) for _ in range(n_games)]
    decl_ids = [int(rng.integers(0, 10)) for _ in range(n_games)]
    world_rngs = [np.random.default_rng(int(rng.integers(0, 2**31))) for _ in range(n_games)]

    oracle = InstrumentedOracle()
    records = generate_eq_games_batched(
        oracle, hands_list, decl_ids,
        n_samples=n_samples,
        world_rngs=world_rngs
    )

    # All games should have exactly 28 decisions
    assert all(len(r.decisions) == 28 for r in records), (
        "All games should complete with exactly 28 decisions (lockstep processing)"
    )

    # Verify lockstep: decision counts should be identical across games
    decision_counts = [len(r.decisions) for r in records]
    assert len(set(decision_counts)) == 1, (
        f"Games should all have same decision count (lockstep), got {decision_counts}"
    )

    print(f"\nLockstep Test:")
    print(f"  All {n_games} games completed with {decision_counts[0]} decisions")
    print(f"  Oracle calls: {oracle.query_count}")
    print(f"  Avg calls per decision round: {oracle.query_count / 28:.1f}")


def test_phase3_correctness_matches_single_game():
    """Verify batched generation produces same results as per-game generation.

    This is a regression test to ensure Phase 3 optimizations don't change
    the generated data.
    """
    n_games = 3
    n_samples = 5

    rng = np.random.default_rng(300)
    hands_list = [deal_from_seed(int(rng.integers(0, 2**31))) for _ in range(n_games)]
    decl_ids = [int(rng.integers(0, 10)) for _ in range(n_games)]

    # Use deterministic oracle for reproducibility
    class DeterministicOracle:
        def __init__(self):
            self.device = "cpu"
            self.call_idx = 0

        def _q_for(self, world, actor, decl_id):
            opp_sum = sum(sum(world[p]) for p in range(4) if p != actor)
            q = torch.zeros(7)
            for local_idx, domino_id in enumerate(world[actor]):
                q[local_idx] = float(domino_id + decl_id) - 0.01 * float(opp_sum)
            return q

        def query_batch(self, worlds, game_state_info, current_player):
            decl_id = game_state_info.get("decl_id", 0)
            return torch.stack([self._q_for(w, current_player, decl_id) for w in worlds])

        def query_batch_multi_state(
            self, *, worlds, decl_id, actors, leaders, trick_plays_list, remaining
        ):
            return torch.stack(
                [self._q_for(w, int(actors[i]), int(decl_id)) for i, w in enumerate(worlds)]
            )

    # Generate with separate oracles but same deterministic logic
    oracle_single = DeterministicOracle()
    records_single = [
        generate_eq_game(
            oracle_single, hands_list[i], decl_ids[i],
            n_samples=n_samples,
            world_rng=np.random.default_rng(1000 + i)
        )
        for i in range(n_games)
    ]

    oracle_batched = DeterministicOracle()
    records_batched = generate_eq_games_batched(
        oracle_batched, hands_list, decl_ids,
        n_samples=n_samples,
        world_rngs=[np.random.default_rng(1000 + i) for i in range(n_games)]
    )

    # Verify identical results
    assert len(records_batched) == len(records_single) == n_games

    for i, (r_batch, r_single) in enumerate(zip(records_batched, records_single)):
        assert len(r_batch.decisions) == len(r_single.decisions) == 28, f"Game {i}"

        for j, (d_batch, d_single) in enumerate(zip(r_batch.decisions, r_single.decisions)):
            # Key fields should match
            assert d_batch.player == d_single.player, f"Game {i}, Decision {j}"
            assert d_batch.action_taken == d_single.action_taken, f"Game {i}, Decision {j}"
            assert torch.allclose(d_batch.e_q_mean, d_single.e_q_mean), f"Game {i}, Decision {j}"
            assert torch.equal(d_batch.legal_mask, d_single.legal_mask), f"Game {i}, Decision {j}"

    print(f"\nCorrectness Test: Batched matches single-game generation ✓")


def test_phase3_handles_different_decl_ids():
    """Verify batching correctly handles games with different decl_ids.

    Games with different decl_ids should be batched separately, as they
    require different trump suit evaluations.
    """
    n_games = 8
    n_samples = 5

    rng = np.random.default_rng(400)
    hands_list = [deal_from_seed(int(rng.integers(0, 2**31))) for _ in range(n_games)]

    # Create controlled decl_id distribution: 4 games with decl_id=0, 4 with decl_id=5
    decl_ids = [0, 0, 0, 0, 5, 5, 5, 5]
    world_rngs = [np.random.default_rng(int(rng.integers(0, 2**31))) for _ in range(n_games)]

    oracle = InstrumentedOracle()
    records = generate_eq_games_batched(
        oracle, hands_list, decl_ids,
        n_samples=n_samples,
        world_rngs=world_rngs
    )

    assert len(records) == n_games

    # With 2 unique decl_ids, we should make ~28 × 2 = 56 calls
    expected_calls = 28 * 2
    assert oracle.query_count == expected_calls, (
        f"Expected {expected_calls} oracle calls (28 rounds × 2 decl groups), "
        f"got {oracle.query_count}"
    )

    # Verify decl_ids in calls
    decl_ids_in_calls = set(c["decl_id"] for c in oracle.multi_state_calls)
    assert decl_ids_in_calls == {0, 5}, (
        f"Expected calls for decl_ids {{0, 5}}, got {decl_ids_in_calls}"
    )

    # Count calls per decl_id (should be ~28 each)
    calls_per_decl = {}
    for call in oracle.multi_state_calls:
        decl_id = call["decl_id"]
        calls_per_decl[decl_id] = calls_per_decl.get(decl_id, 0) + 1

    print(f"\nDecl ID Handling Test:")
    print(f"  Games: {n_games}, Unique decl_ids: 2")
    print(f"  Oracle calls: {oracle.query_count}")
    print(f"  Calls per decl_id: {calls_per_decl}")

    for decl_id, count in calls_per_decl.items():
        assert count == 28, f"Expected 28 calls for decl_id {decl_id}, got {count}"
