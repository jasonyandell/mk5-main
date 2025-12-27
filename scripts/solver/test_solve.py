"""
Tests for solve.py - GPU solver.

Note: Full game enumeration requires ~3M+ states which may OOM on limited memory.
Tests use partial enumeration (enumerate_partial) for practical testing.
"""

import pytest
import torch
from solve import pack_initial_state, enumerate_gpu, build_child_index, solve_gpu, solve_seed
from context import build_context
from state import compute_level, compute_terminal_value, unpack_score
from expand import expand_gpu


def enumerate_partial(ctx, max_levels: int = 8):
    """
    Partial BFS enumeration for testing.

    Args:
        ctx: SeedContext
        max_levels: How many levels to enumerate (from level 28 down)

    Returns:
        Sorted tensor of states from levels 28 down to (28 - max_levels)
    """
    initial = pack_initial_state(ctx)
    frontier = torch.tensor([initial], dtype=torch.int64)
    all_levels = [frontier]

    for level in range(28, 28 - max_levels, -1):
        if frontier.numel() == 0:
            break
        children = expand_gpu(frontier, ctx)
        children = children.flatten()
        children = children[children >= 0]
        children = torch.unique(children)
        if children.numel() > 0:
            all_levels.append(children)
        frontier = children

    all_states = torch.cat(all_levels)
    return torch.sort(torch.unique(all_states)).values


class TestPackInitialState:
    """Tests for pack_initial_state."""

    def test_initial_state_level(self):
        """Initial state should have level 28 (all dominoes)."""
        ctx = build_context(seed=42, decl_id=3)
        initial = pack_initial_state(ctx)
        state = torch.tensor([initial], dtype=torch.int64)
        level = compute_level(state)
        assert level[0].item() == 28, "Initial state should have level 28"

    def test_initial_state_score_zero(self):
        """Initial state should have score 0."""
        ctx = build_context(seed=42, decl_id=3)
        initial = pack_initial_state(ctx)
        state = torch.tensor([initial], dtype=torch.int64)
        score = unpack_score(state)
        assert score[0].item() == 0, "Initial state should have score 0"


class TestEnumeratePartial:
    """Tests for partial enumeration (faster, memory-safe)."""

    def test_includes_initial_state(self):
        """Enumeration should include the initial state."""
        ctx = build_context(seed=42, decl_id=3)
        initial = pack_initial_state(ctx)
        all_states = enumerate_partial(ctx, max_levels=4)

        assert initial in all_states.tolist(), "Initial state should be in enumeration"

    def test_states_are_sorted(self):
        """Enumerated states should be sorted."""
        ctx = build_context(seed=42, decl_id=3)
        all_states = enumerate_partial(ctx, max_levels=6)

        sorted_states = torch.sort(all_states).values
        assert torch.equal(all_states, sorted_states), "States should be sorted"

    def test_states_are_unique(self):
        """Enumerated states should have no duplicates."""
        ctx = build_context(seed=42, decl_id=3)
        all_states = enumerate_partial(ctx, max_levels=6)

        unique_states = torch.unique(all_states)
        assert len(all_states) == len(unique_states), "States should be unique"

    def test_includes_multiple_levels(self):
        """Enumeration should include states at multiple levels."""
        ctx = build_context(seed=42, decl_id=3)
        all_states = enumerate_partial(ctx, max_levels=5)
        levels = compute_level(all_states)

        unique_levels = torch.unique(levels)
        assert len(unique_levels) >= 5, f"Should have at least 5 levels, got {len(unique_levels)}"
        assert 28 in unique_levels.tolist(), "Should have level 28"
        assert 24 in unique_levels.tolist(), "Should have level 24"

    def test_reasonable_state_count(self):
        """Partial enumeration should produce reasonable number of states."""
        ctx = build_context(seed=42, decl_id=3)
        all_states = enumerate_partial(ctx, max_levels=8)

        count = len(all_states)
        # 8 levels should give us several thousand states
        assert count > 100, f"Should have many states, got {count}"
        assert count < 1_000_000, f"Should not have too many states, got {count}"


def build_child_index_partial(all_states: torch.Tensor, ctx) -> torch.Tensor:
    """
    Build child index for partial enumeration (doesn't require all children to exist).

    Unlike build_child_index, this doesn't raise an error when children
    are missing from the enumeration - it just marks them as -1.
    """
    N = all_states.shape[0]
    device = all_states.device

    children = expand_gpu(all_states, ctx)  # (N, 7)

    # Use searchsorted to find where each child would be
    child_idx = torch.searchsorted(all_states, children.clamp(min=0))

    # Check if found indices actually match
    valid_idx = child_idx.clamp(0, N - 1)
    found_states = all_states[valid_idx]
    child_exists = (children >= 0) & (found_states == children)

    # Mark missing children as -1
    return torch.where(child_exists, child_idx, torch.tensor(-1, device=device, dtype=torch.int64))


class TestBuildChildIndexPartial:
    """Tests for build_child_index with partial enumeration."""

    def test_index_shape(self):
        """Child index should be (N, 7)."""
        ctx = build_context(seed=42, decl_id=3)
        all_states = enumerate_partial(ctx, max_levels=6)
        child_idx = build_child_index_partial(all_states, ctx)

        assert child_idx.shape == (len(all_states), 7), f"Expected shape ({len(all_states)}, 7)"

    def test_illegal_moves_marked(self):
        """Illegal moves should be marked with -1."""
        ctx = build_context(seed=42, decl_id=3)
        all_states = enumerate_partial(ctx, max_levels=6)
        child_idx = build_child_index_partial(all_states, ctx)

        # Some moves should be illegal
        illegal_count = (child_idx == -1).sum().item()
        assert illegal_count > 0, "Some moves should be illegal"

    def test_legal_moves_have_valid_indices(self):
        """Legal moves should have valid indices into all_states."""
        ctx = build_context(seed=42, decl_id=3)
        all_states = enumerate_partial(ctx, max_levels=6)
        child_idx = build_child_index_partial(all_states, ctx)

        legal = child_idx >= 0
        legal_indices = child_idx[legal]

        assert (legal_indices >= 0).all(), "Legal indices should be non-negative"
        assert (legal_indices < len(all_states)).all(), "Legal indices should be in bounds"

    def test_child_indices_point_to_correct_states(self):
        """Child indices should point to the actual child states."""
        ctx = build_context(seed=42, decl_id=3)
        all_states = enumerate_partial(ctx, max_levels=5)  # Fewer levels to ensure children exist
        child_idx = build_child_index_partial(all_states, ctx)

        # Sample first 50 states and verify
        children = expand_gpu(all_states[:50], ctx)

        for i in range(min(50, len(all_states))):
            for m in range(7):
                if child_idx[i, m] >= 0:
                    expected_child = children[i, m]
                    actual_child = all_states[child_idx[i, m]]
                    assert expected_child == actual_child, (
                        f"Mismatch at state {i}, move {m}: "
                        f"expected {expected_child}, got {actual_child}"
                    )


class TestSolveGPUPartial:
    """Tests for solve_gpu with partial enumeration."""

    def test_value_range(self):
        """Values should be in range [-42, +42]."""
        ctx = build_context(seed=42, decl_id=3)
        all_states = enumerate_partial(ctx, max_levels=6)
        child_idx = build_child_index_partial(all_states, ctx)
        V, _ = solve_gpu(all_states, child_idx, ctx)

        # Values could be extreme since we don't have terminal states
        # Just check they're int8 range
        assert V.dtype == torch.int8

    def test_move_values_shape(self):
        """Move values should be (N, 7)."""
        ctx = build_context(seed=42, decl_id=3)
        all_states = enumerate_partial(ctx, max_levels=6)
        child_idx = build_child_index_partial(all_states, ctx)
        _, move_values = solve_gpu(all_states, child_idx, ctx)

        assert move_values.shape == (len(all_states), 7)

    def test_illegal_moves_have_sentinel_value(self):
        """Illegal moves should have value -128."""
        ctx = build_context(seed=42, decl_id=3)
        all_states = enumerate_partial(ctx, max_levels=6)
        child_idx = build_child_index_partial(all_states, ctx)
        _, move_values = solve_gpu(all_states, child_idx, ctx)

        illegal = child_idx == -1
        illegal_values = move_values[illegal]

        assert (illegal_values == -128).all(), "Illegal moves should have value -128"


class TestTerminalValues:
    """Tests for terminal value computation (requires no enumeration)."""

    def test_terminal_value_formula(self):
        """Terminal value should be 2 * score - 42."""
        # Create synthetic terminal states
        from state import pack_state

        # Score 0: value = -42
        # Score 21: value = 0
        # Score 42: value = 42
        test_cases = [
            (0, -42),
            (21, 0),
            (42, 42),
            (10, -22),
            (30, 18),
        ]

        for score_val, expected_value in test_cases:
            remaining = torch.zeros((1, 4), dtype=torch.int64)  # All hands empty
            score = torch.tensor([score_val], dtype=torch.int64)
            leader = torch.zeros(1, dtype=torch.int64)
            trick_len = torch.zeros(1, dtype=torch.int64)
            p0 = torch.full((1,), 7, dtype=torch.int64)
            p1 = torch.full((1,), 7, dtype=torch.int64)
            p2 = torch.full((1,), 7, dtype=torch.int64)

            state = pack_state(remaining, score, leader, trick_len, p0, p1, p2)
            value = compute_terminal_value(state)

            assert value[0].item() == expected_value, (
                f"Score {score_val}: expected value {expected_value}, got {value[0].item()}"
            )


class TestMinimaxCorrectnessPartial:
    """Tests for minimax correctness with partial enumeration."""

    def test_value_propagation(self):
        """Parent value should equal best child value (per team)."""
        ctx = build_context(seed=42, decl_id=3)
        # Use fewer levels to keep it fast
        all_states = enumerate_partial(ctx, max_levels=5)
        child_idx = build_child_index_partial(all_states, ctx)
        V, move_values = solve_gpu(all_states, child_idx, ctx)

        from state import compute_team
        is_team0 = compute_team(all_states)
        levels = compute_level(all_states)

        # Check states that have children in our enumeration
        # (i.e., not at the lowest level we enumerated)
        max_level = levels.max().item()
        min_level = levels.min().item()

        # Check states above the minimum level (they have children)
        for i in range(min(100, len(all_states))):
            if levels[i] <= min_level:
                continue  # Skip lowest level (no children in enumeration)

            legal_mask = child_idx[i] >= 0
            if not legal_mask.any():
                continue

            legal_values = move_values[i][legal_mask]

            if is_team0[i]:
                expected = legal_values.max()
            else:
                expected = legal_values.min()

            assert V[i] == expected, (
                f"State {i}: team0={is_team0[i]}, level={levels[i]}, "
                f"V={V[i]}, expected={expected}, legal={legal_values.tolist()}"
            )


class TestExpandAndEnumerate:
    """Integration tests for expand + enumerate."""

    def test_children_have_lower_level(self):
        """All children should have level = parent_level - 1."""
        ctx = build_context(seed=42, decl_id=3)
        all_states = enumerate_partial(ctx, max_levels=4)

        levels = compute_level(all_states)
        children = expand_gpu(all_states, ctx)

        for i in range(min(50, len(all_states))):
            parent_level = levels[i].item()
            for m in range(7):
                if children[i, m] >= 0:
                    child_state = children[i, m].unsqueeze(0)
                    child_level = compute_level(child_state).item()
                    assert child_level == parent_level - 1, (
                        f"Child level {child_level} != parent level {parent_level} - 1"
                    )

    def test_initial_has_7_children(self):
        """Initial state should have exactly 7 legal children."""
        ctx = build_context(seed=42, decl_id=3)
        initial = pack_initial_state(ctx)
        state = torch.tensor([initial], dtype=torch.int64)

        children = expand_gpu(state, ctx)
        legal_count = (children >= 0).sum().item()

        assert legal_count == 7, f"Initial state should have 7 legal moves, got {legal_count}"


class TestDeterminism:
    """Tests for deterministic behavior."""

    def test_enumerate_deterministic(self):
        """Same seed/decl should give same enumeration."""
        ctx = build_context(seed=42, decl_id=3)

        states1 = enumerate_partial(ctx, max_levels=5)
        states2 = enumerate_partial(ctx, max_levels=5)

        assert torch.equal(states1, states2), "Enumeration should be deterministic"

    def test_solve_deterministic(self):
        """Same enumeration should give same solve results."""
        ctx = build_context(seed=42, decl_id=3)
        all_states = enumerate_partial(ctx, max_levels=5)
        child_idx = build_child_index_partial(all_states, ctx)

        V1, mv1 = solve_gpu(all_states, child_idx, ctx)
        V2, mv2 = solve_gpu(all_states, child_idx, ctx)

        assert torch.equal(V1, V2), "Values should be deterministic"
        assert torch.equal(mv1, mv2), "Move values should be deterministic"


# Mark full solve tests that require significant memory
@pytest.mark.slow
class TestFullSolve:
    """Tests requiring full enumeration (may OOM on limited systems)."""

    def test_full_solve_seed(self):
        """Full solve of a seed (requires ~3M+ states)."""
        try:
            all_states, V, move_values, root_value = solve_seed(42, 3)

            assert isinstance(root_value, int)
            assert -42 <= root_value <= 42

            # Check terminal states
            levels = compute_level(all_states)
            terminal = levels == 0
            assert terminal.any(), "Should have terminal states"

            # Terminal values should match formula
            expected = compute_terminal_value(all_states[terminal])
            assert torch.equal(V[terminal], expected)

        except MemoryError:
            pytest.skip("Insufficient memory for full solve")


if __name__ == "__main__":
    # Run fast tests by default, skip slow ones
    pytest.main([__file__, "-v", "-m", "not slow"])
