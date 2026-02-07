"""Tests for GPU-native GameState.

Validates that GPU operations produce identical results to Python GameState.
"""

from __future__ import annotations

import random
import time
import os

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available (GPUGameState is CUDA-only)",
)

from forge.eq.game import GameState
from forge.oracle.tables import (
    can_follow,
    led_suit_for_lead_domino,
    resolve_trick,
)
from forge.zeb.gpu_game_state import (
    GPUGameState,
    apply_action_gpu,
    can_follow_gpu,
    current_player_gpu,
    deal_random_gpu,
    get_domino_tables,
    is_in_called_suit_gpu,
    is_terminal_gpu,
    led_suit_for_lead_domino_gpu,
    legal_actions_gpu,
    trick_rank_gpu,
)


DEVICE = torch.device("cuda")


class TestDealRandomGpu:
    """Test deal_random_gpu produces valid deals."""

    def test_deal_produces_valid_hands(self):
        """Each game should have 4 hands of 7 unique dominoes."""
        n = 100
        states = deal_random_gpu(n, DEVICE)

        assert states.hands.shape == (n, 4, 7)

        for i in range(n):
            # Collect all dominoes in this game
            all_dominoes = states.hands[i].flatten().tolist()

            # Should have exactly 28 dominoes
            assert len(all_dominoes) == 28

            # All unique (no -1 since fresh deal)
            assert len(set(all_dominoes)) == 28

            # All in valid range [0, 27]
            assert all(0 <= d <= 27 for d in all_dominoes)

    def test_deal_sets_correct_initial_state(self):
        """Fresh deals should have no plays, empty tricks, zero scores."""
        n = 50
        states = deal_random_gpu(n, DEVICE)

        # No dominoes played
        assert (states.played_mask == False).all()

        # No plays in history
        assert (states.n_plays == 0).all()

        # Empty current trick
        assert (states.trick_len == 0).all()

        # Scores are zero
        assert (states.scores == 0).all()

    def test_deal_with_fixed_decl_and_leader(self):
        """Test specifying declaration and leader."""
        n = 20
        states = deal_random_gpu(n, DEVICE, decl_ids=5, leaders=2)

        assert (states.decl_id == 5).all()
        assert (states.leader == 2).all()

    def test_deal_with_random_decl(self):
        """Without decl_ids, should get random 0-9."""
        n = 1000
        states = deal_random_gpu(n, DEVICE, decl_ids=None)

        # Should see variety of declarations
        unique_decls = states.decl_id.unique()
        assert len(unique_decls) >= 5  # Statistically should see most declarations


class TestCurrentPlayerGpu:
    """Test current_player_gpu matches Python implementation."""

    def test_leading_player_is_leader(self):
        """When trick is empty, current player is leader."""
        states = deal_random_gpu(100, DEVICE)

        # Set various leaders
        states.leader = torch.randint(0, 4, (100,), dtype=torch.int32, device=DEVICE)
        states.trick_len = torch.zeros(100, dtype=torch.int32, device=DEVICE)

        players = current_player_gpu(states)
        assert (players == states.leader).all()

    def test_following_player_rotates(self):
        """Current player rotates as trick progresses."""
        n = 100
        states = deal_random_gpu(n, DEVICE)

        for trick_pos in range(4):
            states.trick_len = torch.full((n,), trick_pos, dtype=torch.int32, device=DEVICE)
            states.leader = torch.randint(0, 4, (n,), dtype=torch.int32, device=DEVICE)

            players = current_player_gpu(states)
            expected = (states.leader + trick_pos) % 4
            assert (players == expected).all()


class TestLegalActionsGpu:
    """Test legal_actions_gpu matches Python GameState.legal_actions()."""

    def test_leading_all_in_hand_legal(self):
        """When leading, all dominoes in hand are legal."""
        n = 50
        states = deal_random_gpu(n, DEVICE, decl_ids=0)

        # All fresh deals start with leading
        legal = legal_actions_gpu(states)

        # All 7 slots should be legal (all dominoes in hand)
        assert legal.shape == (n, 7)
        assert (legal == True).all()

    # NOTE: Python->GPU GameState conversion is intentionally disabled, so
    # cross-checking arbitrary mid-game Python states is out of scope here.


class TestApplyActionGpu:
    """Test apply_action_gpu matches Python GameState.apply_action()."""

    def test_single_move_removes_domino(self):
        """Playing a domino should mark it as played."""
        states = deal_random_gpu(10, DEVICE, decl_ids=0, leaders=0)

        # Play slot 0 for all games
        actions = torch.zeros(10, dtype=torch.int32, device=DEVICE)
        new_states = apply_action_gpu(states, actions)

        # Slot 0 should now be -1 for player 0
        assert (new_states.hands[:, 0, 0] == -1).all()

        # n_plays should be 1
        assert (new_states.n_plays == 1).all()

        # trick_len should be 1
        assert (new_states.trick_len == 1).all()

    def test_full_trick_resolves_correctly(self):
        """After 4 plays, trick should resolve and clear."""
        n = 20
        states = deal_random_gpu(n, DEVICE, decl_ids=0, leaders=0)

        # Play 4 moves (one complete trick)
        for i in range(4):
            legal = legal_actions_gpu(states)
            # Pick first legal action for each game
            actions = legal.int().argmax(dim=1)
            states = apply_action_gpu(states, actions)

        # After 4 plays, trick should be cleared
        assert (states.trick_len == 0).all()

        # n_plays should be 4
        assert (states.n_plays == 4).all()

        # At least one team should have points > 0 (1 base + count points)
        total_points = states.scores.sum(dim=1)
        assert (total_points > 0).all()

    # NOTE: Python->GPU GameState conversion is intentionally disabled, so
    # full cross-checking against Python mid-game trajectories is out of scope here.


class TestIsTerminalGpu:
    """Test is_terminal_gpu."""

    def test_fresh_deal_not_terminal(self):
        """Fresh deals should not be terminal."""
        states = deal_random_gpu(100, DEVICE)
        terminal = is_terminal_gpu(states)
        assert (terminal == False).all()

    def test_full_game_is_terminal(self):
        """After 28 plays, game is terminal."""
        # Create state with n_plays = 28
        states = deal_random_gpu(10, DEVICE)
        states.n_plays = torch.full((10,), 28, dtype=torch.int32, device=DEVICE)

        terminal = is_terminal_gpu(states)
        assert (terminal == True).all()


class TestGameRulesGpu:
    """Test GPU implementations of game rules match Python."""

    def test_is_in_called_suit_pip_trump(self):
        """Test is_in_called_suit for pip trumps."""
        tables = get_domino_tables(DEVICE)

        # Domino 0 is (0,0) - should be in called suit for decl 0 (blanks)
        domino = torch.tensor([0], dtype=torch.int32, device=DEVICE)
        decl = torch.tensor([0], dtype=torch.int32, device=DEVICE)
        result = is_in_called_suit_gpu(domino, decl, tables)
        assert result[0].item()

        # Domino 27 is (6,6) - should be in called suit for decl 6 (sixes)
        domino = torch.tensor([27], dtype=torch.int32, device=DEVICE)
        decl = torch.tensor([6], dtype=torch.int32, device=DEVICE)
        result = is_in_called_suit_gpu(domino, decl, tables)
        assert result[0].item()

        # Domino 27 (6,6) should NOT be in called suit for decl 0 (blanks)
        decl = torch.tensor([0], dtype=torch.int32, device=DEVICE)
        result = is_in_called_suit_gpu(domino, decl, tables)
        assert not result[0].item()

    def test_is_in_called_suit_doubles(self):
        """Test is_in_called_suit for doubles trump/suit."""
        tables = get_domino_tables(DEVICE)

        # All doubles should be in called suit for doubles trump (7) and suit (8)
        doubles = torch.tensor([0, 2, 5, 9, 14, 20, 27], dtype=torch.int32, device=DEVICE)  # All 7 doubles

        for decl_val in [7, 8]:
            decl = torch.full_like(doubles, decl_val)
            result = is_in_called_suit_gpu(doubles, decl, tables)
            assert result.all(), f"Not all doubles in called suit for decl {decl_val}"

    def test_led_suit_matches_python(self):
        """Test led_suit_for_lead_domino matches Python."""
        from forge.oracle.tables import led_suit_for_lead_domino

        tables = get_domino_tables(DEVICE)

        for domino_id in range(28):
            for decl_id in range(10):
                py_result = led_suit_for_lead_domino(domino_id, decl_id)

                domino = torch.tensor([domino_id], dtype=torch.int32, device=DEVICE)
                decl = torch.tensor([decl_id], dtype=torch.int32, device=DEVICE)
                gpu_result = led_suit_for_lead_domino_gpu(domino, decl, tables)[0].item()

                assert gpu_result == py_result, (
                    f"led_suit mismatch for domino {domino_id}, decl {decl_id}: "
                    f"gpu={gpu_result}, py={py_result}"
                )

    def test_can_follow_matches_python(self):
        """Test can_follow matches Python implementation."""
        from forge.oracle.tables import can_follow

        tables = get_domino_tables(DEVICE)

        for domino_id in range(28):
            for led_suit in range(8):  # 0-6 pip suits, 7 = called suit
                for decl_id in range(10):
                    py_result = can_follow(domino_id, led_suit, decl_id)

                    domino = torch.tensor([domino_id], dtype=torch.int32, device=DEVICE)
                    suit = torch.tensor([led_suit], dtype=torch.int32, device=DEVICE)
                    decl = torch.tensor([decl_id], dtype=torch.int32, device=DEVICE)
                    gpu_result = can_follow_gpu(domino, suit, decl, tables)[0].item()

                    assert gpu_result == py_result, (
                        f"can_follow mismatch for domino {domino_id}, led_suit {led_suit}, "
                        f"decl {decl_id}: gpu={gpu_result}, py={py_result}"
                    )


class TestDisabledPythonBootstrap:
    def test_from_python_states_is_disabled(self):
        from forge.zeb import gpu_game_state
        with pytest.raises(RuntimeError, match="from_python_states\\(\\) is disabled"):
            _ = gpu_game_state.from_python_states([], DEVICE)


class TestBenchmark:
    """Benchmark GPU vs Python performance."""

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA not available for benchmark"
    )
    def test_gpu_faster_than_python(self):
        """GPU batch simulation should be faster than Python loops.

        Note: The actual speedup depends heavily on:
        - GPU warmup
        - Batch size (larger batches = more GPU efficiency)
        - Python interpreter overhead

        For small batches, the speedup may be modest. The real value comes
        from being able to keep all state on GPU for MCTS integration.
        """
        if os.getenv("ZEB_RUN_BENCHMARKS") != "1":
            pytest.skip("Set ZEB_RUN_BENCHMARKS=1 to run GPU/CPU benchmark (not stable across hardware)")

        n_games = 1000
        n_moves = 10  # Play 10 moves per game

        # GPU timing
        device = torch.device("cuda")
        gpu_states = deal_random_gpu(n_games, device, decl_ids=0)

        # Warmup - important for accurate GPU timing
        for _ in range(5):
            legal = legal_actions_gpu(gpu_states)
            actions = legal.int().argmax(dim=1)
            gpu_states = apply_action_gpu(gpu_states, actions)

        # Reset and time
        gpu_states = deal_random_gpu(n_games, device, decl_ids=0)
        torch.cuda.synchronize()
        start = time.perf_counter()

        for _ in range(n_moves):
            legal = legal_actions_gpu(gpu_states)
            actions = legal.int().argmax(dim=1)
            gpu_states = apply_action_gpu(gpu_states, actions)

        torch.cuda.synchronize()
        gpu_time = time.perf_counter() - start

        # Python timing
        random.seed(42)
        py_states = []
        for _ in range(n_games):
            dominoes = list(range(28))
            random.shuffle(dominoes)
            hands = [dominoes[i*7:(i+1)*7] for i in range(4)]
            py_states.append(GameState.from_hands(hands, 0, 0))

        start = time.perf_counter()

        for i in range(n_games):
            state = py_states[i]
            for _ in range(n_moves):
                legal = state.legal_actions()
                if legal:
                    state = state.apply_action(legal[0])
            py_states[i] = state

        py_time = time.perf_counter() - start

        speedup = py_time / gpu_time
        print(f"\nBenchmark: {n_games} games x {n_moves} moves")
        print(f"  Python: {py_time:.3f}s ({n_games * n_moves / py_time:.0f} moves/sec)")
        print(f"  GPU:    {gpu_time:.3f}s ({n_games * n_moves / gpu_time:.0f} moves/sec)")
        print(f"  Speedup: {speedup:.1f}x")

        # Should be at least faster (GPU wins at scale, and avoids CPU-GPU transfers)
        # The main benefit is avoiding Python object overhead in MCTS, not raw speed
        assert speedup > 1.0, f"Expected GPU to be faster, got {speedup:.1f}x"


class TestFullGameSimulation:
    """Test complete game simulations."""

    def test_play_full_game_to_completion(self):
        """Play games until terminal and verify point totals."""
        n = 50
        states = deal_random_gpu(n, DEVICE, decl_ids=0)

        # Play until all games complete
        max_moves = 28
        for _ in range(max_moves):
            terminal = is_terminal_gpu(states)
            if terminal.all():
                break

            legal = legal_actions_gpu(states)
            # Pick random legal action
            # Mask out illegal actions with large negative
            masked = legal.float() - (~legal).float() * 1e9
            actions = masked.argmax(dim=1)
            states = apply_action_gpu(states, actions)

        # All games should be complete
        assert is_terminal_gpu(states).all()

        # Total points per game should be 42 (7 tricks x (1 base + 5 count avg))
        # Actually: 7 tricks base + 35 count points = 42 total
        total_points = states.scores.sum(dim=1)
        assert (total_points == 42).all(), f"Point totals: {total_points.tolist()}"

    def test_multiple_declarations(self):
        """Test games with different declarations all complete correctly."""
        for decl_id in range(10):
            states = deal_random_gpu(10, DEVICE, decl_ids=decl_id)

            for _ in range(28):
                if is_terminal_gpu(states).all():
                    break

                legal = legal_actions_gpu(states)
                masked = legal.float() - (~legal).float() * 1e9
                actions = masked.argmax(dim=1)
                states = apply_action_gpu(states, actions)

            assert is_terminal_gpu(states).all(), f"Games with decl {decl_id} didn't complete"

            # Points should sum to 42
            total_points = states.scores.sum(dim=1)
            assert (total_points == 42).all(), f"Wrong points for decl {decl_id}: {total_points.tolist()}"
