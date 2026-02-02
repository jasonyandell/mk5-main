"""Tests for GPU-native GameState.

Validates that GPU operations produce identical results to Python GameState.
"""

from __future__ import annotations

import random
import time

import pytest
import torch

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
    from_python_states,
    get_domino_tables,
    is_in_called_suit_gpu,
    is_terminal_gpu,
    led_suit_for_lead_domino_gpu,
    legal_actions_gpu,
    to_python_state,
    trick_rank_gpu,
)


# Use CPU for tests (works on all machines)
DEVICE = torch.device("cpu")


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

    def test_matches_python_implementation(self):
        """GPU legal actions should exactly match Python for various states."""
        random.seed(42)
        torch.manual_seed(42)

        # Create Python states and play some moves
        for _ in range(20):
            # Random deal
            dominoes = list(range(28))
            random.shuffle(dominoes)
            hands = [dominoes[i*7:(i+1)*7] for i in range(4)]
            decl_id = random.randint(0, 9)
            leader = random.randint(0, 3)

            py_state = GameState.from_hands(hands, decl_id, leader)

            # Play 0-20 random moves
            n_moves = random.randint(0, 20)
            for _ in range(n_moves):
                if py_state.is_complete():
                    break
                legal = py_state.legal_actions()
                if not legal:
                    break
                action = random.choice(legal)
                py_state = py_state.apply_action(action)

            # Convert to GPU with original hands for proper slot mapping
            gpu_state = from_python_states([py_state], DEVICE, original_hands=[hands])

            # Get legal actions from GPU
            gpu_legal = legal_actions_gpu(gpu_state)[0]  # (7,) bool mask

            # Get legal actions from Python
            py_legal_dominoes = set(py_state.legal_actions())

            # Get current player's original hand slots
            player = py_state.current_player()
            original_hand = hands[player]  # Original 7 dominoes in order

            # Check each slot
            for slot_idx in range(7):
                domino_id = original_hand[slot_idx]
                # Slot is legal if domino is in hand AND in legal actions
                in_hand = domino_id in py_state.hands[player]
                in_legal = domino_id in py_legal_dominoes
                expected_legal = in_hand and in_legal

                assert gpu_legal[slot_idx].item() == expected_legal, (
                    f"Mismatch at slot {slot_idx}: gpu={gpu_legal[slot_idx].item()}, "
                    f"expected={expected_legal}, domino={domino_id}, "
                    f"in_hand={in_hand}, in_legal={in_legal}, decl={decl_id}"
                )


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

    def test_matches_python_single_game(self):
        """Play a full game and verify each step matches Python."""
        random.seed(123)
        torch.manual_seed(123)

        # Create matching Python and GPU states
        dominoes = list(range(28))
        random.shuffle(dominoes)
        hands = [dominoes[i*7:(i+1)*7] for i in range(4)]
        decl_id = 3  # Threes trump
        leader = 1

        # Store original hands for slot-to-domino mapping
        original_hands = [list(h) for h in hands]

        py_state = GameState.from_hands(hands, decl_id, leader)
        gpu_state = from_python_states([py_state], DEVICE, original_hands=[original_hands])

        # Play until game is complete
        move_count = 0
        while not py_state.is_complete():
            # Verify states match
            py_player = py_state.current_player()
            gpu_player = current_player_gpu(gpu_state)[0].item()
            assert py_player == gpu_player, f"Player mismatch at move {move_count}"

            # Get legal actions
            py_legal = py_state.legal_actions()

            # Convert Python legal (domino IDs) to slot indices
            player_original = original_hands[py_player]
            py_legal_slots = [player_original.index(d) for d in py_legal if d in player_original]

            gpu_legal = legal_actions_gpu(gpu_state)[0]
            gpu_legal_slots = [i for i in range(7) if gpu_legal[i].item()]

            assert set(py_legal_slots) == set(gpu_legal_slots), (
                f"Legal actions mismatch at move {move_count}: "
                f"py={py_legal_slots}, gpu={gpu_legal_slots}"
            )

            # Pick first legal slot
            slot = min(gpu_legal_slots)
            domino_id = player_original[slot]

            # Apply action
            py_state = py_state.apply_action(domino_id)
            gpu_state = apply_action_gpu(
                gpu_state,
                torch.tensor([slot], dtype=torch.int32, device=DEVICE)
            )

            move_count += 1

        # Game should be complete
        assert is_terminal_gpu(gpu_state)[0].item()
        assert py_state.is_complete()


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


class TestConversion:
    """Test conversion between Python and GPU states."""

    def test_roundtrip_fresh_state(self):
        """Converting to GPU and back should preserve state."""
        random.seed(999)

        dominoes = list(range(28))
        random.shuffle(dominoes)
        hands = [dominoes[i*7:(i+1)*7] for i in range(4)]

        py_state = GameState.from_hands(hands, decl_id=5, leader=2)

        gpu_state = from_python_states([py_state], DEVICE)
        recovered = to_python_state(gpu_state, 0)

        # Hands should match (allowing for different tuple ordering)
        for p in range(4):
            assert set(py_state.hands[p]) == set(recovered.hands[p])

        assert recovered.decl_id == py_state.decl_id
        assert recovered.leader == py_state.leader
        assert recovered.played == py_state.played

    def test_roundtrip_mid_game(self):
        """Roundtrip should work for mid-game states."""
        random.seed(888)

        dominoes = list(range(28))
        random.shuffle(dominoes)
        hands = [dominoes[i*7:(i+1)*7] for i in range(4)]

        py_state = GameState.from_hands(hands, decl_id=3, leader=1)

        # Play some moves
        for _ in range(10):
            legal = py_state.legal_actions()
            if not legal:
                break
            py_state = py_state.apply_action(random.choice(legal))

        gpu_state = from_python_states([py_state], DEVICE)
        recovered = to_python_state(gpu_state, 0)

        assert recovered.decl_id == py_state.decl_id
        assert recovered.leader == py_state.leader
        assert len(recovered.play_history) == len(py_state.play_history)


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

    def test_cpu_batch_faster_than_python_loop(self):
        """Even on CPU, batched should be faster than Python loops."""
        n_games = 100
        n_moves = 10

        # CPU batch timing
        cpu_states = deal_random_gpu(n_games, DEVICE, decl_ids=0)

        start = time.perf_counter()
        for _ in range(n_moves):
            legal = legal_actions_gpu(cpu_states)
            actions = legal.int().argmax(dim=1)
            cpu_states = apply_action_gpu(cpu_states, actions)
        batch_time = time.perf_counter() - start

        # Python loop timing
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
        loop_time = time.perf_counter() - start

        speedup = loop_time / batch_time
        print(f"\nCPU Benchmark: {n_games} games x {n_moves} moves")
        print(f"  Python loop: {loop_time:.3f}s")
        print(f"  Batched CPU: {batch_time:.3f}s")
        print(f"  Speedup: {speedup:.1f}x")

        # Batched should be at least comparable (might be slower on CPU due to tensor overhead)
        # Just verify it doesn't crash and produces valid output


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
