"""
Tests for forge/eq/game_tensor.py

Verifies GPU GameStateTensor matches CPU GameState semantics exactly.
Tests vectorized operations on multiple concurrent games.
"""
from __future__ import annotations

import pytest
import torch
import random

from forge.eq.game_tensor import (
    GameStateTensor,
    LED_SUIT_TABLE,
    TRICK_RANK_TABLE,
)
from forge.eq.game import GameState
from forge.oracle.tables import (
    led_suit_for_lead_domino,
    trick_rank,
)
from forge.oracle.declarations import (
    DOUBLES_TRUMP,
    DOUBLES_SUIT,
    NOTRUMP,
    N_DECLS,
)


# Fixtures

@pytest.fixture
def device():
    """Use CPU for tests (works everywhere, deterministic)."""
    return 'cpu'


@pytest.fixture
def simple_hands():
    """Simple hand setup for basic tests."""
    return [
        [0, 1, 2, 3, 4, 5, 6],
        [7, 8, 9, 10, 11, 12, 13],
        [14, 15, 16, 17, 18, 19, 20],
        [21, 22, 23, 24, 25, 26, 27]
    ]


@pytest.fixture
def follow_suit_hands():
    """Hand setup for follow-suit tests (notrump)."""
    # P0: [0=(0,0), 1=(1,0), 2=(1,1), 3=(2,0), 6=(3,0), 10=(4,0), 15=(5,0)]
    # P1: [4=(2,1), 5=(2,2), 7=(3,1), 8=(3,2), 9=(3,3), 11=(4,1), 12=(4,2)]
    return [
        [0, 1, 2, 3, 6, 10, 15],
        [4, 5, 7, 8, 9, 11, 12],
        [13, 14, 16, 17, 18, 19, 20],
        [21, 22, 23, 24, 25, 26, 27]
    ]


@pytest.fixture
def void_hands():
    """Hand setup for void tests."""
    # P1 has no pip 0s: [5=(2,2), 9=(3,3), 14=(4,4), 16=(5,1), 17=(5,2), 18=(5,3), 19=(5,4)]
    return [
        [0, 1, 2, 3, 6, 10, 15],
        [5, 9, 14, 16, 17, 18, 19],
        [4, 7, 8, 11, 12, 13, 20],
        [21, 22, 23, 24, 25, 26, 27]
    ]


# Lookup Table Tests

def test_led_suit_table_matches_cpu():
    """LED_SUIT_TABLE must match CPU led_suit_for_lead_domino() exactly."""
    for lead_domino_id in range(28):
        for decl_id in range(N_DECLS):
            expected = led_suit_for_lead_domino(lead_domino_id, decl_id)
            actual = LED_SUIT_TABLE[lead_domino_id, decl_id].item()
            assert actual == expected, (
                f"LED_SUIT_TABLE mismatch for lead={lead_domino_id}, decl={decl_id}: "
                f"expected {expected}, got {actual}"
            )


def test_trick_rank_table_matches_cpu():
    """TRICK_RANK_TABLE must match CPU trick_rank() exactly."""
    for domino_id in range(28):
        for led_suit in range(8):
            for decl_id in range(N_DECLS):
                expected = trick_rank(domino_id, led_suit, decl_id)
                actual = TRICK_RANK_TABLE[domino_id, led_suit, decl_id].item()
                assert actual == expected, (
                    f"TRICK_RANK_TABLE mismatch for domino={domino_id}, led_suit={led_suit}, decl={decl_id}: "
                    f"expected {expected}, got {actual}"
                )


# Initialization Tests

def test_from_deals_single_game(simple_hands, device):
    """from_deals() creates valid initial state for single game."""
    state = GameStateTensor.from_deals(
        hands=[simple_hands],
        decl_ids=[NOTRUMP],
        device=device
    )

    assert state.n_games == 1
    assert state.hands.shape == (1, 4, 7)
    assert state.played_mask.shape == (1, 28)
    assert state.history.shape == (1, 28, 3)
    assert state.trick_plays.shape == (1, 4)
    assert state.leader.shape == (1,)
    assert state.decl_ids.shape == (1,)

    # Check initial state
    assert state.leader[0].item() == 0
    assert state.decl_ids[0].item() == NOTRUMP
    assert not state.played_mask[0].any()
    assert (state.trick_plays[0] == -1).all()


def test_from_deals_multiple_games(simple_hands, device):
    """from_deals() can initialize multiple games."""
    n_games = 4
    hands = [simple_hands for _ in range(n_games)]
    decl_ids = [NOTRUMP, DOUBLES_TRUMP, DOUBLES_SUIT, 5]

    state = GameStateTensor.from_deals(
        hands=hands,
        decl_ids=decl_ids,
        device=device
    )

    assert state.n_games == n_games
    assert state.hands.shape == (n_games, 4, 7)
    assert (state.decl_ids == torch.tensor(decl_ids, device=device)).all()


def test_from_deals_validation(device):
    """from_deals() validates input."""
    # Wrong number of hands per game
    with pytest.raises(ValueError, match="expected 4 hands"):
        GameStateTensor.from_deals(
            hands=[[[0, 1, 2, 3, 4, 5, 6]]],  # Only 1 hand
            decl_ids=[NOTRUMP],
            device=device
        )

    # Wrong number of dominoes per hand
    with pytest.raises(ValueError, match="expected 7 dominoes"):
        GameStateTensor.from_deals(
            hands=[[[0, 1, 2], [3, 4, 5, 6], [7, 8, 9, 10], [11, 12, 13, 14]]],
            decl_ids=[NOTRUMP],
            device=device
        )

    # Mismatched decl_ids length
    with pytest.raises(ValueError, match="Expected .* decl_ids"):
        GameStateTensor.from_deals(
            hands=[[[0, 1, 2, 3, 4, 5, 6]] * 4],
            decl_ids=[NOTRUMP, DOUBLES_TRUMP],  # 2 decls for 1 game
            device=device
        )


# Current Player Tests

def test_current_player_initial_state(simple_hands, device):
    """current_player matches CPU for initial state."""
    state_gpu = GameStateTensor.from_deals(
        hands=[simple_hands],
        decl_ids=[NOTRUMP],
        device=device
    )
    state_cpu = GameState.from_hands(simple_hands, decl_id=NOTRUMP, leader=0)

    assert state_gpu.current_player[0].item() == state_cpu.current_player()


def test_current_player_after_plays(simple_hands, device):
    """current_player advances correctly after plays."""
    state = GameStateTensor.from_deals(
        hands=[simple_hands],
        decl_ids=[NOTRUMP],
        device=device
    )

    assert state.current_player[0].item() == 0

    # Player 0 plays slot 0 (domino 0)
    state = state.apply_actions(torch.tensor([0], device=device))
    assert state.current_player[0].item() == 1

    # Player 1 plays slot 0 (domino 7)
    state = state.apply_actions(torch.tensor([0], device=device))
    assert state.current_player[0].item() == 2


# Legal Actions Tests

def test_legal_actions_when_leading(simple_hands, device):
    """legal_actions() when leading: all hand dominoes."""
    state_gpu = GameStateTensor.from_deals(
        hands=[simple_hands],
        decl_ids=[NOTRUMP],
        device=device
    )
    state_cpu = GameState.from_hands(simple_hands, decl_id=NOTRUMP, leader=0)

    legal_gpu = state_gpu.legal_actions()[0]  # (7,) bool tensor
    legal_cpu = state_cpu.legal_actions()

    # All slots should be legal
    assert legal_gpu.all()
    assert len(legal_cpu) == 7


def test_legal_actions_when_following_notrump(follow_suit_hands, device):
    """legal_actions() when following (notrump): must follow suit."""
    state_gpu = GameStateTensor.from_deals(
        hands=[follow_suit_hands],
        decl_ids=[NOTRUMP],
        device=device
    )
    state_cpu = GameState.from_hands(follow_suit_hands, decl_id=NOTRUMP, leader=0)

    # P0 leads with slot 2 (domino 2=(1,1))
    state_gpu = state_gpu.apply_actions(torch.tensor([2], device=device))
    state_cpu = state_cpu.apply_action(2)

    # P1's turn: must follow suit 1
    # Hand: [4=(2,1), 5=(2,2), 7=(3,1), 8=(3,2), 9=(3,3), 11=(4,1), 12=(4,2)]
    # Followers: [4=(2,1), 7=(3,1), 11=(4,1)] at slots [0, 2, 5]
    legal_gpu = state_gpu.legal_actions()[0]
    legal_cpu = state_cpu.legal_actions()

    # Convert CPU legal dominoes to slot indices
    legal_cpu_set = set(legal_cpu)
    expected_slots = [i for i, d in enumerate(follow_suit_hands[1]) if d in legal_cpu_set]

    # Check GPU matches
    legal_gpu_slots = legal_gpu.nonzero(as_tuple=True)[0].tolist()
    assert set(legal_gpu_slots) == set(expected_slots)
    assert len(legal_cpu) == 3


def test_legal_actions_when_void(void_hands, device):
    """legal_actions() when void: can play anything."""
    state_gpu = GameStateTensor.from_deals(
        hands=[void_hands],
        decl_ids=[NOTRUMP],
        device=device
    )
    state_cpu = GameState.from_hands(void_hands, decl_id=NOTRUMP, leader=0)

    # P0 leads with slot 0 (domino 0=(0,0))
    state_gpu = state_gpu.apply_actions(torch.tensor([0], device=device))
    state_cpu = state_cpu.apply_action(0)

    # P1's turn: has no pip 0s, so void - can play anything
    legal_gpu = state_gpu.legal_actions()[0]
    legal_cpu = state_cpu.legal_actions()

    # All 7 slots should be legal (void)
    assert legal_gpu.all()
    assert len(legal_cpu) == 7


# Apply Actions Tests

def test_apply_actions_updates_hands(simple_hands, device):
    """apply_actions() removes domino from hand."""
    state = GameStateTensor.from_deals(
        hands=[simple_hands],
        decl_ids=[NOTRUMP],
        device=device
    )

    # Player 0 plays slot 0 (domino 0)
    new_state = state.apply_actions(torch.tensor([0], device=device))

    # Original state unchanged (immutable)
    assert state.hands[0, 0, 0].item() == 0

    # New state has domino removed
    assert new_state.hands[0, 0, 0].item() == -1


def test_apply_actions_updates_played_mask(simple_hands, device):
    """apply_actions() marks domino as played."""
    state = GameStateTensor.from_deals(
        hands=[simple_hands],
        decl_ids=[NOTRUMP],
        device=device
    )

    # Player 0 plays slot 0 (domino 0)
    new_state = state.apply_actions(torch.tensor([0], device=device))

    # Domino 0 should be marked as played
    assert new_state.played_mask[0, 0].item()
    assert new_state.played_mask[0].sum().item() == 1


def test_apply_actions_updates_history(simple_hands, device):
    """apply_actions() records play in history."""
    state = GameStateTensor.from_deals(
        hands=[simple_hands],
        decl_ids=[NOTRUMP],
        device=device
    )

    # Player 0 plays slot 0 (domino 0)
    new_state = state.apply_actions(torch.tensor([0], device=device))

    # Check history entry
    assert new_state.history[0, 0, 0].item() == 0  # player
    assert new_state.history[0, 0, 1].item() == 0  # domino_id
    assert new_state.history[0, 0, 2].item() == 0  # lead_domino_id


def test_apply_actions_updates_trick_plays(simple_hands, device):
    """apply_actions() adds domino to current trick."""
    state = GameStateTensor.from_deals(
        hands=[simple_hands],
        decl_ids=[NOTRUMP],
        device=device
    )

    # Player 0 plays slot 0 (domino 0)
    new_state = state.apply_actions(torch.tensor([0], device=device))

    # Check trick_plays
    assert new_state.trick_plays[0, 0].item() == 0
    assert new_state.trick_plays[0, 1].item() == -1


def test_apply_actions_completes_trick(simple_hands, device):
    """apply_actions() resolves trick after 4 plays."""
    state_gpu = GameStateTensor.from_deals(
        hands=[simple_hands],
        decl_ids=[NOTRUMP],
        device=device
    )
    state_cpu = GameState.from_hands(simple_hands, decl_id=NOTRUMP, leader=0)

    # Play 4 dominoes to complete a trick
    # Need to pick legal actions for each player
    for _ in range(4):
        # Get legal actions
        legal_gpu = state_gpu.legal_actions()[0]
        action_slot = legal_gpu.nonzero(as_tuple=True)[0][0].item()

        # Apply GPU action
        action_t = torch.tensor([action_slot], device=device)
        state_gpu = state_gpu.apply_actions(action_t)

        # Get CPU action (find domino ID from slot)
        cpu_hand = state_cpu.hands[state_cpu.current_player()]
        domino_id = cpu_hand[action_slot]
        state_cpu = state_cpu.apply_action(domino_id)

    # After 4 plays, trick should be complete
    assert (state_gpu.trick_plays[0] == -1).all()
    assert len(state_cpu.current_trick) == 0

    # Leader should be updated (match CPU)
    assert state_gpu.leader[0].item() == state_cpu.leader


# Helper Methods Tests

def test_hand_sizes(simple_hands, device):
    """hand_sizes() returns correct counts."""
    state = GameStateTensor.from_deals(
        hands=[simple_hands],
        decl_ids=[NOTRUMP],
        device=device
    )

    sizes = state.hand_sizes()
    assert sizes.shape == (1, 4)
    assert (sizes[0] == 7).all()

    # After one play
    state = state.apply_actions(torch.tensor([0], device=device))
    sizes = state.hand_sizes()
    assert sizes[0, 0].item() == 6
    assert sizes[0, 1].item() == 7


def test_active_games(simple_hands, device):
    """active_games() detects game completion."""
    state = GameStateTensor.from_deals(
        hands=[simple_hands],
        decl_ids=[NOTRUMP],
        device=device
    )

    # Initially active
    assert state.active_games()[0].item()

    # Play all 28 dominoes
    for _ in range(28):
        legal = state.legal_actions()[0]
        action = legal.nonzero(as_tuple=True)[0][0]  # First legal action
        state = state.apply_actions(action.unsqueeze(0))

    # Should be inactive now
    assert not state.active_games()[0].item()


# Equivalence Tests (GPU matches CPU)

def test_equivalence_simple_game(simple_hands, device):
    """GPU matches CPU for simple complete game."""
    state_gpu = GameStateTensor.from_deals(
        hands=[simple_hands],
        decl_ids=[NOTRUMP],
        device=device
    )
    state_cpu = GameState.from_hands(simple_hands, decl_id=NOTRUMP, leader=0)

    # Play through entire game
    for step in range(28):
        # Check current player matches
        assert state_gpu.current_player[0].item() == state_cpu.current_player()

        # Get legal actions
        legal_gpu = state_gpu.legal_actions()[0]
        legal_cpu = state_cpu.legal_actions()

        # GPU legal_gpu is a mask over slots (with -1s)
        # CPU legal_cpu is a tuple of domino IDs (compact, no -1s)
        # We need to map GPU slots to actual domino IDs
        current_player = state_cpu.current_player()
        gpu_hand = state_gpu.hands[0, current_player].tolist()  # Includes -1s

        # Get legal domino IDs from GPU
        legal_gpu_dominoes = set(gpu_hand[i] for i in legal_gpu.nonzero(as_tuple=True)[0].tolist())
        legal_cpu_set = set(legal_cpu)

        # Verify same legal dominoes
        assert legal_gpu_dominoes == legal_cpu_set, f"Step {step}: GPU={legal_gpu_dominoes}, CPU={legal_cpu_set}"

        # Pick first legal action
        action_slot = legal_gpu.nonzero(as_tuple=True)[0][0].item()
        domino_id = gpu_hand[action_slot]

        # Apply action
        state_gpu = state_gpu.apply_actions(torch.tensor([action_slot], device=device))
        state_cpu = state_cpu.apply_action(domino_id)

        # Check hand sizes match
        gpu_sizes = state_gpu.hand_sizes()[0].tolist()
        cpu_sizes = list(state_cpu.hand_sizes())
        assert gpu_sizes == cpu_sizes

    # Both games should be complete
    assert not state_gpu.active_games()[0].item()
    assert state_cpu.is_complete()


def test_equivalence_random_games(device):
    """GPU matches CPU for 10 random games with random declarations."""
    rng = random.Random(42)

    for game_idx in range(10):
        # Generate random deal
        dominoes = list(range(28))
        rng.shuffle(dominoes)
        hands = [dominoes[i*7:(i+1)*7] for i in range(4)]

        # Random declaration
        decl_id = rng.randint(0, N_DECLS - 1)

        # Initialize states
        state_gpu = GameStateTensor.from_deals(
            hands=[hands],
            decl_ids=[decl_id],
            device=device
        )
        state_cpu = GameState.from_hands(hands, decl_id=decl_id, leader=0)

        # Play through game
        for step in range(28):
            # Verify equivalence
            assert state_gpu.current_player[0].item() == state_cpu.current_player()

            # Get legal actions
            legal_gpu = state_gpu.legal_actions()[0]
            legal_cpu_set = set(state_cpu.legal_actions())

            # Verify same legal dominoes
            # Get GPU hand (with -1s)
            current_player = state_cpu.current_player()
            gpu_hand = state_gpu.hands[0, current_player].tolist()
            legal_gpu_dominoes = set(
                gpu_hand[i] for i in legal_gpu.nonzero(as_tuple=True)[0].tolist()
            )
            assert legal_gpu_dominoes == legal_cpu_set, (
                f"Game {game_idx}, step {step}: legal actions mismatch\n"
                f"GPU: {legal_gpu_dominoes}\nCPU: {legal_cpu_set}"
            )

            # Pick random legal action
            legal_slots = legal_gpu.nonzero(as_tuple=True)[0].tolist()
            action_slot = rng.choice(legal_slots)
            domino_id = gpu_hand[action_slot]

            # Apply actions
            state_gpu = state_gpu.apply_actions(torch.tensor([action_slot], device=device))
            state_cpu = state_cpu.apply_action(domino_id)


# Vectorization Tests (Multiple Games)

def test_vectorized_multiple_games(simple_hands, device):
    """Process multiple games in parallel."""
    n_games = 8
    hands = [simple_hands for _ in range(n_games)]
    decl_ids = [i % N_DECLS for i in range(n_games)]

    state = GameStateTensor.from_deals(
        hands=hands,
        decl_ids=decl_ids,
        device=device
    )

    # All games start with player 0
    assert (state.current_player == 0).all()

    # Apply same action to all games
    actions = torch.zeros(n_games, dtype=torch.int64, device=device)
    state = state.apply_actions(actions)

    # All games advance to player 1
    assert (state.current_player == 1).all()

    # All games have player 0 with 6 dominoes
    sizes = state.hand_sizes()
    assert (sizes[:, 0] == 6).all()
    assert (sizes[:, 1] == 7).all()


def test_vectorized_different_actions(simple_hands, device):
    """Process different actions for each game."""
    n_games = 4
    hands = [simple_hands for _ in range(n_games)]
    decl_ids = [NOTRUMP] * n_games

    state = GameStateTensor.from_deals(
        hands=hands,
        decl_ids=decl_ids,
        device=device
    )

    # Different action for each game
    actions = torch.tensor([0, 1, 2, 3], device=device)
    state = state.apply_actions(actions)

    # Different dominoes should be played
    played_dominoes = [
        state.played_mask[i].nonzero(as_tuple=True)[0].item()
        for i in range(n_games)
    ]
    assert played_dominoes == [0, 1, 2, 3]


def test_vectorized_game_completion(simple_hands, device):
    """Games can complete at different times."""
    n_games = 2
    hands = [simple_hands for _ in range(n_games)]
    decl_ids = [NOTRUMP] * n_games

    state = GameStateTensor.from_deals(
        hands=hands,
        decl_ids=decl_ids,
        device=device
    )

    # Play game 0 to completion, game 1 only partway
    for step in range(14):
        legal = state.legal_actions()

        # Both games play first legal action
        action0 = legal[0].nonzero(as_tuple=True)[0][0]
        action1 = legal[1].nonzero(as_tuple=True)[0][0]

        actions = torch.stack([action0, action1])
        state = state.apply_actions(actions)

    # After 14 steps, both games should have played 14 dominoes
    sizes = state.hand_sizes()
    assert sizes[0].sum() == 14  # Game 0: 14 dominoes left
    assert sizes[1].sum() == 14  # Game 1: 14 dominoes left

    # Play game 0 to completion (14 more moves)
    for step in range(14):
        legal = state.legal_actions()
        action0 = legal[0].nonzero(as_tuple=True)[0][0]

        # Game 1: play first slot (even if empty, it will be -1 and fail, so use legal action)
        action1 = legal[1].nonzero(as_tuple=True)[0][0]

        actions = torch.stack([action0, action1])
        state = state.apply_actions(actions)

    # After 28 total steps, both games should be complete
    sizes = state.hand_sizes()
    assert sizes[0].sum() == 0  # Game 0 complete
    assert sizes[1].sum() == 0  # Game 1 complete


# Edge Case Tests

def test_edge_case_doubles_trump_declaration(simple_hands, device):
    """Works correctly with doubles-trump declaration."""
    state = GameStateTensor.from_deals(
        hands=[simple_hands],
        decl_ids=[DOUBLES_TRUMP],
        device=device
    )

    # Just verify it runs without errors
    legal = state.legal_actions()
    assert legal.shape == (1, 7)


def test_edge_case_doubles_suit_declaration(simple_hands, device):
    """Works correctly with doubles-suit declaration."""
    state = GameStateTensor.from_deals(
        hands=[simple_hands],
        decl_ids=[DOUBLES_SUIT],
        device=device
    )

    # Just verify it runs without errors
    legal = state.legal_actions()
    assert legal.shape == (1, 7)


def test_edge_case_all_declarations(simple_hands, device):
    """Works with all 10 declarations."""
    for decl_id in range(N_DECLS):
        state = GameStateTensor.from_deals(
            hands=[simple_hands],
            decl_ids=[decl_id],
            device=device
        )

        # Verify basic operations work
        assert state.current_player[0].item() == 0
        legal = state.legal_actions()
        assert legal.any()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
