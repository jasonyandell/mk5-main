"""Tests for GPU game record collation and transcript reconstruction.

Tests use minimal inputs to keep runtime fast.
"""

from __future__ import annotations

import pytest
import torch

from forge.eq.collate import collate_game_record, collate_batch
from forge.eq.generate_gpu import GameRecordGPU, DecisionRecordGPU
from forge.eq.transcript_tokenize import (
    N_FEATURES,
    MAX_TOKENS,
    FEAT_PLAYER,
    FEAT_IS_IN_HAND,
    FEAT_TOKEN_TYPE,
    TOKEN_TYPE_DECL,
    TOKEN_TYPE_HAND,
    TOKEN_TYPE_PLAY,
)


def test_collate_single_decision():
    """Test collating a single decision from GPU record."""
    # Simple game: one player makes one play
    hands = [
        [0, 1, 2, -1, -1, -1, -1],  # Player 0: dominoes 0, 1, 2
        [3, 4, 5, -1, -1, -1, -1],  # Player 1
        [6, 7, 8, -1, -1, -1, -1],  # Player 2
        [9, 10, 11, -1, -1, -1, -1],  # Player 3
    ]
    decl_id = 5  # Fives

    # Player 0 plays slot 0 (domino 0)
    decision = DecisionRecordGPU(
        player=0,
        e_q=torch.tensor([10.0, 5.0, 3.0, -float('inf'), -float('inf'), -float('inf'), -float('inf')]),
        action_taken=0,  # Slot 0
        legal_mask=torch.tensor([True, True, True, False, False, False, False]),
    )

    game_record = GameRecordGPU(
        decisions=[decision],
        hands=hands,
        decl_id=decl_id,
    )

    # Collate
    examples = collate_game_record(game_record)

    # Should have 1 example
    assert len(examples) == 1
    ex = examples[0]

    # Check fields
    assert ex['player'] == 0
    assert ex['action_taken'] == 0
    assert ex['decision_idx'] == 0
    assert torch.allclose(ex['e_q_mean'], decision.e_q)
    assert torch.equal(ex['legal_mask'], decision.legal_mask)

    # Check transcript tokens
    tokens = ex['transcript_tokens']
    assert tokens.shape[1] == N_FEATURES
    assert ex['transcript_length'] == tokens.shape[0]

    # Should have: 1 decl + 3 hand tokens (before any plays) = 4 tokens
    assert tokens.shape[0] == 4

    # First token is declaration
    assert tokens[0, FEAT_TOKEN_TYPE] == TOKEN_TYPE_DECL

    # Next 3 tokens are hand
    assert tokens[1, FEAT_TOKEN_TYPE] == TOKEN_TYPE_HAND
    assert tokens[1, FEAT_PLAYER] == 0  # Current player
    assert tokens[1, FEAT_IS_IN_HAND] == 1


def test_collate_two_decisions():
    """Test collating two decisions - first play appears in second transcript."""
    hands = [
        [0, 1, 2, -1, -1, -1, -1],
        [3, 4, 5, -1, -1, -1, -1],
        [6, 7, 8, -1, -1, -1, -1],
        [9, 10, 11, -1, -1, -1, -1],
    ]
    decl_id = 5

    # Player 0 plays slot 0 (domino 0)
    decision1 = DecisionRecordGPU(
        player=0,
        e_q=torch.tensor([10.0, 5.0, 3.0, -float('inf'), -float('inf'), -float('inf'), -float('inf')]),
        action_taken=0,
        legal_mask=torch.tensor([True, True, True, False, False, False, False]),
    )

    # Player 1 plays slot 1 (domino 4)
    decision2 = DecisionRecordGPU(
        player=1,
        e_q=torch.tensor([8.0, 6.0, 4.0, -float('inf'), -float('inf'), -float('inf'), -float('inf')]),
        action_taken=1,
        legal_mask=torch.tensor([True, True, True, False, False, False, False]),
    )

    game_record = GameRecordGPU(
        decisions=[decision1, decision2],
        hands=hands,
        decl_id=decl_id,
    )

    # Collate
    examples = collate_game_record(game_record)

    # Should have 2 examples
    assert len(examples) == 2

    # First example: no plays yet
    ex1 = examples[0]
    assert ex1['player'] == 0
    assert ex1['decision_idx'] == 0
    tokens1 = ex1['transcript_tokens']
    # 1 decl + 3 hand tokens = 4 tokens
    assert tokens1.shape[0] == 4

    # Second example: should include first play
    ex2 = examples[1]
    assert ex2['player'] == 1
    assert ex2['decision_idx'] == 1
    tokens2 = ex2['transcript_tokens']
    # 1 decl + 3 hand tokens + 1 play token = 5 tokens
    assert tokens2.shape[0] == 5

    # Last token should be a play
    assert tokens2[-1, FEAT_TOKEN_TYPE] == TOKEN_TYPE_PLAY
    # Played by player 0, seen from player 1's perspective
    # Relative player = (0 - 1 + 4) % 4 = 3
    assert tokens2[-1, FEAT_PLAYER] == 3
    assert tokens2[-1, FEAT_IS_IN_HAND] == 0


def test_collate_batch_single_game():
    """Test batching a single game."""
    hands = [[0, 1, -1, -1, -1, -1, -1] for _ in range(4)]
    decl_id = 0

    decision = DecisionRecordGPU(
        player=0,
        e_q=torch.tensor([5.0, 3.0, -float('inf'), -float('inf'), -float('inf'), -float('inf'), -float('inf')]),
        action_taken=0,
        legal_mask=torch.tensor([True, True, False, False, False, False, False]),
    )

    game_record = GameRecordGPU(
        decisions=[decision],
        hands=hands,
        decl_id=decl_id,
    )

    # Collate batch
    batch = collate_batch([game_record], game_indices=[42], is_val=[False])

    # Check shapes
    assert batch['transcript_tokens'].shape == (1, MAX_TOKENS, N_FEATURES)
    assert batch['transcript_lengths'].shape == (1,)
    assert batch['e_q_mean'].shape == (1, 7)
    assert batch['legal_mask'].shape == (1, 7)
    assert batch['action_taken'].shape == (1,)
    assert batch['game_idx'].shape == (1,)
    assert batch['train_mask'].shape == (1,)

    # Check values
    assert batch['game_idx'][0] == 42
    assert batch['train_mask'][0] == True
    assert batch['action_taken'][0] == 0
    assert batch['decision_idx'][0] == 0

    # Check transcript length (1 decl + 2 hand = 3)
    assert batch['transcript_lengths'][0] == 3


def test_collate_batch_multiple_games():
    """Test batching multiple games."""
    # Create two simple games with one decision each
    games = []
    for g in range(2):
        hands = [[g * 4 + p, -1, -1, -1, -1, -1, -1] for p in range(4)]
        decision = DecisionRecordGPU(
            player=0,
            e_q=torch.tensor([float(g), -float('inf'), -float('inf'), -float('inf'), -float('inf'), -float('inf'), -float('inf')]),
            action_taken=0,
            legal_mask=torch.tensor([True, False, False, False, False, False, False]),
        )
        games.append(GameRecordGPU(decisions=[decision], hands=hands, decl_id=0))

    # Collate batch
    batch = collate_batch(games, game_indices=[10, 20], is_val=[False, True])

    # Should have 2 examples total (1 per game)
    assert batch['transcript_tokens'].shape[0] == 2
    assert batch['game_idx'][0] == 10
    assert batch['game_idx'][1] == 20
    assert batch['train_mask'][0] == True
    assert batch['train_mask'][1] == False


def test_hand_tracking():
    """Test that hands are correctly tracked as dominoes are played."""
    # Game with 3 decisions from same player
    hands = [
        [0, 1, 2, 3, -1, -1, -1],  # Player 0
        [4, 5, -1, -1, -1, -1, -1],  # Others
        [6, 7, -1, -1, -1, -1, -1],
        [8, 9, -1, -1, -1, -1, -1],
    ]
    decl_id = 0

    # Player 0 makes 3 plays: slots 0, 2, 1
    decisions = [
        DecisionRecordGPU(
            player=0,
            e_q=torch.zeros(7),
            action_taken=0,  # Play slot 0 (domino 0)
            legal_mask=torch.tensor([True, True, True, True, False, False, False]),
        ),
        DecisionRecordGPU(
            player=0,
            e_q=torch.zeros(7),
            action_taken=2,  # Play slot 2 (domino 2)
            legal_mask=torch.tensor([False, True, True, True, False, False, False]),
        ),
        DecisionRecordGPU(
            player=0,
            e_q=torch.zeros(7),
            action_taken=1,  # Play slot 1 (domino 1)
            legal_mask=torch.tensor([False, True, False, True, False, False, False]),
        ),
    ]

    game_record = GameRecordGPU(decisions=decisions, hands=hands, decl_id=decl_id)
    examples = collate_game_record(game_record)

    # Decision 0: hand has all 4 dominoes
    assert examples[0]['transcript_length'] == 1 + 4  # decl + 4 hand

    # Decision 1: hand has 3 dominoes (played domino 0)
    # Plus 1 play token
    assert examples[1]['transcript_length'] == 1 + 3 + 1  # decl + 3 hand + 1 play

    # Decision 2: hand has 2 dominoes (played dominoes 0, 2)
    # Plus 2 play tokens
    assert examples[2]['transcript_length'] == 1 + 2 + 2  # decl + 2 hand + 2 plays


def test_optional_fields():
    """Test that optional diagnostic fields are handled correctly."""
    hands = [[0, -1, -1, -1, -1, -1, -1] for _ in range(4)]
    decl_id = 0

    # Decision without optional fields
    decision = DecisionRecordGPU(
        player=0,
        e_q=torch.zeros(7),
        action_taken=0,
        legal_mask=torch.tensor([True, False, False, False, False, False, False]),
    )

    game_record = GameRecordGPU(decisions=[decision], hands=hands, decl_id=decl_id)
    examples = collate_game_record(game_record)

    ex = examples[0]
    # Should have default values
    assert ex['u_mean'] == 0.0
    assert ex['u_max'] == 0.0
    assert ex['ess'] is None
    assert ex['max_w'] is None

    # Batch should convert None to 0.0
    batch = collate_batch([game_record])
    assert batch['ess'][0] == 0.0
    assert batch['max_w'][0] == 0.0


@pytest.mark.slow
def test_integration_with_gpu_pipeline():
    """Integration test: run GPU pipeline and collate output.

    This test requires a trained Stage 1 model and may be slow.
    """
    pytest.importorskip("forge.eq.generate_gpu")
    pytest.importorskip("forge.eq.oracle")

    try:
        from forge.eq.oracle import Stage1Oracle
        from forge.oracle.rng import deal_from_seed
        from forge.eq.generate_gpu import generate_eq_games_gpu
    except ImportError:
        pytest.skip("Required modules not available")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Try to load oracle
    checkpoint_paths = [
        "checkpoints/stage1/best.ckpt",
        "forge/models/domino-qval-large-3.3M-qgap0.071-qmae0.94.ckpt",
        "forge/models/domino-large-817k-valuehead-acc97.8-qgap0.07.ckpt",
    ]

    oracle = None
    for checkpoint_path in checkpoint_paths:
        try:
            oracle = Stage1Oracle(checkpoint_path, device=device, compile=False)
            break
        except Exception:
            continue

    if oracle is None:
        pytest.skip("Could not load oracle model")

    # Generate a small game
    hands = deal_from_seed(42)
    decl_id = 5

    game_records = generate_eq_games_gpu(
        model=oracle.model,
        hands=[hands],
        decl_ids=[decl_id],
        n_samples=10,  # Small for speed
        device=device,
        greedy=True,
    )

    # Collate the output
    batch = collate_batch(game_records, game_indices=[0], is_val=[False])

    # Should have 28 decisions (complete game)
    assert batch['transcript_tokens'].shape[0] == 28
    assert batch['e_q_mean'].shape == (28, 7)
    assert batch['legal_mask'].shape == (28, 7)

    # All transcript lengths should be valid (between 1 and MAX_TOKENS)
    assert (batch['transcript_lengths'] > 0).all()
    assert (batch['transcript_lengths'] <= MAX_TOKENS).all()

    # Transcript lengths should generally increase (more plays accumulate)
    lengths = batch['transcript_lengths'].cpu().numpy()
    assert lengths[-1] > lengths[0], "Transcript should grow as game progresses"

    # All game indices should be 0
    assert (batch['game_idx'] == 0).all()

    # All should be training examples
    assert batch['train_mask'].all()
