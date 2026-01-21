"""Post-game transcript reconstruction for GPU pipeline.

Reconstructs play history from GPU game records and tokenizes
for Stage 2 training data.
"""

from __future__ import annotations

import torch
from torch import Tensor

from forge.eq.generate_gpu import GameRecordGPU, DecisionRecordGPU
from forge.eq.transcript_tokenize import tokenize_transcript, MAX_TOKENS, N_FEATURES


def collate_game_record(
    game_record: GameRecordGPU,
) -> list[dict]:
    """Convert GPU game record to training examples with transcript tokens.

    Reconstructs the play history from the action sequence and tokenizes
    each decision point for Stage 2 training.

    Args:
        game_record: Output from generate_eq_games_gpu()

    Returns:
        List of dicts, one per decision, containing:
        - transcript_tokens: Tensor from tokenize_transcript()
        - transcript_length: int - actual sequence length before padding
        - e_q_mean: Tensor[7]
        - e_q_var: Tensor[7] or None
        - legal_mask: Tensor[7]
        - action_taken: int
        - player: int
        - decision_idx: int
        - Plus any diagnostics fields
    """
    hands = game_record.hands  # list[list[int]] - 4 players × 7 dominoes
    decl_id = game_record.decl_id
    decisions = game_record.decisions

    # Track hand state for each player (copy to avoid mutation)
    current_hands = [list(h) for h in hands]

    # Track plays so far
    plays: list[tuple[int, int]] = []  # (player, domino_id)

    examples = []

    for decision_idx, decision in enumerate(decisions):
        player = decision.player
        slot = decision.action_taken  # Slot index (0-6)

        # Get current player's hand (remaining dominoes)
        my_hand = [d for d in current_hands[player] if d >= 0]

        # Tokenize the transcript at this decision point
        transcript_tokens = tokenize_transcript(
            my_hand=my_hand,
            plays=plays,
            decl_id=decl_id,
            current_player=player,
        )

        # Build example dict
        example = {
            'transcript_tokens': transcript_tokens,
            'transcript_length': transcript_tokens.shape[0],
            'e_q_mean': decision.e_q,
            'e_q_var': getattr(decision, 'e_q_var', None),
            'legal_mask': decision.legal_mask,
            'action_taken': decision.action_taken,
            'player': player,
            'decision_idx': decision_idx,
            # Optional diagnostics
            'u_mean': getattr(decision, 'u_mean', 0.0),
            'u_max': getattr(decision, 'u_max', 0.0),
            'ess': getattr(decision, 'ess', None),
            'max_w': getattr(decision, 'max_w', None),
            'exploration_mode': getattr(decision, 'exploration_mode', None),
            'q_gap': getattr(decision, 'q_gap', None),
        }
        examples.append(example)

        # Update state for next decision
        # 1. Record the play
        domino_id = current_hands[player][slot]
        plays.append((player, domino_id))

        # 2. Remove domino from hand (mark as -1)
        current_hands[player][slot] = -1

    return examples


def collate_batch(
    game_records: list[GameRecordGPU],
    game_indices: list[int] | None = None,
    is_val: list[bool] | None = None,
) -> dict:
    """Collate multiple games into batched tensors for saving.

    Args:
        game_records: List of GPU game records
        game_indices: Optional game index for each record
        is_val: Optional validation flag for each game

    Returns:
        Dict with batched tensors matching CPU output schema:
        - transcript_tokens: [n_examples, MAX_TOKENS, N_FEATURES]
        - transcript_lengths: [n_examples] actual lengths before padding
        - e_q_mean: [n_examples, 7]
        - e_q_var: [n_examples, 7] (zeros if not available)
        - legal_mask: [n_examples, 7]
        - action_taken: [n_examples]
        - game_idx: [n_examples]
        - decision_idx: [n_examples]
        - train_mask: [n_examples] bool
        - u_mean: [n_examples]
        - u_max: [n_examples]
        - ess: [n_examples]
        - max_w: [n_examples]
        - exploration_mode: [n_examples]
        - q_gap: [n_examples]
    """
    all_examples = []

    for g_idx, game_record in enumerate(game_records):
        game_idx = game_indices[g_idx] if game_indices else g_idx
        val_flag = is_val[g_idx] if is_val else False

        examples = collate_game_record(game_record)
        for ex in examples:
            ex['game_idx'] = game_idx
            ex['train_mask'] = not val_flag
            all_examples.append(ex)

    # Stack into tensors
    n_examples = len(all_examples)

    # Pad transcript tokens to MAX_TOKENS
    padded_tokens = []
    for ex in all_examples:
        tokens = ex['transcript_tokens']
        seq_len = tokens.shape[0]
        if seq_len < MAX_TOKENS:
            padding = torch.zeros((MAX_TOKENS - seq_len, N_FEATURES), dtype=tokens.dtype)
            tokens = torch.cat([tokens, padding], dim=0)
        padded_tokens.append(tokens)

    return {
        'transcript_tokens': torch.stack(padded_tokens),
        'transcript_lengths': torch.tensor([ex['transcript_length'] for ex in all_examples], dtype=torch.long),
        'e_q_mean': torch.stack([ex['e_q_mean'] for ex in all_examples]),
        'e_q_var': torch.stack([ex['e_q_var'] if ex['e_q_var'] is not None else torch.zeros(7) for ex in all_examples]),
        'legal_mask': torch.stack([ex['legal_mask'] for ex in all_examples]),
        'action_taken': torch.tensor([ex['action_taken'] for ex in all_examples], dtype=torch.long),
        'game_idx': torch.tensor([ex['game_idx'] for ex in all_examples], dtype=torch.long),
        'decision_idx': torch.tensor([ex['decision_idx'] for ex in all_examples], dtype=torch.long),
        'train_mask': torch.tensor([ex['train_mask'] for ex in all_examples], dtype=torch.bool),
        'u_mean': torch.tensor([ex['u_mean'] for ex in all_examples], dtype=torch.float32),
        'u_max': torch.tensor([ex['u_max'] for ex in all_examples], dtype=torch.float32),
        'ess': torch.tensor([ex['ess'] if ex['ess'] is not None else 0.0 for ex in all_examples], dtype=torch.float32),
        'max_w': torch.tensor([ex['max_w'] if ex['max_w'] is not None else 0.0 for ex in all_examples], dtype=torch.float32),
        'exploration_mode': torch.tensor([ex['exploration_mode'] if ex['exploration_mode'] is not None else 0 for ex in all_examples], dtype=torch.int8),
        'q_gap': torch.tensor([ex['q_gap'] if ex['q_gap'] is not None else 0.0 for ex in all_examples], dtype=torch.float32),
    }
