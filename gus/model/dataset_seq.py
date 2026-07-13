"""PyTorch Dataset over a joint-world corpus yielding tokenized sequences.

Used by the transformer-based v1 student. Sibling of dataset.py (flat MLP).
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset

from .features import extract_belief_target, reconstruct_prior_plays
from .tokenize import SEQ_LEN, tokenize_decision


class JointWorldSequenceDataset(Dataset):
    """One sample per decision. Yields tokenized sequence + belief target."""

    def __init__(self, corpus_path: str | Path):
        blob = torch.load(str(corpus_path), weights_only=False)
        self.games = blob["results"]
        self.seeds = blob.get("seeds", [])

        self.index: list[tuple[int, int]] = []
        for g, game in enumerate(self.games):
            for d in range(len(game.decisions)):
                self.index.append((g, d))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        g_idx, d_idx = self.index[idx]
        game = self.games[g_idx]
        decision = game.decisions[d_idx]
        current_player = int(decision.player)

        tokens, attn_mask = tokenize_decision(
            game.hands,
            int(game.decl_id),
            game.decisions,
            d_idx,
        )
        prior_plays = reconstruct_prior_plays(game.hands, game.decisions, d_idx)
        target, belief_mask = extract_belief_target(
            game.hands,
            prior_plays,
            current_player,
        )

        return {
            "tokens": tokens,                   # [SEQ_LEN, 5] long
            "attention_mask": attn_mask,        # [SEQ_LEN] bool
            "belief_target": target,            # [28] long in {0,1,2}
            "belief_mask": belief_mask,         # [28] bool
            "decision_idx": torch.tensor(d_idx, dtype=torch.long),
            "player": torch.tensor(current_player, dtype=torch.long),
        }
