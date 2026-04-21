"""PyTorch Dataset over a joint-world corpus (.pt) for v0 belief training.

One item per decision. Belief-only signal for v0; v1 will extend to
(decision, world) pairs yielding Q targets.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset

from .features import (
    FEATURE_DIM,
    N_DOMINOES,
    extract_belief_target,
    extract_decision_features,
    reconstruct_prior_plays,
)


class JointWorldDecisionDataset(Dataset):
    """One sample per decision in the corpus. v0 yields (features, belief_target, belief_mask)."""

    def __init__(self, corpus_path: str | Path):
        blob = torch.load(str(corpus_path), weights_only=False)
        self.games = blob["results"]
        self.seeds = blob.get("seeds", [])

        # Flatten: build an index of (game_idx, decision_idx) pairs
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

        prior_plays = reconstruct_prior_plays(game.hands, game.decisions, d_idx)
        current_player = int(decision.player)

        features = extract_decision_features(
            game.hands,
            int(game.decl_id),
            prior_plays,
            current_player,
        )
        target, mask = extract_belief_target(
            game.hands,
            prior_plays,
            current_player,
        )

        return {
            "features": features,              # [FEATURE_DIM]
            "belief_target": target,           # [28] long in {0,1,2}
            "belief_mask": mask,               # [28] bool
            "decision_idx": torch.tensor(d_idx, dtype=torch.long),
            "player": torch.tensor(current_player, dtype=torch.long),
        }
