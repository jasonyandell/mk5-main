"""Exploration (stochastic action selection) helpers for E[Q] generation."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from forge.eq.types import ExplorationPolicy


def _spawn_child_rng(parent: np.random.Generator) -> np.random.Generator:
    """Create a deterministic child RNG stream from a parent RNG.

    Prefers BitGenerator.jump() when available to avoid consuming parent entropy.
    """
    bitgen = parent.bit_generator
    jumped = getattr(bitgen, "jumped", None)
    if callable(jumped):
        return np.random.Generator(bitgen.jumped())

    # Fallback: consume one uint32 from the parent.
    seed_u32 = int(parent.integers(0, 2**32, dtype=np.uint32))
    return np.random.default_rng(seed_u32)


def _select_action_with_exploration(
    e_q_mean: Tensor,
    legal_mask: Tensor,
    policy: ExplorationPolicy | None,
    rng: np.random.Generator | None,
) -> tuple[int, str, float]:
    """Select action index given E[Q] values and legal mask.

    Returns:
        (action_idx, selection_mode, action_entropy)
    """
    legal_indices = legal_mask.nonzero(as_tuple=False).squeeze(-1).tolist()
    n_legal = len(legal_indices)
    if n_legal == 0:
        # Should never happen; be defensive.
        return 0, "greedy", 0.0

    legal_q = e_q_mean[legal_mask]
    p = F.softmax(legal_q, dim=0)
    action_entropy = float(-(p * torch.log(p + 1e-30)).sum().item())

    # Greedy action (always compute as baseline)
    greedy_local = legal_q.argmax().item()
    greedy_idx = legal_indices[greedy_local]
    greedy_q = legal_q[greedy_local].item()

    # No policy = greedy
    if policy is None or rng is None:
        return greedy_idx, "greedy", action_entropy

    # Check for blunder (occasional suboptimal pick with bounded regret)
    if policy.blunder_rate > 0 and rng.random() < policy.blunder_rate:
        regret_threshold = greedy_q - policy.blunder_max_regret
        blunder_candidates = [
            (i, local_i)
            for local_i, i in enumerate(legal_indices)
            if legal_q[local_i].item() >= regret_threshold and i != greedy_idx
        ]
        if blunder_candidates:
            idx, _ = blunder_candidates[int(rng.integers(len(blunder_candidates)))]
            return idx, "blunder", action_entropy

    # Check for epsilon-random
    if policy.epsilon > 0 and rng.random() < policy.epsilon:
        random_idx = legal_indices[int(rng.integers(n_legal))]
        return random_idx, "epsilon", action_entropy

    # Boltzmann sampling or greedy
    if policy.use_boltzmann:
        boltzmann_probs = F.softmax(legal_q / policy.temperature, dim=0).numpy()
        sampled_local = int(rng.choice(n_legal, p=boltzmann_probs))
        sampled_idx = legal_indices[sampled_local]
        return sampled_idx, "boltzmann", action_entropy

    return greedy_idx, "greedy", action_entropy
