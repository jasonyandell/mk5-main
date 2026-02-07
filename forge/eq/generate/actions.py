"""Action selection and decision recording for GPU E[Q] pipeline."""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from forge.eq.game_tensor import GameStateTensor
from forge.eq.types import ExplorationPolicy, PosteriorDiagnostics

from .types import DecisionRecordGPU, GameRecordGPU


def select_actions(
    states: GameStateTensor,
    e_q: Tensor,
    e_q_pdf: Tensor,
    greedy: bool,
    exploration_policy: ExplorationPolicy | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[Tensor, list | None]:
    """Select actions by probability of making the contract (p_make).

    Texas 42 has a threshold-based payoff: the bidding team (P0/P2) needs >= 30 points
    to make the contract. E[Q] = 0 is NOT neutral - it's a 21-21 split, which is a
    LOSS for offense. This function optimizes p_make = P(Q >= threshold | action)
    instead of E[Q], with E[Q] as tie-breaker.

    Win thresholds:
        - Offense (P0, P2): win when Q >= 18 (team scored >= 30) -> bin 60+
        - Defense (P1, P3): win when Q >= -17 (bidder scored <30) -> bin 25+

    Args:
        states: GameStateTensor
        e_q: [n_games, 7] E[Q] values (for tie-breaking)
        e_q_pdf: [n_games, 7, 85] P(Q=q|action) for q in [-42, +42], bin i -> Q = i - 42
        greedy: If True, argmax. If False, softmax sample.
        exploration_policy: Optional exploration policy (overrides greedy if provided)
        rng: NumPy RNG for exploration

    Returns:
        Tuple of (actions, exploration_stats):
            - actions: [n_games] action indices (0-6)
            - exploration_stats: List of ExplorationStats (one per game), or None if no exploration
    """
    from forge.eq.exploration import _select_action_with_exploration

    n_games = states.n_games

    # Get legal actions: [n_games, 7]
    legal_mask = states.legal_actions()

    # Compute p_make from PDF
    # Offense (bidder's team): need Q >= 18 -> bin 60+ (bin = Q + 42)
    # Defense (opponent team): need Q > -18, i.e., Q >= -17 -> bin 25+
    # Player is offense if they're on the same team as the bidder
    is_offense = ((states.current_player % 2) == (states.bidder % 2)).unsqueeze(1)  # [n_games, 1]

    p_make_offense = e_q_pdf[:, :, 60:].sum(dim=2)  # [n_games, 7]
    p_make_defense = e_q_pdf[:, :, 25:].sum(dim=2)  # [n_games, 7]
    p_make = torch.where(is_offense, p_make_offense, p_make_defense)  # [n_games, 7]

    # If exploration policy provided, use it (per-game selection)
    # Note: exploration still uses E[Q] for now (separate concern)
    if exploration_policy is not None:
        from forge.eq.types import ExplorationStats

        actions = []
        exploration_stats = []
        for g in range(n_games):
            action_idx, selection_mode, action_entropy = _select_action_with_exploration(
                e_q_mean=e_q[g].cpu(),  # Move to CPU for numpy conversion
                legal_mask=legal_mask[g].cpu(),
                policy=exploration_policy,
                rng=rng,
            )
            actions.append(action_idx)

            # Compute greedy action (by p_make) and q_gap
            legal = legal_mask[g]
            masked_p_make = p_make[g].clone()
            masked_p_make[~legal] = float('-inf')
            greedy_action = masked_p_make.argmax().item()

            q_greedy = e_q[g][greedy_action].item()
            q_taken = e_q[g][action_idx].item()
            q_gap = q_greedy - q_taken

            stats = ExplorationStats(
                greedy_action=greedy_action,
                action_taken=action_idx,
                was_greedy=(action_idx == greedy_action),
                selection_mode=selection_mode,
                q_gap=q_gap,
                action_entropy=action_entropy,
            )
            exploration_stats.append(stats)
        return torch.tensor(actions, dtype=torch.long, device=e_q.device), exploration_stats

    # Mask illegal actions
    p_make_masked = p_make.clone()
    p_make_masked[~legal_mask] = float('-inf')

    if greedy:
        # Tie-break by E[Q]: add tiny normalized E[Q] term
        # This ensures "lose gracefully" (smaller margin) or "win big" (larger margin)

        # Normalize E[Q] to [0, 1] using only LEGAL action values
        # Use extreme values for illegal so they don't affect min/max
        e_q_for_min = e_q.clone()
        e_q_for_min[~legal_mask] = float('inf')  # Won't be the min
        e_q_for_max = e_q.clone()
        e_q_for_max[~legal_mask] = float('-inf')  # Won't be the max

        e_q_min = e_q_for_min.min(dim=1, keepdim=True).values
        e_q_max = e_q_for_max.max(dim=1, keepdim=True).values
        e_q_range = (e_q_max - e_q_min).clamp(min=1e-10)

        # Normalize E[Q] to [0, 1], zero out illegal actions
        e_q_normalized = (e_q - e_q_min) / e_q_range
        e_q_normalized[~legal_mask] = 0.0  # Don't affect score for illegal

        # p_make is in [0, 1], tie-break term is negligible (1e-6 scale)
        # Note: 1e-9 is too small for float32 precision, gets absorbed
        score = p_make_masked + 1e-6 * e_q_normalized
        actions = score.argmax(dim=1)
    else:
        # Softmax sample over p_make
        probs = torch.softmax(p_make_masked, dim=1)
        actions = torch.multinomial(probs, num_samples=1).squeeze(1)

    return actions, None


def record_decisions(
    states: GameStateTensor,
    e_q: Tensor,
    e_q_var: Tensor,
    e_q_pdf: Tensor,
    actions: Tensor,
    all_decisions: list[list[DecisionRecordGPU]],
    diagnostics: PosteriorDiagnostics | None = None,
    exploration_stats: list | None = None,
    n_samples_used: int | None = None,
    did_converge: bool | None = None,
):
    """Record decisions for each game (in-place).

    Args:
        states: GameStateTensor
        e_q: [n_games, 7] E[Q] mean values
        e_q_var: [n_games, 7] E[Q] variance values
        e_q_pdf: [n_games, 7, 85] full PDF P(Q=q|action) for q in [-42, +42]
        actions: [n_games] action indices
        all_decisions: List of decision lists (one per game)
        diagnostics: Optional posterior diagnostics (aggregated across games)
        exploration_stats: Optional exploration stats (one per game)
        n_samples_used: Optional number of samples used (for adaptive mode)
        did_converge: Optional convergence status (for adaptive mode)
    """
    n_games = states.n_games
    legal_mask = states.legal_actions()
    current_players = states.current_player

    # Mode mapping for exploration (CPU pipeline convention)
    mode_to_int = {"greedy": 0, "boltzmann": 1, "epsilon": 2, "blunder": 3}

    for g in range(n_games):
        # Only record if game is active
        if not states.active_games()[g]:
            continue

        # Compute state uncertainty from variance
        sigma = torch.sqrt(e_q_var[g])  # [7]
        legal = legal_mask[g]
        if legal.any():
            u_mean = sigma[legal].mean().item()
            u_max = sigma[legal].max().item()
        else:
            u_mean = 0.0
            u_max = 0.0

        # Extract exploration stats if available
        exploration_mode = None
        q_gap = None
        greedy_action = None
        if exploration_stats is not None and g < len(exploration_stats):
            stats = exploration_stats[g]
            exploration_mode = mode_to_int.get(stats.selection_mode, 0)
            q_gap = stats.q_gap
            greedy_action = stats.greedy_action

        record = DecisionRecordGPU(
            player=current_players[g].item(),
            e_q=e_q[g].cpu(),
            action_taken=actions[g].item(),
            legal_mask=legal_mask[g].cpu(),
            e_q_var=e_q_var[g].cpu(),
            e_q_pdf=e_q_pdf[g].cpu(),
            u_mean=u_mean,
            u_max=u_max,
            ess=diagnostics.ess if diagnostics else None,
            max_w=diagnostics.max_w if diagnostics else None,
            exploration_mode=exploration_mode,
            q_gap=q_gap,
            greedy_action=greedy_action,
            n_samples=n_samples_used,
            converged=did_converge,
        )
        all_decisions[g].append(record)


def collate_records(
    hands: list[list[list[int]]],
    decl_ids: list[int],
    all_decisions: list[list[DecisionRecordGPU]],
) -> list[GameRecordGPU]:
    """Collate decision records into GameRecordGPU.

    Args:
        hands: Initial deals
        decl_ids: Declaration IDs
        all_decisions: Decision records per game

    Returns:
        List of GameRecordGPU
    """
    return [
        GameRecordGPU(
            decisions=decisions,
            hands=hands[g],
            decl_id=decl_ids[g],
        )
        for g, decisions in enumerate(all_decisions)
    ]
