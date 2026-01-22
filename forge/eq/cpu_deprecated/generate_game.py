"""Single-game E[Q] generation."""

from __future__ import annotations

import numpy as np
import torch

from forge.eq.exploration import _spawn_child_rng, _select_action_with_exploration
from forge.eq.game import GameState
from forge.eq.oracle import Stage1Oracle
from forge.eq.outcomes import _fill_actual_outcomes
from forge.eq.posterior import compute_posterior_weights
from forge.eq.reduction import _reduce_world_q_values
from forge.eq.sampling import sample_consistent_worlds
from forge.eq.transcript_tokenize import tokenize_transcript
from forge.eq.types import (
    DecisionRecord,
    DecisionRecordV2,
    ExplorationPolicy,
    ExplorationStats,
    GameExplorationStats,
    GameRecord,
    GameRecordV2,
    PosteriorConfig,
    PosteriorDiagnostics,
)
from forge.eq.worlds import _build_hypothetical_worlds_batched, _build_legal_mask
from forge.oracle.tables import can_follow, led_suit_for_lead_domino


def generate_eq_game(
    oracle: Stage1Oracle,
    hands: list[list[int]],
    decl_id: int,
    n_samples: int = 100,
    posterior_config: PosteriorConfig | None = None,
    exploration_policy: ExplorationPolicy | None = None,
    world_rng: np.random.Generator | None = None,
) -> GameRecord:
    """Play one game, record all 28 decisions."""
    use_posterior = posterior_config is not None and posterior_config.enabled
    use_exploration = exploration_policy is not None
    use_v2 = use_posterior or use_exploration

    game = GameState.from_hands(hands, decl_id, leader=0)
    decisions: list[DecisionRecord] = []

    if use_exploration:
        if exploration_policy.seed is not None:
            rng = np.random.default_rng(exploration_policy.seed)
        elif world_rng is not None:
            rng = _spawn_child_rng(world_rng)
        else:
            rng = np.random.default_rng()
    else:
        rng = None

    game_exploration_stats = GameExplorationStats() if use_exploration else None
    total_entropy = 0.0

    voids: dict[int, set[int]] = {0: set(), 1: set(), 2: set(), 3: set()}
    plays_processed = 0
    played_by: dict[int, list[int]] = {0: [], 1: [], 2: [], 3: []}

    while not game.is_complete():
        player = game.current_player()
        my_hand = list(game.hands[player])

        for play_idx in range(plays_processed, len(game.play_history)):
            play_player, domino_id, lead_domino_id = game.play_history[play_idx]
            played_by[play_player].append(domino_id)
            led_suit = led_suit_for_lead_domino(lead_domino_id, decl_id)
            if not can_follow(domino_id, led_suit, decl_id):
                voids[play_player].add(led_suit)
        plays_processed = len(game.play_history)

        remaining_worlds = sample_consistent_worlds(
            my_player=player,
            my_hand=my_hand,
            played=game.played,
            hand_sizes=game.hand_sizes(),
            voids=voids,
            decl_id=decl_id,
            n_samples=n_samples,
            rng=world_rng,
        )

        hypothetical_deals, game_state_info = _build_hypothetical_worlds_batched(
            game, remaining_worlds, played_by, player
        )

        n_worlds = len(hypothetical_deals)
        all_logits = oracle.query_batch(hypothetical_deals, game_state_info, player)

        diagnostics: PosteriorDiagnostics | None = None
        if use_posterior:
            weights, diagnostics = compute_posterior_weights(
                oracle=oracle,
                hypothetical_deals=hypothetical_deals,
                play_history=list(game.play_history),
                decl_id=decl_id,
                config=posterior_config,
                rng=world_rng,
            )
        else:
            weights = torch.ones(n_worlds, device=oracle.device) / n_worlds

        e_q_mean, e_q_var = _reduce_world_q_values(
            all_q_values=all_logits,
            weights=weights,
            hypothetical_deals=hypothetical_deals,
            player=player,
            my_hand=my_hand,
        )

        plays_for_transcript = [(p, d) for p, d, _ in game.play_history]
        transcript_tokens = tokenize_transcript(my_hand, plays_for_transcript, decl_id, player)

        legal_actions = game.legal_actions()
        legal_mask = _build_legal_mask(legal_actions, my_hand)

        masked_q = e_q_mean.clone()
        masked_q[~legal_mask] = float("-inf")
        greedy_idx = masked_q.argmax().item()

        action_idx, selection_mode, action_entropy = _select_action_with_exploration(
            e_q_mean=e_q_mean,
            legal_mask=legal_mask,
            policy=exploration_policy,
            rng=rng,
        )
        action_domino = my_hand[action_idx]

        exploration_stats: ExplorationStats | None = None
        if use_exploration:
            q_gap = (e_q_mean[greedy_idx] - e_q_mean[action_idx]).item()
            exploration_stats = ExplorationStats(
                greedy_action=greedy_idx,
                action_taken=action_idx,
                was_greedy=action_idx == greedy_idx,
                selection_mode=selection_mode,
                q_gap=q_gap,
                action_entropy=action_entropy,
            )

            if game_exploration_stats is not None:
                game_exploration_stats.n_decisions += 1
                if selection_mode == "greedy":
                    game_exploration_stats.n_greedy += 1
                elif selection_mode == "boltzmann":
                    game_exploration_stats.n_boltzmann += 1
                elif selection_mode == "epsilon":
                    game_exploration_stats.n_epsilon += 1
                elif selection_mode == "blunder":
                    game_exploration_stats.n_blunder += 1
                game_exploration_stats.total_q_gap += q_gap
                total_entropy += action_entropy

        padded_e_q_mean = torch.full((7,), float("-inf"))
        padded_e_q_mean[: len(my_hand)] = e_q_mean
        padded_legal_mask = torch.zeros(7, dtype=torch.bool)
        padded_legal_mask[: len(my_hand)] = legal_mask

        padded_e_q_var = torch.zeros(7)
        padded_e_q_var[: len(my_hand)] = e_q_var

        legal_std = torch.sqrt(e_q_var)
        legal_std_for_u = legal_std[legal_mask]
        if len(legal_std_for_u) > 0:
            u_mean = legal_std_for_u.mean().item()
            u_max = legal_std_for_u.max().item()
        else:
            u_mean = 0.0
            u_max = 0.0

        if use_v2:
            decisions.append(
                DecisionRecordV2(
                    transcript_tokens=transcript_tokens,
                    e_q_mean=padded_e_q_mean,
                    legal_mask=padded_legal_mask,
                    action_taken=action_idx,
                    player=player,
                    diagnostics=diagnostics,
                    exploration=exploration_stats,
                    e_q_var=padded_e_q_var,
                    u_mean=u_mean,
                    u_max=u_max,
                )
            )
        else:
            decisions.append(
                DecisionRecord(
                    transcript_tokens=transcript_tokens,
                    e_q_mean=padded_e_q_mean,
                    legal_mask=padded_legal_mask,
                    action_taken=action_idx,
                    player=player,
                )
            )

        game = game.apply_action(action_domino)

    if use_exploration and game_exploration_stats is not None and game_exploration_stats.n_decisions > 0:
        game_exploration_stats.mean_action_entropy = total_entropy / game_exploration_stats.n_decisions

    _fill_actual_outcomes(decisions, game, decl_id)

    if use_v2:
        return GameRecordV2(
            decisions=decisions,
            posterior_config=posterior_config,
            exploration_policy=exploration_policy,
            exploration_stats=game_exploration_stats,
        )
    return GameRecord(decisions=decisions)
