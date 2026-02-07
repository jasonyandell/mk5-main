from __future__ import annotations
"""
======================================================================
DEPRECATED CPU PIPELINE - DO NOT USE
======================================================================
This module contains KNOWN BUGS (E[Q] collapse with high sample counts).
It is kept temporarily for reference only and will be deleted soon.

Use the GPU pipeline instead: forge/eq/generate_gpu.py
======================================================================
"""
import sys as _sys
if not _sys.flags.interactive:  # Allow interactive inspection
    raise RuntimeError(
        "\n" + "=" * 70 + "\n"
        "DEPRECATED CPU PIPELINE - DO NOT USE\n"
        + "=" * 70 + "\n"
        "This module contains KNOWN BUGS (E[Q] collapse with high sample counts).\n"
        "It is kept temporarily for reference only and will be deleted soon.\n"
        "\n"
        "Use the GPU pipeline instead: forge/eq/generate_gpu.py\n"
        + "=" * 70
    )
del _sys

"""Cross-game batched E[Q] generation.

This batches oracle calls across multiple concurrently-generated games, grouped by decl_id,
to reduce GPU kernel launch overhead and improve throughput.
"""


from collections import defaultdict

import numpy as np
import torch
from torch import Tensor

from forge.eq.exploration import _spawn_child_rng, _select_action_with_exploration
from forge.eq.game import GameState
from forge.eq.oracle import Stage1Oracle
from forge.eq.outcomes import _fill_actual_outcomes
from forge.eq.posterior import compute_posterior_weights_many
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


def generate_eq_games_batched(
    oracle: Stage1Oracle,
    hands_list: list[list[list[int]]],
    decl_ids: list[int],
    *,
    n_samples: int = 100,
    posterior_config: PosteriorConfig | None = None,
    exploration_policies: list[ExplorationPolicy | None] | None = None,
    world_rngs: list[np.random.Generator | None] | None = None,
) -> list[GameRecord]:
    """Generate multiple games while batching oracle calls across games (by decl_id)."""
    if len(hands_list) != len(decl_ids):
        raise ValueError(
            f"hands_list ({len(hands_list)}) and decl_ids ({len(decl_ids)}) must have same length"
        )
    n_games = len(hands_list)
    if n_games == 0:
        return []

    if exploration_policies is None:
        exploration_policies = [None] * n_games
    if world_rngs is None:
        world_rngs = [None] * n_games
    if len(exploration_policies) != n_games:
        raise ValueError("exploration_policies must be same length as hands_list")
    if len(world_rngs) != n_games:
        raise ValueError("world_rngs must be same length as hands_list")

    use_posterior = posterior_config is not None and posterior_config.enabled

    games = [
        GameState.from_hands(h, decl_id=d, leader=0)
        for h, d in zip(hands_list, decl_ids)
    ]
    decisions_by_game: list[list[DecisionRecord]] = [[] for _ in range(n_games)]

    voids_by_game: list[dict[int, set[int]]] = [
        {0: set(), 1: set(), 2: set(), 3: set()} for _ in range(n_games)
    ]
    plays_processed = [0] * n_games
    played_by: list[dict[int, list[int]]] = [
        {0: [], 1: [], 2: [], 3: []} for _ in range(n_games)
    ]

    exploration_rngs: list[np.random.Generator | None] = []
    for policy, world_rng in zip(exploration_policies, world_rngs):
        if policy is None:
            exploration_rngs.append(None)
            continue
        if policy.seed is not None:
            exploration_rngs.append(np.random.default_rng(policy.seed))
            continue
        if world_rng is not None:
            exploration_rngs.append(_spawn_child_rng(world_rng))
            continue
        exploration_rngs.append(np.random.default_rng())

    game_exploration_stats: list[GameExplorationStats | None] = [
        GameExplorationStats() if p is not None else None for p in exploration_policies
    ]
    total_entropy: list[float] = [0.0] * n_games

    def _batched_query_current_q_values(ctxs: list[dict]) -> dict[int, Tensor]:
        """Return per-game (N,7) Q-values for the current decision, batched across all games."""
        if not hasattr(oracle, "query_batch_multi_state"):
            out: dict[int, Tensor] = {}
            for ctx in ctxs:
                out[ctx["game_idx"]] = oracle.query_batch(
                    ctx["hypothetical_deals"], ctx["game_state_info"], ctx["player"]
                )
            return out

        # Single batch: concatenate all worlds from all games
        worlds_all: list[list[list[int]]] = []
        trick_plays_list: list[list[tuple[int, int]]] = []

        actors_chunks: list[np.ndarray] = []
        leaders_chunks: list[np.ndarray] = []
        remaining_chunks: list[np.ndarray] = []
        decl_ids_chunks: list[np.ndarray] = []

        offsets: dict[int, slice] = {}
        cursor = 0
        for ctx in ctxs:
            deals = ctx["hypothetical_deals"]
            game_state_info = ctx["game_state_info"]
            n_worlds = len(deals)
            worlds_all.extend(deals)

            actors_chunks.append(np.full(n_worlds, ctx["player"], dtype=np.int32))
            leaders_chunks.append(
                np.full(n_worlds, game_state_info["leader"], dtype=np.int32)
            )
            remaining_chunks.append(game_state_info["remaining"].astype(np.int32))
            decl_ids_chunks.append(np.full(n_worlds, ctx["decl_id"], dtype=np.int32))
            trick_plays_list.extend([game_state_info["trick_plays"]] * n_worlds)

            offsets[ctx["game_idx"]] = slice(cursor, cursor + n_worlds)
            cursor += n_worlds

        actors = np.concatenate(actors_chunks, axis=0)
        leaders = np.concatenate(leaders_chunks, axis=0)
        remaining = np.concatenate(remaining_chunks, axis=0)
        decl_ids = np.concatenate(decl_ids_chunks, axis=0)

        # Single unified batch call with per-sample decl_ids
        q_all = oracle.query_batch_multi_state(
            worlds=worlds_all,
            decl_ids=decl_ids,
            actors=actors,
            leaders=leaders,
            trick_plays_list=trick_plays_list,
            remaining=remaining,
        )

        # Split results back to per-game
        out: dict[int, Tensor] = {}
        for game_idx, sl in offsets.items():
            out[game_idx] = q_all[sl]
        return out

    while not all(game.is_complete() for game in games):
        ctxs: list[dict] = []
        for game_idx, game in enumerate(games):
            if game.is_complete():
                continue

            decl_id = decl_ids[game_idx]
            player = game.current_player()
            my_hand = list(game.hands[player])

            for play_idx in range(plays_processed[game_idx], len(game.play_history)):
                play_player, domino_id, lead_domino_id = game.play_history[play_idx]
                played_by[game_idx][play_player].append(domino_id)
                led_suit = led_suit_for_lead_domino(lead_domino_id, decl_id)
                if not can_follow(domino_id, led_suit, decl_id):
                    voids_by_game[game_idx][play_player].add(led_suit)
            plays_processed[game_idx] = len(game.play_history)

            remaining_worlds = sample_consistent_worlds(
                my_player=player,
                my_hand=my_hand,
                played=game.played,
                hand_sizes=game.hand_sizes(),
                voids=voids_by_game[game_idx],
                decl_id=decl_id,
                n_samples=n_samples,
                rng=world_rngs[game_idx],
            )

            hypothetical_deals, game_state_info = _build_hypothetical_worlds_batched(
                game, remaining_worlds, played_by[game_idx], player
            )

            ctxs.append(
                {
                    "game_idx": game_idx,
                    "decl_id": decl_id,
                    "player": player,
                    "my_hand": my_hand,
                    "hypothetical_deals": hypothetical_deals,
                    "game_state_info": game_state_info,
                    "play_history": list(game.play_history),
                }
            )

        q_values_by_game = _batched_query_current_q_values(ctxs)

        weights_by_game: dict[int, Tensor] = {}
        diagnostics_by_game: dict[int, PosteriorDiagnostics | None] = {}
        if use_posterior:
            weights_list, diag_list = compute_posterior_weights_many(
                oracle=oracle,
                hypothetical_deals_list=[ctx["hypothetical_deals"] for ctx in ctxs],
                play_histories=[ctx["play_history"] for ctx in ctxs],
                decl_ids=[ctx["decl_id"] for ctx in ctxs],
                config=posterior_config,
                rngs=[world_rngs[ctx["game_idx"]] for ctx in ctxs],
            )
            for ctx, w, d in zip(ctxs, weights_list, diag_list):
                weights_by_game[ctx["game_idx"]] = w
                diagnostics_by_game[ctx["game_idx"]] = d
        else:
            for ctx in ctxs:
                n_worlds = len(ctx["hypothetical_deals"])
                weights_by_game[ctx["game_idx"]] = (
                    torch.ones(n_worlds, device=oracle.device) / n_worlds
                )
                diagnostics_by_game[ctx["game_idx"]] = None

        for ctx in ctxs:
            game_idx = ctx["game_idx"]
            player = ctx["player"]
            my_hand = ctx["my_hand"]
            decl_id = ctx["decl_id"]
            hypothetical_deals = ctx["hypothetical_deals"]

            game = games[game_idx]
            all_logits = q_values_by_game[game_idx]
            weights = weights_by_game[game_idx]
            diagnostics = diagnostics_by_game[game_idx]

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
                policy=exploration_policies[game_idx],
                rng=exploration_rngs[game_idx],
            )
            action_domino = my_hand[action_idx]

            exploration_stats: ExplorationStats | None = None
            if exploration_policies[game_idx] is not None:
                q_gap = (e_q_mean[greedy_idx] - e_q_mean[action_idx]).item()
                exploration_stats = ExplorationStats(
                    greedy_action=greedy_idx,
                    action_taken=action_idx,
                    was_greedy=action_idx == greedy_idx,
                    selection_mode=selection_mode,
                    q_gap=q_gap,
                    action_entropy=action_entropy,
                )

                ges = game_exploration_stats[game_idx]
                if ges is not None:
                    ges.n_decisions += 1
                    if selection_mode == "greedy":
                        ges.n_greedy += 1
                    elif selection_mode == "boltzmann":
                        ges.n_boltzmann += 1
                    elif selection_mode == "epsilon":
                        ges.n_epsilon += 1
                    elif selection_mode == "blunder":
                        ges.n_blunder += 1
                    ges.total_q_gap += q_gap
                    total_entropy[game_idx] += action_entropy

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

            if use_posterior or exploration_policies[game_idx] is not None:
                decisions_by_game[game_idx].append(
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
                decisions_by_game[game_idx].append(
                    DecisionRecord(
                        transcript_tokens=transcript_tokens,
                        e_q_mean=padded_e_q_mean,
                        legal_mask=padded_legal_mask,
                        action_taken=action_idx,
                        player=player,
                    )
                )

            games[game_idx] = game.apply_action(action_domino)

    records: list[GameRecord] = []
    for game_idx, game in enumerate(games):
        if exploration_policies[game_idx] is not None:
            ges = game_exploration_stats[game_idx]
            if ges is not None and ges.n_decisions > 0:
                ges.mean_action_entropy = total_entropy[game_idx] / ges.n_decisions

        _fill_actual_outcomes(decisions_by_game[game_idx], game, decl_ids[game_idx])

        if use_posterior or exploration_policies[game_idx] is not None:
            records.append(
                GameRecordV2(
                    decisions=decisions_by_game[game_idx],
                    posterior_config=posterior_config,
                    exploration_policy=exploration_policies[game_idx],
                    exploration_stats=game_exploration_stats[game_idx],
                )
            )
        else:
            records.append(GameRecord(decisions=decisions_by_game[game_idx]))

    return records

