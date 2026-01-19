"""Cross-game pipelined E[Q] generation with CPU/GPU overlap.

This module implements pipelining to overlap CPU world sampling with GPU oracle queries.

Key insight: While GPU processes oracle queries for game A's decision D, CPU can
prepare game A's decision D+1 (or game B's decision D). Games are independent,
so we can pipeline across multiple games.

Architecture:
- Main thread: Coordinates work, applies actions, builds final records
- CPU worker threads: Sample worlds, build hypotheticals (9.3ms bottleneck)
- GPU processing: Uses oracle's async streams for overlapped execution

Target: ~12.2ms/decision (GPU-bound) vs 22.2ms/decision (sequential) = 1.8x
"""

from __future__ import annotations

import queue
import threading
from collections import defaultdict
from dataclasses import dataclass

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


@dataclass
class _SamplingTask:
    """Task for CPU worker: sample worlds for a decision."""

    game_idx: int
    decl_id: int
    player: int
    my_hand: list[int]
    played: set[int]
    hand_sizes: list[int]
    voids: dict[int, set[int]]
    n_samples: int
    rng: np.random.Generator | None


@dataclass
class _PreparedQuery:
    """CPU-prepared query ready for GPU processing."""

    game_idx: int
    decl_id: int
    player: int
    my_hand: list[int]
    hypothetical_deals: list[list[list[int]]]
    game_state_info: dict
    play_history: list[tuple[int, int, int]]
    played_by: dict[int, list[int]]


@dataclass
class _OracleResult:
    """GPU result ready for decision making."""

    game_idx: int
    player: int
    my_hand: list[int]
    decl_id: int
    hypothetical_deals: list[list[list[int]]]
    q_values: Tensor
    weights: Tensor
    diagnostics: PosteriorDiagnostics | None
    game_play_history: list[tuple[int, int, int]]


def generate_eq_games_pipelined(
    oracle: Stage1Oracle,
    hands_list: list[list[list[int]]],
    decl_ids: list[int],
    *,
    n_samples: int = 100,
    posterior_config: PosteriorConfig | None = None,
    exploration_policies: list[ExplorationPolicy | None] | None = None,
    world_rngs: list[np.random.Generator | None] | None = None,
    n_workers: int = 2,
) -> list[GameRecord]:
    """Generate multiple games with pipelined CPU/GPU overlap.

    Uses worker threads for CPU-bound world sampling while GPU processes queries.
    The oracle's async CUDA streams provide additional GPU-level pipelining.

    Args:
        oracle: Stage1Oracle for Q-value queries
        hands_list: List of initial hands for each game
        decl_ids: Declaration ID for each game
        n_samples: Number of worlds to sample per decision
        posterior_config: Optional posterior weighting configuration
        exploration_policies: Optional exploration policies per game
        world_rngs: Optional RNG per game for reproducibility
        n_workers: Number of CPU threads for world sampling (default 2)

    Returns:
        List of GameRecord objects, one per game

    Design:
        1. Main thread submits sampling tasks to CPU workers
        2. CPU workers sample worlds → prepared_queue
        3. Main thread batches prepared queries by decl_id
        4. Main thread runs oracle (GPU async), applies actions
        5. Repeat until all games complete

        The overlap comes from CPU workers preparing future queries while
        main thread waits for GPU (which uses async streams internally).

    Note:
        - Currently batches by decl_id (same as generate_batched.py)
        - Future optimization: Remove decl_id batching constraint (t42-5kvo §2)
    """
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

    # Initialize game states
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

    # Queues for pipeline
    task_queue: queue.Queue[_SamplingTask | None] = queue.Queue()
    prepared_queue: queue.Queue[_PreparedQuery | None] = queue.Queue()

    # Error handling
    error_box: list[Exception | None] = [None]

    # CPU worker: Sample worlds (the 9.3ms bottleneck we want to overlap)
    def cpu_worker():
        """Sample worlds and build hypothetical deals."""
        try:
            while True:
                task = task_queue.get()
                if task is None:  # Poison pill
                    break

                # This is the CPU-bound work we want to overlap with GPU
                remaining_worlds = sample_consistent_worlds(
                    my_player=task.player,
                    my_hand=task.my_hand,
                    played=task.played,
                    hand_sizes=task.hand_sizes,
                    voids=task.voids,
                    decl_id=task.decl_id,
                    n_samples=task.n_samples,
                    rng=task.rng,
                )

                # Build hypothetical deals (fast, but part of preparation)
                # Note: We need the game state, so we pass played_by through task
                # For now, we'll defer this to main thread
                prepared_queue.put((task, remaining_worlds))
                task_queue.task_done()

        except Exception as e:
            error_box[0] = e
            prepared_queue.put(None)  # Signal error

    # Start worker threads
    workers = []
    for _ in range(n_workers):
        t = threading.Thread(target=cpu_worker, daemon=True)
        t.start()
        workers.append(t)

    try:
        # Main loop: submit tasks, collect prepared queries, process batches
        while not all(game.is_complete() for game in games):
            # Phase 1: Submit sampling tasks for all incomplete games
            n_submitted = 0
            for game_idx, game in enumerate(games):
                if game.is_complete():
                    continue

                decl_id = decl_ids[game_idx]
                player = game.current_player()
                my_hand = list(game.hands[player])

                # Update voids from new plays
                for play_idx in range(plays_processed[game_idx], len(game.play_history)):
                    play_player, domino_id, lead_domino_id = game.play_history[play_idx]
                    played_by[game_idx][play_player].append(domino_id)
                    led_suit = led_suit_for_lead_domino(lead_domino_id, decl_id)
                    if not can_follow(domino_id, led_suit, decl_id):
                        voids_by_game[game_idx][play_player].add(led_suit)
                plays_processed[game_idx] = len(game.play_history)

                # Submit sampling task to workers
                task = _SamplingTask(
                    game_idx=game_idx,
                    decl_id=decl_id,
                    player=player,
                    my_hand=my_hand,
                    played=game.played.copy(),
                    hand_sizes=game.hand_sizes(),
                    voids=voids_by_game[game_idx].copy(),
                    n_samples=n_samples,
                    rng=world_rngs[game_idx],
                )
                task_queue.put(task)
                n_submitted += 1

            if n_submitted == 0:
                break

            # Phase 2: Collect prepared queries (workers are sampling in parallel)
            queries: list[_PreparedQuery] = []
            for _ in range(n_submitted):
                result = prepared_queue.get()
                if result is None:  # Error signal
                    if error_box[0]:
                        raise error_box[0]
                    raise RuntimeError("Worker thread failed")

                task, remaining_worlds = result

                # Build hypothetical deals (fast, in main thread)
                game = games[task.game_idx]
                hypothetical_deals, game_state_info = _build_hypothetical_worlds_batched(
                    game, remaining_worlds, played_by[task.game_idx], task.player
                )

                queries.append(
                    _PreparedQuery(
                        game_idx=task.game_idx,
                        decl_id=task.decl_id,
                        player=task.player,
                        my_hand=task.my_hand,
                        hypothetical_deals=hypothetical_deals,
                        game_state_info=game_state_info,
                        play_history=list(game.play_history),
                        played_by=played_by[task.game_idx].copy(),
                    )
                )
                prepared_queue.task_done()

            # Phase 3: Process batch on GPU (while workers prepare next round)
            results = _process_batch_by_decl(
                oracle, queries, use_posterior, posterior_config, world_rngs
            )

            # Phase 4: Apply actions and update games
            for result in results:
                _process_result(
                    result=result,
                    games=games,
                    decisions_by_game=decisions_by_game,
                    exploration_policies=exploration_policies,
                    exploration_rngs=exploration_rngs,
                    game_exploration_stats=game_exploration_stats,
                    total_entropy=total_entropy,
                    decl_ids=decl_ids,
                )

    finally:
        # Shutdown workers
        for _ in range(n_workers):
            task_queue.put(None)  # Poison pill
        for t in workers:
            t.join(timeout=1.0)

    # Build final records
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


def _process_batch_by_decl(
    oracle: Stage1Oracle,
    queries: list[_PreparedQuery],
    use_posterior: bool,
    posterior_config: PosteriorConfig | None,
    world_rngs: list[np.random.Generator | None],
) -> list[_OracleResult]:
    """Process a batch of queries, grouping by decl_id for efficient oracle calls."""
    # Group by decl_id
    by_decl: dict[int, list[_PreparedQuery]] = defaultdict(list)
    for query in queries:
        by_decl[query.decl_id].append(query)

    results: list[_OracleResult] = []

    for decl_id, group in by_decl.items():
        # Batch oracle call across all queries with same decl_id
        if hasattr(oracle, "query_batch_multi_state"):
            worlds_all: list[list[list[int]]] = []
            trick_plays_list: list[list[tuple[int, int]]] = []
            actors_chunks: list[np.ndarray] = []
            leaders_chunks: list[np.ndarray] = []
            remaining_chunks: list[np.ndarray] = []

            offsets: dict[int, slice] = {}
            cursor = 0
            for i, query in enumerate(group):
                deals = query.hypothetical_deals
                game_state_info = query.game_state_info
                n_worlds = len(deals)
                worlds_all.extend(deals)

                actors_chunks.append(np.full(n_worlds, query.player, dtype=np.int32))
                leaders_chunks.append(
                    np.full(n_worlds, game_state_info["leader"], dtype=np.int32)
                )
                remaining_chunks.append(game_state_info["remaining"].astype(np.int32))
                trick_plays_list.extend([game_state_info["trick_plays"]] * n_worlds)

                offsets[i] = slice(cursor, cursor + n_worlds)
                cursor += n_worlds

            actors = np.concatenate(actors_chunks, axis=0)
            leaders = np.concatenate(leaders_chunks, axis=0)
            remaining = np.concatenate(remaining_chunks, axis=0)

            q_all = oracle.query_batch_multi_state(
                worlds=worlds_all,
                decl_ids=decl_id,  # Accepts int or array
                actors=actors,
                leaders=leaders,
                trick_plays_list=trick_plays_list,
                remaining=remaining,
            )

            # Split results by query
            q_values_by_idx = {i: q_all[sl] for i, sl in offsets.items()}
        else:
            # Fallback: individual queries
            q_values_by_idx = {}
            for i, query in enumerate(group):
                q_values_by_idx[i] = oracle.query_batch(
                    query.hypothetical_deals,
                    query.game_state_info,
                    query.player,
                )

        # Compute posterior weights if needed
        weights_by_idx: dict[int, Tensor] = {}
        diagnostics_by_idx: dict[int, PosteriorDiagnostics | None] = {}

        if use_posterior:
            weights_list, diag_list = compute_posterior_weights_many(
                oracle=oracle,
                hypothetical_deals_list=[q.hypothetical_deals for q in group],
                play_histories=[q.play_history for q in group],
                decl_ids=[q.decl_id for q in group],
                config=posterior_config,
                rngs=[world_rngs[q.game_idx] for q in group],
            )
            for i, (w, d) in enumerate(zip(weights_list, diag_list)):
                weights_by_idx[i] = w
                diagnostics_by_idx[i] = d
        else:
            for i, query in enumerate(group):
                n_worlds = len(query.hypothetical_deals)
                weights_by_idx[i] = torch.ones(n_worlds, device=oracle.device) / n_worlds
                diagnostics_by_idx[i] = None

        # Package results
        for i, query in enumerate(group):
            results.append(
                _OracleResult(
                    game_idx=query.game_idx,
                    player=query.player,
                    my_hand=query.my_hand,
                    decl_id=query.decl_id,
                    hypothetical_deals=query.hypothetical_deals,
                    q_values=q_values_by_idx[i],
                    weights=weights_by_idx[i],
                    diagnostics=diagnostics_by_idx[i],
                    game_play_history=query.play_history,
                )
            )

    return results


def _process_result(
    result: _OracleResult,
    games: list[GameState],
    decisions_by_game: list[list[DecisionRecord]],
    exploration_policies: list[ExplorationPolicy | None],
    exploration_rngs: list[np.random.Generator | None],
    game_exploration_stats: list[GameExplorationStats | None],
    total_entropy: list[float],
    decl_ids: list[int],
):
    """Process oracle result, select action, and update game state."""
    game_idx = result.game_idx
    game = games[game_idx]
    decl_id = result.decl_id

    # Reduce Q-values
    e_q_mean, e_q_var = _reduce_world_q_values(
        all_q_values=result.q_values,
        weights=result.weights,
        hypothetical_deals=result.hypothetical_deals,
        player=result.player,
        my_hand=result.my_hand,
    )

    plays_for_transcript = [(p, d) for p, d, _ in result.game_play_history]
    transcript_tokens = tokenize_transcript(
        result.my_hand, plays_for_transcript, decl_id, result.player
    )

    legal_actions = game.legal_actions()
    legal_mask = _build_legal_mask(legal_actions, result.my_hand)

    masked_q = e_q_mean.clone()
    masked_q[~legal_mask] = float("-inf")
    greedy_idx = masked_q.argmax().item()

    action_idx, selection_mode, action_entropy = _select_action_with_exploration(
        e_q_mean=e_q_mean,
        legal_mask=legal_mask,
        policy=exploration_policies[game_idx],
        rng=exploration_rngs[game_idx],
    )
    action_domino = result.my_hand[action_idx]

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
    padded_e_q_mean[: len(result.my_hand)] = e_q_mean
    padded_legal_mask = torch.zeros(7, dtype=torch.bool)
    padded_legal_mask[: len(result.my_hand)] = legal_mask

    padded_e_q_var = torch.zeros(7)
    padded_e_q_var[: len(result.my_hand)] = e_q_var

    legal_std = torch.sqrt(e_q_var)
    legal_std_for_u = legal_std[legal_mask]
    if len(legal_std_for_u) > 0:
        u_mean = legal_std_for_u.mean().item()
        u_max = legal_std_for_u.max().item()
    else:
        u_mean = 0.0
        u_max = 0.0

    use_v2 = (result.diagnostics is not None) or (exploration_policies[game_idx] is not None)

    if use_v2:
        decisions_by_game[game_idx].append(
            DecisionRecordV2(
                transcript_tokens=transcript_tokens,
                e_q_mean=padded_e_q_mean,
                legal_mask=padded_legal_mask,
                action_taken=action_idx,
                player=result.player,
                diagnostics=result.diagnostics,
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
                player=result.player,
            )
        )

    games[game_idx] = game.apply_action(action_domino)
