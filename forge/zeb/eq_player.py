"""E[Q] player for evaluation - uses Stage 1 oracle with world sampling.

Bridges the Zeb game infrastructure (ZebGameState) with the E[Q] pipeline
(GameStateTensor, world sampling, oracle queries, P(Make) action selection).

Usage:
    model = DominoLightningModule.load_from_checkpoint(checkpoint_path)
    results = evaluate_eq_vs_random(model, n_games=1000, n_samples=100)
"""

from __future__ import annotations

import random as stdlib_random
import time

import torch
from torch import Tensor

from forge.eq.game_tensor import GameStateTensor
from forge.eq.generate.actions import select_actions
from forge.eq.generate.deals import build_hypothetical_deals
from forge.eq.generate.eq_compute import compute_eq_pdf
from forge.eq.generate.model import query_model
from forge.eq.generate.sampling import sample_worlds_batched
from forge.eq.generate.tokenization import tokenize_batched
from forge.eq.sampling_mrv_gpu import WorldSamplerMRV
from forge.eq.tokenize_gpu import GPUTokenizer

from .evaluate import Player, _get_current_player
from .game import apply_action, game_seed, get_outcome, is_terminal, legal_actions, new_game
from .gpu_training_pipeline import TrainingExamples
from .observation import observe, get_legal_mask
from .types import ZebGameState


def zeb_states_to_game_state_tensor(
    states: list[ZebGameState],
    device: str = 'cuda',
) -> GameStateTensor:
    """Convert list of ZebGameState to batched GameStateTensor.

    Args:
        states: List of N ZebGameState objects (must be in PLAYING phase)
        device: Device for tensors

    Returns:
        GameStateTensor with N games
    """
    n_games = len(states)

    # Build hands tensor: [N, 4, 7] int8
    hands_list = [[list(h) for h in s.hands] for s in states]
    hands = torch.tensor(hands_list, dtype=torch.int8, device=device)

    # Build played_mask: [N, 28] bool
    played_mask = torch.zeros(n_games, 28, dtype=torch.bool, device=device)
    for g, state in enumerate(states):
        for d in state.played:
            played_mask[g, d] = True

    # Mark played slots in hands as -1 (vectorized)
    hands_long = hands.long().clamp(0, 27)
    batch_idx = torch.arange(n_games, device=device).view(-1, 1, 1).expand_as(hands)
    is_played = played_mask[batch_idx.reshape(-1), hands_long.reshape(-1)].reshape(n_games, 4, 7)
    hands[is_played] = -1

    # Build history: [N, 28, 3] int8 - (player, domino_id, lead_domino_id)
    history = torch.full((n_games, 28, 3), -1, dtype=torch.int8, device=device)
    for g, state in enumerate(states):
        for i, (player, domino_id) in enumerate(state.play_history):
            trick_start = (i // 4) * 4
            lead_domino_id = state.play_history[trick_start][1]
            history[g, i, 0] = player
            history[g, i, 1] = domino_id
            history[g, i, 2] = lead_domino_id

    # Build trick_plays: [N, 4] int8
    trick_plays = torch.full((n_games, 4), -1, dtype=torch.int8, device=device)
    for g, state in enumerate(states):
        for i, domino_id in enumerate(state.current_trick):
            trick_plays[g, i] = domino_id

    leader = torch.tensor([s.trick_leader for s in states], dtype=torch.int8, device=device)
    decl_ids = torch.tensor([s.decl_id for s in states], dtype=torch.int8, device=device)
    bidder = torch.tensor([s.bidder for s in states], dtype=torch.int8, device=device)

    return GameStateTensor(
        hands=hands,
        played_mask=played_mask,
        history=history,
        trick_plays=trick_plays,
        leader=leader,
        decl_ids=decl_ids,
        device=device,
        bidder=bidder,
    )


class EQPlayer:
    """E[Q] player implementing the Player protocol.

    Uses Stage 1 oracle with world sampling and P(Make) action selection.
    For use with play_match() or any code expecting the Player protocol.

    Note: This processes one game at a time. For batched evaluation,
    use evaluate_eq_vs_random() directly.
    """

    def __init__(self, model, n_samples: int = 100, device: str = 'cuda'):
        self.model = model
        self.n_samples = n_samples
        self.device = device
        self.sampler = WorldSamplerMRV(max_games=1, max_samples=n_samples, device=device)
        self.tokenizer = GPUTokenizer(max_batch=n_samples, device=device)
        self.model.eval()

    def select_action(self, state: ZebGameState, player: int) -> int:
        """Select action using E[Q] with P(Make) optimization."""
        gst = zeb_states_to_game_state_tensor([state], self.device)

        with torch.no_grad():
            worlds = sample_worlds_batched(gst, self.sampler, self.n_samples)
            deals = build_hypothetical_deals(gst, worlds)
            tokens, masks = tokenize_batched(gst, deals, self.tokenizer)
            q_values = query_model(
                self.model, tokens, masks, gst, self.n_samples, self.device,
            )

            q_reshaped = q_values.view(1, self.n_samples, 7)
            e_q = q_reshaped.mean(dim=1)
            e_q_pdf = compute_eq_pdf(q_reshaped)
            actions, _ = select_actions(gst, e_q, e_q_pdf, greedy=True)

        return actions[0].item()


def evaluate_eq_vs_random(
    model,
    n_games: int = 1000,
    n_samples: int = 100,
    device: str = 'cuda',
    verbose: bool = True,
) -> dict:
    """Evaluate E[Q] player vs random opponents (batched GPU).

    Runs games with both team assignments to eliminate seat advantage.

    Args:
        model: Stage 1 oracle model (DominoLightningModule)
        n_games: Total games (split evenly between team assignments)
        n_samples: Worlds sampled per E[Q] decision
        device: CUDA device
        verbose: Print progress

    Returns:
        Dict with win rates, margins, per-team breakdowns
    """
    half = n_games // 2

    if verbose:
        print(f"  E[Q] as team 0 ({half} games)...")
    t0 = time.time()
    r0 = _run_eq_vs_random_batched(
        model, n_games=half, n_samples=n_samples,
        eq_team=0, base_seed=0, device=device,
    )
    if verbose:
        print(f"    {r0['eq_win_rate']:.1%} win rate, {time.time()-t0:.1f}s")

    if verbose:
        print(f"  E[Q] as team 1 ({half} games)...")
    t1 = time.time()
    r1 = _run_eq_vs_random_batched(
        model, n_games=half, n_samples=n_samples,
        eq_team=1, base_seed=half, device=device,
    )
    if verbose:
        print(f"    {r1['eq_win_rate']:.1%} win rate, {time.time()-t1:.1f}s")

    eq_wins = r0['eq_wins'] + r1['eq_wins']
    total = r0['n_games'] + r1['n_games']

    return {
        'eq_win_rate': eq_wins / total,
        'eq_wins': eq_wins,
        'total_games': total,
        'as_team0': r0,
        'as_team1': r1,
    }


def _run_eq_vs_random_batched(
    model,
    n_games: int,
    n_samples: int,
    eq_team: int,
    base_seed: int,
    device: str,
) -> dict:
    """Run E[Q] as one team vs random (batched GPU evaluation)."""
    # Initialize games
    states = [
        new_game(seed=game_seed(base_seed, i))
        for i in range(n_games)
    ]
    active = [True] * n_games

    # Pre-allocate GPU resources for worst case (all games need E[Q])
    sampler = WorldSamplerMRV(max_games=n_games, max_samples=n_samples, device=device)
    tokenizer = GPUTokenizer(max_batch=n_games * n_samples, device=device)

    model.eval()

    while any(active):
        # Partition active games by player type
        eq_indices = []
        eq_game_states = []
        random_indices = []

        for i, (state, is_active) in enumerate(zip(states, active)):
            if not is_active:
                continue
            player = _get_current_player(state)
            if player % 2 == eq_team:
                eq_indices.append(i)
                eq_game_states.append(state)
            else:
                random_indices.append(i)

        # Batched E[Q] decisions
        if eq_indices:
            n_eq = len(eq_indices)
            gst = zeb_states_to_game_state_tensor(eq_game_states, device)

            with torch.no_grad():
                worlds = sample_worlds_batched(gst, sampler, n_samples)
                deals = build_hypothetical_deals(gst, worlds)

                # Resize tokenizer if batch exceeds capacity
                batch_needed = n_eq * n_samples
                if batch_needed > tokenizer.max_batch:
                    tokenizer = GPUTokenizer(max_batch=batch_needed, device=device)

                tokens, masks = tokenize_batched(gst, deals, tokenizer)
                q_values = query_model(
                    model, tokens, masks, gst, n_samples, device,
                )

                q_reshaped = q_values.view(n_eq, n_samples, 7)
                e_q = q_reshaped.mean(dim=1)
                e_q_pdf = compute_eq_pdf(q_reshaped)
                actions, _ = select_actions(gst, e_q, e_q_pdf, greedy=True)

            for idx, game_idx in enumerate(eq_indices):
                action = actions[idx].item()
                states[game_idx] = apply_action(states[game_idx], action)
                if is_terminal(states[game_idx]):
                    active[game_idx] = False

        # Random decisions
        for game_idx in random_indices:
            legal = legal_actions(states[game_idx])
            action = stdlib_random.choice(legal)
            states[game_idx] = apply_action(states[game_idx], action)
            if is_terminal(states[game_idx]):
                active[game_idx] = False

    # Score results
    eq_wins = 0
    opp_wins = 0
    total_margin = 0

    for state in states:
        team0_pts, team1_pts = state.team_points
        eq_pts = team0_pts if eq_team == 0 else team1_pts
        rand_pts = team1_pts if eq_team == 0 else team0_pts

        if eq_pts > rand_pts:
            eq_wins += 1
        else:
            opp_wins += 1
        total_margin += eq_pts - rand_pts

    return {
        'eq_wins': eq_wins,
        'opp_wins': opp_wins,
        'eq_win_rate': eq_wins / n_games,
        'avg_margin': total_margin / n_games,
        'n_games': n_games,
        'eq_team': eq_team,
    }


def _merge_results(results: list[dict]) -> dict:
    """Merge multiple batch results into one aggregate result."""
    eq_wins = sum(r['eq_wins'] for r in results)
    opp_wins = sum(r['opp_wins'] for r in results)
    n_games = sum(r['n_games'] for r in results)
    total_margin = sum(r['avg_margin'] * r['n_games'] for r in results)
    return {
        'eq_wins': eq_wins,
        'opp_wins': opp_wins,
        'eq_win_rate': eq_wins / n_games,
        'avg_margin': total_margin / n_games,
        'n_games': n_games,
        'eq_team': results[0]['eq_team'],
    }


def evaluate_eq_vs_eq(
    model,
    n_games: int = 1000,
    n_samples_a: int = 100,
    n_samples_b: int = 100,
    device: str = 'cuda',
    batch_size: int = 0,
    verbose: bool = True,
) -> dict:
    """Evaluate E[Q](n_samples_a) vs E[Q](n_samples_b) (batched GPU).

    Both sides share the same oracle model but may differ in sample count.
    Team A uses n_samples_a, Team B uses n_samples_b.
    Runs both team assignments to eliminate seat advantage.

    Returns:
        Dict with win rates, margins, per-team breakdowns (from A's perspective).
    """
    half = n_games // 2

    def _run_chunked(n: int, a_team: int, seed_offset: int) -> dict:
        chunk = batch_size if batch_size > 0 else n
        results = []
        seed = seed_offset
        remaining = n
        batch_num = 0
        while remaining > 0:
            this_batch = min(chunk, remaining)
            r = _run_eq_vs_eq_batched(
                model, n_games=this_batch,
                n_samples_a=n_samples_a, n_samples_b=n_samples_b,
                a_team=a_team, base_seed=seed, device=device,
            )
            results.append(r)
            batch_num += 1
            done = n - remaining + this_batch
            agg = _merge_results(results)
            if verbose:
                print(f"    batch {batch_num}: {done}/{n} games, "
                      f"A {agg['eq_win_rate']:.1%}")
            seed += this_batch
            remaining -= this_batch
        return agg if len(results) > 1 else results[0]

    if verbose:
        print(f"  A(n={n_samples_a}) as team 0 ({half} games)...")
    t0 = time.time()
    r0 = _run_chunked(half, a_team=0, seed_offset=0)
    if verbose:
        print(f"    A {r0['eq_win_rate']:.1%}, {time.time()-t0:.1f}s")

    if verbose:
        print(f"  A(n={n_samples_a}) as team 1 ({half} games)...")
    t1 = time.time()
    r1 = _run_chunked(half, a_team=1, seed_offset=half)
    if verbose:
        print(f"    A {r1['eq_win_rate']:.1%}, {time.time()-t1:.1f}s")

    a_wins = r0['eq_wins'] + r1['eq_wins']
    total = r0['n_games'] + r1['n_games']

    return {
        'eq_win_rate': a_wins / total,
        'eq_wins': a_wins,
        'total_games': total,
        'as_team0': r0,
        'as_team1': r1,
    }


def _run_eq_vs_eq_batched(
    model,
    n_games: int,
    n_samples_a: int,
    n_samples_b: int,
    a_team: int,
    base_seed: int,
    device: str,
) -> dict:
    """Run E[Q](A) vs E[Q](B) batched. A plays as a_team (0 or 1)."""
    states = [
        new_game(seed=game_seed(base_seed, i))
        for i in range(n_games)
    ]
    active = [True] * n_games

    # Separate GPU resources for each side (different n_samples)
    sampler_a = WorldSamplerMRV(max_games=n_games, max_samples=n_samples_a, device=device)
    tokenizer_a = GPUTokenizer(max_batch=n_games * n_samples_a, device=device)
    sampler_b = WorldSamplerMRV(max_games=n_games, max_samples=n_samples_b, device=device)
    tokenizer_b = GPUTokenizer(max_batch=n_games * n_samples_b, device=device)

    model.eval()

    while any(active):
        a_indices = []
        a_game_states = []
        b_indices = []
        b_game_states = []

        for i, (state, is_active) in enumerate(zip(states, active)):
            if not is_active:
                continue
            player = _get_current_player(state)
            if player % 2 == a_team:
                a_indices.append(i)
                a_game_states.append(state)
            else:
                b_indices.append(i)
                b_game_states.append(state)

        # Batched E[Q] decisions for team A
        if a_indices:
            _apply_eq_actions(
                model, a_indices, a_game_states,
                n_samples_a, sampler_a, tokenizer_a,
                states, active, device,
            )

        # Batched E[Q] decisions for team B
        if b_indices:
            _apply_eq_actions(
                model, b_indices, b_game_states,
                n_samples_b, sampler_b, tokenizer_b,
                states, active, device,
            )

    # Score from A's perspective
    a_wins = 0
    opp_wins = 0
    total_margin = 0

    for state in states:
        team0_pts, team1_pts = state.team_points
        a_pts = team0_pts if a_team == 0 else team1_pts
        b_pts = team1_pts if a_team == 0 else team0_pts

        if a_pts > b_pts:
            a_wins += 1
        else:
            opp_wins += 1
        total_margin += a_pts - b_pts

    return {
        'eq_wins': a_wins,
        'opp_wins': opp_wins,
        'eq_win_rate': a_wins / n_games,
        'avg_margin': total_margin / n_games,
        'n_games': n_games,
        'eq_team': a_team,
    }


def _apply_eq_actions(
    model,
    indices: list[int],
    game_states: list[ZebGameState],
    n_samples: int,
    sampler: WorldSamplerMRV,
    tokenizer: GPUTokenizer,
    states: list[ZebGameState],
    active: list[bool],
    device: str,
) -> None:
    """Batched E[Q] inference + action application. Mutates states/active."""
    n_eq = len(indices)
    gst = zeb_states_to_game_state_tensor(game_states, device)

    with torch.no_grad():
        worlds = sample_worlds_batched(gst, sampler, n_samples)
        deals = build_hypothetical_deals(gst, worlds)

        batch_needed = n_eq * n_samples
        if batch_needed > tokenizer.max_batch:
            tokenizer = GPUTokenizer(max_batch=batch_needed, device=device)

        tokens, masks = tokenize_batched(gst, deals, tokenizer)
        q_values = query_model(model, tokens, masks, gst, n_samples, device)

        q_reshaped = q_values.view(n_eq, n_samples, 7)
        e_q = q_reshaped.mean(dim=1)
        e_q_pdf = compute_eq_pdf(q_reshaped)
        actions, _ = select_actions(gst, e_q, e_q_pdf, greedy=True)

    for idx, game_idx in enumerate(indices):
        action = actions[idx].item()
        states[game_idx] = apply_action(states[game_idx], action)
        if is_terminal(states[game_idx]):
            active[game_idx] = False


def _belief_targets_from_owner(owner: Tensor, player: int) -> tuple[Tensor, Tensor]:
    """Build belief supervision from full-deal ownership for one perspective."""
    rel_owner = (owner - int(player) + 4) % 4
    belief_mask = rel_owner != 0
    belief_targets = (rel_owner - 1).clamp(min=0).to(torch.long)
    return belief_targets, belief_mask


def _owner_tensor_from_state(state: ZebGameState) -> Tensor:
    owner = torch.empty(28, dtype=torch.long)
    for p in range(4):
        for domino in state.hands[p]:
            owner[int(domino)] = p
    return owner


def _concat_training_examples(examples: list[TrainingExamples]) -> TrainingExamples:
    if len(examples) == 1:
        return examples[0]
    return TrainingExamples(
        observations=torch.cat([x.observations for x in examples], dim=0),
        masks=torch.cat([x.masks for x in examples], dim=0),
        hand_indices=torch.cat([x.hand_indices for x in examples], dim=0),
        hand_masks=torch.cat([x.hand_masks for x in examples], dim=0),
        policy_targets=torch.cat([x.policy_targets for x in examples], dim=0),
        value_targets=torch.cat([x.value_targets for x in examples], dim=0),
        belief_targets=torch.cat([x.belief_targets for x in examples], dim=0),
        belief_mask=torch.cat([x.belief_mask for x in examples], dim=0),
    )


def generate_eq_vs_zeb_training_examples(
    oracle_model,
    zeb_model,
    *,
    n_games: int,
    n_samples: int,
    device: str = 'cuda',
    zeb_temperature: float = 0.1,
    base_seed: int = 0,
    batch_size: int = 0,
    include_eq_pdf_policy: bool = False,
) -> tuple[TrainingExamples, dict]:
    """Generate source-compatible TrainingExamples from E[Q] vs Zeb games.

    Returns:
        (examples, stats) where examples are CPU tensors ready for save_examples.
    """
    if n_games <= 0:
        raise ValueError("n_games must be positive")

    # Balance seat assignment (E[Q] team 0 vs team 1) to avoid seat bias.
    n_team0 = (n_games + 1) // 2
    n_team1 = n_games - n_team0

    chunk = batch_size if batch_size > 0 else n_games

    chunks: list[TrainingExamples] = []
    stats_chunks: list[dict] = []

    def _run_side(total_games: int, *, eq_team: int, seed_offset: int) -> None:
        remaining = total_games
        seed = seed_offset
        while remaining > 0:
            this_batch = min(chunk, remaining)
            examples, side_stats = _run_eq_vs_zeb_batched_with_examples(
                oracle_model=oracle_model,
                zeb_model=zeb_model,
                n_games=this_batch,
                n_samples=n_samples,
                eq_team=eq_team,
                base_seed=seed,
                device=device,
                zeb_temperature=zeb_temperature,
                include_eq_pdf_policy=include_eq_pdf_policy,
            )
            chunks.append(examples)
            stats_chunks.append(side_stats)
            seed += this_batch
            remaining -= this_batch

    _run_side(n_team0, eq_team=0, seed_offset=base_seed)
    _run_side(n_team1, eq_team=1, seed_offset=base_seed + n_team0)

    merged_examples = _concat_training_examples(chunks)
    eq_wins = sum(s['eq_wins'] for s in stats_chunks)
    total_games_played = sum(s['n_games'] for s in stats_chunks)
    return merged_examples, {
        'eq_wins': eq_wins,
        'total_games': total_games_played,
        'eq_win_rate': eq_wins / max(total_games_played, 1),
        'avg_margin': sum(s['avg_margin'] * s['n_games'] for s in stats_chunks) / max(total_games_played, 1),
    }


def _run_eq_vs_zeb_batched_with_examples(
    oracle_model,
    zeb_model,
    *,
    n_games: int,
    n_samples: int,
    eq_team: int,
    base_seed: int,
    device: str,
    zeb_temperature: float,
    include_eq_pdf_policy: bool,
) -> tuple[TrainingExamples, dict]:
    """Run batched E[Q] vs Zeb and return per-move training examples."""
    states = [
        new_game(seed=game_seed(base_seed, i))
        for i in range(n_games)
    ]
    active = [True] * n_games
    zeb_step = 0

    owners = [_owner_tensor_from_state(s) for s in states]
    per_game_records: list[list[dict]] = [[] for _ in range(n_games)]

    sampler = WorldSamplerMRV(max_games=n_games, max_samples=n_samples, device=device)
    tokenizer = GPUTokenizer(max_batch=n_games * n_samples, device=device)

    oracle_model.eval()
    zeb_model.eval()
    zeb_model.to(device)

    while any(active):
        eq_indices = []
        eq_game_states = []
        zeb_indices = []
        zeb_game_states = []
        zeb_players = []

        for i, (state, is_active) in enumerate(zip(states, active)):
            if not is_active:
                continue
            player = _get_current_player(state)
            if player % 2 == eq_team:
                eq_indices.append(i)
                eq_game_states.append(state)
            else:
                zeb_indices.append(i)
                zeb_game_states.append(state)
                zeb_players.append(player)

        # Batched E[Q] decisions
        if eq_indices:
            n_eq = len(eq_indices)
            gst = zeb_states_to_game_state_tensor(eq_game_states, device)

            with torch.no_grad():
                worlds = sample_worlds_batched(gst, sampler, n_samples)
                deals = build_hypothetical_deals(gst, worlds)

                batch_needed = n_eq * n_samples
                if batch_needed > tokenizer.max_batch:
                    tokenizer = GPUTokenizer(max_batch=batch_needed, device=device)

                tokens, masks = tokenize_batched(gst, deals, tokenizer)
                q_values = query_model(
                    oracle_model, tokens, masks, gst, n_samples, device,
                )

                q_reshaped = q_values.view(n_eq, n_samples, 7)
                e_q = q_reshaped.mean(dim=1)
                e_q_pdf = compute_eq_pdf(q_reshaped)
                actions, _ = select_actions(gst, e_q, e_q_pdf, greedy=True)

                # Compute p_make per action for soft policy targets
                # Offense (bidder's team): need Q >= 18 -> bin 60+
                # Defense (opponent team): need Q >= -17 -> bin 25+
                if include_eq_pdf_policy:
                    is_offense = ((gst.current_player % 2) == (gst.bidder % 2)).unsqueeze(1)
                    p_make_off = e_q_pdf[:, :, 60:].sum(dim=2)
                    p_make_def = e_q_pdf[:, :, 25:].sum(dim=2)
                    p_make_cpu = torch.where(is_offense, p_make_off, p_make_def).detach().cpu()
                else:
                    p_make_cpu = None

            for idx, game_idx in enumerate(eq_indices):
                state = states[game_idx]
                player = _get_current_player(state)
                action = int(actions[idx].item())

                obs, mask, hand_idx = observe(state, player)
                hand_mask = mask[1:8].clone()
                belief_targets, belief_mask = _belief_targets_from_owner(owners[game_idx], player)

                policy = torch.zeros(7, dtype=torch.float32)
                if include_eq_pdf_policy and p_make_cpu is not None:
                    policy = p_make_cpu[idx].to(torch.float32)
                    policy = torch.where(hand_mask, policy, torch.zeros_like(policy))
                    norm = policy.sum()
                    if norm.item() > 0:
                        policy = policy / norm
                    else:
                        policy[action] = 1.0
                else:
                    policy[action] = 1.0

                per_game_records[game_idx].append({
                    'obs': obs.to(torch.int32),
                    'mask': mask,
                    'hand_idx': hand_idx,
                    'hand_mask': hand_mask,
                    'policy': policy,
                    'belief_targets': belief_targets,
                    'belief_mask': belief_mask,
                    'player': player,
                })

                states[game_idx] = apply_action(state, action)
                if is_terminal(states[game_idx]):
                    active[game_idx] = False

        # Batched Zeb decisions
        if zeb_indices:
            with torch.no_grad():
                batch_tokens = []
                batch_masks = []
                batch_hand_indices = []
                batch_legal = []
                batch_hand_masks = []
                batch_cpu_tokens = []
                batch_cpu_masks = []
                batch_cpu_hand_idx = []

                for state, player in zip(zeb_game_states, zeb_players):
                    tokens, mask, hand_indices = observe(state, player)
                    legal = get_legal_mask(state, player)

                    batch_cpu_tokens.append(tokens.to(torch.int32))
                    batch_cpu_masks.append(mask)
                    batch_cpu_hand_idx.append(hand_indices)
                    batch_hand_masks.append(mask[1:8].clone())

                    batch_tokens.append(tokens.to(device))
                    batch_masks.append(mask.to(device))
                    batch_hand_indices.append(hand_indices.to(device))
                    batch_legal.append(legal.to(device))

                batch_tokens_t = torch.stack(batch_tokens)
                batch_masks_t = torch.stack(batch_masks)
                batch_hand_indices_t = torch.stack(batch_hand_indices)
                batch_legal_t = torch.stack(batch_legal)

                torch.manual_seed(hash((base_seed, zeb_step)) & 0xFFFFFFFF)
                zeb_step += 1
                actions, _, _ = zeb_model.get_action(
                    batch_tokens_t, batch_masks_t, batch_hand_indices_t, batch_legal_t,
                    temperature=zeb_temperature,
                )

                for idx, game_idx in enumerate(zeb_indices):
                    state = states[game_idx]
                    player = zeb_players[idx]
                    action = int(actions[idx].item())

                    belief_targets, belief_mask = _belief_targets_from_owner(owners[game_idx], player)
                    policy = torch.zeros(7, dtype=torch.float32)
                    policy[action] = 1.0

                    per_game_records[game_idx].append({
                        'obs': batch_cpu_tokens[idx],
                        'mask': batch_cpu_masks[idx],
                        'hand_idx': batch_cpu_hand_idx[idx],
                        'hand_mask': batch_hand_masks[idx],
                        'policy': policy,
                        'belief_targets': belief_targets,
                        'belief_mask': belief_mask,
                        'player': player,
                    })

                    states[game_idx] = apply_action(state, action)
                    if is_terminal(states[game_idx]):
                        active[game_idx] = False

    obs_list = []
    mask_list = []
    hand_idx_list = []
    hand_mask_list = []
    policy_list = []
    value_list = []
    belief_targets_list = []
    belief_mask_list = []

    eq_wins = 0
    opp_wins = 0
    total_margin = 0

    for game_idx, final_state in enumerate(states):
        team0_pts, team1_pts = final_state.team_points
        eq_pts = team0_pts if eq_team == 0 else team1_pts
        opp_pts = team1_pts if eq_team == 0 else team0_pts
        if eq_pts > opp_pts:
            eq_wins += 1
        else:
            opp_wins += 1
        total_margin += eq_pts - opp_pts

        for rec in per_game_records[game_idx]:
            obs_list.append(rec['obs'])
            mask_list.append(rec['mask'])
            hand_idx_list.append(rec['hand_idx'])
            hand_mask_list.append(rec['hand_mask'])
            policy_list.append(rec['policy'])
            value_list.append(torch.tensor(get_outcome(final_state, rec['player']), dtype=torch.float32))
            belief_targets_list.append(rec['belief_targets'])
            belief_mask_list.append(rec['belief_mask'])

    examples = TrainingExamples(
        observations=torch.stack(obs_list),
        masks=torch.stack(mask_list),
        hand_indices=torch.stack(hand_idx_list),
        hand_masks=torch.stack(hand_mask_list),
        policy_targets=torch.stack(policy_list),
        value_targets=torch.stack(value_list),
        belief_targets=torch.stack(belief_targets_list),
        belief_mask=torch.stack(belief_mask_list),
    )
    stats = {
        'eq_wins': eq_wins,
        'opp_wins': opp_wins,
        'eq_win_rate': eq_wins / n_games,
        'avg_margin': total_margin / n_games,
        'n_games': n_games,
        'eq_team': eq_team,
    }
    return examples, stats


def evaluate_eq_vs_zeb(
    oracle_model,
    zeb_model,
    n_games: int = 1000,
    n_samples: int = 100,
    device: str = 'cuda',
    zeb_temperature: float = 0.1,
    batch_size: int = 0,
    verbose: bool = True,
) -> dict:
    """Evaluate E[Q] player vs Zeb (batched GPU).

    Runs games with both team assignments to eliminate seat advantage.

    Args:
        oracle_model: Stage 1 oracle for E[Q] (DominoLightningModule)
        zeb_model: ZebModel for opponent
        n_games: Total games (split evenly between team assignments)
        n_samples: Worlds sampled per E[Q] decision
        device: CUDA device
        zeb_temperature: Softmax temperature for Zeb (lower = more greedy)
        batch_size: Max games per GPU batch (0 = all at once). Limits VRAM
            usage to batch_size * n_samples allocations.
        verbose: Print progress

    Returns:
        Dict with win rates, margins, per-team breakdowns
    """
    half = n_games // 2

    def _run_chunked(n: int, eq_team: int, seed_offset: int) -> dict:
        chunk = batch_size if batch_size > 0 else n
        results = []
        seed = seed_offset
        remaining = n
        batch_num = 0
        while remaining > 0:
            this_batch = min(chunk, remaining)
            r = _run_eq_vs_zeb_batched(
                oracle_model, zeb_model, n_games=this_batch,
                n_samples=n_samples, eq_team=eq_team, base_seed=seed,
                device=device, zeb_temperature=zeb_temperature,
            )
            results.append(r)
            batch_num += 1
            done = n - remaining + this_batch
            agg = _merge_results(results)
            if verbose:
                print(f"    batch {batch_num}: {done}/{n} games, "
                      f"E[Q] {agg['eq_win_rate']:.1%}")
            seed += this_batch
            remaining -= this_batch
        return agg if len(results) > 1 else results[0]

    if verbose:
        print(f"  E[Q] as team 0 vs Zeb ({half} games)...")
    t0 = time.time()
    r0 = _run_chunked(half, eq_team=0, seed_offset=0)
    if verbose:
        print(f"    E[Q] {r0['eq_win_rate']:.1%}, {time.time()-t0:.1f}s")

    if verbose:
        print(f"  E[Q] as team 1 vs Zeb ({half} games)...")
    t1 = time.time()
    r1 = _run_chunked(half, eq_team=1, seed_offset=half)
    if verbose:
        print(f"    E[Q] {r1['eq_win_rate']:.1%}, {time.time()-t1:.1f}s")

    eq_wins = r0['eq_wins'] + r1['eq_wins']
    total = r0['n_games'] + r1['n_games']

    return {
        'eq_win_rate': eq_wins / total,
        'eq_wins': eq_wins,
        'total_games': total,
        'as_team0': r0,
        'as_team1': r1,
    }


def _run_eq_vs_zeb_batched(
    oracle_model,
    zeb_model,
    n_games: int,
    n_samples: int,
    eq_team: int,
    base_seed: int,
    device: str,
    zeb_temperature: float = 0.1,
) -> dict:
    """Run E[Q] as one team vs Zeb (batched GPU evaluation)."""
    states = [
        new_game(seed=game_seed(base_seed, i))
        for i in range(n_games)
    ]
    active = [True] * n_games
    zeb_step = 0  # For deterministic torch seeding

    # Pre-allocate E[Q] GPU resources
    sampler = WorldSamplerMRV(max_games=n_games, max_samples=n_samples, device=device)
    tokenizer = GPUTokenizer(max_batch=n_games * n_samples, device=device)

    oracle_model.eval()
    zeb_model.eval()
    zeb_model.to(device)

    while any(active):
        eq_indices = []
        eq_game_states = []
        zeb_indices = []
        zeb_game_states = []
        zeb_players = []

        for i, (state, is_active) in enumerate(zip(states, active)):
            if not is_active:
                continue
            player = _get_current_player(state)
            if player % 2 == eq_team:
                eq_indices.append(i)
                eq_game_states.append(state)
            else:
                zeb_indices.append(i)
                zeb_game_states.append(state)
                zeb_players.append(player)

        # Batched E[Q] decisions
        if eq_indices:
            n_eq = len(eq_indices)
            gst = zeb_states_to_game_state_tensor(eq_game_states, device)

            with torch.no_grad():
                worlds = sample_worlds_batched(gst, sampler, n_samples)
                deals = build_hypothetical_deals(gst, worlds)

                batch_needed = n_eq * n_samples
                if batch_needed > tokenizer.max_batch:
                    tokenizer = GPUTokenizer(max_batch=batch_needed, device=device)

                tokens, masks = tokenize_batched(gst, deals, tokenizer)
                q_values = query_model(
                    oracle_model, tokens, masks, gst, n_samples, device,
                )

                q_reshaped = q_values.view(n_eq, n_samples, 7)
                e_q = q_reshaped.mean(dim=1)
                e_q_pdf = compute_eq_pdf(q_reshaped)
                actions, _ = select_actions(gst, e_q, e_q_pdf, greedy=True)

            for idx, game_idx in enumerate(eq_indices):
                action = actions[idx].item()
                states[game_idx] = apply_action(states[game_idx], action)
                if is_terminal(states[game_idx]):
                    active[game_idx] = False

        # Batched Zeb decisions
        if zeb_indices:
            with torch.no_grad():
                batch_tokens = []
                batch_masks = []
                batch_hand_indices = []
                batch_legal = []

                for state, player in zip(zeb_game_states, zeb_players):
                    tokens, mask, hand_indices = observe(state, player)
                    legal = get_legal_mask(state, player)
                    batch_tokens.append(tokens)
                    batch_masks.append(mask)
                    batch_hand_indices.append(hand_indices)
                    batch_legal.append(legal)

                batch_tokens = torch.stack(batch_tokens).to(device)
                batch_masks = torch.stack(batch_masks).to(device)
                batch_hand_indices = torch.stack(batch_hand_indices).to(device)
                batch_legal = torch.stack(batch_legal).to(device)

                torch.manual_seed(hash((base_seed, zeb_step)) & 0xFFFFFFFF)
                zeb_step += 1
                actions, _, _ = zeb_model.get_action(
                    batch_tokens, batch_masks, batch_hand_indices, batch_legal,
                    temperature=zeb_temperature,
                )

                for idx, game_idx in enumerate(zeb_indices):
                    action = actions[idx].item()
                    states[game_idx] = apply_action(states[game_idx], action)
                    if is_terminal(states[game_idx]):
                        active[game_idx] = False

    # Score
    eq_wins = 0
    opp_wins = 0
    total_margin = 0

    for state in states:
        team0_pts, team1_pts = state.team_points
        eq_pts = team0_pts if eq_team == 0 else team1_pts
        opp_pts = team1_pts if eq_team == 0 else team0_pts

        if eq_pts > opp_pts:
            eq_wins += 1
        else:
            opp_wins += 1
        total_margin += eq_pts - opp_pts

    return {
        'eq_wins': eq_wins,
        'opp_wins': opp_wins,
        'eq_win_rate': eq_wins / n_games,
        'avg_margin': total_margin / n_games,
        'n_games': n_games,
        'eq_team': eq_team,
    }
