from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from .context import SeedContext, build_context
from .expand import expand_gpu
from .state import compute_level, compute_team, compute_terminal_value


@dataclass(frozen=True)
class SolveConfig:
    child_index_chunk_size: int = 1_000_000
    solve_chunk_size: int = 1_000_000
    enum_chunk_size: int = 100_000


def enumerate_gpu(ctx: SeedContext, config: SolveConfig | None = None, verbose: bool = True) -> torch.Tensor:
    initial = ctx.initial_state()
    frontier = initial.unsqueeze(0)
    all_levels: list[torch.Tensor] = [frontier]

    enum_chunk_size = config.enum_chunk_size if config else 0

    for depth in range(28, 0, -1):
        if verbose and depth % 2 == 0:
            total_so_far = sum(t.shape[0] for t in all_levels)
            print(f"  enum depth {depth:2d}: frontier={frontier.shape[0]:,} total={total_so_far:,}", flush=True)
        n = int(frontier.shape[0])
        if enum_chunk_size > 0 and n > enum_chunk_size:
            # Chunked expansion: unique per-chunk, then merge
            chunk_uniques: list[torch.Tensor] = []
            for start in range(0, n, enum_chunk_size):
                chunk = frontier[start : start + enum_chunk_size]
                c = expand_gpu(chunk, ctx)
                c = c[c >= 0]
                c = torch.unique(c)
                chunk_uniques.append(c)
            children = torch.unique(torch.cat(chunk_uniques))
        else:
            # Original path for small frontiers
            children = expand_gpu(frontier, ctx)
            children = children[children >= 0]
            children = torch.unique(children)

        all_levels.append(children)
        frontier = children

    return torch.sort(torch.cat(all_levels)).values


def build_child_index(all_states: torch.Tensor, ctx: SeedContext, config: SolveConfig) -> torch.Tensor:
    n = int(all_states.shape[0])
    child_idx = torch.empty((n, 7), dtype=torch.int32, device=all_states.device)
    chunk = int(config.child_index_chunk_size)
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        states_chunk = all_states[start:end]
        children = expand_gpu(states_chunk, ctx)
        idx = torch.searchsorted(all_states, children.clamp(min=0))
        idx = idx.to(torch.int32)
        idx = idx.masked_fill(children < 0, -1)
        child_idx[start:end] = idx
    return child_idx


def compute_transition_rewards(states: torch.Tensor, ctx: SeedContext) -> torch.Tensor:
    """Compute transition rewards for each (state, move) pair.

    For trick-completing moves (trick_len == 3): returns +points if Team 0 wins, -points if Team 1 wins.
    For mid-trick moves (trick_len != 3): returns 0.

    Returns (N, 7) int16 tensor of rewards.
    """
    leader = (states >> 28) & 0x3
    trick_len = (states >> 30) & 0x3
    p0 = (states >> 32) & 0x7
    p1 = (states >> 35) & 0x7
    p2 = (states >> 38) & 0x7

    completes = trick_len == 3
    n = states.shape[0]
    rewards = torch.zeros((n, 7), dtype=torch.int16, device=states.device)

    if not bool(completes.any()):
        return rewards

    # Only compute for completing states
    safe_p0 = p0.clamp(0, 6)
    safe_p1 = p1.clamp(0, 6)
    safe_p2 = p2.clamp(0, 6)

    for move in range(7):
        trick_idx = (leader * 2401 + safe_p0 * 343 + safe_p1 * 49 + safe_p2 * 7 + move).to(torch.int64)
        winner_offset = ctx.TRICK_WINNER[trick_idx].to(torch.int64)
        points = ctx.TRICK_POINTS[trick_idx].to(torch.int16)
        winner = (leader + winner_offset) & 0x3
        team0_wins = (winner & 0x1) == 0

        # +points if Team 0 wins, -points if Team 1 wins
        reward = torch.where(team0_wins, points, -points)
        # Only apply to completing moves
        rewards[:, move] = torch.where(completes, reward, torch.zeros_like(reward))

    return rewards


def solve_gpu(all_states: torch.Tensor, child_idx: torch.Tensor, ctx: SeedContext, config: SolveConfig):
    n = int(all_states.shape[0])
    v = torch.zeros(n, dtype=torch.int16, device=all_states.device)
    move_values = torch.full((n, 7), -128, dtype=torch.int16, device=all_states.device)

    level_of = compute_level(all_states)
    is_team0 = compute_team(all_states)

    terminal = level_of == 0
    if bool(terminal.any()):
        idx = terminal.nonzero(as_tuple=True)[0]
        v[idx] = compute_terminal_value(all_states[idx]).to(torch.int16)

    chunk = int(config.solve_chunk_size)
    for level in range(1, 29):
        idx_all = (level_of == level).nonzero(as_tuple=True)[0]
        if int(idx_all.numel()) == 0:
            continue

        for start in range(0, int(idx_all.numel()), chunk):
            idx = idx_all[start : start + chunk]
            states_chunk = all_states[idx]

            cidx32 = child_idx[idx]
            legal = cidx32 >= 0
            cidx64 = cidx32.clamp(min=0).to(torch.int64)

            # Get child values and add transition rewards
            cv = v[cidx64]
            rewards = compute_transition_rewards(states_chunk, ctx)
            cv_with_rewards = cv + rewards

            move_values[idx] = cv_with_rewards.masked_fill(~legal, -128)

            illegal = ~legal
            max_val = cv_with_rewards.masked_fill(illegal, -128).max(dim=1).values
            min_val = cv_with_rewards.masked_fill(illegal, 127).min(dim=1).values

            v[idx] = torch.where(is_team0[idx], max_val, min_val)

    return v, move_values


def solve_one_seed(
    seed: int,
    decl_id: int,
    output_path: Path | None,
    device: torch.device,
    config: SolveConfig,
):
    ctx = build_context(seed=seed, decl_id=decl_id, device=device)
    all_states = enumerate_gpu(ctx, config=config)
    child_idx = build_child_index(all_states, ctx, config=config)
    v, move_values = solve_gpu(all_states, child_idx, ctx, config=config)

    if output_path is not None:
        from .output import write_result

        write_result(output_path, seed, decl_id, all_states, v, move_values)

    return all_states, v, move_values
