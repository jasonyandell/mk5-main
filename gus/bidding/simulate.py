"""All-gus self-play simulator: roll out full 42 games and return team-0 points.

Reuses forge/eq/GameStateTensor for engine correctness (legal actions, trick
resolution). Unlike gus/eval/arena.py, this puts gus in all four seats and
therefore skips the oracle entirely.

`simulate_all_gus_batch` packs games for multiple declarations into a single
GameStateTensor so the model runs one big forward pass per decision step
instead of one per declaration.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from forge.eq.game_tensor import GameStateTensor
from forge.oracle.tables import resolve_trick

from gus.model.features import reconstruct_prior_plays
from gus.model.tokenize import tokenize_decision
from gus.model.voids import voids_feature_vector


@dataclass
class _Decision:
    player: int
    action_taken: int


def _pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _deal_random(bidder_hand: list[int], n_games: int, rng: random.Random) -> list[list[list[int]]]:
    """Deal 3 opponent hands for each of n_games, keeping bidder at seat 0."""
    others_base = [d for d in range(28) if d not in set(bidder_hand)]
    if len(others_base) != 21:
        raise ValueError("bidder hand must have 7 unique dominoes")
    games: list[list[list[int]]] = []
    for _ in range(n_games):
        shuf = others_base[:]
        rng.shuffle(shuf)
        games.append([
            sorted(bidder_hand),
            sorted(shuf[0:7]),
            sorted(shuf[7:14]),
            sorted(shuf[14:21]),
        ])
    return games


@torch.no_grad()
def _gus_actions(
    model,
    is_voids: bool,
    states: GameStateTensor,
    hands: list[list[list[int]]],
    decl_ids: list[int],
    game_decisions: list[list[_Decision]],
    active_mask: Tensor,
    device: str,
) -> Tensor:
    """Ask gus for an action in every active game. Returns [n_games] long.

    `decl_ids` is one declaration per game so games for different trumps can
    share a single forward pass.
    """
    n_games = states.n_games
    actions = torch.zeros(n_games, dtype=torch.long)

    # One CPU pull per tick for the per-game scalars; avoids 800× .item() calls.
    active_list = active_mask.tolist()
    if not any(active_list):
        return actions
    legal_cpu = states.legal_actions().cpu()
    current_all_list = states.current_player.cpu().long().tolist()

    tokens_list: list[Tensor] = []
    mask_list: list[Tensor] = []
    voids_list: list[Tensor] = []
    legal_list: list[Tensor] = []
    game_indices: list[int] = []

    for g in range(n_games):
        if not active_list[g]:
            continue
        decisions = game_decisions[g]
        d_idx = len(decisions)
        current_player = current_all_list[g]
        decl_id = decl_ids[g]
        # tokenize_decision reads decisions[d_idx].player; stub it.
        stubbed = decisions + [_Decision(player=current_player, action_taken=0)]
        tokens, attn = tokenize_decision(hands[g], decl_id, stubbed, d_idx)
        prior_plays = reconstruct_prior_plays(hands[g], stubbed, d_idx)
        voids = voids_feature_vector(prior_plays, decl_id, current_player)

        tokens_list.append(tokens)
        mask_list.append(attn)
        voids_list.append(voids)
        legal_list.append(legal_cpu[g])
        game_indices.append(g)

    if not game_indices:
        return actions

    # Stack on CPU then transfer in ONE .to() call per tensor.
    batch_tokens = torch.stack(tokens_list).to(device)
    batch_masks = torch.stack(mask_list).to(device)
    batch_voids = torch.stack(voids_list).to(device)
    batch_legal = torch.stack(legal_list).to(device)
    B = batch_tokens.shape[0]
    world = torch.zeros(B, 28, 3, dtype=torch.float32, device=device)

    if is_voids:
        out = model(batch_tokens, batch_masks, world, batch_voids)
    else:
        out = model(batch_tokens, batch_masks, world)
    pi_logits = out["pi_me_logits"]  # [B, 7]

    # Batched masked argmax — single GPU op + single sync at the end.
    masked = pi_logits.masked_fill(~batch_legal, float("-inf"))
    picks = masked.argmax(dim=-1).cpu().tolist()
    for b, g in enumerate(game_indices):
        actions[g] = picks[b]
    return actions


def _score_games(states: GameStateTensor, decl_ids: list[int]) -> Tensor:
    """Walk history tensor → [n_games, 2] team points."""
    n_games = states.n_games
    scores = torch.zeros(n_games, 2, dtype=torch.int32)
    history = states.history.cpu()

    for g in range(n_games):
        h = history[g]
        decl_id = decl_ids[g]
        for t in range(7):
            base = 4 * t
            dominoes = tuple(int(h[base + i, 1].item()) for i in range(4))
            if any(d < 0 for d in dominoes):
                continue
            lead = int(h[base, 1].item())
            outcome = resolve_trick(lead, dominoes, decl_id)
            leader = int(h[base, 0].item())
            winner = (leader + outcome.winner_offset) % 4
            scores[g, winner % 2] += outcome.points
    return scores


def simulate_all_gus_batch(
    model,
    is_voids: bool,
    bidder_hand: list[int],
    trump_ids: list[int],
    n_per_trump: int,
    device: str,
    seed: int = 42,
) -> dict[int, Tensor]:
    """Roll out n_per_trump games for each trump, batched into one tensor.

    Per-trump deals use rng `seed + decl_id` so results match the per-trump
    sequential path byte-for-byte (same deals, deterministic policy).

    Returns {decl_id: bidder_pts_tensor[n_per_trump]}.
    """
    all_hands: list[list[list[int]]] = []
    all_decls: list[int] = []
    block_starts: dict[int, int] = {}
    for decl_id in trump_ids:
        rng = random.Random(seed + decl_id)
        hands_block = _deal_random(bidder_hand, n_per_trump, rng)
        block_starts[decl_id] = len(all_hands)
        all_hands.extend(hands_block)
        all_decls.extend([decl_id] * n_per_trump)

    n_total = len(all_hands)
    states = GameStateTensor.from_deals(all_hands, all_decls, device=device)
    game_decisions: list[list[_Decision]] = [[] for _ in range(n_total)]

    while bool(states.active_games().any().item()):
        active_cpu = states.active_games().cpu()
        active_list = active_cpu.tolist()
        current_list = states.current_player.cpu().long().tolist()
        picks = _gus_actions(
            model, is_voids, states, all_hands, all_decls,
            game_decisions, active_cpu, device,
        )
        picks_list = picks.tolist()
        for g in range(n_total):
            if not active_list[g]:
                continue
            game_decisions[g].append(_Decision(
                player=current_list[g],
                action_taken=picks_list[g],
            ))
        states = states.apply_actions(picks.to(device))

    scores = _score_games(states, all_decls)
    bidder_pts = scores[:, 0]

    return {
        decl: bidder_pts[block_starts[decl]: block_starts[decl] + n_per_trump]
        for decl in trump_ids
    }
