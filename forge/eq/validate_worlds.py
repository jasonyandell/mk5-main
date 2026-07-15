"""Write-time validity checks for stored joint-world tensors.

Issue #52: the April eq corpus fossilized 27-67% malformed sampled worlds per
decision because the pre-repair sampler injected domino 0-0 instead of
rejecting, and nothing at write time noticed. These checks make that class of
rot impossible to store silently: the generator asserts every stored world is
a real 28-domino deal at record time.

A stored world for a decision is VALID iff:
- its in-range entries form EXACTLY the decision's unseen set (each unseen
  domino appears exactly once across the three opponent rows; no seen domino
  appears at all), and
- each relative-seat row carries exactly that opponent's remaining hand count.

The read-side referee that grades regenerated corpora intentionally does NOT
reuse this module (independence discipline, issue #55): it re-derives validity
from the saved records alone.
"""

from __future__ import annotations

import torch
from torch import Tensor

from forge.eq.game_tensor import GameStateTensor

N_DOMINOES = 28


def stored_world_validity(states: GameStateTensor, worlds: Tensor) -> Tensor:
    """Vectorized validity mask for sampled worlds about to be stored.

    Args:
        states: GameStateTensor with N games (source of truth for the unseen
            set: current hands + played mask + current player)
        worlds: [N, M, 3, 7] opponent hands per sampled world, -1 padded

    Returns:
        [N, M] bool — True where world m is a real completion of game n's
        unseen set.
    """
    n_games, n_worlds = worlds.shape[0], worlds.shape[1]
    device = worlds.device

    current_players = states.current_player.long()  # [N]

    # My remaining hand as a [N, 28] mask.
    player_idx = current_players.view(n_games, 1, 1).expand(n_games, 1, 7)
    my_hands = torch.gather(states.hands.long(), 1, player_idx).squeeze(1)  # [N, 7]
    my_mask = torch.zeros(n_games, N_DOMINOES, dtype=torch.bool, device=device)
    in_hand = my_hands >= 0
    batch_idx = torch.arange(n_games, device=device).unsqueeze(1).expand_as(my_hands)
    my_mask[batch_idx[in_hand], my_hands[in_hand]] = True

    # Unseen = not played and not in my hand.
    unseen = ~states.played_mask.to(device=device, dtype=torch.bool) & ~my_mask  # [N, 28]

    # Occupancy of each world over the 28 dominoes.
    flat = worlds.reshape(n_games, n_worlds, 21).long()  # [N, M, 21]
    in_range = (flat >= 0) & (flat < N_DOMINOES)
    occ = torch.zeros(
        n_games, n_worlds, N_DOMINOES, dtype=torch.int32, device=device
    )
    occ.scatter_add_(2, flat.clamp(0, N_DOMINOES - 1), in_range.to(torch.int32))

    # Exact cover: occupancy 1 on unseen, 0 elsewhere.
    cover_ok = (occ == unseen.to(torch.int32).unsqueeze(1)).all(dim=2)  # [N, M]

    # Row cardinality: each relative seat r holds opponent (P+r+1)%4's
    # remaining count.
    hand_counts = (states.hands >= 0).sum(dim=2)  # [N, 4]
    offsets = torch.arange(1, 4, device=states.hands.device).unsqueeze(0)
    opp_idx = (states.current_player.unsqueeze(1) + offsets) % 4  # [N, 3]
    expected = torch.gather(hand_counts, 1, opp_idx.long()).to(device)  # [N, 3]
    row_counts = in_range.reshape(n_games, n_worlds, 3, 7).sum(dim=3)  # [N, M, 3]
    rows_ok = (row_counts == expected.unsqueeze(1)).all(dim=2)  # [N, M]

    return cover_ok & rows_ok


def assert_stored_worlds_valid(
    states: GameStateTensor,
    worlds: Tensor,
    active: Tensor | None = None,
    context: str = "",
) -> None:
    """Raise if any world about to be stored is not a real 28-domino deal.

    Args:
        states: GameStateTensor for the batch
        worlds: [N, M, 3, 7] sampled opponent hands
        active: optional [N] bool — restrict the check to active games
        context: appears in the error message (e.g. decision index)
    """
    valid = stored_world_validity(states, worlds)  # [N, M]
    if active is not None:
        valid = valid | ~active.to(valid.device).unsqueeze(1)
    if bool(valid.all()):
        return
    bad_per_game = (~valid).sum(dim=1)  # [N]
    bad_games = torch.nonzero(bad_per_game, as_tuple=False).flatten().tolist()
    detail = ", ".join(
        f"game {g}: {int(bad_per_game[g])}/{worlds.shape[1]} invalid"
        for g in bad_games[:8]
    )
    raise RuntimeError(
        f"Refusing to store malformed sampled worlds ({context or 'write time'}): "
        f"{int((~valid).sum())} invalid worlds across {len(bad_games)} games "
        f"[{detail}{', ...' if len(bad_games) > 8 else ''}]. "
        f"A stored world must be a real 28-domino deal (issue #52)."
    )


def assert_world_weights_normalized(
    weights: Tensor,
    active: Tensor | None = None,
    atol: float = 1e-4,
    context: str = "",
) -> None:
    """Raise if per-world weights are not a distribution per game.

    Args:
        weights: [N, M] per-world posterior weights
        active: optional [N] bool — restrict the check to active games
        atol: allowed |sum - 1| per game
        context: appears in the error message
    """
    sums = weights.sum(dim=1)  # [N]
    ok = (weights >= 0).all(dim=1) & ((sums - 1.0).abs() <= atol)
    if active is not None:
        ok = ok | ~active.to(ok.device)
    if bool(ok.all()):
        return
    bad = torch.nonzero(~ok, as_tuple=False).flatten().tolist()
    raise RuntimeError(
        f"Refusing to store unnormalized world weights ({context or 'write time'}): "
        f"games {bad[:8]} have sum/negativity violations "
        f"(sums: {[float(sums[g]) for g in bad[:8]]})."
    )
