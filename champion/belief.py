"""Belief posterior extraction for belief-weighted world sampling (rung #25).

The play-evidence Gus belief head (`gus.model.student` BeliefHead) outputs, per
unseen domino, P(seat) over the three RELATIVE opponents (current+1, +2, +3) —
exactly the seat order the forge MRV world sampler uses for its three opponent
rows (`worlds[n, m, rel, slot]`). So a uniformly-sampled world's log-probability
under the belief is the sum of its per-tile seat log-probs, and importance-weighting
the (validity-guaranteed) MRV worlds by that posterior biases the oracle's E[Q]
toward belief-likely worlds without touching the sampler.

`belief_model=None` yields uniform weights, recovering plain LensPlay exactly.
The seat-row alignment holds only because both belief and worlds are taken from
the SAME current-player POV — see [[champion]] rung #25.
"""
from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor

from forge.zeb.game import current_player
from forge.zeb.types import ZebGameState
from gus.model.tokenize import tokenize_decision
from gus.model.voids import voids_feature_vector


class _Stub:
    """Minimal decision proxy: gus tokenization reads .player and .action_taken
    (a slot index into the player's initial hand)."""

    __slots__ = ("player", "action_taken")

    def __init__(self, player: int, action_taken: int):
        self.player = player
        self.action_taken = action_taken


def _decisions_and_idx(state: ZebGameState):
    """Reconstruct the gus `decisions` list (action_taken = slot in the initial
    hand) plus the current-decision index and acting player, from a ZebGameState.

    `play_history` stores `(player, domino_id)`; the slot is the domino's index in
    that player's immutable initial hand."""
    decisions: list[_Stub] = []
    for (player, dom) in state.play_history:
        slot = list(state.hands[player]).index(int(dom))
        decisions.append(_Stub(int(player), slot))
    cp = current_player(state)
    d_idx = len(decisions)
    decisions.append(_Stub(cp, 0))  # current decision; action_taken unused
    return decisions, d_idx, cp


def load_belief(adapter: str | None, device: str):
    """Load a Gus adapter as a belief model. Returns (model, is_voids)."""
    from gus.bidding.evaluate import ADAPTER_DEFAULT, load_gus

    return load_gus(str(adapter or ADAPTER_DEFAULT), device)


def belief_logits_for_states(
    model, is_voids: bool, states: Sequence[ZebGameState], device: str, *, tau: float = 1.0
) -> Tensor:
    """Batched belief forward: log P(seat | domino) as [n, 28, 3].

    The belief head depends only on the pooled state embedding, so a zeros
    placeholder world drives the forward pass (lamir1 does the same)."""
    toks, attns, voids_rows = [], [], []
    for st in states:
        decisions, d_idx, cp = _decisions_and_idx(st)
        tk, am = tokenize_decision([list(h) for h in st.hands], int(st.decl_id), decisions, d_idx)
        toks.append(tk)
        attns.append(am)
        if is_voids:
            voids_rows.append(
                voids_feature_vector(list(st.play_history), int(st.decl_id), cp)
            )
    tokens = torch.stack(toks).to(device)  # [n, 33, 5]
    attn = torch.stack(attns).to(device)   # [n, 33]
    n = tokens.shape[0]
    placeholder_world = torch.zeros(n, 28, 3, device=device)
    if is_voids:
        voids = torch.stack(voids_rows).to(device)  # [n, 24]
        out = model(tokens, attn, placeholder_world, voids)
    else:
        out = model(tokens, attn, placeholder_world)
    return torch.log_softmax(out["belief_logits"] / tau, dim=-1)  # [n, 28, 3]


def weights_from_logP(logP: Tensor, worlds: Tensor, uniform_mix: float = 0.1) -> Tensor:
    """Per-world importance weights w[n, M] from a belief log-posterior.

    For each MRV world, sum the belief log P(seat=rel | domino) over its assigned
    (domino, relative-opponent) pairs, softmax over the M worlds, then mix with a
    uniform floor to bound effective sample size. Pure tensor math, model-free.

    Args:
        logP:   [n, 28, 3] log P(seat | domino), seats = relative opponents.
        worlds: [n, M, 3, 7] domino ids (-1 padded), dim 2 = the 3 relative opponents.
    Returns:
        w: [n, M] a distribution over worlds per game (sums to 1).
    """
    n, M = worlds.shape[0], worlds.shape[1]
    logP = logP.to(worlds.device)
    valid = worlds >= 0  # [n, M, 3, 7]
    safe = worlds.clamp(min=0).long()
    rel = torch.arange(3, device=worlds.device).view(1, 1, 3, 1).expand(n, M, 3, 7)
    n_idx = torch.arange(n, device=worlds.device).view(n, 1, 1, 1).expand(n, M, 3, 7)
    gathered = logP[n_idx, safe, rel]  # [n, M, 3, 7]
    gathered = torch.where(valid, gathered, torch.zeros_like(gathered))
    logw = gathered.sum(dim=(2, 3))  # [n, M]
    w = torch.softmax(logw, dim=1)
    return (1.0 - uniform_mix) * w + uniform_mix / M


def belief_weights_for_worlds(
    model, is_voids: bool, states, worlds: Tensor, device: str,
    *, uniform_mix: float = 0.1, tau: float = 1.0,
) -> Tensor:
    """Per-world weights [n, M] from the belief head; uniform 1/M when model=None."""
    n, M = worlds.shape[0], worlds.shape[1]
    if model is None:
        return torch.full((n, M), 1.0 / M, device=worlds.device)
    logP = belief_logits_for_states(model, is_voids, states, device, tau=tau)
    return weights_from_logP(logP, worlds, uniform_mix)


def effective_sample_size(w: Tensor) -> Tensor:
    """ESS per game: 1 / sum(w^2), in [1, M]. Diagnostic for weight collapse."""
    return 1.0 / (w ** 2).sum(dim=1).clamp(min=1e-12)
