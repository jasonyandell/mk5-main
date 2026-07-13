"""JudSearch (jud v1, JS1): value at the leaves of shallow belief-state search.

Where `JudPlay` prices each legal move by one net query on the post-move
info-state, JudSearch looks the CURRENT TRICK through to resolution: for each
root state it samples N consistent worlds (the same MRV belief lift LensPlay
uses — completions of the hidden hands given voids and the play history),
then inside each world it applies each candidate move and lets the remaining
seats of the trick reply greedily via the jud head — each simulated seat
featurizing ITS OWN info-state within the sampled world (its sampled hand +
the public history), offense argmax E[pts], defense argmin: exactly
JudPlay's rule. At trick resolution the leaf info-state is evaluated from
the ROOT mover's POV with the same head, E[pts] is averaged over the N
worlds, and the root picks argmax (defenders argmin). No oracle anywhere:
belief enters only through the sampled worlds, value only through the head.

Batching: all (state × legal-move × world) rollouts advance in lockstep —
each of the ≤3 reply steps is ONE net forward over every active rollout's
legal children (forced follows are applied without a forward), plus one
final leaf forward. Worlds are sampled once per root state and shared
across its candidate moves: the belief does not depend on the move, and
common worlds cancel sampling noise out of the move comparison. The
determinized state reconstructs each hidden seat's original 7-tile hand as
its sampled remaining tiles plus the tiles it already played (public), so
the engine's own legality and trick resolution drive the rollout unchanged.

The MRV sampler consumes the global torch RNG (as everywhere else in the
arena), so batch composition changes the sampled worlds; the deterministic
core `_choose_given_worlds` is where batched == sequential holds exactly.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

import torch
from torch import Tensor

from arena.lens_play import _max_pool_size
from champion.jud_net import JudNet, featurize_state, mean_points
from forge.eq.generate.sampling import sample_worlds_batched
from forge.eq.sampling_mrv_gpu import WorldSamplerMRV
from forge.zeb.eq_player import zeb_states_to_game_state_tensor
from forge.zeb.game import apply_action, current_player, legal_actions
from forge.zeb.types import ZebGameState


@dataclass
class _Rollout:
    """One determinized line: root (state, move, world), advanced in place."""

    state: ZebGameState
    steps: int  # hidden replies left before the current trick resolves
    pov: int    # root mover — the seat whose info-state the leaf is priced for


class JudSearch:
    """Trick-resolution search over sampled worlds, jud head at every seat."""

    def __init__(self, model: JudNet, n_worlds: int = 10, device: str = "cpu"):
        self.model = model
        self.n_worlds = n_worlds
        self.device = device
        self._sampler: WorldSamplerMRV | None = None
        self.model.eval()

    def choose(
        self,
        states: Sequence[ZebGameState],
        bid_values: Sequence[int],
        marks: Sequence[tuple[int, int]] | None = None,
        marks_to_win: int = 7,
    ) -> list[int]:
        return self._choose_given_worlds(states, self._sample_worlds(states))

    def _sample_worlds(self, states: Sequence[ZebGameState]) -> Tensor:
        """[n_states, n_worlds, 3, 7] hidden-seat completions (-1 padded),
        seat order relative to each state's mover — LensPlay's exact lift."""
        n = len(states)
        if self._sampler is None or self._sampler.max_games < n:
            self._sampler = WorldSamplerMRV(
                max_games=n, max_samples=self.n_worlds, device=self.device,
            )
        gst = zeb_states_to_game_state_tensor(list(states), self.device)
        with torch.no_grad():
            worlds = sample_worlds_batched(
                gst, self._sampler, self.n_worlds,
                max_pool_size=_max_pool_size(states),
            )
        return worlds.cpu()

    def _choose_given_worlds(
        self, states: Sequence[ZebGameState], worlds: Tensor,
    ) -> list[int]:
        """The deterministic core: given the worlds, choices are a pure
        function of the states — identical batched or one state at a time."""
        rollouts: list[_Rollout] = []
        plans: list[tuple[tuple[int, ...], float, int]] = []  # (legal, sign, start)
        for i, s in enumerate(states):
            mover = current_player(s)
            legal = legal_actions(s)
            sign = 1.0 if mover % 2 == s.bidder % 2 else -1.0
            plans.append((legal, sign, len(rollouts)))
            if len(legal) == 1:
                continue  # forced move: nothing to search
            steps = 3 - len(s.current_trick)
            for m in legal:
                for w in range(self.n_worlds):
                    det = replace(s, hands=self._determinize(s, mover, worlds[i, w]))
                    rollouts.append(_Rollout(apply_action(det, m), steps, mover))

        # Lockstep in-trick replies: ≤3 steps, one forward per step.
        for _ in range(3):
            feats: list[Tensor] = []
            spans: list[tuple[_Rollout, list[ZebGameState], float, int]] = []
            for r in rollouts:
                if r.steps == 0:
                    continue
                r.steps -= 1
                legal = legal_actions(r.state)
                if len(legal) == 1:  # forced follow: no query needed
                    r.state = apply_action(r.state, legal[0])
                    continue
                mover = current_player(r.state)
                children = [apply_action(r.state, a) for a in legal]
                sign = 1.0 if mover % 2 == r.state.bidder % 2 else -1.0
                spans.append((r, children, sign, len(feats)))
                feats.extend(featurize_state(c, seat=mover) for c in children)
            if not feats:
                continue
            ev = self._ev(feats)
            for r, children, sign, start in spans:
                best = torch.argmax(sign * ev[start:start + len(children)])
                r.state = children[int(best)]

        # Leaf: post-trick-resolution info-states, root mover's POV.
        ev = (
            self._ev([featurize_state(r.state, seat=r.pov) for r in rollouts])
            if rollouts else torch.empty(0)
        )
        return [
            legal[0] if len(legal) == 1 else legal[int(torch.argmax(
                sign * ev[start:start + len(legal) * self.n_worlds]
                .view(len(legal), self.n_worlds).mean(dim=1)
            ))]
            for legal, sign, start in plans
        ]

    def _determinize(
        self, state: ZebGameState, mover: int, world: Tensor,
    ) -> tuple[tuple[int, ...], ...]:
        """Full deal for one sampled world: each hidden seat's original hand is
        its sampled remaining tiles plus the tiles it already played (public)."""
        hands = list(state.hands)
        for i, row in enumerate(world.tolist()):
            q = (mover + 1 + i) % 4
            played_by_q = [d for p, d in state.play_history if p == q]
            hands[q] = tuple(int(d) for d in row if d >= 0) + tuple(played_by_q)
        return tuple(hands)

    def _ev(self, feats: list[Tensor]) -> Tensor:
        x = torch.stack(feats).to(self.device)
        with torch.no_grad():
            return mean_points(self.model(x)).cpu()

    def __repr__(self) -> str:
        return f"JudSearch(n_worlds={self.n_worlds})"
