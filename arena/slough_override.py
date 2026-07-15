"""Tied-rollout slough override on lens play (otis Phase 1 V1, issue #53).

One flag on the incumbent: when the actor is void in the led suit, lens:ev's
default action is itself a non-trump junk slough, and >= 2 legal non-trump
junk sloughs exist (the otis-v0 W6 genuine-retention predicate), the discard
is chosen by tied-rollout retention price instead of the E[Q] argmax:

- M valid worlds are sampled live (repaired WorldSamplerMRV) and weighted by
  the gus belief head (the otis W5/W6 weighting);
- each candidate junk slough is forced as the root action of a
  tied-by-construction rollout (otis/tiedroll.py: every seat plays the gus
  pi_me argmax over its OWN info-state) on common random worlds;
- the candidate with the highest belief-weighted mean my-team points wins.

Everything else — bidder, utility, sampling, selection off-trigger — is
bit-identical to LensPlay. Per-trigger receipts stream to a JSONL sidecar
(M1 trigger rate, M2 disagreement, V1-m price margins).
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch

from forge.eq.generate.sampling import sample_worlds_batched
from forge.eq.sampling_mrv_gpu import WorldSamplerMRV
from forge.oracle.tables import can_follow, led_suit_for_lead_domino
from forge.zeb.eq_player import zeb_states_to_game_state_tensor
from forge.zeb.game import current_player as zeb_current_player
from forge.zeb.types import ZebGameState
from otis.fates import COUNT_TILE_IDS
from otis.tiedroll import (
    Commitment,
    PlayStub,
    _is_trump,
    aggregate_outcomes,
    belief_weights_for_decision,
    load_tied_policy,
    outcomes_from_trajectories,
    roll_worlds,
)

from .lens_play import LensPlay

DEFAULT_STUDENT = "gus/adapters/v3_consistency_10000g.pt"


@dataclass
class _GameStub:
    """Minimal game-record view of a live ZebGameState for the belief head."""

    hands: list[list[int]]
    decl_id: int
    decisions: list[PlayStub]


class TiedSloughPlay(LensPlay):
    """LensPlay + the tied-rollout retention override at slough decisions."""

    def __init__(
        self,
        model,
        utility: str = "ev",
        n_samples: int = 10,
        device: str = "mps",
        student_path: str = DEFAULT_STUDENT,
        m_worlds: int = 50,
        stats_path: str | None = None,
        shadow: bool = False,
    ):
        super().__init__(model, utility=utility, n_samples=n_samples, device=device)
        self._tied = load_tied_policy(student_path, device)
        self._m = m_worlds
        self._world_sampler: WorldSamplerMRV | None = None
        self._stats_path = Path(stats_path) if stats_path else None
        # Shadow mode: compute + record the override but PLAY the default —
        # the M1/M2 pre-gate measures the trigger/disagreement distribution
        # under the INCUMBENT's play distribution (registered protocol).
        self.shadow = shadow
        self.n_triggers = 0
        self.n_disagreements = 0
        self.n_decisions_seen = 0

    # ------------------------------------------------------------------ #
    # trigger                                                             #
    # ------------------------------------------------------------------ #

    def _trigger_candidates(self, s: ZebGameState, default_slot: int) -> list[int] | None:
        """W6 predicate + registered scope. Returns candidate junk dominoes,
        or None when the override does not fire."""
        if len(s.current_trick) == 0:
            return None  # leading — not a slough
        P = zeb_current_player(s)
        decl = s.decl_id
        led_suit = led_suit_for_lead_domino(s.current_trick[0], decl)
        remaining = [d for d in s.hands[P] if d not in s.played]
        if any(can_follow(d, led_suit, decl) for d in remaining):
            return None  # can follow — not void
        legal_doms = [
            s.hands[P][i]
            for i, d in enumerate(s.hands[P])
            if d not in s.played
        ]
        junk = [
            d for d in legal_doms
            if d not in COUNT_TILE_IDS and not _is_trump(d, decl)
        ]
        if len(junk) < 2:
            return None
        default_dom = s.hands[P][default_slot]
        if default_dom not in junk:
            # lens chose trump-in or a count slough — outside the registered
            # junk-retention scope; leave it.
            return None
        return junk

    # ------------------------------------------------------------------ #
    # pricing                                                             #
    # ------------------------------------------------------------------ #

    def _price_and_choose(
        self, s: ZebGameState, candidates: list[int], default_dom: int,
    ) -> tuple[int, dict]:
        P = zeb_current_player(s)
        decl, bidder = s.decl_id, s.bidder
        t0 = time.time()

        # 1. Live worlds (repaired sampler; validity by construction).
        if self._world_sampler is None:
            self._world_sampler = WorldSamplerMRV(
                max_games=1, max_samples=self._m, device=self.device,
            )
        gst = zeb_states_to_game_state_tensor([s], self.device)
        with torch.no_grad():
            worlds = sample_worlds_batched(gst, self._world_sampler, self._m)
        wh = worlds[0].long().cpu()  # [M, 3, 7], -1 padded

        # 2. Reconstruct full initial deals per world.
        played_by = [[d for (p, d) in s.play_history if p == seat] for seat in range(4)]
        deals: list[list[list[int]]] = []
        for m in range(wh.shape[0]):
            hands_m: list[list[int]] = [None] * 4  # type: ignore[list-item]
            hands_m[P] = list(s.hands[P])
            for r in range(3):
                opp = (P + r + 1) % 4
                sampled = [int(d) for d in wh[m, r].tolist() if d >= 0]
                hands_m[opp] = played_by[opp] + sampled
            deals.append(hands_m)

        # 3. Belief weights over the sampled worlds (live info-state stub).
        prefix = list(s.play_history)
        stub_decisions = [
            PlayStub(p, s.hands[p].index(d)) for (p, d) in prefix
        ] + [PlayStub(P, 0)]
        stub = _GameStub(
            hands=[list(h) for h in s.hands],
            decl_id=decl,
            decisions=stub_decisions,
        )
        w_belief, ess = belief_weights_for_decision(
            self._tied.student, self._tied.is_voids, stub, len(prefix), wh,
        )

        # 4. Tied rollout per candidate on common random worlds.
        prices: dict[int, float] = {}
        for c in candidates:
            traj = roll_worlds(
                deals, decl, bidder, prefix, P, self._tied,
                Commitment(label=f"root-{c}", root_domino=c),
            )
            outcomes = outcomes_from_trajectories(
                traj, deals, decl, bidder, P, list(range(len(deals))),
            )
            agg = aggregate_outcomes(outcomes, w_belief, ess)
            prices[c] = agg.mean_my_points_belief

        chosen = max(candidates, key=lambda c: prices[c])
        receipt = {
            "trick_len": len(s.current_trick),
            "ply": len(prefix),
            "actor": P,
            "decl": decl,
            "candidates": candidates,
            "prices": {str(k): round(v, 3) for k, v in prices.items()},
            "default": default_dom,
            "chosen": chosen,
            "disagree": chosen != default_dom,
            "margin": round(prices[chosen] - prices[default_dom], 3),
            "ess": round(float(ess), 2),
            "secs": round(time.time() - t0, 3),
        }
        return chosen, receipt

    # ------------------------------------------------------------------ #
    # policy                                                              #
    # ------------------------------------------------------------------ #

    def choose(
        self,
        states: Sequence[ZebGameState],
        bid_values: Sequence[int],
        marks: Sequence[tuple[int, int]] | None = None,
        marks_to_win: int = 7,
    ) -> list[int]:
        actions = super().choose(states, bid_values, marks, marks_to_win)
        for i, s in enumerate(states):
            self.n_decisions_seen += 1
            candidates = self._trigger_candidates(s, actions[i])
            if candidates is None:
                continue
            P = zeb_current_player(s)
            default_dom = s.hands[P][actions[i]]
            chosen_dom, receipt = self._price_and_choose(s, candidates, default_dom)
            self.n_triggers += 1
            if receipt["disagree"]:
                self.n_disagreements += 1
                if not self.shadow:
                    actions[i] = s.hands[P].index(chosen_dom)
            if self._stats_path is not None:
                with open(self._stats_path, "a") as f:
                    f.write(json.dumps(receipt) + "\n")
        return actions

    def __repr__(self) -> str:
        return (
            f"TiedSloughPlay(utility={self.utility!r}, n_samples={self.n_samples}, "
            f"m_worlds={self._m})"
        )
