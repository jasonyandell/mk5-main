"""Fate-head slough override on lens play (otis Phase 1 V2, issue #53).

Same trigger and scope as the tied-rollout variant (arena/slough_override.py):
void in the led suit, lens:ev's default is a non-trump junk slough, and >= 2
legal non-trump junk sloughs exist. Instead of tied rollouts, each candidate
discard is scored INSTANTLY by the play-state fate head (OtisPlayNet, built
2026-07-15 per the phase-1 pre-run amendment): featurize the post-discard
state from the actor's perspective and read

    score(c) = sum_t value(t) * P(my team captures t | s_c)
             + sum_k k * P(my-team tricks = k | s_c)

— the fate-ledger estimate of my team's total points (35 count + 7 trick
points = 42). One batched forward per trigger; per-trigger receipts stream
to the same JSONL shape as V1 (prices here are fate-ledger points).
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Sequence

import torch

from forge.zeb.game import apply_action
from forge.zeb.game import current_player as zeb_current_player
from forge.zeb.types import ZebGameState

from .lens_play import LensPlay
from .slough_override import TiedSloughPlay


class FateSloughPlay(LensPlay):
    """LensPlay + the instant fate-head retention override at slough decisions."""

    # Reuse V1's trigger predicate verbatim — one lever, two pricers.
    _trigger_candidates = TiedSloughPlay._trigger_candidates

    def __init__(
        self,
        model,
        utility: str = "ev",
        n_samples: int = 10,
        device: str = "mps",
        playnet_path: str = "otis/models/otis_play_v0.pt",
        stats_path: str | None = None,
        shadow: bool = False,
    ):
        super().__init__(model, utility=utility, n_samples=n_samples, device=device)
        from otis.play_model import OtisPlayNet

        self._playnet = OtisPlayNet.load(playnet_path, map_location="cpu")
        self._playnet.eval()
        self._stats_path = Path(stats_path) if stats_path else None
        self.shadow = shadow
        self.n_triggers = 0
        self.n_disagreements = 0

    def _price_and_choose(
        self, s: ZebGameState, candidates: list[int], default_dom: int,
    ) -> tuple[int, dict]:
        from otis.model import TILE_VALUES
        from otis.play_model import featurize_play_state

        P = zeb_current_player(s)
        t0 = time.time()
        feats = []
        for c in candidates:
            successor = apply_action(s, s.hands[P].index(c))
            feats.append(featurize_play_state(successor, perspective=P))
        x = torch.stack(feats)
        with torch.no_grad():
            out = self._playnet(x)
            fate_p = torch.softmax(out["fate"], dim=-1)  # [k, 5, 8]
            capture = fate_p[:, :, :4].sum(dim=-1)  # [k, 5] P(my team captures)
            trick_p = torch.softmax(out["trick"], dim=-1)  # [k, 8]
            e_tricks = (trick_p * torch.arange(8, dtype=torch.float32)).sum(dim=-1)
        values = torch.tensor(TILE_VALUES, dtype=torch.float32)
        scores = (capture * values).sum(dim=-1) + e_tricks  # [k]

        prices = {c: float(scores[i]) for i, c in enumerate(candidates)}
        chosen = max(candidates, key=lambda c: prices[c])
        receipt = {
            "trick_len": len(s.current_trick),
            "ply": len(s.play_history),
            "actor": P,
            "decl": s.decl_id,
            "candidates": candidates,
            "prices": {str(k): round(v, 3) for k, v in prices.items()},
            "default": default_dom,
            "chosen": chosen,
            "disagree": chosen != default_dom,
            "margin": round(prices[chosen] - prices[default_dom], 3),
            "secs": round(time.time() - t0, 4),
        }
        return chosen, receipt

    def choose(
        self,
        states: Sequence[ZebGameState],
        bid_values: Sequence[int],
        marks: Sequence[tuple[int, int]] | None = None,
        marks_to_win: int = 7,
    ) -> list[int]:
        actions = super().choose(states, bid_values, marks, marks_to_win)
        for i, s in enumerate(states):
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
        return f"FateSloughPlay(utility={self.utility!r}, n_samples={self.n_samples})"
