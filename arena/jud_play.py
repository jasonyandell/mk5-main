"""Jud-backed play policy (jud v1): value-native play, no worlds, no oracle.

At each decision the mover prices every legal move by querying `JudNet` on the
info-state AFTER that move — still the mover's own POV, exactly the
"evaluation" rows the net trains on (`champion.jud_net.hand_samples`) — and
takes the argmax of E[declaring-team points]. The net's target is oriented to
the BIDDING team, so a defender flips the sign and minimizes it; the final
margin is affine in E[pts] (margin = 2·E[pts] − 42), so argmax E[pts] is
argmax EV of the margin distribution.

Where `LensPlay` samples consistent worlds and asks a perfect-information
oracle, JudPlay asks one learned function of the information actually held —
belief is implicit in the conditioning on everyone's bids and plays. All
legal-move queries across the whole batch of games go through ONE forward
pass per tick.
"""
from __future__ import annotations

from typing import Sequence

import torch

from champion.jud_net import JudNet, featurize_state, mean_points
from forge.zeb.game import apply_action, current_player, legal_actions
from forge.zeb.types import ZebGameState


class JudPlay:
    """Argmax-E[pts] player over the jud realized-value head (defenders minimize)."""

    def __init__(self, model: JudNet, device: str = "cpu"):
        self.model = model
        self.device = device
        self.model.eval()

    def choose(
        self,
        states: Sequence[ZebGameState],
        bid_values: Sequence[int],
        marks: Sequence[tuple[int, int]] | None = None,
        marks_to_win: int = 7,
    ) -> list[int]:
        # One flat batch of every legal child across every game this tick.
        feats: list[torch.Tensor] = []
        spans: list[tuple[int, tuple[int, ...], float]] = []  # (start, legal, sign)
        for s in states:
            mover = current_player(s)
            legal = legal_actions(s)
            sign = 1.0 if mover % 2 == s.bidder % 2 else -1.0
            spans.append((len(feats), legal, sign))
            feats.extend(
                featurize_state(apply_action(s, a), seat=mover) for a in legal
            )

        x = torch.stack(feats).to(self.device)
        with torch.no_grad():
            ev = mean_points(self.model(x)).cpu()

        return [
            legal[int(torch.argmax(sign * ev[start:start + len(legal)]))]
            for start, legal, sign in spans
        ]

    def __repr__(self) -> str:
        return "JudPlay()"
