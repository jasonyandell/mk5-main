"""Score-conditioned play risk (rung #27 v2).

The play-phase counterpart to MarksToSeven's bid conditioning: a LensPlay that
picks its utility lens per game from the live mark score via
``champion.utility.score_to_utility`` — protect a lead with the lower-tail-averse
``cvar_10``, chase from behind with ``upside_10``, maximize EV near even. It
reads the same ``race_wp`` table as the bidder, so the champion's play risk and
bid risk move together rather than as two ad-hoc policies.

Only the action-selection seam changes; world sampling and marginalization are
the base LensPlay's, so with no score supplied this is exactly ``LensPlay('ev')``.
"""
from __future__ import annotations

import torch

from arena.lens_play import LensPlay
from champion.utility import score_to_utility
from w42.lens_v1.lens import utility_scores


def select_by_score(chosen, e_q, e_q_pdf, bidder, current_players, bid_values, legal_mask):
    """Argmax legal action per game, each scored under its own chosen utility.

    ``chosen[g]`` names the lens for game g. Runs one ``utility_scores`` pass per
    distinct lens (a small constant — at most the size of UTILITIES), gathers each
    game's row from its lens, masks illegal slots, and argmaxes with the lowest-slot
    tie-break ``argmax_under_utility`` uses. Pure tensor in/out, model-free —
    the testable core of the score-conditioned play policy.
    """
    n = e_q.shape[0]
    bv = list(bid_values)
    final = torch.empty_like(e_q)  # [n, 7]
    for u in set(chosen):
        scores_u = utility_scores(u, e_q, e_q_pdf, bidder, current_players, bv)
        idx = torch.tensor(
            [g for g in range(n) if chosen[g] == u],
            device=e_q.device, dtype=torch.long,
        )
        final[idx] = scores_u[idx]
    masked = torch.where(legal_mask, final, torch.full_like(final, float("-inf")))
    return masked.argmax(dim=1).long()


class ScoreConditionedLensPlay(LensPlay):
    """LensPlay that selects its utility per game by marks-to-7 win probability."""

    def __init__(
        self,
        model,
        *,
        n_samples: int = 10,
        device: str = "mps",
        band: float = 0.15,
        ahead: str = "cvar_10",
        even: str = "ev",
        behind: str = "upside_10",
    ):
        # The base utility is the even-score lens; the per-game choice in
        # _select overrides it whenever the score is supplied.
        super().__init__(model, utility=even, n_samples=n_samples, device=device)
        self.band = band
        self.ahead = ahead
        self.even = even
        self.behind = behind

    def _select(self, e_q, e_q_pdf, gst, bid_values, marks, marks_to_win):
        if marks is None:
            return super()._select(e_q, e_q_pdf, gst, bid_values, marks, marks_to_win)

        n = e_q.shape[0]
        acting_team = (gst.current_player.long() % 2).tolist()
        chosen = [
            score_to_utility(
                tuple(marks[g]), acting_team[g], marks_to_win,
                band=self.band, ahead=self.ahead, even=self.even, behind=self.behind,
            )
            for g in range(n)
        ]

        return select_by_score(
            chosen, e_q, e_q_pdf,
            gst.bidder.long(), gst.current_player.long(),
            bid_values, gst.legal_actions(),
        )

    def __repr__(self) -> str:
        return (
            f"ScoreConditionedLensPlay(n_samples={self.n_samples}, band={self.band}, "
            f"ahead={self.ahead!r}, even={self.even!r}, behind={self.behind!r})"
        )
