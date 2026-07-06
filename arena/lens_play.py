"""Oracle-backed utility-greedy play policy (Lens) for the arena.

The same batched E[Q] path as `w42/lens_v1/parallel_match.py`, with one
upgrade: per-game bid values from the real auction feed the utility
thresholds, instead of a flat bid=30. The oracle itself never sees the bid
(GameStateTensor carries decl and bidder only), so auction-won contracts
are in-distribution for the Q model; only the utility changes.
"""
from __future__ import annotations

from typing import Sequence

import torch

from forge.eq.generate.deals import build_hypothetical_deals
from forge.eq.generate.eq_compute import compute_eq_pdf
from forge.eq.generate.model import query_model
from forge.eq.generate.sampling import sample_worlds_batched
from forge.eq.generate.tokenization import tokenize_batched
from forge.eq.sampling_mrv_gpu import WorldSamplerMRV
from forge.eq.tokenize_gpu import GPUTokenizer
from forge.zeb.eq_player import zeb_states_to_game_state_tensor
from forge.zeb.game import current_player as zeb_current_player
from forge.zeb.types import ZebGameState
from w42.lens_v1.lens import UTILITIES, argmax_under_utility


def _max_pool_size(states: Sequence[ZebGameState]) -> int:
    """Largest unseen-tile pool across the batch, computed on the CPU.

    Pool = 28 - played - (current player's unplayed tiles); matches the GPU
    pool bitmask popcount exactly, so the MRV sampler can skip its
    max-pool-size GPU->CPU sync without changing its step count.
    """
    max_pool = 0
    for s in states:
        played = s.played
        mine_left = sum(1 for d in s.hands[zeb_current_player(s)] if d not in played)
        max_pool = max(max_pool, 28 - len(played) - mine_left)
    return max_pool


class LensPlay:
    """Utility-greedy player over the forge E[Q] PDF."""

    def __init__(self, model, utility: str = "ev", n_samples: int = 10,
                 device: str = "mps"):
        if utility not in UTILITIES:
            raise ValueError(f"Unknown utility: {utility!r}. Known: {UTILITIES}")
        self.model = model
        self.utility = utility
        self.n_samples = n_samples
        self.device = device
        self._sampler: WorldSamplerMRV | None = None
        self._tokenizer: GPUTokenizer | None = None
        self.model.eval()

    def _ensure_capacity(self, n_games: int) -> None:
        if self._sampler is None or self._sampler.max_games < n_games:
            self._sampler = WorldSamplerMRV(
                max_games=n_games, max_samples=self.n_samples, device=self.device,
            )
        batch_needed = n_games * self.n_samples
        if self._tokenizer is None or self._tokenizer.max_batch < batch_needed:
            self._tokenizer = GPUTokenizer(max_batch=batch_needed, device=self.device)

    def choose(
        self,
        states: Sequence[ZebGameState],
        bid_values: Sequence[int],
        marks: Sequence[tuple[int, int]] | None = None,
        marks_to_win: int = 7,
    ) -> list[int]:
        n = len(states)
        self._ensure_capacity(n)
        gst = zeb_states_to_game_state_tensor(list(states), self.device)

        with torch.no_grad():
            worlds = sample_worlds_batched(
                gst, self._sampler, self.n_samples,
                max_pool_size=_max_pool_size(states),
            )
            deals = build_hypothetical_deals(gst, worlds)
            tokens, masks = tokenize_batched(gst, deals, self._tokenizer)
            q_values = query_model(
                self.model, tokens, masks, gst, self.n_samples, self.device,
            )
            q_reshaped = q_values.view(n, self.n_samples, 7)
            e_q, e_q_pdf = self._marginalize(q_reshaped, gst, states, worlds)
            actions = self._select(
                e_q, e_q_pdf, gst, bid_values, marks, marks_to_win,
            )

        return [int(a) for a in actions.tolist()]

    def _marginalize(self, q_reshaped, gst, states, worlds):
        """Reduce per-world Q to (E[Q], E[Q] PDF). Base: uniform over worlds.

        The belief-weighted player (rung #25) overrides this to importance-weight
        the worlds by a learned posterior; the extra ``gst``/``states``/``worlds``
        arguments are the evidence it needs."""
        return q_reshaped.mean(dim=1), compute_eq_pdf(q_reshaped)

    def _select(self, e_q, e_q_pdf, gst, bid_values, marks, marks_to_win):
        """Choose one action per game under the fixed utility. The
        score-conditioned player (rung #27) overrides this to vary the utility
        by the game's mark score."""
        return argmax_under_utility(
            utility=self.utility,
            e_q=e_q,
            e_q_pdf=e_q_pdf,
            bidder=gst.bidder.long(),
            current_players=gst.current_player.long(),
            legal_mask=gst.legal_actions(),
            bid_values=list(bid_values),
        )

    def __repr__(self) -> str:
        return f"LensPlay(utility={self.utility!r}, n_samples={self.n_samples})"
