"""Belief-weighted oracle play (rung #25) — the highest-leverage ML move.

`BeliefLensPlay` keeps the validity-guaranteed MRV world sampler and the oracle
E[Q] path unchanged, and changes only the marginalization: instead of averaging
Q uniformly over the sampled worlds, it importance-weights them by the Gus belief
posterior (`champion.belief`). Likely worlds get more weight, so bidding
evaluation, play, and defense all sharpen from one change.

With `belief_model=None` the weights are uniform and this is byte-identical to
`LensPlay`, so the arena A/B is a pure swap.
"""
from __future__ import annotations

from arena.lens_play import LensPlay
from champion.belief import belief_weights_for_worlds, effective_sample_size, load_belief
from forge.eq.generate.eq_compute import compute_eq_pdf, compute_eq_weighted_mean


class BeliefLensPlay(LensPlay):
    """LensPlay that importance-weights the sampled worlds by the belief posterior."""

    def __init__(
        self,
        model,
        *,
        utility: str = "ev",
        n_samples: int = 10,
        device: str = "mps",
        belief_adapter: str | None = None,
        belief_model=None,
        belief_is_voids: bool = False,
        use_belief: bool = True,
        uniform_mix: float = 0.1,
        tau: float = 1.0,
    ):
        super().__init__(model, utility=utility, n_samples=n_samples, device=device)
        # Load the belief model by default (the default adapter when none given);
        # use_belief=False is the explicit uniform-weights degrade path for tests.
        if use_belief and belief_model is None:
            belief_model, belief_is_voids = load_belief(belief_adapter, device)
        if belief_model is not None:
            belief_model.eval()
        self.belief_model = belief_model
        self.belief_is_voids = belief_is_voids
        self.uniform_mix = uniform_mix
        self.tau = tau
        self.last_ess: float | None = None  # mean ESS of the last decision batch
        self._calls = 0

    def _marginalize(self, q_reshaped, gst, states, worlds):
        w = belief_weights_for_worlds(
            self.belief_model, self.belief_is_voids, states, worlds, self.device,
            uniform_mix=self.uniform_mix, tau=self.tau,
        )  # [n, M]
        if self.belief_model is not None:
            ess = effective_sample_size(w)
            self.last_ess = float(ess.mean().item())
            self._calls += 1
            # Occasional heartbeat: is the play-evidence belief peaked (ESS << M)
            # or near-flat (ESS ~ M)? Governs how to read the arena delta.
            if self._calls % 200 == 1:
                print(
                    f"    [belieflens] ess mean={self.last_ess:.1f} "
                    f"min={float(ess.min().item()):.1f} (M={self.n_samples})",
                    flush=True,
                )
        e_q = compute_eq_weighted_mean(q_reshaped, w)
        e_q_pdf = compute_eq_pdf(q_reshaped, weights=w)
        return e_q, e_q_pdf

    def __repr__(self) -> str:
        tag = "uniform" if self.belief_model is None else f"belief(tau={self.tau})"
        return (
            f"BeliefLensPlay(utility={self.utility!r}, n_samples={self.n_samples}, "
            f"weights={tag}, uniform_mix={self.uniform_mix})"
        )
