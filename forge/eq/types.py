"""Shared datatypes for E[Q] generation.

This file intentionally holds the cross-cutting dataclasses/config used by:
- single-game generation
- batched generation
- posterior weighting
- exploration policy and diagnostics
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from torch import Tensor


# =============================================================================
# Posterior weighting parameters (t42-64uj.3)
# =============================================================================


@dataclass
class PosteriorConfig:
    """Configuration for posterior-weighted E[Q] marginalization."""

    enabled: bool = False  # Whether to use posterior weighting
    tau: float = 10.0  # Temperature for advantage-softmax
    beta: float = 0.10  # Uniform mixture coefficient for robustness
    window_k: int = 8  # Number of past plays to score
    delta: float = 30.0  # Log-weight clipping threshold

    # ESS thresholds for diagnostics/mitigation
    ess_warn: float = 10.0  # Warn if ESS < this
    ess_critical: float = 3.0  # Critical if ESS < this

    # Mitigation parameters
    mitigation_enabled: bool = True  # Whether to apply mitigation when ESS low
    mitigation_alpha: float = 0.3  # Uniform mix when ESS < ess_critical
    mitigation_beta_boost: float = 0.15  # Extra beta when ESS < ess_warn

    # Adaptive K parameters (Phase 7)
    adaptive_k_enabled: bool = False  # Whether to expand window when ESS low
    adaptive_k_max: int = 16  # Maximum window size
    adaptive_k_ess_threshold: float = 5.0  # Expand window if ESS < this
    adaptive_k_step: int = 4  # How much to expand window by

    # Rejuvenation kernel parameters (Phase 8)
    rejuvenation_enabled: bool = False  # Whether to resample+rejuvenate when ESS critical
    rejuvenation_steps: int = 3  # Number of MCMC steps per particle
    rejuvenation_ess_threshold: float = 2.0  # Trigger rejuvenation if ESS < this

    # Mapping integrity (design notes §6)
    strict_integrity: bool = False  # If True, raise on in-hand-but-illegal (dev mode)
    # If False, count and continue (production mode)


@dataclass
class PosteriorDiagnostics:
    """Diagnostics for posterior weighting health."""

    ess: float = 0.0  # Effective sample size
    max_w: float = 0.0  # Maximum weight
    entropy: float = 0.0  # Weight entropy H = -sum(w log w)
    k_eff: float = 0.0  # Effective world count = exp(H)
    n_invalid: int = 0  # Worlds with -inf log-weight (inconsistent)
    n_illegal: int = 0  # Worlds where observed was in-hand but illegal (bug indicator)
    window_nll: float = 0.0  # Average negative log-likelihood over window
    window_k_used: int = 0  # Actual window size used (may differ from config if adaptive)
    rejuvenation_applied: bool = False  # Whether rejuvenation was triggered
    rejuvenation_accepts: int = 0  # Number of accepted MCMC swaps
    mitigation: str = ""  # Description of any mitigation applied


class MappingIntegrityError(Exception):
    """Raised when observed action is in-hand but illegal (rules/state reconstruction bug)."""


# =============================================================================
# Exploration policy parameters (t42-64uj.5)
# =============================================================================


@dataclass
class ExplorationPolicy:
    """Configuration for exploratory action selection during game generation.

    Supports a mixture of policies to diversify transcripts:
    - greedy: Always pick argmax(E[Q]) (default, no exploration)
    - boltzmann: Sample from softmax(E[Q] / temperature)
    - epsilon: With probability epsilon, pick uniformly random legal action
    - blunder: Occasionally pick suboptimal action with bounded regret

    The policies are applied in order: first check blunder, then epsilon,
    then boltzmann vs greedy. This allows combining them.
    """

    # Boltzmann sampling
    temperature: float = 1.0  # Temperature for softmax sampling (lower = greedier)
    use_boltzmann: bool = False  # If False, use greedy argmax

    # Epsilon-greedy
    epsilon: float = 0.0  # Probability of random action (0 = no random)

    # Bounded blunder (occasional suboptimal pick)
    blunder_rate: float = 0.0  # Probability of picking non-best action
    blunder_max_regret: float = 5.0  # Max Q-gap allowed for blunder (in points)

    # Seeding for reproducibility
    seed: int | None = None  # If set, use deterministic RNG

    @classmethod
    def greedy(cls) -> "ExplorationPolicy":
        """Pure greedy policy (no exploration)."""
        return cls()

    @classmethod
    def boltzmann(
        cls, temperature: float = 2.0, seed: int | None = None
    ) -> "ExplorationPolicy":
        """Boltzmann sampling with given temperature."""
        return cls(use_boltzmann=True, temperature=temperature, seed=seed)

    @classmethod
    def epsilon_greedy(
        cls, epsilon: float = 0.1, seed: int | None = None
    ) -> "ExplorationPolicy":
        """Epsilon-greedy with given exploration rate."""
        return cls(epsilon=epsilon, seed=seed)

    @classmethod
    def mixed_exploration(
        cls,
        temperature: float = 3.0,
        epsilon: float = 0.05,
        blunder_rate: float = 0.02,
        blunder_max_regret: float = 3.0,
        seed: int | None = None,
    ) -> "ExplorationPolicy":
        """Default mixed exploration for production dataset generation."""
        return cls(
            use_boltzmann=True,
            temperature=temperature,
            epsilon=epsilon,
            blunder_rate=blunder_rate,
            blunder_max_regret=blunder_max_regret,
            seed=seed,
        )


@dataclass
class ExplorationStats:
    """Exploration statistics for one decision."""

    greedy_action: int
    action_taken: int
    was_greedy: bool
    selection_mode: str  # "greedy", "boltzmann", "epsilon", "blunder"
    q_gap: float  # Q_greedy - Q_taken in points
    action_entropy: float  # Entropy of action distribution (legal-only)


@dataclass
class GameExplorationStats:
    """Aggregate exploration stats for one game."""

    n_decisions: int = 0
    n_greedy: int = 0
    n_boltzmann: int = 0
    n_epsilon: int = 0
    n_blunder: int = 0
    total_q_gap: float = 0.0
    mean_action_entropy: float = 0.0

    @property
    def greedy_rate(self) -> float:
        return (self.n_greedy / self.n_decisions) if self.n_decisions > 0 else 0.0

    @property
    def mean_q_gap(self) -> float:
        return (self.total_q_gap / self.n_decisions) if self.n_decisions > 0 else 0.0


# =============================================================================
# Decision record dataclasses (Stage 2 training schema)
# =============================================================================


@dataclass
class DecisionRecord:
    """One decision point (one player's turn) for Stage 2 training.

    The target field `e_q_mean` contains the E[Q] (expected Q-value) in POINTS,
    computed by averaging Q-values across sampled worlds. These are NOT logits -
    do NOT apply softmax. Values are typically in [-42, +42] range.

    The `actual_outcome` field contains the actual margin (my team - opponent team)
    from this decision point to end of game. This is computed after game completion
    via backward pass and enables direct validation of E[Q] predictions.
    """

    transcript_tokens: Tensor  # Stage 2 input (from tokenize_transcript)
    e_q_mean: Tensor  # (7,) target - E[Q] in points (averaged Q across worlds)
    legal_mask: Tensor  # (7,) which actions were legal
    action_taken: int  # Which action was actually played
    player: int  # Who made this decision (0-3)
    actual_outcome: float | None = None  # Actual margin from here to end (filled after game)


@dataclass
class GameRecord:
    """All decisions from one game."""

    decisions: list[DecisionRecord]
    # 28 decisions per game (4 players × 7 tricks)


@dataclass
class DecisionRecordV2(DecisionRecord):
    """Extended decision record with posterior diagnostics (t42-64uj.3) and exploration stats.

    Inherits from DecisionRecord:
    - player: Who made this decision (0-3)
    - actual_outcome: Actual margin from here to end (filled after game)

    Uncertainty fields (t42-64uj.6):
    - e_q_var: Per-move variance σ²(a) = Var_w[Q(a)] in points²
    - u_mean: State-level uncertainty = mean_{a legal}(σ(a)) in points
    - u_max: State-level uncertainty = max_{a legal}(σ(a)) in points
    """

    diagnostics: PosteriorDiagnostics | None = None  # Posterior health metrics
    exploration: ExplorationStats | None = None  # Exploration statistics (t42-64uj.5)
    # Uncertainty fields (t42-64uj.6)
    e_q_var: Tensor | None = None  # (7,) variance per action in points²
    u_mean: float = 0.0  # State-level uncertainty (mean σ over legal) in points
    u_max: float = 0.0  # State-level uncertainty (max σ over legal) in points


@dataclass
class GameRecordV2(GameRecord):
    """Extended game record with posterior config and exploration stats."""

    posterior_config: PosteriorConfig | None = None
    exploration_policy: ExplorationPolicy | None = None  # Policy used for action selection
    exploration_stats: GameExplorationStats | None = None  # Aggregate exploration stats
