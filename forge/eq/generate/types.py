"""Type definitions for GPU E[Q] generation pipeline."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor


@dataclass
class PosteriorConfig:
    """Configuration for posterior weighting in E[Q] computation.

    Args:
        enabled: Whether to use posterior weighting (default: False)
        window_k: Number of past steps to use for weighting (default: 4)
        tau: Temperature for softmax over Q-values (default: 0.1)
        uniform_mix: Uniform mixture coefficient for robustness (default: 0.1)
    """
    enabled: bool = False
    window_k: int = 4
    tau: float = 0.1
    uniform_mix: float = 0.1


@dataclass
class AdaptiveConfig:
    """Configuration for adaptive convergence-based sampling.

    Instead of using a fixed number of samples, adaptively sample until
    E[Q] estimates converge. This allows:
    - Early stopping for low-variance decisions (saves compute)
    - More samples for high-variance decisions (improves accuracy)

    Convergence criterion: max(SEM) over legal actions < sem_threshold
    where SEM = Standard Error of Mean = sigma / sqrt(n)

    Args:
        enabled: Whether to use adaptive sampling (default: False)
        min_samples: Minimum samples before checking convergence (default: 50)
        max_samples: Maximum samples (hard cap) (default: 2000)
        batch_size: Samples to add per iteration (default: 50)
        sem_threshold: Stop when max(SEM) < this (in Q-value points) (default: 0.5)
        convergence_check_interval: Check convergence every N iterations (default: 4)
    """
    enabled: bool = False
    min_samples: int = 50
    max_samples: int = 2000
    batch_size: int = 50
    sem_threshold: float = 0.5
    convergence_check_interval: int = 4


@dataclass
class DecisionRecordGPU:
    """Record for one decision in GPU pipeline.

    Extended in Phase 1a (t42-xncr) to include variance and diagnostics.
    Extended in t42-qz7f to include full E[Q] PDF (85 bins per action).
    Extended in Schema v2 to include per-seat oracle softmax + bid_value.
    """
    player: int  # Which player made the decision (0-3)
    e_q: Tensor  # [7] E[Q] mean values (padded, -inf for illegal/empty)
    action_taken: int  # Slot index (0-6) of domino played
    legal_mask: Tensor  # [7] boolean mask of legal actions
    e_q_var: Tensor | None = None  # [7] E[Q] variance per action
    e_q_pdf: Tensor | None = None  # [7, 85] full PDF: P(Q=q|action) for q in [-42, +42]
    # State uncertainty
    u_mean: float = 0.0  # mean(sqrt(var)) over legal actions
    u_max: float = 0.0   # max(sqrt(var)) over legal actions
    # Posterior diagnostics (None if posterior disabled)
    ess: float | None = None  # Effective sample size
    max_w: float | None = None  # Maximum weight
    # Exploration stats (None if exploration disabled)
    exploration_mode: int | None = None  # 0=greedy, 1=boltzmann, 2=epsilon, 3=blunder
    q_gap: float | None = None  # Q_greedy - Q_taken
    greedy_action: int | None = None  # What greedy would have chosen (slot index 0-6)
    # Adaptive sampling stats (None if fixed sampling)
    n_samples: int | None = None  # Actual samples used (for adaptive: may vary per decision)
    converged: bool | None = None  # True if SEM < threshold, False if hit max_samples
    # Joint-world tensor (opt-in via --save-joint-worlds; fixed-sampling path only)
    # Keeps the raw (world_layout, Q_per_world) tensor that compute_eq_pdf collapses.
    # Enables distillation of the joint belief-Q distribution, not just the marginal PDF.
    world_hands: Tensor | None = None  # [M, 3, 7] sampled opponent dominoes per world
    q_per_world: Tensor | None = None  # [M, 7] oracle Q per action per world
    # Schema v2 fields (opt-in via --schema v2; superset of v1)
    bid_value: int | None = None  # Actual bid amount (30-42, or mark value)
    oracle_softmax_per_seat: Tensor | None = None  # [4, 7] p_make-based softmax per seat
    legal_mask_per_seat: Tensor | None = None  # [4, 7] legal action mask at each seat
    voids_per_seat: Tensor | None = None  # [4, 3, 8] voids matrix at each seat's POV


@dataclass
class GameRecordGPU:
    """Record for one complete game in GPU pipeline."""
    decisions: list[DecisionRecordGPU]
    hands: list[list[int]]  # Initial deal (4 players x 7 dominoes)
    decl_id: int  # Declaration ID
    bid_value: int | None = None  # Schema v2: actual bid amount for this game
