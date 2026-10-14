# STRATEGY-FUSION FENCE: this file calls the oracle era's clairvoyant values
# (the double-dummy solver, Stage1Oracle, E[Q] or LensPlay): per-world perfect-
# information values, strategy fusion if read as a seat's value. A measuring
# stick only; build on information-state solving (Zeb) instead.
# The ruling: wiki/topics/strategy-fusion-fence.md.
"""GPU-native E[Q] generation package.

This package provides the GPU-native pipeline for generating E[Q] training data
by playing games using a Stage 1 oracle.

Import from submodules directly, e.g.:
    from forge.eq.generate.pipeline import generate_eq_games_gpu
    from forge.eq.generate.types import AdaptiveConfig, PosteriorConfig
"""

from .actions import contract_threshold_bins, select_actions as _select_actions
from .pipeline import generate_eq_games_gpu

__all__ = ["_select_actions", "contract_threshold_bins", "generate_eq_games_gpu"]
