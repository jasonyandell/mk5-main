from __future__ import annotations
"""
======================================================================
DEPRECATED CPU PIPELINE - DO NOT USE
======================================================================
This module contains KNOWN BUGS (E[Q] collapse with high sample counts).
It is kept temporarily for reference only and will be deleted soon.

Use the GPU pipeline instead: forge/eq/generate_gpu.py
======================================================================
"""
import sys as _sys
if not _sys.flags.interactive:  # Allow interactive inspection
    raise RuntimeError(
        "\n" + "=" * 70 + "\n"
        "DEPRECATED CPU PIPELINE - DO NOT USE\n"
        + "=" * 70 + "\n"
        "This module contains KNOWN BUGS (E[Q] collapse with high sample counts).\n"
        "It is kept temporarily for reference only and will be deleted soon.\n"
        "\n"
        "Use the GPU pipeline instead: forge/eq/generate_gpu.py\n"
        + "=" * 70
    )
del _sys

"""Compatibility shim for E[Q] generation.

Historically, all Stage 2 E[Q] generation lived in this module. It has been
split into focused submodules to keep files manageable, but we keep this shim
to avoid churn in downstream imports.
"""


# Public datatypes/configs
from forge.eq.types import (  # noqa: F401
    DecisionRecord,
    DecisionRecordV2,
    ExplorationPolicy,
    ExplorationStats,
    GameExplorationStats,
    GameRecord,
    GameRecordV2,
    MappingIntegrityError,
    PosteriorConfig,
    PosteriorDiagnostics,
)

# Public entrypoints
from forge.eq.generate_batched import generate_eq_games_batched  # noqa: F401
from forge.eq.generate_game import generate_eq_game  # noqa: F401

# Posterior scoring (used by tests/tools)
from forge.eq.posterior import (  # noqa: F401
    compute_posterior_weights,
    compute_posterior_weights_many,
)

# Internal helpers (re-exported for compatibility with tests and tools)
from forge.eq.exploration import (  # noqa: F401
    _select_action_with_exploration,
    _spawn_child_rng,
)
from forge.eq.outcomes import _fill_actual_outcomes  # noqa: F401
from forge.eq.posterior import (  # noqa: F401
    _compute_weights_for_window,
    _get_legal_local_indices,
    _score_all_steps_batched,
    _score_step_likelihood,
)
from forge.eq.reduction import _reduce_world_q_values  # noqa: F401
from forge.eq.rejuvenation import _domino_violates_voids, _rejuvenate_particles  # noqa: F401
from forge.eq.worlds import _build_hypothetical_worlds_batched, _build_legal_mask  # noqa: F401

__all__ = [
    # Entry points
    "generate_eq_game",
    "generate_eq_games_batched",
    # Types
    "DecisionRecord",
    "DecisionRecordV2",
    "GameRecord",
    "GameRecordV2",
    "PosteriorConfig",
    "PosteriorDiagnostics",
    "MappingIntegrityError",
    "ExplorationPolicy",
    "ExplorationStats",
    "GameExplorationStats",
    # Posterior helpers
    "compute_posterior_weights",
    "compute_posterior_weights_many",
]
