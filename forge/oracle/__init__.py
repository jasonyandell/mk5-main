# STRATEGY-FUSION FENCE: exports the oracle's clairvoyant solver
# (`solve_one_seed`) beside the rules tables; the solver's values are never a
# seat's values. Build on information-state solving (Zeb) instead.
# The ruling: wiki/topics/strategy-fusion-fence.md.
"""forge.oracle - Oracle solver for perfect-play analysis."""

from .context import SeedContext, build_context
from .declarations import N_DECLS, DECL_ID_TO_NAME, DECL_NAME_TO_ID
from .rng import deal_from_seed
from .schema import load_file, unpack_state
from .solve import SolveConfig, solve_one_seed

__all__ = [
    "SeedContext",
    "build_context",
    "N_DECLS",
    "DECL_ID_TO_NAME",
    "DECL_NAME_TO_ID",
    "deal_from_seed",
    "load_file",
    "unpack_state",
    "SolveConfig",
    "solve_one_seed",
]
