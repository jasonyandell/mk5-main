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
