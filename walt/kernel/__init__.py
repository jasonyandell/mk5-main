"""walt/kernel — the net-free solve substrate (CONTRACTS.md, 2026-07-18a).

Zero-torch kernel over the wavefront engine's SoA shape: bitboard hands,
`walt.tables` LUT rules, waves. The only net-touching call is
`compile_sigma` (via FieldOracle); everything else — subgame enumeration,
best response, profiles — is numpy only. See subgame.py / profiles.py /
br.py module docstrings for the design; cfr/toys/reference are the CFR
lane's files.
"""
from walt.kernel.br import (
    BRResult,
    br_solve,
    payoff_make,
    payoff_points,
    profile_value,
)
from walt.kernel.profiles import (
    SigmaTable,
    StochasticProfile,
    compile_rule_sigma,
    compile_sigma,
)
from walt.kernel.subgame import (
    DEFAULT_SLOT_BUDGET,
    KernelMemoryError,
    Subgame,
    build_subgame,
    expand_full_width,
    path_hash,
)

__all__ = [
    "BRResult",
    "br_solve",
    "payoff_make",
    "payoff_points",
    "profile_value",
    "SigmaTable",
    "StochasticProfile",
    "compile_rule_sigma",
    "compile_sigma",
    "DEFAULT_SLOT_BUDGET",
    "KernelMemoryError",
    "Subgame",
    "build_subgame",
    "expand_full_width",
    "path_hash",
]
