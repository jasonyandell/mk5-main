"""Shared utilities for analysis notebooks.

Modules:
    loading - Shard loading with caching
    features - Feature extraction from packed states
    viz - Visualization helpers
    compression - Entropy and compressibility metrics
    navigation - State space navigation (PV tracing, children)
    symmetry - Canonical forms and orbit enumeration
"""

from forge.analysis.utils.loading import (
    load_seed,
    load_seeds,
    iterate_shards,
    find_shard_files,
)
from forge.analysis.utils.features import (
    depth,
    team,
    player,
    count_locations,
    counts_remaining,
    extract_all,
)
from forge.analysis.utils.compression import (
    entropy_bits,
    conditional_entropy,
    mutual_information,
    lzma_ratio,
)
from forge.analysis.utils.viz import (
    plot_v_distribution,
    plot_v_by_depth,
    setup_notebook_style,
)
from forge.analysis.utils.navigation import (
    build_state_lookup_fast,
    trace_principal_variation,
    track_count_captures,
    count_capture_signature,
    get_children,
)
from forge.analysis.utils.symmetry import (
    team_swap,
    seat_rotate,
    canonical_form,
    enumerate_orbits,
    orbit_sizes,
    orbit_compression_ratio,
    check_v_consistency,
)

__all__ = [
    # loading
    "load_seed",
    "load_seeds",
    "iterate_shards",
    "find_shard_files",
    # features
    "depth",
    "team",
    "player",
    "count_locations",
    "counts_remaining",
    "extract_all",
    # compression
    "entropy_bits",
    "conditional_entropy",
    "mutual_information",
    "lzma_ratio",
    # viz
    "plot_v_distribution",
    "plot_v_by_depth",
    "setup_notebook_style",
    # navigation
    "build_state_lookup_fast",
    "trace_principal_variation",
    "track_count_captures",
    "count_capture_signature",
    "get_children",
    # symmetry
    "team_swap",
    "seat_rotate",
    "canonical_form",
    "enumerate_orbits",
    "orbit_sizes",
    "orbit_compression_ratio",
    "check_v_consistency",
]
