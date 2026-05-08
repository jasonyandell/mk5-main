from __future__ import annotations

from types import SimpleNamespace

from burl.lab.tools.expected_utility import CALCULATE_EXPECTED_UTILITY


def _dist(mean: float, *, p_make: float, stdev: float = 10.0, shape: str = "unimodal"):
    return SimpleNamespace(
        mean=mean,
        p_make=p_make,
        stdev=stdev,
        n_samples=20,
        distribution_shape=shape,
        modes=[{"center": mean, "mass": 0.7}],
        spike_drivers=[],
    )


def test_calculate_expected_utility_ranks_candidate_plays() -> None:
    dists = {
        19: _dist(-10.0, p_make=0.55),
        25: _dist(-20.0, p_make=0.30, shape="multimodal"),
    }
    ctx = SimpleNamespace(
        me_abs=1,
        get_or_build=lambda play: SimpleNamespace(dist=dists[int(play)]),
    )

    result = CALCULATE_EXPECTED_UTILITY.impl(ctx, {"plays": [25, 19]})

    assert "EXPECTED UTILITY RANKING" in result.evidence["prose"]
    assert "Recommendation by E[Q]: 19(5-4)" in result.evidence["prose"]
    structured = result.evidence["structured"]
    assert structured["recommended_play"] == 19
    assert [row["play"] for row in structured["ranking"]] == [19, 25]
    assert structured["recommendation_basis"] == "max_mean_q"


def test_calculate_expected_utility_schema_names_optional_plays() -> None:
    assert CALCULATE_EXPECTED_UTILITY.name == "calculate_expected_utility"
    assert "plays" in CALCULATE_EXPECTED_UTILITY.params["properties"]
    assert "Expected" in CALCULATE_EXPECTED_UTILITY.description or "expected" in CALCULATE_EXPECTED_UTILITY.description
