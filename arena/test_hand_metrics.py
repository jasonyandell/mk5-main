"""Hand metrics tests, including equivalence with the original w42 hand_eval."""
import importlib.util
import random
from pathlib import Path

from forge.oracle.tables import DOMINOES

from arena.hand_metrics import best_trump, evaluate_trump

NAME_TO_ID = {f"{h}-{l}": i for i, (h, l) in enumerate(DOMINOES)}


def hand(*names: str) -> tuple[int, ...]:
    return tuple(NAME_TO_ID[n] for n in names)


def test_pure_trump_hand_has_full_budget():
    h = hand("5-0", "5-1", "5-2", "5-3", "5-4", "5-5", "6-5")
    e = evaluate_trump(h, 5)
    assert e.trump_count == 7
    assert e.has_trump_double
    assert e.unique_exposed_points == 0
    assert e.bid_ceiling == 42


def test_exposure_arithmetic():
    # Trump 6: offs are 3-2, 1-0, 2-0. The held 3-2 covers its own suits;
    # side 1 exposes 4-1 (5), side 0 exposes 5-0 (5) -> ceiling 32.
    h = hand("6-6", "6-5", "6-4", "3-2", "1-0", "2-0", "4-4")
    e = evaluate_trump(h, 6)
    assert e.trump_count == 3
    assert e.unique_exposed_points == 10
    assert e.bid_ceiling == 32
    assert e.held_count_points == 10 + 5  # 6-4 and 3-2


def test_best_trump_respects_min_trumps():
    h = hand("6-6", "6-5", "6-4", "3-2", "1-0", "2-0", "4-4")
    assert best_trump(h, min_trumps=3).trump == 6
    assert best_trump(h, min_trumps=4) is None


def test_matches_original_w42_hand_eval():
    """The arena port must agree with the validated wave-2.B arithmetic."""
    script = (
        Path(__file__).parent.parent
        / "w42/bidding_risk_budget_claim_validation/validate_bidding_risk_budget.py"
    )
    spec = importlib.util.spec_from_file_location("w42_risk_budget", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    rng = random.Random(42)
    for _ in range(200):
        h = tuple(sorted(rng.sample(range(28), 7)))
        for trump in range(7):
            theirs = mod.hand_eval(h, trump)
            ours = evaluate_trump(h, trump)
            assert ours.trump_count == theirs["trump_count"]
            assert ours.unique_exposed_points == theirs["unique_exposed_points"]
            assert ours.bid_ceiling == theirs["bid_ceiling_proxy"]
            assert ours.held_count_points == theirs["held_count_points"]
