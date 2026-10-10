"""The mold: the compiled count of a deal's positions per layer is the
referee's, on seeded random deals."""
import random

import pytest

from arena.walt_play import _declared_trump
from scenario.atlas import mold
from scenario.compiled import CompiledWalt
from scenario.planner import Config, Rung
from scenario.rules import mask_of


def _deal(seed: int) -> list[int]:
    tiles = list(range(28))
    random.Random(seed).shuffle(tiles)
    return [mask_of(tiles[7 * s:7 * s + 7]) for s in range(4)]


def test_the_mold_of_seed_one_is_the_prototypes():
    """The counts scratch/atlas/atlas1.py logged for its seed-1 deal."""
    hands = _deal(1)
    assert _declared_trump(hands[0]) == 6
    assert mold(hands, 6, 0, max_layer=10) == [1, 7, 9, 13, 67, 402, 936, 2258, 5081, 25405, 50083]


@pytest.mark.parametrize("seed, leader", [(1, 0), (7, 2), (23, 3)])
def test_compiled_mold_matches_the_referee(seed, leader):
    hands = _deal(seed)
    decl = _declared_trump(hands[leader])
    fast = CompiledWalt(Config((Rung(1, None),)))
    assert fast.mold(hands, decl, leader, max_layer=10) == mold(hands, decl, leader, max_layer=10)


def test_mold_refuses_a_misdeal():
    hands = _deal(1)
    with pytest.raises(ValueError):
        mold([hands[0], hands[0], hands[2], hands[3]], 6, 0)
    with pytest.raises(ValueError):
        CompiledWalt(Config((Rung(1, None),))).mold([hands[0], hands[0], hands[2], hands[3]], 6, 0)
