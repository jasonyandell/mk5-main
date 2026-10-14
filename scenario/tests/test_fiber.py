# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0

"""The fiber sampler is exact: lawful always, uniform by count."""
import random
from collections import Counter
from itertools import combinations

from scenario.fiber import fiber_size, sample_fiber, sample_worlds
from scenario.record import record_from_state, remaining_hand
from scenario.rules import ALL_TILES, mask_of, tiles_of
from scenario.tests.test_rules_record import random_playing_states


def _brute_force(pool: int, sizes, forbid) -> set[tuple[int, int, int]]:
    tiles = tiles_of(pool)
    out = set()
    for a in combinations(tiles, sizes[0]):
        rest = [t for t in tiles if t not in a]
        for b in combinations(rest, sizes[1]):
            c = tuple(t for t in rest if t not in b)
            if len(c) != sizes[2]:
                continue
            hands = (mask_of(a), mask_of(b), mask_of(c))
            if all(h & f == 0 for h, f in zip(hands, forbid)):
                out.add(hands)
    return out


def test_count_matches_brute_force_with_voids():
    rng = random.Random(2)
    for _ in range(30):
        tiles = rng.sample(range(28), 9)
        pool = mask_of(tiles)
        sizes = (3, 3, 3)
        forbid = tuple(mask_of(rng.sample(tiles, rng.randint(0, 3))) for _ in range(3))
        assert fiber_size(pool, sizes, forbid) == len(_brute_force(pool, sizes, forbid))


def test_samples_are_lawful():
    rng = random.Random(5)
    for _ in range(200):
        tiles = rng.sample(range(28), 12)
        pool = mask_of(tiles)
        sizes = (5, 4, 3)
        forbid = tuple(mask_of(rng.sample(tiles, rng.randint(0, 2))) for _ in range(3))
        if fiber_size(pool, sizes, forbid) == 0:
            continue
        hands = sample_fiber(pool, sizes, forbid, rng)
        assert hands[0] | hands[1] | hands[2] == pool
        assert hands[0] & hands[1] == hands[0] & hands[2] == hands[1] & hands[2] == 0
        assert tuple(bin(h).count("1") for h in hands) == sizes
        assert all(h & f == 0 for h, f in zip(hands, forbid))


def test_samples_are_uniform():
    rng = random.Random(9)
    tiles = list(range(6))
    pool = mask_of(tiles)
    sizes = (2, 2, 2)
    forbid = (mask_of([0]), mask_of([1, 2]), 0)
    support = _brute_force(pool, sizes, forbid)
    n = 20000
    counts = Counter(sample_fiber(pool, sizes, forbid, rng) for _ in range(n))
    assert set(counts) == support
    expected = n / len(support)
    for c in counts.values():
        assert abs(c - expected) < 5 * (expected ** 0.5)


def test_worlds_from_real_states_contain_the_truth():
    """The true deal is always in the fiber: every world is lawful, and the
    viewer's own hand is carried through unchanged."""
    rng = random.Random(4)
    for state in list(random_playing_states(10, seed=8))[::7]:
        if state.played == frozenset(range(28)):
            continue
        rec = record_from_state(state)
        viewer = rec.seat
        own = remaining_hand(state, viewer)
        sizes = tuple(rec.hand_size(s) for s in range(4))
        worlds, _ = sample_worlds(viewer, own, rec.played, sizes, rec.forbid, 5, rng)
        truth = tuple(remaining_hand(state, s) for s in range(4))
        for w in worlds:
            assert w[viewer] == own
            assert all(bin(w[s]).count("1") == sizes[s] for s in range(4))
            assert all(w[s] & rec.forbid[s] == 0 for s in range(4))
            assert w[0] | w[1] | w[2] | w[3] == ALL_TILES & ~rec.played
        assert all(truth[s] & rec.forbid[s] == 0 for s in range(4))


def test_enumeration_is_the_whole_fiber_and_replaces_sampling_when_it_fits():
    from scenario.fiber import enumerate_fiber
    from scenario.planner import SplitMix
    rng = random.Random(13)
    for _ in range(20):
        tiles = rng.sample(range(28), 9)
        pool = mask_of(tiles)
        sizes = (3, 3, 3)
        forbid = tuple(mask_of(rng.sample(tiles, rng.randint(0, 2))) for _ in range(3))
        full = enumerate_fiber(pool, sizes, forbid)
        assert len(full) == len(set(full)) == fiber_size(pool, sizes, forbid)
        assert set(full) == _brute_force(pool, sizes, forbid)
    # Six hidden tiles, two each: 90 worlds; asking for 160 yields all 90, once each.
    for state in random_playing_states(60, seed=17):
        if len(state.played) == 20 and not state.current_trick:
            rec = record_from_state(state)
            viewer = rec.seat
            own = remaining_hand(state, viewer)
            sizes = tuple(rec.hand_size(s) for s in range(4))
            worlds, exact = sample_worlds(viewer, own, rec.played, sizes, rec.forbid, 160, SplitMix(1))
            assert exact and 1 <= len(worlds) <= 90 and len(set(worlds)) == len(worlds)
            # Asking for fewer than the fiber holds samples instead, and says so.
            if len(worlds) > 1:
                _, exact = sample_worlds(viewer, own, rec.played, sizes, rec.forbid,
                                         len(worlds) - 1, SplitMix(1))
                assert not exact
            truth = tuple(remaining_hand(state, s) for s in range(4))
            assert truth in worlds
            break
    else:
        raise AssertionError("no two-tricks-left state found")
