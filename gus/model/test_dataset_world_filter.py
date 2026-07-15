"""Read-time validity filtering of stored sampled worlds (#52, decided in #55).

The pre-repair eq sampler injected domino 0-0 instead of rejecting infeasible
draws, so 27-67% of stored worlds per decision are malformed. These tests pin
down `valid_world_indices_for_decision` and the `filter_invalid_worlds` path on
both dataset classes against a hand-crafted synthetic corpus, plus one
integration assertion on the real April eval corpus.

A stored world is VALID iff its in-range tiles form EXACTLY the decision's
unseen set (multiset equality) AND each relative-seat row r holds exactly
(7 - plays already made by absolute seat (P + r + 1) % 4) in-range tiles.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from forge.eq.generate.types import DecisionRecordGPU, GameRecordGPU
from gus.model.dataset_seq_world import (
    JointWorldFullDataset,
    JointWorldFullIterable,
    _world_to_assignment,
    valid_world_indices_for_decision,
)

# ---------------------------------------------------------------------------
# Synthetic corpus construction
# ---------------------------------------------------------------------------

# A real 28-domino partition into four 7-tile hands.
HANDS = [
    [0, 1, 2, 3, 4, 5, 6],       # P0
    [7, 8, 9, 10, 11, 12, 13],   # P1
    [14, 15, 16, 17, 18, 19, 20],  # P2
    [21, 22, 23, 24, 25, 26, 27],  # P3
]


def _decision(player: int, action_taken: int, worlds: list[list[list[int]]]) -> DecisionRecordGPU:
    """A minimal DecisionRecordGPU carrying hand-crafted world_hands.

    `worlds` is a list of M layouts, each [3][7] of tile ids (-1 = padding for a
    seat that has already played and holds fewer than 7 tiles).
    """
    world_hands = torch.tensor(worlds, dtype=torch.long)  # [M, 3, 7]
    M = world_hands.shape[0]
    return DecisionRecordGPU(
        player=player,
        e_q=torch.zeros(7),
        action_taken=action_taken,
        legal_mask=torch.ones(7, dtype=torch.bool),
        world_hands=world_hands,
        q_per_world=torch.zeros(M, 7),
    )


# --- Game A, decision 0: actor P0, no prior plays. unseen = {7..27}. ---------
# Relative rows: r0 = P1 (7 tiles), r1 = P2 (7), r2 = P3 (7).
A_D0_WORLDS = [
    # 0: valid (truthful partition)
    [[7, 8, 9, 10, 11, 12, 13], [14, 15, 16, 17, 18, 19, 20], [21, 22, 23, 24, 25, 26, 27]],
    # 1: valid (unseen set re-partitioned across seats — still a valid hypothesis)
    [[7, 8, 9, 14, 15, 16, 17], [10, 11, 12, 18, 19, 20, 21], [13, 22, 23, 24, 25, 26, 27]],
    # 2: INVALID — duplicates tile 7 (and drops tile 8): occupancy != exact cover
    [[7, 7, 9, 10, 11, 12, 13], [14, 15, 16, 17, 18, 19, 20], [21, 22, 23, 24, 25, 26, 27]],
    # 3: INVALID — contains the actor's own tile 0 (and drops tile 7)
    [[0, 8, 9, 10, 11, 12, 13], [14, 15, 16, 17, 18, 19, 20], [21, 22, 23, 24, 25, 26, 27]],
    # 4: INVALID — missing tile 13 via -1 padding: wrong row-0 cardinality (6, not 7)
    [[7, 8, 9, 10, 11, 12, -1], [14, 15, 16, 17, 18, 19, 20], [21, 22, 23, 24, 25, 26, 27]],
]
A_D0_VALID = [0, 1]

# --- Game A, decision 1: actor P1, one prior play (P0 played tile 0). --------
# unseen = 28 - {7..13} - {0} = {1..6} ∪ {14..27} (20 tiles).
# Relative rows from P1: r0 = P2 (7), r1 = P3 (7), r2 = P0 (6, one play made).
A_D1_WORLDS = [
    # 0: valid — P0's remaining {1..6} in row 2 with one -1 pad
    [[14, 15, 16, 17, 18, 19, 20], [21, 22, 23, 24, 25, 26, 27], [1, 2, 3, 4, 5, 6, -1]],
    # 1: INVALID — wrongly contains already-played tile 0 (and drops tile 1)
    [[14, 15, 16, 17, 18, 19, 20], [21, 22, 23, 24, 25, 26, 27], [0, 2, 3, 4, 5, 6, -1]],
]
A_D1_VALID = [0]

# --- Game B, decision 0: actor P0, both worlds malformed → fully dropped. ----
B_D0_WORLDS = [
    [[7, 7, 9, 10, 11, 12, 13], [14, 15, 16, 17, 18, 19, 20], [21, 22, 23, 24, 25, 26, 27]],  # dup 7
    [[0, 8, 9, 10, 11, 12, 13], [14, 15, 16, 17, 18, 19, 20], [21, 22, 23, 24, 25, 26, 27]],  # own tile 0
]


def _build_corpus(tmp_path: Path) -> Path:
    game_a = GameRecordGPU(
        decisions=[
            _decision(player=0, action_taken=0, worlds=A_D0_WORLDS),  # plays HANDS[0][0] = tile 0
            _decision(player=1, action_taken=0, worlds=A_D1_WORLDS),  # plays HANDS[1][0] = tile 7
        ],
        hands=HANDS,
        decl_id=1,
    )
    game_b = GameRecordGPU(
        decisions=[_decision(player=0, action_taken=0, worlds=B_D0_WORLDS)],
        hands=HANDS,
        decl_id=1,
    )
    p = tmp_path / "corpus.pt"
    torch.save({"results": [game_a, game_b], "seeds": []}, str(p))
    return p


def _games():
    game_a = GameRecordGPU(
        decisions=[
            _decision(player=0, action_taken=0, worlds=A_D0_WORLDS),
            _decision(player=1, action_taken=0, worlds=A_D1_WORLDS),
        ],
        hands=HANDS,
        decl_id=1,
    )
    game_b = GameRecordGPU(
        decisions=[_decision(player=0, action_taken=0, worlds=B_D0_WORLDS)],
        hands=HANDS,
        decl_id=1,
    )
    return game_a, game_b


# ---------------------------------------------------------------------------
# valid_world_indices_for_decision
# ---------------------------------------------------------------------------

def test_valid_indices_no_prior_plays():
    game_a, _ = _games()
    idx = valid_world_indices_for_decision(game_a, 0)
    assert torch.equal(idx, torch.tensor(A_D0_VALID, dtype=torch.long))


def test_valid_indices_after_one_play():
    game_a, _ = _games()
    idx = valid_world_indices_for_decision(game_a, 1)
    assert torch.equal(idx, torch.tensor(A_D1_VALID, dtype=torch.long))


def test_valid_indices_all_invalid_is_empty():
    _, game_b = _games()
    idx = valid_world_indices_for_decision(game_b, 0)
    assert idx.numel() == 0
    assert idx.dtype == torch.long


def test_each_defect_class_is_rejected():
    """Duplicate, own-tile, missing-tile (row cardinality), and already-played
    defects are each individually excluded."""
    game_a, _ = _games()
    valid = set(valid_world_indices_for_decision(game_a, 0).tolist())
    assert 2 not in valid  # duplicate tile
    assert 3 not in valid  # actor's own tile
    assert 4 not in valid  # missing tile / wrong row cardinality
    valid_d1 = set(valid_world_indices_for_decision(game_a, 1).tolist())
    assert 1 not in valid_d1  # already-played tile present


# ---------------------------------------------------------------------------
# JointWorldFullDataset (map-style)
# ---------------------------------------------------------------------------

def test_map_filter_off_is_pre_change_semantics(tmp_path):
    p = _build_corpus(tmp_path)
    ds = JointWorldFullDataset(str(p), seed=0, filter_invalid_worlds=False)
    # Every decision carrying world tensors is included, unfiltered.
    assert ds.index == [(0, 0), (0, 1), (1, 0)]
    assert len(ds) == 3
    assert ds._valid_idx is None


def test_map_filter_on_drops_zero_valid_decision(tmp_path):
    p = _build_corpus(tmp_path)
    ds = JointWorldFullDataset(str(p), seed=0, filter_invalid_worlds=True)
    # Game B's only decision has no valid world → dropped; Game A's two remain.
    assert ds.index == [(0, 0), (0, 1)]
    assert len(ds) == 2
    assert (1, 0) not in ds.index
    assert torch.equal(ds._valid_idx[(0, 0)], torch.tensor(A_D0_VALID, dtype=torch.long))
    assert torch.equal(ds._valid_idx[(0, 1)], torch.tensor(A_D1_VALID, dtype=torch.long))


def _which_world(world_assignment: torch.Tensor, worlds: list[list[list[int]]]) -> int:
    """Reconstruct which stored world a sample came from by matching its [28,3]
    seat one-hot against each stored layout's assignment. Layouts here are
    pairwise-distinct as assignments, so the match is unique."""
    for m, layout in enumerate(worlds):
        expected = _world_to_assignment(torch.tensor(layout, dtype=torch.long))
        if torch.equal(expected, world_assignment):
            return m
    return -1


def test_map_filter_on_never_samples_invalid_world(tmp_path):
    p = _build_corpus(tmp_path)
    ds = JointWorldFullDataset(str(p), seed=123, filter_invalid_worlds=True)
    item_idx = ds.index.index((0, 0))
    drawn = set()
    for _ in range(200):
        item = ds[item_idx]
        m = _which_world(item["world_assignment"], A_D0_WORLDS)
        assert m in A_D0_VALID, f"sampled invalid world m={m}"
        drawn.add(m)
    # Both valid worlds are actually reachable (sampling is over the valid set).
    assert drawn == set(A_D0_VALID)


# ---------------------------------------------------------------------------
# JointWorldFullIterable (streaming)
# ---------------------------------------------------------------------------

def test_iterable_filter_off_length_and_count(tmp_path):
    p = _build_corpus(tmp_path)
    ds = JointWorldFullIterable(str(p), shuffle=False, seed=0, filter_invalid_worlds=False)
    assert len(ds) == 3
    assert sum(1 for _ in ds) == 3


def test_iterable_filter_on_skips_and_len_consistent(tmp_path):
    p = _build_corpus(tmp_path)
    ds = JointWorldFullIterable(str(p), shuffle=False, seed=0, filter_invalid_worlds=True)
    assert len(ds) == 2
    # Iteration skips the zero-valid decision; count matches __len__.
    assert sum(1 for _ in ds) == len(ds)


def test_iterable_filter_on_yields_only_valid_worlds(tmp_path):
    p = _build_corpus(tmp_path)
    ds = JointWorldFullIterable(str(p), shuffle=False, seed=7, filter_invalid_worlds=True)
    items = list(ds)
    # First yielded item is Game A decision 0 (chunk-native order, shuffle off).
    m = _which_world(items[0]["world_assignment"], A_D0_WORLDS)
    assert m in A_D0_VALID


def test_iterable_length_cache_invalidated_by_filter_state(tmp_path):
    p = _build_corpus(tmp_path)
    cache = tmp_path / "len_cache.pt"
    on = JointWorldFullIterable(
        str(p), shuffle=False, seed=0, length_cache_path=str(cache),
        filter_invalid_worlds=True,
    )
    assert len(on) == 2  # writes cache under a filter=1 key
    # Same cache path, filter OFF must NOT reuse the stale filter=1 count of 2.
    off = JointWorldFullIterable(
        str(p), shuffle=False, seed=0, length_cache_path=str(cache),
        filter_invalid_worlds=False,
    )
    assert len(off) == 3


# ---------------------------------------------------------------------------
# Integration against the real April eval corpus (skip if absent)
# ---------------------------------------------------------------------------

_CORPUS = Path("/Users/jason/code/mk5-main/gus/data/corpus_eval_20.pt")


@pytest.fixture(scope="module")
def eval_dataset():
    if not _CORPUS.exists():
        pytest.skip(f"corpus not present: {_CORPUS}")
    return JointWorldFullDataset(str(_CORPUS), seed=0, filter_invalid_worlds=True)


def test_integration_constructs(eval_dataset):
    assert len(eval_dataset.index) > 0


def test_integration_d0_valid_fraction_in_band(eval_dataset):
    fracs = []
    for game in eval_dataset.games:
        dec = game.decisions[0]
        if dec.world_hands is None or dec.q_per_world is None:
            continue
        M = int(dec.world_hands.shape[0])
        if M == 0:
            continue
        fracs.append(valid_world_indices_for_decision(game, 0).numel() / M)
    assert fracs, "no d0 decisions carried world tensors"
    mean_frac = sum(fracs) / len(fracs)
    print(f"\n[integration] corpus_eval_20 d0 mean valid fraction = {mean_frac:.4f}")
    assert 0.35 <= mean_frac <= 0.75, f"d0 mean valid fraction {mean_frac:.3f} out of band"


def test_integration_no_decision_fully_dropped(eval_dataset):
    total = sum(
        1
        for game in eval_dataset.games
        for dec in game.decisions
        if dec.world_hands is not None and dec.q_per_world is not None
    )
    assert len(eval_dataset.index) == total, (
        f"{total - len(eval_dataset.index)} decision(s) dropped on corpus_eval_20"
    )
