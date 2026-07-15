"""Tests for write-time joint-world validity (issue #52) and the decl-8 purge
(issue #51).

The end-to-end test re-checks stored worlds with a deliberately independent
pure-Python set-based referee — not forge.eq.validate_worlds — per the
independence discipline in issue #55.
"""

from __future__ import annotations

import pytest
import torch

from forge.eq.game_tensor import GameStateTensor
from forge.eq.validate_worlds import (
    assert_stored_worlds_valid,
    assert_world_weights_normalized,
    stored_world_validity,
)
from forge.oracle.declarations import (
    DOUBLES_SUIT,
    GAME_DECL_IDS,
    N_GAME_DECLS,
    parse_decl_arg,
)


DEAL = [
    list(range(0, 7)),
    list(range(7, 14)),
    list(range(14, 21)),
    list(range(21, 28)),
]


def _states(device: str = "cpu") -> GameStateTensor:
    return GameStateTensor.from_deals([DEAL], [0], device)


def _world(rows: list[list[int]]) -> list[list[int]]:
    """Pad rows to 7 with -1."""
    return [row + [-1] * (7 - len(row)) for row in rows]


class TestStoredWorldValidity:
    def test_masks_each_defect_class(self):
        states = _states()
        # Actor is the leader; rows are relative seats (P+1, P+2, P+3).
        p = int(states.current_player[0])
        opp = [(p + r + 1) % 4 for r in range(3)]
        truth = [DEAL[o] for o in opp]

        worlds = torch.tensor(
            [
                _world(truth),  # exact truth — valid
                # swap a tile between two opponents — still an exact cover
                _world(
                    [
                        [truth[1][0]] + truth[0][1:],
                        [truth[0][0]] + truth[1][1:],
                        truth[2],
                    ]
                ),
                # duplicate: one tile appears twice, another goes missing
                _world([truth[0], truth[1], [truth[0][0]] + truth[2][1:]]),
                # overlap: actor's own first tile leaks into an opponent row
                _world([[DEAL[p][0]] + truth[0][1:], truth[1], truth[2]]),
                # missing tile: a row padded short (19 of 21 unseen covered)
                _world([truth[0][:6], truth[1], truth[2][:6]]),
            ],
            dtype=torch.int8,
        ).unsqueeze(0)  # [1, 5, 3, 7]

        mask = stored_world_validity(states, worlds)
        assert mask.tolist() == [[True, True, False, False, False]]

    def test_played_tile_becomes_seen(self):
        states = _states()
        p0 = int(states.current_player[0])
        played = int(
            states.hands[0, p0, 0]
        )  # slot 0 of the leader's hand is legal on a lead
        states = states.apply_actions(torch.tensor([0]))

        p1 = int(states.current_player[0])
        opp = [(p1 + r + 1) % 4 for r in range(3)]
        truth = []
        for o in opp:
            row = [d for d in DEAL[o] if d != played]
            truth.append(row)
        worlds_good = torch.tensor([_world(truth)], dtype=torch.int8).unsqueeze(0)
        assert stored_world_validity(states, worlds_good).all()

        # Same world but with the already-played tile stuffed back in.
        bad = [row[:] for row in truth]
        short = min(range(3), key=lambda r: len(bad[r]))
        bad[short] = bad[short] + [played]
        worlds_bad = torch.tensor([_world(bad)], dtype=torch.int8).unsqueeze(0)
        assert not stored_world_validity(states, worlds_bad).any()

        # Exact cover but wrong per-row distribution: the seat that already
        # played (6 remaining) is dealt 7 tiles by stealing one from a full
        # row. Only representable mid-game — exactly where the row-cardinality
        # check matters, since exact cover alone cannot see it.
        long_row = max(range(3), key=lambda r: len(truth[r]))
        assert len(truth[short]) == 6 and len(truth[long_row]) == 7
        misdist = [row[:] for row in truth]
        donor = misdist[long_row].pop()
        misdist[short] = misdist[short] + [donor]
        worlds_misdist = torch.tensor([_world(misdist)], dtype=torch.int8).unsqueeze(0)
        assert not stored_world_validity(states, worlds_misdist).any()

    def test_assert_raises_with_context(self):
        states = _states()
        p = int(states.current_player[0])
        opp = [(p + r + 1) % 4 for r in range(3)]
        truth = [DEAL[o] for o in opp]
        bad = torch.tensor(
            [_world([[DEAL[p][0]] + truth[0][1:], truth[1], truth[2]])],
            dtype=torch.int8,
        ).unsqueeze(0)
        with pytest.raises(RuntimeError, match="malformed sampled worlds"):
            assert_stored_worlds_valid(states, bad, context="unit test")

        good = torch.tensor([_world(truth)], dtype=torch.int8).unsqueeze(0)
        assert_stored_worlds_valid(states, good)  # should not raise


class TestWorldWeights:
    def test_normalized_ok(self):
        w = torch.full((2, 50), 1 / 50)
        assert_world_weights_normalized(w)

    def test_unnormalized_raises(self):
        w = torch.full((2, 50), 1 / 25)
        with pytest.raises(RuntimeError, match="unnormalized"):
            assert_world_weights_normalized(w)

    def test_negative_raises(self):
        w = torch.full((1, 4), 0.5)
        w[0, 0] = -0.5
        with pytest.raises(RuntimeError, match="unnormalized"):
            assert_world_weights_normalized(w)


class TestDeclPurge:
    def test_game_decl_ids_exclude_doubles_suit(self):
        assert DOUBLES_SUIT not in GAME_DECL_IDS
        assert N_GAME_DECLS == 9
        assert len(set(GAME_DECL_IDS)) == 9
        assert all(0 <= d < 10 for d in GAME_DECL_IDS)

    def test_parse_all_excludes_doubles_suit(self):
        assert parse_decl_arg("all").decl_ids == GAME_DECL_IDS

    def test_parse_doubles_suit_refused(self):
        for name in ("ds", "doubles-suit", "8"):
            with pytest.raises(ValueError, match="#51"):
                parse_decl_arg(name)

    def test_campaign_decls_for_seed_purged(self):
        from forge.oracle.campaign import decls_for_seed

        for seed in range(200):
            assert DOUBLES_SUIT not in decls_for_seed(seed, k=3)


@pytest.fixture
def gpu_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    pytest.skip("Joint-world generation smoke needs a GPU (cuda or mps).")


def test_cli_end_to_end(gpu_device, tmp_path):
    """The actual CLI entry point runs (catches main()-scope bugs the direct
    pipeline call cannot, e.g. import shadowing)."""
    import subprocess
    import sys

    out = tmp_path / "cli_smoke.pt"
    proc = subprocess.run(
        [sys.executable, "-u", "-m", "forge.eq.generate",
         "--start-seed", "424242", "--n-games", "2", "--n-decl-per-seed", "2",
         "--samples", "10", "--schema", "v2", "--save-joint-worlds",
         "--record-world-weights", "--device", gpu_device, "-o", str(out)],
        capture_output=True, text=True, timeout=600,
    )
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
    blob = torch.load(out, weights_only=False)
    assert len(blob["results"]) == 2
    assert all(g.decl_id != 8 for g in blob["results"])
    assert blob["results"][0].decisions[0].world_weights is not None


def test_generation_records_valid_worlds_and_weights(gpu_device):
    """End-to-end micro-generation: stored worlds are real deals (checked by
    an INDEPENDENT set-based referee) and world_weights are normalized."""
    from forge.eq.generate import generate_eq_games_gpu
    from forge.eq.oracle import Stage1Oracle
    from forge.oracle.rng import deal_from_seed

    oracle = Stage1Oracle(
        "forge/models/domino-qval-large-3.3M-qgap0.071-qmae0.94.ckpt",
        device=gpu_device,
        compile=False,
    )
    hands = [deal_from_seed(424242), deal_from_seed(424243)]
    results = generate_eq_games_gpu(
        model=oracle.model,
        hands=hands,
        decl_ids=[0, 9],
        n_samples=25,
        device=gpu_device,
        greedy=True,
        save_joint_worlds=True,
        record_world_weights=True,
    )

    assert len(results) == 2
    n_checked = 0
    for game in results:
        played: set[int] = set()
        # Track remaining hands per seat as the recorded line replays.
        remaining = [set(h) for h in game.hands]
        for dec in game.decisions:
            p = dec.player
            assert dec.world_hands is not None
            assert dec.world_weights is not None
            w = dec.world_weights
            assert w.shape[0] == dec.world_hands.shape[0]
            assert float(w.sum()) == pytest.approx(1.0, abs=1e-4)
            assert float(w.min()) >= 0.0

            # Independent referee: exact cover of the unseen set, by seats.
            unseen = set(range(28)) - remaining[p] - played
            counts = [len(remaining[(p + r + 1) % 4]) for r in range(3)]
            for m in range(dec.world_hands.shape[0]):
                rows = dec.world_hands[m].tolist()
                seen_tiles: list[int] = []
                for r, row in enumerate(rows):
                    tiles = [t for t in row if 0 <= t < 28]
                    assert len(tiles) == counts[r]
                    seen_tiles.extend(tiles)
                assert sorted(seen_tiles) == sorted(unseen)
                n_checked += 1

            # Advance the recorded line: action_taken is a slot into the
            # actor's INITIAL hand row (fixed-slot engine convention, same
            # replay rule as gus/eval/belief_ceiling.py).
            tile = int(game.hands[p][dec.action_taken])
            assert tile in remaining[p]
            remaining[p].discard(tile)
            played.add(tile)
    assert n_checked > 0
