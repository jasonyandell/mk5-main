"""
Tests for GameStateTensor.from_snapshot / to_snapshot (state injection harness).

Two tests:
- test_from_snapshot_round_trip: deal → apply 5 actions → to_snapshot → from_snapshot,
  assert tensors are bit-identical.
- test_apply_actions_equivalence: extend the above state 3 more steps via apply_actions
  on both the original tensor and the round-tripped tensor, assert equality.
"""
from __future__ import annotations

import random

import pytest
import torch

from forge.eq.game_tensor import GameStateTensor, SNAPSHOT_SCHEMA_VERSION
from forge.oracle.declarations import NOTRUMP, N_DECLS


@pytest.fixture
def device() -> str:
    """CPU for portability — these tests do not require CUDA."""
    return "cpu"


def _random_deal(rng: random.Random) -> list[list[int]]:
    dominoes = list(range(28))
    rng.shuffle(dominoes)
    return [dominoes[i * 7 : (i + 1) * 7] for i in range(4)]


def _apply_n_legal_actions(
    state: GameStateTensor, n: int, rng: random.Random, device: str
) -> GameStateTensor:
    """Apply n randomly selected legal actions."""
    for _ in range(n):
        legal = state.legal_actions()[0]  # (7,) bool
        legal_slots = legal.nonzero(as_tuple=True)[0].tolist()
        slot = rng.choice(legal_slots)
        state = state.apply_actions(torch.tensor([slot], device=device))
    return state


# ---------------------------------------------------------------------------
# Round-trip test
# ---------------------------------------------------------------------------

def test_from_snapshot_round_trip(device: str) -> None:
    """
    Deal a fresh game, apply 5 random legal actions, extract a snapshot via
    to_snapshot, feed back through from_snapshot, and assert tensor equality.
    """
    rng = random.Random(42)
    deal = _random_deal(rng)
    decl_id = rng.randint(0, N_DECLS - 1)

    state_orig = GameStateTensor.from_deals(
        hands=[deal], decl_ids=[decl_id], device=device
    )

    # Advance by 5 actions
    state_mid = _apply_n_legal_actions(state_orig, 5, rng, device)

    # Serialise → deserialise
    snapshots = state_mid.to_snapshot(bid_values=[30])
    assert len(snapshots) == 1
    snap = snapshots[0]

    assert snap["schema_version"] == SNAPSHOT_SCHEMA_VERSION
    assert snap["decl_id"] == decl_id

    state_rt = GameStateTensor.from_snapshot(snapshots, device=device)

    # Every tensor field must be bit-identical
    assert torch.equal(state_mid.hands, state_rt.hands), "hands mismatch"
    assert torch.equal(state_mid.played_mask, state_rt.played_mask), "played_mask mismatch"
    assert torch.equal(state_mid.history, state_rt.history), "history mismatch"
    assert torch.equal(state_mid.trick_plays, state_rt.trick_plays), "trick_plays mismatch"
    assert torch.equal(state_mid.leader, state_rt.leader), "leader mismatch"
    assert torch.equal(state_mid.decl_ids, state_rt.decl_ids), "decl_ids mismatch"
    assert torch.equal(state_mid.bidder, state_rt.bidder), "bidder mismatch"


# ---------------------------------------------------------------------------
# Apply-actions equivalence test
# ---------------------------------------------------------------------------

def test_apply_actions_equivalence(device: str) -> None:
    """
    From the mid-state reached after 5 actions, apply 3 more actions via
    apply_actions on BOTH the original tensor and a round-tripped tensor.
    Assert the resulting tensors are identical.
    """
    rng = random.Random(7)
    deal = _random_deal(rng)
    decl_id = rng.randint(0, N_DECLS - 1)

    state_orig = GameStateTensor.from_deals(
        hands=[deal], decl_ids=[decl_id], device=device
    )

    # Advance by 5 actions — record which slots were chosen
    state_mid = state_orig
    actions_5: list[int] = []
    for _ in range(5):
        legal = state_mid.legal_actions()[0]
        legal_slots = legal.nonzero(as_tuple=True)[0].tolist()
        slot = rng.choice(legal_slots)
        actions_5.append(slot)
        state_mid = state_mid.apply_actions(torch.tensor([slot], device=device))

    # Round-trip the mid-state
    snapshots = state_mid.to_snapshot(bid_values=[35])
    state_rt = GameStateTensor.from_snapshot(snapshots, device=device)

    # Apply the same 3 more actions to both branches
    # Use the same RNG so we pick the same legal slots
    rng_branch = random.Random(99)
    state_fwd_orig = state_mid
    state_fwd_rt = state_rt
    for _ in range(3):
        legal_orig = state_fwd_orig.legal_actions()[0]
        legal_rt = state_fwd_rt.legal_actions()[0]
        # Legal masks must agree before picking
        assert torch.equal(legal_orig, legal_rt), "legal_actions diverged after round-trip"

        legal_slots = legal_orig.nonzero(as_tuple=True)[0].tolist()
        slot = rng_branch.choice(legal_slots)
        action_t = torch.tensor([slot], device=device)

        state_fwd_orig = state_fwd_orig.apply_actions(action_t)
        state_fwd_rt = state_fwd_rt.apply_actions(action_t)

    # Final states must be bit-identical
    assert torch.equal(state_fwd_orig.hands, state_fwd_rt.hands), "hands diverged"
    assert torch.equal(state_fwd_orig.played_mask, state_fwd_rt.played_mask), "played_mask diverged"
    assert torch.equal(state_fwd_orig.history, state_fwd_rt.history), "history diverged"
    assert torch.equal(state_fwd_orig.trick_plays, state_fwd_rt.trick_plays), "trick_plays diverged"
    assert torch.equal(state_fwd_orig.leader, state_fwd_rt.leader), "leader diverged"


# ---------------------------------------------------------------------------
# Validation tests (ValueError on bad input)
# ---------------------------------------------------------------------------

def _minimal_snapshot() -> dict:
    """A valid initial-state snapshot for decl=0, all 28 tiles unplayed."""
    dominoes = list(range(28))
    hands = [dominoes[i * 7 : (i + 1) * 7] for i in range(4)]
    return {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "decl_id": 0,
        "bid_value": 30,
        "bidder": 0,
        "hands": [[d for d in h] for h in hands],
        "played_mask": [False] * 28,
        "history": [[-1, -1, -1]] * 28,
        "trick_plays": [-1, -1, -1, -1],
        "leader": 0,
    }


def test_from_snapshot_bad_schema_version(device: str) -> None:
    snap = _minimal_snapshot()
    snap["schema_version"] = "wrong"
    with pytest.raises(ValueError, match="schema_version"):
        GameStateTensor.from_snapshot([snap], device=device)


def test_from_snapshot_count_mismatch(device: str) -> None:
    snap = _minimal_snapshot()
    # Remove one tile from hands without marking it in played_mask.
    # n_remaining=27, n_played_mask=0, sum=27 != 28 → count mismatch.
    snap["hands"][0][6] = -1
    with pytest.raises(ValueError, match="count mismatch"):
        GameStateTensor.from_snapshot([snap], device=device)


def test_from_snapshot_played_mask_inconsistency(device: str) -> None:
    snap = _minimal_snapshot()
    # Mark tile 0 as played in mask but not in history.
    # Remove from hand so count (n_remaining + n_played_mask = 28) passes.
    # But mask has tile 0 as True while history has no record of it.
    snap["played_mask"][0] = True
    snap["hands"][0][0] = -1  # remove from hand to keep count consistent
    # history still all -1, so mask says played but history doesn't
    # NOTE: count check: n_remaining=27, n_played_mask=1, sum=28 → passes.
    # played_mask consistency check should fire because history has nothing.
    with pytest.raises(ValueError, match="played_mask is inconsistent"):
        GameStateTensor.from_snapshot([snap], device=device)


def test_from_snapshot_trick_plays_gap(device: str) -> None:
    """trick_plays filled at slot 1 but not slot 0 → played_mask inconsistency.

    The trick leader (slot 0) must be played before slot 1, so an empty slot
    0 with a filled slot 1 is invalid.  The validator catches this via the
    played_mask consistency check: the tile in slot 1 is marked in played_mask
    but never appears in history, so it's an 'extra_true' tile.
    """
    snap = _minimal_snapshot()
    # Put tile 5 at slot 1 (leaving slot 0 = -1).
    # Tile 5 is removed from the hand and in played_mask, but never in history.
    snap["trick_plays"] = [-1, 5, -1, -1]
    snap["hands"][0][5] = -1
    snap["played_mask"][5] = True
    with pytest.raises(ValueError, match="played_mask is inconsistent"):
        GameStateTensor.from_snapshot([snap], device=device)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
