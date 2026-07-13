"""Teacher-forced E[Q] labeling — the Lane B aligned-label gate (CPU, fast).

The greedy-replay bridge (``generate_eq_from_snapshots`` without ``--teacher-forced``)
replays the ORACLE's own p_make-greedy line from each deal, so its decision k
only sometimes lands on the arena's play step k — a silent prefix mismatch that
mis-keys aux labels. Teacher-forcing advances the RECORDED line instead, so
decision k IS play step k and every decision row gets its E[Q] label.

These tests use a fake oracle (no checkpoint, CPU, seconds) because the property
under test is STATE ALIGNMENT, not the Q values: the decision states the labeler
visits must be byte-identical to the ones ``champion.jud_net.JudDataset`` builds
from the same snapshots, so the (deal, decision-index, acting-seat) join hits
100% of decision rows. The real-oracle numeric proof lives in
``scratch/lane-b/prove_alignment.py``.
"""
from __future__ import annotations

import json

import pytest
import torch
import torch.nn as nn

from arena.bidders import HeuristicBidder
from arena.engine import ArenaConfig
from arena.match import run_match, snapshot_rows
from arena.play import RandomPlay
from champion.jud_net import JudDataset, load_aux_table
from forge.cli.generate_eq_from_snapshots import (
    attach_auction,
    forced_actions_from_snapshot,
    verify_teacher_forced_alignment,
)
from forge.eq.generate import generate_eq_games_gpu


class _FakeOracle(nn.Module):
    """A stand-in Stage 1 model: a token-dependent, non-degenerate [B,7] Q so the
    greedy line it induces genuinely diverges from the arena's recorded line
    (that divergence is what teacher-forcing exists to defeat)."""

    def __init__(self) -> None:
        super().__init__()
        self.p = nn.Parameter(torch.zeros(1))

    def forward(self, tokens, masks, current_players):
        b = tokens.shape[0]
        q = (tokens.float().sum(dim=(1, 2)).view(b, 1) % 11.0).expand(b, 7) - 5.0
        return q, None


@pytest.fixture(scope="module")
def snaps() -> list[dict]:
    """A handful of real arena snapshot hands (full 28-ply ``plays`` each)."""
    res = run_match(
        bid_a=HeuristicBidder(), bid_b=HeuristicBidder(),
        play_a=RandomPlay(seed=1), play_b=RandomPlay(seed=2),
        n_games=2, cfg=ArenaConfig(marks_to_win=2, base_seed=7),
    )
    rows = snapshot_rows(res)
    assert rows and all(len(s["plays"]) == 28 for s in rows)
    return rows


def _label(snaps: list[dict], *, teacher_forced: bool):
    """Run the fake oracle over the snapshot deals; teacher-forced advances the
    recorded line, greedy advances the oracle's own p_make line (old bridge)."""
    torch.manual_seed(0)
    hands = [[list(h) for h in s["hands"]] for s in snaps]
    decls = [int(s["decl_id"]) for s in snaps]
    bidders = [int(s["bidder"]) for s in snaps]
    bid_values = [int(s["bid_value"]) for s in snaps]
    forced = [forced_actions_from_snapshot(s) for s in snaps] if teacher_forced else None
    recs = generate_eq_games_gpu(
        model=_FakeOracle(), hands=hands, decl_ids=decls, n_samples=4, device="cpu",
        save_joint_worlds=True, bid_values=bid_values, bidders=bidders,
        forced_actions=forced,
    )
    for r, s in zip(recs, snaps):
        attach_auction(r, s)
    return recs


def _decision_row_coverage(ds: JudDataset, snaps: list[dict]) -> tuple[int, int]:
    """(supervised, total) over DECISION rows only. A row is a decision point iff
    its POV is the mover at its step; the eval/child rows (the other half of each
    `hand_samples` pair) are never labeled by design, so whole-dataset coverage
    caps near 0.5 and the meaningful gate is decision-row coverage."""
    plays_by = {(int(s["seed"]), int(s["hand_idx"])): s["plays"] for s in snaps}
    n_sup = n_dec = 0
    for i, key in enumerate(ds.keys):
        seed, hand_idx, step, pov = key[0], key[1], key[5], key[6]
        plays = plays_by[(int(seed), int(hand_idx))]
        if step < len(plays) and int(pov) == int(plays[step][0]):
            n_dec += 1
            if float(ds.aux_masks[i].sum()) > 0:
                n_sup += 1
    return n_sup, n_dec


# --------------------------------------------------------------------- #
#  1. Slot mapping: recorded (seat, domino) -> action slot index         #
# --------------------------------------------------------------------- #

def test_forced_actions_invert_apply_actions(snaps):
    """The forced slot at each step must select exactly the recorded domino out
    of the acting seat's hand — the inverse of ``apply_actions``."""
    for s in snaps:
        slots = forced_actions_from_snapshot(s)
        assert len(slots) == len(s["plays"])
        for (seat, domino), slot in zip(s["plays"], slots):
            assert 0 <= slot < 7
            assert int(s["hands"][int(seat)][slot]) == int(domino)


# --------------------------------------------------------------------- #
#  2. Alignment: decision k == recorded play step k                      #
# --------------------------------------------------------------------- #

def test_teacher_forced_decisions_track_recorded_line(snaps):
    recs = _label(snaps, teacher_forced=True)
    for r, s in zip(recs, snaps):
        assert len(r.decisions) == len(s["plays"]) == 28
        for k, dec in enumerate(r.decisions):
            seat, domino = s["plays"][k]
            # The engine's mover at ply k equals the recorded mover ...
            assert int(dec.player) == int(seat)
            # ... and the action it applied is the recorded domino.
            assert int(s["hands"][int(seat)][dec.action_taken]) == int(domino)
        # And the shipped verifier accepts the aligned record.
        verify_teacher_forced_alignment(r, s)


def test_verify_alignment_rejects_a_corrupted_record(snaps):
    recs = _label(snaps, teacher_forced=True)
    r, s = recs[0], snaps[0]
    r.decisions[5].player = (int(s["plays"][5][0]) + 1) % 4  # desync one ply
    with pytest.raises(ValueError, match="alignment broken"):
        verify_teacher_forced_alignment(r, s)


# --------------------------------------------------------------------- #
#  3. The whole point: 100% decision-row aux coverage vs the greedy bridge #
# --------------------------------------------------------------------- #

def test_teacher_forced_gives_full_decision_row_coverage(tmp_path, snaps):
    sp = tmp_path / "snaps.json"
    sp.write_text(json.dumps({"snapshots": snaps}))

    tf = tmp_path / "tf.pt"
    torch.save({"results": _label(snaps, teacher_forced=True), "seeds": []}, tf)
    ds_tf = JudDataset(sp, split="all", aux_table=load_aux_table(tf))
    sup_tf, dec_tf = _decision_row_coverage(ds_tf, snaps)
    assert dec_tf > 0
    assert sup_tf == dec_tf                       # 100% — every decision row labeled
    assert ds_tf.aux_coverage() > 0.49            # whole-dataset ceiling ~0.5

    # The old greedy-replay bridge drifts off the recorded line, so it labels
    # strictly fewer decision rows — the very failure --teacher-forced fixes.
    gr = tmp_path / "greedy.pt"
    torch.save({"results": _label(snaps, teacher_forced=False), "seeds": []}, gr)
    ds_gr = JudDataset(sp, split="all", aux_table=load_aux_table(gr))
    sup_gr, dec_gr = _decision_row_coverage(ds_gr, snaps)
    assert dec_gr == dec_tf                        # same corpus, same decision rows
    assert sup_gr < dec_gr                          # greedy leaves decisions unlabeled


# --------------------------------------------------------------------- #
#  4. Fail-fast guard: a mismapped/illegal forced action aborts          #
# --------------------------------------------------------------------- #

def test_illegal_forced_action_raises(snaps):
    """Corrupting a forced slot to an already-played (empty) slot must fail fast
    rather than silently advance a wrong state (or index a -1 domino)."""
    s = snaps[0]
    plays = s["plays"]
    forced = forced_actions_from_snapshot(s)
    # Find the first ply whose mover has moved before, and point it at that
    # earlier (now-emptied) slot — guaranteed illegal at this ply.
    seen: dict[int, int] = {}
    corrupt_k = corrupt_slot = None
    for k, (seat, _dom) in enumerate(plays):
        if int(seat) in seen:
            corrupt_k, corrupt_slot = k, seen[int(seat)]
            break
        seen[int(seat)] = forced[k]
    assert corrupt_k is not None
    forced[corrupt_k] = corrupt_slot

    with pytest.raises(ValueError, match="forced action illegal"):
        generate_eq_games_gpu(
            model=_FakeOracle(), hands=[[list(h) for h in s["hands"]]],
            decl_ids=[int(s["decl_id"])], n_samples=4, device="cpu",
            bidders=[int(s["bidder"])], forced_actions=[forced],
        )
