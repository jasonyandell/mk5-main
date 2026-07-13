"""JudNet Lane B arm HC — child-state per-move values on the MAIN head (CPU, fast).

The consumer-aligned residual from the round-1 aux null
(`wiki/experiments/jud-target-granularity.md`, Reading §3): instead of a
parent-side ranking head, supervise the MAIN head's `mean_points` on each
post-move CHILD row with the parent decision's per-move oracle value, converted
to declaring-team points — exactly the quantity `arena.jud_play` argmaxes over
child states. Four load-bearing properties, mirroring `test_jud_aux.py`:

  1. ORIENTATION — the child target is the exact inverse of `acting_seat_margin`
     (defender sign −1, offense +1), hand-computed.
  2. BYTE-COMPAT — child off ⇒ the (x, y) row set and tensors are unchanged.
  3. SLOT — the taken domino reads the correct e_q slot (cross-checked against
     `forced_actions_from_snapshot`, the bridge's own domino→slot map).
  4. LOSS PLUMBING — a training step with child values on runs, adds NO head
     (`aux_per_action=False`), and actually perturbs the main head.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from arena.engine import ArenaConfig
from arena.match import run_match, snapshot_rows
from arena.bidders import HeuristicBidder
from arena.play import RandomPlay
from champion.jud_net import (
    FEATURE_DIM,
    MARGIN_SCALE,
    N_ACTIONS,
    N_POINTS,
    JudDataset,
    JudNet,
    _deal_key,
    acting_seat_margin,
    child_value_target,
    load_jud_net,
    masked_value_mse,
    train,
)
from forge.cli.generate_eq_from_snapshots import forced_actions_from_snapshot
from forge.eq.generate.types import DecisionRecordGPU, GameRecordGPU


def _tiny_snaps(n_games=2, base_seed=7):
    res = run_match(
        bid_a=HeuristicBidder(), bid_b=HeuristicBidder(),
        play_a=RandomPlay(seed=1), play_b=RandomPlay(seed=2),
        n_games=n_games, cfg=ArenaConfig(marks_to_win=2, base_seed=base_seed),
    )
    return snapshot_rows(res)


def _table_from_snaps(snaps, eq_fn, mask_fn=None):
    """Aligned (teacher-forced) aux table built directly, mirroring
    `load_aux_table`'s keying: decision k's player == plays[k][0]. ``eq_fn(k)``
    returns the [7] acting-seat e_q for decision k."""
    table = {}
    for snap in snaps:
        dkey = _deal_key(snap["hands"], snap["decl_id"], snap["bidder"], snap["bids"])
        for k, (player, _dom) in enumerate(snap["plays"]):
            mask = (mask_fn(k) if mask_fn else torch.ones(N_ACTIONS)).float()
            e_q = eq_fn(k).float()
            e_q = torch.where(mask > 0, e_q, torch.zeros_like(e_q))
            table[(dkey, int(k), int(player))] = (e_q, mask)
    return table


def _fake_records(snaps):
    """One aligned GameRecordGPU per snapshot; e_q on every legal slot = float(k)
    so a landed child value is self-identifying by parent decision index."""
    records = []
    for snap in snaps:
        decisions = [
            DecisionRecordGPU(
                player=int(player),
                e_q=torch.full((N_ACTIONS,), float(k)),
                action_taken=0,
                legal_mask=torch.ones(N_ACTIONS, dtype=torch.bool),
            )
            for k, (player, _dom) in enumerate(snap["plays"])
        ]
        records.append(GameRecordGPU(
            decisions=decisions,
            hands=[list(h) for h in snap["hands"]],
            decl_id=int(snap["decl_id"]),
            bid_value=int(snap["bid_value"]),
            bids=tuple(int(b) for b in snap["bids"]),
            bidder=int(snap["bidder"]),
        ))
    return records


# --------------------------------------------------------------------- #
#  1. Orientation: acting-seat E[Q] → declaring-team points               #
# --------------------------------------------------------------------- #

def test_child_value_target_hand_computed():
    # bidder 2 → bidding team {0, 2}. Acting-seat margin +12 at the taken slot.
    assert child_value_target(12.0, seat=2, bidder=2) == 27.0   # offense: (12+42)/2
    assert child_value_target(12.0, seat=0, bidder=2) == 27.0   # partner, also +
    assert child_value_target(12.0, seat=1, bidder=2) == 15.0   # defender: (-12+42)/2
    assert child_value_target(12.0, seat=3, bidder=2) == 15.0
    # Boundaries span the full point range.
    assert child_value_target(42.0, seat=0, bidder=0) == 42.0
    assert child_value_target(-42.0, seat=0, bidder=0) == 0.0


def test_child_value_target_inverts_acting_seat_margin():
    # child_value_target ∘ acting_seat_margin == identity on declaring points.
    for bidder in range(4):
        for seat in range(4):
            for decl_pts in range(0, N_POINTS):
                m = acting_seat_margin(decl_pts, seat, bidder)
                assert child_value_target(m, seat, bidder) == float(decl_pts)


def test_orientation_lands_on_child_rows_both_signs(tmp_path):
    """(mission a) A defender child row target == ((-1)*v+42)/2 and a bidding-team
    child row target == (v+42)/2, hand-computed on real child rows."""
    snap = _tiny_snaps()[0]
    sp = tmp_path / "one.json"
    sp.write_text(json.dumps({"snapshots": [snap]}))
    bidder = int(snap["bidder"])
    plays = snap["plays"]
    V = 12.0
    table = _table_from_snaps([snap], eq_fn=lambda k: torch.full((N_ACTIONS,), V))
    ds = JudDataset(sp, split="all", aux_table=table, child_values=True)

    seen_offense = seen_defense = False
    for i, key in enumerate(ds.keys):
        step, pov = key[5], key[6]
        w = float(ds.child_weights[i])
        t = float(ds.child_targets[i])
        if w > 0:
            # Supervised rows are exactly post-move child rows of pov's decision.
            assert step >= 1 and int(pov) == int(plays[step - 1][0])
            assert t == child_value_target(V, pov, bidder)
            if int(pov) % 2 == bidder % 2:
                assert t == (V + 42) / 2 and t == 27.0
                seen_offense = True
            else:
                assert t == (-V + 42) / 2 and t == 15.0
                seen_defense = True
        else:
            assert t == 0.0
            # Unsupervised rows are the k=0 root or pure decision rows.
            assert step == 0 or int(pov) != int(plays[step - 1][0])
    assert seen_offense and seen_defense  # both a declarer-team and a defender child row


# --------------------------------------------------------------------- #
#  2. Byte-compatibility of the child-off path                            #
# --------------------------------------------------------------------- #

def test_child_off_is_byte_compatible(tmp_path):
    """(mission b) child_values off ⇒ tensors + row set identical to the plain
    dataset, and turning it on never perturbs (x, y)."""
    snaps = _tiny_snaps()
    sp = tmp_path / "snaps.json"
    sp.write_text(json.dumps({"snapshots": snaps}))
    table = _table_from_snaps(snaps, eq_fn=lambda k: torch.full((N_ACTIONS,), 3.0))

    base = JudDataset(sp, split="all")                 # the reference row set
    ds = JudDataset(sp, split="all", aux_table=table, child_values=True)

    assert len(ds) == len(base)
    assert ds.keys == base.keys
    assert not base.child and ds.child
    bx, by = base.tensors()
    cx, cy = ds.tensors()
    assert torch.equal(bx, cx) and torch.equal(by, cy)   # child join rides along untouched

    # Return shapes: plain → 2-tuple, child → 6-tuple (x,y,aux_t,aux_m,child_t,child_w).
    assert len(base[0]) == 2
    assert len(ds[0]) == 6
    x, y, at, am, ct, cw = ds[0]
    assert x.shape == (FEATURE_DIM,) and ct.ndim == 0 and cw.ndim == 0
    # Coverage is the post-move half only — never the whole set.
    assert 0.40 < ds.child_coverage() < 0.60


def test_child_values_requires_aux_table(tmp_path):
    snaps = _tiny_snaps()
    sp = tmp_path / "snaps.json"
    sp.write_text(json.dumps({"snapshots": snaps}))
    with pytest.raises(ValueError):
        JudDataset(sp, split="all", child_values=True)   # no table → refuse


# --------------------------------------------------------------------- #
#  3. Slot mapping: the taken domino reads the right e_q slot              #
# --------------------------------------------------------------------- #

def test_child_reads_taken_slot(tmp_path):
    """(mission c) With a distinct e_q per slot, every supervised child target
    equals the value at the TAKEN slot — cross-checked against the bridge's own
    `forced_actions_from_snapshot`, not a reimplementation."""
    snap = _tiny_snaps()[0]
    sp = tmp_path / "one.json"
    sp.write_text(json.dumps({"snapshots": [snap]}))
    bidder = int(snap["bidder"])
    forced = forced_actions_from_snapshot(snap)          # ground-truth taken slots
    # Distinct, finite value per slot spanning the margin range.
    per_slot = torch.arange(N_ACTIONS, dtype=torch.float32) * 6.0 - 18.0
    table = _table_from_snaps([snap], eq_fn=lambda k: per_slot.clone())
    ds = JudDataset(sp, split="all", aux_table=table, child_values=True)

    n_checked = 0
    for i, key in enumerate(ds.keys):
        if float(ds.child_weights[i]) <= 0:
            continue
        step, pov = key[5], key[6]
        k = step - 1
        slot = forced[k]
        expected = child_value_target(float(per_slot[slot]), int(pov), bidder)
        assert float(ds.child_targets[i]) == expected
        n_checked += 1
    assert n_checked > 0


# --------------------------------------------------------------------- #
#  4. masked value MSE + child training smoke                             #
# --------------------------------------------------------------------- #

def test_masked_value_mse_ignores_unsupervised():
    mean_pts = torch.tensor([21.0, 0.0, 42.0])
    target = torch.tensor([21.0, 42.0, 0.0])
    weight = torch.tensor([1.0, 1.0, 0.0])              # row 2 unsupervised
    # normalized se: (0/42)^2 + (-42/42)^2 = 0 + 1 over 2 supervised → 0.5.
    assert torch.isclose(masked_value_mse(mean_pts, target, weight), torch.tensor(0.5))
    # all-unsupervised → finite 0 (clamp guards the divide).
    assert float(masked_value_mse(mean_pts, target, torch.zeros(3))) == 0.0


def test_child_training_smoke_adds_no_head_and_moves_main(tmp_path):
    """(mission d) One HC run trains, saves aux_per_action=False (no aux head),
    reloads and serves; and against a seed-matched H run the child loss actually
    perturbs the main head — it changes only what it should."""
    snaps = _tiny_snaps(n_games=2)
    sp = tmp_path / "snaps.json"
    sp.write_text(json.dumps({"snapshots": snaps}))
    eqp = tmp_path / "eq.pt"
    torch.save({"results": _fake_records(snaps), "seeds": []}, eqp)

    out_h = tmp_path / "jud_h.pt"
    out_hc = tmp_path / "jud_hc.pt"
    train(corpus=sp, out_model=out_h, epochs=2, batch_size=64, lr=1e-3,
          patience=5, device="cpu", seed=0)                       # arm H
    metrics = train(corpus=sp, out_model=out_hc, epochs=2, batch_size=64, lr=1e-3,
                    patience=5, device="cpu", aux_labels=str(eqp),
                    child_values=True, child_lambda=1.0, seed=0)   # arm HC

    assert out_hc.exists() and "test_ce" in metrics
    saved = torch.load(out_hc, map_location="cpu", weights_only=False)
    assert saved["aux_per_action"] is False                       # HC adds no head
    assert not any(k.startswith("aux.") for k in saved["model_state"])

    hc = load_jud_net(out_hc)
    assert not hc.aux_per_action and not hasattr(hc, "aux")
    assert hc(torch.zeros(3, FEATURE_DIM)).shape == (3, N_POINTS)

    # Same seed ⇒ identical trunk init + batch order; only the child loss differs,
    # so the trained main head must diverge from the pure-H head.
    h = load_jud_net(out_h)
    assert not torch.equal(h.net[4].weight, hc.net[4].weight)
