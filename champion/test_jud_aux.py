"""JudNet Lane B — the per-legal-action E[Q] auxiliary head (CPU, seconds).

Three load-bearing properties:

  1. BYTE-COMPAT — with ``aux_per_action=False`` the module holds exactly the
     ``net.0/2/4`` params of every existing checkpoint, `forward` is identical
     to ``self.net(x)``, and a real jud_net_r*.pt loads strict.
  2. ORIENTATION (the #1 foot-gun) — the aux head lives in ACTING-SEAT
     orientation (matching forge e_q); `acting_seat_margin` is the explicit
     bridge from the main head's declaring-team orientation, hand-checked.
  3. JOIN — the (deal, decision-index, acting-seat) join attaches an E[Q]
     label to exactly the right decision rows and drops misaligned ones.
"""
from __future__ import annotations

import json
from pathlib import Path

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
    acting_seat_margin,
    aux_target_from_eq,
    load_aux_table,
    load_jud_net,
    masked_action_mse,
    mean_points,
    train,
)
from forge.eq.generate.types import DecisionRecordGPU, GameRecordGPU

_EXISTING_CKPT = Path("champion/jud_net_r4.pt")


def _tiny_snaps(n_games=2, base_seed=7):
    res = run_match(
        bid_a=HeuristicBidder(), bid_b=HeuristicBidder(),
        play_a=RandomPlay(seed=1), play_b=RandomPlay(seed=2),
        n_games=n_games, cfg=ArenaConfig(marks_to_win=2, base_seed=base_seed),
    )
    return snapshot_rows(res)


# --------------------------------------------------------------------- #
#  1. Byte-compatibility of the aux-off path                              #
# --------------------------------------------------------------------- #

def test_aux_off_state_dict_is_byte_compatible():
    off = JudNet(aux_per_action=False)
    assert list(off.state_dict()) == [
        "net.0.weight", "net.0.bias", "net.2.weight", "net.2.bias",
        "net.4.weight", "net.4.bias",
    ]
    assert not off.aux_per_action
    # aux head is purely additive: on adds exactly the two aux.* keys.
    on = JudNet(aux_per_action=True)
    assert set(on.state_dict()) - set(off.state_dict()) == {"aux.weight", "aux.bias"}
    assert on.aux.weight.shape == (N_ACTIONS, 512)


def test_forward_is_byte_identical_to_sequential():
    torch.manual_seed(0)
    model = JudNet()
    x = torch.randn(4, FEATURE_DIM)
    assert torch.equal(model(x), model.net(x))
    assert model(x).shape == (4, N_POINTS)


def test_existing_checkpoint_loads_strict_and_serves():
    if not _EXISTING_CKPT.exists():
        return  # checkpoints are optional in a fresh clone
    # Strict load into the default (aux-off) constructor — proves byte-compat.
    ckpt = torch.load(_EXISTING_CKPT, map_location="cpu", weights_only=False)
    JudNet(in_dim=ckpt.get("feature_dim", FEATURE_DIM)).load_state_dict(
        ckpt["model_state"]
    )
    # And the public loader reconstructs + serves a [B,43] head.
    model = load_jud_net(_EXISTING_CKPT)
    assert model(torch.zeros(3, FEATURE_DIM)).shape == (3, N_POINTS)
    assert not model.aux_per_action


# --------------------------------------------------------------------- #
#  2. Orientation: acting-seat vs declaring-team                          #
# --------------------------------------------------------------------- #

def test_acting_seat_margin_hand_computed():
    # Declaring team = bidder's team (bidder 3 → team 1). Declaring team caught
    # 30 of 42 points → declaring margin 2*30-42 = +18.
    assert acting_seat_margin(30, seat=1, bidder=3) == 18   # offense reads +18
    assert acting_seat_margin(30, seat=3, bidder=3) == 18   # bidder's own seat
    assert acting_seat_margin(30, seat=0, bidder=3) == -18  # defense negates
    assert acting_seat_margin(30, seat=2, bidder=3) == -18
    # Boundary points map to the full margin range.
    assert acting_seat_margin(42, seat=0, bidder=0) == 42
    assert acting_seat_margin(0, seat=0, bidder=0) == -42


def test_aux_target_is_eq_scaled_no_flip():
    # forge e_q is ALREADY acting-seat oriented; the only transform is /42.
    e_q = torch.tensor([18.0, -42.0, 0.0, 42.0, -18.0, 5.0, -5.0])
    assert torch.allclose(aux_target_from_eq(e_q), e_q / MARGIN_SCALE)
    assert MARGIN_SCALE == float(N_POINTS - 1) == 42.0


def test_main_head_sign_matches_acting_seat_orientation():
    # A defender's ranking over the MAIN head uses sign*mean_points; the aux
    # head needs no sign. Both must rank a set of positions the same way.
    torch.manual_seed(0)
    # Three constructed positions with declaring-team points 10, 25, 40.
    decl_pts = [10, 25, 40]
    logits = torch.full((3, N_POINTS), -20.0)
    for i, p in enumerate(decl_pts):
        logits[i, p] = 20.0  # mass on that point
    ev = mean_points(logits)  # ≈ decl_pts
    seat, bidder = 0, 3  # seat 0 is a DEFENDER (team 0 vs bidder team 1)
    sign = -1.0
    main_rank = torch.argsort(sign * ev)
    aux_vals = torch.tensor([acting_seat_margin(p, seat, bidder) for p in decl_pts]).float()
    aux_rank = torch.argsort(aux_vals)
    assert torch.equal(main_rank, aux_rank)  # defender prefers FEWER declaring pts


# --------------------------------------------------------------------- #
#  3. The (deal, decision-index, acting-seat) join                        #
# --------------------------------------------------------------------- #

def _fake_records(snaps, *, corrupt_steps=frozenset()):
    """One GameRecordGPU per snapshot; decisions follow the arena play line so
    decision k's player == plays[k][0] (the aligned case). e_q on every legal
    slot is set to float(step) so a landed label is self-identifying by step.
    ``corrupt_steps`` forces those decision indices' movers wrong in EVERY
    record — proving the acting-seat guard drops them (applied to every record
    so paired-half deals cannot re-supply the correct label)."""
    records = []
    for snap in snaps:
        decisions = []
        for k, (player, _dom) in enumerate(snap["plays"]):
            p = (int(player) + 1) % 4 if k in corrupt_steps else int(player)
            decisions.append(DecisionRecordGPU(
                player=p,
                e_q=torch.full((N_ACTIONS,), float(k)),
                action_taken=0,
                legal_mask=torch.ones(N_ACTIONS, dtype=torch.bool),
            ))
        records.append(GameRecordGPU(
            decisions=decisions,
            hands=[list(h) for h in snap["hands"]],
            decl_id=int(snap["decl_id"]),
            bid_value=int(snap["bid_value"]),
            bids=tuple(int(b) for b in snap["bids"]),
            bidder=int(snap["bidder"]),
        ))
    return records


def test_load_aux_table_kills_inf_pads():
    # A single hand-built record (no deal collisions) with an illegal -inf slot.
    rec = GameRecordGPU(
        decisions=[DecisionRecordGPU(
            player=2,
            e_q=torch.tensor([1., 1., 1., float("-inf"), 1., 1., 1.]),
            action_taken=0,
            legal_mask=torch.tensor([1, 1, 1, 0, 1, 1, 1], dtype=torch.bool),
        )],
        hands=[[0, 1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12, 13],
               [14, 15, 16, 17, 18, 19, 20], [21, 22, 23, 24, 25, 26, 27]],
        decl_id=6, bid_value=30, bids=(0, 0, 30, 0), bidder=2,
    )
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".pt") as f:
        torch.save({"results": [rec], "seeds": []}, f.name)
        (e_q, mask), = load_aux_table(f.name).values()
    assert float(e_q[3]) == 0.0 and float(mask[3]) == 0.0  # pad killed under mask
    assert torch.isfinite(e_q).all()


def test_join_lands_labels_on_decision_rows(tmp_path):
    snaps = _tiny_snaps()
    sp = tmp_path / "snaps.json"
    sp.write_text(json.dumps({"snapshots": snaps}))
    eqp = tmp_path / "eq.pt"
    torch.save({"results": _fake_records(snaps), "seeds": []}, eqp)
    table = load_aux_table(eqp)

    base = JudDataset(sp, split="all")            # aux off — the baseline row set
    ds = JudDataset(sp, split="all", aux_table=table)
    assert len(ds) == len(base)                    # aux never changes the row set
    assert ds.aux and not base.aux

    # Exactly the decision-row half is supervised (each hand: ~28 decision rows,
    # ~28 eval/child rows). If eval rows had wrongly matched, coverage → ~1.0.
    assert 0.40 < ds.aux_coverage() < 0.60

    n_sup = 0
    for i, key in enumerate(ds.keys):
        step = key[5]
        mask, target = ds.aux_masks[i], ds.aux_targets[i]
        if float(mask.sum()) > 0:
            n_sup += 1
            assert float(mask.sum()) == N_ACTIONS          # fixture: all 7 slots legal
            # Alignment: e_q was float(step), so a correctly-joined label reads
            # step/42 — a wrong-step join would fail this.
            assert torch.allclose(target, torch.full((N_ACTIONS,), step / MARGIN_SCALE))
        else:
            assert float(target.abs().sum()) == 0.0        # unsupervised → zero target
    assert n_sup > 0
    # The bid-root rows (step 0, pov = bidder) are always decision points → labeled.
    root = [i for i, k in enumerate(ds.keys) if k[5] == 0]
    assert root and all(float(ds.aux_masks[i].sum()) > 0 for i in root)


def test_acting_seat_guard_drops_misaligned_decisions(tmp_path):
    snaps = _tiny_snaps()
    sp = tmp_path / "snaps.json"
    sp.write_text(json.dumps({"snapshots": snaps}))
    # Corrupt decision index 5's mover in EVERY record → no deal can supply a
    # correct (deal, 5, seat) label, so all step-5 decision rows go unsupervised.
    eqp = tmp_path / "eq.pt"
    torch.save({"results": _fake_records(snaps, corrupt_steps={5}), "seeds": []}, eqp)
    ds = JudDataset(sp, split="all", aux_table=load_aux_table(eqp))

    step5 = [i for i, k in enumerate(ds.keys) if k[5] == 5]
    step4 = [i for i, k in enumerate(ds.keys) if k[5] == 4]
    assert step5 and step4
    assert all(float(ds.aux_masks[i].sum()) == 0.0 for i in step5)   # guard dropped
    assert any(float(ds.aux_masks[i].sum()) > 0 for i in step4)      # neighbors intact


# --------------------------------------------------------------------- #
#  4. masked MSE + aux training smoke                                     #
# --------------------------------------------------------------------- #

def test_masked_action_mse_ignores_unsupervised():
    pred = torch.tensor([[0.0, 0.0, 0.0]])
    target = torch.tensor([[1.0, 1.0, 99.0]])
    mask = torch.tensor([[1.0, 1.0, 0.0]])   # slot 2 unsupervised
    # (0-1)^2 + (0-1)^2 = 2 over 2 supervised slots → 1.0; slot 2 ignored.
    assert torch.isclose(masked_action_mse(pred, target, mask), torch.tensor(1.0))
    # All-unsupervised → finite 0 (clamp guards the divide).
    assert float(masked_action_mse(pred, target, torch.zeros_like(mask))) == 0.0


def test_seeded_arms_share_identical_trunk_init():
    # The fairness guarantee: at a fixed seed, arm H (aux off) and arm HP (aux
    # on) get byte-identical trunk weights — `self.net` is built before the
    # optional `self.aux`, so HP's extra draws come after the shared trunk.
    torch.manual_seed(123)
    h = JudNet(aux_per_action=False)
    torch.manual_seed(123)
    hp = JudNet(aux_per_action=True)
    for k in h.state_dict():
        assert torch.equal(h.state_dict()[k], hp.state_dict()[k]), k


def test_forward_aux_shapes():
    model = JudNet(aux_per_action=True)
    logits, aux = model.forward_aux(torch.zeros(5, FEATURE_DIM))
    assert logits.shape == (5, N_POINTS)
    assert aux.shape == (5, N_ACTIONS)


def test_aux_training_smoke_runs_and_saves_flag(tmp_path):
    snaps = _tiny_snaps(n_games=2)
    sp = tmp_path / "snaps.json"
    sp.write_text(json.dumps({"snapshots": snaps}))
    eqp = tmp_path / "eq.pt"
    torch.save({"results": _fake_records(snaps), "seeds": []}, eqp)

    out = tmp_path / "jud_hp.pt"
    metrics = train(
        corpus=sp, out_model=out, epochs=2, batch_size=64, lr=1e-3,
        patience=5, device="cpu", aux_labels=str(eqp), aux_lambda=1.0,
    )
    assert out.exists()
    assert "test_ce" in metrics
    saved = torch.load(out, map_location="cpu", weights_only=False)
    assert saved["aux_per_action"] is True
    # The saved arm-HP head reloads with its aux head intact.
    reloaded = load_jud_net(out)
    assert reloaded.aux_per_action
    logits, aux = reloaded.forward_aux(torch.zeros(2, FEATURE_DIM))
    assert logits.shape == (2, N_POINTS) and aux.shape == (2, N_ACTIONS)
