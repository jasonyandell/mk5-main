# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0

"""The probe kit against the referee: the forward pass is the field net's
own arithmetic, the concept list compiles and reads the viewer only, the
cards show the viewer's information, and the audit statistics behave on
rows whose answer is known."""
from __future__ import annotations

import numpy as np
import pytest

from lem.probe import audit, concepts
from lem.probe.net import forward, load_net, picks
from scenario.field import CANON, FieldNet, features

N_FEATURES = 343     # the shipped net's layout, field-v2
V2 = "field-v2"
from scenario.record import empty_record
from scenario.rules import N_TILES, rules_for, tiles_of

NET = "zeb/net/margin_d250k_h128x64.fnet"


def _deal(rng: np.random.Generator) -> list[int]:
    tiles = rng.permutation(28)
    return [int(sum(1 << int(t) for t in tiles[7 * s:7 * s + 7])) for s in range(4)]


def _random_positions(n: int, seed: int = 1):
    """Random legal positions under random pip declarations, with the pour's
    contract (seat 0 bids 30)."""
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        hands = _deal(rng)
        decl = int(rng.integers(0, 7))
        rules = rules_for(decl)
        rec = empty_record(0)
        plays = []
        for _ in range(int(rng.integers(0, 27))):
            seat = rec.seat
            legal = tiles_of(rules.legal(hands[seat], rec.led))
            t = int(rng.choice(legal))
            hands[seat] &= ~(1 << t)
            plays.append((seat, t))
            rec = rec.advance(t, rules)
        out.append((decl, rec.seat, hands[rec.seat], 0, 30, plays, rec))
    return out


@pytest.fixture(scope="module")
def net() -> FieldNet:
    return load_net(NET)


def test_forward_matches_the_referee(net):
    rows = _random_positions(200)
    x = np.zeros((len(rows), N_FEATURES), np.uint8)
    for i, (decl, seat, own, bidder, bid, plays, rec) in enumerate(rows):
        x[i, features(rules_for(decl), seat, own, rec, bid, bidder % 2, V2)] = 1
    acts = forward(net, x)
    for i, (decl, seat, own, bidder, bid, plays, rec) in enumerate(rows):
        ref = net.scores(features(rules_for(decl), seat, own, rec, bid, bidder % 2, V2))
        assert np.array_equal(acts.out[i], ref)
    assert acts.n_poles == 2 * (net.b1.shape[0] + net.b2.shape[0]) == 384


def test_picks_match_move(net):
    rows = _random_positions(100, seed=2)
    x = np.zeros((len(rows), N_FEATURES), np.uint8)
    actions = np.full((len(rows), 7), -1, np.int8)
    canon = np.zeros((len(rows), N_TILES), np.int64)
    for i, (decl, seat, own, bidder, bid, plays, rec) in enumerate(rows):
        rules = rules_for(decl)
        x[i, features(rules, seat, own, rec, bid, bidder % 2, V2)] = 1
        legal = tiles_of(rules.legal(own, rec.led))
        actions[i, :len(legal)] = legal
        canon[i] = CANON[decl]
    acts = forward(net, x)
    got = picks(acts.out, actions, canon)
    for i, (decl, seat, own, bidder, bid, plays, rec) in enumerate(rows):
        assert got[i] == net.move(rules_for(decl), seat, own, rec, bid, bidder % 2)


def test_pole_signals_are_the_pre_activations(net):
    rows = _random_positions(20, seed=3)
    x = np.zeros((len(rows), N_FEATURES), np.uint8)
    for i, (decl, seat, own, bidder, bid, plays, rec) in enumerate(rows):
        x[i, features(rules_for(decl), seat, own, rec, bid, bidder % 2, V2)] = 1
    acts = forward(net, x)
    assert np.array_equal(acts.pole(0), acts.z1[:, 0])
    assert np.array_equal(acts.pole(1), -acts.z1[:, 0])
    assert np.array_equal(acts.pole(2 * 128 + 6), acts.z2[:, 3])
    assert np.array_equal(acts.pole(2 * 128 + 7), -acts.z2[:, 3])


def test_concepts_compile_and_read_the_viewer_only():
    from scenario_core import SilasBelief, SilasFrame, SilasQuery
    queries = concepts.compile_all()
    assert len(queries) == len(concepts.CONCEPTS) == len(set(concepts.NAMES))
    rows = _random_positions(30, seed=4)
    for decl, seat, own, bidder, bid, plays, rec in rows:
        frame = SilasFrame(decl, seat, own, bidder, bid, plays)
        belief = SilasBelief.uniform_resolving(frame, 0, 1)
        for q, c in zip(queries, concepts.CONCEPTS):
            e = q.summarize(belief, 10_000_000).event_probability
            assert e.authority.label == "exact", c.name
            assert e.value in (0, 1), c.name
    # a query that reads a hidden hand is refused on the same belief: the fence
    decl, seat, own, bidder, bid, plays, rec = rows[0]
    frame = SilasFrame(decl, seat, own, bidder, bid, plays)
    hidden = SilasQuery("(query (viewer me) (partner me p) (holds p d) (double d))")
    with pytest.raises(Exception, match="unresolved"):
        hidden.summarize(SilasBelief.uniform_resolving(frame, 0, 1), 10_000_000)


def test_concepts_agree_with_the_record():
    """A few concepts checked against the record's own reading."""
    from scenario_core import SilasBelief, SilasFrame
    queries = dict(zip(concepts.NAMES, concepts.compile_all()))

    def holds(name, frame):
        return queries[name].summarize(SilasBelief.uniform_resolving(frame, 0, 1), 10_000_000).event_probability.value == 1

    for decl, seat, own, bidder, bid, plays, rec in _random_positions(40, seed=5):
        frame = SilasFrame(decl, seat, own, bidder, bid, plays)
        rules = rules_for(decl)
        assert holds("lead", frame) == (len(rec.trick) == 0)
        assert holds("last", frame) == (len(rec.trick) == 3)
        assert holds("bidding-side", frame) == (seat % 2 == bidder % 2)
        assert holds("holds-double", frame) == any(t in (0, 2, 5, 9, 14, 20, 27) for t in tiles_of(own))
        assert holds("holds-ten", frame) == any(rules.count[t] == 10 for t in tiles_of(own))
        assert holds("forced", frame) == (len(tiles_of(rules.legal(own, rec.led))) == 1)
        assert holds("made", frame) == (rec.banked[bidder % 2] >= bid)
        trump = rules.follow_bits[7]
        assert holds("holds-trump", frame) == bool(own & trump)
        assert holds("no-trump-held", frame) == (not own & trump)


def test_concept_identity_is_the_text():
    a = concepts.identity()
    assert len(a) == 64 and a == concepts.identity()


def test_audit_statistics_on_known_rows():
    rng = np.random.default_rng(0)
    n = 10_000
    signal = rng.standard_normal(n)
    c_clean = signal > np.quantile(signal, 0.985)         # holds on the top 1.5%: clean
    c_half = rng.random(n) < 0.5                           # noise at base 0.5
    c_all = np.ones(n, bool)                               # true everywhere: excluded by the base cap
    labels = np.stack([c_clean, c_half, c_all], axis=1)
    base = labels.mean(axis=0)
    k = int(round(audit.TOP * n))
    top, prec, r = audit.score_pole(signal, labels, base, k)
    best, p, lo, hi, kind, runner, rp = audit.classify(prec, base, ["clean", "half", "all"], k)
    assert best == 0 and kind == "clean" and p == 1.0 and lo > 0.95
    assert abs(r[0]) > abs(r[1])
    lo2, hi2 = audit.wilson(90, 100)
    assert 0.82 < lo2 < 0.9 < hi2 < 0.96


def test_card_shows_the_viewer_only(tmp_path):
    from lem.probe import sample as S
    from lem.probe.cards import card
    # build a one-row sample by hand
    decl, seat, own, bidder, bid, plays, rec = _random_positions(1, seed=6)[0]
    rules = rules_for(decl)
    x = np.zeros((1, N_FEATURES), np.uint8)
    x[0, features(rules, seat, own, rec, bid, bidder % 2, V2)] = 1
    legal = tiles_of(rules.legal(own, rec.led))
    d = dict(x=x, legal=np.zeros((1, 28), bool), chosen=np.array([legal[0]], np.int8),
             actions=np.full((1, 7), -1, np.int8), deal=np.zeros(1, np.int32), index=np.zeros(1, np.int32),
             decl=np.array([decl], np.int8), viewer=np.array([seat], np.int8), own=np.array([own], np.uint32),
             bidder=np.array([bidder], np.int8), bid=np.array([bid], np.int16),
             plays=np.full((1, 28, 2), -1, np.int8), n_plays=np.array([len(plays)], np.int8),
             pour=np.zeros(1, np.int8), source=np.array("test"), layout=np.array(V2))
    for i, (s, t) in enumerate(plays):
        d["plays"][0, i] = (s, t)
    path = tmp_path / "s.npz"
    np.savez(path, **d)
    text = card(S.Sample(path), 0)
    assert "Your hand:" in text and "Trump:" in text and "Shown voids:" in text
    assert "partner holds" not in text.lower()
    assert text.count("\n") == 5


def test_ablate_candidates_equal_direct_fidelity(net):
    from lem.probe import ablate
    rows = _random_positions(300, seed=7)
    x = np.zeros((len(rows), N_FEATURES), np.uint8)
    legal = np.zeros((len(rows), N_TILES), bool)
    for i, (decl, seat, own, bidder, bid, plays, rec) in enumerate(rows):
        rules = rules_for(decl)
        x[i, features(rules, seat, own, rec, bid, bidder % 2, V2)] = 1
        legal[i, [CANON[decl][t] for t in tiles_of(rules.legal(own, rec.led))]] = True
    acts = forward(net, x)
    ref = ablate._pick_slots(acts.out, legal)
    rng = np.random.default_rng(0)
    h = net.b1.shape[0] + net.b2.shape[0]
    keep = (rng.random(h) < 0.3).astype(np.int64)
    cands = ablate._step_candidates(net, acts.a1, legal, ref, keep)
    assert set(cands) == {j for j in range(h) if not keep[j]}
    for j in list(cands)[::7]:
        k2 = keep.copy(); k2[j] = 1
        assert cands[j] == ablate.fidelity(net, acts.a1, legal, ref, k2)


def test_v2_concepts_compile_viewer_only_and_name_the_declaration():
    from scenario_core import SilasBelief, SilasFrame
    from scenario.atlas import deal
    queries = dict(zip(concepts.NAMES_V2, concepts.compile_all(concepts.CONCEPTS_V2)))
    assert concepts.identity(concepts.CONCEPTS_V2) != concepts.identity()
    assert concepts.NAMES_V2[:len(concepts.NAMES)] == concepts.NAMES          # v1 is a prefix of v2

    def holds(name, frame):
        e = queries[name].summarize(SilasBelief.uniform_resolving(frame, 0, 1), 10_000_000).event_probability
        assert e.authority.label == "exact", name
        return e.value == 1

    for s in range(1, 8):
        hands = deal(s)
        for decl in range(7):
            f = SilasFrame(decl, 0, hands[0], 0, 30, [])
            assert [holds(f"trump-is-{p}", f) for p in concepts.PIPS] == [q == decl for q in range(7)]
            assert not holds("trump-is-doubles", f) and not holds("no-trump-game", f)
        f7 = SilasFrame(7, 0, hands[0], 0, 30, [])
        assert holds("trump-is-doubles", f7) and not holds("no-trump-game", f7)
        assert not any(holds(f"trump-is-{p}", f7) for p in concepts.PIPS)
        f9 = SilasFrame(9, 0, hands[0], 0, 30, [])
        assert holds("no-trump-game", f9) and not holds("trump-is-doubles", f9)
        assert holds("lead-no-trump", f9)                 # no trump to hold
        for name in concepts.NAMES_V2:
            holds(name, f7); holds(name, f9)              # every v2 query answers exactly under 7 and 9


def test_belief_targets_are_exact_posteriors_and_ridge_recovers_a_linear_map(tmp_path):
    from lem.probe import belief
    from lem.probe import sample as S
    # a small sample built from random positions
    rows = _random_positions(40, seed=21)
    d = dict(x=np.zeros((len(rows), N_FEATURES), np.uint8), legal=np.zeros((len(rows), 28), bool),
             chosen=np.zeros(len(rows), np.int8), actions=np.full((len(rows), 7), -1, np.int8),
             deal=np.arange(len(rows), dtype=np.int32), index=np.arange(len(rows), dtype=np.int32),
             decl=np.zeros(len(rows), np.int8), viewer=np.zeros(len(rows), np.int8), own=np.zeros(len(rows), np.uint32),
             bidder=np.zeros(len(rows), np.int8), bid=np.full(len(rows), 30, np.int16),
             plays=np.full((len(rows), 28, 2), -1, np.int8), n_plays=np.zeros(len(rows), np.int8),
             pour=np.zeros(len(rows), np.int8), source=np.array("test"), layout=np.array(V2))
    for i, (decl, seat, own, bidder, bid, plays, rec) in enumerate(rows):
        d["decl"][i], d["viewer"][i], d["own"][i], d["n_plays"][i] = decl, seat, own, len(plays)
        for k, (s_, t_) in enumerate(plays):
            d["plays"][i, k] = (s_, t_)
    path = tmp_path / "s.npz"; np.savez(path, **d)
    smp = S.Sample(path)
    post, prior, mask = belief.targets(smp, np.arange(len(rows)))
    for i, (decl, seat, own, bidder, bid, plays, rec) in enumerate(rows):
        hidden = [t for t in range(28) if not (rec.played >> t) & 1 and not (own >> t) & 1]
        canon = CANON[decl]
        assert mask[i].sum() == 3 * len(hidden)
        for t in hidden:
            col = post[i, :, canon[t]]
            assert abs(col.sum() - 1.0) < 1e-9                      # the tile is with exactly one other seat
            if not any(rec.forbid[(seat + rel) % 4] for rel in (1, 2, 3)):
                assert np.allclose(col, prior[i, :, canon[t]])     # no voids shown: the posterior is the counting prior
            for rel in (1, 2, 3):
                if (rec.forbid[(seat + rel) % 4] >> t) & 1:
                    assert col[rel - 1] == 0.0                     # a shown void is a zero
    rng = np.random.default_rng(0)
    F = rng.standard_normal((500, 6)); Wt = rng.standard_normal((7, 3))
    Y = np.concatenate([F, np.ones((500, 1))], axis=1) @ Wt
    M = rng.random((500, 3)) < 0.8
    W = belief.ridge_fit(F, Y, M, lam=1e-6)
    assert belief.r2(belief.ridge_predict(F, W), Y, M) > 0.999


def test_fidelity_collects_the_teachers_q_and_regret_is_zero_at_agreement(tmp_path):
    from lem.probe import fidelity
    from lem.probe import sample as S
    from lem.probe.net import forward, load_net
    pour_dir = "/Users/jason/code/mk5-main/scratch/atlas/pour_smoke"
    import pathlib
    if not pathlib.Path(pour_dir).exists():
        pytest.skip("the smoke pour is not on this machine")
    S.build(pour_dir, tmp_path / "part.npz", every=1, log=lambda m: None)
    S.merge(tmp_path / "sample.npz", [tmp_path / "part.npz"], log=lambda m: None)
    smp = S.Sample(tmp_path / "sample.npz")
    d = fidelity.collect(smp, pour_dir, 0, log=None)
    (tmp_path / "q").mkdir()
    np.savez(tmp_path / "q" / "q_0.npz", **d)
    assert len(d["rows"]) == len(smp) and np.isfinite(d["q"]).sum() == (smp.actions >= 0).sum()
    net = load_net("zeb/net/margin_d250k_h128x64.fnet")
    acts = forward(net, smp.x)
    r, gap, n_legal = fidelity.regret(smp, acts, tmp_path / "q")
    assert np.isfinite(r).all() and (r >= -1e-12).all() and (gap >= 0).all()
    assert set(n_legal.tolist()) <= set(range(2, 8))
    # where the net picks the teacher's own choice, and that choice is the teacher's best, regret is 0
    from lem.probe.net import picks
    pick = picks(acts.out, smp.actions, smp.canon())
    same = pick == smp.chosen
    assert same.any()
    for i in np.flatnonzero(same)[:20]:
        qi = d["q"][i] * d["sign"][i]
        if np.isclose(qi[np.argmax(qi)], qi[list(smp.actions[i]).index(smp.chosen[i])]):
            assert r[i] == 0


def test_aa_calibration_counts_are_near_their_targets_on_a_simulated_calibrated_estimator():
    """A/A bookkeeping (#191): with exact normal intervals on independent halves, the
    interval misses the other half's point about 0.166 of the time, |z| > 1.96 about 0.05,
    and no-overlap about 0.006; the deal bootstrap recovers a cluster mean's spread."""
    from lem.probe import aa
    rng = np.random.default_rng(0)
    est = []
    for _ in range(20000):
        pA, pB = rng.normal(0.5, 0.01, 2)
        est.append({"pA": pA, "loA": pA - 1.96 * 0.01, "hiA": pA + 1.96 * 0.01, "seA": 0.01,
                    "pB": pB, "loB": pB - 1.96 * 0.01, "hiB": pB + 1.96 * 0.01, "seB": 0.01})
    c = aa.compare(est)
    assert abs(c["miss_other_point"] - c["miss_other_point_target"]) < 0.01
    assert abs(c["z_abs_gt_1_96"] - 0.05) < 0.006 and abs(c["z_sd"] - 1.0) < 0.03
    assert c["no_overlap"] < 0.012 and c["degenerate"] == 0
    from lem.probe.audit import wilson
    lo, hi = wilson(2174, 2174)
    one = {"pA": 1.0, "loA": lo, "hiA": hi, "seA": (hi - lo) / 3.92, "pB": 1.0, "loB": lo, "hiB": hi, "seB": (hi - lo) / 3.92}
    assert aa.compare([one])["miss_other_point"] == 0.0            # Wilson's hi at p = 1 is 1 up to rounding
    # deal bootstrap: 400 deals of 8 rows; the rows of a deal share a deal effect
    deal = np.repeat(np.arange(400), 8)
    y = (rng.normal(0, 1, 400)[deal] + rng.normal(0, 0.3, 3200) > 0).astype(float)
    bs = aa.deal_bootstrap(y, deal, rng, boot=2000)
    truth_se = np.std([y[rng.integers(0, 400, 400)[:, None] * 8 + np.arange(8)].mean() for _ in range(2000)])
    assert abs(bs.std() - truth_se) / truth_se < 0.15 and bs.std() > 1.5 * np.sqrt(y.mean() * (1 - y.mean()) / 3200)
    a, b = aa.halves(deal)
    assert a.sum() == b.sum() == 1600 and not (a & b).any()
    # R² bootstrap equals the plain R² when every deal is drawn once
    pred, yy = rng.normal(0, 1, (200, 5)), rng.normal(0, 1, (200, 5)); m = np.ones_like(yy, bool); d = np.arange(200)
    class Once:  # a "rng" that draws every deal exactly once
        def integers(self, lo, hi, size): return np.tile(np.arange(hi), (size[0], 1))
    r = aa.r2_bootstrap(pred, yy, m, d, Once(), boot=3)
    e, v = (yy - pred).ravel(), (yy - yy.mean()).ravel()
    assert np.allclose(r, 1 - e @ e / (v @ v))


def test_squeeze_fold_is_exact_on_pip_positions_and_families_cover_the_list():
    """The folded v2 net scores every pip position exactly as the v3 net it came from,
    and every v1 concept has exactly one family (the squeeze's family test, G2)."""
    from lem.probe import concepts, squeeze
    from scenario.field import N_FEATURES, OFFSET, FieldNet
    rng = np.random.default_rng(3)
    h1, h2 = 16, 8
    v3 = FieldNet(rng.integers(-300, 300, (N_FEATURES, h1)).astype(np.int64), rng.integers(-5000, 5000, h1).astype(np.int64),
                  rng.integers(-300, 300, (h2, h1)).astype(np.int64), rng.integers(-5000, 5000, h2).astype(np.int64),
                  rng.integers(-300, 300, (28, h2)).astype(np.int64), rng.integers(-5000, 5000, 28).astype(np.int64), 6, 6, "field-v3")
    v2 = squeeze.fold_to_v2(v3)
    for _ in range(50):
        active = sorted(rng.choice(OFFSET["kind"], 30, replace=False).tolist())
        assert np.array_equal(v3.scores(active + [OFFSET["kind"]]), v2.scores(active))
    names = [c.name for c in concepts.CONCEPTS]
    members = [c for cs in squeeze.FAMILIES.values() for c in cs]
    assert sorted(members) == sorted(names) and len(members) == len(set(members))
    # the union table: P(a or b) on a toy top set, cap respected, diagonal excluded
    tl = np.array([[1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1]], bool)
    u = squeeze.union_table(tl, np.ones((3, 3), bool))
    assert u[0, 1] == 0.75 and u[0, 2] == 0.75 and u[0, 0] == -1


def test_label_files_carry_their_concept_list_and_the_sample_honours_the_trainers_offset(tmp_path):
    """A label file names the list it was made with (v1 or v2) and the kit reads either; a sample
    of a later pour takes the trainer's deal offset, so its held-out rows are the trainer's."""
    from lem.probe import concepts
    for name, cl in concepts.LISTS.items():
        f = tmp_path / f"{name}.npz"
        concepts.save(f, np.zeros((3, len(cl)), bool), cl)
        with np.load(f) as z:
            assert concepts.list_of(str(z["identity"])) is cl and len(z["names"]) == len(cl)
    with pytest.raises(ValueError):
        concepts.save(tmp_path / "bad.npz", np.zeros((3, 5), bool), concepts.CONCEPTS)
    with pytest.raises(ValueError):
        concepts.list_of("0" * 64)
    assert concepts.identity(concepts.CONCEPTS).startswith("13e4d525")


def test_formation_checkpoints_are_early_steps_then_epoch_ends():
    from lem.probe.squeeze import formation_steps
    s = formation_steps(3_915_000)                    # 15,293 steps an epoch at batch 256
    assert s[:7] == (0, 30, 100, 300, 1_000, 3_000, 10_000) and s[7] == 15_293 and s[-1] == 60 * 15_293 and len(s) == 15
    assert formation_steps(1_000, epochs=3) == (0, 4, 8, 12, 30, 100, 300, 1_000, 3_000, 10_000)


def test_formation_reads_first_clean_steps_and_family_kept():
    """A pole is first clean at the earliest checkpoint after which it stays clean; family kept
    compares its label there with its final label."""
    from lem.probe import formation
    def audit(rows):
        return {"rows": [{"pole": p, "kind": k, "label": l, "precision": 0.95} for p, k, l in rows]}
    ser = [(0, audit([(0, "mixed", "lead"), (1, "clean", "holds-boss-trump"), (2, "dead", "x")])),
           (10, audit([(0, "clean", "last"), (1, "partial", "holds-master"), (2, "clean", "holds-ten")])),
           (20, audit([(0, "clean", "third"), (1, "clean", "holds-master"), (2, "clean", "count-on-trick")]))]
    ps = {p["pole"]: p for p in formation.poles(ser)}
    assert ps[0]["first_clean_step"] == 10 and ps[0]["first_label"] == "last" and ps[0]["final_label"] == "third"
    assert ps[1]["first_clean_step"] == 20 and ps[1]["first_label"] == "holds-master"
    assert ps[2]["first_clean_step"] == 10
    k = formation.kept_share(list(ps.values()))
    assert k["n"] == 3 and k["kept"] == 1.0 and abs(k["null"] - 1 / 3) < 1e-9          # three final families, one pole each
