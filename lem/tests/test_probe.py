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
    ser = [(0, audit([(0, "unnamed", "lead"), (1, "clean", "holds-boss-trump"), (2, "dead", "x")])),
           (10, audit([(0, "clean", "last"), (1, "partial", "holds-master"), (2, "clean", "holds-ten")])),
           (20, audit([(0, "clean", "third"), (1, "clean", "holds-master"), (2, "clean", "count-on-trick")]))]
    ps = {p["pole"]: p for p in formation.poles(ser)}
    assert ps[0]["first_clean_step"] == 10 and ps[0]["first_label"] == "last" and ps[0]["final_label"] == "third"
    assert ps[1]["first_clean_step"] == 20 and ps[1]["first_label"] == "holds-master"
    assert ps[2]["first_clean_step"] == 10
    k = formation.kept_share(list(ps.values()))
    assert k["n"] == 3 and k["kept"] == 1.0 and abs(k["null"] - 1 / 3) < 1e-9          # three final families, one pole each


def test_tree_rows_are_deal_disjoint_and_r2_recovers_a_linear_split():
    from lem.probe import tree
    deal = np.repeat(np.arange(500), 100)
    fit, ev = tree.rows(deal, n=4000)
    assert len(fit) + len(ev) == 4000 and not set(deal[fit]) & set(deal[ev])
    rng = np.random.default_rng(1)
    X, Xv = rng.normal(size=(400, 3)), rng.normal(size=(400, 3))
    w = np.array([1.0, -2.0, 0.5])
    assert tree.r2(X @ w, X, Xv @ w, Xv) > 0.999
    assert tree.r2(rng.normal(size=400), X, rng.normal(size=400), Xv) < 0.05


def test_causal_downstream_is_the_forward_pass_and_ablating_an_off_neuron_changes_nothing(net):
    from lem.probe.causal import downstream
    rows = _random_positions(200, seed=5)
    x = np.zeros((len(rows), N_FEATURES), np.uint8)
    for i, (decl, seat, own, bidder, bid, plays, rec) in enumerate(rows):
        x[i, features(rules_for(decl), seat, own, rec, bid, bidder % 2, V2)] = 1
    acts = forward(net, x)
    assert np.array_equal(downstream(net, x, 1, 0, None), acts.out)
    for layer, a in ((1, acts.a1), (2, acts.a2)):
        j = int(np.argmin((a > 0).sum(axis=0)))
        out = downstream(net, x, layer, j, 0)
        off = a[:, j] == 0
        assert np.array_equal(out[off], acts.out[off])        # rows where it was off are untouched
    assert np.array_equal(downstream(net, x, 2, 0, None, add=0), acts.out)   # a zero dose is no dose


def test_gus_reads_the_last_real_token_and_ignores_right_padding():
    import torch
    from lem.observer.data import SEAT, encode
    from lem.observer.model import Config
    from lem.probe.gus import forward, load_model
    model, _ = load_model(None, untrained_seed=7, config=Config().__dict__)
    rng = np.random.default_rng(3)
    seqs = []
    for n in (0, 3, 9):
        hand = int(sum(1 << int(t) for t in rng.choice(28, 7, replace=False)))
        plays = [(int(i % 4), int(t)) for i, t in enumerate(rng.choice(28, n, replace=False))]
        seqs.append(np.append(encode(hand, 1, 0, 4, plays), SEAT))
    lengths = np.array([len(q) for q in seqs])
    tokens = np.zeros((len(seqs), int(lengths.max())), np.int64)
    for i, q in enumerate(seqs):
        tokens[i, :len(q)] = q
    acts = forward(model, tokens, lengths)
    for i, q in enumerate(seqs):
        with torch.inference_mode():
            logits, traces = model(torch.from_numpy(q[None]), inspect=True)
        assert np.allclose(acts.out[i], logits[0, -1].numpy(), atol=1e-5)
        assert np.allclose(acts.z1[i], traces[0]["pre"][0, -1].numpy(), atol=1e-5)
        assert np.allclose(acts.z2[i], traces[1]["pre"][0, -1].numpy(), atol=1e-5)
    assert acts.n_poles == 2 * 2 * 256


def test_conditioning_on_the_stratum_leaves_a_stratum_only_unit_nothing_to_rank():
    from lem.probe.audit import condition
    from lem.probe.net import Activations
    rng = np.random.default_rng(0)
    strata = rng.integers(0, 28, 5000)
    pos_only = (strata * 7 % 11).astype(np.float64)[:, None]           # reads only the stratum
    real = rng.normal(size=(5000, 1)) + 3 * (strata % 2)[:, None]       # a stratum shift plus a signal
    z = np.concatenate([pos_only, real], axis=1)
    c = condition(Activations(z, z, z, z, z), strata)
    assert np.allclose(c.z1[:, 0], 0)                                    # nothing left to rank
    for v in np.unique(strata):                                         # within every stratum: mean 0, sd 1
        r = strata == v
        assert abs(c.z1[r, 1].mean()) < 1e-9 and abs(c.z1[r, 1].std() - 1) < 1e-9
    assert np.array_equal(c.a1, z)                                       # on/off stays raw


def test_merge_numbers_a_later_part_after_all_of_the_earlier_pours_deals(tmp_path):
    """#225: pour A's pip half has 217,376 deals but its sample may end earlier; the trainer
    numbers the next pour's deals after all of them, and so must the merge."""
    from lem.probe.sample import COLUMNS, merge
    parts = []
    for k, (n_deals, sampled) in enumerate(((217_376, [0, 10, 217_370]), (32_624, [4, 14]))):
        pour = tmp_path / f"pour{k}"
        pour.mkdir()
        np.savez(pour / "pour.npz", hands=np.zeros((n_deals, 4), np.uint32))
        n = len(sampled)
        cols = {c: np.zeros((n, 2), np.uint8) if c == "x" else np.zeros(n, np.int64) for c in COLUMNS}
        cols["deal"] = np.array(sampled, np.int64)
        part = tmp_path / f"part{k}.npz"
        np.savez(part, **cols, source=np.array(str(pour)), layout=np.array("field-v3"))
        parts.append(part)
    merge(tmp_path / "merged.npz", parts, log=lambda m: None)
    with np.load(tmp_path / "merged.npz") as z:
        assert z["deal"].tolist() == [0, 10, 217_370, 217_376 + 4, 217_376 + 14]
        assert z["pour"].tolist() == [0, 0, 0, 1, 1]


def test_stratified_gap_compares_rows_only_within_their_stratum():
    """A concept that holds mostly in an expensive stratum must not look costly by itself (#205)."""
    from lem.probe.causal import stratified_gap
    strata = np.repeat([0, 1], 1000)
    v = np.where(strata == 1, 10.0, 0.0)                       # stratum 1 is costly for everyone
    c = np.zeros(2000, bool)
    c[:100] = True                                             # rare in stratum 0
    c[1000:1900] = True                                        # common in stratum 1
    rows = np.ones(2000, bool)
    assert abs(stratified_gap(v, rows, c, strata)) < 1e-12     # no gap within either stratum
    v2 = v + np.where(c, 1.0, 0.0)                             # a real one-unit gap inside the concept
    assert abs(stratified_gap(v2, rows, c, strata) - 1.0) < 1e-12
    assert np.nanmean(v[c]) - np.nanmean(v[~c]) > 5            # the unstratified gap is fooled


def test_family_u_is_half_when_families_mean_nothing_and_high_for_a_coarse_unit():
    """#200: the subset correction makes a big family no likelier to win than a small one."""
    from lem.probe.squeeze import family_u, subset_max_below
    assert abs(subset_max_below(3, 5, 2) - 3 / 10) < 1e-12          # C(3,2)/C(5,2)
    assert subset_max_below(1, 5, 2) == 0.0
    rng = np.random.default_rng(1)
    c = 12
    fam = np.repeat(np.arange(4), 3)
    valid = np.ones((c, c), bool)
    vals = []
    for _ in range(4000):                                           # random excesses: mean u about 0.5
        e = rng.random((c, c))
        vals.append(family_u(np.maximum(e, e.T), valid, fam))
    assert abs(np.mean(vals) - 0.5) < 0.02
    e = np.zeros((c, c))
    e[0, 1] = e[1, 0] = 1.0                                          # the best union lies in one family:
    assert abs(family_u(e, valid, fam) - 60 / 66) < 1e-12           # misses it 54/66, ties it 12/66 (counted half)


def test_a_planted_fold_is_one_real_unit_with_both_donors_input_and_output(net):
    """#229: the fold merges unit k into unit j (inputs, bias, outgoing weights) and removes k."""
    from lem.probe.coarse import cos, fold
    rows = _random_positions(200, seed=9)
    x = np.zeros((len(rows), N_FEATURES), np.uint8)
    for i, (decl, seat, own, bidder, bid, plays, rec) in enumerate(rows):
        x[i, features(rules_for(decl), seat, own, rec, bid, bidder % 2, V2)] = 1
    a, f = forward(net, x), forward(fold(net, 3, 5), x)
    assert np.array_equal(f.z1[:, 3], a.z1[:, 3] + a.z1[:, 5])          # one unit, both donors' input
    assert (f.a1[:, 5] == 0).all()                                     # k never fires again
    assert np.array_equal(fold(net, 3, 5).w2[:, 3], net.w2[:, 3] + net.w2[:, 5])
    assert cos(np.array([1.0, 0.0]), np.array([0.0, 2.0])) == 0.0 and cos(np.zeros(2), np.ones(2)) is None
