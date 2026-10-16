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
from scenario.field import CANON, N_FEATURES, FieldNet, features
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
        x[i, features(rules_for(decl), seat, own, rec, bid, bidder % 2)] = 1
    acts = forward(net, x)
    for i, (decl, seat, own, bidder, bid, plays, rec) in enumerate(rows):
        ref = net.scores(features(rules_for(decl), seat, own, rec, bid, bidder % 2))
        assert np.array_equal(acts.out[i], ref)
    assert acts.n_poles == 2 * (net.b1.shape[0] + net.b2.shape[0]) == 384


def test_picks_match_move(net):
    rows = _random_positions(100, seed=2)
    x = np.zeros((len(rows), N_FEATURES), np.uint8)
    actions = np.full((len(rows), 7), -1, np.int8)
    canon = np.zeros((len(rows), N_TILES), np.int64)
    for i, (decl, seat, own, bidder, bid, plays, rec) in enumerate(rows):
        rules = rules_for(decl)
        x[i, features(rules, seat, own, rec, bid, bidder % 2)] = 1
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
        x[i, features(rules_for(decl), seat, own, rec, bid, bidder % 2)] = 1
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
    x[0, features(rules, seat, own, rec, bid, bidder % 2)] = 1
    legal = tiles_of(rules.legal(own, rec.led))
    d = dict(x=x, legal=np.zeros((1, 28), bool), chosen=np.array([legal[0]], np.int8),
             actions=np.full((1, 7), -1, np.int8), deal=np.zeros(1, np.int32), index=np.zeros(1, np.int32),
             decl=np.array([decl], np.int8), viewer=np.array([seat], np.int8), own=np.array([own], np.uint32),
             bidder=np.array([bidder], np.int8), bid=np.array([bid], np.int16),
             plays=np.full((1, 28, 2), -1, np.int8), n_plays=np.array([len(plays)], np.int8),
             pour=np.zeros(1, np.int8), source=np.array("test"))
    for i, (s, t) in enumerate(plays):
        d["plays"][0, i] = (s, t)
    path = tmp_path / "s.npz"
    np.savez(path, **d)
    text = card(S.Sample(path), 0)
    assert "Your hand:" in text and "Trump:" in text and "Shown voids:" in text
    assert "partner holds" not in text.lower()
    assert text.count("\n") == 5
