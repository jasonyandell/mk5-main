# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0

"""The convention overlay: a sender that plays the rule's tile at a gated
position, a reader that draws its worlds from the rule-only posterior.
Referee = core to the bit in every arm (S, R, SR), the override flag,
proposals and fallbacks included; the sender fires exactly at its gate;
the reader keeps exactly the worlds the rule allows; with no gate in play
both halves are the plain player byte for byte."""
import random

import pytest
import scenario_core

from forge.zeb.game import legal_actions

from arena.walt_play import parse_walt_spec
from scenario.atlas import _state, deal
from scenario.compiled import CompiledWalt
from scenario.convention import NAMES, Convention, trumps
from scenario.fiber import Fiber
from scenario.planner import Config, Rung, Walt, situation_from_state
from scenario.record import empty_record, remaining_hand
from scenario.rules import bit, rules_for, tiles_of
from scenario.tests.test_compiled import _same, _same_analysis
from scenario.tests.test_field import NET

SIXES = 6
T = {f"{h}{l}": h * (h + 1) // 2 + l for h in range(7) for l in range(h + 1)}  # "64" -> the 6-4's id


def _with(cfg: Config, **kw) -> Config:
    return Config(**{**cfg.__dict__, **kw})


def _game(seed: int, conv: Convention):
    """A seeded deal played by random legal choice, except that every seat
    follows the convention's sender at its gates (the bidder's team obeys
    the rule, so the real deal is in the reader's posterior). Returns the
    plays and the play indices where a gate fired."""
    hands, decl, bidder = deal(seed), seed % 7, seed % 4
    rules, rng = rules_for(decl), random.Random(seed)
    rec, plays, fired = empty_record(bidder), (), []
    while len(plays) < 28:
        seat = rec.seat
        own = hands[seat] & ~rec.played
        legal = tiles_of(rules.legal(own, rec.led))
        allowed = conv.sends(rules, seat, bidder, own, rec)
        if allowed is not None:
            fired.append(len(plays))
            legal = [t for t in legal if (allowed >> t) & 1]
        tile = rng.choice(legal)
        plays += ((seat, tile),)
        rec = rec.advance(tile, rules)
    return hands, decl, bidder, plays, fired


def _at(hands, decl, bidder, plays, k: int):
    """The state with the first k plays down."""
    rules = rules_for(decl)
    rec = empty_record(bidder)
    for _, t in plays[:k]:
        rec = rec.advance(t, rules)
    return _state(tuple(tuple(tiles_of(h)) for h in hands), bidder, 30, decl, plays[:k], rec)


def _positions(name: str, n_send: int = 3, n_read: int = 3):
    """Sender positions (the gated mover to play) and reader positions (the
    gated mover's partner to play, after the gated decision, with a choice)."""
    conv = Convention(name)
    send, read = [], []
    for seed in range(400):
        hands, decl, bidder, plays, fired = _game(seed, conv)
        for k in fired:
            state = _at(hands, decl, bidder, plays, k)
            if len(send) < n_send and len(legal_actions(state)) > 1:
                send.append(state)
            reader = (plays[k][0] + 2) % 4
            for j in range(k + 1, 28):
                if plays[j][0] == reader:
                    later = _at(hands, decl, bidder, plays, j)
                    if len(legal_actions(later)) > 1 and len(read) < n_read:
                        read.append(later)
                    break
        if len(send) == n_send and len(read) == n_read:
            return send, read
    raise AssertionError(f"too few gated positions for {name}")


POSITIONS = {name: _positions(name) for name in NAMES}

PARITY_CONFIGS = [
    Config((Rung(12, 6),), "make", seed=1),
    Config((Rung(3, 3), Rung(8, 6)), "points", seed=2),
    Config((Rung(10, None, 4),), "make", seed=3, field=NET),
]


def _arms(name: str):
    return [Convention(name, send=s, read=r) for s, r in ((True, False), (False, True), (True, True))]


def test_compiled_matches_the_planner_under_every_convention_and_arm():
    sent = filtered = 0
    for name in NAMES:
        send, read = POSITIONS[name]
        for conv in _arms(name):
            for i, base in enumerate(PARITY_CONFIGS):
                if base.field is not None and name != "handoff":
                    continue  # the net plays pip trumps; one convention over it is enough
                cfg = _with(base, convention=conv)
                ref, fast = Walt(cfg), CompiledWalt(cfg)
                for state in send + read:
                    if cfg.field is not None and state.decl_id > 6:
                        continue
                    a = ref.analyze(state, 30)
                    _same_analysis(a, fast.analyze(state, 30))
                    counts = (ref.nodes, ref.proposals, ref.fallbacks, ref.sent)
                    assert counts == (fast.nodes, fast.proposals, fast.fallbacks, fast.sent)
                    _same(ref.decide(state, 30), fast.decide(state, 30))
                    assert counts == (ref.nodes, ref.proposals, ref.fallbacks, ref.sent)
                    assert counts == (fast.nodes, fast.proposals, fast.fallbacks, fast.sent)
                    sent += ref.sent
                    filtered += ref.proposals > len(a.worlds) or 0.0 in a.weights
    assert sent >= 30 and filtered >= 12


def test_a_spent_budget_falls_back_to_the_uniform_in_both_engines():
    falls = 0
    for name in NAMES:
        _, read = POSITIONS[name]
        cfg = Config((Rung(64, 6),), "make", seed=5, draw_budget=64, convention=Convention(name, send=False))
        plain = _with(cfg, convention=None)
        for state in read:
            ref, fast = Walt(cfg), CompiledWalt(cfg)
            a = ref.analyze(state, 30)
            _same_analysis(a, fast.analyze(state, 30))
            assert (ref.proposals, ref.fallbacks) == (fast.proposals, fast.fallbacks)
            if ref.fallbacks:
                _same_analysis(a, Walt(plain).analyze(state, 30))
                falls += 1
    assert falls >= 2


def _position(hands: list[str], plays: list[tuple[int, str]], bidder: int = 0, decl: int = SIXES):
    """The state after `plays` from four whole hands (seven tiles each,
    space-separated "hl" names), every play checked legal."""
    masks = tuple(sum(bit(T[x]) for x in h.split()) for h in hands)
    assert sorted(t for m in masks for t in tiles_of(m)) == list(range(28))
    rules, rec = rules_for(decl), empty_record(bidder)
    for seat, x in plays:
        assert seat == rec.seat and (rules.legal(masks[seat] & ~rec.played, rec.led) >> T[x]) & 1
        rec = rec.advance(T[x], rules)
    return _at(masks, decl, bidder, tuple((s, T[x]) for s, x in plays), len(plays))


def _sends(name, state):
    sit = situation_from_state(state, 30)
    rules = rules_for(state.decl_id)
    own = remaining_hand(state, sit.viewer)
    return Convention(name).sends(rules, sit.viewer, sit.bidder, own, sit.record)


H1 = ["62 44 33 22 11 00 52", "60 50 51 53 54 40 41", "66 61 55 43 42 31 30", "65 64 63 32 21 20 10"]
H2 = ["62 44 33 22 11 00 52", "60 50 51 53 54 40 41", "65 61 55 43 42 31 30", "66 64 63 32 21 20 10"]
B1 = ["44 62 33 22 11 00 54", "40 60 50 51 53 52 41", "55 32 10 20 21 31 30", "66 61 65 64 63 43 42"]
B2 = ["44 62 33 22 11 00 54", "64 60 50 51 53 52 61", "55 32 10 20 21 31 30", "66 65 63 40 41 43 42"]
D1 = ["65 66 62 61 33 22 11", "63 64 00 10 20 21 30", "50 55 32 31 40 41 42", "60 43 44 51 52 53 54"]
D2 = ["66 62 61 33 22 11 00", "63 64 65 10 20 21 30", "55 32 31 40 41 42 43", "60 50 51 52 53 54 44"]

# Sixes trump; seat 0 bids. Each case: (convention, hands, plays before the
# decision, the allowed tiles at the gate, or None for no gate).
GATES = [
    # The partner follows the bidder's trump lead holding the top trump: hand it over.
    ("handoff", H1, [(0, "62"), (1, "60")], "66"),
    # The top trump is with an opponent: no gate.
    ("handoff", H2, [(0, "62"), (1, "60")], None),
    # The bidder led an off, not a trump: no gate.
    ("handoff", H1, [(0, "44"), (1, "40")], None),
    # The partner cannot follow the bidder's winning double: no count but the biggest.
    ("biggest", B1, [(0, "44"), (1, "40")], "55 10 20 21 31 30"),
    # The double is trumped: no gate.
    ("biggest", B2, [(0, "44"), (1, "64")], None),
    # The bidder leads; partner shown void in trump; a trump is out against him: the top.
    ("donation", D1, [(0, "65"), (1, "63"), (2, "50"), (3, "60")], "66"),
    # The partner, void in trump, on the top trump lead: his largest count.
    ("donation", D2, [(0, "66"), (1, "63")], "55"),
    # On a lower trump lead (the 6-6 is still out): no count.
    ("donation", D2, [(0, "62"), (1, "63")], "31 40 42 43"),
]


@pytest.mark.parametrize("name,fixed,plays,allowed", GATES)
def test_the_sender_fires_exactly_at_the_gate(name, fixed, plays, allowed):
    state = _position(fixed, plays)
    got = _sends(name, state)
    assert got == (None if allowed is None else sum(bit(T[x]) for x in allowed.split()))
    plain = Config((Rung(12, None),), "make", seed=7)
    for walt in (Walt(_with(plain, convention=Convention(name, read=False))),
                 CompiledWalt(_with(plain, convention=Convention(name, read=False)))):
        tile, q = walt.decide(state, 30)
        base_tile, base_q = Walt(plain).decide(state, 30)
        assert q == base_q                              # the plan runs, its values are reported
        assert walt.sent == (allowed is not None)
        if allowed is None:
            assert tile == base_tile
        else:
            assert (got >> tile) & 1 and q[tile] == max(q[t] for t in tiles_of(got))


def test_the_handoff_gates_hold_the_book_s_statements():
    """Sixes: the bidder leads the 6-2, the partner holds the 6-6 and the 6-1.
    The table hands over the top; the book's reading says a top trump from
    partner is his only one, so the reader allows the 6-6 only when it is
    alone and otherwise any other trump."""
    state = _position(H1, [(0, "62"), (1, "60")])
    sit = situation_from_state(state, 30)
    rules, own = rules_for(SIXES), remaining_hand(state, 2)
    table, book = Convention("handoff"), Convention("handoff-singleton")
    assert table.reads(rules, 2, 0, own, sit.record) == bit(T["66"])
    assert book.reads(rules, 2, 0, own, sit.record) == bit(T["61"])
    assert book.sends(rules, 2, 0, own, sit.record) == bit(T["66"])   # same sender
    alone = own & ~bit(T["61"]) | bit(T["11"])
    assert book.reads(rules, 2, 0, alone, sit.record) is None         # forced: no gate
    assert trumps(rules) == sum(bit(T[f"6{l}"]) for l in range(7))
    assert trumps(rules_for(9)) == 0                                  # no-trump has none


def _lowest(hands: list[str], plays: list[tuple[int, str]], bidder: int = 0, decl: int = SIXES):
    """A whole hand: `plays`, then every seat plays its lowest legal tile."""
    masks = _position(hands, plays, bidder, decl).hands
    masks = tuple(sum(bit(t) for t in h) for h in masks)
    rules, rec, out = rules_for(decl), empty_record(bidder), []
    for _, x in plays:
        out.append((rec.seat, T[x]))
        rec = rec.advance(T[x], rules)
    while len(out) < 28:
        tile = tiles_of(rules.legal(masks[rec.seat] & ~rec.played, rec.led))[0]
        out.append((rec.seat, tile))
        rec = rec.advance(tile, rules)
    return masks, decl, bidder, tuple(out)


# Biggest, read by the bidder: the partner sluffs the 3-2 on the bidder's
# double-four; in every world where he also held the 5-5 or the 6-4 he broke
# the rule. (The real partner held one count: no gate in the real deal.)
B3 = ["44 62 33 22 11 00 54", "40 60 50 51 53 52 55", "32 10 20 21 31 30 61", "66 41 65 64 63 43 42"]


def _late(cfg: Config):
    """A position with a choice whose fiber is enumerated (it fits the
    rung's worlds) and where the reading deletes some of its worlds but not
    all."""
    conv, n = cfg.convention, cfg.rungs[0].n_worlds
    games = (_game(seed, conv)[:4] for seed in range(400))
    if conv.name == "biggest":
        games = iter([_lowest(B3, [(0, "44"), (1, "40"), (2, "32"), (3, "43")])])
    for hands, decl, bidder, plays in games:
        for j in range(16, 26):
            state = _at(hands, decl, bidder, plays, j)
            sit = situation_from_state(state, 30)
            rec = sit.record
            fiber = Fiber(sit.viewer, remaining_hand(state, sit.viewer), rec.played,
                          tuple(rec.hand_size(x) for x in range(4)), rec.forbid)
            if len(legal_actions(state)) > 1 and fiber.size <= n:
                if set(Walt(cfg).convention_likelihood(sit, fiber.worlds())) == {0.0, 1.0}:
                    return state
    raise AssertionError(f"no late reader position for {conv.name}")


def _handoff_kept(sit, world, singleton: bool) -> bool:
    """The reading, from the rules alone: replay the partner's plays."""
    rules, tr = rules_for(sit.decl_id), trumps(rules_for(sit.decl_id))
    partner = (sit.viewer + 2) % 4
    hand = world[partner] | sit.played_by(partner)
    rec = empty_record(sit.plays[0][0])
    for seat, tile in sit.plays:
        if seat == partner and rec.trick and rec.leader == sit.bidder and (tr >> rec.trick[0]) & 1:
            out = [t for t in tiles_of(tr & ~rec.played)]
            top = max(out, key=lambda t: rules.rank[7][t])
            if (hand >> top) & 1:
                others = hand & tr & ~bit(top)
                if not singleton and tile != top:
                    return False
                if singleton and others and tile == top:
                    return False
        if seat == partner:
            hand &= ~bit(tile)
        rec = rec.advance(tile, rules)
    return True


@pytest.mark.parametrize("name", NAMES)
def test_the_reader_keeps_exactly_the_worlds_the_rule_allows(name):
    cfg = Config((Rung(512, 4),), "make", seed=9, convention=Convention(name, send=False))
    state = _late(cfg)
    sit = situation_from_state(state, 30)
    ref = Walt(cfg)
    a = ref.analyze(state, 30)
    uniform = Walt(_with(cfg, convention=None)).analyze(state, 30)
    assert a.worlds == uniform.worlds and len(a.worlds) < 512        # the whole fiber
    like = ref.convention_likelihood(sit, a.worlds)
    assert set(like) == {0.0, 1.0}                                   # the reading deletes worlds
    assert a.weights == [x / sum(like) for x in like]
    if name.startswith("handoff"):
        assert like == [float(_handoff_kept(sit, w, name == "handoff-singleton")) for w in a.worlds]
    assert ref.proposals == 0 and ref.fallbacks == 0
    _same_analysis(a, CompiledWalt(cfg).analyze(state, 30))


def test_with_no_gate_in_play_both_halves_are_the_plain_player():
    """Before the partner of the seat to play has moved and away from every
    gate (the first two plays), each arm plays exactly as the plain player:
    the reader's proposals are the uniform stream, every one accepted."""
    cfg = Config((Rung(12, 6),), "make", seed=11)
    for seed in range(4):
        hands, decl, bidder, plays, _ = _game(seed, Convention("handoff"))
        for k in (0, 1):
            state = _at(hands, decl, bidder, plays, k)
            if len(legal_actions(state)) == 1:
                continue
            plain = Walt(cfg).analyze(state, 30)
            for name in NAMES:
                for conv in _arms(name):
                    for walt in (Walt(_with(cfg, convention=conv)), CompiledWalt(_with(cfg, convention=conv))):
                        _same_analysis(plain, walt.analyze(state, 30))
                        assert (walt.sent, walt.fallbacks, walt.proposals) == (0, 0, 12 if conv.read else 0)


def test_the_convention_is_validated_and_named_by_the_spec(tmp_path):
    with pytest.raises(ValueError):
        Convention("signal")
    with pytest.raises(ValueError):
        Convention("handoff", send=False, read=False)
    with pytest.raises(ValueError):   # a reader draws from the uniform only
        Config((Rung(8, None),), field=NET, draw="posterior", convention=Convention("handoff"))
    Config((Rung(8, None),), field=NET, draw="posterior", convention=Convention("handoff", read=False))
    with pytest.raises(ValueError):
        scenario_core.Core([(8, None, 1)], "make", 0, NET.to_bytes(), 0.0, "posterior", 10, ("handoff", False, True))
    with pytest.raises(ValueError):
        scenario_core.Core([(8, None, 1)], "make", 0, None, 0.0, "uniform", 10, ("signal", True, True))
    for arm, (s, r) in {"S": (True, False), "R": (False, True), "SR": (True, True)}.items():
        p = parse_walt_spec(f"k0,n8,c=biggest:{arm}")
        assert p.walt.config.convention == Convention("biggest", s, r) and f"biggest:{arm}" in repr(p)
    assert parse_walt_spec("k0,n8").walt.config.convention is None
    for bad in ("c=handoff", "c=handoff:RS", "c=nope:S"):
        with pytest.raises(ValueError):
            parse_walt_spec(f"k0,n8,{bad}")
