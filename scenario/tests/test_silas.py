"""Silas through Python on Zeb's rules: frames from Zeb's decisions, exact
beliefs and summaries in Fractions, float weights read exactly and labeled
by their source, and the three-valued comparison."""
from fractions import Fraction
from pathlib import Path

import pytest
from scenario_core import (
    Core, SilasAuthority, SilasBelief, SilasError, SilasFrame, SilasPolicy, SilasQuery, silas_compare,
)

from forge.zeb.game import is_terminal, legal_actions

from scenario.planner import Situation, situation_from_state
from scenario.record import empty_record, remaining_hand
from scenario.rules import mask_of, rules_for, tiles_of
from scenario.silas import frame
from scenario.tests.test_rules_record import random_playing_states

EXAMPLES = Path(__file__).resolve().parents[2] / "silas" / "examples"
BUDGET = 10_000_000

# Every hidden seat's whole hand: one answer set per world.
HOLDINGS = "(query (roles (chair c) (domino d)) (out c d) (pattern (holds c d)))"


def tile(high: int, low: int) -> int:
    return high * (high + 1) // 2 + low


def name(t: int) -> str:
    high = max(h for h in range(7) if tile(h, 0) <= t)
    return f"{high}-{t - tile(high, 0)}"


def holdings(world) -> frozenset:
    return frozenset((f"S{s}", name(t)) for s in range(4) for t in tiles_of(world[s]))


def eighty_four():
    """The book's four-fives 84 (Winning 42, p. 113): fives trump, the bidder
    in seat 0 holds 5-5 5-4 5-2 5-0 4-4 4-2 2-2, nothing played. (The frame
    reads no bid.)"""
    own = mask_of(tile(h, l) for h, l in [(5, 5), (5, 4), (5, 2), (5, 0), (4, 4), (4, 2), (2, 2)])
    sit = Situation(decl_id=5, viewer=0, bid=42, bidder=0, record=empty_record(0), plays=())
    return frame(sit, own)


def test_the_eighty_four_is_33_over_95_exactly():
    root = eighty_four()
    assert root.physics == "zeb/P5"
    assert root.count_worlds() == 399_072_960
    query = SilasQuery((EXAMPLES / "six-five-guarded.silas").read_text())
    belief = SilasBelief.uniform_resolving(root, mask_of([tile(6, 5), tile(5, 1), tile(5, 3)]), 1000)
    assert len(belief) == 27 and belief.total_mass == 399_072_960
    summary = query.summarize(belief, BUDGET)
    p = summary.event_probability
    assert type(p.value) is Fraction and p.value == Fraction(33, 95)
    assert p.authority.label == "exact" and summary.authority == p.authority
    with pytest.raises(SilasError, match="unresolved"):
        query.summarize(SilasBelief.uniform_resolving(root, mask_of([tile(6, 5)]), 1000), BUDGET)
    with pytest.raises(SilasError, match="above the cap"):
        SilasBelief.uniform(root, 1_000_000)


def _mid_hand(seed: int, plays: int):
    """A real mid-hand decision with a choice: the state and its Situation."""
    for state in random_playing_states(40, seed=seed):
        if not is_terminal(state) and len(state.played) == plays and len(legal_actions(state)) > 1:
            return state, situation_from_state(state, 30)
    raise AssertionError("no such decision")


def test_frames_from_decisions_count_zebs_fiber():
    """A frame built from Zeb's decision holds exactly the worlds Zeb's own
    sampler enumerates there."""
    core = Core([(1024, None, 1)], "make", 0)
    checked = voided = 0
    for state in random_playing_states(12, seed=3):
        if is_terminal(state) or len(state.played) < 13:
            continue
        sit = situation_from_state(state, 30)
        own = remaining_hand(state, sit.viewer)
        f = frame(sit, own)
        rec = sit.record
        req = (sit.decl_id, sit.viewer, own, rec.played, rec.leader, list(rec.trick), rec.banked, rec.forbid,
               sit.bid, sit.bidder, list(sit.plays))
        worlds = core.sample_roots([req])[0]
        if f.count_worlds() <= 1024:  # the whole fiber, enumerated
            assert f.count_worlds() == len(worlds)
            uniform = SilasBelief.uniform(f, 1024)
            law = SilasQuery(HOLDINGS).summarize(uniform, BUDGET).answer_sets
            assert set(law) == {holdings(w) for w in worlds}
            checked += 1
            voided += any(f.voids)
    assert checked > 20 and voided > 10, (checked, voided)


def _sampled():
    """Zeb's sampler's worlds at a mid-hand decision, with float weights."""
    state, sit = _mid_hand(seed=17, plays=9)
    own = remaining_hand(state, sit.viewer)
    rec = sit.record
    seed, n = 5, 64
    core = Core([(n, None, 1)], "make", seed)
    req = (sit.decl_id, sit.viewer, own, rec.played, rec.leader, list(rec.trick), rec.banked, rec.forbid,
           sit.bid, sit.bidder, list(sit.plays))
    worlds = core.sample_roots([req])[0]
    f = frame(sit, own)
    assert len(worlds) == n and f.count_worlds() > n, "sampled, not enumerated"
    weights = [1.0 + ((w[(sit.viewer + 1) % 4] * 2654435761) % 97) / 50.0 for w in worlds]
    return f, worlds, weights, SilasAuthority.sampled(n, seed)


def test_float_weights_enter_exactly_and_say_they_were_sampled():
    f, worlds, weights, sampled = _sampled()
    belief = SilasBelief.from_worlds(f, worlds, weights, sampled)
    assert belief.authority == sampled and belief.authority.label == "sampled(64, 5)"
    summary = SilasQuery(HOLDINGS).summarize(belief, BUDGET)
    assert summary.authority == sampled
    expected: dict[frozenset, Fraction] = {}
    for w, x in zip(worlds, weights):
        expected[holdings(w)] = expected.get(holdings(w), Fraction(0)) + Fraction(x)  # Fraction(float) is exact
    total = sum(expected.values())
    assert belief.total_mass == total
    assert summary.answer_sets == {k: v / total for k, v in expected.items()}
    assert all(type(v) is Fraction for v in summary.answer_sets.values())
    # A presence probability is the mass of the worlds holding the answer.
    some = next(iter(next(iter(expected))))
    held = sum((v for k, v in expected.items() if some in k), Fraction(0))
    assert summary.presence[some] == held / total
    assert summary.event_probability.value == 1 and summary.event_probability.authority == sampled


def test_exact_is_refused_for_float_weights():
    f, worlds, weights, _ = _sampled()
    held = "(query (roles (chair c)) (out) (pattern (holds c 6-5)))"
    exact = SilasQuery(held).summarize(SilasBelief.uniform_resolving(eighty_four(), mask_of([tile(6, 5)]), 10),
                                       BUDGET).authority
    assert exact.label == "exact"
    with pytest.raises(SilasError, match="Exact"):
        SilasBelief.from_worlds(f, worlds, weights, exact)
    with pytest.raises(TypeError):
        SilasBelief.from_worlds(f, worlds, weights, "exact")
    with pytest.raises(SilasError, match="names its model"):
        SilasAuthority.exact_given(" ")
    given = SilasAuthority.exact_given("field:9f1c2e rung=1 seed=5")
    assert SilasBelief.from_worlds(f, worlds, weights, given).authority.label == "exact_given(field:9f1c2e rung=1 seed=5)"
    for bad in (float("nan"), float("inf"), -1.0):
        with pytest.raises(SilasError):
            SilasBelief.from_worlds(f, worlds, [bad] + weights[1:], given)
    swapped = (worlds[0][1], worlds[0][2], worlds[0][3], worlds[0][0])  # every seat holds another's hand
    with pytest.raises(SilasError, match="viewer"):
        SilasBelief.from_worlds(f, [swapped], [1.0], given)


def test_conditioning_on_a_likelihood_gives_its_evidence_exactly():
    """`condition_likelihood`: the evidence is Σ w·L / Σ w exactly, under the
    meet of the belief's authority and the model's (a sample stays a
    sample); every world needs one likelihood, and zero mass is refused."""
    f, worlds, weights, sampled = _sampled()
    belief = SilasBelief.from_worlds(f, worlds, weights, sampled)
    like = [(i % 4) / 4 for i in range(len(worlds))]
    posterior, evidence = belief.condition_likelihood("quarters", "test model v1", worlds, like)
    expected = sum(Fraction(w) * Fraction(x) for w, x in zip(weights, like)) / sum(Fraction(w) for w in weights)
    assert evidence.value == expected and evidence.authority == sampled and posterior.authority == sampled
    assert posterior.total_mass == sum(Fraction(w) * Fraction(x) for w, x in zip(weights, like))
    with pytest.raises(SilasError, match="no likelihood"):
        belief.condition_likelihood("short", "test model v1", worlds[1:], like[1:])
    with pytest.raises(SilasError, match="two different"):
        belief.condition_likelihood("twice", "test model v1", worlds + worlds[:1], like + [0.5 if like[0] != 0.5 else 0.25])
    with pytest.raises(SilasError):
        belief.condition_likelihood("none", "test model v1", worlds, [0.0] * len(worlds))
    with pytest.raises(SilasError):
        belief.condition_likelihood("big", "test model v1", worlds, [1.5] * len(worlds))
    with pytest.raises(SilasError):
        belief.condition_likelihood("anon", " ", worlds, like)


def test_the_three_valued_comparison():
    root = eighty_four()
    belief = SilasBelief.uniform_resolving(root, mask_of([tile(6, 5), tile(5, 1), tile(5, 3)]), 1000)
    guarded = SilasQuery((EXAMPLES / "six-five-guarded.silas").read_text()).summarize(belief, BUDGET)
    held = "(query (roles (chair c)) (out) (pattern (opponent S0 c) (holds c 6-5)))"
    six_five = SilasQuery(held).summarize(belief, BUDGET)
    assert six_five.event_probability.value == Fraction(2, 3)
    a, b = guarded.event_probability, six_five.event_probability
    assert (silas_compare(a, b, Fraction(0)), silas_compare(b, a, Fraction(0))) == ("worse", "better")
    assert silas_compare(a, a, Fraction(0)) == "indistinguishable"
    with pytest.raises(SilasError, match="zero band"):
        silas_compare(a, b, Fraction(1, 2))
    f, worlds, weights, sampled = _sampled()
    s = SilasQuery("(query (roles (chair c)) (out) (pattern (holds c 6-6)))").summarize(
        SilasBelief.from_worlds(f, worlds, weights, sampled), BUDGET).event_probability
    assert s.authority == sampled
    assert silas_compare(s, s, Fraction(1, 10)) == "indistinguishable"
    with pytest.raises(SilasError):
        silas_compare(s, s, Fraction(-1, 10))


def test_text_compiles_or_is_refused_with_silas_message():
    assert "holds" in SilasQuery(HOLDINGS).identity
    with pytest.raises(SilasError, match="unregistered predicate mystery"):
        SilasQuery("(query (roles (chair c)) (out c) (pattern (mystery c)))")
    with pytest.raises(SilasError):
        SilasQuery("(query")
    policy = ("(policy doubles (initial a) (fallback lowest-legal)"
              " (rule lead-double (in a) (next a) (select answer)"
              "  (guard (query (roles (domino d)) (out d) (pattern (own-legal d) (double d))))))")
    for decl in range(10):
        assert SilasPolicy(policy, decl).identity
    with pytest.raises(SilasError, match="world predicate holds"):
        SilasPolicy(policy.replace("(double d)", "(holds S1 d)"), 5)
    with pytest.raises(SilasError, match="no declaration"):
        SilasPolicy(policy, 10)


def test_frames_refuse_incoherent_records():
    own = mask_of(range(7))
    with pytest.raises(SilasError, match="exceeds"):
        SilasFrame(5, 0, own, 0, 43, [])
    with pytest.raises(SilasError, match="out of turn"):
        SilasFrame(5, 0, own, 0, 30, [(1, 27)])
    with pytest.raises(SilasError, match="disagrees"):
        SilasFrame(5, 0, mask_of(range(6)), 0, 30, [])
    with pytest.raises(SilasError, match="illegal"):
        # Blanks trump: seat 0 holds 0-0, 1-0, 2-0, 3-0 and may not slough
        # the 1-1 on seat 3's blank lead.
        SilasFrame(0, 0, own & ~(1 << tile(1, 1)), 3, 30, [(3, tile(6, 0)), (0, tile(1, 1))])
    with pytest.raises(SilasError, match="no seat"):
        SilasFrame(5, 4, own, 0, 30, [])
    with pytest.raises(SilasError, match="no tile"):
        SilasFrame(5, 0, own, 0, 30, [(0, 28)])
    assert SilasFrame(8, 0, own, 0, 30, []).physics == "zeb/DS"


def test_frames_carry_the_contract_and_the_history():
    """The record's facts reach Silas's predicates: who bid, the bid, the
    points still needed to set, the trick number and who led which trick,
    read through a belief that resolves no tile (one exact point)."""
    checked = 0
    for state in random_playing_states(12, seed=11):
        if is_terminal(state) or len(state.played) < 5:
            continue
        sit = situation_from_state(state, 34)
        f = frame(sit, remaining_hand(state, sit.viewer))
        assert (f.bidder, f.bid) == (sit.bidder, 34)
        point = SilasBelief.uniform_resolving(f, 0, 1)

        def answers(text):
            return set(SilasQuery(text).summarize(point, BUDGET).presence)

        assert answers("(query (roles (chair c)) (out c) (pattern (declarer c)))") == {(f"S{sit.bidder}",)}
        assert answers("(query (roles) (out) (pattern (bid 34)))") == {()}
        trick = len(sit.plays) // 4 + 1
        assert answers(f"(query (roles) (out) (pattern (trick-number {trick})))") == {()}
        leads = {(f"S{s}", name(t)) for i, (s, t) in enumerate(sit.plays) if i % 4 == 0}
        assert answers("(query (roles (chair c) (domino d)) (out c d) (pattern (led c d)))") == leads
        setters = sit.record.banked[1 - sit.bidder % 2]
        to_set = max(0, 42 + 1 - 34 - setters)
        assert answers(f"(query (roles) (out) (pattern (to-set {to_set})))") == {()}
        checked += 1
    assert checked > 5, checked
