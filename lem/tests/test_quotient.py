# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0
"""The decision quotient on a hand-built course where every count is known."""
import json
from collections import Counter
from fractions import Fraction
from pathlib import Path

import pytest

from lem.quotient import PREFIX, ROOT_PLY, YOU, analyse, audit, export, load_course, load_root

PREFIX_TOKENS = [0] * PREFIX + [29, 1] * ROOT_PLY


def row(tokens, legal, q, ply):
    best = [a for a in legal if q[a] == max(q.values())]
    return {"key": str(tokens), "tokens": list(tokens), "legal": legal, "optimal": best,
            "q": {str(a): str(v) for a, v in q.items()}, "mass": "1", "ply": ply, "settled": False}


def course(a, b, obs):
    """Root chooses a (wins) or b (loses); one observed play; the last tile is forced."""
    return [row(PREFIX_TOKENS + [YOU], [a, b], {a: Fraction(1), b: Fraction(0)}, 19),
            row(PREFIX_TOKENS + [YOU, a, 30, obs, YOU], [5], {5: Fraction(1)}, 21),
            row(PREFIX_TOKENS + [YOU, b, 30, obs, YOU], [6], {6: Fraction(0)}, 21)]


def write_root(study, name, decisions, value="1"):
    d = study / "roots" / name
    d.mkdir(parents=True)
    (d / "teacher.json").write_text(json.dumps({"root_id": name, "split": name.split("-")[0],
                                                "value": value, "decisions": decisions}))
    return d


@pytest.fixture
def study(tmp_path):
    for i, (a, b, obs) in enumerate([(1, 2, 10), (1, 2, 11), (3, 4, 10)]):
        write_root(tmp_path, f"train-{i:04d}", course(a, b, obs))
    write_root(tmp_path, "test-0000", course(1, 2, 10))
    write_root(tmp_path, "test-0001", course(7, 8, 10))
    return tmp_path


def test_counts_under_each_interface(study):
    rep = analyse(study, eligible_only=False)
    assert rep["roots"] == 5 and rep["nodes"] == 15
    raw, shape, fork = (rep["interfaces"][k] for k in ("raw", "shape", "fork"))
    # raw: four distinct root histories (two observation labels, three menus) and two forced tiles.
    assert raw["states"] == 6
    assert raw["ladder_by_block_depth"] == [5, 6, 6]
    assert raw["test_coverage_by_train_classes"]["root"]["all"] == {"seen": 1, "of": 2}
    # Values are not outputs: both forced leaves are one shape; one root shape.
    assert shape["states"] == 2 and fork["states"] == 2
    assert fork["root_ladder_by_block_depth"] == [1, 1]
    assert rep["decision_classes"] == {2: {"nodes": 5, "distinct_weak_orders": 1,
                                           "weak_orders": {"01": 5}, "tier_types": {"1/1": 5}}}


def test_audits_reject_a_broken_bellman_row(tmp_path):
    decisions = course(1, 2, 10)
    decisions[0]["q"]["1"] = "1/2"           # the root claims less than its children deliver
    d = write_root(tmp_path, "train-0000", decisions, value="1/2")
    with pytest.raises(ValueError, match="Bellman"):
        audit(load_root(d))


def test_audits_reject_a_missing_branch(tmp_path):
    decisions = course(1, 2, 10)[:2]         # action 2 was never explored
    d = write_root(tmp_path, "train-0000", decisions, value="1")
    with pytest.raises(ValueError, match="explored nowhere"):
        audit(load_root(d))


def test_eligibility_filter_reads_complete_json(tmp_path):
    d = write_root(tmp_path, "train-0000", course(1, 2, 10))
    (d / "complete.json").write_text(json.dumps({"eligible": False}))
    e = write_root(tmp_path, "train-0001", course(3, 4, 10))
    (e / "complete.json").write_text(json.dumps({"eligible": True}))
    assert analyse(tmp_path)["roots"] == 1
    assert analyse(tmp_path, eligible_only=False)["roots"] == 2


# ---- tile-free observation labels ----------------------------------------------------------------

def test_play_facts_name_no_tile():
    from scenario.record import empty_record
    from scenario.rules import rules_for
    from lem.quotient import play_facts
    rules, left = rules_for(4), {13, 19, 27}          # fours are trump; the focal seat holds 4-3, 5-4, 6-6
    record = empty_record(1)
    # Seat 1 leads 6-4: trump, a lead, ten count, beats all three of mine, winning; 4-3 and 5-4 follow.
    assert play_facts(rules, record, left, 25) == (1, 1, 2, 10, 1, 1, (0, 1), (0, 1, 2))
    # Seat 2 tops it with 4-4: follows, no count, still beats all of mine, now winning.
    assert play_facts(rules, record.advance(25, rules), left, 14) == (2, 1, 1, 0, 1, 1, (), (0, 1, 2))
    # Seat 3 sloughs 0-0: no trump, fails to follow, beats nothing, not winning.
    after = record.advance(25, rules).advance(14, rules)
    assert play_facts(rules, after, left, 0) == (3, 0, 0, 0, 0, 0, (), ())


def labelled_course(tmp_path, q_after_10, q_after_11):
    """Root plays 1 or 2; after 1 an opponent shows 10 or 11, then a two-way second decision."""
    decisions = [row(PREFIX_TOKENS + [YOU], [1, 2], {1: Fraction(1, 2), 2: Fraction(0)}, 19),
                 row(PREFIX_TOKENS + [YOU, 1, 29, 10, YOU], [5, 6], q_after_10, 21),
                 row(PREFIX_TOKENS + [YOU, 1, 29, 11, YOU], [5, 6], q_after_11, 21),
                 row(PREFIX_TOKENS + [YOU, 2, 29, 10, YOU], [5], {5: Fraction(0)}, 21)]
    nodes = load_root(write_root(tmp_path, "train-0000", decisions, value="1/2"))
    rid = "train-0000"
    facts = {(rid, tuple(PREFIX_TOKENS + [YOU, 1, 29, 10, YOU])): (0, (play(),), BLOCK),
             (rid, tuple(PREFIX_TOKENS + [YOU, 1, 29, 11, YOU])): (0, (play(count=5),), BLOCK),
             (rid, tuple(PREFIX_TOKENS + [YOU, 2, 29, 10, YOU])): (1, (play(),), BLOCK)}
    return nodes, {rid: nodes}, facts


BLOCK = (0, None, None)


def play(count=0, unseen_count=0):
    """The facts of one observed play by seat 1: hand-relative, then posterior."""
    return (1, 0, 0, count, 0, 0, (), ()) + (0, unseen_count, (0, 0, 0), (0, 0, 0), 0, 0)


def route(nodes, per_root, facts, label, *, decisions_only=False):
    from lem.quotient import (JOINT_LABELS, LABELS, POSTERIOR_LABELS, decision_only, measure, output_shape,
                              routing, successors_labelled)
    fields = {**LABELS, **POSTERIOR_LABELS, **JOINT_LABELS}[label]
    succ = successors_labelled(facts, fields)
    _, cls = measure(nodes, per_root, label, output_shape, succ)
    return (decision_only if decisions_only else routing)(nodes, cls, *(() if decisions_only else (succ,)),
                                                          facts, fields)


def test_a_label_that_hides_the_difference_fails_to_route(tmp_path):
    nodes, per_root, facts = labelled_course(tmp_path, {5: Fraction(1), 6: Fraction(0)}, {5: Fraction(0), 6: Fraction(1)})
    # The two observations differ only in count; the seat label merges them, and the next best slot differs.
    coarse = route(nodes, per_root, facts, "seat")
    assert not coarse["routes"] and coarse["states_that_fail"] == 1
    assert coarse["colliding_groups_by_kind"] == {"decision: maximizing slots": 1}
    assert coarse["consecutive_colliding_blocks_told_apart_by"] == {"decision": {"count": 1}}
    # The brief label carries count, so every block from every state leads to one state.
    assert route(nodes, per_root, facts, "brief")["routes"]


def test_identical_facts_with_different_futures_are_told_apart_by_nothing(tmp_path):
    nodes, per_root, facts = labelled_course(tmp_path, {5: Fraction(1), 6: Fraction(0)}, {5: Fraction(0), 6: Fraction(1)})
    key = next(k for k in facts if k[1][-2] == 11)
    facts[key] = (0, (play(),), BLOCK)       # now both observations look alike in every field
    fine = route(nodes, per_root, facts, "hand_relative")
    assert not fine["routes"] and fine["consecutive_colliding_blocks_told_apart_by"] == {"decision": {"nothing": 1}}
    assert not route(nodes, per_root, facts, "everything")["routes"]


def test_a_posterior_fact_outside_the_label_is_named_as_what_tells_blocks_apart(tmp_path):
    nodes, per_root, facts = labelled_course(tmp_path, {5: Fraction(1), 6: Fraction(0)}, {5: Fraction(0), 6: Fraction(1)})
    key = next(k for k in facts if k[1][-2] == 11)
    facts[key] = (0, (play(unseen_count=5),), BLOCK)
    assert route(nodes, per_root, facts, "hand_relative")["consecutive_colliding_blocks_told_apart_by"] == {
        "decision": {"unseen_count": 1}}
    assert route(nodes, per_root, facts, "unseen")["routes"]


def test_posterior_facts_read_only_what_the_focal_seat_has_seen():
    from scenario.record import empty_record
    from scenario.rules import rules_for
    from lem.quotient import posterior_facts
    rules = rules_for(4)                              # fours are trump
    own = {13, 19, 27, 0, 1, 2, 3}                    # 4-3, 5-4, 6-6 and the four lowest tiles
    record = empty_record(1)
    # Seat 1 leads 6-4.  The unseen trumps are the fours less 4-3, 5-4 (mine) and 6-4: 4-0, 4-1, 4-2, 4-4.
    facts = posterior_facts(rules, record, own, 25)
    assert facts[0] == 4
    # The count tiles are 5-5, 6-4, 5-0, 4-1 and 3-2; with 6-4 played and none of them mine, 25 is unseen.
    assert facts[1] == 25
    assert facts[2] == (0, 0, 0)                      # no one has failed to follow yet
    assert facts[4] == 1                              # only 4-4 outranks 6-4 among the unseen tiles
    assert facts[5] == 4                              # four unseen trumps still follow
    # Seat 2 fails to follow with 0-0, a tile the focal seat does not hold this time.
    voids = posterior_facts(rules, record.advance(25, rules), {13, 19, 27, 1, 2, 3, 4}, 0)[2]
    assert voids == (0, 1, 0)                         # seat 2 has shown void in trumps


def test_likelihood_buckets_are_factors_of_two():
    from lem.quotient import likelihood_bucket
    assert [likelihood_bucket(Fraction(n, d)) for n, d in [(1, 1), (3, 4), (1, 2), (1, 3), (1, 4), (1, 5)]] == [
        0, 0, 1, 1, 2, 2]


def test_decision_only_drops_a_domain_collision(tmp_path):
    """After observation 10 the next state accepts one later label, after 11 another; both answer
    alike.  The label routes decision-only but not as an interface that keeps domains."""
    t = PREFIX_TOKENS + [YOU, 1, 29]
    decisions = [row(PREFIX_TOKENS + [YOU], [1, 2], {1: Fraction(1), 2: Fraction(0)}, 19),
                 row(t + [10, YOU], [5, 6], {5: Fraction(1), 6: Fraction(0)}, 21),
                 row(t + [11, YOU], [5, 6], {5: Fraction(1), 6: Fraction(0)}, 21),
                 row(t + [10, YOU, 5, 29, 20, YOU], [7], {7: Fraction(1)}, 23),
                 row(t + [10, YOU, 6, 29, 20, YOU], [8], {8: Fraction(0)}, 23),
                 row(t + [11, YOU, 5, 29, 21, YOU], [7], {7: Fraction(1)}, 23),
                 row(t + [11, YOU, 6, 29, 21, YOU], [8], {8: Fraction(0)}, 23),
                 row(PREFIX_TOKENS + [YOU, 2, 29, 10, YOU], [5], {5: Fraction(0)}, 21)]
    nodes, per_root, facts = {}, {}, {}
    for rid in ("train-0000", "test-0000"):
        per_root[rid] = load_root(write_root(tmp_path, rid, decisions, value="1"))
        nodes.update(per_root[rid])
        facts[(rid, tuple(t + [10, YOU]))] = facts[(rid, tuple(t + [11, YOU]))] = (0, (play(),), BLOCK)
        facts[(rid, tuple(PREFIX_TOKENS + [YOU, 2, 29, 10, YOU]))] = (1, (play(),), BLOCK)
        for obs, count in ((10, 0), (11, 5)):
            for slot, a in ((0, 5), (1, 6)):
                facts[(rid, tuple(t + [obs, YOU, a, 29, 10 + obs, YOU]))] = (slot, (play(count=count),), BLOCK)
    kept = route(nodes, per_root, facts, "brief")
    assert not kept["routes"] and kept["colliding_groups_by_kind"] == {"domain": 2}
    dropped = route(nodes, per_root, facts, "brief", decisions_only=True)
    assert dropped["routes"] and dropped["states_that_fail"] == 0
    assert dropped["fresh_roots_covered_by_a_train_root"] == dropped["of"] == 1


def test_a_decision_collision_survives_decision_only(tmp_path):
    nodes, per_root, facts = labelled_course(tmp_path, {5: Fraction(1), 6: Fraction(0)}, {5: Fraction(0), 6: Fraction(1)})
    dropped = route(nodes, per_root, facts, "seat", decisions_only=True)
    assert not dropped["routes"] and dropped["nodes_that_fail_by_stage"] == {"root": 1}
    # The solved ceiling routes: it names the next decision's answer.
    from lem.quotient import BLOCK_FIELDS, output_shape
    solved = {k: (s, p, (0, output_shape(nodes[k]), None)) for k, (s, p, _) in facts.items()}
    assert BLOCK_FIELDS[1] == "next_output" and route(nodes, per_root, solved, "solved_next_output")["routes"]


# ---- the joint family: the support of the unseen tiles ----------------------------------------------

def test_support_closes_under_hand_sizes():
    from lem.quotient import _closure
    # One tile only seat 1 may hold, one seats 1 or 2, one anyone; each seat holds one.  The first
    # fills seat 1, so the second must go to seat 2 and the third to seat 3: one world.
    classes = (((1,), 1), ((1, 2), 1), ((1, 2, 3), 1))
    assert _closure(classes, (1, 1, 1)) == {(1,): (1,), (1, 2): (2,), (1, 2, 3): (3,)}
    # Seat 1 holding three, nothing binds that the voids do not already say.
    assert _closure(classes, (3, 0, 0)) == {(1,): (1,), (1, 2): (1,), (1, 2, 3): (1,)}
    assert _closure((((1, 2, 3), 3),), (1, 1, 1)) == {(1, 2, 3): (1, 2, 3)}
    with pytest.raises(ValueError, match="no world"):
        _closure((((1,), 2),), (1, 1, 0))


def test_canonical_forgets_which_pip_is_which():
    from lem.quotient import canonical
    a = [(0, 0, "x"), (1, 0, "y"), (3, 2, "y")]
    swapped = [(1, 1, "x"), (1, 0, "y"), (3, 2, "y")]          # pips 0 and 1 exchanged
    assert canonical(a) == canonical(swapped)
    assert canonical(a) != canonical([(0, 0, "x"), (2, 1, "y"), (3, 2, "y")])   # the double now touches nothing
    assert canonical(a) != canonical([(0, 0, "y"), (1, 0, "y"), (3, 2, "y")])   # attributes are kept


def test_joint_facts_lead_with_hand_sizes_and_read_shown_voids():
    from scenario.record import empty_record
    from scenario.rules import rules_for
    from lem.quotient import joint_facts
    rules, own = rules_for(4), {13, 19, 27, 0, 1, 2, 3}       # fours are trump
    record = empty_record(1)
    counts, graph, hand, shape, named = joint_facts(rules, record, own, 25)   # seat 1 leads 6-4
    assert counts[0] == graph[0] == hand[0] == shape[0] == named[0] == (6, 7, 7)
    assert len(named[1]) == 20 and all(s == (1, 2, 3) for _, s in named[1])
    assert sum(n for _, n in counts[1]) == 20 and len(graph[1]) == 20 and len(hand[1]) == len(shape[1]) == 27
    # The control keeps the focal seat's seven tiles as seat 0 and erases every unseen tile's seats.
    assert Counter(e[2][0] for e in shape[1]) == {(): 20, (0,): 7}
    # Seat 2 plays 3-3 under the trump lead and shows void in fours.
    after = record.advance(25, rules)
    named = joint_facts(rules, after, own, 9)[4][1]
    trumps = [t for t, _ in named if (rules.follow_bits[7] >> t) & 1]
    assert trumps and all(dict(named)[t] == (1, 3) for t in trumps)


def test_a_joint_fact_tells_apart_blocks_the_whole_posterior_family_merges(tmp_path):
    nodes, per_root, facts = labelled_course(tmp_path, {5: Fraction(1), 6: Fraction(0)}, {5: Fraction(0), 6: Fraction(1)})
    key = next(k for k in facts if k[1][-2] == 11)
    joint = ("s", "g", "h", "e", "n")
    for k, (slot, plays, block) in facts.items():
        facts[k] = (slot, tuple(p + joint for p in plays), block)
    facts[key] = (0, (play() + ("s", "g2", "h2", "e", "n2"),), BLOCK)
    everything = route(nodes, per_root, facts, "everything")
    assert not everything["routes"]
    assert everything["consecutive_colliding_blocks_told_apart_by"] == {
        "decision": {"support+support_hand+support_named": 1}}
    assert not route(nodes, per_root, facts, "support_counts")["routes"]
    assert route(nodes, per_root, facts, "support")["routes"]
    assert route(nodes, per_root, facts, "joint_everything")["routes"]
    assert not route(nodes, per_root, facts, "shape_everything")["routes"]     # the seats did the work


def test_default_report_has_no_label_interfaces(study):
    rep = analyse(study, eligible_only=False)
    assert "label_interfaces" not in rep and set(rep["interfaces"]) == {"raw", "shape", "fork"}


def test_export_is_the_course_without_hidden_state(study):
    course, _ = load_course(study, eligible_only=False)
    graph = json.loads(json.dumps(export(course)))
    assert graph["schema"] == "silas-decision-course-v1"
    nodes = graph["nodes"]
    assert len(nodes) == 15
    assert all(set(n) == {"group", "split", "stage", "ply", "menu", "best", "q", "mass", "settled", "children"}
               for n in nodes)
    for n in nodes:
        assert len(n["q"]) == len(n["menu"]) and set(n["best"]) <= set(n["menu"])
        for c in n["children"]:
            assert c["play"] in n["menu"] and 0 <= c["next"] < len(nodes)
            assert nodes[c["next"]]["ply"] > n["ply"]
    # Every block of the course appears once.
