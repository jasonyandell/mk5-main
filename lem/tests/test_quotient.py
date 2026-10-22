# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0
"""The decision quotient on a hand-built course where every count is known."""
import json
from fractions import Fraction
from pathlib import Path

import pytest

from lem.quotient import PREFIX, ROOT_PLY, YOU, analyse, audit, load_root

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
    facts = {(rid, tuple(PREFIX_TOKENS + [YOU, 1, 29, 10, YOU])): (0, ((1, 0, 0, 0, 0, 0, (), ()),)),
             (rid, tuple(PREFIX_TOKENS + [YOU, 1, 29, 11, YOU])): (0, ((1, 0, 0, 5, 0, 0, (), ()),)),
             (rid, tuple(PREFIX_TOKENS + [YOU, 2, 29, 10, YOU])): (1, ((1, 0, 0, 0, 0, 0, (), ()),))}
    return nodes, {rid: nodes}, facts


def route(nodes, per_root, facts, label):
    from lem.quotient import LABELS, measure, output_shape, routing, successors_labelled
    succ = successors_labelled(facts, LABELS[label])
    _, cls = measure(nodes, per_root, label, output_shape, succ)
    return routing(nodes, cls, succ, facts, LABELS[label])


def test_a_label_that_hides_the_difference_fails_to_route(tmp_path):
    nodes, per_root, facts = labelled_course(tmp_path, {5: Fraction(1), 6: Fraction(0)}, {5: Fraction(0), 6: Fraction(1)})
    # The two observations differ only in count; the seat label merges them, and the next best slot differs.
    coarse = route(nodes, per_root, facts, "seat")
    assert not coarse["routes"] and coarse["states_that_fail"] == 1
    assert coarse["colliding_groups_by_kind"] == {"decision: maximizing slots": 1}
    assert coarse["consecutive_colliding_blocks_told_apart_by"] == {"count": 1}
    # The brief label carries count, so every block from every state leads to one state.
    assert route(nodes, per_root, facts, "brief")["routes"]


def test_identical_facts_with_different_futures_are_told_apart_by_nothing(tmp_path):
    nodes, per_root, facts = labelled_course(tmp_path, {5: Fraction(1), 6: Fraction(0)}, {5: Fraction(0), 6: Fraction(1)})
    key = next(k for k in facts if k[1][-2] == 11)
    facts[key] = (0, ((1, 0, 0, 0, 0, 0, (), ()),))      # now both observations look alike in every field
    fine = route(nodes, per_root, facts, "hand_relative")
    assert not fine["routes"] and fine["consecutive_colliding_blocks_told_apart_by"] == {"nothing": 1}


def test_default_report_has_no_label_interfaces(study):
    rep = analyse(study, eligible_only=False)
    assert "label_interfaces" not in rep and set(rep["interfaces"]) == {"raw", "shape", "fork"}
