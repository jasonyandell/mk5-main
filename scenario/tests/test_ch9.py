# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0

"""The Ch. 9 study's bookkeeping (`scenario.ch9`, #189): which hands the
menu changes, that the seats it registers are symmetric, and that the summary
reads a ledger by deal, with an unchanged deal reading exactly zero. The
summary is exercised on a lowest-tile ledger, so the test plays no planner."""

import json
from collections import Counter
from itertools import combinations

from arena.ledger import parse_seat, run_ledger
from forge.oracle.tables import DOMINO_IS_DOUBLE
from scenario import ch9


def test_the_menu_changes_a_census_of_the_four_double_hands():
    # Every hand of seven with four doubles or more (51,108), by what the menu
    # chose against the heuristic: the share the study's size rests on.
    doubles = [t for t in range(28) if DOMINO_IS_DOUBLE[t]]
    others = [t for t in range(28) if not DOMINO_IS_DOUBLE[t]]
    kinds = Counter(ch9.menu_kind(ds + rest) for k in range(4, 8)
                    for ds in combinations(doubles, k) for rest in combinations(others, 7 - k))
    assert kinds == {"same": 41744, "NT": 5840, "DT": 3524}


def test_every_seat_plays_the_same_menu_walt():
    field, a, b = ch9.seats()
    plays = {id(s.play) for s in (*field, a, b)}
    assert len(plays) == 1, "one MenuWalt at every seat, both arms (#155's symmetry)"
    assert type(a.bid).__name__ == "MenuBidder" and type(b.bid).__name__ == "HeuristicBidder"
    assert all(type(s.bid).__name__ == "HeuristicBidder" for s in field)
    assert all("DTNT=random" in s.name for s in (*field, a, b))


def test_the_summary_reads_a_ledger_by_deal(tmp_path):
    field = tuple(parse_seat("heuristic+lowest") for _ in range(4))
    run = tmp_path / "menu-s20261017"
    run.mkdir()
    ledger = run_ledger(field=field, a=parse_seat("menu+lowest"), b=parse_seat("heuristic+lowest"), n=10,
                        seed=20261017, seats=ch9.CHAIR, keep=ch9.four_doubles)
    (run / "ledger.json").write_text(json.dumps(ledger))
    (run / "meta.json").write_text(json.dumps({"arm": ch9.ARM, "seed": 20261017, "wall_s": 0.0}))
    out = ch9.summarize([str(run)])
    reg, mech = out["registered"], out["mechanism"]
    assert out["deals"] == 10 and out["label"] == ch9.LABEL
    assert all(v["label"] == ch9.LABEL for v in reg.values())
    counts = mech["menu_kind_counts"]
    assert sum(counts.values()) == 10 and counts["DT"] + counts["NT"] > 0
    assert mech["unchanged_nonzero"] == 0
    primary = reg["primary: a - b, marks per hand, four-double deals"]
    changed = reg["changed deals (menu chose DT or NT): a - b, marks per hand"]
    assert primary["deals"] == 10 and changed["deals"] == counts["DT"] + counts["NT"]
    # The unchanged deals read zero, so the primary is the changed mean, diluted.
    assert abs(primary["mean"] * 10 - changed["mean"] * changed["deals"]) < 1e-9
    assert mech["table_differed"] <= changed["deals"]
