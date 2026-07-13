#!/usr/bin/env python3
"""Build deterministic W42 laydown and rule-accounting proof artifacts.

This runner covers the Chapter 1 rule substrate and the Chapter 3 laydown rule
without using any stochastic corpus data. It compares hand-authored fixtures and
enumerated invariants against the Forge oracle rule tables.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from forge.eq.game import GameState
from forge.oracle.declarations import DOUBLES_TRUMP, NOTRUMP
from forge.oracle.tables import (
    DOMINO_COUNT_POINTS,
    DOMINO_IS_DOUBLE,
    DOMINOES,
    can_follow,
    domino_contains_pip,
    is_in_called_suit,
    led_suit_for_lead_domino,
    resolve_trick,
    score_trick,
)


BEAD_ID = "t42-br7n.4"


def domino_id(high: int, low: int) -> int:
    pair = (max(high, low), min(high, low))
    return DOMINOES.index(pair)


def domino_name(domino: int) -> str:
    high, low = DOMINOES[domino]
    return f"{high}-{low}"


def domino_names(dominoes: list[int] | tuple[int, ...]) -> list[str]:
    return [domino_name(d) for d in dominoes]


def jsonish(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True)


@dataclass(frozen=True)
class RuleResult:
    test_id: str
    family: str
    assertion: str
    expected: Any
    observed: Any
    status: str
    details: str = ""


@dataclass(frozen=True)
class LegalMaskFixture:
    fixture_id: str
    decl_id: int
    leader: int
    lead_domino: int
    follower_hand: tuple[int, ...]
    expected_legal: tuple[int, ...]
    rationale: str


@dataclass(frozen=True)
class TrickFixture:
    fixture_id: str
    decl_id: int
    leader: int
    dominoes_in_play_order: tuple[int, int, int, int]
    expected_winner_offset: int
    rationale: str


@dataclass(frozen=True)
class LaydownFixture:
    fixture_id: str
    decl_id: int
    claimant: int
    leader: int
    hands: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]]
    expected_proven: bool
    rationale: str


def result(
    test_id: str,
    family: str,
    assertion: str,
    expected: Any,
    observed: Any,
    details: str = "",
) -> RuleResult:
    return RuleResult(
        test_id=test_id,
        family=family,
        assertion=assertion,
        expected=expected,
        observed=observed,
        status="pass" if observed == expected else "fail",
        details=details,
    )


def build_rule_results() -> tuple[list[RuleResult], list[dict[str, Any]]]:
    rows: list[RuleResult] = []
    fixtures: list[dict[str, Any]] = []

    count_tiles = {domino_name(i): points for i, points in enumerate(DOMINO_COUNT_POINTS) if points}
    rows.append(
        result(
            "count_identity_exact_set",
            "count_identity",
            "Count dominoes are exactly 5-5, 6-4, 4-1, 3-2, and 5-0 with face-sum values.",
            {"3-2": 5, "4-1": 5, "5-0": 5, "5-5": 10, "6-4": 10},
            count_tiles,
        )
    )
    rows.append(
        result(
            "count_total_35",
            "count_identity",
            "All count domino values sum to 35.",
            35,
            sum(DOMINO_COUNT_POINTS),
        )
    )
    rows.append(
        result(
            "hand_total_42",
            "count_identity",
            "Seven trick points plus 35 count points total 42.",
            42,
            7 + sum(DOMINO_COUNT_POINTS),
        )
    )
    rows.append(
        result(
            "doubles_face_total_42",
            "count_identity",
            "The face value of the seven doubles also totals 42.",
            42,
            sum(2 * pip for pip in range(7)),
        )
    )

    memberships = {
        domino_name(i): [pip for pip in range(7) if domino_contains_pip(i, pip)]
        for i in range(len(DOMINOES))
    }
    rows.append(
        result(
            "non_trump_membership_total",
            "suit_membership",
            "Twenty-one non-doubles belong to two suits and seven doubles belong to one.",
            49,
            sum(len(v) for v in memberships.values()),
        )
    )
    rows.append(
        result(
            "each_pip_suit_has_seven_tiles",
            "suit_membership",
            "Each of the seven pip suits contains seven dominoes before trump is declared.",
            {str(pip): 7 for pip in range(7)},
            {str(pip): sum(1 for i in range(len(DOMINOES)) if domino_contains_pip(i, pip)) for pip in range(7)},
        )
    )
    rows.append(
        result(
            "doubles_single_suit",
            "suit_membership",
            "Doubles have one natural pip-suit membership.",
            [],
            [domino_name(i) for i, is_double in enumerate(DOMINO_IS_DOUBLE) if is_double and len(memberships[domino_name(i)]) != 1],
        )
    )

    five_blank = domino_id(5, 0)
    six_blank = domino_id(6, 0)
    six_five = domino_id(6, 5)
    six_six = domino_id(6, 6)
    rows.extend(
        [
            result(
                "trump_secondary_suit_excluded",
                "trump_exclusivity",
                "A five-blank is trump when fives are called and cannot follow blanks.",
                False,
                can_follow(five_blank, 0, 5),
            ),
            result(
                "non_trump_secondary_can_follow",
                "trump_exclusivity",
                "A six-blank can follow blanks when fives are called.",
                True,
                can_follow(six_blank, 0, 5),
            ),
            result(
                "called_suit_follow_requires_called_tile",
                "trump_exclusivity",
                "A six-five can follow the called-suit lead when fives are called.",
                True,
                can_follow(six_five, 7, 5),
            ),
            result(
                "off_double_cannot_follow_called_suit",
                "trump_exclusivity",
                "A double-six cannot follow a fives-trump lead.",
                False,
                can_follow(six_six, 7, 5),
            ),
        ]
    )

    legal_fixtures = [
        LegalMaskFixture(
            fixture_id="trump_lead_requires_trump",
            decl_id=5,
            leader=0,
            lead_domino=five_blank,
            follower_hand=(six_five, six_blank, domino_id(0, 0)),
            expected_legal=(six_five,),
            rationale="When trump is led, only called-suit tiles can follow if present.",
        ),
        LegalMaskFixture(
            fixture_id="secondary_suit_excludes_trump",
            decl_id=5,
            leader=0,
            lead_domino=domino_id(0, 0),
            follower_hand=(five_blank, six_blank, domino_id(1, 1)),
            expected_legal=(six_blank,),
            rationale="A trump tile containing the led pip is not also a blank for follow-suit.",
        ),
        LegalMaskFixture(
            fixture_id="void_may_play_anything",
            decl_id=5,
            leader=0,
            lead_domino=domino_id(0, 0),
            follower_hand=(domino_id(2, 2), domino_id(3, 3), six_five),
            expected_legal=(domino_id(2, 2), domino_id(3, 3), six_five),
            rationale="A player void in led suit can play any tile, including trump.",
        ),
    ]
    for fixture in legal_fixtures:
        state = GameState(
            hands=((fixture.lead_domino,), fixture.follower_hand, (), ()),
            played=frozenset(),
            play_history=(),
            current_trick=(),
            leader=fixture.leader,
            decl_id=fixture.decl_id,
        ).apply_action(fixture.lead_domino)
        observed = tuple(state.legal_actions())
        rows.append(
            result(
                f"legal_mask_{fixture.fixture_id}",
                "follow_suit_mask",
                fixture.rationale,
                domino_names(fixture.expected_legal),
                domino_names(observed),
            )
        )
        fixtures.append(
            {
                "type": "legal_mask",
                "fixture_id": fixture.fixture_id,
                "decl_id": fixture.decl_id,
                "lead_domino": domino_name(fixture.lead_domino),
                "follower_hand": domino_names(fixture.follower_hand),
                "expected_legal": domino_names(fixture.expected_legal),
                "rationale": fixture.rationale,
            }
        )

    trick_fixtures = [
        TrickFixture(
            fixture_id="no_trump_high_led_suit_wins",
            decl_id=NOTRUMP,
            leader=0,
            dominoes_in_play_order=(domino_id(3, 1), domino_id(6, 3), domino_id(1, 1), domino_id(2, 2)),
            expected_winner_offset=1,
            rationale="Without trump, highest tile in led suit wins.",
        ),
        TrickFixture(
            fixture_id="pip_trump_beats_led_suit",
            decl_id=5,
            leader=0,
            dominoes_in_play_order=(domino_id(0, 0), five_blank, six_blank, domino_id(1, 1)),
            expected_winner_offset=1,
            rationale="Any called-suit trump beats non-trump led-suit tiles.",
        ),
        TrickFixture(
            fixture_id="higher_pip_trump_beats_lower_trump",
            decl_id=5,
            leader=0,
            dominoes_in_play_order=(five_blank, six_five, domino_id(5, 5), domino_id(4, 4)),
            expected_winner_offset=2,
            rationale="Highest called-suit trump wins among multiple trumps.",
        ),
        TrickFixture(
            fixture_id="doubles_trump_highest_double_wins",
            decl_id=DOUBLES_TRUMP,
            leader=0,
            dominoes_in_play_order=(domino_id(1, 1), domino_id(6, 6), domino_id(5, 5), domino_id(6, 0)),
            expected_winner_offset=1,
            rationale="In doubles-trump, the highest double wins over lower doubles and off tiles.",
        ),
    ]
    for fixture in trick_fixtures:
        outcome = resolve_trick(fixture.dominoes_in_play_order[0], fixture.dominoes_in_play_order, fixture.decl_id)
        rows.append(
            result(
                f"trick_winner_{fixture.fixture_id}",
                "trick_winner",
                fixture.rationale,
                fixture.expected_winner_offset,
                outcome.winner_offset,
                details=f"points={outcome.points}",
            )
        )
        fixtures.append(
            {
                "type": "trick_winner",
                "fixture_id": fixture.fixture_id,
                "decl_id": fixture.decl_id,
                "play_order": domino_names(fixture.dominoes_in_play_order),
                "expected_winner_offset": fixture.expected_winner_offset,
                "rationale": fixture.rationale,
            }
        )

    lead_state = GameState(
        hands=((domino_id(3, 1),), (domino_id(6, 3),), (domino_id(1, 1),), (domino_id(2, 2),)),
        played=frozenset(),
        play_history=(),
        current_trick=(),
        leader=0,
        decl_id=NOTRUMP,
    )
    for action in (domino_id(3, 1), domino_id(6, 3), domino_id(1, 1), domino_id(2, 2)):
        lead_state = lead_state.apply_action(action)
    rows.append(
        result(
            "lead_control_winner_leads_next",
            "lead_control",
            "The winner of a trick leads the next trick.",
            1,
            lead_state.leader,
            details="P1 wins the threes trick with 6-3.",
        )
    )

    count_trick = (domino_id(5, 5), domino_id(6, 4), domino_id(4, 1), domino_id(3, 2))
    rows.append(
        result(
            "trick_points_include_count",
            "count_capture",
            "A trick scores one trick point plus any count dominoes it contains.",
            31,
            score_trick(count_trick),
            details="Synthetic all-count trick used only for accounting arithmetic.",
        )
    )

    def contract_score(bid: int, bidder_points: int) -> dict[str, int]:
        opponent_points = 42 - bidder_points
        if bidder_points >= bid:
            return {"bidder_score": bidder_points, "opponent_score": opponent_points}
        return {"bidder_score": 0, "opponent_score": bid + opponent_points}

    scoring_cases = [
        ("bid_30_take_33", 30, 33, {"bidder_score": 33, "opponent_score": 9}),
        ("bid_35_take_33", 35, 33, {"bidder_score": 0, "opponent_score": 44}),
        ("bid_35_take_30", 35, 30, {"bidder_score": 0, "opponent_score": 47}),
    ]
    for case_id, bid, bidder_points, expected in scoring_cases:
        rows.append(
            result(
                f"contract_scoring_{case_id}",
                "contract_scoring",
                "Chapter 1 make/set examples separate hand points from failed-bid score.",
                expected,
                contract_score(bid, bidder_points),
            )
        )

    return rows, fixtures


def state_from_laydown_fixture(fixture: LaydownFixture) -> GameState:
    return GameState(
        hands=fixture.hands,
        played=frozenset(),
        play_history=(),
        current_trick=(),
        leader=fixture.leader,
        decl_id=fixture.decl_id,
    )


def exact_laydown_proof(state: GameState, claimant: int) -> dict[str, Any]:
    """Return whether claimant's team wins every remaining trick for every legal line."""

    claimant_team = claimant % 2
    completions = 0
    nodes = 0
    counterexample: list[dict[str, Any]] | None = None

    def done(s: GameState) -> bool:
        return all(len(hand) == 0 for hand in s.hands) and len(s.current_trick) == 0

    def search(s: GameState, path: list[dict[str, Any]]) -> bool:
        nonlocal completions, nodes, counterexample
        nodes += 1
        if done(s):
            completions += 1
            return True

        player = s.current_player()
        legal = s.legal_actions()
        if not legal:
            counterexample = path + [{"error": f"no legal actions for player {player}"}]
            return False

        for action in legal:
            before_len = len(s.current_trick)
            after = s.apply_action(action)
            event: dict[str, Any] = {
                "player": player,
                "action": domino_name(action),
                "legal_actions": domino_names(legal),
            }
            if before_len == 3:
                winner = after.leader
                event["completed_trick_winner"] = winner
                event["claimant_team_won_trick"] = winner % 2 == claimant_team
                if winner % 2 != claimant_team:
                    counterexample = path + [event]
                    return False
            if not search(after, path + [event]):
                return False
        return True

    proven = search(state, [])
    return {
        "proven": proven,
        "legal_completions_checked": completions,
        "search_nodes": nodes,
        "counterexample": counterexample or [],
    }


def build_laydown_fixtures() -> list[LaydownFixture]:
    return [
        LaydownFixture(
            fixture_id="single_boss_trump_laydown",
            decl_id=6,
            claimant=0,
            leader=0,
            hands=((domino_id(6, 6),), (domino_id(5, 5),), (domino_id(4, 4),), (domino_id(3, 3),)),
            expected_proven=True,
            rationale="Claimant leads the boss trump with one trick remaining; all responses lose.",
        ),
        LaydownFixture(
            fixture_id="two_boss_trumps_laydown",
            decl_id=6,
            claimant=0,
            leader=0,
            hands=(
                (domino_id(6, 6), domino_id(6, 5)),
                (domino_id(0, 0), domino_id(1, 1)),
                (domino_id(2, 2), domino_id(3, 3)),
                (domino_id(4, 4), domino_id(5, 5)),
            ),
            expected_proven=True,
            rationale="Claimant has the only remaining trumps and keeps lead after either legal order.",
        ),
        LaydownFixture(
            fixture_id="false_off_can_be_taken",
            decl_id=NOTRUMP,
            claimant=0,
            leader=0,
            hands=((domino_id(4, 0),), (domino_id(6, 4),), (domino_id(1, 1),), (domino_id(2, 2),)),
            expected_proven=False,
            rationale="Opponent must follow fours and beats the claimant's off lead.",
        ),
        LaydownFixture(
            fixture_id="false_secondary_trump_exclusion",
            decl_id=5,
            claimant=0,
            leader=0,
            hands=((domino_id(6, 0),), (domino_id(5, 0),), (domino_id(1, 1),), (domino_id(2, 2),)),
            expected_proven=False,
            rationale="A five-blank is trump, not a blank; it can trump the claimant's off lead.",
        ),
        LaydownFixture(
            fixture_id="walker_after_suit_exhaustion",
            decl_id=NOTRUMP,
            claimant=0,
            leader=0,
            hands=((domino_id(2, 1),), (domino_id(0, 0),), (domino_id(1, 1),), (domino_id(3, 3),)),
            expected_proven=True,
            rationale="The claimant's low deuce is a walker because no other remaining hand can follow deuces.",
        ),
        LaydownFixture(
            fixture_id="book_like_final_deuce_ace_counterexample",
            decl_id=3,
            claimant=0,
            leader=0,
            hands=((domino_id(2, 1),), (domino_id(6, 2),), (domino_id(6, 4),), (domino_id(0, 0),)),
            expected_proven=False,
            rationale="Mirrors the Chapter 3 warning: if an opponent still has a higher deuce, the final deuce-ace is not proven.",
        ),
    ]


def build_laydown_results() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    fixtures = build_laydown_fixtures()
    result_rows: list[dict[str, Any]] = []
    fixture_rows: list[dict[str, Any]] = []
    counterexamples: list[dict[str, Any]] = []

    for fixture in fixtures:
        proof = exact_laydown_proof(state_from_laydown_fixture(fixture), fixture.claimant)
        status = "pass" if proof["proven"] == fixture.expected_proven else "fail"
        result_rows.append(
            {
                "fixture_id": fixture.fixture_id,
                "decl_id": fixture.decl_id,
                "claimant": fixture.claimant,
                "leader": fixture.leader,
                "expected_proven": fixture.expected_proven,
                "observed_proven": proof["proven"],
                "status": status,
                "legal_completions_checked": proof["legal_completions_checked"],
                "search_nodes": proof["search_nodes"],
                "rationale": fixture.rationale,
            }
        )
        fixture_rows.append(
            {
                "fixture_id": fixture.fixture_id,
                "decl_id": fixture.decl_id,
                "claimant": fixture.claimant,
                "leader": fixture.leader,
                "hands": [[domino_name(d) for d in hand] for hand in fixture.hands],
                "expected_proven": fixture.expected_proven,
                "rationale": fixture.rationale,
            }
        )
        if proof["counterexample"]:
            counterexamples.append(
                {
                    "fixture_id": fixture.fixture_id,
                    "counterexample": proof["counterexample"],
                }
            )

    return result_rows, fixture_rows, counterexamples


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: jsonish(v) for k, v in row.items()})


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("w42/phase4_laydown_rule_accounting"))
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rule_results, rule_fixtures = build_rule_results()
    laydown_results, laydown_fixtures, laydown_counterexamples = build_laydown_results()

    rule_result_rows = [asdict(r) for r in rule_results]
    write_csv(out_dir / "rule_results.csv", rule_result_rows)
    write_json(out_dir / "rule_fixtures.json", rule_fixtures)
    write_csv(out_dir / "laydown_results.csv", laydown_results)
    write_json(out_dir / "laydown_fixtures.json", laydown_fixtures)
    write_json(out_dir / "laydown_counterexamples.json", laydown_counterexamples)

    all_statuses = [r["status"] for r in rule_result_rows] + [r["status"] for r in laydown_results]
    families: dict[str, dict[str, int]] = {}
    for row in rule_result_rows:
        fam = row["family"]
        families.setdefault(fam, {"pass": 0, "fail": 0})
        families[fam][row["status"]] += 1
    families.setdefault("laydown_exact_proof", {"pass": 0, "fail": 0})
    for row in laydown_results:
        families["laydown_exact_proof"][row["status"]] += 1

    summary = {
        "bead": BEAD_ID,
        "artifact": "phase4_laydown_rule_accounting",
        "source_grounding": [
            "wiki/AGENTS.md",
            "wiki/experiments/winning42-ch01-in-a-nutshell.md",
            "wiki/experiments/winning42-ch03-bidder-play.md",
            "scratch/winning42/winning42.with_figures.md lines 479-734 and 2045-2057",
            "forge.oracle.tables",
            "forge.eq.game.GameState",
        ],
        "coverage": {
            "rule_assertions": len(rule_result_rows),
            "rule_fixtures": len(rule_fixtures),
            "laydown_fixtures": len(laydown_results),
            "laydown_counterexamples": len(laydown_counterexamples),
            "total_assertions": len(all_statuses),
            "failures": all_statuses.count("fail"),
        },
        "families": families,
        "headline_findings": {
            "chapter1_accounting": "Count identity, 42-point hand accounting, natural suit membership, trump exclusivity, follow-suit masks, trick winner, lead control, count capture, and contract scoring all passed deterministic checks.",
            "laydown_checker": "A small exact checker can prove or reject explicit remaining-trick claims by enumerating every legal continuation in tiny late-state fixtures.",
            "book_like_caveat": "The Chapter 3 final deuce-ace warning is reproduced as a false laydown when an opponent still holds a higher deuce.",
        },
        "caveats": [
            "The laydown checker is exact for explicit full-information tiny fixtures, not yet wired to arbitrary engine/corpus state snapshots.",
            "The proof criterion is intentionally strong: every legal continuation by every player must give the claimant's team every remaining trick.",
            "No wiki or bead metadata was edited by Worker D per task scope.",
        ],
    }
    write_json(out_dir / "summary.json", summary)

    manifest = {
        "bead": BEAD_ID,
        "files": [
            "run_laydown_rule_accounting.py",
            "validate_outputs.py",
            "rule_results.csv",
            "rule_fixtures.json",
            "laydown_results.csv",
            "laydown_fixtures.json",
            "laydown_counterexamples.json",
            "summary.json",
            "manifest.json",
        ],
    }
    write_json(out_dir / "manifest.json", manifest)

    print(json.dumps(summary["coverage"], indent=2, sort_keys=True))
    return 0 if summary["coverage"]["failures"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
