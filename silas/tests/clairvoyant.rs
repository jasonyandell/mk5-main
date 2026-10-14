// SPDX-FileCopyrightText: Jason Yandell
// SPDX-License-Identifier: CC0-1.0

//! STRATEGY-FUSION FENCE: the tests of the clairvoyant solver itself. The
//! ruling: wiki/topics/strategy-fusion-fence.md.
//!
//! The double-dummy solver against an unpruned exhaustive search, under seat
//! rotation, and on the book's Hand 28 (Roberson, *Winning 42*, pp. 97-100).
mod common;

use common::*;
use silas::reference::Straight42;
use silas::*;

/// A random position with `tricks` tricks still to finish: a random
/// declaration, declarer, leader, bid and banked points, and a random number
/// of plays already on the table.
fn sample(rng: &mut Rng, tricks: usize) -> (Physics, OpenPosition) {
    let rules = Straight42::ALL[rng.below(Straight42::ALL.len())].physics();
    let declarer = Seat::ALL[rng.below(4)];
    let leader = Seat::ALL[rng.below(4)];
    let on_table = rng.below(4);
    let mut tiles = rng.shuffled().into_iter();
    let trick: Vec<Domino> = tiles.by_ref().take(on_table).collect();
    let mut hands = [DominoSet::EMPTY; 4];
    for i in 0..4 {
        let seat = leader.plus(i);
        hands[seat.index()] = tiles.by_ref().take(tricks - usize::from(i < on_table)).collect();
    }
    let banked = [rng.below(20) as u32, rng.below(20) as u32];
    let contract = contract(&rules, declarer, 20 + rng.below(23) as u32);
    (rules, OpenPosition::new(contract, hands, leader, trick, banked).unwrap())
}

/// The final value under unpruned minimax: every legal play of every seat,
/// each side choosing its best, with the rules read from the physics's own
/// methods.
fn exhaustive(position: &OpenPosition, objective: Objective) -> i32 {
    let side = position.contract().declarer().team();
    let Some(seat) = position.next_actor() else {
        let banked = position.banked();
        let ours = banked[side.index()] as i32;
        return match objective {
            Objective::Margin => ours - banked[1 - side.index()] as i32,
            Objective::Make => i32::from(ours >= position.contract().threshold() as i32),
        };
    };
    let values = position.legal().iter().map(|d| exhaustive(&position.play(d).unwrap(), objective));
    if seat.team() == side { values.max().unwrap() } else { values.min().unwrap() }
}

fn exhaustive_root(position: &OpenPosition, objective: Objective) -> Vec<(Domino, i32)> {
    let legal = position.legal();
    legal.iter().map(|d| (d, exhaustive(&position.play(d).unwrap(), objective))).collect()
}

#[test]
fn every_root_value_agrees_with_exhaustive_search_on_sampled_endings() {
    let mut rng = Rng(42);
    for tricks in [1, 2, 3] {
        for _ in 0..1500 {
            let (rules, position) = sample(&mut rng, tricks);
            let mut solver = ClairvoyantSolver::new(&rules);
            for objective in [Objective::Margin, Objective::Make] {
                let values = solver.solve(&position, objective).unwrap();
                assert_eq!(values.moves, exhaustive_root(&position, objective), "{rules:?} {position:?}");
                assert_eq!(solver.value(&position, objective), values.best());
            }
        }
    }
}

#[test]
fn every_root_value_agrees_with_exhaustive_search_with_four_tricks_left() {
    let mut rng = Rng(7);
    for _ in 0..60 {
        let (rules, position) = sample(&mut rng, 4);
        let mut solver = ClairvoyantSolver::new(&rules);
        for objective in [Objective::Margin, Objective::Make] {
            assert_eq!(solver.solve(&position, objective).unwrap().moves, exhaustive_root(&position, objective));
        }
    }
}

/// One table serves both objectives, whichever is asked first: make-or-set
/// is the margin read against the bid.
#[test]
fn make_or_set_is_the_margin_read_against_the_bid() {
    let mut rng = Rng(99);
    for _ in 0..200 {
        let (rules, position) = sample(&mut rng, 5);
        let side = position.contract().declarer().team().index();
        let total = (position.banked()[0] + position.banked()[1] + position.unbanked()) as i32;
        let mut solver = ClairvoyantSolver::new(&rules);
        let make = solver.solve(&position, Objective::Make).unwrap();
        let margin = solver.solve(&position, Objective::Margin).unwrap();
        for ((d, m), (e, v)) in make.moves.iter().zip(&margin.moves) {
            assert_eq!(d, e);
            let ours = (total + v) / 2;
            assert_eq!(*m, i32::from(ours >= position.contract().threshold() as i32), "{position:?} side {side}");
        }
    }
}

/// The same position with every seat moved `n` chairs clockwise: the teams
/// trade indices when `n` is odd.
fn rotated(position: &OpenPosition, n: usize) -> OpenPosition {
    let mut hands = [DominoSet::EMPTY; 4];
    for seat in Seat::ALL {
        hands[seat.plus(n).index()] = position.hand(seat);
    }
    let old = position.contract();
    let contract = Contract::new(old.declarer().plus(n), old.bid(), old.declaration().clone()).unwrap();
    let banked = if n % 2 == 1 { [position.banked()[1], position.banked()[0]] } else { position.banked() };
    OpenPosition::new(contract, hands, position.leader().plus(n), position.trick().to_vec(), banked).unwrap()
}

#[test]
fn rotating_the_seats_leaves_every_value_unchanged() {
    let mut rng = Rng(2026);
    let deal = hand_28();
    let rules = physics("P0");
    let opening = OpenPosition::deal(deal.contract.clone(), deal.hands).unwrap();
    let mut positions = vec![(rules, opening)];
    for tricks in [3, 5, 7] {
        positions.extend((0..40).map(|_| sample(&mut rng, tricks)));
    }
    for (rules, position) in positions {
        let mut solver = ClairvoyantSolver::new(&rules);
        let values = [Objective::Margin, Objective::Make].map(|o| solver.solve(&position, o).unwrap().moves);
        for n in 1..4 {
            let turned = rotated(&position, n);
            let mut fresh = ClairvoyantSolver::new(&rules);
            for (o, expected) in [Objective::Margin, Objective::Make].into_iter().zip(&values) {
                assert_eq!(&fresh.solve(&turned, o).unwrap().moves, expected);
            }
        }
    }
}

// Hand 28 (pp. 97-100): blanks, bid 30. Hands by seat, the bidder in S0 and
// play clockwise: bidder, left opponent, partner, right opponent.
const HAND_28: [[&str; 7]; 4] = [
    ["6-0", "0-0", "3-1", "5-3", "5-0", "6-6", "6-1"],
    ["4-0", "5-1", "3-2", "5-5", "4-1", "3-3", "4-4"],
    ["1-0", "2-0", "4-3", "6-5", "6-4", "6-2", "2-2"],
    ["3-0", "2-1", "6-3", "5-4", "4-2", "1-1", "5-2"],
];

// The narrated line, as reconstructed trick by trick in the archive's
// discrepancy review (§ Hand 28): the bidder and partner take 25, the
// opponents 17, from tricks three and four.
const HAND_28_LINE: [&str; 28] = [
    "6-0", "4-0", "1-0", "3-0", // T1, bidder leads and wins
    "0-0", "5-1", "2-0", "2-1", // T2, bidder
    "3-1", "3-2", "4-3", "6-3", // T3, right opponent wins 6
    "5-4", "5-3", "5-5", "6-5", // T4, left opponent wins 11
    "4-1", "6-4", "4-2", "5-0", // T5, the bidder trumps in for 21
    "6-6", "3-3", "6-2", "1-1", // T6, bidder
    "6-1", "4-4", "2-2", "5-2", // T7, bidder
];

fn hand_28() -> Deal {
    Deal { contract: contract(&physics("P0"), Seat::S0, 30), hands: HAND_28.map(|h| tiles(&h)) }
}

#[test]
fn hand_28_replayed_through_the_solver_comes_to_25_and_17() {
    let deal = hand_28();
    let physics = deal.contract.declaration();
    let line: Vec<Domino> = HAND_28_LINE.iter().map(|d| tile(d)).collect();
    let mut solver = ClairvoyantSolver::new(physics);
    let mut position = OpenPosition::deal(deal.contract.clone(), deal.hands).unwrap();
    for d in &line[..27] {
        position = position.play(*d).unwrap();
    }
    // The last play is forced, and its value is the line's: 25 to 17.
    let last = solver.solve(&position, Objective::Margin).unwrap();
    assert_eq!(last.moves, vec![(tile("5-2"), 25 - 17)]);
    assert_eq!(solver.solve(&position, Objective::Make).unwrap().moves, vec![(tile("5-2"), 0)]);
    let end = position.play(line[27]).unwrap();
    assert_eq!((end.banked(), end.next_actor()), ([25, 17], None));
    assert_eq!(solver.value(&end, Objective::Margin), 8);

    // Every play of the line is graded at the position it was made from, and
    // each grade agrees with the root values it was read from.
    let grades = clairvoyant_grade_line(&deal, &line).unwrap();
    assert_eq!(grades.len(), 28);
    let mut position = OpenPosition::deal(deal.contract.clone(), deal.hands).unwrap();
    for (g, d) in grades.iter().zip(&line) {
        assert_eq!((g.tile, Some(g.seat), g.legal), (*d, position.next_actor(), position.legal()));
        let values = ClairvoyantSolver::new(physics).solve(&position, Objective::Margin).unwrap();
        assert_eq!(
            (g.margin.given, g.margin.best, g.margin.tied),
            (values.value(*d).unwrap(), values.best(), values.tied_best())
        );
        position = position.play(*d).unwrap();
    }

    // Two of the narrated plays fall short double-dummy, and one of them
    // costs the contract: the bidder's 3-1 lead to trick three (his last
    // trump, the 5-0, or the 6-1 or 6-6 there keeps all 42) and the right
    // opponent's 5-4 lead to trick four (the 1-1 holds the bidders to 19).
    let short: Vec<(usize, Seat, i32, i32, i32, i32)> = grades
        .iter()
        .filter(|g| !g.margin.optimal() || !g.make.optimal())
        .map(|g| (g.ply, g.seat, g.margin.given, g.margin.best, g.make.given, g.make.best))
        .collect();
    assert_eq!(short, vec![(8, Seat::S0, -4, 42, 0, 1), (12, Seat::S3, 8, -4, 0, 0)]);
    assert_eq!((grades[8].tile, grades[8].margin.tied), (tile("3-1"), tiles(&["5-0", "6-1", "6-6"])));
    assert_eq!((grades[12].tile, grades[12].margin.tied), (tile("5-4"), tiles(&["1-1"])));
}

/// The declaring side's points still to come, by unpruned minimax memoized
/// only on exact trick-start states: no alpha-beta, no bounds, no move
/// ordering, and the rules read through the physics's own methods.
fn memoized(position: &OpenPosition, memo: &mut std::collections::HashMap<([DominoSet; 4], Seat), u32>) -> u32 {
    let Some(seat) = position.next_actor() else { return 0 };
    let side = position.contract().declarer().team();
    let key = (position.hands(), position.leader());
    if let Some(v) = position.trick().is_empty().then(|| memo.get(&key)).flatten() {
        return *v;
    }
    let before = position.banked()[side.index()];
    let values = position.legal().iter().map(|d| {
        let next = position.play(d).unwrap();
        next.banked()[side.index()] - before + memoized(&next, memo)
    });
    let v = if seat.team() == side { values.max().unwrap() } else { values.min().unwrap() };
    if position.trick().is_empty() {
        memo.insert(key, v);
    }
    v
}

/// The whole deal from the opening lead, every move's value checked against
/// an independent search of the full game tree.
#[test]
fn hand_28_from_the_opening_lead_agrees_with_the_whole_tree() {
    let deal = hand_28();
    let physics = deal.contract.declaration();
    let opening = OpenPosition::deal(deal.contract.clone(), deal.hands).unwrap();
    let mut memo = std::collections::HashMap::new();
    let tree: Vec<(Domino, i32)> = opening
        .legal()
        .iter()
        .map(|d| {
            let next = opening.play(d).unwrap();
            (d, 2 * (next.banked()[0] + memoized(&next, &mut memo)) as i32 - 42)
        })
        .collect();
    let expected = [("0-0", 42), ("3-1", 8), ("5-0", 42), ("5-3", 8), ("6-0", 42), ("6-1", 26), ("6-6", 26)];
    assert_eq!(tree, expected.map(|(d, v)| (tile(d), v)));
    let mut solver = ClairvoyantSolver::new(physics);
    assert_eq!(solver.solve(&opening, Objective::Margin).unwrap().moves, tree);
    let make: Vec<(Domino, i32)> = tree.iter().map(|(d, m)| (*d, i32::from((42 + m) / 2 >= 30))).collect();
    assert_eq!(solver.solve(&opening, Objective::Make).unwrap().moves, make);
}
