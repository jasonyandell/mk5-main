//! The book's frame facts as queries (Roberson, *Winning 42*): what an off
//! can lose and which side a double protects (`off-risk.silas`,
//! `off-exposed.silas`), what a defender of an 84 watches for
//! (`watch-list.silas`, `pair-targets.silas`), and when a hand is over
//! (`settled.silas`). Each reads only the viewer's frame, so its answers are
//! one set per position; each is asked at the book's own positions
//! (`examples/book/hands.json`) and must give the book's list, at all four
//! seatings. "Ours" marks a tile the book does not name.
mod common;

use std::collections::BTreeSet;

use common::fixture::{Position, Role, book};
use common::rules::*;
use common::*;
use silas::*;

const OFF_RISK: &str = include_str!("../examples/book/off-risk.silas");
const OFF_EXPOSED: &str = include_str!("../examples/book/off-exposed.silas");
const WATCH_LIST: &str = include_str!("../examples/book/watch-list.silas");
const PAIR_TARGETS: &str = include_str!("../examples/book/pair-targets.silas");
const SETTLED: &str = include_str!("../examples/book/settled.silas");

/// The query's answers at the position, each tuple printed and joined by a
/// space, the same with the bidder in every chair.
fn answers(source: &str, at: &Position) -> BTreeSet<String> {
    let query = compile(source);
    let at_chair = |declarer: Seat| -> BTreeSet<String> {
        query
            .evaluate_viewer(&at.frame(declarer), &mut budget())
            .unwrap()
            .into_iter()
            .map(|answer| answer.0.iter().map(ToString::to_string).collect::<Vec<_>>().join(" "))
            .collect()
    };
    let found = at_chair(Seat::S0);
    for declarer in [Seat::S1, Seat::S2, Seat::S3] {
        assert_eq!(at_chair(declarer), found, "the bidder in {declarer}");
    }
    found
}

fn set(items: &[&str]) -> BTreeSet<String> {
    items.iter().map(|s| (*s).to_owned()).collect()
}

/// The bidder of book hand `id` before the first lead.
fn bidder(id: &str) -> Position {
    let example = book().example(id);
    Position::holding(example.physics(), 30, Role::Bidder, example.hand(), vec![])
}

// -- What an off can lose (S-0023-a, S-0023-b, S-0023-c, S-0047-b, S-0066-b)

#[test]
fn s0023b_each_off_risks_the_count_of_both_its_suits_less_the_trump_count() {
    // The bidding hands of Ch. 2 as the book prices them, each off with the
    // count of either suit, a trump count tile dropping out (S-0023-c).
    let cases: [(&str, &[&str]); 12] = [
        // Sixes trump: the 2-0 risks the 3-2 and the 5-0.
        ("hand-1", &["2-0 3-2", "2-0 5-0"]),
        // Treys: the 6-1, the 6-4 or the 4-1.
        ("hand-2", &["6-1 6-4", "6-1 4-1"]),
        // Fours: the 5-2 off, the 5-5 and the 5-0 (and the 3-2 on its deuce
        // side, which the book does not price, the five side being higher).
        ("hand-3", &["5-2 5-5", "5-2 5-0", "5-2 3-2"]),
        // Fours: six and ace offs risk only the 5-0 (Hand 15's lesson).
        ("hand-4", &["6-0 5-0", "1-0 5-0"]),
        // Sixes: the 4-3, the 4-1 or the 3-2; the 2-0, the 3-2 or the 5-0.
        ("hand-5", &["4-3 4-1", "4-3 3-2", "2-0 3-2", "2-0 5-0"]),
        // Blanks (the transcription's corrected Hand 6): the 6-3 and the 2-1.
        ("hand-6", &["6-3 6-4", "6-3 3-2", "2-1 3-2", "2-1 4-1"]),
        // Fives: the 6-3, the 6-4 or the 3-2.
        ("hand-7", &["6-3 6-4", "6-3 3-2"]),
        // Treys: the 1-0, the 4-1 or the 5-0; the 2-1, the 4-1.
        ("hand-8", &["1-0 4-1", "1-0 5-0", "2-1 4-1"]),
        // Fives: the 4-0, the 4-1 or the 6-4; its blank count is trump.
        ("hand-9", &["4-0 4-1", "4-0 6-4"]),
        // Fives: the 4-3, the 6-4 and the 4-1 or the 3-2 (the book names the
        // 6-4 and the 3-2, and its doubles protect both sides).
        ("hand-10", &["4-3 6-4", "4-3 4-1", "4-3 3-2"]),
        // Fours: the 2-0, the 3-2 or the 5-0.
        ("hand-11", &["2-0 3-2", "2-0 5-0"]),
        // Hand 15 is Hand 4: harmless six and ace offs with fours trump.
        ("hand-15", &["6-0 5-0", "1-0 5-0"]),
    ];
    for (id, offs) in cases {
        assert_eq!(answers(OFF_RISK, &bidder(id)), set(offs), "{id}");
    }
}

#[test]
fn s0023a_an_off_that_is_count_risks_itself() {
    // Hand 7 on treys (p. 19): the 6-5 risks the 6-4; the 5-0, its own five.
    let at = position("P3", Role::Bidder, &["5-0", "5-3", "5-5", "6-5", "1-1", "6-3", "3-3"], &[]);
    assert_eq!(answers(OFF_RISK, &at), set(&["6-5 6-4", "5-0 5-0"]));
}

#[test]
fn s0066b_with_the_offs_count_trump_the_off_risks_none() {
    // P. 45: the 2-0 off with treys trump calls no count on its deuce side;
    // only the 5-0 on its blank side.
    let at = position("P3", Role::Bidder, &["3-3", "3-6", "3-5", "6-6", "5-5", "4-4", "2-0"], &[]);
    assert_eq!(answers(OFF_RISK, &at), set(&["2-0 5-0"]));
}

// -- The side a double protects (S-0024-b, S-0133-h, S-0125-d)

#[test]
fn s0133h_a_double_protects_one_side_of_an_off() {
    // Hand 10: the 4-3 under the 4-4 and the 3-3 risks nothing; Hand 11: the
    // 3-3 does not cover the 2-0's deuce side; Hand 5: the 2-2 covers the
    // 2-0's deuces, its blank side stays open.
    assert_eq!(answers(OFF_EXPOSED, &bidder("hand-10")), set(&[]));
    assert_eq!(answers(OFF_EXPOSED, &bidder("hand-11")), set(&["2-0 3-2", "2-0 5-0"]));
    assert_eq!(answers(OFF_EXPOSED, &bidder("hand-5")), set(&["4-3 4-1", "4-3 3-2", "2-0 5-0"]));
    // Hand 29, the bidder's view before the lead: the 3-3 covers the 3-2's
    // treys and the 2-2 is out; the setters take it on the deuce side.
    let hand = tiles(&["3-2", "3-3", "5-1", "5-5", "4-0", "4-1", "4-4"]);
    let at = Position::holding(physics("P4"), 30, Role::Bidder, hand, vec![]);
    assert_eq!(answers(OFF_EXPOSED, &at), set(&["3-2 3-2"]));
}

#[test]
fn s0125d_hand_27s_offs_risk_tricks_and_no_count() {
    // Fours trump: the 6-2's ten is trump and the 2-2 covers its deuces; the
    // 3-1's ace side is open, but its five is trump; the 3-3 covers its treys.
    assert_eq!(answers(OFF_EXPOSED, &bidder("hand-27")), set(&[]));
    assert_eq!(answers(OFF_RISK, &bidder("hand-27")), set(&["6-2 3-2", "3-1 3-2"]));
}

// -- The watch list (S-0090-b, S-0091-a, S-0091-b, S-0096-c)

/// A defender's view of an 84 before the first lead: the left setter holds
/// `hand`, the bidder S0 bids 84 under `decl`.
fn defender(decl: &str, hand: &[&str]) -> Position {
    position(decl, Role::Opponent1, hand, &[]).bid(84)
}

#[test]
fn s0091a_the_most_tiles_to_watch_for_each_double() {
    // The double of pip p watches the p tiles below it, one fewer when a
    // lower pip is trump; a trump double watches none, and at no trump each
    // double watches all p. The defender holds the double and six tiles of
    // other suits, none below the double in its own.
    let fillers = ["6-6", "5-5", "4-4", "3-3", "2-2", "1-1", "0-0"];
    for p in 0..=6u8 {
        let double = format!("{p}-{p}");
        let hand: Vec<&str> =
            std::iter::once(double.as_str()).chain(fillers.iter().copied().filter(|d| *d != double).take(6)).collect();
        for (decl, trump) in [("P0", Some(0)), ("P1", Some(1)), ("P2", Some(2)), ("P3", Some(3)), ("P4", Some(4))]
            .into_iter()
            .chain([("P5", Some(5)), ("P6", Some(6)), ("NT", None), ("DT", None)])
        {
            let found = answers(WATCH_LIST, &defender(decl, &hand));
            let mine: usize = found.iter().filter(|a| a.starts_with(&double)).count();
            let want = match (decl, trump) {
                ("DT", _) => 0,
                (_, Some(t)) if t == p => 0,
                (_, Some(t)) if t < p => p as usize - 1,
                _ => p as usize,
            };
            assert_eq!(mine, want, "{double} under {decl}");
        }
    }
    // The book's table: 6-6 five, 5-5 five, 4-4 four, 3-3 three, 2-2 two,
    // 1-1 one, 0-0 none; the 1-1 is worthless with blanks trump (p. 66).
    let hand = ["1-1", "6-6", "5-5", "4-4", "3-3", "2-2", "0-0"];
    let under_blanks = answers(WATCH_LIST, &defender("P0", &hand));
    assert!(!under_blanks.iter().any(|a| a.starts_with("1-1")));
}

#[test]
fn s0096c_the_2_2_the_4_4_and_the_4_1_against_treys_watch_four() {
    let at = defender("P3", &["2-2", "4-4", "4-1", "6-1", "6-5", "5-1", "0-0"]);
    assert_eq!(answers(WATCH_LIST, &at), set(&["2-2 2-1", "2-2 2-0", "4-4 4-2", "4-4 4-0"]));
}

// -- Pair targets (S-0097-c, S-0098-a, S-0098-b, S-0147-b, S-0147-c,
// S-0148-a, S-0151-d)

/// The pair targets in one suit.
fn targets_in(found: &BTreeSet<String>, suit: &str) -> BTreeSet<String> {
    found.iter().filter(|a| a.ends_with(suit)).map(|a| a.split(' ').next().unwrap().to_owned()).collect()
}

#[test]
fn s0098b_the_6_1_and_the_6_5_catch_three_sixes() {
    // The 6-1 and the 6-5 against treys catch a six off led after the 6-6:
    // the 6-4, the 6-2 and the 6-0 (the archive's adjudication; the book's
    // list omits the 6-2). The 6-3 is trump.
    let at = defender("P3", &["2-2", "4-4", "4-1", "6-1", "6-5", "5-1", "0-0"]);
    assert_eq!(targets_in(&answers(PAIR_TARGETS, &at), "q6"), set(&["6-4", "6-2", "6-0"]));
}

#[test]
fn s0147b_hand_34_the_right_setter_watches_four_small_tiles() {
    // Hand 34, fives trump. On trick 1 the right setter, with no doubles,
    // has pairs in aces, deuces, treys and sixes: the 1-0, 2-0, 3-0 and 6-2.
    let found = answers(PAIR_TARGETS, &book_at("hand-34", 3, O2));
    let all: BTreeSet<String> = found.iter().map(|a| a.split(' ').next().unwrap().to_owned()).collect();
    assert_eq!(all, set(&["1-0", "2-0", "3-0", "6-2"]));
    // Trick 2: the bidder's partner has played the 6-2; the sixes are done.
    let found = answers(PAIR_TARGETS, &book_at("hand-34", 7, O2));
    assert_eq!(targets_in(&found, "q6"), set(&[]));
    // Trick 3 (S-0148-a): the left setter's 2-0 leaves the 1-0 and the 3-0.
    let found = answers(PAIR_TARGETS, &book_at("hand-34", 11, O2));
    let all: BTreeSet<String> = found.iter().map(|a| a.split(' ').next().unwrap().to_owned()).collect();
    assert_eq!(all, set(&["1-0", "3-0"]));
    // Trick 4: his partner's 3-0 leaves the 1-0 alone, though the partner's
    // 1-1 is gone: the bidder can hold no double ahead of it.
    let found = answers(PAIR_TARGETS, &book_at("hand-34", 15, O2));
    let all: BTreeSet<String> = found.iter().map(|a| a.split(' ').next().unwrap().to_owned()).collect();
    assert_eq!(all, set(&["1-0"]));
}

#[test]
fn s0147c_hand_34_the_left_setter_weighs_the_3_3_against_two_fours() {
    // Trick 3: the 3-3 watches the 3-2 and the 3-1; the 4-3 and the 6-4
    // catch the 4-2, the 4-1 or the 4-0 behind the 4-4. Nothing else.
    let at = book_at("hand-34", 9, O1);
    assert_eq!(answers(WATCH_LIST, &at), set(&["3-3 3-2", "3-3 3-1"]));
    let found = answers(PAIR_TARGETS, &at);
    assert_eq!(found, set(&["4-2 q4", "4-1 q4", "4-0 q4"]));
}

#[test]
fn s0151d_hand_35_the_right_setter_watches_the_6_3_the_5_3_and_the_5_2() {
    // Trick 3 (S-0151-a): pairs in sixes and fives, the 6-3 or the 6-2, the
    // 5-3, the 5-2 or the 5-0. Trick 4: after the 5-0 and the 6-2 fall, the
    // 6-3, the 5-3 and the 5-2.
    let found = answers(PAIR_TARGETS, &book_at("hand-35", 11, O2));
    assert_eq!(targets_in(&found, "q6"), set(&["6-3", "6-2"]));
    assert_eq!(targets_in(&found, "q5"), set(&["5-3", "5-2", "5-0"]));
    let found = answers(PAIR_TARGETS, &book_at("hand-35", 15, O2));
    assert_eq!(targets_in(&found, "q6"), set(&["6-3"]));
    assert_eq!(targets_in(&found, "q5"), set(&["5-3", "5-2"]));
}

// -- Settled (S-0113-b)

#[test]
fn s0113b_a_hand_is_over_once_made_or_set() {
    // Hand 29: the setters' 16 on the first trick set the 30.
    assert!(!answers(SETTLED, &book_at("hand-29", 4, O1)).is_empty());
    assert!(answers(SETTLED, &book_at("hand-29", 3, O2)).is_empty());
    // Hand 32: the 3-3's trick sets the 36.
    assert!(answers(SETTLED, &book_at("hand-32", 11, O2)).is_empty());
    assert!(!answers(SETTLED, &dealt("hand-32", Role::Opponent2, &prose("hand-32")[..12])).is_empty());
}
