// SPDX-FileCopyrightText: Jason Yandell
// SPDX-License-Identifier: CC0-1.0

//! The book's readings of a bid, exactly (`GAPS.md` 4, #134): the belief of
//! a seat that has heard the calls, conditioned on each caller bidding by a
//! named policy (README, The record as evidence). Each number is exact
//! given those policies, and stands beside the one the auction cannot move,
//! the uniform deal's.
mod common;

use std::sync::Arc;

use common::*;
use num_rational::BigRational;
use silas::*;

fn book_policy(file: &str) -> CompiledPolicy {
    let path = format!("{}/examples/book/bidding/{file}", env!("CARGO_MANIFEST_DIR"));
    let program = PolicyProgram::parse(&std::fs::read_to_string(&path).unwrap(), &path).unwrap();
    program.compile(&Registry::standard(), &physics("P6")).unwrap()
}

fn compiled(text: &str) -> CompiledQuery {
    format!("(include book) {text}").parse::<Query>().unwrap().compile(&Registry::standard()).unwrap()
}

/// The arena's declarations, the higher pip first: its last tie-break.
fn arena() -> Arc<[Physics]> {
    ["P6", "P5", "P4", "P3", "P2", "P1", "P0"].iter().map(|d| physics(d)).collect()
}

/// P(query) at `frame`: uniform, and with the calls of `readers` read.
fn uniform_and_read(
    frame: &Frame,
    readers: &[(Seat, &CompiledPolicy)],
    q: &CompiledQuery,
) -> (BigRational, BigRational) {
    let mut budget = Budget::new(u64::MAX);
    let uniform = Belief::uniform_for(frame.clone(), q, 100_000, &mut budget).unwrap();
    let before = q.summarize(&uniform, &mut budget).unwrap().event_probability();
    let readers = readers.iter().map(|(s, p)| (*s, (*p).clone())).collect();
    let reading = Reading::calls(readers, MatchState::new(Scoring::MARKS, [0, 0]).unwrap(), arena()).unwrap();
    let evidence = reading.evidence(frame, &mut budget).unwrap();
    let read = Belief::given_for(&evidence, q, 100_000, &mut budget).unwrap();
    let after = q.summarize(&read, &mut budget).unwrap().event_probability();
    assert!(before.authority().is_exact());
    assert_eq!(after.authority().models().len(), evidence.seats().len(), "exact given each read policy");
    (before.value().clone(), after.value().clone())
}

#[test]
fn c12b_the_three_passes_move_the_condition_the_bidders_hand_cannot() {
    // S-0232-c, S-0232-d, S-0233-a (the repaired C12B). Sixes; the bidder
    // holds 6-6 6-2 6-1 6-0 5-1 3-3 1-0, opens 30, and the other three pass.
    // His hand alone puts the condition the book's low lead needs (the
    // partner holds the second trump, or a setter holds it with another
    // trump) at 194/285 for every hand of the shape. Read as the arena's
    // bidder would make them, the passes say each seat lacks a suit of three
    // that exposes 10 or less: 65,180 of each seat's 116,280 hands. The
    // condition rises to 69.4%; read as the bid book's count (risk-budget,
    // no suit at risk 11 or less), it falls to 67.0%. The auction moves it,
    // by a point either way.
    let hand = tiles(&["6-6", "6-2", "6-1", "6-0", "5-1", "3-3", "1-0"]);
    let frame = opening(&physics("P6"), hand);
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/examples/book/repaired/double-up-lead-low.silas");
    let condition = Query::parse(&std::fs::read_to_string(path).unwrap(), path).unwrap();
    let condition = condition.compile(&Registry::standard()).unwrap();
    let only = book_policy("bid-only-enough.silas");
    let risk = book_policy("risk-budget.silas");
    let setters_and_partner = |p| [(Seat::S1, p), (Seat::S2, p), (Seat::S3, p)];
    assert_eq!(
        uniform_and_read(&frame, &setters_and_partner(&only), &condition),
        (r(194, 285), r(18_379_909, 26_478_093))
    );
    assert_eq!(uniform_and_read(&frame, &setters_and_partner(&risk), &condition).1, r(4_954_991, 7_391_904));
}

#[test]
fn the_bidders_thirty_read_from_the_partners_seat_in_hand_29() {
    // C4, the bid as a signal; p. 89, the partner's bid the only information
    // before play. Hand 29 (fours, 30): the bidder opens 30, the others pass,
    // and every seat's call there is bid-only-enough's. From the partner's
    // seat before the lead, reading the bidder's 30 and his fours as that
    // policy would make them, and then the setters' passes too:
    let partner = tiles(&["3-0", "4-5", "3-6", "1-1", "2-1", "4-2", "5-2"]);
    let calls = ["30", "pass", "pass", "pass"].iter().map(|c| c.parse().unwrap()).collect();
    let auction = Auction::new(AuctionConfig::STRAIGHT, Seat::S3, calls).unwrap();
    let frame = Frame::new(auction, physics("P4"), Seat::S2, partner, vec![]).unwrap();
    let only = book_policy("bid-only-enough.silas");
    let bidder = [(Seat::S0, &only)];
    let everyone = [(Seat::S0, &only), (Seat::S1, &only), (Seat::S3, &only)];
    for (q, uniform, read, all) in [
        // He holds the 4-4, the trump double: a third of the deal, more
        // than two thirds of the 30s in fours.
        ("(query (holds S0 4-4))", r(1, 3), r(6_003, 8_669), r(2_198_843, 3_237_964)),
        // He holds a double.
        ("(query (some (roles d) (double d) (holds S0 d)))", r(2_441, 2_584), r(8_449, 8_669), r(6_346_177, 6_475_928)),
        // He holds two doubles or more.
        (
            "(query (<= 2 (card (roles d) (double d) (holds S0 d))))",
            r(313, 456),
            r(6_814, 8_669),
            r(5_252_929, 6_475_928),
        ),
    ] {
        let q = compiled(q);
        assert_eq!(uniform_and_read(&frame, &bidder, &q), (uniform.clone(), read), "the bidder read");
        assert_eq!(uniform_and_read(&frame, &everyone, &q), (uniform, all), "every call read");
    }
}

/// The deal of Hands 32 and 33 (one deal, turned a seat), each seat's tiles
/// from the book, by seat at the table of `bidder`'s hand: S0 the bidder,
/// S1 the left setter (opponent 1), S2 the partner, S3 the right setter.
fn hand_33() -> [DominoSet; 4] {
    [
        tiles(&["1-1", "0-2", "0-4", "5-5", "1-2", "4-6", "4-4"]),
        tiles(&["4-2", "4-1", "5-1", "1-6", "3-6", "5-6", "6-6"]),
        tiles(&["3-3", "3-5", "3-4", "2-3", "5-4", "5-2", "6-2"]),
        tiles(&["0-0", "0-6", "0-5", "0-3", "0-1", "1-3", "2-2"]),
    ]
}

fn hand_32() -> [DominoSet; 4] {
    let [o2, p, b, o1] = [hand_33()[2], hand_33()[1], hand_33()[3], hand_33()[0]];
    [b, o1, p, o2]
}

/// `viewer`'s frame before the opening lead, under `calls` from `shaker`.
fn before_the_lead(decl: &str, shaker: Seat, calls: &[&str], viewer: Seat, deal: [DominoSet; 4]) -> Frame {
    let calls = calls.iter().map(|c| c.parse().unwrap()).collect();
    let auction = Auction::new(AuctionConfig::STRAIGHT, shaker, calls).unwrap();
    Frame::new(auction, physics(decl), viewer, deal[viewer.index()], vec![]).unwrap()
}

/// The hand law of `seat` at `frame`: the uniform deal's, and with the calls
/// of `readers` read.
fn laws(frame: &Frame, seat: Seat, readers: &[(Seat, &CompiledPolicy)]) -> (Belief, Belief) {
    let mut budget = Budget::new(u64::MAX);
    let uniform = Belief::uniform_hand_law(frame.clone(), seat, 116_280, &mut budget).unwrap();
    let readers = readers.iter().map(|(s, p)| (*s, (*p).clone())).collect();
    let reading = Reading::calls(readers, MatchState::new(Scoring::MARKS, [0, 0]).unwrap(), arena()).unwrap();
    let evidence = reading.evidence(frame, &mut budget).unwrap();
    let read = Belief::hand_law(&evidence, seat, 116_280).unwrap();
    assert!(uniform.authority().is_exact());
    assert_eq!(read.authority().models().len(), evidence.seats().len(), "exact given each read policy");
    (uniform, read)
}

/// P(query) under `belief`.
fn p(belief: &Belief, q: &CompiledQuery) -> BigRational {
    q.summarize(belief, &mut Budget::new(u64::MAX)).unwrap().event_probability().value().clone()
}

fn example(file: &str) -> CompiledQuery {
    let path = format!("{}/examples/book/bidding/{file}", env!("CARGO_MANIFEST_DIR"));
    Query::parse(&std::fs::read_to_string(&path).unwrap(), &path).unwrap().compile(&Registry::standard()).unwrap()
}

/// "He holds every tile of `hand`", of S0.
fn holds_all(hand: DominoSet) -> CompiledQuery {
    let held: Vec<String> = hand.iter().map(|d| format!("(holds S0 {d})")).collect();
    compiled(&format!("(query {})", held.join(" ")))
}

#[test]
fn a_losing_31_read_from_its_callers_partner() {
    // W42-p054-b: a 31 that is outbid still says the caller probably holds a
    // double or two. The deal of Hand 33 (fours, 35), under an auction of
    // ours: the left setter opens 30, the partner passes, the right setter
    // bids 31, and the bidder, the shaker, takes it at 35. The left setter,
    // the 31's partner, before the lead, reads the right setter's 31 alone.
    // Read as bid-only-enough makes it (a raise worth 42 less what its offs
    // expose), the 31 lifts "a double" from 2,441/2,584 to 24,033/24,989 and
    // "two or more" from 313/456 to 18,684/24,989; read as risk-budget makes
    // it (42 less the risk, so a risk of exactly 11), a double is certain
    // and two or more is 2,158/2,187. The book's right-setter hand (the
    // blanks, risk 6) bids 31 here by bid-only-enough, not by risk-budget.
    let frame = before_the_lead("P4", Seat::S0, &["30", "pass", "31", "35"], Seat::S1, hand_33());
    let only = book_policy("bid-only-enough.silas");
    let risk = book_policy("risk-budget.silas");
    let a_double = example("losing-31-holds-a-double.silas");
    let two = compiled("(query (right-setter s) (<= 2 (card (roles d) (double d) (holds s d))))");
    let read = |p: &CompiledPolicy, q: &CompiledQuery| uniform_and_read(&frame, &[(Seat::S3, p)], q);
    assert_eq!(read(&only, &a_double), (r(2_441, 2_584), r(24_033, 24_989)));
    assert_eq!(read(&risk, &a_double), (r(2_441, 2_584), r(1, 1)));
    assert_eq!(read(&only, &two), (r(313, 456), r(18_684, 24_989)));
    assert_eq!(read(&risk, &two), (r(313, 456), r(2_158, 2_187)));
    // The caller's hand law answers the same, and finds the book's hand
    // among bid-only-enough's 31s and not among risk-budget's.
    let held: Vec<String> = hand_33()[3].iter().map(|d| format!("(holds S3 {d})")).collect();
    let book = compiled(&format!("(query {})", held.join(" ")));
    let (_, under_only) = laws(&frame, Seat::S3, &[(Seat::S3, &only)]);
    let (_, under_risk) = laws(&frame, Seat::S3, &[(Seat::S3, &risk)]);
    assert_eq!((p(&under_only, &a_double), p(&under_only, &two)), (r(24_033, 24_989), r(18_684, 24_989)));
    assert_eq!((p(&under_risk, &a_double), p(&under_risk, &two)), (r(1, 1), r(2_158, 2_187)));
    assert!(p(&under_only, &book) > r(0, 1));
    assert_eq!(p(&under_risk, &book), r(0, 1));
}

#[test]
fn a_35_bidder_holds_two_offs_and_a_36_bidder_one() {
    // W42-p054-a: a 35 probably means two offs. GAPS.md 4 pairs a 36 with
    // one off; no catalog row of pp. 53-54 states it, and W42-p046-k (at 35
    // or 36 the off is probably not a high one) is the nearest. The bidder's
    // offs, a tile neither trump nor double, are a question of his whole
    // hand, which his hand law answers. Hands 33 (fours, 35) and 32 (blanks, 36) are one
    // deal; the bidder opens, the rest pass, and the left setter reads the
    // bidder's call and declaration before the lead, as risk-budget (the bid
    // book's count, 42 less the risk) makes them. bid-only-enough raises to
    // the least bid, so in one round it never bids 35 or 36.
    //
    // Uniformly the bidder holds two offs 77/646 of the time. The 35 makes
    // it 1,587/1,751, one off the rest (164/1,751): seven at risk is two
    // offs and a five, or one off, a five and the trump trick. The 36 makes
    // it one off, certainly: six at risk is one trick and a five. Read with
    // the setters' and the partner's passes too, the 35 is two offs
    // 1,897,452/2,112,947. Both of the book's bidders are among the hands.
    let risk = book_policy("risk-budget.silas");
    let offs = |n: &str| {
        compiled(&format!(
            "(query (bidder b) ({n} (card (roles o) (holds b o) (not (trump-tile o)) (not (double o)))))"
        ))
    };
    let (none, one, two, more) = (offs("= 0"), offs("= 1"), example("bidder-holds-two-offs.silas"), offs("<= 3"));
    let law = |b: &Belief| [&none, &one, &two, &more].map(|q| p(b, q));

    let frame = before_the_lead("P4", Seat::S3, &["35", "pass", "pass", "pass"], Seat::S1, hand_33());
    let (uniform, read) = laws(&frame, Seat::S0, &[(Seat::S0, &risk)]);
    assert_eq!(law(&uniform), [r(1, 969), r(77, 3_876), r(77, 646), r(1_111, 1_292)]);
    assert_eq!(law(&read), [r(0, 1), r(164, 1_751), r(1_587, 1_751), r(0, 1)]);
    assert!(p(&read, &holds_all(hand_33()[0])) > r(0, 1));
    let everyone = [(Seat::S0, &risk), (Seat::S2, &risk), (Seat::S3, &risk)];
    let (_, all) = laws(&frame, Seat::S0, &everyone);
    assert_eq!(law(&all), [r(0, 1), r(215_495, 2_112_947), r(1_897_452, 2_112_947), r(0, 1)]);

    let frame = before_the_lead("P0", Seat::S3, &["36", "pass", "pass", "pass"], Seat::S1, hand_32());
    let (uniform, read) = laws(&frame, Seat::S0, &[(Seat::S0, &risk)]);
    assert_eq!(law(&uniform), [r(1, 14_535), r(91, 29_070), r(182, 4_845), r(1_859, 1_938)]);
    assert_eq!(law(&read), [r(0, 1), r(1, 1), r(0, 1), r(0, 1)]);
    assert!(p(&read, &holds_all(hand_32()[0])) > r(0, 1));
}
