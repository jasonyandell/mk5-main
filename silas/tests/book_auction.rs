//! The book's readings of a bid, exactly (`GAPS.md` 4, #134): the belief of
//! a seat that has heard the calls, conditioned on each caller bidding by a
//! named policy (README, The auction as evidence). Each number is exact
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
    let reading = AuctionReading::new(readers, MatchState::new(Scoring::MARKS, [0, 0]).unwrap(), arena()).unwrap();
    let evidence = reading.evidence(frame, &mut budget).unwrap();
    let read = Belief::given_auction_for(&evidence, q, 100_000, &mut budget).unwrap();
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
