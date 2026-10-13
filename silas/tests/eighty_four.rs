//! The first real query: the book's four-fives 84 (Roberson, Winning 42,
//! p. 113). Fives are trump and the bidder, S0, holds
//! 5-5 5-4 5-2 5-0 4-4 4-2 2-2; the other 21 tiles are hidden, seven to a seat.
//! The book's first way to be set: an opponent holds the six-five together
//! with another five.
//!
//! Independently: the 6-5 sits with an opponent with probability 14/21. Its
//! holder's other six tiles come from the remaining 20, and the event needs
//! one of the two outstanding fives, 5-1 and 5-3, among them:
//! 1 - C(18,6)/C(20,6) = 99/190. So the event has probability
//! 2/3 * 99/190 = 33/95.
mod common;

use common::*;
use silas::*;

fn root() -> Frame {
    opening(&physics("P5"), tiles(&["5-5", "5-4", "5-2", "5-0", "4-4", "4-2", "2-2"]))
}

#[test]
fn an_opponent_guards_the_six_five_with_probability_exactly_33_over_95() {
    let frame = root();
    // 21! / (7!)^3 hidden worlds: far too many to enumerate one by one.
    assert_eq!(frame.count_worlds(), 399_072_960);
    assert!(Belief::uniform(frame.clone(), 1_000_000).is_err());
    let belief = Belief::uniform_resolving(frame, tiles(&["6-5", "5-1", "5-3"]), 1_000).unwrap();
    // One point per placement of the three tiles: 3^3 = 27.
    assert_eq!(belief.len(), 27);
    assert_eq!(belief.total_mass(), &Weight::integer(399_072_960));
    let summary = compile(SIX_FIVE).summarize(&belief, &mut budget()).unwrap();
    assert!(summary.authority.is_exact());
    assert_eq!(exact(summary.event_probability()), r(33, 95));
}

#[test]
fn the_answer_does_not_depend_on_how_much_is_resolved() {
    let query = compile(SIX_FIVE);
    let wider = Belief::uniform_resolving(root(), tiles(&["6-5", "5-1", "5-3", "6-6", "4-1"]), 1_000).unwrap();
    assert_eq!(exact(query.summarize(&wider, &mut budget()).unwrap().event_probability()), r(33, 95));
    // Resolving too little is refused, never approximated: where the 5-1 and
    // 5-3 sit is part of the question.
    let narrow = Belief::uniform_resolving(root(), tiles(&["6-5"]), 1_000).unwrap();
    assert!(query.summarize(&narrow, &mut budget()).is_err());
}
