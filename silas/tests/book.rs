// SPDX-FileCopyrightText: Jason Yandell
// SPDX-License-Identifier: CC0-1.0

//! The study's 21 rule gates (Roberson, *Winning 42*), written as Silas
//! policies in `examples/book/gates/` and checked on positions, most of them
//! the book's own. They are the record of what the 2026-10-12 study priced;
//! the rules as the book means them are `examples/book/*.silas`, checked in
//! `tests/book_rules.rs`.
//!
//! A position names its seats by role, as the book does: the bidder, the seat
//! after him (the "left" setter), his partner, and the last setter, in play
//! order. Every check runs at all four seatings, the bidder in each chair
//! (S0 is the study's pour), and must come out the same: the rules name the
//! bidder through `declarer`, never a chair.
//!
//! A position names the catalog entry it comes from (`W42-pNNN-x`, in the
//! private study notes) when it is the book's; "ours" marks a position or a
//! play the book does not give. The book's hands, deals, declarations, bids and
//! narrated plays are read from `examples/book/hands.json`, the reviewed
//! transcription's facts, and checked there by `book_fixture.rs`. Each rule gets a position where it fires and
//! allows the book's tile, and one where it does not fire. Where the book and
//! the study's Python gate part ways, the test says so: the gate is the
//! reading of #123 that this file checks.
mod common;

use common::fixture::{Position, book};
use common::rules::*;
use common::*;
use silas::*;

const C01A: &str = include_str!("../examples/book/gates/c01a-top-trump-for-the-donation.silas");
const C01B: &str = include_str!("../examples/book/gates/c01b-partner-count-on-a-trump-lead.silas");
const C02: &str = include_str!("../examples/book/gates/c02-high-trump-handoff.silas");
const C02_READING: &str = include_str!("../examples/book/gates/c02-reading-singleton.silas");
const C03: &str = include_str!("../examples/book/gates/c03-biggest-count-first.silas");
const C07: &str = include_str!("../examples/book/gates/c07-pounce.silas");
const C08: &str = include_str!("../examples/book/gates/c08-no-count-on-the-double.silas");
const C09: &str = include_str!("../examples/book/gates/c09-partner-never-leads-trump.silas");
const C10: &str = include_str!("../examples/book/gates/c10-off-on-trick-one.silas");
const C11: &str = include_str!("../examples/book/gates/c11-double-ahead-of-the-off.silas");
const C12A: &str = include_str!("../examples/book/gates/c12a-four-trumps-lead-the-double.silas");
const C12B: &str = include_str!("../examples/book/gates/c12b-four-trumps-lead-low.silas");
const C13A: &str = include_str!("../examples/book/gates/c13a-shed-a-singleton.silas");
const C13B: &str = include_str!("../examples/book/gates/c13b-protect-the-count.silas");
const C13C: &str = include_str!("../examples/book/gates/c13c-low-from-a-pair.silas");
const C13D: &str = include_str!("../examples/book/gates/c13d-lead-a-singleton.silas");
const C14: &str = include_str!("../examples/book/gates/c14-crisis-cash-the-double.silas");
const C15: &str = include_str!("../examples/book/gates/c15-setter-leads-call-for-count.silas");
const C21: &str = include_str!("../examples/book/gates/c21-six-five-bait.silas");
const C22: &str = include_str!("../examples/book/gates/c22-trump-the-double-from-the-left.silas");
const C23: &str = include_str!("../examples/book/gates/c23-partner-leads-doubles.silas");

const POLICIES: [&str; 20] =
    [C01A, C01B, C02, C03, C07, C08, C09, C10, C11, C12A, C12B, C13A, C13B, C13C, C13D, C14, C15, C21, C22, C23];

#[test]
fn every_book_file_parses_round_trips_and_compiles_under_every_declaration() {
    for source in POLICIES {
        let parsed = program(source);
        assert_eq!(parsed.to_string().parse::<PolicyProgram>().unwrap(), parsed);
        assert!(!source.contains("S0") && !source.contains("S2"), "a book rule names no chair");
        for rules in reference::Straight42::ALL {
            parsed.compile(&Registry::standard(), &rules.physics()).unwrap();
        }
    }
    let reading = C02_READING.parse::<Query>().unwrap();
    assert_eq!(reading.to_string().parse::<Query>().unwrap(), reading);
    reading.compile(&Registry::standard()).unwrap();
}

// C1a: the bidder leads the top trump so the partner can give count.

#[test]
fn c01a_fires_in_hand_32_and_leads_the_top_blank() {
    // W42-p109-a: after T1 the partner has shown void (he gave the 4-1), the
    // 4-0 is out; the book leads the 6-0. The gate fires here too.
    let at = book_at("hand-32", 4, B);
    assert_eq!(allows(C01A, &at, "lead-the-top-trump"), tiles(&["6-0"]));
}

#[test]
fn c01a_is_the_gate_and_misses_the_book_in_hand_28() {
    // W42-p098-c: after T1 every seat followed blanks, so the partner has not
    // shown void, and neither the gate nor the encoding fires. One trump (2-0)
    // is out; the book leads the 0-0 to pull it, so that the partner can then
    // give count: its condition is what the partner will know (GAPS.md,
    // another seat's knowledge).
    let at = book_at("hand-28", 4, B);
    assert_eq!(fires(C01A, &at), None);
}

#[test]
fn c01a_does_not_fire_in_hand_33_where_the_book_still_leads_the_trump() {
    // W42-p111-d: after T4 no opposing trump is out, so neither the gate nor
    // the encoding fires. The book leads the 4-0, the bidder's last trump,
    // because the partner has not yet had a safe trick to give count on. The
    // gate's "a trump still out" excludes a book play: the book's condition is
    // what the partner knows, not what the bidder knows.
    let at = book_at("hand-33", 16, B);
    assert_eq!(fires(C01A, &at), None);
}

// C1b: the partner, out of trump, gives count only on the top trump.

#[test]
fn c01b_top_lead_gives_the_count_in_hands_31_and_32() {
    // W42-p106-a: the bidder leads the 2-2; the partner, void in deuces, gives
    // the 5-0, his only count.
    let at = book_at("hand-31", 2, P);
    assert_eq!(allows(C01B, &at, "top-lead-give-five"), tiles(&["5-0"]));
    // W42-p109-a: the bidder leads the 0-0; the partner gives the 4-1.
    let at = book_at("hand-32", 2, P);
    assert_eq!(allows(C01B, &at, "top-lead-give-five"), tiles(&["4-1"]));
}

#[test]
fn c01b_not_top_lead_keeps_the_count() {
    // Ours, on Hand 31's deal: the bidder leads the 6-2 instead of the 2-2.
    // The partner, void in deuces, cannot know the trick is safe (the 2-2 is
    // live), so he throws anything but his 5-0. The book has no worked
    // example of this branch (W42-p038-d states it).
    let at = dealt("hand-31", P, &then(&[], &["6-2", "2-1"]));
    assert_eq!(allows(C01B, &at, "not-top-lead-no-count"), tiles(&["6-6", "0-0", "3-1", "6-0", "6-1", "5-3"]));
}

#[test]
fn c01b_does_not_fire_without_count_or_while_holding_trump() {
    // W42-p109-a, T2: the partner has no count left and throws the 4-2.
    let at = book_at("hand-32", 6, P);
    assert_eq!(fires(C01B, &at), None);
    // W42-p101-b, Hand 29 T1: the partner holds trump and must follow.
    let at = book_at("hand-29", 2, P);
    assert_eq!(fires(C01B, &at), None);
}

// C2: the table's handoff, and the book's reading of it.

#[test]
fn c02_fires_in_hand_30_and_the_book_reading_is_false_there() {
    // W42-p102-c, Hand 30 T1: the bidder leads the 6-5; the partner holds the
    // 6-6, the top six, and plays it. The handoff allows exactly that.
    let at = book_at("hand-30", 2, P);
    assert_eq!(allows(C02, &at, "play-the-top-trump"), tiles(&["6-6"]));
    // The book's p. 25 reads a partner's top trump as his only one. Here he
    // still holds the 6-0 and the 6-1: the reading is false in the book's own
    // deal. From the bidder's chair, over the deals he cannot tell apart, the
    // partner is void in trump with probability 26/57 (his six tiles miss both
    // live sixes, of nineteen hidden): the reading is a convention, not a
    // fact of the deal.
    let bidder = dealt("hand-30", B, &prose("hand-30")[..3]);
    let reading = C02_READING.parse::<Query>().unwrap().compile(&Registry::standard()).unwrap();
    for declarer in Seat::ALL {
        let belief = Belief::uniform_resolving(bidder.frame(declarer), tiles(&["6-0", "6-1"]), 100).unwrap();
        let p = exact(reading.summarize(&belief, &mut budget()).unwrap().event_probability());
        assert_eq!(p, r(26, 57), "the bidder in {declarer}");
    }
}

#[test]
fn c02_does_not_fire_in_hand_29() {
    // W42-p101-b, Hand 29 T1: the bidder leads the 4-0; the top four (4-4) is
    // the bidder's own, so the partner, holding 4-2 and 5-4, does not hold it.
    let at = book_at("hand-29", 2, P);
    assert_eq!(fires(C02, &at), None);
}

// C3: biggest count first.

#[test]
fn c03_gives_the_ten_before_the_five() {
    // W42-p106-c: in Hand 31 the bidder reasons that a partner holding the
    // 5-5 would have given it instead of the 5-0. Ours: that partner, Hand
    // 31's deal with the 5-5 and 6-6 exchanged between partner and right
    // opponent. C1b agrees.
    let hand = book().example("hand-31");
    let mut deal = hand.deal();
    let (double_six, double_five) = (tile("6-6"), tile("5-5"));
    assert!(deal[P.index()].remove(double_six) && deal[O2.index()].remove(double_five));
    deal[P.index()].insert(double_five);
    deal[O2.index()].insert(double_six);
    let at = Position::dealt(hand.physics(), 30, deal, P, &prose("hand-31")[..2]);
    assert_eq!(allows(C03, &at, "give-a-ten"), tiles(&["5-5"]));
    assert_eq!(allows(C01B, &at, "top-lead-give-ten"), tiles(&["5-5"]));
}

#[test]
fn c03_does_not_fire_with_one_count_tile() {
    // W42-p106-a, Hand 31 T1 as dealt: the 5-0 is the partner's only count, so
    // the gate's "two or more" fails; C1b gives it.
    let at = book_at("hand-31", 2, P);
    assert_eq!(fires(C03, &at), None);
}

// C7: the pounce.

#[test]
fn c07_pounces_in_hand_32() {
    // W42-p109-b, Hand 32 T3: the bidder leads his 3-1 off; the left opponent,
    // void in treys and out of trump, holds 6-4 and 5-5. The book plays the
    // 6-4 at once (the partner wins with the 3-3 and the bid is set); the 5-5
    // is the same ten.
    let at = dealt("hand-32", O1, &then(&prose("hand-32")[..8], &["3-1"]));
    assert_eq!(allows(C07, &at, "play-count-now"), tiles(&["6-4", "5-5"]));
}

#[test]
fn c07_does_not_fire_when_the_setter_can_follow() {
    // W42-p098-e, Hand 28 T3: the bidder leads the 3-1; the left opponent holds
    // 3-2 and 3-3 and can follow. The gate needs a void and does not fire, but
    // the book plays the 3-2 here, count before the partner: its pounce
    // (pp. 43-46) is wider than the gate.
    let at = book_at("hand-28", 9, O1);
    assert_eq!(fires(C07, &at), None);
}

// C8 and C13b: no count on the bidder's double; protect the count.

/// W42-p049-a (ours as a position: the book gives two tiles and the bidder's
/// double): treys trump, the bidder leads the 6-6, the left opponent holds
/// the 6-4 and the 6-0.
fn guarded_six_four(extra: &str) -> Position {
    position("P3", O1, &["6-4", "6-0", "5-5", "2-1", "1-1", "5-1", extra], &["6-6"])
}

#[test]
fn c08_and_c13b_follow_the_double_with_the_guard() {
    let at = guarded_six_four("2-0");
    assert_eq!(allows(C08, &at, "follow-without-count"), tiles(&["6-0"]));
    assert_eq!(allows(C13B, &at, "follow-with-the-guard"), tiles(&["6-0"]));
}

#[test]
fn c13b_is_exactly_two_and_leaves_three_to_c08() {
    // With three sixes neither the gate's C13b nor the encoding fires; C8
    // does.
    let at = guarded_six_four("6-1");
    assert_eq!(fires(C13B, &at), None);
    assert_eq!(allows(C08, &at, "follow-without-count"), tiles(&["6-0", "6-1"]));
}

#[test]
fn c08_and_c13b_do_not_fire_on_a_trump_double_where_the_book_protects() {
    // W42-p111-a, Hand 33 T1: the bidder leads the 4-4 (trump); the left
    // opponent holds 4-2 and 4-1 and plays the 4-2, saving the count. The
    // gates take only non-trump doubles, so neither fires on the book's
    // instance of the rule.
    let at = book_at("hand-33", 1, O1);
    assert_eq!(fires(C08, &at), None);
    assert_eq!(fires(C13B, &at), None);
}

// C9: the partner never leads trump, and leads by what a suit can draw.

#[test]
fn c09_orders_the_lead_in_hand_30_as_the_book_does() {
    // W42-p103-d, Hand 30 T3: the partner leads holding 3-1, 5-0, 5-3 and two
    // trumps (6-0, 6-1), no double and no walker. Never trump allows the
    // three; the ordering narrows them to the 3-1, whose suit has only a five
    // out (the 3-2), where the fives' suit still has the 5-5: the book's lead.
    let at = book_at("hand-30", 8, P);
    assert_eq!(allows(C09, &at, "lead-from-a-five-suit"), tiles(&["3-1"]));
}

#[test]
fn c09_orders_the_lead_without_trump_in_hand_31() {
    // W42-p107-b, Hand 31 T4: the partner holds no trump and leads the 3-1.
    // The gate's ordering does not need trump: the treys have no count out.
    let at = book_at("hand-31", 12, P);
    assert_eq!(allows(C09, &at, "lead-from-a-count-safe-suit"), tiles(&["3-1"]));
}

#[test]
fn c09_does_not_fire_off_lead() {
    // Hand 31 T1: the partner follows the bidder's lead.
    let at = book_at("hand-31", 2, P);
    assert_eq!(fires(C09, &at), None);
}

// C10: the off on trick one.

#[test]
fn c10_fires_on_hand_14_where_the_book_leads_trump() {
    // W42-p028-b, Hand 14: fours, the bidder holds five trumps and one off
    // (5-2). The gate makes the off the book's tile; the book's p. 24 only
    // permits it, and in this hand the book leads the 4-4 (and is later set on
    // the 5-2). The gate reads a permission as an instruction.
    let at = book().bidders("hand-14", &[]);
    assert_eq!(allows(C10, &at, "lead-the-off"), tiles(&["5-2"]));
}

#[test]
fn c10_is_exactly_one_off() {
    // Ours: Hand 14 with the 6-6 for the 6-1, two offs. Neither the gate nor
    // the encoding fires.
    let at = position("P4", B, &["5-2", "4-1", "6-1", "6-4", "4-2", "4-3", "4-4"], &[]);
    assert_eq!(fires(C10, &at), None);
}

#[test]
fn c10_does_not_fire_on_hand_27() {
    // W42-p094-d, Hand 27: fours with two trumps. Neither the gate nor the
    // encoding fires; the book leads its 6-2 off on trick one anyway, as a
    // stated exception for two-trump hands (W42-p091-c).
    let at = book().bidders("hand-27", &[]);
    assert_eq!(fires(C10, &at), None);
}

// C11: the double ahead of the off.

#[test]
fn c11_fires_in_hand_28() {
    // W42-p099-b, Hand 28 T6: trumps are gone; the bidder holds 6-6 and the 6-1
    // off (the 6-2 is out) and leads the double.
    let at = book_at("hand-28", 20, B);
    assert_eq!(allows(C11, &at, "lead-the-double"), tiles(&["6-6"]));
}

#[test]
fn c11_is_the_gate_and_misses_the_book_in_hand_30() {
    // W42-p105-a, Hand 30 T5: the last trump (6-1) is in the partner's hand,
    // so "trumps drawn" fails and neither the gate nor the encoding fires. The
    // book leads the 2-2 ahead of the 2-0 off all the same.
    let at = book_at("hand-30", 16, B);
    assert_eq!(fires(C11, &at), None);
}

#[test]
fn c11_does_not_fire_for_a_double_of_another_suit() {
    // W42-p024-a (ours as a hand): a 2-2 does not protect a 6-3 off.
    let at = position("P4", B, &["2-2", "6-3", "4-4", "4-3", "4-1", "4-0", "1-1"], &[]);
    assert_eq!(fires(C11, &at), None);
}

// C12: four trumps on trick one.

#[test]
fn c12a_leads_the_double_in_hand_34() {
    // W42-p113-a, Hand 34 T1: fives, the bidder holds 5-5, 5-4, 5-2, 5-0, the
    // double and the third but not the 6-5; the book leads the 5-5. The bid is
    // 84: all 42 points.
    let at = book_at("hand-34", 0, B);
    assert_eq!(allows(C12A, &at, "lead-the-double"), tiles(&["5-5"]));
    assert_eq!(fires(C12B, &at), None);
}

#[test]
fn c12b_leads_the_lowest_trump() {
    // W42-p183-c (ours as hands): fours, the double without the 6-4 and 5-4.
    let at = position("P4", B, &["4-4", "4-2", "4-1", "4-0", "3-3", "5-5", "6-6"], &[]);
    assert_eq!(allows(C12B, &at, "lead-the-lowest-trump"), tiles(&["4-0"]));
    assert_eq!(fires(C12A, &at), None);
    let at = position("P4", B, &["4-4", "4-3", "4-2", "4-1", "3-3", "5-5", "6-6"], &[]);
    assert_eq!(allows(C12B, &at, "lead-the-lowest-trump"), tiles(&["4-1"]));
    // The same shape under doubles trump, where the gate does not look.
    let at = position("DT", B, &["6-6", "3-3", "2-2", "1-1", "6-5", "5-4", "4-3"], &[]);
    assert_eq!(allows(C12B, &at, "lead-the-lowest-trump"), tiles(&["1-1"]));
}

#[test]
fn c12_does_not_fire_on_three_trumps_where_the_book_leads_low() {
    // W42-p101-a, Hand 29 T1: fours, the bidder holds 4-4, 4-1, 4-0 (missing
    // the 6-4 and 5-4) and leads the 4-0 low, the B rule with three trumps
    // (also W42-p183-g). Both gates take four trumps only.
    let at = book_at("hand-29", 0, B);
    assert_eq!(fires(C12A, &at), None);
    assert_eq!(fires(C12B, &at), None);
}

// C13a: shed a singleton on the bidder's trump lead.

#[test]
fn c13a_sheds_the_books_singleton() {
    // W42-p109-a, Hand 32 T1: the right opponent, no blanks, throws the 6-2,
    // his only six: the gate's set, and the encoding's.
    let at = book_at("hand-32", 3, O2);
    assert_eq!(allows(C13A, &at, "throw-a-lone-tile"), tiles(&["6-2"]));
    // W42-p111-a, Hand 33 T1: the right opponent throws the 6-0, his only six;
    // the 2-2, his only deuce, is the other lone tile.
    let at = book_at("hand-33", 3, O2);
    assert_eq!(allows(C13A, &at, "throw-a-lone-tile"), tiles(&["6-0", "2-2"]));
}

#[test]
fn c13a_does_not_fire_while_the_setter_holds_trump() {
    // Hand 32 T1: the left opponent must follow with a blank.
    let at = book_at("hand-32", 1, O1);
    assert_eq!(fires(C13A, &at), None);
}

// C13c: low from a pair.

#[test]
fn c13c_follows_low_in_hand_35() {
    // W42-p117-b, Hand 35 T5: the bidder leads the 5-5; the right opponent
    // holds two fives, 5-1 and 6-5, neither count, and plays the 5-1 (the book's
    // reason is to keep his sixes; the tile is the gate's).
    let at = book_at("hand-35", 19, O2);
    assert_eq!(allows(C13C, &at, "follow-low"), tiles(&["5-1"]));
}

#[test]
fn c13c_does_not_fire_when_the_pair_holds_count() {
    // W42-p098-e, Hand 28 T3: the left opponent's treys are 3-2 and 3-3.
    let at = book_at("hand-28", 9, O1);
    assert_eq!(fires(C13C, &at), None);
}

// C13d: lead a singleton (the gate's reading; no page states it).

#[test]
fn c13d_fires_in_hand_28_where_the_book_leads_the_ten_caller() {
    // W42-p098-f, Hand 28 T4: the right opponent leads holding 5-4, 4-2, 1-1,
    // 5-2. The gate's tile, and the encoding's, is the 1-1, his only ace; the
    // book leads the 5-4, calling for the 5-5 that sets the bid.
    let at = book_at("hand-28", 12, O2);
    assert_eq!(allows(C13D, &at, "lead-the-lone-tile"), tiles(&["1-1"]));
}

#[test]
fn c13d_does_not_fire_off_lead() {
    let at = book_at("hand-28", 9, O1);
    assert_eq!(fires(C13D, &at), None);
}

// C14: crisis management.

#[test]
fn c14_cashes_a_double_after_one_trump_trick_with_one_follower() {
    // Ours (the gate's trigger, W42-p093-c's shape): fours, the bidder leads
    // the 4-4, the left opponent follows, the right opponent shows out. On
    // lead again with trump still out, the encoding allows every non-trump
    // double (GAP aggregate: the gate's is the suit with most unplayed).
    let at = position("P4", B, &["3-3", "1-1", "2-2", "5-2", "6-4", "4-3"], &["4-4", "4-0", "4-1", "3-0"]);
    assert_eq!(allows(C14, &at, "cash-a-double"), tiles(&["1-1", "2-2", "3-3"]));
    // Both opponents followed: no crisis.
    let at = position("P4", B, &["3-3", "1-1", "2-2", "5-2", "6-4", "4-3"], &["4-4", "4-0", "4-1", "4-2"]);
    assert_eq!(fires(C14, &at), None);
}

#[test]
fn c14_does_not_fire_after_two_trump_tricks_where_the_book_cashes() {
    // W42-p093-c, Hand 26 (fours): after two trump tricks an opponent still
    // holds the 4-2; the bidder cashes the 5-5 on T3, then leads the double of
    // the suit with more tiles out. The tricks' other tiles are ours (the book
    // gives the shape, not the plays). The gate's trigger is one trump trick,
    // so neither the gate nor the encoding fires on the book's own crisis.
    let plays = [
        "4-4", "4-0", "4-1", "4-3", // T1
        "6-4", "4-5", "6-6", "3-0", // T2
        "5-5", "5-3", "5-1", "6-5", // T3
    ];
    let at = book().bidders("hand-26", &plays.map(tile));
    assert_eq!(fires(C14, &at), None);
    // W42-p106-d, Hand 31 T2: the book's crisis lead is the 6-5 bait (C21).
    let at = book_at("hand-31", 4, B);
    assert_eq!(fires(C14, &at), None);
}

// C15: setter leads.

#[test]
fn c15_leads_the_count_calling_double_in_hand_29() {
    // W42-p101-c, Hand 29 T2: the left opponent leads the 2-2, calling for the
    // 3-2.
    let at = book_at("hand-29", 4, O1);
    assert_eq!(allows(C15, &at, "a-double-that-calls-for-count"), tiles(&["2-2"]));
}

#[test]
fn c15_leads_the_double_where_the_book_leads_the_count_that_sets() {
    // W42-p098-f, Hand 28 T4: the setters hold 6 and need 13. The gate's tier
    // A is the 1-1 (it calls the 4-1); the book leads the 5-4, which calls the
    // 5-5 and sets.
    let at = book_at("hand-28", 12, O2);
    assert_eq!(allows(C15, &at, "a-double-that-calls-for-count"), tiles(&["1-1"]));
    // W42-p108-b, Hand 31 T5: tier A is the 4-4 (it calls the 6-4); the book
    // leads the 4-1, which calls the 6-4 for fifteen.
    let at = book_at("hand-31", 16, O2);
    assert_eq!(allows(C15, &at, "a-double-that-calls-for-count"), tiles(&["4-4"]));
}

#[test]
fn c15_does_not_fire_without_a_count_calling_lead() {
    // W42-p101-d, Hand 29 T3: no lead calls for count the book allows; the
    // book leads the 1-0 just to see what happens.
    let at = book_at("hand-29", 8, O1);
    assert_eq!(fires(C15, &at), None);
}

#[test]
fn c15_leads_quietly_only_when_one_point_sets() {
    // Ours: sixes trump, a bid of 41. The left setter trumps the bidder's 1-1
    // for a trick of no count; his side needs two points more to set, the
    // bidder's side forty. On lead, he holds every count tile his leads could
    // call, so none calls for count. One more point sets: any lead does.
    let hand = ["5-5", "6-4", "5-0", "3-2", "2-0", "3-3"];
    let plays = ["1-1", "6-0", "1-0", "2-1"];
    let at = position("P6", O1, &hand, &plays).bid(41);
    assert_eq!(allows(C15, &at, "one-point-sets-lead-quietly"), tiles(&hand));
    // Under a bid of 30 the setters need twelve: no rule fires.
    let at = position("P6", O1, &hand, &plays);
    assert_eq!(fires(C15, &at), None);
}

// C21: the six-five bait.

#[test]
fn c21_leads_the_six_five_in_hand_31() {
    // W42-p106-d, Hand 31 T2: deuces, the bidder holds 6-5 and 6-4 and leads
    // the 6-5.
    let at = book_at("hand-31", 4, B);
    assert_eq!(allows(C21, &at, "lead-the-six-five"), tiles(&["6-5"]));
}

#[test]
fn c21_does_not_fire_when_the_six_five_is_trump() {
    // W42-p103-a, Hand 30 T1: sixes; the 6-5 is a trump, led as one.
    let at = book_at("hand-30", 0, B);
    assert_eq!(fires(C21, &at), None);
}

// C22: in a trump set, trump the bidder's double from the left.

/// Ours (W42-p051-a gives no hand): deuces trump. T1 is the trump set: the
/// bidder leads the 2-2, the trump-rich setter follows with the 2-0, his
/// partner shows out. The bidder then leads the 6-6; the setter holds two
/// trumps and no six.
const TRUMP_SET: [&str; 6] = ["5-2", "4-2", "1-1", "3-3", "1-0", "4-0"];

#[test]
fn c22_trumps_in_from_the_left() {
    let at = position("P2", O1, &TRUMP_SET, &["2-2", "2-0", "2-1", "6-1", "6-6"]);
    assert_eq!(allows(C22, &at, "trump-in"), tiles(&["5-2", "4-2"]));
}

#[test]
fn c22_from_the_last_seat_trumps_only_on_count() {
    // The setters' seats exchanged: the last setter holds the trumps. With the
    // 6-4 on the trick he trumps; with no count on it, he does not. The gate
    // fires on both.
    let at = position("P2", O2, &TRUMP_SET, &["2-2", "6-0", "2-1", "2-0", "6-6", "6-4", "6-3"]);
    assert_eq!(allows(C22, &at, "trump-in-on-count"), tiles(&["5-2", "4-2"]));
    let at = position("P2", O2, &TRUMP_SET, &["2-2", "6-0", "2-1", "2-0", "6-6", "6-3", "6-4"]);
    assert_eq!(allows(C22, &at, "trump-in-on-count"), tiles(&["5-2", "4-2"]));
    let at = position("P2", O2, &TRUMP_SET, &["2-2", "6-0", "2-1", "2-0", "5-5", "5-3", "5-1"]);
    assert_eq!(allows(C22, &at, "trump-in-on-count"), tiles(&["5-2", "4-2"]));
    let at = position("P2", O2, &TRUMP_SET, &["2-2", "6-0", "2-1", "2-0", "6-6", "6-1", "6-3"]);
    assert_eq!(fires(C22, &at), None);
}

#[test]
fn c22_needs_the_trump_set() {
    // Without the trump-set trick the encoding does not fire; the gate does.
    let hand = ["5-2", "4-2", "5-5", "3-3", "1-0", "4-0", "3-1"];
    let at = position("P2", O1, &hand, &["6-6"]);
    assert_eq!(fires(C22, &at), None);
    // Both setters followed the trump lead: no trump set.
    let at = position("P2", O1, &TRUMP_SET, &["2-2", "2-0", "2-1", "3-2", "6-6"]);
    assert_eq!(fires(C22, &at), None);
}

// C23: the partner leads doubles.

#[test]
fn c23_leads_the_double_in_hand_30() {
    // W42-p102-c, Hand 30 T2: the partner, on lead after the 6-6, leads 0-0.
    let at = book_at("hand-30", 4, P);
    assert_eq!(allows(C23, &at, "lead-a-double"), tiles(&["0-0"]));
}

#[test]
fn c23_does_not_fire_out_of_doubles() {
    // W42-p103-d, Hand 30 T3: the partner has no double left.
    let at = book_at("hand-30", 8, P);
    assert_eq!(fires(C23, &at), None);
}
