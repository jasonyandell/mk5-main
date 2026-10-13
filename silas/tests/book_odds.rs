//! The book's odds (Roberson, *Winning 42*), exactly: every probability it
//! prints, as a Silas query in `examples/book/odds/`, summarized over the
//! uniform deal of the tiles the bidder cannot see. Each claim keeps three
//! numbers: the printed one, the book's model (each missing tile equally
//! likely in each of the three other hands, 3^m placements), and the exact
//! one (three hands of seven from 21). The table is
//! `examples/book/odds/README.md`.
//!
//! The book states most claims for a shape ("four trumps, missing the
//! second-highest"), not a hand; each shape is checked on two hands with
//! different trumps and other tiles, and must give the same answer. Each
//! exact answer is also derived independently here, in plain Rust, by
//! weighting each placement of the marked tiles by its completions.
//! Questions over a whole deal, with no hand known, are beyond a Silas frame
//! (`GAPS.md`, deal view); they are enumerated here over all C(28,7) hands.
mod common;

use std::collections::BTreeMap;

use common::*;
use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{One, ToPrimitive, Zero};
use silas::*;

macro_rules! odds {
    ($name:literal) => {
        include_str!(concat!("../examples/book/odds/", $name, ".silas"))
    };
}

const GUARDED_4T: &str = odds!("c16-4t-guarded-second");
const WORKS_4T: &str = odds!("c16-4t-works");
const TWO_OF_THREE_4T: &str = odds!("c16-4t-two-of-three");
const BOTH_HIGH_4T: &str = odds!("c16-4t-both-high");
const FOUR_OF_A_SUIT: &str = odds!("c16-four-of-a-suit");
const GUARDED_3T: &str = odds!("c16-3t-guarded-second");
const PARTNER_SECOND_3T: &str = odds!("c16-3t-partner-second");
const OPPONENT_THREE_3T: &str = odds!("c16-3t-opponent-three");
const SPREAD_2T: &str = odds!("c16-2t-spread");
const PARTNER_THREE_2T: &str = odds!("c16-2t-partner-three");
const FAVORABLE_2T: &str = odds!("c16-2t-favorable");
const SUITS_REPRESENTED: &str = odds!("c16-suits-represented");
const DOUBLES: &str = odds!("c16-doubles");
const TWO_TO_FOUR_DOUBLES: &str = odds!("c16-two-to-four-doubles");
const MODAL_HAND: &str = odds!("c16-modal-hand");
const PARTNER_TWO_DOUBLES: &str = odds!("c16-partner-two-doubles");
const PARTNER_DOUBLE_FIVE: &str = odds!("c02-partner-double-five");
const OPPONENT_DOUBLE_FIVE: &str = odds!("c02-opponent-double-five");
const DOUBLE_AHEAD: &str = odds!("c03-double-ahead-forces-count");
const DOUBLE_AHEAD_LATER: &str = odds!("c03-double-ahead-later");
const OFF_FIRST: &str = odds!("c03-off-first-partner");
const SETTER_PARTNER: &str = odds!("c05-setter-partner-holds-winner");
const STRAIGHT_OFF: &str = odds!("c07-straight-off-double");
const PARTNER_FIVE_BLANK: &str = odds!("c09-partner-five-blank");
const DOUBLES_TRUMP_VOID: &str = odds!("c09-doubles-trump-void");
const OUTTRUMPED: &str = odds!("c12-two-trump-outtrumped");
const ONLY_TREY: &str = odds!("c12-only-trey");

type Hand = (&'static str, [&'static str; 7]);

/// Four trumps with the double, not the second-highest. The first is the
/// four-fives 84 (p. 113).
const FOUR_NO_SECOND: [Hand; 2] = [
    ("P5", ["5-5", "5-4", "5-2", "5-0", "4-4", "4-2", "2-2"]),
    ("P3", ["3-3", "3-2", "3-1", "3-0", "6-6", "5-4", "1-1"]),
];
/// Four trumps with the double, neither the second nor the third highest.
const FOUR_NO_SECOND_THIRD: [Hand; 2] = [
    ("P5", ["5-5", "5-3", "5-2", "5-0", "6-6", "4-4", "1-0"]),
    ("P2", ["2-2", "4-2", "2-1", "2-0", "6-6", "5-5", "3-0"]),
];
/// Three trumps with the double, not the second-highest.
const THREE_NO_SECOND: [Hand; 2] = [
    ("P6", ["6-6", "6-4", "6-0", "5-5", "4-4", "3-1", "2-0"]),
    ("P1", ["1-1", "5-1", "1-0", "6-6", "5-5", "4-3", "2-0"]),
];
/// Only the two top trumps. The first is Hand 26 (p. 91).
const TWO_TOP: [Hand; 2] = [
    ("P4", ["3-3", "1-1", "4-4", "6-4", "5-5", "2-2", "5-2"]),
    ("P6", ["6-6", "6-5", "5-5", "4-4", "3-2", "1-0", "0-0"]),
];
/// The double-deuce ahead of the deuce-blank, four deuces out and not trump.
const DEUCE_AHEAD: [Hand; 2] = [
    ("P6", ["6-6", "6-5", "6-4", "6-2", "2-2", "2-0", "1-1"]),
    ("P5", ["5-5", "5-4", "5-3", "5-2", "2-2", "2-0", "1-1"]),
];
/// An 84 with one straight off: 3-1 without 3-3, 4-2 without 4-4.
const STRAIGHT_OFF_HANDS: [Hand; 2] = [
    ("P5", ["5-5", "5-4", "5-3", "5-2", "6-6", "4-4", "3-1"]),
    ("P6", ["6-6", "6-5", "6-4", "6-3", "5-5", "1-1", "4-2"]),
];
const HAND_3: Hand = ("P4", ["5-2", "4-1", "6-6", "6-4", "4-2", "4-3", "4-4"]);
const HAND_23: Hand = ("P6", ["6-1", "1-1", "6-6", "0-0", "3-3", "5-4", "6-5"]);
const HAND_25: Hand = ("DT", ["5-0", "5-5", "2-2", "4-4", "1-1", "2-1", "3-3"]);

// ---------------------------------------------------------------------------
// Frames, beliefs, and the two placement models.

fn root((decl, hand): Hand) -> Frame {
    opening(&physics(decl), tiles(&hand))
}

/// The tiles following `context` that the viewer cannot see.
fn missing(frame: &Frame, context: Context) -> DominoSet {
    frame.physics().incidence(context).intersection(frame.kernel().pool())
}

fn trumps(frame: &Frame) -> DominoSet {
    missing(frame, Context::Called)
}

fn big_budget() -> Budget {
    Budget::new(1_000_000_000)
}

fn resolved(frame: Frame, tiles: DominoSet) -> Belief {
    let belief = Belief::uniform_resolving(frame, tiles, 100_000).unwrap();
    assert_eq!(belief.authority(), &Authority::exact());
    belief
}

fn probability(query: &str, belief: &Belief) -> BigRational {
    exact(compile(query).summarize(belief, &mut big_budget()).unwrap().event_probability())
}

/// The query's exact probability at a hand, resolving `marked` there.
fn at(query: &str, hand: Hand, marked: fn(&Frame) -> DominoSet) -> BigRational {
    let frame = root(hand);
    let tiles = marked(&frame);
    probability(query, &resolved(frame, tiles))
}

/// The query's answer on a shape: equal on both representative hands.
fn on_shape(query: &str, shape: [Hand; 2], marked: fn(&Frame) -> DominoSet) -> BigRational {
    let [a, b] = shape.map(|hand| at(query, hand, marked));
    assert_eq!(a, b, "the two representatives of the shape disagree");
    a
}

/// A placement puts each marked tile with one of the three other hands,
/// named from the bidder: left, partner, right. Marked tiles are ordered
/// from the highest down where the claim cares.
const LEFT: usize = 0;
const PARTNER: usize = 1;
const RIGHT: usize = 2;

fn opponent(seat: usize) -> bool {
    seat != PARTNER
}

fn placements(m: u32) -> impl Iterator<Item = Vec<usize>> {
    (0..3usize.pow(m)).map(move |mut i| {
        (0..m)
            .map(|_| {
                let seat = i % 3;
                i /= 3;
                seat
            })
            .collect()
    })
}

fn held(placement: &[usize], seat: usize) -> usize {
    placement.iter().filter(|s| **s == seat).count()
}

fn big(n: u128) -> BigRational {
    BigRational::from_integer(BigInt::from(n))
}

/// The book's model: the 3^m placements equally likely.
fn book_model(m: u32, event: impl Fn(&[usize]) -> bool) -> BigRational {
    let hits = placements(m).filter(|p| event(p)).count() as u128;
    big(hits) / big(3u128.pow(m))
}

fn factorial(n: usize) -> u128 {
    (1..=n as u128).product()
}

/// Ways to deal `n` tiles into hands of the given sizes.
fn multinomial(n: usize, sizes: [usize; 3]) -> u128 {
    sizes.iter().fold(factorial(n), |acc, k| acc / factorial(*k))
}

/// The real deal, independently of Silas: hidden hands of `capacity`, each
/// placement weighted by the deals of the other tiles around it.
fn dealt_with(capacity: [usize; 3], m: u32, event: impl Fn(&[usize]) -> bool) -> BigRational {
    let n: usize = capacity.iter().sum();
    let mut hits = 0u128;
    for p in placements(m) {
        let room = [0, 1, 2].map(|s| capacity[s].checked_sub(held(&p, s)));
        if let (true, [Some(a), Some(b), Some(c)]) = (event(&p), room) {
            hits += multinomial(n - m as usize, [a, b, c]);
        }
    }
    big(hits) / big(multinomial(n, capacity))
}

fn dealt(m: u32, event: impl Fn(&[usize]) -> bool) -> BigRational {
    dealt_with([7, 7, 7], m, event)
}

/// Nearest whole percent, as the book prints.
fn percent(x: &BigRational) -> i64 {
    (x * big(100)).round().to_integer().to_i64().unwrap()
}

fn count(n: i64) -> BigRational {
    BigRational::from_integer(BigInt::from(n))
}

// Events on placements.

/// An opponent holds the top marked tile and another beside it.
fn guarded_top(p: &[usize]) -> bool {
    opponent(p[0]) && held(p, p[0]) >= 2
}

fn opponent_holds(p: &[usize], at_least: usize) -> bool {
    held(p, LEFT) >= at_least || held(p, RIGHT) >= at_least
}

// ---------------------------------------------------------------------------
// Chapter 16: four trumps (shot-0232, shot-0233).

#[test]
fn four_trumps_twenty_seven_placements() {
    for hand in FOUR_NO_SECOND.into_iter().chain(FOUR_NO_SECOND_THIRD) {
        let frame = root(hand);
        let tiles = trumps(&frame);
        assert_eq!(resolved(frame, tiles).len(), 27);
    }
}

#[test]
fn four_trumps_missing_the_second_guarded_is_33_over_95_not_10_over_27() {
    let exact = on_shape(GUARDED_4T, FOUR_NO_SECOND, trumps);
    let book = book_model(3, guarded_top);
    assert_eq!(book, r(10, 27));
    assert_eq!(percent(&book), 37);
    assert_eq!(exact, r(33, 95));
    assert_eq!(exact, dealt(3, guarded_top));
    // The four-fives 84 hand: the same number as the six-five query.
    assert_eq!(at(SIX_FIVE, FOUR_NO_SECOND[0], trumps), exact);
    assert_eq!(percent(&exact), 35);
}

#[test]
fn four_trumps_lead_the_double_works_62_over_95() {
    let exact = on_shape(WORKS_4T, FOUR_NO_SECOND, trumps);
    let book = book_model(3, |p| !guarded_top(p));
    assert_eq!((book.clone(), percent(&book)), (r(17, 27), 63));
    assert_eq!(exact, r(62, 95));
    assert_eq!(exact, BigRational::one() - r(33, 95));
    assert_eq!(percent(&exact), 65);
}

#[test]
fn four_trumps_missing_second_and_third_is_47_over_95_not_14_over_27() {
    let event = |p: &[usize]| opponent_holds(p, 2);
    let exact = on_shape(TWO_OF_THREE_4T, FOUR_NO_SECOND_THIRD, trumps);
    let book = book_model(3, event);
    assert_eq!((book.clone(), percent(&book)), (r(14, 27), 52));
    assert_eq!(exact, r(47, 95));
    assert_eq!(exact, dealt(3, event));
    // Under one half: the book's "barely against you" is, exactly, barely for.
    assert!(exact < r(1, 2));
}

#[test]
fn four_trumps_both_high_with_one_opponent_six_of_fourteen() {
    let both = |p: &[usize]| opponent(p[0]) && p[1] == p[0];
    let exact = on_shape(BOTH_HIGH_4T, FOUR_NO_SECOND_THIRD, trumps);
    assert_eq!(book_model(3, both), r(6, 27));
    assert_eq!(exact, r(1, 5));
    assert_eq!(exact, dealt(3, both));
    // Read inside the bad placements: 6/14 in the model, 19/47 exactly.
    assert_eq!(book_model(3, both) / book_model(3, |p| opponent_holds(p, 2)), r(3, 7));
    for hand in FOUR_NO_SECOND_THIRD {
        let frame = root(hand);
        let tiles = trumps(&frame);
        let belief = resolved(frame, tiles);
        let (bad, evidence) = belief.condition(&compile(TWO_OF_THREE_4T), "two-of-three", &mut big_budget()).unwrap();
        assert_eq!(common::exact(evidence), r(47, 95));
        assert_eq!(probability(BOTH_HIGH_4T, &bad), r(19, 47));
    }
}

// ---------------------------------------------------------------------------
// Chapter 16: three trumps (shot-0233).

#[test]
fn three_trumps_eighty_one_placements() {
    for hand in THREE_NO_SECOND {
        let frame = root(hand);
        let tiles = trumps(&frame);
        assert_eq!(resolved(frame, tiles).len(), 81);
    }
}

#[test]
fn three_trumps_missing_the_second_guarded_is_388_over_855_not_38_over_81() {
    let exact = on_shape(GUARDED_3T, THREE_NO_SECOND, trumps);
    let book = book_model(4, guarded_top);
    assert_eq!((book.clone(), percent(&book)), (r(38, 81), 47));
    assert_eq!(exact, r(388, 855));
    assert_eq!(exact, dealt(4, guarded_top));
    assert_eq!(percent(&exact), 45);
}

#[test]
fn three_trumps_partner_holds_the_second_one_third_not_21_over_81() {
    let event = |p: &[usize]| p[0] == PARTNER;
    let exact = on_shape(PARTNER_SECOND_3T, THREE_NO_SECOND, trumps);
    let book = book_model(4, event);
    assert_eq!(book, r(27, 81));
    assert_eq!(exact, r(1, 3));
    assert_eq!(exact, dealt(4, event));
    // The printed 21 of 81 is no simple event's count: the 81 placements
    // split 38 (an opponent guards it), 27 (partner holds it), 16 (an
    // opponent holds it alone).
    let alone = |p: &[usize]| opponent(p[0]) && held(p, p[0]) == 1;
    assert_eq!(book_model(4, alone), r(16, 81));
    assert_eq!(book_model(4, guarded_top) + book + book_model(4, alone), BigRational::one());
    assert_ne!(percent(&exact), 26);
}

#[test]
fn three_trumps_an_opponent_holds_three_is_10_over_57_not_18_over_81() {
    let event = |p: &[usize]| opponent_holds(p, 3);
    let exact = on_shape(OPPONENT_THREE_3T, THREE_NO_SECOND, trumps);
    assert_eq!(book_model(4, event), r(18, 81));
    assert_eq!(exact, r(10, 57));
    assert_eq!(exact, dealt(4, event));
}

// ---------------------------------------------------------------------------
// Chapter 16 (shot-0233) and Chapter 12 (shot-0121, shot-0123): two top trumps.

#[test]
fn two_top_trumps_have_243_placements_not_225() {
    for hand in TWO_TOP {
        let frame = root(hand);
        let tiles = trumps(&frame);
        assert_eq!(tiles.len(), 5);
        assert_eq!(resolved(frame, tiles).len(), 243);
    }
    assert_eq!(placements(5).count(), 243);
}

#[test]
fn two_top_trumps_spread_is_147_over_323() {
    let event = |p: &[usize]| (0..3).all(|s| held(p, s) <= 2);
    let exact = on_shape(SPREAD_2T, TWO_TOP, trumps);
    let book = book_model(5, event);
    // The 90 is right; the printed 0.40 is 90 over the wrong 225.
    assert_eq!(book, count(90) / count(243));
    assert_eq!(percent(&(count(90) / count(225))), 40);
    assert_eq!(percent(&book), 37);
    assert_eq!(exact, r(147, 323));
    assert_eq!(exact, dealt(5, event));
}

#[test]
fn two_top_trumps_partner_holds_three_is_176_over_969() {
    let event = |p: &[usize]| held(p, PARTNER) >= 3;
    let exact = on_shape(PARTNER_THREE_2T, TWO_TOP, trumps);
    let book = book_model(5, event);
    // The model counts 51 placements, not the printed 45.
    assert_eq!(book, count(51) / count(243));
    assert_eq!(percent(&book), 21);
    assert_eq!(exact, r(176, 969));
    assert_eq!(exact, dealt(5, event));
}

#[test]
fn two_top_trumps_favorable_is_617_over_969_in_both_chapters() {
    let event = |p: &[usize]| !opponent_holds(p, 3);
    let exact = on_shape(FAVORABLE_2T, TWO_TOP, trumps);
    let book = book_model(5, event);
    assert_eq!((book.clone(), percent(&book)), (r(47, 81), 58));
    assert_eq!(exact, r(617, 969));
    assert_eq!(exact, dealt(5, event));
    // Favorable is spread or partner-three, never both.
    assert_eq!(exact, r(147, 323) + r(176, 969));
    assert_eq!(percent(&exact), 64);
}

#[test]
fn hand_26_an_opponent_outtrumps_the_bidder_352_over_969() {
    let exact = at(OUTTRUMPED, TWO_TOP[0], trumps);
    let book = book_model(5, |p| opponent_holds(p, 3));
    assert_eq!((book.clone(), percent(&book)), (r(34, 81), 42));
    assert_eq!(exact, r(352, 969));
    assert_eq!(exact, BigRational::one() - r(617, 969));
    assert_eq!(percent(&exact), 36);
}

// ---------------------------------------------------------------------------
// Chapter 16: the whole deal (shot-0233, shot-0234). Silas fixes the viewer's
// hand, so these are enumerated here; each query is checked hand by hand.

struct Deal {
    hands: u64,
    suits: [u64; 8],
    doubles: [u64; 8],
    four_or_more: u64,
    longest_four: u64,
    shape: BTreeMap<(usize, usize), u64>,
}

fn suit_masks() -> [u32; 7] {
    Pip::ALL.map(|p| Domino::ALL.iter().filter(|d| d.has(p)).fold(0, |m, d| m | DominoSet::single(*d).bits()))
}

fn double_mask() -> u32 {
    Domino::ALL.iter().filter(|d| d.is_double()).fold(0, |m, d| m | DominoSet::single(*d).bits())
}

/// Every seven-tile hand, in increasing bit order (Gosper's hack).
fn every_hand() -> impl Iterator<Item = u32> {
    std::iter::successors(Some((1u32 << 7) - 1), |&h| {
        let low = h & h.wrapping_neg();
        let ripple = h + low;
        let next = (((ripple ^ h) >> 2) / low) | ripple;
        (next < 1 << 28).then_some(next)
    })
}

fn deal() -> Deal {
    let masks = suit_masks();
    let doubles = double_mask();
    let mut deal =
        Deal { hands: 0, suits: [0; 8], doubles: [0; 8], four_or_more: 0, longest_four: 0, shape: BTreeMap::new() };
    for hand in every_hand() {
        let lengths = masks.map(|m| (hand & m).count_ones());
        let present = lengths.iter().filter(|n| **n > 0).count();
        let pairs = (hand & doubles).count_ones() as usize;
        let longest = *lengths.iter().max().unwrap();
        deal.hands += 1;
        deal.suits[present] += 1;
        deal.doubles[pairs] += 1;
        deal.four_or_more += u64::from(longest >= 4);
        deal.longest_four += u64::from(longest == 4);
        *deal.shape.entry((pairs, 7 - present)).or_default() += 1;
    }
    deal
}

fn share(n: u64, of: u64) -> BigRational {
    count(n as i64) / count(of as i64)
}

#[test]
fn the_whole_deal() {
    let deal = deal();
    // shot-0234: the number of hands.
    assert_eq!(deal.hands, 1_184_040);
    let of = |n: u64| share(n, deal.hands);

    // shot-0234: suits present. Printed 41, 48, 10, 1.
    assert_eq!(deal.suits[4..], [4200, 122_535, 556_290, 501_015]);
    assert_eq!(deal.suits[..4].iter().sum::<u64>(), 0);
    assert_eq!([7, 6, 5, 4].map(|k| percent(&of(deal.suits[k]))), [42, 47, 10, 0]);

    // shot-0234: doubles held. Printed 10, 32, 36, 18, 4, under 1/2, 1/50.
    let expected = [116_280, 379_848, 427_329, 209_475, 46_550, 4410, 147, 1];
    assert_eq!(deal.doubles, expected);
    for (k, n) in expected.iter().enumerate() {
        let (c7, c21) = (choose(7, k), choose(21, 7 - k));
        assert_eq!(u128::from(*n), c7 * c21);
    }
    assert_eq!([0, 1, 2, 3, 4].map(|k| percent(&of(deal.doubles[k]))), [10, 32, 36, 18, 4]);
    assert!(of(deal.doubles[5]) < r(1, 200));
    // Six doubles: 147 hands, about 1/80 of a percent, not 1/50.
    assert!(of(deal.doubles[6]) < r(1, 7500) && of(deal.doubles[6]) > r(1, 8100));

    // shot-0234: two, three or four doubles; "at least 36 percent" is true
    // but is the two-double bucket alone.
    let two_to_four = of(deal.doubles[2] + deal.doubles[3] + deal.doubles[4]);
    assert_eq!(two_to_four, r(341_677, 592_020));
    assert_eq!(percent(&two_to_four), 58);
    let two_or_more = BigRational::one() - of(deal.doubles[0] + deal.doubles[1]);
    assert_eq!(percent(&two_or_more), 58);

    // shot-0234: the commonest shape is two doubles and one void suit.
    let (mode, n) = deal.shape.iter().max_by_key(|(_, n)| **n).unwrap();
    assert_eq!((*mode, *n), ((2, 1), 213_255));

    // shot-0233: four of one suit, printed 28. Four or more: 0.295; a
    // longest suit of exactly four: 0.268.
    assert_eq!((deal.four_or_more, deal.longest_four), (349_356, 317_450));
    assert_eq!((percent(&of(deal.four_or_more)), percent(&of(deal.longest_four))), (30, 27));
}

fn choose(n: usize, k: usize) -> u128 {
    factorial(n) / (factorial(k) * factorial(n - k))
}

/// The deal-view queries, read on known hands under no trump, agree with
/// the enumeration's predicates.
#[test]
fn the_deal_view_queries_agree_with_the_enumeration_hand_by_hand() {
    let masks = suit_masks();
    let doubles = double_mask();
    let queries = [FOUR_OF_A_SUIT, SUITS_REPRESENTED, DOUBLES, TWO_TO_FOUR_DOUBLES, MODAL_HAND].map(compile);
    let (mut seen, mut hands) = ([0; 5], 0);
    for bits in every_hand().step_by(4999) {
        hands += 1;
        let hand = DominoSet::from_bits(bits).unwrap();
        let frame = opening(&physics("NT"), hand);
        let belief = resolved(frame, DominoSet::EMPTY);
        let lengths = masks.map(|m| (bits & m).count_ones());
        let voids = lengths.iter().filter(|n| **n == 0).count();
        let pairs = (bits & doubles).count_ones();
        let truth = [
            lengths.iter().any(|n| *n >= 4),
            voids == 1,
            pairs == 2,
            (2..=4).contains(&pairs),
            pairs == 2 && voids == 1,
        ];
        for (i, query) in queries.iter().enumerate() {
            let p = exact(query.summarize(&belief, &mut big_budget()).unwrap().event_probability());
            assert_eq!(p, if truth[i] { BigRational::one() } else { BigRational::zero() }, "query {i}");
            seen[i] += usize::from(truth[i]);
        }
    }
    // Every query met hands on both sides.
    assert!(seen.iter().all(|n| *n > 0 && *n < hands));
}

#[test]
fn partner_holds_two_doubles_given_the_bidders_own() {
    fn doubles(frame: &Frame) -> DominoSet {
        frame.kernel().pool().iter().filter(|d| d.is_double()).collect()
    }
    let event = |p: &[usize]| held(p, PARTNER) >= 2;
    let two: [Hand; 2] = [
        ("P6", ["6-6", "5-5", "6-4", "5-3", "4-1", "3-2", "2-0"]),
        ("P6", ["1-1", "0-0", "6-5", "5-4", "4-3", "3-1", "2-1"]),
    ];
    let three: [Hand; 2] = [
        ("P6", ["6-6", "5-5", "4-4", "6-1", "5-3", "3-2", "2-0"]),
        ("P3", ["3-3", "2-2", "0-0", "6-5", "5-4", "4-1", "1-0"]),
    ];
    let given_two = on_shape(PARTNER_TWO_DOUBLES, two, doubles);
    assert_eq!(given_two, r(180, 323));
    assert_eq!(given_two, dealt(5, event));
    assert_eq!(book_model(5, event), r(131, 243));
    let given_three = on_shape(PARTNER_TWO_DOUBLES, three, doubles);
    assert_eq!(given_three, r(116, 285));
    assert_eq!(given_three, dealt(4, event));
    assert_eq!(book_model(4, event), r(11, 27));
    // The study's 0.557 and 0.407, to three places.
    assert_eq!(((given_two * big(1000)).round(), (given_three * big(1000)).round()), (big(557), big(407)));
}

// ---------------------------------------------------------------------------
// The earlier chapters.

#[test]
fn hand_3_the_double_five() {
    // shot-0026 and shot-0047 (Hand 14 is Hand 3's tiles): the partner; shot-0027: an opponent.
    let belief = resolved(root(HAND_3), tiles(&["5-5"]));
    assert_eq!(probability(PARTNER_DOUBLE_FIVE, &belief), r(1, 3));
    assert_eq!(probability(OPPONENT_DOUBLE_FIVE, &belief), r(2, 3));
    assert_eq!(book_model(1, |p| p[0] == PARTNER), r(1, 3));
    assert_eq!(dealt(1, |p| p[0] == PARTNER), r(1, 3));
}

fn deuces(frame: &Frame) -> DominoSet {
    missing(frame, Context::Natural(Pip::new(2).unwrap()))
}

#[test]
fn the_double_deuce_ahead_brings_down_the_deuce_trey_467_over_855() {
    // shot-0042. The 3-2 is marked first, then the other three deuces out.
    let event = |p: &[usize]| p[0] == PARTNER || held(p, p[0]) == 1;
    for hand in DEUCE_AHEAD {
        assert_eq!(deuces(&root(hand)).len(), 4);
    }
    let exact = on_shape(DOUBLE_AHEAD, DEUCE_AHEAD, deuces);
    let book = book_model(4, event);
    assert_eq!((book.clone(), percent(&book)), (r(43, 81), 53));
    assert_eq!(exact, r(467, 855));
    assert_eq!(exact, dealt(4, event));
    assert_eq!(percent(&exact), 55);
    // With five deuces out the reading gives 113/243 in the model.
    assert_eq!(book_model(5, event), r(113, 243));
    // The 5 percent is a two-trick play-out: a gap, no query.
    assert!(!DOUBLE_AHEAD_LATER.contains("(query"));
    assert!(DOUBLE_AHEAD_LATER.contains("GAP play-out"));
}

#[test]
fn leading_the_off_first_the_partner_holds_the_top_deuce_one_third() {
    // shot-0042, the 33 percent.
    assert_eq!(on_shape(OFF_FIRST, DEUCE_AHEAD, deuces), r(1, 3));
    assert_eq!(book_model(4, |p| p[0] == PARTNER), r(1, 3));
}

#[test]
fn the_setters_hold_the_winner_two_to_one_or_seven_in_twenty() {
    // shot-0066. Reading A, the bidder's view of one unseen tile: 2/3 (the
    // next test's relation). Reading B, the setter who plays second to the
    // bidder's off and cannot follow: his partner holds it.
    let setters: [(&str, [&str; 7], &str, &str); 2] = [
        ("P5", ["6-6", "6-5", "4-4", "4-1", "3-3", "3-1", "1-0"], "2-0", "2-2"),
        ("P6", ["5-5", "5-4", "4-4", "2-1", "1-0", "0-0", "6-2"], "3-1", "3-3"),
    ];
    let not_bidders = "(query (roles (chair b) (context q) (domino d)) (out) (pattern (declarer b) (led-context q) (double d) (in d q) (not (holds b d))))";
    for (decl, hand, lead, winner) in setters {
        let contract = contract(&physics(decl), Seat::S0, 30);
        let frame = framed_history(&contract, Seat::S1, tiles(&hand), &[(Seat::S0, tile(lead))]).unwrap();
        let belief = resolved(frame, tiles(&[winner]));
        assert_eq!(probability(SETTER_PARTNER, &belief), r(7, 20));
        let (known, evidence) = belief.condition(&compile(not_bidders), "not-the-bidders", &mut big_budget()).unwrap();
        assert_eq!(exact(evidence), r(14, 20));
        assert_eq!(probability(SETTER_PARTNER, &known), r(1, 2));
    }
    // Hidden hands here: the bidder's six, his partner's seven, the
    // setter's partner's seven (index 2).
    assert_eq!(dealt_with([6, 7, 7], 1, |p| p[0] == 2), r(7, 20));
}

#[test]
fn a_straight_off_double_is_with_an_opponent_two_thirds() {
    // shot-0089 and shot-0101.
    // The query ranges over every double, so every unseen double is
    // resolved, not only the off's.
    fn unseen_doubles(frame: &Frame) -> DominoSet {
        frame.kernel().pool().iter().filter(|d| d.is_double()).collect()
    }
    assert_eq!(on_shape(STRAIGHT_OFF, STRAIGHT_OFF_HANDS, unseen_doubles), r(2, 3));
    assert_eq!(book_model(1, |p| opponent(p[0])), r(2, 3));
}

#[test]
fn hand_23_the_partner_holds_the_five_blank_one_third() {
    // shot-0104.
    let frame = root(HAND_23);
    assert_eq!(probability(PARTNER_FIVE_BLANK, &resolved(frame, tiles(&["5-0"]))), r(1, 3));
}

#[test]
fn hand_25_an_opponent_is_void_in_doubles_23_over_30() {
    // shot-0109. Only 6-6 and 0-0 are out; the event fails only when each
    // opponent holds one.
    let event = |p: &[usize]| !(p.contains(&LEFT) && p.contains(&RIGHT));
    let frame = root(HAND_25);
    let marked = trumps(&frame);
    assert_eq!(marked, tiles(&["6-6", "0-0"]));
    let exact = probability(DOUBLES_TRUMP_VOID, &resolved(frame, marked));
    let book = book_model(2, event);
    assert_eq!((book.clone(), percent(&book)), (r(7, 9), 78));
    assert_eq!(exact, r(23, 30));
    assert_eq!(exact, dealt(2, event));
    assert_eq!(percent(&exact), 77);
}

#[test]
fn hand_26_the_five_trey_was_his_only_trey_7_over_33() {
    // shot-0124, on a record of ours that fits the book's account.
    let plays = [
        (Seat::S0, "4-4"),
        (Seat::S1, "4-0"),
        (Seat::S2, "4-1"),
        (Seat::S3, "4-3"),
        (Seat::S0, "6-4"),
        (Seat::S1, "5-4"),
        (Seat::S2, "6-2"),
        (Seat::S3, "1-0"),
        (Seat::S0, "5-5"),
        (Seat::S1, "5-3"),
        (Seat::S2, "6-5"),
        (Seat::S3, "5-1"),
    ]
    .map(|(s, d)| (s, tile(d)))
    .to_vec();
    let contract = contract(&physics("P4"), Seat::S0, 35);
    let hand = tiles(&["3-3", "1-1", "2-2", "5-2"]);
    let frame = framed_history(&contract, Seat::S0, hand, &plays).unwrap();
    // The left setter holds the 4-2 and three of the other eleven.
    assert_eq!(frame.count_worlds(), 165 * 70);
    let belief = Belief::uniform(frame, 100_000).unwrap();
    let exact = probability(ONLY_TREY, &belief);
    assert_eq!(exact, r(7, 33));
    assert_eq!(exact, count(choose(7, 3) as i64) / count(choose(11, 3) as i64));
    // The book's model, each of the four live treys anywhere: (2/3)^4.
    assert_eq!(book_model(4, |p| p.iter().all(|s| *s != LEFT)), r(16, 81));
    assert!(exact < r(1, 3));
}
