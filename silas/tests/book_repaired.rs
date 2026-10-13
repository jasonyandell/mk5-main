//! The repaired book (`examples/book/repaired/`): two of the book's rules with
//! the condition Zeb's reasons show they omit. Each counterexample deal is
//! checked double-dummy (the solver sees all four hands) and, where the deal
//! is decided in a trick or two, play by play: every play of the bidder's side
//! there is forced, so the setters' count is what they can force. Each
//! condition is read in the deal's own world, and as the bidder can read it,
//! as an exact probability over the worlds he cannot tell apart.
mod common;

use common::fixture::{Position, book};
use common::rules::*;
use common::*;
use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{ToPrimitive, Zero};
use silas::*;

const DOUBLE_UP_LOW: &str = include_str!("../examples/book/repaired/double-up-lead-low.silas");
const ONE_OFF: &str = include_str!("../examples/book/repaired/one-off-lead-trump.silas");
const PARTNER_RUFFS: &str = include_str!("../examples/book/repaired/one-off-partner-ruffs.silas");
const FIRST_LEAD: &str = include_str!("../examples/book/first-lead.silas");

/// A deal with the bidder in S0, the hands in seat order (S1 is the left
/// setter, S2 the partner, S3 the right setter), at a bid of 30.
fn deal(decl: &str, hands: [&str; 4]) -> OpenPosition {
    let hands = hands.map(|h| tiles(&h.split(' ').collect::<Vec<_>>()));
    OpenPosition::deal(contract(&physics(decl), Seat::S0, 30), hands).unwrap()
}

/// Double-dummy: does the bidder's side make after the opening lead `lead`?
fn makes(position: &OpenPosition, lead: &str) -> bool {
    let values = Solver::new(position.physics()).solve(position, Objective::Make).unwrap();
    values.value(tile(lead)).unwrap() == 1
}

/// Play `line` from `position`, each play checked legal; where `forced` marks
/// a play, it must be the seat's only legal tile.
fn play(position: &OpenPosition, line: &[(&str, bool)]) -> OpenPosition {
    line.iter().fold(position.clone(), |at, (d, forced)| {
        if *forced {
            assert_eq!(at.legal(), tiles(&[d]), "{d} is forced");
        }
        at.play(tile(d)).unwrap()
    })
}

/// The bidder's frame before the first lead.
fn bidders_frame(position: &OpenPosition) -> Frame {
    opening(position.physics(), position.hand(Seat::S0))
}

/// The query's truth in the deal's own world.
fn holds_in(query: &str, position: &OpenPosition) -> bool {
    let frame = bidders_frame(position);
    let world = frame.kernel().world([Seat::S1, Seat::S2, Seat::S3].map(|s| position.hand(s)));
    let one = Belief::from_weights(frame, "the deal", given("test: the dealt hands"), [(world, w(1, 1))], 1).unwrap();
    let p = compile(query).summarize(&one, &mut budget()).unwrap().event_probability();
    p.value() == &r(1, 1)
}

/// The query's exact probability over the bidder's worlds before the first
/// lead, resolving only the tiles it reads.
fn probability(query: &str, frame: Frame) -> BigRational {
    let compiled = compile(query);
    let belief = Belief::uniform_for(frame, &compiled, 100_000, &mut Budget::new(1_000_000_000)).unwrap();
    exact(compiled.summarize(&belief, &mut Budget::new(1_000_000_000)).unwrap().event_probability())
}

fn choose(n: u64, k: u64) -> BigInt {
    (0..k).fold(BigInt::from(1), |acc, i| acc * BigInt::from(n - i) / BigInt::from(i + 1))
}

/// The partner's seven of the 21 unseen tiles miss `k` given tiles and meet
/// `m` others.
fn misses_and_meets(k: u64, m: u64) -> BigRational {
    BigRational::new(choose(21 - k, 7) - choose(21 - k - m, 7), choose(21, 7))
}

#[test]
fn the_repaired_rules_parse_round_trip_name_no_chair_and_compile() {
    let policy: PolicyProgram = ONE_OFF.parse().unwrap();
    assert_eq!(policy.to_string().parse::<PolicyProgram>().unwrap(), policy);
    for rules in reference::Straight42::ALL {
        policy.compile(&Registry::standard(), &rules.physics()).unwrap();
    }
    for source in [DOUBLE_UP_LOW, ONE_OFF, PARTNER_RUFFS] {
        for chair in ["S0", "S1", "S2", "S3"] {
            assert!(!source.contains(chair), "a repaired rule names no chair");
        }
    }
    for query in [DOUBLE_UP_LOW, PARTNER_RUFFS] {
        compile(query);
    }
}

// -- C12B: four trumps with the double, missing the second and third.

/// Sixes; the bidder holds 6-6 6-2 6-1 6-0 and three others. The report's
/// two worlds on s200001 deal 43962.
const C12B_BIDDER: &str = "6-6 6-2 6-1 6-0 5-1 3-3 1-0";
const C12B_LOW_FAILS: [&str; 3] =
    ["6-5 5-2 4-2 3-2 3-0 1-1 0-0", "6-4 5-5 5-4 4-1 4-0 2-2 2-1", "6-3 5-3 5-0 4-4 4-3 3-1 2-0"];
const C12B_DOUBLE_FAILS: [&str; 3] =
    ["6-4 6-3 5-2 4-1 3-2 2-1 1-1", "6-5 5-4 5-0 4-4 4-2 3-1 2-2", "5-5 5-3 4-3 4-0 3-0 2-0 0-0"];

fn c12b(others: [&str; 3]) -> OpenPosition {
    deal("P6", [C12B_BIDDER, others[0], others[1], others[2]])
}

#[test]
fn c12b_the_low_lead_fails_where_the_second_trump_is_bare_with_a_setter() {
    let at = c12b(C12B_LOW_FAILS);
    assert!(!holds_in(DOUBLE_UP_LOW, &at));
    assert!(makes(&at, "6-6") && !makes(&at, "6-0"));
    // The 6-6 drops all three sixes out, the partner's 6-4 among them.
    let after = play(&at, &[("6-6", false), ("6-5", true), ("6-4", true), ("6-3", true)]);
    assert_eq!(after.banked(), [11, 0]);
    // The 6-0 loses to the bare 6-5 with the 6-4 under it; the setters lead
    // the 0-0, the bidder's side must follow, and the 5-0 sets the bid.
    let after = play(&at, &[("6-0", false), ("6-5", true), ("6-4", true), ("6-3", true)]);
    assert_eq!(after.banked(), [0, 11]);
    let after = play(&after, &[("0-0", false), ("4-0", true), ("5-0", false), ("1-0", true)]);
    assert_eq!(after.banked(), [0, 17]);
}

#[test]
fn c12b_the_double_fails_where_the_partner_holds_the_second_trump() {
    let at = c12b(C12B_DOUBLE_FAILS);
    assert!(holds_in(DOUBLE_UP_LOW, &at));
    assert!(!makes(&at, "6-6") && makes(&at, "6-0"));
    // On the double the partner's bare 6-5 falls to his own side, and the
    // left setter keeps the 6-4 above every trump the bidder has left.
    let after = play(&at, &[("6-6", false), ("6-3", false), ("6-5", true)]);
    let trump = after.physics().trump().unwrap();
    assert!(after.hand(Seat::S1).contains(tile("6-4")));
    for d in ["6-2", "6-1", "6-0"] {
        assert!(after.physics().beats(trump, tile(d)).contains(tile("6-4")), "{d}");
    }
    // On the low lead the partner's 6-5 takes the trick.
    let after = play(&at, &[("6-0", false), ("6-3", false), ("6-5", true), ("0-0", false)]);
    assert_eq!(after.leader(), Seat::S2);
}

#[test]
fn c12b_the_condition_is_194_of_285_at_every_hand_of_the_shape() {
    // 1/3 for the partner and 33/95 for a setter with another trump: the
    // bidder's own hand cannot move it.
    for (decl, hand) in [
        ("P6", C12B_BIDDER),
        ("P4", "4-4 4-3 4-2 4-0 6-6 6-5 5-2"),
        ("P4", "4-4 4-3 4-2 4-1 3-1 2-2 2-0"),
        ("P2", "2-2 4-2 2-1 2-0 5-5 5-1 3-0"),
        ("P3", "3-3 4-3 3-1 3-0 5-0 2-1 1-1"),
    ] {
        let frame = opening(&physics(decl), tiles(&hand.split(' ').collect::<Vec<_>>()));
        assert_eq!(probability(DOUBLE_UP_LOW, frame), r(194, 285), "{decl} {hand}");
    }
    // Outside the shape it is false: three trumps, or the second held.
    let frame = opening(&physics("P6"), tiles(&["6-6", "6-2", "6-1", "5-1", "3-3", "1-0", "0-0"]));
    assert_eq!(probability(DOUBLE_UP_LOW, frame), r(0, 1));
}

// -- C10: one off and four trumps or more; the off may go first.

/// Deuces; the bidder holds five trumps and the 5-1 off. The report's two
/// worlds on s350001 deal 31347.
const C10_BIDDER: &str = "6-2 5-2 5-1 4-2 2-1 2-0 0-0";
const C10_OFF_FAILS: [&str; 3] =
    ["6-6 6-3 5-5 5-4 5-3 3-2 3-0", "6-5 6-4 4-1 4-0 3-1 2-2 1-1", "6-1 6-0 5-0 4-4 4-3 3-3 1-0"];
const C10_TRUMP_FAILS: [&str; 3] =
    ["6-5 6-1 5-5 5-4 5-0 4-4 4-3", "6-6 4-0 3-2 3-1 3-0 1-1 1-0", "6-4 6-3 6-0 5-3 4-1 3-3 2-2"];

fn c10(others: [&str; 3]) -> OpenPosition {
    deal("P2", [C10_BIDDER, others[0], others[1], others[2]])
}

#[test]
fn c10_the_book_permits_the_off_and_the_repair_leads_trump() {
    let at = Position::holding(physics("P2"), 30, B, tiles(&C10_BIDDER.split(' ').collect::<Vec<_>>()), vec![]);
    assert_eq!(allows(FIRST_LEAD, &at, "one-off-may-go-first"), tiles(&["5-1"]));
    assert_eq!(allows(ONE_OFF, &at, "lead-a-trump"), tiles(&["6-2", "5-2", "4-2", "2-1", "2-0"]));
}

#[test]
fn c10_the_off_fails_where_a_setter_takes_it_with_count() {
    let at = c10(C10_OFF_FAILS);
    assert!(!holds_in(PARTNER_RUFFS, &at));
    assert!(makes(&at, "5-2") && !makes(&at, "5-1"));
    // The partner must follow with the 6-5 and the right setter with the 5-0.
    let after = play(&at, &[("5-1", false), ("5-5", false), ("6-5", true), ("5-0", true)]);
    assert_eq!(after.banked(), [0, 16]);
    // The trump lead pulls the left setter's only trump under the partner's 2-2.
    let after = play(&at, &[("5-2", false), ("3-2", true), ("2-2", false), ("1-0", false)]);
    assert_eq!((after.leader(), after.banked()), (Seat::S2, [6, 0]));
}

#[test]
fn c10_the_trump_lead_fails_where_the_partner_would_ruff_the_off() {
    let at = c10(C10_TRUMP_FAILS);
    assert!(holds_in(PARTNER_RUFFS, &at));
    assert!(!makes(&at, "5-2") && makes(&at, "5-1"));
    // The trump lead pulls the partner's only trump, the 3-2, under the 2-2;
    // the 5-3 then forces the bidder's 5-1 under the 5-5.
    let after = play(&at, &[("5-2", false), ("5-0", false), ("3-2", true), ("2-2", false)]);
    assert_eq!(after.banked(), [0, 11]);
    let after = play(&after, &[("5-3", false), ("5-1", true), ("5-5", false), ("1-0", false)]);
    assert_eq!(after.banked(), [0, 22]);
    // On the off the partner ruffs with the 3-2.
    let after = play(&at, &[("5-1", false), ("5-0", false), ("3-2", false), ("5-3", false)]);
    assert_eq!((after.leader(), after.banked()), (Seat::S2, [11, 0]));
}

/// The off of a hand the book's permission covers, and the tiles of its suit
/// and the trumps the bidder cannot see.
fn off_and_counts(frame: &Frame) -> (Domino, u64, u64) {
    let physics = frame.physics();
    let hand = frame.kernel().viewer_hand();
    let trump = physics.trump().unwrap();
    let offs: Vec<Domino> = hand
        .iter()
        .filter(|d| !physics.follows(*d, trump))
        .filter(|d| {
            let q = physics.led_context(*d);
            physics.incidence(q).difference(hand).iter().any(|x| physics.beats(q, *d).contains(x))
        })
        .collect();
    assert_eq!(offs.len(), 1);
    let pool = frame.kernel().pool();
    let k = physics.incidence(physics.led_context(offs[0])).intersection(pool).len();
    let m = physics.incidence(trump).intersection(pool).len();
    (offs[0], k as u64, m as u64)
}

#[test]
fn c10_the_partners_ruff_is_never_more_likely_than_not_at_the_first_lead() {
    // The closed form, over every count the permission allows: the off can
    // lose (k >= 1), four trumps leave at most three out.
    let mut most = BigRational::zero();
    for k in 1..=6 {
        for m in 0..=3 {
            most = most.max(misses_and_meets(k, m));
        }
    }
    assert_eq!(most, r(7259, 14535));
    assert!(most < r(1, 2));
    // Silas agrees with it exactly, at the deals above, the book's Hands 12
    // and 14, and the first 25 random hands the permission covers (seeded); and
    // a setter's ruff is exactly as likely as the partner's.
    let left = PARTNER_RUFFS.replace("(bidders-partner p)", "(left-setter p)");
    let mut frames = vec![
        bidders_frame(&c10(C10_OFF_FAILS)),
        book().bidders("hand-12", &[]).frame(Seat::S0),
        book().bidders("hand-14", &[]).frame(Seat::S0),
    ];
    let policy = program(FIRST_LEAD);
    let compiled: Vec<CompiledPolicy> =
        (0..7).map(|p| policy.compile(&Registry::standard(), &physics(&format!("P{p}"))).unwrap()).collect();
    let mut rng = Rng(20261013);
    let mut covered = 0;
    while covered < 25 {
        let p = rng.below(7);
        let shuffled = rng.shuffled();
        let frame = opening(&physics(&format!("P{p}")), shuffled[..7].iter().copied().collect());
        let allowed = compiled[p].allowed(&frame, &mut budget()).unwrap();
        if matches!(&allowed.provenance, DecisionProvenance::RelationalRule { name } if name == "one-off-may-go-first")
        {
            frames.push(frame);
            covered += 1;
        }
    }
    for frame in frames {
        let (_, k, m) = off_and_counts(&frame);
        let p = probability(PARTNER_RUFFS, frame.clone());
        assert_eq!(p, misses_and_meets(k, m), "k {k} m {m}");
        assert!(p < r(1, 2), "{}", p.to_f64().unwrap());
        assert_eq!(probability(&left, frame), p);
    }
}

#[test]
fn c10_the_repair_leads_the_books_trump_in_hands_12_14_and_32() {
    for (id, lead, off) in [("hand-12", "6-6", "2-0"), ("hand-14", "4-4", "5-2")] {
        let at = book().bidders(id, &[]);
        let allowed = allows(ONE_OFF, &at, "lead-the-top-trumps");
        assert!(allowed.contains(tile(lead)) && !allowed.contains(tile(off)), "{id}");
    }
    let at = book_at("hand-32", 0, B);
    assert!(
        fires(ONE_OFF, &at).is_some_and(|(_, allowed)| allowed.contains(tile("0-0")) && !allowed.contains(tile("3-1")))
    );
}
