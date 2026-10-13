//! The book's rules (Roberson, *Winning 42*), written as Silas policies in
//! `examples/book/` and checked on positions, most of them the book's own.
//!
//! Seats follow the book study's pour: the bidder is S0, the seat after him
//! (the "left" setter) S1, his partner S2, the last setter S3. The book's play
//! order is the same (bidder, left opponent, partner, right opponent).
//!
//! A position names the catalog entry it comes from (`W42-pNNN-x`, in the
//! private study notes) when it is the book's; "ours" marks a position or a
//! play the book does not give. Each rule gets a position where it fires and
//! allows the book's tile, and one where it does not fire. Where the book and
//! the study's Python gate part ways, the test says so: the gate is the
//! reading of #123 that this file checks.
mod common;

use std::sync::Arc;

use common::*;
use silas::*;

const C01A: &str = include_str!("../examples/book/c01a-top-trump-for-the-donation.silas");
const C01B: &str = include_str!("../examples/book/c01b-partner-count-on-a-trump-lead.silas");
const C02: &str = include_str!("../examples/book/c02-high-trump-handoff.silas");
const C02_READING: &str = include_str!("../examples/book/c02-reading-singleton.silas");
const C03: &str = include_str!("../examples/book/c03-biggest-count-first.silas");
const C07: &str = include_str!("../examples/book/c07-pounce.silas");
const C08: &str = include_str!("../examples/book/c08-no-count-on-the-double.silas");
const C09: &str = include_str!("../examples/book/c09-partner-never-leads-trump.silas");
const C10: &str = include_str!("../examples/book/c10-off-on-trick-one.silas");
const C11: &str = include_str!("../examples/book/c11-double-ahead-of-the-off.silas");
const C12A: &str = include_str!("../examples/book/c12a-four-trumps-lead-the-double.silas");
const C12B: &str = include_str!("../examples/book/c12b-four-trumps-lead-low.silas");
const C13A: &str = include_str!("../examples/book/c13a-shed-a-singleton.silas");
const C13B: &str = include_str!("../examples/book/c13b-protect-the-count.silas");
const C13C: &str = include_str!("../examples/book/c13c-low-from-a-pair.silas");
const C13D: &str = include_str!("../examples/book/c13d-lead-a-singleton.silas");
const C14: &str = include_str!("../examples/book/c14-crisis-cash-the-double.silas");
const C15: &str = include_str!("../examples/book/c15-setter-leads-call-for-count.silas");
const C21: &str = include_str!("../examples/book/c21-six-five-bait.silas");
const C22: &str = include_str!("../examples/book/c22-trump-the-double-from-the-left.silas");
const C23: &str = include_str!("../examples/book/c23-partner-leads-doubles.silas");

const POLICIES: [&str; 20] =
    [C01A, C01B, C02, C03, C07, C08, C09, C10, C11, C12A, C12B, C13A, C13B, C13C, C13D, C14, C15, C21, C22, C23];

const B: Seat = Seat::S0;
const O1: Seat = Seat::S1;
const P: Seat = Seat::S2;
const O2: Seat = Seat::S3;

// The book's full deals, hands in seat order bidder, left opponent, partner,
// right opponent, transcribed in the study catalog from the page images.

/// Hand 28 (pp. 97-100, W42-p097-a): blanks, bid 30, set.
const HAND_28: [[&str; 7]; 4] = [
    ["6-0", "0-0", "3-1", "5-3", "5-0", "6-6", "6-1"],
    ["4-0", "5-1", "3-2", "5-5", "4-1", "3-3", "4-4"],
    ["1-0", "2-0", "4-3", "6-5", "6-4", "6-2", "2-2"],
    ["3-0", "2-1", "6-3", "5-4", "4-2", "1-1", "5-2"],
];
const HAND_28_PLAY: [&str; 24] = [
    "6-0", "4-0", "1-0", "3-0", // T1
    "0-0", "5-1", "2-0", "2-1", // T2
    "3-1", "3-2", "4-3", "6-3", // T3, right opponent wins
    "5-4", "5-3", "5-5", "6-5", // T4, left opponent wins
    "4-1", "6-4", "4-2", "5-0", // T5, bidder trumps in
    "6-6", "3-3", "6-2", "1-1", // T6
];

/// Hand 29 (pp. 100-102, W42-p100-c): the same deal re-seated, fours, set on T1.
const HAND_29: [[&str; 7]; 4] = [
    ["4-0", "3-2", "5-1", "4-1", "4-4", "3-3", "5-5"],
    ["6-4", "2-2", "1-0", "6-2", "4-3", "2-0", "6-5"],
    ["4-2", "2-1", "1-1", "6-3", "5-4", "3-0", "5-2"],
    ["5-0", "6-0", "6-1", "6-6", "3-1", "5-3", "0-0"],
];
const HAND_29_PLAY: [&str; 8] = [
    "4-0", "6-4", "4-2", "5-0", // T1, left opponent wins
    "2-2", "2-1", "6-0", "3-2", // T2
];

/// Hand 30 (pp. 102-105, W42-p102-c): sixes, made 42.
const HAND_30: [[&str; 7]; 4] = [
    ["6-5", "1-0", "4-3", "6-4", "2-2", "2-0", "6-2"],
    ["6-3", "3-0", "2-1", "4-2", "5-2", "5-4", "1-1"],
    ["6-6", "0-0", "3-1", "6-0", "5-0", "6-1", "5-3"],
    ["4-0", "5-1", "3-2", "3-3", "4-4", "5-5", "4-1"],
];
const HAND_30_PLAY: [&str; 16] = [
    "6-5", "6-3", "6-6", "4-0", // T1, partner wins
    "0-0", "5-1", "1-0", "3-0", // T2
    "3-1", "3-2", "4-3", "2-1", // T3, bidder wins
    "6-4", "4-2", "6-0", "3-3", // T4
];

/// Hand 31 (pp. 105-108, W42-p105-d): Hand 30's cards, deuces, set.
const HAND_31: [[&str; 7]; 4] = [
    ["2-2", "6-5", "1-0", "4-3", "6-4", "2-0", "6-2"],
    ["2-1", "6-3", "3-0", "5-4", "4-2", "1-1", "5-2"],
    ["5-0", "6-6", "0-0", "3-1", "6-0", "6-1", "5-3"],
    ["3-2", "4-0", "5-1", "3-3", "4-1", "5-5", "4-4"],
];
const HAND_31_PLAY: [&str; 16] = [
    "2-2", "2-1", "5-0", "3-2", // T1
    "6-5", "6-3", "6-6", "4-0", // T2, partner wins
    "0-0", "5-1", "1-0", "3-0", // T3
    "3-1", "3-3", "4-3", "5-4", // T4, right opponent wins
];

/// Hand 32 (pp. 108-110, W42-p108-c): blanks, bid 36, set on T3.
const HAND_32: [[&str; 7]; 4] = [
    ["0-0", "6-0", "3-1", "1-0", "5-0", "3-0", "2-2"],
    ["2-0", "4-0", "6-4", "1-1", "5-5", "2-1", "4-4"],
    ["4-1", "4-2", "6-3", "5-1", "6-1", "6-5", "6-6"],
    ["6-2", "5-3", "3-3", "5-4", "4-3", "3-2", "5-2"],
];
const HAND_32_PLAY: [&str; 8] = [
    "0-0", "2-0", "4-1", "6-2", // T1
    "6-0", "4-0", "4-2", "5-3", // T2
];

/// Hand 33 (pp. 110-112, W42-p110-c): Hand 32's cards, fours, made. The
/// catalog flags its last two tricks; only T1-T4 are used.
const HAND_33: [[&str; 7]; 4] = [
    ["4-4", "6-4", "2-0", "5-5", "4-0", "1-1", "2-1"],
    ["4-2", "4-1", "5-1", "6-5", "6-1", "6-3", "6-6"],
    ["4-3", "5-4", "5-2", "5-3", "3-2", "6-2", "3-3"],
    ["6-0", "3-0", "2-2", "5-0", "1-0", "0-0", "3-1"],
];
const HAND_33_PLAY: [&str; 16] = [
    "4-4", "4-2", "4-3", "6-0", // T1
    "6-4", "4-1", "5-4", "3-0", // T2
    "2-0", "5-1", "5-2", "2-2", // T3, right opponent wins
    "5-0", "5-5", "6-5", "5-3", // T4, bidder wins
];

/// Hand 34 (pp. 112-115, W42-p112-c): fives, bid 84.
const HAND_34: [[&str; 7]; 4] = [
    ["5-5", "5-4", "5-2", "5-0", "2-2", "4-4", "4-2"],
    ["5-1", "5-3", "2-0", "3-0", "4-3", "6-4", "3-3"],
    ["6-5", "6-2", "4-1", "1-1", "1-0", "4-0", "6-6"],
    ["0-0", "6-0", "2-1", "3-2", "6-1", "6-3", "3-1"],
];

/// Hand 35 (pp. 115-118, W42-p115-f): fours, bid 84, set on the last trick.
const HAND_35: [[&str; 7]; 4] = [
    ["4-4", "6-4", "4-3", "4-0", "5-5", "6-6", "6-3"],
    ["0-0", "1-0", "6-0", "5-0", "5-3", "3-0", "2-2"],
    ["5-4", "3-1", "3-2", "6-2", "5-2", "3-3", "2-1"],
    ["4-2", "4-1", "1-1", "2-0", "5-1", "6-1", "6-5"],
];
const HAND_35_PLAY: [&str; 19] = [
    "4-4", "0-0", "5-4", "4-2", // T1
    "6-4", "1-0", "3-1", "4-1", // T2
    "4-3", "6-0", "3-2", "1-1", // T3
    "4-0", "5-0", "6-2", "2-0", // T4
    "5-5", "5-3", "5-2", // T5, the right opponent to play
];

/// One seat's view of a hand: its frame, the public record, the score.
struct Position {
    physics: Arc<dyn Physics>,
    frame: Frame,
    history: Vec<(Seat, Domino)>,
    banked: [u32; 2],
}

impl Position {
    fn input(&self) -> PolicyInput<'_> {
        PolicyInput::new(&self.frame, &self.history, self.banked, 30, Team::T0).unwrap()
    }
}

/// The seats of `plays`, the bidder leading the first trick.
fn actors(physics: &dyn Physics, plays: &[Domino]) -> Vec<Seat> {
    let mut leader = B;
    let mut trick = Vec::new();
    plays
        .iter()
        .map(|tile| {
            let actor = leader.plus(trick.len());
            trick.push(*tile);
            if let Ok(full) = <[Domino; 4]>::try_from(trick.as_slice()) {
                leader = physics.trick_winner(leader, full);
                trick.clear();
            }
            actor
        })
        .collect()
}

/// `viewer`'s view after `plays`, holding `hand`. The other seats' plays are
/// checked only against their shown voids: a book that gives one hand gives
/// the rest of the record without the hands behind it.
fn position(decl: &str, viewer: Seat, hand: &[&str], plays: &[&str]) -> Position {
    let physics = physics(decl);
    let plays: Vec<Domino> = plays.iter().map(|d| tile(d)).collect();
    let seats = actors(physics.as_ref(), &plays);
    let now = tiles(hand);
    let dealt =
        seats.iter().zip(&plays).filter(|(s, _)| **s == viewer).fold(now, |h, (_, d)| h.union(DominoSet::single(*d)));
    assert_eq!(dealt.len(), 7, "the viewer's hand and plays make a seven-tile deal");
    let hidden = [1, 2, 3].map(|i| Hidden { seat: viewer.plus(i), capacity: 7, voids: ContextSet::EMPTY });
    let kernel = Kernel::new(viewer, dealt, DominoSet::FULL.difference(dealt), hidden).unwrap();
    let mut frame = Frame::new(physics.clone(), kernel, B, vec![], DominoSet::EMPTY).unwrap();
    let mut banked = [0, 0];
    for (actor, tile) in seats.iter().zip(&plays) {
        let class = match frame.led_context() {
            None => PlayClass::Lead,
            Some(q) if physics.follows(*tile, q) => PlayClass::Follow(q),
            Some(q) => PlayClass::Slough(q),
        };
        let mut trick = frame.prefix().to_vec();
        trick.push(*tile);
        let leader = frame.leader();
        frame = step_frame(&frame, ObservedPlay { actor: *actor, tile: *tile, class }, &mut budget()).unwrap();
        if let Ok(full) = <[Domino; 4]>::try_from(trick.as_slice()) {
            banked[physics.trick_winner(leader, full).team().index()] += physics.trick_points(full);
        }
    }
    assert_eq!(frame.kernel().viewer_hand(), now);
    Position { physics, frame, history: seats.into_iter().zip(plays).collect(), banked }
}

/// `viewer`'s view of a full deal after `plays`, every play checked legal
/// against the hand it came from.
fn dealt(decl: &str, deal: &[[&str; 7]; 4], viewer: Seat, plays: &[&str]) -> Position {
    let rules = physics(decl);
    let mut hands = deal.map(|h| tiles(&h));
    assert_eq!(hands.iter().fold(DominoSet::EMPTY, |all, h| all.union(*h)), DominoSet::FULL);
    let tiles_played: Vec<Domino> = plays.iter().map(|d| tile(d)).collect();
    let mut led = None;
    for (i, (seat, d)) in actors(rules.as_ref(), &tiles_played).into_iter().zip(&tiles_played).enumerate() {
        if i % 4 == 0 {
            led = Some(*d);
        }
        let held = hands[seat.index()];
        let lead = if i % 4 == 0 { None } else { led };
        assert!(rules.legal_plays(held, lead).contains(*d), "play {i}: {seat} cannot play {d}");
        hands[seat.index()].remove(*d);
    }
    let hand: Vec<String> = hands[viewer.index()].iter().map(|d| d.to_string()).collect();
    let hand: Vec<&str> = hand.iter().map(String::as_str).collect();
    position(decl, viewer, &hand, plays)
}

fn program(source: &str) -> PolicyProgram {
    source.parse().unwrap()
}

/// The first rule of the policy that selects at this position, and every tile
/// it allows (legal answers of its guard). The policy's own decision must be
/// one of them, by that rule; `None` means the policy falls back.
fn fires(source: &str, at: &Position) -> Option<(String, DominoSet)> {
    let program = program(source);
    let input = at.input();
    let legal = at.physics.legal_plays(at.frame.kernel().viewer_hand(), at.frame.lead());
    let mut found = None;
    for rule in &program.rules {
        let guard = rule.guard.compile(&Registry::standard()).unwrap();
        let answers = guard.evaluate_viewer(&at.frame, &mut budget()).unwrap();
        let allowed: DominoSet = match &rule.selector {
            Selector::FirstOutput => answers
                .iter()
                .filter_map(|a| match a.0.as_slice() {
                    [Value::Domino(d)] => Some(*d),
                    _ => None,
                })
                .collect(),
            Selector::Literal(d) if !answers.is_empty() => DominoSet::single(*d),
            _ => DominoSet::EMPTY,
        };
        let allowed = allowed.intersection(legal);
        if !allowed.is_empty() {
            found = Some((rule.name.clone(), allowed));
            break;
        }
    }
    let compiled = program.compile(&Registry::standard(), &at.physics).unwrap();
    let mut controller = compiled.initialize(&input, &mut budget()).unwrap();
    let trace = compiled.choose_traced(&mut controller, &input, &mut budget()).unwrap();
    match &found {
        Some((name, allowed)) => {
            assert_eq!(trace.provenance, DecisionProvenance::RelationalRule { name: name.clone() });
            assert!(allowed.contains(trace.action));
        }
        None => assert_eq!(trace.provenance, DecisionProvenance::Fallback),
    }
    found
}

/// The tiles the policy allows, asserting it fires by the named rule.
fn allows(source: &str, at: &Position, rule: &str) -> DominoSet {
    let (name, allowed) = fires(source, at).expect("the rule fires");
    assert_eq!(name, rule);
    allowed
}

#[test]
fn every_book_file_parses_round_trips_and_compiles_under_every_declaration() {
    for source in POLICIES {
        let parsed = program(source);
        assert_eq!(parsed.to_string().parse::<PolicyProgram>().unwrap(), parsed);
        for rules in reference::Straight42::ALL {
            let rules: Arc<dyn Physics> = Arc::new(rules);
            parsed.compile(&Registry::standard(), &rules).unwrap();
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
    let at = dealt("P0", &HAND_32, B, &HAND_32_PLAY[..4]);
    assert_eq!(allows(C01A, &at, "lead-the-top-trump"), tiles(&["6-0"]));
}

#[test]
fn c01a_wider_than_the_gate_in_hand_28_and_still_the_book() {
    // W42-p098-c: after T1 every seat followed blanks, so the partner has not
    // shown void and the gate does not fire. One trump (2-0) is out; the book
    // leads the 0-0 to pull it, hoping the partner can then give count. The
    // encoding, blind to the partner's void (GAP shown-void), fires and
    // allows the book's tile.
    let at = dealt("P0", &HAND_28, B, &HAND_28_PLAY[..4]);
    assert_eq!(allows(C01A, &at, "lead-the-top-trump"), tiles(&["0-0"]));
}

#[test]
fn c01a_does_not_fire_in_hand_33_where_the_book_still_leads_the_trump() {
    // W42-p111-d: after T4 no opposing trump is out, so neither the gate nor
    // the encoding fires. The book leads the 4-0, the bidder's last trump,
    // because the partner has not yet had a safe trick to give count on. The
    // gate's "a trump still out" excludes a book play: the book's condition is
    // what the partner knows, not what the bidder knows.
    let at = dealt("P4", &HAND_33, B, &HAND_33_PLAY);
    assert_eq!(fires(C01A, &at), None);
}

// C1b: the partner, out of trump, gives count only on the top trump.

#[test]
fn c01b_top_lead_gives_the_count_in_hands_31_and_32() {
    // W42-p106-a: the bidder leads the 2-2; the partner, void in deuces, gives
    // the 5-0, his only count.
    let at = dealt("P2", &HAND_31, P, &HAND_31_PLAY[..2]);
    assert_eq!(allows(C01B, &at, "top-lead-give-five"), tiles(&["5-0"]));
    // W42-p109-a: the bidder leads the 0-0; the partner gives the 4-1.
    let at = dealt("P0", &HAND_32, P, &HAND_32_PLAY[..2]);
    assert_eq!(allows(C01B, &at, "top-lead-give-five"), tiles(&["4-1"]));
}

#[test]
fn c01b_not_top_lead_keeps_the_count() {
    // Ours, on Hand 31's deal: the bidder leads the 6-2 instead of the 2-2.
    // The partner, void in deuces, cannot know the trick is safe (the 2-2 is
    // live), so he throws anything but his 5-0. The book has no worked
    // example of this branch (W42-p038-d states it).
    let at = dealt("P2", &HAND_31, P, &["6-2", "2-1"]);
    assert_eq!(allows(C01B, &at, "not-top-lead-no-count"), tiles(&["6-6", "0-0", "3-1", "6-0", "6-1", "5-3"]));
}

#[test]
fn c01b_does_not_fire_without_count_or_while_holding_trump() {
    // W42-p109-a, T2: the partner has no count left and throws the 4-2.
    let at = dealt("P0", &HAND_32, P, &HAND_32_PLAY[..6]);
    assert_eq!(fires(C01B, &at), None);
    // W42-p101-b, Hand 29 T1: the partner holds trump and must follow.
    let at = dealt("P4", &HAND_29, P, &HAND_29_PLAY[..2]);
    assert_eq!(fires(C01B, &at), None);
}

// C2: the table's handoff, and the book's reading of it.

#[test]
fn c02_fires_in_hand_30_and_the_book_reading_is_false_there() {
    // W42-p102-c, Hand 30 T1: the bidder leads the 6-5; the partner holds the
    // 6-6, the top six, and plays it. The handoff allows exactly that.
    let at = dealt("P6", &HAND_30, P, &HAND_30_PLAY[..2]);
    assert_eq!(allows(C02, &at, "play-the-top-trump"), tiles(&["6-6"]));
    // The book's p. 25 reads a partner's top trump as his only one. Here he
    // still holds the 6-0 and the 6-1: the reading is false in the book's own
    // deal. From the bidder's chair, over the deals he cannot tell apart, the
    // partner is void in trump with probability 26/57 (his six tiles miss both
    // live sixes, of nineteen hidden): the reading is a convention, not a
    // fact of the deal.
    let bidder = dealt("P6", &HAND_30, B, &HAND_30_PLAY[..3]);
    let reading = C02_READING.parse::<Query>().unwrap().compile(&Registry::standard()).unwrap();
    let belief = Belief::uniform_resolving(bidder.frame.clone(), tiles(&["6-0", "6-1"]), 100).unwrap();
    let p = exact(reading.summarize(&belief, &mut budget()).unwrap().event_probability());
    assert_eq!(p, r(26, 57));
}

#[test]
fn c02_does_not_fire_in_hand_29() {
    // W42-p101-b, Hand 29 T1: the bidder leads the 4-0; the top four (4-4) is
    // the bidder's own, so the partner, holding 4-2 and 5-4, does not hold it.
    let at = dealt("P4", &HAND_29, P, &HAND_29_PLAY[..2]);
    assert_eq!(fires(C02, &at), None);
}

// C3: biggest count first.

#[test]
fn c03_gives_the_ten_before_the_five() {
    // W42-p106-c: in Hand 31 the bidder reasons that a partner holding the
    // 5-5 would have given it instead of the 5-0. Ours: that partner, Hand
    // 31's deal with the 5-5 and 6-6 exchanged between partner and right
    // opponent. C1b agrees.
    let mut deal = HAND_31;
    deal[2][1] = "5-5";
    deal[3][5] = "6-6";
    let at = dealt("P2", &deal, P, &HAND_31_PLAY[..2]);
    assert_eq!(allows(C03, &at, "give-a-ten"), tiles(&["5-5"]));
    assert_eq!(allows(C01B, &at, "top-lead-give-ten"), tiles(&["5-5"]));
}

#[test]
fn c03_does_not_fire_with_one_count_tile() {
    // W42-p106-a, Hand 31 T1 as dealt: the 5-0 is the partner's only count, so
    // the gate's "two or more" fails; C1b gives it.
    let at = dealt("P2", &HAND_31, P, &HAND_31_PLAY[..2]);
    assert_eq!(fires(C03, &at), None);
}

// C7: the pounce.

#[test]
fn c07_pounces_in_hand_32() {
    // W42-p109-b, Hand 32 T3: the bidder leads his 3-1 off; the left opponent,
    // void in treys and out of trump, holds 6-4 and 5-5. The book plays the
    // 6-4 at once (the partner wins with the 3-3 and the bid is set); the 5-5
    // is the same ten.
    let at = dealt("P0", &HAND_32, O1, &[&HAND_32_PLAY[..], &["3-1"]].concat());
    assert_eq!(allows(C07, &at, "play-count-now"), tiles(&["6-4", "5-5"]));
}

#[test]
fn c07_does_not_fire_when_the_setter_can_follow() {
    // W42-p098-e, Hand 28 T3: the bidder leads the 3-1; the left opponent holds
    // 3-2 and 3-3 and can follow. The gate needs a void and does not fire, but
    // the book plays the 3-2 here, count before the partner: its pounce
    // (pp. 43-46) is wider than the gate.
    let at = dealt("P0", &HAND_28, O1, &HAND_28_PLAY[..9]);
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
fn c13b_cannot_say_exactly_two() {
    // GAP none: with three sixes the gate's C13b does not fire; the encoding
    // does, and is then C8.
    let at = guarded_six_four("6-1");
    assert_eq!(allows(C13B, &at, "follow-with-the-guard"), tiles(&["6-0", "6-1"]));
    assert_eq!(allows(C08, &at, "follow-without-count"), tiles(&["6-0", "6-1"]));
}

#[test]
fn c08_and_c13b_do_not_fire_on_a_trump_double_where_the_book_protects() {
    // W42-p111-a, Hand 33 T1: the bidder leads the 4-4 (trump); the left
    // opponent holds 4-2 and 4-1 and plays the 4-2, saving the count. The
    // gates take only non-trump doubles, so neither fires on the book's
    // instance of the rule.
    let at = dealt("P4", &HAND_33, O1, &HAND_33_PLAY[..1]);
    assert_eq!(fires(C08, &at), None);
    assert_eq!(fires(C13B, &at), None);
}

// C9: the partner never leads trump.

#[test]
fn c09_fires_in_hand_30_and_the_book_narrows_it() {
    // W42-p103-d, Hand 30 T3: the partner leads holding 3-1, 5-0, 5-3 and two
    // trumps (6-0, 6-1). Never trump allows the three; the book leads the 3-1,
    // which calls only a five, by the ordering this file cannot write (GAP).
    let at = dealt("P6", &HAND_30, P, &HAND_30_PLAY[..8]);
    let allowed = allows(C09, &at, "lead-anything-but-trump");
    assert_eq!(allowed, tiles(&["3-1", "5-0", "5-3"]));
}

#[test]
fn c09_does_not_fire_without_trump() {
    // W42-p107-b, Hand 31 T4: the partner holds no trump and leads the 3-1.
    let at = dealt("P2", &HAND_31, P, &HAND_31_PLAY[..12]);
    assert_eq!(fires(C09, &at), None);
}

// C10: the off on trick one.

#[test]
fn c10_fires_on_hand_14_where_the_book_leads_trump() {
    // W42-p028-b, Hand 14: fours, the bidder holds five trumps and one off
    // (5-2). The gate makes the off the book's tile; the book's p. 24 only
    // permits it, and in this hand the book leads the 4-4 (and is later set on
    // the 5-2). The gate reads a permission as an instruction.
    let at = position("P4", B, &["5-2", "4-1", "6-6", "6-4", "4-2", "4-3", "4-4"], &[]);
    assert_eq!(allows(C10, &at, "lead-the-off"), tiles(&["5-2"]));
}

#[test]
fn c10_cannot_say_exactly_one_off() {
    // Ours: Hand 14 with the 6-6 for the 6-1, two offs. The gate does not fire
    // (GAP none); the encoding does.
    let at = position("P4", B, &["5-2", "4-1", "6-1", "6-4", "4-2", "4-3", "4-4"], &[]);
    assert_eq!(allows(C10, &at, "lead-the-off"), tiles(&["5-2", "6-1"]));
}

#[test]
fn c10_does_not_fire_on_hand_27() {
    // W42-p094-d, Hand 27: fours with two trumps. Neither the gate nor the
    // encoding fires; the book leads its 6-2 off on trick one anyway, as a
    // stated exception for two-trump hands (W42-p091-c).
    let at = position("P4", B, &["3-3", "6-2", "4-4", "6-4", "0-0", "2-2", "3-1"], &[]);
    assert_eq!(fires(C10, &at), None);
}

// C11: the double ahead of the off.

#[test]
fn c11_fires_in_hand_28() {
    // W42-p099-b, Hand 28 T6: trumps are gone; the bidder holds 6-6 and the 6-1
    // off (the 6-2 is out) and leads the double.
    let at = dealt("P0", &HAND_28, B, &HAND_28_PLAY[..20]);
    assert_eq!(allows(C11, &at, "lead-the-double"), tiles(&["6-6"]));
}

#[test]
fn c11_wider_than_the_gate_in_hand_30_and_still_the_book() {
    // W42-p105-a, Hand 30 T5: the last trump (6-1) is in the partner's hand,
    // so the gate's "trumps drawn" fails. The book leads the 2-2 ahead of the
    // 2-0 off; the encoding, which cannot say "trumps drawn" (GAP none),
    // allows it.
    let at = dealt("P6", &HAND_30, B, &HAND_30_PLAY);
    assert_eq!(allows(C11, &at, "lead-the-double"), tiles(&["2-2"]));
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
    // double and the third but not the 6-5; the book leads the 5-5.
    let at = dealt("P5", &HAND_34, B, &[]);
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
}

#[test]
fn c12_does_not_fire_on_three_trumps_where_the_book_leads_low() {
    // W42-p101-a, Hand 29 T1: fours, the bidder holds 4-4, 4-1, 4-0 (missing
    // the 6-4 and 5-4) and leads the 4-0 low, the B rule with three trumps
    // (also W42-p183-g). Both gates take four trumps only.
    let at = dealt("P4", &HAND_29, B, &[]);
    assert_eq!(fires(C12A, &at), None);
    assert_eq!(fires(C12B, &at), None);
}

// C13a: shed a singleton on the bidder's trump lead.

#[test]
fn c13a_allows_the_books_singleton_and_more() {
    // W42-p109-a, Hand 32 T1: the right opponent, no blanks, throws the 6-2,
    // his only six (the gate's set is exactly that). The encoding cannot say
    // "only one" (GAP none) and allows every non-count tile.
    let at = dealt("P0", &HAND_32, O2, &HAND_32_PLAY[..3]);
    let allowed = allows(C13A, &at, "throw-a-lone-tile");
    assert_eq!(allowed, tiles(&["6-2", "5-3", "3-3", "5-4", "4-3", "5-2"]));
    // W42-p111-a, Hand 33 T1: the right opponent throws the 6-0, his only six.
    let at = dealt("P4", &HAND_33, O2, &HAND_33_PLAY[..3]);
    assert!(allows(C13A, &at, "throw-a-lone-tile").contains(tile("6-0")));
}

#[test]
fn c13a_does_not_fire_while_the_setter_holds_trump() {
    // Hand 32 T1: the left opponent must follow with a blank.
    let at = dealt("P0", &HAND_32, O1, &HAND_32_PLAY[..1]);
    assert_eq!(fires(C13A, &at), None);
}

// C13c: low from a pair.

#[test]
fn c13c_follows_low_in_hand_35() {
    // W42-p117-b, Hand 35 T5: the bidder leads the 5-5; the right opponent
    // holds two fives, 5-1 and 6-5, neither count, and plays the 5-1 (the book's
    // reason is to keep his sixes; the tile is the gate's).
    let at = dealt("P4", &HAND_35, O2, &HAND_35_PLAY);
    assert_eq!(allows(C13C, &at, "follow-low"), tiles(&["5-1"]));
}

#[test]
fn c13c_does_not_fire_when_the_pair_holds_count() {
    // W42-p098-e, Hand 28 T3: the left opponent's treys are 3-2 and 3-3.
    let at = dealt("P0", &HAND_28, O1, &HAND_28_PLAY[..9]);
    assert_eq!(fires(C13C, &at), None);
}

// C13d: lead a singleton (the gate's reading; no page states it).

#[test]
fn c13d_fires_in_hand_28_where_the_book_leads_the_ten_caller() {
    // W42-p098-f, Hand 28 T4: the right opponent leads holding 5-4, 4-2, 1-1,
    // 5-2. The gate's tile is the 1-1, his only ace; the book leads the 5-4,
    // calling for the 5-5 that sets the bid. The encoding (GAP none) allows
    // every non-trump, non-count lead, the book's among them.
    let at = dealt("P0", &HAND_28, O2, &HAND_28_PLAY[..12]);
    assert_eq!(allows(C13D, &at, "lead-the-lone-tile"), tiles(&["5-4", "4-2", "1-1", "5-2"]));
}

#[test]
fn c13d_does_not_fire_off_lead() {
    let at = dealt("P0", &HAND_28, O1, &HAND_28_PLAY[..9]);
    assert_eq!(fires(C13D, &at), None);
}

// C14: crisis management.

#[test]
fn c14_cashes_a_double_in_hand_26() {
    // W42-p093-c, Hand 26 (fours): after two trump tricks an opponent still
    // holds the 4-2; the bidder cashes the 5-5 on T3, then leads the double of
    // the suit with more tiles out. The tricks' other tiles are ours (the book
    // gives the shape, not the plays). The gate does not fire (two trump
    // tricks, not one); the encoding, without trigger or "most unplayed"
    // (GAP history, shown-void, aggregate), allows every double, the book's
    // 2-2 and 3-3 among them, and the 1-1 the book calls worthless here.
    let plays = [
        "4-4", "4-0", "4-1", "4-3", // T1
        "6-4", "4-5", "6-6", "3-0", // T2
        "5-5", "5-3", "5-1", "6-5", // T3
    ];
    let at = position("P4", B, &["3-3", "1-1", "2-2", "5-2"], &plays);
    assert_eq!(allows(C14, &at, "cash-a-double"), tiles(&["1-1", "2-2", "3-3"]));
}

#[test]
fn c14_does_not_fire_in_hand_31() {
    // W42-p106-d, Hand 31 T2: the book's crisis lead is the 6-5 bait (C21);
    // the bidder holds no non-trump double.
    let at = dealt("P2", &HAND_31, B, &HAND_31_PLAY[..4]);
    assert_eq!(fires(C14, &at), None);
}

// C15: setter leads.

#[test]
fn c15_leads_the_count_calling_double_in_hand_29() {
    // W42-p101-c, Hand 29 T2: the left opponent leads the 2-2, calling for the
    // 3-2.
    let at = dealt("P4", &HAND_29, O1, &HAND_29_PLAY[..4]);
    assert_eq!(allows(C15, &at, "a-double-that-calls-for-count"), tiles(&["2-2"]));
}

#[test]
fn c15_leads_the_double_where_the_book_leads_the_count_that_sets() {
    // W42-p098-f, Hand 28 T4: the setters hold 6 and need 13. The gate's tier
    // A is the 1-1 (it calls the 4-1); the book leads the 5-4, which calls the
    // 5-5 and sets.
    let at = dealt("P0", &HAND_28, O2, &HAND_28_PLAY[..12]);
    assert_eq!(allows(C15, &at, "a-double-that-calls-for-count"), tiles(&["1-1"]));
    // W42-p108-b, Hand 31 T5: tier A is the 4-4 (it calls the 6-4); the book
    // leads the 4-1, which calls the 6-4 for fifteen.
    let at = dealt("P2", &HAND_31, O2, &HAND_31_PLAY);
    assert_eq!(allows(C15, &at, "a-double-that-calls-for-count"), tiles(&["4-4"]));
}

#[test]
fn c15_does_not_fire_without_a_count_calling_lead() {
    // W42-p101-d, Hand 29 T3: no lead calls for count the book allows; the
    // book leads the 1-0 just to see what happens.
    let at = dealt("P4", &HAND_29, O1, &HAND_29_PLAY);
    assert_eq!(fires(C15, &at), None);
}

// C21: the six-five bait.

#[test]
fn c21_leads_the_six_five_in_hand_31() {
    // W42-p106-d, Hand 31 T2: deuces, the bidder holds 6-5 and 6-4 and leads
    // the 6-5.
    let at = dealt("P2", &HAND_31, B, &HAND_31_PLAY[..4]);
    assert_eq!(allows(C21, &at, "lead-the-six-five"), tiles(&["6-5"]));
}

#[test]
fn c21_does_not_fire_when_the_six_five_is_trump() {
    // W42-p103-a, Hand 30 T1: sixes; the 6-5 is a trump, led as one.
    let at = dealt("P6", &HAND_30, B, &[]);
    assert_eq!(fires(C21, &at), None);
}

// C22: trump the bidder's double from the left.

/// Ours (W42-p051-b gives no hand): deuces trump, the bidder leads the 6-6;
/// the setter holds two trumps and no six.
const TRUMP_RICH: [&str; 7] = ["5-2", "4-2", "5-5", "3-3", "1-0", "4-0", "3-1"];

#[test]
fn c22_trumps_in_from_the_left() {
    let at = position("P2", O1, &TRUMP_RICH, &["6-6"]);
    assert_eq!(allows(C22, &at, "trump-in"), tiles(&["5-2", "4-2"]));
}

#[test]
fn c22_does_not_fire_from_the_last_seat() {
    // The book leaves the last seat to decide by the count already on the
    // trick; the gate fires there too.
    let at = position("P2", O2, &TRUMP_RICH, &["6-6", "6-1", "6-0"]);
    assert_eq!(fires(C22, &at), None);
}

// C23: the partner leads doubles.

#[test]
fn c23_leads_the_double_in_hand_30() {
    // W42-p102-c, Hand 30 T2: the partner, on lead after the 6-6, leads 0-0.
    let at = dealt("P6", &HAND_30, P, &HAND_30_PLAY[..4]);
    assert_eq!(allows(C23, &at, "lead-a-double"), tiles(&["0-0"]));
}

#[test]
fn c23_does_not_fire_out_of_doubles() {
    // W42-p103-d, Hand 30 T3: the partner has no double left.
    let at = dealt("P6", &HAND_30, P, &HAND_30_PLAY[..8]);
    assert_eq!(fires(C23, &at), None);
}
