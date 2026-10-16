// SPDX-FileCopyrightText: Jason Yandell
// SPDX-License-Identifier: CC0-1.0

//! Bid and declare policies (`docs/cross-game.md`, slice 2), held to the
//! arena's bidder and to the book.
//!
//! `bid-only-enough.silas` is the arena's `HeuristicBidder` (arena/bidders.py,
//! min_trumps 3, caution 0) written in Silas: it is held to the arena's own
//! auctions (`data/heuristic-bidder.txt`, written by the arena's code: every
//! call and declaration of 400 deals at every shaker, with and without the
//! forced bid), and to a transcription of the arena's arithmetic on every one
//! of the 1,184,040 hands a seat can be dealt. `risk-budget.silas` is the bid
//! book's rule as we read it, held to that reading on every hand and run on
//! the book's 30 bid-book hands. Both give an exact histogram of the opening
//! call over all hands.

mod common;

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Instant;

use common::*;
use silas::reference::Straight42;
use silas::*;

const ARENA_DECLARATIONS: [&str; 7] = ["P6", "P5", "P4", "P3", "P2", "P1", "P0"];

/// The arena's declarations, the higher pip first (its last tie-break).
fn arena_menu() -> Arc<[Physics]> {
    ARENA_DECLARATIONS.iter().map(|d| physics(d)).collect()
}

fn policy(file: &str) -> CompiledPolicy {
    let path = format!("{}/examples/book/bidding/{file}", env!("CARGO_MANIFEST_DIR"));
    let program = PolicyProgram::parse(&std::fs::read_to_string(&path).unwrap(), &path).unwrap();
    program.compile(&Registry::standard(), &physics("P6")).unwrap()
}

fn first_call(viewer: Seat, dealt: DominoSet, menu: &Arc<[Physics]>) -> AuctionFrame {
    let auction = Auction::new(AuctionConfig::STRAIGHT, viewer.plus(3), vec![]).unwrap();
    AuctionFrame::new(auction, viewer, dealt, MatchState::new(Scoring::MARKS, [0, 0]).unwrap(), menu.clone()).unwrap()
}

/// Every how many hands the two heaviest all-hands proofs read in the default
/// run: 4. `SILAS_FULL_PROOF=1` reads every hand (STRIDE 1) and checks the
/// all-hands digests (`PINNED_ANSWERS`, `PINNED_RAISES`); the default run
/// checks the digests of every 4th hand (`PINNED_ANSWERS_4TH`,
/// `PINNED_RAISES_4TH`), taken from the same lines. Run the full proof when
/// a bidding policy, the bidding prelude or the evaluator changes.
fn proof_stride() -> usize {
    if std::env::var_os("SILAS_FULL_PROOF").is_some_and(|v| v != "0" && !v.is_empty()) { 1 } else { 4 }
}

/// Every hand of seven tiles, in increasing order of its bits.
fn every_hand() -> impl Iterator<Item = DominoSet> {
    let last = 0x0fe0_0000u32;
    std::iter::successors(Some(0x7fu32), move |&v| {
        (v != last).then(|| {
            // Gosper's hack: the next word with as many bits set.
            let c = v & v.wrapping_neg();
            let r = v + c;
            (((r ^ v) >> 2) / c) | r
        })
    })
    .map(|bits| DominoSet::from_bits(bits).unwrap())
}

/// What a pass over hands found: how often each outcome came, and the first
/// few hands where the policy and its oracle part.
#[derive(Default, Debug)]
struct Tally {
    hands: usize,
    outcomes: BTreeMap<String, usize>,
    mismatches: Vec<String>,
}

impl Tally {
    fn add(&mut self, outcome: String, mismatch: Option<String>) {
        self.hands += 1;
        *self.outcomes.entry(outcome).or_default() += 1;
        if let Some(m) = mismatch.filter(|_| self.mismatches.len() < 5) {
            self.mismatches.push(m);
        }
    }

    fn merge(&mut self, other: Tally) {
        self.hands += other.hands;
        for (k, v) in other.outcomes {
            *self.outcomes.entry(k).or_default() += v;
        }
        self.mismatches.extend(other.mismatches);
    }

    fn of(&self, outcome: &str) -> usize {
        self.outcomes.get(outcome).copied().unwrap_or(0)
    }
}

/// `f` on every hand, on every core: each hand's outcome, and a mismatch
/// when the policy parts from its oracle. Prints the time taken.
fn every_hand_tally(what: &str, f: impl Fn(DominoSet) -> (String, Option<String>) + Sync) -> Tally {
    let hands: Vec<DominoSet> = every_hand().collect();
    // One core on every 97th hand first, for the cost of a hand.
    let started = Instant::now();
    let mut sample = 0;
    for hand in hands.iter().step_by(97) {
        f(*hand);
        sample += 1;
    }
    let each = started.elapsed() / sample as u32;
    println!("{what}: {each:?} a hand on one core ({sample} hands)");
    let started = Instant::now();
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    let parts: Vec<Tally> = std::thread::scope(|s| {
        let handles: Vec<_> = hands
            .chunks(hands.len().div_ceil(threads))
            .map(|part| {
                let f = &f;
                s.spawn(move || {
                    let mut tally = Tally::default();
                    for hand in part {
                        let (outcome, mismatch) = f(*hand);
                        tally.add(outcome, mismatch);
                    }
                    tally
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let mut total = Tally::default();
    for part in parts {
        total.merge(part);
    }
    println!("{what}: {} hands in {:?} on {threads} threads: {:?}", total.hands, started.elapsed(), total.outcomes);
    assert_eq!(total.hands, 1_184_040);
    assert!(total.mismatches.is_empty(), "{what}: {:#?}", total.mismatches);
    total
}

fn unlimited() -> Budget {
    Budget::new(u64::MAX)
}

// -------------------------------------------------- the arena's arithmetic

/// The arena's `evaluate_trump` (arena/hand_metrics.py) under pip `t`: the
/// trump length, whether the hand holds the trump double, and the count its
/// offs expose (each count tile the hand lacks, not a trump, sharing a pip
/// with an off).
fn arena_eval(hand: DominoSet, t: Pip) -> (usize, bool, u32) {
    let trumps = hand.iter().filter(|d| d.has(t)).count();
    let double = hand.contains(Domino::new(t, t));
    let offs: Vec<Domino> = hand.iter().filter(|d| !d.has(t) && !d.is_double()).collect();
    let exposed: u32 = Domino::ALL
        .iter()
        .filter(|c| c.pip_sum() % 5 == 0 && c.pip_sum() > 0)
        .filter(|c| !hand.contains(**c) && !c.has(t))
        .filter(|c| offs.iter().any(|o| c.has(o.hi()) || c.has(o.lo())))
        .map(|c| u32::from(c.pip_sum()))
        .sum();
    (trumps, double, exposed)
}

/// The arena's `best_trump(hand, min)`: the most willing suit (42 less the
/// exposed count), then the longest, then one holding its double, then the
/// higher pip.
fn arena_best(hand: DominoSet, min: usize) -> Option<(Pip, u32)> {
    Pip::ALL
        .into_iter()
        .map(|t| (t, arena_eval(hand, t)))
        .filter(|(_, (n, _, _))| *n >= min)
        .max_by_key(|(t, (n, double, exposed))| (42 - *exposed as i32, *n, *double, t.value()))
        .map(|(t, (_, _, exposed))| (t, 42 - exposed))
}

/// `HeuristicBidder(min_trumps=3, caution=0).bid`: the least legal bid of
/// one mark or less when the best suit's ceiling reaches it, else pass.
fn arena_bid(hand: DominoSet, auction: &Auction) -> Call {
    let Some((_, ceiling)) = arena_best(hand, 3) else { return Call::Pass };
    let willing = ceiling.min(42);
    match auction.least_raise() {
        Some(bid) if bid.value() <= 42 && bid.value() <= willing => Call::Bid(bid),
        _ => Call::Pass,
    }
}

/// `HeuristicBidder.declare`: the best suit of three or more, else of any length.
fn arena_declare(hand: DominoSet) -> Pip {
    arena_best(hand, 3).or_else(|| arena_best(hand, 0)).unwrap().0
}

fn pip_of(physics: &Physics) -> Pip {
    match Trump::of(physics) {
        Some(Trump::Suit(p)) => p,
        other => panic!("{} is not a pip trump: {other:?}", physics.name()),
    }
}

/// The frame where `viewer`, dealt `hand`, has won the auction with `bid`
/// and the other three passed: where he declares.
fn won(viewer: Seat, hand: DominoSet, bid: Bid, menu: &Arc<[Physics]>) -> AuctionFrame {
    let auction = Auction::won_by(viewer, bid).unwrap();
    AuctionFrame::new(auction, viewer, hand, MatchState::new(Scoring::MARKS, [0, 0]).unwrap(), menu.clone()).unwrap()
}

// ------------------------------------------------ bid only enough = the arena

#[test]
fn bid_only_enough_is_the_heuristic_bidder_on_every_hand() {
    let policy = policy("bid-only-enough.silas");
    let menu = arena_menu();
    let tally = every_hand_tally("bid-only-enough, the first call", |hand| {
        let frame = first_call(Seat::S0, hand, &menu);
        let call = policy.bid(&frame, &mut unlimited()).unwrap().choice;
        let expected = arena_bid(hand, frame.auction());
        (call.to_string(), (call != expected).then(|| format!("{hand:?}: {call}, the arena {expected}")))
    });
    // S-0053-d's biddable share under the arena's arithmetic: 403,197 of
    // 1,184,040 hands open (34.05%); the rest pass.
    assert_eq!((tally.of("30"), tally.of("pass")), (403_197, 780_843));
}

#[test]
fn bid_only_enough_declares_the_heuristic_bidders_suit_on_every_hand() {
    let policy = policy("bid-only-enough.silas");
    let menu = arena_menu();
    every_hand_tally("bid-only-enough, the declaration", |hand| {
        let frame = won(Seat::S2, hand, Bid::Points(30), &menu);
        let choice = policy.declare(&frame, &mut unlimited()).unwrap();
        let (declared, expected) = (pip_of(&choice.choice), arena_declare(hand));
        let fallback = choice.provenance == DecisionProvenance::Fallback;
        let outcome = format!("P{declared}{}", if fallback { " (fallback)" } else { "" });
        (outcome, (declared != expected).then(|| format!("{hand:?}: P{declared}, the arena P{expected}")))
    });
}

// --------------------------------------------- the risk budget, as we read it

/// Our reading of the book's count under pip trump `t`
/// (`risk-budget.silas`): each off's trick, the count of the suits the offs
/// lead (their high pips) that the hand lacks and is not trump, each count
/// tile once, and without the trump double one trump trick with the count
/// trumps the hand lacks. Also the number of offs.
fn book_risk(hand: DominoSet, t: Pip) -> (u32, usize) {
    let offs: Vec<Domino> = hand.iter().filter(|d| !d.has(t) && !d.is_double()).collect();
    let double = hand.contains(Domino::new(t, t));
    let points: u32 = Domino::ALL
        .iter()
        .filter(|c| c.pip_sum() % 5 == 0 && c.pip_sum() > 0 && !hand.contains(**c))
        .filter(|c| if c.has(t) { !double } else { offs.iter().any(|o| c.has(o.hi())) })
        .map(|c| u32::from(c.pip_sum()))
        .sum();
    (points + offs.len() as u32 + u32::from(!double), offs.len())
}

/// The book's 84 shape under `t` (S-0083-a), as we read it: three or four
/// trumps with the trump double, two or three doubles besides, at most one
/// off, under its double.
fn eighty_four_shape(hand: DominoSet, t: Pip) -> bool {
    let trumps = hand.iter().filter(|d| d.has(t)).count();
    let doubles = hand.iter().filter(|d| d.is_double() && !d.has(t)).count();
    let offs: Vec<Domino> = hand.iter().filter(|d| !d.has(t) && !d.is_double()).collect();
    (3..=4).contains(&trumps)
        && hand.contains(Domino::new(t, t))
        && (2..=3).contains(&doubles)
        && offs.len() <= 1
        && offs.iter().all(|o| hand.contains(Domino::new(o.hi(), o.hi())))
}

/// The opening call of our reading: 84 on the shape; else 42 less the least
/// risk of a suit of three or more, when at most 12; else 30 with two offs or
/// more and at most 17 at risk; else pass.
fn book_call(hand: DominoSet) -> Call {
    let suits: Vec<Pip> = Pip::ALL.into_iter().filter(|t| hand.iter().filter(|d| d.has(*t)).count() >= 3).collect();
    if suits.iter().any(|t| eighty_four_shape(hand, *t)) {
        return Call::Bid(Bid::Marks(2));
    }
    let risks: Vec<(u32, usize)> = suits.iter().map(|t| book_risk(hand, *t)).collect();
    match risks.iter().map(|(r, _)| *r).min() {
        Some(r) if r <= 12 => Call::Bid(Bid::number(42 - r).unwrap()),
        _ if risks.iter().any(|(r, offs)| *r <= 17 && *offs >= 2) => Call::Bid(Bid::Points(30)),
        _ => Call::Pass,
    }
}

#[test]
fn risk_budget_is_its_reading_on_every_hand() {
    let policy = policy("risk-budget.silas");
    let menu = arena_menu();
    let tally = every_hand_tally("risk-budget, the first call", |hand| {
        let frame = first_call(Seat::S1, hand, &menu);
        let call = policy.bid(&frame, &mut unlimited()).unwrap().choice;
        let expected = book_call(hand);
        (call.to_string(), (call != expected).then(|| format!("{hand:?}: {call}, the reading {expected}")))
    });
    // The exact histogram of the opening call (S-0053-d, S-0153-a): about
    // half of all hands pass (647,781, 54.7%); of the 536,259 that bid, 30 is
    // nearly half, the book's reasoned bids 30, 31, 35 and 36 are 61.5%, and
    // 32 and 33 are 5.3%.
    let expected = [
        ("pass", 647_781),
        ("30", 260_860),
        ("31", 15_465),
        ("32", 4_180),
        ("33", 24_311),
        ("34", 49_637),
        ("35", 42_795),
        ("36", 10_686),
        ("37", 1_993),
        ("38", 14_302),
        ("39", 34_533),
        ("40", 36_101),
        ("41", 11_422),
        ("42", 3_409),
        ("84", 26_565),
    ];
    assert_eq!(tally.outcomes, expected.iter().map(|(k, v)| (k.to_string(), *v)).collect());
    let bids = tally.hands - tally.of("pass");
    let reasoned: usize = ["30", "31", "35", "36"].iter().map(|c| tally.of(c)).sum();
    assert_eq!((bids, reasoned, tally.of("32") + tally.of("33")), (536_259, 329_806, 28_491));
}

// ------------------------------------------- the arena's own auctions, replayed

/// One deal of `data/heuristic-bidder.txt`: the four hands, and per shaker the
/// arena's four calls and its declaration, or a passed-out hand and the
/// forced shaker's declaration.
struct ArenaDeal {
    hands: [DominoSet; 4],
    auctions: Vec<(Vec<Call>, bool, Pip)>,
}

fn arena_deals() -> Vec<ArenaDeal> {
    include_str!("data/heuristic-bidder.txt")
        .lines()
        .filter(|l| !l.starts_with('#'))
        .map(|line| {
            let mut cells = line.split(" | ");
            let deal: Vec<Domino> = cells
                .next()
                .unwrap()
                .split(' ')
                .map(|t| {
                    let pip = |c: u8| Pip::new(c - b'0').unwrap();
                    Domino::new(pip(t.as_bytes()[0]), pip(t.as_bytes()[1]))
                })
                .collect();
            let hands = std::array::from_fn(|s| deal[7 * s..7 * s + 7].iter().copied().collect());
            let auctions = cells
                .map(|cell| {
                    let words: Vec<&str> = cell.split(' ').collect();
                    let calls = words[..4].iter().map(|w| w.parse().unwrap()).collect();
                    let out = words[4] == "out";
                    let pip = Pip::new(words.last().unwrap().parse().unwrap()).unwrap();
                    (calls, out, pip)
                })
                .collect();
            ArenaDeal { hands, auctions }
        })
        .collect()
}

#[test]
fn bid_only_enough_replays_the_arenas_auctions_call_for_call() {
    let policy = policy("bid-only-enough.silas");
    let menu = arena_menu();
    let state = MatchState::new(Scoring::MARKS, [0, 0]).unwrap();
    let deals = arena_deals();
    assert_eq!(deals.len(), 400);
    let (mut raised, mut forced) = (0, 0);
    for (k, deal) in deals.iter().enumerate() {
        for (shaker, (calls, out, declared)) in Seat::ALL.into_iter().zip(&deal.auctions) {
            let config = |forced| AuctionConfig::new(5, forced).unwrap();
            let mut auction = Auction::new(config(false), shaker, vec![]).unwrap();
            for expected in calls {
                let seat = auction.actor().unwrap();
                let frame = AuctionFrame::new(auction.clone(), seat, deal.hands[seat.index()], state, menu.clone());
                let call = policy.bid(&frame.unwrap(), &mut unlimited()).unwrap().choice;
                assert_eq!(&call, expected, "deal {k}, {seat} after {auction}");
                auction = auction.after(call).unwrap();
            }
            raised += usize::from(calls.iter().filter(|c| **c != Call::Pass).count() > 1);
            // A passed-out hand under the forced bid: the shaker's least legal
            // call is 30, the arena's forced bid.
            if *out {
                forced += 1;
                let three = Auction::new(config(true), shaker, calls[..3].to_vec()).unwrap();
                let frame = AuctionFrame::new(three.clone(), shaker, deal.hands[shaker.index()], state, menu.clone());
                let choice = policy.bid(&frame.unwrap(), &mut unlimited()).unwrap();
                assert_eq!(
                    (choice.choice, choice.provenance),
                    (Call::Bid(Bid::Points(30)), DecisionProvenance::Fallback)
                );
                auction = three.after(choice.choice).unwrap();
            }
            let (winner, _) = auction.result().unwrap();
            let frame = AuctionFrame::new(auction.clone(), winner, deal.hands[winner.index()], state, menu.clone());
            let choice = policy.declare(&frame.unwrap(), &mut unlimited()).unwrap();
            assert_eq!(pip_of(&choice.choice), *declared, "deal {k}: {winner} declares after {auction}");
        }
    }
    // The fixture exercises raises and forced bids, not only openings.
    assert!(raised > 100 && forced > 100, "{raised} raised, {forced} forced");
}

// ------------------------------------------- every call and declaration, pinned

/// FNV-1a, 64 bits: a digest of a long text, so a pin is one number.
fn fnv1a(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |h, b| (h ^ u64::from(*b)).wrapping_mul(0x0100_0000_01b3))
}

/// What the three bidding policies say about one hand: each one's call
/// opening the auction, over a 31 and over a 36, and the declarations of
/// `bid-only-enough` (won at 30) and `risk-budget` (won at 30 and at 84).
fn every_answer(policies: &[CompiledPolicy; 3], menu: &Arc<[Physics]>, hand: DominoSet) -> String {
    let state = MatchState::new(Scoring::MARKS, [0, 0]).unwrap();
    let mut line = String::new();
    for (k, policy) in policies.iter().enumerate() {
        for before in [vec![], vec![Call::Bid(Bid::Points(31))], vec![Call::Bid(Bid::Points(36))]] {
            let auction = Auction::new(AuctionConfig::STRAIGHT, Seat::S3, before).unwrap();
            let viewer = auction.actor().unwrap();
            let frame = AuctionFrame::new(auction, viewer, hand, state, menu.clone()).unwrap();
            line += &format!("{} ", policy.bid(&frame, &mut unlimited()).unwrap().choice);
        }
        let bids: &[Bid] = match k {
            0 => &[Bid::Points(30)],
            1 => &[Bid::Points(30), Bid::Marks(2)],
            _ => &[],
        };
        for bid in bids {
            line += &format!(
                "P{} ",
                pip_of(&policy.declare(&won(Seat::S2, hand, *bid, menu), &mut unlimited()).unwrap().choice)
            );
        }
        line += "| ";
    }
    line
}

#[test]
fn every_call_and_declaration_is_pinned_on_every_hand() {
    // The calls and declarations of bid-only-enough, risk-budget and
    // never-32-or-33 on all 1,184,040 hands, as one digest of one line a
    // hand. The digest was taken from the encodings before the aggregate
    // (gap 5, 2026-10-13), which spelled each bound as `none` scopes over the
    // count tiles; the encodings with `card` and `points` give the same line
    // on every hand.
    let policies = [policy("bid-only-enough.silas"), policy("risk-budget.silas"), policy("never-32-or-33.silas")];
    let menu = arena_menu();
    let stride = proof_stride();
    let hands: Vec<DominoSet> = every_hand().step_by(stride).collect();
    let started = Instant::now();
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    let lines: Vec<String> = std::thread::scope(|s| {
        let handles: Vec<_> = hands
            .chunks(hands.len().div_ceil(threads))
            .map(|part| {
                let (policies, menu) = (&policies, &menu);
                s.spawn(move || part.iter().map(|hand| every_answer(policies, menu, *hand)).collect::<Vec<_>>())
            })
            .collect();
        handles.into_iter().flat_map(|h| h.join().unwrap()).collect()
    });
    println!("every answer of three policies: {} hands in {:?} on {threads} threads", lines.len(), started.elapsed());
    assert_eq!(lines.len(), 1_184_040_usize.div_ceil(stride));
    let digest = fnv1a(lines.join("\n").as_bytes());
    println!("digest {digest:#018x}; first line: {}", lines[0]);
    assert_eq!(digest, if stride == 1 { PINNED_ANSWERS } else { PINNED_ANSWERS_4TH });
}

/// The digest of the same lines on every 4th hand (the default run; see
/// [`proof_stride`]).
const PINNED_ANSWERS_4TH: u64 = 0x25ef_df9b_e692_f220;

/// The digest of [`every_call_and_declaration_is_pinned_on_every_hand`]'s
/// lines.
const PINNED_ANSWERS: u64 = 0x927d_9038_f897_51c5;

/// The raising rules as written before the least raise was a number (#144):
/// a rule for each range of it, an `or` of its literals.
const LITERAL_RANGES: [&str; 2] = [
    "(include bidding)
    (policy bid-only-enough
      (bidding
        (rule raise-to-32 (bid least-raise)
          (or (least-raise 30) (least-raise 31) (least-raise 32))
          (some (roles t) (suit-at-least t 3) (exposes-at-most t 10)))
        (rule raise-to-37 (bid least-raise)
          (or (least-raise 33) (least-raise 34) (least-raise 35) (least-raise 36) (least-raise 37))
          (some (roles t) (suit-at-least t 3) (exposes-at-most t 5)))
        (rule raise-to-42 (bid least-raise)
          (or (least-raise 38) (least-raise 39) (least-raise 40) (least-raise 41) (least-raise 42))
          (some (roles t) (suit-at-least t 3) (exposes-at-most t 0)))))",
    "(include bidding)
    (policy never-32-or-33
      (bidding
        (rule raise-to-31 (bid least-raise)
          (or (least-raise 30) (least-raise 31))
          (some (roles t) (suit-at-least t 3) (exposes-at-most t 10)))
        (rule past-33 (bid 34)
          (or (least-raise 32) (least-raise 33))
          (some (roles t) (suit-at-least t 3) (exposes-at-most t 5)))
        (rule raise-to-37 (bid least-raise)
          (or (least-raise 34) (least-raise 35) (least-raise 36) (least-raise 37))
          (some (roles t) (suit-at-least t 3) (exposes-at-most t 5)))
        (rule raise-to-42 (bid least-raise)
          (or (least-raise 38) (least-raise 39) (least-raise 40) (least-raise 41) (least-raise 42))
          (some (roles t) (suit-at-least t 3) (exposes-at-most t 0)))))",
];

/// An auction at every least raise a seat can face, the viewer to call: 30
/// (opening) to 41 and 42 over a points bid, 84, 126 and 168 over marks, and
/// none left once the marks cap is bid.
fn every_least_raise() -> Vec<Auction> {
    let bid = |n: u32| Call::Bid(Bid::number(n).unwrap());
    let straight = |calls: Vec<Call>| Auction::new(AuctionConfig::STRAIGHT, Seat::S3, calls).unwrap();
    let mut auctions: Vec<Auction> =
        (29..=42).map(|high| straight(if high < 30 { vec![] } else { vec![bid(high)] })).collect();
    auctions.push(straight(vec![bid(42), bid(84)]));
    auctions.push(straight(vec![bid(42), bid(84), bid(126)]));
    auctions.push(Auction::new(AuctionConfig::new(2, false).unwrap(), Seat::S3, vec![bid(84)]).unwrap());
    auctions
}

/// The calls of `policies` on `hand` at each of `auctions`, in one line.
fn raises(policies: &[CompiledPolicy], auctions: &[Auction], menu: &Arc<[Physics]>, hand: DominoSet) -> String {
    let state = MatchState::new(Scoring::MARKS, [0, 0]).unwrap();
    let mut line = String::new();
    for auction in auctions {
        let frame = AuctionFrame::new(auction.clone(), auction.actor().unwrap(), hand, state, menu.clone()).unwrap();
        for policy in policies {
            line += &format!("{} ", policy.bid(&frame, &mut unlimited()).unwrap().choice);
        }
    }
    line
}

#[test]
fn a_raise_reads_the_least_raise_as_a_number_and_calls_as_its_literals_did() {
    // bid-only-enough's one raising rule ("the least raise and the count
    // exposed come to 42 at most") and never-32-or-33's two, on every hand at
    // every least raise (`every_least_raise`): one digest of one line a hand,
    // taken from the rules of literal ranges they replace (`LITERAL_RANGES`,
    // before #144). Those rules are run again on every 97th hand. About a
    // minute on 18 cores at `SILAS_FULL_PROOF=1` (every 4th hand otherwise): it is the proof (40 million calls), and it costs
    // what the literal ranges did.
    let auctions = every_least_raise();
    let least: Vec<Option<u32>> = auctions.iter().map(|a| a.least_raise().map(Bid::value)).collect();
    let mut expected: Vec<Option<u32>> = (30..=42).chain([84, 126, 168]).map(Some).collect();
    expected.push(None);
    assert_eq!(least, expected);
    let menu = arena_menu();
    let new = [policy("bid-only-enough.silas"), policy("never-32-or-33.silas")];
    let old: Vec<CompiledPolicy> = LITERAL_RANGES
        .iter()
        .map(|text| {
            PolicyProgram::parse(text, "literal ranges").unwrap().compile(&Registry::standard(), &physics("P6"))
        })
        .collect::<Result<_>>()
        .unwrap();
    let all: Vec<DominoSet> = every_hand().collect();
    let sample: Vec<DominoSet> = all.iter().copied().step_by(97).collect();
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    std::thread::scope(|s| {
        for part in sample.chunks(sample.len().div_ceil(threads)) {
            let (new, old, auctions, menu) = (&new, &old, &auctions, &menu);
            s.spawn(move || {
                for hand in part {
                    assert_eq!(raises(new, auctions, menu, *hand), raises(old, auctions, menu, *hand), "{hand:?}");
                }
            });
        }
    });
    let stride = proof_stride();
    let hands: Vec<DominoSet> = all.iter().copied().step_by(stride).collect();
    let started = Instant::now();
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    let lines: Vec<String> = std::thread::scope(|s| {
        let handles: Vec<_> = hands
            .chunks(hands.len().div_ceil(threads))
            .map(|part| {
                let (new, auctions, menu) = (&new, &auctions, &menu);
                s.spawn(move || part.iter().map(|hand| raises(new, auctions, menu, *hand)).collect::<Vec<_>>())
            })
            .collect();
        handles.into_iter().flat_map(|h| h.join().unwrap()).collect()
    });
    println!("the raises at every least raise: {} hands in {:?} on {threads} threads", lines.len(), started.elapsed());
    assert_eq!(lines.len(), 1_184_040_usize.div_ceil(stride));
    let mut distinct: BTreeMap<&str, usize> = BTreeMap::new();
    for line in &lines {
        *distinct.entry(line).or_default() += 1;
    }
    println!("{distinct:#?}");
    // Four kinds of hand: worth 42 (no count exposed), 37, 32, and none of
    // them; bid-only-enough raises to its worth, never-32-or-33 past 32.
    assert_eq!(distinct.values().copied().collect::<Vec<_>>().len(), 4);
    assert_eq!(fnv1a(lines.join("\n").as_bytes()), if stride == 1 { PINNED_RAISES } else { PINNED_RAISES_4TH });
}

/// The digest of the same lines on every 4th hand (the default run; see
/// [`proof_stride`]).
const PINNED_RAISES_4TH: u64 = 0x84fa_fe77_74bf_5370;

/// The digest of [`a_raise_reads_the_least_raise_as_a_number_and_calls_as_its_literals_did`]'s
/// lines, from the rules of literal ranges.
const PINNED_RAISES: u64 = 0x51e1_17e1_0ab5_8281;

/// `risk-budget` as written before the computed call (#148): a rule per
/// level, 42 down to 30, each its own literal call, and the prelude's
/// `less-at-risk` it declared by (gone from the prelude since; `(< (risk u)
/// (risk t))` says it).
const LITERAL_LEVELS: &str = "(include bidding)
    (define less-at-risk (roles (context u) (context t))
      (<
        (+
          (points (roles c) (risk-tile c u))
          (card (roles o) (off-under o u))
          (card (roles) (not (suit-double u))))
        (+
          (points (roles c) (risk-tile c t))
          (card (roles o) (off-under o t))
          (card (roles) (not (suit-double t))))))
    (policy risk-budget
      (bidding
        (rule eighty-four (bid 84)
          (some (roles t) (eighty-four-shape t)))
        (rule one-mark (bid 42)
          (some (roles t) (suit-at-least t 3) (risk-at-most t 0)))
        (rule bid-41 (bid 41)
          (some (roles t) (suit-at-least t 3) (risk-at-most t 1)))
        (rule bid-40 (bid 40)
          (some (roles t) (suit-at-least t 3) (risk-at-most t 2)))
        (rule bid-39 (bid 39)
          (some (roles t) (suit-at-least t 3) (risk-at-most t 3)))
        (rule bid-38 (bid 38)
          (some (roles t) (suit-at-least t 3) (risk-at-most t 4)))
        (rule bid-37 (bid 37)
          (some (roles t) (suit-at-least t 3) (risk-at-most t 5)))
        (rule bid-36 (bid 36)
          (some (roles t) (suit-at-least t 3) (risk-at-most t 6)))
        (rule bid-35 (bid 35)
          (some (roles t) (suit-at-least t 3) (risk-at-most t 7)))
        (rule bid-34 (bid 34)
          (some (roles t) (suit-at-least t 3) (risk-at-most t 8)))
        (rule bid-33 (bid 33)
          (some (roles t) (suit-at-least t 3) (risk-at-most t 9)))
        (rule bid-32 (bid 32)
          (some (roles t) (suit-at-least t 3) (risk-at-most t 10)))
        (rule bid-31 (bid 31)
          (some (roles t) (suit-at-least t 3) (risk-at-most t 11)))
        (rule bid-30 (bid 30)
          (some (roles t) (suit-at-least t 3) (risk-at-most t 12)))
        (rule two-offs-at-17 (bid 30)
          (some (roles t) (suit-at-least t 3) (not (offs-at-most t 1)) (risk-at-most t 17))))
      (declaring
        (rule the-eighty-four-suit (declare t)
          (viewer v)
          (bid-of v 84)
          (eighty-four-shape t))
        (rule least-at-risk (declare t)
          (suit-at-least t 3)
          (risk-at-most t 12)
          (none (roles u) (suit-at-least u 3) (less-at-risk u t)))
        (rule at-risk-17 (declare t)
          (suit-at-least t 3)
          (risk-at-most t 17))))";

/// The calls of `policy` on `hand` at each of `auctions`, and its
/// declarations won at 30, at 42 and at 84, in one line.
fn calls_and_declarations(
    policy: &CompiledPolicy,
    auctions: &[Auction],
    menu: &Arc<[Physics]>,
    hand: DominoSet,
) -> String {
    let mut line = raises(std::slice::from_ref(policy), auctions, menu, hand);
    for bid in [Bid::Points(30), Bid::Marks(1), Bid::Marks(2)] {
        let declared = policy.declare(&won(Seat::S2, hand, bid, menu), &mut unlimited()).unwrap().choice;
        line += &format!("P{} ", pip_of(&declared));
    }
    line
}

#[test]
fn risk_budget_computes_its_call_as_its_thirteen_levels_did() {
    // risk-budget's one priced rule, (bid (- 42 (risk t))) on the least at
    // risk, and its declarations, at every least raise (30 to 42, 84, 126,
    // 168 and none left) and won at 30, 42 and 84: one digest of one line a
    // hand, the same for the thirteen literal levels it replaces
    // (`LITERAL_LEVELS`), which are run again on every 997th hand. On all
    // 1,184,040 hands (`STRIDE` 1) both encodings' lines digest to
    // 0xe811_d0a0_6259_d24f: 24 million decisions, two and a half minutes on
    // 18 cores, run once when the rule was written (2026-10-14). The suite
    // keeps every 8th hand, about 20 s; the all-hands pin of the opening,
    // over 31 and 36, and the declarations at 30 and 84 is
    // `every_call_and_declaration_is_pinned_on_every_hand`.
    let auctions = every_least_raise();
    let menu = arena_menu();
    let new = policy("risk-budget.silas");
    let old = PolicyProgram::parse(LITERAL_LEVELS, "literal levels")
        .unwrap()
        .compile(&Registry::standard(), &physics("P6"))
        .unwrap();
    // The literal levels are the old file's core, as pinned before #148.
    assert_eq!(fnv1a(old.source().to_string().as_bytes()), 0xe5bf_4c07_ddfe_e065);
    let sample: Vec<DominoSet> = every_hand().step_by(997).collect();
    std::thread::scope(|s| {
        for part in sample.chunks(sample.len().div_ceil(18)) {
            let (new, old, auctions, menu) = (&new, &old, &auctions, &menu);
            s.spawn(move || {
                for hand in part {
                    assert_eq!(
                        calls_and_declarations(new, auctions, menu, *hand),
                        calls_and_declarations(old, auctions, menu, *hand),
                        "{hand:?}"
                    );
                }
            });
        }
    });
    let hands: Vec<DominoSet> = every_hand().step_by(STRIDE).collect();
    let started = Instant::now();
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    let lines: Vec<String> = std::thread::scope(|s| {
        let handles: Vec<_> = hands
            .chunks(hands.len().div_ceil(threads))
            .map(|part| {
                let (new, auctions, menu) = (&new, &auctions, &menu);
                s.spawn(move || {
                    part.iter().map(|hand| calls_and_declarations(new, auctions, menu, *hand)).collect::<Vec<_>>()
                })
            })
            .collect();
        handles.into_iter().flat_map(|h| h.join().unwrap()).collect()
    });
    println!("risk-budget at every least raise: {} hands in {:?} on {threads} threads", lines.len(), started.elapsed());
    assert_eq!(lines.len(), 1_184_040_usize.div_ceil(STRIDE));
    let digest = fnv1a(lines.join("\n").as_bytes());
    println!("digest {digest:#018x}; first line: {}", lines[0]);
    assert_eq!(digest, PINNED_LEVELS);
}

/// Every how many hands [`risk_budget_computes_its_call_as_its_thirteen_levels_did`]
/// reads: 1 is every hand.
const STRIDE: usize = 8;

/// The digest of [`risk_budget_computes_its_call_as_its_thirteen_levels_did`]'s
/// lines on every `STRIDE`th hand, the literal levels' and the rule's alike.
const PINNED_LEVELS: u64 = 0x567b_14a0_afa5_22e4;

// ------------------------------------------------------------- measurability

#[test]
fn a_bid_is_a_function_of_the_auction_frame_alone() {
    // The frame holds the viewer's hand, the auction, the match and the
    // declarations offered, nothing of another seat; and a bidding rule may
    // name no predicate that reads a world, nor one of the play.
    let menu = arena_menu();
    for (rule, why) in [
        ("(bidding (rule r (bid 30) (holds S1 6-6)))", "world predicate holds"),
        ("(bidding (rule r (bid 30) (declarer S0)))", "uses declarer, a predicate of the play"),
        ("(bidding (rule r (bid 30) (own-legal 6-6)))", "uses own-legal, a predicate of the play"),
        ("(declaring (rule r (declare t) (trump t)))", "uses trump, a predicate of the play"),
        ("(rule r (play d) (own-legal d) (least-raise 30))", "uses least-raise, a predicate of the auction"),
        ("(bidding (rule r (play d) (mine d)))", "not a bidding rule's short form"),
        ("(declaring (rule r (bid 30) (no-bid)))", "not a declaring rule's short form"),
        ("(bidding (rule r (select tile 6-6) (guard (query (no-bid)))))", "selects bid"),
        ("(declaring (rule r (declare q*) (no-bid)))", "q0 to q6"),
        ("(bidding (rule r (bid 29) (no-bid)))", "least-raise"),
    ] {
        let text = format!("(policy p {rule})");
        let refused = text.parse::<PolicyProgram>().and_then(|p| p.compile(&Registry::standard(), &physics("P6")));
        let Err(Error(message)) = refused else { panic!("{text} compiled") };
        assert!(message.contains(why) || message.contains("not a"), "{text}: {message}");
    }
    // Same frame, same call: the decision reads nothing else (the declaring
    // rules' answers too), and a different deal of the other hands cannot
    // reach it, since the frame does not hold them.
    let policy = policy("bid-only-enough.silas");
    let mut rng = Rng(7);
    for _ in 0..200 {
        let deck = rng.shuffled();
        let hand: DominoSet = deck[..7].iter().copied().collect();
        let frame = first_call(Seat::S2, hand, &menu);
        let a = policy.explain_bid(&frame, &mut unlimited()).unwrap();
        let b = policy.explain_bid(&frame.clone(), &mut unlimited()).unwrap();
        assert_eq!(a, b);
    }
}

#[test]
fn bidding_sections_print_as_they_read_and_short_forms_are_their_core() {
    let short = "(policy p (bidding (rule r (bid least-raise) (least-raise 30))) (declaring (rule s (declare t) (suit-at-least t 3))))";
    let explicit = "(policy p (bidding (rule r (select bid least-raise) (guard (query (least-raise 30))))) \
                    (declaring (rule s (select declare answer) (guard (query (out t) (suit-at-least t 3))))))";
    let short = PolicyProgram::parse(&format!("(include bidding) {short}"), "short").unwrap();
    let explicit = PolicyProgram::parse(&format!("(include bidding) {explicit}"), "explicit").unwrap();
    assert_eq!(short, explicit);
    assert_eq!(PolicyProgram::parse(&short.to_string(), "printed").unwrap(), short);
    for (form, selector) in [
        ("(bid 30)", Selector::Call(Call::Bid(Bid::Points(30)))),
        ("(bid 84)", Selector::Call(Call::Bid(Bid::Marks(2)))),
        ("(bid pass)", Selector::Call(Call::Pass)),
        ("(bid least-raise)", Selector::LeastRaise),
    ] {
        let program: PolicyProgram = format!("(policy p (bidding (rule r {form} (no-bid))))").parse().unwrap();
        assert_eq!(program.bidding[0].selector, selector, "{form}");
    }
    for (form, declared) in [
        ("(declare q5)", Declared::Suit(Pip::new(5).unwrap())),
        ("(declare doubles)", Declared::Doubles),
        ("(declare no-trump)", Declared::NoTrump),
    ] {
        let program: PolicyProgram = format!("(policy p (declaring (rule r {form} (no-bid))))").parse().unwrap();
        assert_eq!(program.declaring[0].selector, Selector::Declare(declared), "{form}");
    }
    // A policy of the play alone prints as it did before the auction.
    let play = "(policy p (rule r (play d) (own-legal d)))".parse::<PolicyProgram>().unwrap();
    assert!(!play.to_string().contains("bidding"));
}

#[test]
fn declaring_names_doubles_and_no_trump_and_falls_back_to_the_first_offered() {
    let menu: Arc<[Physics]> =
        ["NT", "DT", "P0", "P1", "P2", "P3", "P4", "P5", "P6"].iter().map(|d| physics(d)).collect();
    let hand = tiles(&["6-6", "5-5", "4-4", "3-3", "2-2", "1-1", "0-0"]);
    let frame = won(Seat::S1, hand, Bid::Points(30), &menu);
    let decide = |text: &str| {
        let program: PolicyProgram = text.parse().unwrap();
        let policy = program.compile(&Registry::standard(), &physics("P6")).unwrap();
        let choice = policy.declare(&frame, &mut unlimited()).unwrap();
        (Trump::of(&choice.choice), choice.provenance)
    };
    let rule = |name: &str| DecisionProvenance::RelationalRule { name: name.to_owned() };
    assert_eq!(
        decide("(policy p (declaring (rule d (declare doubles) (mine 6-6))))"),
        (Some(Trump::Doubles), rule("d"))
    );
    assert_eq!(decide("(policy p (declaring (rule n (declare no-trump) (mine 6-6))))"), (Some(Trump::None), rule("n")));
    // Every suit answered: the first offered of them, P0 here.
    let all = "(policy p (declaring (rule s (declare t) (some (roles d) (mine d) (bears d t)))))";
    assert_eq!(decide(all), (Some(Trump::Suit(Pip::new(0).unwrap())), rule("s")));
    // Nothing allowed: the first offered, no trump.
    assert_eq!(
        decide("(policy p (declaring (rule s (declare q4) (mine 6-5))))"),
        (Some(Trump::None), DecisionProvenance::Fallback)
    );
    // Only the winner declares, and only once the auction is over.
    let program: PolicyProgram = "(policy p)".parse().unwrap();
    let policy = program.compile(&Registry::standard(), &physics("P6")).unwrap();
    let loser = AuctionFrame::new(frame.auction().clone(), Seat::S2, hand, *frame.state(), menu.clone()).unwrap();
    assert!(policy.declare(&loser, &mut unlimited()).is_err());
    assert!(policy.bid(&frame, &mut unlimited()).is_err(), "no one is to call once the auction is over");
}

// --------------------------------------------------------------- the bid book

/// The bid book of the 2026-10-12 study (wiki: book-study-2026-10-12): 30
/// hands with the call the book prints for each. The tiles are
/// `examples/book/hands.json`'s (the reviewed transcription; the study read
/// Hand 20 from the prose as a variant, the figure's tiles are these); a
/// single hand opens the auction, and the seats of the two full deals call in
/// the deal's own order: Hand 28's bidder opens 30 and the others pass (p.
/// 97), Hand 32's opponent-1 opens 35, the partner and opponent-2 pass and the
/// bidder, the shaker, bids 36 (p. 108). Each is (hand, role, its seat in
/// the calling order, the printed call).
const BID_BOOK: [(&str, &str, usize, &str); 30] = [
    ("hand-1", "bidder", 0, "36"),
    ("hand-2", "bidder", 0, "31"),
    ("hand-3", "bidder", 0, "pass"),
    ("hand-4", "bidder", 0, "35"),
    ("hand-5", "bidder", 0, "30"),
    ("hand-6", "bidder", 0, "30"),
    ("hand-7", "bidder", 0, "31"),
    ("hand-9", "bidder", 0, "30"),
    ("hand-10", "bidder", 0, "pass"),
    ("hand-11", "bidder", 0, "30"),
    ("hand-16", "bidder", 0, "31"),
    ("hand-18", "bidder", 0, "84"),
    ("hand-19", "bidder", 0, "84"),
    ("hand-20", "bidder", 0, "84"),
    ("hand-21", "bidder", 0, "84"),
    ("hand-22", "bidder", 0, "84"),
    ("hand-23", "bidder", 0, "31"),
    ("hand-24", "bidder", 0, "30"),
    ("hand-25", "bidder", 0, "30"),
    ("hand-26", "bidder", 0, "35"),
    ("hand-34", "bidder", 0, "84"),
    ("hand-36", "bidder", 0, "30"),
    ("hand-28", "bidder", 0, "30"),
    ("hand-28", "opponent-1", 1, "pass"),
    ("hand-28", "partner", 2, "pass"),
    ("hand-28", "opponent-2", 3, "pass"),
    ("hand-32", "opponent-1", 0, "35"),
    ("hand-32", "partner", 1, "pass"),
    ("hand-32", "opponent-2", 2, "pass"),
    ("hand-32", "bidder", 3, "36"),
];

/// The calls before each seat of the two deals, in calling order.
fn deal_calls(hand: &str) -> &'static [&'static str] {
    match hand {
        "hand-28" => &["30", "pass", "pass", "pass"],
        "hand-32" => &["35", "pass", "pass", "36"],
        _ => &[],
    }
}

/// A hand of `examples/book/hands.json`: the tiles of a seat, its printed
/// at-risk count and the book's declaration.
fn book_seat(hand: &str, role: &str) -> (DominoSet, Option<u32>, Option<String>) {
    let json: serde_json::Value = serde_json::from_str(include_str!("../examples/book/hands.json")).unwrap();
    let example = json["examples"].as_array().unwrap().iter().find(|e| e["id"] == hand).unwrap();
    let seat = example["seats"].as_array().unwrap().iter().find(|s| s["role"] == role).unwrap();
    let tiles = seat["tiles"].as_array().unwrap().iter().map(|t| t.as_str().unwrap().parse().unwrap()).collect();
    let at_risk = example["at_risk"].as_u64().map(|r| r as u32);
    (tiles, at_risk, example["declaration"].as_str().map(str::to_owned))
}

#[test]
fn risk_budget_on_the_bid_books_thirty_hands() {
    let policy = policy("risk-budget.silas");
    let menu: Arc<[Physics]> = Straight42::ALL.iter().map(|d| d.physics()).collect();
    let mut rows = Vec::new();
    for (hand, role, position, printed) in BID_BOOK {
        let (tiles, _, _) = book_seat(hand, role);
        let before: Vec<Call> = deal_calls(hand)[..position].iter().map(|c| c.parse().unwrap()).collect();
        let mut calls = BTreeMap::new();
        // Every chair the seat could sit in, under both scorings: the call
        // reads neither (the rule never asks the score).
        for viewer in Seat::ALL {
            for state in
                [MatchState::new(Scoring::MARKS, [0, 0]).unwrap(), MatchState::new(Scoring::POINTS, [0, 0]).unwrap()]
            {
                let auction = Auction::new(AuctionConfig::STRAIGHT, viewer.plus(3 - position), before.clone()).unwrap();
                let frame = AuctionFrame::new(auction, viewer, tiles, state, menu.clone()).unwrap();
                let call = policy.bid(&frame, &mut unlimited()).unwrap().choice;
                *calls.entry(call.to_string()).or_insert(0) += 1;
            }
        }
        assert_eq!(calls.len(), 1, "{hand} {role}: {calls:?}");
        let ours = calls.into_keys().next().unwrap();
        // The call is our reading's, where the seat opens.
        if position == 0 && hand != "hand-32" {
            assert_eq!(ours, book_call(tiles).to_string(), "{hand}");
        }
        rows.push((hand, role, printed, ours));
    }
    let agree = |f: &dyn Fn(&str, &str) -> bool| rows.iter().filter(|(_, _, p, o)| f(p, o)).count();
    let bids = |c: &str| c != "pass";
    let same_call = agree(&|p, o| p == o);
    let bid_or_pass = agree(&|p, o| bids(p) == bids(o));
    // Against the chance baselines: always bid agrees on bid-or-pass where the
    // book bids (23 of 30); the book's commonest call, 30, on its 30s (8).
    let always_bid = rows.iter().filter(|(_, _, p, _)| bids(p)).count();
    let thirties = rows.iter().filter(|(_, _, p, _)| *p == "30").count();
    for (hand, role, printed, ours) in &rows {
        println!("{hand:>8} {role:<11} book {printed:>4}  risk-budget {ours:>4}");
    }
    println!(
        "same call {same_call} of 30 (chance: 30 always, {thirties}); bid or pass {bid_or_pass} of 30 (always bid: {always_bid})"
    );
    assert_eq!((always_bid, thirties), (23, 8));
    assert_eq!((same_call, bid_or_pass), (18, 22));
}

#[test]
fn the_reading_against_the_books_printed_risk_counts() {
    // hands.json carries the risk the book prints for twelve pip-trump hands.
    // Our reading's count under the book's trump agrees on eight; the four it
    // misses are the book pricing by judgment: an off's other pip (Hand 4's
    // 5-0), a 6-4 it does not count (Hand 8), a weak trump suit with its
    // double (Hand 9), and a protected off (Hand 10).
    let mut agree = Vec::new();
    let mut differ = Vec::new();
    for n in 1..=36 {
        let hand = format!("hand-{n}");
        let json: serde_json::Value = serde_json::from_str(include_str!("../examples/book/hands.json")).unwrap();
        let Some(example) = json["examples"].as_array().unwrap().iter().find(|e| e["id"] == hand.as_str()) else {
            continue;
        };
        if example["seats"].as_array().unwrap().iter().all(|s| s["role"] != "bidder") {
            continue;
        }
        let (tiles, at_risk, declaration) = book_seat(&hand, "bidder");
        let (Some(printed), Some(decl)) = (at_risk, declaration) else { continue };
        let Some(pip) = decl.strip_prefix('P').and_then(|p| p.parse().ok()).and_then(Pip::new) else { continue };
        let (ours, _) = book_risk(tiles, pip);
        if ours == printed { agree.push(n) } else { differ.push((n, printed, ours)) }
    }
    assert_eq!(agree, vec![1, 2, 3, 5, 6, 7, 11, 26]);
    assert_eq!(differ, vec![(4, 7, 2), (8, 8, 7), (9, 17, 16), (10, 16, 32)]);
}

#[test]
fn never_32_or_33_raises_past_them() {
    // S-0152-c, S-0153-b: never 32 or 33. Over every legal auction prefix
    // where a seat is to call and 400 seeded hands, the policy never bids
    // either; where bid-only-enough would raise to 32 or 33, it bids 34 or
    // passes.
    let never = policy("never-32-or-33.silas");
    let only = policy("bid-only-enough.silas");
    let menu = arena_menu();
    let state = MatchState::new(Scoring::MARKS, [0, 0]).unwrap();
    let prefixes: Vec<Vec<Call>> = include_str!("data/auction-legal.txt")
        .lines()
        .filter(|l| !l.starts_with('#'))
        .map(|l| {
            l.split_once(':').unwrap().0.split_whitespace().filter(|w| *w != "-").map(|w| w.parse().unwrap()).collect()
        })
        .collect();
    let mut rng = Rng(32);
    let hands: Vec<DominoSet> = (0..400).map(|_| rng.shuffled()[..7].iter().copied().collect()).collect();
    let (mut skipped, mut checked) = (0, 0);
    for prefix in &prefixes {
        let auction = Auction::new(AuctionConfig::STRAIGHT, Seat::S3, prefix.clone()).unwrap();
        let viewer = auction.actor().unwrap();
        for hand in &hands {
            let frame = AuctionFrame::new(auction.clone(), viewer, *hand, state, menu.clone()).unwrap();
            let call = never.bid(&frame, &mut unlimited()).unwrap().choice;
            let theirs = only.bid(&frame, &mut unlimited()).unwrap().choice;
            assert!(!matches!(call, Call::Bid(Bid::Points(32 | 33))), "{call} after {auction}");
            match theirs {
                Call::Bid(Bid::Points(32 | 33)) => {
                    skipped += 1;
                    assert!(matches!(call, Call::Pass | Call::Bid(Bid::Points(34))), "{call} for {theirs}");
                }
                _ => assert_eq!(call, theirs),
            }
            checked += 1;
        }
    }
    assert!(skipped > 1000, "{skipped} of {checked}");
}

// ------------------------------------------- the populations' bidders (#141)

fn population(file: &str) -> CompiledPolicy {
    let path = format!("{}/examples/populations/{file}", env!("CARGO_MANIFEST_DIR"));
    let program = PolicyProgram::parse(&std::fs::read_to_string(&path).unwrap(), &path).unwrap();
    program.compile(&Registry::standard(), &physics("P6")).unwrap()
}

/// The suits of which `hand` holds the two highest tiles, written in Rust
/// from the literals the optimists were first anchored to: the double, and
/// the six of the suit (the 6-5 in sixes).
fn top_two_suits(hand: DominoSet) -> Vec<Pip> {
    let pip = |v: u8| Pip::new(v).unwrap();
    Pip::ALL
        .into_iter()
        .filter(|&t| {
            let next = if t.value() == 6 { pip(5) } else { pip(6) };
            hand.contains(Domino::new(t, t)) && hand.contains(Domino::new(t, next))
        })
        .collect()
}

/// The optimists' declaration as first written, a rule for each length,
/// before `least` (`examples/populations/optimist-2.silas`).
const CASCADE: &str = "(include bidding)
    (policy cascade
      (declaring
        (rule two-tiles (declare t) (holds-top t 2) (not (suit-at-least t 3)))
        (rule three-tiles (declare t) (holds-top t 2) (not (suit-at-least t 4)))
        (rule four-tiles (declare t) (holds-top t 2) (not (suit-at-least t 5)))
        (rule more (declare t) (holds-top t 2))))";

#[test]
fn the_populations_are_their_reading_on_every_hand() {
    // The optimists bid 30 on the top two of two (three) suits and declare
    // the qualifying suit with the fewest tiles, the higher pip on a tie;
    // the marginal 30 bids on 13 or 14 at risk under a suit of three or
    // more. Each against its reading in Rust, and the optimists' `least`
    // against the cascade of rules it replaces, on every hand.
    let (two, three, marginal) =
        (population("optimist-2.silas"), population("optimist-3.silas"), population("marginal-30.silas"));
    let cascade =
        PolicyProgram::parse(CASCADE, "cascade").unwrap().compile(&Registry::standard(), &physics("P6")).unwrap();
    let menu = arena_menu();
    let tally = every_hand_tally("the populations' bidders", |hand| {
        let frame = first_call(Seat::S0, hand, &menu);
        let call = |p: &CompiledPolicy| p.bid(&frame, &mut unlimited()).unwrap().choice;
        let suits = top_two_suits(hand);
        let thirty = |yes: bool| if yes { Call::Bid(Bid::Points(30)) } else { Call::Pass };
        let length = |t: Pip| hand.iter().filter(|d| d.has(t)).count();
        let marginal_suit = |t: Pip| length(t) >= 3 && (13..=14).contains(&book_risk(hand, t).0);
        let expected =
            [thirty(suits.len() >= 2), thirty(suits.len() >= 3), thirty(Pip::ALL.into_iter().any(marginal_suit))];
        let calls = [call(&two), call(&three), call(&marginal)];
        let outcome = format!("{} {} {}", calls[0], calls[1], calls[2]);
        let mut mismatch = (calls != expected).then(|| format!("{hand:?}: {calls:?}, expected {expected:?}"));
        // The declaration, where a suit qualifies (elsewhere no declaring
        // rule speaks, and the fallback declares the first offered).
        if let Some(fewest) = suits.iter().copied().min_by_key(|&t| (length(t), 6 - t.value())) {
            let won = won(Seat::S2, hand, Bid::Points(30), &menu);
            let declared = pip_of(&two.declare(&won, &mut unlimited()).unwrap().choice);
            let cascaded = pip_of(&cascade.declare(&won, &mut unlimited()).unwrap().choice);
            if declared != fewest || cascaded != declared {
                mismatch = Some(format!("{hand:?}: declares P{declared}, the cascade P{cascaded}, expected P{fewest}"));
            }
        }
        (outcome, mismatch)
    });
    // How often each opens, exactly: optimist-2 on 4.2% of hands, optimist-3
    // on 0.16%, the marginal 30 on 13.7%.
    let opens = |k: usize| -> usize {
        tally.outcomes.iter().filter(|(o, _)| o.split(' ').nth(k) == Some("30")).map(|(_, n)| n).sum()
    };
    assert_eq!((opens(0), opens(1), opens(2)), (49_310, 1_895, 161_946));
}

// ------------------------------------------- the book's overbidders (#163)

/// The frame where `viewer`, dealt `hand`, calls after `calls` from the seat
/// `first` counted from him (1 his left, 3 his right).
fn after_calls(viewer: Seat, hand: DominoSet, first: usize, calls: &[Call], menu: &Arc<[Physics]>) -> AuctionFrame {
    let shaker = viewer.plus(first + 3);
    let auction = Auction::new(AuctionConfig::STRAIGHT, shaker, calls.to_vec()).unwrap();
    AuctionFrame::new(auction, viewer, hand, MatchState::new(Scoring::MARKS, [0, 0]).unwrap(), menu.clone()).unwrap()
}

/// Whether `hand` holds a suit of three or more under which an off (its tile,
/// not a double, not of the suit) shares a pip with a ten-count tile it lacks
/// that is not of the suit: the 5-5 or the 6-4. Written in Rust.
fn ten_count_off(hand: DominoSet) -> bool {
    let tens = [
        Domino::new(Pip::new(5).unwrap(), Pip::new(5).unwrap()),
        Domino::new(Pip::new(6).unwrap(), Pip::new(4).unwrap()),
    ];
    Pip::ALL.into_iter().any(|t| {
        hand.iter().filter(|d| d.has(t)).count() >= 3
            && tens.iter().any(|&c| {
                !hand.contains(c)
                    && !c.has(t)
                    && hand
                        .iter()
                        .any(|o| !o.has(t) && !o.is_double() && Pip::ALL.into_iter().any(|p| o.has(p) && c.has(p)))
            })
    })
}

#[test]
fn the_overbidders_are_their_reading_on_every_hand() {
    // The set-bidding fallacy (set-bid) opens exactly as bid-only-enough
    // does: no opponent has bid. The habitual overbidder (ten-count-35)
    // opens 35 exactly on a suit of three with a ten-count off, and as
    // bid-only-enough elsewhere.
    let (set_bid, ten, enough) =
        (population("set-bid.silas"), population("ten-count-35.silas"), policy("bid-only-enough.silas"));
    let menu = arena_menu();
    let tally = every_hand_tally("the overbidders", |hand| {
        let frame = first_call(Seat::S0, hand, &menu);
        let call = |p: &CompiledPolicy| p.bid(&frame, &mut unlimited()).unwrap().choice;
        let (s, t, e) = (call(&set_bid), call(&ten), call(&enough));
        let expected_ten = if ten_count_off(hand) { Call::Bid(Bid::Points(35)) } else { e };
        let mismatch = (s != e || t != expected_ten)
            .then(|| format!("{hand:?}: set-bid {s}, ten-count-35 {t}, bid-only-enough {e}, expected {expected_ten}"));
        (format!("{t}"), mismatch)
    });
    let thirty_fives: usize = tally.outcomes.iter().filter(|(o, _)| o.as_str() == "35").map(|(_, n)| n).sum();
    assert!(thirty_fives > 0);
}

#[test]
fn the_set_bidding_fallacy_raises_over_an_opponent_only() {
    // On 2,000 seeded hands: over the partner's 30 the fallacy calls what
    // bid-only-enough calls; over an opponent's 30 it raises wherever
    // bid-only-enough does, and on some hands bid-only-enough passes it
    // raises anyway: the hands worth 25 to 30 points to bid-only-enough.
    let menu = arena_menu();
    let (set_bid, enough) = (population("set-bid.silas"), policy("bid-only-enough.silas"));
    let thirty = Call::Bid(Bid::Points(30));
    let call = |p: &CompiledPolicy, f: &AuctionFrame| p.bid(f, &mut unlimited()).unwrap().choice;
    let mut rng = Rng(163);
    let mut only_the_fallacy = 0;
    for _ in 0..2000 {
        let hand: DominoSet = rng.shuffled()[..7].iter().copied().collect();
        let over_partner = after_calls(Seat::S0, hand, 2, &[thirty, Call::Pass], &menu);
        assert_eq!(call(&set_bid, &over_partner), call(&enough, &over_partner), "{hand:?} over the partner");
        let over_opponent = after_calls(Seat::S0, hand, 3, &[thirty], &menu);
        let (s, e) = (call(&set_bid, &over_opponent), call(&enough, &over_opponent));
        if e != Call::Pass {
            assert_eq!(s, e, "{hand:?} over an opponent");
        } else if s != Call::Pass {
            assert_eq!(s, Call::Bid(Bid::Points(31)), "{hand:?}");
            only_the_fallacy += 1;
        }
    }
    assert!(only_the_fallacy > 50, "{only_the_fallacy}");
}
