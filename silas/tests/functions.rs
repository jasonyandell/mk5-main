//! Functions are numbers (#144; README, Semantics): a predicate functional in
//! its last argument, a number, read without it, `(to-set)` or `(count d)`, is
//! that number in a comparison. Held to the relation it reads on frames of
//! the play and the auction in all three truth values; its refusals; the
//! chains `(<= a b c)`; and the book prelude's definitions rewritten with it,
//! each held to the family of literals it replaced on every frame of seeded
//! hands at every bid.
mod common;

use std::sync::Arc;

use common::*;
use silas::*;

fn query(text: &str) -> Query {
    text.parse().unwrap_or_else(|e| panic!("{text}\n{e}"))
}

fn same_query(short: &str, explicit: &str) {
    let (a, b) = (query(short), query(explicit));
    assert_eq!(a, b, "\n{short}\n{explicit}");
    assert_eq!(a.to_string(), b.to_string());
    assert_eq!(query(&a.to_string()), a, "the core reads back as itself");
}

fn refused(text: &str) -> String {
    match text.parse::<Query>() {
        Ok(q) => q.compile(&Registry::standard()).map(|_| String::new()).unwrap_or_else(|e| e.0),
        Err(e) => e.0,
    }
}

/// Seeded frames of random deals at every bid a contract can carry, played
/// at random, from every seat.
fn frames_at_every_bid(deals: usize) -> Vec<Frame> {
    let bids: Vec<u32> = (30..=41).chain([42, 84, 126, 168, 210]).collect();
    let mut rng = Rng(0x144);
    let mut out = Vec::new();
    for game in 0..deals {
        let physics = reference::Straight42::ALL[game % 9].physics();
        let deck = rng.shuffled();
        let deal: [DominoSet; 4] = std::array::from_fn(|s| deck[7 * s..7 * s + 7].iter().copied().collect());
        let contract = contract(&physics, Seat::ALL[rng.below(4)], bids[game % bids.len()]);
        let mut hands = deal;
        let mut record = framed(&contract, Seat::S0, deal[0], vec![]).unwrap();
        loop {
            for viewer in Seat::ALL {
                out.push(framed(&contract, viewer, deal[viewer.index()], record.plays().to_vec()).unwrap());
            }
            let Some(actor) = record.next_actor() else { break };
            let legal = physics.legal_plays(hands[actor.index()], record.lead());
            let tile = legal.iter().nth(rng.below(legal.len())).unwrap();
            hands[actor.index()].remove(tile);
            record = record.after(tile).unwrap();
        }
    }
    out
}

fn holds(compiled: &CompiledQuery, frame: &Frame) -> Answers {
    compiled.evaluate_viewer(frame, &mut budget()).unwrap()
}

/// A yes-or-no query's truth: true, false, or undefined (it and its
/// negation both fail).
fn truth(text: &str, frame: &Frame) -> Option<bool> {
    thread_local! {
        /// Each text compiled once, with its negation.
        static COMPILED: std::cell::RefCell<std::collections::HashMap<String, Arc<(CompiledQuery, CompiledQuery)>>> =
            Default::default();
    }
    let both = COMPILED.with(|c| {
        c.borrow_mut()
            .entry(text.to_owned())
            .or_insert_with(|| {
                let body = text.strip_prefix("(query ").and_then(|t| t.strip_suffix(')')).unwrap();
                Arc::new((compile(text), compile(&format!("(query (not {body}))"))))
            })
            .clone()
    });
    let (yes, no) = &*both;
    match (!holds(yes, frame).is_empty(), !holds(no, frame).is_empty()) {
        (true, false) => Some(true),
        (false, true) => Some(false),
        (false, false) => None,
        (true, true) => panic!("{text} and its negation both hold"),
    }
}

#[test]
fn a_function_is_the_number_its_relation_holds_for_in_all_three_values() {
    // Each zero-argument function of the play against its literals: (<= (p)
    // n) is true exactly when (p m) holds for an m <= n, false when for an m
    // > n, and its negation is (> (p) n), exactly.
    let frames = frames_at_every_bid(17);
    let mut values = std::collections::BTreeMap::<&str, std::collections::BTreeSet<u32>>::new();
    for frame in frames.iter().step_by(5) {
        for p in ["bid", "stake", "to-make", "to-set", "trick-number"] {
            let value = (0..=300).find(|n| truth(&format!("(query ({p} {n}))"), frame) == Some(true));
            let undefined = truth(&format!("(query ({p} 1))"), frame).is_none();
            assert!(value.is_some() || undefined, "{p} holds for no number at {frame:?}");
            for n in [0, 1, 6, 7, 13, 35, 41, 42, 84] {
                let compared = truth(&format!("(query (<= ({p}) {n}))"), frame);
                assert_eq!(compared, value.map(|v| v <= n), "(<= ({p}) {n}) at {frame:?}");
                assert_eq!(truth(&format!("(query (> ({p}) {n}))"), frame), compared.map(|c| !c));
                assert_eq!(truth(&format!("(query (= ({p}) {n}))"), frame), value.map(|v| v == n));
            }
            if let Some(v) = value {
                values.entry(p).or_default().insert(v);
            }
        }
    }
    // The frames reach every bid, and to-set and trick-number run their range.
    assert_eq!(values["bid"].len(), 17);
    assert!(values["to-set"].len() > 10 && values["trick-number"].len() == 7, "{values:?}");
}

#[test]
fn a_function_of_roles_reads_them_and_sorts_them() {
    // (count d) is d's count: points, tile by tile. Its role is a domino by
    // its use, so the query needs no declaration.
    let frame = &frames_at_every_bid(1)[0];
    let tens = compile("(query (out d) (= (count d) 10))");
    let found: Vec<String> = holds(&tens, frame).iter().map(|a| a.0[0].to_string()).collect();
    assert_eq!(found, ["5-5", "6-4"]);
    same_query(
        "(query (out d) (= (count d) 10))",
        "(query (roles (domino d)) (out d) (pattern (<= (count d) 10) (<= 10 (count d))))",
    );
    // Sums and aggregates mix with it: a tile and its partner in count.
    let fifteen = compile("(query (out d e) (double d) (= (+ (count d) (count e)) 15))");
    let pairs: Vec<String> = holds(&fifteen, frame).iter().map(|a| format!("{} {}", a.0[0], a.0[1])).collect();
    assert_eq!(pairs, ["5-5 3-2", "5-5 4-1", "5-5 5-0"]);
}

#[test]
fn no_number_leaves_a_comparison_undefined_and_negation_exact() {
    // At the auction: no high bid before the first bid, no least raise for a
    // seat not to call, and none left once the marks cap is bid. Each
    // comparison reading such a function is undefined, as is its negation;
    // the literals it replaces are false (no bid) or undefined (not to call).
    let menu: Arc<[Physics]> = ["P6", "P5"].iter().map(|d| physics(d)).collect();
    let state = MatchState::new(Scoring::MARKS, [0, 0]).unwrap();
    let hand: DominoSet = Rng(7).shuffled()[..7].iter().copied().collect();
    let at = |config: AuctionConfig, calls: Vec<Call>, viewer: Seat| {
        let auction = Auction::new(config, Seat::S3, calls).unwrap();
        AuctionFrame::new(auction, viewer, hand, state, menu.clone()).unwrap()
    };
    let decide = |text: &str, frame: &AuctionFrame| {
        let body = text.strip_prefix("(query ").and_then(|t| t.strip_suffix(')')).unwrap();
        let yes = compile(text).evaluate_auction(frame, &mut budget()).unwrap();
        let no = compile(&format!("(query (not {body}))")).evaluate_auction(frame, &mut budget()).unwrap();
        match (yes.is_empty(), no.is_empty()) {
            (false, true) => Some(true),
            (true, false) => Some(false),
            (true, true) => None,
            (false, false) => panic!("{text} and its negation both hold"),
        }
    };
    let bid = |n: u32| Call::Bid(Bid::number(n).unwrap());
    let opening = at(AuctionConfig::STRAIGHT, vec![], Seat::S0);
    let waiting = at(AuctionConfig::STRAIGHT, vec![], Seat::S1);
    let capped = at(AuctionConfig::new(2, false).unwrap(), vec![bid(84)], Seat::S1);
    let raised = at(AuctionConfig::STRAIGHT, vec![bid(31)], Seat::S1);
    for (frame, high, least) in
        [(&opening, None, Some(30)), (&waiting, None, None), (&capped, Some(84), None), (&raised, Some(31), Some(32))]
    {
        for n in [30, 31, 32, 84] {
            assert_eq!(decide(&format!("(query (<= (high-bid) {n}))"), frame), high.map(|h| h <= n));
            assert_eq!(decide(&format!("(query (<= (least-raise) {n}))"), frame), least.map(|l| l <= n));
            assert_eq!(decide(&format!("(query (> (least-raise) {n}))"), frame), least.map(|l| l > n));
        }
    }
    assert_eq!(decide("(query (least-raise 30))", &waiting), None);
    assert_eq!(decide("(query (least-raise 126))", &capped), Some(false));
    assert_eq!(decide("(query (high-bid 30))", &opening), Some(false));
    // A tile not yet played was played to no trick: (trick-of d) has no number.
    let frame = &frames_at_every_bid(1)[0];
    assert_eq!(truth("(query (<= (trick-of 6-6) 7))", frame), None);
    assert_eq!(truth("(query (trick-of 6-6 1))", frame), Some(false));
}

#[test]
fn chains_read_as_lisp_reads_them() {
    same_query("(query (<= 1 (to-set) 6))", "(query (<= 1 (to-set)) (<= (to-set) 6))");
    same_query("(query (< 0 (to-set) 7))", "(query (<= 1 (to-set)) (<= (+ (to-set) 1) 7))");
    same_query("(query (not (<= 32 (bid) 33)))", "(query (none (roles) (<= 32 (bid)) (<= (bid) 33)))");
    same_query("(query (= 1 (stake) 1))", "(query (<= 1 (stake)) (<= (stake) 1) (<= (stake) 1) (<= 1 (stake)))");
    assert!(refused("(query (/= 1 (stake) 2))").contains("/= compares two numbers"));
}

#[test]
fn a_number_that_is_not_a_function_is_refused_at_its_use() {
    assert!(refused("(query (<= (holds S1) 3))").contains("holds is not functional"));
    assert!(refused("(query (<= (live) 3))").contains("live is not functional"));
    assert!(refused("(query (<= (to-set 3) 4))").contains("to-set expects 0 arguments, got 1"));
    assert!(refused("(query (<= (count S1) 4))").contains("count expects domino, got chair"));
    assert!(refused("(query (<= (no-such) 4))").contains("no-such"));
    assert!(refused("(include book) (query (<= (every-trick) 4))").contains("every-trick ...) is a definition"));
}

#[test]
fn the_core_prints_a_function_and_reads_it_back() {
    let q = query("(include book) (query (out d) (lead-sets d))");
    let core = q.to_string();
    assert!(
        core.contains("(<= 1 (to-set))") && core.contains("(<= (to-set) (+ (count d) (count called) 1))"),
        "{core}"
    );
    assert_eq!(query(&core), q);
}

/// The book prelude's definitions before #144: a family of literals for each
/// range of the bid and of what the setters need, renamed apart from the
/// prelude's own.
const LITERAL_FAMILIES: &str = "(include book)
(define old-every-trick (roles)
  (or (bid 42) (bid 84) (bid 126) (bid 168) (bid 210) (bid 252) (bid 294)))
(define old-bid-35-or-more (roles)
  (or (bid 35) (bid 36) (bid 37) (bid 38) (bid 39) (bid 40) (bid 41) (old-every-trick)))
(define to-set-within-6 (roles)
  (or (to-set 1) (to-set 2) (to-set 3) (to-set 4) (to-set 5) (to-set 6)))
(define to-set-within-11 (roles)
  (or (to-set-within-6) (to-set 7) (to-set 8) (to-set 9) (to-set 10) (to-set 11)))
(define to-set-within-16 (roles)
  (or (to-set-within-11) (to-set 12) (to-set 13) (to-set 14) (to-set 15) (to-set 16)))
(define to-set-within-21 (roles)
  (or (to-set-within-16) (to-set 17) (to-set 18) (to-set 19) (to-set 20) (to-set 21)))
(define sets-with-a-five (roles (domino d))
  (count d 0) (calls-for d 5) (to-set-within-6))
(define sets-with-a-ten (roles (domino d))
  (count d 0) (calls-for d 10) (to-set-within-11))
(define five-sets-with-a-five (roles (domino d))
  (count d 5) (calls-for d 5) (to-set-within-11))
(define five-sets-with-a-ten (roles (domino d))
  (count d 5) (calls-for d 10) (to-set-within-16))
(define ten-sets-with-a-five (roles (domino d))
  (count d 10) (calls-for d 5) (to-set-within-16))
(define ten-sets-with-a-ten (roles (domino d))
  (count d 10) (calls-for d 10) (to-set-within-21))
(define old-lead-sets (roles (domino d))
  (or (sets-with-a-five d) (sets-with-a-ten d) (five-sets-with-a-five d)
      (five-sets-with-a-ten d) (ten-sets-with-a-five d) (ten-sets-with-a-ten d)))
";

#[test]
fn the_book_preludes_definitions_mean_what_their_literals_meant() {
    // every-trick, the bid of 35 or more (setter-leads.silas) and lead-sets,
    // each against the family of literals it replaced, in all three truth
    // values: on every frame of 170 seeded hands, ten at each bid a contract
    // carries, from every seat, every domino for lead-sets. Each reads the
    // frame only (the bid, what the setters need, the count, the live tiles
    // and the viewer's hand), so these frames are its domain.
    let pair = |old: &str, new: &str| {
        let yes =
            |body: &str| compile(&format!("{LITERAL_FAMILIES}(query (roles (domino d)) (out d) (pattern {body}))"));
        [(yes(old), yes(new)), (yes(&format!("(not {old})")), yes(&format!("(not {new})")))]
    };
    let pairs = [
        pair("(old-every-trick)", "(every-trick)"),
        pair("(old-bid-35-or-more)", "(<= 35 (bid))"),
        pair("(old-lead-sets d)", "(lead-sets d)"),
    ];
    let frames = frames_at_every_bid(170);
    let mut sets = 0;
    for frame in &frames {
        for (k, both) in pairs.iter().enumerate() {
            for (old, new) in both {
                let (was, is) = (holds(old, frame), holds(new, frame));
                assert_eq!(was, is, "definition {k} at {frame:?}");
            }
        }
        sets += holds(&pairs[2][0].1, frame).len();
    }
    assert_eq!(frames.len(), 170 * 29 * 4);
    assert!(sets > 1000, "lead-sets answered {sets} times");
}
