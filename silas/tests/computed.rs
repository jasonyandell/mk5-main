//! A computed call (#148; README, The auction): a bidding rule whose call is
//! a number of its guard's answer, `(bid (- 42 (risk t)))`. Held here: the
//! difference `(- a b ...)` as an interval, its negation exact and its caps
//! sound; a definition whose body is one number, read wherever a number is;
//! and the computed call's semantics, each case against the arithmetic it
//! means: the rule allows, at each answer, the bid the table calls the
//! number's one value, when legal, and calls the least; a number that is no
//! bid, no legal bid, or no number at all is no call, and the rule passes to
//! the next. `tests/bidding.rs` holds `risk-budget`'s one rule to its thirteen
//! literal levels on every hand at every least raise.
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

/// Three doubles; sixes four long, fives three, ones and twos two, fours one,
/// treys and blanks one.
fn hand() -> DominoSet {
    tiles(&["6-6", "5-5", "4-4", "3-1", "2-0", "6-2", "6-5"])
}

fn menu() -> Arc<[Physics]> {
    ["P6", "P5", "P4", "P3", "P2", "P1", "P0"].iter().map(|d| physics(d)).collect()
}

fn bid(n: u32) -> Call {
    Call::Bid(Bid::number(n).unwrap())
}

/// The hand's frame with the viewer to call after `calls` (S3 shook), or
/// with `viewer` waiting.
fn frame_after(config: AuctionConfig, calls: Vec<Call>, viewer: Option<Seat>) -> AuctionFrame {
    let auction = Auction::new(config, Seat::S3, calls).unwrap();
    let viewer = viewer.unwrap_or_else(|| auction.actor().unwrap());
    AuctionFrame::new(auction, viewer, hand(), MatchState::new(Scoring::MARKS, [0, 0]).unwrap(), menu()).unwrap()
}

/// True, false or undefined at an auction frame.
fn decide(text: &str, frame: &AuctionFrame) -> Option<bool> {
    let head = text.find("(query ").unwrap();
    let body = &text[head + "(query ".len()..text.len() - 1];
    let yes = compile(text).evaluate_auction(frame, &mut budget()).unwrap();
    let negated = format!("{}(query (not {body}))", &text[..head]);
    let no = compile(&negated).evaluate_auction(frame, &mut budget()).unwrap();
    match (yes.is_empty(), no.is_empty()) {
        (false, true) => Some(true),
        (true, false) => Some(false),
        (true, true) => None,
        (false, false) => panic!("{text} and its negation both hold"),
    }
}

fn policy(text: &str) -> CompiledPolicy {
    let program = PolicyProgram::parse(text, "policy").unwrap_or_else(|e| panic!("{text}\n{e}"));
    program.compile(&Registry::standard(), &physics("P6")).unwrap_or_else(|e| panic!("{text}\n{e}"))
}

/// The call of a policy of `rules` (a bidding section, `(include bidding)`
/// in force) at `frame`, and the rule that made it.
fn call(rules: &str, frame: &AuctionFrame) -> (Call, String) {
    let policy = policy(&format!("(include bidding) (policy p (bidding {rules}))"));
    let choice = policy.bid(frame, &mut budget()).unwrap();
    let by = match choice.provenance {
        DecisionProvenance::RelationalRule { name } => name,
        _ => "fallback".to_owned(),
    };
    (choice.choice, by)
}

// ----------------------------------------------------------- differences

#[test]
fn a_difference_is_an_interval_and_its_negation_is_exact() {
    let opening = frame_after(AuctionConfig::STRAIGHT, vec![], None);
    let waiting = frame_after(AuctionConfig::STRAIGHT, vec![], Some(Seat::S1));
    let raised = frame_after(AuctionConfig::STRAIGHT, vec![bid(31)], None);
    // 42 less the least raise, against every bound and below zero: the
    // integers, not the naturals. Where the least raise has no number (the
    // viewer not to call) the difference has none, and the comparison and
    // its negation are undefined.
    for (frame, least) in [(&opening, Some(30i64)), (&waiting, None), (&raised, Some(32))] {
        for n in [0u32, 9, 10, 11, 12, 13] {
            let text = format!("(query (<= (- 42 (least-raise)) {n}))");
            assert_eq!(decide(&text, frame), least.map(|l| 42 - l <= i64::from(n)), "{text}");
            let text = format!("(query (> (- 42 (least-raise)) {n}))");
            assert_eq!(decide(&text, frame), least.map(|l| 42 - l > i64::from(n)), "{text}");
        }
        for (text, holds) in [
            ("(query (<= (- 30 (least-raise)) 0))", least.map(|l| 30 - l <= 0)),
            ("(query (< (- 30 (least-raise)) 0))", least.map(|l| 30 - l < 0)),
            ("(query (= (+ (- 30 (least-raise)) 2) 0))", least.map(|l| 32 - l == 0)),
        ] {
            assert_eq!(decide(text, frame), holds, "{text}");
        }
    }
    // Over the hand, a count less a count, where a part past a cap could
    // still fall: seven tiles less three doubles is four, at every bound.
    for n in 0..=8u32 {
        let text = format!("(query (<= {n} (+ (card (roles d) (mine d)) (- 0 (card (roles d) (mine d) (double d))))))");
        assert_eq!(decide(&text, &opening), Some(n <= 4), "{text}");
        let text = format!("(query (<= (- (card (roles d) (mine d)) (card (roles d) (mine d) (double d))) {n}))");
        assert_eq!(decide(&text, &opening), Some(4 <= n), "{text}");
    }
}

#[test]
fn a_difference_prints_and_reads_back_and_takes_two_numbers_or_more() {
    same_query("(query (<= (- 42 (to-set)) 6))", "(query (<= (- 42 (to-set)) 6))");
    same_query("(query (< (- 42 (to-set) 1) 6))", "(query (<= (+ (- 42 (to-set) 1) 1) 6))");
    same_query("(query (<= 1 (- 42 (bid)) 12))", "(query (<= 1 (- 42 (bid))) (<= (- 42 (bid)) 12))");
    assert!(refused("(query (<= (- 42) 6))").contains("- takes a number less one or more others"));
    assert!(refused("(query (- 42 1))").contains("is a number, not a clause"));
}

// ----------------------------------------------------- number definitions

#[test]
fn a_definition_of_one_number_is_that_number() {
    // The prelude's risk is a number; risk-at-most is (<= (risk t) n), and
    // reads to the core it read to when it spelled the sum out.
    same_query("(include bidding) (query (<= (risk q6) 5))", "(include bidding) (query (risk-at-most q6 5))");
    same_query(
        "(define plus-two (roles (number n)) (+ n 2)) (query (<= (plus-two 3) (to-set)))",
        "(query (<= (+ 3 2) (to-set)))",
    );
    same_query(
        "(define doubles-held (roles) (card (roles d) (mine d) (double d))) (query (<= 2 (doubles-held)))",
        "(query (<= 2 (card (roles d) (mine d) (double d))))",
    );
    // Its role takes its sort from the call, as a function's does.
    let q = query("(include bidding) (query (out t) (<= (risk t) 5))");
    assert_eq!(q.roles[0].sort, Sort::Context);
    // A number is not a clause, negated or not.
    for text in ["(include bidding) (query (risk q6))", "(include bidding) (query (not (risk q6)))"] {
        assert!(refused(text).contains("(risk ...) is a number, not a clause"), "{}", refused(text));
    }
    assert!(refused("(include bidding) (query (<= (risk q6 3) 5))").contains("risk takes 1 argument(s), got 2"));
    // The risk of each suit of the hand, as tests/bidding.rs counts it.
    let opening = frame_after(AuctionConfig::STRAIGHT, vec![], None);
    let risk = |q: &str| {
        (0..=60).find(|n| decide(&format!("(include bidding) (query (= (risk {q}) {n}))"), &opening) == Some(true))
    };
    assert!(Pip::ALL.iter().all(|p| risk(&Value::Context(Context::Natural(*p)).to_string()).is_some()));
}

// --------------------------------------------------------- computed calls

#[test]
fn a_computed_call_is_the_bid_its_number_names_when_legal() {
    let doubles = "(card (roles d) (mine d) (double d))";
    let at = |calls: Vec<Call>| frame_after(AuctionConfig::STRAIGHT, calls, None);
    // Thirty and a point a double: 33, legal opening and over 32.
    let rule = format!("(rule priced (bid (+ 30 {doubles})) (viewer v))");
    assert_eq!(call(&rule, &at(vec![])), (bid(33), "priced".to_owned()));
    assert_eq!(call(&rule, &at(vec![bid(32)])), (bid(33), "priced".to_owned()));
    // Over 33 it is no legal bid: the rule passes to the next, as a literal
    // level does, and never clamps to a legal one.
    let next = format!("{rule} (rule then (bid 41) (viewer v))");
    assert_eq!(call(&next, &at(vec![bid(33)])), (bid(41), "then".to_owned()));
    assert_eq!(call(&rule, &at(vec![bid(33)])), (Call::Pass, "fallback".to_owned()));
    // A number that names no bid is no call: below 30, between the points
    // and a mark, between marks, below zero.
    for number in [
        format!("(- 32 {doubles})"),
        format!("(+ 40 {doubles})"),
        format!("(+ 82 {doubles})"),
        format!("(- {doubles} 30)"),
    ] {
        let rule = format!("(rule priced (bid {number}) (viewer v))");
        assert_eq!(call(&rule, &at(vec![])), (Call::Pass, "fallback".to_owned()), "{number}");
    }
    // Marks are numbers too: 39 plus three doubles is 42, one mark; twice
    // 42 is 84, legal over 42.
    assert_eq!(call(&format!("(rule m (bid (+ 39 {doubles})) (viewer v))"), &at(vec![])), (bid(42), "m".to_owned()));
    assert_eq!(call("(rule m (bid (+ 42 42)) (viewer v))", &at(vec![bid(42)])), (bid(84), "m".to_owned()));
    // No number, no call: no high bid before the first bid.
    let raise = "(rule over (bid (+ (high-bid) 2)) (viewer v))";
    assert_eq!(call(raise, &at(vec![])), (Call::Pass, "fallback".to_owned()));
    assert_eq!(call(raise, &at(vec![bid(31)])), (bid(33), "over".to_owned()));
    // (bid (least-raise)) is (bid least-raise), at every least raise.
    let capped = frame_after(AuctionConfig::new(2, false).unwrap(), vec![bid(84)], None);
    for frame in (29..=41).map(|h| at(if h < 30 { vec![] } else { vec![bid(h)] })).chain([at(vec![bid(42)]), capped]) {
        assert_eq!(
            call("(rule r (bid (least-raise)) (viewer v))", &frame).0,
            call("(rule r (bid least-raise) (viewer v))", &frame).0
        );
    }
}

#[test]
fn a_computed_call_reads_each_answer_and_calls_the_least() {
    // Thirty and the length of each suit held: every answer gives its own
    // bid, and the rule calls the least legal one (as a play rule plays the
    // least tile it allows). The witnesses are the answers that call it.
    let length = |p: Pip| hand().iter().filter(|d| d.has(p)).count() as u32;
    let lengths: Vec<u32> = Pip::ALL.iter().map(|p| length(*p)).filter(|n| *n > 0).collect();
    let rule = "(rule by-length (bid (+ 30 (card (roles d) (of-suit d t)))) (suit-at-least t 1))";
    let policy = policy(&format!("(include bidding) (policy p (bidding {rule}))"));
    for high in [None, Some(31), Some(32), Some(33), Some(34)] {
        let frame = frame_after(AuctionConfig::STRAIGHT, high.map(bid).into_iter().collect(), None);
        let least = lengths.iter().map(|n| 30 + n).filter(|b| high.is_none_or(|h| *b > h)).min();
        let choice = policy.explain_bid(&frame, &mut budget()).unwrap();
        assert_eq!(choice.choice, least.map_or(Call::Pass, bid), "over {high:?}");
        if let Some(least) = least {
            let named: Vec<Pip> = Pip::ALL.into_iter().filter(|p| 30 + length(*p) == least).collect();
            let witnessed: Vec<Value> = choice.witnesses.iter().map(|(answer, _)| answer[0]).collect();
            let expected: Vec<Value> = named.iter().map(|p| Value::Context(Context::Natural(*p))).collect();
            assert_eq!(witnessed, expected, "over {high:?}");
        }
    }
}

#[test]
fn a_computed_call_prints_reads_back_and_is_refused_where_it_cannot_mean() {
    let short = "(include bidding) (policy p (bidding (rule r (bid (- 42 (risk t))) (suit-at-least t 3))))";
    let explicit = "(include bidding) (policy p (bidding (rule r (select bid (- 42 (risk t))) \
                    (guard (query (out t) (suit-at-least t 3))))))";
    let short = PolicyProgram::parse(short, "short").unwrap();
    assert_eq!(short, PolicyProgram::parse(explicit, "explicit").unwrap());
    assert_eq!(PolicyProgram::parse(&short.to_string(), "printed").unwrap(), short);
    assert!(short.to_string().contains("(select bid (- 42 (+ (points"), "{short}");
    for (rules, why) in [
        // The guard answers what the number reads, no more and no less.
        (
            "(rule r (select bid (- 42 (risk t))) (guard (query (suit-at-least t 3))))",
            "must answer the roles its call reads, t",
        ),
        ("(rule r (select bid 30) (guard (query (out t) (suit-at-least t 3))))", "must have 0 output(s)"),
        // The number sees what the guard may: the viewer, at the auction.
        ("(rule r (bid (+ 30 (card (roles d) (holds S1 d)))) (viewer v))", "world predicate holds"),
        ("(rule r (bid (to-set)) (viewer v))", "to-set, a predicate of the play"),
        ("(rule r (bid (+ 30 (risk t))) (viewer v))", "cannot infer the sort of t"),
    ] {
        let text = format!("(include bidding) (policy p (bidding {rules}))");
        let refused = text.parse::<PolicyProgram>().and_then(|p| p.compile(&Registry::standard(), &physics("P6")));
        let Err(Error(message)) = refused else { panic!("{text} compiled") };
        assert!(message.contains(why), "{text}: {message}");
    }
    // Only a bid is computed: a play or a declaration names its answer.
    for rules in ["(rule r (play (+ 1 2)) (viewer v))", "(declaring (rule r (declare (+ 1 2)) (viewer v)))"] {
        let text = format!("(policy p {rules})");
        assert!(text.parse::<PolicyProgram>().is_err(), "{text}");
    }
}
