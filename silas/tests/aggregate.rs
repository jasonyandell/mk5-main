//! The aggregate (`GAPS.md`, 5): `(card (roles ...) clause ...)` counts the
//! bindings that satisfy a body, `(points (roles d) clause ...)` sums their
//! count, and `(<= a b)` compares two of them, literals or sums (README,
//! Aggregates). Each is held to its definition: to plain Rust over the
//! viewer's hand and over every world of a small frame, to `some` and `none`
//! in all three truth values, and its short forms to the core they spell.
mod common;

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

/// Seeded frames of random deals, played at random, from every seat to play.
fn random_frames(n: usize) -> Vec<Frame> {
    let mut rng = Rng(0xa66);
    let mut out = Vec::new();
    for game in 0..n {
        let physics = reference::Straight42::ALL[game % 9].physics();
        let deck = rng.shuffled();
        let deal: [DominoSet; 4] = std::array::from_fn(|s| deck[7 * s..7 * s + 7].iter().copied().collect());
        let contract = contract(&physics, Seat::ALL[rng.below(4)], 30 + rng.below(13) as u32);
        let mut hands = deal;
        let mut record = framed(&contract, Seat::S0, deal[0], vec![]).unwrap();
        while let Some(actor) = record.next_actor() {
            // Every seat's view, the one to play and the three that wait.
            for viewer in Seat::ALL {
                out.push(framed(&contract, viewer, deal[viewer.index()], record.plays().to_vec()).unwrap());
            }
            let legal = physics.legal_plays(hands[actor.index()], record.lead());
            let tile = legal.iter().nth(rng.below(legal.len())).unwrap();
            hands[actor.index()].remove(tile);
            record = record.after(tile).unwrap();
        }
    }
    out
}

/// The answers of a yes-or-no viewer query: true, false, or undefined (the
/// query and its negation both fail).
fn truth(text: &str, frame: &Frame) -> Option<bool> {
    let holds = |t: &str| !compile(t).evaluate_viewer(frame, &mut budget()).unwrap().is_empty();
    let body = text.strip_prefix("(query ").and_then(|t| t.strip_suffix(')')).unwrap();
    match (holds(text), holds(&format!("(query (not {body}))"))) {
        (true, false) => Some(true),
        (false, true) => Some(false),
        (false, false) => None,
        (true, true) => panic!("{text} and its negation both hold"),
    }
}

#[test]
fn card_and_points_count_the_viewers_hand_exactly() {
    // Suit lengths and count over the viewer's own hand: always decided, and
    // the numbers plain Rust gives.
    for frame in random_frames(12).iter().step_by(7) {
        let hand = frame.kernel().viewer_hand();
        let physics = frame.physics();
        for p in Pip::ALL {
            let length = hand.iter().filter(|d| d.has(p)).count() as u32;
            let count: u32 = hand.iter().filter(|d| d.has(p)).map(|d| physics.count(d)).sum();
            for n in 0..=8 {
                let card = format!("(query (<= {n} (card (roles d) (mine d) (bears d q{p}))))");
                let points = format!("(query (<= (points (roles d) (mine d) (bears d q{p})) {}))", n * 5);
                assert_eq!(truth(&card, frame), Some(length >= n), "{card}");
                assert_eq!(truth(&points, frame), Some(count <= n * 5), "{points}");
            }
        }
    }
}

#[test]
fn some_and_none_are_card_at_least_one_and_at_most_zero_in_all_three_values() {
    // Kleene's quantifiers are the aggregate's two thresholds: `(some R B)`
    // is `(<= 1 (card R B))` and `(none R B)` is `(<= (card R B) 0)`, true,
    // false and undefined alike, the bodies below undefined wherever the
    // viewer is not to play or no trick is led.
    let bodies = [
        "(mine d) (own-legal d)",
        "(own-legal d) (double d)",
        "(live d) (led-context q) (in d q)",
        "(mine d) (count d 10)",
        "(trick-play c d)",
    ];
    let frames = random_frames(6);
    let mut undefined = 0;
    for body in bodies {
        let roles = if body.contains(" c ") || body.contains("(trick-play c") {
            "(roles d c)"
        } else if body.contains(" q") {
            "(roles d q)"
        } else {
            "(roles d)"
        };
        for frame in &frames {
            let some = truth(&format!("(query (some {roles} {body}))"), frame);
            let none = truth(&format!("(query (none {roles} {body}))"), frame);
            assert_eq!(truth(&format!("(query (<= 1 (card {roles} {body})))"), frame), some, "{body}");
            assert_eq!(truth(&format!("(query (<= (card {roles} {body}) 0))"), frame), none, "{body}");
            undefined += usize::from(some.is_none());
        }
    }
    assert!(undefined > 100, "the undefined case is exercised: {undefined}");
}

#[test]
fn an_undefined_body_makes_an_interval_and_a_comparison_decides_where_it_can() {
    // Not the viewer's turn: own-legal is undefined at each of his seven
    // tiles, so the card of his legal tiles is anywhere from 0 to 7. Asked
    // against 7 it is decided; against 3, not.
    let frame = opening(&physics("P6"), tiles(&["6-6", "6-5", "6-4", "5-5", "4-4", "3-3", "0-0"]));
    let s1 = tiles(&["1-0", "1-1", "2-0", "2-1", "2-2", "3-0", "3-1"]);
    let waiting = framed(frame.contract(), Seat::S1, s1, vec![]).unwrap();
    assert_eq!(waiting.next_actor(), Some(Seat::S0));
    assert_eq!(truth("(query (<= (card (roles d) (mine d) (own-legal d)) 7))", &waiting), Some(true));
    assert_eq!(truth("(query (<= (card (roles d) (mine d) (own-legal d)) 3))", &waiting), None);
    assert_eq!(truth("(query (<= 8 (card (roles d) (mine d) (own-legal d))))", &waiting), Some(false));
    // At his turn every one of the seven is legal on the lead.
    assert_eq!(truth("(query (= (card (roles d) (mine d) (own-legal d)) 7))", &frame), Some(true));
}

#[test]
fn over_hidden_hands_an_aggregate_is_decided_in_each_world_and_a_belief_gives_its_law() {
    // The small frame: S0 holds 6-6 and 0-0; 6-5 5-5 3-2 6-4 2-1 1-0 are
    // split two to a seat over 90 worlds. "S1 holds at least ten points of
    // count" is a world fact; its probability is the number of worlds where
    // it holds, over 90, exactly.
    let frame = small_frame(physics("P6"));
    let belief = Belief::uniform(frame.clone(), 90).unwrap();
    for n in [0u32, 5, 10, 15, 20] {
        let q = compile(&format!("(query (<= {n} (points (roles d) (not (count d 0)) (holds S1 d))))"));
        let p = q.summarize(&belief, &mut budget()).unwrap().event_probability();
        let worlds = frame
            .worlds()
            .filter(|w| w.hand(Seat::S1).iter().map(|d| frame.physics().count(d)).sum::<u32>() >= n)
            .count();
        assert_eq!(exact(p), r(worlds as i64, 90), "at least {n}");
    }
    // Comparing two hidden hands: S1 longer in sixes than S3.
    let q =
        compile("(query (< (card (roles d) (bears d q6) (holds S3 d)) (card (roles d) (bears d q6) (holds S1 d))))");
    let p = q.summarize(&belief, &mut budget()).unwrap().event_probability();
    let sixes = |w: &World, s: Seat| w.hand(s).iter().filter(|d| d.has(Pip::new(6).unwrap())).count();
    let worlds = frame.worlds().filter(|w| sixes(w, Seat::S3) < sixes(w, Seat::S1)).count();
    assert_eq!(exact(p), r(worlds as i64, 90));
    // A policy may not read one: a hidden hand is not the viewer's.
    let text = "(policy p (rule r (play d) (mine d) (<= 1 (card (roles e) (holds S1 e)))))";
    let refused = text.parse::<PolicyProgram>().unwrap().compile(&Registry::standard(), &physics("P6"));
    let Err(Error(message)) = refused else { panic!("a policy read a hidden hand") };
    assert!(message.contains("world predicate holds"), "{message}");
}

#[test]
fn comparisons_are_short_forms_of_at_most() {
    let card = "(card (roles d) (mine d) (double d))";
    let other = "(card (roles d) (mine d) (count d 5))";
    same_query(&format!("(query (< {card} 3))"), &format!("(query (<= (+ {card} 1) 3))"));
    same_query(&format!("(query (> {card} 3))"), &format!("(query (<= 4 {card}))"));
    same_query(&format!("(query (>= {card} {other}))"), &format!("(query (<= {other} {card}))"));
    same_query(&format!("(query (= {card} 2))"), &format!("(query (<= {card} 2) (<= 2 {card}))"));
    same_query(&format!("(query (/= {card} 2))"), &format!("(query (none (roles) (<= 2 {card}) (<= {card} 2)))"));
    // Negation flips the comparison: not a <= b is b + 1 <= a.
    same_query(&format!("(query (not (<= {card} {other})))"), &format!("(query (<= (+ {other} 1) {card}))"));
    same_query(&format!("(query (not (< {card} 2)))"), &format!("(query (<= 2 {card}))"));
    // A definition's number parameter takes its argument; negated, the
    // comparison flips.
    let defined = "(define doubles-at-least (roles (number n)) (<= n (card (roles d) (mine d) (double d))))";
    same_query(
        &format!("{defined} (query (doubles-at-least 2) (not (doubles-at-least 4)))"),
        &format!("(query (<= 2 {card}) (<= (+ {card} 1) 4))"),
    );
    // points binds one domino, its sort given.
    same_query("(query (<= 10 (points (roles c) (mine c))))", "(query (<= 10 (points (roles (domino c)) (mine c))))");
}

#[test]
fn an_aggregate_that_cannot_be_read_says_why() {
    for (text, why) in [
        ("(query (card (roles d) (mine d)))", "is a number, not a clause"),
        ("(query (roles (domino d)) (<= d 3))", "never a role"),
        ("(query (<= x 3))", "never a role"),
        ("(query (<= 1 (points (roles a b) (mine a) (mine b))))", "one domino"),
        ("(query (<= 1 (points (roles (chair c)) (viewer c))))", "one domino"),
        ("(query (<= 1 (card (mine d))))", "(roles ...)"),
        ("(query (/= 1 2 3))", "compares two numbers"),
        ("(query (< 1))", "compares two numbers"),
        ("(query (<= 1 6-6))", "is not a number"),
    ] {
        let message = refused(text);
        assert!(message.contains(why), "{text}: {message}");
    }
    // A card role the body never names counts every value of its sort.
    let warnings = query("(query (<= 28 (card (roles (domino d) e) (mine e))))").lint();
    assert!(warnings.iter().any(|w| w.0.contains("d is bound and never used")), "{warnings:?}");
}

#[test]
fn rules_that_share_an_aggregate_count_it_once_and_answer_as_alone() {
    // Two rules of one section name the same aggregate under different
    // bindings of t: the decision's shared count must give each rule its own
    // answer. Each rule's guard is also compiled alone and asked directly.
    let program: PolicyProgram = "(include bidding)
        (policy p
          (declaring
            (rule long (declare t) (suit-at-least t 4))
            (rule short (declare t) (suit-at-least t 2) (not (suit-at-least t 3)))))"
        .parse()
        .unwrap();
    let policy = program.compile(&Registry::standard(), &physics("P6")).unwrap();
    let menu: std::sync::Arc<[Physics]> =
        ["P0", "P1", "P2", "P3", "P4", "P5", "P6"].iter().map(|d| physics(d)).collect();
    let mut rng = Rng(4);
    for _ in 0..300 {
        let hand: DominoSet = rng.shuffled()[..7].iter().copied().collect();
        let auction = Auction::won_by(Seat::S1, Bid::Points(30)).unwrap();
        let frame =
            AuctionFrame::new(auction, Seat::S1, hand, MatchState::new(Scoring::MARKS, [0, 0]).unwrap(), menu.clone())
                .unwrap();
        let length = |p: Pip| hand.iter().filter(|d| d.has(p)).count();
        let long: Vec<Pip> = Pip::ALL.into_iter().filter(|p| length(*p) >= 4).collect();
        let short: Vec<Pip> = Pip::ALL.into_iter().filter(|p| length(*p) == 2).collect();
        let expected = long.first().or(short.first());
        let choice = policy.declare(&frame, &mut budget()).unwrap();
        match (expected, Trump::of(&choice.choice)) {
            (Some(p), Some(Trump::Suit(q))) => assert_eq!(*p, q, "{hand:?}"),
            (None, _) => assert_eq!(choice.provenance, DecisionProvenance::Fallback),
            (e, got) => panic!("{hand:?}: {e:?} against {got:?}"),
        }
        for (rule, guard) in policy.declaring() {
            let alone = rule.guard.compile(&Registry::standard()).unwrap();
            assert_eq!(
                guard.evaluate_auction(&frame, &mut budget()).unwrap(),
                alone.evaluate_auction(&frame, &mut budget()).unwrap()
            );
        }
    }
}
