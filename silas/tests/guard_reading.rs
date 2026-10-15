// SPDX-FileCopyrightText: Jason Yandell
// SPDX-License-Identifier: CC0-1.0

//! A guard that reads (#128; `GAPS.md` 1): a policy's rule acting on a
//! reading of another seat's plays. The book's own readings of Hand 31
//! (`tests/book_reading.rs`) are asked as guard atoms, at all four seatings:
//! each holds where the reading makes its event certain, fails where only the
//! uniform deal is read, and moves the policy's decision exactly there.
mod common;

use common::rules::{B, O1, O2, book_at, prose};
use common::*;
use num_rational::BigRational;
use silas::*;

fn query_file(file: &str) -> CompiledQuery {
    let path = format!("{}/examples/book/{file}", env!("CARGO_MANIFEST_DIR"));
    Query::parse(&std::fs::read_to_string(&path).unwrap(), &path).unwrap().compile(&Registry::standard()).unwrap()
}

fn book_policy(file: &str, frame: &Frame) -> CompiledPolicy {
    let path = format!("{}/examples/book/{file}", env!("CARGO_MANIFEST_DIR"));
    PolicyProgram::parse(&std::fs::read_to_string(&path).unwrap(), &path)
        .unwrap()
        .compile(&Registry::standard(), frame.physics())
        .unwrap()
}

/// The guard `name`: the right setter's plays read under protect-the-count
/// (consistently), the `event` at probability at least `bound`.
fn right_setter_read(name: &str, event: &str, bound: BigRational, frame: &Frame) -> ReadingPredicate {
    let right = (ReadChair::RightSetter, book_policy("protect-the-count.silas", frame));
    ReadingPredicate::new(name, vec![right], Likelihood::Consistent, query_file(event), bound).unwrap()
}

/// The same event, nothing read: the uniform deal.
fn unread(name: &str, event: &str, bound: BigRational) -> ReadingPredicate {
    ReadingPredicate::new(name, vec![], Likelihood::Consistent, query_file(event), bound).unwrap()
}

/// A reading runs a policy over every hand a seat could hold: a budget for
/// that, not a test's default.
fn unbounded() -> Budget {
    Budget::new(u64::MAX)
}

fn holds(guard: &ReadingPredicate, frame: &Frame) -> Option<bool> {
    guard.evaluate(PredicateContext::Viewer(frame), &[], &mut unbounded()).unwrap()
}

fn probability(guard: &ReadingPredicate, frame: &Frame) -> BigRational {
    guard.probability(frame, &mut unbounded()).unwrap().expect("a possible record").0
}

const LEFT_HOLDS: &str = "reading/left-setter-holds-the-trumps-out.silas";
const BIDDER_HOLDS: &str = "reading/bidder-holds-the-trumps-out.silas";

#[test]
fn s0138e_the_bidders_reading_after_trick_one_is_a_guard() {
    // Hand 31 trick 2, the bidder to lead: read under protect-the-count, the
    // right setter's 3-2 was his only trump, so both trumps out are the left
    // setter's. Certain read (5/22 uniformly), at every seating.
    let at = book_at("hand-31", 4, B);
    for declarer in Seat::ALL {
        let frame = at.frame(declarer);
        let read = right_setter_read("left-holds-the-trumps", LEFT_HOLDS, r(1, 1), &frame);
        assert_eq!(probability(&read, &frame), r(1, 1), "the bidder in {declarer}");
        assert_eq!(holds(&read, &frame), Some(true));
        let uniform = unread("left-holds-the-trumps", LEFT_HOLDS, r(1, 1));
        assert_eq!(probability(&uniform, &frame), r(5, 22));
        assert_eq!(holds(&uniform, &frame), Some(false));
        // A bound the uniform deal clears, and one it does not.
        assert_eq!(holds(&unread("p", LEFT_HOLDS, r(5, 22)), &frame), Some(true));
        assert_eq!(holds(&unread("p", LEFT_HOLDS, r(6, 22)), &frame), Some(false));
    }
}

#[test]
fn s0140e_the_left_setters_certainty_is_a_guard() {
    // Hand 31 trick 6, the left setter on lead: the bidder holds both trumps
    // out. 2/15 uniformly; certain read.
    let at = book_at("hand-31", 20, O1);
    for declarer in Seat::ALL {
        let frame = at.frame(declarer);
        let read = right_setter_read("bidder-holds-the-trumps", BIDDER_HOLDS, r(1, 1), &frame);
        assert_eq!(holds(&read, &frame), Some(true), "the bidder in {declarer}");
        assert_eq!(probability(&unread("u", BIDDER_HOLDS, r(1, 1)), &frame), r(2, 15));
    }
}

#[test]
fn the_viewers_own_chair_is_not_read() {
    // Hand 31 trick one, the right setter to play: asked from his own seat,
    // his record is no evidence to him, so the guard reads nothing and asks
    // the uniform deal (where Reading::evidence would refuse the viewer).
    let at = book_at("hand-31", 3, O2);
    for declarer in Seat::ALL {
        let frame = at.frame(declarer);
        assert_eq!(frame.viewer(), declarer.plus(3));
        let refused = Reading::plays(
            vec![(frame.viewer(), book_policy("protect-the-count.silas", &frame))],
            Likelihood::Consistent,
        );
        assert!(refused.unwrap().evidence(&frame, &mut unbounded()).is_err());
        let read = right_setter_read("left-holds-the-trumps", LEFT_HOLDS, r(1, 1), &frame);
        let (_, authority) = read.probability(&frame, &mut unbounded()).unwrap().unwrap();
        assert!(authority.is_exact(), "the uniform deal's, no model read");
    }
}

#[test]
fn an_impossible_record_is_no_reading() {
    // Hand 31 trick one: the partner, out of trump, may discard any of his
    // tiles. Read under a policy that never discards the tile he did discard
    // while he holds another, the record is impossible in every world, so the
    // guard is undefined and neither it nor its negation holds.
    let at = book_at("hand-31", 4, B);
    let discarded = prose("hand-31")[2];
    let never = format!(
        "(include book)
        (policy never-that-tile
          (rule anything-else (play x) (viewer me) (own-legal x) (not (tile x {discarded}))))"
    );
    for declarer in Seat::ALL {
        let frame = at.frame(declarer);
        let policy = PolicyProgram::parse(&never, "never").unwrap().compile(&Registry::standard(), frame.physics());
        let guard = ReadingPredicate::new(
            "impossible",
            vec![(ReadChair::Partner, policy.unwrap())],
            Likelihood::Consistent,
            query_file(LEFT_HOLDS),
            r(1, 1),
        )
        .unwrap();
        assert_eq!(holds(&guard, &frame), None, "the bidder in {declarer}");
        let mut registry = Registry::standard();
        registry.register(guard).unwrap();
        let positive = "(include book) (policy p (rule r (play x) (own-legal x) (impossible)))";
        let negative = "(include book) (policy p (rule r (play x) (own-legal x) (not (impossible))))";
        for text in [positive, negative] {
            let p = PolicyProgram::parse(text, "p").unwrap().compile(&registry, frame.physics()).unwrap();
            assert_eq!(p.allowed(&frame, &mut unbounded()).unwrap().provenance, DecisionProvenance::Fallback);
        }
    }
}

/// A rule that acts on the reading: the bidder, on lead, plays a trump when
/// the reading puts every trump out with the left setter.
const ACTS: &str = "(include book)
    (policy acts-on-the-reading
      (rule lead-trump-into-the-left-setter (play x)
        (viewer b) (bidder b) (on-lead b)
        (own-legal x) (trump-tile x)
        (left-holds-the-trumps)))";

#[test]
fn a_rule_acts_on_the_reading() {
    let at = book_at("hand-31", 4, B);
    for declarer in Seat::ALL {
        let frame = at.frame(declarer);
        let trumps: DominoSet = frame
            .physics()
            .legal_plays(frame.kernel().viewer_hand(), frame.lead())
            .iter()
            .filter(|d| frame.physics().trump().is_some_and(|q| frame.physics().incidence(q).contains(*d)))
            .collect();
        assert!(!trumps.is_empty(), "the bidder holds a trump at Hand 31 trick 2");
        // Read: the rule fires and allows exactly his trumps.
        let mut registry = Registry::standard();
        registry.register(right_setter_read("left-holds-the-trumps", LEFT_HOLDS, r(1, 1), &frame)).unwrap();
        let policy = PolicyProgram::parse(ACTS, "acts").unwrap().compile(&registry, frame.physics()).unwrap();
        let allowed = policy.allowed(&frame, &mut unbounded()).unwrap();
        let fired = DecisionProvenance::RelationalRule { name: "lead-trump-into-the-left-setter".into() };
        assert_eq!((allowed.provenance, allowed.tiles), (fired, trumps), "the bidder in {declarer}");
        // The same frame twice: the same decision (the policy is a function of it).
        assert_eq!(policy.decide(&frame, &mut unbounded()).unwrap(), policy.decide(&frame, &mut unbounded()).unwrap());
        // Unread: the rule waits, and the fallback decides.
        let mut registry = Registry::standard();
        registry.register(unread("left-holds-the-trumps", LEFT_HOLDS, r(1, 1))).unwrap();
        let policy = PolicyProgram::parse(ACTS, "acts").unwrap().compile(&registry, frame.physics()).unwrap();
        assert_eq!(policy.allowed(&frame, &mut unbounded()).unwrap().provenance, DecisionProvenance::Fallback);
    }
}

#[test]
fn the_standard_registry_does_not_know_a_reading() {
    // A reading is registered by the host that names it; the standard
    // registry refuses the atom.
    let frame = book_at("hand-31", 4, B).frame(Seat::S0);
    let refused = PolicyProgram::parse(ACTS, "acts").unwrap().compile(&Registry::standard(), frame.physics());
    assert!(refused.is_err());
}

#[test]
fn the_reading_names_its_models_in_the_policys_identity() {
    // Two readings under different policies are different predicates, and
    // the policy that reads one is not the policy that reads the other.
    let frame = book_at("hand-31", 4, B).frame(Seat::S0);
    let identity = |reader: &str| {
        let mut registry = Registry::standard();
        let right = (ReadChair::RightSetter, book_policy(reader, &frame));
        let guard = ReadingPredicate::new(
            "left-holds-the-trumps",
            vec![right],
            Likelihood::Consistent,
            query_file(LEFT_HOLDS),
            r(1, 1),
        );
        registry.register(guard.unwrap()).unwrap();
        PolicyProgram::parse(ACTS, "acts").unwrap().compile(&registry, frame.physics()).unwrap().identity().to_owned()
    };
    assert_ne!(identity("protect-the-count.silas"), identity("follow-low.silas"));
}

#[test]
fn refusals() {
    let frame = book_at("hand-31", 4, B).frame(Seat::S0);
    let p = || book_policy("protect-the-count.silas", &frame);
    let event = || query_file(LEFT_HOLDS);
    let twice = vec![(ReadChair::RightSetter, p()), (ReadChair::RightSetter, p())];
    assert!(ReadingPredicate::new("a", twice, Likelihood::Consistent, event(), r(1, 1)).is_err());
    let outputs = compile("(query (roles (domino d)) (out d) (live d))");
    assert!(ReadingPredicate::new("a", vec![], Likelihood::Consistent, outputs, r(1, 1)).is_err());
    assert!(ReadingPredicate::new("a", vec![], Likelihood::Consistent, event(), r(0, 1)).is_err());
    assert!(ReadingPredicate::new("a", vec![], Likelihood::Consistent, event(), r(3, 2)).is_err());
    assert!(ReadingPredicate::new("2bad", vec![], Likelihood::Consistent, event(), r(1, 2)).is_err());
}

#[test]
fn the_answer_is_remembered_by_the_record() {
    // A reading costs a policy run per hand a seat could hold; asked again at
    // the same record it costs nothing, so a guard asked once per binding, or
    // a policy priced over a pour, pays for each record once.
    let frame = book_at("hand-31", 4, B).frame(Seat::S0);
    let read = right_setter_read("left-holds-the-trumps", LEFT_HOLDS, r(1, 1), &frame);
    let mut first = unbounded();
    assert_eq!(read.evaluate(PredicateContext::Viewer(&frame), &[], &mut first).unwrap(), Some(true));
    assert!(first.spent() > 0);
    let mut again = Budget::new(0);
    assert_eq!(read.evaluate(PredicateContext::Viewer(&frame), &[], &mut again).unwrap(), Some(true));
    assert_eq!(again.spent(), 0);
}
