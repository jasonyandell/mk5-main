// SPDX-FileCopyrightText: Jason Yandell
// SPDX-License-Identifier: CC0-1.0

//! A reading written in a policy file (#162): `(reading name (read chair
//! policy) ... clause ...)` declares a guard atom (#128) its rules may name,
//! the reader found by name. The book's crisis as Hand 31 reads it
//! (`examples/book/crisis-read.silas`) fires where the reading makes one
//! setter's holding certain, and waits where no setter's play says which.
mod common;

use common::rules::{B, book_at, position};
use common::*;
use silas::*;

const CRISIS: &str = include_str!("../examples/book/crisis.silas");
const CRISIS_READ: &str = include_str!("../examples/book/crisis-read.silas");

/// A book policy by name, from `examples/book/`.
fn book(name: &str) -> Result<PolicyProgram> {
    let path = format!("{}/examples/book/{name}.silas", env!("CARGO_MANIFEST_DIR"));
    let text = std::fs::read_to_string(&path).map_err(|e| Error(format!("{path}: {e}")))?;
    PolicyProgram::parse(&text, &path)
}

fn compiled(source: &str, frame: &Frame) -> CompiledPolicy {
    let program = PolicyProgram::parse(source, "policy").unwrap();
    program.compile_reading(&Registry::standard(), frame.physics(), &book).unwrap()
}

/// The deciding rule and its tiles at one seating; `None` where it falls back.
fn fires_at(source: &str, frame: &Frame) -> Option<(String, DominoSet)> {
    let allowed = compiled(source, frame).allowed(frame, &mut Budget::new(u64::MAX)).unwrap();
    match allowed.provenance {
        DecisionProvenance::RelationalRule { name } => Some((name, allowed.tiles)),
        _ => None,
    }
}

/// [`fires_at`] with the bidder in each chair, which must agree.
fn fires(source: &str, at: &common::fixture::Position) -> Option<(String, DominoSet)> {
    let found = fires_at(source, &at.frame(Seat::S0));
    for declarer in [Seat::S1, Seat::S2, Seat::S3] {
        assert_eq!(fires_at(source, &at.frame(declarer)), found, "the bidder in {declarer}");
    }
    found
}

#[test]
fn a_reading_prints_as_it_reads() {
    let program = PolicyProgram::parse(CRISIS_READ, "crisis-read").unwrap();
    assert_eq!(program.readings.len(), 1);
    let reading = &program.readings[0];
    assert_eq!(reading.name, "one-setter-holds-the-trumps-out");
    assert_eq!(
        reading.readers,
        vec![
            (ReadChair::LeftSetter, "protect-the-count".to_owned()),
            (ReadChair::RightSetter, "protect-the-count".to_owned())
        ]
    );
    assert_eq!(reading.likelihood, Likelihood::Consistent);
    assert_eq!(reading.bound, r(1, 1));
    let printed = program.to_string();
    assert!(printed.contains("(reading one-setter-holds-the-trumps-out (read left-setter protect-the-count)"));
    assert_eq!(PolicyProgram::parse(&printed, "printed").unwrap(), program);
    // The likelihood and the bound, when said.
    let said = "(policy p (reading r (read partner donate-when-safe) (likelihood uniform) (at-least 3/4) (live 5-5))
        (rule x (play d) (own-legal d) (r)))";
    let said = PolicyProgram::parse(said, "said").unwrap();
    assert_eq!((said.readings[0].likelihood, said.readings[0].bound.clone()), (Likelihood::Uniform, r(3, 4)));
    assert_eq!(PolicyProgram::parse(&said.to_string(), "again").unwrap(), said);
}

#[test]
fn a_policy_that_reads_is_compiled_with_its_readers() {
    let frame = book_at("hand-31", 4, B).frame(Seat::S0);
    let program = PolicyProgram::parse(CRISIS_READ, "crisis-read").unwrap();
    let refused = program.compile(&Registry::standard(), frame.physics()).err().unwrap();
    assert!(refused.0.contains("compile_reading"), "{}", refused.0);
    let missing = |name: &str| Err(Error(format!("no policy {name}")));
    let refused = program.compile_reading(&Registry::standard(), frame.physics(), &missing).err().unwrap();
    assert!(refused.0.contains("the reader protect-the-count: no policy protect-the-count"), "{}", refused.0);
    // A policy that reads itself, however far round, is refused.
    let a = "(policy a (reading r (read partner b) (live 5-5)) (rule x (play d) (own-legal d) (r)))";
    let b = "(policy b (reading r (read partner a) (live 5-5)) (rule x (play d) (own-legal d) (r)))";
    let resolve = |name: &str| PolicyProgram::parse(if name == "a" { a } else { b }, name);
    let refused = resolve("a").unwrap().compile_reading(&Registry::standard(), frame.physics(), &resolve);
    assert!(refused.err().unwrap().0.contains("a reading reads itself: a reads b reads a"));
    // Malformed readings.
    for text in [
        "(policy p (reading r (live 5-5)) (rule x (play d) (own-legal d)))",
        "(policy p (reading r (read pard follow-low) (live 5-5)) (rule x (play d) (own-legal d)))",
        "(policy p (reading r (read partner a) (read partner b) (live 5-5)) (rule x (play d) (own-legal d)))",
        "(policy p (reading r (read partner a) (at-least half) (live 5-5)) (rule x (play d) (own-legal d)))",
    ] {
        assert!(PolicyProgram::parse(text, "p").is_err(), "{text}");
    }
    let outside = "(policy p (reading r (read partner a) (at-least 2) (live 5-5)) (rule x (play d) (own-legal d)))";
    let a_policy = |_: &str| PolicyProgram::parse("(policy a (rule x (play d) (own-legal d)))", "a");
    let outside = PolicyProgram::parse(outside, "p").unwrap();
    assert!(outside.compile_reading(&Registry::standard(), frame.physics(), &a_policy).is_err());
}

#[test]
fn s0138e_s0139a_hand_31s_crisis_as_read() {
    // Hand 31 trick 2: the right setter's 3-2 on the bidder's 2-2, read under
    // protect-the-count, puts both trumps out with the left setter, so the
    // crisis is certain and the bait is led, as in crisis.silas.
    let at = book_at("hand-31", 4, B);
    let bait = Some(("the-six-five-bait".to_owned(), tiles(&["6-5"])));
    assert_eq!(fires(CRISIS_READ, &at), bait);
    assert_eq!(fires(CRISIS, &at), bait);
}

#[test]
fn two_trumps_out_that_no_play_places_are_no_crisis_yet() {
    // Ours: fours. The bidder leads the 6-4, sure while he holds the 4-4; the
    // setters follow with the 4-0 and the 4-3, neither of them count, and the
    // partner shows out with the 6-2. Out: the 4-5 and the 4-1; the 4-4 beats
    // one, so if one setter holds both, one outlives the draw. Nothing either
    // setter played says whether one holds both: crisis.silas cashes the 5-5
    // on two trumps out; the crisis as read waits for a play that says.
    let hand = ["4-4", "4-2", "5-5", "3-3", "2-2", "1-0"];
    let at = position("P4", B, &hand, &["6-4", "4-0", "6-2", "4-3"]);
    assert_eq!(fires(CRISIS, &at), Some(("cash-a-double-that-is-count".to_owned(), tiles(&["5-5"]))));
    assert_eq!(fires(CRISIS_READ, &at), None);
}
