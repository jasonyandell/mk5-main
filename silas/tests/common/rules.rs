//! Asking a book rule what it allows at a position, at all four seatings.

use silas::*;

use super::fixture::{Position, Role, book};
use super::{budget, physics, tile, tiles};

// The book's hands, deals, declarations, bids and narrated plays come from
// `examples/book/hands.json` (see `tests/book_fixture.rs`), read from the
// reviewed transcription; an id such as `hand-28` names its example there.
// A position names its seats by role, counted clockwise from the bidder.

pub const B: Role = Role::Bidder;
pub const O1: Role = Role::Opponent1;
pub const P: Role = Role::Partner;
pub const O2: Role = Role::Opponent2;

/// The tiles of a narrated line, in play order.
pub fn prose(id: &str) -> Vec<Domino> {
    book().line(&format!("{id}/prose")).plays()
}

/// `plays` and then `more`.
pub fn then(plays: &[Domino], more: &[&str]) -> Vec<Domino> {
    plays.iter().copied().chain(more.iter().map(|d| tile(d))).collect()
}

/// The book's deal `id` after its narrated line's first `plays`, `role` to move.
pub fn book_at(id: &str, plays: usize, role: Role) -> Position {
    book().at(id, plays, role)
}

/// The book's deal `id` after `plays`, `role` to move.
pub fn dealt(id: &str, role: Role, plays: &[Domino]) -> Position {
    book().dealt(id, role, plays)
}

/// The viewer's view after `plays` under `decl` at a bid of 30, holding `hand`.
pub fn position(decl: &str, viewer: Role, hand: &[&str], plays: &[&str]) -> Position {
    Position::holding(physics(decl), 30, viewer, tiles(hand), plays.iter().map(|d| tile(d)).collect())
}

pub fn program(source: &str) -> PolicyProgram {
    source.parse().unwrap()
}

/// At one seating: the first rule of the policy that allows a legal tile,
/// and every tile it allows. The policy's decision is one of them, and is a
/// function of the frame; `None` means the policy falls back.
pub fn fires_at(source: &str, frame: &Frame) -> Option<(String, DominoSet)> {
    let compiled = program(source).compile(&Registry::standard(), frame.physics()).unwrap();
    let allowed = compiled.allowed(frame, &mut budget()).unwrap();
    let decision = compiled.decide(frame, &mut budget()).unwrap();
    assert_eq!((&decision.provenance, decision.allowed), (&allowed.provenance, allowed.tiles));
    assert!(allowed.tiles.contains(decision.action));
    match allowed.provenance {
        DecisionProvenance::RelationalRule { name } => Some((name, allowed.tiles)),
        DecisionProvenance::Fallback => None,
        DecisionProvenance::Exact => panic!("the book's rules have no exact table"),
    }
}

/// [`fires_at`] with the bidder in each of the four chairs, which must agree.
pub fn fires(source: &str, at: &Position) -> Option<(String, DominoSet)> {
    let found = fires_at(source, &at.frame(Seat::S0));
    for declarer in [Seat::S1, Seat::S2, Seat::S3] {
        assert_eq!(fires_at(source, &at.frame(declarer)), found, "the bidder in {declarer}");
    }
    found
}

/// The tiles the policy allows, asserting it fires by the named rule.
pub fn allows(source: &str, at: &Position, rule: &str) -> DominoSet {
    let (name, allowed) = fires(source, at).expect("the rule fires");
    assert_eq!(name, rule);
    allowed
}
