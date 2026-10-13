//! The study's exact-check leads (the private coverage matrix, § 5), checked
//! with Silas's own exact tools: 14 double-dummy rows on the book's eight full
//! deals (Hands 28-35) and 21 probability or arithmetic rows. The leads came
//! from throwaway tools; each test is named by its row id and pins the value
//! *we* compute. CONFIRMED: ours equals the lead's claim. REFUTED: it does not;
//! the comment gives both. BLOCKED: the claim needs a capability Silas lacks,
//! named by its `GAPS.md` number; nothing is faked in its place. The table is
//! `examples/book/leads.md`. Claims are restated in our words.
//!
//! Double-dummy values are read at the position the narrated line reaches
//! (`examples/book/hands.json`), with the bidder in S0, under both objectives:
//! the bidder's side's final points (from `Objective::Margin`) and whether the
//! deal's bid is made (`Objective::Make`; a marks bid needs all 42). A
//! "value" in a lead is the bidder's side's points, banked plus to come,
//! under best play by all four seats. The solver sees all four hands: these
//! are facts about open positions, not about what a seat could know.
//!
//! Probabilities are exact over the uniform deal of the tiles the viewer
//! cannot see, given what the record shows (`Belief::uniform_resolving`,
//! `Authority::Exact`).
mod common;
#[path = "common/leads.rs"]
mod leads;

use common::fixture::{Role as R, book};
use common::*;
use leads::*;
use num_rational::BigRational;
use silas::*;

macro_rules! odds {
    ($name:literal) => {
        include_str!(concat!("../examples/book/odds/", $name, ".silas"))
    };
}

// ---------------------------------------------------------------------------
// 5a. Double-dummy leads.

/// CONFIRMED. Hand 28 (blanks, bid 30) is worth 42 to the bidder's side from
/// the opening lead; at trick 1 the bidder's 6-0, 0-0 and 5-0 each keep 42 and
/// so does either partner trump; at trick 2 the 0-0 lead, the left opponent's
/// 5-1 and the right opponent's 2-1 all keep 42.
#[test]
fn dd_01() {
    let h = narrated("hand-28");
    assert_eq!(h.worth(&h.opening), (42, 1));
    h.values(&h.at(1, R::Bidder)).assert(&[("6-0", 42, 1), ("0-0", 42, 1), ("5-0", 42, 1)]);
    h.values(&h.at(1, R::Partner)).assert(&[("1-0", 42, 1), ("2-0", 42, 1)]);
    h.values(&h.at(2, R::Bidder)).assert(&[("0-0", 42, 1)]);
    h.values(&h.at(2, R::Opponent1)).assert(&[("5-1", 42, 1)]);
    h.values(&h.at(2, R::Opponent2)).assert(&[("2-1", 42, 1)]);
}

/// CONFIRMED. Hand 28, trick 3, the bidder leads: the 3-1 or 5-3 holds his
/// side to 19 (set); the 5-0, 6-1 or 6-6 keeps 42. After the 6-1 the partner
/// overtakes with the 6-4 and leads the 2-2, 6-2 and 6-5 in turn, every play
/// margin-best; the bidder throws his 6-6 on the 2-2, then both offs (3-1,
/// 5-3) on the sixes. (Whether the book's
/// diagnosis is fair is a judgment about what the partner could know; the
/// solver only says the line exists.)
#[test]
fn dd_02() {
    let h = narrated("hand-28");
    let t3 = h.at(3, R::Bidder);
    h.values(&t3).assert(&[("3-1", 19, 0), ("5-3", 19, 0), ("5-0", 42, 1), ("6-1", 42, 1), ("6-6", 42, 1)]);
    let (end, played) = h.follow(
        &t3,
        &[
            Some("6-1"),
            None,
            Some("6-4"),
            None,
            Some("2-2"),
            None,
            Some("6-6"),
            None,
            Some("6-2"),
            None,
            Some("3-1"),
            None,
            Some("6-5"),
            None,
            Some("5-3"),
            None,
        ],
    );
    assert_eq!(played[2], tile("6-4"));
    assert_eq!(h.worth(&end), (42, 1));
}

/// CONFIRMED. Hand 28, trick 3, the left opponent follows: the 3-2 holds the
/// bidder to 19; the 3-3 gives him exactly 30, made.
#[test]
fn dd_03() {
    let h = narrated("hand-28");
    h.values(&h.at(3, R::Opponent1)).assert(&[("3-2", 19, 0), ("3-3", 30, 1)]);
}

/// CONFIRMED. Hand 28, trick 4, the right opponent leads: the 1-1 holds the
/// bidder to 19, the 5-4 or 5-2 to 25, the 4-2 lets him to 36. Played at 31
/// instead of 30 the 1-1 is still the margin-best lead (19) and sets.
#[test]
fn dd_04() {
    let h = narrated("hand-28");
    let t4 = h.at(4, R::Opponent2);
    h.values(&t4).assert(&[("1-1", 19, 0), ("5-4", 25, 0), ("5-2", 25, 0), ("4-2", 36, 1)]);
    let at_31 = rebid(h.physics(), &t4, 31);
    let v = values(h.physics(), &at_31);
    v.assert(&[("1-1", 19, 0), ("5-4", 25, 0), ("5-2", 25, 0), ("4-2", 36, 1)]);
    let best = v.moves.iter().map(|(_, p, _)| *p).min().unwrap();
    assert_eq!((best, v.of("1-1").0), (19, 19));
    assert_eq!(v.moves.iter().filter(|(_, p, _)| *p == best).count(), 1);
}

/// CONFIRMED. Hand 28, tricks 5 and 6: the left opponent's 4-1 and 4-4 leads
/// both give 25, and the bidder, void in fours, can ruff either with the
/// 5-0; the bidder's ruff, his 6-6-then-6-1 order, and both opponents'
/// trick-6 throws are all worth 25.
#[test]
fn dd_05() {
    let h = narrated("hand-28");
    let p = h.physics();
    h.values(&h.at(5, R::Opponent1)).assert(&[("4-1", 25, 0), ("4-4", 25, 0)]);
    let bidder = h.at(5, R::Bidder).hand(Seat::S0);
    for lead in ["4-1", "4-4"] {
        assert!(p.legal_plays(bidder, Some(tile(lead))).contains(tile("5-0")));
        assert!(p.beats(p.led_context(tile(lead)), tile(lead)).contains(tile("5-0")));
    }
    h.values(&h.at(5, R::Bidder)).assert(&[("5-0", 25, 0), ("6-1", 25, 0), ("6-6", 25, 0)]);
    h.values(&h.at(6, R::Bidder)).assert(&[("6-6", 25, 0), ("6-1", 25, 0)]);
    h.values(&h.at(6, R::Opponent1)).assert(&[("3-3", 25, 0), ("4-4", 25, 0)]);
    h.values(&h.at(6, R::Opponent2)).assert(&[("1-1", 25, 0), ("5-2", 25, 0)]);
}

/// CONFIRMED. Hand 29 (fours, bid 30) is worth 26: 30 cannot be made. Trick 1:
/// the bidder's 4-0 gives 20, the 4-4, 3-2 or 5-5 26; then the left
/// opponent's 6-4 gives 20 and his 4-3 42.
#[test]
fn dd_06() {
    let h = narrated("hand-29");
    assert_eq!(h.worth(&h.opening), (26, 0));
    h.values(&h.at(1, R::Bidder)).assert(&[("4-0", 20, 0), ("4-4", 26, 0), ("3-2", 26, 0), ("5-5", 26, 0)]);
    h.values(&h.at(1, R::Opponent1)).assert(&[("6-4", 20, 0), ("4-3", 42, 1)]);
}

/// CONFIRMED. Hand 29: at trick 2 the left opponent's 2-2 gives 20 and every
/// other lead 26; at trick 4 the partner's 6-3, 5-2 or 5-4 gives 20 and his
/// 3-0 19, the bidder's 4-1 or 4-4 20 and a thrown 3-3 or 5-5 8; at trick 5
/// the 4-4 gives 20.
#[test]
fn dd_07() {
    let h = narrated("hand-29");
    let t2 = h.values(&h.at(2, R::Opponent1));
    assert_eq!(t2.of("2-2"), (20, 0));
    assert!(t2.others(&["2-2"]).iter().all(|v| *v == (26, 0)));
    h.values(&h.at(4, R::Partner)).assert(&[("6-3", 20, 0), ("5-2", 20, 0), ("5-4", 20, 0), ("3-0", 19, 0)]);
    h.values(&h.at(4, R::Bidder)).assert(&[("4-1", 20, 0), ("4-4", 20, 0), ("3-3", 8, 0), ("5-5", 8, 0)]);
    h.values(&h.at(5, R::Bidder)).assert(&[("4-4", 20, 0)]);
}

/// CONFIRMED. Hand 30 (sixes) is worth 42; the same deal in deuces (Hand 31)
/// is worth 29. Trick 1: only the bidder's 1-0 lead drops to 35. Trick 2,
/// the partner leads: 0-0, 5-0 or 5-3 keep 42, a six or the 3-1 give 41.
/// Trick 3: the partner's 3-1 gives 41, his 5-0 or 5-3 42; the right
/// opponent's 3-2 then gives 42 and his 3-3 41. Every other narrated play
/// with a choice is margin-best.
#[test]
fn dd_08() {
    let h = narrated("hand-30");
    assert_eq!(h.worth(&h.opening), (42, 1));
    assert_eq!(narrated("hand-31").worth(&narrated("hand-31").opening), (29, 0));
    let t1 = h.values(&h.at(1, R::Bidder));
    assert_eq!(t1.of("1-0"), (35, 1));
    assert!(t1.others(&["1-0"]).iter().all(|v| *v == (42, 1)));
    h.values(&h.at(2, R::Partner)).assert(&[
        ("0-0", 42, 1),
        ("5-0", 42, 1),
        ("5-3", 42, 1),
        ("6-0", 41, 1),
        ("6-1", 41, 1),
        ("3-1", 41, 1),
    ]);
    h.values(&h.at(3, R::Partner)).assert(&[("3-1", 41, 1), ("5-0", 42, 1), ("5-3", 42, 1)]);
    h.values(&h.at(3, R::Opponent2)).assert(&[("3-2", 42, 1), ("3-3", 41, 1)]);
    let short: Vec<(usize, Seat)> =
        narrated_grades(&h).filter(|g| !g.margin.optimal()).map(|g| (g.ply, g.seat)).collect();
    // Trick 3: the partner (ply 8) and the right opponent (ply 9).
    assert_eq!(short, vec![(8, Seat::S2), (9, Seat::S3)]);
}

/// Partly REFUTED. Hand 31 (deuces, bid 30), worth 29. Confirmed: trick 1,
/// the bidder's 2-2 or 6-2 keeps 29 and anything else gives at most 12, the
/// partner's 5-0 keeps 29 and anything else at most 25; trick 2, the bidder's
/// 6-5 gives 24 and 6-4 29, the partner's 6-6 24 and his other sixes 23, the
/// right opponent's 4-0 24 and 5-5 29; trick 3, the partner's 0-0 is best
/// (24); trick 4, the partner's 3-1 gives 24 and a six 19, and the left
/// opponent's two non-trump throws (1-1, 5-4) both 24; trick 5, the right
/// opponent's 4-4 and 5-5 24, the left opponent's thrown 1-1 30.
///
/// REFUTED, the lead's value then ours: trick 2, the bidder's 6-2: 24, ours
/// 25; the right opponent's 4-1: 29, ours 30, which *makes* the bid; trick 4,
/// the partner's 5-3: 24, ours 25; trick 5, the right opponent's 4-1: 24,
/// ours 25; the left opponent's trumps (4-2, 5-2): 24, ours 25. From trick 4
/// on, the narrated position is worth 25, not 24: the right opponent's
/// narrated trick-3 throw (5-1) gives up a point against his 3-3 or 4-4.
///
/// The book's two post-mortem alternatives (the left opponent keeps the 5-4
/// at trick 4, throwing the 1-1; the right opponent leads a double at trick
/// 5) are each worth 24 and set: CONFIRMED that neither lets the bidder make
/// it. (Read as throws: the left opponent's trumps at trick 4 are not
/// "keeping the 5-4", and they would give 41.)
#[test]
fn dd_09() {
    let h = narrated("hand-31");
    assert_eq!(h.worth(&h.opening), (29, 0));
    let t1 = h.values(&h.at(1, R::Bidder));
    t1.assert(&[("2-2", 29, 0), ("6-2", 29, 0)]);
    assert_eq!(t1.others(&["2-2", "6-2"]).iter().map(|v| v.0).max(), Some(12));
    let p1 = h.values(&h.at(1, R::Partner));
    assert_eq!(p1.of("5-0"), (29, 0));
    assert_eq!(p1.others(&["5-0"]).iter().map(|v| v.0).max(), Some(25));
    // Refuted: the 6-2 is 25, not 24.
    h.values(&h.at(2, R::Bidder)).assert(&[("6-5", 24, 0), ("6-4", 29, 0), ("6-2", 25, 0)]);
    h.values(&h.at(2, R::Partner)).assert(&[("6-6", 24, 0), ("6-0", 23, 0), ("6-1", 23, 0)]);
    // Refuted: the 4-1 is 30 and makes, not 29.
    h.values(&h.at(2, R::Opponent2)).assert(&[("4-0", 24, 0), ("4-1", 30, 1), ("5-5", 29, 0)]);
    let p3 = h.values(&h.at(3, R::Partner));
    assert_eq!(p3.of("0-0"), (24, 0));
    assert!(p3.others(&["0-0"]).iter().all(|v| v.0 < 24));
    // Refuted: the 5-3 is 25, not 24.
    h.values(&h.at(4, R::Partner)).assert(&[("3-1", 24, 0), ("5-3", 25, 0), ("6-0", 19, 0), ("6-1", 19, 0)]);
    h.values(&h.at(4, R::Opponent1)).assert(&[("1-1", 24, 0), ("5-4", 24, 0), ("4-2", 41, 1), ("5-2", 41, 1)]);
    // Refuted: the 4-1 is 25, not 24.
    h.values(&h.at(5, R::Opponent2)).assert(&[("4-1", 25, 0), ("4-4", 24, 0), ("5-5", 24, 0)]);
    // Refuted: either trump is 25, not 24.
    h.values(&h.at(5, R::Opponent1)).assert(&[("4-2", 25, 0), ("5-2", 25, 0), ("1-1", 30, 1)]);
    // The narrated position from trick 4 is worth 25.
    assert_eq!(h.worth(&h.at(4, R::Partner)), (25, 0));
    assert_eq!(h.values(&h.at(3, R::Opponent2)).of("5-1"), (25, 0));
}

/// CONFIRMED. Hand 32 (blanks, bid 36) is worth 31: 36 cannot be made. Trick
/// 1: the partner's 4-1 keeps 31 and his 6-3 gives 26; the right opponent's
/// 6-2 keeps 31 and his 3-3 gives 42. Trick 2: the right opponent's 5-3 keeps
/// 31, his 3-3 gives 42. Trick 3: the left opponent's 6-4 or 5-5 sets (31),
/// anything else gives 41; the right opponent's 3-3 is his only setting play.
/// Trick 4: the bidder's 1-0 keeps 31, his 2-2 gives 20.
#[test]
fn dd_10() {
    let h = narrated("hand-32");
    assert_eq!(h.worth(&h.opening), (31, 0));
    h.values(&h.at(1, R::Partner)).assert(&[("4-1", 31, 0), ("6-3", 26, 0)]);
    h.values(&h.at(1, R::Opponent2)).assert(&[("6-2", 31, 0), ("3-3", 42, 1)]);
    h.values(&h.at(2, R::Opponent2)).assert(&[("5-3", 31, 0), ("3-3", 42, 1)]);
    let o1 = h.values(&h.at(3, R::Opponent1));
    o1.assert(&[("6-4", 31, 0), ("5-5", 31, 0)]);
    assert!(o1.others(&["6-4", "5-5"]).iter().all(|v| *v == (41, 1)));
    let o2 = h.values(&h.at(3, R::Opponent2));
    assert_eq!(o2.of("3-3"), (31, 0));
    assert!(o2.others(&["3-3"]).iter().all(|v| v.1 == 1));
    h.values(&h.at(4, R::Bidder)).assert(&[("1-0", 31, 0), ("2-2", 20, 0)]);
}

/// Partly REFUTED. Hand 33 (fours, bid 35) is worth 42. Confirmed: trick 2,
/// the bidder's 1-1, 4-0 or 5-5 keeps 42 and the right opponent's 2-2 gives
/// 42; trick 4, the right opponent's 5-0, 1-0 and 3-1 give 41; the bid is
/// made after every one of these; every other narrated play with a choice
/// is margin-best.
///
/// REFUTED, the lead's value then ours: trick 2, the bidder's 6-4: 40, ours
/// 41; the right opponent's 3-0: 40, ours 41; trick 3, the partner's 5-2 or
/// 6-2: 40, ours 41, and his 3-2: 35, ours 36; trick 4, the right
/// opponent's 0-0: 40, ours 41.
#[test]
fn dd_11() {
    let h = narrated("hand-33");
    assert_eq!(h.worth(&h.opening), (42, 1));
    h.values(&h.at(2, R::Bidder)).assert(&[("6-4", 41, 1), ("1-1", 42, 1), ("4-0", 42, 1), ("5-5", 42, 1)]);
    h.values(&h.at(2, R::Opponent2)).assert(&[("3-0", 41, 1), ("2-2", 42, 1)]);
    h.values(&h.at(3, R::Partner)).assert(&[("5-2", 41, 1), ("6-2", 41, 1), ("3-2", 36, 1)]);
    h.values(&h.at(4, R::Opponent2)).assert(&[("5-0", 41, 1), ("1-0", 41, 1), ("3-1", 41, 1), ("0-0", 41, 1)]);
    let short: Vec<usize> = narrated_grades(&h).filter(|g| !g.margin.optimal()).map(|g| g.ply).collect();
    assert_eq!(short, vec![4]); // the bidder's 6-4 at trick 2
}

/// CONFIRMED. Hand 34 (fives, 84) is made from the opening lead. Trick 1:
/// the bidder's 5-5 sets, and every line after it is set; his 5-4, 5-2, 5-0
/// or 2-2 makes. After the 5-4 the partner overtakes with the 6-5 and leads
/// the 6-2, and the left opponent's only legal play is the 6-4. Trick 3: the
/// left opponent's 2-0, 3-0 or 3-3 keeps the set, his 4-3 or 6-4 lets it be
/// made; the right opponent's 2-1 or 3-2 gives 31, anything else 26. Trick 4:
/// the left opponent's 3-0 or 3-3 keeps the set. Trick 5: his 3-3 sets, his
/// 4-3 or 6-4 lets it be made. Trick 6: the bidder's 4-4 makes, his 4-2 is
/// set. The left opponent holds the 6-4 and the 4-3.
#[test]
fn dd_12() {
    let h = narrated("hand-34");
    let p = h.physics();
    assert_eq!(h.worth(&h.opening), (42, 1));
    h.values(&h.opening).assert(&[("5-5", 30, 0), ("5-4", 42, 1), ("5-2", 42, 1), ("5-0", 42, 1), ("2-2", 42, 1)]);
    assert_eq!(h.worth(&h.opening.play(p, tile("5-5")).unwrap()).1, 0);
    let (after, _) = h.follow(&h.opening, &[Some("5-4"), None, Some("6-5"), None]);
    assert_eq!(after.leader(), Seat::S2);
    let (before_o1, _) = h.follow(&after, &[Some("6-2"), None, None]);
    assert_eq!(before_o1.next_actor(), Some(Seat::S1));
    assert_eq!(before_o1.legal(p), tiles(&["6-4"]));
    h.values(&h.at(3, R::Opponent1)).assert(&[
        ("2-0", 26, 0),
        ("3-0", 26, 0),
        ("3-3", 26, 0),
        ("4-3", 42, 1),
        ("6-4", 42, 1),
    ]);
    let o2 = h.values(&h.at(3, R::Opponent2));
    o2.assert(&[("2-1", 31, 0), ("3-2", 31, 0)]);
    assert!(o2.others(&["2-1", "3-2"]).iter().all(|v| *v == (26, 0)));
    h.values(&h.at(4, R::Opponent1)).assert(&[("3-0", 31, 0), ("3-3", 31, 0)]);
    h.values(&h.at(5, R::Opponent1)).assert(&[("3-3", 31, 0), ("4-3", 42, 1), ("6-4", 42, 1)]);
    h.values(&h.at(6, R::Bidder)).assert(&[("4-4", 42, 1), ("4-2", 30, 0)]);
    assert!(tiles(&["6-4", "4-3"]).is_subset_of(h.opening.hand(Seat::S1)));
}

/// CONFIRMED. Hand 35 (fours, 84) is made from the opening lead. Trick 1:
/// the bidder's 4-4 sets; his 4-3, 4-0, 5-5 or 6-6 makes. After the 4-3 the
/// partner's 5-4 wins and he leads the 6-2, the right opponent follows with
/// the 6-1, the bidder's 6-3 wins, and the bidder's 6-6 later forces out the
/// 6-5. After the 4-4 every left-opponent play at tricks 1 and 2 keeps the
/// set. Trick 3: the right opponent's 1-1 or 2-0 keeps the set, his 5-1, 6-1
/// or 6-5 lets it be made. Trick 4: only his 2-0 keeps it. Trick 5: his 5-1
/// sets, his 6-5 makes. Trick 6: the 6-1 is his only setting play.
#[test]
fn dd_13() {
    let h = narrated("hand-35");
    let p = h.physics();
    assert_eq!(h.worth(&h.opening), (42, 1));
    h.values(&h.opening).assert(&[("4-4", 36, 0), ("4-3", 42, 1), ("4-0", 42, 1), ("5-5", 42, 1), ("6-6", 42, 1)]);
    let (after, _) = h.follow(&h.opening, &[Some("4-3"), None, Some("5-4"), None]);
    assert_eq!(after.leader(), Seat::S2);
    let (after, _) = h.follow(&after, &[Some("6-2"), Some("6-1"), Some("6-3"), None]);
    assert_eq!(after.leader(), Seat::S0);
    let (before_o2, _) = h.follow(&after, &[Some("6-6"), None, None]);
    assert_eq!(before_o2.legal(p), tiles(&["6-5"]));
    for trick in [1, 2] {
        assert!(h.values(&h.at(trick, R::Opponent1)).moves.iter().all(|(_, _, m)| *m == 0));
    }
    h.values(&h.at(3, R::Opponent2)).assert(&[
        ("1-1", 36, 0),
        ("2-0", 36, 0),
        ("5-1", 42, 1),
        ("6-1", 42, 1),
        ("6-5", 42, 1),
    ]);
    let t4 = h.values(&h.at(4, R::Opponent2));
    assert_eq!(t4.of("2-0").1, 0);
    assert!(t4.others(&["2-0"]).iter().all(|v| v.1 == 1));
    h.values(&h.at(5, R::Opponent2)).assert(&[("5-1", 41, 0), ("6-5", 42, 1)]);
    let t6 = h.values(&h.at(6, R::Opponent2));
    assert_eq!(t6.of("6-1"), (41, 0));
    assert!(t6.others(&["6-1"]).iter().all(|v| v.1 == 1));
}

/// CONFIRMED. Each narrated line, played as far as it is legal and finished
/// by best play where the prose hands the rest to one side, comes to the
/// printed score: Hand 28 the setters 17; Hand 29 the setters 22 after two
/// tricks; Hand 30 42 to 0; Hand 31 the setters 17; Hand 32 the setters 11;
/// Hand 33 41 to 1, with the right opponent's trick-6 0-0 a revoke (his only
/// legal tile is the 3-1); Hand 34 all 42; Hand 35 set on the last trick.
/// (The annotation-order conflicts are the fixture's, `tests/book_fixture.rs`.)
#[test]
fn dd_14() {
    let banked = |id: &str, n: usize| {
        let h = narrated(id);
        h.after(n).banked()
    };
    let rest = |id: &str| {
        let h = narrated(id);
        let end = h.after(h.plays.len());
        h.worth(&end).0
    };
    assert_eq!(banked("hand-28", 28), [25, 17]);
    assert_eq!(banked("hand-29", 8)[1], 22);
    assert_eq!(rest("hand-30"), 42);
    assert_eq!(42 - rest("hand-31"), 17);
    assert_eq!(42 - rest("hand-32"), 11);
    let h33 = narrated("hand-33");
    let before = h33.after(23);
    assert_eq!(before.next_actor(), Some(Seat::S3));
    assert_eq!(h33.plays[23], tile("0-0"));
    assert_eq!(before.legal(h33.physics()), tiles(&["3-1"]));
    assert_eq!(h33.worth(&before).0, 41);
    assert_eq!(rest("hand-34"), 42);
    let h35 = narrated("hand-35");
    let end = h35.after(28);
    assert_eq!((end.banked(), end.leader()), ([41, 1], Seat::S3));
}

fn narrated_grades(h: &Narrated) -> impl Iterator<Item = Grade> {
    // The legal prefix of the narrated line.
    let mut line = Vec::new();
    let mut position = h.opening.clone();
    for d in &h.plays {
        match position.play(h.physics(), *d) {
            Ok(next) => position = next,
            Err(_) => break,
        }
        line.push(*d);
    }
    let deal = Deal { physics: h.physics.clone(), contract: h.opening.contract(), hands: h.opening.hands() };
    grade_line(&deal, &line).unwrap().into_iter().filter(|g| g.legal.len() > 1)
}

// ---------------------------------------------------------------------------
// 5b. Probability and arithmetic leads.

fn big_budget() -> Budget {
    Budget::new(1_000_000_000)
}

fn hand_of(id: &str) -> DominoSet {
    book().example(id).hand()
}

fn root(decl: &str, hand: DominoSet) -> Frame {
    let kernel = Kernel::new(Seat::S0, hand, DominoSet::FULL.difference(hand), open_hidden([7, 7, 7])).unwrap();
    Frame::new(physics(decl), kernel, Seat::S0, vec![], DominoSet::EMPTY).unwrap()
}

fn trumps(frame: &Frame) -> DominoSet {
    frame.physics().incidence(Context::Called).intersection(frame.kernel().pool())
}

fn resolved(frame: Frame, tiles: DominoSet) -> Belief {
    let belief = Belief::uniform_resolving(frame, tiles, 100_000).unwrap();
    assert_eq!(belief.authority(), &Authority::Exact);
    belief
}

fn probability(query: &str, belief: &Belief) -> BigRational {
    exact(compile(query).summarize(belief, &mut big_budget()).unwrap().event_probability())
}

/// The query's exact probability at a root frame, resolving its unseen trumps.
fn over_trumps(query: &str, decl: &str, hand: DominoSet) -> BigRational {
    let frame = root(decl, hand);
    let marked = trumps(&frame);
    probability(query, &resolved(frame, marked))
}

const OPPONENT_THREE: &str = odds!("c16-3t-opponent-three");

/// `x` holds three distinct trumps.
const HIDDEN_THREE: &str = "(query (roles (chair v) (chair x) (domino a) (domino b) (domino c)) (out)
  (pattern (viewer v) (in a q*) (in b q*) (in c q*) (holds x a) (holds x b) (holds x c)))";
const LEFT_THREE: &str = "(query (roles (chair x) (domino a) (domino b) (domino c)) (out)
  (pattern (chair x S1) (in a q*) (in b q*) (in c q*) (holds x a) (holds x b) (holds x c)))";

/// CONFIRMED. Hand 33, the bidder's information after trick 1 (his tiles;
/// the left opponent followed 4-2, the partner 4-3, the right opponent threw
/// the 6-0 and so is out of trumps): the left opponent holds at least one of
/// the two trumps still out, 17/22; the partner holds both, 5/22. In Hand
/// 32 at the same point (both the partner and the right opponent off trumps)
/// the left opponent holds the last one for certain.
#[test]
fn ex_01() {
    let at = |id: &str, decl: &str, bid: u32, plays: [&str; 4]| {
        let plays: Vec<(Seat, Domino)> =
            [Seat::S0, Seat::S1, Seat::S2, Seat::S3].into_iter().zip(plays.map(tile)).collect();
        let hand = hand_of(id).difference(DominoSet::single(plays[0].1));
        let frame =
            Frame::from_record(physics(decl), Seat::S0, hand, Contract { declarer: Seat::S0, bid }, plays).unwrap();
        let marked = trumps(&frame);
        (marked, resolved(frame, marked))
    };
    let left_one = "(query (roles (domino t)) (out) (pattern (in t q*) (holds S1 t)))";
    let partner_two =
        "(query (roles (domino a) (domino b)) (out) (pattern (in a q*) (in b q*) (holds S2 a) (holds S2 b)))";
    let (out, belief) = at("hand-33", "P4", 35, ["4-4", "4-2", "4-3", "6-0"]);
    assert_eq!(out, tiles(&["4-1", "5-4"]));
    assert_eq!(probability(left_one, &belief), r(17, 22));
    assert_eq!(probability(partner_two, &belief), r(5, 22));
    let (out, belief) = at("hand-32", "P0", 36, ["0-0", "2-0", "4-1", "6-2"]);
    assert_eq!(out, tiles(&["4-0"]));
    assert_eq!(probability(left_one, &belief), r(1, 1));
}

/// CONFIRMED, with the refinements computed. Four trumps out (Hand 2's
/// threes): an opponent holds three or more, 10/57; one given opponent,
/// 5/57. Hand 13 (Hand 2's tiles played): the 3-5 beats the bidder's 3-4,
/// and an opponent holds three or more trumps with the 3-5 among them 23/171
/// (the lead names no number). Hand 19, read as sixes (its only three-tile
/// suit; the book never names it): an opponent with three or more of the four
/// trumps out always keeps one above the 6-2 (probability of the contrary,
/// 0).
#[test]
fn ex_02() {
    let hand_2 = hand_of("hand-2");
    assert_eq!(over_trumps(OPPONENT_THREE, "P3", hand_2), r(10, 57));
    assert_eq!(over_trumps(LEFT_THREE, "P3", hand_2), r(5, 57));
    let p3 = physics("P3");
    assert!(p3.beats(Context::Called, tile("3-4")).contains(tile("3-5")));
    assert_eq!(hand_of("hand-13"), hand_2);
    let with_35 = "(query (roles (chair v) (chair x) (domino a) (domino b)) (out)
      (pattern (viewer v) (opponent v x) (in a q*) (in b q*) (not (tile a 3-5)) (not (tile b 3-5))
        (holds x 3-5) (holds x a) (holds x b)))";
    assert_eq!(over_trumps(with_35, "P3", hand_2), r(23, 171));
    let hand_19 = hand_of("hand-19");
    assert_eq!(over_trumps(OPPONENT_THREE, "P6", hand_19), r(10, 57));
    let none_above = "(query (roles (chair v) (chair x) (domino a) (domino b) (domino c)) (out)
      (pattern (viewer v) (opponent v x) (in a q*) (in b q*) (in c q*) (holds x a) (holds x b) (holds x c)
        (none (roles (domino u)) (in u q*) (beats u 6-2 q*) (holds x u))))";
    assert_eq!(over_trumps(none_above, "P6", hand_19), r(0, 1));
}

/// CONFIRMED. Hand 2 (threes) has four treys out: one given hand holds all
/// four 1/171, an opponent 2/171, any of the three hands 1/57.
#[test]
fn ex_03() {
    let hand = hand_of("hand-2");
    let four = |who: &str| {
        format!(
            "(query (roles (chair v) (chair x) (domino a) (domino b) (domino c) (domino d)) (out)
              (pattern (viewer v) {who} (in a q*) (in b q*) (in c q*) (in d q*)
                (holds x a) (holds x b) (holds x c) (holds x d)))"
        )
    };
    assert_eq!(over_trumps(&four("(chair x S1)"), "P3", hand), r(1, 171));
    assert_eq!(over_trumps(&four("(opponent v x)"), "P3", hand), r(2, 171));
    assert_eq!(over_trumps(&four(""), "P3", hand), r(1, 57));
}

/// CONFIRMED for the numbers; BLOCKED for "always sets". Three trumps out
/// (Hands 17 and 18, each with four trumps): one given hand holds all three
/// 1/38, any hand 3/38, an opponent 1/19. When an opponent holds all three he
/// holds a trump above the bidder's second-lowest (5-4 and 5-3 over the 5-2
/// in Hand 17; 5-2 and 4-2 over the 3-2 in Hand 18): the contrary has
/// probability 0. That such a holding *always sets* the 84 is an outcome of
/// play under every seat's choices: gap 7 (play-out over a belief).
#[test]
fn ex_04() {
    for (id, decl, low) in [("hand-17", "P5", "5-2"), ("hand-18", "P2", "3-2")] {
        let hand = hand_of(id);
        assert_eq!(trumps(&root(decl, hand)).len(), 3);
        assert_eq!(over_trumps(LEFT_THREE, decl, hand), r(1, 38));
        assert_eq!(over_trumps(HIDDEN_THREE, decl, hand), r(3, 38));
        assert_eq!(over_trumps(OPPONENT_THREE, decl, hand), r(1, 19));
        let kept_low = format!(
            "(query (roles (chair v) (chair x) (domino a) (domino b) (domino c)) (out)
              (pattern (viewer v) (opponent v x) (in a q*) (in b q*) (in c q*) (holds x a) (holds x b) (holds x c)
                (none (roles (domino u)) (in u q*) (beats u {low} q*) (holds x u))))"
        );
        assert_eq!(over_trumps(&kept_low, decl, hand), r(0, 1));
    }
}

/// CONFIRMED (already by `tests/book_odds.rs`; pinned again). One missing
/// tile is with the partner 1/3 and an opponent 2/3. From the setter's seat,
/// unable to follow the bidder's off, his partner holds the winning double
/// 7/20, or 1/2 once he knows the bidder lacks it.
#[test]
fn ex_05() {
    let hand_3 = ("P4", ["5-2", "4-1", "6-6", "6-4", "4-2", "4-3", "4-4"]);
    let belief = resolved(root(hand_3.0, tiles(&hand_3.1)), tiles(&["5-5"]));
    assert_eq!(probability(odds!("c02-partner-double-five"), &belief), r(1, 3));
    assert_eq!(probability(odds!("c02-opponent-double-five"), &belief), r(2, 3));
    let hand_23 = ("P6", ["6-1", "1-1", "6-6", "0-0", "3-3", "5-4", "6-5"]);
    let belief = resolved(root(hand_23.0, tiles(&hand_23.1)), tiles(&["5-0"]));
    assert_eq!(probability(odds!("c09-partner-five-blank"), &belief), r(1, 3));
    let straight_off = root("P5", tiles(&["5-5", "5-4", "5-3", "5-2", "6-6", "4-4", "3-1"]));
    let doubles: DominoSet = straight_off.kernel().pool().iter().filter(|d| d.is_double()).collect();
    assert_eq!(probability(odds!("c07-straight-off-double"), &resolved(straight_off, doubles)), r(2, 3));
    let setter = tiles(&["6-6", "6-5", "4-4", "4-1", "3-3", "3-1", "1-0"]);
    let contract = Contract { declarer: Seat::S0, bid: 30 };
    let frame = Frame::from_record(physics("P5"), Seat::S1, setter, contract, vec![(Seat::S0, tile("2-0"))]).unwrap();
    let belief = resolved(frame, tiles(&["2-2"]));
    let winner = odds!("c05-setter-partner-holds-winner");
    assert_eq!(probability(winner, &belief), r(7, 20));
    let not_bidders = "(query (roles (chair b) (context q) (domino d)) (out) (pattern (declarer b) (led-context q) (double d) (in d q) (not (holds b d))))";
    let (known, _) = belief.condition(&compile(not_bidders), "not-the-bidders", &mut big_budget()).unwrap();
    assert_eq!(probability(winner, &known), r(1, 2));
}

/// CONFIRMED (already by `tests/book_odds.rs`; pinned again). Hand 25's two
/// outstanding doubles (read with doubles trump, as `odds/` reads it): some
/// opponent holds neither, 23/30.
#[test]
fn ex_06() {
    let hand = tiles(&["5-0", "5-5", "2-2", "4-4", "1-1", "2-1", "3-3"]);
    assert_eq!(hand, hand_of("hand-25"));
    assert_eq!(over_trumps(odds!("c09-doubles-trump-void"), "DT", hand), r(23, 30));
}

/// CONFIRMED (already by `tests/book_odds.rs`; pinned again). Five trumps
/// out (Hand 26): neither opponent holds three or more 617/969, some
/// opponent does 352/969, no hand holds more than two 147/323, the partner
/// holds three or more 176/969.
#[test]
fn ex_07() {
    let hand = hand_of("hand-26");
    assert_eq!(over_trumps(odds!("c16-2t-favorable"), "P4", hand), r(617, 969));
    assert_eq!(over_trumps(odds!("c12-two-trump-outtrumped"), "P4", hand), r(352, 969));
    assert_eq!(over_trumps(odds!("c16-2t-spread"), "P4", hand), r(147, 323));
    assert_eq!(over_trumps(odds!("c16-2t-partner-three"), "P4", hand), r(176, 969));
}

/// CONFIRMED (already by `tests/book_odds.rs`; pinned again). Four trumps
/// held without the second: an opponent guards it 33/95. Without the second
/// and third: an opponent holds two of the three above the bidder's next
/// 47/95; one opponent holds both top trumps out 1/5, or 19/47 given the
/// previous event. Three trumps held without the second: guarded 388/855;
/// the partner holds it 1/3; an opponent holds three or four of the four out
/// 10/57.
#[test]
fn ex_08() {
    let four_no_second = tiles(&["5-5", "5-4", "5-2", "5-0", "4-4", "4-2", "2-2"]);
    assert_eq!(over_trumps(odds!("c16-4t-guarded-second"), "P5", four_no_second), r(33, 95));
    let no_second_third = tiles(&["5-5", "5-3", "5-2", "5-0", "6-6", "4-4", "1-0"]);
    let two_of_three = odds!("c16-4t-two-of-three");
    assert_eq!(over_trumps(two_of_three, "P5", no_second_third), r(47, 95));
    assert_eq!(over_trumps(odds!("c16-4t-both-high"), "P5", no_second_third), r(1, 5));
    let frame = root("P5", no_second_third);
    let marked = trumps(&frame);
    let (bad, _) =
        resolved(frame, marked).condition(&compile(two_of_three), "two-of-three", &mut big_budget()).unwrap();
    assert_eq!(probability(odds!("c16-4t-both-high"), &bad), r(19, 47));
    let three_no_second = tiles(&["6-6", "6-4", "6-0", "5-5", "4-4", "3-1", "2-0"]);
    assert_eq!(over_trumps(odds!("c16-3t-guarded-second"), "P6", three_no_second), r(388, 855));
    assert_eq!(over_trumps(odds!("c16-3t-partner-second"), "P6", three_no_second), r(1, 3));
    assert_eq!(over_trumps(OPPONENT_THREE, "P6", three_no_second), r(10, 57));
}

/// CONFIRMED, the two figures `odds/` left open included. Over all
/// 1,184,040 seven-tile hands: longest suit exactly four 26.8% (reading
/// "exactly four of a suit" as the longest suit, the reading that gives the
/// lead's figure), four or more 29.5%; seven, six, five or four suits present
/// 42.31, 46.98, 10.35, 0.35%; zero to six doubles 9.82, 32.08, 36.09, 17.69,
/// 3.93, 0.37, 0.0124% (147 hands with six); two to four doubles 57.7%, two
/// or more 58.1%; the commonest shape (two doubles, one void suit) 18.0%,
/// the next (one double, no void) 16.2%. A whole-deal question is beyond a
/// Silas frame (gap 6, deal view), so this is an exact enumeration in plain
/// Rust, not a belief.
#[test]
fn ex_09() {
    let suits: [u32; 7] =
        Pip::ALL.map(|p| Domino::ALL.iter().filter(|d| d.has(p)).fold(0, |m, d| m | DominoSet::single(*d).bits()));
    let doubles = Domino::ALL.iter().filter(|d| d.is_double()).fold(0, |m, d| m | DominoSet::single(*d).bits());
    let (mut hands, mut present, mut pairs) = (0u64, [0u64; 8], [0u64; 8]);
    let (mut longest_four, mut four_or_more) = (0u64, 0u64);
    let mut shape = std::collections::BTreeMap::<(usize, usize), u64>::new();
    let mut hand = (1u32 << 7) - 1;
    while hand < 1 << 28 {
        let lengths = suits.map(|m| (hand & m).count_ones());
        let longest = *lengths.iter().max().unwrap();
        let shown = lengths.iter().filter(|n| **n > 0).count();
        let doubled = (hand & doubles).count_ones() as usize;
        hands += 1;
        present[shown] += 1;
        pairs[doubled] += 1;
        longest_four += u64::from(longest == 4);
        four_or_more += u64::from(longest >= 4);
        *shape.entry((doubled, 7 - shown)).or_default() += 1;
        let low = hand & hand.wrapping_neg();
        let ripple = hand + low;
        hand = (((ripple ^ hand) >> 2) / low) | ripple;
    }
    assert_eq!(hands, 1_184_040);
    // Rounded to `places` decimals of a percent.
    let pct = |n: u64, places: u32| {
        let scale = 100 * 10u64.pow(places);
        (2 * n * scale + hands) / (2 * hands)
    };
    assert_eq!((pct(longest_four, 1), pct(four_or_more, 1)), (268, 295));
    assert_eq!([7, 6, 5, 4].map(|k| pct(present[k], 2)), [4231, 4698, 1035, 35]);
    assert_eq!((0..6).map(|k| pct(pairs[k], 2)).collect::<Vec<_>>(), [982, 3208, 3609, 1769, 393, 37]);
    assert_eq!((pairs[6], pct(pairs[6], 4)), (147, 124));
    assert_eq!(pct(pairs[2] + pairs[3] + pairs[4], 1), 577);
    assert_eq!(pct(hands - pairs[0] - pairs[1], 1), 581);
    let mut ranked: Vec<_> = shape.into_iter().collect();
    ranked.sort_by_key(|(_, n)| std::cmp::Reverse(*n));
    assert_eq!((ranked[0].0, pct(ranked[0].1, 1)), ((2, 1), 180));
    assert_eq!((ranked[1].0, pct(ranked[1].1, 1)), ((1, 0), 162));
}

/// BLOCKED (gap 7). Hand 3 with fours trump: that the bidder takes six of
/// seven tricks in every deal is a guarantee for one line of play against
/// every defence in every world, a play-out over the belief. Pinned here
/// only the tile facts the argument starts from: two trumps are out, the
/// 4-0 and 5-4, and the bidder's 4-4 and 6-4 beat both.
#[test]
fn ex_10() {
    let frame = root("P4", hand_of("hand-3"));
    assert_eq!(trumps(&frame), tiles(&["4-0", "5-4"]));
    let p = frame.physics();
    for top in ["4-4", "6-4"] {
        assert!(p.beats(Context::Called, tile(top)).is_disjoint(tiles(&["4-0", "5-4"])));
    }
}

/// CONFIRMED for 1/3; ours for the 53%; BLOCKED for the 5%. With the double
/// deuce ahead of the deuce-blank (four deuces out), the partner holds the
/// top deuce 1/3. The 53% needs a follower convention; on `odds/`'s reading
/// (the partner gives the 3-2, an opponent only when forced) it is exactly
/// 467/855 = 54.6%, so the lead's "not reproducible without a stated
/// policy" stands. The 5% is a two-trick play-out: gap 7.
#[test]
fn ex_11() {
    let frame = root("P6", tiles(&["6-6", "6-5", "6-4", "6-2", "2-2", "2-0", "1-1"]));
    let deuces = frame.physics().incidence(Context::Natural(Pip::new(2).unwrap())).intersection(frame.kernel().pool());
    assert_eq!(deuces.len(), 4);
    let belief = resolved(frame, deuces);
    assert_eq!(probability(odds!("c03-double-ahead-forces-count"), &belief), r(467, 855));
    assert_eq!(probability(odds!("c03-off-first-partner"), &belief), r(1, 3));
}

/// BLOCKED for the main claim; one tile fact REFUTED. Hand 16 (= Hand 8,
/// threes): that after the narrated tricks 1-4 the bidder is set only on one
/// layout is an outcome of play over the worlds (gap 7), and the record
/// cannot even be built: the prose leaves four of trick 1's and trick 4's
/// plays unnamed. The separate trick-3 claim, checked on the tiles: with the
/// 0-0 led, the left opponent overtrumps the bidder's 3-2 with the 3-4 or
/// 3-5, CONFIRMED; with the 3-0, REFUTED: the 3-0 is a trump below the 3-2
/// (it is no blank, so it cannot follow the 0-0 either). The 3-3, the top
/// trump, is safe: CONFIRMED.
#[test]
fn ex_12() {
    let p = physics("P3");
    let led = p.led_context(tile("0-0"));
    let over = p.beats(led, tile("3-2"));
    assert_eq!(over.intersection(tiles(&["3-4", "3-5", "3-0"])), tiles(&["3-4", "3-5"]));
    assert!(!p.follows(tile("3-0"), led));
    assert!(p.beats(led, tile("3-3")).is_empty());
}

/// CONFIRMED for the 6-5; the named threat REFUTED. Hand 12 (sixes), trick
/// 2, after the 6-6 fell at trick 1: the 6-5 lead cannot be beaten, so a
/// donation on it is safe. A 6-4 or 6-3 lead could lose, but not to the
/// 6-2, which beats neither: only to the 6-5 (or, over the 6-3, the 6-4).
#[test]
fn ex_13() {
    let p = physics("P6");
    let beaten_by = |lead: &str| p.beats(p.led_context(tile(lead)), tile(lead));
    assert_eq!(beaten_by("6-5"), tiles(&["6-6"]));
    assert_eq!(beaten_by("6-4"), tiles(&["6-6", "6-5"]));
    assert_eq!(beaten_by("6-3"), tiles(&["6-6", "6-5", "6-4"]));
    assert!(!beaten_by("6-4").contains(tile("6-2")) && !beaten_by("6-3").contains(tile("6-2")));
}

/// BLOCKED (gap 7). Hand 23 (doubles trump): the set paths are lines of play
/// over the unseen deal. Nothing is computed.
#[test]
fn ex_14() {
    assert_eq!(hand_of("hand-23").len(), 7);
}

/// Arithmetic CONFIRMED; the rest BLOCKED (gap 7; the book's override also
/// reads the trump holder's follow, gap 1). Hand 26 at 35 with 28 banked and
/// 8 left (the 5-0 and three tricks): 35 needs 7 of the 8, the 5-0 and two
/// tricks; the 5-0 and one trick reach only 34.
#[test]
fn ex_15() {
    let five_blank = physics("P4").count(tile("5-0"));
    assert_eq!(five_blank + 3, 8);
    assert_eq!(35 - 28, five_blank + 2);
    assert_eq!(28 + five_blank + 1, 34);
}

/// Arithmetic CONFIRMED; the set path BLOCKED (gap 7). Hand 25 at 30 may lose
/// 12; a lost off trick with the 6-4 costs 11 plus the count of the off.
#[test]
fn ex_16() {
    assert_eq!(42 - 30, 12);
    assert_eq!(physics("NT").count(tile("6-4")) + 1, 11);
}

/// Arithmetic CONFIRMED; the set probability BLOCKED (gap 7, under a stated
/// setter policy). Hand 1 at 36 may lose 6, and a ten-count trick costs 11:
/// a ten sets alone.
#[test]
fn ex_17() {
    let p = physics("P6");
    assert_eq!(42 - 36, 6);
    for ten in ["6-4", "5-5"] {
        assert!(p.count(tile(ten)) + 1 > 6);
    }
}

/// CONFIRMED. Hand 9 (fives): the 4-0 off can lose the 6-4 and 4-1 in one
/// trick, 16 with the trick (against the book's 11); at 31 the bidder may
/// lose 11, so 11 and one more trick (12) fails. Four trumps are out; the
/// bidder's 5-5 is the top trump, and three of those out (6-5, 5-4, 5-3)
/// beat his next one, the 5-2, so a second trump loser is possible.
#[test]
fn ex_18() {
    let p = physics("P5");
    assert_eq!(p.count(tile("6-4")) + p.count(tile("4-1")) + 1, 16);
    let allowance = |bid: u32| 42 - bid;
    let charged = p.count(tile("6-4")) + 1;
    assert_eq!((charged, allowance(31)), (11, 11));
    assert!(charged + 1 > allowance(31));
    let frame = root("P5", hand_of("hand-9"));
    let out = trumps(&frame);
    assert_eq!(out, tiles(&["5-0", "5-3", "5-4", "6-5"]));
    assert!(p.beats(Context::Called, tile("5-5")).is_empty());
    assert_eq!(p.beats(Context::Called, tile("5-2")).intersection(out), tiles(&["6-5", "5-4", "5-3"]));
}

/// BLOCKED. Nel-O is not a straight declaration and `Straight42` omits it;
/// with no Nel-O physics there is nothing exact to compute.
#[test]
fn ex_19() {
    assert!("nello".parse::<silas::reference::Straight42>().is_err());
}

/// CONFIRMED for the tile arithmetic and the 1/5; the sample-hand claim
/// BLOCKED (no Sevens physics). By pip sum: three sevens, four sixes, three
/// eights; twelve tiles sum to 0-5 and twelve to 7-12. Holding one seven, an
/// opponent holds both other sevens 1/5 (it does not depend on the
/// declaration: only who holds what is asked).
#[test]
fn ex_20() {
    let sums = |f: fn(u8) -> bool| Domino::ALL.iter().filter(|d| f(d.pip_sum())).count();
    assert_eq!((sums(|s| s == 7), sums(|s| s == 6), sums(|s| s == 8)), (3, 4, 3));
    assert_eq!((sums(|s| s <= 5), sums(|s| s >= 7)), (12, 12));
    let both =
        "(query (roles (chair v) (chair x)) (out) (pattern (viewer v) (opponent v x) (holds x 5-2) (holds x 4-3)))";
    for hand in [["6-1", "6-6", "5-5", "4-4", "3-3", "2-2", "0-0"], ["6-1", "6-0", "5-0", "4-0", "3-0", "2-0", "1-0"]] {
        let belief = resolved(root("NT", tiles(&hand)), tiles(&["5-2", "4-3"]));
        assert_eq!(probability(both, &belief), r(1, 5));
    }
}

/// CONFIRMED. Losing exactly 42 - bid points: 10 (a 32 bid) can only be five
/// tricks and one five-count; 9 (33) four tricks and a five; 3, 2 and 1 (39,
/// 40, 41) three, two and one tricks with no count. 42 - 15 = 27.
#[test]
fn ex_21() {
    let p = physics("NT");
    let counted: Vec<u32> = Domino::ALL.iter().map(|d| p.count(*d)).filter(|c| *c > 0).collect();
    assert_eq!(counted.len(), 5);
    // Each way to lose `loss`: tricks, and the count tiles lost, sorted.
    let ways = |loss: u32| {
        let mut out = std::collections::BTreeSet::new();
        for tricks in 1..=7u32 {
            for subset in 0u32..1 << counted.len() {
                let lost: Vec<u32> = (0..counted.len()).filter(|i| subset >> i & 1 == 1).map(|i| counted[i]).collect();
                if lost.len() as u32 <= 4 * tricks && tricks + lost.iter().sum::<u32>() == loss {
                    let mut lost = lost;
                    lost.sort();
                    out.insert((tricks, lost));
                }
            }
        }
        out.into_iter().collect::<Vec<_>>()
    };
    assert_eq!(ways(42 - 32), vec![(5, vec![5])]);
    assert_eq!(ways(42 - 33), vec![(4, vec![5])]);
    for (bid, tricks) in [(39, 3), (40, 2), (41, 1)] {
        assert_eq!(ways(42 - bid), vec![(tricks, vec![])]);
    }
    assert_eq!(42 - 15, 27);
}
