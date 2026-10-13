//! The match frame and the field-free cross-game claims (`docs/cross-game.md`,
//! slice 1): the settlement under both scorings, the book's scoring
//! definitions (cluster F: S-0015-c, S-0017-c+S-0132-a, S-0020-a,
//! S-0063-a+S-0154-b, S-0063-b, S-0072-b, S-0073-a, S-0113-a+S-0209-b,
//! S-0114-a, S-0156-a, S-0166-a, S-0166-b, S-0168-b, S-0199-b, S-0208-b,
//! S-0208-c, S-0208-e, S-0210-e), the dominances of clusters A and B by
//! enumeration of the hand's 43 outcomes, the linear-marks line of the 84,
//! the properties a match-win value must have, and the score-grid programme.

use std::num::NonZeroU64;

use num_bigint::BigInt;
use num_rational::BigRational;
use silas::reference::Straight42;
use silas::*;

fn physics() -> Physics {
    "P5".parse::<Straight42>().unwrap().physics()
}

fn contract(declarer: Seat, bid: Bid) -> Contract {
    Contract::new(declarer, bid, physics()).unwrap()
}

fn ratio(n: u64, d: u64) -> Weight {
    Weight::ratio(n, NonZeroU64::new(d).unwrap())
}

fn rational(n: i64, d: i64) -> BigRational {
    BigRational::new(BigInt::from(n), BigInt::from(d))
}

fn dominance(file: &str) -> Dominance {
    let path = format!("{}/examples/book/cross-game/{file}", env!("CARGO_MANIFEST_DIR"));
    Dominance::parse(&std::fs::read_to_string(&path).unwrap(), &path).unwrap()
}

#[test]
fn settle_reproduces_the_books_points_vectors() {
    // S-0017-c: set, the bidder's side scores nothing and the setters the bid
    // plus their points; made, each side scores what it took.
    for declarer in Seat::ALL {
        let (us, them) = (declarer.team(), declarer.team().other());
        for (bid, outcome, ours, theirs) in
            [(35, Outcome::new(30, 12), 0, 47), (30, Outcome::new(33, 9), 33, 9), (35, Outcome::new(33, 9), 0, 44)]
        {
            let k = contract(declarer, Bid::Points(bid));
            for reading in SetReading::BOTH {
                let t = k.settle_reading(Scoring::POINTS, outcome, reading).unwrap();
                assert_eq!((t.to(us), t.to(them)), (ours, theirs), "bid {bid}, {outcome}");
            }
        }
    }
    // A marks bid under points: made, its value; set, the readings part.
    let k = contract(Seat::S1, Bid::Marks(2));
    let made = k.settle(Scoring::POINTS, Outcome::new(42, 0)).unwrap();
    assert_eq!((made.to(Team::T1), made.to(Team::T0)), (84, 0));
    let set = |reading| k.settle_reading(Scoring::POINTS, Outcome::new(30, 12), reading).unwrap().to(Team::T0);
    assert_eq!((set(SetReading::BidAndPoints), set(SetReading::BidAlone)), (96, 84));
    // An outcome must split the hand's 42 points.
    assert!(k.settle(Scoring::POINTS, Outcome::new(30, 11)).unwrap_err().0.contains("splits the hand's 42 points"));
}

#[test]
fn settle_under_marks_is_the_award() {
    let bids = (30..=41).map(Bid::Points).chain((1..=4).map(Bid::Marks));
    for bid in bids {
        for declarer in Seat::ALL {
            let k = contract(declarer, bid);
            for outcome in Outcome::all(42) {
                let (team, stake) = k.award(outcome.declaring());
                for reading in SetReading::BOTH {
                    let t = k.settle_reading(Scoring::MARKS, outcome, reading).unwrap();
                    assert_eq!((t.to(team), t.to(team.other())), (stake, 0), "{bid} by {declarer}, {outcome}");
                }
            }
        }
    }
    // An 84 is two marks (S-0082-a); all seven of a suit, bid at three marks
    // and laid down, is three (S-0188-a; the lay-down is `lay-down.silas`).
    let made = |m| contract(Seat::S0, Bid::Marks(m)).settle(Scoring::MARKS, Outcome::new(42, 0)).unwrap();
    assert_eq!((made(2).to(Team::T0), made(3).to(Team::T0)), (2, 3));
}

#[test]
fn a_set_is_worth_at_least_43_and_outscores_any_make() {
    // S-0063-b: set, the setters took at least 43 - b, so they score b + (43 - b)
    // or more; the floor is reached. Under either reading: they agree on points bids.
    for b in 30..=41 {
        let k = contract(Seat::S0, Bid::Points(b));
        let mut least = u32::MAX;
        for outcome in Outcome::all(42).filter(|o| !k.makes(o.declaring())) {
            for reading in SetReading::BOTH {
                least = least.min(k.settle_reading(Scoring::POINTS, outcome, reading).unwrap().to(Team::T1));
            }
        }
        assert_eq!(least, 43, "bid {b}");
        // The set outscores every make the bid could have had (42 at most).
        let best_make = Outcome::all(42)
            .filter(|o| k.makes(o.declaring()))
            .map(|o| k.settle(Scoring::POINTS, o).unwrap().to(Team::T0))
            .max();
        assert_eq!(best_make, Some(42));
    }
    // A marks bid under the book's rule clears the floor too; under S-0082-a's
    // reading a set 42 pays the setters 42, one short of it, and a set marks bid
    // pays exactly what it would have made: the readings part here, and only here.
    for m in 1..=3 {
        let k = contract(Seat::S0, Bid::Marks(m));
        let set = |reading| k.settle_reading(Scoring::POINTS, Outcome::new(41, 1), reading).unwrap().to(Team::T1);
        assert_eq!((set(SetReading::BidAndPoints), set(SetReading::BidAlone)), (42 * m + 1, 42 * m));
    }
}

#[test]
fn needing_the_last_mark_one_mark_dominates_two() {
    let declaration = physics();
    // Marks: one mark short (S-0170-a); points: within 42 of 250 (S-0089-b),
    // under both readings of a set marks bid.
    let marks = dominance("last-mark.silas").decide(&declaration).unwrap();
    assert!(marks.holds() && marks.authority().is_exact(), "{marks:?}");
    assert_eq!((marks.states, marks.outcomes, marks.readings.len()), (7, 43, 0));
    let points = dominance("last-mark-points.silas").decide(&declaration).unwrap();
    assert!(points.holds(), "{points:?}");
    assert_eq!((points.states, points.outcomes, points.readings), (42 * 250, 43, SetReading::BOTH.to_vec()));
    // Two marks out, the 84 is not dominated: a made 84 wins, a made 42 does not.
    let out = dominance("two-marks-out.silas").decide(&declaration).unwrap();
    let Some(Counterexample::States { state, outcome, after, .. }) = &out.counterexample else {
        panic!("two marks out is not a dominance: {out:?}")
    };
    assert_eq!((state.short(Team::T0), outcome.declaring()), (2, 42));
    assert_eq!((after[0].won(), after[1].won()), (None, Some(Team::T0)));
    // The same at the boundary under points: short 43 or more, a made 42 falls short.
    let mut far = dominance("last-mark-points.silas");
    far.place = Place::ShortAtMost(43);
    assert!(!far.decide(&declaration).unwrap().holds());
}

#[test]
fn a_lower_point_bid_dominates_a_higher_one_under_both_scorings() {
    let declaration = physics();
    for file in ["bid-only-enough.silas", "bid-only-enough-points.silas"] {
        let decision = dominance(file).decide(&declaration).unwrap();
        assert!(decision.holds() && decision.authority().is_exact(), "{file}: {decision:?}");
    }
    // The same pair compared by the hand's own marks or points, and the reverse fails.
    for criterion in [Criterion::Marks, Criterion::Points] {
        let mut d = dominance("bid-only-enough.silas");
        d.criterion = criterion;
        assert!(d.decide(&declaration).unwrap().holds(), "{criterion:?}");
        d.contracts.reverse();
        assert!(!d.decide(&declaration).unwrap().holds(), "{criterion:?} reversed");
    }
    // The form prints as it reads.
    let d = dominance("bid-only-enough-points.silas");
    assert_eq!(Dominance::parse(&d.to_string(), "printed").unwrap(), d);
}

#[test]
fn in_linear_marks_the_84_wins_above_one_half_and_loses_below() {
    // S-0170-a's first half is a belief statement, not a dominance
    // (`eighty-four-in-marks.silas` fails): with make probability p for both
    // contracts, the 84 nets 2(2p - 1) and the 42 nets 2p - 1.
    let state = MatchState::new(Scoring::MARKS, [0, 0]).unwrap();
    assert!(!dominance("eighty-four-in-marks.silas").decide(&physics()).unwrap().holds());
    let value = |marks: u32, p: (u64, u64)| {
        let k = contract(Seat::S0, Bid::Marks(marks));
        let law =
            vec![(k.clone(), Outcome::new(42, 0), ratio(p.0, p.1)), (k, Outcome::new(30, 12), ratio(p.1 - p.0, p.1))];
        Utility::Marks.expect(Team::T0, &state, &format!("toy make p = {}/{}", p.0, p.1), &law).unwrap()
    };
    let zero = BigRational::from_integer(BigInt::from(0));
    for (p, verdict, eighty_four) in
        [((3, 5), Verdict::Better, rational(2, 5)), ((2, 5), Verdict::Worse, rational(-2, 5))]
    {
        let (a, b) = (value(2, p), value(1, p));
        assert_eq!((a.value(), b.value()), (&eighty_four, &(eighty_four.clone() / BigInt::from(2))));
        let (v, authority) = compare(&a, &b, &zero).unwrap();
        assert_eq!(v, verdict, "p = {}/{}", p.0, p.1);
        assert!(!authority.is_exact());
    }
}

/// The marks match where `team` declares every hand with a points bid it makes
/// with probability `p`.
fn always_bids(team: Team, p: (u64, u64)) -> impl Fn(&MatchState) -> Result<HandLaw> {
    let k = contract(if team == Team::T0 { Seat::S0 } else { Seat::S1 }, Bid::Points(30));
    move |_| {
        Ok(vec![
            (k.clone(), Outcome::new(30, 12), ratio(p.0, p.1)),
            (k.clone(), Outcome::new(29, 13), ratio(p.1 - p.0, p.1)),
        ])
    }
}

#[test]
fn a_value_failing_the_properties_is_refused() {
    let t = 7;
    let model = || Authority::given("a hand-written table").unwrap();
    // A constant one half has every property.
    let half = vec![rational(1, 2); t * t];
    let value = WinValue::new(Scoring::MARKS, Team::T0, half.clone(), model()).unwrap();
    // Rising with the opponents' score is refused.
    let mut rising = half.clone();
    rising[3 * t + 4] = rational(3, 4);
    let e = WinValue::new(Scoring::MARKS, Team::T0, rising, model()).unwrap_err().0;
    assert!(e.contains("V(3-4) = 3/4 exceeds V(3-3) = 1/2") && e.contains("other side's score"), "{e}");
    // So is falling with the side's own, leaving [0, 1], and a model unnamed.
    let mut falling = half.clone();
    falling[4 * t + 3] = rational(1, 4);
    assert!(WinValue::new(Scoring::MARKS, Team::T0, falling, model()).unwrap_err().0.contains("own score"));
    let mut big = half.clone();
    big[0] = rational(3, 2);
    assert!(WinValue::new(Scoring::MARKS, Team::T0, big, model()).unwrap_err().0.contains("not a probability"));
    assert!(WinValue::new(Scoring::MARKS, Team::T0, half, Authority::exact()).is_err());
    // A value is one on a won state and zero on a lost one, and is its side's.
    let won = MatchState::new(Scoring::MARKS, [7, 3]).unwrap();
    let lost = MatchState::new(Scoring::MARKS, [3, 8]).unwrap();
    assert_eq!((value.at(&won).unwrap(), value.at(&lost).unwrap()), (rational(1, 1), rational(0, 1)));
    let state = MatchState::new(Scoring::MARKS, [6, 6]).unwrap();
    let k = contract(Seat::S0, Bid::Points(30));
    assert!(Utility::MatchWin(value).of(Team::T1, &state, &k, Outcome::new(30, 12)).is_err());
}

#[test]
fn the_score_grid_programme_gives_the_next_hands_make_at_six_all() {
    let p = (3, 5);
    let v =
        WinValue::solve(Scoring::MARKS, Team::T0, "toy: T0 bids 30 every hand, makes 3/5", always_bids(Team::T0, p))
            .unwrap();
    assert_eq!(v.authority().to_string(), "exact_given(toy: T0 bids 30 every hand, makes 3/5)");
    let at = |a, b| v.at(&MatchState::new(Scoring::MARKS, [a, b]).unwrap()).unwrap();
    // 6-6: the next hand decides the match.
    assert_eq!(at(6, 6), rational(3, 5));
    // 6-5: make now, or be set and make the next: 3/5 + 2/5 · 3/5.
    assert_eq!(at(6, 5), rational(21, 25));
    // The other side's chance is the complement, and its bidding is its own.
    let w = WinValue::solve(Scoring::MARKS, Team::T1, "toy: T0 bids", always_bids(Team::T0, p)).unwrap();
    let state = MatchState::new(Scoring::MARKS, [2, 4]).unwrap();
    assert_eq!(w.at(&state).unwrap(), rational(1, 1) - v.at(&state).unwrap());
    // The order is sound for it: a state at least as good has at least the value.
    let states: Vec<MatchState> =
        (0..7).flat_map(|a| (0..7).map(move |b| MatchState::new(Scoring::MARKS, [a, b]).unwrap())).collect();
    for a in &states {
        for b in &states {
            if a.at_least(b, Team::T0) {
                assert!(v.at(a).unwrap() >= v.at(b).unwrap(), "{a} against {b}");
            }
        }
    }
    // Under points the grid is 250 by 250: the bidder of the deciding hand wins
    // a tie at 250, and the programme solves it whole.
    let points = WinValue::solve(Scoring::POINTS, Team::T0, "toy: T0 bids 30, makes 3/5", always_bids(Team::T0, p));
    let points = points.unwrap();
    let tie = MatchState::new(Scoring::POINTS, [240, 245]).unwrap();
    assert!(points.at(&tie).unwrap() > rational(1, 2));
    let both = tie.apply(&contract(Seat::S0, Bid::Points(30)), Outcome::new(30, 12)).unwrap();
    assert_eq!(both.won(), Some(Team::T0));
}
