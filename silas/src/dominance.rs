//! The `dominates` form: one contract against another, compared over every
//! way the hand can end. No deal, belief or field enters, so the answer is
//! exact (`docs/cross-game.md` § 2.4).
//!
//! ```text
//! (dominates name
//!   (contracts (marks 1) (marks 2))
//!   (frame (scoring marks 7) (short 1))
//!   (utility match-win))
//! ```
//!
//! Both contracts have the same bidder (S0) and declaration. The form holds
//! when, from every state of the frame, for every outcome of the hand (43
//! splits of 42 points) and under both readings of a set marks bid under
//! points, the first contract is at least as good for the bidder's side as
//! the second: by the hand's marks or points, or by the match-win order
//! ([`MatchState::at_least`]), which is "at least as good under every
//! match-win value", so no value is needed.

use std::fmt;

use num_bigint::BigInt;
use num_rational::BigRational;

use crate::frame::{Bid, Contract};
use crate::measure::Authority;
use crate::nouns::{Seat, Team};
use crate::physics::Physics;
use crate::scoring::{MatchState, Outcome, Scoring, SetReading};
use crate::syntax::Expr;
use crate::{Result, error};

/// The bidder of both contracts.
const BIDDER: Seat = Seat::S0;

/// What a dominance compares by.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Criterion {
    /// The hand's net marks for the bidder's side.
    Marks,
    /// The hand's net points, under both readings of a set marks bid.
    Points,
    /// The match-win order: at least as good under every match-win value.
    MatchWin,
}

/// Where the bidder's side stands in the match. The other side is anywhere
/// short of the target.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Place {
    /// Every undecided state.
    Any,
    /// The bidder's side needs exactly `n`.
    Short(u32),
    /// The bidder's side needs `n` or less.
    ShortAtMost(u32),
    /// One state: the bidder's side's score, then the other's.
    Score(u32, u32),
}

impl Place {
    /// Whether some undecided state of a match to `target` is here.
    fn fits(self, target: u32) -> bool {
        match self {
            Place::Any => target > 0,
            Place::Short(n) | Place::ShortAtMost(n) => (1..=target).contains(&n),
            Place::Score(a, b) => a < target && b < target,
        }
    }
}

/// A `(dominates ...)` form: whether the first contract is at least as good
/// as the second for the bidder's side.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Dominance {
    pub name: String,
    pub contracts: [Bid; 2],
    pub scoring: Scoring,
    pub place: Place,
    pub criterion: Criterion,
}

/// Where a dominance fails. The reading is the one of a set marks bid under
/// points that was in force, and none where nothing is scored in points.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Counterexample {
    /// The first contract's net marks or points fall short of the second's.
    Utility { outcome: Outcome, reading: Option<SetReading>, values: [BigRational; 2] },
    /// From `state`, the first contract leaves a state that is not at least
    /// as good for the bidder's side as the second's.
    States { state: MatchState, outcome: Outcome, reading: Option<SetReading>, after: [MatchState; 2] },
}

/// A dominance decided: how much was enumerated (the readings of a set marks
/// bid are those checked, none when nothing is scored in points), and the
/// first place it fails, if any. Always exact.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Decision {
    pub states: usize,
    pub outcomes: usize,
    pub readings: Vec<SetReading>,
    pub counterexample: Option<Counterexample>,
}

impl Decision {
    pub fn holds(&self) -> bool {
        self.counterexample.is_none()
    }

    /// An enumeration over outcomes: no model, no sample.
    pub fn authority(&self) -> Authority {
        Authority::exact()
    }
}

fn number(expr: &Expr) -> Result<u32> {
    let word = expr.word()?;
    word.parse().map_err(|_| expr.span.error(format!("expected a nonnegative number, found {word}")))
}

/// The `N` items of `(tag ...)`.
fn arity<'a, const N: usize>(expr: &'a Expr, tag: &str, shape: &str) -> Result<&'a [Expr; N]> {
    expr.tagged(tag)?.try_into().map_err(|_| expr.span.error(format!("{tag} syntax is {shape}")))
}

fn bid(expr: &Expr) -> Result<Bid> {
    match expr.head() {
        Some("points") => Ok(Bid::Points(number(&arity::<1>(expr, "points", "(points n)")?[0])?)),
        Some("marks") => Ok(Bid::Marks(number(&arity::<1>(expr, "marks", "(marks m)")?[0])?)),
        _ => Err(expr.span.error("a contract is (points n) or (marks m)")),
    }
}

fn spelled(bid: Bid) -> String {
    match bid {
        Bid::Points(n) => format!("(points {n})"),
        Bid::Marks(m) => format!("(marks {m})"),
    }
}

impl Dominance {
    /// Parse a `(dominates ...)` text, naming it `name` in errors: one form,
    /// nothing before it.
    pub fn parse(text: &str, name: &str) -> Result<Dominance> {
        let forms = Expr::read_all(text, name, 1 << 20, 8, false)?;
        let [form] = forms.as_slice() else {
            let at = forms.get(1).or(forms.first());
            return Err(match at {
                Some(x) => x.span.error("a dominance file holds one (dominates ...) form and nothing else"),
                None => error(format!("{name}: expected a (dominates ...) form, found nothing")),
            });
        };
        let shape = "(dominates name (contracts c c) (frame (scoring kind n) place) (utility u))";
        let [name, contracts, frame, utility] = arity(form, "dominates", shape)?;
        let [first, second] = arity(contracts, "contracts", "(contracts c c)")?;
        let contracts = [bid(first)?, bid(second)?];
        let frame_items = frame.tagged("frame")?;
        let (scoring, place) = match frame_items {
            [scoring] => (scoring, None),
            [scoring, place] => (scoring, Some(place)),
            _ => return Err(frame.span.error("frame syntax is (frame (scoring kind n)) or with one place after it")),
        };
        let [kind, target] = arity(scoring, "scoring", "(scoring points n) or (scoring marks n)")?;
        let target = number(target)?;
        if target == 0 {
            return Err(scoring.span.error("a match is to one or more"));
        }
        let scoring = match kind.word()? {
            "points" => Scoring::Points { target },
            "marks" => Scoring::Marks { target },
            _ => return Err(kind.span.error("the scoring is points or marks")),
        };
        let place = match place {
            None => Place::Any,
            Some(p) => {
                let place = match p.head() {
                    Some("short") => Place::Short(number(&arity::<1>(p, "short", "(short n)")?[0])?),
                    Some("short-at-most") => {
                        Place::ShortAtMost(number(&arity::<1>(p, "short-at-most", "(short-at-most n)")?[0])?)
                    }
                    Some("score") => {
                        let [a, b] = arity(p, "score", "(score bidders-side other-side)")?;
                        Place::Score(number(a)?, number(b)?)
                    }
                    _ => return Err(p.span.error("a place is (short n), (short-at-most n) or (score n m)")),
                };
                if !place.fits(target) {
                    return Err(p.span.error(format!("no undecided state of a match to {target} is there")));
                }
                place
            }
        };
        let [utility] = arity(utility, "utility", "(utility marks), (utility points) or (utility match-win)")?;
        let criterion = match utility.word()? {
            "marks" => Criterion::Marks,
            "points" => Criterion::Points,
            "match-win" => Criterion::MatchWin,
            _ => return Err(utility.span.error("the utility is marks, points or match-win")),
        };
        Ok(Dominance { name: name.word()?.to_owned(), contracts, scoring, place, criterion })
    }

    /// The two contracts under `declaration`, bid by S0. Refused as
    /// [`Contract::new`] refuses.
    pub fn contracts(&self, declaration: &Physics) -> Result<[Contract; 2]> {
        let [a, b] = self.contracts;
        Ok([Contract::new(BIDDER, a, declaration.clone())?, Contract::new(BIDDER, b, declaration.clone())?])
    }

    /// The states the form ranges over: the bidder's side (S0's, `T0`) at
    /// its place, the other side anywhere short of the target.
    pub fn states(&self) -> Result<Vec<MatchState>> {
        let t = self.scoring.target();
        if !self.place.fits(t) {
            return Err(error(format!("no undecided state of a match to {t} is at {:?}", self.place)));
        }
        let own: Vec<u32> = match self.place {
            Place::Any => (0..t).collect(),
            Place::Short(n) => vec![t - n],
            Place::ShortAtMost(n) => (t - n..t).collect(),
            Place::Score(a, _) => vec![a],
        };
        let other: Vec<u32> = match self.place {
            Place::Score(_, b) => vec![b],
            _ => (0..t).collect(),
        };
        own.iter().flat_map(|&a| other.iter().map(move |&b| MatchState::new(self.scoring, [a, b]))).collect()
    }

    /// Decide the form under `declaration` by enumeration: every state, every
    /// outcome, and both readings of a set marks bid wherever points are
    /// scored. Exact.
    pub fn decide(&self, declaration: &Physics) -> Result<Decision> {
        let [first, second] = self.contracts(declaration)?;
        let side = BIDDER.team();
        let states = self.states()?;
        let total = declaration.total_points();
        let points = match self.criterion {
            Criterion::Points => true,
            Criterion::Marks => false,
            Criterion::MatchWin => matches!(self.scoring, Scoring::Points { .. }),
        };
        let readings: Vec<Option<SetReading>> = if points { SetReading::BOTH.map(Some).to_vec() } else { vec![None] };
        let mut decision = Decision {
            states: states.len(),
            outcomes: total as usize + 1,
            readings: readings.iter().flatten().copied().collect(),
            counterexample: None,
        };
        for &reading in &readings {
            let read = reading.unwrap_or(SetReading::BidAndPoints);
            for outcome in Outcome::all(total) {
                decision.counterexample = match self.criterion {
                    Criterion::Marks | Criterion::Points => {
                        let scoring = if points { Scoring::POINTS } else { Scoring::MARKS };
                        let net =
                            |k: &Contract| -> Result<i64> { Ok(k.settle_reading(scoring, outcome, read)?.net(side)) };
                        let values = [net(&first)?, net(&second)?];
                        (values[0] < values[1]).then(|| Counterexample::Utility {
                            outcome,
                            reading,
                            values: values.map(|v| BigRational::from_integer(BigInt::from(v))),
                        })
                    }
                    Criterion::MatchWin => {
                        let mut found = None;
                        for state in &states {
                            let after = [
                                state.apply_reading(&first, outcome, read)?,
                                state.apply_reading(&second, outcome, read)?,
                            ];
                            if !after[0].at_least(&after[1], side) {
                                found = Some(Counterexample::States { state: *state, outcome, reading, after });
                                break;
                            }
                        }
                        found
                    }
                };
                if decision.counterexample.is_some() {
                    return Ok(decision);
                }
            }
        }
        Ok(decision)
    }

    /// Whose side the form is about: the bidder's.
    pub fn side(&self) -> Team {
        BIDDER.team()
    }
}

impl fmt::Display for Dominance {
    /// The canonical text, as `silas fmt` writes it.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (kind, target) = match self.scoring {
            Scoring::Points { target } => ("points", target),
            Scoring::Marks { target } => ("marks", target),
        };
        let place = match self.place {
            Place::Any => String::new(),
            Place::Short(n) => format!(" (short {n})"),
            Place::ShortAtMost(n) => format!(" (short-at-most {n})"),
            Place::Score(a, b) => format!(" (score {a} {b})"),
        };
        let utility = match self.criterion {
            Criterion::Marks => "marks",
            Criterion::Points => "points",
            Criterion::MatchWin => "match-win",
        };
        write!(
            f,
            "(dominates {}\n  (contracts {} {})\n  (frame (scoring {kind} {target}){place})\n  (utility {utility}))",
            self.name,
            spelled(self.contracts[0]),
            spelled(self.contracts[1])
        )
    }
}

fn read(reading: Option<SetReading>) -> String {
    reading.map_or_else(String::new, |r| format!(" ({r})"))
}

impl fmt::Display for Counterexample {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Counterexample::Utility { outcome, reading, values } => write!(
                f,
                "the hand ending {outcome}{}: the first nets {} and the second {}",
                read(*reading),
                values[0],
                values[1]
            ),
            Counterexample::States { state, outcome, reading, after } => write!(
                f,
                "from {state}, the hand ending {outcome}{}: the first leaves {} and the second {}",
                read(*reading),
                after[0],
                after[1]
            ),
        }
    }
}
