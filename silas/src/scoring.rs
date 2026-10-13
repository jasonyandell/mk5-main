//! The match: the scoring, the score, and what one hand settles.
//!
//! A hand ends in an [`Outcome`], the points each side took. Its [`Contract`]
//! settles the outcome under a [`Scoring`] into a [`Transfer`]: under marks
//! the contract's award, under points the book's rule. A [`MatchState`] is
//! the score of a match under one scoring, and applying a hand moves it. A
//! [`Utility`] values a hand for one side: its marks, its points, or its
//! chance of winning the match, a [`WinValue`]. Silas checks that a value has
//! the properties the dominance theorems need ([`MatchState::at_least`]), or
//! builds one itself from a hand-outcome model by a dynamic programme over
//! the score.
//!
//! This is the book's scoring frame (Roberson, *Winning 42*; `docs/cross-game.md`
//! § 2.1, cluster F).

use std::fmt;

use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{One, Zero};

use crate::frame::{Bid, Contract};
use crate::measure::{Authority, Estimate, Weight};
use crate::nouns::Team;
use crate::{Result, error};

/// How a match is scored, and how far it goes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Scoring {
    /// First to `target` points: the book's game, to 250.
    Points { target: u32 },
    /// First to `target` marks: the tournament's game, to seven.
    Marks { target: u32 },
}

impl Scoring {
    /// The book's game.
    pub const POINTS: Scoring = Scoring::Points { target: 250 };
    /// The tournament's game.
    pub const MARKS: Scoring = Scoring::Marks { target: 7 };

    pub fn target(self) -> u32 {
        match self {
            Scoring::Points { target } | Scoring::Marks { target } => target,
        }
    }
}

impl fmt::Display for Scoring {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Scoring::Points { target } => write!(f, "points to {target}"),
            Scoring::Marks { target } => write!(f, "marks to {target}"),
        }
    }
}

/// What a set marks bid pays the setters under points. The book says it two
/// ways: its rule (S-0017-c) is the bid plus the points they took, and its 84
/// (S-0082-a) is "they get 84". The readings differ only for a set marks bid
/// under points; [`Contract::settle`] takes the rule.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SetReading {
    /// S-0017-c: the bid's value and the setters' own points.
    BidAndPoints,
    /// S-0082-a: the bid's value alone.
    BidAlone,
}

impl SetReading {
    pub const BOTH: [SetReading; 2] = [SetReading::BidAndPoints, SetReading::BidAlone];
}

impl fmt::Display for SetReading {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            SetReading::BidAndPoints => "bid and points (S-0017-c)",
            SetReading::BidAlone => "bid alone (S-0082-a)",
        })
    }
}

/// How a hand ended: the points the declaring side took and the points the
/// other side took. The two make the hand's total (42), which
/// [`Contract::settle`] checks.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Outcome {
    declaring: u32,
    other: u32,
}

impl Outcome {
    pub fn new(declaring: u32, other: u32) -> Outcome {
        Outcome { declaring, other }
    }

    pub fn declaring(self) -> u32 {
        self.declaring
    }

    pub fn other(self) -> u32 {
        self.other
    }

    /// Every split of `total` points, the declaring side's share ascending:
    /// `total + 1` outcomes, 43 for a hand of 42.
    pub fn all(total: u32) -> impl Iterator<Item = Outcome> {
        (0..=total).map(move |declaring| Outcome { declaring, other: total - declaring })
    }
}

impl fmt::Display for Outcome {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} to {}", self.declaring, self.other)
    }
}

/// What each team gains from one hand, in the scoring's unit.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Transfer {
    to: [u32; 2],
}

impl Transfer {
    pub fn to(self, team: Team) -> u32 {
        self.to[team.index()]
    }

    /// What `team` gains less what the other side gains.
    pub fn net(self, team: Team) -> i64 {
        i64::from(self.to(team)) - i64::from(self.to(team.other()))
    }
}

impl Contract {
    /// The hand's settlement under `scoring`. Marks: the award, the stake to
    /// the side that gets it. Points, by the book's rule (S-0017-c): made, the
    /// bidder's side scores the points it took (a marks bid, its value: 42 a
    /// mark) and the other side the points it took; set, the bidder's side
    /// scores nothing and the other side the bid's value plus the points it
    /// took. The other reading of a set marks bid, "they get 84" (S-0082-a),
    /// is [`Contract::settle_reading`] with [`SetReading::BidAlone`]. Refused
    /// unless the outcome splits the hand's points.
    pub fn settle(&self, scoring: Scoring, outcome: Outcome) -> Result<Transfer> {
        self.settle_reading(scoring, outcome, SetReading::BidAndPoints)
    }

    /// [`Contract::settle`] under either reading of a set marks bid under
    /// points. The readings agree everywhere else.
    pub fn settle_reading(&self, scoring: Scoring, outcome: Outcome, reading: SetReading) -> Result<Transfer> {
        let total = self.declaration().total_points();
        if outcome.declaring.checked_add(outcome.other) != Some(total) {
            return Err(error(format!(
                "an outcome splits the hand's {total} points; {} and {} do not",
                outcome.declaring, outcome.other
            )));
        }
        let declaring = self.declarer().team();
        let mut to = [0; 2];
        match scoring {
            Scoring::Marks { .. } => {
                let (team, marks) = self.award(outcome.declaring);
                to[team.index()] = marks;
            }
            Scoring::Points { .. } if self.makes(outcome.declaring) => {
                to[declaring.index()] = match self.bid() {
                    Bid::Points(_) => outcome.declaring,
                    Bid::Marks(_) => self.bid().value(),
                };
                to[declaring.other().index()] = outcome.other;
            }
            Scoring::Points { .. } => {
                let theirs = match (self.bid(), reading) {
                    (Bid::Marks(_), SetReading::BidAlone) => 0,
                    _ => outcome.other,
                };
                to[declaring.other().index()] = self.bid().value() + theirs;
            }
        }
        Ok(Transfer { to })
    }
}

/// The score of a match under one scoring, by team, and who has won it. A
/// side wins on reaching the target; when both reach it on one hand (under
/// points), the side that bid that hand wins (the book's tie at 250), which
/// is why the winner is kept and not read off the score.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MatchState {
    scoring: Scoring,
    score: [u32; 2],
    winner: Option<Team>,
}

impl MatchState {
    /// The match at `score` (indexed by team). A side at or past the target
    /// has won. Refused when both are: who won is the deciding hand's
    /// bidder, which a score does not record.
    pub fn new(scoring: Scoring, score: [u32; 2]) -> Result<MatchState> {
        let target = scoring.target();
        let winner = match score.map(|s| s >= target) {
            [false, false] => None,
            [true, false] => Some(Team::T0),
            [false, true] => Some(Team::T1),
            [true, true] => {
                return Err(error(format!(
                    "both sides at or past {target}: the winner is the deciding hand's bidder, not the score's"
                )));
            }
        };
        Ok(MatchState { scoring, score, winner })
    }

    pub fn scoring(&self) -> Scoring {
        self.scoring
    }

    pub fn score(&self, team: Team) -> u32 {
        self.score[team.index()]
    }

    /// The side that has won the match, if either has.
    pub fn won(&self) -> Option<Team> {
        self.winner
    }

    /// What `team` still needs to reach the target; zero once it has.
    pub fn short(&self, team: Team) -> u32 {
        self.scoring.target().saturating_sub(self.score(team))
    }

    /// The match after a hand played under `contract` ends in `outcome`,
    /// settled by the book's rule. Refused once the match is decided.
    pub fn apply(&self, contract: &Contract, outcome: Outcome) -> Result<MatchState> {
        self.apply_reading(contract, outcome, SetReading::BidAndPoints)
    }

    /// [`MatchState::apply`] under either reading of a set marks bid.
    pub fn apply_reading(&self, contract: &Contract, outcome: Outcome, reading: SetReading) -> Result<MatchState> {
        if let Some(team) = self.winner {
            return Err(error(format!("the match is over: {team} has won it")));
        }
        let transfer = contract.settle_reading(self.scoring, outcome, reading)?;
        let mut score = self.score;
        for team in [Team::T0, Team::T1] {
            score[team.index()] =
                score[team.index()].checked_add(transfer.to(team)).ok_or_else(|| error("the score overflows"))?;
        }
        let target = self.scoring.target();
        let winner = match score.map(|s| s >= target) {
            [false, false] => None,
            [true, false] => Some(Team::T0),
            [false, true] => Some(Team::T1),
            [true, true] => Some(contract.declarer().team()),
        };
        Ok(MatchState { scoring: self.scoring, score, winner })
    }

    /// Whether this state is at least as good for `team` as `other` under
    /// every match-win value ([`WinValue`]): this one is won by `team`, or
    /// `other` is lost by it, or neither is decided and here `team`'s score
    /// is no lower and the other side's no higher. The last case is the
    /// chain `V(a, b) ≥ V(a', b) ≥ V(a', b')` for `a ≥ a'` and `b ≤ b'`,
    /// through the undecided state `(a', b)`. The order is exact: whenever it
    /// fails, some value has `V(self) < V(other)` (a step in the side's own
    /// score, a step in the other's, or a constant one half). States under
    /// different scorings are never comparable.
    pub fn at_least(&self, other: &MatchState, team: Team) -> bool {
        if self.scoring != other.scoring {
            return false;
        }
        match (self.winner, other.winner) {
            (Some(w), _) if w == team => true,
            (_, Some(w)) if w != team => true,
            (None, None) => {
                self.score(team) >= other.score(team) && self.score(team.other()) <= other.score(team.other())
            }
            _ => false,
        }
    }
}

impl fmt::Display for MatchState {
    /// `6-4` (T0's score, then T1's), and the winner once there is one.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}-{}", self.score[0], self.score[1])?;
        match self.winner {
            Some(team) => write!(f, " (won by {team})"),
            None => Ok(()),
        }
    }
}

/// One hand's law: the contract reached and how it ends, with their weights,
/// which sum to one.
pub type HandLaw = Vec<(Contract, Outcome, Weight)>;

fn check_law(law: &[(Contract, Outcome, Weight)], at: &MatchState) -> Result<()> {
    let total: Weight = law.iter().map(|(_, _, w)| w).sum();
    if total != Weight::one() {
        return Err(error(format!("a hand's law at {at} sums to {total}, not one")));
    }
    Ok(())
}

/// One side's chance of winning the match from each state: one on the states
/// it has won, zero on those it has lost, and on the undecided states a table
/// Silas checks lies between zero and one, never rises with the other side's
/// score and never falls with its own. Those are the properties the
/// dominance order needs ([`MatchState::at_least`]); a table without them is
/// refused, never used.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WinValue {
    scoring: Scoring,
    team: Team,
    /// `own * target + other`, for the undecided states.
    table: Vec<BigRational>,
    authority: Authority,
}

impl WinValue {
    /// `team`'s value from a table over the undecided states, row by row:
    /// entry `own * target + other` is the value at `team`'s score `own` and
    /// the other side's `other`, both below the target. Its numbers come from
    /// a model, so the authority names it: `exact` is refused.
    pub fn new(scoring: Scoring, team: Team, table: Vec<BigRational>, authority: Authority) -> Result<WinValue> {
        if authority.is_exact() {
            return Err(error("a match-win value comes from a model; its authority names it"));
        }
        let t = scoring.target() as usize;
        if table.len() != t * t {
            return Err(error(format!("a match-win value to {t} has {} undecided states, not {}", t * t, table.len())));
        }
        let at = |own: usize, other: usize| &table[own * t + other];
        let (zero, one) = (BigRational::zero(), BigRational::one());
        for own in 0..t {
            for other in 0..t {
                let v = at(own, other);
                if *v < zero || *v > one {
                    return Err(error(format!("not a match-win value: V({own}-{other}) = {v} is not a probability")));
                }
                if other + 1 < t && at(own, other + 1) > v {
                    return Err(error(format!(
                        "not a match-win value: V({own}-{}) = {} exceeds V({own}-{other}) = {v}; it must not rise \
                         with the other side's score",
                        other + 1,
                        at(own, other + 1)
                    )));
                }
                if own + 1 < t && at(own + 1, other) < v {
                    return Err(error(format!(
                        "not a match-win value: V({}-{other}) = {} is below V({own}-{other}) = {v}; it must not fall \
                         with the side's own score",
                        own + 1,
                        at(own + 1, other)
                    )));
                }
            }
        }
        Ok(WinValue { scoring, team, table, authority })
    }

    /// `team`'s value when every hand from every undecided state follows
    /// `hand`, a hand-outcome model named `model` (its identity is the
    /// value's authority, `exact_given(model)`). A dynamic programme over
    /// the score grid: every hand raises the total score, so the states are
    /// solved from the highest total down. Refused when a hand moves no
    /// score, when a law does not sum to one, or when the value fails the
    /// properties ([`WinValue::new`]).
    pub fn solve(
        scoring: Scoring,
        team: Team,
        model: &str,
        hand: impl Fn(&MatchState) -> Result<HandLaw>,
    ) -> Result<WinValue> {
        let authority = Authority::given(model)?;
        let t = scoring.target();
        if t == 0 {
            return Err(error("a match to zero is decided before it starts"));
        }
        let mut table = vec![BigRational::zero(); (t as usize) * (t as usize)];
        for total in (0..=2 * t.saturating_sub(1)).rev() {
            for own in total.saturating_sub(t - 1)..=total.min(t - 1) {
                let other = total - own;
                let mut score = [0; 2];
                score[team.index()] = own;
                score[team.other().index()] = other;
                let state = MatchState::new(scoring, score)?;
                let law = hand(&state)?;
                check_law(&law, &state)?;
                let mut value = BigRational::zero();
                for (contract, outcome, weight) in &law {
                    let next = state.apply(contract, *outcome)?;
                    if next.score[0] + next.score[1] == own + other {
                        return Err(error(format!(
                            "a hand at {state} moves no score: the programme needs every hand to"
                        )));
                    }
                    let v = match next.winner {
                        Some(w) => BigRational::from_integer(BigInt::from(u8::from(w == team))),
                        None => table[(next.score(team) * t + next.score(team.other())) as usize].clone(),
                    };
                    value += weight.rational() * v;
                }
                table[(own * t + other) as usize] = value;
            }
        }
        WinValue::new(scoring, team, table, authority)
    }

    pub fn scoring(&self) -> Scoring {
        self.scoring
    }

    /// The side whose chance it is.
    pub fn team(&self) -> Team {
        self.team
    }

    pub fn authority(&self) -> &Authority {
        &self.authority
    }

    /// The value at `state`: one if the side has won, zero if it has lost,
    /// else the table's. Refused for a state under another scoring.
    pub fn at(&self, state: &MatchState) -> Result<BigRational> {
        if state.scoring != self.scoring {
            return Err(error(format!("a value for {} read at a state of {}", self.scoring, state.scoring)));
        }
        Ok(match state.winner {
            Some(w) => BigRational::from_integer(BigInt::from(u8::from(w == self.team))),
            None => {
                let t = self.scoring.target();
                self.table[(state.score(self.team) * t + state.score(self.team.other())) as usize].clone()
            }
        })
    }
}

/// What a hand is worth to one side.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Utility {
    /// The marks the side gains less the marks the other gains.
    Marks,
    /// The same in points, by the book's rule (S-0017-c).
    Points,
    /// The side's chance of winning the match from the state the hand
    /// leaves. The value's side is the side valued.
    MatchWin(WinValue),
}

impl Utility {
    /// What the hand under `contract` ending in `outcome`, played from
    /// `state`, is worth to `team`. Marks and points read the hand alone.
    pub fn of(&self, team: Team, state: &MatchState, contract: &Contract, outcome: Outcome) -> Result<BigRational> {
        let net = |scoring| -> Result<BigRational> {
            Ok(BigRational::from_integer(BigInt::from(contract.settle(scoring, outcome)?.net(team))))
        };
        match self {
            Utility::Marks => net(Scoring::MARKS),
            Utility::Points => net(Scoring::POINTS),
            Utility::MatchWin(value) => {
                if value.team != team {
                    return Err(error(format!("a match-win value for {} asked for {team}", value.team)));
                }
                value.at(&state.apply(contract, outcome)?)
            }
        }
    }

    /// The expected utility to `team` of a hand whose law is `law`, from a
    /// model named `model`. Its authority is the model's, met with the
    /// value's.
    pub fn expect(
        &self,
        team: Team,
        state: &MatchState,
        model: &str,
        law: &[(Contract, Outcome, Weight)],
    ) -> Result<Estimate> {
        check_law(law, state)?;
        let mut total = BigRational::zero();
        for (contract, outcome, weight) in law {
            total += weight.rational() * self.of(team, state, contract, *outcome)?;
        }
        Ok(Estimate::new(total, Authority::given(model)?.meet(&self.authority())))
    }

    /// What the utility rests on: `exact` for marks and points, the value's
    /// model for the match.
    pub fn authority(&self) -> Authority {
        match self {
            Utility::MatchWin(value) => value.authority.clone(),
            _ => Authority::exact(),
        }
    }
}
