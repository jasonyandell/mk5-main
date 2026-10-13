//! Silas: a typed relational language over the worlds consistent with what a
//! seat of Texas 42 knows.
//!
//! A [`Query`] is a relation: the set of output tuples for which some
//! [`Pattern`] has a satisfying assignment of its roles. A [`Belief`] is an
//! exact measure over the hidden [`World`]s of a [`Frame`]; a query summarized
//! over a belief yields exact probabilities, each carrying its [`Authority`].
//! A [`PolicyProgram`] chooses plays from what the viewer alone can see. A
//! [`Solver`] is the perfect-information reference: it sees all four hands.
//! [`forces`] and [`lay_down`] are what the viewer can guarantee without it.
//! An [`Auction`] settles the hand's [`Contract`]; a policy's bidding and
//! declaring rules read an [`AuctionFrame`]. A [`Contract`] settles a hand
//! under a [`Scoring`] and moves a
//! [`MatchState`]; a [`Dominance`] compares two contracts over every outcome
//! of the hand, exactly.
//!
//! Silas owns the nouns of the game, the language, and every derived rule. A
//! rules engine supplies only one declaration's tables, through
//! [`Declaration`]; Silas reads them into a [`Physics`].
//! [`reference::Straight42`] is the reference declarations. See `README.md`.

mod auction;
mod belief;
mod coverage;
mod dominance;
mod dynamics;
mod eval;
mod frame;
mod lint;
mod measure;
mod nouns;
mod outcome;
mod physics;
mod policy;
mod reading;
pub mod reference;
mod registry;
mod scoring;
mod solve;
mod sugar;
mod syntax;

pub use auction::{Auction, AuctionConfig, AuctionFrame, Call, MIN_BID};
pub use belief::{Belief, Certainty, Comparison, Difference, QuerySummary, Selection, SelectionLaw};
pub use coverage::{Coverage, Funnel, RuleCoverage, why_not};
pub use dominance::{Counterexample, Criterion, Decision, Dominance, Place};
pub use dynamics::{
    AnswerDynamics, BeliefStep, ObservedPlay, PlayClass, anchor_back, compare_answers, play_is_legal, step_belief,
    step_belief_given, step_frame, step_world, transport_answers,
};
pub use eval::{Answer, Answers, Budget, CompiledQuery, Witness};
pub use frame::{Bid, Contract, Frame, HIDDEN_SEATS, Hidden, Holdings, Kernel, MAX_CAPACITY, World};
pub use lint::Warning;
pub use measure::{Authority, Estimate, Sample, Verdict, Weight, compare};
pub use nouns::{Context, ContextSet, Domino, DominoSet, Pip, Seat, Team};
pub use outcome::{Goal, OUTCOME_SCAN_CAP, forces, lay_down};
pub use physics::{Declaration, Physics};
pub use policy::{
    Allowed, Choice, CompiledPolicy, DecisionProvenance, Declared, ExactRule, Fallback, GuardWitnesses,
    PolicyDecisionTrace, PolicyKey, PolicyProgram, PolicyRule, Program, Selector, Trump,
};
pub use reading::{AuctionEvidence, AuctionReading, SeatEvidence};
pub use registry::{Access, Phase, Predicate, PredicateContext, PredicateSpec, Registry};
pub use scoring::{HandLaw, MatchState, Outcome, Scoring, SetReading, Transfer, Utility, WinValue};
pub use solve::{Deal, Grade, Judged, Objective, OpenPosition, RootValues, Solver, grade_line};
pub use syntax::{
    Aggregate, AtMost, Atom, Pattern, Quantifier, Quantity, Query, Role, Scope, Sort, Span, Term, Value, format,
};

/// An invalid program, an invalid input, or a refused computation. Nothing
/// partial is returned alongside it. An error in program text names the
/// source, line and column, and shows the line with a caret under the fault.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Error(pub String);

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;

pub(crate) fn error(message: impl Into<String>) -> Error {
    Error(message.into())
}
