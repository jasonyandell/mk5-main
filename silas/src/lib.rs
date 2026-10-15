// SPDX-FileCopyrightText: Jason Yandell
// SPDX-License-Identifier: CC0-1.0

//! Silas: a typed relational language over the worlds consistent with what a
//! seat of Texas 42 knows.
//!
//! A [`Query`] is a relation: the set of output tuples for which some
//! [`Pattern`] has a satisfying assignment of its roles. A [`Belief`] is an
//! exact measure over the hidden [`World`]s of a [`Frame`]; a query summarized
//! over a belief yields exact probabilities, each carrying its [`Authority`].
//! A [`PolicyProgram`] chooses plays from what the viewer alone can see.
//! [`forces`] and [`lay_down`] are what the viewer can guarantee: exact
//! information-state solves, one choice per frame, never one per world.
//! An [`Auction`] settles the hand's [`Contract`]; a policy's bidding and
//! declaring rules read an [`AuctionFrame`]. A [`Contract`] settles a hand
//! under a [`Scoring`] and moves a
//! [`MatchState`]; a [`Dominance`] compares two contracts over every outcome
//! of the hand, exactly. A [`Field`] of four [`Player`]s plays a hand out
//! ([`play_out`]); a [`Trial`] compares two players in its chair over deals,
//! paired, given the field.
//!
//! Silas owns the nouns of the game, the language, and every derived rule. A
//! rules engine supplies only one declaration's tables, through
//! [`Declaration`]; Silas reads them into a [`Physics`].
//! [`reference::Straight42`] is the reference declarations. See `README.md`.
//!
//! STRATEGY-FUSION FENCE: the clairvoyant (double-dummy) reference,
//! `ClairvoyantSolver`, `clairvoyant_grade_line` and the World atom
//! `(clairvoyant-dd-make)`, sees all four hands, so its value used as a value
//! is strategy fusion. It is behind the non-default `clairvoyant` feature and a
//! policy cannot read it. Allowed instead: information-state solving
//! ([`forces`], [`lay_down`], Zeb's plan). The ruling:
//! `wiki/topics/strategy-fusion-fence.md`.

mod auction;
mod belief;
// STRATEGY-FUSION FENCE: compiled always, because `forces` refutes with it (a
// world double-dummy play fails refutes every strategy: an upper bound, never
// a value); exported only under the `clairvoyant` feature.
#[cfg_attr(not(feature = "clairvoyant"), allow(dead_code))]
mod clairvoyant;
mod coverage;
mod dominance;
mod dynamics;
mod eval;
mod field;
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
pub use eval::{Answer, Answers, Budget, CompiledNumber, CompiledQuery, Witness};
pub use field::{Deals, Field, Paired, Player, Seats, Settlement, Table, Trial, play_out};
pub use frame::{Bid, Contract, Frame, HIDDEN_SEATS, Hidden, Holdings, Kernel, MAX_CAPACITY, World};
pub use lint::Warning;
pub use measure::{Authority, Estimate, Sample, Verdict, Weight, compare};
pub use nouns::{Context, ContextSet, Domino, DominoSet, Pip, Seat, Team};
pub use outcome::{Goal, OUTCOME_SCAN_CAP, forces, lay_down};
pub use physics::{Declaration, Physics};
pub use policy::{
    Allowed, Choice, CompiledPolicy, DecisionProvenance, Declared, ExactRule, Fallback, GuardWitnesses,
    PolicyDecisionTrace, PolicyKey, PolicyProgram, PolicyRule, Program, ReadingDecl, Selector, Trump,
};
pub use reading::{
    CallEvidence, Evidence, Likelihood, PlayEvidence, READING_CAP, ReadChair, Reading, ReadingPredicate, SeatEvidence,
};
pub use registry::{Access, NumberAt, Phase, Predicate, PredicateContext, PredicateSpec, Registry};
pub use scoring::{HandLaw, MatchState, Outcome, Scoring, SetReading, Transfer, Utility, WinValue};
// STRATEGY-FUSION FENCE: the clairvoyant reference, for tests and showcases only.
#[cfg(feature = "clairvoyant")]
pub use clairvoyant::{
    ClairvoyantSolver, Deal, Grade, Judged, Objective, OpenPosition, RootValues, clairvoyant_grade_line,
};
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
