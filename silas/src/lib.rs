//! Silas: a typed relational language over the worlds consistent with what a
//! seat of Texas 42 knows.
//!
//! A [`Query`] is a relation: the set of output tuples for which some
//! [`Pattern`] has a satisfying assignment of its roles. A [`Belief`] is an
//! exact measure over the hidden [`World`]s of a [`Frame`]; a query summarized
//! over a belief yields exact probabilities, each carrying its [`Authority`].
//! A [`PolicyProgram`] chooses plays from what the viewer alone can see. A
//! [`Solver`] is the perfect-information reference: it sees all four hands.
//!
//! Silas owns the nouns of the game, the language, and every derived rule. A
//! rules engine supplies only one declaration's tables, through
//! [`Declaration`]; Silas reads them into a [`Physics`].
//! [`reference::Straight42`] is the reference declarations. See `README.md`.

mod belief;
mod dynamics;
mod eval;
mod frame;
mod measure;
mod nouns;
mod physics;
mod policy;
pub mod reference;
mod registry;
mod solve;
mod syntax;

pub use belief::{Belief, Certainty, Comparison, Difference, QuerySummary, Selection, SelectionLaw};
pub use dynamics::{
    AnswerDynamics, BeliefStep, ObservedPlay, PlayClass, anchor_back, compare_answers, play_is_legal, step_belief,
    step_belief_given, step_frame, step_world, transport_answers,
};
pub use eval::{Answer, Answers, Budget, CompiledQuery, Witness};
pub use frame::{Bid, Contract, Frame, HIDDEN_SEATS, Hidden, Holdings, Kernel, MAX_CAPACITY, World};
pub use measure::{Authority, Estimate, Sample, Verdict, Weight, compare};
pub use nouns::{Context, ContextSet, Domino, DominoSet, Pip, Seat, Team};
pub use physics::{Declaration, Physics};
pub use policy::{
    Allowed, CompiledPolicy, DecisionProvenance, ExactRule, Fallback, GuardWitnesses, PolicyDecisionTrace, PolicyKey,
    PolicyProgram, PolicyRule, Program, Selector,
};
pub use registry::{Access, Predicate, PredicateContext, PredicateSpec, Registry};
pub use solve::{Deal, Grade, Judged, Objective, OpenPosition, RootValues, Solver, grade_line};
pub use syntax::{Atom, Pattern, Quantifier, Query, Role, Scope, Sort, Span, Term, Value, format};

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
