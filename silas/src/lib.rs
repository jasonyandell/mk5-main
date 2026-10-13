//! Silas: a typed relational language over the worlds consistent with what a
//! seat of Texas 42 knows.
//!
//! A [`Query`] is a relation: the set of output tuples for which some
//! [`Pattern`] has a satisfying assignment of its roles. A [`Belief`] is an
//! exact measure over the hidden [`World`]s of a [`Frame`]; a query summarized
//! over a belief yields exact probabilities, each carrying its [`Authority`].
//! A [`PolicyProgram`] chooses plays from what the viewer alone can see.
//!
//! Silas owns the nouns of the game and the language. The rules live behind
//! the [`Physics`] trait; [`reference::Straight42`] is the reference physics.
//! See `README.md`.

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
mod syntax;

pub use belief::{Belief, Certainty, Comparison, Difference, QuerySummary, Selection, SelectionLaw};
pub use dynamics::{
    AnswerDynamics, BeliefStep, ObservedPlay, PlayClass, anchor_back, compare_answers, play_is_legal, step_belief,
    step_belief_given, step_frame, step_world, transport_answers,
};
pub use eval::{Answer, Answers, Budget, CompiledQuery};
pub use frame::{Frame, HIDDEN_SEATS, Hidden, Holdings, Kernel, MAX_CAPACITY, World};
pub use measure::{Authority, Estimate, Verdict, Weight, compare};
pub use nouns::{Context, ContextSet, Domino, DominoSet, Pip, Seat, Team};
pub use physics::Physics;
pub use policy::{
    CompiledPolicy, DecisionProvenance, ExactRule, Fallback, PolicyController, PolicyControllerState,
    PolicyDecisionTrace, PolicyInput, PolicyKey, PolicyProgram, PolicyRule, RigidBinding, Selector,
};
pub use registry::{Access, Predicate, PredicateContext, PredicateSpec, Registry};
pub use syntax::{Atom, Pattern, Query, Role, Sort, Term, Value};

/// An invalid program, an invalid input, or a refused computation. Nothing
/// partial is returned alongside it.
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
