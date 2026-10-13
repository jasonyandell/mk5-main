//! Silas in `scenario_core` (the `python` and `silas` features): queries and
//! policies compiled from text, frames over [`ZebPhysics`], exact beliefs,
//! summaries in `fractions.Fraction`, and the three-valued comparison.
//! `scenario/silas.py` builds a frame from Zeb's own decision.
//!
//! Tiles, tile masks, seats and led suits cross in Zeb's numbering
//! (`crate::physics` maps them); answers come back in Silas's spelling
//! (`"6-5"`, `"S1"`, `"q*"`, `"T0"`, or an int). Every refusal is a
//! `SilasError` carrying Silas's message. No number crosses as a float except
//! the weights handed to `SilasBelief.from_worlds`, each read exactly.

use std::num::NonZeroU64;
use std::sync::Arc;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyFrozenSet, PyTuple};
use pyo3::IntoPyObjectExt;

use silas::{
    Answer, Authority, Belief, Budget, CompiledPolicy, CompiledQuery, ContextSet, DominoSet, Estimate, Frame, Hidden,
    Kernel, Physics, PolicyProgram, Query, QuerySummary, Registry, Value, Verdict, Weight, HIDDEN_SEATS,
};

use crate::physics::{context, domino, dominoes, seat, ZebPhysics};

pyo3::create_exception!(scenario_core, SilasError, PyValueError, "A refusal from Silas, with its message.");

fn refused(e: silas::Error) -> PyErr {
    SilasError::new_err(e.0)
}

fn invalid(message: impl Into<String>) -> PyErr {
    SilasError::new_err(message.into())
}

fn physics(decl: u8) -> PyResult<Arc<dyn Physics>> {
    Ok(Arc::new(ZebPhysics::new(decl).ok_or_else(|| invalid(format!("no declaration {decl}")))?))
}

fn tiles(mask: u32) -> PyResult<DominoSet> {
    dominoes(mask).ok_or_else(|| invalid(format!("{mask:#x} is not a mask of the 28 tiles")))
}

/// An exact rational as a `fractions.Fraction`, through its exact text.
fn fraction<'py>(py: Python<'py>, value: &impl ToString) -> PyResult<Bound<'py, PyAny>> {
    py.import("fractions")?.getattr("Fraction")?.call1((value.to_string(),))
}

/// The labels that name an authority: `exact`, `exact_given(<model>)`,
/// `sampled(<n>, <seed>)`.
fn label(authority: &Authority) -> String {
    match authority {
        Authority::Exact => "exact".into(),
        Authority::ExactGiven { model } => format!("exact_given({model})"),
        Authority::Sampled { n, seed } => format!("sampled({n}, {seed})"),
    }
}

/// Where a number's exactness comes from. Callers make only the two kinds a
/// caller can vouch for, `sampled` and `exact_given`; `exact` belongs to
/// the uniform beliefs and reaches Python only on what they produce.
#[pyclass(frozen, eq, module = "scenario_core")]
#[derive(Clone, PartialEq)]
pub struct SilasAuthority(Authority);

#[pymethods]
impl SilasAuthority {
    /// `n` worlds drawn, not enumerated, from the stream seeded `seed`.
    #[staticmethod]
    fn sampled(n: usize, seed: u64) -> Self {
        SilasAuthority(Authority::Sampled { n, seed })
    }

    /// Exact arithmetic on weights from a pinned model. `model_id` must name
    /// the model completely: its content hash, rung and seed.
    #[staticmethod]
    fn exact_given(model_id: &str) -> PyResult<Self> {
        if model_id.trim().is_empty() {
            return Err(invalid("exact_given names its model: content hash, rung and seed"));
        }
        Ok(SilasAuthority(Authority::ExactGiven { model: model_id.to_owned() }))
    }

    #[getter]
    fn label(&self) -> String {
        label(&self.0)
    }

    fn __str__(&self) -> String {
        label(&self.0)
    }

    fn __repr__(&self) -> String {
        format!("SilasAuthority({})", label(&self.0))
    }
}

/// What one seat of Zeb knows at a decision, under one declaration.
#[pyclass(frozen, module = "scenario_core")]
pub struct SilasFrame(Frame);

#[pymethods]
impl SilasFrame {
    /// `decl` is Zeb's declaration; `viewer` and `leader` are seats;
    /// `own` and `played` tile masks; `trick` the open trick's tiles in play
    /// order; `voids[s]` the led suits seat `s` has shown void in (bit k for
    /// Zeb's led suit k, 7 the called suit); `sizes[s]` the tiles seat `s`
    /// still holds. The pool is every tile neither held by the viewer nor
    /// played.
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        decl: u8,
        viewer: u8,
        own: u32,
        played: u32,
        leader: u8,
        trick: Vec<u8>,
        voids: [u8; 4],
        sizes: [u8; 4],
    ) -> PyResult<Self> {
        let physics = physics(decl)?;
        let at = |s: u8| seat(s).ok_or_else(|| invalid(format!("no seat {s}")));
        let viewer = at(viewer)?;
        let leader = at(leader)?;
        let own = tiles(own)?;
        let played = tiles(played)?;
        let trick = trick
            .iter()
            .map(|&t| domino(t).ok_or_else(|| invalid(format!("no tile {t}"))))
            .collect::<PyResult<Vec<_>>>()?;
        let shown: [ContextSet; 4] = voids.map(|v| (0..8).filter(|k| v >> k & 1 == 1).filter_map(context).collect());
        if usize::from(sizes[viewer.index()]) != own.len() {
            return Err(invalid(format!("the viewer holds {} tiles, not {}", own.len(), sizes[viewer.index()])));
        }
        if shown[viewer.index()].iter().any(|q| !own.is_disjoint(physics.incidence(q))) {
            return Err(invalid("the viewer holds a tile of a suit it has shown void in"));
        }
        let hidden: [Hidden; HIDDEN_SEATS] = std::array::from_fn(|i| {
            let seat = viewer.plus(i + 1);
            Hidden { seat, capacity: usize::from(sizes[seat.index()]), voids: shown[seat.index()] }
        });
        let pool = DominoSet::FULL.difference(own).difference(played);
        let kernel = Kernel::new(viewer, own, pool, hidden).map_err(refused)?;
        Frame::new(physics, kernel, leader, trick, played).map(SilasFrame).map_err(refused)
    }

    /// The physics identity, such as `zeb/P5`.
    #[getter]
    fn physics(&self) -> String {
        self.0.physics().identity()
    }

    /// How many hidden worlds fit the frame.
    fn count_worlds(&self) -> u128 {
        self.0.count_worlds()
    }
}

/// A query compiled against Silas's standard predicates.
#[pyclass(frozen, module = "scenario_core")]
pub struct SilasQuery(CompiledQuery);

#[pymethods]
impl SilasQuery {
    #[new]
    fn new(text: &str) -> PyResult<Self> {
        let query: Query = text.parse().map_err(refused)?;
        query.compile(&Registry::standard()).map(SilasQuery).map_err(refused)
    }

    /// The query's complete text identity, predicate versions included.
    #[getter]
    fn identity(&self) -> String {
        self.0.identity()
    }

    /// The query over a belief, spending at most `budget` units of work.
    fn summarize(&self, belief: &SilasBelief, budget: u64) -> PyResult<SilasSummary> {
        self.0.summarize(&belief.0, &mut Budget::new(budget)).map(SilasSummary).map_err(refused)
    }
}

/// A policy compiled under one of Zeb's declarations.
#[pyclass(frozen, module = "scenario_core")]
pub struct SilasPolicy(CompiledPolicy);

#[pymethods]
impl SilasPolicy {
    #[new]
    fn new(text: &str, decl: u8) -> PyResult<Self> {
        let program: PolicyProgram = text.parse().map_err(refused)?;
        program.compile(&Registry::standard(), &physics(decl)?).map(SilasPolicy).map_err(refused)
    }

    #[getter]
    fn identity(&self) -> String {
        self.0.identity().to_owned()
    }
}

/// An exact finite measure over the worlds of a frame.
#[pyclass(frozen, module = "scenario_core")]
pub struct SilasBelief(Belief);

#[pymethods]
impl SilasBelief {
    /// Every world of the frame, weight one: `exact`. Refused when the frame
    /// has more than `cap` worlds.
    #[staticmethod]
    fn uniform(frame: &SilasFrame, cap: usize) -> PyResult<Self> {
        Belief::uniform(frame.0.clone(), cap).map(SilasBelief).map_err(refused)
    }

    /// The uniform measure resolved only on the pool tiles of the mask
    /// `resolving`, one point per placement, each weighted by its exact number
    /// of completions: `exact`. A query reading any other hidden tile is
    /// refused, never approximated.
    #[staticmethod]
    fn uniform_resolving(frame: &SilasFrame, resolving: u32, cap: usize) -> PyResult<Self> {
        Belief::uniform_resolving(frame.0.clone(), tiles(resolving)?, cap).map(SilasBelief).map_err(refused)
    }

    /// Worlds (four remaining-hand masks, absolute seats) with float
    /// weights, each read as the exact rational it denotes. The caller says
    /// where the weights came from: `SilasAuthority.sampled(n, seed)` or
    /// `SilasAuthority.exact_given(model_id)`; `exact` is refused.
    /// Repeated worlds add their weights; zero weights drop out.
    #[staticmethod]
    fn from_worlds(
        frame: &SilasFrame,
        worlds: Vec<[u32; 4]>,
        weights: Vec<f64>,
        authority: &SilasAuthority,
    ) -> PyResult<Self> {
        if worlds.len() != weights.len() {
            return Err(invalid(format!("{} worlds but {} weights", worlds.len(), weights.len())));
        }
        let kernel = frame.0.kernel();
        let viewer = kernel.viewer();
        let mut entries = Vec::with_capacity(worlds.len());
        for (hands, weight) in worlds.iter().zip(weights) {
            if hands[viewer.index()] != kernel.viewer_hand().bits() {
                return Err(invalid("a world gives the viewer a hand other than its own"));
            }
            let hidden = kernel.hidden().map(|h| tiles(hands[h.seat.index()]));
            let [a, b, c] = hidden;
            let world = kernel.world([a?, b?, c?]);
            entries.push((world, Weight::from_f64_exact(weight).map_err(refused)?));
        }
        let id = format!("worlds {}", label(&authority.0));
        Belief::from_weights(frame.0.clone(), id, authority.0.clone(), entries, worlds.len())
            .map(SilasBelief)
            .map_err(refused)
    }

    #[getter]
    fn identity(&self) -> String {
        self.0.id().to_owned()
    }

    #[getter]
    fn authority(&self) -> SilasAuthority {
        SilasAuthority(self.0.authority().clone())
    }

    /// The belief's total mass, exactly.
    #[getter]
    fn total_mass<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        fraction(py, self.0.total_mass())
    }

    fn __len__(&self) -> usize {
        self.0.len()
    }
}

/// A number read off a belief: an exact `Fraction` and its authority.
#[pyclass(frozen, module = "scenario_core")]
pub struct SilasEstimate(Estimate);

#[pymethods]
impl SilasEstimate {
    #[getter]
    fn value<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        fraction(py, self.0.value())
    }

    #[getter]
    fn authority(&self) -> SilasAuthority {
        SilasAuthority(self.0.authority().clone())
    }
}

/// What a query says about a belief. Probabilities are exact `Fraction`s.
#[pyclass(frozen, module = "scenario_core")]
pub struct SilasSummary(QuerySummary);

fn answer<'py>(py: Python<'py>, a: &Answer) -> PyResult<Bound<'py, PyTuple>> {
    let values = a.0.iter().map(|v| match v {
        Value::Number(n) => n.into_bound_py_any(py),
        other => other.to_string().into_bound_py_any(py),
    });
    PyTuple::new(py, values.collect::<PyResult<Vec<_>>>()?)
}

#[pymethods]
impl SilasSummary {
    #[getter]
    fn authority(&self) -> SilasAuthority {
        SilasAuthority(self.0.authority.clone())
    }

    #[getter]
    fn query_identity(&self) -> &str {
        &self.0.query_identity
    }

    #[getter]
    fn belief_identity(&self) -> &str {
        &self.0.belief_identity
    }

    /// The probability that the query has an answer.
    #[getter]
    fn event_probability(&self) -> SilasEstimate {
        SilasEstimate(self.0.event_probability())
    }

    /// Each answer tuple's presence probability: the mass of the worlds where
    /// it is an answer, over the total. They need not sum to one.
    #[getter]
    fn presence<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let out = PyDict::new(py);
        for (a, mass) in &self.0.answer_presence_mass {
            out.set_item(answer(py, a)?, fraction(py, &(mass.rational() / self.0.total_mass.rational()))?)?;
        }
        Ok(out)
    }

    /// The law of the whole answer set: each frozenset of answer tuples, and
    /// the probability that it is exactly the query's answer.
    #[getter]
    fn answer_sets<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let out = PyDict::new(py);
        for (answers, mass) in &self.0.answer_set_mass {
            let set = PyFrozenSet::new(py, answers.iter().map(|a| answer(py, a)).collect::<PyResult<Vec<_>>>()?)?;
            out.set_item(set, fraction(py, &(mass.rational() / self.0.total_mass.rational()))?)?;
        }
        Ok(out)
    }
}

/// Compare `a` against `b` with an explicit nonnegative `band` (a
/// `Fraction`, zero when both are exact): "better" when `a` exceeds `b` by
/// more than the band, "worse" when it falls short by more, else
/// "indistinguishable".
#[pyfunction]
fn silas_compare(a: &SilasEstimate, b: &SilasEstimate, band: &Bound<'_, PyAny>) -> PyResult<&'static str> {
    let part = |name: &str| band.getattr(name)?.extract::<u64>();
    let (numerator, denominator) = (part("numerator"), part("denominator").map(NonZeroU64::new));
    let band = match (numerator, denominator) {
        (Ok(n), Ok(Some(d))) => Weight::ratio(n, d),
        _ => return Err(invalid("a band is a nonnegative Fraction below 2**64")),
    };
    Ok(match silas::compare(&a.0, &b.0, band.rational()).map_err(refused)? {
        Verdict::Better => "better",
        Verdict::Worse => "worse",
        Verdict::Indistinguishable => "indistinguishable",
    })
}

/// Add the Silas classes to `scenario_core`.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("SilasError", m.py().get_type::<SilasError>())?;
    m.add_class::<SilasAuthority>()?;
    m.add_class::<SilasFrame>()?;
    m.add_class::<SilasQuery>()?;
    m.add_class::<SilasPolicy>()?;
    m.add_class::<SilasBelief>()?;
    m.add_class::<SilasEstimate>()?;
    m.add_class::<SilasSummary>()?;
    m.add_function(wrap_pyfunction!(silas_compare, m)?)?;
    Ok(())
}
