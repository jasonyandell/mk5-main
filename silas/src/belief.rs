// SPDX-FileCopyrightText: Jason Yandell
// SPDX-License-Identifier: CC0-1.0

//! Beliefs: exact finite measures over the worlds consistent with a frame,
//! and what a query says about them.

use std::cell::Cell;
use std::collections::BTreeMap;
use std::num::NonZeroU64;

use crate::eval::{Answer, Answers, Budget, CompiledQuery};
use crate::frame::{Frame, HIDDEN_SEATS, Hidden, Holdings, Kernel, World};
use crate::measure::{Authority, Estimate, Weight};
use crate::nouns::{Domino, DominoSet, Seat};
use crate::reading::Evidence;
use crate::{Result, error};

/// A finite measure over worlds of one frame. Each point carries an exact
/// weight; probabilities divide by the total mass, which is always positive.
///
/// A belief may leave some pool tiles *unresolved*: a point then stands for
/// every world that agrees with it on the resolved tiles, weighted by how many
/// there are, and queries may read only resolved tiles (see [`Holdings`]).
#[derive(Clone, Debug)]
pub struct Belief {
    frame: Frame,
    id: String,
    authority: Authority,
    unresolved: DominoSet,
    points: BTreeMap<[u32; Seat::COUNT], (World, Weight)>,
    total: Weight,
}

impl Belief {
    /// A measure supplied by the caller. Its weights come from somewhere other
    /// than the support itself, so its authority must say where: `Exact` is
    /// refused. Repeated worlds add their weights; zero weights drop out.
    pub fn from_weights(
        frame: Frame,
        id: impl Into<String>,
        authority: Authority,
        entries: impl IntoIterator<Item = (World, Weight)>,
        max_worlds: usize,
    ) -> Result<Self> {
        if authority.is_exact() {
            return Err(error("caller-supplied weights name a model or a sample; Exact is the support's own"));
        }
        Self::assemble(frame, id.into(), authority, DominoSet::EMPTY, entries, max_worlds)
    }

    fn assemble(
        frame: Frame,
        id: String,
        authority: Authority,
        unresolved: DominoSet,
        entries: impl IntoIterator<Item = (World, Weight)>,
        max_worlds: usize,
    ) -> Result<Self> {
        if id.trim().is_empty() {
            return Err(error("a belief requires a declared identity"));
        }
        let mut points: BTreeMap<_, (World, Weight)> = BTreeMap::new();
        for (world, weight) in entries {
            if !frame.contains(&world) {
                return Err(error("a belief world is outside the frame's support"));
            }
            if weight.is_zero() {
                continue;
            }
            let key = Seat::ALL.map(|s| world.hand(s).difference(unresolved).bits());
            if !points.contains_key(&key) && points.len() >= max_worlds {
                return Err(error(format!("the belief exceeds its cap of {max_worlds} points")));
            }
            points.entry(key).or_insert_with(|| (world, Weight::zero())).1 += &weight;
        }
        let total: Weight = points.values().map(|(_, w)| w).sum();
        if total.is_zero() {
            return Err(error("a zero-mass belief (the conditioning event may be impossible)"));
        }
        Ok(Self { frame, id, authority, unresolved, points, total })
    }

    /// The uniform measure on every world consistent with the frame. The
    /// support is counted before it is enumerated, and refused above the cap.
    pub fn uniform(frame: Frame, max_worlds: usize) -> Result<Self> {
        let count = frame.count_worlds();
        if count > max_worlds as u128 {
            return Err(error(format!("the support has {count} worlds, above the cap of {max_worlds}")));
        }
        let entries: Vec<_> = frame.worlds().map(|w| (w, Weight::one())).collect();
        Self::assemble(frame, "uniform-full-support".into(), Authority::exact(), DominoSet::EMPTY, entries, max_worlds)
    }

    /// The uniform measure on every world consistent with the frame, resolved
    /// only on the pool tiles in `resolved`: one point per placement of those
    /// tiles, weighted by the exact number of worlds that complete it. Queries
    /// that read only resolved tiles get the uniform measure's exact answers
    /// without enumerating the support.
    pub fn uniform_resolving(frame: Frame, resolved: DominoSet, max_points: usize) -> Result<Self> {
        let kernel = frame.kernel();
        let physics = frame.physics();
        let tiles: Vec<Domino> = resolved.intersection(kernel.pool()).iter().collect();
        let unresolved = kernel.pool().difference(resolved);
        let allowed: [DominoSet; HIDDEN_SEATS] = std::array::from_fn(|slot| kernel.allowed(physics, slot));
        let mut entries = Vec::new();
        let mut point = |placed: &[DominoSet; HIDDEN_SEATS]| {
            let rest = Kernel::new(
                kernel.viewer(),
                kernel.viewer_hand(),
                unresolved,
                std::array::from_fn(|slot| Hidden {
                    capacity: kernel.hidden()[slot].capacity - placed[slot].len(),
                    ..kernel.hidden()[slot]
                }),
            )?;
            if let Some(first) = physics.worlds(&rest).next() {
                if entries.len() == max_points {
                    return Err(error(format!("the belief exceeds its cap of {max_points} points")));
                }
                let hands = std::array::from_fn(|slot| placed[slot].union(first.hand(kernel.hidden()[slot].seat)));
                entries.push((kernel.world(hands), Weight::integer(physics.count_worlds(&rest))));
            }
            Ok(())
        };
        place(
            &tiles,
            &allowed,
            kernel.hidden().map(|h| h.capacity),
            &mut [DominoSet::EMPTY; HIDDEN_SEATS],
            &mut point,
        )?;
        let id = format!("uniform-resolving {:?}", DominoSet::from_iter(tiles));
        Self::assemble(frame, id, Authority::exact(), unresolved, entries, max_points)
    }

    /// The uniform measure resolved on exactly the pool tiles `query` reads:
    /// starting from none, every tile a refused read needed is resolved, until
    /// no read is refused. The query's summary over the result is the
    /// uniform measure's, exactly. Refused above `max_points` points.
    pub fn uniform_for(frame: Frame, query: &CompiledQuery, max_points: usize, budget: &mut Budget) -> Result<Self> {
        let mut resolved = DominoSet::EMPTY;
        loop {
            let belief = Belief::uniform_resolving(frame.clone(), resolved, max_points)?;
            let needed = Cell::new(DominoSet::EMPTY);
            for (world, _) in belief.worlds() {
                // A refused read records what it needed; any other refusal is
                // the query's own.
                if let Err(e) = query.evaluate_in(belief.holdings(world).probing(&needed), budget)
                    && needed.get().is_empty()
                {
                    return Err(e);
                }
            }
            if needed.get().is_empty() {
                return Ok(belief);
            }
            resolved = resolved.union(needed.get());
        }
    }

    /// The uniform measure on the frame's worlds conditioned on the record as
    /// `evidence` reads it: each world weighs the product of its read seats'
    /// weights (one or zero for the calls and for plays read as consistent; a
    /// product of `1/|allowed|` for plays read as uniform), resolved on the
    /// pool tiles of `resolved` as [`Belief::uniform_resolving`] is. Its
    /// authority is `exact_given` the read policies. Refused when no world
    /// weighs anything (the record is impossible under those policies) and
    /// above `max_points`.
    pub fn given(evidence: &Evidence, resolved: DominoSet, max_points: usize) -> Result<Self> {
        let frame = evidence.frame().clone();
        let tiles = resolved.intersection(frame.kernel().pool());
        let unresolved = frame.kernel().pool().difference(tiles);
        let entries = evidence.count(tiles)?;
        let id = format!("given [{}] resolving {tiles:?}", evidence.authority());
        Self::assemble(frame, id, evidence.authority().clone(), unresolved, entries, max_points)
    }

    /// [`Belief::given`] resolved on the tiles `query` reads: those
    /// [`Belief::uniform_for`] finds. A point sums the weights of the worlds
    /// it stands for, and the query is constant on them, so the summary is
    /// the fully resolved belief's.
    pub fn given_for(
        evidence: &Evidence,
        query: &CompiledQuery,
        max_points: usize,
        budget: &mut Budget,
    ) -> Result<Self> {
        let uniform = Belief::uniform_for(evidence.frame().clone(), query, max_points, budget)?;
        let resolved = evidence.frame().kernel().pool().difference(uniform.unresolved());
        Self::given(evidence, resolved, max_points)
    }

    pub fn frame(&self) -> &Frame {
        &self.frame
    }

    pub fn id(&self) -> &str {
        &self.id
    }

    pub fn authority(&self) -> &Authority {
        &self.authority
    }

    /// Pool tiles this belief does not resolve; empty unless built by
    /// [`Belief::uniform_resolving`].
    pub fn unresolved(&self) -> DominoSet {
        self.unresolved
    }

    /// The positive-weight points. Under an unresolved tile set each world is
    /// one representative of the worlds its point stands for.
    pub fn worlds(&self) -> impl Iterator<Item = (&World, &Weight)> {
        self.points.values().map(|(world, weight)| (world, weight))
    }

    pub fn len(&self) -> usize {
        self.points.len()
    }

    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    pub fn total_mass(&self) -> &Weight {
        &self.total
    }

    pub(crate) fn holdings<'a>(&'a self, world: &'a World) -> Holdings<'a> {
        Holdings::new(&self.frame, world, self.unresolved)
    }

    pub(crate) fn rebuild(
        &self,
        frame: Frame,
        id: String,
        authority: Authority,
        entries: Vec<(World, Weight)>,
        max_worlds: usize,
    ) -> Result<Self> {
        Self::assemble(frame, id, authority, self.unresolved, entries, max_worlds)
    }

    /// Condition on caller-declared evidence: each world's weight times its
    /// likelihood in `[0, 1]`, a number from the pinned `model`. Returns the
    /// posterior and the evidence's prior probability. Nothing is built on
    /// failure, and a zero-mass posterior is refused.
    pub fn condition_likelihood(
        &self,
        posterior_id: impl Into<String>,
        model: &str,
        mut likelihood: impl FnMut(Holdings<'_>) -> Result<Weight>,
    ) -> Result<(Self, Estimate)> {
        let authority = self.authority.meet(&Authority::given(model)?);
        let mut entries = Vec::with_capacity(self.len());
        for (world, weight) in self.worlds() {
            entries.push((*world, weight * &probability(likelihood(self.holdings(world))?)?));
        }
        self.posterior(posterior_id.into(), authority, entries)
    }

    /// Condition on the EXISTENCE of an answer, not on how many there are.
    pub fn condition(
        &self,
        query: &CompiledQuery,
        posterior_id: impl Into<String>,
        budget: &mut Budget,
    ) -> Result<(Self, Estimate)> {
        let mut entries = Vec::with_capacity(self.len());
        for (world, weight) in self.worlds() {
            if !query.evaluate_in(self.holdings(world), budget)?.is_empty() {
                entries.push((*world, weight.clone()));
            }
        }
        self.posterior(posterior_id.into(), self.authority.clone(), entries)
    }

    fn posterior(&self, id: String, authority: Authority, entries: Vec<(World, Weight)>) -> Result<(Self, Estimate)> {
        let posterior = self.rebuild(self.frame.clone(), id, authority, entries, self.len())?;
        let evidence =
            Estimate::new(posterior.total.share_of(&self.total).rational().clone(), posterior.authority.clone());
        Ok((posterior, evidence))
    }
}

/// Every placement of `tiles` into the hidden slots that respects what each
/// slot may hold and its remaining capacity.
fn place(
    tiles: &[Domino],
    allowed: &[DominoSet; HIDDEN_SEATS],
    room: [usize; HIDDEN_SEATS],
    placed: &mut [DominoSet; HIDDEN_SEATS],
    visit: &mut dyn FnMut(&[DominoSet; HIDDEN_SEATS]) -> Result<()>,
) -> Result<()> {
    let Some((tile, rest)) = tiles.split_first() else {
        return visit(placed);
    };
    for slot in 0..HIDDEN_SEATS {
        if allowed[slot].contains(*tile) && placed[slot].len() < room[slot] {
            placed[slot].insert(*tile);
            place(rest, allowed, room, placed, visit)?;
            placed[slot].remove(*tile);
        }
    }
    Ok(())
}

/// A likelihood is a probability: at most one.
pub(crate) fn probability(w: Weight) -> Result<Weight> {
    if w > Weight::one() {
        return Err(error(format!("a likelihood of {w} exceeds one")));
    }
    Ok(w)
}

/// What a query says about a belief. Every statement concerns the
/// positive-weight points of the NAMED measure only.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Certainty {
    pub event_possible: bool,
    pub event_certain: bool,
    pub constant_multiplicity: Option<usize>,
    pub constant_answers: Option<Answers>,
    pub world_functional: bool,
    pub identity: Option<Answer>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QuerySummary {
    pub frame: Frame,
    pub query_identity: String,
    pub belief_identity: String,
    pub authority: Authority,
    pub positive_worlds: usize,
    pub total_mass: Weight,
    pub event_mass: Weight,
    /// Each answer gets a world's weight ONCE when present. These masses may
    /// sum above the total: they are presence events, not a law over answers.
    pub answer_presence_mass: BTreeMap<Answer, Weight>,
    /// The law of the whole answer SET; hidden witnesses disappear.
    pub answer_set_mass: BTreeMap<Answers, Weight>,
    pub certainty: Certainty,
}

impl CompiledQuery {
    pub fn summarize(&self, belief: &Belief, budget: &mut Budget) -> Result<QuerySummary> {
        let mut event_mass = Weight::zero();
        let mut presence: BTreeMap<Answer, Weight> = BTreeMap::new();
        let mut sets: BTreeMap<Answers, Weight> = BTreeMap::new();
        for (world, weight) in belief.worlds() {
            let answers = self.evaluate_in(belief.holdings(world), budget)?;
            if !answers.is_empty() {
                event_mass += weight;
            }
            for answer in &answers {
                *presence.entry(answer.clone()).or_default() += weight;
            }
            *sets.entry(answers).or_default() += weight;
        }
        let first = sets.keys().next().expect("a belief has positive mass");
        let constant_answers = (sets.len() == 1).then(|| first.clone());
        let constant_multiplicity = sets.keys().all(|s| s.len() == first.len()).then_some(first.len());
        let world_functional = constant_multiplicity == Some(1);
        let identity = if world_functional && sets.len() == 1 { first.first().cloned() } else { None };
        let certainty = Certainty {
            event_possible: !event_mass.is_zero(),
            event_certain: event_mass == belief.total,
            constant_multiplicity,
            constant_answers,
            world_functional,
            identity,
        };
        Ok(QuerySummary {
            frame: belief.frame.clone(),
            query_identity: self.identity(),
            belief_identity: belief.id.clone(),
            authority: belief.authority.clone(),
            positive_worlds: belief.len(),
            total_mass: belief.total.clone(),
            event_mass,
            answer_presence_mass: presence,
            answer_set_mass: sets,
            certainty,
        })
    }

    /// The first point of the belief where the two queries differ, if any.
    /// Under unresolved tiles, every world the point stands for differs.
    pub fn compare(
        &self,
        other: &Self,
        belief: &Belief,
        comparison: Comparison,
        budget: &mut Budget,
    ) -> Result<Option<Difference>> {
        if comparison == Comparison::Answers && self.outputs() != other.outputs() {
            return Err(error("comparing answers requires the same ordered outputs"));
        }
        for (world, _) in belief.worlds() {
            let left = self.evaluate_in(belief.holdings(world), budget)?;
            let right = other.evaluate_in(belief.holdings(world), budget)?;
            let differs = match comparison {
                Comparison::Existence => left.is_empty() != right.is_empty(),
                Comparison::Answers => left != right,
            };
            if differs {
                return Ok(Some(Difference { world: *world, left, right }));
            }
        }
        Ok(None)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Comparison {
    Existence,
    Answers,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Difference {
    pub world: World,
    pub left: Answers,
    pub right: Answers,
}

/// An explicit convention for turning answer sets into one answer per world.
/// Neither is implied by the language.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Selection {
    LexicographicFirst,
    UniformWithinWorld,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SelectionLaw {
    pub selection: Selection,
    pub authority: Authority,
    /// Unconditional: these sum to the event probability, not always to one.
    pub probabilities: BTreeMap<Answer, Weight>,
    pub no_answer_probability: Weight,
}

impl QuerySummary {
    pub fn event_probability(&self) -> Estimate {
        self.estimate(&self.event_mass)
    }

    pub fn presence_probability(&self, answer: &Answer) -> Estimate {
        self.estimate(self.answer_presence_mass.get(answer).unwrap_or(&Weight::zero()))
    }

    fn estimate(&self, mass: &Weight) -> Estimate {
        Estimate::new(mass.share_of(&self.total_mass).rational().clone(), self.authority.clone())
    }

    pub fn select(&self, selection: Selection) -> SelectionLaw {
        let mut probabilities: BTreeMap<Answer, Weight> = BTreeMap::new();
        let mut no_answer_probability = Weight::zero();
        for (answers, mass) in &self.answer_set_mass {
            let p = mass.share_of(&self.total_mass);
            match (answers.first(), selection) {
                (None, _) => no_answer_probability += &p,
                (Some(first), Selection::LexicographicFirst) => *probabilities.entry(first.clone()).or_default() += &p,
                (Some(_), Selection::UniformWithinWorld) => {
                    let n = NonZeroU64::new(answers.len() as u64).expect("a nonempty answer set");
                    let share = &p * &Weight::ratio(1, n);
                    for answer in answers {
                        *probabilities.entry(answer.clone()).or_default() += &share;
                    }
                }
            }
        }
        SelectionLaw { selection, authority: self.authority.clone(), probabilities, no_answer_probability }
    }
}
