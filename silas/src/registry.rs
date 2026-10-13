//! Predicates: the typed, versioned relations a query may name.

use std::collections::BTreeMap;
use std::sync::Arc;

use crate::eval::Budget;
use crate::frame::{Frame, Holdings};
use crate::syntax::{Sort, Value, valid_name};
use crate::{Result, error};

/// What a predicate may read.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Access {
    /// The frame only: the viewer's hand and the public residue.
    Viewer,
    /// The frame and one hidden world's [`Holdings`].
    World,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PredicateSpec {
    pub name: String,
    /// The semantic version: changing what the predicate computes changes it.
    pub version: String,
    pub parameters: Vec<Sort>,
    /// Zero for relations on the current state; positive for bounded
    /// continuations.
    pub horizon_plies: u32,
    pub access: Access,
}

#[derive(Clone, Copy, Debug)]
pub enum PredicateContext<'a> {
    Viewer(&'a Frame),
    World(Holdings<'a>),
}

impl<'a> PredicateContext<'a> {
    pub fn frame(self) -> &'a Frame {
        match self {
            Self::Viewer(frame) => frame,
            Self::World(holdings) => holdings.frame(),
        }
    }

    pub fn world(self) -> Option<Holdings<'a>> {
        match self {
            Self::Viewer(_) => None,
            Self::World(holdings) => Some(holdings),
        }
    }
}

/// A registered computation. It must be deterministic, respect its declared
/// horizon, charge its inner work to the budget, and consult nothing it does
/// not declare. The runtime checks argument sorts and withholds the world
/// from `Viewer` predicates; it cannot audit what a closure captures.
pub trait Predicate: Send + Sync {
    fn spec(&self) -> &PredicateSpec;

    /// `None` means undefined: neither the atom nor its negation holds.
    fn evaluate(&self, context: PredicateContext<'_>, args: &[Value], budget: &mut Budget) -> Result<Option<bool>>;
}

/// Words the grammar reserves.
const RESERVED: [&str; 6] = ["query", "roles", "out", "pattern", "same", "not"];

#[derive(Clone, Default)]
pub struct Registry {
    pub(crate) predicates: BTreeMap<String, Arc<dyn Predicate>>,
}

impl Registry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(&mut self, predicate: impl Predicate + 'static) -> Result<()> {
        let spec = predicate.spec();
        if !valid_name(&spec.name) || RESERVED.contains(&spec.name.as_str()) || spec.version.trim().is_empty() {
            return Err(error("a predicate needs an unreserved name and a semantic version"));
        }
        if self.predicates.contains_key(&spec.name) {
            return Err(error(format!("predicate {} is already registered", spec.name)));
        }
        self.predicates.insert(spec.name.clone(), Arc::new(predicate));
        Ok(())
    }

    pub fn specs(&self) -> Vec<PredicateSpec> {
        self.predicates.values().map(|p| p.spec().clone()).collect()
    }

    /// The built-in relations, version `silas-v0/core`.
    pub fn standard() -> Self {
        use Access::{Viewer, World};
        use Sort::{Chair as C, Context as Q, Domino as D, Number as N, Team as T};
        let mut registry = Self::new();
        for (name, parameters, access) in [
            ("live", vec![D], Viewer),
            ("holds", vec![C, D], World),
            ("in", vec![D, Q], Viewer),
            ("double", vec![D], Viewer),
            ("beats", vec![D, D, Q], Viewer),
            ("boss", vec![D, Q], Viewer),
            ("master", vec![D], Viewer),
            ("void", vec![C, Q], World),
            ("quota", vec![C, N], Viewer),
            ("tile", vec![D, D], Viewer),
            ("chair", vec![C, C], Viewer),
            ("context", vec![Q, Q], Viewer),
            ("team", vec![C, T], Viewer),
            ("partner", vec![C, C], Viewer),
            ("opponent", vec![C, C], Viewer),
            ("successor", vec![C, C], Viewer),
            ("viewer", vec![C], Viewer),
            ("leader", vec![C], Viewer),
            ("next-actor", vec![C], Viewer),
            ("led-context", vec![Q], Viewer),
            ("current-winner", vec![C], Viewer),
            ("played", vec![D], Viewer),
            ("count", vec![D, N], Viewer),
            ("legal", vec![C, D], World),
            ("forced", vec![C, D], World),
            ("own-legal", vec![D], Viewer),
            ("trick-play", vec![C, D], Viewer),
            ("leads-context", vec![D, Q], Viewer),
        ] {
            let spec = PredicateSpec {
                name: name.to_owned(),
                version: "silas-v0/core".to_owned(),
                parameters,
                horizon_plies: 0,
                access,
            };
            registry.register(Builtin(spec)).expect("built-in names are unique and valid");
        }
        registry
    }
}

struct Builtin(PredicateSpec);

impl Predicate for Builtin {
    fn spec(&self) -> &PredicateSpec {
        &self.0
    }

    fn evaluate(&self, context: PredicateContext<'_>, args: &[Value], _: &mut Budget) -> Result<Option<bool>> {
        use Value::{Chair as C, Context as Q, Domino as D, Number as N, Team as T};
        let frame = context.frame();
        let physics = frame.physics();
        let kernel = frame.kernel();
        let live = kernel.live();
        let world = || context.world().ok_or_else(|| error("a world predicate requires a world"));
        let value = match (self.0.name.as_str(), args) {
            ("live", [D(d)]) => live.contains(*d),
            ("holds", [C(c), D(d)]) => world()?.holds(*c, *d)?,
            ("in", [D(d), Q(q)]) => physics.follows(*d, *q),
            ("double", [D(d)]) => d.is_double(),
            ("beats", [D(a), D(b), Q(q)]) => physics.trick_key(*a, *q) > physics.trick_key(*b, *q),
            ("boss", [D(d), Q(q)]) => {
                live.contains(*d) && physics.follows(*d, *q) && physics.beats(*q, *d).is_disjoint(live)
            }
            ("master", [D(d)]) => live.contains(*d) && physics.threat(*d).is_disjoint(live),
            ("void", [C(c), Q(q)]) => world()?.hand_within(*c, physics.incidence(*q))?.is_empty(),
            ("quota", [C(c), N(n)]) => kernel.capacity(*c) == *n as usize,
            ("tile", [D(a), D(b)]) => a == b,
            ("chair", [C(a), C(b)]) => a == b,
            ("context", [Q(a), Q(b)]) => a == b,
            ("team", [C(c), T(t)]) => c.team() == *t,
            ("partner", [C(a), C(b)]) => a.partner() == *b,
            ("opponent", [C(a), C(b)]) => a.team() != b.team(),
            ("successor", [C(a), C(b)]) => a.successor() == *b,
            ("viewer", [C(c)]) => *c == kernel.viewer(),
            ("leader", [C(c)]) => *c == frame.leader(),
            ("next-actor", [C(c)]) => frame.next_actor() == Some(*c),
            ("led-context", [Q(q)]) => match frame.led_context() {
                Some(led) => *q == led,
                None => return Ok(None),
            },
            ("current-winner", [C(c)]) => match frame.current_winner() {
                Some(winner) => *c == winner,
                None => return Ok(None),
            },
            ("played", [D(d)]) => frame.played().contains(*d),
            ("count", [D(d), N(n)]) => physics.count(*d) == *n,
            ("own-legal", [D(d)]) => {
                if frame.next_actor() != Some(kernel.viewer()) {
                    return Ok(None);
                }
                physics.legal_plays(kernel.viewer_hand(), frame.lead()).contains(*d)
            }
            ("trick-play", [C(c), D(d)]) => {
                frame.prefix().iter().enumerate().any(|(i, tile)| frame.leader().plus(i) == *c && tile == d)
            }
            ("leads-context", [D(d), Q(q)]) => physics.led_context(*d) == *q,
            ("legal" | "forced", [C(c), D(d)]) => {
                if frame.next_actor() != Some(*c) {
                    return Ok(None);
                }
                let legal = physics.legal_plays(world()?.hand(*c)?, frame.lead());
                if self.0.name == "forced" { legal == crate::DominoSet::single(*d) } else { legal.contains(*d) }
            }
            _ => return Err(error("built-in argument mismatch (an invalid compiled query)")),
        };
        Ok(Some(value))
    }
}
