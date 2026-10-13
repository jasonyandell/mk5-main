//! Compilation and evaluation of queries.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use crate::frame::{Frame, Holdings, World};
use crate::nouns::DominoSet;
use crate::registry::{Access, Predicate, PredicateContext, PredicateSpec, Registry};
use crate::syntax::{Query, Role, Sort, Term, Value, valid_name};
use crate::{Result, error};

/// An explicit finite budget of work, shared across a whole computation. No
/// clocks: exceeding it is a refusal, never a partial result.
#[derive(Clone, Debug)]
pub struct Budget {
    limit: u64,
    spent: u64,
}

impl Budget {
    pub fn new(limit: u64) -> Self {
        Self { limit, spent: 0 }
    }

    pub fn spent(&self) -> u64 {
        self.spent
    }

    pub fn remaining(&self) -> u64 {
        self.limit - self.spent
    }

    pub fn spend(&mut self, n: u64) -> Result<()> {
        if n > self.remaining() {
            return Err(error(format!("work budget exhausted ({}/{})", self.spent, self.limit)));
        }
        self.spent += n;
        Ok(())
    }
}

/// One output tuple, in the query's declared output order.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Answer(pub Vec<Value>);

pub type Answers = BTreeSet<Answer>;

#[derive(Clone)]
enum Bound {
    Variable(usize),
    Literal(Value),
}

#[derive(Clone)]
struct CompiledAtom {
    predicate: Arc<dyn Predicate>,
    access: Access,
    terms: Vec<Bound>,
    negated: bool,
}

#[derive(Clone)]
struct CompiledPattern {
    /// The sort of each equality class.
    sorts: Vec<Sort>,
    /// The class of each output.
    outputs: Vec<usize>,
    atoms: Vec<CompiledAtom>,
}

#[derive(Clone)]
pub struct CompiledQuery {
    source: Query,
    outputs: Vec<Role>,
    patterns: Vec<CompiledPattern>,
    specs: Vec<PredicateSpec>,
}

impl Query {
    pub fn compile(&self, registry: &Registry) -> Result<CompiledQuery> {
        if self.roles.len() > 64 {
            return Err(error("at most 64 roles are supported"));
        }
        let mut names = BTreeMap::new();
        for (i, role) in self.roles.iter().enumerate() {
            if !valid_name(&role.name) || Value::literal(&role.name).is_some() || !role.sort.is_role() {
                return Err(error(format!("invalid role {}:{}", role.name, role.sort)));
            }
            if names.insert(role.name.as_str(), i).is_some() {
                return Err(error(format!("duplicate role {}", role.name)));
            }
        }
        let resolve = |name: &str| names.get(name).copied().ok_or_else(|| error(format!("undeclared role {name}")));
        let mut outputs = Vec::new();
        for name in &self.outputs {
            let i = resolve(name)?;
            if outputs.contains(&i) {
                return Err(error(format!("duplicate output {name}")));
            }
            outputs.push(i);
        }
        let mut patterns = Vec::new();
        let mut used = BTreeMap::new();
        for pattern in &self.patterns {
            // Union the equality groups, then number the classes densely.
            let mut class: Vec<usize> = (0..self.roles.len()).collect();
            for group in &pattern.equal {
                if group.len() < 2 {
                    return Err(error("same requires at least two names"));
                }
                let first = resolve(&group[0])?;
                for name in &group[1..] {
                    let next = resolve(name)?;
                    if self.roles[first].sort != self.roles[next].sort {
                        return Err(error("an equality group cannot mix sorts"));
                    }
                    let (from, to) = (class[next], class[first]);
                    class.iter_mut().filter(|c| **c == from).for_each(|c| *c = to);
                }
            }
            let mut dense = BTreeMap::new();
            let mut sorts = Vec::new();
            for (i, c) in class.iter_mut().enumerate() {
                let next = dense.len();
                *c = *dense.entry(*c).or_insert_with(|| {
                    sorts.push(self.roles[i].sort);
                    next
                });
            }
            let mut atoms = Vec::new();
            for atom in &pattern.atoms {
                let predicate = registry
                    .predicates
                    .get(&atom.predicate)
                    .ok_or_else(|| error(format!("unregistered predicate {}", atom.predicate)))?
                    .clone();
                let spec = predicate.spec().clone();
                if spec.parameters.len() != atom.args.len() {
                    return Err(error(format!(
                        "{} expects {} arguments, got {}",
                        spec.name,
                        spec.parameters.len(),
                        atom.args.len()
                    )));
                }
                let mut terms = Vec::new();
                for (arg, expected) in atom.args.iter().zip(&spec.parameters) {
                    let (term, actual) = match arg {
                        Term::Role(name) => {
                            let i = resolve(name)?;
                            (Bound::Variable(class[i]), self.roles[i].sort)
                        }
                        Term::Literal(value) => (Bound::Literal(*value), value.sort()),
                    };
                    if actual != *expected {
                        return Err(error(format!("{} expects {expected}, got {actual}", spec.name)));
                    }
                    terms.push(term);
                }
                atoms.push(CompiledAtom { predicate, access: spec.access, terms, negated: atom.negated });
                used.insert(spec.name.clone(), spec);
            }
            patterns.push(CompiledPattern { sorts, outputs: outputs.iter().map(|i| class[*i]).collect(), atoms });
        }
        Ok(CompiledQuery {
            source: self.clone(),
            outputs: outputs.iter().map(|i| self.roles[*i].clone()).collect(),
            patterns,
            specs: used.into_values().collect(),
        })
    }
}

/// Where atoms read from: the frame alone, or a world through its holdings.
#[derive(Clone, Copy)]
enum Scope<'a> {
    Viewer(&'a Frame),
    World(Holdings<'a>),
}

impl CompiledAtom {
    fn ready(&self, values: &[Option<Value>]) -> bool {
        self.terms.iter().all(|t| match t {
            Bound::Literal(_) => true,
            Bound::Variable(i) => values[*i].is_some(),
        })
    }

    fn check(&self, scope: Scope<'_>, values: &[Option<Value>], budget: &mut Budget) -> Result<bool> {
        budget.spend(1)?;
        let args: Vec<_> = self
            .terms
            .iter()
            .map(|t| match t {
                Bound::Variable(i) => values[*i].expect("a ready atom"),
                Bound::Literal(v) => *v,
            })
            .collect();
        let context = match (self.access, scope) {
            (Access::Viewer, Scope::Viewer(frame)) => PredicateContext::Viewer(frame),
            (Access::Viewer, Scope::World(holdings)) => PredicateContext::Viewer(holdings.frame()),
            (Access::World, Scope::World(holdings)) => PredicateContext::World(holdings),
            (Access::World, Scope::Viewer(_)) => return Err(error("viewer evaluation refused a world predicate")),
        };
        Ok(self.predicate.evaluate(context, &args, budget)?.is_some_and(|value| value != self.negated))
    }
}

impl CompiledQuery {
    pub fn source(&self) -> &Query {
        &self.source
    }

    pub fn outputs(&self) -> &[Role] {
        &self.outputs
    }

    pub fn predicate_specs(&self) -> &[PredicateSpec] {
        &self.specs
    }

    /// A stable, complete textual identity: the syntax version, the source,
    /// and every predicate spec it uses. Not a normal form under logical
    /// equivalence.
    pub fn identity(&self) -> String {
        format!("silas-query-v0\n{}\nregistry: {:?}", self.source, self.specs)
    }

    /// The answers in one world consistent with `frame`.
    pub fn evaluate(&self, frame: &Frame, world: &World, budget: &mut Budget) -> Result<Answers> {
        if !frame.contains(world) {
            return Err(error("the world is outside this frame's support"));
        }
        self.evaluate_in(Holdings::new(frame, world, DominoSet::EMPTY), budget)
    }

    pub(crate) fn evaluate_in(&self, holdings: Holdings<'_>, budget: &mut Budget) -> Result<Answers> {
        self.run(Scope::World(holdings), budget)
    }

    /// Evaluate a query that reads no world. Refused, before any work, if it
    /// names a `World` predicate.
    pub fn evaluate_viewer(&self, frame: &Frame, budget: &mut Budget) -> Result<Answers> {
        if let Some(spec) = self.specs.iter().find(|spec| spec.access == Access::World) {
            return Err(error(format!("viewer evaluation refused the world predicate {}", spec.name)));
        }
        self.run(Scope::Viewer(frame), budget)
    }

    fn run(&self, scope: Scope<'_>, budget: &mut Budget) -> Result<Answers> {
        budget.spend(1)?;
        let mut answers = Answers::new();
        for pattern in &self.patterns {
            pattern.evaluate(scope, budget, &mut answers)?;
        }
        Ok(answers)
    }
}

impl CompiledPattern {
    fn evaluate(&self, scope: Scope<'_>, budget: &mut Budget, answers: &mut Answers) -> Result<()> {
        let mut values = vec![None; self.sorts.len()];
        // Constant atoms are checked once; single-variable atoms shrink their
        // variable's domain before the join; the rest are checked as soon as
        // every argument is bound.
        let mut constant = Vec::new();
        let mut unary = vec![Vec::new(); self.sorts.len()];
        let mut joined = Vec::new();
        for (i, atom) in self.atoms.iter().enumerate() {
            let vars: BTreeSet<usize> = atom
                .terms
                .iter()
                .filter_map(|t| match t {
                    Bound::Variable(v) => Some(*v),
                    Bound::Literal(_) => None,
                })
                .collect();
            match (vars.len(), vars.first()) {
                (0, _) => constant.push(i),
                (1, Some(v)) => unary[*v].push(i),
                _ => joined.push(i),
            }
        }
        for i in constant {
            if !self.atoms[i].check(scope, &values, budget)? {
                return Ok(());
            }
        }
        let mut domains = Vec::new();
        for (v, sort) in self.sorts.iter().enumerate() {
            let mut domain = Vec::new();
            for candidate in sort.values() {
                budget.spend(1)?;
                values[v] = Some(candidate);
                let mut admitted = true;
                for a in &unary[v] {
                    if !self.atoms[*a].check(scope, &values, budget)? {
                        admitted = false;
                        break;
                    }
                }
                if admitted {
                    domain.push(candidate);
                }
            }
            values[v] = None;
            if domain.is_empty() {
                return Ok(());
            }
            domains.push(domain);
        }
        let mut order: Vec<usize> = (0..self.sorts.len()).collect();
        order.sort_by_key(|v| (domains[*v].len(), *v));
        Search { pattern: self, scope, budget, answers, domains: &domains, order: &order }.walk(0, &mut values, &joined)
    }
}

struct Search<'a, 's> {
    pattern: &'a CompiledPattern,
    scope: Scope<'s>,
    budget: &'a mut Budget,
    answers: &'a mut Answers,
    domains: &'a [Vec<Value>],
    order: &'a [usize],
}

impl Search<'_, '_> {
    fn walk(&mut self, depth: usize, values: &mut [Option<Value>], pending: &[usize]) -> Result<()> {
        self.budget.spend(1)?;
        // Once an output tuple has a witness, further witnesses cannot change
        // the relation, even across patterns.
        let projected: Option<Vec<Value>> = self.pattern.outputs.iter().map(|v| values[*v]).collect();
        if projected.is_some_and(|row| self.answers.contains(&Answer(row))) {
            return Ok(());
        }
        let mut rest = Vec::new();
        for i in pending {
            let atom = &self.pattern.atoms[*i];
            if !atom.ready(values) {
                rest.push(*i);
            } else if !atom.check(self.scope, values, self.budget)? {
                return Ok(());
            }
        }
        if depth == self.order.len() {
            let row = self.pattern.outputs.iter().map(|v| values[*v].expect("a complete binding")).collect();
            self.answers.insert(Answer(row));
            return Ok(());
        }
        let v = self.order[depth];
        for candidate in &self.domains[v] {
            // Equality has already merged named roles, so distinct classes of
            // one sort are distinct values. A value's sort tag keeps sorts apart.
            if values.iter().flatten().any(|bound| bound == candidate) {
                continue;
            }
            values[v] = Some(*candidate);
            self.walk(depth + 1, values, &rest)?;
            values[v] = None;
        }
        Ok(())
    }
}
