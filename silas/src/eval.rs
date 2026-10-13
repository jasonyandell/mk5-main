//! Compilation and evaluation of queries.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use crate::frame::{Frame, Holdings, World};
use crate::nouns::DominoSet;
use crate::registry::{Access, Predicate, PredicateContext, PredicateSpec, Registry};
use crate::syntax::{Atom, NoneClause, Pattern, Query, Role, Sort, Term, Value, valid_name};
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

/// The most roles a query may declare, its `none` clauses' included.
const MAX_ROLES: usize = 64;

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

/// A `none` clause: its own classes, its body, and the enclosing classes the
/// body reads.
#[derive(Clone)]
struct CompiledNone {
    locals: Vec<usize>,
    body: Vec<Clause>,
    reads: BTreeSet<usize>,
}

#[derive(Clone)]
enum Clause {
    Atom(CompiledAtom),
    None(CompiledNone),
}

#[derive(Clone)]
struct CompiledPattern {
    /// The sort of each equality class: the pattern's own classes first,
    /// then each `none` clause's.
    sorts: Vec<Sort>,
    /// How many of the classes are the pattern's own.
    outer: usize,
    /// The class of each output.
    outputs: Vec<usize>,
    clauses: Vec<Clause>,
}

#[derive(Clone)]
pub struct CompiledQuery {
    source: Query,
    outputs: Vec<Role>,
    patterns: Vec<CompiledPattern>,
    specs: Vec<PredicateSpec>,
}

/// What compiling one pattern accumulates: its classes' sorts, the roles
/// declared so far, and the predicate specs used.
struct Compiler<'r> {
    registry: &'r Registry,
    sorts: Vec<Sort>,
    roles: usize,
    used: BTreeMap<String, PredicateSpec>,
}

/// The names in scope: each role's class and sort.
type Scope = BTreeMap<String, (usize, Sort)>;

impl Compiler<'_> {
    fn atom(&mut self, atom: &Atom, scope: &Scope) -> Result<CompiledAtom> {
        let predicate = self
            .registry
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
                    let (class, sort) = scope.get(name).ok_or_else(|| error(format!("undeclared role {name}")))?;
                    (Bound::Variable(*class), *sort)
                }
                Term::Literal(value) => (Bound::Literal(*value), value.sort()),
            };
            if actual != *expected {
                return Err(error(format!("{} expects {expected}, got {actual}", spec.name)));
            }
            terms.push(term);
        }
        let compiled = CompiledAtom { predicate, access: spec.access, terms, negated: atom.negated };
        self.used.insert(spec.name.clone(), spec);
        Ok(compiled)
    }

    /// Declare `roles` with the equality groups `equal` over them, numbering
    /// their classes after every class so far. Returns each role's class.
    fn declare(&mut self, roles: &[Role], equal: &[Vec<String>], scope: &Scope) -> Result<Vec<usize>> {
        self.roles += roles.len();
        if self.roles > MAX_ROLES {
            return Err(error(format!("at most {MAX_ROLES} roles are supported")));
        }
        let mut names = BTreeMap::new();
        for (i, role) in roles.iter().enumerate() {
            if !valid_name(&role.name) || Value::literal(&role.name).is_some() || !role.sort.is_role() {
                return Err(error(format!("invalid role {}:{}", role.name, role.sort)));
            }
            if scope.contains_key(&role.name) || names.insert(role.name.as_str(), i).is_some() {
                return Err(error(format!("duplicate role {}", role.name)));
            }
        }
        // Union the equality groups, then number the classes densely.
        let mut class: Vec<usize> = (0..roles.len()).collect();
        for group in equal {
            if group.len() < 2 {
                return Err(error("same requires at least two names"));
            }
            let resolve = |name: &String| {
                names.get(name.as_str()).copied().ok_or_else(|| {
                    error(format!(
                        "same names {name}, which is not a role declared with it (a none joins only its own roles)"
                    ))
                })
            };
            let first = resolve(&group[0])?;
            for name in &group[1..] {
                let next = resolve(name)?;
                if roles[first].sort != roles[next].sort {
                    return Err(error("an equality group cannot mix sorts"));
                }
                let (from, to) = (class[next], class[first]);
                class.iter_mut().filter(|c| **c == from).for_each(|c| *c = to);
            }
        }
        let mut dense = BTreeMap::new();
        for (i, c) in class.iter_mut().enumerate() {
            *c = *dense.entry(*c).or_insert_with(|| {
                self.sorts.push(roles[i].sort);
                self.sorts.len() - 1
            });
        }
        Ok(class)
    }

    fn clauses(&mut self, pattern: &Pattern, scope: &Scope) -> Result<Vec<Clause>> {
        let mut out = Vec::new();
        for atom in &pattern.atoms {
            out.push(Clause::Atom(self.atom(atom, scope)?));
        }
        for none in &pattern.nones {
            out.push(Clause::None(self.none(none, scope)?));
        }
        Ok(out)
    }

    fn none(&mut self, none: &NoneClause, scope: &Scope) -> Result<CompiledNone> {
        let classes = self.declare(&none.roles, &none.body.equal, scope)?;
        let mut inner = scope.clone();
        for (role, class) in none.roles.iter().zip(&classes) {
            inner.insert(role.name.clone(), (*class, role.sort));
        }
        let locals: BTreeSet<usize> = classes.iter().copied().collect();
        let body = self.clauses(&none.body, &inner)?;
        let reads = body.iter().flat_map(Clause::vars).filter(|v| !locals.contains(v)).collect();
        Ok(CompiledNone { locals: locals.into_iter().collect(), body, reads })
    }
}

impl Query {
    pub fn compile(&self, registry: &Registry) -> Result<CompiledQuery> {
        if self.roles.len() > MAX_ROLES {
            return Err(error(format!("at most {MAX_ROLES} roles are supported")));
        }
        let mut names = BTreeMap::new();
        for (i, role) in self.roles.iter().enumerate() {
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
            for group in &pattern.equal {
                group.iter().try_for_each(|name| resolve(name).map(drop))?;
            }
            let mut compiler = Compiler { registry, sorts: Vec::new(), roles: 0, used: BTreeMap::new() };
            let class = compiler.declare(&self.roles, &pattern.equal, &Scope::new())?;
            let outer = compiler.sorts.len();
            let scope: Scope =
                self.roles.iter().zip(&class).map(|(role, c)| (role.name.clone(), (*c, role.sort))).collect();
            let clauses = compiler.clauses(pattern, &scope)?;
            used.append(&mut compiler.used);
            patterns.push(CompiledPattern {
                sorts: compiler.sorts,
                outer,
                outputs: outputs.iter().map(|i| class[*i]).collect(),
                clauses,
            });
        }
        if self.patterns.is_empty() {
            // Still refuse invalid role declarations.
            Compiler { registry, sorts: Vec::new(), roles: 0, used: BTreeMap::new() }.declare(
                &self.roles,
                &[],
                &Scope::new(),
            )?;
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
enum Reader<'a> {
    Viewer(&'a Frame),
    World(Holdings<'a>),
}

impl CompiledAtom {
    /// `Some(true)` when the atom (or its negation) holds, `Some(false)` when
    /// it fails, `None` when the predicate is undefined here.
    fn evaluate(&self, reader: Reader<'_>, values: &[Option<Value>], budget: &mut Budget) -> Result<Option<bool>> {
        budget.spend(1)?;
        let args: Vec<_> = self
            .terms
            .iter()
            .map(|t| match t {
                Bound::Variable(i) => values[*i].expect("a ready atom"),
                Bound::Literal(v) => *v,
            })
            .collect();
        let context = match (self.access, reader) {
            (Access::Viewer, Reader::Viewer(frame)) => PredicateContext::Viewer(frame),
            (Access::Viewer, Reader::World(holdings)) => PredicateContext::Viewer(holdings.frame()),
            (Access::World, Reader::World(holdings)) => PredicateContext::World(holdings),
            (Access::World, Reader::Viewer(_)) => return Err(error("viewer evaluation refused a world predicate")),
        };
        Ok(self.predicate.evaluate(context, &args, budget)?.map(|value| value != self.negated))
    }
}

impl Clause {
    /// The classes the clause reads from outside itself.
    fn vars(&self) -> BTreeSet<usize> {
        match self {
            Clause::Atom(atom) => atom
                .terms
                .iter()
                .filter_map(|t| match t {
                    Bound::Variable(v) => Some(*v),
                    Bound::Literal(_) => None,
                })
                .collect(),
            Clause::None(none) => none.reads.clone(),
        }
    }

    fn ready(&self, values: &[Option<Value>]) -> bool {
        self.vars().iter().all(|v| values[*v].is_some())
    }

    fn evaluate(
        &self,
        reader: Reader<'_>,
        sorts: &[Sort],
        values: &mut [Option<Value>],
        budget: &mut Budget,
    ) -> Result<Option<bool>> {
        match self {
            Clause::Atom(atom) => atom.evaluate(reader, values, budget),
            Clause::None(none) => none.evaluate(reader, sorts, values, budget),
        }
    }

    /// Whether the clause holds: an undefined clause does not.
    fn holds(
        &self,
        reader: Reader<'_>,
        sorts: &[Sort],
        values: &mut [Option<Value>],
        budget: &mut Budget,
    ) -> Result<bool> {
        Ok(self.evaluate(reader, sorts, values, budget)? == Some(true))
    }
}

impl CompiledNone {
    /// `Some(true)` when every binding of the clause's own roles has a body
    /// clause that fails; `Some(false)` when some binding satisfies the whole
    /// body; `None` otherwise: some binding is refuted by nothing, only left
    /// undefined.
    fn evaluate(
        &self,
        reader: Reader<'_>,
        sorts: &[Sort],
        values: &mut [Option<Value>],
        budget: &mut Budget,
    ) -> Result<Option<bool>> {
        let mut undefined = false;
        let pending: Vec<usize> = (0..self.body.len()).collect();
        let found = self.witness(0, reader, sorts, values, &pending, false, &mut undefined, budget);
        for v in &self.locals {
            values[*v] = None;
        }
        Ok(if found? {
            Some(false)
        } else if undefined {
            None
        } else {
            Some(true)
        })
    }

    /// Whether some extension of the binding so far satisfies the body.
    /// Clauses are checked as soon as their roles are bound; a failed one
    /// refutes every extension, an undefined one leaves them unrefuted unless
    /// a later clause fails.
    #[allow(clippy::too_many_arguments)]
    fn witness(
        &self,
        depth: usize,
        reader: Reader<'_>,
        sorts: &[Sort],
        values: &mut [Option<Value>],
        pending: &[usize],
        mut unsure: bool,
        undefined: &mut bool,
        budget: &mut Budget,
    ) -> Result<bool> {
        budget.spend(1)?;
        let mut rest = Vec::new();
        for i in pending {
            let clause = &self.body[*i];
            if !clause.ready(values) {
                rest.push(*i);
                continue;
            }
            match clause.evaluate(reader, sorts, values, budget)? {
                Some(true) => {}
                Some(false) => return Ok(false),
                None => unsure = true,
            }
        }
        if depth == self.locals.len() {
            *undefined |= unsure;
            return Ok(!unsure);
        }
        let v = self.locals[depth];
        for candidate in sorts[v].values() {
            // The roles of one none are distinct from one another, not from
            // the roles around it.
            if self.locals[..depth].iter().any(|u| values[*u] == Some(candidate)) {
                continue;
            }
            values[v] = Some(candidate);
            if self.witness(depth + 1, reader, sorts, values, &rest, unsure, undefined, budget)? {
                return Ok(true);
            }
        }
        values[v] = None;
        Ok(false)
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
        format!("silas-query-v1\n{}\nregistry: {:?}", self.source, self.specs)
    }

    /// The answers in one world consistent with `frame`.
    pub fn evaluate(&self, frame: &Frame, world: &World, budget: &mut Budget) -> Result<Answers> {
        if !frame.contains(world) {
            return Err(error("the world is outside this frame's support"));
        }
        self.evaluate_in(Holdings::new(frame, world, DominoSet::EMPTY), budget)
    }

    pub(crate) fn evaluate_in(&self, holdings: Holdings<'_>, budget: &mut Budget) -> Result<Answers> {
        self.run(Reader::World(holdings), budget)
    }

    /// Evaluate a query that reads no world. Refused, before any work, if it
    /// names a `World` predicate.
    pub fn evaluate_viewer(&self, frame: &Frame, budget: &mut Budget) -> Result<Answers> {
        if let Some(spec) = self.specs.iter().find(|spec| spec.access == Access::World) {
            return Err(error(format!("viewer evaluation refused the world predicate {}", spec.name)));
        }
        self.run(Reader::Viewer(frame), budget)
    }

    fn run(&self, reader: Reader<'_>, budget: &mut Budget) -> Result<Answers> {
        budget.spend(1)?;
        let mut answers = Answers::new();
        for pattern in &self.patterns {
            pattern.evaluate(reader, budget, &mut answers)?;
        }
        Ok(answers)
    }
}

impl CompiledPattern {
    fn evaluate(&self, reader: Reader<'_>, budget: &mut Budget, answers: &mut Answers) -> Result<()> {
        let mut values = vec![None; self.sorts.len()];
        // Constant clauses are checked once; single-variable clauses shrink
        // their variable's domain before the join; the rest are checked as
        // soon as every role they read is bound.
        let mut constant = Vec::new();
        let mut unary = vec![Vec::new(); self.outer];
        let mut joined = Vec::new();
        for (i, clause) in self.clauses.iter().enumerate() {
            let vars = clause.vars();
            match (vars.len(), vars.first()) {
                (0, _) => constant.push(i),
                (1, Some(v)) => unary[*v].push(i),
                _ => joined.push(i),
            }
        }
        for i in constant {
            if !self.clauses[i].holds(reader, &self.sorts, &mut values, budget)? {
                return Ok(());
            }
        }
        let mut domains = Vec::new();
        for v in 0..self.outer {
            let mut domain = Vec::new();
            for candidate in self.sorts[v].values() {
                budget.spend(1)?;
                values[v] = Some(candidate);
                let mut admitted = true;
                for c in &unary[v] {
                    if !self.clauses[*c].holds(reader, &self.sorts, &mut values, budget)? {
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
        let mut order: Vec<usize> = (0..self.outer).collect();
        order.sort_by_key(|v| (domains[*v].len(), *v));
        Search { pattern: self, reader, budget, answers, domains: &domains, order: &order }.walk(
            0,
            &mut values,
            &joined,
        )
    }
}

struct Search<'a, 's> {
    pattern: &'a CompiledPattern,
    reader: Reader<'s>,
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
            let clause = &self.pattern.clauses[*i];
            if !clause.ready(values) {
                rest.push(*i);
            } else if !clause.holds(self.reader, &self.pattern.sorts, values, self.budget)? {
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
            // one sort are distinct values. A value's sort tag keeps sorts
            // apart; a none's classes are unbound here.
            if values[..self.pattern.outer].iter().flatten().any(|bound| bound == candidate) {
                continue;
            }
            values[v] = Some(*candidate);
            self.walk(depth + 1, values, &rest)?;
            values[v] = None;
        }
        Ok(())
    }
}
