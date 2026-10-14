// SPDX-FileCopyrightText: Jason Yandell
// SPDX-License-Identifier: CC0-1.0

//! Compilation and evaluation of queries.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use crate::auction::AuctionFrame;
use crate::frame::{Frame, Holdings, World};
use crate::nouns::{Domino, DominoSet};
use crate::registry::{Access, NumberAt, Phase, Predicate, PredicateContext, PredicateSpec, Registry};
use crate::syntax::{
    Atom, Pattern, Quantifier, Quantity, Query, Role, Scope as ScopeClause, Sort, Span, Term, Value, valid_name,
};
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
const MAX_ROLES: usize = 256;

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

/// What a scope or an aggregate binds: its own classes, its body, and the
/// enclosing classes the body reads.
#[derive(Clone)]
struct Binder {
    locals: Vec<usize>,
    body: Vec<Clause>,
    reads: BTreeSet<usize>,
    /// The body clauses that become ready at each depth of the search, in
    /// source order: `schedule[d]` once the first `d` locals are bound. A
    /// scope is read only once every enclosing role it names is bound, so
    /// when a clause is ready depends on its locals alone.
    schedule: Vec<Vec<usize>>,
}

/// A `some` or `none` clause.
#[derive(Clone)]
struct CompiledScope {
    quantifier: Quantifier,
    binder: Binder,
}

/// A number a comparison reads, as an interval of integers: decided when its
/// bounds meet.
#[derive(Clone)]
enum CompiledQuantity {
    Number(u64),
    /// A functional predicate's number: the atom of its other arguments.
    Function(CompiledAtom),
    Aggregate(CompiledAggregate),
    Sum(Vec<CompiledQuantity>),
    /// The first part less the rest.
    Difference(Vec<CompiledQuantity>),
}

/// `(card ...)` or `(points ...)`. Its value at a frame is a function of
/// its text and the values of the enclosing roles it names, so it is counted
/// once per evaluation for each binding of those: `id` numbers its text
/// among the aggregates evaluated together (one query, or one policy's rules
/// of a section), and `free` lists the classes of the enclosing names it
/// reads, in the order of their names.
#[derive(Clone)]
struct CompiledAggregate {
    binder: Binder,
    /// The class of the domino whose count is summed; none for `card`.
    points: Option<usize>,
    shape: String,
    free: Vec<usize>,
    id: usize,
}

/// `(<= left right)`, and the enclosing classes its aggregates read.
#[derive(Clone)]
struct CompiledAtMost {
    left: CompiledQuantity,
    right: CompiledQuantity,
    reads: BTreeSet<usize>,
}

#[derive(Clone)]
enum Clause {
    Atom(CompiledAtom),
    Scope(CompiledScope),
    AtMost(Box<CompiledAtMost>),
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
    /// The class of each of the query's roles, by name.
    names: Vec<(String, usize)>,
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

/// A located compile error, or a plain one for syntax built in Rust.
fn at(span: &Span, message: String) -> crate::Error {
    span.error(message)
}

impl Compiler<'_> {
    fn atom(&mut self, atom: &Atom, scope: &Scope) -> Result<CompiledAtom> {
        self.application(&atom.predicate, &atom.args, atom.negated, false, scope, &atom.span)
    }

    /// `(p a ...)`, an atom; or, for a `function`, a functional predicate
    /// given every argument but its last, a number.
    fn application(
        &mut self,
        name: &str,
        args: &[Term],
        negated: bool,
        function: bool,
        scope: &Scope,
        span: &Span,
    ) -> Result<CompiledAtom> {
        let predicate =
            self.registry.predicates.get(name).ok_or_else(|| at(span, crate::outcome::unregistered(name)))?.clone();
        let spec = predicate.spec().clone();
        let mut parameters = &spec.parameters[..];
        if function {
            if !predicate.functional() {
                return Err(at(
                    span,
                    format!(
                        "{name} is not functional, so ({name} ...) is no number: compare a functional predicate's, as (to-set)"
                    ),
                ));
            }
            parameters = &parameters[..parameters.len() - 1];
        }
        if parameters.len() != args.len() {
            return Err(at(span, format!("{} expects {} arguments, got {}", spec.name, parameters.len(), args.len())));
        }
        let mut terms = Vec::new();
        for (arg, expected) in args.iter().zip(parameters) {
            let (term, actual) = match arg {
                Term::Role(name) => {
                    let (class, sort) = scope.get(name).ok_or_else(|| at(span, format!("undeclared role {name}")))?;
                    (Bound::Variable(*class), *sort)
                }
                Term::Literal(value) => (Bound::Literal(*value), value.sort()),
            };
            if actual != *expected {
                return Err(at(span, format!("{} expects {expected}, got {actual}", spec.name)));
            }
            terms.push(term);
        }
        let compiled = CompiledAtom { predicate, access: spec.access, terms, negated };
        self.used.insert(spec.name.clone(), spec);
        Ok(compiled)
    }

    /// Declare `roles` with the equality groups `equal` over them, numbering
    /// their classes after every class so far. Returns each role's class.
    fn declare(&mut self, roles: &[Role], equal: &[Vec<String>], scope: &Scope, span: &Span) -> Result<Vec<usize>> {
        self.roles += roles.len();
        if self.roles > MAX_ROLES {
            return Err(at(span, format!("at most {MAX_ROLES} roles are supported")));
        }
        let mut names = BTreeMap::new();
        for (i, role) in roles.iter().enumerate() {
            if !valid_name(&role.name) || Value::literal(&role.name).is_some() || !role.sort.is_role() {
                return Err(at(&role.span, format!("invalid role {}:{}", role.name, role.sort)));
            }
            if scope.contains_key(&role.name) || names.insert(role.name.as_str(), i).is_some() {
                return Err(at(&role.span, format!("duplicate role {}", role.name)));
            }
        }
        // Union the equality groups, then number the classes densely.
        let mut class: Vec<usize> = (0..roles.len()).collect();
        for group in equal {
            if group.len() < 2 {
                return Err(at(span, "same requires at least two names".into()));
            }
            let resolve = |name: &String| {
                names.get(name.as_str()).copied().ok_or_else(|| {
                    at(
                        span,
                        format!(
                            "same names {name}, which is not a role declared with it (a scope joins only its own roles)"
                        ),
                    )
                })
            };
            let first = resolve(&group[0])?;
            for name in &group[1..] {
                let next = resolve(name)?;
                if roles[first].sort != roles[next].sort {
                    return Err(at(span, "an equality group cannot mix sorts".into()));
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
        for clause in &pattern.scopes {
            out.push(Clause::Scope(self.scope(clause, scope)?));
        }
        for bound in &pattern.bounds {
            let left = self.quantity(&bound.left, scope, &bound.span)?;
            let right = self.quantity(&bound.right, scope, &bound.span)?;
            let mut reads = left.reads();
            reads.extend(right.reads());
            out.push(Clause::AtMost(Box::new(CompiledAtMost { left, right, reads })));
        }
        Ok(out)
    }

    fn scope(&mut self, clause: &ScopeClause, scope: &Scope) -> Result<CompiledScope> {
        let (binder, _) = self.binder(&clause.roles, &clause.body, scope, &clause.span)?;
        Ok(CompiledScope { quantifier: clause.quantifier, binder })
    }

    /// Declare `roles` and compile `body` under them. Returns the binder and
    /// each role's class.
    fn binder(&mut self, roles: &[Role], body: &Pattern, scope: &Scope, span: &Span) -> Result<(Binder, Vec<usize>)> {
        let classes = self.declare(roles, &body.equal, scope, span)?;
        let mut inner = scope.clone();
        for (role, class) in roles.iter().zip(&classes) {
            inner.insert(role.name.clone(), (*class, role.sort));
        }
        let locals: BTreeSet<usize> = classes.iter().copied().collect();
        let body = self.clauses(body, &inner)?;
        let reads = body.iter().flat_map(Clause::vars).filter(|v| !locals.contains(v)).collect();
        let locals: Vec<usize> = locals.into_iter().collect();
        let mut schedule = vec![Vec::new(); locals.len() + 1];
        for (i, c) in body.iter().enumerate() {
            let depth = c.vars().iter().filter_map(|v| locals.iter().position(|l| l == v)).map(|k| k + 1).max();
            schedule[depth.unwrap_or(0)].push(i);
        }
        Ok((Binder { locals, body, reads, schedule }, classes))
    }

    fn quantity(&mut self, quantity: &Quantity, scope: &Scope, span: &Span) -> Result<CompiledQuantity> {
        Ok(match quantity {
            Quantity::Number(n) => CompiledQuantity::Number(u64::from(*n)),
            Quantity::Function(f) => {
                CompiledQuantity::Function(self.application(&f.predicate, &f.args, false, true, scope, &f.span)?)
            }
            Quantity::Parameter(name) => {
                return Err(at(span, format!("{name} is a definition's number parameter, outside its definition")));
            }
            Quantity::Card(a) | Quantity::Points(a) => {
                let points = matches!(quantity, Quantity::Points(_));
                if points {
                    if !matches!(&a.roles[..], [role] if role.sort == Sort::Domino) {
                        return Err(at(&a.span, "points sums the count of one domino role".into()));
                    }
                    // The sum reads each tile's count, as the count predicate does.
                    let count =
                        self.registry.predicates.get("count").ok_or_else(|| {
                            at(&a.span, "points reads the count, and the registry has no count".into())
                        })?;
                    self.used.insert("count".to_owned(), count.spec().clone());
                }
                let (binder, classes) = self.binder(&a.roles, &a.body, scope, &a.span)?;
                // The enclosing names the aggregate's text reads, in order.
                let mut names = BTreeSet::new();
                let own = a.roles.iter().map(|r| r.name.clone()).collect();
                crate::syntax::free_names(&a.body, &own, &mut names);
                let free = names.iter().map(|n| scope[n].0).collect();
                CompiledQuantity::Aggregate(CompiledAggregate {
                    points: points.then(|| classes[0]),
                    shape: crate::syntax::shape_text(quantity, &names),
                    free,
                    id: 0,
                    binder,
                })
            }
            Quantity::Sum(parts) => {
                CompiledQuantity::Sum(parts.iter().map(|q| self.quantity(q, scope, span)).collect::<Result<_>>()?)
            }
            Quantity::Difference(parts) => CompiledQuantity::Difference(
                parts.iter().map(|q| self.quantity(q, scope, span)).collect::<Result<_>>()?,
            ),
        })
    }
}

impl CompiledQuantity {
    /// The enclosing classes its aggregates read.
    fn reads(&self) -> BTreeSet<usize> {
        match self {
            CompiledQuantity::Number(_) => BTreeSet::new(),
            CompiledQuantity::Function(f) => Clause::Atom(f.clone()).vars(),
            CompiledQuantity::Aggregate(a) => a.binder.reads.clone(),
            CompiledQuantity::Sum(parts) | CompiledQuantity::Difference(parts) => {
                parts.iter().flat_map(CompiledQuantity::reads).collect()
            }
        }
    }
}

/// Number every aggregate by its text, in `table`: two aggregates of one
/// text get one number, so an evaluation that reads both counts it once.
fn intern(clauses: &mut [Clause], table: &mut BTreeMap<String, usize>) {
    fn quantity(q: &mut CompiledQuantity, table: &mut BTreeMap<String, usize>) {
        match q {
            CompiledQuantity::Number(_) | CompiledQuantity::Function(_) => {}
            CompiledQuantity::Aggregate(a) => {
                let next = table.len();
                a.id = *table.entry(a.shape.clone()).or_insert(next);
                intern(&mut a.binder.body, table);
            }
            CompiledQuantity::Sum(parts) | CompiledQuantity::Difference(parts) => {
                parts.iter_mut().for_each(|p| quantity(p, table));
            }
        }
    }
    for clause in clauses {
        match clause {
            Clause::Atom(_) => {}
            Clause::Scope(scope) => intern(&mut scope.binder.body, table),
            Clause::AtMost(bound) => {
                quantity(&mut bound.left, table);
                quantity(&mut bound.right, table);
            }
        }
    }
}

impl Query {
    pub fn compile(&self, registry: &Registry) -> Result<CompiledQuery> {
        if self.roles.len() > MAX_ROLES {
            return Err(at(&self.span, format!("at most {MAX_ROLES} roles are supported")));
        }
        let mut names = BTreeMap::new();
        for (i, role) in self.roles.iter().enumerate() {
            if names.insert(role.name.as_str(), i).is_some() {
                return Err(at(&role.span, format!("duplicate role {}", role.name)));
            }
        }
        let resolve =
            |name: &str| names.get(name).copied().ok_or_else(|| at(&self.span, format!("undeclared role {name}")));
        let mut outputs = Vec::new();
        for name in &self.outputs {
            let i = resolve(name)?;
            if outputs.contains(&i) {
                return Err(at(&self.span, format!("duplicate output {name}")));
            }
            outputs.push(i);
        }
        let mut patterns = Vec::new();
        let mut used = BTreeMap::new();
        for pattern in &self.patterns {
            for group in &pattern.equal {
                group.iter().try_for_each(|name| {
                    names
                        .get(name.as_str())
                        .map(drop)
                        .ok_or_else(|| at(&pattern.span, format!("undeclared role {name}")))
                })?;
            }
            let mut compiler = Compiler { registry, sorts: Vec::new(), roles: 0, used: BTreeMap::new() };
            let class = compiler.declare(&self.roles, &pattern.equal, &Scope::new(), &pattern.span)?;
            let outer = compiler.sorts.len();
            let scope: Scope =
                self.roles.iter().zip(&class).map(|(role, c)| (role.name.clone(), (*c, role.sort))).collect();
            let clauses = compiler.clauses(pattern, &scope)?;
            used.append(&mut compiler.used);
            patterns.push(CompiledPattern {
                sorts: compiler.sorts,
                outer,
                outputs: outputs.iter().map(|i| class[*i]).collect(),
                names: self.roles.iter().zip(&class).map(|(role, c)| (role.name.clone(), *c)).collect(),
                clauses,
            });
        }
        let mut shapes = BTreeMap::new();
        for pattern in &mut patterns {
            intern(&mut pattern.clauses, &mut shapes);
        }
        if self.patterns.is_empty() {
            // Still refuse invalid role declarations.
            Compiler { registry, sorts: Vec::new(), roles: 0, used: BTreeMap::new() }.declare(
                &self.roles,
                &[],
                &Scope::new(),
                &self.span,
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

/// Where atoms read from: the frame alone, a world through its holdings, or
/// a seat's view of the auction.
#[derive(Clone, Copy)]
enum Reader<'a> {
    Viewer(&'a Frame),
    World(Holdings<'a>),
    Auction(&'a AuctionFrame),
}

/// What one evaluation reads: the reader, and the aggregates it has counted
/// so far. An aggregate's interval is a function of the reader and the
/// values of the enclosing roles it reads, so within one evaluation each is
/// counted once per binding of those.
#[derive(Clone, Copy)]
struct Cx<'a> {
    reader: Reader<'a>,
    memo: &'a Memo,
}

/// Counted aggregates, keyed by the binder (its place in the compiled
/// query) and the values its reads were bound to: the interval, and whether
/// the count ran to the end (else it stopped at a cap, its low end there).
pub(crate) type Memo = std::cell::RefCell<BTreeMap<(usize, Vec<Value>), (u64, u64, bool)>>;

impl CompiledAtom {
    /// `Some(true)` when the atom (or its negation) holds, `Some(false)` when
    /// it fails, `None` when the predicate is undefined here.
    fn evaluate(&self, cx: Cx<'_>, values: &[Option<Value>], budget: &mut Budget) -> Result<Option<bool>> {
        self.read(cx, values, budget, |predicate, context, args, budget| {
            Ok(predicate.evaluate(context, args, budget)?.map(|value| value != self.negated))
        })
    }

    /// A functional predicate's number at its arguments.
    fn value(&self, cx: Cx<'_>, values: &[Option<Value>], budget: &mut Budget) -> Result<NumberAt> {
        self.read(cx, values, budget, |predicate, context, args, budget| predicate.value(context, args, budget))
    }

    /// `f` of the predicate, its context and its arguments at this binding.
    fn read<T>(
        &self,
        cx: Cx<'_>,
        values: &[Option<Value>],
        budget: &mut Budget,
        f: impl FnOnce(&dyn Predicate, PredicateContext<'_>, &[Value], &mut Budget) -> Result<T>,
    ) -> Result<T> {
        budget.spend(1)?;
        let arg = |t: &Bound| match t {
            Bound::Variable(i) => values[*i].expect("a ready atom"),
            Bound::Literal(v) => *v,
        };
        // The built-ins take at most three arguments: those are read on the
        // stack, and only a longer registered predicate's are collected.
        let mut inline = [Value::Number(0); 3];
        let collected: Vec<Value>;
        let args: &[Value] = if self.terms.len() <= inline.len() {
            for (slot, t) in inline.iter_mut().zip(&self.terms) {
                *slot = arg(t);
            }
            &inline[..self.terms.len()]
        } else {
            collected = self.terms.iter().map(arg).collect();
            &collected
        };
        let context = match (self.access, cx.reader) {
            (Access::Viewer, Reader::Viewer(frame)) => PredicateContext::Viewer(frame),
            (Access::Viewer, Reader::World(holdings)) => PredicateContext::Viewer(holdings.frame()),
            (Access::World | Access::Clairvoyant, Reader::World(holdings)) => PredicateContext::World(holdings),
            (access @ (Access::World | Access::Clairvoyant), Reader::Viewer(_) | Reader::Auction(_)) => {
                return Err(error(format!("viewer evaluation refused a {}{}", access.noun(), access.fence())));
            }
            (Access::Viewer, Reader::Auction(frame)) => PredicateContext::Auction(frame),
        };
        f(self.predicate.as_ref(), context, args, budget)
    }
}

impl Reader<'_> {
    /// The count points on `tile`, from the frame's tables.
    fn count(&self, tile: Domino) -> u64 {
        u64::from(match self {
            Reader::Viewer(frame) => frame.physics().count(tile),
            Reader::World(holdings) => holdings.frame().physics().count(tile),
            Reader::Auction(frame) => frame.count(tile),
        })
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
            Clause::Scope(scope) => scope.binder.reads.clone(),
            Clause::AtMost(bound) => bound.reads.clone(),
        }
    }

    fn ready(&self, values: &[Option<Value>]) -> bool {
        match self {
            Clause::Atom(atom) => atom.terms.iter().all(|t| match t {
                Bound::Variable(v) => values[*v].is_some(),
                Bound::Literal(_) => true,
            }),
            Clause::Scope(scope) => scope.binder.reads.iter().all(|v| values[*v].is_some()),
            Clause::AtMost(bound) => bound.reads.iter().all(|v| values[*v].is_some()),
        }
    }

    fn evaluate(
        &self,
        cx: Cx<'_>,
        sorts: &[Sort],
        values: &mut [Option<Value>],
        budget: &mut Budget,
    ) -> Result<Option<bool>> {
        match self {
            Clause::Atom(atom) => atom.evaluate(cx, values, budget),
            Clause::Scope(scope) => scope.evaluate(cx, sorts, values, budget),
            Clause::AtMost(bound) => bound.evaluate(cx, sorts, values, budget),
        }
    }

    /// Whether the clause holds: an undefined clause does not.
    fn holds(&self, cx: Cx<'_>, sorts: &[Sort], values: &mut [Option<Value>], budget: &mut Budget) -> Result<bool> {
        Ok(self.evaluate(cx, sorts, values, budget)? == Some(true))
    }
}

impl CompiledScope {
    /// Kleene's existential over the clause's own roles, negated for `none`.
    /// A binding is satisfied when every body clause holds, refuted when some
    /// body clause fails, and unsettled otherwise. `some` is true when some
    /// binding is satisfied, false when every binding is refuted, undefined
    /// otherwise; `none` is its negation, undefined where it is.
    fn evaluate(
        &self,
        cx: Cx<'_>,
        sorts: &[Sort],
        values: &mut [Option<Value>],
        budget: &mut Budget,
    ) -> Result<Option<bool>> {
        let (mut found, mut undefined) = (false, false);
        self.binder.search(cx, sorts, values, budget, &mut |unsure, _| {
            if unsure {
                undefined = true;
                false
            } else {
                found = true;
                true
            }
        })?;
        let exists = if found {
            Some(true)
        } else if undefined {
            None
        } else {
            Some(false)
        };
        Ok(match self.quantifier {
            Quantifier::Some => exists,
            Quantifier::None => exists.map(|e| !e),
        })
    }
}

impl Binder {
    /// Visit every binding of the locals, distinct from one another, that no
    /// body clause refutes: `visit(unsure, values)`, `unsure` when some
    /// clause was undefined, until `visit` says stop. Clauses are checked as
    /// soon as their roles are bound (the schedule), so a refuted clause
    /// prunes every extension. The locals are unbound again on return.
    fn search(
        &self,
        cx: Cx<'_>,
        sorts: &[Sort],
        values: &mut [Option<Value>],
        budget: &mut Budget,
        visit: &mut dyn FnMut(bool, &[Option<Value>]) -> bool,
    ) -> Result<()> {
        let result = self.extend(0, cx, sorts, values, false, budget, visit);
        for v in &self.locals {
            values[*v] = None;
        }
        result.map(drop)
    }

    /// Whether `visit` said stop below this binding so far.
    #[allow(clippy::too_many_arguments)]
    fn extend(
        &self,
        depth: usize,
        cx: Cx<'_>,
        sorts: &[Sort],
        values: &mut [Option<Value>],
        mut unsure: bool,
        budget: &mut Budget,
        visit: &mut dyn FnMut(bool, &[Option<Value>]) -> bool,
    ) -> Result<bool> {
        budget.spend(1)?;
        for i in &self.schedule[depth] {
            let clause = &self.body[*i];
            debug_assert!(clause.ready(values), "a scheduled clause is ready");
            match clause.evaluate(cx, sorts, values, budget)? {
                Some(true) => {}
                Some(false) => return Ok(false),
                None => unsure = true,
            }
        }
        if depth == self.locals.len() {
            return Ok(visit(unsure, values));
        }
        let v = self.locals[depth];
        for candidate in sorts[v].values().iter().copied() {
            // The roles of one scope are distinct from one another, not from
            // the roles around it.
            if self.locals[..depth].iter().any(|u| values[*u] == Some(candidate)) {
                continue;
            }
            values[v] = Some(candidate);
            if self.extend(depth + 1, cx, sorts, values, unsure, budget, visit)? {
                return Ok(true);
            }
        }
        values[v] = None;
        Ok(false)
    }
}

impl CompiledAggregate {
    /// The aggregate's interval: the weight of the satisfied bindings, and
    /// that plus the weight of the unsettled ones. Each binding weighs one,
    /// or the count of its domino for `points`. Once the satisfied weight
    /// reaches `cap` the count stops: past it the caller's comparison is
    /// decided.
    fn interval(
        &self,
        cap: Option<u64>,
        cx: Cx<'_>,
        sorts: &[Sort],
        values: &mut [Option<Value>],
        budget: &mut Budget,
    ) -> Result<(u64, u64)> {
        let key = (self.id, self.free.iter().map(|v| values[*v].expect("a read role is bound")).collect());
        if let Some(&(low, high, whole)) = cx.memo.borrow().get(&key)
            && (whole || cap.is_some_and(|c| low >= c))
        {
            return Ok((low, high));
        }
        let (mut satisfied, mut unsettled, mut stopped) = (0u64, 0u64, false);
        let points = self.points;
        self.binder.search(cx, sorts, values, budget, &mut |unsure, values| {
            let weight = match points.map(|class| values[class]) {
                Some(Some(Value::Domino(tile))) => cx.reader.count(tile),
                Some(_) => unreachable!("points sums a domino role"),
                None => 1,
            };
            if unsure {
                unsettled += weight;
            } else {
                satisfied += weight;
            }
            stopped = cap.is_some_and(|c| satisfied >= c);
            stopped
        })?;
        cx.memo.borrow_mut().insert(key, (satisfied, satisfied + unsettled, !stopped));
        Ok((satisfied, satisfied + unsettled))
    }
}

impl CompiledQuantity {
    /// Whether a function's number is part of it.
    fn reads_function(&self) -> bool {
        match self {
            CompiledQuantity::Function(_) => true,
            CompiledQuantity::Sum(parts) | CompiledQuantity::Difference(parts) => {
                parts.iter().any(CompiledQuantity::reads_function)
            }
            CompiledQuantity::Number(_) | CompiledQuantity::Aggregate(_) => false,
        }
    }

    /// Whether a difference is part of it: then it may fall as a part
    /// grows, and no part may stop counting at a cap.
    fn has_difference(&self) -> bool {
        match self {
            CompiledQuantity::Difference(_) => true,
            CompiledQuantity::Sum(parts) => parts.iter().any(CompiledQuantity::has_difference),
            CompiledQuantity::Number(_) | CompiledQuantity::Function(_) | CompiledQuantity::Aggregate(_) => false,
        }
    }

    /// The interval of values the quantity takes at this binding: exact
    /// unless an aggregate's body is undefined at some binding. With `cap`,
    /// the low end may stop at the cap once it reaches it (only where no
    /// difference is part of it). `None` when a function in it has no number
    /// here (undefined, or holding for none): a comparison that reads it is
    /// undefined.
    fn interval(
        &self,
        cap: Option<u64>,
        cx: Cx<'_>,
        sorts: &[Sort],
        values: &mut [Option<Value>],
        budget: &mut Budget,
    ) -> Result<Option<(i64, i64)>> {
        match self {
            CompiledQuantity::Number(n) => Ok(Some((signed(*n), signed(*n)))),
            CompiledQuantity::Function(f) => Ok(match f.value(cx, values, budget)? {
                NumberAt::Is(n) => Some((i64::from(n), i64::from(n))),
                NumberAt::Absent | NumberAt::Undefined => None,
            }),
            CompiledQuantity::Aggregate(aggregate) => {
                let (low, high) = aggregate.interval(cap, cx, sorts, values, budget)?;
                Ok(Some((signed(low), signed(high))))
            }
            CompiledQuantity::Difference(parts) => {
                // a - b over intervals: the least a less the most b, to the
                // most a less the least b.
                let Some((mut low, mut high)) = parts[0].interval(None, cx, sorts, values, budget)? else {
                    return Ok(None);
                };
                for part in &parts[1..] {
                    let Some((l, h)) = part.interval(None, cx, sorts, values, budget)? else {
                        return Ok(None);
                    };
                    (low, high) = (low - h, high - l);
                }
                Ok(Some((low, high)))
            }
            CompiledQuantity::Sum(parts) => {
                let (mut low, mut high) = (0, 0);
                for (i, part) in parts.iter().enumerate() {
                    let need = cap.map(|c| c.saturating_sub(u64::try_from(low).unwrap_or(0)));
                    let Some((l, h)) = part.interval(need, cx, sorts, values, budget)? else {
                        return Ok(None);
                    };
                    (low, high) = (low + l, high + h);
                    // With a cap no part is a difference, so every part is
                    // nonnegative: each counts only to what the sum still
                    // needs to reach the cap, and once there the sum is
                    // decided, unless a function after it has no number,
                    // which leaves the sum none.
                    if cap.is_some_and(|c| u64::try_from(low).is_ok_and(|low| low >= c))
                        && !parts[i + 1..].iter().any(CompiledQuantity::reads_function)
                    {
                        break;
                    }
                }
                Ok(Some((low, high)))
            }
        }
    }
}

/// A count or a literal as a signed number: differences may go below zero.
fn signed(n: u64) -> i64 {
    i64::try_from(n).expect("a number within i64")
}

impl CompiledAtMost {
    /// `left <= right` over the two intervals: true when every value of the
    /// left is at most every value of the right, false when none is,
    /// undefined otherwise. Against a literal the aggregate stops counting
    /// once that is settled.
    fn evaluate(
        &self,
        cx: Cx<'_>,
        sorts: &[Sort],
        values: &mut [Option<Value>],
        budget: &mut Budget,
    ) -> Result<Option<bool>> {
        // A side that holds a difference is counted whole: a part of it
        // past a cap could still fall.
        let capped = !self.left.has_difference() && !self.right.has_difference();
        let (left, right) = match (&self.left, &self.right) {
            (left, CompiledQuantity::Number(n)) if capped => {
                (left.interval(Some(n + 1), cx, sorts, values, budget)?, Some((signed(*n), signed(*n))))
            }
            (CompiledQuantity::Number(n), right) if capped => {
                (Some((signed(*n), signed(*n))), right.interval(Some(*n), cx, sorts, values, budget)?)
            }
            (left, right) => {
                (left.interval(None, cx, sorts, values, budget)?, right.interval(None, cx, sorts, values, budget)?)
            }
        };
        let (Some((left_low, left_high)), Some((right_low, right_high))) = (left, right) else {
            return Ok(None);
        };
        Ok(if left_high <= right_low {
            Some(true)
        } else if left_low > right_high {
            Some(false)
        } else {
            None
        })
    }
}

/// A number read at a query's answers: compiled over the roles of its
/// outputs, which an answer binds, so it is a function of the answer. A
/// computed call's number (README, The auction).
#[derive(Clone)]
pub struct CompiledNumber {
    source: Quantity,
    /// The outputs' classes first, then each aggregate's own.
    sorts: Vec<Sort>,
    quantity: CompiledQuantity,
    specs: Vec<PredicateSpec>,
}

impl CompiledNumber {
    /// Compile `quantity` over `over`, the roles an answer binds, in order.
    /// Every role it names must be one of them.
    pub fn compile(quantity: &Quantity, over: &[Role], span: &Span, registry: &Registry) -> Result<CompiledNumber> {
        let mut compiler = Compiler { registry, sorts: Vec::new(), roles: 0, used: BTreeMap::new() };
        let classes = compiler.declare(over, &[], &Scope::new(), span)?;
        let scope: Scope = over.iter().zip(&classes).map(|(role, c)| (role.name.clone(), (*c, role.sort))).collect();
        if let Some(name) = crate::syntax::quantity_names(quantity).into_iter().find(|n| !scope.contains_key(n)) {
            return Err(at(span, format!("the number reads {name}, which is not an answer of its guard")));
        }
        let compiled = compiler.quantity(quantity, &scope, span)?;
        Ok(CompiledNumber {
            source: quantity.clone(),
            sorts: compiler.sorts,
            quantity: compiled,
            specs: compiler.used.into_values().collect(),
        })
    }

    pub fn source(&self) -> &Quantity {
        &self.source
    }

    pub fn predicate_specs(&self) -> &[PredicateSpec] {
        &self.specs
    }

    /// Number its aggregates by their texts in `table`, with the queries it
    /// is evaluated alongside.
    pub(crate) fn intern(&mut self, table: &mut BTreeMap<String, usize>) {
        let mut clause = [Clause::AtMost(Box::new(CompiledAtMost {
            left: self.quantity.clone(),
            right: CompiledQuantity::Number(0),
            reads: BTreeSet::new(),
        }))];
        intern(&mut clause, table);
        let [Clause::AtMost(bound)] = clause else { unreachable!("one bound") };
        self.quantity = bound.left;
    }

    /// The number's interval at a seat's view of the auction, the outputs
    /// bound to `answer`; `None` where a function in it has no number.
    pub(crate) fn at_auction(
        &self,
        frame: &AuctionFrame,
        answer: &[Value],
        memo: &Memo,
        budget: &mut Budget,
    ) -> Result<Option<(i64, i64)>> {
        let cx = Cx { reader: Reader::Auction(frame), memo };
        let mut values = vec![None; self.sorts.len()];
        for (slot, value) in values.iter_mut().zip(answer) {
            *slot = Some(*value);
        }
        self.quantity.interval(None, cx, &self.sorts, &mut values, budget)
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

    /// The phase every predicate the query names is defined in: the play,
    /// the auction, or both. Refused when it names one of each.
    pub fn phase(&self) -> Result<Phase> {
        let reads = |at: Phase| self.specs.iter().all(|spec| spec.phase.reads(at));
        match (reads(Phase::Play), reads(Phase::Auction)) {
            (true, true) => Ok(Phase::Both),
            (true, false) => Ok(Phase::Play),
            (false, true) => Ok(Phase::Auction),
            (false, false) => Err(error("the query names predicates of the play and of the auction together")),
        }
    }

    /// Refused, before any work, when the query names a predicate not
    /// defined at a frame of `at`, or (`viewer`) a `World` predicate.
    fn admits(&self, at: Phase, viewer: bool) -> Result<()> {
        if let Some(spec) = self.specs.iter().find(|spec| !spec.phase.reads(at)) {
            return Err(error(format!(
                "{} is a predicate of {}, and is not defined at a frame of {at}",
                spec.name, spec.phase
            )));
        }
        if viewer && let Some(spec) = self.specs.iter().find(|spec| spec.access.reads_world()) {
            let access = spec.access;
            return Err(error(format!(
                "viewer evaluation refused the {} {}{}",
                access.noun(),
                spec.name,
                access.fence()
            )));
        }
        Ok(())
    }

    /// The answers in one world consistent with `frame`.
    pub fn evaluate(&self, frame: &Frame, world: &World, budget: &mut Budget) -> Result<Answers> {
        if !frame.contains(world) {
            return Err(error("the world is outside this frame's support"));
        }
        self.evaluate_in(Holdings::new(frame, world, DominoSet::EMPTY), budget)
    }

    pub(crate) fn evaluate_in(&self, holdings: Holdings<'_>, budget: &mut Budget) -> Result<Answers> {
        self.admits(Phase::Play, false)?;
        self.run(Reader::World(holdings), budget)
    }

    /// Evaluate a query that reads no world. Refused, before any work, if it
    /// names a `World` predicate or one of the auction alone.
    pub fn evaluate_viewer(&self, frame: &Frame, budget: &mut Budget) -> Result<Answers> {
        self.evaluate_viewer_with(frame, &Memo::default(), budget)
    }

    /// [`Self::evaluate_viewer`], counting each aggregate once with every
    /// query evaluated at `frame` under `memo` (queries interned together).
    pub(crate) fn evaluate_viewer_with(&self, frame: &Frame, memo: &Memo, budget: &mut Budget) -> Result<Answers> {
        self.admits(Phase::Play, true)?;
        self.run_with(Reader::Viewer(frame), memo, budget)
    }

    /// Evaluate a query at a seat's view of the auction. Refused, before any
    /// work, if it names a `World` predicate or one of the play alone.
    pub fn evaluate_auction(&self, frame: &AuctionFrame, budget: &mut Budget) -> Result<Answers> {
        self.evaluate_auction_with(frame, &Memo::default(), budget)
    }

    /// [`Self::evaluate_auction`] under a shared `memo`, as
    /// [`Self::evaluate_viewer_with`].
    pub(crate) fn evaluate_auction_with(
        &self,
        frame: &AuctionFrame,
        memo: &Memo,
        budget: &mut Budget,
    ) -> Result<Answers> {
        self.admits(Phase::Auction, true)?;
        self.run_with(Reader::Auction(frame), memo, budget)
    }

    /// Number this query's aggregates by their texts in `table`, shared with
    /// the other queries that will be evaluated under one memo: equal texts,
    /// one number.
    pub(crate) fn intern(&mut self, table: &mut BTreeMap<String, usize>) {
        for pattern in &mut self.patterns {
            intern(&mut pattern.clauses, table);
        }
    }

    /// The answers at a frame, as [`Self::evaluate_viewer`], each with one
    /// witness: a binding of every role of the query (outputs and existential
    /// roles alike) that satisfies a pattern. The first found is shown.
    pub fn witnesses(&self, frame: &Frame, budget: &mut Budget) -> Result<Vec<(Answer, Witness)>> {
        self.admits(Phase::Play, true)?;
        self.witnessed(Reader::Viewer(frame), budget)
    }

    /// [`Self::witnesses`] at a seat's view of the auction.
    pub fn witnesses_at_auction(&self, frame: &AuctionFrame, budget: &mut Budget) -> Result<Vec<(Answer, Witness)>> {
        self.admits(Phase::Auction, true)?;
        self.witnessed(Reader::Auction(frame), budget)
    }

    fn witnessed(&self, reader: Reader<'_>, budget: &mut Budget) -> Result<Vec<(Answer, Witness)>> {
        let mut witnesses = BTreeMap::new();
        budget.spend(1)?;
        let memo = Memo::default();
        let cx = Cx { reader, memo: &memo };
        let mut answers = Answers::new();
        for pattern in &self.patterns {
            pattern.evaluate(cx, budget, &mut answers, Some(&mut witnesses))?;
        }
        Ok(witnesses.into_iter().collect())
    }

    fn run(&self, reader: Reader<'_>, budget: &mut Budget) -> Result<Answers> {
        self.run_with(reader, &Memo::default(), budget)
    }

    fn run_with(&self, reader: Reader<'_>, memo: &Memo, budget: &mut Budget) -> Result<Answers> {
        budget.spend(1)?;
        let cx = Cx { reader, memo };
        let mut answers = Answers::new();
        for pattern in &self.patterns {
            pattern.evaluate(cx, budget, &mut answers, None)?;
        }
        Ok(answers)
    }
}

/// A satisfying binding of a query's roles, by name.
pub type Witness = Vec<(String, Value)>;

impl CompiledPattern {
    fn evaluate(
        &self,
        cx: Cx<'_>,
        budget: &mut Budget,
        answers: &mut Answers,
        witnesses: Option<&mut BTreeMap<Answer, Witness>>,
    ) -> Result<()> {
        let mut values = vec![None; self.sorts.len()];
        // Constant clauses are checked once; single-variable clauses shrink
        // their variable's domain before the join; the rest are checked as
        // soon as every role they read is bound. A conjunction's order is
        // not its meaning: atoms first, then comparisons, then scopes, the
        // likely cheaper first, so a scope runs only where the rest hold.
        let mut constant = Vec::new();
        let mut unary = vec![Vec::new(); self.outer];
        let mut joined = Vec::new();
        let rank = |c: &Clause| match c {
            Clause::Atom(_) => 0,
            Clause::AtMost(_) => 1,
            Clause::Scope(_) => 2,
        };
        let mut order: Vec<usize> = (0..self.clauses.len()).collect();
        order.sort_by_key(|i| rank(&self.clauses[*i]));
        for (i, clause) in order.into_iter().map(|i| (i, &self.clauses[i])) {
            let vars = clause.vars();
            match (vars.len(), vars.first()) {
                (0, _) => constant.push(i),
                (1, Some(v)) => unary[*v].push(i),
                _ => joined.push(i),
            }
        }
        for i in constant {
            if !self.clauses[i].holds(cx, &self.sorts, &mut values, budget)? {
                return Ok(());
            }
        }
        let mut domains = Vec::new();
        for v in 0..self.outer {
            let mut domain = Vec::new();
            for candidate in self.sorts[v].values().iter().copied() {
                budget.spend(1)?;
                values[v] = Some(candidate);
                let mut admitted = true;
                for c in &unary[v] {
                    if !self.clauses[*c].holds(cx, &self.sorts, &mut values, budget)? {
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
        Search { pattern: self, cx, budget, answers, witnesses, domains: &domains, order: &order }.walk(
            0,
            &mut values,
            &joined,
        )
    }
}

struct Search<'a, 's> {
    pattern: &'a CompiledPattern,
    cx: Cx<'s>,
    budget: &'a mut Budget,
    answers: &'a mut Answers,
    witnesses: Option<&'a mut BTreeMap<Answer, Witness>>,
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
            } else if !clause.holds(self.cx, &self.pattern.sorts, values, self.budget)? {
                return Ok(());
            }
        }
        if depth == self.order.len() {
            let row = Answer(self.pattern.outputs.iter().map(|v| values[*v].expect("a complete binding")).collect());
            if let Some(witnesses) = self.witnesses.as_deref_mut() {
                let binding = self.pattern.names.iter().map(|(n, c)| (n.clone(), values[*c].expect("bound"))).collect();
                witnesses.entry(row.clone()).or_insert(binding);
            }
            self.answers.insert(row);
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
