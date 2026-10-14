//! The surface sugar: shorter forms that rewrite, tree to tree, into the core
//! forms before the core reader sees them. Nothing here adds meaning. Each
//! rewrite is an equation of the semantics (Kleene's De Morgan laws, `none`
//! as the negation of `some`), and a sugared text reads to exactly the core
//! its rewrite spells: the same `Query`, printed the same.
//!
//! - `(and c ...)` is its clauses, spliced into the body around it.
//! - `(or c ...)` is `(none (roles) (not c) ...)`: no disjunct holds is false.
//! - `(not c)` negates any clause: `(not (not c))` is `c`; `(not (some R B))`
//!   is `(none R B)` and `(not (none R B))` is `(some R B)`; `(not (and c
//!   ...))` is `(none (roles) c ...)`; `(not (or c ...))` is `(not c) ...`. An
//!   atom or a definition call stays `(not (p ...))`, the core's own negation.
//! - `(least t q c ...)` says `t` is a least candidate by the number `q`: it
//!   is `c ...` and `(none (roles u) c' ... (< q' q))`, where `u` is a name
//!   fresh to the form and to the names around it, and `c'` and `q'` are `c`
//!   and `q` with `u` for `t`. `(most t q c ...)` is the same with `>`.
//! - In any `(roles ...)`, a bare name takes its sort from its uses
//!   ([`Uses`]); a query may omit `(roles ...)`, `(out ...)` and `(pattern
//!   ...)`, and a policy rule its query (`syntax.rs` and `policy.rs` apply
//!   these, with the definitions in force).

use std::collections::{BTreeMap, BTreeSet};

use crate::Result;
use crate::syntax::{Expr, Kind, Sort, Span, Value};

/// Heads that are clause forms of the core or the sugar, not atoms.
const FORMS: [&str; 14] =
    ["same", "some", "none", "not", "and", "or", "least", "most", "<=", "<", ">=", ">", "=", "/="];

/// Whether a name is taken around a clause: a fresh name avoids it.
pub(crate) type Taken<'a> = &'a dyn Fn(&str) -> bool;

/// The comparisons read into the core's `(<= a b)`, each with the
/// comparison that negates it.
const COMPARISONS: [(&str, &str); 6] = [("<=", ">"), ("<", ">="), (">=", "<"), (">", "<="), ("=", "/="), ("/=", "=")];

/// `(head a b)`'s two operands.
fn operands<'a>(x: &'a Expr, head: &str) -> Result<(&'a Expr, &'a Expr)> {
    match x.list()? {
        [_, a, b] => Ok((a, b)),
        _ => Err(x.span.error(format!("{head} compares two numbers: ({head} a b)"))),
    }
}

/// `q + 1`: a literal folded, anything else `(+ q 1)`.
fn plus_one(q: &Expr) -> Expr {
    match q.word().ok().and_then(|w| w.parse::<u32>().ok()).and_then(|n| n.checked_add(1)) {
        Some(n) => Expr::word_at(&n.to_string(), &q.span),
        None => Expr::list_at(vec![Expr::word_at("+", &q.span), q.clone(), Expr::word_at("1", &q.span)], &q.span),
    }
}

/// `(<= a b)` at `x`'s place.
fn at_most(a: Expr, b: Expr, x: &Expr) -> Expr {
    Expr::list_at(vec![Expr::word_at("<=", &x.span), a, b], &x.span)
}

/// A comparison's short form as core `(<= ...)` clauses (and, for `/=`, the
/// `or` it is): over the integers, `a < b` is `a + 1 <= b`, `a >= b` is `b <=
/// a`, `a > b` is `b + 1 <= a`, `a = b` is both `a <= b` and `b <= a`, and `a
/// /= b` is `a < b` or `a > b`.
fn comparison(x: &Expr, head: &str, taken: Taken<'_>, out: &mut Vec<Expr>) -> Result<()> {
    let (a, b) = operands(x, head)?;
    let (a, b) = (a.clone(), b.clone());
    match head {
        "<=" => out.push(x.clone()),
        "<" => out.push(at_most(plus_one(&a), b, x)),
        ">=" => out.push(at_most(b, a, x)),
        ">" => out.push(at_most(plus_one(&b), a, x)),
        "=" => {
            out.push(at_most(a.clone(), b.clone(), x));
            out.push(at_most(b, a, x));
        }
        _ => {
            let less = Expr::list_at(vec![Expr::word_at("<", &x.span), a.clone(), b.clone()], &x.span);
            let more = Expr::list_at(vec![Expr::word_at(">", &x.span), a, b], &x.span);
            clause(&Expr::list_at(vec![Expr::word_at("or", &x.span), less, more], &x.span), taken, out)?;
        }
    }
    Ok(())
}

/// A `(roles ...)` whose bare names are dominoes: an aggregate summing count
/// points binds one.
pub(crate) fn dominoes(declared: &Expr) -> Result<Expr> {
    let entries = declared.list()?;
    let mut out = vec![entries[0].clone()];
    for entry in &entries[1..] {
        out.push(match &entry.kind {
            Kind::Word(_) => Expr::list_at(vec![Expr::word_at("domino", &entry.span), entry.clone()], &entry.span),
            _ => entry.clone(),
        });
    }
    Ok(Expr::list_at(out, &declared.span))
}

/// A clause list with every `and`, `or` and compound `not` rewritten, at
/// this level (a scope's body is rewritten when it is read).
pub(crate) fn clauses(xs: &[Expr], taken: Taken<'_>) -> Result<Vec<Expr>> {
    let mut out = Vec::new();
    for x in xs {
        clause(x, taken, &mut out)?;
    }
    Ok(out)
}

fn clause(x: &Expr, taken: Taken<'_>, out: &mut Vec<Expr>) -> Result<()> {
    match x.head() {
        Some("and") => {
            for c in &x.list()?[1..] {
                clause(c, taken, out)?;
            }
        }
        Some("or") => {
            let mut body = vec![Expr::list_at(vec![Expr::word_at("roles", &x.span)], &x.span)];
            for c in &x.list()?[1..] {
                negation(c, taken, &mut body)?;
            }
            out.push(Expr::list_at([vec![Expr::word_at("none", &x.span)], body].concat(), &x.span));
        }
        Some("not") => match x.list()? {
            [_, c] if c.head().is_some_and(|h| FORMS.contains(&h)) => {
                let mut negated = Vec::new();
                negation(c, taken, &mut negated)?;
                for n in &negated {
                    clause(n, taken, out)?;
                }
            }
            _ => out.push(x.clone()),
        },
        Some(head @ ("least" | "most")) => clause(&extremum(x, head, taken)?, taken, out)?,
        Some(head) if COMPARISONS.iter().any(|(c, _)| *c == head) => comparison(x, head, taken, out)?,
        _ => out.push(x.clone()),
    }
    Ok(())
}

/// `(least t q c ...)` as the `and` it is: `t` meets `c ...`, and no
/// candidate `u` that meets them has a smaller `q` (a larger, for `most`).
/// `u` is fresh: no word of the form and no name taken around it.
fn extremum(x: &Expr, head: &str, taken: Taken<'_>) -> Result<Expr> {
    let [_, role, quantity, candidates @ ..] = x.list()? else {
        return Err(x.span.error(format!("{head} syntax is ({head} role number clause ...)")));
    };
    let name = role.word()?;
    if Value::literal(name).is_some() {
        return Err(role.span.error(format!("{head} ranges a role, and {name} is a literal")));
    }
    let mut words = BTreeSet::new();
    collect_words(x, &mut words);
    let other = (2..).map(|k| format!("{name}-{k}")).find(|n| !words.contains(n) && !taken(n)).expect("an unused name");
    let at = |w: &str| Expr::word_at(w, &x.span);
    let compare = if head == "least" { "<" } else { ">" };
    let mut rival = vec![at("none"), Expr::list_at(vec![at("roles"), at(&other)], &x.span)];
    rival.extend(candidates.iter().map(|c| substitute(c, name, &other)));
    rival.push(Expr::list_at(vec![at(compare), substitute(quantity, name, &other), quantity.clone()], &x.span));
    let mut and = vec![at("and")];
    and.extend(candidates.iter().cloned());
    and.push(Expr::list_at(rival, &x.span));
    Ok(Expr::list_at(and, &x.span))
}

/// Every word of `x`.
fn collect_words(x: &Expr, out: &mut BTreeSet<String>) {
    match &x.kind {
        Kind::Word(w) => {
            out.insert(w.clone());
        }
        Kind::List(xs) => xs.iter().for_each(|x| collect_words(x, out)),
        Kind::Comment(_) => {}
    }
}

/// `x` with the word `from` read as `to` throughout. A form that declared
/// `from` again inside would already be refused around it (a scope's role
/// may not repeat a name in force), so no binding is crossed.
fn substitute(x: &Expr, from: &str, to: &str) -> Expr {
    match &x.kind {
        Kind::Word(w) if w == from => Expr::word_at(to, &x.span),
        Kind::List(xs) => Expr::list_at(xs.iter().map(|x| substitute(x, from, to)).collect(), &x.span),
        _ => x.clone(),
    }
}

/// The clauses whose conjunction is the negation of `c`, still to be read
/// as clauses (they may hold sugar of their own).
fn negation(c: &Expr, taken: Taken<'_>, out: &mut Vec<Expr>) -> Result<()> {
    let xs = c.list()?;
    let renamed = |tag: &str| {
        let mut items = xs.to_vec();
        items[0] = Expr::word_at(tag, &xs[0].span);
        Expr::list_at(items, &c.span)
    };
    match c.head() {
        Some("not") => match xs {
            [_, inner] => out.push(inner.clone()),
            _ => return Err(c.span.error("not takes exactly one clause")),
        },
        Some("some") => out.push(renamed("none")),
        Some("none") => out.push(renamed("some")),
        Some("and") => {
            let roles = Expr::list_at(vec![Expr::word_at("roles", &c.span)], &c.span);
            out.push(Expr::list_at([vec![Expr::word_at("none", &c.span), roles], xs[1..].to_vec()].concat(), &c.span));
        }
        Some("or") => {
            for d in &xs[1..] {
                negation(d, taken, out)?;
            }
        }
        Some(head @ ("least" | "most")) => negation(&extremum(c, head, taken)?, taken, out)?,
        Some("same") => {
            return Err(c.span.error("same cannot be negated: roles declared together are already distinct"));
        }
        Some(head) if COMPARISONS.iter().any(|(c, _)| *c == head) => {
            let (_, opposite) = COMPARISONS.iter().find(|(c, _)| *c == head).expect("a comparison");
            operands(c, head)?;
            out.push(renamed(opposite));
        }
        _ => out.push(Expr::list_at(vec![Expr::word_at("not", &c.span), c.clone()], &c.span)),
    }
    Ok(())
}

/// The free names of a body, in order of first use, each with the sorts its
/// uses give it: an argument takes its predicate's or definition's parameter
/// sort, and a `same` joins its names' sorts. A name bound by a scope inside
/// the body is that scope's, not the body's.
#[derive(Default)]
pub(crate) struct Uses {
    order: Vec<String>,
    sorts: BTreeMap<String, Vec<(Sort, Span)>>,
    joined: Vec<Vec<String>>,
}

impl Uses {
    /// Walk `xs`, with `bound` the names scopes around them bind, reading
    /// each atom's parameter sorts from `signature`.
    pub(crate) fn walk(
        &mut self,
        xs: &[Expr],
        bound: &[String],
        signature: &dyn Fn(&str) -> Option<Vec<Sort>>,
    ) -> Result<()> {
        for x in clauses(xs, &|n| bound.iter().any(|b| b == n))? {
            let items = x.list()?;
            match x.head() {
                Some("same") => {
                    let mut group = Vec::new();
                    for n in &items[1..] {
                        let name = n.word()?;
                        if !bound.iter().any(|b| b == name) {
                            self.seen(name);
                            group.push(name.to_owned());
                        }
                    }
                    self.joined.push(group);
                }
                Some("some" | "none") => self.scope(items, bound, signature)?,
                Some("<=") => {
                    for q in &items[1..] {
                        self.quantity(q, bound, signature)?;
                    }
                }
                Some("not") => {
                    if let [_, atom] = items {
                        self.atom(atom, bound, signature)?;
                    }
                }
                _ => self.atom(&x, bound, signature)?,
            }
        }
        Ok(())
    }

    /// `(some|none|card|points (roles ...) clause ...)`: its body, with its
    /// roles bound.
    fn scope(&mut self, items: &[Expr], bound: &[String], signature: &dyn Fn(&str) -> Option<Vec<Sort>>) -> Result<()> {
        let Some(declared) = items.get(1) else { return Ok(()) };
        let mut inner = bound.to_vec();
        for role in declared.tagged("roles")? {
            inner.push(match &role.kind {
                Kind::List(pair) if pair.len() == 2 => pair[1].word()?.to_owned(),
                _ => role.word()?.to_owned(),
            });
        }
        self.walk(&items[2..], &inner, signature)
    }

    /// A comparison's operand: a name standing for a number is a use of the
    /// number sort; an aggregate's body is walked as a scope's.
    fn quantity(&mut self, q: &Expr, bound: &[String], signature: &dyn Fn(&str) -> Option<Vec<Sort>>) -> Result<()> {
        match (&q.kind, q.head()) {
            (Kind::Word(name), _) if Value::literal(name).is_none() && !bound.iter().any(|b| b == name) => {
                self.seen(name);
                self.sorts.get_mut(name.as_str()).expect("seen").push((Sort::Number, q.span.clone()));
            }
            (Kind::List(items), Some("+")) => {
                for p in &items[1..] {
                    self.quantity(p, bound, signature)?;
                }
            }
            (Kind::List(items), Some("card" | "points")) => self.scope(items, bound, signature)?,
            _ => {}
        }
        Ok(())
    }

    fn atom(&mut self, atom: &Expr, bound: &[String], signature: &dyn Fn(&str) -> Option<Vec<Sort>>) -> Result<()> {
        let items = atom.list()?;
        let Some(head) = atom.head() else { return Ok(()) };
        let sorts = signature(head).filter(|s| s.len() + 1 == items.len());
        for (i, arg) in items[1..].iter().enumerate() {
            let Ok(name) = arg.word() else { continue };
            if Value::literal(name).is_some() || bound.iter().any(|b| b == name) {
                continue;
            }
            self.seen(name);
            if let Some(sort) = sorts.as_ref().map(|s| s[i]) {
                self.sorts.get_mut(name).expect("seen").push((sort, arg.span.clone()));
            }
        }
        Ok(())
    }

    fn seen(&mut self, name: &str) {
        if !self.sorts.contains_key(name) {
            self.order.push(name.to_owned());
            self.sorts.insert(name.to_owned(), Vec::new());
        }
    }

    /// The free names, in order of first use.
    pub(crate) fn names(&self) -> &[String] {
        &self.order
    }

    /// The sort of `name` from its uses, or an error at `at` saying why
    /// there is none: no use gives one, two give different ones, or the
    /// slot takes a literal only.
    pub(crate) fn sort(&self, name: &str, at: &Span, literal_sorts: bool) -> Result<Sort> {
        // Every name a `same` joins to this one, transitively.
        let mut class = vec![name.to_owned()];
        let mut grown = true;
        while grown {
            grown = false;
            for group in &self.joined {
                if group.iter().any(|n| class.contains(n)) {
                    for n in group {
                        if !class.contains(n) {
                            class.push(n.clone());
                            grown = true;
                        }
                    }
                }
            }
        }
        let uses: Vec<&(Sort, Span)> = class.iter().filter_map(|n| self.sorts.get(n)).flatten().collect();
        let Some((sort, first)) = uses.first() else {
            return Err(at.error(format!(
                "cannot infer the sort of {name}: no predicate or definition argument names it; declare it, as (domino {name})"
            )));
        };
        if let Some((other, span)) = uses.iter().find(|(s, _)| s != sort) {
            let (line, column) = first.line_column().unwrap_or((0, 0));
            return Err(span.error(format!("{name} is a {other} here and a {sort} at {line}:{column}")));
        }
        if !literal_sorts && !sort.is_role() {
            return Err(
                first.error(format!("{name} stands where a {sort} goes, and a {sort} is a literal, never a role"))
            );
        }
        Ok(*sort)
    }
}
