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
//! - In any `(roles ...)`, a bare name takes its sort from its uses
//!   ([`Uses`]); a query may omit `(roles ...)`, `(out ...)` and `(pattern
//!   ...)`, and a policy rule its query (`syntax.rs` and `policy.rs` apply
//!   these, with the definitions in force).

use std::collections::BTreeMap;

use crate::Result;
use crate::syntax::{Expr, Kind, Sort, Span, Value};

/// Heads that are clause forms of the core or the sugar, not atoms.
const FORMS: [&str; 6] = ["same", "some", "none", "not", "and", "or"];

/// A clause list with every `and`, `or` and compound `not` rewritten, at
/// this level (a scope's body is rewritten when it is read).
pub(crate) fn clauses(xs: &[Expr]) -> Result<Vec<Expr>> {
    let mut out = Vec::new();
    for x in xs {
        clause(x, &mut out)?;
    }
    Ok(out)
}

fn clause(x: &Expr, out: &mut Vec<Expr>) -> Result<()> {
    match x.head() {
        Some("and") => {
            for c in &x.list()?[1..] {
                clause(c, out)?;
            }
        }
        Some("or") => {
            let mut body = vec![Expr::list_at(vec![Expr::word_at("roles", &x.span)], &x.span)];
            for c in &x.list()?[1..] {
                negation(c, &mut body)?;
            }
            out.push(Expr::list_at([vec![Expr::word_at("none", &x.span)], body].concat(), &x.span));
        }
        Some("not") => match x.list()? {
            [_, c] if c.head().is_some_and(|h| FORMS.contains(&h)) => {
                let mut negated = Vec::new();
                negation(c, &mut negated)?;
                for n in &negated {
                    clause(n, out)?;
                }
            }
            _ => out.push(x.clone()),
        },
        _ => out.push(x.clone()),
    }
    Ok(())
}

/// The clauses whose conjunction is the negation of `c`, still to be read
/// as clauses (they may hold sugar of their own).
fn negation(c: &Expr, out: &mut Vec<Expr>) -> Result<()> {
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
                negation(d, out)?;
            }
        }
        Some("same") => {
            return Err(c.span.error("same cannot be negated: roles declared together are already distinct"));
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
        for x in clauses(xs)? {
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
                Some("some" | "none") => {
                    let Some(declared) = items.get(1) else { continue };
                    let mut inner = bound.to_vec();
                    for role in declared.tagged("roles")? {
                        inner.push(match &role.kind {
                            Kind::List(pair) if pair.len() == 2 => pair[1].word()?.to_owned(),
                            _ => role.word()?.to_owned(),
                        });
                    }
                    self.walk(&items[2..], &inner, signature)?;
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
