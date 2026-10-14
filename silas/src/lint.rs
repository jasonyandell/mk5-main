// SPDX-FileCopyrightText: Jason Yandell
// SPDX-License-Identifier: CC0-1.0

//! Warnings: what, on its face, makes a pattern never hold, a clause say
//! nothing, or a rule never fire. Each is read off the core, so a sugared
//! text and its spelled-out form warn alike. A warning is advice, not a
//! refusal: the program means what the semantics says it means.

use std::collections::{BTreeMap, BTreeSet};

use crate::policy::{PolicyProgram, Selector};
use crate::syntax::{Aggregate, Atom, Pattern, Quantifier, Quantity, Query, Role, Sort, Span, Term};

/// A located warning, shown as an error is: the source, line and column, and
/// the line with a caret under the fault.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Warning(pub String);

impl std::fmt::Display for Warning {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

fn warn(out: &mut Vec<Warning>, span: &Span, message: String) {
    out.push(Warning(span.warning(message)));
}

/// How many values a role of the sort can take.
fn domain(sort: Sort) -> usize {
    sort.values().len()
}

/// The equality classes of `roles` under the `same` groups: each role's
/// class representative.
fn classes(roles: &[Role], equal: &[Vec<String>]) -> BTreeMap<String, String> {
    let mut class: BTreeMap<String, String> = roles.iter().map(|r| (r.name.clone(), r.name.clone())).collect();
    for group in equal {
        let Some(first) = group.first().and_then(|n| class.get(n)).cloned() else { continue };
        for name in group {
            if let Some(old) = class.get(name).cloned() {
                class.values_mut().filter(|c| **c == old).for_each(|c| *c = first.clone());
            }
        }
    }
    class
}

/// Warnings about roles declared together: more of one sort than it has
/// values, and an equality atom between two of them, which distinctness
/// decides.
fn declaration(roles: &[Role], body: &Pattern, at: &Span, what: &str, out: &mut Vec<Warning>) {
    let class = classes(roles, &body.equal);
    for sort in [Sort::Chair, Sort::Context, Sort::Domino] {
        let n = roles.iter().filter(|r| r.sort == sort).map(|r| &class[&r.name]).collect::<BTreeSet<_>>().len();
        if n > domain(sort) {
            warn(
                out,
                at,
                format!(
                    "{what} needs {n} distinct {sort}s and there are {}: roles declared together are distinct, so it never holds",
                    domain(sort)
                ),
            );
        }
    }
    for atom in &body.atoms {
        if let ("tile" | "chair" | "context", [Term::Role(a), Term::Role(b)]) =
            (atom.predicate.as_str(), &atom.args[..])
            && a != b
            && let (Some(x), Some(y)) = (class.get(a), class.get(b))
            && x != y
        {
            let message = if atom.negated {
                format!("always true: {a} and {b} are declared together, so they are already distinct")
            } else {
                format!(
                    "never true: {a} and {b} are declared together, so they are distinct; join them with (same {a} {b})"
                )
            };
            warn(out, &atom.span, message);
        }
    }
}

/// Warnings within one body and the scopes inside it.
fn body(pattern: &Pattern, out: &mut Vec<Warning>) {
    for (k, atom) in pattern.atoms.iter().enumerate() {
        let opposite = |other: &&Atom| {
            other.predicate == atom.predicate && other.args == atom.args && other.negated != atom.negated
        };
        if pattern.atoms[..k].iter().any(|other| opposite(&other)) {
            warn(
                out,
                &atom.span,
                format!("never holds: this clause and its negation are asked together ({})", atom.predicate),
            );
        }
    }
    for scope in &pattern.scopes {
        let what = match scope.quantifier {
            Quantifier::Some => "this (some ...)",
            Quantifier::None => "this (none ...)",
        };
        binder(&scope.roles, &scope.body, &scope.span, what, true, out);
    }
    for (aggregate, points) in
        pattern.bounds.iter().flat_map(|b| aggregates(&b.left).into_iter().chain(aggregates(&b.right)))
    {
        let what = if points { "this (points ...)" } else { "this (card ...)" };
        // A points sum reads its domino's count: a use of its own.
        binder(&aggregate.roles, &aggregate.body, &aggregate.span, what, !points, out);
    }
}

/// Warnings about a scope's or an aggregate's own roles and its body.
fn binder(roles: &[Role], pattern: &Pattern, at: &Span, what: &str, unused: bool, out: &mut Vec<Warning>) {
    declaration(roles, pattern, at, what, out);
    for role in roles.iter().filter(|r| unused && !mentions(pattern, &r.name)) {
        warn(out, &role.span, format!("{} is bound and never used", role.name));
    }
    body(pattern, out);
}

/// The aggregates a quantity reads, each with whether it sums points.
fn aggregates(quantity: &Quantity) -> Vec<(&Aggregate, bool)> {
    match quantity {
        Quantity::Number(_) | Quantity::Parameter(_) | Quantity::Function(_) => vec![],
        Quantity::Card(a) => vec![(a, false)],
        Quantity::Points(a) => vec![(a, true)],
        Quantity::Sum(parts) | Quantity::Difference(parts) => parts.iter().flat_map(aggregates).collect(),
    }
}

/// Whether a quantity gives `name` to a function, as `(count d)` does.
fn applies(quantity: &Quantity, name: &str) -> bool {
    match quantity {
        Quantity::Function(f) => f.args.contains(&Term::Role(name.to_owned())),
        Quantity::Sum(parts) | Quantity::Difference(parts) => parts.iter().any(|p| applies(p, name)),
        Quantity::Number(_) | Quantity::Parameter(_) | Quantity::Card(_) | Quantity::Points(_) => false,
    }
}

fn mentions(pattern: &Pattern, name: &str) -> bool {
    let inside = |roles: &[Role], body: &Pattern| !roles.iter().any(|r| r.name == name) && mentions(body, name);
    pattern.equal.iter().flatten().any(|n| n == name)
        || pattern.atoms.iter().any(|a| a.args.contains(&Term::Role(name.to_owned())))
        || pattern.scopes.iter().any(|s| inside(&s.roles, &s.body))
        || pattern
            .bounds
            .iter()
            .flat_map(|b| aggregates(&b.left).into_iter().chain(aggregates(&b.right)))
            .any(|(a, _)| inside(&a.roles, &a.body))
        || pattern.bounds.iter().any(|b| applies(&b.left, name) || applies(&b.right, name))
}

/// The polarity of each use of `name` in `pattern`: true where a use is
/// positive, false where it is under an odd number of negations (a negated
/// atom, a `none`). The first span of each.
fn polarities(pattern: &Pattern, name: &str, positive: bool, uses: &mut Vec<(bool, Span)>) {
    if pattern.equal.iter().flatten().any(|n| n == name) {
        uses.push((positive, pattern.span.clone()));
    }
    for atom in &pattern.atoms {
        if atom.args.contains(&Term::Role(name.to_owned())) {
            uses.push((positive != atom.negated, atom.span.clone()));
        }
    }
    for scope in &pattern.scopes {
        if !scope.roles.iter().any(|r| r.name == name) {
            polarities(&scope.body, name, positive == (scope.quantifier == Quantifier::Some), uses);
        }
    }
    // A comparison counts its aggregates' bindings up on one side and down
    // on the other: a use inside one is not a negation.
    for bound in &pattern.bounds {
        let inside = aggregates(&bound.left)
            .into_iter()
            .chain(aggregates(&bound.right))
            .any(|(a, _)| !a.roles.iter().any(|r| r.name == name) && mentions(&a.body, name));
        if inside || applies(&bound.left, name) || applies(&bound.right, name) {
            uses.push((true, bound.span.clone()));
        }
    }
}

impl Query {
    /// What, on its face, makes a pattern of this query never hold or a
    /// clause say nothing.
    pub fn lint(&self) -> Vec<Warning> {
        let mut out = Vec::new();
        for role in &self.roles {
            if !self.patterns.is_empty()
                && !self.outputs.contains(&role.name)
                && !self.patterns.iter().any(|p| mentions(p, &role.name))
            {
                warn(
                    &mut out,
                    &role.span,
                    format!(
                        "{} is declared and never used, yet still takes a value distinct from its sort's other roles",
                        role.name
                    ),
                );
            }
        }
        for (i, pattern) in self.patterns.iter().enumerate() {
            let what = if self.patterns.len() == 1 { "the pattern".to_owned() } else { format!("pattern {}", i + 1) };
            let at = if pattern.span.range().is_some() { &pattern.span } else { &self.span };
            declaration(&self.roles, pattern, at, &what, &mut out);
            body(pattern, &mut out);
            for role in self.roles.iter().filter(|r| !self.outputs.contains(&r.name)) {
                let mut uses = Vec::new();
                polarities(pattern, &role.name, true, &mut uses);
                if let Some((_, span)) = uses.first().filter(|_| uses.iter().all(|(positive, _)| !positive)) {
                    let name = &role.name;
                    warn(
                        &mut out,
                        span,
                        format!(
                            "{name} is used only under a negation: as a role of the query it is existential, so this \
                             asks for some {name} the clause fails, which almost any {name} does; to say no {name}, \
                             bind it inside: (none (roles {name}) ...)"
                        ),
                    );
                }
            }
        }
        out
    }
}

impl PolicyProgram {
    /// The guards' warnings, and a rule that can never decide: one with no
    /// pattern, or one whose answer is not asked to be the viewer's tile,
    /// where answers no hand holds are dropped without a word.
    pub fn lint(&self) -> Vec<Warning> {
        let mut out = Vec::new();
        for rule in self.rules.iter().chain(&self.bidding).chain(&self.declaring) {
            out.extend(rule.guard.lint());
            if rule.guard.patterns.is_empty() {
                warn(&mut out, &rule.span, format!("rule {} has no pattern, so it never fires", rule.name));
            }
            if rule.selector == Selector::FirstOutput
                && let [answer] = &rule.guard.outputs[..]
            {
                for (i, pattern) in rule.guard.patterns.iter().enumerate() {
                    let held = pattern.atoms.iter().any(|a| {
                        !a.negated
                            && matches!(a.predicate.as_str(), "own-legal" | "mine")
                            && a.args == [Term::Role(answer.clone())]
                    });
                    if !held {
                        let which = if rule.guard.patterns.len() == 1 {
                            String::new()
                        } else {
                            format!(" (pattern {})", i + 1)
                        };
                        warn(
                            &mut out,
                            &rule.span,
                            format!(
                                "rule {}{which} plays {answer} without asking (own-legal {answer}) or (mine {answer}): \
                                 an answer the viewer cannot play is dropped without a word",
                                rule.name
                            ),
                        );
                    }
                }
            }
        }
        out
    }
}
