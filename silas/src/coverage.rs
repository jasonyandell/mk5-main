//! Where a rule stops, and how often each rule and each clause holds: the
//! diagnostic for a rule that silently never fires.
//!
//! A pattern's *funnel* is its clauses in source order, grouped by the source
//! form each came from (a definition call is one step, however many
//! primitives it expands to; so is a sugared `or`). Step `k` asks whether the
//! first `k` steps hold together: some assignment of the query's roles makes
//! them all true. Adding a clause only removes assignments, so the steps hold
//! on a prefix, and the first step that fails is where the pattern stops.
//! Every step is a query of its own, read under the same semantics: nothing
//! here is a second evaluator.

use crate::eval::{Budget, CompiledQuery};
use crate::frame::Frame;
use crate::policy::{CompiledPolicy, DecisionProvenance};
use crate::registry::Registry;
use crate::syntax::{Pattern, Query};
use crate::{Result, error};

/// One pattern's funnel: each step's label and the query of the steps so far.
#[derive(Clone)]
pub struct Funnel {
    steps: Vec<(String, CompiledQuery)>,
}

impl Funnel {
    /// The funnels of every pattern of `query`.
    pub fn of(query: &Query, registry: &Registry) -> Result<Vec<Funnel>> {
        query.patterns.iter().map(|pattern| Funnel::new(query, pattern, registry)).collect()
    }

    fn new(query: &Query, pattern: &Pattern, registry: &Registry) -> Result<Funnel> {
        // Each clause with where it came from: atoms first, then scopes, as
        // the core keeps them; a stable sort by source position restores the
        // text's order, and a clause built in Rust keeps its core place.
        enum Clause<'a> {
            Atom(&'a crate::syntax::Atom),
            Scope(&'a crate::syntax::Scope),
        }
        let mut clauses: Vec<(Option<(usize, usize)>, Clause<'_>)> = pattern
            .atoms
            .iter()
            .map(|a| (a.span.range(), Clause::Atom(a)))
            .chain(pattern.scopes.iter().map(|s| (s.span.range(), Clause::Scope(s))))
            .collect();
        clauses.sort_by_key(|(range, _)| range.map_or(usize::MAX, |r| r.0));
        let mut steps = Vec::new();
        let mut prefix = Pattern { equal: pattern.equal.clone(), span: pattern.span.clone(), ..Pattern::default() };
        let mut k = 0;
        while k < clauses.len() {
            let range = clauses[k].0;
            let label = match &clauses[k].1 {
                Clause::Atom(a) => a.span.text(),
                Clause::Scope(s) => s.span.text(),
            };
            // One source form: every clause from the same place.
            loop {
                match &clauses[k].1 {
                    Clause::Atom(a) => prefix.atoms.push((*a).clone()),
                    Clause::Scope(s) => prefix.scopes.push((*s).clone()),
                }
                k += 1;
                if range.is_none() || clauses.get(k).is_none_or(|c| c.0 != range) {
                    break;
                }
            }
            let step = Query { patterns: vec![prefix.clone()], ..query.clone() };
            let label = label.unwrap_or_else(|| format!("clause {}", steps.len() + 1));
            steps.push((label, step.compile(registry)?));
        }
        Ok(Funnel { steps })
    }

    /// Each step's label: its source text, on one line.
    pub fn labels(&self) -> Vec<&str> {
        self.steps.iter().map(|(label, _)| label.as_str()).collect()
    }

    /// How many steps hold together at `frame`, from the first. A funnel of
    /// a query that reads a hidden world is refused, as viewer evaluation is.
    pub fn depth(&self, frame: &Frame, budget: &mut Budget) -> Result<usize> {
        // The steps that hold are a prefix: bisect for its length.
        let (mut holds, mut fails) = (0, self.steps.len() + 1);
        while fails - holds > 1 {
            let k = (holds + fails) / 2;
            if self.steps[k - 1].1.evaluate_viewer(frame, budget)?.is_empty() {
                fails = k;
            } else {
                holds = k;
            }
        }
        Ok(holds)
    }
}

/// How often each rule of a policy answered, allowed a legal tile and
/// decided, and how often each step of each of its patterns held, over the
/// frames observed.
#[derive(Clone)]
pub struct Coverage {
    /// Frames observed.
    pub frames: u64,
    /// Frames the exact table decided.
    pub exact: u64,
    /// Frames no layer above the fallback decided.
    pub fallback: u64,
    pub rules: Vec<RuleCoverage>,
}

#[derive(Clone)]
pub struct RuleCoverage {
    pub name: String,
    /// Frames where the guard had an answer.
    pub answered: u64,
    /// Frames where the rule allowed a legal tile: it would decide, were it
    /// first.
    pub allowed: u64,
    /// Frames where it decided.
    pub decided: u64,
    /// Each pattern's steps: the label and the frames where the steps so far
    /// held together.
    pub patterns: Vec<Vec<(String, u64)>>,
    funnels: Vec<Funnel>,
}

impl Coverage {
    pub fn new(policy: &CompiledPolicy, registry: &Registry) -> Result<Coverage> {
        let rules = policy
            .rules()
            .iter()
            .map(|(rule, _)| {
                let funnels = Funnel::of(&rule.guard, registry)?;
                Ok(RuleCoverage {
                    name: rule.name.clone(),
                    answered: 0,
                    allowed: 0,
                    decided: 0,
                    patterns: funnels
                        .iter()
                        .map(|f| f.labels().into_iter().map(|l| (l.to_owned(), 0)).collect())
                        .collect(),
                    funnels,
                })
            })
            .collect::<Result<_>>()?;
        Ok(Coverage { frames: 0, exact: 0, fallback: 0, rules })
    }

    /// Observe `policy` at `frame`, where its viewer is to play: every rule
    /// is read, not only the rules up to the one that decides.
    pub fn observe(&mut self, policy: &CompiledPolicy, frame: &Frame, budget: &mut Budget) -> Result<()> {
        if frame.next_actor() != Some(frame.viewer()) {
            return Err(error("a policy acts only when its viewer is next"));
        }
        let legal = frame.physics().legal_plays(frame.kernel().viewer_hand(), frame.lead());
        let mut decided = policy.has_exact(frame);
        self.frames += 1;
        self.exact += u64::from(decided);
        for (k, rule) in self.rules.iter_mut().enumerate() {
            let (answers, tiles) = policy.rule_allows(k, frame, legal, budget)?;
            rule.answered += u64::from(!answers.is_empty());
            rule.allowed += u64::from(!tiles.is_empty());
            if !tiles.is_empty() && !decided {
                rule.decided += 1;
                decided = true;
            }
            for (funnel, counts) in rule.funnels.iter().zip(&mut rule.patterns) {
                for count in counts.iter_mut().take(funnel.depth(frame, budget)?) {
                    count.1 += 1;
                }
            }
        }
        self.fallback += u64::from(!decided);
        Ok(())
    }
}

/// Why each rule before the deciding one did not decide at `frame`: for each
/// pattern, the label of the step where it stops; or, when the guard answered,
/// that no answer was legal.
pub fn why_not(
    policy: &CompiledPolicy,
    provenance: &DecisionProvenance,
    frame: &Frame,
    registry: &Registry,
    budget: &mut Budget,
) -> Result<Vec<(String, String)>> {
    let legal = frame.physics().legal_plays(frame.kernel().viewer_hand(), frame.lead());
    let mut out = Vec::new();
    if *provenance == DecisionProvenance::Exact {
        return Ok(out);
    }
    for (k, (rule, _)) in policy.rules().iter().enumerate() {
        if *provenance == (DecisionProvenance::RelationalRule { name: rule.name.clone() }) {
            break;
        }
        let (answers, _) = policy.rule_allows(k, frame, legal, budget)?;
        let reason = if !answers.is_empty() {
            "its guard answered, but no answer is a legal play".to_owned()
        } else {
            let funnels = Funnel::of(&rule.guard, registry)?;
            let mut stops = Vec::new();
            for (i, funnel) in funnels.iter().enumerate() {
                let depth = funnel.depth(frame, budget)?;
                let at = funnel.labels().get(depth).map_or("its roles", |l| l).to_owned();
                stops.push(if funnels.len() == 1 {
                    format!("stops at {at}")
                } else {
                    format!("pattern {} stops at {at}", i + 1)
                });
            }
            if stops.is_empty() { "it has no pattern".to_owned() } else { stops.join("; ") }
        };
        out.push((rule.name.clone(), reason));
    }
    Ok(out)
}
