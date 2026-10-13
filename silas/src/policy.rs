//! Executable policies: a pure function of the viewer's information. A
//! decision tries an exact table keyed by the complete record, then ordered
//! relational rules guarded by viewer-only queries, then a total fallback.
//! Nothing is carried from one decision to the next: the same frame always
//! gets the same decision.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::str::FromStr;

use crate::eval::{Budget, CompiledQuery, Witness};
use crate::frame::{Bid, Contract, Frame};
use crate::nouns::{Domino, DominoSet, Seat};
use crate::physics::Physics;
use crate::registry::{Access, Registry};
use crate::syntax::{Expr, Library, Query, Sort, Span, Value, valid_name};
use crate::{Error, Result, error};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PolicyProgram {
    pub name: String,
    pub exact_rules: Vec<ExactRule>,
    pub rules: Vec<PolicyRule>,
    pub fallback: Fallback,
}

/// The complete viewer information at a decision: the frame's own source.
/// The declaration's physics (by identity), the viewer, the hand it was
/// dealt, the contract, and every play so far. Everything else a frame holds
/// is derived from these.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct PolicyKey {
    pub physics: String,
    pub viewer: Seat,
    pub dealt: DominoSet,
    pub declarer: Seat,
    pub bid: Bid,
    pub history: Vec<Domino>,
}

impl PolicyKey {
    pub fn of(frame: &Frame) -> PolicyKey {
        PolicyKey {
            physics: frame.physics().identity().to_owned(),
            viewer: frame.viewer(),
            dealt: frame.dealt(),
            declarer: frame.contract().declarer(),
            bid: frame.contract().bid(),
            history: frame.plays().to_vec(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExactRule {
    pub key: PolicyKey,
    pub action: Domino,
    pub span: Span,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PolicyRule {
    pub name: String,
    pub selector: Selector,
    pub guard: Query,
    pub span: Span,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Selector {
    /// The least legal domino of the guard's one-column answer relation.
    FirstOutput,
    /// This tile, when the guard holds and the tile is legal.
    Literal(Domino),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fallback {
    LowestLegal,
}

#[derive(Clone)]
pub struct CompiledPolicy {
    identity: String,
    source: PolicyProgram,
    exact: BTreeMap<PolicyKey, Domino>,
    rules: Vec<(PolicyRule, CompiledQuery)>,
}

/// Which layer of the policy decided.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DecisionProvenance {
    /// The full information key was in the exact table.
    Exact,
    /// The named relational rule was the first to allow a legal tile.
    RelationalRule { name: String },
    /// Nothing else allowed one, so the total fallback ran.
    Fallback,
}

/// What a policy allows at a frame: the layer that decides, and the legal
/// tiles it allows. A rule allows the legal tiles its guard answers (`answer`)
/// or its tile when the guard holds and the tile is legal (`tile`); the exact
/// table and the fallback allow one tile each. The decision is the least
/// allowed tile.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Allowed {
    pub provenance: DecisionProvenance,
    pub tiles: DominoSet,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PolicyDecisionTrace {
    pub action: Domino,
    pub provenance: DecisionProvenance,
    /// Every tile the deciding layer allows.
    pub allowed: DominoSet,
    /// `Some(true)`: the declaring side has made its bid. `Some(false)`: the
    /// points still unbanked cannot make it.
    pub contract_resolved: Option<bool>,
}

/// The deciding rule's witnesses: each answer of its guard, and one binding
/// of the guard's roles that gives it.
pub type GuardWitnesses = Vec<(Vec<Value>, Witness)>;

/// Layer names a rule may not take.
const LAYERS: [&str; 2] = ["exact", "fallback"];

impl PolicyProgram {
    /// Compile against a registry. Exact rules are validated under `physics`,
    /// and every exact key must name it; relational rules run under any.
    pub fn compile(&self, registry: &Registry, physics: &Physics) -> Result<CompiledPolicy> {
        if !valid_name(&self.name) {
            return Err(error(format!("{} is not a valid policy name", self.name)));
        }
        let mut exact = BTreeMap::new();
        for rule in &self.exact_rules {
            let frame = validate_key(&rule.key, physics).map_err(|e| rule.span.error(e.0))?;
            if frame.next_actor() != Some(rule.key.viewer)
                || !physics.legal_plays(frame.kernel().viewer_hand(), frame.lead()).contains(rule.action)
            {
                return Err(rule.span.error(format!("the exact action {} is illegal at its key", rule.action)));
            }
            if exact.insert(rule.key.clone(), rule.action).is_some() {
                return Err(rule.span.error("a duplicate exact policy key"));
            }
        }
        let mut names = BTreeSet::new();
        let mut rules = Vec::new();
        for rule in &self.rules {
            if !valid_name(&rule.name) || LAYERS.contains(&rule.name.as_str()) || !names.insert(rule.name.as_str()) {
                return Err(rule.span.error(format!("invalid, reserved or duplicate policy rule {}", rule.name)));
            }
            let guard = rule.guard.compile(registry)?;
            if let Some(spec) = guard.predicate_specs().iter().find(|spec| spec.access == Access::World) {
                return Err(rule.span.error(format!(
                    "rule {} uses the world predicate {}; a policy sees only its viewer",
                    rule.name, spec.name
                )));
            }
            let wanted = usize::from(rule.selector == Selector::FirstOutput);
            if guard.outputs().len() != wanted {
                return Err(rule
                    .span
                    .error(format!("rule {}'s guard must have {wanted} output(s) for its selector", rule.name)));
            }
            if wanted == 1 && guard.outputs()[0].sort != Sort::Domino {
                return Err(rule.span.error(format!("rule {}'s action output must be a domino", rule.name)));
            }
            rules.push((rule.clone(), guard));
        }
        let mut identity = format!("silas-policy-v2\n{self}");
        for (_, query) in &rules {
            identity.push_str("\nrule-query: ");
            identity.push_str(&query.identity());
        }
        Ok(CompiledPolicy { identity, source: self.clone(), exact, rules })
    }
}

/// Rebuild the key's frame under the key's physics.
fn validate_key(key: &PolicyKey, physics: &Physics) -> Result<Frame> {
    if key.physics != physics.identity() {
        return Err(error(format!(
            "the key's physics {} is not {} ({})",
            key.physics,
            physics.identity(),
            physics.name()
        )));
    }
    let contract = Contract::new(key.declarer, key.bid, physics.clone())?;
    Frame::new(contract, key.viewer, key.dealt, key.history.clone())
}

impl CompiledPolicy {
    pub fn source(&self) -> &PolicyProgram {
        &self.source
    }

    pub fn identity(&self) -> &str {
        &self.identity
    }

    /// The tile the policy plays at `frame`, where its viewer is to play.
    pub fn choose(&self, frame: &Frame, budget: &mut Budget) -> Result<Domino> {
        self.decide(frame, budget).map(|trace| trace.action)
    }

    /// The decision at `frame`, with the layer that made it and every tile
    /// that layer allows. A function of the frame alone.
    pub fn decide(&self, frame: &Frame, budget: &mut Budget) -> Result<PolicyDecisionTrace> {
        let allowed = self.allowed(frame, budget)?;
        let action = allowed.tiles.iter().next().expect("a deciding layer allows a tile");
        Ok(PolicyDecisionTrace {
            action,
            provenance: allowed.provenance,
            allowed: allowed.tiles,
            contract_resolved: contract_resolution(frame),
        })
    }

    /// The layer that decides at `frame` and the tiles it allows.
    pub fn allowed(&self, frame: &Frame, budget: &mut Budget) -> Result<Allowed> {
        Ok(self.fired(frame, budget, false)?.0)
    }

    /// [`Self::allowed`], with the deciding rule's witnesses: each answer of
    /// its guard and one binding of the guard's roles that gives it. Empty for
    /// the exact table and the fallback.
    pub fn explain(&self, frame: &Frame, budget: &mut Budget) -> Result<(Allowed, GuardWitnesses)> {
        self.fired(frame, budget, true)
    }

    fn fired(&self, frame: &Frame, budget: &mut Budget, explain: bool) -> Result<(Allowed, GuardWitnesses)> {
        if frame.next_actor() != Some(frame.viewer()) {
            return Err(error("a policy acts only when its viewer is next"));
        }
        let legal = frame.physics().legal_plays(frame.kernel().viewer_hand(), frame.lead());
        if let Some(tile) = self.exact.get(&PolicyKey::of(frame)) {
            if !legal.contains(*tile) {
                return Err(error(format!("the exact action {tile} is illegal")));
            }
            return Ok((Allowed { provenance: DecisionProvenance::Exact, tiles: DominoSet::single(*tile) }, vec![]));
        }
        for (rule, guard) in &self.rules {
            let answers = guard.evaluate_viewer(frame, budget)?;
            let tiles = match &rule.selector {
                Selector::FirstOutput => answers
                    .iter()
                    .filter_map(|answer| match answer.0.as_slice() {
                        [Value::Domino(tile)] => Some(*tile),
                        _ => None,
                    })
                    .collect::<DominoSet>()
                    .intersection(legal),
                Selector::Literal(tile) if !answers.is_empty() => DominoSet::single(*tile).intersection(legal),
                Selector::Literal(_) => DominoSet::EMPTY,
            };
            if !tiles.is_empty() {
                let witnesses = if explain {
                    guard.witnesses(frame, budget)?.into_iter().map(|(answer, w)| (answer.0, w)).collect()
                } else {
                    vec![]
                };
                let provenance = DecisionProvenance::RelationalRule { name: rule.name.clone() };
                return Ok((Allowed { provenance, tiles }, witnesses));
            }
        }
        let lowest = legal.iter().next().ok_or_else(|| error("the policy has no legal action"))?;
        match self.source.fallback {
            Fallback::LowestLegal => {
                Ok((Allowed { provenance: DecisionProvenance::Fallback, tiles: DominoSet::single(lowest) }, vec![]))
            }
        }
    }
}

fn contract_resolution(frame: &Frame) -> Option<bool> {
    match frame.to_make_and_set() {
        (0, _) => Some(true),
        (_, 0) => Some(false),
        _ => None,
    }
}

/// The body of the one `(tag value)` among `xs`.
fn one<'a>(xs: &'a [Expr], tag: &str, at: &Span) -> Result<&'a Expr> {
    let bodies: Vec<_> = xs.iter().filter(|x| x.head() == Some(tag)).collect();
    match bodies.as_slice() {
        [form] => match form.tagged(tag)? {
            [value] => Ok(value),
            _ => Err(form.span.error(format!("{tag} takes one value"))),
        },
        _ => Err(at.error(format!("expected exactly one ({tag} ...)"))),
    }
}

fn parse<T: FromStr>(expr: &Expr, what: &str) -> Result<T> {
    let word = expr.word()?;
    word.parse().map_err(|_| expr.span.error(format!("invalid {what}: {word}")))
}

fn parse_seat(expr: &Expr) -> Result<Seat> {
    let word = expr.word()?;
    word.strip_prefix('S')
        .and_then(|n| n.parse().ok())
        .and_then(Seat::from_index)
        .ok_or_else(|| expr.span.error(format!("invalid chair {word}")))
}

impl PolicyProgram {
    /// Parse a policy text, naming it `name` in errors: any `include` and
    /// `define` forms, then the `(policy ...)`. At most 4 MiB, nested at most
    /// 40 deep.
    pub fn parse(text: &str, name: &str) -> Result<PolicyProgram> {
        let (library, tree) = Library::program(text, name, "policy", 4 << 20, 40)?;
        let forms = tree.tagged("policy")?;
        let name = forms.first().ok_or_else(|| tree.span.error("a policy requires a name"))?.word()?.to_owned();
        let forms = &forms[1..];
        let mut program = PolicyProgram { name, exact_rules: vec![], rules: vec![], fallback: Fallback::LowestLegal };
        let mut fallbacks = 0;
        for form in forms {
            match form.head() {
                Some("fallback") => {
                    fallbacks += 1;
                    if form.tagged("fallback")?.iter().map(Expr::word).collect::<Result<Vec<_>>>()? != ["lowest-legal"]
                    {
                        return Err(form.span.error("the fallback is (fallback lowest-legal)"));
                    }
                }
                Some("rule") => program.rules.push(rule(&library, form)?),
                Some("exact") => {
                    let [key, play] = form.tagged("exact")? else {
                        return Err(form.span.error("exact syntax is (exact (key ...) (play D))"));
                    };
                    let action = match play.tagged("play")? {
                        [tile] => parse(tile, "exact action")?,
                        _ => return Err(play.span.error("play takes one domino")),
                    };
                    program.exact_rules.push(ExactRule { key: parse_key(key)?, action, span: form.span.clone() });
                }
                Some(tag @ ("initial" | "bind")) => {
                    return Err(form.span.error(format!(
                        "({tag} ...) is gone: a policy is a function of the frame, with no modes and no bindings"
                    )));
                }
                _ => return Err(form.span.error("a policy form is (fallback ...), (rule ...) or (exact ...)")),
            }
        }
        if fallbacks != 1 {
            return Err(tree.span.error("a policy requires exactly one (fallback lowest-legal)"));
        }
        Ok(program)
    }
}

/// `(rule name (select ...) (guard (query ...)))`.
fn rule(library: &Library, form: &Expr) -> Result<PolicyRule> {
    let body = form.tagged("rule")?;
    let [name, fields @ ..] = body else {
        return Err(form.span.error("rule syntax is (rule name (select ...) (guard (query ...)))"));
    };
    if fields.len() != 2 || fields.iter().any(|f| !matches!(f.head(), Some("select" | "guard"))) {
        return Err(form.span.error("rule syntax is (rule name (select ...) (guard (query ...))), with no modes"));
    }
    let select = fields.iter().find(|f| f.head() == Some("select")).expect("a select form");
    let selector = match select.tagged("select")? {
        [answer] if answer.word()? == "answer" => Selector::FirstOutput,
        [tile, d] if tile.word()? == "tile" => Selector::Literal(parse(d, "selector domino")?),
        _ => return Err(select.span.error("select is answer, or tile D")),
    };
    Ok(PolicyRule {
        name: name.word()?.to_owned(),
        selector,
        guard: Query::from_expr(library, one(fields, "guard", &form.span)?)?,
        span: form.span.clone(),
    })
}

/// A Silas text: a query or a policy, after its `include` and `define`
/// forms.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Program {
    Query(Query),
    Policy(PolicyProgram),
}

impl Program {
    /// Parse a text whose final form is a `(query ...)` or a `(policy ...)`,
    /// naming it `name` in errors.
    pub fn parse(text: &str, name: &str) -> Result<Program> {
        let forms = Expr::read_all(text, name, 4 << 20, 40, false)?;
        match forms.last().and_then(Expr::head) {
            Some("query") => Query::parse(text, name).map(Program::Query),
            Some("policy") => PolicyProgram::parse(text, name).map(Program::Policy),
            _ => Err(match forms.last() {
                Some(last) => last.span.error("expected a final (query ...) or (policy ...) form"),
                None => error(format!("{name}: expected a (query ...) or a (policy ...), found nothing")),
            }),
        }
    }
}

impl FromStr for PolicyProgram {
    type Err = Error;

    fn from_str(source: &str) -> Result<Self> {
        PolicyProgram::parse(source, "policy")
    }
}

impl fmt::Display for PolicyProgram {
    /// The program with every definition expanded: it reads back as itself.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "(policy {}", self.name)?;
        writeln!(f, "  (fallback lowest-legal)")?;
        for rule in &self.exact_rules {
            writeln!(f, "  (exact {} (play {}))", KeyDisplay(&rule.key), rule.action)?;
        }
        for rule in &self.rules {
            let selector = match &rule.selector {
                Selector::FirstOutput => "answer".to_owned(),
                Selector::Literal(d) => format!("tile {d}"),
            };
            writeln!(f, "  (rule {} (select {selector}) (guard {}))", rule.name, rule.guard)?;
        }
        write!(f, ")")
    }
}

const KEY_FIELDS: [&str; 6] = ["physics", "viewer", "dealt-bits", "declarer", "bid", "history"];

fn parse_key(expr: &Expr) -> Result<PolicyKey> {
    let xs = expr.tagged("key")?;
    let tags: BTreeSet<&str> = xs.iter().filter_map(Expr::head).collect();
    if xs.len() != KEY_FIELDS.len() || tags != KEY_FIELDS.into_iter().collect() {
        return Err(expr.span.error("an exact key has missing, duplicate, or unknown fields"));
    }
    let bits: u32 = parse(one(xs, "dealt-bits", &expr.span)?, "domino bits")?;
    let history = xs
        .iter()
        .find(|x| x.head() == Some("history"))
        .expect("a history field")
        .tagged("history")?
        .iter()
        .map(|d| parse(d, "history domino"))
        .collect::<Result<_>>()?;
    Ok(PolicyKey {
        physics: one(xs, "physics", &expr.span)?.word()?.to_owned(),
        viewer: parse_seat(one(xs, "viewer", &expr.span)?)?,
        dealt: DominoSet::from_bits(bits).ok_or_else(|| expr.span.error("domino bits outside the deck"))?,
        declarer: parse_seat(one(xs, "declarer", &expr.span)?)?,
        bid: parse(one(xs, "bid", &expr.span)?, "bid")?,
        history,
    })
}

struct KeyDisplay<'a>(&'a PolicyKey);

impl fmt::Display for KeyDisplay<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let k = self.0;
        write!(
            f,
            "(key (physics {}) (viewer {}) (dealt-bits {}) (declarer {}) (bid {}) (history",
            k.physics,
            k.viewer,
            k.dealt.bits(),
            k.declarer,
            k.bid
        )?;
        for d in &k.history {
            write!(f, " {d}")?;
        }
        write!(f, "))")
    }
}
