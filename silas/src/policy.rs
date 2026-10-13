//! Executable policies: exact information-state rules, ordered relational
//! rules guarded by viewer-only queries, rigid bindings, a controller mode,
//! and a total fallback.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::str::FromStr;
use std::sync::Arc;

use crate::eval::{Budget, CompiledQuery};
use crate::frame::{Frame, Hidden, Kernel};
use crate::nouns::{Context, ContextSet, Domino, DominoSet, Seat, Team};
use crate::physics::{Physics, total_points};
use crate::registry::{Access, Registry};
use crate::syntax::{Expr, Query, Sort, Value, valid_name};
use crate::{Error, Result, error};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PolicyProgram {
    pub name: String,
    pub initial_mode: String,
    pub bindings: Vec<RigidBinding>,
    pub exact_rules: Vec<ExactRule>,
    pub rules: Vec<PolicyRule>,
    pub fallback: Fallback,
}

/// Everything a policy may see at a decision: the viewer's frame and the full
/// public history, score and contract.
#[derive(Clone, Debug)]
pub struct PolicyInput<'a> {
    pub frame: &'a Frame,
    pub history: &'a [(Seat, Domino)],
    pub banked: [u32; 2],
    pub bid: u8,
    pub declaring_team: Team,
}

impl<'a> PolicyInput<'a> {
    /// Checks that the history is a coherent public predecessor of the frame.
    pub fn new(
        frame: &'a Frame,
        history: &'a [(Seat, Domino)],
        banked: [u32; 2],
        bid: u8,
        declaring_team: Team,
    ) -> Result<Self> {
        let input = Self { frame, history, banked, bid, declaring_team };
        validate_key(&PolicyKey::from_input(&input), frame.physics_handle())?;
        Ok(input)
    }
}

/// The complete viewer information at a decision, under one physics.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct PolicyKey {
    pub physics: String,
    pub viewer: Seat,
    pub hand: DominoSet,
    pub pool: DominoSet,
    pub hidden: Vec<(Seat, usize, Vec<Context>)>,
    pub leader: Seat,
    pub prefix: Vec<Domino>,
    pub played: DominoSet,
    pub history: Vec<(Seat, Domino)>,
    pub banked: [u32; 2],
    pub bid: u8,
    pub declaring_team: Team,
}

impl PolicyKey {
    pub fn from_input(input: &PolicyInput<'_>) -> Self {
        let frame = input.frame;
        let kernel = frame.kernel();
        Self {
            physics: frame.physics().identity(),
            viewer: kernel.viewer(),
            hand: kernel.viewer_hand(),
            pool: kernel.pool(),
            hidden: kernel.hidden().iter().map(|h| (h.seat, h.capacity, h.voids.iter().collect())).collect(),
            leader: frame.leader(),
            prefix: frame.prefix().to_vec(),
            played: frame.played(),
            history: input.history.to_vec(),
            banked: input.banked,
            bid: input.bid,
            declaring_team: input.declaring_team,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExactRule {
    pub key: PolicyKey,
    pub action: Domino,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RigidBinding {
    pub name: String,
    pub query: Query,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PolicyRule {
    pub name: String,
    pub in_mode: String,
    pub next_mode: String,
    pub selector: Selector,
    pub guard: Query,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Selector {
    /// The least legal domino in the guard's one-column answer relation.
    FirstOutput,
    Literal(Domino),
    Rigid(String),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fallback {
    LowestLegal,
}

#[derive(Clone)]
pub struct CompiledPolicy {
    identity: Arc<str>,
    source: PolicyProgram,
    bindings: Vec<(String, CompiledQuery)>,
    exact: BTreeMap<PolicyKey, Domino>,
    rules: Vec<(PolicyRule, CompiledQuery)>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PolicyController {
    program_identity: Arc<str>,
    mode: String,
    rigid: BTreeMap<String, Value>,
}

/// Which layer of the policy chose.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DecisionProvenance {
    /// The full information key was in the exact table.
    Exact,
    /// The named relational rule was the first to select a legal action.
    RelationalRule { name: String },
    /// Nothing else selected, so the total fallback ran.
    Fallback,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PolicyDecisionTrace {
    pub action: Domino,
    pub provenance: DecisionProvenance,
    pub mode_before: String,
    pub mode_after: String,
    /// `Some(true)`: the declaring side has made its bid. `Some(false)`: the
    /// points still unbanked cannot make it.
    pub contract_resolved: Option<bool>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PolicyControllerState {
    pub mode: String,
    pub rigid: BTreeMap<String, Value>,
}

impl PolicyProgram {
    /// Compile against a registry. Exact rules are validated under `physics`,
    /// and every exact key must name it; relational rules run under any.
    pub fn compile(&self, registry: &Registry, physics: &Arc<dyn Physics>) -> Result<CompiledPolicy> {
        if !valid_name(&self.name) || !valid_name(&self.initial_mode) {
            return Err(error("a policy's name and initial mode must be valid names"));
        }
        let mut names = BTreeSet::new();
        let mut bindings = Vec::new();
        for binding in &self.bindings {
            if !valid_name(&binding.name) || !names.insert(binding.name.as_str()) {
                return Err(error(format!("invalid or duplicate rigid binding {}", binding.name)));
            }
            let query = binding.query.compile(registry)?;
            viewer_only(&query, &format!("binding {}", binding.name))?;
            if query.outputs().len() != 1 {
                return Err(error(format!("binding {} must have one output", binding.name)));
            }
            bindings.push((binding.name.clone(), query));
        }
        let mut exact = BTreeMap::new();
        for rule in &self.exact_rules {
            let frame = validate_key(&rule.key, physics)?;
            if frame.next_actor() != Some(rule.key.viewer)
                || !physics.legal_plays(rule.key.hand, frame.led_context()).contains(rule.action)
            {
                return Err(error(format!("the exact action {} is illegal at its key", rule.action)));
            }
            if exact.insert(rule.key.clone(), rule.action).is_some() {
                return Err(error("a duplicate exact policy key"));
            }
        }
        let mut rule_names = BTreeSet::new();
        let mut rules = Vec::new();
        for rule in &self.rules {
            if ![&rule.name, &rule.in_mode, &rule.next_mode].into_iter().all(|n| valid_name(n))
                || !rule_names.insert(rule.name.as_str())
            {
                return Err(error(format!("invalid or duplicate policy rule {}", rule.name)));
            }
            if let Selector::Rigid(name) = &rule.selector
                && !self.bindings.iter().any(|b| b.name == *name)
            {
                return Err(error(format!("rule {} names the unknown binding {name}", rule.name)));
            }
            let guard = rule.guard.compile(registry)?;
            viewer_only(&guard, &format!("rule {}", rule.name))?;
            let wanted = usize::from(rule.selector == Selector::FirstOutput);
            if guard.outputs().len() != wanted {
                return Err(error(format!("rule {}'s guard must have {wanted} output(s) for its selector", rule.name)));
            }
            if wanted == 1 && guard.outputs()[0].sort != Sort::Domino {
                return Err(error(format!("rule {}'s action output must be a domino", rule.name)));
            }
            rules.push((rule.clone(), guard));
        }
        let mut identity = format!("silas-policy-v0\n{self}");
        for (_, query) in &bindings {
            identity.push_str("\nbinding-query: ");
            identity.push_str(&query.identity());
        }
        for (_, query) in &rules {
            identity.push_str("\nrule-query: ");
            identity.push_str(&query.identity());
        }
        Ok(CompiledPolicy { identity: Arc::from(identity), source: self.clone(), bindings, exact, rules })
    }
}

fn viewer_only(query: &CompiledQuery, owner: &str) -> Result<()> {
    match query.predicate_specs().iter().find(|spec| spec.access == Access::World) {
        Some(spec) => {
            Err(error(format!("{owner} uses the world predicate {}; a policy sees only its viewer", spec.name)))
        }
        None => Ok(()),
    }
}

/// Rebuild the key's frame and replay its history, checking that every part
/// of the key agrees with it.
fn validate_key(key: &PolicyKey, physics: &Arc<dyn Physics>) -> Result<Frame> {
    if key.physics != physics.identity() {
        return Err(error(format!("the key's physics {} is not {}", key.physics, physics.identity())));
    }
    let hidden: Vec<Hidden> = key
        .hidden
        .iter()
        .map(|(seat, capacity, voids)| {
            let set: ContextSet = voids.iter().copied().collect();
            if set.len() != voids.len() {
                return Err(error("an exact key repeats a void"));
            }
            Ok(Hidden { seat: *seat, capacity: *capacity, voids: set })
        })
        .collect::<Result<_>>()?;
    let hidden = hidden.try_into().map_err(|_| error("an exact key describes three hidden seats"))?;
    let kernel = Kernel::new(key.viewer, key.hand, key.pool, hidden)?;
    if physics.count_worlds(&kernel) == 0 {
        return Err(error("the exact key's kernel has no worlds"));
    }
    let frame = Frame::new(physics.clone(), kernel, key.leader, key.prefix.clone(), key.played)?;
    if frame.kernel().live().union(key.played) != DominoSet::FULL {
        return Err(error("the key's live and played tiles must partition the deck"));
    }
    let history: DominoSet = key.history.iter().map(|(_, tile)| *tile).collect();
    if history.len() != key.history.len() || history != key.played {
        return Err(error("the history must list every played domino exactly once"));
    }
    let mut remaining = [7usize; Seat::COUNT];
    let mut voids = [ContextSet::EMPTY; Seat::COUNT];
    let mut banked = [0u32; 2];
    let mut prefix: Vec<Domino> = Vec::new();
    let mut leader = key.history.first().map_or(key.leader, |(seat, _)| *seat);
    let mut viewer_hand =
        key.hand.union(key.history.iter().filter(|(s, _)| *s == key.viewer).map(|(_, d)| *d).collect());
    for (seat, tile) in &key.history {
        if *seat != leader.plus(prefix.len()) || remaining[seat.index()] == 0 {
            return Err(error("the history violates turn order or a seat's capacity"));
        }
        let led = prefix.first().map(|d| physics.led_context(*d));
        if *seat == key.viewer {
            if !physics.legal_plays(viewer_hand, led).contains(*tile) {
                return Err(error("the history contains an illegal play by the viewer"));
            }
            viewer_hand.remove(*tile);
        }
        if let Some(q) = led {
            if !physics.follows(*tile, q) {
                voids[seat.index()].insert(q);
            } else if voids[seat.index()].contains(q) {
                return Err(error("the history follows a context its seat had shown void"));
            }
        }
        remaining[seat.index()] -= 1;
        prefix.push(*tile);
        if let Ok(trick) = <[Domino; 4]>::try_from(prefix.as_slice()) {
            leader = physics.trick_winner(leader, trick);
            banked[leader.team().index()] += physics.trick_points(trick);
            prefix.clear();
        }
    }
    if leader != key.leader || prefix != key.prefix || banked != key.banked {
        return Err(error("the key's leader, prefix, or banked score disagrees with its history"));
    }
    if remaining[key.viewer.index()] != key.hand.len() {
        return Err(error("the viewer's hand size disagrees with the history"));
    }
    for (seat, capacity, shown) in &key.hidden {
        if remaining[seat.index()] != *capacity || voids[seat.index()] != shown.iter().copied().collect() {
            return Err(error("a hidden capacity or void disagrees with the history"));
        }
    }
    Ok(frame)
}

impl CompiledPolicy {
    pub fn source(&self) -> &PolicyProgram {
        &self.source
    }

    pub fn identity(&self) -> &str {
        &self.identity
    }

    pub fn initialize(&self, input: &PolicyInput<'_>, budget: &mut Budget) -> Result<PolicyController> {
        let mut rigid = BTreeMap::new();
        for (name, query) in &self.bindings {
            let answers = query.evaluate_viewer(input.frame, budget)?;
            match answers.first() {
                Some(answer) if answers.len() == 1 => rigid.insert(name.clone(), answer.0[0]),
                _ => {
                    return Err(error(format!("rigid binding {name} needs exactly one answer, got {}", answers.len())));
                }
            };
        }
        Ok(PolicyController { program_identity: self.identity.clone(), mode: self.source.initial_mode.clone(), rigid })
    }

    pub fn choose(
        &self,
        controller: &mut PolicyController,
        input: &PolicyInput<'_>,
        budget: &mut Budget,
    ) -> Result<Domino> {
        self.choose_traced(controller, input, budget).map(|trace| trace.action)
    }

    /// Choose as [`Self::choose`] does, recording which layer answered. A
    /// refusal, including budget exhaustion partway through the rules,
    /// restores the controller; work already done stays charged.
    pub fn choose_traced(
        &self,
        controller: &mut PolicyController,
        input: &PolicyInput<'_>,
        budget: &mut Budget,
    ) -> Result<PolicyDecisionTrace> {
        let checkpoint = controller.clone();
        let result = self.decide(controller, input, budget);
        if result.is_err() {
            *controller = checkpoint;
        }
        result
    }

    fn decide(
        &self,
        controller: &mut PolicyController,
        input: &PolicyInput<'_>,
        budget: &mut Budget,
    ) -> Result<PolicyDecisionTrace> {
        let frame = input.frame;
        if controller.program_identity != self.identity {
            return Err(error("the controller belongs to a different compiled policy"));
        }
        if frame.next_actor() != Some(frame.kernel().viewer()) {
            return Err(error("a policy acts only when its viewer is next"));
        }
        let legal = frame.physics().legal_plays(frame.kernel().viewer_hand(), frame.led_context());
        let lowest = legal.iter().next().ok_or_else(|| error("the policy has no legal action"))?;
        let mode_before = controller.mode.clone();
        let contract_resolved = contract_resolution(input);
        let trace = |action, provenance, mode_after| PolicyDecisionTrace {
            action,
            provenance,
            mode_before: mode_before.clone(),
            mode_after,
            contract_resolved,
        };
        if let Some(tile) = self.exact.get(&PolicyKey::from_input(input)) {
            if !legal.contains(*tile) {
                return Err(error(format!("the exact action {tile} is illegal")));
            }
            return Ok(trace(*tile, DecisionProvenance::Exact, mode_before.clone()));
        }
        for (rule, guard) in &self.rules {
            if rule.in_mode != controller.mode {
                continue;
            }
            let answers = guard.evaluate_viewer(frame, budget)?;
            let selected = match &rule.selector {
                Selector::FirstOutput => answers
                    .iter()
                    .filter_map(|answer| match answer.0.as_slice() {
                        [Value::Domino(tile)] if legal.contains(*tile) => Some(*tile),
                        _ => None,
                    })
                    .min(),
                Selector::Literal(tile) => (!answers.is_empty() && legal.contains(*tile)).then_some(*tile),
                Selector::Rigid(name) => match controller.rigid.get(name) {
                    Some(Value::Domino(tile)) => (!answers.is_empty() && legal.contains(*tile)).then_some(*tile),
                    Some(_) => return Err(error(format!("rigid binding {name} is not a domino"))),
                    None => return Err(error(format!("missing rigid binding {name}"))),
                },
            };
            if let Some(tile) = selected {
                controller.mode = rule.next_mode.clone();
                let provenance = DecisionProvenance::RelationalRule { name: rule.name.clone() };
                return Ok(trace(tile, provenance, controller.mode.clone()));
            }
        }
        match self.source.fallback {
            Fallback::LowestLegal => Ok(trace(lowest, DecisionProvenance::Fallback, mode_before.clone())),
        }
    }
}

fn contract_resolution(input: &PolicyInput<'_>) -> Option<bool> {
    let total = total_points(input.frame.physics());
    let declared = input.banked[input.declaring_team.index()];
    let unbanked = total - input.banked[0] - input.banked[1];
    let bid = u32::from(input.bid);
    if declared >= bid {
        Some(true)
    } else if declared + unbanked < bid {
        Some(false)
    } else {
        None
    }
}

impl PolicyController {
    pub fn mode(&self) -> &str {
        &self.mode
    }

    pub fn rigid(&self, name: &str) -> Option<Value> {
        self.rigid.get(name).copied()
    }

    pub fn state(&self) -> PolicyControllerState {
        PolicyControllerState { mode: self.mode.clone(), rigid: self.rigid.clone() }
    }
}

/// The body of the one `(tag value)` among `xs`.
fn one<'a>(xs: &'a [Expr], tag: &str) -> Result<&'a Expr> {
    let bodies: Vec<_> = xs.iter().filter_map(|x| x.tagged(tag).ok()).collect();
    match bodies.as_slice() {
        [[value]] => Ok(value),
        [_] => Err(error(format!("{tag} takes one value"))),
        _ => Err(error(format!("expected exactly one ({tag} ...)"))),
    }
}

/// The words of the one `(tag word ...)` among `xs`.
fn words<'a>(xs: &'a [Expr], tag: &str) -> Result<Vec<&'a str>> {
    let bodies: Vec<_> = xs.iter().filter_map(|x| x.tagged(tag).ok()).collect();
    match bodies.as_slice() {
        [body] => body.iter().map(Expr::word).collect(),
        _ => Err(error(format!("expected exactly one ({tag} ...)"))),
    }
}

fn parse<T: FromStr>(word: &str, what: &str) -> Result<T> {
    word.parse().map_err(|_| error(format!("invalid {what}: {word}")))
}

fn parse_seat(word: &str) -> Result<Seat> {
    word.strip_prefix('S')
        .and_then(|n| n.parse().ok())
        .and_then(Seat::from_index)
        .ok_or_else(|| error(format!("invalid chair {word}")))
}

fn parse_bits(word: &str) -> Result<DominoSet> {
    DominoSet::from_bits(parse(word, "domino bits")?).ok_or_else(|| error("domino bits outside the deck"))
}

impl FromStr for PolicyProgram {
    type Err = Error;

    /// At most 4 MiB of text, nested at most 40 deep.
    fn from_str(source: &str) -> Result<Self> {
        let tree = Expr::read(source, "policy", 4 << 20, 40)?;
        let forms = tree.tagged("policy")?;
        let name = forms.first().ok_or_else(|| error("a policy requires a name"))?.word()?.to_owned();
        let forms = &forms[1..];
        let mut counts = BTreeMap::new();
        for form in forms {
            let tag = form.list()?.first().ok_or_else(|| error("an empty policy form"))?.word()?;
            if !["initial", "fallback", "bind", "exact", "rule"].contains(&tag) {
                return Err(error(format!("unknown policy form {tag}")));
            }
            *counts.entry(tag).or_insert(0) += 1;
        }
        if counts.get("initial") != Some(&1) || counts.get("fallback") != Some(&1) {
            return Err(error("a policy requires exactly one initial and one fallback form"));
        }
        let initial_mode = one(forms, "initial")?.word()?.to_owned();
        if one(forms, "fallback")?.word()? != "lowest-legal" {
            return Err(error("the fallback must be lowest-legal"));
        }
        let mut program = PolicyProgram {
            name,
            initial_mode,
            bindings: vec![],
            exact_rules: vec![],
            rules: vec![],
            fallback: Fallback::LowestLegal,
        };
        for form in forms {
            if let Ok(body) = form.tagged("bind") {
                let [name, query] = body else {
                    return Err(error("bind syntax is (bind name (query ...))"));
                };
                program.bindings.push(RigidBinding { name: name.word()?.to_owned(), query: Query::from_expr(query)? });
            } else if let Ok(body) = form.tagged("rule") {
                if body.len() != 5 {
                    return Err(error(
                        "rule syntax is (rule name (in mode) (next mode) (select ...) (guard (query ...)))",
                    ));
                }
                let fields = &body[1..];
                let selector = match words(fields, "select")?.as_slice() {
                    ["answer"] => Selector::FirstOutput,
                    ["tile", tile] => Selector::Literal(parse(tile, "selector domino")?),
                    ["rigid", name] => Selector::Rigid((*name).to_owned()),
                    _ => return Err(error("select is answer, tile D, or rigid NAME")),
                };
                program.rules.push(PolicyRule {
                    name: body[0].word()?.to_owned(),
                    in_mode: one(fields, "in")?.word()?.to_owned(),
                    next_mode: one(fields, "next")?.word()?.to_owned(),
                    selector,
                    guard: Query::from_expr(one(fields, "guard")?)?,
                });
            } else if let Ok(body) = form.tagged("exact") {
                let [key, play] = body else {
                    return Err(error("exact syntax is (exact (key ...) (play D))"));
                };
                let action = play.tagged("play")?.first().ok_or_else(|| error("play requires a domino"))?.word()?;
                program.exact_rules.push(ExactRule { key: parse_key(key)?, action: parse(action, "exact action")? });
            }
        }
        Ok(program)
    }
}

impl fmt::Display for PolicyProgram {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "(policy {}", self.name)?;
        writeln!(f, "  (initial {})", self.initial_mode)?;
        writeln!(f, "  (fallback lowest-legal)")?;
        for binding in &self.bindings {
            writeln!(f, "  (bind {} {})", binding.name, binding.query)?;
        }
        for rule in &self.exact_rules {
            writeln!(f, "  (exact {} (play {}))", KeyDisplay(&rule.key), rule.action)?;
        }
        for rule in &self.rules {
            let selector = match &rule.selector {
                Selector::FirstOutput => "answer".to_owned(),
                Selector::Literal(d) => format!("tile {d}"),
                Selector::Rigid(n) => format!("rigid {n}"),
            };
            writeln!(
                f,
                "  (rule {} (in {}) (next {}) (select {selector}) (guard {}))",
                rule.name, rule.in_mode, rule.next_mode, rule.guard
            )?;
        }
        write!(f, ")")
    }
}

const KEY_FIELDS: [&str; 12] = [
    "physics",
    "viewer",
    "hand-bits",
    "pool-bits",
    "hidden",
    "leader",
    "prefix",
    "played-bits",
    "history",
    "banked",
    "bid",
    "declaring-team",
];

fn parse_key(expr: &Expr) -> Result<PolicyKey> {
    let xs = expr.tagged("key")?;
    let tags: BTreeSet<&str> = xs.iter().filter_map(|x| x.list().ok()?.first()?.word().ok()).collect();
    if xs.len() != KEY_FIELDS.len() || tags != KEY_FIELDS.into_iter().collect() {
        return Err(error("an exact key has missing, duplicate, or unknown fields"));
    }
    let scalar = |tag: &str| match words(xs, tag)?.as_slice() {
        [word] => Ok(*word),
        _ => Err(error(format!("{tag} takes one value"))),
    };
    let banked = match words(xs, "banked")?.as_slice() {
        [a, b] => [parse(a, "banked score")?, parse(b, "banked score")?],
        _ => return Err(error("banked takes two values")),
    };
    let hidden = one_body(xs, "hidden")?
        .iter()
        .map(|entry| match entry.list()? {
            [seat, capacity, voids] => Ok((
                parse_seat(seat.word()?)?,
                parse(capacity.word()?, "hidden capacity")?,
                voids.tagged("voids")?.iter().map(|q| parse(q.word()?, "void context")).collect::<Result<_>>()?,
            )),
            _ => Err(error("a hidden entry is (chair capacity (voids ...))")),
        })
        .collect::<Result<_>>()?;
    let history = one_body(xs, "history")?
        .iter()
        .map(|entry| match entry.list()? {
            [seat, tile] => Ok((parse_seat(seat.word()?)?, parse(tile.word()?, "history domino")?)),
            _ => Err(error("a history entry is (chair domino)")),
        })
        .collect::<Result<_>>()?;
    Ok(PolicyKey {
        physics: scalar("physics")?.to_owned(),
        viewer: parse_seat(scalar("viewer")?)?,
        hand: parse_bits(scalar("hand-bits")?)?,
        pool: parse_bits(scalar("pool-bits")?)?,
        hidden,
        leader: parse_seat(scalar("leader")?)?,
        prefix: words(xs, "prefix")?.into_iter().map(|d| parse(d, "prefix domino")).collect::<Result<_>>()?,
        played: parse_bits(scalar("played-bits")?)?,
        history,
        banked,
        bid: parse(scalar("bid")?, "bid")?,
        declaring_team: match scalar("declaring-team")? {
            "T0" => Team::T0,
            "T1" => Team::T1,
            other => return Err(error(format!("invalid declaring team {other}"))),
        },
    })
}

fn one_body<'a>(xs: &'a [Expr], tag: &str) -> Result<&'a [Expr]> {
    xs.iter().find_map(|x| x.tagged(tag).ok()).ok_or_else(|| error(format!("missing ({tag} ...)")))
}

struct KeyDisplay<'a>(&'a PolicyKey);

impl fmt::Display for KeyDisplay<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let k = self.0;
        write!(
            f,
            "(key (physics {}) (viewer {}) (hand-bits {}) (pool-bits {}) (hidden",
            k.physics,
            k.viewer,
            k.hand.bits(),
            k.pool.bits()
        )?;
        for (seat, capacity, voids) in &k.hidden {
            write!(f, " ({seat} {capacity} (voids")?;
            for q in voids {
                write!(f, " {q}")?;
            }
            write!(f, "))")?;
        }
        write!(f, ") (leader {}) (prefix", k.leader)?;
        for d in &k.prefix {
            write!(f, " {d}")?;
        }
        write!(f, ") (played-bits {}) (history", k.played.bits())?;
        for (seat, d) in &k.history {
            write!(f, " ({seat} {d})")?;
        }
        write!(f, ") (banked {} {}) (bid {}) (declaring-team {}))", k.banked[0], k.banked[1], k.bid, k.declaring_team)
    }
}
