//! Executable policies: a pure function of the viewer's information. A
//! decision tries an exact table keyed by the complete record, then ordered
//! relational rules guarded by viewer-only queries, then a total fallback.
//! Nothing is carried from one decision to the next: the same frame always
//! gets the same decision.
//!
//! A policy has three sections, each a list of rules: the play (its
//! top-level rules), `(bidding ...)` and `(declaring ...)`. The play's rules
//! read a [`Frame`]; the bidding and declaring rules read an
//! [`AuctionFrame`], so a whole hand's policy is a function of the frame of
//! each decision.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::str::FromStr;

use crate::auction::{Auction, AuctionConfig, AuctionFrame, Call};
use crate::dominance::Dominance;
use crate::eval::{Budget, CompiledQuery, Memo, Witness};
use crate::frame::{Bid, Frame};
use crate::nouns::{Context, Domino, DominoSet, Pip, Seat};
use crate::physics::Physics;
use crate::registry::{Phase, Registry};
use crate::syntax::{Expr, Library, Query, Sort, Span, Value, valid_name};
use crate::{Error, Result, error};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PolicyProgram {
    pub name: String,
    pub exact_rules: Vec<ExactRule>,
    /// The play's rules.
    pub rules: Vec<PolicyRule>,
    /// The bidding rules: each selects a call.
    pub bidding: Vec<PolicyRule>,
    /// The declaring rules: each selects a declaration.
    pub declaring: Vec<PolicyRule>,
    pub fallback: Fallback,
}

/// The complete viewer information at a decision of the play: the frame's
/// own source. The declaration's physics (by identity), the viewer, the hand
/// it was dealt, the auction, and every play so far. Everything else a frame
/// holds is derived from these, the contract included.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct PolicyKey {
    pub physics: String,
    pub viewer: Seat,
    pub dealt: DominoSet,
    pub auction: Auction,
    pub history: Vec<Domino>,
}

impl PolicyKey {
    pub fn of(frame: &Frame) -> PolicyKey {
        PolicyKey {
            physics: frame.physics().identity().to_owned(),
            viewer: frame.viewer(),
            dealt: frame.dealt(),
            auction: frame.auction().clone(),
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

/// What a rule selects when its guard holds.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Selector {
    /// Play: the least legal domino of the guard's one-column answer relation.
    FirstOutput,
    /// Play: this tile, when the guard holds and the tile is legal.
    Literal(Domino),
    /// Bidding: this call, when the guard holds and the call is legal.
    Call(Call),
    /// Bidding: the least legal bid, when the guard holds.
    LeastRaise,
    /// Declaring: a declaration offered, when the guard holds.
    Declare(Declared),
}

/// What a declaring rule names: the pip suits its guard answers (one context
/// column), one pip suit, the doubles, or no trump. A declaration offered is
/// named by what it makes trump ([`Trump::of`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Declared {
    Answer,
    Suit(Pip),
    Doubles,
    NoTrump,
}

/// What a declaration makes trump, read off its tables: the tiles bearing one
/// pip, the doubles, or nothing. A declaration whose trump is neither has no
/// name a rule can give, and is chosen only by the fallback.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Trump {
    Suit(Pip),
    Doubles,
    None,
}

impl Trump {
    pub fn of(physics: &Physics) -> Option<Trump> {
        let Some(q) = physics.trump() else {
            return Some(Trump::None);
        };
        let tiles = physics.incidence(q);
        let doubles: DominoSet = Domino::ALL.into_iter().filter(|d| d.is_double()).collect();
        if tiles == doubles {
            return Some(Trump::Doubles);
        }
        Pip::ALL.into_iter().find(|p| tiles == Domino::ALL.into_iter().filter(|d| d.has(*p)).collect()).map(Trump::Suit)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fallback {
    /// The lowest legal tile; the least legal call (a pass when it is
    /// legal); the first declaration offered.
    LowestLegal,
}

/// The three sections of a policy, by the frames their rules read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Section {
    Play,
    Bidding,
    Declaring,
}

impl Section {
    fn phase(self) -> Phase {
        match self {
            Section::Play => Phase::Play,
            Section::Bidding | Section::Declaring => Phase::Auction,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Section::Play => "play",
            Section::Bidding => "bidding",
            Section::Declaring => "declaring",
        }
    }
}

#[derive(Clone)]
pub struct CompiledPolicy {
    identity: String,
    source: PolicyProgram,
    exact: BTreeMap<PolicyKey, Domino>,
    rules: Vec<(PolicyRule, CompiledQuery)>,
    bidding: Vec<(PolicyRule, CompiledQuery)>,
    declaring: Vec<(PolicyRule, CompiledQuery)>,
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

/// A decision at the auction: the call or the declaration, the layer that
/// made it, and the deciding rule's witnesses when asked for.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Choice<T> {
    pub choice: T,
    pub provenance: DecisionProvenance,
    pub witnesses: GuardWitnesses,
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
        let mut section = |rules: &[PolicyRule], section: Section| -> Result<Vec<(PolicyRule, CompiledQuery)>> {
            rules.iter().map(|rule| compile_rule(rule, section, registry, &mut names)).collect()
        };
        let mut rules = section(&self.rules, Section::Play)?;
        let mut bidding = section(&self.bidding, Section::Bidding)?;
        let mut declaring = section(&self.declaring, Section::Declaring)?;
        // A decision evaluates a section's guards at one frame: an aggregate
        // written alike in two rules is counted once.
        let mut shapes = BTreeMap::new();
        for (_, guard) in rules.iter_mut().chain(&mut bidding).chain(&mut declaring) {
            guard.intern(&mut shapes);
        }
        let mut identity = format!("silas-policy-v3\n{self}");
        for (_, query) in rules.iter().chain(&bidding).chain(&declaring) {
            identity.push_str("\nrule-query: ");
            identity.push_str(&query.identity());
        }
        Ok(CompiledPolicy { identity, source: self.clone(), exact, rules, bidding, declaring })
    }
}

/// One rule, checked against its section: a name of its own, a guard of
/// viewer predicates defined at the section's frames, and as many outputs as
/// its selector reads (one domino for a play's `answer`, one context for a
/// declaring `answer`, none otherwise).
fn compile_rule(
    rule: &PolicyRule,
    section: Section,
    registry: &Registry,
    names: &mut BTreeSet<String>,
) -> Result<(PolicyRule, CompiledQuery)> {
    if !valid_name(&rule.name) || LAYERS.contains(&rule.name.as_str()) || !names.insert(rule.name.clone()) {
        return Err(rule.span.error(format!("invalid, reserved or duplicate policy rule {}", rule.name)));
    }
    let fits = matches!(
        (&rule.selector, section),
        (Selector::FirstOutput | Selector::Literal(_), Section::Play)
            | (Selector::Call(_) | Selector::LeastRaise, Section::Bidding)
            | (Selector::Declare(_), Section::Declaring)
    );
    if !fits {
        return Err(rule.span.error(format!(
            "rule {} selects {}, which is not a {} rule's",
            rule.name,
            Choose(&rule.selector),
            section.name()
        )));
    }
    let guard = rule.guard.compile(registry)?;
    if let Some(spec) = guard.predicate_specs().iter().find(|spec| spec.access.reads_world()) {
        return Err(rule.span.error(format!(
            "rule {} uses the {} {}; a policy sees only its viewer{}",
            rule.name,
            spec.access.noun(),
            spec.name,
            spec.access.fence()
        )));
    }
    if let Some(spec) = guard.predicate_specs().iter().find(|spec| !spec.phase.reads(section.phase())) {
        return Err(rule.span.error(format!(
            "rule {} is a {} rule and uses {}, a predicate of {}",
            rule.name,
            section.name(),
            spec.name,
            spec.phase
        )));
    }
    let output = match &rule.selector {
        Selector::FirstOutput => Some(Sort::Domino),
        Selector::Declare(Declared::Answer) => Some(Sort::Context),
        _ => None,
    };
    if guard.outputs().len() != usize::from(output.is_some()) {
        return Err(rule.span.error(format!(
            "rule {}'s guard must have {} output(s) for its selector",
            rule.name,
            usize::from(output.is_some())
        )));
    }
    if let Some(sort) = output
        && guard.outputs()[0].sort != sort
    {
        return Err(rule.span.error(format!("rule {}'s answer must be a {sort}", rule.name)));
    }
    Ok((rule.clone(), guard))
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
    Frame::new(key.auction.clone(), physics.clone(), key.viewer, key.dealt, key.history.clone())
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

    /// The set the policy means at `frame`: what its deciding layer allows,
    /// or every legal tile where only the fallback decides, the policy's
    /// silence. A reading weighs a play against this set, and a play model
    /// draws from it.
    pub fn permits(&self, frame: &Frame, budget: &mut Budget) -> Result<DominoSet> {
        let allowed = self.allowed(frame, budget)?;
        Ok(match allowed.provenance {
            DecisionProvenance::Fallback => frame.physics().legal_plays(frame.kernel().viewer_hand(), frame.lead()),
            _ => allowed.tiles,
        })
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
        let memo = Memo::default();
        if let Some(tile) = self.exact.get(&PolicyKey::of(frame)) {
            if !legal.contains(*tile) {
                return Err(error(format!("the exact action {tile} is illegal")));
            }
            return Ok((Allowed { provenance: DecisionProvenance::Exact, tiles: DominoSet::single(*tile) }, vec![]));
        }
        for k in 0..self.rules.len() {
            let (rule, guard) = &self.rules[k];
            let (_, tiles) = self.rule_allows(k, frame, legal, &memo, budget)?;
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

    /// The call the policy makes at `frame`, where its viewer is to call: the
    /// first bidding rule whose guard holds and whose call is legal (a
    /// `least-raise` rule's call is the least legal bid), else the least legal
    /// call. A function of the auction frame alone.
    pub fn bid(&self, frame: &AuctionFrame, budget: &mut Budget) -> Result<Choice<Call>> {
        self.bid_explained(frame, budget, false)
    }

    /// [`Self::bid`] with the deciding rule's witnesses.
    pub fn explain_bid(&self, frame: &AuctionFrame, budget: &mut Budget) -> Result<Choice<Call>> {
        self.bid_explained(frame, budget, true)
    }

    fn bid_explained(&self, frame: &AuctionFrame, budget: &mut Budget, explain: bool) -> Result<Choice<Call>> {
        let auction = frame.auction();
        if auction.actor() != Some(frame.viewer()) {
            return Err(error(format!("{} is not to call after {auction}", frame.viewer())));
        }
        let legal = auction.legal_calls();
        let memo = Memo::default();
        for (rule, guard) in &self.bidding {
            if guard.evaluate_auction_with(frame, &memo, budget)?.is_empty() {
                continue;
            }
            let call = match rule.selector {
                Selector::Call(call) => Some(call).filter(|c| legal.contains(c)),
                Selector::LeastRaise => auction.least_raise().map(Call::Bid),
                _ => unreachable!("a bidding rule selects a call"),
            };
            if let Some(call) = call {
                return chosen(call, rule, guard, frame, budget, explain);
            }
        }
        let least = *legal.first().ok_or_else(|| error("no call is legal"))?;
        Ok(Choice { choice: least, provenance: DecisionProvenance::Fallback, witnesses: vec![] })
    }

    /// The declaration the policy makes at `frame`, where its viewer has won
    /// the auction: of the declarations offered that the first declaring rule
    /// to name one allows, the first offered; else the first offered. A
    /// function of the auction frame alone.
    pub fn declare(&self, frame: &AuctionFrame, budget: &mut Budget) -> Result<Choice<Physics>> {
        self.declare_explained(frame, budget, false)
    }

    /// [`Self::declare`] with the deciding rule's witnesses.
    pub fn explain_declare(&self, frame: &AuctionFrame, budget: &mut Budget) -> Result<Choice<Physics>> {
        self.declare_explained(frame, budget, true)
    }

    fn declare_explained(&self, frame: &AuctionFrame, budget: &mut Budget, explain: bool) -> Result<Choice<Physics>> {
        match frame.auction().result() {
            Some((winner, _)) if winner == frame.viewer() => {}
            _ => return Err(error(format!("{} has not won {}", frame.viewer(), frame.auction()))),
        }
        let offered: Vec<(Option<Trump>, &Physics)> = frame.declarations().iter().map(|d| (Trump::of(d), d)).collect();
        let memo = Memo::default();
        for (rule, guard) in &self.declaring {
            let answers = guard.evaluate_auction_with(frame, &memo, budget)?;
            if answers.is_empty() {
                continue;
            }
            let named: Vec<Trump> = match rule.selector {
                Selector::Declare(Declared::Answer) => answers
                    .iter()
                    .filter_map(|answer| match answer.0.as_slice() {
                        [Value::Context(Context::Natural(p))] => Some(Trump::Suit(*p)),
                        _ => None,
                    })
                    .collect(),
                Selector::Declare(Declared::Suit(p)) => vec![Trump::Suit(p)],
                Selector::Declare(Declared::Doubles) => vec![Trump::Doubles],
                Selector::Declare(Declared::NoTrump) => vec![Trump::None],
                _ => unreachable!("a declaring rule selects a declaration"),
            };
            if let Some((_, physics)) = offered.iter().find(|(trump, _)| trump.is_some_and(|t| named.contains(&t))) {
                return chosen((*physics).clone(), rule, guard, frame, budget, explain);
            }
        }
        let first = frame.declarations()[0].clone();
        Ok(Choice { choice: first, provenance: DecisionProvenance::Fallback, witnesses: vec![] })
    }

    /// The relational rules of the play, in order, each with its compiled guard.
    pub fn rules(&self) -> &[(PolicyRule, CompiledQuery)] {
        &self.rules
    }

    /// The bidding rules, in order, each with its compiled guard.
    pub fn bidding(&self) -> &[(PolicyRule, CompiledQuery)] {
        &self.bidding
    }

    /// The declaring rules, in order, each with its compiled guard.
    pub fn declaring(&self) -> &[(PolicyRule, CompiledQuery)] {
        &self.declaring
    }

    /// Whether the exact table holds the frame's key.
    pub(crate) fn has_exact(&self, frame: &Frame) -> bool {
        self.exact.contains_key(&PolicyKey::of(frame))
    }

    /// What rule `k` says at `frame`: its guard's answers, and the tiles of
    /// `legal` it allows. An answer that is not legal is no tile.
    pub(crate) fn rule_allows(
        &self,
        k: usize,
        frame: &Frame,
        legal: DominoSet,
        memo: &Memo,
        budget: &mut Budget,
    ) -> Result<(crate::Answers, DominoSet)> {
        let (rule, guard) = &self.rules[k];
        let answers = guard.evaluate_viewer_with(frame, memo, budget)?;
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
            _ => DominoSet::EMPTY,
        };
        Ok((answers, tiles))
    }
}

/// A rule's decision at the auction, with its witnesses when asked for.
fn chosen<T>(
    choice: T,
    rule: &PolicyRule,
    guard: &CompiledQuery,
    frame: &AuctionFrame,
    budget: &mut Budget,
    explain: bool,
) -> Result<Choice<T>> {
    let witnesses = if explain {
        guard.witnesses_at_auction(frame, budget)?.into_iter().map(|(answer, w)| (answer.0, w)).collect()
    } else {
        vec![]
    };
    Ok(Choice { choice, provenance: DecisionProvenance::RelationalRule { name: rule.name.clone() }, witnesses })
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
        let mut program = PolicyProgram {
            name,
            exact_rules: vec![],
            rules: vec![],
            bidding: vec![],
            declaring: vec![],
            fallback: Fallback::LowestLegal,
        };
        let mut fallbacks = 0;
        let mut sections = BTreeSet::new();
        for form in forms {
            match form.head() {
                Some("fallback") => {
                    fallbacks += 1;
                    if form.tagged("fallback")?.iter().map(Expr::word).collect::<Result<Vec<_>>>()? != ["lowest-legal"]
                    {
                        return Err(form.span.error("the fallback is (fallback lowest-legal)"));
                    }
                }
                Some("rule") => program.rules.push(rule(&library, form, Section::Play)?),
                Some(tag @ ("bidding" | "declaring")) => {
                    if !sections.insert(tag) {
                        return Err(form.span.error(format!("a policy has one ({tag} ...) section")));
                    }
                    let section = if tag == "bidding" { Section::Bidding } else { Section::Declaring };
                    let rules = form
                        .tagged(tag)?
                        .iter()
                        .map(|r| match r.head() {
                            Some("rule") => rule(&library, r, section),
                            _ => Err(r.span.error(format!("a ({tag} ...) section holds (rule ...) forms"))),
                        })
                        .collect::<Result<Vec<_>>>()?;
                    if section == Section::Bidding {
                        program.bidding = rules;
                    } else {
                        program.declaring = rules;
                    }
                }
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
                _ => {
                    return Err(form.span.error(
                        "a policy form is (fallback ...), (rule ...), (exact ...), (bidding ...) or (declaring ...)",
                    ));
                }
            }
        }
        if fallbacks > 1 {
            return Err(tree.span.error("a policy has one fallback, (fallback lowest-legal), and may leave it unsaid"));
        }
        Ok(program)
    }
}

/// What a short form's head selects, and the output it names, if any.
type ShortForm = (Selector, Option<Expr>);

/// The short form `(play x)`, `(bid c)` or `(declare k)` of `section`'s
/// rules: `(play x)`, `x` a name, is `(select answer)` answering `x`; `(play
/// D)` is `(select tile D)`; `(bid c)` is `(select bid c)`; `(declare t)`, `t`
/// a name, is `(select declare answer)` answering `t`, and `(declare k)` with
/// `k` a pip suit (`q0` to `q6`), `doubles` or `no-trump` is `(select declare
/// k)`.
fn short_form(head: &Expr, section: Section) -> Result<ShortForm> {
    let tag = head.head().expect("a short form has a head");
    let [action] = head.tagged(tag)? else {
        return Err(head.span.error(format!("{tag} takes one value")));
    };
    let word = action.word()?;
    let named = || Some(action.clone());
    match (tag, section) {
        ("play", Section::Play) => match Value::literal(word) {
            Some(Value::Domino(tile)) => Ok((Selector::Literal(tile), None)),
            Some(_) => Err(action.span.error(format!("{word} is not a domino or a name to play"))),
            None => Ok((Selector::FirstOutput, named())),
        },
        ("bid", Section::Bidding) => Ok((bid_choice(action)?, None)),
        ("declare", Section::Declaring) => match declared(action) {
            Ok(kind) => Ok((Selector::Declare(kind), None)),
            Err(_) if Value::literal(word).is_none() && word != "answer" => {
                Ok((Selector::Declare(Declared::Answer), named()))
            }
            Err(e) => Err(e),
        },
        _ => Err(head.span.error(format!("({tag} ...) is not a {} rule's short form", section.name()))),
    }
}

/// A bid choice: a bid as the table says it, `pass`, or `least-raise`.
fn bid_choice(expr: &Expr) -> Result<Selector> {
    match expr.word()? {
        "least-raise" => Ok(Selector::LeastRaise),
        word => match word.parse::<Call>() {
            Ok(Call::Bid(Bid::Points(n))) if n < crate::auction::MIN_BID => Err(expr.span.error(format!(
                "a bid is 30 to 41 points or a number of marks (42 each), pass or least-raise, not {word}"
            ))),
            Ok(call) => Ok(Selector::Call(call)),
            Err(_) => Err(expr.span.error(format!(
                "a bid is 30 to 41 points or a number of marks (42 each), pass or least-raise, not {word}"
            ))),
        },
    }
}

/// A declaration's name: `answer`, a pip suit `q0` to `q6`, `doubles` or
/// `no-trump`.
fn declared(expr: &Expr) -> Result<Declared> {
    match expr.word()? {
        "answer" => Ok(Declared::Answer),
        "doubles" => Ok(Declared::Doubles),
        "no-trump" => Ok(Declared::NoTrump),
        word => match Value::literal(word) {
            Some(Value::Context(Context::Natural(p))) => Ok(Declared::Suit(p)),
            _ => Err(expr.span.error(format!("a declaration is answer, q0 to q6, doubles or no-trump, not {word}"))),
        },
    }
}

/// `(rule name (select ...) (guard (query ...)))`, or a short form `(rule
/// name (play x) query-body ...)`, `(rule name (bid c) query-body ...)` or
/// `(rule name (declare k) query-body ...)`: the guard is `(query
/// query-body ...)`, answering the short form's name when it gives one. The
/// query body is what follows `(out ...)` in a query, `(roles ...)` first if
/// given.
fn rule(library: &Library, form: &Expr, section: Section) -> Result<PolicyRule> {
    let body = form.tagged("rule")?;
    let [name, fields @ ..] = body else {
        return Err(form.span.error(
            "rule syntax is (rule name (play ...) clause ...) or (rule name (select ...) (guard (query ...)))",
        ));
    };
    if let Some((head, rest)) =
        fields.split_first().filter(|(f, _)| matches!(f.head(), Some("play" | "bid" | "declare")))
    {
        let (selector, output) = short_form(head, section)?;
        let at = &form.span;
        let (roles, rest) = match rest.split_first() {
            Some((roles, tail)) if roles.head() == Some("roles") => (vec![roles.clone()], tail),
            _ => (vec![], rest),
        };
        let out = Expr::list_at([vec![Expr::word_at("out", at)], output.into_iter().collect()].concat(), at);
        let query = Expr::list_at([vec![Expr::word_at("query", at)], roles, vec![out], rest.to_vec()].concat(), at);
        return Ok(PolicyRule {
            name: name.word()?.to_owned(),
            selector,
            guard: Query::from_expr(library, &query)?,
            span: form.span.clone(),
        });
    }
    if fields.len() != 2 || fields.iter().any(|f| !matches!(f.head(), Some("select" | "guard"))) {
        return Err(form.span.error("rule syntax is (rule name (select ...) (guard (query ...))), with no modes"));
    }
    let select = fields.iter().find(|f| f.head() == Some("select")).expect("a select form");
    let words = select.tagged("select")?;
    let selector = match (section, words) {
        (Section::Play, [answer]) if answer.word()? == "answer" => Selector::FirstOutput,
        (Section::Play, [tile, d]) if tile.word()? == "tile" => Selector::Literal(parse(d, "selector domino")?),
        (Section::Play, _) => return Err(select.span.error("a play rule selects answer, or tile D")),
        (Section::Bidding, [bid, choice]) if bid.word()? == "bid" => bid_choice(choice)?,
        (Section::Bidding, _) => {
            return Err(select.span.error("a bidding rule selects bid N, bid pass or bid least-raise"));
        }
        (Section::Declaring, [declare, kind]) if declare.word()? == "declare" => Selector::Declare(declared(kind)?),
        (Section::Declaring, _) => {
            return Err(select.span.error("a declaring rule selects declare answer, q0 to q6, doubles or no-trump"));
        }
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
    Dominance(Dominance),
}

impl Program {
    /// Parse a text whose final form is a `(query ...)`, a `(policy ...)` or
    /// a `(dominates ...)`, naming it `name` in errors.
    pub fn parse(text: &str, name: &str) -> Result<Program> {
        let forms = Expr::read_all(text, name, 4 << 20, 40, false)?;
        match forms.last().and_then(Expr::head) {
            Some("query") => Query::parse(text, name).map(Program::Query),
            Some("policy") => PolicyProgram::parse(text, name).map(Program::Policy),
            Some("dominates") => Dominance::parse(text, name).map(Program::Dominance),
            _ => Err(match forms.last() {
                Some(last) => last.span.error("expected a final (query ...), (policy ...) or (dominates ...) form"),
                None => {
                    error(format!("{name}: expected a (query ...), a (policy ...) or a (dominates ...), found nothing"))
                }
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

/// A selector as its explicit `(select ...)` says it.
struct Choose<'a>(&'a Selector);

impl fmt::Display for Choose<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.0 {
            Selector::FirstOutput => f.write_str("answer"),
            Selector::Literal(d) => write!(f, "tile {d}"),
            Selector::Call(call) => write!(f, "bid {call}"),
            Selector::LeastRaise => f.write_str("bid least-raise"),
            Selector::Declare(Declared::Answer) => f.write_str("declare answer"),
            Selector::Declare(Declared::Suit(p)) => write!(f, "declare q{p}"),
            Selector::Declare(Declared::Doubles) => f.write_str("declare doubles"),
            Selector::Declare(Declared::NoTrump) => f.write_str("declare no-trump"),
        }
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
            writeln!(f, "  (rule {} (select {}) (guard {}))", rule.name, Choose(&rule.selector), rule.guard)?;
        }
        for (tag, rules) in [("bidding", &self.bidding), ("declaring", &self.declaring)] {
            if rules.is_empty() {
                continue;
            }
            writeln!(f, "  ({tag}")?;
            for rule in rules {
                writeln!(f, "    (rule {} (select {}) (guard {}))", rule.name, Choose(&rule.selector), rule.guard)?;
            }
            writeln!(f, "  )")?;
        }
        write!(f, ")")
    }
}

const KEY_FIELDS: [&str; 5] = ["physics", "viewer", "dealt-bits", "auction", "history"];
const AUCTION_FIELDS: [&str; 4] = ["max-marks", "forced", "shaker", "calls"];

/// The tagged fields of `(tag (field ...) ...)`, each once, exactly `fields`.
fn fields<'a>(expr: &'a Expr, tag: &str, fields: &[&str]) -> Result<&'a [Expr]> {
    let xs = expr.tagged(tag)?;
    let tags: BTreeSet<&str> = xs.iter().filter_map(Expr::head).collect();
    if xs.len() != fields.len() || tags != fields.iter().copied().collect() {
        return Err(expr
            .span
            .error(format!("{tag} has missing, duplicate, or unknown fields; it takes {}", fields.join(", "))));
    }
    Ok(xs)
}

fn parse_key(expr: &Expr) -> Result<PolicyKey> {
    let xs = fields(expr, "key", &KEY_FIELDS)?;
    let bits: u32 = parse(one(xs, "dealt-bits", &expr.span)?, "domino bits")?;
    let history = xs
        .iter()
        .find(|x| x.head() == Some("history"))
        .expect("a history field")
        .tagged("history")?
        .iter()
        .map(|d| parse(d, "history domino"))
        .collect::<Result<_>>()?;
    let auction = xs.iter().find(|x| x.head() == Some("auction")).expect("an auction field");
    Ok(PolicyKey {
        physics: one(xs, "physics", &expr.span)?.word()?.to_owned(),
        viewer: parse_seat(one(xs, "viewer", &expr.span)?)?,
        dealt: DominoSet::from_bits(bits).ok_or_else(|| expr.span.error("domino bits outside the deck"))?,
        auction: parse_auction(auction)?,
        history,
    })
}

/// `(auction (max-marks m) (forced yes|no) (shaker S) (calls c ...))`.
fn parse_auction(expr: &Expr) -> Result<Auction> {
    let xs = fields(expr, "auction", &AUCTION_FIELDS)?;
    let forced = match one(xs, "forced", &expr.span)?.word()? {
        "yes" => true,
        "no" => false,
        word => return Err(expr.span.error(format!("forced is yes or no, not {word}"))),
    };
    let config = AuctionConfig::new(parse(one(xs, "max-marks", &expr.span)?, "max-marks")?, forced)
        .map_err(|e| expr.span.error(e.0))?;
    let calls = xs
        .iter()
        .find(|x| x.head() == Some("calls"))
        .expect("a calls field")
        .tagged("calls")?
        .iter()
        .map(|c| parse(c, "call"))
        .collect::<Result<_>>()?;
    Auction::new(config, parse_seat(one(xs, "shaker", &expr.span)?)?, calls).map_err(|e| expr.span.error(e.0))
}

struct KeyDisplay<'a>(&'a PolicyKey);

impl fmt::Display for KeyDisplay<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let k = self.0;
        let a = &k.auction;
        write!(
            f,
            "(key (physics {}) (viewer {}) (dealt-bits {}) (auction (max-marks {}) (forced {}) (shaker {}) (calls",
            k.physics,
            k.viewer,
            k.dealt.bits(),
            a.config().max_marks(),
            if a.config().forced() { "yes" } else { "no" },
            a.shaker()
        )?;
        for call in a.calls() {
            write!(f, " {call}")?;
        }
        write!(f, ")) (history")?;
        for d in &k.history {
            write!(f, " {d}")?;
        }
        write!(f, "))")
    }
}
