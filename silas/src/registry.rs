//! Predicates: the typed, versioned relations a query may name.

use std::collections::BTreeMap;
use std::sync::Arc;

use crate::auction::{Auction, AuctionFrame, Call};
use crate::eval::Budget;
use crate::frame::{Bid, Frame, Holdings};
use crate::nouns::{Context, Pip, Seat};
use crate::scoring::Scoring;
use crate::syntax::{RESERVED, Sort, Value, valid_name};
use crate::{Result, error};

/// What a predicate may read.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Access {
    /// The frame only: the viewer's hand and the public residue.
    Viewer,
    /// The frame and one hidden world's [`Holdings`].
    World,
}

/// Where a predicate is defined: at a frame of the play (after the
/// declaration), at an auction frame (before it), or at both. A policy's play
/// rules may name only `Play` and `Both` predicates, its bidding and
/// declaring rules only `Auction` and `Both`; a query is refused at a frame of
/// the other phase before any work.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Phase {
    Play,
    Auction,
    Both,
}

impl Phase {
    /// Whether a predicate of this phase is defined at a frame of `at`.
    pub fn reads(self, at: Phase) -> bool {
        self == Phase::Both || self == at
    }
}

impl std::fmt::Display for Phase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Phase::Play => "the play",
            Phase::Auction => "the auction",
            Phase::Both => "the play and the auction",
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PredicateSpec {
    pub name: String,
    /// The semantic version: changing what the predicate computes changes it.
    pub version: String,
    pub parameters: Vec<Sort>,
    /// Zero for relations on the current state; positive for bounded
    /// continuations.
    pub horizon_plies: u32,
    pub access: Access,
    pub phase: Phase,
}

#[derive(Clone, Copy, Debug)]
pub enum PredicateContext<'a> {
    Viewer(&'a Frame),
    World(Holdings<'a>),
    /// A seat's view of the auction, before the declaration.
    Auction(&'a AuctionFrame),
}

impl<'a> PredicateContext<'a> {
    /// The play frame. Refused at the auction, where no predicate of the
    /// play is asked (its phase is checked first).
    pub fn frame(self) -> Result<&'a Frame> {
        match self {
            Self::Viewer(frame) => Ok(frame),
            Self::World(holdings) => Ok(holdings.frame()),
            Self::Auction(_) => Err(error("a predicate of the play was asked at the auction")),
        }
    }

    pub fn world(self) -> Option<Holdings<'a>> {
        match self {
            Self::Viewer(_) | Self::Auction(_) => None,
            Self::World(holdings) => Some(holdings),
        }
    }

    /// The auction frame, before the declaration.
    pub fn auction(self) -> Option<&'a AuctionFrame> {
        match self {
            Self::Auction(frame) => Some(frame),
            _ => None,
        }
    }
}

/// A registered computation. It must be deterministic, respect its declared
/// horizon, charge its inner work to the budget, and consult nothing it does
/// not declare. The runtime checks argument sorts and withholds the world
/// from `Viewer` predicates; it cannot audit what a closure captures.
pub trait Predicate: Send + Sync {
    fn spec(&self) -> &PredicateSpec;

    /// `None` means undefined: neither the atom nor its negation holds.
    fn evaluate(&self, context: PredicateContext<'_>, args: &[Value], budget: &mut Budget) -> Result<Option<bool>>;
}

/// The built-ins added with the record, version `silas-v1/core`.
const V1: [&str; 8] = ["declarer", "banked", "to-make", "to-set", "led", "played-to", "trick-of", "shown-void"];

/// The built-ins whose meaning last changed with the one-frame record and
/// the full contract, version `silas-v2/core`: `played` is every tile of the
/// record, `bid` reads the bid as the table says it (84 for two marks),
/// `trick-number` is undefined once the hand is over; `stake`, `trump` and
/// `mine` are new.
const V2: [&str; 6] = ["played", "bid", "trick-number", "stake", "trump", "mine"];

/// The built-ins that came with the auction, version `silas-v4/auction`.
const V4: [&str; 16] = [
    "bears",
    "high-pip",
    "shaker",
    "bid-of",
    "passed",
    "no-bid",
    "high-bid",
    "high-bidder",
    "legal-bid",
    "least-raise",
    "scoring-marks",
    "scoring-points",
    "score",
    "short",
    "short-at-most",
    "behind",
];

/// The built-ins defined at the auction as well as in the play: the tiles'
/// own facts, the seats, the dealt hand, the count, and the auction's record.
const BOTH: [&str; 19] = [
    "double",
    "tile",
    "chair",
    "context",
    "team",
    "partner",
    "opponent",
    "successor",
    "viewer",
    "mine",
    "count",
    "bears",
    "high-pip",
    "shaker",
    "bid-of",
    "passed",
    "no-bid",
    "high-bid",
    "high-bidder",
];

#[derive(Clone, Default)]
pub struct Registry {
    pub(crate) predicates: BTreeMap<String, Arc<dyn Predicate>>,
}

impl Registry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(&mut self, predicate: impl Predicate + 'static) -> Result<()> {
        let spec = predicate.spec();
        if !valid_name(&spec.name) || RESERVED.contains(&spec.name.as_str()) || spec.version.trim().is_empty() {
            return Err(error("a predicate needs an unreserved name and a semantic version"));
        }
        if self.predicates.contains_key(&spec.name) {
            return Err(error(format!("predicate {} is already registered", spec.name)));
        }
        self.predicates.insert(spec.name.clone(), Arc::new(predicate));
        Ok(())
    }

    pub fn specs(&self) -> Vec<PredicateSpec> {
        self.predicates.values().map(|p| p.spec().clone()).collect()
    }

    /// The built-in relations. Each carries the version in which its meaning
    /// last changed: `silas-v0/core` for the first set, `silas-v1/core` and
    /// `silas-v2/core` for those that read the hand's record, and
    /// `silas-v3/outcome` for the solved outcomes, which look to the hand's end.
    pub fn standard() -> Self {
        use Access::{Viewer, World};
        use Sort::{Chair as C, Context as Q, Domino as D, Number as N, Team as T};
        let mut registry = Self::new();
        for (name, parameters, access) in [
            ("live", vec![D], Viewer),
            ("holds", vec![C, D], World),
            ("in", vec![D, Q], Viewer),
            ("double", vec![D], Viewer),
            ("beats", vec![D, D, Q], Viewer),
            ("boss", vec![D, Q], Viewer),
            ("master", vec![D], Viewer),
            ("void", vec![C, Q], World),
            ("quota", vec![C, N], Viewer),
            ("tile", vec![D, D], Viewer),
            ("chair", vec![C, C], Viewer),
            ("context", vec![Q, Q], Viewer),
            ("team", vec![C, T], Viewer),
            ("partner", vec![C, C], Viewer),
            ("opponent", vec![C, C], Viewer),
            ("successor", vec![C, C], Viewer),
            ("viewer", vec![C], Viewer),
            ("leader", vec![C], Viewer),
            ("next-actor", vec![C], Viewer),
            ("led-context", vec![Q], Viewer),
            ("current-winner", vec![C], Viewer),
            ("played", vec![D], Viewer),
            ("count", vec![D, N], Viewer),
            ("legal", vec![C, D], World),
            ("forced", vec![C, D], World),
            ("own-legal", vec![D], Viewer),
            ("trick-play", vec![C, D], Viewer),
            ("leads-context", vec![D, Q], Viewer),
            ("declarer", vec![C], Viewer),
            ("bid", vec![N], Viewer),
            ("banked", vec![C, N], Viewer),
            ("to-make", vec![N], Viewer),
            ("to-set", vec![N], Viewer),
            ("trick-number", vec![N], Viewer),
            ("led", vec![C, D], Viewer),
            ("played-to", vec![D, C, D], Viewer),
            ("trick-of", vec![D, N], Viewer),
            ("shown-void", vec![C, Q], Viewer),
            ("stake", vec![N], Viewer),
            ("trump", vec![Q], Viewer),
            ("mine", vec![D], Viewer),
            ("bears", vec![D, Q], Viewer),
            ("high-pip", vec![D, Q], Viewer),
            ("shaker", vec![C], Viewer),
            ("bid-of", vec![C, N], Viewer),
            ("passed", vec![C], Viewer),
            ("no-bid", vec![], Viewer),
            ("high-bid", vec![N], Viewer),
            ("high-bidder", vec![C], Viewer),
            ("legal-bid", vec![N], Viewer),
            ("least-raise", vec![N], Viewer),
            ("scoring-marks", vec![], Viewer),
            ("scoring-points", vec![], Viewer),
            ("score", vec![C, N], Viewer),
            ("short", vec![C, N], Viewer),
            ("short-at-most", vec![C, N], Viewer),
            ("behind", vec![C, N], Viewer),
        ] {
            let phase = if BOTH.contains(&name) {
                Phase::Both
            } else if V4.contains(&name) {
                Phase::Auction
            } else {
                Phase::Play
            };
            let version = if V4.contains(&name) {
                "silas-v4/auction"
            } else if V2.contains(&name) {
                "silas-v2/core"
            } else if V1.contains(&name) {
                "silas-v1/core"
            } else {
                "silas-v0/core"
            };
            let spec = PredicateSpec {
                name: name.to_owned(),
                version: version.to_owned(),
                parameters,
                horizon_plies: 0,
                access,
                phase,
            };
            registry.register(Builtin(spec)).expect("built-in names are unique and valid");
        }
        for outcome in crate::outcome::predicates() {
            registry.register(outcome).expect("outcome names are unique and valid");
        }
        registry
    }
}

struct Builtin(PredicateSpec);

impl Predicate for Builtin {
    fn spec(&self) -> &PredicateSpec {
        &self.0
    }

    fn evaluate(&self, context: PredicateContext<'_>, args: &[Value], _: &mut Budget) -> Result<Option<bool>> {
        use Value::{Chair as C, Context as Q, Domino as D, Number as N};
        let name = self.0.name.as_str();
        if let Some(value) = tile_fact(name, args) {
            return Ok(Some(value));
        }
        if let PredicateContext::Auction(frame) = context {
            return at_auction(frame, name, args);
        }
        let frame = context.frame()?;
        if let Some(value) = auction_fact(frame.auction(), name, args) {
            return Ok(Some(value));
        }
        let physics = frame.physics();
        let kernel = frame.kernel();
        let live = kernel.live();
        let world = || context.world().ok_or_else(|| error("a world predicate requires a world"));
        let value = match (name, args) {
            ("live", [D(d)]) => live.contains(*d),
            ("holds", [C(c), D(d)]) => world()?.holds(*c, *d)?,
            ("in", [D(d), Q(q)]) => physics.follows(*d, *q),
            ("beats", [D(a), D(b), Q(q)]) => physics.trick_key(*a, *q) > physics.trick_key(*b, *q),
            ("boss", [D(d), Q(q)]) => {
                live.contains(*d) && physics.follows(*d, *q) && physics.beats(*q, *d).is_disjoint(live)
            }
            ("master", [D(d)]) => live.contains(*d) && physics.threat(*d).is_disjoint(live),
            ("void", [C(c), Q(q)]) => world()?.hand_within(*c, physics.incidence(*q))?.is_empty(),
            ("quota", [C(c), N(n)]) => kernel.capacity(*c) == *n as usize,
            ("viewer", [C(c)]) => *c == kernel.viewer(),
            ("leader", [C(c)]) => *c == frame.leader(),
            ("next-actor", [C(c)]) => frame.next_actor() == Some(*c),
            ("led-context", [Q(q)]) => match frame.led_context() {
                Some(led) => *q == led,
                None => return Ok(None),
            },
            ("current-winner", [C(c)]) => match frame.current_winner() {
                Some(winner) => *c == winner,
                None => return Ok(None),
            },
            ("played", [D(d)]) => frame.played().contains(*d),
            ("count", [D(d), N(n)]) => physics.count(*d) == *n,
            ("own-legal", [D(d)]) => {
                if frame.next_actor() != Some(frame.viewer()) {
                    return Ok(None);
                }
                physics.legal_plays(kernel.viewer_hand(), frame.lead()).contains(*d)
            }
            ("trick-play", [C(c), D(d)]) => {
                frame.prefix().iter().enumerate().any(|(i, tile)| frame.leader().plus(i) == *c && tile == d)
            }
            ("leads-context", [D(d), Q(q)]) => physics.led_context(*d) == *q,
            ("legal" | "forced", [C(c), D(d)]) => {
                if frame.next_actor() != Some(*c) {
                    return Ok(None);
                }
                let legal = physics.legal_plays(world()?.hand(*c)?, frame.lead());
                if self.0.name == "forced" { legal == crate::DominoSet::single(*d) } else { legal.contains(*d) }
            }
            ("shown-void", [C(c), Q(q)]) => frame.shown_voids(*c).contains(*q),
            ("trump", [Q(q)]) => physics.trump() == Some(*q),
            ("mine", [D(d)]) => kernel.viewer_hand().contains(*d),
            ("declarer", [C(c)]) => *c == frame.contract().declarer(),
            ("bid", [N(n)]) => *n == frame.contract().bid().value(),
            ("stake", [N(n)]) => *n == frame.contract().stake(),
            ("banked", [C(c), N(n)]) => frame.banked()[c.team().index()] == *n,
            ("to-make", [N(n)]) => frame.to_make_and_set().0 == *n,
            ("to-set", [N(n)]) => frame.to_make_and_set().1 == *n,
            ("trick-number", [N(n)]) => match frame.trick_number() {
                Some(number) => number == *n as usize,
                None => return Ok(None),
            },
            ("led", [C(c), D(d)]) => frame.tricks().any(|trick| trick[0] == (*c, *d)),
            ("played-to", [D(lead), C(c), D(d)]) => {
                frame.tricks().any(|trick| trick[0].1 == *lead && trick.contains(&(*c, *d)))
            }
            ("trick-of", [D(d), N(n)]) => {
                let trick = frame.tricks().position(|trick| trick.iter().any(|(_, tile)| tile == d));
                trick.map(|k| k + 1) == Some(*n as usize)
            }
            _ => return Err(error("built-in argument mismatch (an invalid compiled query)")),
        };
        Ok(Some(value))
    }
}

/// The natural pip of a context; none for the called suit.
fn pip_of(q: Context) -> Option<Pip> {
    match q {
        Context::Natural(p) => Some(p),
        Context::Called => None,
    }
}

/// The facts of the tiles and the seats themselves, the same at every frame.
fn tile_fact(name: &str, args: &[Value]) -> Option<bool> {
    use Value::{Chair as C, Context as Q, Domino as D, Team as T};
    Some(match (name, args) {
        ("double", [D(d)]) => d.is_double(),
        ("tile", [D(a), D(b)]) => a == b,
        ("chair", [C(a), C(b)]) => a == b,
        ("context", [Q(a), Q(b)]) => a == b,
        ("team", [C(c), T(t)]) => c.team() == *t,
        ("partner", [C(a), C(b)]) => a.partner() == *b,
        ("opponent", [C(a), C(b)]) => a.team() != b.team(),
        ("successor", [C(a), C(b)]) => a.successor() == *b,
        ("bears", [D(d), Q(q)]) => pip_of(*q).is_some_and(|p| d.has(p)),
        ("high-pip", [D(d), Q(q)]) => pip_of(*q) == Some(d.hi()),
        _ => return None,
    })
}

/// The auction's record, at every frame: complete in the play, under way at
/// the auction. `None` when `name` is not such a fact.
fn auction_fact(auction: &Auction, name: &str, args: &[Value]) -> Option<bool> {
    use Value::{Chair as C, Number as N};
    let high = auction.high();
    Some(match (name, args) {
        ("shaker", [C(c)]) => auction.shaker() == *c,
        ("bid-of", [C(c), N(n)]) => Bid::number(*n).is_ok_and(|bid| auction.call_of(*c) == Some(Call::Bid(bid))),
        ("passed", [C(c)]) => auction.call_of(*c) == Some(Call::Pass),
        ("no-bid", []) => high.is_none(),
        ("high-bid", [N(n)]) => high.is_some_and(|(_, bid)| bid.value() == *n),
        ("high-bidder", [C(c)]) => high.is_some_and(|(seat, _)| seat == *c),
        _ => return None,
    })
}

/// A predicate at the auction frame: the viewer's dealt hand, the count, the
/// auction's record and its legal bids, and the match state.
fn at_auction(frame: &AuctionFrame, name: &str, args: &[Value]) -> Result<Option<bool>> {
    use Value::{Chair as C, Domino as D, Number as N};
    let auction = frame.auction();
    if let Some(value) = auction_fact(auction, name, args) {
        return Ok(Some(value));
    }
    let state = frame.state();
    let side = |c: &Seat| c.team();
    let value = match (name, args) {
        ("viewer", [C(c)]) => *c == frame.viewer(),
        ("mine", [D(d)]) => frame.dealt().contains(*d),
        ("count", [D(d), N(n)]) => frame.count(*d) == *n,
        ("legal-bid" | "least-raise", [N(_)]) if auction.actor() != Some(frame.viewer()) => return Ok(None),
        ("legal-bid", [N(n)]) => Bid::number(*n).is_ok_and(|bid| auction.legal(Call::Bid(bid))),
        ("least-raise", [N(n)]) => auction.least_raise().is_some_and(|bid| bid.value() == *n),
        ("scoring-marks", []) => matches!(state.scoring(), Scoring::Marks { .. }),
        ("scoring-points", []) => matches!(state.scoring(), Scoring::Points { .. }),
        ("score", [C(c), N(n)]) => state.score(side(c)) == *n,
        ("short", [C(c), N(n)]) => state.short(side(c)) == *n,
        ("short-at-most", [C(c), N(n)]) => state.short(side(c)) <= *n,
        ("behind", [C(c), N(n)]) => *n > 0 && state.score(side(c).other()) >= state.score(side(c)) + *n,
        _ => return Err(error(format!("{name} is not defined at the auction (an invalid compiled query)"))),
    };
    Ok(Some(value))
}
