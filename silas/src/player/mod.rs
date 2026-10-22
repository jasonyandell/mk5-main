// SPDX-FileCopyrightText: Jason Yandell
// SPDX-License-Identifier: CC0-1.0

//! The exact player (the exact-player track, 2026-10-22; `wiki/experiments/exact-player-s0-2026-10-22.md`).
//!
//! One seat, its own information only. The other three seats are a declared model: a
//! library of table types, each one seat kernel per other seat, under a prior that is
//! affine in a parameter `p` (the family of [`crate::family`]). The belief is the exact
//! counted posterior over (world, type) atoms: every world of the viewer's fiber, each
//! weighted by the type's prior times the likelihood of every play the other seats have
//! made, replayed in that world. The decision is the exact information-state optimum:
//! from the viewer's turn, every atom is carried through the others' kernels to the
//! viewer's next turn, atoms the viewer cannot tell apart are merged into one block, and
//! the value is a sum over blocks and a maximum over the viewer's own plays only, never a
//! maximum per world (STRATEGY-FUSION FENCE: nothing here is clairvoyant). The course is
//! solved by [`Family`] for every `p` at once, in exact rationals.
//!
//! The endgame is enumerable, so the value is exact: no tail and no bound is consulted.
//! Every budget is a count (kernel reads, atoms), and a budget that would be exceeded
//! refuses; nothing is truncated.

use std::collections::BTreeMap;

use num_rational::BigRational;
use num_traits::{One, Zero};

use crate::family::{Affine, Choice, Family};
use crate::frame::{Frame, World};
use crate::nouns::{Domino, DominoSet, Seat};
use crate::physics::Physics;
use crate::quotient::Block;
use crate::{Result, error};

pub mod recount;

type Q = BigRational;

/// What the viewer sees between two of its decisions: the other seats' plays, in order.
pub type Seen = Vec<Domino>;

/// How one other seat plays, from its own remaining hand and the public record alone.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SeatLaw {
    /// Its lowest legal tile (in index order).
    Lowest,
    /// Its highest legal tile.
    Highest,
    /// Every legal tile with equal probability.
    UniformLegal,
    /// Half the time its lowest legal tile, otherwise uniform: [`SeatLaw::Lowest`] with a
    /// tremble, so every legal play has positive probability and no record is impossible.
    LeanLow,
    /// Half the time its highest legal tile, otherwise uniform.
    LeanHigh,
}

impl SeatLaw {
    pub fn name(self) -> &'static str {
        match self {
            SeatLaw::Lowest => "lowest",
            SeatLaw::Highest => "highest",
            SeatLaw::UniformLegal => "uniform",
            SeatLaw::LeanLow => "lean-low",
            SeatLaw::LeanHigh => "lean-high",
        }
    }

    /// The law of the seat's play, ascending by tile, every probability positive.
    pub fn law(self, physics: &Physics, hand: DominoSet, lead: Option<Domino>) -> Vec<(Domino, Q)> {
        let legal = physics.legal_plays(hand, lead);
        match self {
            SeatLaw::Lowest => legal.iter().next().map(|d| (d, Q::one())).into_iter().collect(),
            SeatLaw::Highest => legal.iter().last().map(|d| (d, Q::one())).into_iter().collect(),
            SeatLaw::UniformLegal => {
                let share = Q::new(1.into(), (legal.len() as i64).into());
                legal.iter().map(|d| (d, share.clone())).collect()
            }
            SeatLaw::LeanLow | SeatLaw::LeanHigh => {
                let n = legal.len() as i64;
                let lean = if self == SeatLaw::LeanLow { legal.iter().next() } else { legal.iter().last() };
                let (rest, top) = (Q::new(1.into(), (2 * n).into()), Q::new((n + 1).into(), (2 * n).into()));
                legal.iter().map(|d| (d, if Some(d) == lean { top.clone() } else { rest.clone() })).collect()
            }
        }
    }
}

/// One type of the table: a kernel for each other seat, indexed by seat.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TableType {
    pub name: String,
    pub seats: [SeatLaw; Seat::COUNT],
}

impl TableType {
    /// Every other seat played by `law`.
    pub fn all(law: SeatLaw) -> TableType {
        TableType { name: law.name().to_string(), seats: [law; Seat::COUNT] }
    }
}

/// The declared model of the other seats: types and a prior over them, each weight
/// `at0 (1 - p) + at1 p`, nonnegative at both ends and summing to one at both.
#[derive(Clone, Debug)]
pub struct Library {
    types: Vec<TableType>,
    prior: Vec<Affine>,
}

impl Library {
    pub fn new(types: Vec<TableType>, prior: Vec<Affine>) -> Result<Library> {
        if types.is_empty() || types.len() != prior.len() {
            return Err(error("a library has one prior weight per type, and at least one type"));
        }
        if prior.iter().any(|w| w.at0 < Q::zero() || w.at1 < Q::zero()) {
            return Err(error("a prior weight is negative"));
        }
        let total = prior.iter().fold(Affine::zero(), |s, w| s.plus(w));
        if total != Affine::new(Q::one(), Q::one()) {
            return Err(error("the prior does not sum to one at both ends"));
        }
        Ok(Library { types, prior })
    }

    /// One type, certain.
    pub fn one(t: TableType) -> Library {
        Library { types: vec![t], prior: vec![Affine::new(Q::one(), Q::one())] }
    }

    pub fn types(&self) -> &[TableType] {
        &self.types
    }

    /// The prior does not move with `p`: one parameter class.
    pub fn constant(&self) -> bool {
        self.prior.iter().all(|w| w.at0 == w.at1)
    }
}

/// How the belief at the root reads the record.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Root {
    /// The exact posterior: prior times the likelihood of the other seats' every play.
    Posterior,
    /// Every world of the fiber with its type's prior alone: the record enters only
    /// through the voids it shows (Walt's belief kind, and the note's fixture).
    FiberUniform,
}

/// When the viewer's success is settled.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Settle {
    /// As soon as the contract is made or set.
    Decided,
    /// When the last tile is played.
    Hand,
}

/// The declared deterministic budget: a decision that would read the kernels more often
/// refuses.
#[derive(Clone, Copy, Debug)]
pub struct Budget {
    pub reads: u64,
}

/// What a decision cost, in counts.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Counts {
    /// Worlds of the fiber, and the atoms of positive weight at the root.
    pub worlds: usize,
    pub atoms: usize,
    /// Kernel reads: the likelihood of the record, then the course.
    pub reads: u64,
    /// The course's viewer decisions.
    pub choices: usize,
}

/// The public record of the hand as play goes on, shared by every atom of a decision.
#[derive(Clone, Debug)]
struct Table {
    played: DominoSet,
    trick: Vec<Domino>,
    leader: Seat,
    banked: [u32; 2],
}

impl Table {
    fn actor(&self) -> Seat {
        self.leader.plus(self.trick.len())
    }

    fn play(&self, physics: &Physics, d: Domino) -> Table {
        let mut t = self.clone();
        t.played.insert(d);
        t.trick.push(d);
        if let Ok(plays) = <[Domino; 4]>::try_from(t.trick.as_slice()) {
            let w = physics.trick_winner(t.leader, plays);
            t.banked[w.team().index()] += physics.trick_points(plays);
            t.leader = w;
            t.trick.clear();
        }
        t
    }

    fn done(&self) -> bool {
        self.played.len() == Domino::COUNT
    }
}

/// One atom of the belief: a world of the fiber, a type, and its weight.
#[derive(Clone, Debug)]
struct Atom {
    world: u32,
    kind: u16,
    weight: Affine,
}

fn scaled(w: &Affine, x: &Q) -> Affine {
    Affine::new(&w.at0 * x, &w.at1 * x)
}

fn mass(atoms: &[Atom]) -> Affine {
    atoms.iter().fold(Affine::zero(), |m, a| m.plus(&a.weight))
}

/// The exact belief at a frame: the fiber's worlds and the atoms of positive weight.
pub struct Belief {
    worlds: Vec<World>,
    atoms: Vec<Atom>,
    reads: u64,
}

impl Belief {
    pub fn new(frame: &Frame, library: &Library, root: Root) -> Result<Belief> {
        let physics = frame.physics();
        let worlds: Vec<World> = physics.worlds(frame.kernel()).collect();
        let mut atoms = Vec::new();
        let mut reads = 0u64;
        for (w, world) in worlds.iter().enumerate() {
            for (k, t) in library.types.iter().enumerate() {
                let prior = &library.prior[k];
                if prior.at0.is_zero() && prior.at1.is_zero() {
                    continue;
                }
                let like = match root {
                    Root::FiberUniform => Q::one(),
                    Root::Posterior => likelihood(frame, world, t, &mut reads),
                };
                if !like.is_zero() {
                    atoms.push(Atom { world: w as u32, kind: k as u16, weight: scaled(prior, &like) });
                }
            }
        }
        if atoms.is_empty() {
            return Err(error("the record has probability zero under every type of the library"));
        }
        Ok(Belief { worlds, atoms, reads })
    }

    /// The belief's total weight: the probability of the record, affine in `p`.
    pub fn mass(&self) -> Affine {
        mass(&self.atoms)
    }

    pub fn worlds(&self) -> usize {
        self.worlds.len()
    }

    pub fn atoms(&self) -> usize {
        self.atoms.len()
    }
}

/// The probability, under type `t` in `world`, of every play the other seats made.
fn likelihood(frame: &Frame, world: &World, t: &TableType, reads: &mut u64) -> Q {
    let physics = frame.physics();
    let viewer = frame.viewer();
    let mut held: [DominoSet; Seat::COUNT] = Seat::ALL.map(|s| world.hand(s));
    for &(s, d) in frame.history() {
        held[s.index()].insert(d);
    }
    let mut like = Q::one();
    for trick in frame.tricks() {
        let lead = trick.first().map(|(_, d)| *d);
        for (i, &(s, d)) in trick.iter().enumerate() {
            if s != viewer {
                *reads += 1;
                let law = t.seats[s.index()].law(physics, held[s.index()], if i == 0 { None } else { lead });
                match law.iter().find(|(x, _)| *x == d) {
                    Some((_, q)) => like *= q,
                    None => return Q::zero(),
                }
            }
            held[s.index()].remove(d);
        }
    }
    like
}

/// The solved decision: the course for every `p`, its root, and what it cost.
pub struct Solved {
    pub family: Family<Domino, Seen>,
    pub root: usize,
    pub counts: Counts,
    viewer_bidding: bool,
}

impl Solved {
    /// The root menu.
    pub fn menu(&self) -> &[Domino] {
        &self.family.choice(self.root).menu
    }

    /// Each root play's value at `p`, the viewer side's success probability given the
    /// record; `None` where the record has no mass at `p`.
    pub fn values(&self, p: &Q) -> Option<Vec<Q>> {
        let m = self.family.choice(self.root).mass.at(p);
        (!m.is_zero()).then(|| self.family.value(self.root).iter().map(|v| v.at(p) / &m).collect())
    }

    /// The maximizing set at `p`.
    pub fn best(&self, p: &Q) -> Vec<Domino> {
        self.family.best_at(self.root, p)
    }

    /// The act: the least maximizing tile.
    pub fn tile(&self, p: &Q) -> Domino {
        self.best(p)[0]
    }

    /// Whether the viewer's side is the declaring side.
    pub fn viewer_bidding(&self) -> bool {
        self.viewer_bidding
    }
}

struct Builder<'a> {
    physics: &'a Physics,
    viewer: Seat,
    worlds: &'a [World],
    types: &'a [TableType],
    settle: Settle,
    need: u32,
    declaring: usize,
    viewer_bidding: bool,
    budget: Budget,
    reads: u64,
    choices: Vec<Choice<Domino, Seen>>,
}

impl Builder<'_> {
    /// The viewer's side's success, once settled.
    fn settled(&self, t: &Table) -> Option<bool> {
        let total = self.physics.total_points();
        let made = t.banked[self.declaring] >= self.need;
        let set = t.banked[1 - self.declaring] > total - self.need;
        let over = match self.settle {
            Settle::Decided => made || set || t.done(),
            Settle::Hand => t.done(),
        };
        over.then_some(made == self.viewer_bidding)
    }

    /// Each atom's law at the seat to play, the atoms split by the tile played.
    fn split(&mut self, t: &Table, atoms: Vec<Atom>) -> Result<BTreeMap<Domino, Vec<Atom>>> {
        let s = t.actor();
        let lead = t.trick.first().copied();
        let mut out: BTreeMap<Domino, Vec<Atom>> = BTreeMap::new();
        for a in atoms {
            self.reads += 1;
            if self.reads > self.budget.reads {
                return Err(error(format!("the decision needs more than {} kernel reads", self.budget.reads)));
            }
            let hand = self.worlds[a.world as usize].hand(s).difference(t.played);
            let law = self.types[a.kind as usize].seats[s.index()].law(self.physics, hand, lead);
            if let [(d, _)] = law.as_slice() {
                out.entry(*d).or_default().push(a);
                continue;
            }
            for (d, q) in law {
                out.entry(d).or_default().push(Atom { world: a.world, kind: a.kind, weight: scaled(&a.weight, &q) });
            }
        }
        Ok(out)
    }

    /// The viewer side's success mass when the others play the hand out.
    fn playout(&mut self, t: &Table, atoms: Vec<Atom>) -> Result<Affine> {
        if let Some(u) = self.settled(t) {
            return Ok(if u { mass(&atoms) } else { Affine::zero() });
        }
        let mut total = Affine::zero();
        for (d, sub) in self.split(t, atoms)? {
            total = total.plus(&self.playout(&t.play(self.physics, d), sub)?);
        }
        Ok(total)
    }

    /// The others play until the viewer's turn: each block of what the viewer saw.
    fn observe(
        &mut self,
        t: Table,
        atoms: Vec<Atom>,
        seen: Seen,
        out: &mut Vec<(Seen, Table, Vec<Atom>)>,
    ) -> Result<()> {
        if t.actor() == self.viewer || self.settled(&t).is_some() {
            out.push((seen, t, atoms));
            return Ok(());
        }
        for (d, sub) in self.split(&t, atoms)? {
            let mut s = seen.clone();
            s.push(d);
            self.observe(t.play(self.physics, d), sub, s, out)?;
        }
        Ok(())
    }

    /// The viewer's decision at `t` over `atoms`: its index in the course.
    fn decide(&mut self, t: Table, hand: DominoSet, atoms: Vec<Atom>) -> Result<usize> {
        let menu: Vec<Domino> = self.physics.legal_plays(hand, t.trick.first().copied()).iter().collect();
        let i = self.choices.len();
        let m = mass(&atoms);
        self.choices.push(Choice { menu: menu.clone(), mass: m.clone(), payoff: vec![], blocks: vec![] });
        if let Some(u) = self.settled(&t) {
            let pay = if u { m } else { Affine::zero() };
            self.choices[i].payoff = vec![pay; menu.len()];
            return Ok(i);
        }
        let (mut payoff, mut blocks) = (Vec::with_capacity(menu.len()), Vec::new());
        for &a in &menu {
            let after = t.play(self.physics, a);
            let mut left = hand;
            left.remove(a);
            if left.is_empty() {
                payoff.push(self.playout(&after, atoms.clone())?);
                continue;
            }
            payoff.push(Affine::zero());
            let mut cells = Vec::new();
            self.observe(after, atoms.clone(), Vec::new(), &mut cells)?;
            for (seen, t2, sub) in cells {
                let next = self.decide(t2, left, sub)?;
                blocks.push(Block { play: a, seen, next });
            }
        }
        self.choices[i].payoff = payoff;
        self.choices[i].blocks = blocks;
        Ok(i)
    }
}

/// The exact decision at `frame`, the viewer to play, for every `p` at once.
pub fn solve(frame: &Frame, library: &Library, belief: &Belief, settle: Settle, budget: Budget) -> Result<Solved> {
    if frame.next_actor() != Some(frame.viewer()) {
        return Err(error("the viewer is not to play"));
    }
    let physics = frame.physics();
    let contract = frame.contract();
    let declaring = contract.declarer().team().index();
    let viewer_bidding = frame.viewer().team() == contract.declarer().team();
    let table = Table {
        played: frame.played(),
        trick: frame.prefix().to_vec(),
        leader: frame.leader(),
        banked: frame.banked(),
    };
    let hand = frame.kernel().viewer_hand();
    let mut b = Builder {
        physics,
        viewer: frame.viewer(),
        worlds: &belief.worlds,
        types: &library.types,
        settle,
        need: contract.threshold(),
        declaring,
        viewer_bidding,
        budget,
        reads: belief.reads,
        choices: Vec::new(),
    };
    let root = b.decide(table, hand, belief.atoms.clone())?;
    let counts =
        Counts { worlds: belief.worlds.len(), atoms: belief.atoms.len(), reads: b.reads, choices: b.choices.len() };
    Ok(Solved { family: Family::new(b.choices)?, root, counts, viewer_bidding })
}

/// One hand played on from a frame to its last tile: the viewer by the exact player,
/// re-planning at each of its turns from its own frame alone; the other seats by one type
/// of the library in the true deal, a law with more than one tile drawn by a seeded
/// generator. Once the contract is decided (under [`Settle::Decided`]) the viewer plays
/// its lowest legal tile without deciding.
pub struct Played {
    pub history: Vec<(Seat, Domino)>,
    /// The viewer's side got its way: the declaring side made, or the other side set.
    pub success: bool,
    /// The viewer's decisions, and how many of them had, in the previous decision's course,
    /// the same mass and the same maximizing set at `p` (every decision after the first,
    /// under the exact posterior: Bayes' rule and Bellman's, checked).
    pub decisions: usize,
    pub consistent: usize,
}

/// The seat's draw from `law` at play number `n`: splitmix64 of the seed and `n`, read as
/// a fraction of 2^64 and compared exactly with the law's running sum.
fn draw(law: &[(Domino, Q)], seed: u64, n: usize) -> Domino {
    if let [(d, _)] = law {
        return *d;
    }
    let mut z = (seed ^ (n as u64).wrapping_mul(0xd1b5_4a32_d192_ed03)).wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^= z >> 31;
    let u = Q::new(z.into(), num_bigint::BigInt::from(1u8) << 64u32);
    let mut sum = Q::zero();
    for (d, q) in law {
        sum += q;
        if u < sum {
            return *d;
        }
    }
    law.last().expect("a legal tile").0
}

/// Plays the hand on from `start`, the true deal `truth`, the others by `library`'s type
/// number `table`. `visit` sees every decision with its wall time (reported, never a
/// budget).
#[allow(clippy::too_many_arguments)]
pub fn play_hand(
    start: &Frame,
    truth: &World,
    library: &Library,
    table: usize,
    p: &Q,
    root: Root,
    settle: Settle,
    budget: Budget,
    seed: u64,
    mut visit: impl FnMut(&Frame, &Solved, std::time::Duration) -> Result<()>,
) -> Result<Played> {
    let kind = library.types.get(table).ok_or_else(|| error("no such type in the library"))?;
    let viewer = start.viewer();
    let (mut decisions, mut consistent) = (0, 0);
    let mut last: Option<(Solved, Domino)> = None;
    let mut seen: Seen = Vec::new();
    let mut frame = start.clone();
    while let Some(actor) = frame.next_actor() {
        let physics = frame.physics();
        let (to_make, to_set) = frame.to_make_and_set();
        let decided = settle == Settle::Decided && (to_make == 0 || to_set == 0);
        let tile = if actor != viewer {
            let hand = truth.hand(actor).difference(frame.played());
            let d = draw(&kind.seats[actor.index()].law(physics, hand, frame.lead()), seed, frame.plays().len());
            seen.push(d);
            d
        } else if decided {
            physics.legal_plays(frame.kernel().viewer_hand(), frame.lead()).iter().next().expect("a legal tile")
        } else {
            let t0 = std::time::Instant::now();
            let belief = Belief::new(&frame, library, root)?;
            let solved = solve(&frame, library, &belief, settle, budget)?;
            let wall = t0.elapsed();
            if let Some((prev, a)) = last.take() {
                let next = prev
                    .family
                    .choice(prev.root)
                    .blocks
                    .iter()
                    .find(|b| b.play == a && b.seen == seen)
                    .ok_or_else(|| error("the previous course has no block for what was seen"))?
                    .next;
                if prev.family.choice(next).mass == belief.mass() && prev.family.best_at(next, p) == solved.best(p) {
                    consistent += 1;
                }
            }
            visit(&frame, &solved, wall)?;
            decisions += 1;
            let d = solved.tile(p);
            last = Some((solved, d));
            seen.clear();
            d
        };
        frame = frame.after(tile)?;
    }
    let contract = frame.contract();
    let made = contract.makes(frame.banked()[contract.declarer().team().index()]);
    let success = made == (viewer.team() == contract.declarer().team());
    Ok(Played { history: frame.history().to_vec(), success, decisions, consistent })
}

/// Positions and libraries read from JSON, and each decision written as one JSON row that
/// `lem/exact_ref.py check` re-derives. Tiles are domino indices (`h(h+1)/2 + l`, the
/// forge ids); seats 0..3; `decl` is forge's declaration id (0..6 a pip suit, 7 doubles,
/// 9 no-trump).
#[cfg(feature = "course")]
pub mod file {
    use num_rational::BigRational;
    use num_traits::Zero;
    use serde_json::{Value, json};

    use super::{Library, Root, SeatLaw, Settle, Solved, TableType};
    use crate::auction::Auction;
    use crate::family::{Affine, Reach, representatives};
    use crate::frame::{Bid, Frame, World};
    use crate::nouns::{Domino, DominoSet, Seat};
    use crate::quotient::Interface;
    use crate::reference::Straight42;
    use crate::{Result, error};

    fn law(name: &str) -> Result<SeatLaw> {
        [SeatLaw::Lowest, SeatLaw::Highest, SeatLaw::UniformLegal, SeatLaw::LeanLow, SeatLaw::LeanHigh]
            .into_iter()
            .find(|l| l.name() == name)
            .ok_or_else(|| error(format!("no seat law {name}")))
    }

    fn rational(v: &Value) -> Result<BigRational> {
        v.as_str().and_then(|s| s.parse().ok()).ok_or_else(|| error("a rational is a string such as \"1/3\""))
    }

    /// `{"types": [{"name", "seats": [law × 4]}], "prior": [[at 0, at 1], ...]}`; the
    /// viewer's own entry is never read.
    pub fn library(v: &Value) -> Result<Library> {
        let types = v
            .get("types")
            .and_then(Value::as_array)
            .ok_or_else(|| error("a library has types"))?
            .iter()
            .map(|t| {
                let seats = t.get("seats").and_then(Value::as_array).filter(|s| s.len() == 4);
                let seats = seats.ok_or_else(|| error("a type names a law for each of the four seats"))?;
                let mut out = [SeatLaw::Lowest; Seat::COUNT];
                for (i, s) in seats.iter().enumerate() {
                    out[i] = law(s.as_str().unwrap_or(""))?;
                }
                Ok(TableType { name: t.get("name").and_then(Value::as_str).unwrap_or("").to_string(), seats: out })
            })
            .collect::<Result<Vec<_>>>()?;
        let prior = v
            .get("prior")
            .and_then(Value::as_array)
            .ok_or_else(|| error("a library has a prior"))?
            .iter()
            .map(|w| match w.as_array().map(Vec::as_slice) {
                Some([a, b]) => Ok(Affine::new(rational(a)?, rational(b)?)),
                _ => Err(error("a prior weight is a pair [at 0, at 1]")),
            })
            .collect::<Result<Vec<_>>>()?;
        Library::new(types, prior)
    }

    pub fn root(name: &str) -> Result<Root> {
        match name {
            "posterior" => Ok(Root::Posterior),
            "fiber" => Ok(Root::FiberUniform),
            _ => Err(error("the root is posterior or fiber")),
        }
    }

    pub fn settle(name: &str) -> Result<Settle> {
        match name {
            "decided" => Ok(Settle::Decided),
            "hand" => Ok(Settle::Hand),
            _ => Err(error("settle is decided or hand")),
        }
    }

    /// A position: `{"id", "decl", "bidder", "bid", "hands": [dealt mask × 4], "plays": [[seat, tile], ...]}`.
    pub struct Position {
        pub id: Value,
        pub row: Value,
        pub truth: World,
        /// The frame of the seat to play.
        pub frame: Frame,
    }

    fn uint(v: Option<&Value>, what: &str) -> Result<u64> {
        v.and_then(Value::as_u64).ok_or_else(|| error(format!("a position's {what} is a number")))
    }

    fn tile(n: u64) -> Result<Domino> {
        usize::try_from(n).ok().and_then(Domino::from_index).ok_or_else(|| error("a tile is 0..27"))
    }

    impl Position {
        pub fn from_json(text_in: &str) -> Result<Position> {
            let row: Value = serde_json::from_str(text_in).map_err(|e| error(format!("position: {e}")))?;
            let decl = match uint(row.get("decl"), "decl")? {
                d @ 0..=6 => format!("P{d}"),
                7 => "DT".to_string(),
                9 => "NT".to_string(),
                d => return Err(error(format!("declaration {d} is not a straight declaration"))),
            };
            let physics = decl.parse::<Straight42>()?.physics();
            let bidder =
                Seat::from_index(uint(row.get("bidder"), "bidder")? as usize).ok_or_else(|| error("a seat is 0..3"))?;
            let bid = Bid::number(uint(row.get("bid"), "bid")? as u32)?;
            let hands: Vec<DominoSet> = row
                .get("hands")
                .and_then(Value::as_array)
                .filter(|h| h.len() == 4)
                .ok_or_else(|| error("a position has four hands"))?
                .iter()
                .map(|h| DominoSet::from_bits(uint(Some(h), "hand")? as u32).ok_or_else(|| error("a hand is a mask")))
                .collect::<Result<_>>()?;
            let truth = World::deal([hands[0], hands[1], hands[2], hands[3]])?;
            let plays: Vec<(Seat, Domino)> = row
                .get("plays")
                .and_then(Value::as_array)
                .ok_or_else(|| error("a position has plays"))?
                .iter()
                .map(|p| match p.as_array().map(Vec::as_slice) {
                    Some([s, t]) => Ok((
                        Seat::from_index(uint(Some(s), "seat")? as usize).ok_or_else(|| error("a seat is 0..3"))?,
                        tile(uint(Some(t), "tile")?)?,
                    )),
                    _ => Err(error("a play is [seat, tile]")),
                })
                .collect::<Result<_>>()?;
            let auction = Auction::won_by(bidder, bid)?;
            let tiles: Vec<Domino> = plays.iter().map(|(_, d)| *d).collect();
            let actor = Frame::new(auction.clone(), physics.clone(), bidder, hands[bidder.index()], tiles.clone())?
                .next_actor()
                .ok_or_else(|| error("the hand is over"))?;
            let frame = Frame::new(auction, physics, actor, hands[actor.index()], tiles)?;
            if frame.history() != plays.as_slice() {
                return Err(error("a play's seat disagrees with the record"));
            }
            Ok(Position { id: row.get("id").cloned().unwrap_or(Value::Null), row, truth, frame })
        }
    }

    fn masks(hands: &World) -> Vec<u32> {
        Seat::ALL.iter().map(|s| hands.hand(*s).bits()).collect()
    }

    /// One decision's row: the position at this decision, the declared model, the root's
    /// exact values and maximizing set in every parameter class, the act at `p`, the counts
    /// and the wall time; with `quotient` at `Some(max)`, the controller's states (every,
    /// path, plan; raw and fork) and root ladders, or where the course has more than `max`
    /// parameter classes, only that count (`quotient_skipped`). With `recount` n > 0, the second construction
    /// ([`super::recount`]) is run at n parameter classes (all of them when there are at
    /// most n, else n spaced evenly from the first to the last) and must agree exactly in
    /// mass, value and maximizing set, or the row is refused.
    #[allow(clippy::too_many_arguments)]
    pub fn row(
        position: &Position,
        frame: &Frame,
        library_spec: &Value,
        library: &Library,
        root: Root,
        root_name: &str,
        settle_name: &str,
        solved: &Solved,
        p: &BigRational,
        wall_ns: u128,
        quotient: Option<usize>,
        recount: usize,
    ) -> Result<Value> {
        let f = &solved.family;
        let reach = f.reachable(&[solved.root]);
        let thresholds = if library.constant() { Vec::new() } else { f.thresholds(&reach) };
        let reps = if library.constant() { vec![BigRational::zero()] } else { representatives(&thresholds) };
        let mut recounted = 0usize;
        let checked: Vec<&BigRational> = match (recount, reps.len()) {
            (0, _) => Vec::new(),
            (n, k) if k <= n => reps.iter().collect(),
            (1, _) => vec![&reps[0]],
            (n, k) => (0..n).map(|i| &reps[i * (k - 1) / (n - 1)]).collect(),
        };
        for r in checked {
            let mass = f.choice(solved.root).mass.at(r);
            if mass.is_zero() {
                continue;
            }
            let (m, v) = super::recount::values(frame, library, root, r)?;
            let best: Vec<Domino> = {
                let top = v.iter().map(|(_, x)| x).max().expect("a tile");
                v.iter().filter(|(_, x)| x == top).map(|(d, _)| *d).collect()
            };
            let mine = solved.values(r).expect("positive mass");
            if m != mass
                || v.iter().map(|(d, _)| *d).collect::<Vec<_>>() != solved.menu()
                || v.iter().map(|(_, x)| x.clone()).collect::<Vec<_>>() != mine
                || best != solved.best(r)
            {
                return Err(error(format!("the second construction disagrees at p = {r}")));
            }
            recounted += 1;
        }
        let ids = |v: &[Domino]| v.iter().map(|d| d.index()).collect::<Vec<_>>();
        let classes: Vec<Value> = reps
            .iter()
            .map(|r| {
                let mass = f.choice(solved.root).mass.at(r);
                let value: Vec<String> =
                    solved.values(r).map(|v| v.iter().map(|x| x.to_string()).collect()).unwrap_or_default();
                json!({"p": r.to_string(), "mass": mass.to_string(), "value": value, "best": ids(&solved.best(r))})
            })
            .collect();
        let mut out = json!({
            "kind": "decision",
            "id": position.id,
            "decl": position.row["decl"], "bidder": position.row["bidder"], "bid": position.row["bid"],
            "hands": masks(&position.truth),
            "plays": frame.history().iter().map(|(s, d)| [s.index(), d.index()]).collect::<Vec<_>>(),
            "viewer": frame.viewer().index(),
            "own": frame.kernel().viewer_hand().len(),
            "library": library_spec, "root": root_name, "settle": settle_name,
            "menu": ids(solved.menu()),
            "thresholds": thresholds.iter().map(|t| t.to_string()).collect::<Vec<_>>(),
            "classes": classes,
            "p": p.to_string(), "tile": solved.tile(p).index(),
            "counts": {"worlds": solved.counts.worlds, "atoms": solved.counts.atoms,
                       "reads": solved.counts.reads, "choices": solved.counts.choices},
            "wall_ns": wall_ns as u64,
            "recounted": recounted,
        });
        if quotient.is_some_and(|max| reps.len() > max) {
            out["quotient_skipped"] = Value::from(reps.len());
        } else if quotient.is_some() {
            let mut q = serde_json::Map::new();
            for (name, reach) in [("every", Reach::Every), ("path", Reach::Path), ("plan", Reach::Plan)] {
                let (course, roots) = f.controller(solved.root, &reps, reach)?;
                let mut by = serde_json::Map::new();
                by.insert("contexts".to_string(), Value::from(course.decisions().len()));
                for iface in [Interface::Raw, Interface::Fork] {
                    let qt = course.quotient(iface)?;
                    by.insert(
                        iface.name().to_string(),
                        json!({"states": qt.states, "root_ladder": qt.ladder_of(&roots)}),
                    );
                }
                q.insert(name.to_string(), Value::Object(by));
            }
            out["quotient"] = Value::Object(q);
        }
        Ok(out)
    }
}
