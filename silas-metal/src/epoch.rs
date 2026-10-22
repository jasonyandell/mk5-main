// SPDX-FileCopyrightText: Jason Yandell
// SPDX-License-Identifier: CC0-1.0

//! The field epoch: from one focal action to the focal seat's next decision (or the hand's
//! end), the declared field played in bulk. Its input is a batch of work items, each a
//! segment (one information state and one focal action, everything public) and a particle
//! (one hidden world under one table type, with an exact mass); its output is every line the
//! field can play, one child per line, in a fixed order: work item by work item, and within
//! one, depth first with tiles ascending. A child's mass is its parent's times the line's
//! likelihood at [`SCALE`] per field play, so a mass after `r` field plays from the root
//! means `mass / (W * SCALE^r)`. Nothing is sampled or truncated.
//!
//! Three constructions compute the same children: [`Reference`] from Silas's [`Physics`] and
//! the laws as exact rationals (the authority), [`Cpu`] from the tables as integers (the
//! shaders' scalar mirror, threaded), and [`crate::metal::Gpu`] on Metal. They agree bit for
//! bit, or the batch is refused.

use num_rational::BigRational;
use num_traits::Zero;
use silas::{Domino, DominoSet, Physics, Seat};

use crate::tables::Tables;
use crate::wide::U256;

/// Every law's probabilities are multiples of 1/840 on legal sets of one to seven tiles:
/// 840 = lcm(2n) for n in 1..=7.
pub const SCALE: u32 = 840;

/// Field plays between two focal plays: the rest of the focal's trick and the start of the
/// next, at most 3 + 3.
pub const MAX_FIELD: usize = 6;

/// The empty observation: six five-bit fields, each 31.
pub const NO_OBS: u32 = 0x3fff_ffff;

/// How one other seat plays from its own hand and the trick so far (the exact player's
/// seat laws, `silas::player::SeatLaw`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum Law {
    /// Its lowest legal tile, in index order.
    Lowest = 0,
    /// Its highest legal tile.
    Highest = 1,
    /// Every legal tile with equal probability.
    Uniform = 2,
    /// Half the time its lowest legal tile, otherwise uniform.
    LeanLow = 3,
    /// Half the time its highest legal tile, otherwise uniform.
    LeanHigh = 4,
}

impl Law {
    pub const ALL: [Law; 5] = [Law::Lowest, Law::Highest, Law::Uniform, Law::LeanLow, Law::LeanHigh];

    /// The law's probability on each legal tile, exactly, ascending by tile.
    pub fn law(self, legal: DominoSet) -> Vec<(Domino, BigRational)> {
        let n = legal.len() as i64;
        let q = |a: i64, b: i64| BigRational::new(a.into(), b.into());
        let (lo, hi) = (legal.iter().next(), legal.iter().last());
        legal
            .iter()
            .map(|d| {
                let p = match self {
                    Law::Lowest => if Some(d) == lo { q(1, 1) } else { q(0, 1) },
                    Law::Highest => if Some(d) == hi { q(1, 1) } else { q(0, 1) },
                    Law::Uniform => q(1, n),
                    Law::LeanLow => if Some(d) == lo { q(n + 1, 2 * n) } else { q(1, 2 * n) },
                    Law::LeanHigh => if Some(d) == hi { q(n + 1, 2 * n) } else { q(1, 2 * n) },
                };
                (d, p)
            })
            .filter(|(_, p)| !p.is_zero())
            .collect()
    }
}

/// One information state and one focal action: everything public. `trick` holds the open
/// trick's first `n` plays; the focal seat is the next to play.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Segment {
    pub focal: u8,
    pub leader: u8,
    pub n: u8,
    pub trick: [u8; 4],
    pub done: u8,
    pub focal_hand: u32,
    pub action: u8,
}

pub const SEG_WORDS: usize = 12;

impl Segment {
    pub fn words(&self) -> [u32; SEG_WORDS] {
        let mut w = [0u32; SEG_WORDS];
        w[0] = u32::from(self.focal);
        w[1] = u32::from(self.leader);
        w[2] = u32::from(self.n);
        for i in 0..4 {
            w[3 + i] = u32::from(self.trick[i]);
        }
        w[7] = u32::from(self.done);
        w[8] = self.focal_hand;
        w[9] = u32::from(self.action);
        w
    }
}

/// One hidden world under one table type, with its exact mass. The focal seat's entry of
/// `hands` is ignored (the segment carries the focal hand). `world` is carried through
/// untouched: a label for the caller (the world-split control groups by it).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Particle {
    pub hands: [u32; 4],
    pub kind: u32,
    pub world: u32,
    pub mass: U256,
}

pub const PART_WORDS: usize = 16;

impl Particle {
    pub fn words(&self) -> [u32; PART_WORDS] {
        let mut w = [0u32; PART_WORDS];
        w[..4].copy_from_slice(&self.hands);
        w[4] = self.kind;
        w[5] = self.world;
        w[8..].copy_from_slice(&self.mass.0);
        w
    }
}

/// One line of one work item: the field plays seen (five bits each, 31 after the last), the
/// hands left (the focal entry zero), and the mass times the line's likelihood.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Child {
    pub item: u32,
    pub obs: u32,
    pub hands: [u32; 4],
    pub kind: u32,
    pub world: u32,
    pub mass: U256,
}

pub const CHILD_WORDS: usize = 16;

impl Child {
    pub fn from_words(w: &[u32]) -> Child {
        Child {
            item: w[0],
            obs: w[1],
            hands: [w[2], w[3], w[4], w[5]],
            kind: w[6],
            world: w[7],
            mass: U256(std::array::from_fn(|i| w[8 + i])),
        }
    }

    /// The field plays seen, in order.
    pub fn seen(&self) -> Vec<Domino> {
        obs_tiles(self.obs)
    }
}

/// The tiles of an observation, in order.
pub fn obs_tiles(obs: u32) -> Vec<Domino> {
    (0..MAX_FIELD)
        .map(|i| (obs >> (5 * i)) & 31)
        .take_while(|t| *t != 31)
        .map(|t| Domino::from_index(t as usize).expect("a tile index"))
        .collect()
}

/// A batch of work: items name a segment and a particle by index.
#[derive(Clone, Copy, Debug)]
pub struct Batch<'a> {
    pub segments: &'a [Segment],
    pub particles: &'a [Particle],
    pub items: &'a [(u32, u32)],
}

/// Why a batch was refused. Nothing partial is returned.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EpochError {
    Overflow,
    Depth,
    Count,
    Illegal,
    Law,
    Device(String),
    /// The host would hold more than its leaf budget: a memory refusal, reported, not a crash.
    Memory { leaves: usize, budget: usize },
}

impl EpochError {
    pub fn from_code(code: u32) -> Option<EpochError> {
        match code {
            0 => None,
            1 => Some(EpochError::Overflow),
            2 => Some(EpochError::Depth),
            3 => Some(EpochError::Count),
            4 => Some(EpochError::Illegal),
            5 => Some(EpochError::Law),
            c => Some(EpochError::Device(format!("unknown error code {c}"))),
        }
    }
}

impl std::fmt::Display for EpochError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "the epoch was refused: {self:?}")
    }
}

impl std::error::Error for EpochError {}

/// A way to run epochs. Every backend returns the same children in the same order.
pub trait Backend {
    fn name(&self) -> &'static str;
    fn epoch(&mut self, tables: &Tables, batch: Batch<'_>) -> Result<Vec<Child>, EpochError>;
}

// ---------------------------------------------------------------------------------------
// The scalar mirror: the shader's integer algorithm on the CPU, threaded.

#[derive(Clone, Copy)]
struct State {
    hands: [u32; 4],
    leader: u32,
    n: u32,
    t: [u32; 4],
    done: u32,
    obs: u32,
    nobs: u32,
}

fn legal(tab: &Tables, s: &State, seat: u32) -> u32 {
    let hand = s.hands[seat as usize];
    if s.n == 0 {
        return hand;
    }
    let follow = hand & tab.incidence(tab.led(s.t[0]));
    if follow != 0 { follow } else { hand }
}

fn play(tab: &Tables, s: &mut State, tile: u32) {
    let seat = (s.leader + s.n) & 3;
    s.hands[seat as usize] &= !(1u32 << tile);
    s.t[s.n as usize] = tile;
    s.n += 1;
    if s.n == 4 {
        let led = tab.led(s.t[0]);
        let mut best = 0;
        for i in 1..4 {
            if tab.rank(led, s.t[i]) > tab.rank(led, s.t[best]) {
                best = i;
            }
        }
        s.leader = (s.leader + best as u32) & 3;
        s.n = 0;
        s.done += 1;
    }
}

fn support(law: u32, legal: u32) -> u32 {
    match law {
        0 => 1 << legal.trailing_zeros(),
        1 => 1 << (31 - legal.leading_zeros()),
        _ => legal,
    }
}

fn weight(law: u32, legal: u32, tile: u32) -> Result<u32, EpochError> {
    let n = legal.count_ones();
    let (lo, hi) = (legal.trailing_zeros(), 31 - legal.leading_zeros());
    Ok(match law {
        0 => if tile == lo { 840 } else { 0 },
        1 => if tile == hi { 840 } else { 0 },
        2 => 840 / n,
        3 => if tile == lo { 420 * (n + 1) / n } else { 420 / n },
        4 => if tile == hi { 420 * (n + 1) / n } else { 420 / n },
        _ => return Err(EpochError::Law),
    })
}

fn mirror_item(tab: &Tables, batch: &Batch<'_>, item: u32, out: &mut Vec<Child>) -> Result<(), EpochError> {
    let (seg, part) = batch.items[item as usize];
    let g = &batch.segments[seg as usize];
    let p = &batch.particles[part as usize];
    let focal = u32::from(g.focal);
    let mut s0 = State {
        hands: p.hands,
        leader: u32::from(g.leader),
        n: u32::from(g.n),
        t: g.trick.map(u32::from),
        done: u32::from(g.done),
        obs: NO_OBS,
        nobs: 0,
    };
    s0.hands[focal as usize] = g.focal_hand;
    let action = u32::from(g.action);
    if (legal(tab, &s0, focal) >> action) & 1 == 0 || (s0.leader + s0.n) & 3 != focal {
        return Err(EpochError::Illegal);
    }
    play(tab, &mut s0, action);
    let mut st = [s0; MAX_FIELD + 1];
    let mut cand = [0u32; MAX_FIELD + 1];
    let mut fac = [0u32; MAX_FIELD + 1];
    let mut d = 0usize;
    let mut fresh = true;
    loop {
        if fresh {
            fresh = false;
            let s = st[d];
            let actor = (s.leader + s.n) & 3;
            if s.done == 7 || actor == focal {
                let mut mass = p.mass;
                for f in &fac[1..=d] {
                    mass = mass.checked_mul_small(*f).ok_or(EpochError::Overflow)?;
                }
                let mut hands = s.hands;
                hands[focal as usize] = 0;
                out.push(Child { item, obs: s.obs, hands, kind: p.kind, world: p.world, mass });
                if d == 0 {
                    break;
                }
                d -= 1;
                continue;
            }
            if d == MAX_FIELD {
                return Err(EpochError::Depth);
            }
            cand[d] = support(tab.law(p.kind, actor), legal(tab, &s, actor));
        }
        if cand[d] == 0 {
            if d == 0 {
                break;
            }
            d -= 1;
            continue;
        }
        let tile = cand[d].trailing_zeros();
        cand[d] &= cand[d] - 1;
        let s = st[d];
        let actor = (s.leader + s.n) & 3;
        fac[d + 1] = weight(tab.law(p.kind, actor), legal(tab, &s, actor), tile)?;
        let mut next = s;
        play(tab, &mut next, tile);
        let shift = 5 * next.nobs;
        next.obs = (next.obs & !(31 << shift)) | (tile << shift);
        next.nobs += 1;
        st[d + 1] = next;
        d += 1;
        fresh = true;
    }
    Ok(())
}

/// The shaders' algorithm on the CPU, over `threads` threads.
pub struct Cpu {
    pub threads: usize,
}

impl Backend for Cpu {
    fn name(&self) -> &'static str {
        "cpu"
    }

    fn epoch(&mut self, tables: &Tables, batch: Batch<'_>) -> Result<Vec<Child>, EpochError> {
        let n = batch.items.len();
        let threads = self.threads.max(1).min(n.max(1));
        let chunk = n.div_ceil(threads).max(1);
        let parts: Vec<Result<Vec<Child>, EpochError>> = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..n)
                .step_by(chunk)
                .map(|start| {
                    scope.spawn(move || {
                        let mut out = Vec::new();
                        for item in start..(start + chunk).min(n) {
                            mirror_item(tables, &batch, item as u32, &mut out)?;
                        }
                        Ok(out)
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().expect("an epoch thread panicked")).collect()
        });
        let mut out = Vec::new();
        for part in parts {
            out.extend(part?);
        }
        Ok(out)
    }
}

// ---------------------------------------------------------------------------------------
// The reference: Silas's Physics and the laws as exact rationals. No table is read.

/// The authority: legality and winners from Silas's [`Physics`], probabilities from
/// [`Law::law`] as exact rationals, the mass scaled by [`SCALE`] and checked to be whole.
pub struct Reference {
    pub physics: Physics,
    pub kinds: Vec<[Law; Seat::COUNT]>,
}

struct RefState {
    hands: [DominoSet; 4],
    leader: Seat,
    trick: Vec<Domino>,
    done: u32,
    seen: Vec<Domino>,
}

impl Reference {
    fn walk(
        &self,
        focal: Seat,
        s: RefState,
        mass: &BigRational,
        p: &Particle,
        item: u32,
        out: &mut Vec<Child>,
    ) -> Result<(), EpochError> {
        let actor = s.leader.plus(s.trick.len());
        if s.done == 7 || actor == focal {
            let whole = mass.to_integer();
            if !mass.is_integer() || whole.sign() == num_bigint::Sign::Minus {
                return Err(EpochError::Law);
            }
            let mass = U256::from_biguint(whole.magnitude()).ok_or(EpochError::Overflow)?;
            let mut hands = s.hands.map(|h| h.bits());
            hands[focal.index()] = 0;
            let mut obs = NO_OBS;
            for (i, d) in s.seen.iter().enumerate() {
                obs = (obs & !(31 << (5 * i))) | ((d.index() as u32) << (5 * i));
            }
            out.push(Child { item, obs, hands, kind: p.kind, world: p.world, mass });
            return Ok(());
        }
        if s.seen.len() == MAX_FIELD {
            return Err(EpochError::Depth);
        }
        let legal = self.physics.legal_plays(s.hands[actor.index()], s.trick.first().copied());
        let law = self.kinds[p.kind as usize][actor.index()];
        for (tile, prob) in law.law(legal) {
            let next = self.after(&s, actor, tile);
            let scaled = mass * &prob * BigRational::from_integer(SCALE.into());
            self.walk(focal, next, &scaled, p, item, out)?;
        }
        Ok(())
    }

    fn after(&self, s: &RefState, actor: Seat, tile: Domino) -> RefState {
        let mut hands = s.hands;
        hands[actor.index()].remove(tile);
        let mut trick = s.trick.clone();
        trick.push(tile);
        let mut seen = s.seen.clone();
        seen.push(tile);
        let (mut leader, mut done) = (s.leader, s.done);
        if let Ok(four) = <[Domino; 4]>::try_from(trick.as_slice()) {
            leader = self.physics.trick_winner(s.leader, four);
            trick.clear();
            done += 1;
        }
        RefState { hands, leader, trick, done, seen }
    }
}

impl Backend for Reference {
    fn name(&self) -> &'static str {
        "reference"
    }

    fn epoch(&mut self, _tables: &Tables, batch: Batch<'_>) -> Result<Vec<Child>, EpochError> {
        let mut out = Vec::new();
        for (item, (seg, part)) in batch.items.iter().enumerate() {
            let g = &batch.segments[*seg as usize];
            let p = &batch.particles[*part as usize];
            let focal = Seat::from_index(usize::from(g.focal)).ok_or(EpochError::Illegal)?;
            let leader = Seat::from_index(usize::from(g.leader)).ok_or(EpochError::Illegal)?;
            let mut hands = p.hands.map(|h| DominoSet::from_bits(h).expect("a hand"));
            hands[focal.index()] = DominoSet::from_bits(g.focal_hand).ok_or(EpochError::Illegal)?;
            let trick: Vec<Domino> =
                g.trick[..usize::from(g.n)].iter().map(|t| Domino::from_index(usize::from(*t)).expect("a tile")).collect();
            let action = Domino::from_index(usize::from(g.action)).ok_or(EpochError::Illegal)?;
            if leader.plus(trick.len()) != focal
                || !self.physics.legal_plays(hands[focal.index()], trick.first().copied()).contains(action)
            {
                return Err(EpochError::Illegal);
            }
            let s = RefState { hands, leader, trick, done: u32::from(g.done), seen: Vec::new() };
            let s = self.after(&s, focal, action);
            let s = RefState { seen: Vec::new(), ..s };
            let mass = BigRational::from_integer(p.mass.to_biguint().into());
            self.walk(focal, s, &mass, p, item as u32, &mut out)?;
        }
        Ok(out)
    }
}

/// `SCALE^k` as a mass.
pub fn scale_pow(k: u32) -> U256 {
    (0..k).fold(U256::ONE, |m, _| m.checked_mul_small(SCALE).expect("840^k fits for k <= 26"))
}

/// A mass as a probability against `total` at exponent `r`: `mass / (total * SCALE^r)`.
pub fn probability(mass: U256, total: &num_bigint::BigUint, r: u32) -> BigRational {
    let den = total * scale_pow(r).to_biguint();
    BigRational::new(mass.to_biguint().into(), den.into())
}
