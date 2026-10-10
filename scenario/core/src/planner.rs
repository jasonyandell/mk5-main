//! The finite planning rule and the ladder — `scenario/planner.py`.
//!
//!     V(C) = Σ w·u                      finished
//!          = Σ w·L                      at the search frontier
//!          = max_a Σ_o V(C_{a,o})       the deciding player chooses
//!          = Σ_o V(C_o)                 another seat acts
//!
//! Rung k's modeled seats are rung k−1 players; rung −1 is random legal play
//! from a tape keyed on the public record. The belief weights the live
//! player's scenarios; modeled minds weight theirs uniformly. Below the root,
//! a deciding cell thinner than its rung's floor does not maximize: the
//! decider plays the rung k−1 move for its own remaining hand.

use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hash, Hasher};

use crate::fiber::sample_worlds;

pub const ALL_TILES: u32 = (1 << 28) - 1;
pub const HAND_TOTAL: i32 = 42;
pub const MAX_WORLDS: usize = 1024;
const WORDS: usize = MAX_WORLDS / 64;
const HAND_SIZE: u8 = 7;
const GOLDEN: u64 = 0x9E3779B97F4A7C15;

// -- hashing ------------------------------------------------------------------

/// A small multiplicative hasher for the memo tables (keys are a few words).
#[derive(Default)]
pub struct Fx(u64);

impl Fx {
    #[inline]
    fn add(&mut self, w: u64) {
        self.0 = (self.0.rotate_left(5) ^ w).wrapping_mul(0x517c_c1b7_2722_0a95);
    }
}

impl Hasher for Fx {
    #[inline]
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.add(b as u64);
        }
    }
    #[inline]
    fn write_u8(&mut self, v: u8) {
        self.add(v as u64);
    }
    #[inline]
    fn write_u32(&mut self, v: u32) {
        self.add(v as u64);
    }
    #[inline]
    fn write_u64(&mut self, v: u64) {
        self.add(v);
    }
    #[inline]
    fn write_usize(&mut self, v: usize) {
        self.add(v as u64);
    }
}

type Map<K, V> = HashMap<K, V, BuildHasherDefault<Fx>>;

// -- randomness ---------------------------------------------------------------

#[inline]
pub fn splitmix(mut x: u64) -> u64 {
    x = x.wrapping_add(GOLDEN);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

fn mix(parts: &[u64]) -> u64 {
    let mut h = 0u64;
    for &p in parts {
        h = splitmix(h ^ p);
    }
    h
}

/// The SplitMix64 stream: `splitmix` of a state that advances by GOLDEN.
pub struct SplitMix(u64);

impl SplitMix {
    pub fn new(seed: u64) -> Self {
        SplitMix(seed)
    }
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let out = splitmix(self.0);
        self.0 = self.0.wrapping_add(GOLDEN);
        out
    }
    /// Uniform on [0, 1) with 53 random bits.
    #[inline]
    pub fn random(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
}

// -- rules tables -------------------------------------------------------------

pub struct Rules {
    pub led_suit: [u8; 28],     // led suit when the tile leads
    pub follow_bits: [u32; 8],  // tiles that follow each led suit
    pub rank: [[u8; 28]; 8],    // trick rank under each led suit
    pub count: [u8; 28],        // count points on the tile
}

impl Rules {
    #[inline]
    fn followers(&self, led: u8) -> u32 {
        self.follow_bits[self.led_suit[led as usize] as usize]
    }
    #[inline]
    pub fn legal(&self, hand: u32, led: Option<u8>) -> u32 {
        match led {
            None => hand,
            Some(l) => {
                let f = hand & self.followers(l);
                if f != 0 { f } else { hand }
            }
        }
    }
    #[inline]
    fn winner_offset(&self, trick: &[u8; 4]) -> u8 {
        let ranks = &self.rank[self.led_suit[trick[0] as usize] as usize];
        let mut best = 0u8;
        let mut best_rank = ranks[trick[0] as usize];
        for i in 1..4 {
            let r = ranks[trick[i] as usize];
            if r > best_rank {
                best = i as u8;
                best_rank = r;
            }
        }
        best
    }
    #[inline]
    fn trick_points(&self, trick: &[u8; 4]) -> u8 {
        1 + trick.iter().map(|&t| self.count[t as usize]).sum::<u8>()
    }
}

// -- the public record --------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Rec {
    pub played: u32,
    pub leader: u8,
    pub n: u8,            // tiles in the open trick
    pub trick: [u8; 3],   // slots >= n are zero
    pub banked: [u8; 2],
    pub forbid: [u32; 4], // per seat: tiles it provably cannot hold
}

impl Hash for Rec {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_u64(self.pack());
        for f in self.forbid {
            state.write_u32(f);
        }
    }
}

#[inline]
fn pack(played: u32, leader: u8, trick: &[u8], banked: [u8; 2]) -> u64 {
    let mut v = played as u64 | (leader as u64) << 28 | (trick.len() as u64) << 30;
    for (i, &t) in trick.iter().enumerate() {
        v |= (t as u64) << (32 + 5 * i);
    }
    v | (banked[0] as u64) << 47 | (banked[1] as u64) << 53
}

impl Rec {
    #[inline]
    pub fn seat(&self) -> u8 {
        (self.leader + self.n) % 4
    }
    #[inline]
    pub fn led(&self) -> Option<u8> {
        if self.n > 0 { Some(self.trick[0]) } else { None }
    }
    #[inline]
    pub fn finished(&self) -> bool {
        self.played == ALL_TILES
    }
    #[inline]
    pub fn plays(&self) -> u8 {
        self.played.count_ones() as u8
    }
    pub fn hand_size(&self, seat: u8) -> u8 {
        let completed = (self.plays() - self.n) / 4;
        let in_trick = if (seat + 4 - self.leader) % 4 < self.n { 1 } else { 0 };
        HAND_SIZE - completed - in_trick
    }
    /// The record less voids as one integer below 2^59: the tape key.
    #[inline]
    pub fn pack(&self) -> u64 {
        pack(self.played, self.leader, &self.trick[..self.n as usize], self.banked)
    }
    pub fn advance(&self, tile: u8, rules: &Rules) -> Rec {
        let seat = self.seat() as usize;
        let mut forbid = self.forbid;
        if let Some(led) = self.led() {
            let shown = rules.followers(led);
            if (shown >> tile) & 1 == 0 {
                forbid[seat] |= shown;
            }
        }
        let played = self.played | 1 << tile;
        if self.n < 3 {
            let mut trick = self.trick;
            trick[self.n as usize] = tile;
            return Rec { played, leader: self.leader, n: self.n + 1, trick, banked: self.banked, forbid };
        }
        let t4 = [self.trick[0], self.trick[1], self.trick[2], tile];
        let winner = (self.leader + rules.winner_offset(&t4)) % 4;
        let mut banked = self.banked;
        banked[(winner % 2) as usize] += rules.trick_points(&t4);
        Rec { played, leader: winner, n: 0, trick: [0; 3], banked, forbid }
    }
}

// -- summation --------------------------------------------------------------------

/// Σ as CPython's `sum()` computes it for floats (Neumaier's compensated
/// summation, Python ≥ 3.12), so the referee and the core agree to the bit.
fn fsum(items: impl Iterator<Item = f64>) -> f64 {
    let mut s = 0.0f64;
    let mut c = 0.0f64;
    for x in items {
        let t = s + x;
        if s.abs() >= x.abs() {
            c += (s - t) + x;
        } else {
            c += (x - t) + s;
        }
        s = t;
    }
    if c != 0.0 && c.is_finite() {
        s += c;
    }
    s
}

// -- cells ----------------------------------------------------------------------

/// A set of scenario indices (which worlds are alive in a cell).
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Bits([u64; WORDS]);

impl Hash for Bits {
    fn hash<H: Hasher>(&self, state: &mut H) {
        for w in self.0 {
            state.write_u64(w);
        }
    }
}

impl Bits {
    fn none() -> Self {
        Bits([0; WORDS])
    }
    fn everyone(n: usize) -> Self {
        let mut b = Bits::none();
        for r in 0..n {
            b.set(r);
        }
        b
    }
    #[inline]
    fn set(&mut self, r: usize) {
        self.0[r / 64] |= 1 << (r % 64);
    }
    fn count(&self) -> usize {
        self.0.iter().map(|w| w.count_ones() as usize).sum()
    }
    fn iter(&self) -> impl Iterator<Item = usize> + '_ {
        self.0.iter().enumerate().flat_map(|(i, &w)| {
            let mut w = w;
            std::iter::from_fn(move || {
                if w == 0 {
                    None
                } else {
                    let b = w.trailing_zeros() as usize;
                    w &= w - 1;
                    Some(i * 64 + b)
                }
            })
        })
    }
}

/// Ascending tile ids of a mask.
#[inline]
fn tiles(mut mask: u32) -> impl Iterator<Item = u8> {
    std::iter::from_fn(move || {
        if mask == 0 {
            None
        } else {
            let t = mask.trailing_zeros() as u8;
            mask &= mask - 1;
            Some(t)
        }
    })
}

// -- the ladder -----------------------------------------------------------------

/// The built-in evidence g: how hidden seats' discretionary plays are scored.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum TiltPlays {
    None,
    Dice,  // a play with n legal options has likelihood 1/n
    Field, // ε/n + (1−ε)·[the rung-0 player holding that hand would play it]
}

pub struct Config {
    pub rungs: Vec<(usize, Option<usize>, usize)>, // (n_worlds, horizon, floor) per rung; the live player is last
    pub make: bool,
    pub seed: u64,
    pub tilt: TiltPlays,
    pub eps: f64,
}

impl Config {
    fn level(&self) -> usize {
        self.rungs.len() - 1
    }
}

/// One decision's shared state: the modeled-move cache and the node count.
struct Ctx<'a> {
    cfg: &'a Config,
    rules: &'a Rules,
    bid: i32,
    bt: usize,
    pi: Map<(u8, u8, u32, Rec), u8>,
    nodes: u64,
}

fn invocation_seed(cfg: &Config, k: usize, viewer: u8, own: u32, rec: &Rec) -> u64 {
    mix(&[
        cfg.seed, k as u64, viewer as u64, own as u64, rec.pack(),
        rec.forbid[0] as u64, rec.forbid[1] as u64, rec.forbid[2] as u64, rec.forbid[3] as u64,
    ])
}

fn worlds_for(cfg: &Config, k: usize, viewer: u8, own: u32, rec: &Rec, seed: u64) -> Vec<[u32; 4]> {
    let mut rng = SplitMix::new(seed);
    let sizes = [rec.hand_size(0), rec.hand_size(1), rec.hand_size(2), rec.hand_size(3)];
    sample_worlds(viewer, own, rec.played, sizes, rec.forbid, cfg.rungs[k].0, &mut rng)
}

/// The live rung's scenarios at a root — what `decide` plans over.
pub fn root_worlds(cfg: &Config, viewer: u8, own: u32, rec: &Rec) -> Vec<[u32; 4]> {
    let k = cfg.level();
    worlds_for(cfg, k, viewer, own, rec, invocation_seed(cfg, k, viewer, own, rec))
}

/// e^g for each scenario: the likelihood of the hidden seats' observed
/// discretionary plays, replayed from the first lead with each seat holding
/// the hand the scenario implies it was dealt (`Walt.tilt_weights`).
fn tilt_weights(ctx: &mut Ctx, viewer: u8, worlds: &[[u32; 4]], plays: &[(u8, u8)]) -> Vec<f64> {
    let mut w = vec![1.0f64; worlds.len()];
    if ctx.cfg.tilt == TiltPlays::None || plays.is_empty() {
        return w;
    }
    let dice = ctx.cfg.tilt == TiltPlays::Dice;
    let eps = ctx.cfg.eps;
    let mut played_by = [0u32; 4];
    for &(s, t) in plays {
        played_by[s as usize] |= 1 << t;
    }
    let mut full: Vec<[u32; 4]> = worlds
        .iter()
        .map(|h| [h[0] | played_by[0], h[1] | played_by[1], h[2] | played_by[2], h[3] | played_by[3]])
        .collect();
    let mut rec = Rec { played: 0, leader: plays[0].0, n: 0, trick: [0; 3], banked: [0; 2], forbid: [0; 4] };
    for &(seat, tile) in plays {
        if seat != viewer {
            for (r, hands) in full.iter_mut().enumerate() {
                let legal = ctx.rules.legal(hands[seat as usize], rec.led());
                let n = legal.count_ones();
                if n > 1 {
                    if dice {
                        w[r] *= 1.0 / n as f64;
                    } else {
                        let m = ctx.modeled_move(0, seat, hands[seat as usize], &rec);
                        w[r] *= eps / n as f64 + (1.0 - eps) * if m == tile { 1.0 } else { 0.0 };
                    }
                }
                hands[seat as usize] &= !(1 << tile);
            }
        }
        rec = rec.advance(tile, ctx.rules);
    }
    w
}

/// One planning problem: a viewer, a rung, a sample, and its memo.
struct Inv {
    k: usize,
    viewer: u8,
    root: Rec,
    maximize: bool,
    horizon: Option<usize>,
    floor: usize,
    root_plays: u8,
    worlds: Vec<[u32; 4]>,
    weights: Vec<f64>,
    tapes: Vec<u64>,
    memo: Map<(Rec, Bits), f64>,
}

impl Inv {
    /// `live` carries the public play history and any external evidence at the
    /// live rung only: modeled minds weigh their scenarios uniformly.
    fn new(ctx: &mut Ctx, k: usize, viewer: u8, own: u32, rec: &Rec,
           live: Option<(&[(u8, u8)], Option<&[f64]>)>) -> Inv {
        let cfg = ctx.cfg;
        let seed = invocation_seed(cfg, k, viewer, own, rec);
        let worlds = worlds_for(cfg, k, viewer, own, rec, seed);
        let weights = match live {
            Some((plays, external)) => {
                let mut w = tilt_weights(ctx, viewer, &worlds, plays);
                if let Some(x) = external {
                    assert_eq!(x.len(), w.len(), "external weights must match the live scenarios");
                    for (wi, xi) in w.iter_mut().zip(x) {
                        *wi *= xi;
                    }
                }
                w
            }
            None => vec![1.0; worlds.len()],
        };
        let tapes = (0..worlds.len()).map(|i| mix(&[seed, i as u64])).collect();
        Inv {
            k,
            viewer,
            root: *rec,
            maximize: (viewer as usize) % 2 == ctx.bt,
            horizon: cfg.rungs[k].1,
            floor: cfg.rungs[k].2,
            root_plays: rec.plays(),
            worlds,
            weights,
            tapes,
            memo: Map::default(),
        }
    }

    /// Q(a) for each legal first action, ascending by tile: value per unit of mass.
    fn action_values(&mut self, ctx: &mut Ctx) -> Vec<(u8, f64)> {
        let own = self.worlds[0][self.viewer as usize] & !self.root.played;
        let everyone = Bits::everyone(self.worlds.len());
        let total = self.mass(&everyone);
        let root = self.root;
        tiles(ctx.rules.legal(own, root.led()))
            .map(|t| (t, self.value(ctx, root.advance(t, ctx.rules), everyone) / total))
            .collect()
    }

    fn value(&mut self, ctx: &mut Ctx, rec: Rec, alive: Bits) -> f64 {
        if let Some(&v) = self.memo.get(&(rec, alive)) {
            return v;
        }
        let v = self.compute(ctx, rec, alive);
        self.memo.insert((rec, alive), v);
        v
    }

    fn compute(&mut self, ctx: &mut Ctx, rec: Rec, alive: Bits) -> f64 {
        ctx.nodes += 1;
        if let Some(u) = self.settled(ctx, &rec) {
            return u * self.mass(&alive);
        }
        if self.at_frontier(&rec) {
            let ctx: &Ctx = ctx;
            return fsum(alive.iter().map(|r| self.weights[r] * self.playout(ctx, r, &rec, true)));
        }
        let seat = rec.seat();
        if seat == self.viewer {
            if self.floored(&alive) {
                let t = self.handed_down(ctx, &rec);
                return self.value(ctx, rec.advance(t, ctx.rules), alive);
            }
            let hand = self.worlds[0][seat as usize] & !rec.played;
            let mut best = if self.maximize { f64::NEG_INFINITY } else { f64::INFINITY };
            for t in tiles(ctx.rules.legal(hand, rec.led())) {
                let v = self.value(ctx, rec.advance(t, ctx.rules), alive);
                best = if self.maximize { best.max(v) } else { best.min(v) };
            }
            return best;
        }
        let values: Vec<f64> = self
            .partition(ctx, seat, &rec, &alive)
            .into_iter()
            .map(|(t, cell)| self.value(ctx, rec.advance(t, ctx.rules), cell))
            .collect();
        fsum(values.into_iter())
    }

    /// The utility every scenario shares once the hand is over — or, under
    /// `make`, once the contract is decided and the rest cannot change it.
    fn settled(&self, ctx: &Ctx, rec: &Rec) -> Option<f64> {
        let bt = ctx.bt;
        if ctx.cfg.make {
            if rec.banked[bt] as i32 >= ctx.bid {
                return Some(1.0);
            }
            if rec.banked[1 - bt] as i32 > HAND_TOTAL - ctx.bid {
                return Some(0.0);
            }
        }
        if rec.finished() {
            return Some(self.score(ctx, rec));
        }
        None
    }

    fn at_frontier(&self, rec: &Rec) -> bool {
        self.horizon.is_some_and(|h| (rec.plays() - self.root_plays) as usize >= h)
    }

    /// The field acts: the cell split by the tile the viewer sees, in
    /// first-seen order.
    fn partition(&self, ctx: &mut Ctx, seat: u8, rec: &Rec, alive: &Bits) -> Vec<(u8, Bits)> {
        let mut buckets: Vec<(u8, Bits)> = Vec::with_capacity(8);
        for r in alive.iter() {
            let hand = self.worlds[r][seat as usize] & !rec.played;
            let tile = self.field_move(ctx, seat, r, hand, rec);
            match buckets.iter_mut().find(|(t, _)| *t == tile) {
                Some((_, cell)) => cell.set(r),
                None => {
                    let mut cell = Bits::none();
                    cell.set(r);
                    buckets.push((tile, cell));
                }
            }
        }
        buckets
    }

    // -- the trace -------------------------------------------------------------

    /// Per legal root action (ascending), the leaf value each scenario reaches
    /// under the plan: a replay of the solved tree through the memo, one path
    /// per scenario (`_Invocation.outcomes`).
    fn outcomes(&mut self, ctx: &mut Ctx) -> Vec<(u8, Vec<f64>)> {
        let own = self.worlds[0][self.viewer as usize] & !self.root.played;
        let everyone = Bits::everyone(self.worlds.len());
        let root = self.root;
        tiles(ctx.rules.legal(own, root.led()))
            .map(|t| {
                let mut leaf = vec![0.0; self.worlds.len()];
                self.trace(ctx, root.advance(t, ctx.rules), everyone, &mut leaf);
                (t, leaf)
            })
            .collect()
    }

    /// Leaves of the trace are bidder-team margins under every utility; where
    /// the plan stopped short of the hand's end, the tape plays it out.
    fn trace(&mut self, ctx: &mut Ctx, rec: Rec, alive: Bits, leaf: &mut [f64]) {
        if rec.finished() {
            for r in alive.iter() {
                leaf[r] = self.margin(ctx, &rec);
            }
            return;
        }
        if self.settled(ctx, &rec).is_some() || self.at_frontier(&rec) {
            for r in alive.iter() {
                leaf[r] = self.playout(ctx, r, &rec, false);
            }
            return;
        }
        let seat = rec.seat();
        if seat == self.viewer {
            if self.floored(&alive) {
                let t = self.handed_down(ctx, &rec);
                self.trace(ctx, rec.advance(t, ctx.rules), alive, leaf);
                return;
            }
            let hand = self.worlds[0][seat as usize] & !rec.played;
            let q: Vec<(u8, f64)> = tiles(ctx.rules.legal(hand, rec.led()))
                .map(|t| (t, self.value(ctx, rec.advance(t, ctx.rules), alive)))
                .collect();
            let t = best(&q, self.maximize);
            self.trace(ctx, rec.advance(t, ctx.rules), alive, leaf);
            return;
        }
        for (t, cell) in self.partition(ctx, seat, &rec, &alive) {
            self.trace(ctx, rec.advance(t, ctx.rules), cell, leaf);
        }
    }

    // -- the floor -------------------------------------------------------------

    /// Too few scenarios left in the cell for a max over them to be honest.
    fn floored(&self, alive: &Bits) -> bool {
        alive.count() < self.floor
    }

    /// The decider's move in a floored cell: the rung k−1 player's, holding
    /// the decider's remaining hand (the same in every scenario of the cell).
    fn handed_down(&self, ctx: &mut Ctx, rec: &Rec) -> u8 {
        let hand = self.worlds[0][self.viewer as usize] & !rec.played;
        ctx.modeled_move(self.k - 1, self.viewer, hand, rec)
    }

    // -- the field -------------------------------------------------------------

    fn field_move(&self, ctx: &mut Ctx, seat: u8, r: usize, hand: u32, rec: &Rec) -> u8 {
        let legal = ctx.rules.legal(hand, rec.led());
        if self.k == 0 {
            return dice(self.tapes[r], rec.pack(), legal);
        }
        ctx.modeled_move(self.k - 1, seat, hand, rec)
    }

    // -- leaves ------------------------------------------------------------------

    /// The frontier continuation: every seat (the viewer too) plays from the
    /// tape. With `settle` the utility returns as soon as it is decided;
    /// without it the hand is played out and the margin returned (the trace).
    fn playout(&self, ctx: &Ctx, r: usize, rec: &Rec, settle: bool) -> f64 {
        let world = self.worlds[r];
        let rules = ctx.rules;
        let tape = self.tapes[r];
        let bt = ctx.bt;
        let (make, bid) = (ctx.cfg.make && settle, ctx.bid);
        let mut played = rec.played;
        let mut leader = rec.leader;
        let mut n = rec.n as usize;
        let mut trick = [rec.trick[0], rec.trick[1], rec.trick[2], 0u8];
        let mut banked = rec.banked;
        while played != ALL_TILES {
            if make {
                if banked[bt] as i32 >= bid {
                    return 1.0;
                }
                if banked[1 - bt] as i32 > HAND_TOTAL - bid {
                    return 0.0;
                }
            }
            let seat = (leader as usize + n) % 4;
            let hand = world[seat] & !played;
            let legal = if n > 0 {
                let f = hand & rules.followers(trick[0]);
                if f != 0 { f } else { hand }
            } else {
                hand
            };
            let tile = if legal & (legal - 1) == 0 {
                legal.trailing_zeros() as u8
            } else {
                dice(tape, pack(played, leader, &trick[..n], banked), legal)
            };
            played |= 1 << tile;
            trick[n] = tile;
            n += 1;
            if n == 4 {
                leader = (leader + rules.winner_offset(&trick)) % 4;
                banked[(leader % 2) as usize] += rules.trick_points(&trick);
                n = 0;
                trick = [0; 4];
            }
        }
        if make {
            if banked[bt] as i32 >= bid { 1.0 } else { 0.0 }
        } else {
            banked[bt] as f64 - banked[1 - bt] as f64
        }
    }

    fn score(&self, ctx: &Ctx, rec: &Rec) -> f64 {
        if ctx.cfg.make {
            if rec.banked[ctx.bt] as i32 >= ctx.bid { 1.0 } else { 0.0 }
        } else {
            self.margin(ctx, rec)
        }
    }

    fn margin(&self, ctx: &Ctx, rec: &Rec) -> f64 {
        let bt = ctx.bt;
        rec.banked[bt] as f64 - rec.banked[1 - bt] as f64
    }

    fn mass(&self, alive: &Bits) -> f64 {
        fsum(alive.iter().map(|r| self.weights[r]))
    }
}

/// Random legal play from a scenario's tape, keyed on the record.
#[inline]
fn dice(tape: u64, key: u64, legal: u32) -> u8 {
    if legal & (legal - 1) == 0 {
        return legal.trailing_zeros() as u8;
    }
    let n = legal.count_ones() as u64;
    let idx = (splitmix(tape ^ key) % n) as u32;
    tiles(legal).nth(idx as usize).expect("index within the legal tiles")
}

/// The tie rule: among equal-valued moves, the lowest tile id.
fn best(q: &[(u8, f64)], maximize: bool) -> u8 {
    let mut it = q.iter();
    let (mut bt, mut bv) = *it.next().expect("a legal play exists");
    for &(t, v) in it {
        if (maximize && v > bv) || (!maximize && v < bv) {
            bt = t;
            bv = v;
        }
    }
    bt
}

impl<'a> Ctx<'a> {
    /// What a rung-k player holding `own` at `seat` plays at this record.
    fn modeled_move(&mut self, k: usize, seat: u8, own: u32, rec: &Rec) -> u8 {
        let key = (k as u8, seat, own, *rec);
        if let Some(&t) = self.pi.get(&key) {
            return t;
        }
        let legal = self.rules.legal(own, rec.led());
        let tile = if legal & (legal - 1) == 0 {
            legal.trailing_zeros() as u8
        } else {
            let mut inv = Inv::new(self, k, seat, own, rec, None);
            let q = inv.action_values(self);
            best(&q, inv.maximize)
        };
        self.pi.insert(key, tile);
        tile
    }
}

/// One decision at the live rung: (tile, action values ascending by tile, cells evaluated).
pub fn decide(
    cfg: &Config,
    rules: &Rules,
    viewer: u8,
    own: u32,
    rec: &Rec,
    bid: i32,
    bidder_team: u8,
    plays: &[(u8, u8)],
    external: Option<&[f64]>,
) -> (u8, Vec<(u8, f64)>, u64) {
    let mut ctx = Ctx { cfg, rules, bid, bt: bidder_team as usize, pi: Map::default(), nodes: 0 };
    let mut inv = Inv::new(&mut ctx, cfg.level(), viewer, own, rec, Some((plays, external)));
    let q = inv.action_values(&mut ctx);
    (best(&q, inv.maximize), q, ctx.nodes)
}

/// `decide`, opened up (`Walt.analyze`): the decision plus the live
/// scenarios, their weights, and each root action's per-scenario outcomes.
pub struct Analysis {
    pub tile: u8,
    pub q: Vec<(u8, f64)>,
    pub nodes: u64,
    pub worlds: Vec<[u32; 4]>,
    pub weights: Vec<f64>,
    pub outcomes: Vec<(u8, Vec<f64>)>,
}

pub fn analyze(
    cfg: &Config,
    rules: &Rules,
    viewer: u8,
    own: u32,
    rec: &Rec,
    bid: i32,
    bidder_team: u8,
    plays: &[(u8, u8)],
    external: Option<&[f64]>,
) -> Analysis {
    let mut ctx = Ctx { cfg, rules, bid, bt: bidder_team as usize, pi: Map::default(), nodes: 0 };
    let mut inv = Inv::new(&mut ctx, cfg.level(), viewer, own, rec, Some((plays, external)));
    let q = inv.action_values(&mut ctx);
    let outcomes = inv.outcomes(&mut ctx);
    Analysis {
        tile: best(&q, inv.maximize),
        q,
        nodes: ctx.nodes,
        worlds: std::mem::take(&mut inv.worlds),
        weights: std::mem::take(&mut inv.weights),
        outcomes,
    }
}
