//! The finite planning rule and the ladder — `scenario/planner.py`.
//!
//! ```text
//! V(C) = Σ w·u                      finished
//!      = Σ w·L                      at the search frontier
//!      = max_a Σ_o V(C_{a,o})       the deciding player chooses
//!      = Σ_o V(C_o)                 another seat acts
//! ```
//!
//! Rung k's modeled seats are rung k−1 players; rung −1 is random legal play
//! from a tape keyed on the public record — or the field net (`field.rs`),
//! which then also takes rung 0's floored moves. With `field_eps` = ε > 0
//! the net's field is F_σ: a modeled seat explores with probability ε,
//! playing the dice's tile, else the net's move, so
//! P_σ(t | hand, record, world) = ε/n + (1 − ε)·[t = the net's move]. The belief weights the live
//! player's scenarios; modeled minds weight theirs uniformly. Below the root,
//! a deciding cell thinner than its rung's floor does not maximize: the
//! decider plays the rung k−1 move for its own remaining hand. The floor
//! guards only a sample: an invocation whose fiber was enumerated outright
//! holds the exact posterior, and is never floored.
//!
//! The live rung may draw its scenarios from the field's posterior β^F
//! instead of the uniform ([`Draw::Posterior`]): the enumerated fiber weighted
//! by the likelihood L normalized, or n draws by rejection from the uniform
//! stream, equally weighted; the uniform sample when no world explains the
//! record or the proposal budget runs out (`Walt._posterior`).

use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hash, Hasher};

use crate::fiber::{sample_worlds, Fiber};
use crate::field::FieldNet;
use crate::rules::Rules;

/// Every tile: the record of a finished hand.
pub const ALL_TILES: u32 = (1 << 28) - 1;
/// Points in a hand: 35 count and one per trick.
pub const HAND_TOTAL: i32 = 42;
/// The most scenarios a rung may sample (a cell is a bitset of this width).
pub const MAX_WORLDS: usize = 1024;
const WORDS: usize = MAX_WORLDS / 64;
const HAND_SIZE: u8 = 7;
const GOLDEN: u64 = 0x9E3779B97F4A7C15;
/// "coin": the posterior's acceptance stream is `mix(seed, COIN)`.
const COIN: u64 = 0x636F_696E;
/// The posterior's default proposal budget (`scenario.planner.DRAW_BUDGET`).
pub const DRAW_BUDGET: u64 = 1 << 22;

// -- hashing ------------------------------------------------------------------

/// A small multiplicative hasher for the memo tables (keys are a few words).
#[derive(Default)]
struct Fx(u64);

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

/// One SplitMix64 output for the state `x`.
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
    /// The stream whose first state is `seed`.
    pub fn new(seed: u64) -> Self {
        SplitMix(seed)
    }
    /// The next 64 random bits.
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

// -- the public record --------------------------------------------------------

/// The public record (`scenario/record.py`): everything every seat has
/// seen, which is everything a cell's scenarios share.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Rec {
    /// Every tile played so far, the open trick's included.
    pub played: u32,
    /// The seat that led the open trick.
    pub leader: u8,
    /// Tiles in the open trick.
    pub n: u8,
    /// The open trick in play order; slots >= n are zero.
    pub trick: [u8; 3],
    /// Points banked by team 0 (seats 0 and 2) and team 1 (seats 1 and 3).
    pub banked: [u8; 2],
    /// Per seat: the tiles it provably cannot hold, from the suits it failed to follow.
    pub forbid: [u32; 4],
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
    /// The record before the opening lead, `leader` to lead (`empty_record`).
    pub fn opening(leader: u8) -> Rec {
        Rec { played: 0, leader, n: 0, trick: [0; 3], banked: [0; 2], forbid: [0; 4] }
    }
    /// The seat to play next.
    #[inline]
    pub fn seat(&self) -> u8 {
        (self.leader + self.n) % 4
    }
    /// The tile that led the open trick, if one has.
    #[inline]
    pub fn led(&self) -> Option<u8> {
        if self.n > 0 {
            Some(self.trick[0])
        } else {
            None
        }
    }
    /// All 28 tiles are down.
    #[inline]
    pub fn finished(&self) -> bool {
        self.played == ALL_TILES
    }
    /// Tiles played so far.
    #[inline]
    pub fn plays(&self) -> u8 {
        self.played.count_ones() as u8
    }
    /// Tiles `seat` still holds, implied by the record alone.
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
    /// The record after the seat to play puts down `tile`: a seat that
    /// fails to follow is marked void in the led suit, and a fourth tile
    /// banks the trick for its winner, who leads next.
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
pub fn fsum(items: impl Iterator<Item = f64>) -> f64 {
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
struct Bits([u64; WORDS]);

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
    /// Every scenario weighs the same.
    None,
    /// A play with n legal options has likelihood 1/n.
    Dice,
    /// ε/n + (1−ε)·[the rung-0 player holding that hand would play it].
    Field,
}

/// The measure the live rung draws its scenarios from (`Config.draw`).
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Draw {
    /// Uniform on the fiber.
    Uniform,
    /// The field's posterior β^F, by rejection within this many uniform proposals.
    Posterior(u64),
}

/// How a decision's live scenarios were drawn.
#[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
pub struct Drawn {
    /// Uniform proposals the posterior examined.
    pub proposals: u64,
    /// The posterior fell back to the uniform sample.
    pub fell_back: bool,
}

/// A particular Walt (`scenario.planner.Config`): the decision is a pure function of this and the position.
pub struct Config {
    /// (n_worlds, horizon, floor) per rung; the live player is the last.
    pub rungs: Vec<(usize, Option<usize>, usize)>,
    /// The utility: the bidding side makes its contract (true) or the signed point margin.
    pub make: bool,
    /// Seeds every random stream of every invocation.
    pub seed: u64,
    /// The built-in evidence on the live rung's scenarios.
    pub tilt: TiltPlays,
    /// The field tilt's ε.
    pub eps: f64,
    /// Rung −1: the field net, or None for random legal play.
    pub field: Option<FieldNet>,
    /// ε of the stochastic field F_σ, in [0, 1]; 0 is the deterministic net.
    pub field_eps: f64,
    /// The live rung's scenarios: uniform, or the field's posterior.
    pub draw: Draw,
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
    drawn: Drawn,
}

/// The seed of an invocation's worlds and tapes: its coordinate, never its
/// rung, so every rung samples alike and the ladder iterates one operator.
fn invocation_seed(cfg: &Config, viewer: u8, own: u32, rec: &Rec) -> u64 {
    mix(&[
        cfg.seed,
        viewer as u64,
        own as u64,
        rec.pack(),
        rec.forbid[0] as u64,
        rec.forbid[1] as u64,
        rec.forbid[2] as u64,
        rec.forbid[3] as u64,
    ])
}

/// The rung's scenarios, and whether they are the whole fiber.
fn worlds_for(cfg: &Config, k: usize, viewer: u8, own: u32, rec: &Rec, seed: u64) -> (Vec<[u32; 4]>, bool) {
    let mut rng = SplitMix::new(seed);
    let sizes = [rec.hand_size(0), rec.hand_size(1), rec.hand_size(2), rec.hand_size(3)];
    sample_worlds(viewer, own, rec.played, sizes, rec.forbid, cfg.rungs[k].0, &mut rng)
}

/// The live rung's scenarios at a root — what `decide` plans over.
pub fn root_worlds(cfg: &Config, d: &Decision) -> Vec<[u32; 4]> {
    let (_, inv) = live(cfg, d, None);
    inv.worlds
}

/// How many scenarios the live rung holds at a root: the fiber's size, or n
/// when it is larger — under either draw.
pub fn root_count(cfg: &Config, d: &Decision) -> usize {
    let sizes = [d.rec.hand_size(0), d.rec.hand_size(1), d.rec.hand_size(2), d.rec.hand_size(3)];
    let size = Fiber::new(d.viewer, d.own, d.rec.played, sizes, d.rec.forbid).size() as usize;
    size.min(cfg.rungs[cfg.level()].0)
}

/// F_σ's explore probability ε̂ = ⌈ε·2^53⌉ / 2^53 (`Config.field_eps_exact`).
fn eps_exact(eps: f64) -> f64 {
    (eps * (1u64 << 53) as f64).ceil() * (1.0 / (1u64 << 53) as f64)
}

/// The other seats' plays the field must explain, from the viewer's chair:
/// (seat, the record before the play, the tile, the tiles that seat had
/// played before it), and every seat's tiles played so far.
struct Evidence {
    plays: Vec<(u8, Rec, u8, u32)>,
    played_by: [u32; 4],
}

impl Evidence {
    fn new(rules: &Rules, viewer: u8, plays: &[(u8, u8)]) -> Evidence {
        let mut out = Vec::new();
        let mut played_by = [0u32; 4];
        if let Some(&(first, _)) = plays.first() {
            let mut rec = Rec::opening(first);
            for &(seat, tile) in plays {
                if seat != viewer {
                    out.push((seat, rec, tile, played_by[seat as usize]));
                }
                played_by[seat as usize] |= 1 << tile;
                rec = rec.advance(tile, rules);
            }
        }
        Evidence { plays: out, played_by }
    }
}

/// L(world) = ∏ P_σ(observed tile | that seat's hand in the world, record)
/// over the other seats' discretionary plays (`Walt.field_likelihood`) —
/// returned early, as the partial product, once it is ≤ `u`: every factor is
/// at most 1, so the product can only fall, and `u < L` is decided.
fn likelihood(ctx: &mut Ctx, ev: &Evidence, world: &[u32; 4], eps: f64, u: f64) -> f64 {
    let mut p = 1.0f64;
    for &(seat, rec, tile, before) in &ev.plays {
        let hand = (world[seat as usize] | ev.played_by[seat as usize]) & !before;
        let n = ctx.rules.legal(hand, rec.led()).count_ones();
        if n > 1 {
            let hit = eps < 1.0 && ctx.modeled_move(None, seat, hand, &rec) == tile;
            p *= eps / n as f64 + (1.0 - eps) * if hit { 1.0 } else { 0.0 };
            if p <= u {
                return p;
            }
        }
    }
    p
}

/// The live rung's scenarios under β^F and their weights, given the uniform
/// ones (`Walt._posterior`): the fiber weighted by L normalized when `exact`,
/// else n draws by rejection, equally weighted — or the uniform, when the
/// field explains no world of the fiber or the budget runs out.
fn posterior(
    ctx: &mut Ctx,
    viewer: u8,
    rec: &Rec,
    plays: &[(u8, u8)],
    uniform: Vec<[u32; 4]>,
    exact: bool,
    seed: u64,
) -> (Vec<[u32; 4]>, Vec<f64>) {
    let Draw::Posterior(budget) = ctx.cfg.draw else { unreachable!("the posterior draw only") };
    let n = uniform.len();
    let eps = eps_exact(ctx.cfg.field_eps);
    let ev = Evidence::new(ctx.rules, viewer, plays);
    if exact {
        let like: Vec<f64> = uniform.iter().map(|w| likelihood(ctx, &ev, w, eps, -1.0)).collect();
        let total = fsum(like.iter().copied());
        if total > 0.0 {
            return (uniform, like.into_iter().map(|x| x / total).collect());
        }
        ctx.drawn.fell_back = true;
        return (uniform, vec![1.0; n]);
    }
    let sizes = [rec.hand_size(0), rec.hand_size(1), rec.hand_size(2), rec.hand_size(3)];
    let mut fiber = Fiber::new(viewer, uniform[0][viewer as usize], rec.played, sizes, rec.forbid);
    let mut rng = SplitMix::new(seed);
    let mut coins = (eps > 0.0).then(|| SplitMix::new(mix(&[seed, COIN])));
    let mut kept = Vec::with_capacity(n);
    while kept.len() < n && ctx.drawn.proposals < budget {
        let world = fiber.draw(&mut rng);
        ctx.drawn.proposals += 1;
        let u = coins.as_mut().map_or(0.0, |c| c.random());
        if u < likelihood(ctx, &ev, &world, eps, u) {
            kept.push(world);
        }
    }
    if kept.len() < n {
        ctx.drawn.fell_back = true;
        return (uniform, vec![1.0; n]);
    }
    (kept, vec![1.0; n])
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
    let mut rec = Rec::opening(plays[0].0);
    for &(seat, tile) in plays {
        if seat != viewer {
            for (r, hands) in full.iter_mut().enumerate() {
                let legal = ctx.rules.legal(hands[seat as usize], rec.led());
                let n = legal.count_ones();
                if n > 1 {
                    if dice {
                        w[r] *= 1.0 / n as f64;
                    } else {
                        let m = ctx.modeled_move(Some(0), seat, hands[seat as usize], &rec);
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

/// What only the live rung sees: the public play history and any external
/// evidence over its scenarios.
type Live<'a> = (&'a [(u8, u8)], Option<&'a [f64]>);

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
    fn new(ctx: &mut Ctx, k: usize, viewer: u8, own: u32, rec: &Rec, live: Option<Live>) -> Inv {
        let cfg = ctx.cfg;
        let seed = invocation_seed(cfg, viewer, own, rec);
        let (mut worlds, exact) = worlds_for(cfg, k, viewer, own, rec, seed);
        let weights = match live {
            Some((plays, external)) => {
                let mut base = None;
                if cfg.draw != Draw::Uniform {
                    let (w, b) = posterior(ctx, viewer, rec, plays, worlds, exact, seed);
                    worlds = w;
                    base = Some(b);
                }
                let mut w = tilt_weights(ctx, viewer, &worlds, plays);
                if let Some(b) = base {
                    for (wi, bi) in w.iter_mut().zip(b) {
                        *wi *= bi;
                    }
                }
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
            floor: if exact { 1 } else { cfg.rungs[k].2 },
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

    /// The decider's move in a floored cell: the rung k−1 player's (the field
    /// net's at rung 0), holding the decider's remaining hand (the same in every scenario of the cell).
    fn handed_down(&self, ctx: &mut Ctx, rec: &Rec) -> u8 {
        let hand = self.worlds[0][self.viewer as usize] & !rec.played;
        ctx.modeled_move(self.k.checked_sub(1), self.viewer, hand, rec)
    }

    // -- the field -------------------------------------------------------------

    /// The modeled seat's tile in scenario r. At rung 0 the field is the
    /// dice, or F_σ: the dice when scenario r's coin at this record explores,
    /// else the net (ε = 0 never draws the coin).
    fn field_move(&self, ctx: &mut Ctx, seat: u8, r: usize, hand: u32, rec: &Rec) -> u8 {
        let cfg = ctx.cfg;
        if self.k == 0 && (cfg.field.is_none() || explores(cfg.field_eps, self.tapes[r], rec.pack())) {
            return dice(self.tapes[r], rec.pack(), ctx.rules.legal(hand, rec.led()));
        }
        ctx.modeled_move(self.k.checked_sub(1), seat, hand, rec)
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
                if f != 0 {
                    f
                } else {
                    hand
                }
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
            if banked[bt] as i32 >= bid {
                1.0
            } else {
                0.0
            }
        } else {
            banked[bt] as f64 - banked[1 - bt] as f64
        }
    }

    fn score(&self, ctx: &Ctx, rec: &Rec) -> f64 {
        if ctx.cfg.make {
            if rec.banked[ctx.bt] as i32 >= ctx.bid {
                1.0
            } else {
                0.0
            }
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

/// F_σ's coin for a scenario's tape at a record: the top 53 bits of one more
/// SplitMix step past the dice's draw, below ⌈ε·2^53⌉ (`_Invocation._explores`;
/// the scaling by a power of two is exact, so both engines hold one integer).
#[inline]
fn explores(eps: f64, tape: u64, key: u64) -> bool {
    eps > 0.0 && splitmix(splitmix(tape ^ key)) >> 11 < (eps * (1u64 << 53) as f64).ceil() as u64
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
    /// What a rung-k player holding `own` at `seat` plays at this record;
    /// `None` is rung −1, the field net.
    fn modeled_move(&mut self, k: Option<usize>, seat: u8, own: u32, rec: &Rec) -> u8 {
        let key = (k.map_or(u8::MAX, |k| k as u8), seat, own, *rec);
        if let Some(&t) = self.pi.get(&key) {
            return t;
        }
        let legal = self.rules.legal(own, rec.led());
        let tile = if legal & (legal - 1) == 0 {
            legal.trailing_zeros() as u8
        } else if let Some(k) = k {
            let mut inv = Inv::new(self, k, seat, own, rec, None);
            let q = inv.action_values(self);
            best(&q, inv.maximize)
        } else {
            let net = self.cfg.field.as_ref().expect("rung −1 is random play, not a mind");
            net.decide(self.rules, seat, own, rec, self.bid, self.bt)
        };
        self.pi.insert(key, tile);
        tile
    }
}

/// One decision to plan at the live rung: the seat to play, what it holds,
/// the public record and play history, and the contract.
#[derive(Clone, Copy)]
pub struct Decision<'a> {
    /// The declaration's rules.
    pub rules: &'a Rules,
    /// The seat to play: the record's next seat.
    pub viewer: u8,
    /// The viewer's remaining hand.
    pub own: u32,
    /// The public record.
    pub rec: Rec,
    /// The contract, in points (42 for a mark bid).
    pub bid: i32,
    /// The bidding side: 0 for seats 0 and 2, 1 for seats 1 and 3.
    pub bidder_team: u8,
    /// Every play so far as (seat, tile), from the opening lead.
    pub plays: &'a [(u8, u8)],
}

/// The live invocation at a decision and the context it plans in.
fn live<'a>(cfg: &'a Config, d: &'a Decision, external: Option<&'a [f64]>) -> (Ctx<'a>, Inv) {
    let mut ctx = Ctx {
        cfg,
        rules: d.rules,
        bid: d.bid,
        bt: d.bidder_team as usize,
        pi: Map::default(),
        nodes: 0,
        drawn: Drawn::default(),
    };
    let inv = Inv::new(&mut ctx, cfg.level(), d.viewer, d.own, &d.rec, Some((d.plays, external)));
    (ctx, inv)
}

/// One decision: (tile, action values ascending by tile, cells evaluated,
/// how the scenarios were drawn). `external` is evidence over the live
/// scenarios ([`root_worlds`]), multiplied into the built-in tilt.
pub fn decide(cfg: &Config, d: &Decision, external: Option<&[f64]>) -> (u8, Vec<(u8, f64)>, u64, Drawn) {
    let (mut ctx, mut inv) = live(cfg, d, external);
    let q = inv.action_values(&mut ctx);
    (best(&q, inv.maximize), q, ctx.nodes, ctx.drawn)
}

/// `decide`, opened up (`Walt.analyze`): the decision plus the live
/// scenarios, their weights, and each root action's per-scenario outcomes.
pub struct Analysis {
    /// The decision.
    pub tile: u8,
    /// Each legal tile's value, ascending by tile.
    pub q: Vec<(u8, f64)>,
    /// Cells evaluated.
    pub nodes: u64,
    /// The live scenarios, remaining hands by absolute seat, in sample order.
    pub worlds: Vec<[u32; 4]>,
    /// The belief over them, unnormalized.
    pub weights: Vec<f64>,
    /// Per legal tile, ascending: the bidder-team final margin each scenario reaches under the plan.
    pub outcomes: Vec<(u8, Vec<f64>)>,
    /// How the scenarios were drawn.
    pub drawn: Drawn,
}

/// `decide`, opened up: see [`Analysis`].
pub fn analyze(cfg: &Config, d: &Decision, external: Option<&[f64]>) -> Analysis {
    let (mut ctx, mut inv) = live(cfg, d, external);
    let q = inv.action_values(&mut ctx);
    let outcomes = inv.outcomes(&mut ctx);
    Analysis {
        tile: best(&q, inv.maximize),
        q,
        nodes: ctx.nodes,
        worlds: std::mem::take(&mut inv.worlds),
        weights: std::mem::take(&mut inv.weights),
        outcomes,
        drawn: ctx.drawn,
    }
}
