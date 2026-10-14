//! The record read as evidence: policy likelihood (#128), for the auction and
//! the play. A seat that has seen the record conditions its belief on named
//! seats having acted by named policies: their calls (`GAPS.md` 4, #134) and
//! their plays.
//!
//! **Calls.** A world is kept when, in it, every seat whose calls are read
//! makes, by its bidding policy at its own auction frame (the auction before
//! its call, its dealt hand, the match state, the declarations offered), the
//! call it made and, for the declarer, the declaration. A bidding policy is a
//! function of the auction frame, so this is an event: weight zero or one.
//!
//! **Plays.** A play policy allows a set of tiles at a frame; its decision,
//! the least of them, is a tie-break by tile order and is not read. Where no
//! rule and no exact entry decides, the fallback stands for whatever decides
//! where the policy is silent, so the reading takes it to allow every legal
//! tile. Under [`Likelihood::Consistent`] a world is kept when every past play
//! of every seat whose plays are read is one its policy allows at that seat's
//! own frame of the time (its dealt hand and the record before the play): an
//! event again, weight zero or one. Under [`Likelihood::Uniform`] each such
//! play weighs `1/|allowed|`, the chance of that tile to a seat choosing
//! uniformly among what its policy allows, and a world weighs the product.
//!
//! **Why it factorizes.** A seat's dealt hand is its hand in the world and the
//! tiles it has played; its auction frame and its frame at each of its past
//! plays are that hand and public facts. So each read seat's weight is a
//! function of its own hand in the world, and a world's weight is the product
//! over the read seats. [`Reading::evidence`] runs each read seat's policy on
//! every hand it holds in some world (at most C(21,7) = 116,280), once per call
//! and once per past play, and keeps the agreeing hands with their weights.
//! [`crate::Belief::given`] then sums, by placement of the tiles a query reads,
//! the weights of the worlds whose read hands all agree: one read seat by the
//! binomial count of the other two seats' completions, two or three by
//! enumerating the second seat's hands for each agreeing hand of the first.
//! Nothing is sampled.
//!
//! The arithmetic is exact, but the number is the uniform deal's only under
//! the hypothesis that each read seat acts by its policy: its authority is
//! `exact_given` one model per seat and phase, which names the seat, the
//! policy and a digest of its identity (with the match state and the
//! declarations offered for the calls), and says which likelihood the plays
//! are read under.

use std::collections::HashMap;
use std::sync::Arc;

use num_bigint::BigInt;
use num_rational::BigRational;

use crate::auction::{Auction, AuctionFrame, Call};
use crate::eval::Budget;
use crate::frame::{Frame, HIDDEN_SEATS, World};
use crate::measure::{Authority, Weight};
use crate::nouns::{Domino, DominoSet, Seat};
use crate::outcome::possible_hands;
use crate::physics::Physics;
use crate::policy::{CompiledPolicy, DecisionProvenance};
use crate::scoring::MatchState;
use crate::{Result, error};

/// How a read seat's plays weigh a world: whether its policy allows each of
/// them (an event), or the chance of each to a seat choosing uniformly among
/// what its policy allows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Likelihood {
    /// Weight one when every read play is allowed, else zero.
    Consistent,
    /// The product over the read plays of `1/|allowed|` when the play is
    /// allowed, else zero.
    Uniform,
}

/// Which hidden seats' calls and plays are read, by which policies.
#[derive(Clone)]
pub struct Reading {
    calls: Option<Calls>,
    plays: Option<Plays>,
}

#[derive(Clone)]
struct Calls {
    readers: Vec<(Seat, CompiledPolicy)>,
    state: MatchState,
    declarations: Arc<[Physics]>,
}

#[derive(Clone)]
struct Plays {
    readers: Vec<(Seat, CompiledPolicy)>,
    likelihood: Likelihood,
}

/// One read seat's evidence at a frame: what of its record was read, and
/// how many of the hands it holds in some world agree.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SeatEvidence {
    pub seat: Seat,
    pub call: Option<CallEvidence>,
    pub plays: Option<PlayEvidence>,
    pub agreeing: usize,
    pub candidates: usize,
}

/// A seat's call read, by its bidding policy, and whether its declaration
/// was read too.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CallEvidence {
    pub policy: String,
    pub call: Call,
    pub declares: bool,
}

/// A seat's past plays read, by its play policy, under a likelihood.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlayEvidence {
    pub policy: String,
    pub tiles: Vec<Domino>,
    pub likelihood: Likelihood,
}

/// The record's evidence at one frame: for each read seat, the hands it
/// could hold now that its policies agree with, and their weights.
#[derive(Clone)]
pub struct Evidence {
    frame: Frame,
    agreeing: [Option<Hands>; HIDDEN_SEATS],
    seats: Vec<SeatEvidence>,
    authority: Authority,
}

/// A set of hands within one pool: the list, a bitmap over the pool's
/// subsets for membership, and each hand's weight as an integer multiple of
/// `1/scale` (none: every weight one).
#[derive(Clone)]
struct Hands {
    list: Vec<DominoSet>,
    bits: Vec<u64>,
    within: Compress,
    multiples: Option<HashMap<u32, u128>>,
    scale: u128,
}

/// A pool's tiles numbered densely, so a hand within it is an index below
/// 2^|pool|: by table, a byte of the tile bits at a time.
#[derive(Clone)]
struct Compress {
    tables: Box<[[u32; 256]; 4]>,
}

impl Compress {
    fn new(pool: DominoSet) -> Compress {
        let mut tables = Box::new([[0u32; 256]; 4]);
        let mut rank = [0u32; 32];
        let mut next = 0;
        for (i, slot) in rank.iter_mut().enumerate().take(28) {
            if pool.bits() >> i & 1 == 1 {
                *slot = 1 << next;
                next += 1;
            }
        }
        for (j, table) in tables.iter_mut().enumerate() {
            for (byte, entry) in table.iter_mut().enumerate() {
                *entry = (0..8).filter(|b| byte >> b & 1 == 1).map(|b| rank[8 * j + b]).fold(0, |a, r| a | r);
            }
        }
        Compress { tables }
    }

    fn index(&self, hand: DominoSet) -> usize {
        let b = hand.bits();
        (self.tables[0][(b & 0xff) as usize]
            | self.tables[1][(b >> 8 & 0xff) as usize]
            | self.tables[2][(b >> 16 & 0xff) as usize]
            | self.tables[3][(b >> 24 & 0xff) as usize]) as usize
    }
}

impl Hands {
    /// The hands `kept`, each with the denominator of its weight (`1/d`).
    /// The weights are scaled to integers by the least common multiple of the
    /// denominators. Each denominator is a product of allowed sets' sizes, the
    /// `k`-th at most `8 - k`, so it divides 420·60·60·12·6·2 = 217,728,000,
    /// and so does the scale.
    fn new(pool: DominoSet, kept: Weighed) -> Hands {
        let within = Compress::new(pool);
        let mut bits = vec![0u64; (1usize << pool.len()).div_ceil(64)];
        for (hand, _) in &kept {
            let i = within.index(*hand);
            bits[i >> 6] |= 1 << (i & 63);
        }
        let scale = kept.iter().fold(1u128, |l, (_, d)| lcm(l, u128::from(*d)));
        let multiples = (scale > 1).then(|| kept.iter().map(|(h, d)| (h.bits(), scale / u128::from(*d))).collect());
        Hands { list: kept.into_iter().map(|(h, _)| h).collect(), bits, within, multiples, scale }
    }

    fn contains(&self, hand: DominoSet) -> bool {
        let i = self.within.index(hand);
        self.bits[i >> 6] >> (i & 63) & 1 == 1
    }

    /// The weight of an agreeing hand, times the scale.
    fn multiple(&self, hand: DominoSet) -> u128 {
        self.multiples.as_ref().map_or(1, |m| m[&hand.bits()])
    }
}

fn lcm(a: u128, b: u128) -> u128 {
    let gcd = |mut x: u128, mut y: u128| {
        while y != 0 {
            (x, y) = (y, x % y);
        }
        x
    };
    a / gcd(a, b) * b
}

/// Every `k`-tile subset of `set`, in increasing order of the subset's index
/// among `set`'s tiles.
fn subsets(set: DominoSet, k: usize) -> impl Iterator<Item = DominoSet> {
    let tiles: Vec<u32> = (0..28).filter(|i| set.bits() >> i & 1 == 1).map(|i| 1u32 << i).collect();
    let n = tiles.len();
    let first: Option<u64> = (k <= n).then(|| (1u64 << k) - 1);
    let last = first.map(|f| f << (n - k));
    std::iter::successors(first, move |&v| {
        (Some(v) != last && v != 0).then(|| {
            // Gosper's hack: the next word with as many bits set.
            let c = v & v.wrapping_neg();
            let r = v + c;
            (((r ^ v) >> 2) / c) | r
        })
    })
    .map(move |v| {
        let mut bits = 0;
        let mut rest = v;
        while rest != 0 {
            bits |= tiles[rest.trailing_zeros() as usize];
            rest &= rest - 1;
        }
        DominoSet::from_bits(bits).expect("tiles of the deck")
    })
}

/// FNV-1a, 64 bits: a digest that names a model, not a security property.
fn digest(text: &str) -> u64 {
    text.bytes().fold(0xcbf2_9ce4_8422_2325, |h, b| (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3))
}

/// Refused when a seat is named twice.
fn once(readers: &[(Seat, CompiledPolicy)]) -> Result<()> {
    for (i, (seat, _)) in readers.iter().enumerate() {
        if readers[..i].iter().any(|(s, _)| s == seat) {
            return Err(error(format!("{seat} is read twice")));
        }
    }
    Ok(())
}

impl Reading {
    /// Read the calls of `readers`, each by its bidding policy, the match at
    /// `state` and `declarations` offered to the winner. Refused for a seat
    /// named twice.
    pub fn calls(
        readers: Vec<(Seat, CompiledPolicy)>,
        state: MatchState,
        declarations: Arc<[Physics]>,
    ) -> Result<Reading> {
        once(&readers)?;
        if declarations.is_empty() {
            return Err(error("the winner must have a declaration to choose"));
        }
        Ok(Reading { calls: Some(Calls { readers, state, declarations }), plays: None })
    }

    /// Read the past plays of `readers`, each by its play policy, under
    /// `likelihood`. Refused for a seat named twice.
    pub fn plays(readers: Vec<(Seat, CompiledPolicy)>, likelihood: Likelihood) -> Result<Reading> {
        once(&readers)?;
        Ok(Reading { calls: None, plays: Some(Plays { readers, likelihood }) })
    }

    /// This reading of the calls, and the plays of `readers` too: a seat in
    /// both is read on its calls and its plays, from the one hand.
    pub fn with_plays(self, readers: Vec<(Seat, CompiledPolicy)>, likelihood: Likelihood) -> Result<Reading> {
        if self.plays.is_some() {
            return Err(error("the plays are already read"));
        }
        Ok(Reading { plays: Reading::plays(readers, likelihood)?.plays, ..self })
    }

    /// The seats read, calls first, each once.
    fn seats(&self) -> Vec<Seat> {
        let calls = self.calls.iter().flat_map(|c| &c.readers);
        let plays = self.plays.iter().flat_map(|p| &p.readers);
        let mut seats: Vec<Seat> = Vec::new();
        for (seat, _) in calls.chain(plays) {
            if !seats.contains(seat) {
                seats.push(*seat);
            }
        }
        seats
    }

    fn call_reader(&self, seat: Seat) -> Option<(&Calls, &CompiledPolicy)> {
        let calls = self.calls.as_ref()?;
        calls.readers.iter().find(|(s, _)| *s == seat).map(|(_, p)| (calls, p))
    }

    fn play_reader(&self, seat: Seat) -> Option<(&Plays, &CompiledPolicy)> {
        let plays = self.plays.as_ref()?;
        plays.readers.iter().find(|(s, _)| *s == seat).map(|(_, p)| (plays, p))
    }

    /// The models the reading rests on, one per seat and phase: `S2 bids by
    /// NAME#DIGEST` (the digest of the policy's identity, the match state and
    /// the declarations offered, in order), and `S2 plays within NAME#DIGEST`
    /// or `S2 plays uniformly within NAME#DIGEST` (the digest of the policy's
    /// identity).
    pub fn models(&self) -> Vec<String> {
        let mut models = Vec::new();
        if let Some(calls) = &self.calls {
            let offered: Vec<&str> = calls.declarations.iter().map(Physics::identity).collect();
            for (seat, policy) in &calls.readers {
                let pinned = format!("{}\n{}\n{}", policy.identity(), calls.state, offered.join(" "));
                models.push(format!("{seat} bids by {}#{:016x}", policy.source().name, digest(&pinned)));
            }
        }
        if let Some(plays) = &self.plays {
            let how = match plays.likelihood {
                Likelihood::Consistent => "plays within",
                Likelihood::Uniform => "plays uniformly within",
            };
            for (seat, policy) in &plays.readers {
                models.push(format!("{seat} {how} {}#{:016x}", policy.source().name, digest(policy.identity())));
            }
        }
        models
    }

    /// The evidence at `frame`: each read seat's agreeing hands. Refused when
    /// a read seat is the viewer (his record is no evidence to him), when the
    /// frame's declaration is not offered and the declarer's calls are read,
    /// and when a policy refuses a frame. The policies run on every core;
    /// `budget` pays for all of it.
    pub fn evidence(&self, frame: &Frame, budget: &mut Budget) -> Result<Evidence> {
        let kernel = frame.kernel();
        let auction = frame.auction();
        let declarer = frame.contract().declarer();
        let mut agreeing: [Option<Hands>; HIDDEN_SEATS] = [None, None, None];
        let mut seats = Vec::new();
        for seat in self.seats() {
            if seat == frame.viewer() {
                return Err(error(format!("{seat} is the viewer: his own record tells him nothing")));
            }
            let call = match self.call_reader(seat) {
                None => None,
                Some((calls, policy)) => {
                    let declares = seat == declarer;
                    if declares && !calls.declarations.contains(frame.physics()) {
                        return Err(error(format!(
                            "the declaration {} is not among those offered, so {seat}'s cannot be read",
                            frame.physics().name()
                        )));
                    }
                    let k = (0..auction.calls().len()).find(|i| auction.seat_of(*i) == seat);
                    let k = k.ok_or_else(|| error(format!("{seat} has not called")))?;
                    let before = Auction::new(auction.config(), auction.shaker(), auction.calls()[..k].to_vec())?;
                    Some((calls, policy, before, auction.calls()[k], declares))
                }
            };
            let plays = self.play_reader(seat).map(|(plays, policy)| {
                let at: Vec<usize> = (0..frame.history().len()).filter(|i| frame.history()[*i].0 == seat).collect();
                (plays.likelihood, policy, at)
            });
            let slot = kernel.slot(seat);
            let played: DominoSet = frame.history().iter().filter(|(s, _)| *s == seat).map(|(_, d)| *d).collect();
            let candidates = possible_hands(frame.physics(), kernel, slot, budget)?;
            // The denominator of the hand's weight, or none when it is zero.
            let agrees = |hand: DominoSet, budget: &mut Budget| -> Result<Option<u64>> {
                let dealt = hand.union(played);
                if let Some((calls, policy, before, call, declares)) = &call {
                    let (state, offered) = (calls.state, calls.declarations.clone());
                    let at = AuctionFrame::new(before.clone(), seat, dealt, state, offered.clone())?;
                    if policy.bid(&at, budget)?.choice != *call {
                        return Ok(None);
                    }
                    if *declares {
                        let won = AuctionFrame::new(auction.clone(), seat, dealt, state, offered)?;
                        if policy.declare(&won, budget)?.choice != *frame.physics() {
                            return Ok(None);
                        }
                    }
                }
                let mut denominator = 1;
                if let Some((likelihood, policy, at)) = &plays {
                    for &i in at {
                        let before = frame.plays()[..i].to_vec();
                        let then = Frame::new(auction.clone(), frame.physics().clone(), seat, dealt, before)?;
                        let allowed = policy.allowed(&then, budget)?;
                        let allowed = match allowed.provenance {
                            DecisionProvenance::Fallback => {
                                then.physics().legal_plays(then.kernel().viewer_hand(), then.lead())
                            }
                            _ => allowed.tiles,
                        };
                        if !allowed.contains(frame.plays()[i]) {
                            return Ok(None);
                        }
                        if *likelihood == Likelihood::Uniform {
                            denominator *= allowed.len() as u64;
                        }
                    }
                }
                Ok(Some(denominator))
            };
            let kept = on_every_core(&candidates, budget, agrees)?;
            seats.push(SeatEvidence {
                seat,
                call: call.as_ref().map(|(_, policy, _, call, declares)| CallEvidence {
                    policy: policy.source().name.clone(),
                    call: *call,
                    declares: *declares,
                }),
                plays: plays.as_ref().map(|(likelihood, policy, at)| PlayEvidence {
                    policy: policy.source().name.clone(),
                    tiles: at.iter().map(|i| frame.plays()[*i]).collect(),
                    likelihood: *likelihood,
                }),
                agreeing: kept.len(),
                candidates: candidates.len(),
            });
            agreeing[slot] = Some(Hands::new(kernel.pool(), kept));
        }
        let authority = self
            .models()
            .into_iter()
            .try_fold(Authority::exact(), |a, m| Ok::<_, crate::Error>(a.meet(&Authority::given(m)?)))?;
        Ok(Evidence { frame: frame.clone(), agreeing, seats, authority })
    }
}

/// Hands, each with the denominator of its weight.
type Weighed = Vec<(DominoSet, u64)>;

/// The candidates `keep` weighs above zero, in order, each with its weight's
/// denominator; `keep` runs on every core. Each core draws on an equal share
/// of what `budget` has left, and `budget` is charged what they spent
/// together.
fn on_every_core(
    candidates: &[DominoSet],
    budget: &mut Budget,
    keep: impl Fn(DominoSet, &mut Budget) -> Result<Option<u64>> + Sync,
) -> Result<Weighed> {
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get()).min(candidates.len().max(1));
    let share = budget.remaining() / threads as u64;
    let parts: Vec<Result<(Weighed, u64)>> = std::thread::scope(|s| {
        let handles: Vec<_> = candidates
            .chunks(candidates.len().div_ceil(threads).max(1))
            .map(|part| {
                let keep = &keep;
                s.spawn(move || {
                    let mut budget = Budget::new(share);
                    let mut kept = Vec::new();
                    for hand in part {
                        if let Some(denominator) = keep(*hand, &mut budget)? {
                            kept.push((*hand, denominator));
                        }
                    }
                    Ok((kept, budget.spent()))
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().expect("a policy thread")).collect()
    });
    let mut kept = Vec::new();
    for part in parts {
        let (mut some, spent) = part?;
        budget.spend(spent)?;
        kept.append(&mut some);
    }
    Ok(kept)
}

impl Evidence {
    pub fn frame(&self) -> &Frame {
        &self.frame
    }

    /// Each read seat's evidence, in the order read.
    pub fn seats(&self) -> &[SeatEvidence] {
        &self.seats
    }

    /// `exact_given` every model read.
    pub fn authority(&self) -> &Authority {
        &self.authority
    }

    /// The worlds every read seat's hand agrees in, weighed and summed by
    /// their placement of `resolved` (the pool tiles of it): one
    /// representative world and the total weight for each placement that has
    /// any.
    pub(crate) fn count(&self, resolved: DominoSet) -> Result<Vec<(World, Weight)>> {
        let kernel = self.frame.kernel();
        let physics = self.frame.physics();
        let pool = kernel.pool();
        let shown = resolved.intersection(pool);
        let allowed: [DominoSet; HIDDEN_SEATS] = std::array::from_fn(|slot| kernel.allowed(physics, slot));
        let capacity: [usize; HIDDEN_SEATS] = std::array::from_fn(|slot| kernel.hidden()[slot].capacity);
        let mut read: Vec<usize> = (0..HIDDEN_SEATS).filter(|s| self.agreeing[*s].is_some()).collect();
        read.sort_by_key(|s| self.agreeing[*s].as_ref().map_or(0, |h| h.list.len()));
        let unread: Vec<usize> = (0..HIDDEN_SEATS).filter(|s| self.agreeing[*s].is_none()).collect();
        let Some(&first) = read.first() else {
            return Err(error("no seat is read: the belief is the uniform one"));
        };
        let agreeing = |slot: usize| self.agreeing[slot].as_ref().expect("a read seat");
        // Each world's weight times the product of the read seats' scales (at
        // most 217,728,000 each), times the worlds of a placement (at most
        // C(21,7)·C(14,7)): below 2^115.
        let scale: u128 = read.iter().map(|s| agreeing(*s).scale).product();
        // The worlds that extend each of the first read seat's agreeing hands,
        // by placement of the shown tiles; the hands split across the cores.
        let extend = |h0: DominoSet, placed: &mut Placed| {
            let rest = pool.difference(h0);
            let m0 = agreeing(first).multiple(h0);
            let mut hands = [DominoSet::EMPTY; HIDDEN_SEATS];
            hands[first] = h0;
            let mut add = |hands: [DominoSet; HIDDEN_SEATS], worlds: u128| {
                let key = hands.map(|h| h.intersection(shown).bits());
                placed.entry(key).or_insert((hands, 0)).1 += m0 * worlds;
            };
            if read.len() == 1 {
                complete(&mut hands, rest, [unread[0], unread[1]], &allowed, &capacity, shown, &mut add);
                return;
            }
            // The second read seat's agreeing hands within the rest; the third
            // seat takes what is left, and agrees too when it is read.
            let (second, third) = (read[1], read.get(2).copied().unwrap_or_else(|| unread[0]));
            let mut local: HashMap<u32, (DominoSet, u128)> = HashMap::new();
            for h1 in subsets(rest.intersection(allowed[second]), capacity[second]) {
                let h2 = rest.difference(h1);
                if !agreeing(second).contains(h1)
                    || !h2.is_subset_of(allowed[third])
                    || self.agreeing[third].as_ref().is_some_and(|h| !h.contains(h2))
                {
                    continue;
                }
                let m2 = self.agreeing[third].as_ref().map_or(1, |h| h.multiple(h2));
                local.entry(h1.intersection(shown).bits()).or_insert((h1, 0)).1 += agreeing(second).multiple(h1) * m2;
            }
            for (h1, worlds) in local.into_values() {
                hands[second] = h1;
                hands[third] = rest.difference(h1);
                add(hands, worlds);
            }
        };
        let list = &agreeing(first).list;
        let threads = std::thread::available_parallelism().map_or(4, |n| n.get()).min(list.len().max(1));
        let parts: Vec<Placed> = std::thread::scope(|s| {
            let handles: Vec<_> = list
                .chunks(list.len().div_ceil(threads).max(1))
                .map(|part| {
                    let extend = &extend;
                    s.spawn(move || {
                        let mut placed = Placed::new();
                        part.iter().for_each(|h0| extend(*h0, &mut placed));
                        placed
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().expect("a counting thread")).collect()
        });
        // Merged in the order of the parts, each part's placements in order,
        // so a placement's representative does not depend on the cores.
        let mut placed: Vec<([DominoSet; HIDDEN_SEATS], u128)> = Vec::new();
        let mut at: HashMap<[u32; HIDDEN_SEATS], usize> = HashMap::new();
        for part in parts {
            let mut part: Vec<_> = part.into_iter().collect();
            part.sort_by_key(|(key, _)| *key);
            for (key, (hands, worlds)) in part {
                match at.get(&key) {
                    Some(i) => placed[*i].1 += worlds,
                    None => {
                        at.insert(key, placed.len());
                        placed.push((hands, worlds));
                    }
                }
            }
        }
        let scale = BigInt::from(scale);
        placed
            .into_iter()
            .map(|(hands, n)| Ok((kernel.world(hands), Weight::new(BigRational::new(BigInt::from(n), scale.clone()))?)))
            .collect()
    }
}

/// The worlds counted so far, by their placement of the shown tiles: a
/// representative's hidden hands and their weight, times the scale.
type Placed = HashMap<[u32; HIDDEN_SEATS], ([DominoSet; HIDDEN_SEATS], u128)>;

/// The completions of `hands` when two seats, `open`, share the tiles
/// `rest`, counted by how they place the `shown` tiles: per placement, the
/// binomial number of ways the free tiles (those either seat may hold) fill
/// the first seat's hand, and one representative.
fn complete(
    hands: &mut [DominoSet; HIDDEN_SEATS],
    rest: DominoSet,
    open: [usize; 2],
    allowed: &[DominoSet; HIDDEN_SEATS],
    capacity: &[usize; HIDDEN_SEATS],
    shown: DominoSet,
    add: &mut impl FnMut([DominoSet; HIDDEN_SEATS], u128),
) {
    let [b, c] = open;
    let hidden = rest.difference(shown);
    // A hidden tile only one seat may hold is that seat's; one neither may
    // hold leaves no world.
    let only_b = hidden.difference(allowed[c]);
    let only_c = hidden.difference(allowed[b]);
    if !only_b.is_subset_of(allowed[b]) || !only_c.is_subset_of(allowed[c]) {
        return;
    }
    let free = hidden.intersection(allowed[b]).intersection(allowed[c]);
    let shown_here = rest.intersection(shown);
    for k in 0..=shown_here.len() {
        for to_b in subsets(shown_here, k) {
            let to_c = shown_here.difference(to_b);
            if !to_b.is_subset_of(allowed[b]) || !to_c.is_subset_of(allowed[c]) {
                continue;
            }
            let Some(need) = capacity[b].checked_sub(only_b.len() + to_b.len()) else { continue };
            if need > free.len() {
                continue;
            }
            let chosen = subsets(free, need).next().expect("enough free tiles");
            hands[b] = to_b.union(only_b).union(chosen);
            hands[c] = rest.difference(hands[b]);
            add(*hands, binomial(free.len(), need));
        }
    }
}

fn binomial(n: usize, k: usize) -> u128 {
    (0..k).fold(1u128, |acc, i| acc * (n - i) as u128 / (i + 1) as u128)
}
