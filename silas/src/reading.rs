//! The auction read as evidence (`GAPS.md` 4, #134; policy likelihood, #128,
//! for the auction). A seat that has heard the calls conditions its belief on
//! each named seat's calls being what that seat's bidding policy makes: a world
//! is kept exactly when, in it, every named seat's policy, at that seat's own
//! auction frame (the auction before its call, its dealt hand, the match
//! state, the declarations offered), makes the call it made, and, for the
//! declarer, declares what the frame's declaration is. A bidding policy is a
//! function of the auction frame, deterministic, so the likelihood of a world
//! is zero or one and the posterior is the uniform measure on the worlds kept.
//!
//! The weights are counted, never sampled. A seat's dealt hand is its hand in
//! the world and the tiles it has played, so whether a world is kept depends,
//! seat by seat, only on that seat's hand: [`AuctionReading::evidence`] runs
//! each named seat's policy once on every hand it could hold (at most
//! C(21,7) = 116,280) and keeps the agreeing ones. [`crate::Belief::given_auction`]
//! then counts the worlds whose named hands all agree, by placement of the
//! tiles a query reads: one named seat by the binomial count of the other two
//! seats' completions, two or three by enumerating the second seat's hands
//! for each agreeing hand of the first.
//!
//! The arithmetic is exact, but the number is the uniform deal's only under
//! the hypothesis that each named seat bids by its policy: its authority is
//! `exact_given` those policies (each named by its seat, its name and a
//! digest of its identity, the match state and the declarations offered).

use std::collections::HashMap;
use std::sync::Arc;

use crate::auction::{Auction, AuctionFrame, Call};
use crate::eval::Budget;
use crate::frame::{Frame, HIDDEN_SEATS, World};
use crate::measure::{Authority, Weight};
use crate::nouns::{DominoSet, Seat};
use crate::physics::Physics;
use crate::policy::CompiledPolicy;
use crate::scoring::MatchState;
use crate::{Result, error};

/// Which hidden seats' calls are read, under which bidding policies, at
/// which match state and with which declarations offered.
#[derive(Clone)]
pub struct AuctionReading {
    readers: Vec<(Seat, CompiledPolicy)>,
    state: MatchState,
    declarations: Arc<[Physics]>,
}

/// One named seat's evidence at a frame: its call (and whether its
/// declaration is read too), and how many of the hands it could hold agree.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SeatEvidence {
    pub seat: Seat,
    pub policy: String,
    pub call: Call,
    pub declares: bool,
    pub agreeing: usize,
    pub candidates: usize,
}

/// The auction's evidence at one frame: for each hidden seat that is read,
/// the hands it could hold now that its policy agrees with.
#[derive(Clone)]
pub struct AuctionEvidence {
    frame: Frame,
    agreeing: [Option<Hands>; HIDDEN_SEATS],
    seats: Vec<SeatEvidence>,
    authority: Authority,
}

/// A set of hands within one pool: the list, and a bitmap over the pool's
/// subsets for membership.
#[derive(Clone)]
struct Hands {
    list: Vec<DominoSet>,
    bits: Vec<u64>,
    within: Compress,
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
    fn new(pool: DominoSet, list: Vec<DominoSet>) -> Hands {
        let within = Compress::new(pool);
        let mut bits = vec![0u64; (1usize << pool.len()).div_ceil(64)];
        for hand in &list {
            let i = within.index(*hand);
            bits[i >> 6] |= 1 << (i & 63);
        }
        Hands { list, bits, within }
    }

    fn contains(&self, hand: DominoSet) -> bool {
        let i = self.within.index(hand);
        self.bits[i >> 6] >> (i & 63) & 1 == 1
    }
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

impl AuctionReading {
    /// Read the calls of `readers`, each by its policy, the match at `state`
    /// and `declarations` offered to the winner. Refused for a seat named
    /// twice.
    pub fn new(
        readers: Vec<(Seat, CompiledPolicy)>,
        state: MatchState,
        declarations: Arc<[Physics]>,
    ) -> Result<AuctionReading> {
        for (i, (seat, _)) in readers.iter().enumerate() {
            if readers[..i].iter().any(|(s, _)| s == seat) {
                return Err(error(format!("{seat} is read twice")));
            }
        }
        if declarations.is_empty() {
            return Err(error("the winner must have a declaration to choose"));
        }
        Ok(AuctionReading { readers, state, declarations })
    }

    /// The model each seat's reading rests on: the seat, its policy's name,
    /// and a digest of the policy's identity, the match state and the
    /// declarations offered, in order.
    pub fn models(&self) -> Vec<String> {
        let offered: Vec<&str> = self.declarations.iter().map(Physics::identity).collect();
        self.readers
            .iter()
            .map(|(seat, policy)| {
                let pinned = format!("{}\n{}\n{}", policy.identity(), self.state, offered.join(" "));
                format!("{seat} bids by {}#{:016x}", policy.source().name, digest(&pinned))
            })
            .collect()
    }

    /// The evidence at `frame`: each read seat's agreeing hands. Refused when
    /// a read seat is the viewer (his hand is no evidence to him), when the
    /// frame's declaration is not offered and the declarer is read, and when
    /// a policy refuses a frame. The policies run on every core; `budget`
    /// pays for all of it.
    pub fn evidence(&self, frame: &Frame, budget: &mut Budget) -> Result<AuctionEvidence> {
        let kernel = frame.kernel();
        let auction = frame.auction();
        let declarer = frame.contract().declarer();
        let mut agreeing: [Option<Hands>; HIDDEN_SEATS] = [None, None, None];
        let mut seats = Vec::new();
        let mut authority = Authority::exact();
        for ((seat, policy), model) in self.readers.iter().zip(self.models()) {
            if *seat == frame.viewer() {
                return Err(error(format!("{seat} is the viewer: his own call tells him nothing")));
            }
            let declares = *seat == declarer;
            if declares && !self.declarations.contains(frame.physics()) {
                return Err(error(format!(
                    "the declaration {} is not among those offered, so {seat}'s cannot be read",
                    frame.physics().name()
                )));
            }
            let k = auction.calls().iter().enumerate().position(|(i, _)| auction.seat_of(i) == *seat);
            let k = k.ok_or_else(|| error(format!("{seat} has not called")))?;
            let call = auction.calls()[k];
            let before = Auction::new(auction.config(), auction.shaker(), auction.calls()[..k].to_vec())?;
            let slot = kernel.slot(*seat);
            let played: DominoSet = frame.history().iter().filter(|(s, _)| s == seat).map(|(_, d)| *d).collect();
            let candidates: Vec<DominoSet> =
                subsets(kernel.allowed(frame.physics(), slot), kernel.hidden()[slot].capacity).collect();
            let agrees = |hand: DominoSet, budget: &mut Budget| -> Result<bool> {
                let dealt = hand.union(played);
                let at = AuctionFrame::new(before.clone(), *seat, dealt, self.state, self.declarations.clone())?;
                if policy.bid(&at, budget)?.choice != call {
                    return Ok(false);
                }
                if !declares {
                    return Ok(true);
                }
                let won = AuctionFrame::new(auction.clone(), *seat, dealt, self.state, self.declarations.clone())?;
                Ok(policy.declare(&won, budget)?.choice == *frame.physics())
            };
            let kept = on_every_core(&candidates, budget, agrees)?;
            seats.push(SeatEvidence {
                seat: *seat,
                policy: policy.source().name.clone(),
                call,
                declares,
                agreeing: kept.len(),
                candidates: candidates.len(),
            });
            agreeing[slot] = Some(Hands::new(kernel.pool(), kept));
            authority = authority.meet(&Authority::given(model)?);
        }
        Ok(AuctionEvidence { frame: frame.clone(), agreeing, seats, authority })
    }
}

/// The candidates `keep` agrees with, in order, `keep` run on every core.
/// Each core draws on an equal share of what `budget` has left, and `budget`
/// is charged what they spent together.
fn on_every_core(
    candidates: &[DominoSet],
    budget: &mut Budget,
    keep: impl Fn(DominoSet, &mut Budget) -> Result<bool> + Sync,
) -> Result<Vec<DominoSet>> {
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get()).min(candidates.len().max(1));
    let share = budget.remaining() / threads as u64;
    let parts: Vec<Result<(Vec<DominoSet>, u64)>> = std::thread::scope(|s| {
        let handles: Vec<_> = candidates
            .chunks(candidates.len().div_ceil(threads).max(1))
            .map(|part| {
                let keep = &keep;
                s.spawn(move || {
                    let mut budget = Budget::new(share);
                    let mut kept = Vec::new();
                    for hand in part {
                        if keep(*hand, &mut budget)? {
                            kept.push(*hand);
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

impl AuctionEvidence {
    pub fn frame(&self) -> &Frame {
        &self.frame
    }

    /// Each read seat's evidence, in the order read.
    pub fn seats(&self) -> &[SeatEvidence] {
        &self.seats
    }

    /// `exact_given` every read seat's model.
    pub fn authority(&self) -> &Authority {
        &self.authority
    }

    /// The worlds every read seat's hand agrees in, counted by their
    /// placement of `resolved` (the pool tiles of it): one representative
    /// world and the number of worlds for each placement that has any.
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
        // The worlds that extend each of the first read seat's agreeing hands,
        // by placement of the shown tiles; the hands split across the cores.
        let extend = |h0: DominoSet, placed: &mut Placed| {
            let rest = pool.difference(h0);
            let mut hands = [DominoSet::EMPTY; HIDDEN_SEATS];
            hands[first] = h0;
            let mut add = |hands: [DominoSet; HIDDEN_SEATS], worlds: u128| {
                let key = hands.map(|h| h.intersection(shown).bits());
                placed.entry(key).or_insert((hands, 0)).1 += worlds;
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
                local.entry(h1.intersection(shown).bits()).or_insert((h1, 0)).1 += 1;
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
        Ok(placed.into_iter().map(|(hands, n)| (kernel.world(hands), Weight::integer(n))).collect())
    }
}

/// The worlds counted so far, by their placement of the shown tiles: a
/// representative's hidden hands and how many worlds.
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
