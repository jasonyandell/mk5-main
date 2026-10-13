//! The Physics boundary. A rules engine supplies one declaration's four rule
//! tables through [`Declaration`]: the led context of each tile, the tiles
//! following each context, the trick key, and the count. Silas reads them
//! once into a [`Physics`], a value, and computes everything else itself from
//! those tables alone: legality, the trick's winner and points, the trump
//! context, and the fiber of hidden worlds (counted and enumerated). No
//! engine can override a derived rule, so a number Silas labels `Exact`
//! rests on Silas's code and the tables, nothing else.
//!
//! A physics's identity is a content hash of its tables, with the trick keys
//! reduced to their order (only the order matters). Two physics are equal
//! exactly when their tables are, whatever engine supplied them or whatever
//! they are called.

use std::fmt;
use std::sync::Arc;

use crate::frame::{HIDDEN_SEATS, Kernel, MAX_CAPACITY, World};
use crate::nouns::{Context, Domino, DominoSet, Seat};
use crate::{Result, error};

/// One declaration's rule tables, as a rules engine supplies them. Silas
/// reads them once ([`Physics::new`]) and derives everything else.
pub trait Declaration {
    /// A name for people, such as `straight42/P6`. It is not the identity.
    fn name(&self) -> String;

    /// The context a tile leads.
    fn led_context(&self, tile: Domino) -> Context;

    /// The tiles that follow `context` (its effective incidence): the tiles a
    /// seat must play, if it holds any, when `context` is led.
    fn incidence(&self, context: Context) -> DominoSet;

    /// The tile's strength in a trick led in `led`. Only the order matters:
    /// the greatest key among the four plays takes the trick, the first of
    /// equal keys.
    fn trick_key(&self, tile: Domino, led: Context) -> u8;

    /// The count points on a tile.
    fn count(&self, tile: Domino) -> u32;
}

/// One declaration's rules: its four tables, and everything Silas derives
/// from them. Cheap to clone (the tables are shared).
#[derive(Clone)]
pub struct Physics(Arc<Tables>);

struct Tables {
    name: String,
    identity: String,
    led: [Context; Domino::COUNT],
    incidence: [DominoSet; Context::COUNT],
    /// Each tile's trick key in each led context, as its rank among that
    /// context's distinct keys: zero is the weakest.
    rank: [[u8; Domino::COUNT]; Context::COUNT],
    count: [u32; Domino::COUNT],
    trump: Option<Context>,
}

impl Physics {
    /// Read a declaration's tables. Refused when a tile does not follow the
    /// context it leads.
    pub fn new(declaration: &dyn Declaration) -> Result<Physics> {
        let led = Domino::ALL.map(|d| declaration.led_context(d));
        let incidence = Context::ALL.map(|q| declaration.incidence(q));
        let count = Domino::ALL.map(|d| declaration.count(d));
        if let Some(d) = Domino::ALL.into_iter().find(|d| !incidence[led[d.index()].index()].contains(*d)) {
            return Err(error(format!("{d} does not follow the context it leads, {}", led[d.index()])));
        }
        if count.iter().try_fold(0u32, |total, c| total.checked_add(*c)).is_none() {
            return Err(error("the count overflows"));
        }
        let rank = Context::ALL.map(|q| {
            let keys = Domino::ALL.map(|d| declaration.trick_key(d, q));
            let mut distinct = keys.to_vec();
            distinct.sort_unstable();
            distinct.dedup();
            keys.map(|k| distinct.binary_search(&k).expect("a key is among the keys") as u8)
        });
        let mut tables =
            Tables { name: declaration.name(), identity: String::new(), led, incidence, rank, count, trump: None };
        tables.identity = format!("tables-{:032x}", tables.content_hash());
        tables.trump = tables.find_trump();
        Ok(Physics(Arc::new(tables)))
    }

    /// The name the declaration gave itself, for people.
    pub fn name(&self) -> &str {
        &self.0.name
    }

    /// The content hash of the tables: one word, so policy text can carry it.
    /// Equal identities are equal tables.
    pub fn identity(&self) -> &str {
        &self.0.identity
    }

    pub fn led_context(&self, tile: Domino) -> Context {
        self.0.led[tile.index()]
    }

    pub fn incidence(&self, context: Context) -> DominoSet {
        self.0.incidence[context.index()]
    }

    /// The tile's trick key in `led`: its rank among the context's keys.
    pub fn trick_key(&self, tile: Domino, led: Context) -> u8 {
        self.0.rank[led.index()][tile.index()]
    }

    pub fn count(&self, tile: Domino) -> u32 {
        self.0.count[tile.index()]
    }

    /// Whether `tile` follows `context`.
    pub fn follows(&self, tile: Domino, context: Context) -> bool {
        self.incidence(context).contains(tile)
    }

    /// The tiles that would take a trick led in `led` from `tile`.
    pub fn beats(&self, led: Context, tile: Domino) -> DominoSet {
        let key = self.trick_key(tile, led);
        Domino::ALL.into_iter().filter(|e| self.trick_key(*e, led) > key).collect()
    }

    /// The tiles that beat `tile` when `tile` itself is led.
    pub fn threat(&self, tile: Domino) -> DominoSet {
        self.beats(self.led_context(tile), tile)
    }

    /// The legal plays from `hand` in a trick led by the tile `lead`:
    /// anything when leading (`lead` is `None`), otherwise the tiles that
    /// follow the context `lead` leads, or anything when the hand holds none
    /// of them.
    pub fn legal_plays(&self, hand: DominoSet, lead: Option<Domino>) -> DominoSet {
        match lead.map(|t| hand.intersection(self.incidence(self.led_context(t)))) {
            Some(follows) if !follows.is_empty() => follows,
            _ => hand,
        }
    }

    /// The seat that takes a trick of four distinct plays, in play order from
    /// `leader`: the first play with the greatest key in the led context.
    pub fn trick_winner(&self, leader: Seat, plays: [Domino; 4]) -> Seat {
        let led = self.led_context(plays[0]);
        let mut best = 0;
        for i in 1..4 {
            if self.trick_key(plays[i], led) > self.trick_key(plays[best], led) {
                best = i;
            }
        }
        leader.plus(best)
    }

    /// One point for the trick plus the count on its tiles.
    pub fn trick_points(&self, plays: [Domino; 4]) -> u32 {
        1 + plays.iter().map(|d| self.count(*d)).sum::<u32>()
    }

    /// Every point a hand can bank: a point per trick plus all the count.
    pub fn total_points(&self) -> u32 {
        (Domino::COUNT / Seat::COUNT) as u32 + self.0.count.iter().sum::<u32>()
    }

    /// The trump context: the one whose tiles follow no other context and,
    /// played to a trick led in any other context, beat every tile that
    /// follows it. None under no trump, and when the called suit has no trump
    /// power (the doubles as their own suit).
    pub fn trump(&self) -> Option<Context> {
        self.0.trump
    }

    /// The number of hidden worlds consistent with `kernel`: assignments of
    /// its pool to the hidden seats that meet every capacity and respect every
    /// shown void.
    pub fn count_worlds(&self, kernel: &Kernel) -> u128 {
        Fiber::new(self, kernel).count()
    }

    /// Those worlds, each exactly once, lazily, in a fixed order.
    pub fn worlds<'a>(&'a self, kernel: &'a Kernel) -> impl Iterator<Item = World> + 'a {
        let fiber = Fiber::new(self, kernel);
        (0..fiber.count()).map(move |k| kernel.world(fiber.unrank(k)))
    }
}

impl Tables {
    /// FNV-1a, 128 bits, over the tables' canonical bytes (the name excluded).
    fn content_hash(&self) -> u128 {
        const OFFSET: u128 = 0x6c62_272e_07bb_0142_62b8_2175_6295_c58d;
        const PRIME: u128 = 0x0000_0000_0100_0000_0000_0000_0000_013b;
        let mut bytes = b"silas-tables-v1".to_vec();
        bytes.extend(self.led.iter().map(|q| q.index() as u8));
        bytes.extend(self.incidence.iter().flat_map(|s| s.bits().to_le_bytes()));
        bytes.extend(self.rank.iter().flatten());
        bytes.extend(self.count.iter().flat_map(|c| c.to_le_bytes()));
        bytes.iter().fold(OFFSET, |h, b| (h ^ u128::from(*b)).wrapping_mul(PRIME))
    }

    fn find_trump(&self) -> Option<Context> {
        let trumps: Vec<Context> = Context::ALL
            .into_iter()
            .filter(|q| {
                let called = self.incidence[q.index()];
                !called.is_empty()
                    && Context::ALL.into_iter().filter(|r| r != q).all(|r| {
                        let followers = self.incidence[r.index()];
                        let rank = &self.rank[r.index()];
                        called.is_disjoint(followers)
                            && called.iter().all(|t| followers.iter().all(|f| rank[t.index()] > rank[f.index()]))
                    })
            })
            .collect();
        match trumps.as_slice() {
            [q] => Some(*q),
            _ => None,
        }
    }
}

impl PartialEq for Physics {
    /// Equal tables, whatever the names.
    fn eq(&self, other: &Physics) -> bool {
        let (a, b) = (&self.0, &other.0);
        (a.led, a.incidence, a.rank, a.count) == (b.led, b.incidence, b.rank, b.count)
    }
}

impl Eq for Physics {}

impl std::hash::Hash for Physics {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.identity.hash(state);
    }
}

impl fmt::Debug for Physics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Physics({} {})", self.0.name, self.0.identity)
    }
}

/// The hidden worlds of a kernel, counted tile by tile: `ways[i][c]` is the
/// number of ways to deal the pool's tiles from the `i`-th on so that hidden
/// slot `j` receives exactly `c[j]` of them. A world is numbered by the order
/// of its choices, so the `k`-th world is read off the table directly.
struct Fiber {
    tiles: Vec<Domino>,
    allowed: [DominoSet; HIDDEN_SEATS],
    capacities: [usize; HIDDEN_SEATS],
    ways: Vec<u128>,
}

const SIDE: usize = MAX_CAPACITY + 1;

impl Fiber {
    fn new(physics: &Physics, kernel: &Kernel) -> Fiber {
        let tiles: Vec<Domino> = kernel.pool().iter().collect();
        let mut fiber = Fiber {
            allowed: std::array::from_fn(|slot| kernel.allowed(physics, slot)),
            capacities: kernel.hidden().map(|h| h.capacity),
            ways: vec![0; (tiles.len() + 1) * SIDE * SIDE * SIDE],
            tiles,
        };
        let n = fiber.tiles.len();
        fiber.ways[Fiber::at(n, [0; HIDDEN_SEATS])] = 1;
        for i in (0..n).rev() {
            for c0 in 0..SIDE {
                for c1 in 0..SIDE {
                    for c2 in 0..SIDE {
                        let c = [c0, c1, c2];
                        let total =
                            (0..HIDDEN_SEATS).filter_map(|j| fiber.after(i, c, j)).map(|next| fiber.ways[next]).sum();
                        fiber.ways[Fiber::at(i, c)] = total;
                    }
                }
            }
        }
        fiber
    }

    fn at(i: usize, c: [usize; HIDDEN_SEATS]) -> usize {
        ((i * SIDE + c[0]) * SIDE + c[1]) * SIDE + c[2]
    }

    /// The table entry after giving tile `i` to slot `j`, if that is allowed.
    fn after(&self, i: usize, mut c: [usize; HIDDEN_SEATS], j: usize) -> Option<usize> {
        if c[j] == 0 || !self.allowed[j].contains(self.tiles[i]) {
            return None;
        }
        c[j] -= 1;
        Some(Fiber::at(i + 1, c))
    }

    fn count(&self) -> u128 {
        self.ways[Fiber::at(0, self.capacities)]
    }

    /// The hidden hands of world number `k`, for `k < count()`.
    fn unrank(&self, mut k: u128) -> [DominoSet; HIDDEN_SEATS] {
        let mut hands = [DominoSet::EMPTY; HIDDEN_SEATS];
        let mut c = self.capacities;
        for (i, tile) in self.tiles.iter().enumerate() {
            for j in 0..HIDDEN_SEATS {
                if let Some(next) = self.after(i, c, j) {
                    if k < self.ways[next] {
                        hands[j].insert(*tile);
                        c[j] -= 1;
                        break;
                    }
                    k -= self.ways[next];
                }
            }
        }
        hands
    }
}
