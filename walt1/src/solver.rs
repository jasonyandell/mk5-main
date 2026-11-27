// SPDX-FileCopyrightText: Jason Yandell
// SPDX-License-Identifier: CC0-1.0

//! The solver of WALT.md §5.1, with its two fields: Dice (§5.3) and the level-0 mind (§5.2).
//!
//! A value is carried as the number of alive worlds in which the bidding team makes. That is
//! the rational of §5.1 exactly: a field node's buckets partition `alive`, so their counts
//! add; a deciding node's children share one alive set, so comparing counts compares values.

use crate::rng::{derive, mix64, splitmix_once, FastMap, Rng, GOLDEN};
use crate::rules::{Rules, State};
use crate::sampler::{inner_world, World};
use std::hash::Hash;

/// A set of world indices.
pub trait Alive: Clone + Eq + Hash {
    fn full(n: usize) -> Self;
    fn empty_like(&self) -> Self;
    fn insert(&mut self, i: usize);
    fn count(&self) -> u32;
    /// The lowest index; the set is not empty.
    fn first(&self) -> usize;
    fn for_each(&self, f: impl FnMut(usize));
}

impl Alive for u64 {
    fn full(n: usize) -> Self {
        if n >= 64 {
            !0
        } else {
            (1 << n) - 1
        }
    }
    #[inline]
    fn empty_like(&self) -> Self {
        0
    }
    #[inline]
    fn insert(&mut self, i: usize) {
        *self |= 1 << i;
    }
    #[inline]
    fn count(&self) -> u32 {
        self.count_ones()
    }
    #[inline]
    fn first(&self) -> usize {
        self.trailing_zeros() as usize
    }
    #[inline]
    fn for_each(&self, mut f: impl FnMut(usize)) {
        let mut m = *self;
        while m != 0 {
            f(m.trailing_zeros() as usize);
            m &= m - 1;
        }
    }
}

fn fill_words(words: &mut [u64], n: usize) {
    for (k, w) in words.iter_mut().enumerate() {
        *w = u64::full(n.saturating_sub(64 * k).min(64));
    }
}

fn first_in_words(words: &[u64]) -> usize {
    let k = words.iter().position(|&w| w != 0).expect("an alive set is never empty");
    64 * k + words[k].trailing_zeros() as usize
}

fn each_in_words(words: &[u64], mut f: impl FnMut(usize)) {
    for (k, w) in words.iter().enumerate() {
        w.for_each(|i| f(64 * k + i));
    }
}

impl<const W: usize> Alive for [u64; W] {
    fn full(n: usize) -> Self {
        let mut s = [0; W];
        fill_words(&mut s, n);
        s
    }
    fn empty_like(&self) -> Self {
        [0; W]
    }
    #[inline]
    fn insert(&mut self, i: usize) {
        self[i >> 6] |= 1 << (i & 63);
    }
    fn count(&self) -> u32 {
        self.iter().map(|w| w.count_ones()).sum()
    }
    fn first(&self) -> usize {
        first_in_words(self)
    }
    fn for_each(&self, f: impl FnMut(usize)) {
        each_in_words(self, f)
    }
}

impl Alive for Vec<u64> {
    fn full(n: usize) -> Self {
        let mut s = vec![0; n.div_ceil(64)];
        fill_words(&mut s, n);
        s
    }
    fn empty_like(&self) -> Self {
        vec![0; self.len()]
    }
    fn insert(&mut self, i: usize) {
        self[i >> 6] |= 1 << (i & 63);
    }
    fn count(&self) -> u32 {
        self.iter().map(|w| w.count_ones()).sum()
    }
    fn first(&self) -> usize {
        first_in_words(self)
    }
    fn for_each(&self, f: impl FnMut(usize)) {
        each_in_words(self, f)
    }
}

/// What a seat other than the deciding one plays in world `w` at κ.
pub trait Field {
    /// `legal` has at least two tiles; `key` is `st.key()`.
    fn field_move(&mut self, w: usize, hand: u32, legal: u32, st: &State, key: u64) -> u8;
}

impl<F: Field> Field for &mut F {
    #[inline]
    fn field_move(&mut self, w: usize, hand: u32, legal: u32, st: &State, key: u64) -> u8 {
        (**self).field_move(w, hand, legal, st, key)
    }
}

/// The contract and the seat whose decision is being solved.
#[derive(Clone, Copy)]
pub struct Table<'r> {
    pub rules: &'r Rules,
    pub bid: u8,
    /// 0 for {0,2}, 1 for {1,3}.
    pub bid_team: u8,
}

pub type Memo<A> = FastMap<(u64, A), u32>;

pub struct Solver<'a, A: Alive, F: Field> {
    pub table: Table<'a>,
    pub me: u8,
    pub worlds: &'a [World],
    pub field: F,
    pub memo: Memo<A>,
}

impl<A: Alive, F: Field> Solver<'_, A, F> {
    /// The number of worlds of `alive` in which the bidding team makes, from κ = `st`.
    pub fn value(&mut self, st: &State, alive: A) -> u32 {
        let Table { rules, bid, bid_team } = self.table;
        let bt = bid_team as usize;
        if st.pts[bt] >= bid {
            return alive.count();
        }
        if st.pts[1 - bt] > 42 - bid {
            return 0;
        }
        let total = alive.count();
        if total == 1 {
            return self.makes_in(st, alive.first()) as u32;
        }
        let key = st.key();
        let memo_key = (key, alive);
        if let Some(&v) = self.memo.get(&memo_key) {
            return v;
        }
        let alive = &memo_key.1;
        let seat = st.to_act();
        let v = if seat == self.me {
            // One tile for every alive world: the seat does not see the deal.
            let mut legal = rules.legal(self.worlds[alive.first()][seat as usize] & !st.played, st);
            let maximize = seat & 1 == bid_team;
            let mut best = if maximize { 0 } else { total };
            while legal != 0 {
                let t = legal.trailing_zeros() as u8;
                legal &= legal - 1;
                let v = self.value(&st.play(rules, t), alive.clone());
                if maximize {
                    best = best.max(v);
                    if best == total {
                        break;
                    }
                } else {
                    best = best.min(v);
                    if best == 0 {
                        break;
                    }
                }
            }
            best
        } else {
            // Each world says what this seat plays there; the worlds split by move.
            // Hands differ by world, so the moves can be any of the 28 tiles.
            let mut tiles = [0u8; 28];
            let mut sets: [A; 28] = std::array::from_fn(|_| alive.empty_like());
            let mut n = 0;
            let (worlds, field) = (self.worlds, &mut self.field);
            alive.for_each(|w| {
                let hand = worlds[w][seat as usize] & !st.played;
                let legal = rules.legal(hand, st);
                let t = if legal & (legal - 1) == 0 {
                    legal.trailing_zeros() as u8
                } else {
                    field.field_move(w, hand, legal, st, key)
                };
                let mut k = 0;
                while k < n && tiles[k] != t {
                    k += 1;
                }
                if k == n {
                    tiles[n] = t;
                    n += 1;
                }
                sets[k].insert(w);
            });
            let mut sum = 0;
            for (k, set) in sets.into_iter().take(n).enumerate() {
                sum += self.value(&st.play(rules, tiles[k]), set);
            }
            sum
        };
        self.memo.insert(memo_key, v);
        v
    }

    /// `value` on the single world `w`: does the bidding team make? With one world alive the
    /// field's moves are fixed and only the deciding seat branches, so this walks the tree
    /// without a memo (almost every node of a solve is one of these, and almost none repeat).
    fn makes_in(&mut self, st: &State, w: usize) -> bool {
        let Table { rules, bid, bid_team } = self.table;
        let bt = bid_team as usize;
        let mut st = *st;
        loop {
            if st.pts[bt] >= bid {
                return true;
            }
            if st.pts[1 - bt] > 42 - bid {
                return false;
            }
            let seat = st.to_act();
            let hand = self.worlds[w][seat as usize] & !st.played;
            let mut legal = rules.legal(hand, &st);
            if legal & (legal - 1) == 0 {
                st = st.play(rules, legal.trailing_zeros() as u8);
            } else if seat != self.me {
                let t = self.field.field_move(w, hand, legal, &st, st.key());
                st = st.play(rules, t);
            } else {
                let maximize = seat & 1 == bid_team;
                while legal != 0 {
                    let t = legal.trailing_zeros() as u8;
                    legal &= legal - 1;
                    if self.makes_in(&st.play(rules, t), w) == maximize {
                        return maximize;
                    }
                }
                return !maximize;
            }
        }
    }
}

/// Dice: the fixed tape of §5.3. World `w` at record κ plays legal tile number
/// `SplitMix64(seed_w ⊕ hash(κ)) mod |legal|`, legal tiles in ascending id.
pub struct Dice {
    pub seeds: Vec<u64>,
}

#[inline]
fn nth_tile(mut mask: u32, n: u32) -> u8 {
    for _ in 0..n {
        mask &= mask - 1;
    }
    mask.trailing_zeros() as u8
}

impl Field for Dice {
    #[inline]
    fn field_move(&mut self, w: usize, _hand: u32, legal: u32, _st: &State, key: u64) -> u8 {
        let roll = splitmix_once(self.seeds[w] ^ mix64(key ^ GOLDEN));
        nth_tile(legal, (roll % legal.count_ones() as u64) as u32)
    }
}

/// The level-0 field: every other seat plays its level-0 mind's tile.
pub struct Minds<'r> {
    pub table: Table<'r>,
    pub n0: usize,
    /// The fixed seed of the field. The field is one policy, the same for every request.
    pub seed: u64,
    /// PiKey = (seat, hand, κ) → tile. The seat is the one to act at κ, so (hand, κ) names it.
    cache: FastMap<(u32, u64), u8>,
    memo: Memo<u64>,
    worlds: Vec<World>,
    seeds: Vec<u64>,
    /// Minds solved (cache misses), for the cost report.
    pub solved: u64,
}

pub const FIELD_SEED: u64 = 0x243F_6A88_85A3_08D3;

impl<'r> Minds<'r> {
    pub fn new(table: Table<'r>, n0: usize) -> Self {
        Minds {
            table,
            n0,
            seed: FIELD_SEED,
            cache: FastMap::default(),
            memo: Memo::default(),
            worlds: Vec::new(),
            seeds: Vec::new(),
            solved: 0,
        }
    }

    /// §5.2: draw n0 inner worlds, best-respond to Dice, return the best tile for the seat's
    /// team; ascending id, replaced only on strict improvement.
    fn solve(&mut self, hand: u32, legal: u32, st: &State, key: u64) -> u8 {
        self.solved += 1;
        let seat = st.to_act();
        let mut rng = Rng::new(derive(derive(self.seed, key), hand as u64));
        self.worlds.clear();
        self.seeds.clear();
        for _ in 0..self.n0 {
            self.worlds.push(inner_world(seat, hand, st, &mut rng));
            self.seeds.push(rng.next());
        }
        let dice = Dice { seeds: std::mem::take(&mut self.seeds) };
        let (tile, dice) = if self.n0 <= 64 {
            let mut memo = std::mem::take(&mut self.memo);
            memo.clear();
            let (tile, dice, memo) = best_tile::<u64>(self.table, seat, &self.worlds, dice, memo, legal, st);
            self.memo = memo;
            (tile, dice)
        } else {
            let (tile, dice, _) =
                best_tile::<Vec<u64>>(self.table, seat, &self.worlds, dice, Memo::default(), legal, st);
            (tile, dice)
        };
        self.seeds = dice.seeds;
        tile
    }
}

fn best_tile<A: Alive>(
    table: Table,
    seat: u8,
    worlds: &[World],
    dice: Dice,
    memo: Memo<A>,
    legal: u32,
    st: &State,
) -> (u8, Dice, Memo<A>) {
    let maximize = seat & 1 == table.bid_team;
    let mut solver = Solver { table, me: seat, worlds, field: dice, memo };
    let full = A::full(worlds.len());
    let mut best: Option<(u8, u32)> = None;
    let mut rest = legal;
    while rest != 0 {
        let t = rest.trailing_zeros() as u8;
        rest &= rest - 1;
        let v = solver.value(&st.play(table.rules, t), full.clone());
        let better = match best {
            None => true,
            Some((_, b)) => {
                if maximize {
                    v > b
                } else {
                    v < b
                }
            }
        };
        if better {
            best = Some((t, v));
        }
    }
    (best.expect("a mind is asked only with legal tiles").0, solver.field, solver.memo)
}

impl Field for Minds<'_> {
    fn field_move(&mut self, _w: usize, hand: u32, legal: u32, st: &State, key: u64) -> u8 {
        if let Some(&t) = self.cache.get(&(hand, key)) {
            return t;
        }
        let t = self.solve(hand, legal, st, key);
        self.cache.insert((hand, key), t);
        t
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alive_sets_agree() {
        for n in [1usize, 8, 40, 64, 65, 160, 640] {
            let v = Vec::<u64>::full(n);
            assert_eq!(v.count() as usize, n);
            let mut seen = Vec::new();
            v.for_each(|i| seen.push(i));
            assert_eq!(seen, (0..n).collect::<Vec<_>>());
            if n <= 64 {
                assert_eq!(u64::full(n).count() as usize, n);
            }
            if n <= 256 {
                assert_eq!(<[u64; 4]>::full(n).count() as usize, n);
            }
        }
        let mut s = <[u64; 4]>::full(0);
        s.insert(130);
        let mut seen = Vec::new();
        s.for_each(|i| seen.push(i));
        assert_eq!(seen, vec![130]);
    }

    fn table(rules: &Rules) -> Table<'_> {
        Table { rules, bid: 30, bid_team: 0 }
    }

    /// A mind is a pure function of its key: the same tile on every call, in any order.
    #[test]
    fn minds_are_pure_functions_of_the_key() {
        let rules = Rules::new(6).unwrap();
        let st = State::start(0).play(&rules, 19);
        let hand = 1 << 3 | 1 << 4 | 1 << 12 | 1 << 14 | 1 << 22 | 1 << 23 | 1 << 24;
        let legal = rules.legal(hand, &st);
        assert!(legal.count_ones() > 1);
        let other = State::start(0).play(&rules, 8);
        let mut a = Minds::new(table(&rules), 8);
        let first = a.field_move(0, hand, legal, &st, st.key());
        a.field_move(0, hand, rules.legal(hand, &other), &other, other.key());
        let mut b = Minds::new(table(&rules), 8);
        b.field_move(0, hand, rules.legal(hand, &other), &other, other.key());
        assert_eq!(b.field_move(5, hand, legal, &st, st.key()), first);
        assert!(legal >> first & 1 == 1);
    }

    /// With every hand fixed and Dice silenced by forced plays, the count is the plain
    /// minimax of the deciding seat: check a two-trick ending by hand.
    #[test]
    fn value_counts_makes_on_a_known_ending() {
        // Sixes trump, bid 30, team 0 has banked 28 with two tricks left (2 + whatever counts).
        // Seat 0 leads holding 6-6 and 5-5: it wins both tricks, so it makes.
        let rules = Rules::new(6).unwrap();
        let t = crate::rules::tile_id;
        let world: World = [
            1 << t(6, 6) | 1 << t(5, 5),
            1 << t(1, 0) | 1 << t(2, 0),
            1 << t(1, 1) | 1 << t(2, 1),
            1 << t(3, 0) | 1 << t(3, 1),
        ];
        let held = world.iter().fold(0, |a, b| a | b);
        let st = State {
            played: crate::rules::ALL_TILES & !held,
            trick: [0; 3],
            len: 0,
            leader: 0,
            pts: [28, 2],
        };
        let worlds = [world; 3];
        let mut solver = Solver {
            table: table(&rules),
            me: 0,
            worlds: &worlds,
            field: Dice { seeds: vec![1, 2, 3] },
            memo: Memo::<u64>::default(),
        };
        assert_eq!(solver.value(&st, u64::full(3)), 3);
        // The same ending for a seat that cannot win a trick: bid 42 is already lost.
        let mut lost = Solver {
            table: Table { rules: &rules, bid: 42, bid_team: 0 },
            me: 0,
            worlds: &worlds,
            field: Dice { seeds: vec![1, 2, 3] },
            memo: Memo::<u64>::default(),
        };
        assert_eq!(lost.value(&st, u64::full(3)), 0);
    }
}
