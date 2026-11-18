// SPDX-FileCopyrightText: Jason Yandell
// SPDX-License-Identifier: CC0-1.0

//! The two beliefs of WALT.md §4. A world is `[u32; 4]`: each seat's remaining tiles.

use crate::rng::Rng;
use crate::rules::{State, ALL_TILES};

pub type World = [u32; 4];

fn tiles_of(mask: u32, out: &mut [u8; 28]) -> usize {
    let mut m = mask;
    let mut n = 0;
    while m != 0 {
        out[n] = m.trailing_zeros() as u8;
        n += 1;
        m &= m - 1;
    }
    n
}

fn shuffle(tiles: &mut [u8], rng: &mut Rng) {
    for i in (1..tiles.len()).rev() {
        let j = rng.below(i as u64 + 1) as usize;
        tiles.swap(i, j);
    }
}

/// The inner belief: the unseen tiles dealt uniformly to the other three seats by hand size.
/// Voids are ignored on purpose.
pub fn inner_world(seat: u8, hand: u32, st: &State, rng: &mut Rng) -> World {
    let mut tiles = [0u8; 28];
    let n = tiles_of(ALL_TILES & !st.played & !hand, &mut tiles);
    shuffle(&mut tiles[..n], rng);
    let mut world = [0u32; 4];
    world[seat as usize] = hand;
    let mut at = 0;
    for o in 0..4u8 {
        if o == seat {
            continue;
        }
        for _ in 0..st.hand_size(o) {
            world[o as usize] |= 1 << tiles[at];
            at += 1;
        }
    }
    debug_assert_eq!(at, n);
    world
}

/// The outer belief: uniform on the worlds that give each other seat its forced hand size and
/// no tile of a suit it has shown void in.
///
/// Rather than deal by sizes and reject, this counts. Unseen tiles fall into seven classes by
/// which of the three seats may hold them; `ways[i][a][b]` is the number of ways to place the
/// tiles of classes `i..` when the first two seats still need `a` and `b` tiles. A draw walks
/// the classes, choosing each split in proportion to the completions it leaves. Same
/// distribution as rejection, and it cannot stall late in a hand when few deals survive.
pub struct OuterSampler {
    seat: u8,
    hand: u32,
    others: [u8; 3],
    caps: [u8; 3],
    class_tiles: [Vec<u8>; 8],
    /// Tiles in classes `i..`.
    rest: [u8; 9],
    ways: Box<[[[u64; 8]; 8]; 9]>,
    binom: [[u64; 22]; 22],
}

impl OuterSampler {
    /// `forbidden[s]`: tiles seat `s` cannot hold. `None` when no world fits the record.
    pub fn new(seat: u8, hand: u32, st: &State, forbidden: &[u32; 4]) -> Option<OuterSampler> {
        let mut others = [0u8; 3];
        let mut caps = [0u8; 3];
        let mut k = 0;
        for o in 0..4u8 {
            if o != seat {
                others[k] = o;
                caps[k] = st.hand_size(o) as u8;
                k += 1;
            }
        }
        let mut class_tiles: [Vec<u8>; 8] = Default::default();
        let mut tiles = [0u8; 28];
        let n = tiles_of(ALL_TILES & !st.played & !hand, &mut tiles);
        for &t in &tiles[..n] {
            let mut class = 0;
            for (i, &o) in others.iter().enumerate() {
                if forbidden[o as usize] >> t & 1 == 0 {
                    class |= 1 << i;
                }
            }
            class_tiles[class].push(t);
        }
        if !class_tiles[0].is_empty() || n != caps.iter().map(|&c| c as usize).sum::<usize>() {
            return None;
        }
        let mut binom = [[0u64; 22]; 22];
        for i in 0..22 {
            binom[i][0] = 1;
            for j in 1..=i {
                binom[i][j] = binom[i - 1][j - 1] + if j < i { binom[i - 1][j] } else { 0 };
            }
        }
        let mut rest = [0u8; 9];
        for i in (1..8).rev() {
            rest[i] = rest[i + 1] + class_tiles[i].len() as u8;
        }
        let mut s = OuterSampler {
            seat,
            hand,
            others,
            caps,
            class_tiles,
            rest,
            ways: Box::new([[[0; 8]; 8]; 9]),
            binom,
        };
        s.ways[8][0][0] = 1;
        for i in (1..8).rev() {
            for a in 0..8usize {
                for b in 0..8usize {
                    let mut total = 0u64;
                    s.splits(i, a, b, |_, _, w| total += w);
                    s.ways[i][a][b] = total;
                }
            }
        }
        if s.count() == 0 {
            return None;
        }
        Some(s)
    }

    /// Every split `(x0, x1)` of class `i`'s tiles between the first two seats (the third
    /// takes the rest) with seats needing `a` and `b`, and the number of deals it leaves.
    fn splits(&self, i: usize, a: usize, b: usize, mut f: impl FnMut(usize, usize, u64)) {
        let m = self.class_tiles[i].len();
        let max0 = if i & 1 != 0 { m.min(a) } else { 0 };
        for x0 in 0..=max0 {
            let max1 = if i & 2 != 0 { (m - x0).min(b) } else { 0 };
            for x1 in 0..=max1 {
                let x2 = m - x0 - x1;
                if x2 != 0 && i & 4 == 0 {
                    continue;
                }
                // The third seat's need after this class must still be a hand size.
                let need2 = self.rest[i + 1] as isize - (a - x0) as isize - (b - x1) as isize;
                if !(0..=7).contains(&need2) {
                    continue;
                }
                let w = self.binom[m][x0] * self.binom[m - x0][x1] * self.ways[i + 1][a - x0][b - x1];
                if w != 0 {
                    f(x0, x1, w);
                }
            }
        }
    }

    /// The number of worlds in the belief's support.
    pub fn count(&self) -> u64 {
        self.ways[1][self.caps[0] as usize][self.caps[1] as usize]
    }

    pub fn draw(&self, rng: &mut Rng) -> World {
        let mut world = [0u32; 4];
        world[self.seat as usize] = self.hand;
        let (mut a, mut b) = (self.caps[0] as usize, self.caps[1] as usize);
        for i in 1..8 {
            let m = self.class_tiles[i].len();
            if m == 0 {
                continue;
            }
            let mut r = rng.below(self.ways[i][a][b]);
            let mut pick = None;
            self.splits(i, a, b, |x0, x1, w| {
                if pick.is_none() {
                    if r < w {
                        pick = Some((x0, x1));
                    } else {
                        r -= w;
                    }
                }
            });
            let (x0, x1) = pick.expect("split weights sum to the node's count");
            let mut tiles = [0u8; 28];
            tiles[..m].copy_from_slice(&self.class_tiles[i]);
            shuffle(&mut tiles[..m], rng);
            for (j, &t) in tiles[..m].iter().enumerate() {
                let k = if j < x0 {
                    0
                } else if j < x0 + x1 {
                    1
                } else {
                    2
                };
                world[self.others[k] as usize] |= 1 << t;
            }
            a -= x0;
            b -= x1;
        }
        world
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rules::Rules;
    use std::collections::HashMap;

    #[test]
    fn inner_world_deals_by_size() {
        let r = Rules::new(6).unwrap();
        let st = State::start(0).play(&r, 19).play(&r, 16);
        let hand = 0b111_1111 << 20;
        let mut rng = Rng::new(1);
        for _ in 0..50 {
            let w = inner_world(2, hand, &st, &mut rng);
            assert_eq!(w[2], hand);
            assert_eq!(w[0].count_ones(), 6);
            assert_eq!(w[1].count_ones(), 6);
            assert_eq!(w[3].count_ones(), 7);
            assert_eq!(w[0] | w[1] | w[2] | w[3] | st.played, ALL_TILES);
        }
    }

    #[test]
    fn no_voids_counts_the_whole_fiber() {
        let st = State::start(0);
        let s = OuterSampler::new(0, 0b111_1111, &st, &[0; 4]).unwrap();
        assert_eq!(s.count(), 399_072_960); // 21! / (7!)^3
    }

    /// A late position small enough to enumerate: the count matches brute force and the draws
    /// are uniform over exactly the lawful worlds.
    #[test]
    fn voids_are_respected_and_draws_uniform() {
        // Four tricks done: 16 played, viewer (seat 1) holds 3, nine unseen tiles 3/3/3.
        let st = State { played: (1 << 16) - 1, trick: [0; 3], len: 0, leader: 0, pts: [10, 6] };
        let hand = 0b111 << 16;
        let unseen: Vec<u8> = (19..28).collect();
        let mut forbidden = [0u32; 4];
        forbidden[0] = 1 << 19 | 1 << 20 | 1 << 21;
        forbidden[2] = 1 << 21 | 1 << 22;
        forbidden[3] = 1 << 27;
        // Brute force over 3^9 assignments.
        let mut lawful = Vec::new();
        for code in 0..3u32.pow(9) {
            let mut w = [0u32; 4];
            w[1] = hand;
            let mut c = code;
            for &t in &unseen {
                w[[0, 2, 3][(c % 3) as usize]] |= 1 << t;
                c /= 3;
            }
            let sized = [0, 2, 3].iter().all(|&s| w[s].count_ones() == 3);
            let clean = (0..4).all(|s| w[s] & forbidden[s] == 0);
            if sized && clean {
                lawful.push(w);
            }
        }
        let s = OuterSampler::new(1, hand, &st, &forbidden).unwrap();
        assert_eq!(s.count(), lawful.len() as u64);
        let mut rng = Rng::new(99);
        let mut seen: HashMap<World, u32> = HashMap::new();
        let per = 200;
        for _ in 0..lawful.len() * per {
            *seen.entry(s.draw(&mut rng)).or_default() += 1;
        }
        assert_eq!(seen.len(), lawful.len());
        for w in &lawful {
            let c = seen[w];
            assert!(c > 130 && c < 280, "count {c} for a world expected near {per}");
        }
    }

    #[test]
    fn an_impossible_record_has_no_sampler() {
        let st = State::start(0);
        let mut forbidden = [0u32; 4];
        forbidden[1] = 1 << 27;
        forbidden[2] = 1 << 27;
        forbidden[3] = 1 << 27;
        assert!(OuterSampler::new(0, 0b111_1111, &st, &forbidden).is_none());
    }
}
