//! Exact-uniform sampling of hidden-hand completions — `scenario/fiber.py`.
//!
//! The number of lawful assignments of the unseen pool to the three hidden
//! seats (given their hand sizes and the tiles each provably cannot hold) is
//! counted by a dynamic program over (next tile, remaining capacities), and a
//! sample is drawn tile by tile in proportion to those counts. The draws
//! consume the SplitMix stream exactly as the Python does. A fiber no larger
//! than the sample asked for is enumerated outright, in the Python's order.

use crate::planner::{SplitMix, ALL_TILES};

struct Fiber<'a> {
    tiles: &'a [u8],
    forbid: [u32; 3],
    dims: [usize; 3],
    memo: Vec<i64>, // -1 = not yet counted
}

impl<'a> Fiber<'a> {
    fn new(tiles: &'a [u8], sizes: [u8; 3], forbid: [u32; 3]) -> Self {
        let dims = [sizes[0] as usize + 1, sizes[1] as usize + 1, sizes[2] as usize + 1];
        let memo = vec![-1; (tiles.len() + 1) * dims[0] * dims[1] * dims[2]];
        Fiber { tiles, forbid, dims, memo }
    }

    #[inline]
    fn slot(&self, i: usize, c: [u8; 3]) -> usize {
        ((i * self.dims[0] + c[0] as usize) * self.dims[1] + c[1] as usize) * self.dims[2] + c[2] as usize
    }

    fn ways(&mut self, i: usize, c: [u8; 3]) -> i64 {
        let slot = self.slot(i, c);
        let known = self.memo[slot];
        if known >= 0 {
            return known;
        }
        let v = if i == self.tiles.len() {
            if c == [0, 0, 0] {
                1
            } else {
                0
            }
        } else {
            let t = self.tiles[i];
            let mut total = 0;
            for j in 0..3 {
                if c[j] > 0 && (self.forbid[j] >> t) & 1 == 0 {
                    let mut d = c;
                    d[j] -= 1;
                    total += self.ways(i + 1, d);
                }
            }
            total
        };
        self.memo[slot] = v;
        v
    }

    /// Every assignment, tile by tile, seat 0 first (`enumerate_fiber`).
    fn enumerate(&mut self, sizes: [u8; 3]) -> Vec<[u32; 3]> {
        let mut out = Vec::new();
        let mut caps = sizes;
        let mut masks = [0u32; 3];
        self.walk(0, &mut caps, &mut masks, &mut out);
        out
    }

    fn walk(&mut self, i: usize, caps: &mut [u8; 3], masks: &mut [u32; 3], out: &mut Vec<[u32; 3]>) {
        if i == self.tiles.len() {
            out.push(*masks);
            return;
        }
        let t = self.tiles[i];
        for j in 0..3 {
            if caps[j] > 0 && (self.forbid[j] >> t) & 1 == 0 {
                caps[j] -= 1;
                if self.ways(i + 1, *caps) != 0 {
                    masks[j] |= 1 << t;
                    self.walk(i + 1, caps, masks, out);
                    masks[j] ^= 1 << t;
                }
                caps[j] += 1;
            }
        }
    }

    /// One assignment of the pool to the three seats, uniform over the fiber.
    fn sample(&mut self, sizes: [u8; 3], rng: &mut SplitMix) -> [u32; 3] {
        let mut caps = sizes;
        let mut total = self.ways(0, caps);
        assert!(total > 0, "empty fiber: no lawful completion exists");
        let mut masks = [0u32; 3];
        for i in 0..self.tiles.len() {
            let t = self.tiles[i];
            let u = rng.random() * total as f64;
            let mut acc: i64 = 0;
            let mut chosen: i32 = -1;
            for j in 0..3 {
                if caps[j] > 0 && (self.forbid[j] >> t) & 1 == 0 {
                    caps[j] -= 1;
                    let w = self.ways(i + 1, caps);
                    caps[j] += 1;
                    acc += w;
                    if u < acc as f64 {
                        chosen = j as i32;
                        total = w;
                        break;
                    }
                }
            }
            if chosen < 0 {
                // floating-point edge: take the last admissible seat
                for j in (0..3).rev() {
                    if caps[j] > 0 && (self.forbid[j] >> t) & 1 == 0 {
                        caps[j] -= 1;
                        total = self.ways(i + 1, caps);
                        caps[j] += 1;
                        chosen = j as i32;
                        break;
                    }
                }
            }
            let j = chosen as usize;
            caps[j] -= 1;
            masks[j] |= 1 << t;
        }
        masks
    }
}

/// The viewer's view of the hidden seats: who they are (absolute seats, in
/// order), the unseen pool, and each one's hand size and provable voids.
struct Hidden {
    others: [usize; 3],
    tiles: Vec<u8>,
    sizes: [u8; 3],
    forbid: [u32; 3],
}

impl Hidden {
    fn new(viewer: u8, own: u32, played: u32, sizes4: [u8; 4], forbid4: [u32; 4]) -> Hidden {
        let mut others = [0usize; 3];
        let mut k = 0;
        for s in 0..4 {
            if s != viewer as usize {
                others[k] = s;
                k += 1;
            }
        }
        let pool = ALL_TILES & !played & !own;
        Hidden {
            others,
            tiles: (0..28u8).filter(|&t| (pool >> t) & 1 == 1).collect(),
            sizes: others.map(|s| sizes4[s]),
            forbid: others.map(|s| forbid4[s]),
        }
    }
}

/// How many lawful deals of the unseen pool to the three hidden seats fit
/// their hand sizes and the voids they have shown. Zero means no deal
/// explains the record from the viewer's chair.
pub fn completions(viewer: u8, own: u32, played: u32, sizes4: [u8; 4], forbid4: [u32; 4]) -> i64 {
    let h = Hidden::new(viewer, own, played, sizes4, forbid4);
    Fiber::new(&h.tiles, h.sizes, h.forbid).ways(0, h.sizes)
}

/// `n` scenarios from the viewer's fiber, as remaining-hand masks in absolute
/// seat order (the viewer's own hand in its slot): the whole fiber when it has
/// at most `n` members, else `n` uniform draws. The flag says which: true
/// when the scenarios ARE the fiber (the exact belief), false for a sample.
pub fn sample_worlds(
    viewer: u8,
    own: u32,
    played: u32,
    sizes4: [u8; 4],
    forbid4: [u32; 4],
    n: usize,
    rng: &mut SplitMix,
) -> (Vec<[u32; 4]>, bool) {
    let Hidden { others, tiles, sizes, forbid } = Hidden::new(viewer, own, played, sizes4, forbid4);
    assert_eq!(
        sizes.iter().map(|&s| s as usize).sum::<usize>(),
        tiles.len(),
        "pool size must equal the hidden hand sizes"
    );
    let mut fiber = Fiber::new(&tiles, sizes, forbid);
    let exact = fiber.ways(0, sizes) as usize <= n;
    let drawn: Vec<[u32; 3]> = if exact {
        fiber.enumerate(sizes)
    } else {
        (0..n).map(|_| fiber.sample(sizes, rng)).collect()
    };
    let worlds = drawn
        .into_iter()
        .map(|hidden| {
            let mut hands = [0u32; 4];
            hands[viewer as usize] = own;
            for j in 0..3 {
                hands[others[j]] = hidden[j];
            }
            hands
        })
        .collect();
    (worlds, exact)
}
