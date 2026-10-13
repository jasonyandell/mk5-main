//! `Straight42`: the reference physics. Plain, spec-faithful and slow; the
//! oracle the tests run against.
//!
//! It covers the nine straight declarations: a pip suit trump (`P0`..`P6`),
//! doubles trump (`DT`), and no trump (`NT`). Doubles as their own suit
//! without trump power (nello's algebra) is not a straight declaration and is
//! omitted.
//!
//! The rules, for a declaration with called set `K` (the trump suit: the tiles
//! bearing the trump pip, or the doubles, or nothing):
//!
//! - a tile in `K` leads the called context `q*`; any other leads its high pip;
//! - the tiles following `q*` are `K`; those following pip `p` are the tiles
//!   bearing `p`, minus `K`;
//! - a tile's trick key in led context `q` is the pair (tier, rank), compared
//!   lexicographically: tier 2 for a trump, 1 for a follower of `q`, 0 for a
//!   slough; rank is the pip sum, except that a double ranks above every
//!   other tile of its suit, and under doubles trump the doubles rank by pip;
//! - the 5-0, 4-1 and 3-2 count five, the 5-5 and 6-4 ten: 35 in all.

use std::fmt;
use std::str::FromStr;

use crate::frame::{HIDDEN_SEATS, Kernel, World};
use crate::nouns::{Context, Domino, DominoSet, Pip};
use crate::physics::Physics;
use crate::{Error, Result, error};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Decl {
    PipTrump(Pip),
    DoublesTrump,
    NoTrump,
}

impl fmt::Display for Decl {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Decl::PipTrump(p) => write!(f, "P{p}"),
            Decl::DoublesTrump => f.write_str("DT"),
            Decl::NoTrump => f.write_str("NT"),
        }
    }
}

impl FromStr for Decl {
    type Err = Error;

    /// `P0`..`P6`, `DT` or `NT`.
    fn from_str(s: &str) -> Result<Decl> {
        match s {
            "DT" => Ok(Decl::DoublesTrump),
            "NT" => Ok(Decl::NoTrump),
            _ => s
                .strip_prefix('P')
                .and_then(|p| p.parse().ok())
                .and_then(Pip::new)
                .map(Decl::PipTrump)
                .ok_or_else(|| error(format!("not a straight declaration: {s}"))),
        }
    }
}

/// Straight 42 under one declaration.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Straight42 {
    decl: Decl,
}

impl Straight42 {
    /// The nine straight declarations.
    pub const ALL: [Straight42; 9] = {
        let mut out = [Straight42 { decl: Decl::NoTrump }; 9];
        let mut p = 0;
        while p < Pip::COUNT {
            out[p] = Straight42 { decl: Decl::PipTrump(Pip::ALL[p]) };
            p += 1;
        }
        out[7] = Straight42 { decl: Decl::DoublesTrump };
        out
    };

    pub const fn new(decl: Decl) -> Straight42 {
        Straight42 { decl }
    }

    pub const fn decl(self) -> Decl {
        self.decl
    }

    fn called(self, tile: Domino) -> bool {
        match self.decl {
            Decl::PipTrump(p) => tile.has(p),
            Decl::DoublesTrump => tile.is_double(),
            Decl::NoTrump => false,
        }
    }
}

impl FromStr for Straight42 {
    type Err = Error;

    fn from_str(s: &str) -> Result<Straight42> {
        s.parse().map(Straight42::new)
    }
}

impl Physics for Straight42 {
    fn identity(&self) -> String {
        format!("straight42/{}", self.decl)
    }

    fn led_context(&self, tile: Domino) -> Context {
        if self.called(tile) { Context::Called } else { Context::Natural(tile.hi()) }
    }

    fn incidence(&self, context: Context) -> DominoSet {
        Domino::ALL
            .into_iter()
            .filter(|d| match context {
                Context::Called => self.called(*d),
                Context::Natural(p) => d.has(p) && !self.called(*d),
            })
            .collect()
    }

    fn trick_key(&self, tile: Domino, led: Context) -> u8 {
        let tier = if self.called(tile) {
            2
        } else if self.follows(tile, led) {
            1
        } else {
            // Tier-zero tiles tie at the bottom: an off-suit play takes
            // nothing and beats nothing (texas-42 lean `Declaration.key`).
            return 0;
        };
        let rank = match (tile.is_double(), self.decl) {
            (true, Decl::DoublesTrump) => tile.hi().value(),
            (true, _) => 12,
            (false, _) => tile.pip_sum(),
        };
        // Ranks are at most 12, so this order is the lexicographic one.
        tier * 16 + rank
    }

    fn count(&self, tile: Domino) -> u32 {
        match tile.pip_sum() {
            5 => 5,
            10 => 10,
            _ => 0,
        }
    }

    fn count_worlds(&self, kernel: &Kernel) -> u128 {
        Fiber::new(self, kernel).count()
    }

    fn worlds<'a>(&'a self, kernel: &'a Kernel) -> Box<dyn Iterator<Item = World> + 'a> {
        let fiber = Fiber::new(self, kernel);
        Box::new((0..fiber.count()).map(move |k| kernel.world(fiber.unrank(k))))
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

const SIDE: usize = crate::frame::MAX_CAPACITY + 1;

impl Fiber {
    fn new(physics: &dyn Physics, kernel: &Kernel) -> Fiber {
        let tiles: Vec<Domino> = kernel.pool().iter().collect();
        let mut fiber = Fiber {
            allowed: std::array::from_fn(|slot| kernel.allowed(physics, slot)),
            capacities: kernel.hidden().map(|h| h.capacity),
            ways: vec![0; (tiles.len() + 1) * SIDE * SIDE * SIDE],
            tiles,
        };
        let n = fiber.tiles.len();
        let at = Fiber::at(n, [0; HIDDEN_SEATS]);
        fiber.ways[at] = 1;
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
