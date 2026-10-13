//! `Straight42`: the reference declarations. Plain and spec-faithful: the
//! tables the tests run against, read into a [`Physics`] by
//! [`Straight42::physics`].
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

use crate::nouns::{Context, Domino, DominoSet, Pip};
use crate::physics::{Declaration, Physics};
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

impl Straight42 {
    /// These rules as a [`Physics`].
    pub fn physics(self) -> Physics {
        Physics::new(&self).expect("the reference tables are well formed")
    }
}

impl Declaration for Straight42 {
    fn name(&self) -> String {
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
        } else if self.incidence(led).contains(tile) {
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
}
