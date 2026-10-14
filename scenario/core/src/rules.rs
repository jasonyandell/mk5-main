// SPDX-FileCopyrightText: Jason Yandell
// SPDX-License-Identifier: CC0-1.0

//! The rules of 42 as tables, one set per declaration — `scenario/rules.py`.
//!
//! The Python side tabulates `forge.oracle.tables`, the rules authority;
//! this module derives the same tables from the same definitions, so the
//! core needs nothing from Python to play. `scenario/tests/test_compiled.py`
//! holds the two equal, entry for entry, for all ten declarations.
//!
//! Tiles are the double-six set in triangular order: (high, low) with
//! high ≥ low has id high·(high+1)/2 + low, so 0-0 is 0 and 6-6 is 27.
//! A tile set is a bitmask, bit t for tile t.

/// Doubles are trump (declarations 0..=6 make that pip suit trump).
pub const DOUBLES_TRUMP: u8 = 7;
/// The doubles form their own suit with no trump power: nello's algebra.
pub const DOUBLES_SUIT: u8 = 8;
/// No trump: a tile leads its high pip.
pub const NO_TRUMP: u8 = 9;
/// Declarations are the ids 0..N_DECLS.
pub const N_DECLS: u8 = 10;

/// The led suit of a called-suit lead (pip suits are 0..=6).
const CALLED: u8 = 7;

/// The tile id of (high, low), high ≥ low.
pub const fn tile(high: u8, low: u8) -> u8 {
    high * (high + 1) / 2 + low
}

/// (high, low) of a tile id in 0..28.
pub const fn pips(t: u8) -> (u8, u8) {
    let mut high = 0;
    while tile(high + 1, 0) <= t {
        high += 1;
    }
    (high, t - tile(high, 0))
}

/// One declaration's rules as lookup tables.
pub struct Rules {
    /// The declaration these tables are for.
    pub decl: u8,
    /// The led suit when the tile leads: its high pip, or 7 for the called suit.
    pub led_suit: [u8; 28],
    /// The tiles that follow each led suit.
    pub follow_bits: [u32; 8],
    /// Trick rank under each led suit; the highest rank takes the trick.
    pub rank: [[u8; 28]; 8],
    /// Count points on the tile.
    pub count: [u8; 28],
}

impl Rules {
    /// The tables for declaration `decl`, or None outside 0..N_DECLS.
    pub fn new(decl: u8) -> Option<Rules> {
        if decl >= N_DECLS {
            return None;
        }
        let called = |t: u8| {
            let (high, low) = pips(t);
            match decl {
                DOUBLES_TRUMP | DOUBLES_SUIT => high == low,
                NO_TRUMP => false,
                pip => high == pip || low == pip,
            }
        };
        let follows = |t: u8, led: u8| {
            let (high, low) = pips(t);
            if led == CALLED {
                called(t)
            } else {
                (high == led || low == led) && !called(t)
            }
        };
        // Within a pip suit the double is highest, then the pip sum; within
        // the doubles (as a led suit, or as trumps under doubles-trump) the
        // high pip. Trumps outrank followers, which outrank everything else.
        let in_pips = |t: u8| {
            let (high, low) = pips(t);
            if high == low {
                14
            } else {
                high + low
            }
        };
        let rank = |t: u8, led: u8| {
            if decl <= DOUBLES_TRUMP && called(t) {
                32 + if decl == DOUBLES_TRUMP { pips(t).0 } else { in_pips(t) }
            } else if follows(t, led) {
                16 + if led == CALLED { pips(t).0 } else { in_pips(t) }
            } else {
                0
            }
        };
        let count = |t: u8| match pips(t) {
            (5, 5) | (6, 4) => 10,
            (5, 0) | (4, 1) | (3, 2) => 5,
            _ => 0,
        };
        Some(Rules {
            decl,
            led_suit: std::array::from_fn(|t| if called(t as u8) { CALLED } else { pips(t as u8).0 }),
            follow_bits: std::array::from_fn(|led| {
                (0..28u8).filter(|&t| follows(t, led as u8)).fold(0, |m, t| m | 1 << t)
            }),
            rank: std::array::from_fn(|led| std::array::from_fn(|t| rank(t as u8, led as u8))),
            count: std::array::from_fn(|t| count(t as u8)),
        })
    }

    /// The tiles that follow the suit the tile `lead` leads.
    #[inline]
    pub fn followers(&self, lead: u8) -> u32 {
        self.follow_bits[self.led_suit[lead as usize] as usize]
    }

    /// The follow-suit-legal subset of `hand` when the trick was led by
    /// `led`: all of it when leading (None) or when it holds none of the suit.
    #[inline]
    pub fn legal(&self, hand: u32, led: Option<u8>) -> u32 {
        match led {
            None => hand,
            Some(l) => {
                let f = hand & self.followers(l);
                if f != 0 {
                    f
                } else {
                    hand
                }
            }
        }
    }

    /// The position (0..4, play order) of the tile that takes the trick; the first of equals.
    #[inline]
    pub fn winner_offset(&self, trick: &[u8; 4]) -> u8 {
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

    /// A trick is worth one point plus the count it holds.
    #[inline]
    pub fn trick_points(&self, trick: &[u8; 4]) -> u8 {
        1 + trick.iter().map(|&t| self.count[t as usize]).sum::<u8>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tile_ids_are_triangular() {
        assert_eq!((tile(0, 0), tile(1, 0), tile(1, 1), tile(6, 6)), (0, 1, 2, 27));
        for t in 0..28 {
            let (high, low) = pips(t);
            assert!(low <= high && high <= 6 && tile(high, low) == t);
        }
    }

    #[test]
    fn ten_declarations_and_no_more() {
        assert!((0..N_DECLS).all(|d| Rules::new(d).is_some_and(|r| r.decl == d)));
        assert!(Rules::new(N_DECLS).is_none());
    }

    #[test]
    fn the_count_is_35_points_on_five_tiles() {
        let r = Rules::new(0).unwrap();
        assert_eq!(r.count.iter().map(|&c| c as u32).sum::<u32>(), 35);
        assert_eq!(r.count.iter().filter(|&&c| c > 0).count(), 5);
    }

    #[test]
    fn every_tile_follows_its_own_lead() {
        for d in 0..N_DECLS {
            let r = Rules::new(d).unwrap();
            for t in 0..28u8 {
                assert_eq!(r.followers(t) >> t & 1, 1, "decl {d} tile {t}");
            }
        }
    }

    #[test]
    fn a_trump_takes_a_led_double() {
        let trick = [tile(5, 5), tile(5, 1), tile(6, 0), tile(5, 2)];
        assert_eq!(Rules::new(6).unwrap().winner_offset(&trick), 2);
        assert_eq!(Rules::new(NO_TRUMP).unwrap().winner_offset(&trick), 0);
    }
}
