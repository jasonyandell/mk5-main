// SPDX-FileCopyrightText: Jason Yandell
// SPDX-License-Identifier: CC0-1.0

//! Straight 42 for one declaration (WALT.md §2): tiles, suits, trick winner, points, and the
//! reduced record κ of §3.

pub const ALL_TILES: u32 = (1 << 28) - 1;
pub const DOUBLES_TRUMP: u8 = 7;
pub const NO_TRUMP: u8 = 9;

/// `id(hi, lo) = hi·(hi+1)/2 + lo`, hi ≥ lo.
pub const fn tile_id(hi: u8, lo: u8) -> u8 {
    hi * (hi + 1) / 2 + lo
}

const fn pips() -> [(u8, u8); 28] {
    let mut out = [(0, 0); 28];
    let mut hi = 0;
    while hi < 7 {
        let mut lo = 0;
        while lo <= hi {
            out[tile_id(hi, lo) as usize] = (hi, lo);
            lo += 1;
        }
        hi += 1;
    }
    out
}

/// (hi, lo) of each tile id.
pub const PIPS: [(u8, u8); 28] = pips();

/// The tables of one declaration.
pub struct Rules {
    /// `follow[lead]`: the tiles that follow suit when `lead` is led (`lead` among them).
    pub follow: [u32; 28],
    /// `power[lead][t]`: how `t` ranks in a trick `lead` opened. Trumps over the led suit over
    /// everything else (0); distinct among the tiles that can win.
    pub power: [[u8; 28]; 28],
    pub pts: [u8; 28],
}

impl Rules {
    pub fn new(decl: u8) -> Option<Rules> {
        if !(decl <= DOUBLES_TRUMP || decl == NO_TRUMP) {
            return None;
        }
        let is_trump = |t: usize| {
            let (hi, lo) = PIPS[t];
            match decl {
                0..=6 => hi == decl || lo == decl,
                DOUBLES_TRUMP => hi == lo,
                _ => false,
            }
        };
        let trump_mask: u32 = (0..28).filter(|&t| is_trump(t)).map(|t| 1u32 << t).sum();
        // A non-trump suit is every non-trump tile carrying the pip.
        let suit_mask = |p: u8| -> u32 {
            (0..28)
                .filter(|&t| !is_trump(t) && (PIPS[t].0 == p || PIPS[t].1 == p))
                .map(|t| 1u32 << t)
                .sum()
        };
        // Rank inside a suit named by pip `p`: the double highest, then by the other pip.
        let rank_in = |t: usize, p: u8| -> u8 {
            let (hi, lo) = PIPS[t];
            if hi == lo {
                7
            } else if hi == p {
                lo
            } else {
                hi
            }
        };
        let mut follow = [0u32; 28];
        let mut power = [[0u8; 28]; 28];
        for lead in 0..28 {
            let led_pip = PIPS[lead].0;
            let led = if is_trump(lead) { trump_mask } else { suit_mask(led_pip) };
            follow[lead] = led;
            for t in 0..28 {
                power[lead][t] = if is_trump(t) {
                    20 + if decl == DOUBLES_TRUMP { PIPS[t].0 } else { rank_in(t, decl) }
                } else if led >> t & 1 == 1 {
                    10 + rank_in(t, led_pip)
                } else {
                    0
                };
            }
        }
        let mut pts = [0u8; 28];
        for (hi, lo, p) in [(5, 5, 10), (6, 4, 10), (5, 0, 5), (4, 1, 5), (3, 2, 5)] {
            pts[tile_id(hi, lo) as usize] = p;
        }
        Some(Rules { follow, power, pts })
    }

    /// The tiles of `hand` that may be played at `st`.
    #[inline]
    pub fn legal(&self, hand: u32, st: &State) -> u32 {
        if st.len == 0 {
            return hand;
        }
        let f = hand & self.follow[st.trick[0] as usize];
        if f != 0 {
            f
        } else {
            hand
        }
    }
}

/// The reduced record κ: played tiles, the trick's leader and plays, points banked by team.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct State {
    pub played: u32,
    pub trick: [u8; 3],
    pub len: u8,
    pub leader: u8,
    /// Banked points of teams {0,2} and {1,3}.
    pub pts: [u8; 2],
}

impl State {
    pub fn start(bidder: u8) -> State {
        State { played: 0, trick: [0; 3], len: 0, leader: bidder, pts: [0; 2] }
    }

    #[inline]
    pub fn to_act(&self) -> u8 {
        (self.leader + self.len) & 3
    }

    /// κ in one word: the memo key, the mind's cache key and what Dice hashes.
    #[inline]
    pub fn key(&self) -> u64 {
        self.played as u64
            | (self.leader as u64) << 28
            | (self.len as u64) << 30
            | (self.trick[0] as u64) << 32
            | (self.trick[1] as u64) << 37
            | (self.trick[2] as u64) << 42
            | (self.pts[0] as u64) << 47
            | (self.pts[1] as u64) << 53
    }

    /// κ after the seat to act plays `tile`.
    #[inline]
    pub fn play(&self, rules: &Rules, tile: u8) -> State {
        let mut s = *self;
        s.played |= 1 << tile;
        if s.len < 3 {
            s.trick[s.len as usize] = tile;
            s.len += 1;
            return s;
        }
        let lead = s.trick[0] as usize;
        let row = &rules.power[lead];
        let four = [s.trick[0], s.trick[1], s.trick[2], tile];
        let mut win = 0usize;
        let mut best = row[lead];
        let mut points = 1u8;
        for (i, &t) in four.iter().enumerate() {
            points += rules.pts[t as usize];
            if row[t as usize] > best {
                best = row[t as usize];
                win = i;
            }
        }
        s.leader = (s.leader + win as u8) & 3;
        s.pts[(s.leader & 1) as usize] += points;
        s.trick = [0; 3];
        s.len = 0;
        s
    }

    /// Tiles `seat` still holds at κ, given that every seat was dealt seven.
    #[inline]
    pub fn hand_size(&self, seat: u8) -> u32 {
        let done = self.played.count_ones() / 4;
        let in_trick = (seat.wrapping_sub(self.leader) & 3) < self.len;
        7 - done - in_trick as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mask(tiles: &[(u8, u8)]) -> u32 {
        tiles.iter().map(|&(h, l)| 1u32 << tile_id(h, l)).sum()
    }

    #[test]
    fn ids() {
        assert_eq!(tile_id(0, 0), 0);
        assert_eq!(tile_id(1, 0), 1);
        assert_eq!(tile_id(1, 1), 2);
        assert_eq!(tile_id(6, 6), 27);
        assert_eq!(PIPS[19], (5, 4));
    }

    #[test]
    fn points_total_42() {
        let r = Rules::new(3).unwrap();
        assert_eq!(r.pts.iter().map(|&p| p as u32).sum::<u32>() + 7, 42);
    }

    #[test]
    fn pip_trump_suits() {
        let r = Rules::new(6).unwrap();
        // A trump lead calls trump: all seven sixes.
        assert_eq!(r.follow[tile_id(6, 2) as usize].count_ones(), 7);
        // 5-4 leads fives; 6-5 is trump, never a five.
        let fives = r.follow[tile_id(5, 4) as usize];
        assert_eq!(fives, mask(&[(5, 0), (5, 1), (5, 2), (5, 3), (5, 4), (5, 5)]));
        // Rank: the double highest, then the other pip descending.
        let p = &r.power[tile_id(5, 4) as usize];
        assert!(p[tile_id(5, 5) as usize] > p[tile_id(5, 4) as usize]);
        assert!(p[tile_id(5, 4) as usize] > p[tile_id(5, 3) as usize]);
        assert!(p[tile_id(6, 0) as usize] > p[tile_id(5, 5) as usize]);
        assert_eq!(p[tile_id(4, 4) as usize], 0);
        let t = &r.power[tile_id(6, 0) as usize];
        assert!(t[tile_id(6, 6) as usize] > t[tile_id(6, 5) as usize]);
        assert!(t[tile_id(6, 5) as usize] > t[tile_id(6, 0) as usize]);
    }

    #[test]
    fn doubles_trump_suits() {
        let r = Rules::new(DOUBLES_TRUMP).unwrap();
        assert_eq!(r.follow[tile_id(3, 3) as usize].count_ones(), 7);
        // 5-4 leads fives: the six non-double fives.
        assert_eq!(
            r.follow[tile_id(5, 4) as usize],
            mask(&[(5, 0), (5, 1), (5, 2), (5, 3), (5, 4), (6, 5)])
        );
        let p = &r.power[tile_id(3, 3) as usize];
        assert!(p[tile_id(6, 6) as usize] > p[tile_id(5, 5) as usize]);
        assert!(p[tile_id(1, 1) as usize] > p[tile_id(0, 0) as usize]);
        let q = &r.power[tile_id(5, 4) as usize];
        assert!(q[tile_id(6, 5) as usize] > q[tile_id(5, 4) as usize]);
        assert!(q[tile_id(0, 0) as usize] > q[tile_id(6, 5) as usize]);
    }

    #[test]
    fn no_trump_doubles_high() {
        let r = Rules::new(NO_TRUMP).unwrap();
        let fives = r.follow[tile_id(5, 5) as usize];
        assert_eq!(fives.count_ones(), 7);
        let p = &r.power[tile_id(5, 2) as usize];
        assert!(p[tile_id(5, 5) as usize] > p[tile_id(6, 5) as usize]);
        assert!(p[tile_id(6, 5) as usize] > p[tile_id(5, 4) as usize]);
        assert_eq!(p[tile_id(6, 6) as usize], 0);
    }

    #[test]
    fn trick_banks_to_the_winner() {
        // Sixes trump, seat 0 leads 5-4; seat 1 trumps with 6-0; 5-5 and 6-4 fall.
        let r = Rules::new(6).unwrap();
        let mut s = State::start(0);
        for t in [tile_id(5, 4), tile_id(6, 0), tile_id(5, 5), tile_id(6, 4)] {
            s = s.play(&r, t);
        }
        assert_eq!(s.leader, 3); // 6-4 outranks 6-0 in trumps
        assert_eq!(s.pts, [0, 21]);
        assert_eq!(s.len, 0);
        assert_eq!(s.hand_size(2), 6);
    }

    #[test]
    fn must_follow_when_able() {
        let r = Rules::new(6).unwrap();
        let s = State::start(0).play(&r, tile_id(5, 4));
        let hand = mask(&[(5, 1), (6, 5), (2, 2)]);
        assert_eq!(r.legal(hand, &s), mask(&[(5, 1)]));
        let hand = mask(&[(6, 5), (2, 2)]);
        assert_eq!(r.legal(hand, &s), hand);
        assert_eq!(s.hand_size(0), 6);
        assert_eq!(s.hand_size(1), 7);
    }
}
