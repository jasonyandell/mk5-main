// SPDX-FileCopyrightText: Jason Yandell
// SPDX-License-Identifier: MIT

//! A position as a seat sees it, rebuilt from the deal's public plays and
//! checked play by play — so the planner never sees an impossible record.

use std::fmt;

use scenario_core::fiber::completions;
use scenario_core::planner::{Decision, Rec};
use scenario_core::rules::{Rules, DOUBLES_SUIT, N_DECLS};

/// A request Zeb refuses, with a plain-language reason.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Error(pub(crate) String);

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Error {}

macro_rules! refuse {
    ($($arg:tt)*) => { return Err(Error(format!($($arg)*))) };
}
pub(crate) use refuse;

/// The rules of a declaration Zeb plays: 0..=6 (that pip is trump),
/// 7 (doubles are trump) or 9 (no trump).
pub(crate) fn rules_for(decl: u8) -> Result<Rules, Error> {
    if decl == DOUBLES_SUIT {
        refuse!("declaration 8 (doubles as their own suit) is nello's; Zeb plays the straight ones, 0 to 7 and 9");
    }
    match Rules::new(decl) {
        Some(rules) => Ok(rules),
        None => refuse!("declaration {decl} is not a declaration (0 to {})", N_DECLS - 1),
    }
}

/// Seven distinct tile ids in 0..=27, as a tile mask.
pub(crate) fn hand_mask(hand: &[u8]) -> Result<u32, Error> {
    let mut mask = 0u32;
    for &t in hand {
        if t > 27 {
            refuse!("tile {t} is not a tile id (0 to 27)");
        }
        if mask >> t & 1 == 1 {
            refuse!("tile {t} appears twice in the hand");
        }
        mask |= 1 << t;
    }
    if hand.len() != 7 {
        refuse!("a hand is 7 tiles, not {}", hand.len());
    }
    Ok(mask)
}

/// A decision Zeb can plan: the seat to play, the hand it was dealt less
/// what it has played, and the public record of every play so far.
pub struct Position {
    rules: Rules,
    seat: u8,
    own: u32,
    rec: Rec,
    bid: u8,
    bidder: u8,
    plays: Vec<(u8, u8)>,
}

impl Position {
    /// The position after `plays` — (seat, tile) in play order from the
    /// opening lead, which is the bidder's — for `seat`, dealt `hand`, in
    /// a hand where `bidder` won the contract `bid` (30..=42 points; 42 is
    /// a mark bid) and declared `decl`. Every play is checked as it is
    /// replayed: the right seat in turn, a tile not yet down, the seat's own
    /// plays from its hand and following suit, no other seat playing a tile
    /// it was dealt or one of a suit that seat already showed it lacks.
    /// Then it must be `seat`'s turn, and some deal of the unseen tiles must
    /// fit the voids the plays have shown.
    pub fn new(decl: u8, bidder: u8, bid: u8, seat: u8, hand: &[u8], plays: &[(u8, u8)]) -> Result<Position, Error> {
        let rules = rules_for(decl)?;
        if bidder > 3 {
            refuse!("bidder {bidder} is not a seat (0 to 3)");
        }
        if seat > 3 {
            refuse!("seat {seat} is not a seat (0 to 3)");
        }
        if !(30..=42).contains(&bid) {
            refuse!("bid {bid} is not a contract: 30 to 41 points, or 42 for a mark bid");
        }
        let dealt = hand_mask(hand)?;
        let mut own = dealt;
        let mut rec = Rec::opening(bidder);
        for (i, &(s, t)) in plays.iter().enumerate() {
            if rec.finished() {
                refuse!("play {i}: the hand is already over");
            }
            if t > 27 {
                refuse!("play {i}: tile {t} is not a tile id (0 to 27)");
            }
            if s != rec.seat() {
                refuse!("play {i}: it is seat {}'s turn, not seat {s}'s", rec.seat());
            }
            if rec.played >> t & 1 == 1 {
                refuse!("play {i}: tile {t} was already played");
            }
            if s == seat {
                if own >> t & 1 == 0 {
                    refuse!("play {i}: seat {s} does not hold tile {t}");
                }
                if rules.legal(own, rec.led()) >> t & 1 == 0 {
                    refuse!("play {i}: seat {s} must follow suit and tile {t} does not");
                }
                own &= !(1 << t);
            } else {
                if dealt >> t & 1 == 1 {
                    refuse!("play {i}: tile {t} was dealt to seat {seat}, not seat {s}");
                }
                if rec.forbid[s as usize] >> t & 1 == 1 {
                    refuse!("play {i}: seat {s} already failed to follow this suit, so cannot hold tile {t}");
                }
            }
            rec = rec.advance(t, &rules);
        }
        if rec.finished() {
            refuse!("the hand is over: there is no play to make");
        }
        if rec.seat() != seat {
            refuse!("it is seat {}'s turn, not seat {seat}'s", rec.seat());
        }
        let sizes = [0, 1, 2, 3].map(|s| rec.hand_size(s));
        if completions(seat, own, rec.played, sizes, rec.forbid) == 0 {
            refuse!("no deal of the unseen tiles fits the suits the other seats have shown they lack");
        }
        Ok(Position { rules, seat, own, rec, bid, bidder, plays: plays.to_vec() })
    }

    /// The declaration.
    pub fn decl(&self) -> u8 {
        self.rules.decl
    }

    /// The seat's legal plays, as a tile mask.
    pub fn legal(&self) -> u32 {
        self.rules.legal(self.own, self.rec.led())
    }

    /// The decision as the planner takes it.
    pub(crate) fn decision(&self) -> Decision<'_> {
        Decision {
            rules: &self.rules,
            viewer: self.seat,
            own: self.own,
            rec: self.rec,
            bid: self.bid as i32,
            bidder: self.bidder,
            plays: &self.plays,
            auction: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Seat 0's hand in the tests: 6-6 6-5 6-4 5-5 5-4 4-4 3-3.
    const HAND: [u8; 7] = [27, 26, 25, 20, 19, 14, 9];

    fn refused(decl: u8, bidder: u8, bid: u8, seat: u8, hand: &[u8], plays: &[(u8, u8)]) -> String {
        Position::new(decl, bidder, bid, seat, hand, plays).err().expect("refused").0
    }

    #[test]
    fn an_opening_lead_is_a_position() {
        let p = Position::new(6, 0, 30, 0, &HAND, &[]).unwrap();
        assert_eq!(p.legal().count_ones(), 7);
        assert_eq!(p.decl(), 6);
    }

    #[test]
    fn the_contract_is_checked() {
        assert!(refused(8, 0, 30, 0, &HAND, &[]).contains("nello"));
        assert!(refused(10, 0, 30, 0, &HAND, &[]).contains("not a declaration"));
        assert!(refused(6, 4, 30, 0, &HAND, &[]).contains("bidder 4"));
        assert!(refused(6, 0, 30, 4, &HAND, &[]).contains("seat 4"));
        assert!(refused(6, 0, 29, 0, &HAND, &[]).contains("bid 29"));
        assert!(refused(6, 0, 43, 0, &HAND, &[]).contains("bid 43"));
    }

    #[test]
    fn the_hand_is_seven_distinct_tiles() {
        assert!(refused(6, 0, 30, 0, &HAND[..6], &[]).contains("7 tiles"));
        assert!(refused(6, 0, 30, 0, &[27, 27, 25, 20, 19, 14, 9], &[]).contains("twice"));
        assert!(refused(6, 0, 30, 0, &[28, 26, 25, 20, 19, 14, 9], &[]).contains("tile 28"));
    }

    #[test]
    fn every_play_is_checked() {
        // Seat 0 bid and leads; seat 1 is to play after the lead.
        assert!(refused(6, 0, 30, 1, &[0, 1, 2, 3, 4, 5, 6], &[]).contains("seat 0's turn"));
        assert!(refused(6, 0, 30, 0, &HAND, &[(1, 0)]).contains("seat 0's turn, not seat 1's"));
        assert!(refused(6, 0, 30, 0, &HAND, &[(0, 0)]).contains("does not hold"));
        assert!(refused(6, 0, 30, 0, &HAND, &[(0, 27), (1, 27)]).contains("already played"));
        assert!(refused(6, 0, 30, 0, &HAND, &[(0, 27), (1, 26)]).contains("dealt to seat 0"));
        assert!(refused(6, 0, 30, 0, &HAND, &[(0, 27), (1, 0), (2, 1)]).contains("seat 3's turn"));
        // Seat 3 bid and leads 6-0, a trump; seat 0 holds trumps, so 3-3 does not follow.
        assert!(refused(6, 3, 30, 0, &HAND, &[(3, 21), (0, 9)]).contains("must follow suit"));
    }

    #[test]
    fn a_seat_that_showed_void_cannot_follow_later() {
        // Trump (6s) led twice; seat 1 shows void on the first, then plays a six.
        let plays = [(0, 27), (1, 0), (2, 21), (3, 22), (0, 26), (1, 23)];
        assert!(refused(6, 0, 30, 0, &HAND, &plays).contains("already failed to follow"));
    }

    #[test]
    fn an_impossible_record_is_refused() {
        // Every other seat shows void in sixes, but seat 0 does not hold them all.
        let plays = [(0, 27), (1, 0), (2, 1), (3, 2)];
        assert!(refused(6, 0, 30, 0, &HAND, &plays).contains("no deal of the unseen tiles"));
    }
}
