// SPDX-FileCopyrightText: Jason Yandell
// SPDX-License-Identifier: CC0-1.0

//! The auction: each seat calls once, from the shaker's left, a pass or a bid
//! above the high bid. The hand's [`Contract`] is derived from it: the last
//! seat to bid declares, at its bid. An [`AuctionFrame`] is the auction as one
//! seat sees it before the declaration, the frame a bid rule reads.
//!
//! Legality is texas-42's Lean `AuctionState.legalBid` (`Auction.lean`),
//! which the TypeScript engine and the arena implement too
//! (`tests/auction.rs` holds Silas to both): a point bid is 30 to 41 and
//! above the high bid; before any marks bid, one or two marks (42, 84) may be
//! bid; after a marks bid only one mark more; marks up to the configured cap.
//! Four passes are a passed-out hand: no contract, and the next seat shakes
//! (the tournament's reshake), unless the configuration forces the shaker to
//! bid when the other three have passed (the common variation).

use std::fmt;
use std::str::FromStr;
use std::sync::Arc;

use crate::frame::{Bid, Contract};
use crate::nouns::{Domino, DominoSet, Seat};
use crate::physics::Physics;
use crate::scoring::MatchState;
use crate::{Error, Result, error};

/// The least point bid.
pub const MIN_BID: u32 = 30;

/// One seat's turn: a pass or a bid. Calls are ordered as the auction orders
/// them: a pass below every bid, point bids by points, then marks bids by
/// marks.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Call {
    Pass,
    Bid(Bid),
}

impl fmt::Display for Call {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Call::Pass => f.write_str("pass"),
            Call::Bid(bid) => bid.fmt(f),
        }
    }
}

impl FromStr for Call {
    type Err = Error;

    /// `pass`, or a bid as the table says it (`30`, `84`).
    fn from_str(s: &str) -> Result<Call> {
        match s {
            "pass" => Ok(Call::Pass),
            _ => s.parse().map(Call::Bid),
        }
    }
}

/// The rules of one auction beyond the fixed ones: the highest marks bid,
/// and whether the shaker must bid when the other three have passed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct AuctionConfig {
    max_marks: u32,
    forced: bool,
}

impl AuctionConfig {
    /// No cap below what four calls can reach (a first marks bid is at most
    /// two and each overcall adds one: five, Lean's `mark_le_ceiling`), and a
    /// passed-out hand is reshaken: the tournament's auction
    /// (`wiki/topics/rules-of-42.md`), and the arena's without its forced
    /// bid.
    pub const STRAIGHT: AuctionConfig = AuctionConfig { max_marks: 5, forced: false };

    /// Marks bids up to `max_marks` (at least one); with `forced`, the shaker
    /// may not pass when the first three have.
    pub fn new(max_marks: u32, forced: bool) -> Result<AuctionConfig> {
        if max_marks == 0 {
            return Err(error("an auction allows a marks bid of at least one mark"));
        }
        Ok(AuctionConfig { max_marks, forced })
    }

    pub fn max_marks(self) -> u32 {
        self.max_marks
    }

    pub fn forced(self) -> bool {
        self.forced
    }
}

impl Default for AuctionConfig {
    fn default() -> AuctionConfig {
        AuctionConfig::STRAIGHT
    }
}

/// An auction so far: its rules, the shaker, and the calls in order, the
/// shaker's left first. Every call was legal when made ([`Auction::new`]
/// refuses one that was not), so the high bid is the last bid.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Auction {
    config: AuctionConfig,
    shaker: Seat,
    calls: Vec<Call>,
}

impl Auction {
    /// The auction `shaker` shook, after `calls`. Refused when a call was
    /// illegal when made, or when there are more than four.
    pub fn new(config: AuctionConfig, shaker: Seat, calls: Vec<Call>) -> Result<Auction> {
        let mut auction = Auction { config, shaker, calls: Vec::with_capacity(calls.len()) };
        for call in calls {
            auction = auction.after(call)?;
        }
        Ok(auction)
    }

    /// The shortest auction under the straight rules that `declarer` wins
    /// with `bid`: he speaks first and the others pass, except that a bid of
    /// three marks or more needs the seats before him to have raised from 84
    /// one mark at a time. This is the record of a hand given with its
    /// contract alone.
    pub fn won_by(declarer: Seat, bid: Bid) -> Result<Auction> {
        let raises = match bid {
            Bid::Marks(m) if m > 2 => m - 2,
            _ => 0,
        } as usize;
        if raises >= Seat::COUNT {
            return Err(error(format!("no auction of four calls reaches {bid}")));
        }
        let shaker = declarer.plus(Seat::COUNT - 1 - raises);
        let mut calls: Vec<Call> = (0..raises).map(|k| Call::Bid(Bid::Marks(2 + k as u32))).collect();
        calls.push(Call::Bid(bid));
        calls.resize(Seat::COUNT, Call::Pass);
        Auction::new(AuctionConfig::STRAIGHT, shaker, calls)
    }

    /// The auction one call later. Refused when the call is illegal.
    pub fn after(&self, call: Call) -> Result<Auction> {
        if !self.legal(call) {
            let legal: Vec<String> = self.legal_calls().iter().map(Call::to_string).collect();
            return Err(error(format!(
                "{} may not call {call} after {self}; the legal calls are {}",
                self.actor().map_or("no seat".to_owned(), |s| s.to_string()),
                if legal.is_empty() { "none".to_owned() } else { legal.join(", ") }
            )));
        }
        let mut calls = self.calls.clone();
        calls.push(call);
        Ok(Auction { config: self.config, shaker: self.shaker, calls })
    }

    pub fn config(&self) -> AuctionConfig {
        self.config
    }

    pub fn shaker(&self) -> Seat {
        self.shaker
    }

    /// The calls so far, in order, the shaker's left first.
    pub fn calls(&self) -> &[Call] {
        &self.calls
    }

    /// The seat that made the `k`-th call (from zero).
    pub fn seat_of(&self, k: usize) -> Seat {
        self.shaker.plus(1 + k)
    }

    /// The calls so far with the seats that made them.
    pub fn history(&self) -> impl Iterator<Item = (Seat, Call)> + '_ {
        self.calls.iter().enumerate().map(|(k, call)| (self.seat_of(k), *call))
    }

    /// `seat`'s call, once it has made one.
    pub fn call_of(&self, seat: Seat) -> Option<Call> {
        self.history().find(|(s, _)| *s == seat).map(|(_, call)| call)
    }

    /// Every seat has called.
    pub fn is_complete(&self) -> bool {
        self.calls.len() == Seat::COUNT
    }

    /// The seat to call next; none once every seat has.
    pub fn actor(&self) -> Option<Seat> {
        (!self.is_complete()).then(|| self.seat_of(self.calls.len()))
    }

    /// The high bid so far and its bidder: the last bid.
    pub fn high(&self) -> Option<(Seat, Bid)> {
        self.history()
            .filter_map(|(seat, call)| if let Call::Bid(bid) = call { Some((seat, bid)) } else { None })
            .last()
    }

    /// Whether `call` is legal for the seat to call: a pass, unless the
    /// configuration forces the shaker to bid over three passes; a point bid
    /// from 30 to 41 above the high bid; one or two marks before any marks
    /// bid, and only one mark more after one, up to the cap.
    pub fn legal(&self, call: Call) -> bool {
        if self.is_complete() {
            return false;
        }
        let high = self.high().map(|(_, bid)| bid);
        match call {
            Call::Pass => !(self.config.forced && self.calls.len() == Seat::COUNT - 1 && high.is_none()),
            Call::Bid(Bid::Points(n)) => {
                (MIN_BID..=41).contains(&n)
                    && match high {
                        None => true,
                        Some(Bid::Points(h)) => h < n,
                        Some(Bid::Marks(_)) => false,
                    }
            }
            Call::Bid(Bid::Marks(m)) => {
                (1..=self.config.max_marks).contains(&m)
                    && match high {
                        Some(Bid::Marks(r)) => m == r + 1,
                        _ => m <= 2,
                    }
            }
        }
    }

    /// The legal calls, in the auction's order (a pass first, when legal).
    pub fn legal_calls(&self) -> Vec<Call> {
        let candidates = std::iter::once(Call::Pass)
            .chain((MIN_BID..=41).map(|n| Call::Bid(Bid::Points(n))))
            .chain((1..=self.config.max_marks.min(5)).map(|m| Call::Bid(Bid::Marks(m))));
        candidates.filter(|call| self.legal(*call)).collect()
    }

    /// The least legal bid: what "bid only enough" raises to.
    pub fn least_raise(&self) -> Option<Bid> {
        self.legal_calls().into_iter().find_map(|call| if let Call::Bid(bid) = call { Some(bid) } else { None })
    }

    /// What the complete auction settled: the last bidder and his bid. None
    /// while it is under way, and none for a passed-out hand.
    pub fn result(&self) -> Option<(Seat, Bid)> {
        if self.is_complete() { self.high() } else { None }
    }

    /// The contract the complete auction settles, under `declaration`.
    /// Refused while it is under way and for a passed-out hand.
    pub fn contract(&self, declaration: Physics) -> Result<Contract> {
        match self.result() {
            Some((declarer, bid)) => Contract::new(declarer, bid, declaration),
            None if self.is_complete() => Err(error(format!("{self} passed out: no contract; the next seat shakes"))),
            None => Err(error(format!("{self} is under way: no contract yet"))),
        }
    }
}

impl fmt::Display for Auction {
    /// `S3 shook: S0 30, S1 pass, ...`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} shook:", self.shaker)?;
        if self.calls.is_empty() {
            return f.write_str(" no calls");
        }
        for (k, (seat, call)) in self.history().enumerate() {
            write!(f, "{} {seat} {call}", if k == 0 { "" } else { "," })?;
        }
        Ok(())
    }
}

/// The auction as one seat sees it before the declaration: the auction so
/// far, the viewer, the seven tiles it was dealt, the match state, and the
/// declarations the winner may choose from. A bid or declaring rule reads
/// nothing else, so a bid policy is a function of this frame.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AuctionFrame {
    auction: Auction,
    viewer: Seat,
    dealt: DominoSet,
    state: MatchState,
    declarations: Arc<[Physics]>,
}

impl AuctionFrame {
    /// Refused unless `dealt` is seven tiles, the match is undecided, and the
    /// declarations are some, distinct, and agree on every tile's count (the
    /// count is the one table the auction reads before a declaration).
    pub fn new(
        auction: Auction,
        viewer: Seat,
        dealt: DominoSet,
        state: MatchState,
        declarations: Arc<[Physics]>,
    ) -> Result<AuctionFrame> {
        if dealt.len() != crate::frame::MAX_CAPACITY {
            return Err(error(format!("a seat is dealt seven tiles, not {}", dealt.len())));
        }
        if state.won().is_some() {
            return Err(error(format!("the match is decided at {state}: no hand is bid")));
        }
        let Some(first) = declarations.first() else {
            return Err(error("the winner must have a declaration to choose"));
        };
        for (i, d) in declarations.iter().enumerate() {
            if declarations[..i].contains(d) {
                return Err(error(format!("the declaration {} is offered twice", d.name())));
            }
            if Domino::ALL.iter().any(|t| d.count(*t) != first.count(*t)) {
                return Err(error(format!("{} and {} count the tiles differently", d.name(), first.name())));
            }
        }
        Ok(AuctionFrame { auction, viewer, dealt, state, declarations })
    }

    pub fn auction(&self) -> &Auction {
        &self.auction
    }

    pub fn viewer(&self) -> Seat {
        self.viewer
    }

    /// The seven tiles the viewer was dealt.
    pub fn dealt(&self) -> DominoSet {
        self.dealt
    }

    pub fn state(&self) -> &MatchState {
        &self.state
    }

    /// The declarations the winner may choose from, in order.
    pub fn declarations(&self) -> &[Physics] {
        &self.declarations
    }

    /// The count points on a tile, which every declaration offered agrees on.
    pub fn count(&self, tile: Domino) -> u32 {
        self.declarations[0].count(tile)
    }

    /// The frame one call later, the same seat viewing.
    pub fn after(&self, call: Call) -> Result<AuctionFrame> {
        Ok(AuctionFrame { auction: self.auction.after(call)?, ..self.clone() })
    }
}
