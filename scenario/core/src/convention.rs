// SPDX-FileCopyrightText: Jason Yandell
// SPDX-License-Identifier: CC0-1.0

//! Conventions: a joint deviation of two partners — `scenario/convention.py`,
//! whose docstring states every gate in the mover's chair.
//!
//! A rule reads the mover's seat role (bidder, or the bidder's partner), its
//! own remaining hand and the public record, and where its gate holds names
//! the allowed tiles. The sender plays the plan's best among them; the reader
//! keeps a world only if every gated decision of its partner, replayed with
//! the hand that world gives him, played an allowed tile. A gate that allows
//! every legal tile is no gate.

use crate::planner::Rec;
use crate::rules::Rules;

/// Trick rank tier 2: a tile with trump power.
const TRUMP_RANK: u8 = 32;

/// The named conventions (`scenario.convention.NAMES`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Rule {
    /// C2: the bidder's partner hands the bidder the top trump.
    Handoff,
    /// C2's sender, read as the book reads it: a top trump is a singleton.
    HandoffSingleton,
    /// C3: when the partner donates count on the bidder's trick, his largest.
    Biggest,
    /// C1: the bidder makes the trick guaranteed; the partner donates only on it.
    Donation,
}

impl Rule {
    /// The rule of a name, or None.
    pub fn named(name: &str) -> Option<Rule> {
        match name {
            "handoff" => Some(Rule::Handoff),
            "handoff-singleton" => Some(Rule::HandoffSingleton),
            "biggest" => Some(Rule::Biggest),
            "donation" => Some(Rule::Donation),
            _ => None,
        }
    }
}

/// A convention and which halves the live seat runs (S, R or SR).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Convention {
    /// The rule.
    pub rule: Rule,
    /// The live seat plays the rule's tile at its gates.
    pub send: bool,
    /// The live seat draws its worlds from the rule-only posterior.
    pub read: bool,
}

/// The tiles with trump power.
pub fn trumps(rules: &Rules) -> u32 {
    (0..28u8).filter(|&t| rules.rank[7][t as usize] >= TRUMP_RANK).fold(0, |m, t| m | 1 << t)
}

fn bits(mut mask: u32) -> impl Iterator<Item = u8> {
    std::iter::from_fn(move || {
        if mask == 0 {
            None
        } else {
            let t = mask.trailing_zeros() as u8;
            mask &= mask - 1;
            Some(t)
        }
    })
}

/// The highest trump of a set of trumps.
fn top(rules: &Rules, mask: u32) -> Option<u8> {
    bits(mask).max_by_key(|&t| rules.rank[7][t as usize])
}

fn counts(rules: &Rules, hand: u32) -> u32 {
    bits(hand).filter(|&t| rules.count[t as usize] > 0).fold(0, |m, t| m | 1 << t)
}

fn partner(seat: u8) -> u8 {
    (seat + 2) % 4
}

fn trick_mask(rec: &Rec) -> u32 {
    rec.trick[..rec.n as usize].iter().fold(0, |m, &t| m | 1 << t)
}

/// The handoff's gate: the top unplayed trump, when the mover is the bidder's
/// partner following the bidder's trump lead and holds it.
fn handoff_top(rules: &Rules, seat: u8, bidder: u8, own: u32, rec: &Rec) -> Option<u8> {
    if seat != partner(bidder) || rec.n == 0 || rec.leader != bidder {
        return None;
    }
    let t = trumps(rules);
    if (t >> rec.trick[0]) & 1 == 0 {
        return None;
    }
    top(rules, t & !rec.played).filter(|&x| (own >> x) & 1 == 1)
}

fn handoff(rules: &Rules, seat: u8, bidder: u8, own: u32, rec: &Rec) -> Option<u32> {
    handoff_top(rules, seat, bidder, own, rec).map(|x| 1 << x)
}

fn singleton(rules: &Rules, seat: u8, bidder: u8, own: u32, rec: &Rec) -> Option<u32> {
    let x = handoff_top(rules, seat, bidder, own, rec)?;
    let others = own & trumps(rules) & !(1 << x);
    Some(if others != 0 { others } else { 1 << x })
}

fn biggest(rules: &Rules, seat: u8, bidder: u8, own: u32, rec: &Rec) -> Option<u32> {
    if seat != partner(bidder) || rec.n == 0 || rec.leader != bidder {
        return None;
    }
    let lead = rec.trick[0];
    let suit = rules.led_suit[lead as usize] as usize;
    if own & rules.follow_bits[suit] != 0 {
        return None;
    }
    let rank = &rules.rank[suit];
    let lr = rank[lead as usize];
    if rec.trick[1..rec.n as usize].iter().any(|&t| rank[t as usize] > lr) {
        return None;
    }
    if (trumps(rules) >> lead) & 1 == 0 && bits(rules.follow_bits[suit] & !rec.played).any(|t| rank[t as usize] > lr) {
        return None;
    }
    let c = counts(rules, own);
    if c.count_ones() < 2 {
        return None;
    }
    let most = bits(c).map(|t| rules.count[t as usize]).max()?;
    let smaller = bits(c).filter(|&t| rules.count[t as usize] < most).fold(0, |m, t| m | 1 << t);
    Some(own & !smaller)
}

fn donation(rules: &Rules, seat: u8, bidder: u8, own: u32, rec: &Rec) -> Option<u32> {
    let t = trumps(rules);
    if t == 0 {
        return None;
    }
    let p = partner(bidder);
    if seat == bidder && rec.n == 0 {
        let out = t & !rec.played;
        if t & !rec.forbid[p as usize] != 0 || out & !own == 0 {
            return None;
        }
        return top(rules, out).filter(|&x| (own >> x) & 1 == 1).map(|x| 1 << x);
    }
    if seat != p || rec.n == 0 || rec.leader != bidder {
        return None;
    }
    let lead = rec.trick[0];
    if (t >> lead) & 1 == 0 || own & t != 0 {
        return None;
    }
    let earlier = rec.played & !trick_mask(rec);
    let rank = &rules.rank[7];
    let top_lead = bits(t & !earlier & !(1 << lead)).all(|x| rank[x as usize] < rank[lead as usize]);
    let c = counts(rules, own);
    if top_lead {
        let most = bits(c).map(|x| rules.count[x as usize]).max()?;
        return Some(bits(c).filter(|&x| rules.count[x as usize] == most).fold(0, |m, x| m | 1 << x));
    }
    let rest = own & !c;
    (rest != 0).then_some(rest)
}

/// Where a gate holds and constrains: the allowed tiles, else None.
fn gated(allowed: Option<u32>, rules: &Rules, own: u32, rec: &Rec) -> Option<u32> {
    let legal = rules.legal(own, rec.led());
    let a = allowed.filter(|&a| a != legal)?;
    debug_assert!(a != 0 && a & !legal == 0, "a rule allows legal tiles only");
    Some(a)
}

impl Convention {
    /// The allowed tiles where the sender's gate holds for the mover.
    pub fn sends(&self, rules: &Rules, seat: u8, bidder: u8, own: u32, rec: &Rec) -> Option<u32> {
        let a = match self.rule {
            Rule::Handoff | Rule::HandoffSingleton => handoff(rules, seat, bidder, own, rec),
            Rule::Biggest => biggest(rules, seat, bidder, own, rec),
            Rule::Donation => donation(rules, seat, bidder, own, rec),
        };
        gated(a, rules, own, rec)
    }

    /// The allowed tiles the reader expects of `seat` holding `own`.
    pub fn reads(&self, rules: &Rules, seat: u8, bidder: u8, own: u32, rec: &Rec) -> Option<u32> {
        let a = match self.rule {
            Rule::Handoff => handoff(rules, seat, bidder, own, rec),
            Rule::HandoffSingleton => singleton(rules, seat, bidder, own, rec),
            Rule::Biggest => biggest(rules, seat, bidder, own, rec),
            Rule::Donation => donation(rules, seat, bidder, own, rec),
        };
        gated(a, rules, own, rec)
    }
}
