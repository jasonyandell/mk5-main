// SPDX-FileCopyrightText: Jason Yandell
// SPDX-License-Identifier: CC0-1.0

//! STRATEGY-FUSION FENCE. The Demon: a fenced tournament opponent; measuring stick only.
//! The arena heuristic's declaration, evaluated on an original seven-tile hand.
//! This models a public choice; it does not see the bidder's actual hand.

use scenario_core::rules::{pips, tile};
use std::sync::LazyLock;

struct Tables {
    suits: [u32; 7],
    doubles: u32,
    exposed: [[u32; 28]; 7],
    counts: u32,
    tens: u32,
}

static TABLES: LazyLock<Tables> = LazyLock::new(|| {
    let suits: [u32; 7] = std::array::from_fn(|p| {
        (0..28)
            .filter(|&t| {
                let (a, b) = pips(t);
                a == p as u8 || b == p as u8
            })
            .fold(0, |m, t| m | 1 << t)
    });
    let doubles = (0..7).fold(0, |m, p| m | 1 << tile(p, p));
    let tens = (1 << tile(5, 5)) | (1 << tile(6, 4));
    let counts = tens | (1 << tile(5, 0)) | (1 << tile(4, 1)) | (1 << tile(3, 2));
    let exposed = std::array::from_fn(|trump| {
        std::array::from_fn(|t| {
            let (a, b) = pips(t as u8);
            (suits[a as usize] | suits[b as usize]) & counts & !suits[trump]
        })
    });
    Tables { suits, doubles, exposed, counts, tens }
});

/// The arena's `HeuristicBidder(min_trumps=3).declare`, including the fallback
/// for a forced bid. Held count is constant across trumps and cancels in ties.
pub fn heuristic_declaration(hand: u32) -> u8 {
    let t = &*TABLES;
    let minimum = if t.suits.iter().any(|s| (hand & s).count_ones() >= 3) { 3 } else { 0 };
    (0..7u8)
        .filter(|&p| (hand & t.suits[p as usize]).count_ones() >= minimum)
        .max_by_key(|&p| {
            let mut offs = hand & !t.suits[p as usize] & !t.doubles;
            let mut exposed = 0;
            while offs != 0 {
                let off = offs.trailing_zeros() as usize;
                offs &= offs - 1;
                exposed |= t.exposed[p as usize][off];
            }
            exposed &= !hand;
            let points = 5 * ((exposed & t.counts).count_ones() + (exposed & t.tens).count_ones());
            (42 - points, (hand & t.suits[p as usize]).count_ones(), hand >> tile(p, p) & 1, p)
        })
        .expect("at least one candidate")
}

/// A soft categorical declaration model. All original hands keep positive
/// likelihood; a match receives `ratio`, a mismatch 1. Dividing by ratio+6
/// gives a normalized distribution over the seven pip declarations.
pub fn declaration_likelihood(hand: u32, declaration: u8, ratio: u32) -> u32 {
    if declaration < 7 && heuristic_declaration(hand) == declaration { ratio } else { 1 }
}
