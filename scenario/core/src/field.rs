//! The field net — `scenario/field.py`: a small integer policy that plays
//! rung −1, the modeled seats' bottom rung, in place of the dice.
//!
//! Integer arithmetic throughout, so the referee and the core agree by
//! construction: bit features, the first layer a sum of the active rows,
//! each hidden layer max(acc >> shift, 0), the move the highest-scoring
//! legal tile with ties to the lowest tile id. The weights and the
//! per-declaration tile relabeling (`canon`) arrive from Python.

use crate::planner::{Rec, Rules};

const N: usize = 28;
const STEPS: usize = 21; // thermometer thresholds 2, 4, ..., 42 points
const BIDS: usize = 13; // the contract one-hot, 30..=42 points

const HAND: usize = 0;
const GONE: usize = HAND + N;
const TRICK: usize = GONE + N;
const VOID: usize = TRICK + 3 * N;
const COUNT: usize = VOID + 3 * N;
const SIDE: usize = COUNT + 2 * N;
const OPEN: usize = SIDE + 1;
const WINNING: usize = OPEN + 4;
const NEED: usize = WINNING + 3;
const SLACK: usize = NEED + STEPS;
const BID: usize = SLACK + STEPS;
pub const N_FEATURES: usize = BID + BIDS;

pub struct FieldNet {
    pub h1: usize,
    pub h2: usize,
    pub w1: Vec<i64>, // [features][h1]
    pub b1: Vec<i64>,
    pub w2: Vec<i64>, // [h2][h1]
    pub b2: Vec<i64>,
    pub w3: Vec<i64>, // [28][h2]
    pub b3: Vec<i64>,
    pub s1: u32,
    pub s2: u32,
    pub canon: [Option<[u8; N]>; 10], // per declaration: tile id -> canonical tile id
}

#[inline]
fn tiles(mut mask: u32) -> impl Iterator<Item = usize> {
    std::iter::from_fn(move || {
        if mask == 0 {
            None
        } else {
            let t = mask.trailing_zeros() as usize;
            mask &= mask - 1;
            Some(t)
        }
    })
}

impl FieldNet {
    /// The active feature indices of a modeled seat's decision (`field.features`).
    pub fn features(&self, rules: &Rules, seat: u8, own: u32, rec: &Rec, bid: i32, bt: usize) -> Vec<usize> {
        let canon = self.canon[rules.decl as usize].as_ref().expect("the field net plays pip trumps only");
        let mut on = Vec::with_capacity(64);
        on.extend(tiles(own).map(|t| HAND + canon[t] as usize));
        on.extend(tiles(rec.played).map(|t| GONE + canon[t] as usize));
        let n = rec.n as usize;
        for i in 0..n {
            on.push(TRICK + ((i + 4 - n) % 4 - 1) * N + canon[rec.trick[i] as usize] as usize);
        }
        for rel in 1..4usize {
            let f = rec.forbid[(seat as usize + rel) % 4];
            on.extend(tiles(f).map(|t| VOID + (rel - 1) * N + canon[t] as usize));
        }
        for (&c, &k) in canon.iter().zip(&rules.count) {
            match k {
                10 => on.push(COUNT + c as usize),
                5 => on.push(COUNT + N + c as usize),
                _ => {}
            }
        }
        if seat as usize % 2 == bt {
            on.push(SIDE);
        }
        on.push(OPEN + n);
        if n > 0 {
            let ranks = &rules.rank[rules.led_suit[rec.trick[0] as usize] as usize];
            let mut best = 0;
            for i in 1..n {
                if ranks[rec.trick[i] as usize] > ranks[rec.trick[best] as usize] {
                    best = i;
                }
            }
            on.push(WINNING + (best + 4 - n) % 4 - 1);
        }
        let need = bid - rec.banked[bt] as i32;
        let slack = (43 - bid) - rec.banked[1 - bt] as i32;
        for i in 0..STEPS {
            if need >= 2 * (i as i32 + 1) {
                on.push(NEED + i);
            }
        }
        for i in 0..STEPS {
            if slack >= 2 * (i as i32 + 1) {
                on.push(SLACK + i);
            }
        }
        on.push(BID + (bid.clamp(30, 42) - 30) as usize);
        on
    }

    /// [28] scores over canonical tiles.
    pub fn scores(&self, active: &[usize]) -> Vec<i64> {
        let mut a1 = self.b1.clone();
        for &i in active {
            for (a, w) in a1.iter_mut().zip(&self.w1[i * self.h1..(i + 1) * self.h1]) {
                *a += w;
            }
        }
        let h1: Vec<i64> = a1.into_iter().map(|a| (a >> self.s1).max(0)).collect();
        let h2: Vec<i64> = (0..self.h2)
            .map(|j| {
                let row = &self.w2[j * self.h1..(j + 1) * self.h1];
                let acc = self.b2[j] + row.iter().zip(&h1).map(|(w, h)| w * h).sum::<i64>();
                (acc >> self.s2).max(0)
            })
            .collect();
        (0..N)
            .map(|t| self.b3[t] + self.w3[t * self.h2..(t + 1) * self.h2].iter().zip(&h2).map(|(w, h)| w * h).sum::<i64>())
            .collect()
    }

    /// The highest-scoring legal tile; ties to the lowest tile id.
    pub fn decide(&self, rules: &Rules, seat: u8, own: u32, rec: &Rec, bid: i32, bt: usize) -> u8 {
        let z = self.scores(&self.features(rules, seat, own, rec, bid, bt));
        let canon = self.canon[rules.decl as usize].as_ref().expect("the field net plays pip trumps only");
        let mut best: Option<(usize, i64)> = None;
        for t in tiles(rules.legal(own, rec.led())) {
            let v = z[canon[t] as usize];
            if best.is_none_or(|(_, b)| v > b) {
                best = Some((t, v));
            }
        }
        best.expect("a legal play exists").0 as u8
    }
}
