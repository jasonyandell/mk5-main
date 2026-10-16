// SPDX-FileCopyrightText: Jason Yandell
// SPDX-License-Identifier: MIT

//! The field net — `scenario/field.py`: a small integer policy that plays
//! rung −1, the modeled seats' bottom rung, in place of the dice.
//!
//! Integer arithmetic throughout, so the referee and the core agree by
//! construction: bit features, the first layer a sum of the active rows,
//! each hidden layer max(acc >> shift, 0), the move the highest-scoring
//! legal tile with ties to the lowest tile id.
//!
//! The weights arrive as one blob, `FieldNet.to_bytes()` in Python, all
//! integers little-endian:
//!
//! | bytes | field |
//! |---|---|
//! | 4 | magic `FNET` |
//! | 1 + len | the feature layout's name, length-prefixed ASCII: `field-v2` or `field-v3` ([`Layout`]) |
//! | 4 × u32 | h1, h2 (hidden widths), s1, s2 (hidden-layer shifts) |
//! | i16 × features·h1 | w1, a row of h1 per feature |
//! | i64 × h1 | b1 |
//! | i16 × h2·h1 | w2, a row of h1 per hidden unit |
//! | i64 × h2 | b2 |
//! | i16 × 28·h2 | w3, a row of h2 per canonical tile |
//! | i64 × 28 | b3 |
//!
//! Weights are int16 because the trainer quantizes them to the largest
//! power-of-two scale that fits int16 (`field_train.quantize`). Biases are
//! int64 because they live in accumulator units — a bias carries the scale
//! of a weight times an activation, plus the folded rounding — which no
//! narrower width bounds; int64 is the referee's own arithmetic, so the
//! format is exact by construction.

use std::fmt;

use crate::planner::Rec;
use crate::rules::{pips, tile, Rules};

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
const KIND: usize = BID + BIDS; // field-v3 only: the declaration's kind, one-hot
const KINDS: usize = 3; // a pip trump, doubles trump, no trump

/// The blob's first four bytes.
pub const MAGIC: &[u8; 4] = b"FNET";

/// A feature layout: which declarations a net plays and how it sees them.
/// `field-v2` is the shipped net's: the seven pip trumps, the trump pip
/// relabelled to 6 so one trump suit is learned. `field-v3` appends one
/// block, the declaration's kind, and plays doubles trump (7) and no trump
/// (9) as well, with the tiles unrelabelled; every v2 feature keeps its
/// index, so a v2 net is a v3 net with three zero rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Layout {
    V2,
    V3,
}

impl Layout {
    /// The name a blob carries.
    pub const fn name(self) -> &'static str {
        match self {
            Layout::V2 => "field-v2",
            Layout::V3 => "field-v3",
        }
    }

    /// The layout a blob names, if this module computes it.
    pub fn parse(name: &[u8]) -> Option<Layout> {
        [Layout::V2, Layout::V3].into_iter().find(|l| l.name().as_bytes() == name)
    }

    /// The width of the feature row.
    pub const fn n_features(self) -> usize {
        match self {
            Layout::V2 => KIND,
            Layout::V3 => KIND + KINDS,
        }
    }

    /// Tile id → canonical tile id under `decl`, or None where this layout
    /// does not play: under a pip trump the trump pip becomes 6 and the
    /// others keep their order (`field.py`'s `_canon`); under doubles trump
    /// and no trump (v3 only) the tiles stay as they are, and the kind bit
    /// says which game it is. Doubles-suit (8) is no game's declaration.
    pub fn canon(self, decl: u8) -> Option<[u8; N]> {
        match (self, decl) {
            (_, 0..=6) => {
                let pip = |q: u8| match q.cmp(&decl) {
                    std::cmp::Ordering::Less => q,
                    std::cmp::Ordering::Equal => 6,
                    std::cmp::Ordering::Greater => q - 1,
                };
                Some(std::array::from_fn(|t| {
                    let (high, low) = pips(t as u8);
                    let (a, b) = (pip(high), pip(low));
                    tile(a.max(b), a.min(b))
                }))
            }
            (Layout::V3, 7 | 9) => Some(std::array::from_fn(|t| t as u8)),
            _ => None,
        }
    }

    /// The kind bit of a declaration this layout plays: 0 a pip trump, 1 doubles trump, 2 no trump.
    fn kind(decl: u8) -> usize {
        match decl {
            0..=6 => 0,
            7 => 1,
            _ => 2,
        }
    }
}

/// Why a blob is not a field net.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlobError(&'static str);

impl fmt::Display for BlobError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.0)
    }
}

impl std::error::Error for BlobError {}

/// A cursor over the blob that refuses to read past its end.
struct Blob<'a>(&'a [u8]);

impl<'a> Blob<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], BlobError> {
        if self.0.len() < n {
            return Err(BlobError("field net blob is truncated"));
        }
        let (head, rest) = self.0.split_at(n);
        self.0 = rest;
        Ok(head)
    }

    fn u32(&mut self) -> Result<u32, BlobError> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().expect("four bytes")))
    }

    fn i16s(&mut self, n: usize) -> Result<Vec<i64>, BlobError> {
        let bytes = self.take(n.checked_mul(2).ok_or(BlobError("field net shape overflows"))?)?;
        Ok(bytes.chunks_exact(2).map(|b| i16::from_le_bytes([b[0], b[1]]) as i64).collect())
    }

    fn i64s(&mut self, n: usize) -> Result<Vec<i64>, BlobError> {
        let bytes = self.take(n.checked_mul(8).ok_or(BlobError("field net shape overflows"))?)?;
        Ok(bytes.chunks_exact(8).map(|b| i64::from_le_bytes(b.try_into().expect("eight bytes"))).collect())
    }
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

/// An integer MLP, features → 28 canonical tile scores; built only by
/// [`FieldNet::from_bytes`], so its shapes always fit.
#[derive(Clone)]
pub struct FieldNet {
    layout: Layout,
    h1: usize,
    h2: usize,
    w1: Vec<i64>, // [features][h1]
    b1: Vec<i64>,
    w2: Vec<i64>, // [h2][h1]
    b2: Vec<i64>,
    w3: Vec<i64>, // [28][h2]
    b3: Vec<i64>,
    s1: u32,
    s2: u32,
}

impl FieldNet {
    /// Parse a blob in the format above, checking the magic, the layout,
    /// the shapes and that nothing trails.
    pub fn from_bytes(bytes: &[u8]) -> Result<FieldNet, BlobError> {
        let mut b = Blob(bytes);
        if b.take(MAGIC.len())? != MAGIC {
            return Err(BlobError("not a field net blob (bad magic)"));
        }
        let len = b.take(1)?[0] as usize;
        let layout = Layout::parse(b.take(len)?).ok_or(BlobError("field net blob names a layout this core does not compute"))?;
        let (h1, h2, s1, s2) = (b.u32()? as usize, b.u32()? as usize, b.u32()?, b.u32()?);
        if h1 == 0 || h2 == 0 {
            return Err(BlobError("field net hidden widths must be positive"));
        }
        if s1 >= 63 || s2 >= 63 {
            return Err(BlobError("field net shifts must be below 63"));
        }
        let overflow = BlobError("field net shape overflows");
        let w1 = b.i16s(layout.n_features().checked_mul(h1).ok_or(overflow.clone())?)?;
        let b1 = b.i64s(h1)?;
        let w2 = b.i16s(h2.checked_mul(h1).ok_or(overflow.clone())?)?;
        let b2 = b.i64s(h2)?;
        let w3 = b.i16s(N.checked_mul(h2).ok_or(overflow)?)?;
        let b3 = b.i64s(N)?;
        if !b.0.is_empty() {
            return Err(BlobError("field net blob has trailing bytes"));
        }
        Ok(FieldNet { layout, h1, h2, w1, b1, w2, b2, w3, b3, s1, s2 })
    }

    /// The feature layout the net was trained on.
    pub fn layout(&self) -> Layout {
        self.layout
    }

    /// Whether the net plays declaration `decl`: the pip trumps (0..=6) under
    /// every layout, doubles trump and no trump under `field-v3`.
    pub fn plays(&self, decl: u8) -> bool {
        self.layout.canon(decl).is_some()
    }

    /// The active feature indices of a modeled seat's decision (`field.features`).
    pub fn features(&self, rules: &Rules, seat: u8, own: u32, rec: &Rec, bid: i32, bt: usize) -> Vec<usize> {
        let canon = self.layout.canon(rules.decl).expect("the field net does not play this declaration");
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
        if self.layout == Layout::V3 {
            on.push(KIND + Layout::kind(rules.decl));
        }
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
            .map(|t| {
                self.b3[t] + self.w3[t * self.h2..(t + 1) * self.h2].iter().zip(&h2).map(|(w, h)| w * h).sum::<i64>()
            })
            .collect()
    }

    /// The highest-scoring legal tile; ties to the lowest tile id.
    pub fn decide(&self, rules: &Rules, seat: u8, own: u32, rec: &Rec, bid: i32, bt: usize) -> u8 {
        let z = self.scores(&self.features(rules, seat, own, rec, bid, bt));
        let canon = self.layout.canon(rules.decl).expect("the field net does not play this declaration");
        let mut best: Option<(usize, i64)> = None;
        for t in tiles(rules.legal(own, rec.led())) {
            let v = z[canon[t] as usize];
            if best.is_none_or(|(_, b)| v > b) {
                best = Some((t, v));
            }
        }
        best.expect("a legal play exists").0 as u8
    }

    /// The legal tiles scoring within `margin` of the best legal score, as a
    /// mask: the set a reading holds the seat to (`FieldNet.allowed`). Margin 0
    /// is the best score and its ties; a margin wider than every gap is every
    /// legal tile.
    #[allow(clippy::too_many_arguments)]
    pub fn allowed(&self, rules: &Rules, seat: u8, own: u32, rec: &Rec, bid: i32, bt: usize, margin: i64) -> u32 {
        let z = self.scores(&self.features(rules, seat, own, rec, bid, bt));
        let canon = self.layout.canon(rules.decl).expect("the field net does not play this declaration");
        let legal = rules.legal(own, rec.led());
        let best = tiles(legal).map(|t| z[canon[t] as usize]).max().expect("a legal play exists");
        tiles(legal).filter(|&t| z[canon[t] as usize] >= best.saturating_sub(margin)).fold(0, |a, t| a | 1 << t)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rules::N_DECLS;

    /// A blob of the given layout and widths: weights 1, biases 0, shifts 0.
    fn blob(layout: Layout, h1: u32, h2: u32) -> Vec<u8> {
        let mut out = MAGIC.to_vec();
        out.push(layout.name().len() as u8);
        out.extend(layout.name().as_bytes());
        for v in [h1, h2, 0, 0] {
            out.extend(v.to_le_bytes());
        }
        let (h1, h2) = (h1 as usize, h2 as usize);
        for (weights, biases) in [(layout.n_features() * h1, h1), (h2 * h1, h2), (N * h2, N)] {
            out.extend(std::iter::repeat_n(1i16.to_le_bytes(), weights).flatten());
            out.extend(std::iter::repeat_n(0i64.to_le_bytes(), biases).flatten());
        }
        out
    }

    #[test]
    fn a_blob_parses_and_scores() {
        let net = FieldNet::from_bytes(&blob(Layout::V2, 3, 2)).unwrap();
        // Two active features: each first-layer unit is 2, each second 3·2, each score 2·6.
        assert_eq!(net.scores(&[0, 1]), vec![12; N]);
        assert_eq!(net.layout(), Layout::V2);
        assert!((0..=6).all(|d| net.plays(d)) && !(7..=10).any(|d| net.plays(d)));
        let v3 = FieldNet::from_bytes(&blob(Layout::V3, 3, 2)).unwrap();
        assert_eq!(v3.layout(), Layout::V3);
        assert!((0..=6).all(|d| v3.plays(d)) && v3.plays(7) && v3.plays(9) && !v3.plays(8) && !v3.plays(10));
        // A v2 blob is three rows short of a v3 net, and the other way round.
        let mut short = blob(Layout::V3, 3, 2);
        let name_at = MAGIC.len() + 1;
        short[name_at + 7] = b'2';
        assert_eq!(FieldNet::from_bytes(&short).err().map(|e| e.to_string()).as_deref(), Some("field net blob has trailing bytes"));
    }

    #[test]
    fn a_bad_blob_is_refused() {
        let good = blob(Layout::V2, 3, 2);
        let refused = |b: &[u8]| FieldNet::from_bytes(b).err().map(|e| e.to_string());
        let mut magic = good.clone();
        magic[0] = b'X';
        assert_eq!(refused(&magic).as_deref(), Some("not a field net blob (bad magic)"));
        let mut layout = good.clone();
        layout[MAGIC.len() + Layout::V2.name().len()] = b'1';
        assert_eq!(refused(&layout).as_deref(), Some("field net blob names a layout this core does not compute"));
        assert_eq!(refused(&good[..good.len() - 1]).as_deref(), Some("field net blob is truncated"));
        let mut trailing = good.clone();
        trailing.push(0);
        assert_eq!(refused(&trailing).as_deref(), Some("field net blob has trailing bytes"));
        assert_eq!(refused(&blob(Layout::V2, 0, 2)).as_deref(), Some("field net hidden widths must be positive"));
        assert_eq!(refused(&[]).as_deref(), Some("field net blob is truncated"));
    }

    #[test]
    fn canon_sends_trump_to_six_and_keeps_the_set() {
        for layout in [Layout::V2, Layout::V3] {
            assert_eq!(layout.canon(6), Some(std::array::from_fn(|t| t as u8)));
            for d in 0..=6 {
                let c = layout.canon(d).unwrap();
                let mut sorted = c;
                sorted.sort_unstable();
                assert_eq!(sorted, std::array::from_fn(|t| t as u8));
                let mut trumps = (0..28u8).filter(|&t| pips(t).0 == d || pips(t).1 == d);
                assert!(trumps.all(|t| pips(c[t as usize]).0 == 6), "decl {d}");
            }
        }
        assert!((7..N_DECLS).all(|d| Layout::V2.canon(d).is_none()));
        let identity: [u8; N] = std::array::from_fn(|t| t as u8);
        assert_eq!(Layout::V3.canon(7), Some(identity));
        assert_eq!(Layout::V3.canon(9), Some(identity));
        assert_eq!(Layout::V3.canon(8), None);
    }

    #[test]
    fn v3_rows_carry_the_kind_bit_and_v2_rows_do_not() {
        let v2 = FieldNet::from_bytes(&blob(Layout::V2, 3, 2)).unwrap();
        let v3 = FieldNet::from_bytes(&blob(Layout::V3, 3, 2)).unwrap();
        let rec = Rec::opening(0);
        for (decl, kind) in [(4u8, 0usize), (7, 1), (9, 2)] {
            let rules = Rules::new(decl).unwrap();
            let v3_on = v3.features(&rules, 1, 0b1111111, &rec, 30, 0);
            assert_eq!(v3_on.iter().filter(|&&i| i >= KIND).copied().collect::<Vec<_>>(), vec![KIND + kind], "decl {decl}");
            if decl <= 6 {
                let v2_on = v2.features(&rules, 1, 0b1111111, &rec, 30, 0);
                assert_eq!(v2_on, v3_on[..v3_on.len() - 1].to_vec(), "a v2 row is the v3 row without its kind bit");
            }
        }
    }
}
