// SPDX-FileCopyrightText: Jason Yandell
// SPDX-License-Identifier: CC0-1.0

//! The Metal gather against Exact-T1's CPU gather (`silas::player::trick1`), on the trick-1
//! lead of deal 0 of `pour_make_s1001_n512_d5000` (P4, S0 bids 30, every other seat lowest),
//! shards 0..1000. The pins are Exact-T1's own `probe-k0.bin` and `probe-k1.bin` (its CPU
//! run, 2026-10-22 08:26): the Metal bytes equalled those files byte for byte, and the pins
//! are their counts and a checksum of their bytes.
//!
//! STRATEGY-FUSION FENCE: the gather's upper column is clairvoyant, a bound only.

use silas::player::{Root, SeatLaw};
use silas::reference::Straight42;
use silas::{Auction, Bid, DominoSet, Frame, Seat};
use silas_metal::Gpu;
use silas_metal::ladder::{Gathered, Limits, Problem, Work};

fn fnv(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |h, b| (h ^ u64::from(*b)).wrapping_mul(0x0100_0000_01b3))
}

fn deal0() -> (Frame, Problem) {
    let physics = Straight42::ALL[4].physics();
    let hand = DominoSet::from_bits(67_732_752).unwrap();
    let auction = Auction::won_by(Seat::S0, Bid::Points(30)).unwrap();
    let frame = Frame::new(auction, physics, Seat::S0, hand, vec![]).unwrap();
    let problem = Problem::new(&frame, [SeatLaw::Lowest; 4], Root::Posterior).unwrap();
    (frame, problem)
}

#[test]
fn deal0_probes_equal_exact_t1() {
    let (_, problem) = deal0();
    assert_eq!(problem.shards(), 116_280);
    let mut gpu = Gpu::new().expect("a Metal device");
    let pins: [(usize, usize, Work, u64); 2] = [
        (0, 12_060, Work { tree: 128_186_278, tails: 641_230_312, god: 1_984_392_674 }, 0xd884_3588_3c66_cf84),
        (1, 509_459, Work { tree: 393_476_864, tails: 1_216_747_116, god: 2_317_763_457 }, 0x284b_3c22_c2f2_6d6f),
    ];
    for (k, leaves, work, sum) in pins {
        let g: Gathered = problem
            .gather(&mut gpu, k, u32::MAX, u32::MAX, 0..1000, Limits { table_bits: 22, ..Limits::default() })
            .unwrap();
        assert_eq!(g.worlds, 3_432_000);
        assert_eq!(g.leaves.len(), leaves, "k {k}");
        assert_eq!(g.work, work, "k {k}");
        let bytes = g.to_bytes();
        assert_eq!(Gathered::from_bytes(&bytes).unwrap(), g);
        assert_eq!(fnv(&bytes), sum, "k {k}: the bytes of Exact-T1's probe file");
    }
}

#[test]
fn per_root_gathers_partition_the_whole() {
    // Each root tile's gather holds exactly the whole gather's leaves under that tile, with
    // the whole's world count; their leaves and plays add up to the whole's.
    let (_, problem) = deal0();
    let mut gpu = Gpu::new().expect("a Metal device");
    let limits = Limits { table_bits: 20, ..Limits::default() };
    let whole = problem.gather(&mut gpu, 2, u32::MAX, u32::MAX, 0..20, limits).unwrap();
    let mut joined = Vec::new();
    let mut work = Work::default();
    for tile in 0..28 {
        if 67_732_752u32 >> tile & 1 == 0 {
            continue;
        }
        let g = problem.gather(&mut gpu, 2, 1 << tile, u32::MAX, 0..20, limits).unwrap();
        assert_eq!(g.worlds, whole.worlds);
        assert!(g.leaves.iter().all(|(bits, _)| (bits >> 120) as u32 == tile + 1));
        joined.extend(g.leaves);
        work = Work { tree: work.tree + g.work.tree, tails: work.tails + g.work.tails, god: work.god + g.work.god };
    }
    joined.sort_by_key(|x: &(u128, _)| x.0);
    assert_eq!(joined, whole.leaves);
    assert_eq!(work, whole.work);
}

#[test]
fn reply_pieces_partition_a_root_tile() {
    // Below one root tile (the hand's highest), the pieces by the next seat's reply hold the
    // tile's leaves, plays and worlds exactly, each world in one piece.
    let (_, problem) = deal0();
    let mut gpu = Gpu::new().expect("a Metal device");
    let limits = Limits { table_bits: 20, ..Limits::default() };
    let root = 1 << (31 - 67_732_752u32.leading_zeros());
    let tile = problem.gather(&mut gpu, 2, root, u32::MAX, 0..20, limits).unwrap();
    let mut joined = Vec::new();
    let (mut work, mut worlds, mut pieces) = (Work::default(), 0, 0);
    for bit in (0..28).chain([31]) {
        let g = problem.gather(&mut gpu, 2, root, 1 << bit, 0..20, limits).unwrap();
        pieces += usize::from(g.worlds > 0);
        assert_eq!(g.worlds == 0, g.leaves.is_empty());
        joined.extend(g.leaves);
        worlds += g.worlds;
        work = Work { tree: work.tree + g.work.tree, tails: work.tails + g.work.tails, god: work.god + g.work.god };
    }
    assert!(pieces > 1, "the tile splits");
    joined.sort_by_key(|x: &(u128, _)| x.0);
    assert_eq!(joined, tile.leaves);
    assert_eq!((worlds, work), (tile.worlds, tile.work));
}

#[test]
fn a_tiny_table_drains_and_still_agrees() {
    // Shards 0..40 through a leaf table of 2^15 slots (drained many times) equal the same
    // shards through 2^20 slots.
    let (_, problem) = deal0();
    let mut gpu = Gpu::new().expect("a Metal device");
    let big =
        problem.gather(&mut gpu, 1, u32::MAX, u32::MAX, 0..40, Limits { table_bits: 20, ..Limits::default() }).unwrap();
    let small =
        problem.gather(&mut gpu, 1, u32::MAX, u32::MAX, 0..40, Limits { table_bits: 15, ..Limits::default() }).unwrap();
    assert_eq!(big, small);
}
