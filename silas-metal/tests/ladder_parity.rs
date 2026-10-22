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
use silas::{Auction, Bid, Domino, DominoSet, Frame, Seat};
use silas_metal::Gpu;
use silas_metal::ladder::{Gathered, Limits, Piece, Problem, Work, viewer_moves_after};

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
        let g: Gathered =
            problem.gather(&mut gpu, k, Piece::ALL, 0..1000, Limits { table_bits: 22, ..Limits::default() }).unwrap();
        assert_eq!(g.worlds, 3_432_000);
        assert_eq!(g.leaves.len(), leaves, "k {k}");
        assert_eq!(g.work, work, "k {k}");
        let bytes = g.to_bytes();
        assert_eq!(Gathered::from_bytes(&bytes).unwrap(), g);
        assert_eq!(fnv(&bytes), sum, "k {k}: the bytes of Exact-T1's probe file");
        // The count walk keeps no leaves and counts the same worlds and plays, through a table
        // small enough to drain many times.
        assert_eq!(problem.count(&mut gpu, k, Piece::ALL, 0..1000, 15).unwrap(), (3_432_000, work));
    }
}

#[test]
fn a_follow_root_equals_exact_t1() {
    // Deal 5 (P2), viewer S1 behind S0's lead of tile 5: a FiberUniform root, shards 0..2000.
    // The pins are Exact-T1's CPU run of the same piece (2026-10-22 19:20), equal byte for byte.
    let hand = DominoSet::from_bits(221_806_592).unwrap();
    let auction = Auction::won_by(Seat::S0, Bid::Points(30)).unwrap();
    let lead = vec![Domino::from_index(5).unwrap()];
    let frame = Frame::new(auction, Straight42::ALL[2].physics(), Seat::S1, hand, lead).unwrap();
    let problem = Problem::new(&frame, [SeatLaw::Lowest; 4], Root::FiberUniform).unwrap();
    assert_eq!(problem.shards(), 77_520);
    let mut gpu = Gpu::new().expect("a Metal device");
    let pins: [(usize, usize, Work, u64); 2] = [
        (0, 1_365, Work { tree: 96_096_000, tails: 825_392_632, god: 3_962_954_061 }, 0x7abe_6cf0_b014_a781),
        (1, 507_456, Work { tree: 638_570_902, tails: 2_655_761_044, god: 6_445_408_721 }, 0x3dbd_a10d_001d_0c79),
    ];
    for (k, leaves, work, sum) in pins {
        let g =
            problem.gather(&mut gpu, k, Piece::ALL, 0..2000, Limits { table_bits: 22, ..Limits::default() }).unwrap();
        assert_eq!(g.worlds, 3_432_000);
        assert_eq!((g.leaves.len(), g.work), (leaves, work), "k {k}");
        assert_eq!(fnv(&g.to_bytes()), sum, "k {k}: the bytes of Exact-T1's CPU piece");
    }
}

#[test]
fn per_root_gathers_partition_the_whole() {
    // Each root tile's gather holds exactly the whole gather's leaves under that tile, with
    // the whole's world count; their leaves and plays add up to the whole's.
    let (_, problem) = deal0();
    let mut gpu = Gpu::new().expect("a Metal device");
    let limits = Limits { table_bits: 20, ..Limits::default() };
    let whole = problem.gather(&mut gpu, 2, Piece::ALL, 0..20, limits).unwrap();
    let mut joined = Vec::new();
    let mut work = Work::default();
    for tile in 0..28 {
        if 67_732_752u32 >> tile & 1 == 0 {
            continue;
        }
        let g = problem.gather(&mut gpu, 2, Piece::roots(1 << tile), 0..20, limits).unwrap();
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
    let tile = problem.gather(&mut gpu, 2, Piece::roots(root), 0..20, limits).unwrap();
    let mut joined = Vec::new();
    let (mut work, mut worlds, mut pieces) = (Work::default(), 0, 0);
    for bit in (0..28).chain([31]) {
        let g = problem.gather(&mut gpu, 2, Piece { replies: 1 << bit, ..Piece::roots(root) }, 0..20, limits).unwrap();
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

/// On one follow root of `pour_make_s1001_n512_d5000` (trump `decl`, the viewer's hand, the
/// trick so far), every shard, horizon `k`: the first reply whose next node is the viewer's
/// (`max`) or another seat's, split by the play after it into two pieces or more (one at k =
/// 0). The pieces
/// must hold the reply's leaves and plays exactly and fold to the reply's rung: summed, or each
/// side's maximum with every piece holding the node's worlds. False when no such reply exists.
fn split_seconds(decl: usize, viewer: Seat, hand: u32, trick: &[usize], k: usize, max: bool) -> bool {
    let auction = Auction::won_by(Seat::S0, Bid::Points(30)).unwrap();
    let prefix: Vec<Domino> = trick.iter().map(|t| Domino::from_index(*t).unwrap()).collect();
    let frame =
        Frame::new(auction, Straight42::ALL[decl].physics(), viewer, DominoSet::from_bits(hand).unwrap(), prefix)
            .unwrap();
    let problem = Problem::new(&frame, [SeatLaw::Lowest; 4], Root::FiberUniform).unwrap();
    let mut gpu = Gpu::new().expect("a Metal device");
    let limits = Limits { table_bits: 22, ..Limits::default() };
    let all = 0..problem.shards();
    let bits = || (0..28u32).chain([31]);
    let tile = |b: u32| Domino::from_index(b as usize).unwrap();
    for root in (0..28).filter(|t| hand >> t & 1 == 1) {
        for r in bits() {
            let plays = if r == 31 { vec![tile(root)] } else { vec![tile(root), tile(r)] };
            if viewer_moves_after(&frame, &plays) != max {
                continue;
            }
            let at = Piece { replies: 1 << r, ..Piece::roots(1 << root) };
            let whole = problem.gather(&mut gpu, k, at, all.clone(), limits).unwrap();
            if whole.leaves.is_empty() {
                continue;
            }
            let pieces: Vec<Gathered> = bits()
                .map(|s| problem.gather(&mut gpu, k, Piece { seconds: 1 << s, ..at }, all.clone(), limits).unwrap())
                .filter(|g| !g.leaves.is_empty())
                .collect();
            // (At k = 0 a viewer's node is a leaf: its one piece, bit 31, is the whole reply.)
            if pieces.len() < if k == 0 { 1 } else { 2 } {
                continue;
            }
            let (mut joined, mut work, mut fold) = (Vec::new(), Work::default(), (0u64, 0u64, 0u64));
            for g in pieces {
                let (_, lo, up) = problem.rung(&frame, &g).unwrap().actions[0];
                fold = if max {
                    assert_eq!(g.worlds, whole.worlds, "every piece holds the node's worlds");
                    (fold.0.max(lo), fold.1.max(up), g.worlds)
                } else {
                    (fold.0 + lo, fold.1 + up, fold.2 + g.worlds)
                };
                work = Work {
                    tree: work.tree + g.work.tree,
                    tails: work.tails + g.work.tails,
                    god: work.god + g.work.god,
                };
                joined.extend(g.leaves);
            }
            joined.sort_by_key(|x: &(u128, _)| x.0);
            assert_eq!(joined, whole.leaves, "max {max}");
            assert_eq!(work, whole.work, "max {max}");
            let r = problem.rung(&frame, &whole).unwrap();
            assert_eq!(fold, (r.actions[0].1, r.actions[0].2, r.mass), "max {max}");
            return true;
        }
    }
    false
}

#[test]
fn second_pieces_fold_to_their_reply() {
    // Another seat's node: deal 5, viewer S3 last in trick 1.
    assert!(split_seconds(2, Seat::S3, 280_771, &[5, 21, 4], 1, false));
    // The viewer's node: the first of deals 12, 15 and 39 where viewer S3, last in trick 1,
    // moves again after the reply (winning trick 1, or following S2's lead of trick 2); at
    // k = 1, and at k = 0, where the horizon ends at that node and its one piece is the reply.
    let last = [(3, 96_502_048, [6, 7, 18]), (0, 159_520_800, [0, 6, 15]), (1, 6_556_320, [1, 17, 16])];
    for k in [1, 0] {
        assert!(last.iter().any(|(decl, hand, trick)| split_seconds(*decl, Seat::S3, *hand, trick, k, true)));
    }
}

#[test]
fn a_tiny_table_drains_and_still_agrees() {
    // Shards 0..40 through a leaf table of 2^15 slots (drained many times) equal the same
    // shards through 2^20 slots.
    let (_, problem) = deal0();
    let mut gpu = Gpu::new().expect("a Metal device");
    let big = problem.gather(&mut gpu, 1, Piece::ALL, 0..40, Limits { table_bits: 20, ..Limits::default() }).unwrap();
    let small = problem.gather(&mut gpu, 1, Piece::ALL, 0..40, Limits { table_bits: 15, ..Limits::default() }).unwrap();
    assert_eq!(big, small);
}
