// SPDX-FileCopyrightText: Jason Yandell
// SPDX-License-Identifier: CC0-1.0

//! The focal-horizon ladder (`silas::player::trick1`) against the player's second
//! construction (`player::recount`, written apart): at every horizon its intervals contain
//! the exact root values and nest, and once the horizon covers the viewer's every decision
//! left they collapse onto them. Positions are seeded deals played to a few tricks left,
//! the other seats by their declared law (so the exact posterior has mass) and the viewer
//! at random.
//!
//! STRATEGY-FUSION FENCE: the ladder's upper side is clairvoyant, a bound only; these tests
//! check it is one (it never falls below the exact value).

mod common;

use common::{Rng, physics};
use num_rational::BigRational;
use num_traits::Zero;
use silas::player::trick1::{Gathered, Ladder, Rules, Rung};
use silas::player::{Library, Root, SeatLaw, TableType, recount};
use silas::{Auction, Bid, Domino, DominoSet, Frame, Seat};

type Q = BigRational;

fn q(n: u64, d: u64) -> Q {
    Q::new(n.into(), d.into())
}

/// The flat tables agree with Physics on legality, the winner and the points.
#[test]
fn the_flat_rules_are_physics() {
    let mut rng = Rng(7);
    for decl in ["NT", "P0", "P3", "P6", "DT"] {
        let p = physics(decl);
        let r = Rules::new(&p);
        for _ in 0..2_000 {
            let deck = rng.shuffled();
            let hand: DominoSet = deck[4..11].iter().copied().collect();
            let lead = (rng.below(2) == 0).then_some(deck[0]);
            let legal = p.legal_plays(hand, lead);
            assert_eq!(r.legal(hand.bits(), lead.map(|d| d.index() as u8)), legal.bits());
            let plays = [deck[0], deck[1], deck[2], deck[3]];
            let leader = Seat::ALL[rng.below(4)];
            let flat = plays.map(|d| d.index() as u8);
            assert_eq!(leader.plus(r.winner(flat)), p.trick_winner(leader, plays));
            assert_eq!(u32::from(r.points(flat)), p.trick_points(plays));
        }
    }
}

/// A seeded position: the other seats play `laws`, the viewer at random, stopped at the
/// viewer's turn once `n` plays are made.
fn position(rng: &mut Rng, decl: &str, laws: [SeatLaw; 4], viewer: Seat, n: usize) -> Option<Frame> {
    let deck = rng.shuffled();
    let hands: [DominoSet; 4] = std::array::from_fn(|s| deck[7 * s..7 * s + 7].iter().copied().collect());
    let p = physics(decl);
    let declarer = Seat::ALL[rng.below(4)];
    let auction = Auction::won_by(declarer, Bid::Points(30 + rng.below(5) as u32)).unwrap();
    let mut frame = Frame::new(auction.clone(), p.clone(), declarer, hands[declarer.index()], vec![]).unwrap();
    while frame.plays().len() < n || frame.next_actor() != Some(viewer) {
        let s = frame.next_actor()?;
        let legal = p.legal_plays(hands[s.index()].difference(frame.played()), frame.lead());
        let d = if s == viewer {
            let l: Vec<Domino> = legal.iter().collect();
            l[rng.below(l.len())]
        } else {
            match laws[s.index()] {
                SeatLaw::Lowest => legal.iter().next().unwrap(),
                _ => legal.iter().last().unwrap(),
            }
        };
        frame = frame.after(d).unwrap();
    }
    Frame::new(auction, p, viewer, hands[viewer.index()], frame.plays().to_vec()).ok()
}

#[test]
fn the_ladder_contains_nests_and_collapses_onto_the_second_construction() {
    let decls = ["NT", "P5", "P0", "DT", "P3", "P6"];
    let mut rng = Rng(2026);
    let (mut checked, mut open) = (0, 0);
    for i in 0..24 {
        let laws = if i % 3 == 2 {
            [SeatLaw::Highest, SeatLaw::Lowest, SeatLaw::Highest, SeatLaw::Lowest]
        } else {
            [SeatLaw::Lowest; 4]
        };
        let viewer = Seat::ALL[i % 4];
        let n = [12, 13, 14, 16, 18][i % 5];
        let Some(frame) = position(&mut rng, decls[i % decls.len()], laws, viewer, n) else { continue };
        let mut t = TableType::all(SeatLaw::Lowest);
        t.seats = laws;
        let library = Library::one(t);
        let root = if i % 2 == 0 { Root::Posterior } else { Root::FiberUniform };
        let (_, exact) = recount::values(&frame, &library, root, &Q::zero()).unwrap();
        let ladder = Ladder::new(&frame, laws, root).unwrap();
        let left = frame.kernel().viewer_hand().len();
        let mut prev: Option<Rung> = None;
        for k in 0..left {
            let rung = if i % 4 == 3 {
                // Resumed: the shards gathered in three pieces, through bytes, then merged.
                let (n, menu) = (ladder.shards(), ladder.menu());
                let piece = |a, b| Gathered::from_bytes(&ladder.gather(k, menu, a, b, 2).to_bytes()).unwrap();
                let g = piece(n / 3, n).merge(piece(0, n / 3 / 2)).unwrap().merge(piece(n / 3 / 2, n / 3)).unwrap();
                assert!(g.clone().merge(piece(0, 1)).is_err(), "a shard gathered twice is refused");
                let whole = ladder.gather(k, menu, 0, n, 1);
                assert!(g.agrees(&whole), "pieces merged are the whole gather");
                assert_eq!(g.to_bytes(), whole.to_bytes(), "and write its bytes");
                assert!(ladder.rung([piece(1, n)], u64::MAX).is_err(), "a rung needs every shard");
                // Split by root tile: one gather per tile, joined by the rung.
                let tiles: Vec<u32> = (0..32).filter(|t| menu >> t & 1 == 1).map(|t| 1 << t).collect();
                let parts: Vec<Gathered> = tiles.iter().map(|t| ladder.gather(k, *t, 0, n, 2)).collect();
                assert!(parts.iter().zip(&tiles).all(|(p, t)| p.roots() == *t));
                let solved = ladder.solve(k, 3, u64::MAX).unwrap();
                let by_tile = ladder.rung(parts.clone(), u64::MAX).unwrap();
                assert_eq!((&by_tile.actions, by_tile.mass), (&solved.actions, solved.mass), "split by tile");
                if parts.len() > 1 {
                    assert!(ladder.rung(parts[1..].to_vec(), u64::MAX).is_err(), "a rung needs every root tile");
                    let twice = [parts.clone(), vec![parts[0].clone()]].concat();
                    assert!(ladder.rung(twice, u64::MAX).is_err(), "and each once");
                    assert!(parts[0].clone().merge(parts[1].clone()).is_err(), "tiles join in a rung, not a merge");
                }
                let r = ladder.rung([g], u64::MAX).unwrap();
                assert_eq!(r, solved, "a resumed rung is the rung");
                r
            } else {
                ladder.solve(k, 2, u64::MAX).unwrap()
            };
            assert_eq!(rung.actions.len(), exact.len());
            for ((d, lo, up), (e, v)) in rung.actions.iter().zip(&exact) {
                assert_eq!(d, e);
                let (l, u) = (q(*lo, rung.mass), q(*up, rung.mass));
                assert!(l <= *v && *v <= u, "k {k}: {d} [{lo}, {up}] / {} vs {v}", rung.mass);
                if k + 1 >= left {
                    assert_eq!((l, u), (v.clone(), v.clone()), "collapse at k {k}");
                }
            }
            if let Some(p) = &prev {
                for (a, b) in p.actions.iter().zip(&rung.actions) {
                    assert!(a.1 <= b.1 && b.2 <= a.2, "nesting at k {k}: {a:?} then {b:?}");
                }
            }
            if rung.gamma() > 0 {
                open += 1;
            }
            prev = Some(rung);
        }
        checked += 1;
    }
    assert!(checked >= 20, "{checked}");
    assert!(open >= 5, "the bound must be open somewhere for the test to bite: {open}");
}
