// SPDX-FileCopyrightText: Jason Yandell
// SPDX-License-Identifier: CC0-1.0

//! Counting is held to independent assignment enumeration, construction to
//! exact belief conditioning and the book's records, not just random draws.
mod common;

use common::fixture::book;
use common::*;
use silas::*;
use std::collections::BTreeSet;

fn work() -> Budget {
    Budget::new(10_000_000_000)
}

fn root() -> Frame {
    let e = book().example("hand-34");
    framed(&contract(&e.physics(), Seat::S0, 84), Seat::S0, e.hand(), vec![]).unwrap()
}

#[test]
fn book_case_has_exact_mass_and_complete_draws_not_cell_representatives() {
    let frame = root();
    let q = compile(include_str!("../examples/six-five-guarded.silas"));
    let space = WorldSpace::new(frame.clone(), &q, 1000, &mut work()).unwrap();
    assert_eq!((space.base_count(), space.count(), space.cell_count()), (399_072_960, 138_625_344, 10));
    assert_eq!(space.resolved(), tiles(&["6-5", "5-1", "5-3"]));
    assert_eq!(exact(space.probability()), r(33, 95));
    let mut rng = Rng(42);
    let mut draws = BTreeSet::new();
    for _ in 0..1000 {
        let w = space.sample(&mut || rng.next(), &mut work()).unwrap();
        assert!(frame.contains(&w));
        let holder = [Seat::S1, Seat::S3].into_iter().find(|s| w.hand(*s).contains(tile("6-5"))).unwrap();
        assert!(w.hand(holder).contains(tile("5-1")) || w.hand(holder).contains(tile("5-3")));
        draws.insert(w);
    }
    assert!(draws.len() > 900, "samples must complete cells, not just return ten representatives");
    assert_ne!(space.unrank(0, &mut work()).unwrap(), space.unrank(space.count() - 1, &mut work()).unwrap());
    assert!(space.unrank(space.count(), &mut work()).is_err());
}

#[test]
fn prepared_fibers_match_independent_assignments_with_voids_and_rotated_slots() {
    let p = physics("P6");
    let pool = tiles(&["0-0", "1-0", "1-1", "2-0", "2-1", "6-0"]);
    let ts: Vec<_> = pool.iter().collect();
    for viewer in Seat::ALL {
        for void_seat in 0..3 {
            let mut hidden =
                std::array::from_fn(|i| Hidden { seat: viewer.plus(i + 1), capacity: 2, voids: ContextSet::EMPTY });
            hidden[void_seat].voids.insert("q0".parse().unwrap());
            let kernel = Kernel::new(viewer, DominoSet::EMPTY, pool, hidden).unwrap();
            let prepared = p.prepare_worlds(&kernel);
            let mut expected = BTreeSet::new();
            for code in 0usize..3usize.pow(6) {
                let mut n = code;
                let mut hands = [DominoSet::EMPTY; 3];
                for d in &ts {
                    hands[n % 3].insert(*d);
                    n /= 3;
                }
                let w = kernel.world(hands);
                if kernel.contains(&p, &w) {
                    expected.insert(w);
                }
            }
            let actual: BTreeSet<_> = (0..prepared.count()).map(|k| prepared.unrank(k).unwrap()).collect();
            assert_eq!(actual.len() as u128, prepared.count());
            assert_eq!(actual, expected);
            assert!(prepared.unrank(prepared.count()).is_err());
            assert_eq!(
                p.worlds(&kernel).collect::<Vec<_>>(),
                (0..prepared.count()).map(|k| prepared.unrank(k).unwrap()).collect::<Vec<_>>()
            );
        }
    }
}

#[test]
fn every_small_world_matches_query_conditioning_with_projection_negation_and_aggregates() {
    for decl in ["P6", "P0", "DT", "NT"] {
        let frame = small_frame(physics(decl));
        let belief = Belief::uniform(frame.clone(), 1000).unwrap();
        for source in [
            "(query)",
            "(query (holds c d))", // many witnesses, each world counted once
            "(query (pattern (holds S1 d)) (pattern (holds S1 d)))", // union
            "(query (out) (holds S1 d) (holds S2 e) (tile d e))", // injectivity
            "(query (out) (some (roles d) (holds S1 d) (double d)))",
            "(query (out) (none (roles d) (holds S1 d) (double d)))",
            "(query (out) (>= (card (roles d) (holds S1 d) (double d)) 1))",
            "(query (out) (not (led-context q0)))", // semantic undefined at a closed trick
            "(query (out) (mine 6-6))",             // fixed-viewer event
        ] {
            let q = compile(source);
            let space = WorldSpace::new(frame.clone(), &q, 1000, &mut work()).unwrap();
            let expected: BTreeSet<_> = belief
                .worlds()
                .filter_map(|(w, _)| (!q.evaluate(&frame, w, &mut work()).unwrap().is_empty()).then_some(*w))
                .collect();
            let actual: BTreeSet<_> = (0..space.count()).map(|k| space.unrank(k, &mut work()).unwrap()).collect();
            assert_eq!(actual.len() as u128, space.count(), "{decl}: {source}");
            assert_eq!(actual, expected, "{decl}: {source}");
            assert_eq!(space.probability(), q.summarize(&belief, &mut work()).unwrap().event_probability());
        }
    }
}

#[test]
fn impossible_is_zero_but_exhaustion_and_invalid_queries_are_errors() {
    let empty = WorldSpace::new(root(), &compile("(query (holds S1 5-5))"), 100, &mut work()).unwrap();
    assert_eq!(empty.count(), 0);
    assert_eq!(exact(empty.probability()), r(0, 1));
    assert!(empty.sample(&mut || 0, &mut work()).is_err());
    assert!(empty.unrank(0, &mut work()).is_err());
    let q = compile(include_str!("../examples/six-five-guarded.silas"));
    assert!(WorldSpace::new(root(), &q, 1, &mut work()).err().unwrap().0.contains("cap"));
    assert!(WorldSpace::new(root(), &q, 1000, &mut Budget::new(1)).err().unwrap().0.contains("budget"));
    assert!(WorldSpace::new(root(), &compile("(query (least-raise 30))"), 100, &mut work()).is_err());
    let space = WorldSpace::new(root(), &q, 100, &mut work()).unwrap();
    // All-zero words stay in the rejected tail for this non-power-of-two count.
    assert!(space.sample(&mut || 0, &mut Budget::new(3)).err().unwrap().0.contains("budget"));
}

#[test]
fn opening_family_reproduces_the_prototype_and_varies_the_viewers_hand() {
    let q: Query = include_str!("../examples/construction/four-fives.silas").parse().unwrap();
    let space = OpeningSpace::new(
        Auction::won_by(Seat::S0, Bid::number(84).unwrap()).unwrap(),
        physics("P5"),
        Seat::S0,
        &q,
        20_000,
        &mut work(),
    )
    .unwrap();
    assert_eq!(space.base_count(), 472_518_347_558_400);
    assert_eq!((space.count(), space.cell_count()), (1_843_717_075_200, 100));
    assert_eq!(space.resolved(), physics("P5").incidence("q*".parse().unwrap()));
    let mut rng = Rng(2026);
    let mut hands = BTreeSet::new();
    for _ in 0..100 {
        let (f, w) = space.sample(&mut || rng.next(), &mut work()).unwrap();
        assert!(f.contains(&w));
        assert_eq!(f.dealt().intersection(space.resolved()).len(), 4);
        assert!(f.dealt().contains(tile("5-5")) && !f.dealt().contains(tile("6-5")));
        hands.insert(f.dealt());
    }
    assert!(hands.len() > 90);
    assert!(space.unrank(space.count(), &mut work()).is_err());
}

#[test]
fn opening_fragment_refuses_unchecked_viewer_reads_and_invalid_auctions() {
    for source in ["(query (mine 5-5))", "(query (own-legal d))", "(query (lay-down))"] {
        let result = OpeningSpace::new(
            root().auction().clone(),
            physics("P5"),
            Seat::S0,
            &source.parse().unwrap(),
            100,
            &mut work(),
        );
        assert!(result.err().unwrap().0.contains("does not admit"));
    }
    let q = "(query)".parse().unwrap();
    let invalid = Auction::new(AuctionConfig::STRAIGHT, Seat::S3, vec![]).unwrap();
    assert!(OpeningSpace::new(invalid, physics("P5"), Seat::S0, &q, 100, &mut work()).is_err());
    let impossible = OpeningSpace::restricted(
        root().auction().clone(),
        physics("P5"),
        Seat::S0,
        &q,
        [DominoSet::EMPTY; 4],
        100,
        &mut work(),
    )
    .unwrap();
    assert_eq!(impossible.count(), 0);
    assert_eq!(impossible.probability(), None);
}

#[test]
fn four_seat_restricted_openings_are_a_bijection_against_independent_permutations() {
    let fixed: [DominoSet; 4] = std::array::from_fn(|s| Domino::ALL[s * 6..s * 6 + 6].iter().copied().collect());
    let free: DominoSet = Domino::ALL[24..].iter().copied().collect();
    let allowed = fixed.map(|hand| hand.union(free));
    let q: Query = "(query (out) (or (holds S0 6-3) (holds S1 6-4)))".parse().unwrap();
    let space =
        OpeningSpace::restricted(root().auction().clone(), physics("P5"), Seat::S0, &q, allowed, 100, &mut work())
            .unwrap();
    let mut expected = BTreeSet::new();
    for code in 0usize..256 {
        let mut hands = fixed;
        let mut n = code;
        for d in free.iter() {
            hands[n % 4].insert(d);
            n /= 4;
        }
        if hands.iter().all(|h| h.len() == 7) && (hands[0].contains(tile("6-3")) || hands[1].contains(tile("6-4"))) {
            expected.insert(hands.map(|h| h.bits()));
        }
    }
    let actual: BTreeSet<_> = (0..space.count())
        .map(|k| {
            let (_, world) = space.unrank(k, &mut work()).unwrap();
            Seat::ALL.map(|s| world.hand(s).bits())
        })
        .collect();
    assert_eq!(space.base_count(), 24);
    assert_eq!(actual.len() as u128, space.count());
    assert_eq!(actual, expected);
}

#[test]
fn constructed_completions_replay_every_full_deals_legal_record_prefix_at_every_seating() {
    let q = compile("(query (roles) (out) (pattern))");
    let mut checked = 0;
    let mut stopped = Vec::new();
    for e in book().examples.iter().filter(|e| e.kind == "deal") {
        let p = e.physics();
        let plays = book().line(&format!("{}/prose", e.id)).plays();
        let mut hands = e.deal();
        let mut leader = Seat::S0;
        for n in 0..=plays.len() {
            for rotation in Seat::ALL {
                for role in Seat::ALL {
                    let viewer = rotation.plus(role.index());
                    let frame = framed(
                        &contract(&p, rotation, e.bid.number()),
                        viewer,
                        e.deal()[role.index()],
                        plays[..n].to_vec(),
                    )
                    .unwrap();
                    let space = WorldSpace::new(frame.clone(), &q, 1, &mut work()).unwrap();
                    assert_eq!(space.count(), frame.count_worlds());
                    for rank in [0, space.count() - 1] {
                        let world = space.unrank(rank, &mut work()).unwrap();
                        for seat in Seat::ALL {
                            let own_plays: DominoSet =
                                frame.history().iter().filter(|(s, _)| *s == seat).map(|(_, d)| *d).collect();
                            let replayed = Frame::new(
                                frame.auction().clone(),
                                p.clone(),
                                seat,
                                world.hand(seat).union(own_plays),
                                frame.plays().to_vec(),
                            )
                            .unwrap();
                            assert!(replayed.contains(&world));
                        }
                    }
                    checked += 1;
                }
            }
            if n == plays.len() {
                break;
            }
            let seat = leader.plus(n % 4);
            let lead = if n % 4 == 0 { None } else { Some(plays[n - n % 4]) };
            if !p.legal_plays(hands[seat.index()], lead).contains(plays[n]) {
                stopped.push((e.id.as_str(), n));
                break;
            }
            hands[seat.index()].remove(plays[n]);
            if n % 4 == 3 {
                leader = p.trick_winner(leader, plays[n - 3..=n].try_into().unwrap());
            }
        }
    }
    assert_eq!(stopped, vec![("hand-33", 23)]); // the fixture's recorded follow-suit conflict
    assert!(checked > 2500, "checked {checked} frames");
}
