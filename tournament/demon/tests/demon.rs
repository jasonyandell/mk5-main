// SPDX-FileCopyrightText: Jason Yandell
// SPDX-License-Identifier: CC0-1.0

//! STRATEGY-FUSION FENCE. The Demon's tests.
//! The Demon: a fenced tournament opponent; measuring stick only (`wiki/topics/the-demon.md`).
//!
//! - its solve agrees with an independent double-dummy search (a brute-force
//!   minimax over Zeb's own rules tables, not Silas's physics) on every world of
//!   enumerated fibers, make counts and margin tiebreaks both;
//! - it is deterministic given the seed;
//! - it plays only legal tiles, whole hands at all four seats, under every
//!   declaration;
//! - every source file carries the fence token.

use demon::{Config, Request, decide, record};
use scenario_core::fiber::Fiber;
use scenario_core::planner::{Rec, SplitMix};
use scenario_core::rules::Rules;

fn below(rng: &mut SplitMix, n: u64) -> u64 {
    rng.next_u64() % n
}

fn deal(rng: &mut SplitMix) -> [u32; 4] {
    let mut tiles: Vec<u8> = (0..28).collect();
    for i in (1..28).rev() {
        tiles.swap(i, below(rng, i as u64 + 1) as usize);
    }
    std::array::from_fn(|s| tiles[7 * s..7 * s + 7].iter().fold(0, |m, &t| m | 1 << t))
}

fn tiles_of(mask: u32) -> Vec<u8> {
    (0..28).filter(|&t| mask >> t & 1 == 1).collect()
}

/// A random hand played `k` tiles in by uniform legal play.
struct Spot {
    decl: u8,
    bid: u32,
    bidder: u8,
    hands: [u32; 4],
    plays: Vec<(u8, u8)>,
    rec: Rec,
}

fn spot(rng: &mut SplitMix, k: usize) -> Spot {
    let decl = below(rng, 10) as u8;
    let bid = [30, 31, 32, 34, 36, 41, 42, 84][below(rng, 8) as usize];
    let bidder = below(rng, 4) as u8;
    let mut hands = deal(rng);
    let (rules, mut rec) = record(decl, bidder, &[]).unwrap();
    let mut plays = vec![];
    for _ in 0..k {
        let s = rec.seat();
        let legal = tiles_of(rules.legal(hands[s as usize], rec.led()));
        let t = legal[below(rng, legal.len() as u64) as usize];
        hands[s as usize] &= !(1 << t);
        plays.push((s, t));
        rec = rec.advance(t, &rules);
    }
    Spot { decl, bid, bidder, hands, plays, rec }
}

/// The independent double-dummy search: plain minimax over Zeb's rules
/// tables, no pruning, no table. Returns the declaring side's final points.
fn points(rules: &Rules, hands: [u32; 4], rec: Rec, side: usize, threshold: u32, make: bool) -> i32 {
    if rec.finished() {
        let ours = rec.banked[side] as i32;
        return if make { (ours as u32 >= threshold) as i32 } else { ours - rec.banked[1 - side] as i32 };
    }
    let s = rec.seat() as usize;
    let declaring = s % 2 == side;
    let values = tiles_of(rules.legal(hands[s], rec.led())).into_iter().map(|t| {
        let mut h = hands;
        h[s] &= !(1 << t);
        points(rules, h, rec.advance(t, rules), side, threshold, make)
    });
    if declaring { values.max().unwrap() } else { values.min().unwrap() }
}

#[test]
fn the_solve_agrees_with_brute_force_minimax_on_every_world_of_the_fiber() {
    let mut rng = SplitMix::new(42);
    let mut checked = 0;
    let mut tiebreaks = 0;
    while checked < 30 {
        let k = 16 + below(&mut rng, 4) as usize; // three tricks to go, give or take a trick's plays
        let sp = spot(&mut rng, k);
        let viewer = sp.rec.seat();
        let rules = Rules::new(sp.decl).unwrap();
        let sizes: [u8; 4] = std::array::from_fn(|s| sp.rec.hand_size(s as u8));
        let mut fiber = Fiber::new(viewer, sp.hands[viewer as usize], sp.rec.played, sizes, sp.rec.forbid);
        let worlds = fiber.worlds();
        let legal = tiles_of(rules.legal(sp.hands[viewer as usize], sp.rec.led()));
        if legal.len() < 2 || worlds.len() > 2000 {
            continue;
        }
        let req = Request {
            seat: viewer,
            hand: sp.hands[viewer as usize],
            decl: sp.decl,
            bid: sp.bid,
            bidder: sp.bidder,
            plays: sp.plays.clone(),
            seed: 7,
        };
        let d = decide(Config { worlds: 2000, margin_tiebreak: true, ..Config::default() }, &req).unwrap();
        assert!(d.exact && d.worlds == worlds.len());
        let no_margin = decide(Config { worlds: 2000, margin_tiebreak: false, ..Config::default() }, &req).unwrap();
        assert_eq!(no_margin.worlds, d.worlds);
        assert!(no_margin.exact);
        assert_eq!(no_margin.moves, d.moves.iter().map(|&(t, m, _)| (t, m, None)).collect::<Vec<_>>());
        let declaring = viewer % 2 == sp.bidder % 2;
        let score = |m: u32| if declaring { m } else { d.worlds as u32 - m };
        let best = d.moves.iter().map(|&(_, m, _)| score(m)).max().unwrap();
        let first_best = d.moves.iter().find(|&&(_, m, _)| score(m) == best).unwrap().0;
        assert_eq!(no_margin.tile, first_best, "without margins, the lowest tile wins the make-count tie");
        let side = (sp.bidder % 2) as usize;
        let threshold = if sp.bid >= 42 { 42 } else { sp.bid };
        for &(t, makes, margin) in &d.moves {
            let after = |w: &[u32; 4]| {
                let mut h = *w;
                h[viewer as usize] &= !(1 << t);
                (h, sp.rec.advance(t, &rules))
            };
            let want: i32 = worlds
                .iter()
                .map(|w| {
                    let (h, r) = after(w);
                    points(&rules, h, r, side, threshold, true)
                })
                .sum();
            assert_eq!(makes as i32, want, "make count of tile {t}");
            if let Some(m) = margin {
                let want: i64 = worlds
                    .iter()
                    .map(|w| {
                        let (h, r) = after(w);
                        points(&rules, h, r, side, threshold, false) as i64
                    })
                    .sum();
                assert_eq!(m, want, "margin of tile {t}");
                tiebreaks += 1;
            }
        }
        checked += 1;
    }
    assert!(tiebreaks > 0, "some position exercised the margin tiebreak");
}

#[test]
fn exact_posterior_values_agree_with_independent_weighted_minimax() {
    let mut rng = SplitMix::new(666123);
    let mut checked = 0;
    while checked < 20 {
        let sp = spot(&mut rng, 19);
        let viewer = sp.rec.seat();
        if viewer == sp.bidder || sp.decl >= 7 {
            continue;
        }
        let rules = Rules::new(sp.decl).unwrap();
        let legal = tiles_of(rules.legal(sp.hands[viewer as usize], sp.rec.led()));
        if legal.len() < 2 {
            continue;
        }
        let mut fiber = Fiber::new(
            viewer,
            sp.hands[viewer as usize],
            sp.rec.played,
            std::array::from_fn(|s| sp.rec.hand_size(s as u8)),
            sp.rec.forbid,
        );
        let worlds = fiber.worlds();
        let known = sp.plays.iter().filter(|&&(s, _)| s == sp.bidder).fold(0, |m, &(_, t)| m | 1 << t);
        let weights: Vec<_> = worlds
            .iter()
            .map(|w| demon::hand_model::declaration_likelihood(w[sp.bidder as usize] | known, sp.decl, 32))
            .collect();
        let req = Request {
            seat: viewer,
            hand: sp.hands[viewer as usize],
            decl: sp.decl,
            bid: sp.bid,
            bidder: sp.bidder,
            plays: sp.plays.clone(),
            seed: 1,
        };
        let d = decide(Config { worlds: 2000, margin_tiebreak: true, declaration_weight: 32 }, &req).unwrap();
        assert!(d.exact);
        assert_eq!(d.mass, weights.iter().sum());
        for &(t, makes, margin) in &d.moves {
            let mut expected_make = 0;
            let mut expected_margin = 0;
            for (w, &weight) in worlds.iter().zip(&weights) {
                let mut after = *w;
                after[viewer as usize] &= !(1 << t);
                let r = sp.rec.advance(t, &rules);
                expected_make +=
                    weight * points(&rules, after, r.clone(), sp.bidder as usize % 2, sp.bid.min(42), true) as u32;
                expected_margin +=
                    weight as i64 * points(&rules, after, r, sp.bidder as usize % 2, sp.bid.min(42), false) as i64;
            }
            assert_eq!(makes, expected_make);
            if let Some(m) = margin {
                assert_eq!(m, expected_margin);
            }
        }
        let sign = if viewer % 2 == sp.bidder % 2 { 1 } else { -1 };
        let expected =
            d.moves.iter().max_by_key(|&&(t, m, g)| (sign * m as i64, sign * g.unwrap_or(0), -(t as i64))).unwrap().0;
        assert_eq!(d.tile, expected);
        checked += 1;
    }
}

#[test]
fn contract_forecast_recognizes_all_trumps_and_is_repeatable() {
    let sixes = (21..28).fold(0, |m, t| m | 1 << t);
    let declarations: Vec<u8> = (0..10).collect();
    let a = demon::bidding::forecast(sixes, 2, 42, 2, 666, &declarations).unwrap();
    let b = demon::bidding::forecast(sixes, 2, 42, 2, 666, &declarations).unwrap();
    assert_eq!(a.makes[6], Some(2), "seven trumps sweep every hidden deal");
    assert_eq!(a.makes, b.makes);
    assert_eq!(a.nodes, b.nodes);
    assert!(demon::bidding::forecast(sixes, 4, 30, 2, 666, &declarations).is_err());
    assert!(demon::bidding::forecast(sixes & !(1 << 27), 0, 30, 2, 666, &declarations).is_err());
    for bad in [vec![], vec![10], vec![6, 6]] {
        assert!(demon::bidding::forecast(sixes, 0, 30, 2, 666, &bad).is_err());
    }
}

#[test]
fn requested_declarations_keep_the_same_worlds_and_exact_counts() {
    let all: Vec<u8> = (0..10).collect();
    let mut rng = SplitMix::new(73);
    for seat in 0..4 {
        let hand = deal(&mut rng)[seat as usize];
        let full = demon::bidding::forecast(hand, seat, 31, 4, 91, &all).unwrap();
        // Reordered and incomplete: only the selected physics is searched.
        let partial = demon::bidding::forecast(hand, seat, 31, 4, 91, &[9, 6, 0, 7]).unwrap();
        assert_eq!(full.worlds, partial.worlds);
        assert!(partial.nodes < full.nodes);
        for d in 0..10 {
            assert_eq!(partial.makes[d], if [9, 6, 0, 7].contains(&d) { full.makes[d] } else { None });
        }
    }
}

#[test]
fn reading_own_declaration_or_an_unmodeled_contract_changes_nothing() {
    let sixes = (21..28).fold(0, |m, t| m | 1 << t);
    for decl in [6, 7, 8, 9] {
        let req = Request { seat: 0, hand: sixes, decl, bid: 30, bidder: 0, plays: vec![], seed: 8 };
        let cfg = Config { worlds: 8, ..Config::default() };
        assert_eq!(decide(cfg, &req).unwrap(), decide(Config { declaration_weight: 32, ..cfg }, &req).unwrap());
    }
}

#[test]
fn the_same_seed_gives_the_same_decision_and_the_seed_moves_the_worlds() {
    let mut rng = SplitMix::new(3);
    let mut moved = false;
    for _ in 0..12 {
        let k = 1 + below(&mut rng, 6) as usize;
        let sp = spot(&mut rng, k);
        let viewer = sp.rec.seat();
        let req = |seed| Request {
            seat: viewer,
            hand: sp.hands[viewer as usize],
            decl: sp.decl,
            bid: sp.bid,
            bidder: sp.bidder,
            plays: sp.plays.clone(),
            seed,
        };
        let cfg = Config { worlds: 24, margin_tiebreak: true, ..Config::default() };
        let a = decide(cfg, &req(11)).unwrap();
        assert_eq!(a, decide(cfg, &req(11)).unwrap());
        moved |= a.moves != decide(cfg, &req(12)).unwrap().moves;
    }
    assert!(moved, "a different seed draws different worlds somewhere");
}

#[test]
fn whole_hands_at_all_four_seats_are_legal_under_every_declaration() {
    let mut rng = SplitMix::new(9);
    for decl in 0..10u8 {
        let bidder = below(&mut rng, 4) as u8;
        let mut hands = deal(&mut rng);
        let (rules, mut rec) = record(decl, bidder, &[]).unwrap();
        let mut plays = vec![];
        while !rec.finished() {
            let s = rec.seat();
            let req = Request {
                seat: s,
                hand: hands[s as usize],
                decl,
                bid: 30,
                bidder,
                plays: plays.clone(),
                seed: rng.next_u64(),
            };
            let d = decide(Config { worlds: 8, margin_tiebreak: true, ..Config::default() }, &req).unwrap();
            assert!(rules.legal(hands[s as usize], rec.led()) >> d.tile & 1 == 1, "decl {decl}: an illegal tile");
            hands[s as usize] &= !(1 << d.tile);
            plays.push((s, d.tile));
            rec = rec.advance(d.tile, &rules);
        }
        assert_eq!(rec.banked[0] + rec.banked[1], 42);
    }
}

#[test]
fn a_request_that_misstates_the_record_is_refused() {
    let base = Request { seat: 0, hand: 0x7f, decl: 3, bid: 30, bidder: 0, plays: vec![], seed: 1 };
    let cfg = Config::default();
    assert!(decide(cfg, &base).is_ok());
    assert!(decide(cfg, &Request { seat: 1, ..base.clone() }).unwrap_err().contains("is to play"));
    assert!(decide(cfg, &Request { hand: 0x3f, ..base.clone() }).unwrap_err().contains("tiles"));
    assert!(decide(cfg, &Request { decl: 10, ..base.clone() }).is_err());
    assert!(decide(cfg, &Request { plays: vec![(1, 0)], ..base.clone() }).is_err());
}

#[test]
fn every_source_file_carries_the_fence_token() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    for f in [
        "Cargo.toml",
        "src/lib.rs",
        "src/main.rs",
        "src/sampling.rs",
        "src/hand_model.rs",
        "src/bidding.rs",
        "src/search.rs",
        "tests/demon.rs",
    ] {
        let text = std::fs::read_to_string(root.join(f)).unwrap();
        let head: String = text.lines().take(12).collect::<Vec<_>>().join("\n");
        assert!(head.contains("STRATEGY-FUSION FENCE"), "{f} lacks the fence token at its top");
        assert!(head.contains("a fenced tournament opponent; measuring stick only"), "{f} lacks the stated reason");
    }
}
