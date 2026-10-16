// SPDX-FileCopyrightText: Jason Yandell
// SPDX-License-Identifier: CC0-1.0
// STRATEGY-FUSION FENCE. The Demon: a fenced tournament opponent; measuring stick only.
use demon::bidding::{best_forecast, forecast};
use scenario_core::planner::SplitMix;
#[test]
fn bounded_forecast_preserves_full_forecast_winner_score_and_ties() {
    let mut rng = SplitMix::new(819279);
    let mut saved = 0;
    let mut tied = 0;
    for i in 0..48 {
        let mut tiles: Vec<u8> = (0..28).collect();
        for j in (1..28).rev() {
            tiles.swap(j, (rng.next_u64() % (j + 1) as u64) as usize);
        }
        let hand = tiles[..7].iter().fold(0, |m, &t| m | 1 << t);
        let ds: Vec<u8> = match i % 4 {
            0 => (0..10).rev().collect(),
            1 => vec![0, 1, 2, 3, 4, 5, 6, 7, 9],
            2 => vec![9, 6, 0, 7],
            _ => vec![6],
        };
        let n = if i < 8 {
            1
        } else if i < 40 {
            8
        } else {
            64
        };
        let bid = [30, 31, 35, 42, 84][i % 5];
        let seat = (i % 4) as u8;
        let seed = rng.next_u64();
        let full = forecast(hand, seat, bid, n, seed, &ds).unwrap();
        let fast = best_forecast(hand, seat, bid, n, seed, &ds).unwrap();
        let preferred = demon::hand_model::heuristic_declaration(hand);
        let want = *ds.iter().max_by_key(|&&d| (full.makes[d as usize].unwrap(), d == preferred, -(d as i16))).unwrap();
        assert_eq!(fast.decl, want);
        assert_eq!(fast.makes, full.makes[want as usize].unwrap());
        assert_eq!(fast.worlds, full.worlds);
        assert!(fast.nodes <= full.nodes);
        assert!(fast.solves <= n * ds.len());
        saved += n * ds.len() - fast.solves;
        tied += usize::from(ds.iter().filter(|&&d| full.makes[d as usize] == Some(fast.makes)).count() > 1);
    }
    assert!(saved > 0);
    assert!(tied > 0);
}
#[test]
fn a_perfect_preferred_declaration_needs_no_other_search() {
    let hand = (21..28).fold(0, |m, t| m | 1 << t);
    let f = best_forecast(hand, 2, 42, 16, 666, &[0, 1, 2, 3, 4, 5, 6, 7, 9]).unwrap();
    assert_eq!(f.decl, 6);
    assert_eq!(f.makes, 16);
    assert_eq!(f.solves, 16);
}
#[test]
fn bounded_forecast_rejects_bad_inputs() {
    let hand = (21..28).fold(0, |m, t| m | 1 << t);
    for ds in [vec![], vec![10], vec![6, 6]] {
        assert!(best_forecast(hand, 0, 30, 2, 1, &ds).is_err());
    }
    for n in [0, 4097] {
        assert!(best_forecast(hand, 0, 30, n, 1, &[6]).is_err());
    }
    assert!(best_forecast(hand, 4, 30, 2, 1, &[6]).is_err());
    assert!(best_forecast(hand | 1, 0, 30, 2, 1, &[6]).is_err());
}

#[test]
fn common_world_make_counts_are_monotone_in_the_point_target() {
    let mut rng = SplitMix::new(741923);
    let declarations = [0, 1, 2, 3, 4, 5, 6, 7, 9];
    for seat in 0..4 {
        let mut tiles: Vec<u8> = (0..28).collect();
        for j in (1..28).rev() {
            tiles.swap(j, (rng.next_u64() % (j + 1) as u64) as usize);
        }
        let hand = tiles[..7].iter().fold(0, |m, &t| m | 1 << t);
        let seed = rng.next_u64();
        let mut previous = vec![Some(8); 10];
        for bid in 30..=42 {
            let f = forecast(hand, seat, bid, 8, seed, &declarations).unwrap();
            for &d in &declarations {
                assert!(f.makes[d as usize].unwrap() <= previous[d as usize].unwrap());
            }
            previous = f.makes;
        }
    }
}
