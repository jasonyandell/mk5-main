// SPDX-FileCopyrightText: Jason Yandell
// SPDX-License-Identifier: CC0-1.0
// STRATEGY-FUSION FENCE. The Demon: a fenced tournament opponent; measuring stick only.
use demon::bidding::{best_forecast, supports};
use scenario_core::planner::SplitMix;
#[test]
fn exact_support_matches_the_best_count_and_carries_valid_bounds() {
    let mut rng = SplitMix::new(409122);
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
        let best = best_forecast(hand, seat, bid, n, seed, &ds).unwrap();
        for k in [0, n / 2, (n * 7 + 7) / 8, n] {
            let f = supports(hand, seat, bid, n, seed, &ds, k as u32).unwrap();
            assert_eq!(f.supported, best.makes >= k as u32);
            assert_eq!(f.worlds, n);
            assert!(f.nodes <= best.nodes);
            assert!(f.solves <= best.solves);
            assert_eq!(f.solves, f.checked.iter().map(|x| x.1).sum::<usize>());
            for &(d, tested, made) in &f.checked {
                assert!(ds.contains(&d));
                assert!(made as usize <= tested && tested <= n);
            }
            if f.supported {
                assert!(f.checked.last().unwrap().2 >= k as u32);
            } else {
                assert_eq!(f.checked.len(), ds.len());
                assert!(f.checked.iter().all(|&(_, tested, made)| made + ((n - tested) as u32) < k as u32));
            }
        }
    }
}
#[test]
fn impossible_required_count_is_refused() {
    assert!(supports((21..28).fold(0, |m, t| m | 1 << t), 0, 30, 2, 7, &[6], 3).is_err());
}
