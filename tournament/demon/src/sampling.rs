// SPDX-FileCopyrightText: Jason Yandell
// SPDX-License-Identifier: CC0-1.0

//! STRATEGY-FUSION FENCE. The Demon: a fenced tournament opponent; measuring stick only.
//! Exact-uniform subsets of the fiber. The fiber's completion-count dynamic
//! program un-ranks a selected world without enumerating the worlds before it.

use scenario_core::fiber::Fiber;
use scenario_core::planner::SplitMix;
use std::collections::HashSet;

/// Floyd's algorithm: a uniform subset of n distinct integers in 0..total.
/// Every stage extends a uniform subset by one. Insertion order is retained
/// explicitly; randomized hash-table iteration never enters the decision.
fn ranks(total: u64, n: usize, mut below: impl FnMut(u64) -> u64) -> Vec<u64> {
    let mut seen = HashSet::with_capacity(n);
    let mut result = Vec::with_capacity(n);
    for j in total - n as u64..total {
        let t = below(j + 1);
        let k = if seen.insert(t) {
            t
        } else {
            seen.insert(j);
            j
        };
        result.push(k);
    }
    result
}

/// Integer rejection sampling avoids modulo bias and a floating-point draw.
fn below(rng: &mut SplitMix, bound: u64) -> u64 {
    let reject = bound.wrapping_neg() % bound;
    loop {
        let x = rng.next_u64();
        if x >= reject {
            return x % bound;
        }
    }
}

pub(crate) fn sample(fiber: &mut Fiber, n: usize, rng: &mut SplitMix) -> Result<(Vec<[u32; 4]>, bool), String> {
    let total = fiber.size();
    if total <= 0 {
        return Err("no hidden deal fits the record".into());
    }
    if total as usize <= n {
        return Ok((fiber.worlds(), true));
    }
    let worlds =
        ranks(total as u64, n, |bound| below(rng, bound)).into_iter().map(|k| fiber.world_at(k as i64)).collect();
    Ok((worlds, false))
}

/// Exact enumeration with integer likelihoods, or IID draws from the modeled
/// posterior by rejection. No self-normalized importance approximation, sample
/// impoverishment, hard exclusion, or silent return to a uniform belief.
pub(crate) fn posterior(
    fiber: &mut Fiber,
    n: usize,
    rng: &mut SplitMix,
    bound: u32,
    likelihood: impl Fn(&[u32; 4]) -> u32,
) -> Result<(Vec<[u32; 4]>, Vec<u32>, bool, usize), String> {
    let total = fiber.size();
    if total <= 0 {
        return Err("no hidden deal fits the record".into());
    }
    if bound == 1 {
        let (worlds, exact) = sample(fiber, n, rng)?;
        let len = worlds.len();
        return Ok((worlds, vec![1; len], exact, len));
    }
    if total as usize <= n {
        let worlds = fiber.worlds();
        let weights = worlds.iter().map(&likelihood).collect();
        return Ok((worlds, weights, true, total as usize));
    }
    let mut worlds = Vec::with_capacity(n);
    let mut proposals = 0;
    while worlds.len() < n {
        let w = fiber.world_at(below(rng, total as u64) as i64);
        let weight = likelihood(&w);
        assert!(weight > 0 && weight <= bound);
        proposals += 1;
        if below(rng, bound as u64) < weight as u64 {
            worlds.push(w);
        }
    }
    Ok((worlds, vec![1; n], false, proposals))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    #[test]
    fn every_three_of_five_subset_has_exactly_the_same_number_of_draws() {
        let mut counts = BTreeMap::new();
        for a in 0..3 {
            for b in 0..4 {
                for c in 0..5 {
                    let mut draws = [a, b, c].into_iter();
                    let mut selected = ranks(5, 3, |bound| {
                        let x = draws.next().unwrap();
                        assert!(x < bound);
                        x
                    });
                    selected.sort_unstable();
                    assert!(selected.windows(2).all(|p| p[0] < p[1]));
                    *counts.entry(selected).or_insert(0) += 1;
                }
            }
        }
        assert_eq!(counts.len(), 10);
        assert!(counts.values().all(|&n| n == 6));
    }

    #[test]
    fn sampled_worlds_are_distinct_and_repeatable_and_the_full_fiber_is_exact() {
        // One tile in each hand, 24 down: the three unseen tiles have six deals.
        let own = 1;
        let played = ((1 << 28) - 1) & !0xf;
        let mut fiber = Fiber::new(0, own, played, [1; 4], [0; 4]);
        let all = fiber.worlds();
        for seed in 0..100 {
            let (worlds, exact) = sample(&mut fiber, 5, &mut SplitMix::new(seed)).unwrap();
            assert!(!exact);
            assert_eq!(worlds.iter().collect::<HashSet<_>>().len(), 5);
            assert!(worlds.iter().all(|w| all.contains(w)));
            assert_eq!(sample(&mut fiber, 5, &mut SplitMix::new(seed)).unwrap().0, worlds);
        }
        assert_eq!(sample(&mut fiber, 6, &mut SplitMix::new(7)).unwrap(), (all, true));
        let mut impossible = Fiber::new(0, own, played, [1; 4], [0, 0xf, 0xf, 0xf]);
        assert!(sample(&mut impossible, 6, &mut SplitMix::new(7)).is_err());
    }

    #[test]
    fn posterior_draws_match_enumerated_likelihoods_and_never_exclude_a_world() {
        let played = ((1 << 28) - 1) & !0xf;
        let mut fiber = Fiber::new(0, 1, played, [1; 4], [0; 4]);
        let likelihood = |w: &[u32; 4]| if w[1] == 2 { 9 } else { 1 };
        let (all, weights, exact, proposals) = posterior(&mut fiber, 6, &mut SplitMix::new(7), 9, likelihood).unwrap();
        assert!(exact);
        assert_eq!(proposals, 6);
        assert_eq!(weights.iter().sum::<u32>(), 22); // two high, four low
        assert_eq!(weights, all.iter().map(likelihood).collect::<Vec<_>>());
        let mut counts = vec![0; 6];
        let mut rng = SplitMix::new(123456);
        for _ in 0..30_000 {
            let (draws, weights, exact, _) = posterior(&mut fiber, 1, &mut rng, 9, likelihood).unwrap();
            assert!(!exact);
            assert_eq!(weights, vec![1]);
            counts[all.iter().position(|w| w == &draws[0]).unwrap()] += 1;
        }
        for (i, w) in all.iter().enumerate() {
            let want = 30_000.0 * likelihood(w) as f64 / 22.0;
            assert!((counts[i] as f64 - want).abs() < 350.0, "{counts:?}");
        }
        let a = posterior(&mut fiber, 4, &mut SplitMix::new(8), 9, likelihood).unwrap();
        let b = posterior(&mut fiber, 4, &mut SplitMix::new(8), 9, likelihood).unwrap();
        assert_eq!(a, b);
    }
}
