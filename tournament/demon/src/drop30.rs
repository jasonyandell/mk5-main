// SPDX-FileCopyrightText: Jason Yandell
// SPDX-License-Identifier: CC0-1.0

//! STRATEGY-FUSION FENCE. The Demon: a fenced tournament opponent; measuring stick only.
//! Infer holdings from choices. Each observed tile competes only against the
//! alternatives this hypothetical player could legally have held at that time.
use super::{Request, hand_model, sampling};
use scenario_core::{
    fiber::Fiber,
    planner::{Rec, SplitMix},
    rules::Rules,
};

struct Observation {
    seat: usize,
    later: u32,
    follow: u32,
    chosen: usize,
    preference: [f64; 28],
}

fn each(mut mask: u32, mut f: impl FnMut(usize)) {
    while mask != 0 {
        f(mask.trailing_zeros() as usize);
        mask &= mask - 1;
    }
}

impl Observation {
    fn probability(&self, world: &[u32; 4]) -> f64 {
        let hand = world[self.seat] | self.later;
        let follows = hand & self.follow;
        let legal = if follows == 0 { hand } else { follows };
        let mut total = 0.0;
        each(legal, |tile| total += self.preference[tile]);
        self.preference[self.chosen] / total
    }
}

// A positive, deliberately approximate play model. Lead top tiles; while
// following, capture or save count and conserve winners when discarding.
fn preferences(r: &Rules, rec: &Rec, bidder: u8) -> [f64; 28] {
    let actor = rec.seat();
    let suit = rec.led().map(|t| r.led_suit[t as usize] as usize);
    let trick = &rec.trick[..rec.n as usize];
    let high = suit.and_then(|l| trick.iter().enumerate().max_by_key(|(_, t)| r.rank[l][**t as usize]));
    let trick_count: u8 = trick.iter().map(|&t| r.count[t as usize]).sum();
    std::array::from_fn(|t| {
        let lead = r.led_suit[t] as usize;
        let trump = lead == 7;
        let rank = r.rank[lead][t];
        let stronger = (0..28)
            .filter(|&j| rec.played & (1 << j) == 0 && r.follow_bits[lead] & (1 << j) != 0 && r.rank[lead][j] > rank)
            .count();
        let boss = stronger == 0;
        let count = r.count[t] as f64 / 5.0;
        let score = if let (Some(l), Some((pos, &tile))) = (suit, high) {
            let beats = r.rank[l][t] > r.rank[l][tile as usize];
            let partner = (pos % 2) == (rec.n as usize % 2);
            let keep = partner || beats;
            let win_value = if !partner && beats { 1.2 + trick_count as f64 * 0.09 } else { 0.0 };
            win_value + if keep { 1.3 * count } else { -1.7 * count }
                - if boss { 1.0 } else { 0.0 }
                - if trump { 1.2 } else { 0.0 }
                - if partner && beats { 1.0 } else { 0.0 }
        } else {
            (if boss { 2.4 } else { -0.22 * stronger as f64 })
                + (if trump && actor == bidder {
                    1.25
                } else if trump {
                    -1.1
                } else {
                    0.0
                })
                + (if boss { 0.3 } else { -1.4 }) * count
        };
        score.exp2()
    })
}

pub(super) fn draw(
    fiber: &mut Fiber,
    n: usize,
    req: &Request,
    rules: &Rules,
) -> Result<(Vec<[u32; 4]>, Vec<u32>, bool, usize), String> {
    let mut rng = SplitMix::new(req.seed);
    let mut remaining_public = [0u32; 4];
    for &(s, t) in &req.plays {
        remaining_public[s as usize] |= 1 << t;
    }
    let bidder_known = remaining_public[req.bidder as usize];
    let mut observations = Vec::new();
    let mut rec = Rec::opening(req.bidder);
    for &(s, t) in &req.plays {
        if s != req.seat {
            observations.push(Observation {
                seat: s as usize,
                later: remaining_public[s as usize],
                chosen: t as usize,
                follow: rec.led().map_or(0, |lead| rules.followers(lead)),
                preference: preferences(rules, &rec, req.bidder),
            });
        }
        remaining_public[s as usize] ^= 1 << t;
        rec = rec.advance(t, rules);
    }
    if observations.is_empty() && req.seat == req.bidder {
        let (ws, exact) = sampling::sample(fiber, n, &mut rng)?;
        let k = ws.len();
        return Ok((ws, vec![1; k], exact, k));
    }
    let limit = n * if req.seat == req.bidder { 4 } else { 12 };
    let (pool, _) = sampling::sample(fiber, limit, &mut rng)?;
    let mut cumulative = Vec::with_capacity(pool.len());
    let mut total = 0.0;
    for w in &pool {
        let consistent =
            req.seat == req.bidder || hand_model::bid30_declaration(w[req.bidder as usize] | bidder_known) == req.decl;
        let likelihood: f64 = observations.iter().map(|obs| obs.probability(w)).product();
        total += if consistent { likelihood.sqrt() } else { 0.0 };
        cumulative.push(total);
    }
    // A finite proposal pool can miss a rare declaration. The public plays
    // still provide evidence in that case. Never retry until the clock expires.
    if total == 0.0 {
        for (w, c) in pool.iter().zip(&mut cumulative) {
            total += observations.iter().map(|obs| obs.probability(w)).product::<f64>().sqrt();
            *c = total;
        }
    }
    // Systematic resampling assigns integer multiplicities. Solve each
    // selected deal once and use the multiplicity in both make and margin.
    let offset = rng.random();
    let mut i = 0;
    let mut last = None;
    let mut worlds = Vec::new();
    let mut weights = Vec::new();
    for j in 0..n {
        // Rounding the last draw up to total would select a trailing zero-weight deal.
        let target = ((j as f64 + offset) / n as f64 * total).min(total.next_down());
        while i + 1 < pool.len() && cumulative[i] <= target {
            i += 1;
        }
        if last == Some(i) {
            *weights.last_mut().unwrap() += 1;
        } else {
            worlds.push(pool[i]);
            weights.push(1);
            last = Some(i);
        }
    }
    Ok((worlds, weights, false, pool.len()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_only_over_choices_available_at_the_time() {
        let mut preference = [1.0; 28];
        preference[2] = 3.0;
        let obs = Observation { seat: 1, later: 1 << 2, follow: (1 << 2) | (1 << 3), chosen: 2, preference };
        let mut world = [0u32; 4];
        world[1] = 1 << 5;
        assert_eq!(obs.probability(&world), 1.0);
        world[1] |= 1 << 3;
        assert_eq!(obs.probability(&world), 0.75);
        let lead = Observation { follow: 0, ..obs };
        assert_eq!(lead.probability(&world), 0.6);
    }
}
