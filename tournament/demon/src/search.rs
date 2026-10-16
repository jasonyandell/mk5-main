// SPDX-FileCopyrightText: Jason Yandell
// SPDX-License-Identifier: CC0-1.0

//! STRATEGY-FUSION FENCE. The Demon: a fenced tournament opponent; measuring stick only.
//! Share immutable declaration tables, never world-specific search state.

use scenario_core::physics::ZebPhysics;
use scenario_core::rules::{N_DECLS, Rules};
use silas::ClairvoyantSolver;
use std::sync::OnceLock;

/// Only called after the request's declaration has been validated. The
/// cached solver is never searched: `fresh` gives each world its own empty
/// transposition table, owners and node counter, sharing only the rules.
pub(crate) fn template(decl: u8) -> &'static ClairvoyantSolver {
    static TEMPLATES: [OnceLock<ClairvoyantSolver>; N_DECLS as usize] = [const { OnceLock::new() }; N_DECLS as usize];
    TEMPLATES[decl as usize]
        .get_or_init(|| ClairvoyantSolver::new(&ZebPhysics::new(decl).expect("validated declaration").physics()))
}

/// Index of the first equivalent root tile. The outside mask includes the
/// open trick and every hidden tile, independent of which seat owns it.
///
/// Swapping two tiles in the mover's hand preserves every continuation when
/// they have equal count, led context, follow membership and comparisons
/// against all outside tiles. They cannot meet in one trick; once one is
/// played, the other is renamed in the paired continuation. Thus every
/// world's make and margin values agree, under either side's minimax.
pub(crate) fn representatives(rules: &Rules, tiles: &[u8], outside: u32) -> Vec<usize> {
    let mut reps = Vec::with_capacity(tiles.len());
    for (i, &tile) in tiles.iter().enumerate() {
        let a = tile as usize;
        let same = |j: usize| {
            let b = tiles[j] as usize;
            rules.count[a] == rules.count[b]
                && rules.led_suit[a] == rules.led_suit[b]
                && rules.follow_bits.iter().all(|f| (f >> a & 1) == (f >> b & 1))
                && (0..28)
                    .filter(|t| outside >> t & 1 == 1)
                    .all(|t| rules.rank.iter().all(|r| r[a].cmp(&r[t]) == r[b].cmp(&r[t])))
        };
        reps.push((0..i).find(|&j| reps[j] == j && same(j)).unwrap_or(i));
    }
    reps
}

#[cfg(test)]
mod tests {
    use super::*;
    use scenario_core::planner::SplitMix;
    use scenario_core::rules::tile;

    #[test]
    fn outside_trump_and_count_distinguish_otherwise_adjacent_root_tiles() {
        let r = Rules::new(6).unwrap();
        let (a, b, c) = (tile(6, 0), tile(6, 1), tile(6, 2));
        assert_eq!(representatives(&r, &[a, c], 0), vec![0, 0]);
        // A tile in the open trick must remain in `outside` too.
        assert_eq!(representatives(&r, &[a, c], 1 << b), vec![0, 1]);
        assert_eq!(representatives(&r, &[a, tile(6, 4)], 0), vec![0, 1]);
    }

    #[test]
    fn other_suits_cannot_be_renamed_even_when_their_ranks_agree() {
        let r = Rules::new(6).unwrap();
        // Same high pip, zero count: different follow membership.
        assert_eq!(representatives(&r, &[tile(4, 2), tile(4, 3)], 0), vec![0, 1]);
        // Same zero count: different led contexts.
        assert_eq!(representatives(&r, &[tile(2, 0), tile(3, 0)], 0), vec![0, 1]);
    }

    // Independent, unpruned minimax over the scenario rule tables. No Silas
    // solver, transpositions or equivalence pruning enters this referee.
    fn margin(r: &Rules, h: [u32; 4], leader: usize, trick: Vec<u8>) -> i32 {
        if trick.len() == 4 {
            let t: [u8; 4] = trick.try_into().unwrap();
            let winner = (leader + r.winner_offset(&t) as usize) % 4;
            let points = r.trick_points(&t) as i32 * if winner % 2 == 0 { 1 } else { -1 };
            return points + margin(r, h, winner, vec![]);
        }
        let seat = (leader + trick.len()) % 4;
        if h[seat] == 0 {
            return 0;
        }
        let legal = r.legal(h[seat], trick.first().copied());
        let values = (0..28).filter(|t| legal >> t & 1 == 1).map(|t| {
            let mut next = h;
            next[seat] &= !(1 << t);
            let mut plays = trick.clone();
            plays.push(t);
            margin(r, next, leader, plays)
        });
        if seat % 2 == 0 { values.max().unwrap() } else { values.min().unwrap() }
    }

    #[test]
    fn root_swaps_preserve_independent_minimax_with_every_prefix_length() {
        let mut rng = SplitMix::new(186);
        let mut collapsed = [0; 4];
        for decl in 0..N_DECLS {
            let r = Rules::new(decl).unwrap();
            for case in 0..1000 {
                let mut tiles: Vec<u8> = (0..28).collect();
                for i in (1..28).rev() {
                    tiles.swap(i, rng.next_u64() as usize % (i + 1));
                }
                let mut h = std::array::from_fn(|s| (1 << tiles[2 * s]) | (1 << tiles[2 * s + 1]));
                let leader = case % 4;
                let n = (case / 4) % 4;
                let mut trick = vec![];
                for i in 0..n {
                    let seat = (leader + i) % 4;
                    let legal = r.legal(h[seat], trick.first().copied());
                    let tile = legal.trailing_zeros() as u8;
                    h[seat] &= !(1 << tile);
                    trick.push(tile);
                }
                let seat = (leader + n) % 4;
                let legal = r.legal(h[seat], trick.first().copied());
                let tiles: Vec<u8> = (0..28).filter(|t| legal >> t & 1 == 1).collect();
                let outside = h.iter().enumerate().filter(|&(s, _)| s != seat).fold(0, |m, (_, h)| m | h)
                    | trick.iter().fold(0, |m, t| m | 1 << t);
                let reps = representatives(&r, &tiles, outside);
                let values: Vec<_> = tiles
                    .iter()
                    .map(|&t| {
                        let mut next = h;
                        next[seat] &= !(1 << t);
                        let mut plays = trick.clone();
                        plays.push(t);
                        margin(&r, next, leader, plays)
                    })
                    .collect();
                for (i, &rep) in reps.iter().enumerate() {
                    assert_eq!(values[i], values[rep], "decl {decl}, case {case}");
                    collapsed[n] += usize::from(i != rep);
                }
            }
        }
        assert!(collapsed.iter().all(|&n| n > 0), "every prefix exercised a root swap: {collapsed:?}");
    }
}
