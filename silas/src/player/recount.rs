// SPDX-FileCopyrightText: Jason Yandell
// SPDX-License-Identifier: CC0-1.0

//! A second construction of the player's root values, written apart from the first, for
//! the check at every decision (gate G1 of the exact-player design).
//!
//! It shares only the rules ([`Physics`]) and the declared laws with the player. It
//! enumerates the fiber itself (every deal of the unseen tiles that fits the seats' counts
//! and shown voids, by combinations), weights each atom by its own replay of the record,
//! evaluates at one parameter `p` in plain rationals (no course, no [`crate::family`]),
//! and settles only when the last tile is played. Settling early never changes a value,
//! so the two constructions must agree exactly wherever the player settles.
//!
//! STRATEGY-FUSION FENCE: nothing here is clairvoyant. The only maximum is over the
//! viewer's legal tiles at a history, after the sum over every atom it cannot tell apart.

use std::collections::BTreeMap;

use num_rational::BigRational;
use num_traits::{One, Zero};

use super::{Library, Root};
use crate::frame::Frame;
use crate::nouns::{ContextSet, Domino, DominoSet, Seat};
use crate::physics::Physics;
use crate::{Result, error};

type Q = BigRational;

/// One atom: the remaining hands of the four seats at the root, its type, its weight.
#[derive(Clone)]
struct Atom {
    hands: [DominoSet; Seat::COUNT],
    kind: usize,
    weight: Q,
}

/// Every way to give `pool` to the seats of `want`, each its count, none a tile of a
/// context it has shown void in.
fn deals(
    physics: &Physics,
    pool: DominoSet,
    want: &[(Seat, usize, ContextSet)],
    out: &mut Vec<Vec<DominoSet>>,
    acc: &mut Vec<DominoSet>,
) {
    let Some(&(_, n, voids)) = want.first() else {
        if pool.is_empty() {
            out.push(acc.clone());
        }
        return;
    };
    let allowed: Vec<Domino> = pool.iter().filter(|d| !voids.iter().any(|q| physics.follows(*d, q))).collect();
    for chosen in subsets(&allowed, n) {
        acc.push(chosen);
        deals(physics, pool.difference(chosen), &want[1..], out, acc);
        acc.pop();
    }
}

/// Every `n`-tile subset of `tiles`.
fn subsets(tiles: &[Domino], n: usize) -> Vec<DominoSet> {
    match (n, tiles.split_first()) {
        (0, _) => vec![DominoSet::EMPTY],
        (_, None) => vec![],
        (_, Some((&d, rest))) => {
            let mut with = subsets(rest, n - 1);
            for s in &mut with {
                s.insert(d);
            }
            with.extend(subsets(rest, n));
            with
        }
    }
}

struct Walk<'a> {
    physics: &'a Physics,
    library: &'a Library,
    viewer: Seat,
    declaring: usize,
    need: u32,
    viewer_bidding: bool,
}

#[derive(Clone)]
struct State {
    played: DominoSet,
    trick: Vec<Domino>,
    leader: Seat,
    banked: [u32; 2],
}

impl Walk<'_> {
    fn next(&self, s: &State, d: Domino) -> State {
        let mut t = s.clone();
        t.played.insert(d);
        t.trick.push(d);
        if t.trick.len() == Seat::COUNT {
            let plays = [t.trick[0], t.trick[1], t.trick[2], t.trick[3]];
            let w = self.physics.trick_winner(t.leader, plays);
            t.banked[w.team().index()] += self.physics.trick_points(plays);
            t.leader = w;
            t.trick.clear();
        }
        t
    }

    /// The viewer side's best success mass from `s` over `atoms`, all sharing its history.
    fn value(&self, s: &State, atoms: Vec<Atom>) -> Q {
        if s.played == DominoSet::FULL {
            let made = s.banked[self.declaring] >= self.need;
            return if made == self.viewer_bidding { atoms.iter().map(|a| &a.weight).sum() } else { Q::zero() };
        }
        let seat = s.leader.plus(s.trick.len());
        let lead = s.trick.first().copied();
        if seat == self.viewer {
            return self.tiles(s, &atoms).into_iter().map(|(_, v)| v).max().expect("a legal tile");
        }
        let mut split: BTreeMap<Domino, Vec<Atom>> = BTreeMap::new();
        for a in atoms {
            let hand = a.hands[seat.index()].difference(s.played);
            for (d, q) in self.library.types[a.kind].seats[seat.index()].law(self.physics, hand, lead) {
                split.entry(d).or_default().push(Atom { weight: &a.weight * q, ..a.clone() });
            }
        }
        split.into_iter().map(|(d, sub)| self.value(&self.next(s, d), sub)).sum()
    }

    /// Each of the viewer's legal tiles at `s`, with its success mass.
    fn tiles(&self, s: &State, atoms: &[Atom]) -> Vec<(Domino, Q)> {
        let hand = atoms[0].hands[self.viewer.index()].difference(s.played);
        self.physics
            .legal_plays(hand, s.trick.first().copied())
            .iter()
            .map(|d| (d, self.value(&self.next(s, d), atoms.to_vec())))
            .collect()
    }
}

/// The record's mass at `p` and each root tile's success probability given the record,
/// the viewer to play at `frame`. Refused when the record has no mass at `p`.
pub fn values(frame: &Frame, library: &Library, root: Root, p: &Q) -> Result<(Q, Vec<(Domino, Q)>)> {
    let physics = frame.physics();
    let viewer = frame.viewer();
    if frame.next_actor() != Some(viewer) {
        return Err(error("the viewer is not to play"));
    }
    // The record, replayed: who holds what was played, the voids, the open trick.
    let mut voids = [ContextSet::EMPTY; Seat::COUNT];
    let mut count = [7usize; Seat::COUNT];
    let mut leader = frame.contract().declarer();
    let mut banked = [0u32; 2];
    let mut trick: Vec<(Seat, Domino)> = Vec::new();
    for &(s, d) in frame.history() {
        if let Some(&(_, lead)) = trick.first() {
            let q = physics.led_context(lead);
            if !physics.follows(d, q) {
                voids[s.index()].insert(q);
            }
        }
        count[s.index()] -= 1;
        trick.push((s, d));
        if trick.len() == Seat::COUNT {
            let plays = [trick[0].1, trick[1].1, trick[2].1, trick[3].1];
            leader = physics.trick_winner(leader, plays);
            banked[leader.team().index()] += physics.trick_points(plays);
            trick.clear();
        }
    }
    let played: DominoSet = frame.history().iter().map(|(_, d)| *d).collect();
    let own = frame.dealt().difference(played);
    let pool = DominoSet::FULL.difference(frame.dealt()).difference(played);
    let others: Vec<(Seat, usize, ContextSet)> =
        (1..Seat::COUNT).map(|i| viewer.plus(i)).map(|s| (s, count[s.index()], voids[s.index()])).collect();
    let mut dealt = Vec::new();
    deals(physics, pool, &others, &mut dealt, &mut Vec::new());
    let mut atoms = Vec::new();
    for d in &dealt {
        let mut hands = [DominoSet::EMPTY; Seat::COUNT];
        hands[viewer.index()] = own;
        for (k, (s, _, _)) in others.iter().enumerate() {
            hands[s.index()] = d[k];
        }
        for (k, t) in library.types.iter().enumerate() {
            let prior = library.prior[k].at(p);
            if prior.is_zero() {
                continue;
            }
            let mut like = Q::one();
            if root == Root::Posterior {
                let mut held = hands;
                for &(s, x) in frame.history() {
                    held[s.index()].insert(x);
                }
                for chunk in frame.history().chunks(Seat::COUNT) {
                    for (i, &(s, x)) in chunk.iter().enumerate() {
                        if s != viewer {
                            let lead = (i > 0).then(|| chunk[0].1);
                            let law = t.seats[s.index()].law(physics, held[s.index()], lead);
                            like *= law.into_iter().find(|(y, _)| *y == x).map(|(_, q)| q).unwrap_or_else(Q::zero);
                        }
                        held[s.index()].remove(x);
                    }
                }
            }
            if !like.is_zero() {
                atoms.push(Atom { hands, kind: k, weight: prior * like });
            }
        }
    }
    let mass: Q = atoms.iter().map(|a| &a.weight).sum();
    if mass.is_zero() {
        return Err(error("the record has no mass at this parameter"));
    }
    let contract = frame.contract();
    let walk = Walk {
        physics,
        library,
        viewer,
        declaring: contract.declarer().team().index(),
        need: contract.threshold(),
        viewer_bidding: viewer.team() == contract.declarer().team(),
    };
    let state = State { played, trick: trick.iter().map(|(_, d)| *d).collect(), leader, banked };
    let tiles = walk.tiles(&state, &atoms).into_iter().map(|(d, v)| (d, v / &mass)).collect();
    Ok((mass, tiles))
}
