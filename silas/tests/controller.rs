// SPDX-FileCopyrightText: Jason Yandell
// SPDX-License-Identifier: CC0-1.0

//! The note's whole-ending family, rebuilt on Silas's rules (Jason's note *A minimal
//! comparison algebra for a declared 42 family*, 2026-10-21; texas-42
//! `lean/Texas42/Witness.lean`, § 10.4): seat 3 bid 31 at no-trump, and holds 2-1,
//! 3-1, 4-1, 5-1 before its play to the fourth trick. Its frame leaves 560 deals. The
//! partner is High (plays its highest legal tile) with probability `p` and Low (its
//! lowest) otherwise; the opponents play their lowest. The viewer maximizes the chance
//! of making 31. Every count of the note is reproduced: 24,089 information nodes,
//! fourteen thresholds, 31 parameter classes, 767 decision states with a root ladder 1,
//! 18, 31, 31, and, with the first play 2-1 committed, 302 states and 1, 10, 23, 23.

mod common;

use std::collections::BTreeMap;

use common::{physics, tile, tiles};
use num_rational::BigRational;
use silas::comparison::{Play, dimensions};
use silas::family::{Affine, Choice, Family, Piecewise, Reach, representatives};
use silas::quotient::{Block, Interface};
use silas::*;

type Q = BigRational;

#[derive(Clone)]
struct Game {
    hands: [DominoSet; 4],
    trick: Vec<Domino>,
    leader: Seat,
    banked: [u32; 2],
}

fn team(s: Seat) -> usize {
    if s.team() == Team::T0 { 0 } else { 1 }
}

impl Game {
    fn actor(&self) -> Seat {
        self.leader.plus(self.trick.len())
    }

    fn legal(&self, p: &Physics) -> DominoSet {
        p.legal_plays(self.hands[self.actor().index()], self.trick.first().copied())
    }

    fn play(&self, p: &Physics, d: Domino) -> Game {
        let mut g = self.clone();
        g.hands[self.actor().index()].remove(d);
        g.trick.push(d);
        if g.trick.len() == 4 {
            let plays = [g.trick[0], g.trick[1], g.trick[2], g.trick[3]];
            let w = p.trick_winner(g.leader, plays);
            g.banked[team(w)] += p.trick_points(plays);
            g.leader = w;
            g.trick.clear();
        }
        g
    }

    fn done(&self) -> bool {
        self.hands.iter().all(|h| h.is_empty()) && self.trick.is_empty()
    }
}

const VIEWER: Seat = Seat::S3;
const PARTNER: Seat = Seat::S1;

/// The others play until the viewer is to play or the hand is over.
fn advance(p: &Physics, mut g: Game, high: bool) -> (Game, Vec<Domino>) {
    let mut seen = Vec::new();
    while !g.done() && g.actor() != VIEWER {
        let legal = g.legal(p);
        let d = if g.actor() == PARTNER && high { legal.iter().last() } else { legal.iter().next() };
        let d = d.expect("a legal play");
        seen.push(d);
        g = g.play(p, d);
    }
    (g, seen)
}

struct Built {
    family: Family<Domino, Vec<Domino>>,
    root: usize,
    nodes: usize,
    /// Each decision's plays over atoms (world, type), for the comparison dimension.
    plays: Vec<Vec<Play>>,
    /// The note's local history: after 2-1 and the observed 1-0, 6-2, 5-2.
    local: usize,
}

type Atom = (usize, bool, Game);

fn q(n: i64, d: i64) -> Q {
    Q::new(n.into(), d.into())
}

fn build() -> Built {
    let p = physics("NT");
    let plays: Vec<Domino> =
        ["6-3", "6-1", "6-4", "6-0", "0-0", "2-2", "5-0", "2-0", "4-3", "4-2", "4-0", "5-4", "1-1", "3-0", "3-3"]
            .iter()
            .map(|s| tile(s))
            .collect();
    let dealt = tiles(&["6-3", "5-0", "4-0", "2-1", "5-1", "4-1", "3-1"]);
    let frame =
        Frame::new(Auction::won_by(Seat::S3, Bid::Points(31)).unwrap(), p.clone(), VIEWER, dealt, plays.clone())
            .unwrap();
    assert_eq!(frame.banked(), [1, 17]);
    assert_eq!(frame.leader(), Seat::S0);
    let worlds: Vec<World> = p.worlds(frame.kernel()).collect();
    assert_eq!(worlds.len(), 560, "the frame leaves 560 deals");
    let weight = q(1, 560);
    let start = |w: &World| Game {
        hands: Seat::ALL.map(|s| w.hand(s)),
        trick: plays[12..].to_vec(),
        leader: Seat::S0,
        banked: frame.banked(),
    };
    let atoms: Vec<Atom> =
        worlds.iter().enumerate().flat_map(|(k, w)| [(2 * k, false, start(w)), (2 * k + 1, true, start(w))]).collect();
    let need = frame.contract().threshold();
    let mut choices: Vec<Choice<Domino, Vec<Domino>>> = Vec::new();
    let mut plays: Vec<Vec<Play>> = Vec::new();
    fn mass(atoms: &[Atom], weight: &Q) -> Affine {
        let high = atoms.iter().filter(|(_, h, _)| *h).count() as i64;
        let low = atoms.len() as i64 - high;
        Affine::new(weight * Q::from_integer(low.into()), weight * Q::from_integer(high.into()))
    }
    fn rec(
        p: &Physics,
        atoms: Vec<Atom>,
        weight: &Q,
        need: u32,
        choices: &mut Vec<Choice<Domino, Vec<Domino>>>,
        plays: &mut Vec<Vec<Play>>,
    ) -> usize {
        let menu: Vec<Domino> = atoms[0].2.legal(p).iter().collect();
        let i = choices.len();
        choices.push(Choice { menu: menu.clone(), mass: mass(&atoms, weight), payoff: vec![], blocks: vec![] });
        plays.push(Vec::new());
        let (mut payoff, mut blocks, mut over) = (Vec::new(), Vec::new(), Vec::new());
        for &a in &menu {
            let mut groups: BTreeMap<Vec<Domino>, Vec<Atom>> = BTreeMap::new();
            let mut made: Vec<Atom> = Vec::new();
            let mut ends = false;
            for (k, high, g) in &atoms {
                let (g2, seen) = advance(p, g.play(p, a), *high);
                if g2.done() {
                    ends = true;
                    if g2.banked[team(VIEWER)] >= need {
                        made.push((*k, *high, g2));
                    }
                } else {
                    groups.entry(seen).or_default().push((*k, *high, g2));
                }
            }
            payoff.push(mass(&made, weight));
            let mut next = Vec::new();
            for (seen, sub) in groups {
                let n = rec(p, sub, weight, need, choices, plays);
                next.push(n);
                blocks.push(Block { play: a, seen, next: n });
            }
            assert!(!ends || next.is_empty(), "a play ends the course in every world or in none");
            over.push(if ends { Play::Ends(made.iter().map(|(k, _, _)| *k).collect()) } else { Play::Continues(next) });
        }
        choices[i].payoff = payoff;
        choices[i].blocks = blocks;
        plays[i] = over;
        i
    }
    let root = rec(&p, atoms, &weight, need, &mut choices, &mut plays);
    let local = choices[root]
        .blocks
        .iter()
        .find(|b| b.play == tile("2-1") && b.seen == vec![tile("1-0"), tile("6-2"), tile("5-2")])
        .expect("the note's local history")
        .next;
    let nodes = choices.len();
    Built { family: Family::new(choices).unwrap(), root, nodes, plays, local }
}

fn qs(v: &[(i64, i64)]) -> Vec<Q> {
    v.iter().map(|&(n, d)| q(n, d)).collect()
}

/// The fewest fixed plans whose best is the root's optimum at every `p`: one per affine
/// piece of the envelope, each the plan optimal at the piece's midpoint, checked to meet
/// the envelope at both ends of its piece.
fn plans(family: &Family<Domino, Vec<Domino>>, root: usize) -> usize {
    let envelope = family.envelope(root);
    let two = Q::from_integer(2.into());
    for w in envelope.knots().windows(2) {
        let plan = family.plan_at(root, &((&w[0].0 + &w[1].0) / &two));
        assert_eq!((plan.at(&w[0].0), plan.at(&w[1].0)), (w[0].1.clone(), w[1].1.clone()));
    }
    envelope.pieces()
}

#[test]
fn the_whole_ending_has_the_notes_thresholds_and_its_767_state_controller() {
    let Built { family, root, nodes, .. } = build();
    assert_eq!(nodes, 24_089);
    let all: Vec<usize> = (0..family.len()).collect();
    let thresholds = family.thresholds(&all);
    assert_eq!(
        thresholds,
        qs(&[
            (1, 16),
            (1, 7),
            (2, 11),
            (1, 5),
            (1, 4),
            (1, 3),
            (2, 5),
            (5, 12),
            (3, 7),
            (1, 2),
            (7, 13),
            (4, 7),
            (2, 3),
            (8, 11)
        ])
    );
    // Each threshold has a history whose maximizing sets on its two sides are disjoint:
    // no play there stays optimal across it.
    assert_eq!(family.witnessed(&all, &thresholds), thresholds);
    let reps = representatives(&thresholds);
    assert_eq!(reps.len(), 31);
    // The root move is 2-1 for every parameter; the root optimum has five affine pieces.
    assert!(reps.iter().all(|r| family.best_at(root, r) == vec![tile("2-1")]));
    let envelope = family.envelope(root);
    assert_eq!(envelope.pieces(), 5);
    assert_eq!(
        envelope.knots().iter().map(|(x, _)| x.clone()).collect::<Vec<_>>(),
        qs(&[(0, 1), (1, 16), (1, 5), (1, 3), (1, 2), (1, 1)])
    );
    // Ex ante five fixed plans are enough and are needed. Each plan's value is affine, so
    // none spans a knot of the convex envelope; the plan optimal inside a piece is
    // optimal across it.
    assert_eq!(plans(&family, root), 5);
    let (course, roots) = family.controller(root, &reps, Reach::Every).unwrap();
    assert_eq!(course.decisions().len(), 722_863);
    let raw = course.quotient(Interface::Raw).unwrap();
    assert_eq!(raw.states, 767);
    assert_eq!(raw.ladder_of(&roots), vec![1, 18, 31, 31]);
    assert_eq!(raw.ladder(), vec![47, 545, 767, 767]);
    let shape = course.quotient(Interface::Shape).unwrap();
    assert_eq!((shape.states, shape.ladder_of(&roots)), (445, vec![1, 18, 31, 31]));
    let fork = course.quotient(Interface::Fork).unwrap();
    assert_eq!((fork.states, fork.ladder_of(&roots)), (76, vec![1, 3, 22, 22]));
}

#[test]
fn with_two_one_committed_the_controller_has_302_states() {
    let Built { family, root, .. } = build();
    let family = family.restricted(root, &[tile("2-1")]).unwrap();
    let later: Vec<usize> = family.reachable(&[root]).into_iter().filter(|&i| i != root).collect();
    assert_eq!(later.len() + 1, 5_935);
    let thresholds = family.thresholds(&later);
    assert_eq!(thresholds.len(), 10);
    assert_eq!(family.witnessed(&later, &thresholds), thresholds);
    let reps = representatives(&thresholds);
    assert_eq!(reps.len(), 23);
    assert_eq!(plans(&family, root), 5);
    let (course, roots) = family.controller(root, &reps, Reach::Every).unwrap();
    let raw = course.quotient(Interface::Raw).unwrap();
    assert_eq!(raw.states, 302);
    assert_eq!(raw.ladder_of(&roots), vec![1, 10, 23, 23]);
}

/// A two-play decision whose plays' values cross at 1/5 (the note's gap
/// (5p - 1)/(1 + 18p), times the mass 1 + 18p): one threshold, disjoint sides, three
/// classes; the controller separates the classes in one block.
#[test]
fn a_single_crossing_gives_three_classes() {
    let a = |x: i64, y: i64| Affine::new(q(x, 1), q(y, 1));
    // Play 1 banks 1 - p + 6p, play 2 banks 2(1 - p) + 2p: equal at p = 1/5.
    let family = Family::new(vec![
        Choice {
            menu: vec![1, 2],
            mass: a(2, 19),
            payoff: vec![a(0, 0), a(0, 0)],
            blocks: vec![Block { play: 1, seen: 0u8, next: 1 }, Block { play: 2, seen: 0u8, next: 2 }],
        },
        Choice { menu: vec![9], mass: a(2, 19), payoff: vec![a(1, 6)], blocks: vec![] },
        Choice { menu: vec![9], mass: a(2, 19), payoff: vec![a(2, 2)], blocks: vec![] },
    ])
    .unwrap();
    let t = family.thresholds(&[0]);
    assert_eq!(t, vec![q(1, 5)]);
    assert_eq!(family.witnessed(&[0], &t), t);
    let reps = representatives(&t);
    assert_eq!(reps, vec![q(0, 1), q(1, 10), q(1, 5), q(3, 5), q(1, 1)]);
    let best: Vec<Vec<u32>> = reps.iter().map(|r| family.best_at(0, r)).collect();
    assert_eq!(best, vec![vec![2], vec![2], vec![1, 2], vec![1], vec![1]]);
    let (course, roots) = family.controller(0, &reps, Reach::Every).unwrap();
    let raw = course.quotient(Interface::Raw).unwrap();
    assert_eq!(raw.ladder_of(&roots), vec![3, 3]);
    assert_eq!(family.envelope(0).pieces(), 2);
}

#[test]
fn a_block_that_does_not_partition_its_mass_is_refused() {
    let a = |x: i64, y: i64| Affine::new(q(x, 1), q(y, 1));
    let e = Family::new(vec![
        Choice {
            menu: vec![1u32],
            mass: a(2, 2),
            payoff: vec![a(0, 0)],
            blocks: vec![Block { play: 1, seen: 0u8, next: 1 }],
        },
        Choice { menu: vec![9], mass: a(1, 2), payoff: vec![a(1, 1)], blocks: vec![] },
    ])
    .err()
    .unwrap();
    assert!(e.0.contains("partition"), "{}", e.0);
}

/// The note's two contrasts, generalized: at every history, with a free belief over the
/// 1,120 deal-and-type atoms, the fewest linear coordinates that preserve every numerical
/// comparison of the viewer's programs. Two at the note's local history (its `x` and
/// `y`); 168 at the root; at most five at the second decision and one at the third.
#[test]
fn how_many_numbers_the_viewer_must_keep() {
    let Built { root, plays, local, .. } = build();
    let order: Vec<usize> = (0..plays.len()).rev().collect();
    let dim = dimensions(&plays, &order);
    assert_eq!(dim[local], 2);
    assert_eq!(dim[root], 168);
    // By depth (the viewer's decision number): 1, 160, 480 and 23,448 histories.
    let mut depth = vec![0usize; plays.len()];
    for i in 0..plays.len() {
        for play in &plays[i] {
            if let Play::Continues(next) = play {
                for &n in next {
                    depth[n] = depth[i] + 1;
                }
            }
        }
    }
    let at = |d: usize| (0..plays.len()).filter(|&i| depth[i] == d).map(|i| dim[i]).collect::<Vec<_>>();
    let tally = |v: Vec<usize>| {
        let mut t = BTreeMap::new();
        for x in v {
            *t.entry(x).or_insert(0usize) += 1;
        }
        t.into_iter().collect::<Vec<_>>()
    };
    assert_eq!(tally(at(1)), vec![(0, 8), (1, 10), (2, 83), (3, 43), (4, 14), (5, 2)]);
    assert_eq!(tally(at(2)), vec![(0, 65), (1, 415)]);
    assert_eq!(tally(at(3)), vec![(0, 23_448)]);
}

/// Good play meets four of the fourteen thresholds, and they are the knots of the root's
/// optimum. Kept to the histories the plan reaches, the 31 classes fall to 11, which are
/// the envelope's five pieces, its four knots and the two pure partners (at `p` = 0 or 1
/// one type of partner has no mass, so fewer histories live). The plan's own play falls
/// to 7: the five plans and the two ends. The other ten thresholds move the best play
/// only after a play the plan never makes.
#[test]
fn good_play_meets_only_the_knots_of_the_optimum() {
    let Built { family, root, .. } = build();
    let all: Vec<usize> = (0..family.len()).collect();
    let reps = representatives(&family.thresholds(&all));
    let knots: Vec<Q> = family.envelope(root).knots().iter().map(|(x, _)| x.clone()).collect();
    let piece = |p: &Q| knots.iter().position(|k| p <= k).unwrap();
    let classes = |reach| {
        let (course, roots) = family.controller(root, &reps, reach).unwrap();
        let quotient = course.quotient(Interface::Raw).unwrap();
        let mut seen: Vec<usize> = Vec::new();
        let named: Vec<usize> = roots
            .iter()
            .map(|r| {
                let c = quotient.class[*r];
                seen.iter().position(|x| *x == c).unwrap_or_else(|| {
                    seen.push(c);
                    seen.len() - 1
                })
            })
            .collect();
        (quotient.states, quotient.ladder_of(&roots), named)
    };
    // On the path: one class per open piece and per knot, so a representative's class is
    // fixed by its piece and by whether it is a knot.
    let (states, ladder, named) = classes(Reach::Path);
    assert_eq!((states, ladder), (142, vec![1, 10, 11, 11]));
    let expected: Vec<(usize, bool)> = reps.iter().map(|p| (piece(p), knots.contains(p))).collect();
    for (a, b) in (0..reps.len()).flat_map(|a| (0..reps.len()).map(move |b| (a, b))) {
        assert_eq!(named[a] == named[b], expected[a] == expected[b], "{} and {}", reps[a], reps[b]);
    }
    // The plan: one class per open piece and per end; each knot joins a piece beside it
    // (which one is the tie-break's).
    let (states, ladder, named) = classes(Reach::Plan);
    assert_eq!((states, ladder), (132, vec![1, 5, 7, 7]));
    let open = |k: usize| -> Vec<usize> {
        (0..reps.len()).filter(|&r| piece(&reps[r]) == k && !knots.contains(&reps[r])).map(|r| named[r]).collect()
    };
    let pieces: Vec<usize> = (1..knots.len())
        .map(|k| {
            let v = open(k);
            assert!(v.iter().all(|c| *c == v[0]), "piece {k} is one class");
            v[0]
        })
        .collect();
    let (first, last) = (named[0], named[reps.len() - 1]);
    let mut distinct = pieces.clone();
    distinct.extend([first, last]);
    distinct.sort_unstable();
    distinct.dedup();
    assert_eq!(distinct.len(), 7);
    for k in 1..knots.len() - 1 {
        let c = named[reps.iter().position(|p| *p == knots[k]).unwrap()];
        assert!(c == pieces[k - 1] || c == pieces[k], "knot {} joins a piece beside it", knots[k]);
    }
}

/// What Lean takes as given (`scenario/lean/Stab/Mixing.lean`), derived here from the
/// rules: the note's local history has mass (1 + 18p)/560, and after 3-1 the gap between
/// 4-1 and 5-1 is (5p - 1)/(1 + 18p). And the note's 0.08 and 0.12 share a cell of the
/// atlas: every decision has the same maximizing set at both.
#[test]
fn the_notes_gap_and_cell_are_the_fixtures() {
    let Built { family, local, .. } = build();
    let c = family.choice(local);
    assert_eq!(c.menu, tiles(&["3-1", "4-1", "5-1"]).iter().collect::<Vec<_>>());
    assert_eq!(c.mass, Affine::new(q(1, 560), q(19, 560)));
    let after = c.blocks.iter().find(|b| b.play == tile("3-1")).unwrap().next;
    assert_eq!(family.choice(after).menu, vec![tile("4-1"), tile("5-1")]);
    // 4-1 makes on 11 High deals; 5-1 on one deal against either partner and 6 more
    // High: (11p - (1 + 6p))/560 = (5p - 1)/560, over the mass (1 + 18p)/560.
    let v = family.value(after);
    assert_eq!(v[0], Piecewise::affine(&Affine::new(q(0, 1), q(11, 560))));
    assert_eq!(v[1], Piecewise::affine(&Affine::new(q(1, 560), q(7, 560))));
    let (a, b) = (q(2, 25), q(3, 25));
    assert!((0..family.len()).all(|i| family.best_at(i, &a) == family.best_at(i, &b)));
    assert!((0..family.len()).any(|i| family.best_at(i, &a) != family.best_at(i, &q(1, 5))));
}
