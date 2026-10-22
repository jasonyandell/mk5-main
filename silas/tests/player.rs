// SPDX-FileCopyrightText: Jason Yandell
// SPDX-License-Identifier: CC0-1.0

//! The exact player (`silas::player`): its course builder against the note's ending, its
//! values against a second construction written apart from it (`player::recount`), its
//! re-planning against its own plan (Bayes and Bellman), and its determinism.

mod common;

use common::{Rng, physics, tile, tiles};
use num_rational::BigRational;
use num_traits::{One, Zero};
use silas::family::{Affine, Reach, representatives};
use silas::player::{Belief, Budget, Library, Root, SeatLaw, Settle, Solved, TableType, play_hand, recount, solve};
use silas::quotient::Interface;
use silas::{Auction, Bid, Domino, DominoSet, Frame, Seat, World};

type Q = BigRational;

fn q(n: i64, d: i64) -> Q {
    Q::new(n.into(), d.into())
}

const BUDGET: Budget = Budget { reads: 100_000_000 };

/// The note's ending (`tests/controller.rs`): seat 3 at 31 no-trump before its trick-4 play.
fn note_frame() -> Frame {
    let plays =
        ["6-3", "6-1", "6-4", "6-0", "0-0", "2-2", "5-0", "2-0", "4-3", "4-2", "4-0", "5-4", "1-1", "3-0", "3-3"]
            .iter()
            .map(|s| tile(s))
            .collect();
    let dealt = tiles(&["6-3", "5-0", "4-0", "2-1", "5-1", "4-1", "3-1"]);
    Frame::new(Auction::won_by(Seat::S3, Bid::Points(31)).unwrap(), physics("NT"), Seat::S3, dealt, plays).unwrap()
}

/// The note's family: the partner (seat 1) Low with probability 1 - p, High with p; the
/// opponents low.
fn note_library() -> Library {
    let low = TableType::all(SeatLaw::Lowest);
    let mut high = TableType::all(SeatLaw::Lowest);
    high.name = "partner-high".to_string();
    high.seats[Seat::S1.index()] = SeatLaw::Highest;
    Library::new(vec![low, high], vec![Affine::new(Q::one(), Q::zero()), Affine::new(Q::zero(), Q::one())]).unwrap()
}

fn uniform() -> Library {
    Library::one(TableType::all(SeatLaw::UniformLegal))
}

/// The second construction agrees with the player in every class: the record's mass, each
/// root tile's value, the maximizing set. Returns the classes checked.
fn recounted(frame: &Frame, library: &Library, root: Root, solved: &Solved) -> usize {
    let f = &solved.family;
    let reps =
        if library.constant() { vec![Q::zero()] } else { representatives(&f.thresholds(&f.reachable(&[solved.root]))) };
    let mut n = 0;
    for r in &reps {
        let mass = f.choice(solved.root).mass.at(r);
        if mass.is_zero() {
            continue;
        }
        let (m, v) = recount::values(frame, library, root, r).unwrap();
        assert_eq!(m, mass, "mass at {r}");
        assert_eq!(v.iter().map(|(d, _)| *d).collect::<Vec<_>>(), solved.menu());
        assert_eq!(v.into_iter().map(|(_, x)| x).collect::<Vec<_>>(), solved.values(r).unwrap(), "values at {r}");
        n += 1;
    }
    n
}

#[test]
fn the_players_course_is_the_notes_ending_and_its_767_state_controller() {
    let frame = note_frame();
    let library = note_library();
    let belief = Belief::new(&frame, &library, Root::FiberUniform).unwrap();
    assert_eq!((belief.worlds(), belief.atoms()), (560, 1_120));
    let solved = solve(&frame, &library, &belief, Settle::Hand, BUDGET).unwrap();
    let f = &solved.family;
    assert_eq!(f.len(), 24_089);
    let all: Vec<usize> = (0..f.len()).collect();
    let thresholds = f.thresholds(&all);
    assert_eq!(
        thresholds,
        [
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
        ]
        .map(|(n, d)| q(n, d))
    );
    let reps = representatives(&thresholds);
    assert_eq!(reps.len(), 31);
    assert!(reps.iter().all(|p| solved.best(p) == vec![tile("2-1")]));
    let (course, roots) = f.controller(solved.root, &reps, Reach::Every).unwrap();
    let raw = course.quotient(Interface::Raw).unwrap();
    assert_eq!((raw.states, raw.ladder_of(&roots)), (767, vec![1, 18, 31, 31]));
    let fork = course.quotient(Interface::Fork).unwrap();
    assert_eq!((fork.states, fork.ladder_of(&roots)), (76, vec![1, 3, 22, 22]));
    // The second construction agrees in all 31 classes.
    assert_eq!(recounted(&frame, &library, Root::FiberUniform, &solved), 31);
}

/// Settling as soon as the contract is decided changes the course, never a value.
#[test]
fn settling_when_decided_keeps_every_value() {
    let frame = note_frame();
    let library = note_library();
    let belief = Belief::new(&frame, &library, Root::FiberUniform).unwrap();
    let hand = solve(&frame, &library, &belief, Settle::Hand, BUDGET).unwrap();
    let decided = solve(&frame, &library, &belief, Settle::Decided, BUDGET).unwrap();
    assert!(decided.family.len() < hand.family.len());
    assert_eq!(decided.family.value(decided.root), hand.family.value(hand.root));
}

/// Under the exact posterior the note's record is impossible for both of the note's types
/// (no other seat played only its lowest, nor the partner only its highest), so the player
/// refuses rather than act on an empty belief. Under the uniform-legal kernel every world
/// of the fiber lives, and the posterior is not the fiber's uniform measure.
#[test]
fn the_posterior_reads_the_record() {
    let frame = note_frame();
    let e = Belief::new(&frame, &note_library(), Root::Posterior).err().unwrap();
    assert!(e.0.contains("probability zero"), "{}", e.0);
    let library = uniform();
    let belief = Belief::new(&frame, &library, Root::Posterior).unwrap();
    assert_eq!((belief.worlds(), belief.atoms()), (560, 560));
    assert_eq!(belief.mass(), Affine::new(q(1, 384), q(1, 384)));
    let exact = solve(&frame, &library, &belief, Settle::Decided, BUDGET).unwrap();
    let flat = Belief::new(&frame, &library, Root::FiberUniform).unwrap();
    let walt_kind = solve(&frame, &library, &flat, Settle::Decided, BUDGET).unwrap();
    let p = Q::zero();
    assert_eq!(
        exact.values(&p).unwrap(),
        vec![q(150_523, 311_040), q(24_533, 51_840), q(31_717, 103_680), q(153_337, 311_040)]
    );
    assert_eq!(
        walt_kind.values(&p).unwrap(),
        vec![q(415, 896), q(54_557, 120_960), q(75_883, 241_920), q(14_293, 30_240)]
    );
    assert_eq!((exact.best(&p), walt_kind.best(&p)), (vec![tile("5-1")], vec![tile("5-1")]));
    assert_eq!((exact.counts.choices, exact.counts.reads), (362_446, 1_735_486));
    assert_eq!(recounted(&frame, &library, Root::Posterior, &exact), 1);
    assert_eq!(recounted(&frame, &library, Root::FiberUniform, &walt_kind), 1);
}

/// The full deal behind a frame: its first world, with every played tile given back.
fn truth(frame: &Frame) -> World {
    let w = frame.physics().worlds(frame.kernel()).next().unwrap();
    let mut hands = Seat::ALL.map(|s| w.hand(s));
    for &(s, d) in frame.history() {
        hands[s.index()].insert(d);
    }
    World::deal(hands).unwrap()
}

/// Every decision after the first is the previous course's child: the fresh posterior's
/// mass is the child's mass and the fresh maximizing set is the child's (Bayes' rule and
/// Bellman's, checked at every turn). The belief that ignores the likelihood does not
/// have this property in general.
#[test]
fn replanning_from_the_posterior_is_the_plan() {
    let frame = note_frame();
    let library = uniform();
    let world = truth(&frame);
    let p = Q::zero();
    for seed in 0..6 {
        let mut walls = 0;
        let played =
            play_hand(&frame, &world, &library, 0, &p, Root::Posterior, Settle::Decided, BUDGET, seed, |_, _, _| {
                walls += 1;
                Ok(())
            })
            .unwrap();
        assert_eq!(played.history.len(), Domino::COUNT);
        assert!(played.decisions >= 1 && played.decisions == walls);
        assert_eq!(played.consistent, played.decisions - 1, "seed {seed}");
    }
}

/// The same frame, library and prior give the same values, sets and counts.
#[test]
fn the_player_is_deterministic() {
    let frame = note_frame();
    let library = note_library();
    let run = || {
        let b = Belief::new(&frame, &library, Root::FiberUniform).unwrap();
        let s = solve(&frame, &library, &b, Settle::Decided, BUDGET).unwrap();
        (s.family.value(s.root).to_vec(), s.counts)
    };
    assert_eq!(run(), run());
}

/// A budget of kernel reads refuses; it never truncates.
#[test]
fn a_budget_refuses() {
    let frame = note_frame();
    let library = uniform();
    let b = Belief::new(&frame, &library, Root::Posterior).unwrap();
    let e = solve(&frame, &library, &b, Settle::Decided, Budget { reads: 10_000 }).err().unwrap();
    assert!(e.0.contains("kernel reads"), "{}", e.0);
}

/// Seeded deals played by every seat uniformly at random to two or three tricks left; at
/// each, the player and the second construction agree: under the exact posterior of the
/// uniform-legal kernel, and under the note's two-type family on the fiber, in every class.
#[test]
fn the_second_construction_agrees_on_random_endgames() {
    let decls = ["NT", "P5", "P0", "DT", "P3"];
    let mut rng = Rng(42);
    let (mut positions, mut classes) = (0, 0);
    for k in 0..20 {
        let deck = rng.shuffled();
        let hands: [DominoSet; 4] = std::array::from_fn(|s| deck[7 * s..7 * s + 7].iter().copied().collect());
        let p = physics(decls[k % decls.len()]);
        let declarer = Seat::ALL[k % 4];
        let auction = Auction::won_by(declarer, Bid::Points(30 + (k % 5) as u32)).unwrap();
        let n = if k % 2 == 0 { 16 } else { 20 };
        let mut frame = Frame::new(auction.clone(), p.clone(), declarer, hands[declarer.index()], vec![]).unwrap();
        while frame.plays().len() < n {
            let s = frame.next_actor().unwrap();
            let legal: Vec<Domino> =
                p.legal_plays(hands[s.index()].difference(frame.played()), frame.lead()).iter().collect();
            let d = legal[rng.below(legal.len())];
            frame = frame.after(d).unwrap();
        }
        let viewer = frame.next_actor().unwrap();
        let frame = Frame::new(auction, p, viewer, hands[viewer.index()], frame.plays().to_vec()).unwrap();
        for (library, root) in [(uniform(), Root::Posterior), (note_library(), Root::FiberUniform)] {
            let b = Belief::new(&frame, &library, root).unwrap();
            let s = solve(&frame, &library, &b, Settle::Decided, BUDGET).unwrap();
            classes += recounted(&frame, &library, root, &s);
        }
        positions += 1;
    }
    assert_eq!(positions, 20);
    assert!(classes >= 40, "{classes}");
}
