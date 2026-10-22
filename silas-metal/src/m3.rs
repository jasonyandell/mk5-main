// SPDX-FileCopyrightText: Jason Yandell
// SPDX-License-Identifier: CC0-1.0

//! The M3 gate (texas-42 `walt/math/gpu_native_trick1_m3_rebrief_v0.1.md`, frozen there, run
//! here): one complete grade-4 carrier, every root, two arms, the lawful solve against its
//! world-split control, the GPU engine against an independent CPU oracle.
//!
//! The carrier is receipt hand h8 before trick 4: S1 declared fives (P5) at 30 and leads,
//! holding 2-1 3-1 3-3 5-5; twelve hidden tiles; the record replayed by Silas's [`Frame`]
//! (which derives the voids). The belief is uniform over the hidden worlds the record allows
//! (texas-42's analysis profile: feasibility only, no likelihood reweighting of the past),
//! and from the root every other seat plays uniformly among its legal tiles.
//!
//! The oracle is a second construction, sharing only Silas's rules: a depth-first recursion
//! over information nodes that keeps each node's worlds with their probabilities as `u128`
//! numerators at scale 420 per field play (the engine uses 840 and 256-bit masses), groups
//! a node's lines by the field plays seen in a `BTreeMap`, and sums then maximizes.
//!
//! STRATEGY-FUSION FENCE: [`oracle`] with `split` true solves each world alone (treatment C,
//! clairvoyant): the sanctioned measurement of fusion, a negative control, never a value.

use std::collections::BTreeMap;

use silas::reference::Straight42;
use silas::{Auction, Bid, ContextSet, Domino, DominoSet, Frame, Physics, Seat};

use crate::epoch::{Law, Particle};
use crate::net::{Root, Tally};
use crate::wide::U256;

pub fn tile(s: &str) -> Domino {
    let (a, b) = s.split_at(1);
    format!("{a}-{b}").parse().expect("a domino")
}

/// The h8 carrier before trick 4, from Silas's replay.
pub struct Carrier {
    pub frame: Frame,
    pub physics: Physics,
    pub worlds: Vec<[DominoSet; Seat::COUNT]>,
}

pub const ROOTS: [&str; 4] = ["21", "31", "33", "55"];

pub fn carrier() -> Carrier {
    let physics = Straight42::new("P5".parse().expect("a declaration")).physics();
    let auction = Auction::won_by(Seat::S1, Bid::number(30).expect("a bid")).expect("an auction");
    let history: Vec<(Seat, Domino)> = [
        (Seat::S1, "52"),
        (Seat::S2, "63"),
        (Seat::S3, "51"),
        (Seat::S0, "65"),
        (Seat::S0, "42"),
        (Seat::S1, "54"),
        (Seat::S2, "44"),
        (Seat::S3, "32"),
        (Seat::S1, "11"),
        (Seat::S2, "22"),
        (Seat::S3, "10"),
        (Seat::S0, "61"),
    ]
    .iter()
    .map(|(s, t)| (*s, tile(t)))
    .collect();
    let hand: DominoSet = ROOTS.iter().map(|t| tile(t)).collect();
    let frame = Frame::from_history(auction, physics.clone(), Seat::S1, hand, &history).expect("the h8 record replays");
    let worlds = frame.worlds().map(|w| Seat::ALL.map(|s| w.hand(s))).collect();
    Carrier { frame, physics, worlds }
}

impl Carrier {
    pub fn voids(&self, seat: Seat) -> ContextSet {
        self.frame.shown_voids(seat)
    }

    /// The solve's root for one physical root action: every world at mass one, every other
    /// seat uniform.
    pub fn root(&self, action: Domino) -> Root {
        let particles = self
            .worlds
            .iter()
            .enumerate()
            .map(|(w, hands)| Particle { hands: hands.map(|h| h.bits()), kind: 0, world: w as u32, mass: U256::ONE })
            .collect();
        Root {
            physics: self.physics.clone(),
            kinds: vec![[Law::Uniform; Seat::COUNT]],
            focal: Seat::S1,
            leader: self.frame.leader(),
            trick: self.frame.prefix().to_vec(),
            done: 3,
            focal_hand: self.frame.kernel().viewer_hand(),
            particles,
            action: Some(action),
        }
    }
}

/// The two arms. A: the future trick count of the declaring team (the differential is
/// `2k - 4`). B: the declaring team makes 30, the defenders' future points at most 11.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Arm {
    Tricks,
    Makes30,
}

impl Arm {
    pub fn utility(self, t: &Tally) -> u32 {
        let declarers = Seat::S1.team().index();
        match self {
            Arm::Tricks => t.tricks[declarers],
            Arm::Makes30 => u32::from(t.points[1 - declarers] <= 11),
        }
    }
}

// ---------------------------------------------------------------------------------------
// The oracle.

/// The oracle's answer: the root value as a numerator over `1200 * 420^12` (worlds times
/// scale), and the focal information nodes it visited below the root.
pub struct OracleAnswer {
    pub value: u128,
    pub nodes: u64,
}

struct Oracle<'a> {
    physics: &'a Physics,
    arm: Arm,
    nodes: u64,
}

#[derive(Clone)]
struct Line {
    hands: [DominoSet; Seat::COUNT],
    p: u128,
}

#[derive(Clone)]
struct Pub {
    leader: Seat,
    trick: Vec<Domino>,
    done: u32,
    tally: Tally,
}

impl Oracle<'_> {
    fn play(&self, s: &mut Pub, line: &mut Line, seat: Seat, d: Domino) {
        line.hands[seat.index()].remove(d);
        s.trick.push(d);
        if s.trick.len() == 4 {
            let four: [Domino; 4] = [s.trick[0], s.trick[1], s.trick[2], s.trick[3]];
            let w = self.physics.trick_winner(s.leader, four);
            s.tally.points[w.team().index()] += self.physics.trick_points(four);
            s.tally.tricks[w.team().index()] += 1;
            s.leader = w;
            s.trick.clear();
            s.done += 1;
        }
    }

    /// The field's lines from `s` until S1 must act or the hand ends, each with its seen plays.
    fn field(&self, s: Pub, line: Line, seen: Vec<Domino>, out: &mut Vec<(Vec<Domino>, Pub, Line)>) {
        let actor = s.leader.plus(s.trick.len());
        if s.done == 7 || actor == Seat::S1 {
            out.push((seen, s, line));
            return;
        }
        let legal = self.physics.legal_plays(line.hands[actor.index()], s.trick.first().copied());
        let share = 420 / legal.len() as u128;
        for d in legal.iter() {
            let (mut s2, mut l2) = (s.clone(), line.clone());
            l2.p *= share;
            self.play(&mut s2, &mut l2, actor, d);
            let mut seen2 = seen.clone();
            seen2.push(d);
            self.field(s2, l2, seen2, out);
        }
    }

    /// The value of an information node: S1 to act, `lines` the worlds it cannot tell apart.
    fn node(&mut self, s: &Pub, lines: &[Line], only: Option<Domino>) -> u128 {
        let hand = lines[0].hands[Seat::S1.index()];
        let legal = self.physics.legal_plays(hand, s.trick.first().copied());
        let mut best = 0u128;
        for a in legal.iter().filter(|a| only.is_none_or(|o| o == *a)) {
            let mut groups: BTreeMap<Vec<Domino>, (Pub, Vec<Line>)> = BTreeMap::new();
            let mut terminal = 0u128;
            for line in lines {
                let (mut s2, mut l2) = (s.clone(), line.clone());
                self.play(&mut s2, &mut l2, Seat::S1, a);
                let mut out = Vec::new();
                self.field(s2, l2, Vec::new(), &mut out);
                for (seen, s3, l3) in out {
                    if s3.done == 7 {
                        terminal += l3.p * u128::from(self.arm.utility(&s3.tally));
                    } else {
                        groups.entry(seen).or_insert_with(|| (s3, Vec::new())).1.push(l3);
                    }
                }
            }
            let mut value = terminal;
            for (s3, group) in groups.values() {
                self.nodes += 1;
                value += self.node(s3, group, None);
            }
            best = best.max(value);
        }
        best
    }
}

/// The oracle's lawful value (or, `split`, the world-split control's) at one physical root.
pub fn oracle(carrier: &Carrier, root: Domino, arm: Arm, split: bool) -> OracleAnswer {
    let s = Pub {
        leader: carrier.frame.leader(),
        trick: carrier.frame.prefix().to_vec(),
        done: 3,
        tally: Tally::default(),
    };
    let lines: Vec<Line> = carrier.worlds.iter().map(|h| Line { hands: *h, p: 1 }).collect();
    let mut o = Oracle { physics: &carrier.physics, arm, nodes: 0 };
    let value = if split {
        lines.iter().map(|l| o.node(&s, std::slice::from_ref(l), Some(root))).sum()
    } else {
        o.node(&s, &lines, Some(root))
    };
    OracleAnswer { value, nodes: o.nodes }
}
