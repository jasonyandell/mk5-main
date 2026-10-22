// SPDX-FileCopyrightText: Jason Yandell
// SPDX-License-Identifier: CC0-1.0

//! The three constructions of the field epoch agree bit for bit: Silas's rules with the laws
//! as rationals, the tables as integers on the CPU, and the tables on Metal. Random deals
//! under all nine declarations, random positions at every trick and seat, every legal focal
//! action, every law and mixed tables, masses near the top of the arithmetic.

use silas::reference::Straight42;
use silas::{Domino, DominoSet, Physics, Seat};
use silas_metal::{Backend, Batch, Cpu, Gpu, Law, Particle, Reference, Segment, Tables, U256};

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

fn deal(rng: &mut Rng) -> [DominoSet; 4] {
    let mut tiles: Vec<Domino> = Domino::ALL.to_vec();
    for i in (1..tiles.len()).rev() {
        tiles.swap(i, rng.below(i + 1));
    }
    std::array::from_fn(|s| tiles[7 * s..7 * s + 7].iter().copied().collect())
}

/// A random position: a deal, `plays` uniformly random legal plays, and whose turn it is.
fn position(rng: &mut Rng, physics: &Physics, plays: usize) -> ([DominoSet; 4], Seat, Vec<Domino>, u8) {
    let mut hands = deal(rng);
    let mut leader = Seat::from_index(rng.below(4)).unwrap();
    let mut trick = Vec::new();
    let mut done = 0u8;
    for _ in 0..plays {
        let actor = leader.plus(trick.len());
        let legal: Vec<Domino> = physics.legal_plays(hands[actor.index()], trick.first().copied()).iter().collect();
        let tile = legal[rng.below(legal.len())];
        hands[actor.index()].remove(tile);
        trick.push(tile);
        if trick.len() == 4 {
            leader = physics.trick_winner(leader, trick.clone().try_into().unwrap());
            trick.clear();
            done += 1;
        }
    }
    (hands, leader, trick, done)
}

/// Other worlds with the same public state: the hidden tiles dealt again, sizes kept.
fn reshuffle(rng: &mut Rng, hands: &[DominoSet; 4], focal: Seat) -> [u32; 4] {
    let mut pool: Vec<Domino> =
        Seat::ALL.iter().filter(|s| **s != focal).flat_map(|s| hands[s.index()].iter()).collect();
    for i in (1..pool.len()).rev() {
        pool.swap(i, rng.below(i + 1));
    }
    let mut out = [0u32; 4];
    let mut at = 0;
    for s in Seat::ALL {
        if s == focal {
            continue;
        }
        let k = hands[s.index()].len();
        out[s.index()] = pool[at..at + k].iter().copied().collect::<DominoSet>().bits();
        at += k;
    }
    out
}

fn batch_for(rng: &mut Rng, physics: &Physics, kinds: usize) -> (Vec<Segment>, Vec<Particle>, Vec<(u32, u32)>) {
    let (mut segments, mut particles, mut items) = (Vec::new(), Vec::new(), Vec::new());
    for _ in 0..12 {
        let plays = rng.below(26);
        let (hands, leader, trick, done) = position(rng, physics, plays);
        let focal = leader.plus(trick.len());
        let first = particles.len() as u32;
        for w in 0..1 + rng.below(6) {
            let hidden = if w == 0 { hands.map(|h| h.bits()) } else { reshuffle(rng, &hands, focal) };
            // Masses up to 2^160, so six plays at 840 stay below 2^256.
            let mass = U256(std::array::from_fn(|i| if i < 5 { rng.next() as u32 } else { 0 }));
            particles.push(Particle { hands: hidden, kind: rng.below(kinds) as u32, world: w as u32, mass });
        }
        let last = particles.len() as u32;
        let mut t = [0u8; 4];
        for (i, d) in trick.iter().enumerate() {
            t[i] = d.index() as u8;
        }
        for action in physics.legal_plays(hands[focal.index()], trick.first().copied()).iter() {
            let seg = segments.len() as u32;
            segments.push(Segment {
                focal: focal.index() as u8,
                leader: leader.index() as u8,
                n: trick.len() as u8,
                trick: t,
                done,
                focal_hand: hands[focal.index()].bits(),
                action: action.index() as u8,
            });
            items.extend((first..last).map(|p| (seg, p)));
        }
    }
    (segments, particles, items)
}

#[test]
fn three_constructions_agree() {
    let mut rng = Rng(0x5eed_0042);
    let mut gpu = Gpu::new().expect("a Metal device");
    let mut checked = 0usize;
    for (round, decl) in Straight42::ALL.iter().enumerate() {
        let physics = decl.physics();
        let mut kinds: Vec<[Law; 4]> = Law::ALL.iter().map(|l| [*l; 4]).collect();
        for _ in 0..4 {
            kinds.push(std::array::from_fn(|_| Law::ALL[rng.below(5)]));
        }
        let tables = Tables::new(&physics, &kinds);
        for _ in 0..3 {
            let (segments, particles, items) = batch_for(&mut rng, &physics, kinds.len());
            let batch = Batch { segments: &segments, particles: &particles, items: &items };
            let reference = Reference { physics: physics.clone(), kinds: kinds.clone() }.epoch(&tables, batch).unwrap();
            let cpu = Cpu { threads: 4 }.epoch(&tables, batch).unwrap();
            let metal = gpu.epoch(&tables, batch).unwrap();
            assert!(!reference.is_empty());
            assert_eq!(cpu, reference, "cpu mirror vs reference, declaration {round}");
            assert_eq!(metal, reference, "metal vs reference, declaration {round}");
            checked += reference.len();
        }
    }
    assert!(checked > 10_000, "only {checked} children checked");
    eprintln!("{checked} children agree across reference, cpu and metal");
}

#[test]
fn mass_is_conserved_per_work_item() {
    // Every law sums to one, so a work item's lines, each lifted from its own exponent r to
    // the common 840^6, sum to the parent's mass times 840^6. Exact big integers.
    use num_bigint::BigUint;
    let mut rng = Rng(7);
    let physics = Straight42::ALL[3].physics();
    let kinds: Vec<[Law; 4]> = Law::ALL.iter().map(|l| [*l; 4]).collect();
    let tables = Tables::new(&physics, &kinds);
    let (segments, particles, items) = batch_for(&mut rng, &physics, kinds.len());
    let batch = Batch { segments: &segments, particles: &particles, items: &items };
    let children = Cpu { threads: 1 }.epoch(&tables, batch).unwrap();
    for (item, (_, part)) in items.iter().enumerate() {
        let parent = particles[*part as usize].mass.to_biguint();
        let mut total = BigUint::from(0u32);
        for c in children.iter().filter(|c| c.item == item as u32) {
            let r = c.seen().len() as u32;
            total += c.mass.to_biguint() * BigUint::from(840u32).pow(6 - r);
        }
        assert_eq!(total, parent * BigUint::from(840u32).pow(6), "item {item}");
    }
}

#[test]
fn an_illegal_action_refuses_the_batch() {
    let physics = Straight42::ALL[0].physics();
    let kinds = vec![[Law::Uniform; 4]];
    let tables = Tables::new(&physics, &kinds);
    let hands = deal(&mut Rng(1));
    let off = Domino::ALL.iter().find(|d| !hands[0].contains(**d)).unwrap();
    let seg = Segment {
        focal: 0,
        leader: 0,
        n: 0,
        trick: [0; 4],
        done: 0,
        focal_hand: hands[0].bits(),
        action: off.index() as u8,
    };
    let p = Particle { hands: hands.map(|h| h.bits()), kind: 0, world: 0, mass: U256::ONE };
    let batch = Batch { segments: &[seg], particles: &[p], items: &[(0, 0)] };
    assert!(Cpu { threads: 1 }.epoch(&tables, batch).is_err());
    assert!(Gpu::new().unwrap().epoch(&tables, batch).is_err());
    assert!(Reference { physics, kinds }.epoch(&tables, batch).is_err());
}
