// SPDX-FileCopyrightText: Jason Yandell
// SPDX-License-Identifier: CC0-1.0

//! The lawful information-state solve on epochs, level by level.
//!
//! A level is the set of the focal seat's information states at one depth of its decisions.
//! For each state and each legal focal action, one segment; for each segment, an epoch over
//! the state's particles. The children of one segment are merged by what the focal seat
//! sees (the field plays, in order): the complete public record extends the parent's by the
//! action and those plays, so equal records can only come from the same parent and action
//! (perfect recall), and grouping inside the segment is grouping by the full record. A
//! group ends the hand or is one information state of the next level, carrying its
//! particles. After the deepest level, values back up: an action's value is its terminal
//! mass plus the values of its child states, summed; a state's value is the maximum over
//! its actions. Sum over what the seat sees, then max over its own plays, never a maximum
//! per world.
//!
//! STRATEGY-FUSION FENCE: [`Treatment::WorldSplit`] also groups by the particle's world
//! label, so each later focal choice sees the world (treatment C, clairvoyant). It is the
//! sanctioned measurement of fusion (C − H), a negative control; never a value or a choice.
//!
//! Values are exact: terminal masses times an integer utility, all at the one terminal
//! scale `W * 840^R` (every terminal line has made all `R` field plays from the root, and
//! the solve checks it).

use std::ops::Range;

use num_bigint::BigUint;
use silas::{Domino, DominoSet, Physics, Seat};

use crate::epoch::{Backend, Batch, Child, EpochError, Law, Particle, Segment, obs_tiles};
use crate::tables::Tables;
use crate::wide::U256;

/// What a terminal line banked from the root on, by team index.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Tally {
    pub points: [u32; 2],
    pub tricks: [u32; 2],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Treatment {
    /// One choice per information state (H).
    Lawful,
    /// One choice per (world, information state): the clairvoyant control (C).
    WorldSplit,
}

/// The position the solve starts from: the focal seat to act, everything public, and the
/// belief as particles.
#[derive(Clone, Debug)]
pub struct Root {
    pub physics: Physics,
    pub kinds: Vec<[Law; Seat::COUNT]>,
    pub focal: Seat,
    pub leader: Seat,
    pub trick: Vec<Domino>,
    pub done: u8,
    pub focal_hand: DominoSet,
    pub particles: Vec<Particle>,
    /// A fixed physical root action, or every legal one.
    pub action: Option<Domino>,
}

/// A solve's exact root values and its census.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Solved {
    /// Each root action and its value, `numerator / (W * 840^R)`.
    pub actions: Vec<(Domino, U256)>,
    /// `R`: the field plays from the root to the hand's end.
    pub field_plays: u32,
    /// Focal information states by level (level 0 is the root).
    pub states: Vec<u64>,
    /// Children emitted by the epochs (one per particle per line).
    pub arrivals: u64,
    /// Terminal groups (end-of-hand records).
    pub terminals: u64,
    /// Work items run (particle by segment).
    pub items: u64,
}

impl Solved {
    pub fn value(&self) -> U256 {
        self.actions.iter().map(|(_, v)| *v).max().unwrap_or(U256::ZERO)
    }
}

#[derive(Clone, Debug)]
struct State {
    leader: Seat,
    trick: Vec<Domino>,
    done: u8,
    focal_hand: DominoSet,
    tally: Tally,
    /// Field plays since the root.
    r: u32,
}

struct Key {
    state: State,
    parts: Range<usize>,
    parent: Option<(usize, usize)>,
}

struct Level {
    keys: Vec<Key>,
    particles: Vec<Particle>,
}

/// Why a solve was refused.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NetError {
    Epoch(EpochError),
    Overflow,
    /// The children of one record disagree with the host's replay.
    Replay(String),
}

impl From<EpochError> for NetError {
    fn from(e: EpochError) -> NetError {
        NetError::Epoch(e)
    }
}

impl std::fmt::Display for NetError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "the solve was refused: {self:?}")
    }
}

impl std::error::Error for NetError {}

fn add(a: U256, b: U256) -> Result<U256, NetError> {
    a.checked_add(b).ok_or(NetError::Overflow)
}

/// The state after the focal `action` and the field's `seen` plays, replayed with Silas's
/// rules; refused unless the seen plays end exactly where the focal seat must decide again
/// or the hand ends.
fn advance(physics: &Physics, focal: Seat, s: &State, action: Domino, seen: &[Domino]) -> Result<State, NetError> {
    let mut next = s.clone();
    next.focal_hand.remove(action);
    let play = |next: &mut State, tile: Domino| {
        next.trick.push(tile);
        if let Ok(four) = <[Domino; 4]>::try_from(next.trick.as_slice()) {
            let winner = physics.trick_winner(next.leader, four);
            let team = winner.team().index();
            next.tally.points[team] += physics.trick_points(four);
            next.tally.tricks[team] += 1;
            next.leader = winner;
            next.trick.clear();
            next.done += 1;
        }
    };
    play(&mut next, action);
    for (i, tile) in seen.iter().enumerate() {
        let actor = next.leader.plus(next.trick.len());
        if actor == focal || next.done == 7 {
            return Err(NetError::Replay(format!("play {i} of {seen:?} comes after the focal seat's turn")));
        }
        play(&mut next, *tile);
    }
    next.r += seen.len() as u32;
    if next.done < 7 && next.leader.plus(next.trick.len()) != focal {
        return Err(NetError::Replay(format!("{seen:?} stops before the focal seat's turn")));
    }
    Ok(next)
}

/// Solve from `root` with any epoch backend, under `utility` on the terminal tally.
pub fn solve(
    backend: &mut dyn Backend,
    root: &Root,
    treatment: Treatment,
    utility: &dyn Fn(&Tally) -> u32,
) -> Result<Solved, NetError> {
    let physics = &root.physics;
    let tables = Tables::new(physics, &root.kinds);
    let focal = root.focal;
    let state = State {
        leader: root.leader,
        trick: root.trick.clone(),
        done: root.done,
        focal_hand: root.focal_hand,
        tally: Tally::default(),
        r: 0,
    };
    let mut levels: Vec<Level> = Vec::new();
    // acc[level][key] = the value of each of the key's actions, accumulated.
    let mut acc: Vec<Vec<Vec<(Domino, U256)>>> = Vec::new();
    let mut next = Level {
        keys: vec![Key { state, parts: 0..root.particles.len(), parent: None }],
        particles: root.particles.clone(),
    };
    let (mut arrivals, mut terminals, mut items_run) = (0u64, 0u64, 0u64);
    let mut field_plays: Option<u32> = None;
    while !next.keys.is_empty() {
        let level = next;
        let depth = levels.len();
        let mut actions_of: Vec<Vec<(Domino, U256)>> = Vec::with_capacity(level.keys.len());
        let mut segments = Vec::new();
        let mut seg_key = Vec::new();
        let mut items = Vec::new();
        for (k, key) in level.keys.iter().enumerate() {
            let legal = physics.legal_plays(key.state.focal_hand, key.state.trick.first().copied());
            let actions: Vec<Domino> = match (depth, root.action) {
                (0, Some(a)) if legal.contains(a) => vec![a],
                (0, Some(a)) => return Err(NetError::Replay(format!("the root action {a} is not legal"))),
                _ => legal.iter().collect(),
            };
            for (slot, a) in actions.iter().enumerate() {
                let seg = segments.len() as u32;
                let mut trick = [0u8; 4];
                for (i, d) in key.state.trick.iter().enumerate() {
                    trick[i] = d.index() as u8;
                }
                segments.push(Segment {
                    focal: focal.index() as u8,
                    leader: key.state.leader.index() as u8,
                    n: key.state.trick.len() as u8,
                    trick,
                    done: key.state.done,
                    focal_hand: key.state.focal_hand.bits(),
                    action: a.index() as u8,
                });
                seg_key.push((k, slot));
                items.extend(key.parts.clone().map(|p| (seg, p as u32)));
            }
            actions_of.push(actions.into_iter().map(|a| (a, U256::ZERO)).collect());
        }
        items_run += items.len() as u64;
        let children =
            backend.epoch(&tables, Batch { segments: &segments, particles: &level.particles, items: &items })?;
        arrivals += children.len() as u64;
        next = Level { keys: Vec::new(), particles: Vec::new() };
        // Children arrive in item order, so each segment's children are contiguous.
        let mut start = 0;
        while start < children.len() {
            let seg = items[children[start].item as usize].0 as usize;
            let mut end = start;
            while end < children.len() && items[children[end].item as usize].0 as usize == seg {
                end += 1;
            }
            let (k, slot) = seg_key[seg];
            let action = actions_of[k][slot].0;
            let mut run: Vec<&Child> = children[start..end].iter().collect();
            match treatment {
                Treatment::Lawful => run.sort_by_key(|c| c.obs),
                Treatment::WorldSplit => run.sort_by_key(|c| (c.world, c.obs)),
            }
            let mut g = 0;
            while g < run.len() {
                let same = |c: &&Child| {
                    c.obs == run[g].obs && (treatment == Treatment::Lawful || c.world == run[g].world)
                };
                let h = g + run[g..].iter().take_while(|c| same(c)).count();
                let state = advance(physics, focal, &level.keys[k].state, action, &obs_tiles(run[g].obs))?;
                if state.done == 7 {
                    terminals += 1;
                    match field_plays {
                        None => field_plays = Some(state.r),
                        Some(r) if r != state.r => {
                            return Err(NetError::Replay(format!("terminal lines at {r} and {} field plays", state.r)));
                        }
                        Some(_) => {}
                    }
                    let u = utility(&state.tally);
                    let mut mass = U256::ZERO;
                    for c in &run[g..h] {
                        mass = add(mass, c.mass)?;
                    }
                    let value = mass.checked_mul_small(u).ok_or(NetError::Overflow)?;
                    actions_of[k][slot].1 = add(actions_of[k][slot].1, value)?;
                } else {
                    let first = next.particles.len();
                    next.particles.extend(run[g..h].iter().map(|c| Particle {
                        hands: c.hands,
                        kind: c.kind,
                        world: c.world,
                        mass: c.mass,
                    }));
                    next.keys.push(Key { state, parts: first..next.particles.len(), parent: Some((k, slot)) });
                }
                g = h;
            }
            start = end;
        }
        acc.push(actions_of);
        levels.push(Level { keys: level.keys, particles: Vec::new() });
    }
    // Back up: deepest level first; a key's value is the max over its actions, added to the
    // action of its parent that reached it.
    for depth in (1..levels.len()).rev() {
        let (upper, lower) = acc.split_at_mut(depth);
        for (k, key) in levels[depth].keys.iter().enumerate() {
            let v = lower[0][k].iter().map(|(_, v)| *v).max().unwrap_or(U256::ZERO);
            let (pk, slot) = key.parent.expect("a key below the root has a parent");
            let cell = &mut upper[depth - 1][pk][slot].1;
            *cell = add(*cell, v)?;
        }
    }
    Ok(Solved {
        actions: acc.swap_remove(0).swap_remove(0),
        field_plays: field_plays.unwrap_or(0),
        states: levels.iter().map(|l| l.keys.len() as u64).collect(),
        arrivals,
        terminals,
        items: items_run,
    })
}

/// The root's total mass `W`.
pub fn total_mass(root: &Root) -> BigUint {
    root.particles.iter().map(|p| p.mass.to_biguint()).sum()
}
