// SPDX-FileCopyrightText: Jason Yandell
// SPDX-License-Identifier: CC0-1.0

//! Fixtures shared by the integration tests. Expected answers in the tests are
//! built directly from hands and the reference rules, independently of the
//! query compiler, joins, projection and aggregation. Every frame is built
//! from a record.
#![allow(dead_code)]

pub mod fixture;
pub mod rules;

use std::num::NonZeroU64;

use num_bigint::BigInt;
use num_rational::BigRational;
use silas::reference::Straight42;
use silas::*;

/// A small deterministic generator (splitmix64), so every sampled position is
/// reproducible from its seed.
pub struct Rng(pub u64);

impl Rng {
    pub fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    pub fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }

    pub fn shuffled(&mut self) -> Vec<Domino> {
        let mut tiles = Domino::ALL.to_vec();
        for i in (1..tiles.len()).rev() {
            tiles.swap(i, self.below(i + 1));
        }
        tiles
    }
}

pub fn r(n: i64, d: i64) -> BigRational {
    BigRational::new(BigInt::from(n), BigInt::from(d))
}

pub fn w(n: u64, d: u64) -> Weight {
    Weight::ratio(n, NonZeroU64::new(d).unwrap())
}

pub fn tile(s: &str) -> Domino {
    s.parse().unwrap()
}

pub fn tiles(ss: &[&str]) -> DominoSet {
    ss.iter().map(|s| tile(s)).collect()
}

pub fn physics(decl: &str) -> Physics {
    decl.parse::<Straight42>().unwrap().physics()
}

/// The frame of a hand given with its contract alone: its record is the
/// shortest auction the declarer wins with the bid (`Auction::won_by`).
pub fn framed(contract: &Contract, viewer: Seat, dealt: DominoSet, plays: Vec<Domino>) -> Result<Frame> {
    let auction = Auction::won_by(contract.declarer(), contract.bid())?;
    Frame::new(auction, contract.declaration().clone(), viewer, dealt, plays)
}

/// [`framed`] from a scorekeeper's record.
pub fn framed_history(contract: &Contract, viewer: Seat, hand: DominoSet, history: &[(Seat, Domino)]) -> Result<Frame> {
    let auction = Auction::won_by(contract.declarer(), contract.bid())?;
    Frame::from_history(auction, contract.declaration().clone(), viewer, hand, history)
}

/// `declarer`'s contract at the bid the table calls `bid`, under `physics`.
pub fn contract(physics: &Physics, declarer: Seat, bid: u32) -> Contract {
    Contract::new(declarer, Bid::number(bid).unwrap(), physics.clone()).unwrap()
}

pub fn open_hidden(capacities: [usize; 3]) -> [Hidden; 3] {
    let seats = [Seat::S1, Seat::S2, Seat::S3];
    std::array::from_fn(|i| Hidden { seat: seats[i], capacity: capacities[i], voids: ContextSet::EMPTY })
}

/// S0's frame, declaring at 30 with nothing played, dealt `hand`.
pub fn opening(physics: &Physics, hand: DominoSet) -> Frame {
    framed(&contract(physics, Seat::S0, 30), Seat::S0, hand, vec![]).unwrap()
}

/// Five tricks from S0's lead under `physics` after which S0 holds `kept`
/// and leads, the other seats hold two each of `pool`, and no seat has shown
/// a void that a pool tile follows: every split of the pool is a world. A
/// deterministic search: a seat follows suit where it can; a hidden seat
/// sloughs only in a context no pool tile follows, and S0 only in one no
/// tile it keeps follows (its later plays then follow none either).
pub fn five_tricks_to(physics: &Physics, kept: DominoSet, pool: DominoSet) -> Vec<Domino> {
    fn search(
        physics: &Physics,
        kept: DominoSet,
        pool: DominoSet,
        rest: DominoSet,
        plays: &mut Vec<Domino>,
        leader: Seat,
        voids: &mut [ContextSet; 4],
    ) -> bool {
        let start = plays.len() - plays.len() % 4;
        if plays.len() == 20 {
            return leader == Seat::S0;
        }
        let seat = leader.plus(plays.len() - start);
        let lead = (plays.len() > start).then(|| plays[start]);
        let barred = |d: Domino| voids[seat.index()].iter().any(|q| physics.follows(d, q));
        let mut candidates: Vec<(Domino, Option<Context>)> = Vec::new();
        for d in rest.iter().filter(|d| !barred(*d)) {
            match lead.map(|l| physics.led_context(l)) {
                None => candidates.push((d, None)),
                Some(q) if physics.follows(d, q) => candidates.push((d, None)),
                Some(q) if physics.incidence(q).is_disjoint(if seat == Seat::S0 { kept } else { pool }) => {
                    candidates.push((d, Some(q)));
                }
                Some(_) => {}
            }
        }
        // Following before sloughing.
        candidates.sort_by_key(|(d, slough)| (slough.is_some(), d.index()));
        for (d, slough) in candidates {
            let mut next_voids = *voids;
            if let Some(q) = slough {
                next_voids[seat.index()].insert(q);
            }
            plays.push(d);
            let mut next_leader = leader;
            if plays.len().is_multiple_of(4) {
                let trick: [Domino; 4] = plays[plays.len() - 4..].try_into().unwrap();
                next_leader = physics.trick_winner(leader, trick);
            }
            let mut rest = rest;
            rest.remove(d);
            if search(physics, kept, pool, rest, plays, next_leader, &mut next_voids) {
                *voids = next_voids;
                return true;
            }
            plays.pop();
        }
        false
    }
    let rest = DominoSet::FULL.difference(kept.union(pool));
    let mut plays = Vec::new();
    let found = search(physics, kept, pool, rest, &mut plays, Seat::S0, &mut [ContextSet::EMPTY; 4]);
    assert!(found, "no record reaches the small frame under {physics:?}");
    plays
}

/// Viewer S0 holds 6-6 and 0-0 and is on lead; six tiles are hidden, two per
/// seat, and no void constrains them: 90 worlds. Reached by a record of five
/// tricks under `physics`.
pub fn small_frame(physics: Physics) -> Frame {
    let kept = tiles(&["6-6", "0-0"]);
    let pool = tiles(&["6-5", "5-5", "3-2", "6-4", "2-1", "1-0"]);
    let plays = five_tricks_to(&physics, kept, pool);
    let dealt = dealt_with(&physics, Seat::S0, kept, &plays);
    let frame = framed(&contract(&physics, Seat::S0, 30), Seat::S0, dealt, plays).unwrap();
    assert_eq!((frame.kernel().viewer_hand(), frame.kernel().pool(), frame.count_worlds()), (kept, pool, 90));
    frame
}

/// The hand `seat` was dealt: `kept` and its own plays in `plays`, the
/// declarer S0 leading the first trick.
pub fn dealt_with(physics: &Physics, seat: Seat, kept: DominoSet, plays: &[Domino]) -> DominoSet {
    actors(physics, Seat::S0, plays).into_iter().zip(plays).filter(|(s, _)| *s == seat).fold(kept, |h, (_, d)| {
        let mut h = h;
        h.insert(*d);
        h
    })
}

/// The seats of `plays`, `declarer` leading the first trick.
pub fn actors(physics: &Physics, declarer: Seat, plays: &[Domino]) -> Vec<Seat> {
    let mut leader = declarer;
    let mut trick = Vec::new();
    plays
        .iter()
        .map(|tile| {
            let actor = leader.plus(trick.len());
            trick.push(*tile);
            if let Ok(full) = <[Domino; 4]>::try_from(trick.as_slice()) {
                leader = physics.trick_winner(leader, full);
                trick.clear();
            }
            actor
        })
        .collect()
}

pub fn uniform() -> Belief {
    Belief::uniform(small_frame(physics("P6")), 90).unwrap()
}

pub fn compile(s: &str) -> CompiledQuery {
    s.parse::<Query>().unwrap().compile(&Registry::standard()).unwrap()
}

pub fn budget() -> Budget {
    Budget::new(20_000_000)
}

/// The value of an estimate that must be exact.
pub fn exact(e: Estimate) -> BigRational {
    assert!(e.authority().is_exact(), "{}", e.authority());
    e.value().clone()
}

pub fn given(model: &str) -> Authority {
    Authority::given(model).unwrap()
}

pub const COUNTS: &str = include_str!("../../examples/partner-count.silas");
pub const HAS_COUNT: &str = include_str!("../../examples/partner-has-count.silas");
pub const MASTER: &str = include_str!("../../examples/partner-master.silas");
pub const FORCED: &str = include_str!("../../examples/current-forced-play.silas");
pub const SIX_FIVE: &str = include_str!("../../examples/six-five-guarded.silas");
