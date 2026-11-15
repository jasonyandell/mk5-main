// SPDX-FileCopyrightText: Jason Yandell
// SPDX-License-Identifier: CC0-1.0

//! Walt, the level-1 scenario player for Texas 42: the `walt1 play` request of WALT.md v1.0.
//!
//! [`handle`] takes one request and returns one JSON line. [`decide`] is the same decision as
//! a Rust value. No clock, no threads, no global state: a request's answer is a pure function
//! of its text.

pub mod rng;
pub mod rules;
pub mod sampler;
pub mod solver;
#[cfg(target_arch = "wasm32")]
mod wasm;

use rng::{derive, Rng};
use rules::{Rules, State};
use sampler::{OuterSampler, World};
use solver::{Alive, Memo, Minds, Solver, Table};

pub const DEFAULT_N: usize = 40;
pub const DEFAULT_N0: usize = 8;
pub const DEFAULT_SEED: u64 = 0xB7E1_5162_8AED_2A6B;
/// The largest `n` accepted; refinement reaches 16·n worlds.
pub const MAX_N: usize = 4096;
pub const MAX_N0: usize = 4096;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Request {
    pub decl: u8,
    pub bid: u8,
    pub seat: u8,
    pub bidder: u8,
    /// The seat's seven dealt tile ids.
    pub hand: Vec<u8>,
    /// `(seat, tile)` in order.
    pub plays: Vec<(u8, u8)>,
    pub n: usize,
    pub n0: usize,
    pub seed: u64,
}

/// A tile's final estimate: the bidding team makes in `makes` of `worlds` sampled worlds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Opt {
    pub tile: u8,
    pub makes: u32,
    pub worlds: u32,
}

impl Opt {
    /// The value × 10000, truncated.
    pub fn bp(&self) -> u32 {
        (self.makes as u64 * 10000 / self.worlds as u64) as u32
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Decision {
    pub choice: u8,
    pub forced: bool,
    /// Every legal tile in ascending id; empty when forced.
    pub opts: Vec<Opt>,
    pub leader: u8,
    pub points: [u8; 2],
    /// Level-0 minds solved for this decision (cost, not part of the response).
    pub minds: u64,
}

pub fn parse(req: &str) -> Result<Request, String> {
    let mut lines = req.lines().map(str::trim).filter(|l| !l.is_empty());
    if lines.next().map(|l| l.split_whitespace().collect::<Vec<_>>()) != Some(vec!["walt1", "play"]) {
        return Err("the first line must be: walt1 play".into());
    }
    let mut fields: [Option<Vec<u64>>; 9] = Default::default();
    const KEYS: [&str; 9] = ["decl", "bid", "seat", "bidder", "hand", "plays", "n", "n0", "seed"];
    for line in lines {
        let mut words = line.split_whitespace();
        let key = words.next().unwrap();
        let slot = KEYS.iter().position(|&k| k == key).ok_or(format!("unknown key {key}"))?;
        if fields[slot].is_some() {
            return Err(format!("{key} given twice"));
        }
        let values: Result<Vec<u64>, _> = words.map(str::parse::<u64>).collect();
        fields[slot] = Some(values.map_err(|_| format!("{key} takes unsigned integers"))?);
    }
    let one = |slot: usize, max: u64, default: Option<u64>| -> Result<u64, String> {
        match (&fields[slot], default) {
            (Some(v), _) if v.len() == 1 && v[0] <= max => Ok(v[0]),
            (Some(_), _) => Err(format!("{} takes one integer, at most {max}", KEYS[slot])),
            (None, Some(d)) => Ok(d),
            (None, None) => Err(format!("{} is missing", KEYS[slot])),
        }
    };
    let decl = one(0, 9, None)? as u8;
    let bid = one(1, 42, None)? as u8;
    let seat = one(2, 3, None)? as u8;
    let bidder = one(3, 3, None)? as u8;
    let n = one(6, MAX_N as u64, Some(DEFAULT_N as u64))? as usize;
    let n0 = one(7, MAX_N0 as u64, Some(DEFAULT_N0 as u64))? as usize;
    let seed = one(8, u64::MAX, Some(DEFAULT_SEED))?;
    if bid < 30 {
        return Err("bid must be 30 to 42".into());
    }
    if n == 0 || n0 == 0 {
        return Err("n and n0 must be at least 1".into());
    }
    let tile = |x: u64| if x < 28 { Ok(x as u8) } else { Err(format!("{x} is not a tile id")) };
    let hand = fields[4].as_ref().ok_or("hand is missing")?;
    let hand = hand.iter().map(|&x| tile(x)).collect::<Result<Vec<u8>, _>>()?;
    let flat = fields[5].clone().unwrap_or_default();
    if flat.len() % 2 != 0 {
        return Err("plays must be seat tile pairs".into());
    }
    let mut plays = Vec::with_capacity(flat.len() / 2);
    for pair in flat.chunks(2) {
        if pair[0] > 3 {
            return Err(format!("{} is not a seat", pair[0]));
        }
        plays.push((pair[0] as u8, tile(pair[1])?));
    }
    Ok(Request { decl, bid, seat, bidder, hand, plays, n, n0, seed })
}

/// The record replayed from one seat's point of view.
struct Replay {
    st: State,
    /// The seat's tiles still in hand.
    hand: u32,
    /// Per seat, the tiles it has shown it cannot hold.
    forbidden: [u32; 4],
}

fn replay(req: &Request, rules: &Rules) -> Result<Replay, String> {
    let mut dealt = 0u32;
    for &t in &req.hand {
        dealt |= 1 << t;
    }
    if req.hand.len() != 7 || dealt.count_ones() != 7 {
        return Err("hand must be seven different tiles".into());
    }
    let mut st = State::start(req.bidder);
    let mut forbidden = [0u32; 4];
    for (i, &(seat, tile)) in req.plays.iter().enumerate() {
        let at = |why: &str| format!("play {i} (seat {seat}, tile {tile}): {why}");
        if st.played == rules::ALL_TILES {
            return Err(at("the hand is over"));
        }
        if seat != st.to_act() {
            return Err(at("not that seat's turn"));
        }
        if st.played >> tile & 1 == 1 {
            return Err(at("tile already played"));
        }
        let follows = st.len == 0 || rules.follow[st.trick[0] as usize] >> tile & 1 == 1;
        if seat == req.seat {
            if dealt >> tile & 1 == 0 {
                return Err(at("tile is not in the seat's hand"));
            }
            if rules.legal(dealt & !st.played, &st) >> tile & 1 == 0 {
                return Err(at("the seat could have followed suit"));
            }
        } else {
            if dealt >> tile & 1 == 1 {
                return Err(at("tile is in the deciding seat's hand"));
            }
            if forbidden[seat as usize] >> tile & 1 == 1 {
                return Err(at("the seat showed void in that suit"));
            }
            if !follows {
                forbidden[seat as usize] |= rules.follow[st.trick[0] as usize];
            }
        }
        st = st.play(rules, tile);
    }
    if st.played == rules::ALL_TILES {
        return Err("the hand is over".into());
    }
    if st.to_act() != req.seat {
        return Err(format!("seat {} is to act, not seat {}", st.to_act(), req.seat));
    }
    Ok(Replay { st, hand: dealt & !st.played, forbidden })
}

/// One evaluation round of §5.4: the make count of each tile of `tiles` on `worlds`.
fn evaluate<A: Alive>(
    table: Table,
    seat: u8,
    worlds: &[World],
    minds: &mut Minds,
    st: &State,
    tiles: &[u8],
) -> Vec<u32> {
    let mut solver = Solver { table, me: seat, worlds, field: minds, memo: Memo::<A>::default() };
    let full = A::full(worlds.len());
    tiles.iter().map(|&t| solver.value(&st.play(table.rules, t), full.clone())).collect()
}

fn evaluate_sized(
    table: Table,
    seat: u8,
    worlds: &[World],
    minds: &mut Minds,
    st: &State,
    tiles: &[u8],
) -> Vec<u32> {
    match worlds.len() {
        0..=64 => evaluate::<u64>(table, seat, worlds, minds, st, tiles),
        65..=256 => evaluate::<[u64; 4]>(table, seat, worlds, minds, st, tiles),
        257..=1024 => evaluate::<[u64; 16]>(table, seat, worlds, minds, st, tiles),
        _ => evaluate::<Vec<u64>>(table, seat, worlds, minds, st, tiles),
    }
}

pub fn decide(req: &Request) -> Result<Decision, String> {
    let rules = Rules::new(req.decl).ok_or("decl must be 0 to 7, or 9")?;
    let Replay { st, hand, forbidden } = replay(req, &rules)?;
    let seat = req.seat;
    let legal_mask = rules.legal(hand, &st);
    let mut legal = Vec::with_capacity(7);
    let mut m = legal_mask;
    while m != 0 {
        legal.push(m.trailing_zeros() as u8);
        m &= m - 1;
    }
    let mut decision = Decision {
        choice: legal[0],
        forced: legal.len() == 1,
        opts: Vec::new(),
        leader: st.leader,
        points: st.pts,
        minds: 0,
    };
    if decision.forced {
        return Ok(decision);
    }
    let sampler = OuterSampler::new(seat, hand, &st, &forbidden).ok_or("no deal fits the record")?;
    let table = Table { rules: &rules, bid: req.bid, bid_team: req.bidder & 1 };
    let maximize = seat & 1 == table.bid_team;
    let mut minds = Minds::new(table, req.n0);

    // Evaluate, then refine ties (§5.4). Round r draws a fresh outer sample of its own stream.
    let mut size = req.n;
    let mut round = 0u64;
    let mut tied: Vec<u8> = legal.clone();
    let mut opts: Vec<Opt> = legal.iter().map(|&tile| Opt { tile, makes: 0, worlds: 1 }).collect();
    loop {
        let mut rng = Rng::new(derive(req.seed, round));
        let worlds: Vec<World> = (0..size).map(|_| sampler.draw(&mut rng)).collect();
        let makes = evaluate_sized(table, seat, &worlds, &mut minds, &st, &tied);
        for (&tile, &makes) in tied.iter().zip(&makes) {
            let o = opts.iter_mut().find(|o| o.tile == tile).unwrap();
            *o = Opt { tile, makes, worlds: size as u32 };
        }
        // Values of different sample sizes compare as rationals, by cross-multiplying.
        let better = |a: &Opt, b: &Opt| {
            let (x, y) = (a.makes as u64 * b.worlds as u64, b.makes as u64 * a.worlds as u64);
            if maximize {
                x > y
            } else {
                x < y
            }
        };
        let mut best = opts[0];
        for o in &opts[1..] {
            if better(o, &best) {
                best = *o;
            }
        }
        tied = opts.iter().filter(|o| !better(&best, o)).map(|o| o.tile).collect();
        decision.choice = best.tile;
        if tied.len() < 2 || size >= 16 * req.n {
            break;
        }
        size *= 4;
        round += 1;
    }
    decision.opts = opts;
    decision.minds = minds.solved;
    Ok(decision)
}

/// One `walt1 play` request in, one JSON line out.
pub fn handle(req: &str) -> String {
    match parse(req).and_then(|r| decide(&r)) {
        Ok(d) => {
            let opts: Vec<String> = d.opts.iter().map(|o| format!("[{},{}]", o.tile, o.bp())).collect();
            format!(
                "{{\"v\":1,\"kind\":\"play\",\"choice\":{},\"forced\":{},\"raced\":false,\"opts\":[{}],\"leader\":{},\"points\":[{},{}]}}",
                d.choice,
                d.forced,
                opts.join(","),
                d.leader,
                d.points[0],
                d.points[1]
            )
        }
        Err(e) => {
            let safe: String = e.chars().filter(|c| !matches!(c, '"' | '\\') && !c.is_control()).collect();
            format!("{{\"v\":1,\"error\":\"{safe}\"}}")
        }
    }
}
