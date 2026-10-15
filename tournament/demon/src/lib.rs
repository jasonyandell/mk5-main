// SPDX-FileCopyrightText: Jason Yandell
// SPDX-License-Identifier: CC0-1.0

//! STRATEGY-FUSION FENCE.
//! The Demon: a fenced tournament opponent; measuring stick only.
//!
//! Classic Perfect-Information Monte Carlo, the textbook strategy-fusion
//! player. At each decision the seat
//!
//! 1. replays the public record (Walt's [`Rec`]: who played what, the banked
//!    points, the voids each seat has shown),
//! 2. draws `n` worlds from its fiber without replacement, using the fiber
//!    completion-count dynamic program (the whole fiber when it has at most `n` members),
//! 3. solves each world double dummy with Silas's [`ClairvoyantSolver`]: the
//!    exact make-or-set of the contract under perfect play by all four seats
//!    with every hand face up,
//! 4. plays the legal tile that makes (or, defending, sets) in the most worlds;
//!    among exact ties, the best summed double-dummy margin; then the lowest
//!    tile id.
//!
//! Step 3 credits each world with a line no seat that cannot see the hands
//! could choose (E[max] ≥ max E), which is exactly why it is fenced: it is
//! here so Walt and Zeb can be measured against strategy fusion, never as a
//! component of either. Nothing in Zeb, Walt or Silas may depend on this
//! crate. The ruling: `wiki/topics/strategy-fusion-fence.md`; the Demon's
//! page: `wiki/topics/the-demon.md`.
//!
//! Deterministic: the worlds are a function of the request's seed, the
//! solver's values are exact (its transposition table changes the work, never
//! a value), and every tie is broken by a fixed rule.

mod sampling;

use scenario_core::fiber::Fiber;
use scenario_core::physics::{ZebPhysics, domino, dominoes, seat};
use scenario_core::planner::{Rec, SplitMix};
use scenario_core::rules::{N_DECLS, Rules};
use silas::{Bid, ClairvoyantSolver, Contract, Domino, DominoSet, Objective, OpenPosition, Seat};

/// The Demon's two knobs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Config {
    /// Worlds drawn per decision (the whole fiber when it is no larger).
    pub worlds: usize,
    /// Break exact make-count ties by summed double-dummy margin.
    pub margin_tiebreak: bool,
}

impl Default for Config {
    fn default() -> Self {
        Config { worlds: 32, margin_tiebreak: true }
    }
}

/// One decision, as the seat sees it: its own remaining tiles and the public
/// record. Never another seat's hand.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Request {
    pub seat: u8,
    /// The seat's remaining tiles, a mask over Zeb's tile ids.
    pub hand: u32,
    /// Zeb's declaration id: 0..=6 pip trump, 7 doubles trump, 8 doubles as a suit, 9 no trump.
    pub decl: u8,
    /// The bid as the table calls it: 30..=41 points, 42 per mark.
    pub bid: u32,
    pub bidder: u8,
    /// Every play so far, (seat, tile), in order; the bidder led the first.
    pub plays: Vec<(u8, u8)>,
    /// The decision's seed: the worlds are a function of it.
    pub seed: u64,
}

/// What the Demon chose and why.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Decision {
    pub tile: u8,
    /// Each legal tile with the worlds where the declaring side makes after
    /// it, and (when a margin tiebreak ran for it) its summed double-dummy
    /// margin for the declaring side.
    pub moves: Vec<(u8, u32, Option<i64>)>,
    /// Worlds solved (zero when only one tile was legal).
    pub worlds: usize,
    /// The worlds were the whole fiber.
    pub exact: bool,
    /// Positions the solver searched.
    pub nodes: u64,
}

/// The public record after `plays`, each play checked as the seat to move's
/// and as a tile not yet down. The bidder leads the first trick.
pub fn record(decl: u8, bidder: u8, plays: &[(u8, u8)]) -> Result<(Rules, Rec), String> {
    let rules = Rules::new(decl).ok_or_else(|| format!("declaration {decl} is outside 0..{N_DECLS}"))?;
    if bidder > 3 {
        return Err(format!("bidder {bidder} is not a seat"));
    }
    let mut rec = Rec::opening(bidder);
    for (i, &(s, t)) in plays.iter().enumerate() {
        if t > 27 || rec.played >> t & 1 == 1 {
            return Err(format!("play {i}: tile {t} is not in the pool"));
        }
        if s != rec.seat() {
            return Err(format!("play {i}: seat {s} played but seat {} was to play", rec.seat()));
        }
        rec = rec.advance(t, &rules);
    }
    Ok((rules, rec))
}

/// STRATEGY-FUSION FENCE: the Demon's decision. Per-world clairvoyant values,
/// averaged: a fenced opponent's choice, never a seat's value anywhere else.
pub fn decide(cfg: Config, req: &Request) -> Result<Decision, String> {
    if cfg.worlds == 0 {
        return Err("the Demon needs at least one world".into());
    }
    let (rules, rec) = record(req.decl, req.bidder, &req.plays)?;
    if rec.finished() {
        return Err("every tile is played".into());
    }
    if rec.seat() != req.seat {
        return Err(format!("seat {} asked, but seat {} is to play", req.seat, rec.seat()));
    }
    if req.hand & rec.played != 0 || req.hand >> 28 != 0 {
        return Err("the hand holds a tile already played".into());
    }
    let sizes: [u8; 4] = std::array::from_fn(|s| rec.hand_size(s as u8));
    if req.hand.count_ones() != sizes[req.seat as usize] as u32 {
        return Err(format!(
            "the hand has {} tiles; the record says {}",
            req.hand.count_ones(),
            sizes[req.seat as usize]
        ));
    }
    if req.hand & rec.forbid[req.seat as usize] != 0 {
        return Err("the hand holds a tile of a suit this seat has shown void in".into());
    }
    let legal = rules.legal(req.hand, rec.led());
    let tiles: Vec<u8> = (0..28u8).filter(|&t| legal >> t & 1 == 1).collect();
    if tiles.len() == 1 {
        return Ok(Decision { tile: tiles[0], moves: vec![(tiles[0], 0, None)], worlds: 0, exact: false, nodes: 0 });
    }

    let mut fiber = Fiber::new(req.seat, req.hand, rec.played, sizes, rec.forbid);
    let (worlds, exact) = sampling::sample(&mut fiber, cfg.worlds, &mut SplitMix::new(req.seed))?;
    let physics = ZebPhysics::new(req.decl).expect("checked above").physics();
    let bid = Bid::number(req.bid).map_err(|e| e.0)?;
    let contract = Contract::new(seat(req.bidder).expect("checked above"), bid, physics.clone()).map_err(|e| e.0)?;
    let leader = seat(rec.leader).expect("a seat");
    let trick: Vec<Domino> = rec.trick[..rec.n as usize].iter().map(|&t| domino(t).expect("a tile")).collect();
    let banked = [rec.banked[0] as u32, rec.banked[1] as u32];
    let positions: Vec<OpenPosition> = worlds
        .iter()
        .map(|w| {
            let hands: [DominoSet; Seat::COUNT] = std::array::from_fn(|s| dominoes(w[s]).expect("tile masks"));
            OpenPosition::new(contract.clone(), hands, leader, trick.clone(), banked).map_err(|e| e.0)
        })
        .collect::<Result<_, _>>()?;

    // Keep each world's bounds in a small table, and retain its make search
    // for the possible margin pass. Tables across unrelated worlds cost more
    // to look up than they save in shared states. Values remain exact.
    let template = ClairvoyantSolver::new(&physics);
    let mut solvers: Vec<_> = positions.iter().map(|_| template.fresh()).collect();
    let mut makes = vec![0u32; tiles.len()];
    for (p, solver) in positions.iter().zip(&mut solvers) {
        let values = solver.solve(p, Objective::Make).map_err(|e| e.0)?;
        for (i, &t) in tiles.iter().enumerate() {
            makes[i] += values.value(domino(t).expect("a tile")).expect("a legal tile") as u32;
        }
    }
    let declaring = req.seat % 2 == req.bidder % 2;
    let n = positions.len() as u32;
    let score = |m: u32| if declaring { m } else { n - m };
    let best = makes.iter().map(|&m| score(m)).max().expect("a legal tile");
    let tied: Vec<usize> = (0..tiles.len()).filter(|&i| score(makes[i]) == best).collect();

    let mut nodes = 0;
    let mut margins: Vec<Option<i64>> = vec![None; tiles.len()];
    // Finish all tied tiles of one world together, then drop its table. The
    // much larger margin searches never accumulate across the whole sample.
    for (p, mut solver) in positions.iter().zip(solvers) {
        if cfg.margin_tiebreak && tied.len() > 1 {
            for &i in &tied {
                let tile = domino(tiles[i]).expect("a tile");
                let margin = solver.value(&p.play(tile).map_err(|e| e.0)?, Objective::Margin) as i64;
                *margins[i].get_or_insert(0) += margin;
            }
        }
        nodes += solver.nodes();
    }
    // Make count first, then margin for the mover's side, then the lowest tile.
    let key = |i: usize| (score(makes[i]), margins[i].map_or(0, |m| if declaring { m } else { -m }));
    let pick = tied.iter().copied().fold(tied[0], |b, i| if key(i) > key(b) { i } else { b });
    Ok(Decision {
        tile: tiles[pick],
        moves: (0..tiles.len()).map(|i| (tiles[i], makes[i], margins[i])).collect(),
        worlds: positions.len(),
        exact,
        nodes,
    })
}
