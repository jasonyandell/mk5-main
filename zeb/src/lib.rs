//! Zeb — the Texas 42 scenario player, compiled to WebAssembly.
//!
//! A seat decides by sampling deals of the tiles it cannot see that fit
//! everything the table has shown, planning one contingent line of play
//! across all of them at once, and taking the first move of the plan
//! (`scenario/planner.py`, compiled as `scenario_core`). The other seats
//! are modeled by the field net, a small integer policy trained on Zeb's
//! own play, under the pip-trump declarations; under doubles and no-trump
//! they are modeled by a cheaper Walt one rung down. The planner never sees
//! another seat's hand.
//!
//! The JavaScript surface (`forecast`, `play`, `analyze`, `version`,
//! `netId`) lives in the `js` module; this module is the same player as
//! plain Rust, which the native tests drive.

#![warn(missing_docs)]

mod js;
mod position;

pub use position::{Error, Position};

use std::sync::LazyLock;

use position::{hand_mask, refuse, rules_for};
use scenario_core::field::FieldNet;
use scenario_core::planner::{analyze, decide, fsum, Analysis, Config, Draw, DRAW_BUDGET, MAX_WORLDS};
use serde::Serialize;

include!(concat!(env!("OUT_DIR"), "/net.rs"));

/// The embedded net, parsed once.
static FIELD: LazyLock<FieldNet> =
    LazyLock::new(|| FieldNet::from_bytes(NET).expect("the embedded net parses (cargo test checks it)"));

/// Scenarios per decision when a request does not say.
pub const DEFAULT_WORLDS: usize = 160;

/// The contract targets a tail is read at: 30..=42 points.
pub const TARGETS: [u8; 13] = [30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42];

/// The declarations a forecast covers when a request does not say: the
/// straight ones, every pip suit, doubles and no-trump.
pub const DECLARATIONS: [u8; 9] = [0, 1, 2, 3, 4, 5, 6, 7, 9];

/// Below the root, a live cell thinner than this hands the decider's move
/// down a rung instead of maximizing over a few clairvoyant worlds — when
/// the worlds are a sample. An enumerated fiber is the exact posterior and
/// is never floored.
const FLOOR: usize = 16;

/// The modeled rung of the plan for doubles and no-trump: 8 scenarios, a
/// 4-play horizon (`walt:k1,m8`).
const MODELED: (usize, Option<usize>, usize) = (8, Some(4), 1);

/// The package version.
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// The embedded field net's sha256, lowercase hex: the same digest
/// `shasum -a 256 net/margin_d250k_h128x64.fnet` prints.
pub fn net_id() -> &'static str {
    NET_ID
}

/// One declaration's forecast for the bidder, before the opening lead.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Forecast {
    /// The declaration.
    pub decl: u8,
    /// The bidding side's expected final margin (its points less the
    /// defenders'), playing the planned opening lead.
    pub mean: f64,
    /// For each target in [`TARGETS`], the probability the bidding side
    /// banks at least that many points.
    pub pmake: [f64; 13],
}

/// A seat's move.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Move {
    /// The tile to play.
    pub tile: u8,
    /// Only one tile was legal; nothing was planned.
    pub forced: bool,
    /// Each legal tile (ascending) and the probability the bidding side
    /// makes its contract if it is played: the bidding side plays the
    /// largest, the defenders the smallest. Empty when forced.
    pub q: Vec<(u8, f64)>,
}

/// A seat's move, opened up for a hint.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Hint {
    /// The tile to play.
    pub tile: u8,
    /// Only one tile was legal; nothing was planned.
    pub forced: bool,
    /// As in [`Move`].
    pub q: Vec<(u8, f64)>,
    /// Each legal tile (ascending) and, for each target in [`TARGETS`], the
    /// probability the bidding side banks at least that many points if it
    /// is played. Empty when forced.
    pub pmake: Vec<(u8, [f64; 13])>,
}

/// The player: how many scenarios it plans over and the seed every random
/// draw is keyed on. The same player at the same position always answers
/// the same, bit for bit, as the Python referee does.
pub struct Player {
    worlds: usize,
    seed: u64,
}

impl Player {
    /// A player planning over `worlds` scenarios (1..=1024) per decision.
    pub fn new(worlds: usize, seed: u64) -> Result<Player, Error> {
        if !(1..=MAX_WORLDS).contains(&worlds) {
            refuse!("worlds must be an integer from 1 to {MAX_WORLDS}, not {worlds}");
        }
        Ok(Player { worlds, seed })
    }

    /// The plan for a declaration. Under a pip trump the live seat plans
    /// alone over the field net (`walt:k0,n<worlds>,F=<net>`); the net does
    /// not play doubles or no-trump, so there the modeled seats are a
    /// one-rung Walt (`walt:k1,n<worlds>,m8`), which costs more.
    fn config(&self, decl: u8) -> Config {
        let (rungs, field) = if FIELD.plays(decl) {
            (vec![(self.worlds, None, FLOOR)], Some(FIELD.clone()))
        } else {
            (vec![MODELED, (self.worlds, None, FLOOR)], None)
        };
        // The field is the deterministic net (field_eps 0), and the worlds
        // are uniform on the fiber: every scenario weighs the same.
        Config {
            rungs,
            make: true,
            seed: self.seed,
            field,
            field_eps: 0.0,
            draw: Draw::Uniform,
            draw_budget: DRAW_BUDGET,
            convention: None,
            reading: None,
        }
    }

    /// For each declaration in `decls` (distinct, in order), how the bidder
    /// holding `hand` fares if it wins a 30 bid and declares it: one plan
    /// from the bidder's chair before the opening lead, its tails read at
    /// every target. The plan is made as seat 0; a seat number only
    /// relabels the table (and so reseeds the scenarios).
    pub fn forecast(&self, hand: &[u8], decls: &[u8]) -> Result<Vec<Forecast>, Error> {
        hand_mask(hand)?;
        for (i, &d) in decls.iter().enumerate() {
            rules_for(d)?;
            if decls[..i].contains(&d) {
                refuse!("declaration {d} is listed twice");
            }
        }
        decls
            .iter()
            .map(|&decl| {
                let a = self.analysis(&Position::new(decl, 0, TARGETS[0], 0, hand, &[])?);
                let (_, margins) = a.outcomes.iter().find(|(t, _)| *t == a.tile).expect("the chosen lead has outcomes");
                let total = fsum(a.weights.iter().copied());
                let mean = fsum(margins.iter().zip(&a.weights).map(|(m, w)| m * w)) / total;
                Ok(Forecast { decl, mean, pmake: tails(margins, &a.weights) })
            })
            .collect()
    }

    /// The move at a position: the first move of the plan.
    pub fn play(&self, p: &Position) -> Move {
        match forced(p) {
            Some(tile) => Move { tile, forced: true, q: Vec::new() },
            None => {
                let (tile, q, _, _) = decide(&self.config(p.decl()), &p.decision(), None);
                Move { tile, forced: false, q }
            }
        }
    }

    /// [`Player::play`], with each legal tile's tails.
    pub fn analyze(&self, p: &Position) -> Hint {
        match forced(p) {
            Some(tile) => Hint { tile, forced: true, q: Vec::new(), pmake: Vec::new() },
            None => {
                let a = self.analysis(p);
                let pmake = a.outcomes.iter().map(|(t, margins)| (*t, tails(margins, &a.weights))).collect();
                Hint { tile: a.tile, forced: false, q: a.q, pmake }
            }
        }
    }

    /// The planner's analysis of a position.
    fn analysis(&self, p: &Position) -> Analysis {
        analyze(&self.config(p.decl()), &p.decision(), None)
    }
}

/// The only legal tile, when there is one: such a move is not planned.
fn forced(p: &Position) -> Option<u8> {
    let legal = p.legal();
    (legal.count_ones() == 1).then(|| legal.trailing_zeros() as u8)
}

/// For each target b, the share of the belief in which the bidding side
/// banks at least b: its final margin is at least 2b − 42.
fn tails(margins: &[f64], weights: &[f64]) -> [f64; 13] {
    let total = fsum(weights.iter().copied());
    TARGETS.map(|b| {
        let need = 2.0 * b as f64 - 42.0;
        fsum(margins.iter().zip(weights).filter(|(m, _)| **m >= need).map(|(_, w)| *w)) / total
    })
}

#[cfg(test)]
mod tests;
