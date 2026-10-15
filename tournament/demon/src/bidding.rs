// SPDX-FileCopyrightText: Jason Yandell
// SPDX-License-Identifier: CC0-1.0

//! STRATEGY-FUSION FENCE. The Demon: a fenced tournament opponent; measuring stick only.
//! Contract forecasts from the bidder's own hand. Requested declarations use
//! the same hidden deals; excluded declarations do no search work. These are
//! double-dummy frequencies, not calibrated wins.

use scenario_core::fiber::Fiber;
use scenario_core::physics::{ZebPhysics, dominoes, seat};
use scenario_core::planner::SplitMix;
use silas::{Bid, ClairvoyantSolver, Contract, OpenPosition};

pub struct Forecast {
    /// Make counts indexed by declaration. Unrequested declarations are None.
    pub makes: Vec<Option<u32>>,
    pub worlds: usize,
    pub nodes: u64,
}

/// STRATEGY-FUSION FENCE: the fenced opponent's exact DD forecast at each
/// declaration. The declaration is selected once, from these averages.
pub fn forecast(
    hand: u32,
    bidder: u8,
    bid: u32,
    worlds: usize,
    seed: u64,
    declarations: &[u8],
) -> Result<Forecast, String> {
    if hand >> 28 != 0 || hand.count_ones() != 7 || bidder > 3 || worlds == 0 || worlds > 4096 {
        return Err("forecast needs seven tiles, a seat 0..4, and worlds 1..=4096".into());
    }
    if declarations.is_empty()
        || declarations.iter().enumerate().any(|(i, &d)| d >= 10 || declarations[..i].contains(&d))
    {
        return Err("forecast declarations must be nonempty, distinct ids in 0..9".into());
    }
    let bid = Bid::number(bid).map_err(|e| e.0)?;
    let mut fiber = Fiber::new(bidder, hand, 0, [7; 4], [0; 4]);
    let (deals, _) = super::sampling::sample(&mut fiber, worlds, &mut SplitMix::new(seed))?;
    let mut makes = vec![None; 10];
    let mut nodes = 0;
    for &decl in declarations {
        let mut count = 0;
        let physics = ZebPhysics::new(decl).expect("declaration").physics();
        let contract = Contract::new(seat(bidder).expect("seat"), bid, physics.clone()).map_err(|e| e.0)?;
        let template = ClairvoyantSolver::new(&physics);
        for w in &deals {
            let position =
                OpenPosition::deal(contract.clone(), w.map(|h| dominoes(h).expect("hand"))).map_err(|e| e.0)?;
            let mut solver = template.fresh();
            count += solver.value(&position, silas::Objective::Make) as u32;
            nodes += solver.nodes();
        }
        makes[decl as usize] = Some(count);
    }
    Ok(Forecast { makes, worlds: deals.len(), nodes })
}
