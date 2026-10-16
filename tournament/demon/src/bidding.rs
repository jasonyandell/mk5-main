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

/// STRATEGY-FUSION FENCE: the exact winning declaration and its fully evaluated DD count.
pub struct BestForecast {
    pub decl: u8,
    pub makes: u32,
    pub worlds: usize,
    pub nodes: u64,
    pub solves: usize,
}

/// STRATEGY-FUSION FENCE: exact winner of the fenced DD forecast. Declarations
/// are searched in tie order. Once successes plus unsearched worlds cannot
/// exceed the incumbent, the remaining worlds cannot change the winner.
pub fn best_forecast(
    hand: u32,
    bidder: u8,
    bid: u32,
    worlds: usize,
    seed: u64,
    declarations: &[u8],
) -> Result<BestForecast, String> {
    let (bid, deals) = prepare(hand, bidder, bid, worlds, seed, declarations)?;
    let preferred = super::hand_model::heuristic_declaration(hand);
    let mut order = declarations.to_vec();
    order.sort_by_key(|&d| (d != preferred, d));
    let mut best = BestForecast { decl: order[0], makes: 0, worlds: deals.len(), nodes: 0, solves: 0 };
    for (rank, &decl) in order.iter().enumerate() {
        let physics = ZebPhysics::new(decl).expect("declaration").physics();
        let contract = Contract::new(seat(bidder).expect("seat"), bid, physics.clone()).map_err(|e| e.0)?;
        let template = ClairvoyantSolver::new(&physics);
        let mut count = 0;
        for (i, w) in deals.iter().enumerate() {
            if rank > 0 && count + (deals.len() - i) as u32 <= best.makes {
                break;
            }
            let position =
                OpenPosition::deal(contract.clone(), w.map(|h| dominoes(h).expect("hand"))).map_err(|e| e.0)?;
            let mut solver = template.fresh();
            count += solver.value(&position, silas::Objective::Make) as u32;
            best.nodes += solver.nodes();
            best.solves += 1;
        }
        // A pruned declaration cannot beat the incumbent; if count does beat
        // it, every world was searched, and the winning count is exact.
        if rank == 0 || count > best.makes {
            best.decl = decl;
            best.makes = count;
        }
    }
    Ok(best)
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
    let (bid, deals) = prepare(hand, bidder, bid, worlds, seed, declarations)?;
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

fn prepare(
    hand: u32,
    bidder: u8,
    bid: u32,
    worlds: usize,
    seed: u64,
    declarations: &[u8],
) -> Result<(Bid, Vec<[u32; 4]>), String> {
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
    Ok((bid, deals))
}

/// STRATEGY-FUSION FENCE: a threshold certificate on the fixed DD sample.
pub struct Support {
    pub supported: bool,
    pub worlds: usize,
    pub nodes: u64,
    pub solves: usize,
    pub checked: Vec<(u8, usize, u32)>,
}

/// STRATEGY-FUSION FENCE: exact predicate max(make_count) >= required. A
/// declaration passes once enough worlds make, and fails once even every
/// unsearched world making would leave it below the required count.
pub fn supports(
    hand: u32,
    bidder: u8,
    bid: u32,
    worlds: usize,
    seed: u64,
    declarations: &[u8],
    required: u32,
) -> Result<Support, String> {
    let (bid, deals) = prepare(hand, bidder, bid, worlds, seed, declarations)?;
    if required as usize > deals.len() {
        return Err("required count exceeds sampled worlds".into());
    }
    let preferred = super::hand_model::heuristic_declaration(hand);
    let mut order = declarations.to_vec();
    order.sort_by_key(|&d| (d != preferred, d));
    let mut out = Support { supported: false, worlds: deals.len(), nodes: 0, solves: 0, checked: vec![] };
    for decl in order {
        let physics = ZebPhysics::new(decl).expect("declaration").physics();
        let contract = Contract::new(seat(bidder).expect("seat"), bid, physics.clone()).map_err(|e| e.0)?;
        let template = ClairvoyantSolver::new(&physics);
        let (mut successes, mut tested) = (0, 0);
        for w in &deals {
            if successes >= required || successes + ((deals.len() - tested) as u32) < required {
                break;
            }
            let position =
                OpenPosition::deal(contract.clone(), w.map(|h| dominoes(h).expect("hand"))).map_err(|e| e.0)?;
            let mut solver = template.fresh();
            successes += solver.value(&position, silas::Objective::Make) as u32;
            out.nodes += solver.nodes();
            out.solves += 1;
            tested += 1;
        }
        out.checked.push((decl, tested, successes));
        if successes >= required {
            out.supported = true;
            break;
        }
    }
    Ok(out)
}
