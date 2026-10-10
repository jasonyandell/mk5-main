//! The compiled core of the scenario player.
//!
//! This crate is `scenario/planner.py` and `scenario/fiber.py` in Rust, line
//! for line: the same SplitMix64 streams, the same packed record key, the
//! same summation order, the same tie rule. The Python planner is the
//! referee — `scenario/tests/test_compiled.py` asserts the two return
//! identical tiles and identical action values on random states — and this
//! is the runtime. Nothing here knows the rules of 42: the per-declaration
//! tables arrive from Python (`forge.oracle.tables` stays the authority).
//!
//! Interface (see `scenario/compiled.py`): `Core(rungs, utility, seed)`,
//! `set_rules(decl, ...)`, `sample_roots(reqs)` for the belief seam, and
//! `solve(reqs, weights, threads)`, which runs a batch of decisions in
//! parallel with the GIL released.

mod fiber;
mod planner;

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use planner::{decide, Config, Rec, Rules, TiltPlays, MAX_WORLDS};

/// One decision request:
/// (decl, viewer, own, played, leader, trick, (banked0, banked1),
///  (forbid0, forbid1, forbid2, forbid3), bid, bidder_team, plays as (seat, tile))
type Req = (u8, u8, u32, u32, u8, Vec<u8>, (u8, u8), (u32, u32, u32, u32), i32, u8, Vec<(u8, u8)>);

#[pyclass]
struct Core {
    cfg: Config,
    rules: HashMap<u8, Rules>,
}

fn rec_of(req: &Req) -> PyResult<Rec> {
    let (_, _, _, played, leader, trick, banked, forbid, _, _, _) = req;
    if trick.len() > 3 {
        return Err(PyValueError::new_err("a record's open trick holds at most three tiles"));
    }
    let mut t = [0u8; 3];
    t[..trick.len()].copy_from_slice(trick);
    Ok(Rec {
        played: *played,
        leader: *leader,
        n: trick.len() as u8,
        trick: t,
        banked: [banked.0, banked.1],
        forbid: [forbid.0, forbid.1, forbid.2, forbid.3],
    })
}

#[pymethods]
impl Core {
    #[new]
    fn new(rungs: Vec<(usize, Option<usize>)>, utility: &str, seed: u64, tilt: &str, eps: f64) -> PyResult<Self> {
        if rungs.is_empty() {
            return Err(PyValueError::new_err("a player needs at least one rung"));
        }
        if rungs.iter().any(|&(n, _)| n == 0 || n > MAX_WORLDS) {
            return Err(PyValueError::new_err(format!("n_worlds must be in 1..={MAX_WORLDS}")));
        }
        let make = match utility {
            "make" => true,
            "points" => false,
            other => return Err(PyValueError::new_err(format!("unknown utility {other:?}"))),
        };
        let tilt = match tilt {
            "none" => TiltPlays::None,
            "dice" => TiltPlays::Dice,
            "field" if rungs.len() > 1 => TiltPlays::Field,
            "field" => return Err(PyValueError::new_err("the field tilt needs a modeled rung")),
            other => return Err(PyValueError::new_err(format!("unknown tilt {other:?}"))),
        };
        if !(eps > 0.0 && eps <= 1.0) {
            return Err(PyValueError::new_err("eps must lie in (0, 1]"));
        }
        Ok(Core { cfg: Config { rungs, make, seed, tilt, eps }, rules: HashMap::new() })
    }

    /// Register one declaration's tables (tabulated in Python from the rules authority).
    fn set_rules(
        &mut self,
        decl: u8,
        led_suit: Vec<u8>,
        follow_bits: Vec<u32>,
        rank: Vec<Vec<u8>>,
        count: Vec<u8>,
    ) -> PyResult<()> {
        if led_suit.len() != 28 || follow_bits.len() != 8 || count.len() != 28
            || rank.len() != 8 || rank.iter().any(|r| r.len() != 28)
        {
            return Err(PyValueError::new_err("rules tables have the wrong shape"));
        }
        let mut r = Rules { led_suit: [0; 28], follow_bits: [0; 8], rank: [[0; 28]; 8], count: [0; 28] };
        r.led_suit.copy_from_slice(&led_suit);
        r.follow_bits.copy_from_slice(&follow_bits);
        r.count.copy_from_slice(&count);
        for (row, src) in r.rank.iter_mut().zip(&rank) {
            row.copy_from_slice(src);
        }
        self.rules.insert(decl, r);
        Ok(())
    }

    /// The live rung's scenarios for each request — exactly the worlds `solve`
    /// will plan over — so a belief can weight them.
    fn sample_roots(&self, reqs: Vec<Req>) -> PyResult<Vec<Vec<(u32, u32, u32, u32)>>> {
        reqs.iter()
            .map(|req| {
                let rec = rec_of(req)?;
                let worlds = planner::root_worlds(&self.cfg, req.1, req.2, &rec);
                Ok(worlds.into_iter().map(|w| (w[0], w[1], w[2], w[3])).collect())
            })
            .collect()
    }

    /// Decide every request: (tile, [(tile, Q)], cells evaluated). `weights[i]`
    /// is external evidence over request i's live scenarios (multiplied into
    /// the built-in tilt), or None. Runs on `threads` OS threads (0 = all
    /// cores) with the GIL released.
    fn solve(
        &self,
        py: Python<'_>,
        reqs: Vec<Req>,
        weights: Vec<Option<Vec<f64>>>,
        threads: usize,
    ) -> PyResult<Vec<(u8, Vec<(u8, f64)>, u64)>> {
        if weights.len() != reqs.len() {
            return Err(PyValueError::new_err("one weights entry per request"));
        }
        let mut jobs = Vec::with_capacity(reqs.len());
        for (req, w) in reqs.iter().zip(&weights) {
            let rec = rec_of(req)?;
            if let Some(w) = w {
                let n = planner::root_worlds(&self.cfg, req.1, req.2, &rec).len();
                if w.len() != n {
                    return Err(PyValueError::new_err(format!(
                        "weights has {} entries but the live rung holds {n} scenarios", w.len()
                    )));
                }
            }
            let rules = self.rules.get(&req.0).ok_or_else(|| {
                PyValueError::new_err(format!("no rules registered for declaration {}", req.0))
            })?;
            jobs.push((rules, req.1, req.2, rec, req.8, req.9, req.10.as_slice(), w.as_deref()));
        }
        let cfg = &self.cfg;
        let out: Mutex<Vec<Option<(u8, Vec<(u8, f64)>, u64)>>> = Mutex::new(vec![None; jobs.len()]);
        let next = AtomicUsize::new(0);
        let workers = if threads == 0 {
            std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1)
        } else {
            threads
        }
        .min(jobs.len().max(1));
        py.allow_threads(|| {
            std::thread::scope(|s| {
                for _ in 0..workers {
                    s.spawn(|| loop {
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        if i >= jobs.len() {
                            break;
                        }
                        let (rules, viewer, own, rec, bid, bt, plays, w) = jobs[i];
                        let r = decide(cfg, rules, viewer, own, &rec, bid, bt, plays, w);
                        out.lock().unwrap()[i] = Some(r);
                    });
                }
            });
        });
        Ok(out.into_inner().unwrap().into_iter().map(|r| r.expect("every job ran")).collect())
    }
}

#[pymodule]
fn scenario_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Core>()?;
    m.add("MAX_WORLDS", MAX_WORLDS)?;
    Ok(())
}
