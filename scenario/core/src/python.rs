//! The Python module `scenario_core` (the `python` feature), which
//! `scenario/compiled.py` drives.
//!
//! Interface: `Core(rungs, utility, seed, tilt, eps, field)` with rungs as
//! (n_worlds, horizon, floor) and `field` the field net's blob
//! (`FieldNet.to_bytes()`) or None; `sample_roots(reqs)` for the belief seam;
//! `solve(reqs, weights, threads)`, which runs a batch of decisions in
//! parallel with the GIL released; `analyze` (the same, opened up: the
//! scenarios and each root action's per-scenario outcomes); and `mold`, the
//! per-layer position counts of one deal's perfect-information tree.
//! `field_scores` runs the field net alone, and the module functions `rules`
//! and `canon` hand back the core's own tables, for the parity tests.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::field::{self, FieldNet};
use crate::mold;
use crate::planner::{self, analyze, decide, Config, Decision, Rec, TiltPlays, MAX_WORLDS};
use crate::rules::{Rules, N_DECLS};

/// One decision request:
/// (decl, viewer, own, played, leader, trick, (banked0, banked1),
///  (forbid0, forbid1, forbid2, forbid3), bid, bidder_team, plays as (seat, tile))
type Req = (u8, u8, u32, u32, u8, Vec<u8>, (u8, u8), (u32, u32, u32, u32), i32, u8, Vec<(u8, u8)>);

/// One request ready to plan, with its external weights.
type Job<'a> = (Decision<'a>, Option<&'a [f64]>);

/// A scenario: the four remaining hands as tile masks, absolute seats.
type World = (u32, u32, u32, u32);

/// A decision as Python receives it: (tile, [(tile, Q)], cells evaluated).
type Decided = (u8, Vec<(u8, f64)>, u64);

/// An analysis as Python receives it: (tile, [(tile, Q)], cells evaluated,
/// worlds, weights, [(tile, per-world outcomes)]).
type Analyzed = (u8, Vec<(u8, f64)>, u64, Vec<World>, Vec<f64>, Vec<(u8, Vec<f64>)>);

/// One declaration's tables: (led_suit, follow_bits, rank, count).
type Tables = (Vec<u8>, Vec<u32>, Vec<Vec<u8>>, Vec<u8>);

#[pyclass]
struct Core {
    cfg: Config,
    rules: Vec<Rules>, // by declaration id
}

fn workers(threads: usize, jobs: usize) -> usize {
    let all = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    (if threads == 0 { all } else { threads }).min(jobs.max(1))
}

/// `f` over every job on `threads` OS threads with the GIL released, results in job order.
fn run<T: Send>(py: Python<'_>, jobs: &[Job], threads: usize, f: impl Fn(&Job) -> T + Sync) -> Vec<T> {
    let out: Mutex<Vec<Option<T>>> = Mutex::new((0..jobs.len()).map(|_| None).collect());
    let next = AtomicUsize::new(0);
    py.allow_threads(|| {
        std::thread::scope(|s| {
            for _ in 0..workers(threads, jobs.len()) {
                s.spawn(|| loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    if i >= jobs.len() {
                        break;
                    }
                    let r = f(&jobs[i]);
                    out.lock().unwrap()[i] = Some(r);
                });
            }
        });
    });
    out.into_inner().unwrap().into_iter().map(|r| r.expect("every job ran")).collect()
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
    #[pyo3(signature = (rungs, utility, seed, tilt, eps, field=None))]
    fn new(
        rungs: Vec<(usize, Option<usize>, usize)>,
        utility: &str,
        seed: u64,
        tilt: &str,
        eps: f64,
        field: Option<&[u8]>,
    ) -> PyResult<Self> {
        if rungs.is_empty() {
            return Err(PyValueError::new_err("a player needs at least one rung"));
        }
        if rungs.iter().any(|&(n, _, _)| n == 0 || n > MAX_WORLDS) {
            return Err(PyValueError::new_err(format!("n_worlds must be in 1..={MAX_WORLDS}")));
        }
        if rungs.iter().any(|&(_, _, f)| f == 0) {
            return Err(PyValueError::new_err("a cell floor is at least 1"));
        }
        if rungs[0].2 > 1 && field.is_none() {
            return Err(PyValueError::new_err("rung 0 hands a floored move down only to a field net"));
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
        let field = field.map(FieldNet::from_bytes).transpose().map_err(|e| PyValueError::new_err(e.to_string()))?;
        let rules = (0..N_DECLS).map(|d| Rules::new(d).expect("every declaration below N_DECLS has rules")).collect();
        Ok(Core { cfg: Config { rungs, make, seed, tilt, eps, field }, rules })
    }

    /// The live rung's scenarios for each request — exactly the worlds `solve`
    /// will plan over — so a belief can weight them.
    fn sample_roots(&self, reqs: Vec<Req>) -> PyResult<Vec<Vec<World>>> {
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
    ) -> PyResult<Vec<Decided>> {
        let jobs = self.jobs(&reqs, &weights)?;
        let cfg = &self.cfg;
        Ok(run(py, &jobs, threads, |(d, w)| decide(cfg, d, *w)))
    }

    /// `solve`, opened up: per request (tile, [(tile, Q)], cells evaluated,
    /// the live worlds, their weights, [(tile, per-world outcomes)]) — see
    /// `scenario.planner.Analysis`.
    fn analyze(
        &self,
        py: Python<'_>,
        reqs: Vec<Req>,
        weights: Vec<Option<Vec<f64>>>,
        threads: usize,
    ) -> PyResult<Vec<Analyzed>> {
        let jobs = self.jobs(&reqs, &weights)?;
        let cfg = &self.cfg;
        Ok(run(py, &jobs, threads, |(d, w)| {
            let a = analyze(cfg, d, *w);
            let worlds = a.worlds.into_iter().map(|w| (w[0], w[1], w[2], w[3])).collect();
            (a.tile, a.q, a.nodes, worlds, a.weights, a.outcomes)
        }))
    }

    /// The field net alone on each request's position: (tile, the 28
    /// canonical scores) — the referee's `FieldNet.move` and `scores`.
    fn field_scores(&self, reqs: Vec<Req>) -> PyResult<Vec<(u8, Vec<i64>)>> {
        let net = self.cfg.field.as_ref().ok_or_else(|| PyValueError::new_err("no field net"))?;
        reqs.iter()
            .map(|req| {
                let rules = self.rules_for(req.0)?;
                if !net.plays(rules.decl) {
                    return Err(PyValueError::new_err("the field net plays pip-trump declarations only"));
                }
                let rec = rec_of(req)?;
                let bt = req.9 as usize;
                let z = net.scores(&net.features(rules, req.1, req.2, &rec, req.8, bt));
                Ok((net.decide(rules, req.1, req.2, &rec, req.8, bt), z))
            })
            .collect()
    }

    /// Positions per layer of one deal's perfect-information tree, layers
    /// 0..=max_layer (default 27: the terminal layer is left out), with
    /// transposition on the whole public record. `hands` are the four dealt
    /// hands as tile masks, absolute seats.
    #[pyo3(signature = (decl, hands, leader, max_layer=None, threads=0))]
    fn mold(
        &self,
        py: Python<'_>,
        decl: u8,
        hands: (u32, u32, u32, u32),
        leader: u8,
        max_layer: Option<usize>,
        threads: usize,
    ) -> PyResult<Vec<u64>> {
        let rules = self.rules_for(decl)?;
        let hands = [hands.0, hands.1, hands.2, hands.3];
        let dealt = hands.iter().fold(0u32, |m, &h| m | h);
        if dealt != planner::ALL_TILES || hands.iter().any(|h| h.count_ones() != 7) || leader > 3 {
            return Err(PyValueError::new_err("a deal is four disjoint seven-tile hands and a leader in 0..4"));
        }
        let last = max_layer.unwrap_or(27);
        if last > 28 {
            return Err(PyValueError::new_err("a hand has layers 0..=28"));
        }
        let n = workers(threads, usize::MAX);
        Ok(py.allow_threads(|| mold::mold(rules, hands, leader, last, n)))
    }
}

impl Core {
    fn rules_for(&self, decl: u8) -> PyResult<&Rules> {
        self.rules.get(decl as usize).ok_or_else(|| PyValueError::new_err(format!("no declaration {decl}")))
    }

    fn jobs<'a>(&'a self, reqs: &'a [Req], weights: &'a [Option<Vec<f64>>]) -> PyResult<Vec<Job<'a>>> {
        if weights.len() != reqs.len() {
            return Err(PyValueError::new_err("one weights entry per request"));
        }
        let mut jobs = Vec::with_capacity(reqs.len());
        for (req, w) in reqs.iter().zip(weights) {
            let rec = rec_of(req)?;
            if let Some(w) = w {
                let n = planner::root_worlds(&self.cfg, req.1, req.2, &rec).len();
                if w.len() != n {
                    return Err(PyValueError::new_err(format!(
                        "weights has {} entries but the live rung holds {n} scenarios",
                        w.len()
                    )));
                }
            }
            let d = Decision {
                rules: self.rules_for(req.0)?,
                viewer: req.1,
                own: req.2,
                rec,
                bid: req.8,
                bidder_team: req.9,
                plays: &req.10,
            };
            jobs.push((d, w.as_deref()));
        }
        Ok(jobs)
    }
}

/// The core's tables for one declaration.
#[pyfunction]
fn rules(decl: u8) -> PyResult<Tables> {
    let r = Rules::new(decl).ok_or_else(|| PyValueError::new_err(format!("no declaration {decl}")))?;
    Ok((r.led_suit.to_vec(), r.follow_bits.to_vec(), r.rank.iter().map(|row| row.to_vec()).collect(), r.count.to_vec()))
}

/// The field net's tile relabeling for one declaration, or None where the net does not play.
#[pyfunction]
fn canon(decl: u8) -> Option<Vec<u8>> {
    field::canon(decl).map(|c| c.to_vec())
}

#[pymodule]
fn scenario_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Core>()?;
    m.add_function(wrap_pyfunction!(rules, m)?)?;
    m.add_function(wrap_pyfunction!(canon, m)?)?;
    m.add("MAX_WORLDS", MAX_WORLDS)?;
    Ok(())
}
