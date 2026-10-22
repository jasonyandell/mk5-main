// SPDX-FileCopyrightText: Jason Yandell
// SPDX-License-Identifier: CC0-1.0

//! The pre-registered summary of `silas exact` rows (PREREG-S0, PREREG-S1 in the desk folder).
//!
//! ```text
//! cargo run --release --example exact_report -- [--p 1/2] FILE.jsonl ...
//! ```
//!
//! Descriptive statistics only, in floating point: medians, Wilson intervals, the
//! calibration z, deal-bootstrap intervals. Nothing here feeds a decision; the player's own
//! numbers stay exact, and an exact rational is converted only to be summarized. A file's
//! cost line from `scripts/cpu-slot` is read from `FILE.jsonl.cost` when it exists.
#![allow(clippy::float_arithmetic)]

use std::collections::BTreeMap;

use serde_json::Value;

fn frac(s: &str) -> f64 {
    match s.split_once('/') {
        Some((n, d)) => n.parse::<f64>().unwrap() / d.parse::<f64>().unwrap(),
        None => s.parse().unwrap(),
    }
}

fn quantiles(mut v: Vec<f64>) -> String {
    if v.is_empty() {
        return "n 0".into();
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let at = |q: f64| v[((v.len() - 1) as f64 * q).floor() as usize];
    format!("n {} median {:.4e} p90 {:.4e} max {:.4e}", v.len(), at(0.5), at(0.9), v[v.len() - 1])
}

fn wilson(k: usize, n: usize) -> String {
    if n == 0 {
        return "0/0".into();
    }
    let (k, n, z) = (k as f64, n as f64, 1.959964);
    let p = k / n;
    let c = (p + z * z / (2.0 * n)) / (1.0 + z * z / n);
    let h = z * (p * (1.0 - p) / n + z * z / (4.0 * n * n)).sqrt() / (1.0 + z * z / n);
    format!("{}/{} = {:.4} [{:.4}, {:.4}]", k as u64, n as u64, p, c - h, c + h)
}

struct SplitMix(u64);
impl SplitMix {
    fn below(&mut self, n: usize) -> usize {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        ((z ^ (z >> 31)) % n as u64) as usize
    }
}

/// A statistic of per-deal groups, with its deal-bootstrap 95% percentile interval
/// (4,000 resamples, seed 0).
fn bootstrap(groups: &[Vec<(f64, f64)>], stat: impl Fn(&[(f64, f64)]) -> f64) -> String {
    let all: Vec<(f64, f64)> = groups.iter().flatten().copied().collect();
    if all.is_empty() {
        return "n 0".into();
    }
    let mut rng = SplitMix(0);
    let mut s: Vec<f64> = (0..4000)
        .map(|_| {
            let pick: Vec<(f64, f64)> =
                (0..groups.len()).flat_map(|_| groups[rng.below(groups.len())].iter().copied()).collect();
            stat(&pick)
        })
        .collect();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    format!("{:.4} [{:.4}, {:.4}] (n {})", stat(&all), s[100], s[3899], all.len())
}

// Over the positions where Walt's recorded time is above its timer's resolution (1 ms); the
// zeros are counted beside it.
fn median_ratio(v: &[(f64, f64)]) -> f64 {
    let mut r: Vec<f64> = v.iter().filter(|(_, b)| *b > 0.0).map(|(a, b)| a / b).collect();
    r.sort_by(|a, b| a.partial_cmp(b).unwrap());
    if r.is_empty() { f64::NAN } else { r[(r.len() - 1) / 2] }
}

fn total_ratio(v: &[(f64, f64)]) -> f64 {
    v.iter().map(|x| x.0).sum::<f64>() / v.iter().map(|x| x.1).sum::<f64>()
}

#[derive(Default)]
struct Cell {
    rows: usize,
    decided: usize,
    choices: Vec<f64>,
    reads: Vec<f64>,
    wall_ms: Vec<f64>,
    recounted: usize,
    q: BTreeMap<String, Vec<f64>>,
    thresholds: Vec<f64>,
    switches: Vec<f64>,
    constant: usize,
    act_moves: usize,
    undecided: usize,
    no_quotient: usize,
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let files: Vec<&String> = args.iter().filter(|a| !a.starts_with("--") && a.ends_with(".jsonl")).collect();
    let mut cells: BTreeMap<String, Cell> = BTreeMap::new();
    let mut hands: BTreeMap<String, (usize, usize, usize, f64, f64, usize, usize)> = BTreeMap::new();
    let mut walt: BTreeMap<String, BTreeMap<String, Vec<(f64, f64)>>> = BTreeMap::new();
    let mut shard_ins: Vec<(f64, usize)> = Vec::new();
    let mut walt_rows: BTreeMap<String, Value> = BTreeMap::new();
    for path in &args {
        if let Some(p) = path.strip_prefix("--walt=") {
            for line in std::fs::read_to_string(p).unwrap().lines().filter(|l| !l.is_empty()) {
                let v: Value = serde_json::from_str(line).unwrap();
                walt_rows.insert(v["id"].as_str().unwrap().to_string(), v["walt"].clone());
            }
        }
    }
    for path in files {
        let text = std::fs::read_to_string(path).unwrap();
        let mut decisions_here = 0usize;
        let mut first: Option<f64> = None;
        for line in text.lines().filter(|l| !l.is_empty()) {
            let v: Value = serde_json::from_str(line).unwrap();
            let id = v["id"].as_str().unwrap_or("").to_string();
            let pour = id.split('/').next().unwrap_or("").to_string();
            if v["kind"] == "hand" {
                let h = hands.entry(pour).or_default();
                let (dec, con) =
                    (v["decisions"].as_u64().unwrap() as usize, v["consistent"].as_u64().unwrap() as usize);
                let s = v["success"].as_bool().unwrap();
                // A hand with no decision was decided before the viewer's first turn: its value
                // was its outcome, so it adds nothing to the calibration's sums.
                let v1 = first.take().unwrap_or(s as u8 as f64);
                h.0 += 1;
                h.1 += s as usize;
                h.2 += (dec == 0 || con + 1 == dec) as usize;
                h.6 += (dec == 0) as usize;
                h.3 += s as u8 as f64 - v1;
                h.4 += v1 * (1.0 - v1);
                h.5 += dec;
                continue;
            }
            decisions_here += 1;
            let classes = v["classes"].as_array().unwrap();
            let top = |c: &Value| {
                c["value"].as_array().unwrap().iter().map(|x| frac(x.as_str().unwrap())).fold(0.0, f64::max)
            };
            if first.is_none() {
                first = Some(top(&classes[0]));
            }
            let own = v["own"].as_u64().unwrap();
            let c = cells.entry(format!("{pour} own {own}")).or_default();
            c.rows += 1;
            let choices = v["counts"]["choices"].as_f64().unwrap();
            c.decided += (choices == 1.0) as usize;
            c.choices.push(choices);
            c.reads.push(v["counts"]["reads"].as_f64().unwrap());
            let wall = v["wall_ns"].as_f64().unwrap() / 1e6;
            c.wall_ms.push(wall);
            c.recounted += (v["recounted"].as_u64().unwrap_or(0) > 0) as usize;
            if choices > 1.0 {
                c.undecided += 1;
                c.thresholds.push(v["thresholds"].as_array().unwrap().len() as f64);
                let bests: Vec<&Value> = classes.iter().map(|c| &c["best"]).collect();
                let switches = bests.windows(2).filter(|w| w[0] != w[1]).count();
                c.switches.push(switches as f64);
                c.constant += (switches == 0) as usize;
                let least = |b: &Value| b.as_array().unwrap()[0].clone();
                let act = &v["tile"];
                c.act_moves += (*act != least(bests[0]) || *act != least(bests[bests.len() - 1])) as usize;
                match v.get("quotient") {
                    Some(q) => {
                        for reach in ["every", "plan"] {
                            for iface in ["raw", "fork"] {
                                c.q.entry(format!("{reach}/{iface}"))
                                    .or_default()
                                    .push(q[reach][iface]["states"].as_f64().unwrap());
                            }
                        }
                    }
                    None => c.no_quotient += 1,
                }
            }
            if let Some(w) = walt_rows.get(&id) {
                let deal = id.split('/').nth(1).unwrap_or("").to_string();
                let by = walt.entry(format!("{pour} own {own}")).or_default();
                by.entry(format!("live:{deal}")).or_default().push((wall / 1e3, w["cpu_live_s"].as_f64().unwrap()));
                by.entry(format!("exact:{deal}")).or_default().push((wall / 1e3, w["cpu_exact_s"].as_f64().unwrap()));
            }
        }
        if let Ok(cost) = std::fs::read_to_string(format!("{path}.cost")) {
            // `/usr/bin/time -l` counts the solver ("N instructions retired"); without it,
            // cpu-slot's own count is the solver's, since cpu-slot then runs it directly.
            let words: Vec<&str> = cost.split_whitespace().collect();
            let ins = match words.windows(3).find(|w| w[1] == "instructions" && w[2] == "retired") {
                Some(w) => Some(w[0]),
                None => words.windows(2).find(|w| w[0] == "instructions").map(|w| w[1]),
            };
            if let (Some(ins), true) = (ins, decisions_here > 0) {
                shard_ins.push((ins.parse::<f64>().unwrap(), decisions_here));
            }
        }
    }
    for (name, c) in &cells {
        println!(
            "== {name}: {} decisions, recounted {}, decided at the root {}",
            c.rows,
            c.recounted,
            wilson(c.decided, c.rows)
        );
        println!("  choices   {}", quantiles(c.choices.clone()));
        println!("  reads     {}", quantiles(c.reads.clone()));
        println!("  wall ms   {} (descriptive)", quantiles(c.wall_ms.clone()));
        if c.undecided > 0 {
            println!("  undecided {}: course thresholds {}", c.undecided, quantiles(c.thresholds.clone()));
            println!("    root switch points {}", quantiles(c.switches.clone()));
            println!("    root set constant in p {}", wilson(c.constant, c.undecided));
            println!("    act at --p differs from an endpoint's {}", wilson(c.act_moves, c.undecided));
            for (k, v) in &c.q {
                println!("    controller {k} {}", quantiles(v.clone()));
            }
            if c.no_quotient > 0 {
                println!("    controller not computed at {}", c.no_quotient);
            }
        }
    }
    for (pour, (n, s, cons, diff, var, dec, settled)) in &hands {
        println!(
            "== hands {pour}: {n} hands ({settled} decided before the first turn), {dec} decisions; G2 consistent on {cons}; success {}; G7 mean(success - v1) {:.4}, z {:.3}",
            wilson(*s, *n),
            diff / *n as f64,
            diff / var.sqrt()
        );
    }
    if !shard_ins.is_empty() {
        let (ins, dec) = shard_ins.iter().fold((0.0, 0usize), |(i, d), (x, n)| (i + x, d + n));
        println!(
            "== instructions over {} processes: {:.4e} for {dec} decisions, {:.4e} per decision (pooled)",
            shard_ins.len(),
            ins,
            ins / dec as f64
        );
    }
    for (name, by) in &walt {
        for arm in ["live", "exact"] {
            let groups: Vec<Vec<(f64, f64)>> =
                by.iter().filter(|(k, _)| k.starts_with(arm)).map(|(_, v)| v.clone()).collect();
            let zeros = groups.iter().flatten().filter(|(_, b)| *b == 0.0).count();
            println!(
                "== {name} ours / Walt cpu_{arm}_s: median ratio {} ({zeros} positions where Walt's time reads 0 left out)",
                bootstrap(&groups, median_ratio)
            );
            println!("   total ratio {}", bootstrap(&groups, total_ratio));
        }
    }
}
