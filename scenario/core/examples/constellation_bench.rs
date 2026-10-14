//! Recognition speed: what a constellation lookup costs at a late-hand
//! decision, against a plain cache probe on the exact record and against
//! just planning (deployed Zeb: one rung, 512 worlds, floor 16, the field net).
//!
//!     cargo run --release --example constellation_bench -- ../../zeb/net/margin_d250k_h128x64.fnet [deals]
//!
//! Positions come from random legal play of random deals (pip-trump
//! declarations, bid 30), every decision with a real choice from play
//! `lo` on. Prints one JSON line per window.

use std::collections::{HashMap, HashSet};
use std::time::Instant;

use scenario_core::constellation::{seat_form, Contract, Form};
use scenario_core::field::FieldNet;
use scenario_core::planner::{decide, Config, Decision, Draw, Tie, Rec, SplitMix, DRAW_BUDGET};
use scenario_core::rules::Rules;

struct Pos {
    decl: u8,
    viewer: u8,
    own: u32,
    rec: Rec,
    plays: Vec<(u8, u8)>,
    bidder: u8,
}

fn positions(deals: u64, lo: u8) -> Vec<Pos> {
    let mut out = Vec::new();
    for seed in 0..deals {
        let mut rng = SplitMix::new(seed.wrapping_mul(0x9E37) ^ 0x5EED);
        let mut deck: [u8; 28] = std::array::from_fn(|i| i as u8);
        for i in (1..28).rev() {
            deck.swap(i, (rng.next_u64() % (i as u64 + 1)) as usize);
        }
        let mut hands = [0u32; 4];
        for (i, &t) in deck.iter().enumerate() {
            hands[i / 7] |= 1 << t;
        }
        let decl = (rng.next_u64() % 7) as u8;
        let bidder = (rng.next_u64() % 4) as u8;
        let rules = Rules::new(decl).unwrap();
        let mut rec = Rec::opening(bidder);
        let mut plays = Vec::new();
        while rec.plays() < 28 {
            let seat = rec.seat();
            let legal = rules.legal(hands[seat as usize], rec.led());
            if rec.plays() >= lo && legal.count_ones() > 1 {
                out.push(Pos { decl, viewer: seat, own: hands[seat as usize], rec, plays: plays.clone(), bidder });
            }
            let ts: Vec<u8> = (0..28).filter(|t| legal >> t & 1 == 1).collect();
            let t = ts[(rng.next_u64() % ts.len() as u64) as usize];
            hands[seat as usize] &= !(1 << t);
            plays.push((seat, t));
            rec = rec.advance(t, &rules);
        }
    }
    out
}

#[derive(Hash, PartialEq, Eq)]
struct Exact(u8, u8, u32, u64, [u32; 4], u8);

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let net = FieldNet::from_bytes(&std::fs::read(&args[1]).expect("net blob")).expect("a field net");
    let deals: u64 = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(2000);
    let rules: Vec<Rules> = (0..10).map(|d| Rules::new(d).unwrap()).collect();
    let cfg = Config {
        rungs: vec![(512, None, 16)],
        make: true,
        seed: 0,
        field: Some(net),
        field_eps: 0.0,
        draw: Draw::Uniform,
        draw_budget: DRAW_BUDGET,
        convention: None,
        reading: None,
        tie: Tie::Lowest,
    };
    for (lo, label) in [(16u8, "tricks 5-7"), (12u8, "tricks 4-7")] {
        let ps = positions(deals, lo);
        let contract = |p: &Pos| Contract::Make { bid: 30, bidder_team: p.bidder % 2 };

        // Recognition: canonicalize every position, three passes, best pass.
        let mut canon_ns = f64::MAX;
        let mut forms: Vec<Form> = Vec::new();
        for _ in 0..3 {
            let t0 = Instant::now();
            let fs: Vec<Form> = ps
                .iter()
                .map(|p| seat_form(&rules[p.decl as usize], p.viewer, p.own, &p.rec, contract(p)).form)
                .collect();
            canon_ns = canon_ns.min(t0.elapsed().as_nanos() as f64 / ps.len() as f64);
            forms = fs;
        }
        // A constellation table probe (canonicalize + hash lookup) and a plain cache probe.
        let table: HashMap<&Form, u32> = forms.iter().enumerate().map(|(i, f)| (f, i as u32)).collect();
        let exact = |p: &Pos| Exact(p.decl, p.viewer, p.own, p.rec.pack(), p.rec.forbid, p.bidder % 2);
        let plain: HashMap<Exact, u32> = ps.iter().enumerate().map(|(i, p)| (exact(p), i as u32)).collect();
        let t0 = Instant::now();
        let mut hits = 0u64;
        for p in &ps {
            let f = seat_form(&rules[p.decl as usize], p.viewer, p.own, &p.rec, contract(p)).form;
            hits += table.contains_key(&f) as u64;
        }
        let table_ns = t0.elapsed().as_nanos() as f64 / ps.len() as f64;
        let t0 = Instant::now();
        for p in &ps {
            hits += plain.contains_key(&exact(p)) as u64;
        }
        let plain_ns = t0.elapsed().as_nanos() as f64 / ps.len() as f64;
        assert_eq!(hits, 2 * ps.len() as u64);

        // Just planning: deployed Zeb on a stratified subsample.
        let sample: Vec<&Pos> = ps.iter().step_by((ps.len() / 400).max(1)).collect();
        let t0 = Instant::now();
        for p in &sample {
            let d = Decision {
                rules: &rules[p.decl as usize],
                viewer: p.viewer,
                own: p.own,
                rec: p.rec,
                bid: 30,
                bidder: p.bidder,
                plays: &p.plays,
                auction: None,
            };
            std::hint::black_box(decide(&cfg, &d, None));
        }
        let plan_us = t0.elapsed().as_micros() as f64 / sample.len() as f64;

        let distinct_forms: HashSet<&Form> = forms.iter().collect();
        let mut tiles = 0usize;
        for f in &forms {
            tiles += f.words.len() - 1;
        }
        println!(
            "{{\"window\":\"{label}\",\"positions\":{},\"distinct_exact\":{},\"distinct_forms\":{},\"mean_tiles\":{:.1},\
             \"canon_ns\":{:.0},\"table_probe_ns\":{:.0},\"plain_probe_ns\":{:.0},\"plan_us\":{:.0},\"plan_sample\":{}}}",
            ps.len(),
            plain.len(),
            distinct_forms.len(),
            tiles as f64 / forms.len() as f64,
            canon_ns,
            table_ns,
            plain_ns,
            plan_us,
            sample.len()
        );
    }
}
