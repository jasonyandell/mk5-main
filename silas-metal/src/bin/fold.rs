// SPDX-FileCopyrightText: Jason Yandell
// SPDX-License-Identifier: CC0-1.0

//! `fold`: one root's rung from its triples lines, in exact integers. A line is `KEY TILE L U
//! MASS …`; a split tile's own line is its pieces' sums, so `TILE/REPLY` pieces are skipped,
//! and so are COUNT and REFUSED lines. When a tile appears twice, the later line stands.
//! Every tile's mass must agree. Settled means one survivor, or Γ = 0.
//!
//! STRATEGY-FUSION FENCE: U is the clairvoyant bound that refutes; the choice reads L only.
//!
//! ```text
//! fold follow-k3-triples.txt "0 2 3"
//! ```

use std::collections::BTreeMap;

use silas::Domino;
use silas_metal::ladder::{Rung, Work};

/// `n / m` to `places` decimals, rounded half up, in integers.
fn ratio(n: u64, m: u64, places: u32) -> String {
    let scale = 10u128.pow(places);
    let q = (u128::from(n) * scale + u128::from(m) / 2) / u128::from(m);
    format!("{}.{:0width$}", q / scale, q % scale, width = places as usize)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (path, key) = (&args[1], format!("{} ", args[2]));
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let mut tiles: BTreeMap<Domino, (u64, u64, u64)> = BTreeMap::new();
    for line in text.lines() {
        let Some(rest) = line.strip_prefix(&key) else { continue };
        let f: Vec<&str> = rest.split(' ').collect();
        if f.len() < 4 || f[0].contains('/') {
            continue;
        }
        let (Ok(tile), Ok(lo), Ok(up), Ok(mass)) =
            (f[0].parse::<Domino>(), f[1].parse::<u64>(), f[2].parse::<u64>(), f[3].parse::<u64>())
        else {
            continue;
        };
        tiles.insert(tile, (lo, up, mass));
    }
    assert!(!tiles.is_empty(), "no tile lines under key {:?}", args[2]);
    let mass = tiles.values().next().unwrap().2;
    assert!(tiles.values().all(|t| t.2 == mass), "the tiles' masses disagree");
    let actions: Vec<(Domino, u64, u64)> = tiles.iter().map(|(t, (lo, up, _))| (*t, *lo, *up)).collect();
    let r = Rung { k: 0, actions, mass, leaves: 0, work: Work::default() };
    let iv: Vec<String> =
        r.actions.iter().map(|(a, lo, up)| format!("{a}[{},{}]", ratio(*lo, mass, 4), ratio(*up, mass, 4))).collect();
    let settled = r.survivors().len() == 1 || r.gamma() == 0;
    println!(
        "{} | tiles {} | choice {} survivors {} gamma {}‰ ({} of {mass}) | {}",
        iv.join(" "),
        r.actions.len(),
        r.choice(),
        r.survivors().len(),
        ratio(r.gamma() * 1000, mass, 1),
        r.gamma(),
        if settled { "settled" } else { "unsettled" }
    );
}
