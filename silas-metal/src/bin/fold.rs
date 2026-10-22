// SPDX-FileCopyrightText: Jason Yandell
// SPDX-License-Identifier: CC0-1.0

//! `fold`: one root's rung from its triples lines, in exact integers. A line is `KEY TILE L U
//! MASS …`; a split tile's own line is its pieces' sums, so `TILE/REPLY` pieces count only for
//! a tile with no line of its own (one a reply refused), summed (the next seat's node sums);
//! a refused reply's own line comes from `--per-second`, its `TILE/REPLY/SECOND` pieces are
//! skipped, and so are COUNT and REFUSED lines. When a name appears twice, the later line
//! stands. Every tile's mass must agree, which a missing reply would break. Settled means one
//! survivor, or Γ = 0.
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
    let mut replies: BTreeMap<(Domino, &str), (u64, u64, u64)> = BTreeMap::new();
    for line in text.lines() {
        let Some(rest) = line.strip_prefix(&key) else { continue };
        let f: Vec<&str> = rest.split(' ').collect();
        if f.len() < 4 {
            continue;
        }
        let mut name = f[0].split('/');
        let (Some(Ok(tile)), reply, None, Ok(lo), Ok(up), Ok(mass)) = (
            name.next().map(str::parse::<Domino>),
            name.next(),
            name.next(),
            f[1].parse::<u64>(),
            f[2].parse::<u64>(),
            f[3].parse::<u64>(),
        ) else {
            continue;
        };
        match reply {
            None => tiles.insert(tile, (lo, up, mass)),
            Some(r) => replies.insert((tile, r), (lo, up, mass)),
        };
    }
    let whole: Vec<Domino> = tiles.keys().copied().collect();
    for ((tile, _), (lo, up, mass)) in replies.into_iter().filter(|((t, _), _)| !whole.contains(t)) {
        let t = tiles.entry(tile).or_default();
        *t = (t.0 + lo, t.1 + up, t.2 + mass);
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
