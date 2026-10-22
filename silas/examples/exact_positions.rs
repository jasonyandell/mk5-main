// SPDX-FileCopyrightText: Jason Yandell
// SPDX-License-Identifier: CC0-1.0

//! Positions for `silas exact` from a Walt pour, one JSON object a line.
//!
//! ```text
//! cargo run --release --example exact_positions -- DIR probe          # the probe's positions
//! cargo run --release --example exact_positions -- DIR own N A:B NAME # decisions of deals A..B with N own tiles
//! ```
//!
//! DIR holds the pour's `hands.npy`, `decl.npy`, `plays.npy` (as `unzip -p pour.npz` writes
//! them), its `manifest.json`, and for `probe` the probe's rows (`probe.jsonl`,
//! `wiki/experiments/walt-endgame-probe-2026-10-22.md`). A probe position carries Walt's tile,
//! its maximizing set under Walt's model, the fiber and Walt's CPU seconds, as data: the
//! exact player never reads them.

use std::path::Path;

use serde_json::{Value, json};

fn npy(path: &Path) -> (Vec<u8>, String) {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    assert_eq!(&bytes[..6], b"\x93NUMPY", "{}: not an .npy file", path.display());
    let (len, start) = match bytes[6] {
        1 => (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10),
        _ => (u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize, 12),
    };
    let header = String::from_utf8_lossy(&bytes[start..start + len]).to_string();
    assert!(header.contains("'fortran_order': False"), "{}: C order only", path.display());
    (bytes[start + len..].to_vec(), header)
}

struct Pour {
    hands: Vec<[u32; 4]>,
    decl: Vec<i8>,
    plays: Vec<[[i8; 2]; 28]>,
    bidder: u64,
    bid: u64,
}

fn pour(dir: &Path) -> Pour {
    let (h, hh) = npy(&dir.join("hands.npy"));
    assert!(hh.contains("'<u4'"), "hands are little-endian u32");
    let (d, dh) = npy(&dir.join("decl.npy"));
    let (p, ph) = npy(&dir.join("plays.npy"));
    assert!(dh.contains("'|i1'") && ph.contains("'|i1'"), "decl and plays are int8");
    let hands = h
        .chunks_exact(16)
        .map(|c| std::array::from_fn(|i| u32::from_le_bytes([c[4 * i], c[4 * i + 1], c[4 * i + 2], c[4 * i + 3]])))
        .collect::<Vec<[u32; 4]>>();
    let decl = d.iter().map(|&b| b as i8).collect::<Vec<_>>();
    let plays = p.chunks_exact(56).map(|c| std::array::from_fn(|i| [c[2 * i] as i8, c[2 * i + 1] as i8])).collect();
    assert_eq!((hands.len(), decl.len()), (decl.len(), hands.len()));
    let man: Value = serde_json::from_str(&std::fs::read_to_string(dir.join("manifest.json")).unwrap()).unwrap();
    Pour { hands, decl, plays, bidder: man["bidder"].as_u64().unwrap(), bid: man["bid"].as_u64().unwrap() }
}

fn position(pour: &Pour, id: String, deal: usize, n: usize) -> Value {
    json!({
        "id": id, "decl": pour.decl[deal], "bidder": pour.bidder, "bid": pour.bid,
        "hands": pour.hands[deal],
        "plays": pour.plays[deal][..n].iter().map(|p| [p[0], p[1]]).collect::<Vec<_>>(),
    })
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let dir = Path::new(&args[0]);
    let pour = pour(dir);
    match args[1].as_str() {
        "probe" => {
            for line in std::fs::read_to_string(dir.join("probe.jsonl")).unwrap().lines().filter(|l| !l.is_empty()) {
                let r: Value = serde_json::from_str(line).unwrap();
                let (deal, n) = (r["deal"].as_u64().unwrap() as usize, r["n_plays"].as_u64().unwrap() as usize);
                let mut v = position(&pour, r["id"].as_str().unwrap().to_string(), deal, n);
                v["walt"] = json!({
                    "tile": r["tile"], "best": r["best"], "menu": r["menu"], "fiber": r["fiber"],
                    "sampled": r["sampled"], "own": r["own"], "cpu_live_s": r["cpu_live_s"],
                    "cpu_exact_s": r["cpu_exact_s"],
                });
                println!("{v}");
            }
        }
        "own" => {
            let own: usize = args[2].parse().unwrap();
            let (a, b) = args[3].split_once(':').unwrap();
            let (a, b): (usize, usize) = (a.parse().unwrap(), b.parse().unwrap());
            for deal in a..b.min(pour.hands.len()) {
                // The plays of trick 8 - own: the actor holds `own` tiles. Kept when it has
                // more than one legal tile.
                for n in 4 * (7 - own)..4 * (8 - own) {
                    let v = position(&pour, format!("{}/{deal}/{n}", args[4]), deal, n);
                    let frame = silas::player::file::Position::from_json(&v.to_string()).unwrap().frame;
                    let hand = frame.dealt().difference(frame.played());
                    if frame.physics().legal_plays(hand, frame.lead()).len() > 1 {
                        println!("{v}");
                    }
                }
            }
        }
        m => panic!("no mode {m}"),
    }
}
