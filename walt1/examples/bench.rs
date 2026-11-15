// SPDX-FileCopyrightText: Jason Yandell
// SPDX-License-Identifier: CC0-1.0

//! Time `handle` in-process over the requests of a corpus.jsonl (one JSON object per line
//! with a "req" string). usage: cargo run --release --example bench -- corpus.jsonl
use std::time::Instant;

fn main() {
    let path = std::env::args().nth(1).expect("corpus.jsonl path");
    let text = std::fs::read_to_string(path).unwrap();
    let mut times = Vec::new();
    let mut minds = 0u64;
    for line in text.lines() {
        let start = line.find("\"req\":\"").unwrap() + 7;
        let end = start + line[start..].find('"').unwrap();
        let req = line[start..end].replace("\\n", "\n");
        let parsed = walt::parse(&req).unwrap();
        let t = Instant::now();
        let d = walt::decide(&parsed).unwrap();
        if !d.forced {
            times.push(t.elapsed().as_secs_f64() * 1e3);
            minds += d.minds;
        }
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = times.len();
    println!(
        "decisions {n}  mean {:.2} ms  median {:.2}  p90 {:.2}  max {:.2}  minds/decision {}",
        times.iter().sum::<f64>() / n as f64,
        times[n / 2],
        times[n * 9 / 10],
        times[n - 1],
        minds / n as u64
    );
}
