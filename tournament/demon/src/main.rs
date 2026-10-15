// SPDX-FileCopyrightText: Jason Yandell
// SPDX-License-Identifier: CC0-1.0

//! STRATEGY-FUSION FENCE.
//! The Demon: a fenced tournament opponent; measuring stick only.
//! See `wiki/topics/the-demon.md` and `wiki/topics/strategy-fusion-fence.md`.
//!
//! `demon <worlds> <margin-tiebreak 0|1>`: one JSON request per stdin line,
//! one JSON reply per stdout line, in order. A request:
//!
//! ```json
//! {"op": "play", "seat": 1, "hand": [3, 9, 27], "decl": 5, "bid": 31, "bidder": 0,
//!  "plays": [[0, 12], [1, 4]], "seed": 1234, "leader": 0, "points": [0, 0]}
//! ```
//!
//! `hand` is the seat's remaining tiles and `plays` the public record
//! (`leader` and `points`, optional, are the table's, checked against it); the
//! reply is `{"tile": t, "worlds": w, "exact": b, "nodes": k, "ms": x, "moves": [[t, makes, margin|null], ...]}`,
//! or `{"error": "..."}`. `{"op": "hello"}` answers with the configuration.

use std::io::{BufRead, Write};
use std::time::Instant;

use demon::{Config, Request, decide};
use serde_json::{Value, json};

fn int(v: &Value, key: &str) -> Result<u64, String> {
    v.get(key).and_then(Value::as_u64).ok_or_else(|| format!("missing integer {key:?}"))
}

fn request(v: &Value) -> Result<Request, String> {
    let small = |key: &str| int(v, key).and_then(|x| u8::try_from(x).map_err(|_| format!("{key} out of range")));
    let hand = v
        .get("hand")
        .and_then(Value::as_array)
        .ok_or("missing array \"hand\"")?
        .iter()
        .map(|t| t.as_u64().filter(|&t| t < 28).map(|t| 1u32 << t).ok_or("a hand tile is not 0..28"))
        .try_fold(0u32, |m, b| b.map(|b| m | b))?;
    let plays = v
        .get("plays")
        .and_then(Value::as_array)
        .ok_or("missing array \"plays\"")?
        .iter()
        .map(|p| match p.as_array().map(|a| a.as_slice()) {
            Some([s, t]) => match (s.as_u64(), t.as_u64()) {
                (Some(s), Some(t)) if s < 4 && t < 28 => Ok((s as u8, t as u8)),
                _ => Err("a play is [seat 0..4, tile 0..28]"),
            },
            _ => Err("a play is [seat, tile]"),
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Request {
        seat: small("seat")?,
        hand,
        decl: small("decl")?,
        bid: u32::try_from(int(v, "bid")?).map_err(|_| "bid out of range")?,
        bidder: small("bidder")?,
        plays,
        seed: int(v, "seed")?,
    })
}

/// When the table sends its trick leader and banked points, the record the
/// Demon replayed must agree with them.
fn table_agrees(v: &Value, r: &Request) -> Result<Request, String> {
    let (_, rec) = demon::record(r.decl, r.bidder, &r.plays)?;
    if let Some(leader) = v.get("leader").and_then(Value::as_u64) {
        if leader != rec.leader as u64 {
            return Err(format!("the table's leader is {leader}; the record's is {}", rec.leader));
        }
    }
    if let Some(points) = v.get("points").and_then(Value::as_array) {
        let ours: Vec<u64> = rec.banked.iter().map(|&p| p as u64).collect();
        if points.iter().map(Value::as_u64).collect::<Option<Vec<_>>>() != Some(ours.clone()) {
            return Err(format!("the table's points are {points:?}; the record's are {ours:?}"));
        }
    }
    Ok(r.clone())
}

fn answer(cfg: Config, line: &str) -> Value {
    let v: Value = match serde_json::from_str(line) {
        Ok(v) => v,
        Err(e) => return json!({"error": format!("not JSON: {e}")}),
    };
    match v.get("op").and_then(Value::as_str) {
        Some("hello") => json!({"demon": env!("CARGO_PKG_VERSION"), "worlds": cfg.worlds,
                                "margin_tiebreak": cfg.margin_tiebreak}),
        Some("play") => {
            let t0 = Instant::now();
            match request(&v).and_then(|r| table_agrees(&v, &r)).and_then(|r| decide(cfg, &r)) {
                Ok(d) => json!({
                    "tile": d.tile, "worlds": d.worlds, "exact": d.exact, "nodes": d.nodes,
                    "mass": d.mass, "proposals": d.proposals,
                    "ms": t0.elapsed().as_secs_f64() * 1e3,
                    "moves": d.moves.iter().map(|&(t, m, g)| json!([t, m, g])).collect::<Vec<_>>(),
                }),
                Err(e) => json!({"error": e}),
            }
        }
        Some("declaration_model") => match v.get("hand_mask").and_then(Value::as_u64) {
            Some(h) if h < (1 << 28) && h.count_ones() == 7 => {
                json!({"decl": demon::hand_model::heuristic_declaration(h as u32)})
            }
            _ => json!({"error": "hand_mask must contain seven of the 28 tiles"}),
        },
        Some("forecast") => {
            let t0 = Instant::now();
            let run = || -> Result<_, String> {
                let h = u32::try_from(int(&v, "hand_mask")?).map_err(|_| "hand_mask out of range")?;
                let s = u8::try_from(int(&v, "seat")?).map_err(|_| "seat out of range")?;
                let b = u32::try_from(int(&v, "bid")?).map_err(|_| "bid out of range")?;
                let n = usize::try_from(int(&v, "worlds")?).map_err(|_| "worlds out of range")?;
                let declarations = v
                    .get("declarations")
                    .and_then(Value::as_array)
                    .ok_or("forecast requires declarations")?
                    .iter()
                    .map(|d| d.as_u64().and_then(|d| u8::try_from(d).ok()).ok_or("declaration out of range"))
                    .collect::<Result<Vec<_>, _>>()?;
                demon::bidding::forecast(h, s, b, n, int(&v, "seed")?, &declarations)
            };
            match run() {
                Ok(f) => json!({"makes": f.makes, "worlds": f.worlds, "nodes": f.nodes,
                               "ms": t0.elapsed().as_secs_f64() * 1e3}),
                Err(e) => json!({"error":e}),
            }
        }
        _ => json!({"error": "op must be play, hello, forecast or declaration_model"}),
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let cfg = match args.as_slice() {
        [_, n, tb] => Config {
            worlds: n.parse().expect("worlds is a positive integer"),
            margin_tiebreak: tb == "1",
            ..Config::default()
        },
        [_, n, tb, ratio] => Config {
            worlds: n.parse().expect("worlds is a positive integer"),
            margin_tiebreak: tb == "1",
            declaration_weight: ratio.parse().expect("declaration weight is an integer 1..=1024"),
        },
        [_] => Config::default(),
        _ => panic!("usage: demon [<worlds> <margin-tiebreak 0|1> [<declaration-weight>]]"),
    };
    let stdin = std::io::stdin();
    let mut out = std::io::stdout().lock();
    for line in stdin.lock().lines() {
        let line = line.expect("stdin");
        if line.trim().is_empty() {
            continue;
        }
        writeln!(out, "{}", answer(cfg, &line)).expect("stdout");
        out.flush().expect("stdout");
    }
}
