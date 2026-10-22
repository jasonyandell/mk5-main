// SPDX-FileCopyrightText: Jason Yandell
// SPDX-License-Identifier: CC0-1.0

//! `ladder`: Exact-T1's trick-1 roots (`silas/examples/trick1_roots.rs`), gathered on Metal.
//! Same pour columns, same frames, same shard ranges; `--save` writes Exact-T1's gather bytes,
//! which its `--load` reads unchanged. `--per-root` gathers each root tile alone (a k = 2
//! rung's leaves do not fit at once) and joins the tiles' actions into the rung.
//!
//! STRATEGY-FUSION FENCE: the gather's upper column is clairvoyant, a bound only.
//!
//! ```text
//! ladder DIR --deals 0,1 --viewer 0 --k 0 [--from 0 --to 1000] [--root 6-5 | --per-root]
//!        [--save deal{d}-{tile}.bin] [--compare their.bin] [--table-bits 23] [--max-leaves N]
//! ```

use std::time::Instant;

use silas::player::{Root, SeatLaw};
use silas::reference::Straight42;
use silas::{Auction, Bid, Domino, DominoSet, Frame, Seat};
use silas_metal::Gpu;
use silas_metal::ladder::{Gathered, Limits, Problem, Rung, Work};

fn npy(path: &str) -> Vec<u8> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    assert_eq!(&bytes[..6], b"\x93NUMPY", "{path} is not .npy");
    let header = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
    bytes[10 + header..].to_vec()
}

/// `n / m` to `places` decimals, rounded half up, in integers.
fn ratio(n: u64, m: u64, places: u32) -> String {
    let scale = 10u128.pow(places);
    let q = (u128::from(n) * scale + u128::from(m) / 2) / u128::from(m);
    format!("{}.{:0width$}", q / scale, q % scale, width = places as usize)
}

fn arg<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    args.iter().position(|a| a == name).map(|i| args[i + 1].as_str())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dir = &args[1];
    let hands: Vec<u32> =
        npy(&format!("{dir}/hands.npy")).chunks(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    let decls: Vec<i8> = npy(&format!("{dir}/decl.npy")).into_iter().map(|b| b as i8).collect();
    let plays: Vec<i8> = npy(&format!("{dir}/plays.npy")).into_iter().map(|b| b as i8).collect();
    let deals: Vec<usize> = arg(&args, "--deals").unwrap_or("0").split(',').map(|s| s.parse().unwrap()).collect();
    let viewer = Seat::ALL[arg(&args, "--viewer").unwrap_or("0").parse::<usize>().unwrap()];
    let k: usize = arg(&args, "--k").unwrap_or("0").parse().unwrap();
    let limits = Limits {
        table_bits: arg(&args, "--table-bits").unwrap_or("23").parse().unwrap(),
        max_leaves: arg(&args, "--max-leaves").map_or(usize::MAX, |s| s.parse().unwrap()),
    };
    let law = match arg(&args, "--law").unwrap_or("lowest") {
        "lowest" => SeatLaw::Lowest,
        "highest" => SeatLaw::Highest,
        x => panic!("unknown law {x}"),
    };
    let mut gpu = Gpu::new().expect("a Metal device");
    let mut ok = true;
    for &d in &deals {
        let decl = decls[d] as usize;
        assert!(decl < 7, "deal {d} declares {decl}, not a pip suit");
        let physics = Straight42::ALL[decl].physics();
        let dealt = |s: usize| DominoSet::from_bits(hands[4 * d + s]).unwrap();
        let record: Vec<(usize, Domino)> = (0..28)
            .map(|i| (plays[56 * d + 2 * i] as usize, Domino::from_index(plays[56 * d + 2 * i + 1] as usize).unwrap()))
            .collect();
        let at = record.iter().position(|(s, _)| *s == viewer.index()).unwrap();
        assert!(at < 4, "the viewer plays in trick 1");
        let prefix: Vec<Domino> = record[..at].iter().map(|(_, t)| *t).collect();
        let auction = Auction::won_by(Seat::S0, Bid::Points(30)).unwrap();
        let frame = Frame::new(auction, physics.clone(), viewer, dealt(viewer.index()), prefix).unwrap();
        let root = if at == 0 { Root::Posterior } else { Root::FiberUniform };
        let problem = Problem::new(&frame, [law; 4], root).unwrap();
        let from: usize = arg(&args, "--from").unwrap_or("0").parse().unwrap();
        let to = arg(&args, "--to").map_or(problem.shards(), |s| s.parse::<usize>().unwrap().min(problem.shards()));
        let whole = from == 0 && to == problem.shards();
        // The pieces: every root tile at once, the listed ones, or each legal one alone.
        let legal = physics.legal_plays(dealt(viewer.index()), frame.lead());
        let pieces: Vec<(String, u32)> = if args.iter().any(|a| a == "--per-root") {
            legal.iter().map(|t| (t.to_string(), 1u32 << t.index())).collect()
        } else if let Some(list) = arg(&args, "--root") {
            let mask = list.split(',').fold(0u32, |m, t| m | 1 << t.parse::<Domino>().unwrap().index());
            vec![(list.replace(',', "+"), mask)]
        } else {
            vec![("all".to_string(), u32::MAX)]
        };
        let mut actions = Vec::new();
        let mut mass = None;
        for (label, mask) in &pieces {
            let t = Instant::now();
            let g = match problem.gather(&mut gpu, k, *mask, from, to, limits) {
                Ok(g) => g,
                Err(e) => {
                    println!("deal {d} decl P{decl} k {k} root {label}: refused, {e:?}");
                    ok = false;
                    continue;
                }
            };
            let ms = t.elapsed().as_millis().max(1);
            println!(
                "deal {d} decl P{decl} viewer {} shards {} k {k} root {label} gathered {from}..{to} worlds {} leaves {} plays {} (tree {} tails {} god {}) metal wall {ms} ms ({} plays/s)",
                viewer.index(),
                problem.shards(),
                g.worlds,
                g.leaves.len(),
                g.work.total(),
                g.work.tree,
                g.work.tails,
                g.work.god,
                u128::from(g.work.total()) * 1000 / ms,
            );
            if whole {
                let r = problem.rung(&frame, &g).expect("the rung");
                actions.extend(r.actions);
                mass = Some(r.mass);
            }
            let bytes = g.to_bytes();
            if let Some(save) = arg(&args, "--save") {
                // `{d}` in the path names the deal, `{tile}` the piece's root tiles.
                std::fs::write(save.replace("{d}", &d.to_string()).replace("{tile}", label), &bytes).unwrap();
            }
            if let Some(theirs) = arg(&args, "--compare") {
                let other = std::fs::read(theirs).unwrap();
                let same = other == bytes;
                ok &= same;
                println!("  byte-equal to {theirs}: {same}");
                if !same {
                    let o = Gathered::from_bytes(&other).unwrap();
                    println!("  theirs: worlds {} leaves {} work {:?}; ours work {:?}", o.worlds, o.leaves.len(), o.work, g.work);
                    let diff = o.leaves.iter().filter(|(p, l)| g.leaves.get(p) != Some(l)).count();
                    println!("  leaves differing or missing: {diff}");
                }
            }
        }
        if let (Some(m), true) = (mass, actions.len() == legal.len()) {
            actions.sort_by_key(|a| a.0);
            let r = Rung { k, actions, mass: m, leaves: 0, work: Work::default() };
            let iv: Vec<String> =
                r.actions.iter().map(|(a, lo, up)| format!("{a}[{},{}]", ratio(*lo, m, 4), ratio(*up, m, 4))).collect();
            println!(
                "  rung k {k}: {} | choice {} survivors {} gamma {}‰ | exact: {:?}",
                iv.join(" "),
                r.choice(),
                r.survivors().len(),
                ratio(r.gamma() * 1000, m, 1),
                r.actions.iter().map(|(a, lo, up)| format!("{a}:{lo}/{up}")).collect::<Vec<_>>(),
            );
        }
    }
    if !ok {
        std::process::exit(1);
    }
}
