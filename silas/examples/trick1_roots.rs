// SPDX-FileCopyrightText: Jason Yandell
// SPDX-License-Identifier: CC0-1.0

//! Trick-1 roots from a recorded pour, through the focal-horizon ladder
//! (`silas::player::trick1`). Reads the pour's deal columns as raw `.npy` files (extract
//! them from `pour.npz` with `unzip`): `hands.npy` (u32 [D, 4]), `decl.npy` (i8 [D]),
//! `plays.npy` (i8 [D, 28, 2]). One line per root and horizon.
//!
//! STRATEGY-FUSION FENCE: the ladder's upper side is clairvoyant, a bound only; the line
//! reports it as `U` and never chooses by it.
//!
//! ```text
//! # whole rungs, k = 0..=1, in one run
//! cargo run --release --example trick1_roots -- DIR --deals 0,1 --viewer 0 --k 1 --threads 8
//! # a rung in pieces across capped runs: gather shard ranges (and, with `--root 6-5`,
//! # one root tile at a time), then load them all
//! cargo run --release --example trick1_roots -- DIR --deals 0 --k 1 --from 0 --to 20000 --save a.bin
//! cargo run --release --example trick1_roots -- DIR --deals 0 --load a.bin,b.bin,...
//! ```

use std::time::Instant;

use silas::player::trick1::{Gathered, Ladder, Rung, Work};
use silas::player::{Root, SeatLaw};
use silas::reference::Straight42;
use silas::{Auction, Bid, Domino, DominoSet, Frame, Seat};

fn npy(path: &str) -> Vec<u8> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    assert_eq!(&bytes[..6], b"\x93NUMPY", "{path} is not .npy");
    let header = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
    bytes[10 + header..].to_vec()
}

/// `n / m` to `places` decimals, rounded half up, in integers.
fn fixed(n: u64, m: u64, places: u32) -> String {
    let scale = 10u128.pow(places);
    let x = (2 * u128::from(n) * scale + u128::from(m)) / (2 * u128::from(m));
    format!("{}.{:0w$}", x / scale, x % scale, w = places as usize)
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
    let kmax: usize = arg(&args, "--k").unwrap_or("0").parse().unwrap();
    let kmin: usize = arg(&args, "--kmin").unwrap_or("0").parse().unwrap();
    let threads: usize = arg(&args, "--threads").unwrap_or("8").parse().unwrap();
    let budget = arg(&args, "--budget").unwrap_or("1e15").parse::<f64>().unwrap() as u64;
    let law = match arg(&args, "--law").unwrap_or("lowest") {
        "lowest" => SeatLaw::Lowest,
        "highest" => SeatLaw::Highest,
        x => panic!("unknown law {x}"),
    };
    for &d in &deals {
        let decl = decls[d] as usize;
        assert!(decl < 7, "deal {d} declares {decl}, not a pip suit");
        let physics = Straight42::ALL[decl].physics();
        let dealt = |s: usize| DominoSet::from_bits(hands[4 * d + s]).unwrap();
        let record: Vec<(usize, Domino)> = (0..28)
            .map(|i| (plays[56 * d + 2 * i] as usize, Domino::from_index(plays[56 * d + 2 * i + 1] as usize).unwrap()))
            .collect();
        // The viewer's first decision: its play in trick 1.
        let at = record.iter().position(|(s, _)| *s == viewer.index()).unwrap();
        assert!(at < 4, "the viewer plays in trick 1");
        let prefix: Vec<Domino> = record[..at].iter().map(|(_, t)| *t).collect();
        let chosen = record[at].1;
        let auction = Auction::won_by(Seat::S0, Bid::Points(30)).unwrap();
        let frame = Frame::new(auction, physics.clone(), viewer, dealt(viewer.index()), prefix).unwrap();
        // The lead reads the exact posterior (nothing is observed yet); a follower's belief
        // reads the record through its voids (the bidder's recorded lead has probability
        // zero under the declared law).
        let root = if at == 0 { Root::Posterior } else { Root::FiberUniform };
        let ladder = Ladder::new(&frame, [law; 4], root).unwrap();
        let own = physics.legal_plays(dealt(viewer.index()), frame.lead());
        let law_tile = match law {
            SeatLaw::Lowest => own.iter().next().unwrap(),
            _ => own.iter().last().unwrap(),
        };
        let head = format!("deal {d} decl P{decl} viewer {} shards {}", viewer.index(), ladder.shards());
        // Pieces merged into one gather (one root-tile set), saved to `--save`.
        if let (Some(files), Some(save)) = (arg(&args, "--load"), arg(&args, "--save")) {
            let g = files
                .split(',')
                .map(|f| Gathered::from_bytes(&std::fs::read(f).unwrap()).unwrap())
                .reduce(|a, b| a.merge(b).unwrap())
                .unwrap();
            std::fs::write(save, g.to_bytes()).unwrap();
            println!(
                "{head} k {} merged worlds {} leaves {} plays {} -> {save}",
                g.k,
                g.worlds,
                g.leaves(),
                g.work.total()
            );
            continue;
        }
        // A piece of a rung: the shards `--from..--to` of horizon `--k`, saved to `--save`.
        if let Some(save) = arg(&args, "--save") {
            let from: usize = arg(&args, "--from").unwrap_or("0").parse().unwrap();
            let to = arg(&args, "--to").map_or(ladder.shards(), |s| s.parse::<usize>().unwrap().min(ladder.shards()));
            let t = Instant::now();
            let roots = match arg(&args, "--root") {
                Some(t) => 1 << t.parse::<Domino>().unwrap().index(),
                None => ladder.menu(),
            };
            let g = ladder.gather(kmax, roots, from, to, threads);
            std::fs::write(save, g.to_bytes()).unwrap();
            println!(
                "{head} k {kmax} gathered {from}..{to} worlds {} leaves {} plays {} (tree {} tails {} god {}) wall {:.1}s -> {save}",
                g.worlds,
                g.leaves(),
                g.work.total(),
                g.work.tree,
                g.work.tails,
                g.work.god,
                t.elapsed().as_secs_f64()
            );
            continue;
        }
        // A rung from another engine's per-tile counts (`--triples FILE --k K`): lines
        // `deal tile L U mass leaves plays tree tails god …`; a tile split into pieces has one
        // line per piece (`deal tile/reply …`), and its counts are their sums (the next seat's
        // node sums). Other lines (refusals, history) are skipped. Refused past `--budget`.
        if let Some(file) = arg(&args, "--triples") {
            let mut tiles: Vec<(Domino, [u64; 7])> = Vec::new();
            for line in std::fs::read_to_string(file).unwrap().lines() {
                let mut w: Vec<&str> = line.split_whitespace().collect();
                // A follow root's line carries its viewer after the deal (`d v tile …`), and past
                // k = 2 its horizon too (`d v k tile …`).
                if let Some(v) = w.get(1).and_then(|s| s.parse::<usize>().ok()) {
                    if v != viewer.index() {
                        continue;
                    }
                    w.remove(1);
                    if let Some(k) = w.get(1).and_then(|s| s.parse::<usize>().ok()) {
                        if k != kmax {
                            continue;
                        }
                        w.remove(1);
                    }
                } else if viewer.index() != 0 {
                    continue;
                }
                let Some(n) =
                    w.get(2..10).and_then(|x| x.iter().map(|s| s.parse::<u64>().ok()).collect::<Option<Vec<u64>>>())
                else {
                    continue;
                };
                // A split tile's own summary line repeats its pieces' sums: read the pieces.
                if w[0].parse::<usize>() != Ok(d) || w.contains(&"split") {
                    continue;
                }
                let tile: Domino = w[1].split('/').next().unwrap().parse().unwrap();
                // L, U, mass, leaves, tree, tails, god (plays is their sum).
                let add = [n[0], n[1], n[2], n[3], n[5], n[6], n[7]];
                match tiles.iter_mut().find(|t| t.0 == tile) {
                    Some(t) => t.1.iter_mut().zip(add).for_each(|(a, b)| *a += b),
                    None => tiles.push((tile, add)),
                }
            }
            tiles.sort_unstable_by_key(|t| t.0);
            let menu: Vec<Domino> = own.iter().collect();
            let mass = tiles.first().map(|t| t.1[2]);
            let sum = |i: usize| tiles.iter().map(|t| t.1[i]).sum();
            let work = Work { tree: sum(4), tails: sum(5), god: sum(6) };
            let rung = if tiles.iter().map(|t| t.0).collect::<Vec<_>>() != menu {
                Err(format!(
                    "the triples hold {:?}, not the menu {menu:?}",
                    tiles.iter().map(|t| t.0).collect::<Vec<_>>()
                ))
            } else if tiles.iter().any(|t| Some(t.1[2]) != mass) {
                Err("the tiles' masses differ".to_string())
            } else if work.total() > budget {
                Err(format!("refused: horizon {kmax} took {} plays, past {budget}", work.total()))
            } else {
                Ok(Rung {
                    k: kmax,
                    actions: tiles.iter().map(|t| (t.0, t.1[0], t.1[1])).collect(),
                    mass: mass.unwrap(),
                    leaves: tiles.iter().map(|t| t.1[3] as usize).sum(),
                    work,
                })
            };
            match rung {
                Ok(r) => print_rung(&head, &r, law_tile, chosen, 0.0, args.iter().any(|a| a == "--counts"), d),
                Err(e) => println!("{head} k {kmax} {e}"),
            }
            continue;
        }
        // The rung from saved gathers, or whole in this run.
        let rungs: Vec<_> = match arg(&args, "--load") {
            Some(files) => {
                let t = Instant::now();
                // One whole gather per file (merge pieces first with `--load … --save`), read
                // one at a time as the rung consumes them; the root tiles join in the rung.
                let gathers = files.split(',').map(|f| Gathered::from_bytes(&std::fs::read(f).unwrap()).unwrap());
                let rung = ladder.rung(gathers, budget);
                vec![(rung.as_ref().map_or(kmax, |r| r.k), rung, t.elapsed().as_secs_f64())]
            }
            None => (kmin..=kmax)
                .map(|k| {
                    let t = Instant::now();
                    (k, ladder.solve(k, threads, budget), t.elapsed().as_secs_f64())
                })
                .collect(),
        };
        for (k, rung, wall) in rungs {
            match rung {
                Ok(r) => print_rung(&head, &r, law_tile, chosen, wall, args.iter().any(|a| a == "--counts"), d),
                Err(e) => println!("{head} k {k} {} wall {wall:.1}s", e.0),
            }
        }
    }
}

/// One rung's line; with `counts`, first each root tile's exact (L, U, mass).
fn print_rung(head: &str, r: &Rung, law_tile: Domino, chosen: Domino, wall: f64, counts: bool, d: usize) {
    let m = r.mass;
    if counts {
        for (a, lo, up) in &r.actions {
            println!("{d} {a} {lo} {up} {m}");
        }
    }
    let iv: Vec<String> =
        r.actions.iter().map(|(a, lo, up)| format!("{a}[{},{}]", fixed(*lo, m, 4), fixed(*up, m, 4))).collect();
    println!(
        "{head} k {} worlds {} | {} | choice {} survivors {} gamma {}‰ ({}) settled {} | law {law_tile} walt {chosen} | leaves {} plays {} (tree {} tails {} god {}) wall {wall:.1}s",
        r.k,
        r.mass,
        iv.join(" "),
        r.choice(),
        r.survivors().len(),
        fixed(1000 * r.gamma(), m, 1),
        r.gamma(),
        r.settled().map(|x| x.to_string()).unwrap_or("-".into()),
        r.leaves,
        r.work.total(),
        r.work.tree,
        r.work.tails,
        r.work.god,
    );
}
