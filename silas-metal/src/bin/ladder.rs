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
//!        [--min-free-gb 10] [--triples k2-triples.txt] [--budget 3000000000000]
//! ```

use std::fs::File;
use std::io::{BufWriter, Write};
use std::time::Instant;

use sha2::{Digest, Sha256};
use silas::player::{Root, SeatLaw};
use silas::reference::Straight42;
use silas::{Auction, Bid, Domino, DominoSet, Frame, Seat};
use silas_metal::Gpu;
use silas_metal::epoch::EpochError;
use silas_metal::ladder::{Gathered, Limits, Problem, Rung, Work, available_bytes};

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
    let min_free: u64 = arg(&args, "--min-free-gb").map_or(0, |s| s.parse::<u64>().unwrap() << 30);
    let limits = Limits {
        table_bits: arg(&args, "--table-bits").unwrap_or("23").parse().unwrap(),
        max_leaves: arg(&args, "--max-leaves").map_or(usize::MAX, |s| s.parse().unwrap()),
        min_free,
    };
    let budget: u64 = arg(&args, "--budget").map_or(u64::MAX, |s| s.parse().unwrap());
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
        // A triples line's key: the deal for a lead root; the deal, the viewer and k for a follow root.
        let key = if at == 0 { d.to_string() } else { format!("{d} {} {k}", viewer.index()) };
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
        // `--count-only` walks each piece keeping no leaves: its plays, the evidence for a play
        // budget, and no rung. A COUNT line per piece, then the rung's verdict on the budget.
        if args.iter().any(|a| a == "--count-only") {
            let (mut spent, mut counted) = (0u64, 0usize);
            for (label, mask) in &pieces {
                let t = Instant::now();
                let (worlds, w) = problem.count(&mut gpu, k, *mask, u32::MAX, from..to, limits.table_bits).unwrap();
                println!(
                    "deal {d} decl P{decl} viewer {} k {k} root {label} counted {from}..{to} worlds {worlds} plays {} (tree {} tails {} god {}) metal wall {} ms",
                    viewer.index(),
                    w.total(),
                    w.tree,
                    w.tails,
                    w.god,
                    t.elapsed().as_millis()
                );
                if let Some(path) = arg(&args, "--triples") {
                    let mut out = std::fs::OpenOptions::new().create(true).append(true).open(path).unwrap();
                    writeln!(
                        out,
                        "{key} {label} COUNT worlds {worlds} plays {} {} {} {} metal",
                        w.total(),
                        w.tree,
                        w.tails,
                        w.god
                    )
                    .unwrap();
                }
                spent += w.total();
                counted += 1;
                // Past the budget is all a refusal needs; the remaining pieces are not walked.
                if spent > budget {
                    break;
                }
            }
            let (verdict, tally) = match (spent > budget, counted == pieces.len()) {
                (false, _) => ("within", String::new()),
                (true, true) => ("past", String::new()),
                (true, false) => ("past", format!(" at least, {counted} of {} tiles counted", pieces.len())),
            };
            println!("  rung k {k}: {spent} plays{tally}, {verdict} the budget {budget}");
            if spent > budget
                && whole
                && let Some(path) = arg(&args, "--triples")
            {
                let mut out = std::fs::OpenOptions::new().create(true).append(true).open(path).unwrap();
                writeln!(out, "{key} REFUSED budget {budget} at k {k}, plays {spent}{tally} (counted)").unwrap();
            }
            continue;
        }
        let mut actions = Vec::new();
        let mut mass = None;
        let mut spent = 0u64;
        // `--per-reply` splits each root tile by the next seat's answer (that seat's node sums,
        // so the pieces' (L, U, mass), leaves and plays add to the tile's); empty pieces are skipped.
        let per_reply = args.iter().any(|a| a == "--per-reply");
        for (tile_label, mask) in &pieces {
            let replies: Vec<(String, u32)> = if per_reply {
                assert_eq!(mask.count_ones(), 1, "--per-reply splits one root tile at a time");
                let answers = (0..28).map(|b| (format!("{tile_label}/{}", Domino::from_index(b).unwrap()), 1u32 << b));
                answers.chain([(format!("{tile_label}/-"), 1u32 << 31)]).collect()
            } else {
                vec![(tile_label.clone(), u32::MAX)]
            };
            // The tile's sums over its pieces: L, U, mass, leaves, work, pieces.
            let mut sum = (0u64, 0u64, 0u64, 0usize, Work::default(), 0usize);
            for (label, reply) in &replies {
                let label = label.as_str();
                let free = available_bytes();
                if free < min_free {
                    println!(
                        "deal {d} decl P{decl} k {k} root {label}: refused, {free} bytes available, under {min_free}"
                    );
                    ok = false;
                    continue;
                }
                let t = Instant::now();
                let g = match problem.gather(&mut gpu, k, *mask, *reply, from..to, limits) {
                    Ok(g) => g,
                    Err(e) => {
                        println!("deal {d} decl P{decl} k {k} root {label}: refused, {e:?}");
                        let why = match e {
                            EpochError::Memory { leaves, budget, shard, plays } => {
                                Some(format!("max-leaves {budget} at shard {shard}, leaves {leaves} plays {plays}"))
                            }
                            EpochError::Free { leaves, available, floor, shard, plays } => Some(format!(
                                "free-memory {available} under {floor} at shard {shard}, leaves {leaves} plays {plays}"
                            )),
                            _ => None,
                        };
                        if let (Some(why), Some(path)) = (why, arg(&args, "--triples")) {
                            let mut out = std::fs::OpenOptions::new().create(true).append(true).open(path).unwrap();
                            writeln!(out, "{key} {label} REFUSED {why}").unwrap();
                        }
                        ok = false;
                        continue;
                    }
                };
                if per_reply && g.worlds == 0 {
                    continue;
                }
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
                // The bytes streamed once: to the file when saved, through SHA-256 (the receipt)
                // and FNV-64 (Exact-T1's key).
                let mut sha = Sha256::new();
                let mut fnv = 0xcbf2_9ce4_8422_2325u64;
                let mut file = arg(&args, "--save").map(|save| {
                    // `{d}` in the path names the deal, `{tile}` the piece's root tiles.
                    BufWriter::new(File::create(save.replace("{d}", &d.to_string()).replace("{tile}", label)).unwrap())
                });
                g.stream(|b| {
                    sha.update(b);
                    fnv = b.iter().fold(fnv, |h, x| (h ^ u64::from(*x)).wrapping_mul(0x0100_0000_01b3));
                    if let Some(f) = file.as_mut() {
                        f.write_all(b).unwrap();
                    }
                });
                if let Some(mut f) = file {
                    f.flush().unwrap();
                }
                let sha: String = sha.finalize().iter().map(|b| format!("{b:02x}")).collect();
                println!("  sha256 {sha} fnv64 {fnv:016x}");
                if whole {
                    let r = problem.rung(&frame, &g).expect("the rung");
                    if let Some(path) = arg(&args, "--triples") {
                        // One line per piece: deal tile L U mass leaves plays tree tails god fnv64 sha256 engine.
                        let mut out = std::fs::OpenOptions::new().create(true).append(true).open(path).unwrap();
                        for (a, lo, up) in &r.actions {
                            let name = if per_reply { label.to_string() } else { a.to_string() };
                            writeln!(
                                out,
                                "{key} {name} {lo} {up} {} {} {} {} {} {} {fnv:016x} {sha} metal",
                                r.mass,
                                g.leaves.len(),
                                g.work.total(),
                                g.work.tree,
                                g.work.tails,
                                g.work.god
                            )
                            .unwrap();
                        }
                    }
                    if per_reply {
                        let (_, lo, up) = r.actions[0];
                        let w = &mut sum.4;
                        *w =
                            Work { tree: w.tree + g.work.tree, tails: w.tails + g.work.tails, god: w.god + g.work.god };
                        sum = (sum.0 + lo, sum.1 + up, sum.2 + r.mass, sum.3 + g.leaves.len(), sum.4, sum.5 + 1);
                    } else {
                        actions.extend(r.actions);
                        mass = Some(r.mass);
                    }
                    spent += g.work.total();
                }
                if let Some(theirs) = arg(&args, "--compare") {
                    let bytes = g.to_bytes();
                    let other = std::fs::read(theirs).unwrap();
                    let same = other == bytes;
                    ok &= same;
                    println!("  byte-equal to {theirs}: {same}");
                    if !same {
                        let o = Gathered::from_bytes(&other).unwrap();
                        println!(
                            "  theirs: worlds {} leaves {} work {:?}; ours work {:?}",
                            o.worlds,
                            o.leaves.len(),
                            o.work,
                            g.work
                        );
                        let diff = o
                            .leaves
                            .iter()
                            .filter(|(p, l)| g.leaves.binary_search_by_key(p, |x| x.0).map(|i| &g.leaves[i].1) != Ok(l))
                            .count();
                        println!("  leaves differing or missing: {diff}");
                    }
                }
            }
            if per_reply && whole && ok {
                let (lo, up, m, leaves, work, n) = sum;
                let a: Domino = tile_label.parse().unwrap();
                println!("  tile {a}: {n} pieces, L {lo} U {up} mass {m} leaves {leaves} plays {}", work.total());
                if let Some(path) = arg(&args, "--triples") {
                    // The tile's line: its pieces' sums, and how many pieces.
                    let mut out = std::fs::OpenOptions::new().create(true).append(true).open(path).unwrap();
                    writeln!(
                        out,
                        "{key} {a} {lo} {up} {m} {leaves} {} {} {} {} split {n} metal",
                        work.total(),
                        work.tree,
                        work.tails,
                        work.god
                    )
                    .unwrap();
                }
                actions.push((a, lo, up));
                mass = Some(m);
            }
        }
        if spent > budget {
            println!("  rung k {k}: refused, {spent} plays past the budget {budget}");
            if let Some(path) = arg(&args, "--triples") {
                let mut out = std::fs::OpenOptions::new().create(true).append(true).open(path).unwrap();
                writeln!(out, "{key} REFUSED budget {budget} at k {k}, plays {spent}").unwrap();
            }
            ok = false;
        } else if let (Some(m), true) = (mass, actions.len() == legal.len()) {
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
