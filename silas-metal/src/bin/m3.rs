// SPDX-FileCopyrightText: Jason Yandell
// SPDX-License-Identifier: CC0-1.0

//! `m3`: run the M3 gate and print its receipt. Every root, both arms; the lawful solve on
//! Metal and on the CPU mirror against the independent oracle; the world-split control on
//! Metal against the oracle's; strict C > H on the trick arm.
//!
//! STRATEGY-FUSION FENCE: the world-split control is the sanctioned measurement of fusion,
//! a negative control, never a value.
//!
//! `m3 [--threads T] [--skip-cpu] [--root 21]`

use std::time::Instant;

use num_bigint::BigUint;
use num_rational::BigRational;
use silas::Seat;
use silas_metal::m3::{Arm, ROOTS, carrier, oracle, tile};
use silas_metal::net::{Root, Solved, Treatment, solve};
use silas_metal::{Backend, Cpu, Gpu, U256};

type Fraction = (i64, i64);

/// The pre-freeze audit's values in texas-42 (orientation, from another codebase's
/// ScalarHidden): root, H, C, for the trick differential.
const AUDIT: [(&str, Fraction, Fraction); 4] = [
    ("21", (2_647_711, 1_036_800), (21_483_353, 8_294_400)),
    ("31", (490_069, 276_480), (424_483, 230_400)),
    ("33", (448_279, 138_240), (6_912_821, 2_073_600)),
    ("55", (6_817_057, 2_073_600), (699_919, 207_360)),
];

fn split(backend: &mut dyn Backend, root: &Root, arm: Arm, chunk: usize) -> U256 {
    let mut total = U256::ZERO;
    for part in root.particles.chunks(chunk) {
        let r = Root { particles: part.to_vec(), ..root.clone() };
        let s = solve(backend, &r, Treatment::WorldSplit, &|t| arm.utility(t)).expect("the world-split solve");
        total = total.checked_add(s.value()).expect("no overflow");
    }
    total
}

fn frac(num: &BigUint, den: &BigUint) -> BigRational {
    BigRational::new(num.clone().into(), den.clone().into())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let flag = |name: &str| args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned();
    let threads: usize = flag("--threads").map(|t| t.parse().expect("threads")).unwrap_or(1);
    let skip_cpu = args.iter().any(|a| a == "--skip-cpu");
    let only = flag("--root");
    let c = carrier();
    println!("carrier h8 before trick 4: P5, S1 declared 30 and leads; hand {:?}", c.frame.kernel().viewer_hand());
    println!(
        "  worlds {}; banked {:?}; voids S2 {:?} S3 {:?} S0 {:?}",
        c.worlds.len(),
        c.frame.banked(),
        c.voids(Seat::S2),
        c.voids(Seat::S3),
        c.voids(Seat::S0)
    );
    let mut gpu = Gpu::new().expect("a Metal device");
    println!("  device {}", gpu.device_name());
    let worlds = BigUint::from(c.worlds.len());
    let den840 = &worlds * BigUint::from(840u32).pow(12);
    let mut ok = true;
    for arm in [Arm::Tricks, Arm::Makes30] {
        for r in ROOTS.iter().filter(|r| only.as_deref().is_none_or(|o| o == **r)) {
            let root = c.root(tile(r));
            let t = Instant::now();
            let h: Solved = solve(&mut gpu, &root, Treatment::Lawful, &|x| arm.utility(x)).expect("the lawful solve");
            let gpu_h = t.elapsed();
            let cpu_h = (!skip_cpu).then(|| {
                let t = Instant::now();
                let s = solve(&mut Cpu { threads }, &root, Treatment::Lawful, &|x| arm.utility(x)).expect("cpu solve");
                (s, t.elapsed())
            });
            let t = Instant::now();
            let oh = oracle(&c, tile(r), arm, false);
            let oracle_h = t.elapsed();
            let t = Instant::now();
            let cv = split(&mut gpu, &root, arm, 100);
            let gpu_c = t.elapsed();
            let t = Instant::now();
            let oc = oracle(&c, tile(r), arm, true);
            let oracle_c = t.elapsed();
            // The oracle's numerators are over 1200 * 420^12; the engine's over 1200 * 840^12.
            let lift = BigUint::from(1u32) << 12;
            let h_ok = h.value().to_biguint() == BigUint::from(oh.value) * &lift;
            let c_ok = cv.to_biguint() == BigUint::from(oc.value) * &lift;
            let cpu_ok = cpu_h.as_ref().is_none_or(|(s, _)| *s == h);
            let states: u64 = h.states[1..].iter().sum();
            let states_ok = states == oh.nodes;
            let hv = frac(&h.value().to_biguint(), &den840);
            let cvv = frac(&cv.to_biguint(), &den840);
            let strict = cvv > hv;
            let (shown_h, shown_c) = match arm {
                Arm::Tricks => {
                    let two = BigRational::from_integer(2.into());
                    let four = BigRational::from_integer(4.into());
                    (&two * &hv - &four, &two * &cvv - &four)
                }
                Arm::Makes30 => (hv.clone(), cvv.clone()),
            };
            println!(
                "{arm:?} root {r}: H {shown_h} C {shown_c} C-H {}  | states {} (levels {:?}) arrivals {} terminals {} items {}",
                &shown_c - &shown_h,
                states,
                h.states,
                h.arrivals,
                h.terminals,
                h.items
            );
            println!(
                "    parity: metal=oracle {h_ok}  cpu=metal {cpu_ok}  C metal=oracle {c_ok}  states=oracle {states_ok} ({})  strict C>H {strict}",
                oh.nodes
            );
            print!("    wall: metal H {gpu_h:?}");
            if let Some((_, w)) = &cpu_h {
                print!("  cpu({threads}) H {w:?}");
            }
            println!("  oracle H {oracle_h:?}  metal C {gpu_c:?}  oracle C {oracle_c:?}");
            if arm == Arm::Tricks {
                let (_, (hn, hd), (cn, cd)) = AUDIT.iter().find(|(a, _, _)| a == r).expect("an audit row");
                let ah = BigRational::new((*hn).into(), (*hd).into());
                let ac = BigRational::new((*cn).into(), (*cd).into());
                println!("    texas-42 audit: H {ah} C {ac}  agree H {} C {}", ah == shown_h, ac == shown_c);
                ok &= ah == shown_h && ac == shown_c && strict;
            }
            ok &= h_ok && c_ok && cpu_ok && states_ok;
        }
    }
    println!("{}", if ok { "M3 GATE: PASS" } else { "M3 GATE: FAIL" });
    if !ok {
        std::process::exit(1);
    }
}
