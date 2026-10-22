// SPDX-FileCopyrightText: Jason Yandell
// SPDX-License-Identifier: CC0-1.0

//! The M3 gate on one root, kept as a test: the h8 carrier replays to its 1,200 worlds and
//! its voids; the lawful solve on Metal equals the oracle's and texas-42's audit value; the
//! world-split control (STRATEGY-FUSION FENCE: the sanctioned measurement of fusion, never a
//! value) equals the oracle's and exceeds H. The full receipt is `cargo run --release --bin m3`.

use num_bigint::BigUint;
use num_rational::BigRational;
use silas::{Context, ContextSet, Pip, Seat};
use silas_metal::Gpu;
use silas_metal::m3::{Arm, carrier, oracle, tile};
use silas_metal::net::{Root, Treatment, solve};

#[test]
fn h8_root_31_passes() {
    let c = carrier();
    assert_eq!(c.worlds.len(), 1200);
    let natural = |p: u8| Context::Natural(Pip::new(p).unwrap());
    assert_eq!(c.voids(Seat::S2), [natural(1), Context::Called].into_iter().collect::<ContextSet>());
    assert_eq!(c.voids(Seat::S3), [natural(4)].into_iter().collect::<ContextSet>());
    assert!(c.voids(Seat::S0).is_empty());

    let mut gpu = Gpu::new().expect("a Metal device");
    let root = c.root(tile("31"));
    let arm = Arm::Tricks;
    let h = solve(&mut gpu, &root, Treatment::Lawful, &|t| arm.utility(t)).unwrap();
    assert_eq!(h.states[1..].iter().sum::<u64>(), 3_286_841);
    let lift = BigUint::from(1u32) << 12;
    assert_eq!(h.value().to_biguint(), BigUint::from(oracle(&c, tile("31"), arm, false).value) * &lift);

    let mut cv = BigUint::from(0u32);
    for part in root.particles.chunks(100) {
        let r = Root { particles: part.to_vec(), ..root.clone() };
        cv += solve(&mut gpu, &r, Treatment::WorldSplit, &|t| arm.utility(t)).unwrap().value().to_biguint();
    }
    assert_eq!(cv, BigUint::from(oracle(&c, tile("31"), arm, true).value) * &lift);

    let den = BigUint::from(1200u32) * BigUint::from(840u32).pow(12);
    let diff = |v: &BigUint| BigRational::new((v * 2u32).into(), den.clone().into()) - BigRational::from_integer(4.into());
    assert_eq!(diff(&h.value().to_biguint()), BigRational::new(490_069.into(), 276_480.into()));
    assert_eq!(diff(&cv), BigRational::new(424_483.into(), 230_400.into()));
    assert!(cv > h.value().to_biguint());
}
