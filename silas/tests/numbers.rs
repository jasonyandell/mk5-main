// SPDX-FileCopyrightText: Jason Yandell
// SPDX-License-Identifier: CC0-1.0

//! Weights, floats, authority and the three-valued comparison.
mod common;

use common::*;
use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{One, Signed};
use silas::*;

/// `m * 2^e`, exactly.
fn dyadic(m: u64, e: i32) -> BigRational {
    let two = |n: i32| BigInt::one() << usize::try_from(n).unwrap();
    if e >= 0 {
        BigRational::from_integer(BigInt::from(m) * two(e))
    } else {
        BigRational::new(BigInt::from(m), two(-e))
    }
}

fn f64_weight(x: f64) -> BigRational {
    Weight::from_f64_exact(x).unwrap().rational().clone()
}

fn f32_weight(x: f32) -> BigRational {
    Weight::from_f32_exact(x).unwrap().rational().clone()
}

#[test]
fn f64_converts_to_the_exact_dyadic_it_denotes() {
    assert_eq!(f64_weight(0.0), dyadic(0, 0));
    assert_eq!(f64_weight(-0.0), dyadic(0, 0));
    assert_eq!(f64_weight(1.0), dyadic(1, 0));
    assert_eq!(f64_weight(0.5), dyadic(1, -1));
    assert_eq!(f64_weight(3.0), dyadic(3, 0));
    // 0.1 is not one tenth: it is the nearest double, 0x1.999999999999ap-4.
    assert_eq!(f64_weight(0.1), dyadic(0x0019_9999_9999_999a, -56));
    assert_ne!(f64_weight(0.1), r(1, 10));
    // The extremes: the largest finite, the smallest normal, and the
    // smallest and largest subnormals.
    assert_eq!(f64_weight(f64::MAX), dyadic((1 << 53) - 1, 971));
    assert_eq!(f64_weight(f64::MIN_POSITIVE), dyadic(1, -1022));
    assert_eq!(f64_weight(f64::from_bits(1)), dyadic(1, -1074));
    assert_eq!(f64_weight(f64::from_bits((1 << 52) - 1)), dyadic((1 << 52) - 1, -1074));
    // Bits just above one: 1 + 2^-52.
    assert_eq!(f64_weight(f64::from_bits(1.0f64.to_bits() + 1)), dyadic((1 << 52) + 1, -52));
}

#[test]
fn f32_converts_to_the_exact_dyadic_it_denotes() {
    assert_eq!(f32_weight(0.0), dyadic(0, 0));
    assert_eq!(f32_weight(-0.0), dyadic(0, 0));
    assert_eq!(f32_weight(0.75), dyadic(3, -2));
    assert_eq!(f32_weight(0.1), dyadic(13_421_773, -27));
    assert_eq!(f32_weight(f32::MAX), dyadic((1 << 24) - 1, 104));
    assert_eq!(f32_weight(f32::MIN_POSITIVE), dyadic(1, -126));
    assert_eq!(f32_weight(f32::from_bits(1)), dyadic(1, -149));
    assert_eq!(f32_weight(f32::from_bits((1 << 23) - 1)), dyadic((1 << 23) - 1, -149));
}

#[test]
fn widening_an_f32_never_changes_its_exact_value() {
    // Widening f32 to f64 is exact, so both conversions must agree, bit
    // pattern by bit pattern, across every exponent and a spread of fractions.
    for exponent in 0u32..0xff {
        for fraction in [0, 1, 2, 0x40_0000, 0x55_5555, 0x7f_fffe, 0x7f_ffff] {
            let x = f32::from_bits(exponent << 23 | fraction);
            assert_eq!(f32_weight(x), f64_weight(f64::from(x)), "{:#x}", x.to_bits());
        }
    }
}

#[test]
fn non_finite_and_negative_floats_are_refused() {
    for x in
        [f64::NAN, -f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -1.0, -f64::MIN_POSITIVE, -f64::from_bits(1), f64::MIN]
    {
        assert!(Weight::from_f64_exact(x).is_err(), "{x:?}");
    }
    for x in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -1.0, -f32::from_bits(1), f32::MIN] {
        assert!(Weight::from_f32_exact(x).is_err(), "{x:?}");
    }
}

#[test]
fn authority_never_upgrades() {
    let exact = Authority::exact();
    let a = given("model-a");
    let b = given("model-b");
    let sampled = Authority::sampled(1000, 7);
    assert_eq!(exact.meet(&exact), Authority::exact());
    assert_eq!(exact.meet(&a), a);
    assert_eq!(a.meet(&exact), a);
    assert_eq!(a.meet(&a), a);
    let both = a.meet(&b);
    assert_eq!(both.models().iter().map(String::as_str).collect::<Vec<_>>(), ["model-a", "model-b"]);
    assert_eq!(both.to_string(), "exact_given(model-a + model-b)");
    assert_eq!(b.meet(&a), both);
    assert_eq!(both.meet(&a), both);
    assert_eq!(sampled.meet(&exact), sampled);
    assert_eq!(sampled.meet(&sampled), sampled);
    // A sampled number from a pinned model names both: the model's identity
    // is kept through the meet (it was dropped before).
    let drawn = a.meet(&sampled);
    assert_eq!((drawn.models(), drawn.samples()), (a.models(), sampled.samples()));
    assert_eq!(drawn.to_string(), "exact_given(model-a) and sampled(1000, 7)");
    // Two different samples combine into a set of samples (they were refused
    // before): the meet is total.
    let other = Authority::sampled(1000, 8);
    assert_eq!(sampled.meet(&other).samples().len(), 2);
    assert!(Authority::given(" ").is_err());
}

/// The meet is a semilattice with `exact` its unit, and never upgrades: a
/// number computed from two inputs is at most as strong as either.
#[test]
fn authority_is_a_provenance_lattice() {
    let atoms =
        [Authority::exact(), given("model-a"), given("model-b"), Authority::sampled(10, 1), Authority::sampled(10, 2)];
    let mut all = atoms.to_vec();
    for x in &atoms {
        for y in &atoms {
            all.push(x.meet(y));
        }
    }
    for x in &all {
        assert_eq!(x.meet(x), *x);
        assert_eq!(x.meet(&Authority::exact()), *x);
        assert!(x.at_least(x) && Authority::exact().at_least(x));
        assert_eq!(x.is_exact(), *x == Authority::exact());
        for y in &all {
            let m = x.meet(y);
            assert_eq!(m, y.meet(x));
            assert!(x.at_least(&m) && y.at_least(&m), "{x} {y}");
            assert_eq!(m.is_exact(), x.is_exact() && y.is_exact());
            for z in &all {
                assert_eq!(m.meet(z), x.meet(&y.meet(z)));
            }
        }
    }
}

#[test]
fn authority_propagates_through_conditioning_and_stepping() {
    let exact_prior = uniform();
    let event = compile(HAS_COUNT);
    let (posterior, p) = exact_prior.condition(&event, "partner has count", &mut budget()).unwrap();
    assert_eq!((posterior.authority(), p.authority()), (&Authority::exact(), &Authority::exact()));
    let (soft, p) = posterior.condition_likelihood("soft", "model-a", |_| Ok(w(1, 2))).unwrap();
    assert_eq!((soft.authority(), p.authority()), (&given("model-a"), &given("model-a")));
    // Once a model has entered, nothing returns to Exact.
    let (again, p) = soft.condition(&event, "again", &mut budget()).unwrap();
    assert_eq!((again.authority(), p.authority()), (&given("model-a"), &given("model-a")));
    let lead = ObservedPlay { actor: Seat::S0, tile: tile("6-6"), class: PlayClass::Lead };
    let stepped = step_belief(&again, lead, "led", 90, &mut budget()).unwrap();
    assert_eq!(stepped.belief.authority(), &given("model-a"));
    let summary = compile(COUNTS).summarize(&stepped.belief, &mut budget()).unwrap();
    assert_eq!(summary.authority, given("model-a"));
    assert_eq!(summary.event_probability().authority(), &given("model-a"));
    // A drawn sample stays a sample, whatever exact arithmetic follows, and
    // a model that enters it is named beside it.
    let worlds: Vec<_> = exact_prior.worlds().take(4).map(|(w, _)| (*w, Weight::one())).collect();
    let sampled = Authority::sampled(4, 11);
    let drawn = Belief::from_weights(exact_prior.frame().clone(), "four draws", sampled.clone(), worlds, 4).unwrap();
    let (_, p) = drawn.condition_likelihood("soft", "model-a", |_| Ok(w(1, 3))).unwrap();
    assert!(p.authority().is_sampled());
    assert_eq!(p.authority(), &sampled.meet(&given("model-a")));
}

#[test]
fn comparison_is_three_valued_with_an_explicit_band() {
    let b = uniform();
    let counts = compile(HAS_COUNT).summarize(&b, &mut budget()).unwrap().event_probability(); // 4/5
    let holder = compile("(query (roles) (out) (pattern (holds S2 5-5)))").summarize(&b, &mut budget()).unwrap();
    let five = holder.event_probability(); // 1/3
    assert_eq!(compare(&counts, &five, &r(0, 1)).unwrap(), (Verdict::Better, Authority::exact()));
    assert_eq!(compare(&five, &counts, &r(0, 1)).unwrap().0, Verdict::Worse);
    assert_eq!(compare(&five, &five, &r(0, 1)).unwrap().0, Verdict::Indistinguishable);
    // Exact numbers are never blurred, and no band is negative.
    assert!(compare(&counts, &five, &r(1, 2)).is_err());
    assert!(compare(&counts, &five, &r(-1, 2)).is_err());
    // A sampled estimate may carry a band.
    let worlds: Vec<_> = b.worlds().take(10).map(|(w, _)| (*w, Weight::one())).collect();
    let drawn = Belief::from_weights(b.frame().clone(), "ten draws", Authority::sampled(10, 3), worlds, 10).unwrap();
    let sampled = compile(HAS_COUNT).summarize(&drawn, &mut budget()).unwrap().event_probability();
    assert_eq!(sampled.authority(), &Authority::sampled(10, 3));
    let gap = (sampled.value() - counts.value()).abs();
    assert!(gap > r(0, 1), "ten draws do not reproduce 4/5 exactly");
    assert_ne!(compare(&sampled, &counts, &r(0, 1)).unwrap().0, Verdict::Indistinguishable);
    // The verdict carries the meet: a sampled number against an exact one is
    // a sampled verdict.
    assert_eq!(compare(&sampled, &counts, &gap).unwrap(), (Verdict::Indistinguishable, Authority::sampled(10, 3)));
    // A model's number may carry a band too; a model and a sample meet.
    let soft = drawn.condition_likelihood("soft", "model-a", |_| Ok(w(1, 2))).unwrap().0;
    let modeled = compile(HAS_COUNT).summarize(&soft, &mut budget()).unwrap().event_probability();
    let (_, met) = compare(&modeled, &counts, &r(1, 100)).unwrap();
    assert_eq!(met, given("model-a").meet(&Authority::sampled(10, 3)));
}

#[test]
fn weights_are_never_negative_and_likelihoods_never_exceed_one() {
    assert!(Weight::new(r(-1, 3)).is_err());
    assert_eq!(Weight::new(r(2, 6)).unwrap(), w(1, 3));
    assert!(uniform().condition_likelihood("too likely", "model", |_| Ok(w(3, 2))).is_err());
}
