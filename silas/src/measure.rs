//! Numbers: exact nonnegative weights, the authority every belief and summary
//! carries, and the three-valued comparison.
//!
//! Silas never computes with floating point. A float can enter only through
//! [`Weight::from_f32_exact`] or [`Weight::from_f64_exact`], which return the
//! dyadic rational the float denotes, bit for bit.

use std::collections::BTreeSet;
use std::fmt;
use std::iter::Sum;
use std::num::NonZeroU64;
use std::ops::{Add, AddAssign, Mul};

use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{One, Zero};

use crate::{Result, error};

/// An exact nonnegative rational: a world's mass, a likelihood, a probability.
/// Negative weights cannot be constructed.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct Weight(BigRational);

impl Weight {
    pub fn new(value: BigRational) -> Result<Weight> {
        if value < BigRational::zero() {
            return Err(error(format!("negative weight {value}")));
        }
        Ok(Weight(value))
    }

    pub fn zero() -> Weight {
        Weight(BigRational::zero())
    }

    pub fn one() -> Weight {
        Weight(BigRational::one())
    }

    pub fn integer(n: u128) -> Weight {
        Weight(BigRational::from_integer(BigInt::from(n)))
    }

    pub fn ratio(numerator: u64, denominator: NonZeroU64) -> Weight {
        Weight(BigRational::new(BigInt::from(numerator), BigInt::from(denominator.get())))
    }

    pub fn rational(&self) -> &BigRational {
        &self.0
    }

    pub fn is_zero(&self) -> bool {
        self.0.is_zero()
    }

    /// `self / total`, for a positive `total`.
    pub(crate) fn share_of(&self, total: &Weight) -> Weight {
        assert!(!total.is_zero(), "a share of zero mass");
        Weight(&self.0 / &total.0)
    }

    /// The exact value of a finite, nonnegative `f32`: `m * 2^e` for its
    /// integer significand `m` and exponent `e`. Subnormals are exact too.
    /// NaN, the infinities and negative values are refused; `-0.0` denotes
    /// zero and is accepted as zero.
    pub fn from_f32_exact(x: f32) -> Result<Weight> {
        // 1 sign bit, 8 exponent bits (bias 127), 23 fraction bits.
        let bits = x.to_bits();
        Self::from_binary(bits >> 31 == 1, (bits >> 23) & 0xff, u64::from(bits & 0x7f_ffff), 0xff, 127, 23, x)
    }

    /// The exact value of a finite, nonnegative `f64`; see
    /// [`Weight::from_f32_exact`].
    pub fn from_f64_exact(x: f64) -> Result<Weight> {
        // 1 sign bit, 11 exponent bits (bias 1023), 52 fraction bits.
        let bits = x.to_bits();
        let exponent = ((bits >> 52) & 0x7ff) as u32;
        Self::from_binary(bits >> 63 == 1, exponent, bits & ((1 << 52) - 1), 0x7ff, 1023, 52, x)
    }

    fn from_binary(
        negative: bool,
        exponent: u32,
        fraction: u64,
        all_ones: u32,
        bias: i64,
        fraction_bits: u32,
        shown: impl fmt::Debug,
    ) -> Result<Weight> {
        if exponent == all_ones {
            return Err(error(format!("{shown:?} is not finite")));
        }
        if exponent == 0 && fraction == 0 {
            return Ok(Weight::zero());
        }
        if negative {
            return Err(error(format!("{shown:?} is negative")));
        }
        // Normal numbers carry the implicit leading one; subnormals use the
        // minimum exponent without it.
        let (significand, scale) = if exponent == 0 {
            (fraction, 1 - bias - i64::from(fraction_bits))
        } else {
            (fraction | 1 << fraction_bits, i64::from(exponent) - bias - i64::from(fraction_bits))
        };
        let two_to = |n: i64| BigInt::one() << usize::try_from(n).expect("exponent fits");
        let value = if scale >= 0 {
            BigRational::from_integer(BigInt::from(significand) * two_to(scale))
        } else {
            BigRational::new(BigInt::from(significand), two_to(-scale))
        };
        Ok(Weight(value))
    }
}

impl fmt::Display for Weight {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl fmt::Debug for Weight {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl Add for &Weight {
    type Output = Weight;
    fn add(self, other: &Weight) -> Weight {
        Weight(&self.0 + &other.0)
    }
}

impl AddAssign<&Weight> for Weight {
    fn add_assign(&mut self, other: &Weight) {
        self.0 += &other.0;
    }
}

impl Mul for &Weight {
    type Output = Weight;
    fn mul(self, other: &Weight) -> Weight {
        Weight(&self.0 * &other.0)
    }
}

impl<'a> Sum<&'a Weight> for Weight {
    fn sum<I: Iterator<Item = &'a Weight>>(iter: I) -> Weight {
        iter.fold(Weight::zero(), |total, w| &total + w)
    }
}

/// Where a number's exactness comes from. Ordered from strongest to weakest:
/// `Exact`, then `ExactGiven`, then `Sampled`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Authority {
    /// Uniform or combinatorial support; no model contributed a weight.
    Exact,
    /// The arithmetic is exact, but weights came from a pinned model, named by
    /// a string that carries its content hash, rung and seed. Combined models
    /// are listed, sorted, separated by `" + "`.
    ExactGiven { model: String },
    /// The support was drawn, not enumerated.
    Sampled { n: usize, seed: u64 },
}

impl Authority {
    /// The authority of a number computed from two inputs: the weaker of the
    /// two. It never upgrades: `Exact` results only from two `Exact` inputs.
    /// Two pinned models combine into one label naming both. Two different
    /// samples cannot share one label, so combining them is refused.
    pub fn meet(&self, other: &Authority) -> Result<Authority> {
        use Authority::{Exact, ExactGiven, Sampled};
        Ok(match (self, other) {
            (Exact, x) | (x, Exact) => x.clone(),
            (ExactGiven { model: a }, ExactGiven { model: b }) => {
                let models: BTreeSet<&str> = a.split(" + ").chain(b.split(" + ")).collect();
                ExactGiven { model: models.into_iter().collect::<Vec<_>>().join(" + ") }
            }
            (s @ Sampled { .. }, ExactGiven { .. }) | (ExactGiven { .. }, s @ Sampled { .. }) => s.clone(),
            (a @ Sampled { .. }, b @ Sampled { .. }) if a == b => a.clone(),
            (a, b) => return Err(error(format!("cannot combine two different samples {a:?} and {b:?}"))),
        })
    }
}

/// A number read off a belief, with the authority it was computed under.
/// Only Silas constructs one, so an `Exact` estimate is always exact.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Estimate {
    value: BigRational,
    authority: Authority,
}

impl Estimate {
    pub(crate) fn new(value: BigRational, authority: Authority) -> Estimate {
        Estimate { value, authority }
    }

    pub fn value(&self) -> &BigRational {
        &self.value
    }

    pub fn authority(&self) -> &Authority {
        &self.authority
    }
}

/// The outcome of [`compare`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Verdict {
    Better,
    Worse,
    Indistinguishable,
}

/// Compare `a` against `b` with an explicit band: `a` is `Better` when it
/// exceeds `b` by more than `band`, `Worse` when it falls short by more than
/// `band`, and `Indistinguishable` otherwise. The band must be nonnegative,
/// and zero when both values are `Exact`: exact numbers are never blurred.
pub fn compare(a: &Estimate, b: &Estimate, band: &BigRational) -> Result<Verdict> {
    if *band < BigRational::zero() {
        return Err(error("a comparison band cannot be negative"));
    }
    if a.authority == Authority::Exact && b.authority == Authority::Exact && !band.is_zero() {
        return Err(error("exact values compare with a zero band"));
    }
    let difference = &a.value - &b.value;
    Ok(if difference > *band {
        Verdict::Better
    } else if -difference > *band {
        Verdict::Worse
    } else {
        Verdict::Indistinguishable
    })
}
