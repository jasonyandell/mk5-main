// SPDX-FileCopyrightText: Jason Yandell
// SPDX-License-Identifier: CC0-1.0

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

/// One draw of worlds: how many, from which seeded stream.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Sample {
    pub n: usize,
    pub seed: u64,
}

/// Where a number's exactness comes from: the pinned models whose weights
/// entered it and the samples it was drawn from. No model and no sample is
/// `exact`: uniform or combinatorial support. A model alone is exact
/// arithmetic given that model; a sample means the support was drawn, not
/// enumerated.
///
/// Authorities form a lattice under [`Authority::meet`], the provenance of a
/// number computed from two inputs: the union of their models and of their
/// samples. It is total, commutative, associative and idempotent, `exact` is
/// its unit, and it never upgrades: the meet is at most as strong as either
/// input ([`Authority::at_least`]).
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct Authority {
    models: BTreeSet<String>,
    samples: BTreeSet<Sample>,
}

impl Authority {
    /// No model, no sample.
    pub fn exact() -> Authority {
        Authority::default()
    }

    /// Exact arithmetic on weights from the pinned `model`, named completely:
    /// its content hash, rung and seed.
    pub fn given(model: impl Into<String>) -> Result<Authority> {
        let model = model.into();
        if model.trim().is_empty() {
            return Err(error("a likelihood names its pinned model: content hash, rung and seed"));
        }
        Ok(Authority { models: BTreeSet::from([model]), samples: BTreeSet::new() })
    }

    /// `n` worlds drawn from the stream seeded `seed`.
    pub fn sampled(n: usize, seed: u64) -> Authority {
        Authority { models: BTreeSet::new(), samples: BTreeSet::from([Sample { n, seed }]) }
    }

    pub fn is_exact(&self) -> bool {
        self.models.is_empty() && self.samples.is_empty()
    }

    pub fn is_sampled(&self) -> bool {
        !self.samples.is_empty()
    }

    /// The pinned models the number depends on.
    pub fn models(&self) -> &BTreeSet<String> {
        &self.models
    }

    /// The samples the number was drawn from.
    pub fn samples(&self) -> &BTreeSet<Sample> {
        &self.samples
    }

    /// The authority of a number computed from two inputs: every model and
    /// every sample of either.
    pub fn meet(&self, other: &Authority) -> Authority {
        Authority {
            models: self.models.union(&other.models).cloned().collect(),
            samples: self.samples.union(&other.samples).copied().collect(),
        }
    }

    /// Whether this authority is at least as strong as `other`: it rests on
    /// no model and no sample that `other` does not.
    pub fn at_least(&self, other: &Authority) -> bool {
        self.models.is_subset(&other.models) && self.samples.is_subset(&other.samples)
    }
}

impl fmt::Display for Authority {
    /// `exact`; otherwise `exact_given(a + b)` for the models and
    /// `sampled(n, seed)` for each sample, joined by `and`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_exact() {
            return f.write_str("exact");
        }
        let mut parts = Vec::new();
        if !self.models.is_empty() {
            parts.push(format!("exact_given({})", self.models.iter().cloned().collect::<Vec<_>>().join(" + ")));
        }
        parts.extend(self.samples.iter().map(|s| format!("sampled({}, {})", s.n, s.seed)));
        f.write_str(&parts.join(" and "))
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
/// and zero unless a side rests on a model or a sample: exact numbers are
/// never blurred. The verdict carries the meet of the two authorities.
pub fn compare(a: &Estimate, b: &Estimate, band: &BigRational) -> Result<(Verdict, Authority)> {
    if *band < BigRational::zero() {
        return Err(error("a comparison band cannot be negative"));
    }
    let authority = a.authority.meet(&b.authority);
    if authority.is_exact() && !band.is_zero() {
        return Err(error("exact values compare with a zero band"));
    }
    let difference = &a.value - &b.value;
    let verdict = if difference > *band {
        Verdict::Better
    } else if -difference > *band {
        Verdict::Worse
    } else {
        Verdict::Indistinguishable
    };
    Ok((verdict, authority))
}
