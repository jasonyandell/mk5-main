// SPDX-FileCopyrightText: Jason Yandell
// SPDX-License-Identifier: CC0-1.0

//! `U256`: an exact unsigned mass in eight little-endian 32-bit limbs, the layout the
//! shaders use. Every operation is checked; overflow is an error, never a wrap.

use std::cmp::Ordering;
use std::fmt;

use num_bigint::BigUint;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct U256(pub [u32; 8]);

impl U256 {
    pub const ZERO: U256 = U256([0; 8]);
    pub const ONE: U256 = U256([1, 0, 0, 0, 0, 0, 0, 0]);

    pub fn from_u128(v: u128) -> U256 {
        U256(std::array::from_fn(|i| if i < 4 { (v >> (32 * i)) as u32 } else { 0 }))
    }

    pub fn is_zero(self) -> bool {
        self.0.iter().all(|w| *w == 0)
    }

    pub fn checked_add(self, other: U256) -> Option<U256> {
        let mut out = [0u32; 8];
        let mut carry = 0u64;
        for (i, w) in out.iter_mut().enumerate() {
            let s = u64::from(self.0[i]) + u64::from(other.0[i]) + carry;
            *w = s as u32;
            carry = s >> 32;
        }
        (carry == 0).then_some(U256(out))
    }

    pub fn checked_mul_small(self, f: u32) -> Option<U256> {
        let mut out = [0u32; 8];
        let mut carry = 0u64;
        for (i, w) in out.iter_mut().enumerate() {
            let p = u64::from(self.0[i]) * u64::from(f) + carry;
            *w = p as u32;
            carry = p >> 32;
        }
        (carry == 0).then_some(U256(out))
    }

    pub fn checked_mul_u64(self, f: u64) -> Option<U256> {
        self.checked_mul_small(f as u32)?
            .checked_add(self.checked_mul_small((f >> 32) as u32)?.checked_shl32()?)
    }

    /// Times 2^32.
    fn checked_shl32(self) -> Option<U256> {
        (self.0[7] == 0).then(|| U256(std::array::from_fn(|i| if i == 0 { 0 } else { self.0[i - 1] })))
    }

    pub fn to_biguint(self) -> BigUint {
        BigUint::from_slice(&self.0)
    }

    pub fn from_biguint(v: &BigUint) -> Option<U256> {
        let digits = v.to_u32_digits();
        (digits.len() <= 8).then(|| U256(std::array::from_fn(|i| digits.get(i).copied().unwrap_or(0))))
    }
}

impl Ord for U256 {
    fn cmp(&self, other: &U256) -> Ordering {
        self.0.iter().rev().cmp(other.0.iter().rev())
    }
}

impl PartialOrd for U256 {
    fn partial_cmp(&self, other: &U256) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl fmt::Debug for U256 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_biguint())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arithmetic_agrees_with_bigint() {
        let mut x = U256::ONE;
        let mut big = BigUint::from(1u32);
        for f in [840u32, 420, 7, 1, 65_535, 4_000_000_000, 840, 839, 3] {
            x = x.checked_mul_small(f).unwrap();
            big *= f;
            assert_eq!(x.to_biguint(), big);
            let y = x.checked_add(x).unwrap();
            assert_eq!(y.to_biguint(), &big + &big);
            assert_eq!(x.checked_mul_u64(0x1234_5678_9abc_def0).unwrap().to_biguint(), &big * 0x1234_5678_9abc_def0u64);
        }
        assert_eq!(U256::from_biguint(&big), Some(x));
        let top = U256([u32::MAX; 8]);
        assert_eq!(top.checked_add(U256::ONE), None);
        assert_eq!(top.checked_mul_small(2), None);
        assert!(U256::from_u128(5) < U256::from_u128(1 << 100));
    }
}
