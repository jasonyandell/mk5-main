// SPDX-FileCopyrightText: Jason Yandell
// SPDX-License-Identifier: CC0-1.0

//! SplitMix64, and the fast hasher the memo tables use. Every stream is derived from a seed
//! and structural coordinates; nothing reads a clock or global state (WALT.md §5.5).

use std::hash::{BuildHasherDefault, Hasher};

pub const GOLDEN: u64 = 0x9E37_79B9_7F4A_7C15;

/// The SplitMix64 output function.
#[inline]
pub fn mix64(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// The first output of a SplitMix64 generator seeded with `seed`.
#[inline]
pub fn splitmix_once(seed: u64) -> u64 {
    mix64(seed.wrapping_add(GOLDEN))
}

/// A seed for a sub-stream named by `coord` under `seed`.
#[inline]
pub fn derive(seed: u64, coord: u64) -> u64 {
    mix64(mix64(seed ^ 0x6A09_E667_F3BC_C909).wrapping_add(mix64(coord.wrapping_add(GOLDEN))))
}

pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(seed)
    }

    #[inline]
    pub fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(GOLDEN);
        mix64(self.0)
    }

    /// Uniform on `0..bound`, unbiased (Lemire's multiply-and-reject). `bound > 0`.
    #[inline]
    pub fn below(&mut self, bound: u64) -> u64 {
        let mut m = (self.next() as u128) * (bound as u128);
        if (m as u64) < bound {
            let threshold = bound.wrapping_neg() % bound;
            while (m as u64) < threshold {
                m = (self.next() as u128) * (bound as u128);
            }
        }
        (m >> 64) as u64
    }
}

/// A multiply-rotate hasher for keys that are already a few machine words.
#[derive(Default, Clone, Copy)]
pub struct FastHasher(u64);

const K: u64 = 0xF135_7AEA_2E62_A9C5;

impl FastHasher {
    #[inline]
    fn add(&mut self, x: u64) {
        self.0 = (self.0.rotate_left(26) ^ x).wrapping_mul(K);
    }
}

impl Hasher for FastHasher {
    #[inline]
    fn finish(&self) -> u64 {
        // hashbrown reads the top bits for its control bytes and the low bits for the slot.
        self.0 ^ (self.0 >> 29)
    }
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        let mut chunks = bytes.chunks_exact(8);
        for c in &mut chunks {
            self.add(u64::from_le_bytes(c.try_into().unwrap()));
        }
        let rest = chunks.remainder();
        if !rest.is_empty() {
            let mut buf = [0u8; 8];
            buf[..rest.len()].copy_from_slice(rest);
            self.add(u64::from_le_bytes(buf));
        }
    }
    #[inline]
    fn write_u8(&mut self, x: u8) {
        self.add(x as u64)
    }
    #[inline]
    fn write_u32(&mut self, x: u32) {
        self.add(x as u64)
    }
    #[inline]
    fn write_u64(&mut self, x: u64) {
        self.add(x)
    }
    #[inline]
    fn write_usize(&mut self, x: usize) {
        self.add(x as u64)
    }
}

pub type FastMap<K, V> = std::collections::HashMap<K, V, BuildHasherDefault<FastHasher>>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splitmix64_reference_vector() {
        // The published SplitMix64 outputs for seed 0.
        let mut r = Rng::new(0);
        assert_eq!(r.next(), 0xE220_A839_7B1D_CDAF);
        assert_eq!(r.next(), 0x6E78_9E6A_A1B9_65F4);
        assert_eq!(r.next(), 0x06C4_5D18_8009_454F);
    }

    #[test]
    fn below_stays_in_range_and_covers_it() {
        let mut r = Rng::new(7);
        let mut seen = [0u32; 5];
        for _ in 0..5000 {
            seen[r.below(5) as usize] += 1;
        }
        assert!(seen.iter().all(|&c| c > 800 && c < 1200), "{seen:?}");
    }
}
