//! The compiled core of the scenario player.
//!
//! This crate is `scenario/planner.py` and `scenario/fiber.py` in Rust, line
//! for line: the same SplitMix64 streams, the same packed record key, the
//! same summation order, the same tie rule. The Python planner is the
//! referee — `scenario/tests/test_compiled.py` asserts the two return
//! identical tiles and identical action values on random states — and this
//! is the runtime.
//!
//! The crate is plain Rust: the rules tables ([`rules`], held equal to
//! `forge.oracle.tables` by the Python tests), the field net and its blob
//! format ([`field`]), the hidden-hand sampler ([`fiber`]), the planner
//! ([`planner`]), the convention overlay ([`convention`]) and the mold
//! ([`mold`]). The `python` feature adds the pyo3 module `scenario_core` that `scenario/compiled.py` drives; the
//! browser package `zeb/` builds on the plain crate.

pub mod convention;
pub mod fiber;
pub mod field;
pub mod mold;
pub mod planner;
pub mod rules;

#[cfg(feature = "python")]
mod python;
