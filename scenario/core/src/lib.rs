// SPDX-FileCopyrightText: Jason Yandell
// SPDX-License-Identifier: CC0-1.0

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
//! ([`planner`]), the convention overlay ([`convention`]), the mold
//! ([`mold`]) and constellations, a position's canonical form over the live
//! tiles ([`constellation`]). The `python` feature adds the pyo3 module `scenario_core` that `scenario/compiled.py` drives; the
//! browser package `zeb/` builds on the plain crate. The `silas` feature adds
//! `physics`, Silas's `Physics` over these rules and this fiber, and with
//! `python` the Silas classes of `scenario_core` (`scenario/silas.py`).

pub mod constellation;
pub mod convention;
pub mod fiber;
pub mod field;
pub mod mold;
#[cfg(feature = "silas")]
pub mod physics;
pub mod planner;
pub mod rules;

#[cfg(feature = "python")]
mod python;
#[cfg(all(feature = "python", feature = "silas"))]
mod python_silas;
