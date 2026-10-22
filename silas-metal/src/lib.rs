// SPDX-FileCopyrightText: Jason Yandell
// SPDX-License-Identifier: CC0-1.0

//! The exact player's bulk engine (the exact-player track, Exact-GPU, 2026-10-22).
//!
//! [`epoch`]: the field epoch, from one focal action to the focal seat's next decision, over
//! explicit worlds with exact integer masses, in three constructions that agree bit for bit
//! (Silas's rules and rationals; the tables as integers on the CPU; the tables on Metal).
//! [`net`]: the lawful information-state solve built on epochs: children merged by the
//! complete public record (one choice per information state, sum over what the seat sees,
//! then max over its own plays), and its world-split control.
//!
//! STRATEGY-FUSION FENCE: the world-split control (treatment C, every later focal choice
//! allowed to see the world) is clairvoyant. It is computed only as a measurement of fusion,
//! the sanctioned "measuring how loose the bound is" (wiki/topics/strategy-fusion-fence.md):
//! a negative control that C exceeds H, never a value and never a choice.

pub mod epoch;
pub mod ladder;
pub mod m3;
pub mod metal;
pub mod net;
pub mod tables;
pub mod wide;

pub use epoch::{Backend, Batch, Child, Cpu, EpochError, Law, Particle, Reference, SCALE, Segment};
pub use metal::Gpu;
pub use tables::Tables;
pub use wide::U256;
