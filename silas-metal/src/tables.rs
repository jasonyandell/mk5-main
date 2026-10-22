// SPDX-FileCopyrightText: Jason Yandell
// SPDX-License-Identifier: CC0-1.0

//! The rule tables the engine reads, generated from Silas's [`Physics`] (the semantic
//! authority) and the declared table types. The engine never re-derives a rule: legality is
//! a mask from `incidence` and `led`, a trick's winner a maximum over `rank`.

use silas::{Context, Domino, Physics, Seat};

use crate::epoch::Law;

pub const T_INC: usize = 0;
pub const T_LED: usize = 8;
pub const T_RANK: usize = 36;
pub const T_LAW: usize = 260;

/// The tables as the shaders lay them out: incidence[8], led[28], rank[8 * 28], then each
/// table type's law for each seat.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Tables {
    words: Vec<u32>,
    kinds: usize,
}

impl Tables {
    /// One declaration's rules and the declared table types (each a law per seat; the
    /// focal seat's entry is never read).
    pub fn new(physics: &Physics, kinds: &[[Law; Seat::COUNT]]) -> Tables {
        let mut words = vec![0u32; T_LAW + kinds.len() * Seat::COUNT];
        for q in Context::ALL {
            words[T_INC + q.index()] = physics.incidence(q).bits();
            for d in Domino::ALL {
                words[T_RANK + q.index() * Domino::COUNT + d.index()] = u32::from(physics.trick_key(d, q));
            }
        }
        for d in Domino::ALL {
            words[T_LED + d.index()] = physics.led_context(d).index() as u32;
        }
        for (k, laws) in kinds.iter().enumerate() {
            for (s, law) in laws.iter().enumerate() {
                words[T_LAW + k * Seat::COUNT + s] = *law as u32;
            }
        }
        Tables { words, kinds: kinds.len() }
    }

    pub fn words(&self) -> &[u32] {
        &self.words
    }

    pub fn kinds(&self) -> usize {
        self.kinds
    }

    pub(crate) fn incidence(&self, context: u32) -> u32 {
        self.words[T_INC + context as usize]
    }

    pub(crate) fn led(&self, tile: u32) -> u32 {
        self.words[T_LED + tile as usize]
    }

    pub(crate) fn rank(&self, context: u32, tile: u32) -> u32 {
        self.words[T_RANK + context as usize * Domino::COUNT + tile as usize]
    }

    pub(crate) fn law(&self, kind: u32, seat: u32) -> u32 {
        self.words[T_LAW + kind as usize * Seat::COUNT + seat as usize]
    }
}
