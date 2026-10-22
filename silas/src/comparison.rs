// SPDX-FileCopyrightText: Jason Yandell
// SPDX-License-Identifier: CC0-1.0

//! How many numbers a viewer must keep: the comparison dimension of a course over atoms
//! (Jason's note *A minimal comparison algebra for a declared 42 family*, 2026-10-21,
//! § A concrete local algebra; Lean `scenario/lean/Stab/Hexagon.lean`).
//!
//! A deterministic viewer program from a decision succeeds on a set of atoms, so under
//! any belief over the atoms its value is the belief's expectation of a 0/1 vector.
//! Every numerical comparison of programs is then a linear functional of the belief,
//! and the comparison dimension is the dimension of the span of all differences of
//! program vectors: the fewest linear coordinates that preserve every comparison. At the
//! note's local history it is two (its `x` and `y`). Plays split the atoms among their
//! blocks, so the programs that open with a play are sums of independent programs of
//! its successors, and
//!
//! `D(h) = Σ_a ⊕_{blocks of a} D(next) + span { r_a - r_b }`
//!
//! with `r_a` any one program vector opening with `a`. Exact over the rationals.

use std::collections::BTreeMap;

use num_rational::BigRational;
use num_traits::{One, Zero};

type Q = BigRational;

/// What one play leads to: the atoms on which the course ends in success, or the
/// decisions its blocks lead to (which partition its atoms).
#[derive(Clone, Debug)]
pub enum Play {
    Ends(Vec<usize>),
    Continues(Vec<usize>),
}

/// A subspace of `Q^atoms`, rows in reduced echelon form keyed by pivot.
#[derive(Clone, Debug, Default)]
pub struct Span {
    rows: BTreeMap<usize, BTreeMap<usize, Q>>,
}

impl Span {
    pub fn dim(&self) -> usize {
        self.rows.len()
    }

    fn reduce(&self, mut v: BTreeMap<usize, Q>) -> BTreeMap<usize, Q> {
        loop {
            let Some(k) = v.keys().find(|k| self.rows.contains_key(k)).copied() else {
                return v;
            };
            let c = v[&k].clone();
            for (j, y) in &self.rows[&k] {
                let x = v.get(j).cloned().unwrap_or_else(Q::zero) - &c * y;
                if x.is_zero() {
                    v.remove(j);
                } else {
                    v.insert(*j, x);
                }
            }
        }
    }

    /// Adds `v`; whether the span grew.
    pub fn add(&mut self, v: BTreeMap<usize, Q>) -> bool {
        let v = self.reduce(v);
        let Some((&k, c)) = v.iter().next() else {
            return false;
        };
        let inv = Q::one() / c;
        let v: BTreeMap<usize, Q> = v.iter().map(|(j, x)| (*j, x * &inv)).collect();
        for row in self.rows.values_mut() {
            if let Some(f) = row.get(&k).cloned() {
                for (j, y) in &v {
                    let x = row.get(j).cloned().unwrap_or_else(Q::zero) - &f * y;
                    if x.is_zero() {
                        row.remove(j);
                    } else {
                        row.insert(*j, x);
                    }
                }
            }
        }
        self.rows.insert(k, v);
        true
    }
}

/// The comparison dimension of every decision of a course whose decisions are given as
/// their plays, in an order that puts every decision after its successors.
pub fn dimensions(decisions: &[Vec<Play>], order: &[usize]) -> Vec<usize> {
    let mut rep: Vec<Option<Vec<usize>>> = vec![None; decisions.len()];
    let mut span: Vec<Option<Span>> = vec![None; decisions.len()];
    let mut dim = vec![0; decisions.len()];
    for &i in order {
        let mut s = Span::default();
        let mut reps: Vec<Vec<usize>> = Vec::new();
        for play in &decisions[i] {
            let r = match play {
                Play::Ends(made) => made.clone(),
                Play::Continues(next) => {
                    let mut r = Vec::new();
                    for &n in next {
                        r.extend(rep[n].clone().expect("successors first"));
                        for row in span[n].as_ref().expect("successors first").rows.values() {
                            s.add(row.clone());
                        }
                    }
                    r
                }
            };
            reps.push(r);
        }
        let first: BTreeMap<usize, Q> = reps[0].iter().map(|&a| (a, Q::one())).collect();
        for r in &reps[1..] {
            let mut d: BTreeMap<usize, Q> = first.iter().map(|(a, x)| (*a, -x)).collect();
            for &a in r {
                let x = d.get(&a).cloned().unwrap_or_else(Q::zero) + Q::one();
                if x.is_zero() {
                    d.remove(&a);
                } else {
                    d.insert(a, x);
                }
            }
            s.add(d);
        }
        dim[i] = s.dim();
        rep[i] = reps.into_iter().next();
        span[i] = Some(s);
    }
    dim
}
