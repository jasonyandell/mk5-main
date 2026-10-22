// SPDX-FileCopyrightText: Jason Yandell
// SPDX-License-Identifier: CC0-1.0

//! A course over a one-parameter family of beliefs, and its controller (Jason's note
//! *A minimal comparison algebra for a declared 42 family*, 2026-10-21, § Whole-ending
//! family).
//!
//! Every atom of the course carries a weight affine in a parameter `p` in `[0, 1]`, as
//! when a partner's type is High with probability `p` and Low otherwise. Then a
//! decision's mass is affine in `p`, the value of each play (its mass times its
//! conditional value) is a convex piecewise-linear function of `p`, and the maximizing
//! set is constant between finitely many thresholds. The parameter classes are the
//! endpoints, the thresholds and the open cells between them. The controller is the
//! course at one representative of each class, joined and quotiented: contexts are
//! (history, class), and two contexts are one state when every block sequence gives
//! equal outputs. All arithmetic is exact.

use num_rational::BigRational;
use num_traits::{One, Zero};

use crate::quotient::{Block, Course, Decision};
use crate::{Result, error};

type Q = BigRational;

/// `at0 (1 - p) + at1 p`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Affine {
    pub at0: Q,
    pub at1: Q,
}

impl Affine {
    pub fn new(at0: Q, at1: Q) -> Affine {
        Affine { at0, at1 }
    }

    pub fn zero() -> Affine {
        Affine::new(Q::zero(), Q::zero())
    }

    pub fn at(&self, p: &Q) -> Q {
        &self.at0 + (&self.at1 - &self.at0) * p
    }

    pub fn plus(&self, other: &Affine) -> Affine {
        Affine::new(&self.at0 + &other.at0, &self.at1 + &other.at1)
    }

    /// Positive somewhere on the open interval, and so everywhere on it.
    fn lives(&self) -> bool {
        self.at0 > Q::zero() || self.at1 > Q::zero()
    }
}

/// A continuous piecewise-linear function on `[0, 1]`: its knots, ascending from 0 to 1,
/// with no three consecutive knots collinear.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Piecewise {
    knots: Vec<(Q, Q)>,
}

impl Piecewise {
    pub fn affine(f: &Affine) -> Piecewise {
        Piecewise { knots: vec![(Q::zero(), f.at0.clone()), (Q::one(), f.at1.clone())] }
    }

    pub fn knots(&self) -> &[(Q, Q)] {
        &self.knots
    }

    /// The number of affine pieces.
    pub fn pieces(&self) -> usize {
        self.knots.len() - 1
    }

    pub fn at(&self, x: &Q) -> Q {
        let i = self.knots.partition_point(|(k, _)| k < x);
        let (x1, y1) = &self.knots[i.min(self.knots.len() - 1)];
        if x1 == x || i == 0 {
            return y1.clone();
        }
        let (x0, y0) = &self.knots[i - 1];
        y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    }

    fn simplified(knots: Vec<(Q, Q)>) -> Piecewise {
        let mut out: Vec<(Q, Q)> = Vec::with_capacity(knots.len());
        for k in knots {
            while out.len() >= 2 {
                let (x0, y0) = &out[out.len() - 2];
                let (x1, y1) = &out[out.len() - 1];
                if (y1 - y0) * (&k.0 - x0) == (&k.1 - y0) * (x1 - x0) {
                    out.pop();
                } else {
                    break;
                }
            }
            out.push(k);
        }
        Piecewise { knots: out }
    }

    fn grid(fs: &[&Piecewise]) -> Vec<Q> {
        let mut xs: Vec<Q> = fs.iter().flat_map(|f| f.knots.iter().map(|(x, _)| x.clone())).collect();
        xs.sort();
        xs.dedup();
        xs
    }

    pub fn sum(fs: &[&Piecewise]) -> Piecewise {
        let xs = Piecewise::grid(fs);
        Piecewise::simplified(
            xs.into_iter()
                .map(|x| {
                    let y = fs.iter().map(|f| f.at(&x)).sum();
                    (x, y)
                })
                .collect(),
        )
    }

    /// The knots of every function and every point where two of them cross.
    pub fn crossings(fs: &[&Piecewise]) -> Vec<Q> {
        let xs = Piecewise::grid(fs);
        let mut out = xs.clone();
        for w in xs.windows(2) {
            let v0: Vec<Q> = fs.iter().map(|f| f.at(&w[0])).collect();
            let v1: Vec<Q> = fs.iter().map(|f| f.at(&w[1])).collect();
            for i in 0..fs.len() {
                for j in i + 1..fs.len() {
                    let (d0, d1) = (&v0[i] - &v0[j], &v1[i] - &v1[j]);
                    if (d0 < Q::zero() && d1 > Q::zero()) || (d0 > Q::zero() && d1 < Q::zero()) {
                        out.push(&w[0] + (&w[1] - &w[0]) * &d0 / (&d0 - &d1));
                    }
                }
            }
        }
        out.sort();
        out.dedup();
        out
    }

    pub fn max(fs: &[&Piecewise]) -> Piecewise {
        Piecewise::simplified(
            Piecewise::crossings(fs)
                .into_iter()
                .map(|x| {
                    let y = fs.iter().map(|f| f.at(&x)).max().expect("a function");
                    (x, y)
                })
                .collect(),
        )
    }
}

/// Which decisions a controller keeps.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reach {
    /// Every decision reachable from the root, with its maximizing set.
    Every,
    /// Only those the plan reaches (at every decision, the least maximizing play), with
    /// their maximizing sets: what a viewer who plays well can meet.
    Path,
    /// The plan's path, outputting the plan's play.
    Plan,
}

/// One decision of a family course: its menu (strictly ascending), its mass, the
/// payoff each play banks when the course ends with it (as mass, aligned with the
/// menu), and its blocks.
#[derive(Clone, Debug)]
pub struct Choice<A, X> {
    pub menu: Vec<A>,
    pub mass: Affine,
    pub payoff: Vec<Affine>,
    pub blocks: Vec<Block<A, X>>,
}

/// A validated family course and every play's value, a piecewise-linear function of `p`.
pub struct Family<A, X> {
    choices: Vec<Choice<A, X>>,
    value: Vec<Vec<Piecewise>>,
}

impl<A: Clone + Ord + std::hash::Hash, X: Clone + Ord + std::hash::Hash> Family<A, X> {
    /// Refused when the structure is not a course, a payoff is not aligned, or the blocks
    /// under a play neither partition its mass nor are absent.
    pub fn new(choices: Vec<Choice<A, X>>) -> Result<Family<A, X>> {
        let structure = Course::new(
            choices
                .iter()
                .map(|c| Decision { menu: c.menu.clone(), best: c.menu.clone(), blocks: c.blocks.clone() })
                .collect(),
        )?;
        for (i, c) in choices.iter().enumerate() {
            if c.payoff.len() != c.menu.len() {
                return Err(error(format!("choice {i}: the payoffs are not aligned with the menu")));
            }
            for a in &c.menu {
                let under: Vec<&Block<A, X>> = c.blocks.iter().filter(|b| b.play == *a).collect();
                if under.is_empty() {
                    continue;
                }
                let mass = under.iter().fold(Affine::zero(), |m, b| m.plus(&choices[b.next].mass));
                if mass != c.mass {
                    return Err(error(format!("choice {i}: the blocks under a play do not partition its mass")));
                }
            }
        }
        let mut value: Vec<Vec<Piecewise>> = vec![Vec::new(); choices.len()];
        let mut best: Vec<Option<Piecewise>> = vec![None; choices.len()];
        for &i in structure.order() {
            let c = &choices[i];
            let row: Vec<Piecewise> = c
                .menu
                .iter()
                .zip(&c.payoff)
                .map(|(a, pay)| {
                    let mut parts = vec![Piecewise::affine(pay)];
                    parts.extend(
                        c.blocks
                            .iter()
                            .filter(|b| b.play == *a)
                            .map(|b| best[b.next].clone().expect("successors first")),
                    );
                    Piecewise::sum(&parts.iter().collect::<Vec<_>>())
                })
                .collect();
            best[i] = Some(Piecewise::max(&row.iter().collect::<Vec<_>>()));
            value[i] = row;
        }
        Ok(Family { choices, value })
    }

    pub fn len(&self) -> usize {
        self.choices.len()
    }

    pub fn is_empty(&self) -> bool {
        self.choices.is_empty()
    }

    pub fn choice(&self, i: usize) -> &Choice<A, X> {
        &self.choices[i]
    }

    /// Each play's value at choice `i` (mass times conditional value), aligned with its menu.
    pub fn value(&self, i: usize) -> &[Piecewise] {
        &self.value[i]
    }

    /// The best value at choice `i`: the upper envelope of its plays' values.
    pub fn envelope(&self, i: usize) -> Piecewise {
        Piecewise::max(&self.value[i].iter().collect::<Vec<_>>())
    }

    /// The maximizing set at choice `i` and parameter `p`.
    pub fn best_at(&self, i: usize, p: &Q) -> Vec<A> {
        let v: Vec<Q> = self.value[i].iter().map(|f| f.at(p)).collect();
        let top = v.iter().max().expect("a nonempty menu").clone();
        self.choices[i].menu.iter().zip(&v).filter(|(_, x)| **x == top).map(|(a, _)| a.clone()).collect()
    }

    /// The interior parameters where the maximizing set of some choice among `which`
    /// changes, ascending.
    pub fn thresholds(&self, which: &[usize]) -> Vec<Q> {
        let mut out = Vec::new();
        for &i in which {
            if self.choices[i].menu.len() < 2 || !self.choices[i].mass.lives() {
                continue;
            }
            let xs: Vec<Q> = Piecewise::crossings(&self.value[i].iter().collect::<Vec<_>>())
                .into_iter()
                .filter(|x| *x > Q::zero() && *x < Q::one())
                .collect();
            let grid: Vec<Q> =
                std::iter::once(Q::zero()).chain(xs.iter().cloned()).chain(std::iter::once(Q::one())).collect();
            for k in 1..grid.len() - 1 {
                let two = Q::from_integer(2.into());
                let (left, right) = ((&grid[k - 1] + &grid[k]) / &two, (&grid[k] + &grid[k + 1]) / &two);
                let here = self.best_at(i, &grid[k]);
                if self.best_at(i, &left) != here || self.best_at(i, &right) != here {
                    out.push(grid[k].clone());
                }
            }
        }
        out.sort();
        out.dedup();
        out
    }

    /// The thresholds at which some choice among `which` has disjoint maximizing sets just
    /// left and just right: where one fixed play cannot serve both sides.
    pub fn witnessed(&self, which: &[usize], thresholds: &[Q]) -> Vec<Q> {
        let grid = cells_grid(thresholds);
        let two = Q::from_integer(2.into());
        thresholds
            .iter()
            .enumerate()
            .filter(|(k, t)| {
                let (left, right) = ((&grid[*k] + *t) / &two, (*t + &grid[k + 2]) / &two);
                which.iter().any(|&i| {
                    self.choices[i].menu.len() > 1 && {
                        let (l, r) = (self.best_at(i, &left), self.best_at(i, &right));
                        l.iter().all(|a| !r.contains(a))
                    }
                })
            })
            .map(|(_, t)| t.clone())
            .collect()
    }

    /// The value from choice `i` of one fixed plan: at every decision, the least
    /// maximizing play at `p`. A fixed plan's value is affine in `p` (mass times success),
    /// and it meets the envelope at `p`.
    pub fn plan_at(&self, i: usize, p: &Q) -> Affine {
        let mut memo: Vec<Option<Affine>> = vec![None; self.choices.len()];
        self.plan_from(i, p, &mut memo)
    }

    fn plan_from(&self, i: usize, p: &Q, memo: &mut Vec<Option<Affine>>) -> Affine {
        if let Some(v) = &memo[i] {
            return v.clone();
        }
        let c = &self.choices[i];
        let a = self.best_at(i, p).into_iter().next().expect("a nonempty menu");
        let k = c.menu.iter().position(|x| *x == a).expect("a play of the menu");
        let mut v = c.payoff[k].clone();
        for b in c.blocks.iter().filter(|b| b.play == a) {
            v = v.plus(&self.plan_from(b.next, p, memo));
        }
        memo[i] = Some(v.clone());
        v
    }

    /// The decisions reachable from `roots`.
    pub fn reachable(&self, roots: &[usize]) -> Vec<usize> {
        let mut seen = vec![false; self.choices.len()];
        let mut stack = roots.to_vec();
        let mut out = Vec::new();
        while let Some(i) = stack.pop() {
            if std::mem::replace(&mut seen[i], true) {
                continue;
            }
            out.push(i);
            stack.extend(self.choices[i].blocks.iter().map(|b| b.next));
        }
        out.sort_unstable();
        out
    }

    /// The controller's course: for each parameter class (`representatives`), the
    /// decisions of positive mass that `reach` keeps, with their outputs. Returns the
    /// course and the context of the root in each class.
    pub fn controller(&self, root: usize, representatives: &[Q], reach: Reach) -> Result<(Course<A, X>, Vec<usize>)> {
        let mut decisions = Vec::new();
        let mut roots = Vec::new();
        for p in representatives {
            let plan = |i: usize| self.best_at(i, p).into_iter().next().expect("a nonempty menu");
            let kept = |i: usize, b: &Block<A, X>| reach == Reach::Every || b.play == plan(i);
            let mut live = Vec::new();
            let mut seen = vec![false; self.choices.len()];
            let mut stack = vec![root];
            while let Some(i) = stack.pop() {
                if std::mem::replace(&mut seen[i], true) || self.choices[i].mass.at(p) <= Q::zero() {
                    continue;
                }
                live.push(i);
                stack.extend(self.choices[i].blocks.iter().filter(|b| kept(i, b)).map(|b| b.next));
            }
            live.sort_unstable();
            let base = decisions.len();
            let mut index: Vec<Option<usize>> = vec![None; self.choices.len()];
            for (k, &i) in live.iter().enumerate() {
                index[i] = Some(base + k);
            }
            for &i in &live {
                let c = &self.choices[i];
                let blocks = c
                    .blocks
                    .iter()
                    .filter(|b| kept(i, b))
                    .filter_map(|b| {
                        index[b.next].map(|next| Block { play: b.play.clone(), seen: b.seen.clone(), next })
                    })
                    .collect();
                let best = if reach == Reach::Plan { vec![plan(i)] } else { self.best_at(i, p) };
                decisions.push(Decision { menu: c.menu.clone(), best, blocks });
            }
            roots.push(index[root].ok_or_else(|| error("the root has no mass at a representative"))?);
        }
        Ok((Course::new(decisions)?, roots))
    }

    /// The same family with choice `i`'s menu cut to `keep` (and its blocks with it).
    pub fn restricted(mut self, i: usize, keep: &[A]) -> Result<Family<A, X>> {
        let c = &mut self.choices[i];
        let slots: Vec<usize> = (0..c.menu.len()).filter(|&k| keep.contains(&c.menu[k])).collect();
        if slots.is_empty() {
            return Err(error("a restriction keeps no play"));
        }
        c.payoff = slots.iter().map(|&k| c.payoff[k].clone()).collect();
        c.menu = slots.iter().map(|&k| c.menu[k].clone()).collect();
        c.blocks.retain(|b| keep.contains(&b.play));
        let choices = std::mem::take(&mut self.choices);
        Family::new(choices)
    }
}

fn cells_grid(thresholds: &[Q]) -> Vec<Q> {
    std::iter::once(Q::zero()).chain(thresholds.iter().cloned()).chain(std::iter::once(Q::one())).collect()
}

/// One representative of each parameter class: 0, then for each cell its midpoint and its
/// right end.
pub fn representatives(thresholds: &[Q]) -> Vec<Q> {
    let grid = cells_grid(thresholds);
    let two = Q::from_integer(2.into());
    let mut out = vec![Q::zero()];
    for w in grid.windows(2) {
        out.push((&w[0] + &w[1]) / &two);
        out.push(w[1].clone());
    }
    out
}
