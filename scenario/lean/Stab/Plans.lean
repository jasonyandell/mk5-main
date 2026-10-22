-- SPDX-FileCopyrightText: Jason Yandell
-- SPDX-License-Identifier: CC0-1.0

/-!
# Fixed plans in a one-parameter family

(Jason's note *A minimal comparison algebra for a declared 42 family*, 2026-10-21,
section "Less demanding tasks have smaller exact answers"; Night-Math, 2026-10-22;
`silas/src/family.rs`, `silas/tests/controller.rs`.)

The partner is High with probability `p` and Low otherwise. A fixed plan of the viewer's
(one play at every decision) succeeds on a fixed set of (deal, type) atoms, whose
weights are affine in `p`, so the plan's value is affine in `p`; the optimum is the
upper envelope of finitely many of them, convex and piecewise linear.

* `optimal_between`: the parameters at which a plan is optimal form an interval.
* `optimal_on_piece`: where the optimum is attained by one plan across `[u, w]`, any plan
  optimal at one interior point is optimal across all of it.

Together: the fewest fixed plans that are optimal at every `p` is the number of affine
pieces of the optimum. A plan's optimal set is an interval on which the optimum equals
that plan's line, so `k` plans give at most `k` pieces; and the plan optimal at a
piece's midpoint serves the whole piece. In the note's 560-deal no-trump ending the
optimum has five pieces, knots 1/16, 1/5, 1/3, 1/2, so five plans are enough and are
needed (the note's 5), against 31 classes for the complete maximizing sets.
-/

namespace Stab.Plans

/-- A line in the parameter: `at0` at `p = 0`, `at1` at `p = 1`. -/
structure Line where
  at0 : Rat
  at1 : Rat

def Line.at (f : Line) (p : Rat) : Rat := f.at0 + (f.at1 - f.at0) * p

/-- Plan `f` is optimal among `plans` at `p`. -/
def Optimal (plans : List Line) (f : Line) (p : Rat) : Prop := ∀ g ∈ plans, g.at p ≤ f.at p

theorem slope_nonneg (e a b : Rat) (he : 0 ≤ e) (hab : a ≤ b) : e * a ≤ e * b := by
  have := Rat.mul_nonneg he (by grind : (0 : Rat) ≤ b - a)
  grind

theorem optimal_between (plans : List Line) (f : Line) (x z y : Rat) (hxz : x ≤ z) (hzy : z ≤ y)
    (hx : Optimal plans f x) (hy : Optimal plans f y) : Optimal plans f z := by
  intro g hg
  have gx := hx g hg
  have gy := hy g hg
  simp only [Line.at] at *
  by_cases he : 0 ≤ (f.at1 - f.at0) - (g.at1 - g.at0)
  · have := slope_nonneg _ _ _ he hxz
    grind
  · have := slope_nonneg (-((f.at1 - f.at0) - (g.at1 - g.at0))) z y (by grind) hzy
    grind

theorem optimal_on_piece (plans : List Line) (f h : Line) (u m w : Rat) (hum : u < m) (hmw : m < w)
    (hf : f ∈ plans) (hh : h ∈ plans) (hopt : ∀ p, u ≤ p → p ≤ w → Optimal plans h p)
    (hm : Optimal plans f m) : ∀ p, u ≤ p → p ≤ w → Optimal plans f p := by
  -- `h - f` is a line, nonnegative on `[u, w]` and zero at the interior `m`: it is zero.
  intro p hup hpw g hg
  have hfu := hopt u (by grind) (by grind) f hf
  have hfw := hopt w (by grind) (by grind) f hf
  have hhm := hm h hh
  have hfm := hopt m (by grind) (by grind) f hf
  have hgp := hopt p hup hpw g hg
  have hfp := hopt p hup hpw f hf
  simp only [Line.at] at *
  let s := (h.at1 - h.at0) - (f.at1 - f.at0)
  have flat : s = 0 := by
    by_cases hneg : s < 0
    · have := Rat.mul_pos (by grind : (0 : Rat) < -s) (by grind : (0 : Rat) < w - m)
      grind
    by_cases hpos : 0 < s
    · have := Rat.mul_pos hpos (by grind : (0 : Rat) < m - u)
      grind
    grind
  have e1 : s * p = 0 := by rw [flat]; exact Rat.zero_mul p
  have e2 : s * m = 0 := by rw [flat]; exact Rat.zero_mul m
  grind

end Stab.Plans
