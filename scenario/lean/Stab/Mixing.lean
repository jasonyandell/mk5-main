-- SPDX-FileCopyrightText: Jason Yandell
-- SPDX-License-Identifier: CC0-1.0

import Stab.Hexagon

/-!
# Q3: the decision quotient is not a congruence for belief mixing
(Jason's note *A minimal comparison algebra for a declared 42 family*, 2026-10-21,
section "Why the richer algebra cannot be replaced by this quotient")

The partner is Low with probability `1 - p` and High with probability `p`; deals are
uniform and independent of the type, so mixing two beliefs with weight `λ` mixes their
parameters: `λ p + (1 - λ) q`. At one later history the note computes that two lawful
continuations differ by `QA - QB = (5p - 1)/(1 + 18p)`. That formula is the fixture's
computed fact and is taken here as the definition of the threshold decision.

* `mass_injective`: the observation `o` after the initial 2-1 has mass
  `(1 + 18p)/560`, an injective function of `p`: exact prediction keeps every parameter.
* `mixing_forces_equality`: a relation on `[0, 1]` that preserves the sign of `QA - QB`
  and is preserved by every convex mixture relates only equal parameters. The witness
  uses `λ = 1/5` and `q = (1 - m)/4` with `m` the midpoint, so no "sufficiently small"
  limit is needed.
* `note_example`: the note's `0.08`, `0.12`, mixed equally with `0.30`, fall on
  opposite sides of `1/5`. That `0.08` and `0.12` share a decision cell `(1/16, 1/7)` is
  the note's atlas, not proved here; only that both lie strictly between `1/16` and
  `1/7` is.
-/

namespace Stab.Mixing
open Stab.Hexagon

/-- The threshold gap at the later history. -/
def gap (p : Rat) : Rat := (5 * p - 1) / (1 + 18 * p)

/-- The threshold decision: which continuation is better, or a tie. -/
def decision (p : Rat) : Int := sgn (gap p)

def mix (lam p q : Rat) : Rat := lam * p + (1 - lam) * q

theorem decision_eq (p : Rat) (hp : 0 ≤ p) : decision p = sgn (5 * p - 1) :=
  sgn_div_pos _ _ (by grind)

/-- The mass of the observation `(1-0, 6-2, 5-2)` after the initial 2-1. -/
def obsMass (p : Rat) : Rat := (1 + 18 * p) / 560

theorem mass_injective (p p' : Rat) (h : obsMass p = obsMass p') : p = p' := by
  simp only [obsMass] at h; grind

/-- Convex mixing does not commute with the decision quotient: any relation that keeps
the threshold decision and is kept by every mixture is equality on `[0, 1]`. -/
theorem mixing_forces_equality (r : Rat → Rat → Prop)
    (hdec : ∀ p p', 0 ≤ p → p ≤ 1 → 0 ≤ p' → p' ≤ 1 → r p p' → decision p = decision p')
    (hmix : ∀ p p' q lam, 0 ≤ q → q ≤ 1 → 0 ≤ lam → lam ≤ 1 →
      r p p' → r (mix lam p q) (mix lam p' q))
    (p p' : Rat) (hp0 : 0 ≤ p) (hp1 : p ≤ 1) (hq0 : 0 ≤ p') (hq1 : p' ≤ 1) (h : r p p') :
    p = p' := by
  -- Mix both with q = (1 - m)/4 at λ = 1/5: the results are (1 + p - m)/5 and
  -- (1 + p' - m)/5, on opposite sides of 1/5 whenever p ≠ p'.
  let m := (p + p') / 2
  let q := (1 - m) / 4
  have hm := hmix p p' q (1/5) (by grind) (by grind) (by grind) (by grind) h
  have e1 : mix (1/5) p q = (1 + p - m) / 5 := by simp only [mix, q, m]; grind
  have e2 : mix (1/5) p' q = (1 + p' - m) / 5 := by simp only [mix, q, m]; grind
  have hd := hdec _ _ (by rw [e1]; grind) (by rw [e1]; grind) (by rw [e2]; grind)
    (by rw [e2]; grind) hm
  rw [decision_eq _ (by rw [e1]; grind), decision_eq _ (by rw [e2]; grind), e1, e2] at hd
  have f1 : 5 * ((1 + p - m) / 5) - 1 = (p - p') / 2 := by simp only [m]; grind
  have f2 : 5 * ((1 + p' - m) / 5) - 1 = (p' - p) / 2 := by simp only [m]; grind
  rw [f1, f2] at hd
  rcases trich (p - p') with a | a | a
  · rw [sgn_neg (by grind), sgn_pos (by grind)] at hd; exact absurd hd (by decide)
  · grind
  · rw [sgn_pos (by grind), sgn_neg (by grind)] at hd; exact absurd hd (by decide)

/-- The note's numbers. -/
theorem note_example :
    (1/16 : Rat) < 8/100 ∧ (8/100 : Rat) < 12/100 ∧ (12/100 : Rat) < 1/7 ∧
      mix (1/2) (8/100) (30/100) = 19/100 ∧ mix (1/2) (12/100) (30/100) = 21/100 ∧
      decision (19/100) = -1 ∧ decision (21/100) = 1 := by
  refine ⟨by grind, by grind, by grind, by simp only [mix]; grind, by simp only [mix]; grind,
    ?_, ?_⟩
  · rw [decision_eq _ (by grind), sgn_neg (by grind)]
  · rw [decision_eq _ (by grind), sgn_pos (by grind)]

end Stab.Mixing
