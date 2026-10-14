-- SPDX-FileCopyrightText: Jason Yandell
-- SPDX-License-Identifier: CC0-1.0

import Stab.Ladder

/-!
# Appendix: the floor, and rung-indexed operators

`floored_local`: an operator that, at some coordinates, returns the lower-rung
policy's value at a later coordinate `f c` with `dep c (f c)`, and otherwise
responds as a local `R`, is local for the same `dep`; so L1 and L2 apply to it
unchanged (same dependency relation, same height bound).

`rung_seeded_no_settle`: L1 iterates ONE operator. If the operator at rung `k`
depends on `k` (Walt seeds its sample with `k`, planner.py line 304), locality alone
does not give stabilization: two local operators with no dependencies at all can
alternate forever.
-/

namespace Stab

variable {C A : Type}

/-- **The floored operator is local.** -/
theorem floored_local {dep : C → C → Prop} {R : (C → A) → (C → A)}
    (hloc : Local dep R) (floored : C → Prop) [DecidablePred floored] (f : C → C)
    (hf : ∀ c, floored c → dep c (f c)) :
    Local dep (fun p c => if floored c then p (f c) else R p c) := by
  intro c p q h
  by_cases hc : floored c
  · simp only [hc, if_true]; exact h _ (hf c hc)
  · simp only [hc, if_false]; exact hloc c p q h

/-- Iterating a rung-indexed family: `iterS Rs (k+1) p = Rs k (iterS Rs k p)`. -/
def iterS (Rs : Nat → (C → A) → (C → A)) : Nat → (C → A) → (C → A)
  | 0, p => p
  | k + 1, p => Rs k (iterS Rs k p)

/-- Each `Rs k` is local for the empty dependency relation (height 0 everywhere),
yet the iterates never settle. -/
def flip (k : Nat) : (Unit → Bool) → (Unit → Bool) := fun _ _ => decide (k % 2 = 0)

theorem flip_local (k : Nat) : Local (fun (_ _ : Unit) => False) (flip k) :=
  fun _ _ _ _ => rfl

theorem rung_seeded_no_settle (p : Unit → Bool) (k : Nat) :
    iterS flip (k + 2) p () ≠ iterS flip (k + 1) p () := by
  show decide ((k + 1) % 2 = 0) ≠ decide (k % 2 = 0)
  by_cases h : k % 2 = 0
  · have : ¬ (k + 1) % 2 = 0 := by omega
    simp [h, this]
  · have : (k + 1) % 2 = 0 := by omega
    simp [h, this]

end Stab
