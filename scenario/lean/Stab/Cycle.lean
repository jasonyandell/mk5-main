-- SPDX-FileCopyrightText: Jason Yandell
-- SPDX-License-Identifier: CC0-1.0

import Stab.Ladder
import Stab.Coherence

/-!
# L4: without locality the ladder need not settle (note 1, section 4)

Hidden matching pennies. Two coordinates, `false` ("0") and `true` ("1"), actions
`Bool`. Coordinate 0 wants to match coordinate 1; coordinate 1 wants to mismatch
coordinate 0. Each reads the other's previous value. This is the shape of a response
whose beliefs are recomputed from the rung below: the answer at a coordinate reads
the lower rung at a coordinate that in turn reads it back.

# L5: the early-stopping certificate (note 2, section 2.4)

If every provisional value `Q_k(a)` is within `ε_a` of the eventual value `Q⋆(a)`,
and `Q_k(a) - ε_a > Q_k(b) + ε_b` for every other `b`, then `a` is the unique
maximizer of `Q⋆`, and the deterministic scan under `Q⋆` returns `a`.
Values are integers (rationals over a common denominator).
-/

namespace Stab

/-! ## L4 -/

/-- Hidden matching pennies as a response operator on `Bool → Bool`. -/
def pennies (p : Bool → Bool) : Bool → Bool :=
  fun c => if c then !p false else p true

/-- `pennies` has no fixed point at all. -/
theorem pennies_no_fixed_point (q : Bool → Bool) : pennies q ≠ q := by
  intro h
  have h0 := congrFun h false
  have h1 := congrFun h true
  simp [pennies] at h0 h1
  cases hq : q false <;> simp_all

/-- Hence no iterate, from any start, is a fixed point. -/
theorem pennies_never_settles (p : Bool → Bool) (k : Nat) :
    pennies (iter pennies k p) ≠ iter pennies k p :=
  pennies_no_fixed_point _

/-- Consecutive iterates always differ. -/
theorem pennies_moves (p : Bool → Bool) (k : Nat) :
    iter pennies (k + 1) p ≠ iter pennies k p :=
  pennies_no_fixed_point _

/-- The ladder cycles with period 4. -/
theorem pennies_period_four (p : Bool → Bool) (k : Nat) :
    iter pennies (k + 4) p = iter pennies k p := by
  show pennies (pennies (pennies (pennies (iter pennies k p)))) = iter pennies k p
  funext c
  cases c <;> simp [pennies]

/-- Any dependency relation for which `pennies` is local must contain both edges
`0 → 1` and `1 → 0`. -/
theorem pennies_local_needs_cycle (dep : Bool → Bool → Prop) (h : Local dep pennies) :
    dep false true ∧ dep true false := by
  constructor
  · apply Classical.byContradiction
    intro hn
    have := h false (fun _ => false) (fun c => c) (by
      intro d hd
      cases d
      · rfl
      · exact absurd hd hn)
    simp [pennies] at this
  · apply Classical.byContradiction
    intro hn
    have := h true (fun _ => false) (fun c => !c) (by
      intro d hd
      cases d
      · exact absurd hd hn
      · rfl)
    simp [pennies] at this

/-- **No height exists.** There is no dependency relation that both makes `pennies`
local and admits a height function: L1's hypotheses cannot be met. -/
theorem pennies_no_height :
    ¬ ∃ (dep : Bool → Bool → Prop) (h : Bool → Nat), Local dep pennies ∧ IsHeight dep h := by
  intro ⟨dep, h, hloc, hh⟩
  obtain ⟨h01, h10⟩ := pennies_local_needs_cycle dep hloc
  have := hh _ _ h01
  have := hh _ _ h10
  omega

/-! ## L5 -/

section EarlyStop
variable {α : Type}

/-- **Early-stopping certificate.** Under the hypotheses, `a` strictly beats every
other listed action under `Q⋆`. -/
theorem early_stop_strict (acts : List α) (Qk Qs ε : α → Int) (a : α)
    (hclose : ∀ b, Qk b - ε b ≤ Qs b ∧ Qs b ≤ Qk b + ε b)
    (hgap : ∀ b, b ∈ acts → b ≠ a → Qk a - ε a > Qk b + ε b) :
    ∀ b, b ∈ acts → b ≠ a → Qs b < Qs a := by
  intro b hb hba
  have h1 := hclose a
  have h2 := hclose b
  have h3 := hgap b hb hba
  omega

/-- ... so `a` is the unique maximizer of `Q⋆`, and the deterministic scan under
`Q⋆` returns it. -/
theorem early_stop_argmax (a0 : α) (l : List α) (Qk Qs ε : α → Int) (a : α)
    (ha : a ∈ a0 :: l)
    (hclose : ∀ b, Qk b - ε b ≤ Qs b ∧ Qs b ≤ Qk b + ε b)
    (hgap : ∀ b, b ∈ a0 :: l → b ≠ a → Qk a - ε a > Qk b + ε b) :
    best Qs a0 l = a ∧ ∀ m, m ∈ a0 :: l → (∀ x, x ∈ a0 :: l → Qs x ≤ Qs m) → m = a := by
  have strict := early_stop_strict (a0 :: l) Qk Qs ε a hclose hgap
  have uniq : ∀ m, m ∈ a0 :: l → (∀ x, x ∈ a0 :: l → Qs x ≤ Qs m) → m = a := by
    intro m hm hmax
    apply Classical.byContradiction
    intro hne
    have h1 := strict m hm hne
    have h2 := hmax a ha
    omega
  exact ⟨uniq _ (best_mem Qs a0 l) (best_max Qs a0 l), uniq⟩

end EarlyStop

end Stab
