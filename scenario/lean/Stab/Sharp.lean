import Stab.Ladder
namespace Stab
def shiftR (n : Nat) (p : Nat → Bool) : Nat → Bool := fun i => if i < n then p (i + 1) else true
theorem shiftR_iter (n : Nat) : ∀ k i, iter (shiftR n) k (fun _ => false) i = decide (n - i < k) := by
  intro k
  induction k with
  | zero => intro i; simp [iter]
  | succ k ih =>
    intro i
    show (if i < n then iter (shiftR n) k (fun _ => false) (i + 1) else true) = _
    split
    · rw [ih]; apply decide_eq_decide.mpr; omega
    · simp; omega

/-- `shiftR n` is local for the chain `i → i + 1` (below `n`), with height `n - i`. -/
theorem shiftR_local (n : Nat) :
    Local (fun i d => i < n ∧ d = i + 1) (shiftR n) := by
  intro c p q h
  unfold shiftR
  split
  · next hc => exact h (c + 1) ⟨hc, rfl⟩
  · rfl

theorem shiftR_height (n : Nat) : IsHeight (fun i d => i < n ∧ d = i + 1) (fun i => n - i) := by
  intro c d ⟨hc, hd⟩; subst hd; show n - (c + 1) < n - c; omega

/-- **The bound `h c + 1` is attained.** For the local operator `shiftR n`, at
coordinate `0` (height `n`) the iterate from the all-`false` policy is still
`false` at iterate `n` and flips at iterate `n + 1`: the root keeps one answer for
`n` applications and changes at the last one (note 2, section 2.3, in miniature). -/
theorem bound_attained (n : Nat) :
    iter (shiftR n) n (fun _ => false) 0 = false ∧
    iter (shiftR n) (n + 1) (fun _ => false) 0 = true := by
  rw [shiftR_iter, shiftR_iter]; simp

end Stab
