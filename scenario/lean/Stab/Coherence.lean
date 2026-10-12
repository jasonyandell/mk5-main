/-!
# L3: the best response is a finite, exact argmax; the coherence bound
(note 1, sections 2-3)

Everything is in exact integer arithmetic. A belief over a finite list of worlds is
an integer weight vector (probabilities over a common denominator `D`); two beliefs
`μ, μ'` are compared only when their totals agree. Each policy `π` has a per-world
value `val π w ∈ [0, M]` (`M = 1` for the make indicator; `M` = the margin range for
points, after shifting). Then

* `V μ π = Σ_w μ w * val π w` (the belief-weighted value, times `D`);
* `L1 μ μ' = Σ_w |μ w - μ' w|`, so `TV(μ, μ') = L1 / (2 D)`.

The theorems, divided through by `2 D`, read:

* `|V_μ(π) - V_μ'(π)| ≤ M · TV(μ, μ')` for every policy (`value_shift`);
* the policy chosen by argmax under `μ'` loses at most `2 M · TV` under `μ`
  (`regret_le`);
* if an action beats another under `μ'` by more than `2 M · TV`, it still beats it
  under `μ` (`certificate`): note 1's `Δ > 2δ`.

Weights need not be nonnegative: the bound holds for any two signed weight vectors
with equal totals.
-/

namespace Stab

/-! ## Argmax with a fixed tie rule -/

section Argmax
variable {α : Type}

/-- Scan a nonempty list `a :: l`, keeping the current champion unless a later
element is strictly better. Ties keep the earlier element (the "lowest tile id"
rule, when the list is sorted by id). -/
def best (v : α → Int) : α → List α → α
  | a, [] => a
  | a, b :: bs => best v (if v a < v b then b else a) bs

/-- The chosen element is in the list. -/
theorem best_mem (v : α → Int) : ∀ (a : α) (l : List α), best v a l ∈ a :: l
  | a, [] => List.Mem.head _
  | a, b :: bs => by
    show best v (if v a < v b then b else a) bs ∈ a :: b :: bs
    by_cases h : v a < v b
    · rw [if_pos h]; exact List.Mem.tail _ (best_mem v b bs)
    · rw [if_neg h]
      have ih := best_mem v a bs
      generalize best v a bs = m at ih
      cases ih with
      | head => exact List.Mem.head _
      | tail _ h' => exact List.Mem.tail _ (List.Mem.tail _ h')

/-- The chosen element is a maximizer: no list element has a larger value. -/
theorem best_max (v : α → Int) :
    ∀ (a : α) (l : List α) (x : α), x ∈ a :: l → v x ≤ v (best v a l)
  | a, [], x, hx => by
    cases hx with
    | head => exact Int.le_refl _
    | tail _ h => cases h
  | a, b :: bs, x, hx => by
    simp only [best]
    have ih := best_max v (if v a < v b then b else a) bs
    have hc : v a ≤ v (if v a < v b then b else a) ∧ v b ≤ v (if v a < v b then b else a) := by
      split <;> omega
    cases hx with
    | head => exact Int.le_trans hc.1 (ih _ (List.Mem.head _))
    | tail _ h =>
      cases h with
      | head => exact Int.le_trans hc.2 (ih _ (List.Mem.head _))
      | tail _ h => exact ih x (List.Mem.tail _ h)

/-- **T1, abstract form.** On a finite nonempty list of policies, a maximizer
exists and the deterministic scan returns one. -/
theorem exists_argmax (v : α → Int) (a : α) (l : List α) :
    ∃ m, m ∈ a :: l ∧ ∀ x, x ∈ a :: l → v x ≤ v m :=
  ⟨best v a l, best_mem v a l, best_max v a l⟩

end Argmax

/-! ## Sums over a finite list of worlds -/

section Sums
variable {W : Type}

/-- Integer absolute value (kept local and explicit). -/
def iabs (x : Int) : Int := if x < 0 then -x else x

/-- `S ws f = Σ_{w ∈ ws} f w`. -/
def S : List W → (W → Int) → Int
  | [], _ => 0
  | w :: ws, f => f w + S ws f

theorem S_sub (ws : List W) (f g : W → Int) :
    S ws (fun w => f w - g w) = S ws f - S ws g := by
  induction ws with
  | nil => rfl
  | cons w ws ih => simp only [S, ih]; omega

theorem S_sub_mul (ws : List W) (f g v : W → Int) :
    S ws (fun w => f w * v w) - S ws (fun w => g w * v w) =
      S ws (fun w => (f w - g w) * v w) := by
  induction ws with
  | nil => rfl
  | cons w ws ih =>
    simp only [S]
    have : (f w - g w) * v w = f w * v w - g w * v w := Int.sub_mul _ _ _
    omega

/-- One world: `x (2v - M) ≤ |x| M` when `0 ≤ v ≤ M`. -/
theorem term_up (x v M : Int) (h0 : 0 ≤ v) (h1 : v ≤ M) :
    x * (2 * v - M) ≤ iabs x * M := by
  unfold iabs
  split
  · have := Int.mul_le_mul_of_nonneg_left (show M - 2 * v ≤ M by omega)
      (show 0 ≤ -x by omega)
    grind
  · have := Int.mul_le_mul_of_nonneg_left (show 2 * v - M ≤ M by omega)
      (show 0 ≤ x by omega)
    grind

/-- Summed, both directions: `±(2 Σ x v - M Σ x) ≤ M Σ |x|`. -/
theorem sum_bound (ws : List W) (x v : W → Int) (M : Int)
    (hv : ∀ w, 0 ≤ v w ∧ v w ≤ M) :
    2 * S ws (fun w => x w * v w) - M * S ws x ≤ M * S ws (fun w => iabs (x w)) ∧
    -(2 * S ws (fun w => x w * v w) - M * S ws x) ≤
      M * S ws (fun w => iabs (x w)) := by
  induction ws with
  | nil => simp [S]
  | cons w ws ih =>
    simp only [S]
    have hu := term_up (x w) (v w) M (hv w).1 (hv w).2
    have hd := term_up (x w) (M - v w) M (by have := hv w; omega) (by have := hv w; omega)
    obtain ⟨ih1, ih2⟩ := ih
    constructor <;> grind

end Sums

/-! ## The coherence bound -/

section Coherence
variable {W P : Type}

/-- Belief-weighted value of a per-world value function (times the denominator). -/
def V (ws : List W) (μ : W → Int) (u : W → Int) : Int := S ws (fun w => μ w * u w)

/-- `L1 = Σ |μ - μ'|`, i.e. `2 D · TV(μ, μ')`. -/
def L1 (ws : List W) (μ μ' : W → Int) : Int := S ws (fun w => iabs (μ w - μ' w))

/-- **Value shift.** Same totals, per-world values in `[0, M]`:
`2 |V_μ(u) - V_μ'(u)| ≤ M · L1(μ, μ')`, i.e. `|ΔV| ≤ M · TV`. -/
theorem value_shift (ws : List W) (μ μ' u : W → Int) (M : Int)
    (hmass : S ws μ = S ws μ') (hu : ∀ w, 0 ≤ u w ∧ u w ≤ M) :
    2 * iabs (V ws μ u - V ws μ' u) ≤ M * L1 ws μ μ' := by
  have hb := sum_bound ws (fun w => μ w - μ' w) u M hu
  have hlin := S_sub_mul ws μ μ' u
  have hx : S ws (fun w => μ w - μ' w) = 0 := by rw [S_sub]; omega
  unfold V L1
  rw [hlin]
  rw [hx] at hb
  generalize S ws (fun w => (μ w - μ' w) * u w) = s at hb ⊢
  generalize M * S ws (fun w => iabs (μ w - μ' w)) = K at hb ⊢
  unfold iabs
  split <;> omega

/-- **Regret of the wrong belief.** Choose the policy by argmax under `μ'` (the
belief Walt uses) from a nonempty finite list; under the true belief `μ` it loses at
most `M · L1 = 2 D · M · TV` to every listed policy. -/
theorem regret_le (ws : List W) (μ μ' : W → Int) (M : Int) (val : P → W → Int)
    (hmass : S ws μ = S ws μ') (hval : ∀ π w, 0 ≤ val π w ∧ val π w ≤ M)
    (a : P) (l : List P) (πstar : P) (hstar : πstar ∈ a :: l) :
    V ws μ (val πstar) - V ws μ (val (best (fun π => V ws μ' (val π)) a l)) ≤
      M * L1 ws μ μ' := by
  generalize hhat : best (fun π => V ws μ' (val π)) a l = πhat
  have hmax := best_max (fun π => V ws μ' (val π)) a l πstar hstar
  rw [hhat] at hmax
  have h1 := value_shift ws μ μ' (val πstar) M hmass (hval πstar)
  have h2 := value_shift ws μ μ' (val πhat) M hmass (hval πhat)
  generalize M * L1 ws μ μ' = K at h1 h2 ⊢
  unfold iabs at h1 h2
  split at h1 <;> split at h2 <;> omega

/-- **The action certificate (`Δ > 2δ`).** If, under the belief used, `a` beats `b`
by more than `M · L1` (that is, by more than `2 M · TV` after dividing by `D`), then
`a` also beats `b` under the true belief. -/
theorem certificate (ws : List W) (μ μ' : W → Int) (M : Int) (ua ub : W → Int)
    (hmass : S ws μ = S ws μ') (ha : ∀ w, 0 ≤ ua w ∧ ua w ≤ M)
    (hb : ∀ w, 0 ≤ ub w ∧ ub w ≤ M)
    (hgap : V ws μ' ua - V ws μ' ub > M * L1 ws μ μ') :
    V ws μ ua > V ws μ ub := by
  have h1 := value_shift ws μ μ' ua M hmass ha
  have h2 := value_shift ws μ μ' ub M hmass hb
  generalize M * L1 ws μ μ' = K at h1 h2 hgap
  unfold iabs at h1 h2
  split at h1 <;> split at h2 <;> omega

end Coherence

end Stab
