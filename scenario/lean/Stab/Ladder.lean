/-!
# L1: the dependency-height stabilization theorem (note 2, section 2.1)

Abstract setting: coordinates `C`, actions `A` (any types; no finiteness needed),
policies `C → A`, a response operator `R : (C → A) → (C → A)`, and a dependency
relation `dep`. Locality: `R p c` reads `p` only at the direct dependencies of `c`.
-/

namespace Stab

variable {C A : Type}

/-- The `k`-th iterate of the response operator: `iter R 0 p = p`,
`iter R (k+1) p = R (iter R k p)`. In note 2's indexing, `P^k = iter R (k+1) U`. -/
def iter (R : (C → A) → (C → A)) : Nat → (C → A) → (C → A)
  | 0, p => p
  | k + 1, p => R (iter R k p)

/-- Locality: the response at `c` depends on the input policy only at the
coordinates `d` with `dep c d`. -/
def Local (dep : C → C → Prop) (R : (C → A) → (C → A)) : Prop :=
  ∀ (c : C) (p q : C → A), (∀ d, dep c d → p d = q d) → R p c = R q c

/-- `Chain dep c n`: a dependency chain with `n` edges starting at `c`. -/
inductive Chain (dep : C → C → Prop) : C → Nat → Prop
  | nil (c : C) : Chain dep c 0
  | cons {c d : C} {n : Nat} : dep c d → Chain dep d n → Chain dep c (n + 1)

/-- A height function: every dependency points strictly down. -/
def IsHeight (dep : C → C → Prop) (h : C → Nat) : Prop :=
  ∀ c d, dep c d → h d < h c

/-- Any height function bounds every chain. So the longest-chain height `h(c)` of
note 2 is below every height function. -/
theorem chain_le_height {dep : C → C → Prop} {h : C → Nat} (hh : IsHeight dep h) :
    ∀ {c n}, Chain dep c n → n ≤ h c := by
  intro c n hc
  induction hc with
  | nil c => exact Nat.zero_le _
  | cons hcd _ ih => have := hh _ _ hcd; omega

/-- **The stabilization theorem (chain form).** If every dependency chain from `c`
has at most `m` edges, then from iterate `m + 1` on, the response at `c` is fixed,
and it is the same value whatever the starting policy. -/
theorem settle {dep : C → C → Prop} {R : (C → A) → (C → A)} (hloc : Local dep R) :
    ∀ (m : Nat) (c : C), (∀ n, Chain dep c n → n ≤ m) →
      ∀ (p q : C → A) (k j : Nat), m < k → m < j → iter R k p c = iter R j q c := by
  intro m
  induction m with
  | zero =>
    intro c hc p q k j hk hj
    match k, j, hk, hj with
    | k + 1, j + 1, _, _ =>
      apply hloc
      intro d hd
      have := hc 1 (Chain.cons hd (Chain.nil d))
      omega
  | succ m ih =>
    intro c hc p q k j hk hj
    match k, j, hk, hj with
    | k + 1, j + 1, hk, hj =>
      apply hloc
      intro d hd
      apply ih d _ p q k j (by omega) (by omega)
      intro n hn
      have := hc (n + 1) (Chain.cons hd hn)
      omega

/-- **The stabilization theorem (height form).** With a height function `h`, for
every `k ≥ h c + 1` the iterate's action at `c` equals the one at iterate `h c + 1`.
(In note 2's indexing, `P^k(c) = P^⋆(c)` for `k ≥ h(c)`.) -/
theorem stable_from_height {dep : C → C → Prop} {R : (C → A) → (C → A)} {h : C → Nat}
    (hloc : Local dep R) (hh : IsHeight dep h) (p : C → A) (c : C) (k : Nat)
    (hk : h c + 1 ≤ k) : iter R k p c = iter R (h c + 1) p c :=
  settle hloc (h c) c (fun _ hn => chain_le_height hh hn) p p k (h c + 1)
    (by omega) (by omega)

/-- The limit is independent of the starting policy. -/
theorem limit_indep {dep : C → C → Prop} {R : (C → A) → (C → A)} {h : C → Nat}
    (hloc : Local dep R) (hh : IsHeight dep h) (p q : C → A) (c : C) (k : Nat)
    (hk : h c + 1 ≤ k) : iter R k p c = iter R k q c :=
  settle hloc (h c) c (fun _ hn => chain_le_height hh hn) p q k k (by omega) (by omega)

/-- **Uniform corollary.** If `h c ≤ H` everywhere, `iter R (H+1) p` is a fixed
point of `R`. -/
theorem fixed_point {dep : C → C → Prop} {R : (C → A) → (C → A)} {h : C → Nat}
    {H : Nat} (hloc : Local dep R) (hh : IsHeight dep h) (hH : ∀ c, h c ≤ H)
    (p : C → A) : R (iter R (H + 1) p) = iter R (H + 1) p := by
  funext c
  show iter R (H + 2) p c = iter R (H + 1) p c
  exact settle hloc (h c) c (fun _ hn => chain_le_height hh hn) p p _ _
    (by have := hH c; omega) (by have := hH c; omega)

/-- ... and it is the unique fixed point: every fixed point of `R` equals it. -/
theorem fixed_point_unique {dep : C → C → Prop} {R : (C → A) → (C → A)} {h : C → Nat}
    {H : Nat} (hloc : Local dep R) (hh : IsHeight dep h) (hH : ∀ c, h c ≤ H)
    (p q : C → A) (hq : R q = q) : q = iter R (H + 1) p := by
  have hiter : ∀ k, iter R k q = q := by
    intro k
    induction k with
    | zero => rfl
    | succ k ih => show R (iter R k q) = q; rw [ih, hq]
  funext c
  have := limit_indep hloc hh q p c (H + 1) (by have := hH c; omega)
  rw [hiter] at this
  exact this

/-- The brief's literal indexing (`R^[k] p0 c = R^[h c] p0 c` for all `k ≥ h c`) is
off by one: a coordinate with no dependencies (`h c = 0`) still needs one
application. Counterexample: one coordinate, no dependencies (so `h = 0`), `R` the
constant-`true` response, start policy `false`. -/
theorem brief_indexing_off_by_one :
    ¬ (∀ k, 0 ≤ k →
        iter (C := Unit) (A := Bool) (fun _ _ => true) k (fun _ => false) () =
        iter (C := Unit) (A := Bool) (fun _ _ => true) 0 (fun _ => false) ()) := by
  intro hall
  have := hall 1 (Nat.zero_le _)
  simp [iter] at this

end Stab
