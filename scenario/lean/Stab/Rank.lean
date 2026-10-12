import Stab.Ladder

/-!
# L2: the rank bound and its Texas 42 instance (note 2, section 2.2)

Part A (abstract). A rank that strictly decreases along every dependency is a height
function, so it bounds the dependency height. If, in addition, every coordinate is a
genuine decision of rank at least one, the height is at most `rank - 1`, and the
response at `c` is settled by iterate `rank c` (note 2's `P^{R-1}`).

Part B (42 tile counts). The rank `R = Σ_j max(n_j - 1, 0)` over the four seats'
tile counts. Every play weakly lowers `R`; a genuine play (the seat holds at least
two tiles) lowers it by exactly one; so every decision strictly after a genuine
decision has strictly smaller `R`. At the opening lead `R = 24`.

What is NOT modeled: dominoes, suits, legality, or the claim that Walt's response at
an information set reads the lower rung only at strictly later decisions. That last
claim is the hypothesis `dep c d → later c d`, which holds because the plan from an
information set only contains decisions after it; it is assumed, not derived.
-/

namespace Stab

variable {C A : Type}

/-! ## Part A: abstract rank bound -/

/-- A rank that drops along every dependency is a height function. -/
theorem rank_isHeight {dep : C → C → Prop} {rank : C → Nat}
    (hr : ∀ c d, dep c d → rank d < rank c) : IsHeight dep rank := hr

/-- Every dependency chain from `c` has at most `rank c` edges. -/
theorem chain_le_rank {dep : C → C → Prop} {rank : C → Nat}
    (hr : ∀ c d, dep c d → rank d < rank c) {c : C} {n : Nat} (hc : Chain dep c n) :
    n ≤ rank c :=
  chain_le_height (rank_isHeight hr) hc

/-- If every coordinate has rank at least one (all coordinates are genuine
decisions), then `rank - 1` is a height function. -/
theorem pred_rank_isHeight {dep : C → C → Prop} {rank : C → Nat}
    (hr : ∀ c d, dep c d → rank d < rank c) (hpos : ∀ c, 1 ≤ rank c) :
    IsHeight dep (fun c => rank c - 1) := by
  intro c d hcd
  have h1 := hr c d hcd
  have h2 := hpos d
  show rank d - 1 < rank c - 1
  omega

/-- **The rank bound.** With all coordinates genuine (rank ≥ 1), the response at `c`
is settled from iterate `rank c` on: `iter R k p c = iter R (rank c) p c` for every
`k ≥ rank c`, independent of the starting policy. -/
theorem settled_by_rank {dep : C → C → Prop} {R : (C → A) → (C → A)}
    {rank : C → Nat} (hloc : Local dep R)
    (hr : ∀ c d, dep c d → rank d < rank c) (hpos : ∀ c, 1 ≤ rank c)
    (p q : C → A) (c : C) (k : Nat) (hk : rank c ≤ k) :
    iter R k p c = iter R (rank c) q c := by
  have hh := pred_rank_isHeight hr hpos
  have := hpos c
  exact settle hloc (rank c - 1) c (fun _ hn => chain_le_height hh hn) p q k (rank c)
    (by omega) (by omega)

/-- Uniform form: if every rank is at most `Rmax` (and at least one), then
`iter R Rmax p` is the unique fixed point of `R`. -/
theorem fixed_point_by_rank {dep : C → C → Prop} {R : (C → A) → (C → A)}
    {rank : C → Nat} {Rmax : Nat} (hloc : Local dep R)
    (hr : ∀ c d, dep c d → rank d < rank c) (hpos : ∀ c, 1 ≤ rank c)
    (hmax : ∀ c, rank c ≤ Rmax) (p : C → A) :
    R (iter R Rmax p) = iter R Rmax p ∧ ∀ q, R q = q → q = iter R Rmax p := by
  have hh := pred_rank_isHeight hr hpos
  have key : ∀ c (p q : C → A) k j, Rmax ≤ k → Rmax ≤ j →
      iter R k p c = iter R j q c := by
    intro c p q k j hk hj
    have h1 := hpos c
    have h2 := hmax c
    exact settle hloc (rank c - 1) c (fun _ hn => chain_le_height hh hn) p q k j
      (by omega) (by omega)
  refine ⟨?_, ?_⟩
  · funext c
    exact key c p p (Rmax + 1) Rmax (by omega) (Nat.le_refl _)
  · intro q hq
    have hiter : ∀ k, iter R k q = q := by
      intro k
      induction k with
      | zero => rfl
      | succ k ih => show R (iter R k q) = q; rw [ih, hq]
    funext c
    have := key c q p Rmax Rmax (Nat.le_refl _) (Nat.le_refl _)
    rw [hiter] at this
    exact this

/-! ## Part B: Texas 42 tile counts -/

/-- Tile counts of the four seats. -/
abbrev Counts := Fin 4 → Nat

/-- The rank `R = Σ_j max(n_j - 1, 0)` (truncated subtraction is `max(· - 1, 0)`). -/
def rk (n : Counts) : Nat := (n 0 - 1) + (n 1 - 1) + (n 2 - 1) + (n 3 - 1)

/-- Seat `j` plays one tile. -/
def play (n : Counts) (j : Fin 4) : Counts := fun i => if i = j then n i - 1 else n i

/-- At the opening lead of an ordinary hand every seat holds seven tiles: `R = 24`. -/
theorem rk_opening : rk (fun _ => 7) = 24 := by decide

/-- Every play weakly lowers the rank. -/
theorem rk_play_le (n : Counts) (j : Fin 4) : rk (play n j) ≤ rk n := by
  unfold rk play
  match j with
  | 0 => simp <;> omega
  | 1 => simp <;> omega
  | 2 => simp <;> omega
  | 3 => simp <;> omega

/-- A genuine play (the acting seat holds at least two tiles) lowers the rank by
exactly one. -/
theorem rk_play_genuine (n : Counts) (j : Fin 4) (hj : 2 ≤ n j) :
    rk (play n j) + 1 = rk n := by
  unfold rk play
  match j with
  | 0 => simp <;> omega
  | 1 => simp <;> omega
  | 2 => simp <;> omega
  | 3 => simp <;> omega

/-- A genuine decision has rank at least one. -/
theorem rk_pos_of_genuine (n : Counts) (j : Fin 4) (hj : 2 ≤ n j) : 1 ≤ rk n := by
  have := rk_play_genuine n j hj
  omega

/-- The counts after a sequence of plays. -/
def playAll (n : Counts) : List (Fin 4) → Counts
  | [] => n
  | j :: js => playAll (play n j) js

/-- Any sequence of plays weakly lowers the rank. -/
theorem rk_playAll_le (n : Counts) (js : List (Fin 4)) : rk (playAll n js) ≤ rk n := by
  induction js generalizing n with
  | nil => exact Nat.le_refl _
  | cons j js ih => exact Nat.le_trans (ih (play n j)) (rk_play_le n j)

/-- **Strict decrease across a genuine decision.** If seat `j` makes a genuine
decision at counts `n` and any plays follow, every later position has strictly
smaller rank. This is the hypothesis `dep c d → rank d < rank c` of Part A, for
dependencies that point to later decisions. -/
theorem rk_later_lt (n : Counts) (j : Fin 4) (hj : 2 ≤ n j) (js : List (Fin 4)) :
    rk (playAll (play n j) js) < rk n := by
  have h1 := rk_playAll_le (play n j) js
  have h2 := rk_play_genuine n j hj
  omega

/-- **The 42 ceiling.** Abstract coordinates with a 42 position attached; if
dependencies point only to later positions (reached by a genuine play and then any
plays), and every coordinate is a genuine decision, then the response at a
coordinate whose position is the opening lead is settled by iterate 24 (that is,
`P^23` in note 2's indexing), from any starting policy. -/
theorem settled_42 {dep : C → C → Prop} {R : (C → A) → (C → A)}
    (hloc : Local dep R) (pos : C → Counts) (seat : C → Fin 4)
    (hgen : ∀ c, 2 ≤ pos c (seat c))
    (hlater : ∀ c d, dep c d → ∃ js, pos d = playAll (play (pos c) (seat c)) js)
    (p q : C → A) (c : C) (hc : pos c = fun _ => 7) (k : Nat) (hk : 24 ≤ k) :
    iter R k p c = iter R 24 q c := by
  have hr : ∀ c d, dep c d → rk (pos d) < rk (pos c) := by
    intro c d hcd
    obtain ⟨js, hjs⟩ := hlater c d hcd
    rw [hjs]
    exact rk_later_lt (pos c) (seat c) (hgen c) js
  have hpos : ∀ c, 1 ≤ rk (pos c) := fun c => rk_pos_of_genuine _ _ (hgen c)
  have h24 : rk (pos c) = 24 := by rw [hc]; exact rk_opening
  have := settled_by_rank (rank := fun c => rk (pos c)) hloc hr hpos p q c k
    (by rw [h24]; exact hk)
  rw [h24] at this
  exact this

end Stab
