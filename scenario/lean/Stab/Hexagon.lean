-- SPDX-FileCopyrightText: Jason Yandell
-- SPDX-License-Identifier: CC0-1.0

/-!
# Q1: the local comparison algebra and its hexagon
(Jason's note *A minimal comparison algebra for a declared 42 family*, 2026-10-21,
section "A concrete local algebra")

The viewer holds `{3-1, 4-1, 5-1}`, every first tile wins the trick, the second lead is
chosen with no observation in between, and the last tile is forced. A viewer program is
one of the six orders of the three tiles. The note computes, atom by atom, that the six
orders collapse to three success-indicator vectors `A, B, C` (its table, `vec` below).
That collapse is the fixture's computed fact; here it is the definition of an order's
success, so everything proved below is conditional on the note's table.

Atoms are any finite list `ws` with masses `μ`, rational and unnormalized; the total mass
is `m = E[1]`. With `x = E[A - B]`, `y = E[C - B]`, `β = E[B]`:

* `Q31 = β + max x 0`, `Q41 = β + max y 0`, `Q51 = β + max x y` (`Q_formulas`);
* after each first tile the next comparison is `x`, `y`, `x - y` (`next_comparison`);
* no mixture of the six programs beats the best first tile (`mixing_no_gain`);
* no single linear statistic carries both `x` and `y` (`no_scalar_statistic`): the
  contrast rank is two;
* the cone: `(x, y, m)` comes from a nonnegative belief iff `0 ≤ m`, `|x| ≤ m`,
  `|y| ≤ m`, `|x - y| ≤ m` (`cone_of_belief`, `belief_of_cone`); with `m = 1`, the
  hexagon;
* the sign patterns of `(x, y, x - y)`, which are the weak orders of
  `(E[A], E[B], E[C])`, number exactly 13 and all 13 occur on the hexagon
  (`thirteen_patterns`); the three second decisions read them off, and the root decision
  is a function of them (`root_of_pattern`), so these are the sequential decision classes
  (`seq_classes`);
* the weak orders of the first-action triple `(Q31, Q41, Q51)` alone number exactly 4
  (`four_root_orders`): `Q51` is never the unique maximum. The 13 are classes of the
  whole sequential behaviour, not of the root values;
* the note's one-parameter segment `x = (5p-1)/(1+18p)`, `y = (p-1)/(1+18p)` satisfies
  `19x - 23y = 4` and crosses exactly five classes (`segment_*`).
-/

namespace Stab.Hexagon

/-! ## Atoms, masses, expectations -/

/-- A Boolean as a 0/1 rational. -/
def ind (b : Bool) : Rat := if b then 1 else 0

/-- The unnormalized expectation of `f` under masses `μ` over the atom list `ws`. -/
def E {W : Type} (ws : List W) (μ : W → Rat) (f : W → Rat) : Rat :=
  (ws.map (fun w => μ w * f w)).sum

theorem E_nil {W : Type} (μ f : W → Rat) : E [] μ f = 0 := rfl

theorem E_cons {W : Type} (w : W) (ws : List W) (μ f : W → Rat) :
    E (w :: ws) μ f = μ w * f w + E ws μ f := by
  simp [E, List.map_cons, List.sum_cons]

theorem E_sub {W : Type} (ws : List W) (μ f g : W → Rat) :
    E ws μ (fun w => f w - g w) = E ws μ f - E ws μ g := by
  induction ws with
  | nil => simp only [E_nil]; grind
  | cons w ws ih => rw [E_cons, E_cons, E_cons, ih]; grind

/-! ## The six programs and the note's table -/

/-- The viewer's three tiles: 3-1, 4-1, 5-1. -/
inductive Tile | t31 | t41 | t51
  deriving DecidableEq, Repr

/-- The three success vectors of the note. -/
inductive Vec | A | B | C
  deriving DecidableEq, Repr

/-- A viewer program: the first tile and the second; the third is forced. -/
inductive Order
  | o31_41_51 | o31_51_41 | o41_31_51 | o41_51_31 | o51_31_41 | o51_41_31
  deriving DecidableEq, Repr

def Order.first : Order → Tile
  | .o31_41_51 | .o31_51_41 => .t31
  | .o41_31_51 | .o41_51_31 => .t41
  | .o51_31_41 | .o51_41_31 => .t51

/-- The note's table: which success vector each order realizes. -/
def vec : Order → Vec
  | .o31_41_51 => .A
  | .o31_51_41 => .B
  | .o41_31_51 => .C
  | .o41_51_31 => .B
  | .o51_31_41 => .C
  | .o51_41_31 => .A

def allOrders : List Order :=
  [.o31_41_51, .o31_51_41, .o41_31_51, .o41_51_31, .o51_31_41, .o51_41_31]

/-- The two orders that start with a tile: the menu of the second decision. -/
def after : Tile → Order × Order
  | .t31 => (.o31_41_51, .o31_51_41)
  | .t41 => (.o41_31_51, .o41_51_31)
  | .t51 => (.o51_41_31, .o51_31_41)

/-- `after t` lists exactly the orders whose first tile is `t`. -/
theorem after_spec (t : Tile) (o : Order) :
    o.first = t ↔ (o = (after t).1 ∨ o = (after t).2) := by
  cases t <;> cases o <;> decide

section Belief
variable {W : Type} (ws : List W) (μ : W → Rat) (A B C : W → Bool)

def pick : Vec → W → Bool
  | .A => A
  | .B => B
  | .C => C

/-- The expected success of an order. -/
def val (o : Order) : Rat := E ws μ (fun w => ind (pick A B C (vec o) w))

def x : Rat := E ws μ (fun w => ind (A w) - ind (B w))
def y : Rat := E ws μ (fun w => ind (C w) - ind (B w))
def β : Rat := E ws μ (fun w => ind (B w))
def mass : Rat := E ws μ (fun _ => 1)

/-- The optimized value of a first tile: the better of its two continuations. No
observation intervenes, so the belief is the same at the second decision. -/
def Q (t : Tile) : Rat := max (val ws μ A B C (after t).1) (val ws μ A B C (after t).2)

theorem x_eq : x ws μ A B = val ws μ A B C .o31_41_51 - val ws μ A B C .o31_51_41 := by
  simp only [x, val, vec, pick]; exact E_sub ws μ _ _

theorem y_eq : y ws μ B C = val ws μ A B C .o41_31_51 - val ws μ A B C .o41_51_31 := by
  simp only [y, val, vec, pick]; exact E_sub ws μ _ _

/-- After each first tile, the next comparison (first listed continuation minus the
second) is `x`, `y` and `x - y`. -/
theorem next_comparison :
    val ws μ A B C (after .t31).1 - val ws μ A B C (after .t31).2 = x ws μ A B ∧
    val ws μ A B C (after .t41).1 - val ws μ A B C (after .t41).2 = y ws μ B C ∧
    val ws μ A B C (after .t51).1 - val ws μ A B C (after .t51).2
      = x ws μ A B - y ws μ B C := by
  refine ⟨(x_eq ws μ A B C).symm, (y_eq ws μ A B C).symm, ?_⟩
  have hx := x_eq ws μ A B C
  have hy := y_eq ws μ A B C
  simp only [after]
  simp only [val, vec, pick] at hx hy ⊢
  grind

/-- The note's first-action values. -/
theorem Q_formulas :
    Q ws μ A B C .t31 = β ws μ B + max (x ws μ A B) 0 ∧
    Q ws μ A B C .t41 = β ws μ B + max (y ws μ B C) 0 ∧
    Q ws μ A B C .t51 = β ws μ B + max (x ws μ A B) (y ws μ B C) := by
  have hx := x_eq ws μ A B C
  have hy := y_eq ws μ A B C
  simp only [Q, after, β]
  simp only [val, vec, pick] at hx hy ⊢
  refine ⟨?_, ?_, ?_⟩ <;> grind

/-- Randomizing over the six programs cannot beat the best first tile. -/
theorem mixing_no_gain (wt : Order → Rat) (hw : ∀ o, 0 ≤ wt o)
    (hsum : (allOrders.map wt).sum = 1) :
    (allOrders.map (fun o => wt o * val ws μ A B C o)).sum ≤
      max (Q ws μ A B C .t31) (max (Q ws μ A B C .t41) (Q ws μ A B C .t51)) := by
  generalize hM : max (Q ws μ A B C .t31) (max (Q ws μ A B C .t41) (Q ws μ A B C .t51)) = M
  have hle : ∀ o, val ws μ A B C o ≤ M := by
    intro o; rw [← hM]; cases o <;> simp only [Q, after] <;> grind
  have term : ∀ o, wt o * val ws μ A B C o ≤ wt o * M :=
    fun o => Rat.mul_le_mul_of_nonneg_left (hle o) (hw o)
  have hs : (allOrders.map (fun o => wt o * M)).sum = M := by
    simp only [allOrders, List.map_cons, List.map_nil, List.sum_cons, List.sum_nil] at hsum ⊢
    grind
  simp only [allOrders, List.map_cons, List.map_nil, List.sum_cons, List.sum_nil] at hs ⊢
  have := term .o31_41_51; have := term .o31_51_41; have := term .o41_31_51
  have := term .o41_51_31; have := term .o51_31_41; have := term .o51_41_31
  grind

end Belief

/-! ## Contrast rank two -/

/-- The six non-constant atom types `(A, B, C)`; `000` and `111` add mass and nothing else. -/
inductive Ty | t100 | t010 | t001 | t110 | t101 | t011
  deriving DecidableEq, Repr

def Ty.a : Ty → Bool
  | .t100 | .t110 | .t101 => true
  | _ => false
def Ty.b : Ty → Bool
  | .t010 | .t110 | .t011 => true
  | _ => false
def Ty.c : Ty → Bool
  | .t001 | .t101 | .t011 => true
  | _ => false

def allTy : List Ty := [.t100, .t010, .t001, .t110, .t101, .t011]

/-- A belief concentrated on one type. -/
def point (t : Ty) : Ty → Rat := fun s => if s = t then 1 else 0

/-- No scalar linear statistic `s(μ) = Σ μ w · c w` determines both `x` and `y` linearly:
the two contrasts are independent, so any linear statistic preserving the comparisons
has rank at least two. -/
theorem no_scalar_statistic (c : Ty → Rat) :
    ¬ ∃ a b : Rat, ∀ μ : Ty → Rat,
      x allTy μ Ty.a Ty.b = a * E allTy μ c ∧ y allTy μ Ty.b Ty.c = b * E allTy μ c := by
  rintro ⟨a, b, h⟩
  have h1 := h (point .t100)
  have h2 := h (point .t001)
  simp [x, y, E, allTy, point, ind, Ty.a, Ty.b, Ty.c] at h1 h2
  obtain ⟨h1x, h1y⟩ := h1
  obtain ⟨h2x, h2y⟩ := h2
  -- at t100, x = 1 and y = 0; at t001, x = 0 and y = 1
  have e1 : a * c .t100 = 1 := by grind
  have e2 : b * c .t100 = 0 := by grind
  have e4 : b * c .t001 = 1 := by grind
  rcases Rat.mul_eq_zero.mp e2 with hb | hc
  · rw [hb] at e4; grind
  · rw [hc] at e1; grind

/-! ## The cone and the hexagon -/

/-- The note's cone `m ≥ 0, |x| ≤ m, |y| ≤ m, |x - y| ≤ m`. -/
def InCone (x y m : Rat) : Prop :=
  0 ≤ m ∧ -m ≤ x ∧ x ≤ m ∧ -m ≤ y ∧ y ≤ m ∧ -m ≤ x - y ∧ x - y ≤ m

/-- Every nonnegative belief over any atoms lands in the cone. -/
theorem cone_of_belief {W : Type} (ws : List W) (μ : W → Rat) (A B C : W → Bool)
    (hμ : ∀ w ∈ ws, 0 ≤ μ w) :
    InCone (x ws μ A B) (y ws μ B C) (mass ws μ) := by
  induction ws with
  | nil => simp only [InCone, x, y, mass, E_nil]; grind
  | cons w ws ih =>
    have ih := ih (fun v hv => hμ v (List.mem_cons_of_mem w hv))
    have hw := hμ w (List.mem_cons_self)
    simp only [InCone, x, y, mass, E_cons] at ih ⊢
    generalize E ws μ (fun w => ind (A w) - ind (B w)) = X at ih ⊢
    generalize E ws μ (fun w => ind (C w) - ind (B w)) = Y at ih ⊢
    generalize E ws μ (fun _ => 1) = M at ih ⊢
    generalize μ w = u at hw ⊢
    cases A w <;> cases B w <;> cases C w <;> simp [ind] <;> grind

/-- Every point of the cone is realized by a nonnegative belief on the six types. -/
theorem belief_of_cone (x0 y0 m : Rat) (h : InCone x0 y0 m) :
    ∃ μ : Ty → Rat, (∀ t, 0 ≤ μ t) ∧
      x allTy μ Ty.a Ty.b = x0 ∧ y allTy μ Ty.b Ty.c = y0 ∧ mass allTy μ = m := by
  obtain ⟨hm, h1, h2, h3, h4, h5, h6⟩ := h
  -- Each sector of the hexagon is spanned by two adjacent vertices; the unused mass is
  -- split evenly between the opposite vertices (1,0) and (-1,0).
  let mk (a100 a010 a001 a110 a101 a011 : Rat) : Ty → Rat
    | .t100 => a100 | .t010 => a010 | .t001 => a001
    | .t110 => a110 | .t101 => a101 | .t011 => a011
  have ev : ∀ a100 a010 a001 a110 a101 a011 : Rat,
      x allTy (mk a100 a010 a001 a110 a101 a011) Ty.a Ty.b = a100 - a010 + a101 - a011 ∧
      y allTy (mk a100 a010 a001 a110 a101 a011) Ty.b Ty.c = -a010 + a001 - a110 + a101 ∧
      mass allTy (mk a100 a010 a001 a110 a101 a011)
        = a100 + a010 + a001 + a110 + a101 + a011 := by
    intro a100 a010 a001 a110 a101 a011
    simp [x, y, mass, E, allTy, mk, ind, Ty.a, Ty.b, Ty.c]; grind
  have pos : ∀ a100 a010 a001 a110 a101 a011 : Rat, 0 ≤ a100 → 0 ≤ a010 → 0 ≤ a001 →
      0 ≤ a110 → 0 ≤ a101 → 0 ≤ a011 → ∀ t, 0 ≤ mk a100 a010 a001 a110 a101 a011 t := by
    intro _ _ _ _ _ _ p1 p2 p3 p4 p5 p6 t; cases t <;> assumption
  by_cases hx : 0 ≤ x0 <;> by_cases hy : 0 ≤ y0 <;> by_cases hxy : y0 ≤ x0
  · -- x ≥ y ≥ 0: (x-y)·(1,0) + y·(1,1)
    let s := (m - x0) / 2
    refine ⟨mk (x0 - y0 + s) 0 0 0 y0 s, pos _ _ _ _ _ _ ?_ ?_ ?_ ?_ ?_ ?_, ?_⟩ <;>
      first | (obtain ⟨e1, e2, e3⟩ := ev (x0 - y0 + s) 0 0 0 y0 s; grind) | grind
  · -- y > x ≥ 0: x·(1,1) + (y-x)·(0,1)
    let s := (m - y0) / 2
    refine ⟨mk s 0 (y0 - x0) 0 x0 s, pos _ _ _ _ _ _ ?_ ?_ ?_ ?_ ?_ ?_, ?_⟩ <;>
      first | (obtain ⟨e1, e2, e3⟩ := ev s 0 (y0 - x0) 0 x0 s; grind) | grind
  · -- x ≥ 0 > y: x·(1,0) + (-y)·(0,-1)
    let s := (m - x0 + y0) / 2
    refine ⟨mk (x0 + s) 0 0 (-y0) 0 s, pos _ _ _ _ _ _ ?_ ?_ ?_ ?_ ?_ ?_, ?_⟩ <;>
      first | (obtain ⟨e1, e2, e3⟩ := ev (x0 + s) 0 0 (-y0) 0 s; grind) | grind
  · exfalso; grind
  · exfalso; grind
  · -- y ≥ 0 > x: (-x)·(-1,0) + y·(0,1)
    let s := (m - y0 + x0) / 2
    refine ⟨mk s 0 y0 0 0 (-x0 + s), pos _ _ _ _ _ _ ?_ ?_ ?_ ?_ ?_ ?_, ?_⟩ <;>
      first | (obtain ⟨e1, e2, e3⟩ := ev s 0 y0 0 0 (-x0 + s); grind) | grind
  · -- 0 > x ≥ y: (-x)·(-1,-1) + (x-y)·(0,-1)
    let s := (m + y0) / 2
    refine ⟨mk s (-x0) 0 (x0 - y0) 0 s, pos _ _ _ _ _ _ ?_ ?_ ?_ ?_ ?_ ?_, ?_⟩ <;>
      first | (obtain ⟨e1, e2, e3⟩ := ev s (-x0) 0 (x0 - y0) 0 s; grind) | grind
  · -- 0 > y > x: (-y)·(-1,-1) + (y-x)·(-1,0)
    let s := (m + x0) / 2
    refine ⟨mk s (-y0) 0 0 0 (y0 - x0 + s), pos _ _ _ _ _ _ ?_ ?_ ?_ ?_ ?_ ?_, ?_⟩ <;>
      first | (obtain ⟨e1, e2, e3⟩ := ev s (-y0) 0 0 0 (y0 - x0 + s); grind) | grind

/-- The realization theorem, normalized: a pair `(x, y)` is attained by a probability
belief on the six non-constant types iff it lies in the hexagon. -/
theorem hexagon (x0 y0 : Rat) :
    (∃ μ : Ty → Rat, (∀ t, 0 ≤ μ t) ∧ mass allTy μ = 1 ∧
      x allTy μ Ty.a Ty.b = x0 ∧ y allTy μ Ty.b Ty.c = y0) ↔ InCone x0 y0 1 := by
  constructor
  · rintro ⟨μ, hμ, hm, rfl, rfl⟩
    rw [← hm]; exact cone_of_belief allTy μ _ _ _ (fun w _ => hμ w)
  · intro h
    obtain ⟨μ, hμ, hx, hy, hm⟩ := belief_of_cone x0 y0 1 h
    exact ⟨μ, hμ, hm, hx, hy⟩

/-! ## Sign patterns: thirteen sequential classes, four root orders -/

/-- The sign of a rational, as `-1, 0, 1`. -/
def sgn (q : Rat) : Int := if q < 0 then -1 else if q = 0 then 0 else 1

/-- The decision class at this context: the signs of `x`, `y`, `x - y`, which are the
signs of `E[A] - E[B]`, `E[C] - E[B]`, `E[A] - E[C]`: the weak order of the three
success vectors. -/
def pattern (x y : Rat) : Int × Int × Int := (sgn x, sgn y, sgn (x - y))

/-- The thirteen weak orders: six strict, six with one tie, the triple tie. -/
def thirteen : List (Int × Int × Int) :=
  [(1, 1, 1), (1, 1, -1), (-1, -1, 1), (-1, -1, -1), (1, -1, 1), (-1, 1, -1),
   (1, 1, 0), (-1, -1, 0), (1, 0, 1), (-1, 0, -1), (0, 1, -1), (0, -1, 1),
   (0, 0, 0)]

theorem thirteen_nodup : thirteen.Nodup := by decide

theorem thirteen_length : thirteen.length = 13 := rfl

/-- Every pair of rationals has one of the thirteen patterns. -/
theorem pattern_mem (x y : Rat) : pattern x y ∈ thirteen := by
  simp only [pattern, sgn]
  by_cases a : x < 0 <;> by_cases b : x = 0 <;> by_cases c : y < 0 <;> by_cases d : y = 0 <;>
    by_cases e : x - y < 0 <;> by_cases f : x - y = 0 <;> simp [a, b, c, d, e, f, thirteen] <;>
    grind

/-- Exactly the thirteen patterns occur on the hexagon, every one of them. -/
theorem thirteen_patterns (s : Int × Int × Int) :
    (∃ x y, InCone x y 1 ∧ pattern x y = s) ↔ s ∈ thirteen := by
  constructor
  · rintro ⟨x, y, _, rfl⟩; exact pattern_mem x y
  · intro h
    simp only [thirteen, List.mem_cons, List.not_mem_nil, or_false] at h
    have w : ∀ x y : Rat, InCone x y 1 → pattern x y = s → ∃ x y, InCone x y 1 ∧ pattern x y = s :=
      fun x y h1 h2 => ⟨x, y, h1, h2⟩
    rcases h with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl
    · exact w (2/3) (1/3) (by simp [InCone]; grind) (by simp [pattern, sgn]; grind)
    · exact w (1/3) (2/3) (by simp [InCone]; grind) (by simp [pattern, sgn]; grind)
    · exact w (-1/3) (-2/3) (by simp [InCone]; grind) (by simp [pattern, sgn]; grind)
    · exact w (-2/3) (-1/3) (by simp [InCone]; grind) (by simp [pattern, sgn]; grind)
    · exact w (1/2) (-1/2) (by simp [InCone]; grind) (by simp [pattern, sgn]; grind)
    · exact w (-1/2) (1/2) (by simp [InCone]; grind) (by simp [pattern, sgn]; grind)
    · exact w (1/2) (1/2) (by simp [InCone]; grind) (by simp [pattern, sgn]; grind)
    · exact w (-1/2) (-1/2) (by simp [InCone]; grind) (by simp [pattern, sgn]; grind)
    · exact w (1/2) 0 (by simp [InCone]; grind) (by simp [pattern, sgn]; grind)
    · exact w (-1/2) 0 (by simp [InCone]; grind) (by simp [pattern, sgn]; grind)
    · exact w 0 (1/2) (by simp [InCone]; grind) (by simp [pattern, sgn]; grind)
    · exact w 0 (-1/2) (by simp [InCone]; grind) (by simp [pattern, sgn]; grind)
    · exact w 0 0 (by simp [InCone]; grind) (by simp [pattern, sgn]; grind)

/-- The root decision class: the signs of `Q31 - Q41`, `Q31 - Q51`, `Q41 - Q51`, written
with the note's formulas (the baseline `β` cancels). -/
def rootClass (x y : Rat) : Int × Int × Int :=
  (sgn (max x 0 - max y 0), sgn (max x 0 - max x y), sgn (max y 0 - max x y))

/-- The root class as a function of the pattern. Both contrasts negative: `Q31 = Q41 > Q51`.
Otherwise `Q51 = max Q31 Q41`, and the larger contrast wins unless it is zero. -/
def rootOf (t : Int × Int × Int) : Int × Int × Int :=
  if t.1 = -1 ∧ t.2.1 = -1 then (0, 1, 1)
  else if t.2.2 = 1 then (if t.1 = 0 then (0, 0, 0) else (1, 0, -1))
  else if t.2.2 = -1 then (if t.2.1 = 0 then (0, 0, 0) else (-1, -1, 0))
  else (0, 0, 0)

theorem sgn_neg {q : Rat} (h : q < 0) : sgn q = -1 := by simp [sgn, h]
theorem sgn_zero {q : Rat} (h : q = 0) : sgn q = 0 := by simp [sgn, h]
theorem sgn_pos {q : Rat} (h : 0 < q) : sgn q = 1 := by
  simp only [sgn]; rw [if_neg (by grind), if_neg (by grind)]

theorem trich (q : Rat) : q < 0 ∨ q = 0 ∨ 0 < q := by grind

theorem sgn_eq_iff (q : Rat) (s : Int) :
    sgn q = s ↔ (s = -1 ∧ q < 0) ∨ (s = 0 ∧ q = 0) ∨ (s = 1 ∧ 0 < q) := by
  simp only [sgn]
  by_cases a : q < 0
  · rw [if_pos a]; grind
  · rw [if_neg a]; by_cases b : q = 0
    · rw [if_pos b]; grind
    · rw [if_neg b]; grind

/-- The root decision is a function of the pattern: the three second decisions, which
read the pattern off directly (`next_comparison`), determine everything. -/
theorem root_of_pattern (x y : Rat) : rootClass x y = rootOf (pattern x y) := by
  simp only [rootClass, pattern]
  rcases trich x with hx | hx | hx <;>
  rcases trich y with hy | hy | hy <;>
  rcases trich (x - y) with hxy | hxy | hxy <;>
  simp (disch := assumption) only [sgn_neg, sgn_zero, sgn_pos] <;>
  first
  | (exfalso; grind)
  | (simp only [rootOf, Int.reduceEq, Int.reduceNeg, if_true, if_false, and_true, and_false]
     simp only [Prod.mk.injEq]
     rw [sgn_eq_iff, sgn_eq_iff, sgn_eq_iff]
     grind)

theorem pattern_eq_root (x y x' y' : Rat) (h : pattern x y = pattern x' y') :
    rootClass x y = rootClass x' y' := by
  rw [root_of_pattern, root_of_pattern, h]

/-- The whole local decision output: the root class and the three second decisions
(after 3-1, 4-1, 5-1 the comparisons are `x`, `y`, `x - y`, by `next_comparison`). -/
def seqClass (x y : Rat) : (Int × Int × Int) × Int × Int × Int :=
  (rootClass x y, sgn x, sgn y, sgn (x - y))

/-- Two beliefs have the same sequential decision behaviour iff they have the same
pattern: the sequential classes are the thirteen patterns. -/
theorem seq_classes (x y x' y' : Rat) :
    seqClass x y = seqClass x' y' ↔ pattern x y = pattern x' y' := by
  constructor
  · intro h
    simp only [seqClass, Prod.mk.injEq] at h
    simp only [pattern, h]
  · intro h
    have hr := pattern_eq_root x y x' y' h
    simp only [pattern, Prod.mk.injEq] at h
    simp only [seqClass, hr, h]

/-- The four root orders: `Q31 = Q41 > Q51`, `Q31 = Q51 > Q41`, `Q41 = Q51 > Q31`, and the
triple tie. -/
def four : List (Int × Int × Int) := [(0, 1, 1), (1, 0, -1), (-1, -1, 0), (0, 0, 0)]

/-- The first-action triple alone takes exactly four weak orders on the hexagon: `Q51` is
never the unique maximum, and `Q51` is strictly worst only when `x, y < 0`. Thirteen is
the count of the sequential classes (`thirteen_patterns`), not of the root values. -/
theorem four_root_orders (s : Int × Int × Int) :
    (∃ x y, InCone x y 1 ∧ rootClass x y = s) ↔ s ∈ four := by
  constructor
  · rintro ⟨x, y, _, rfl⟩
    rw [root_of_pattern]
    have h := pattern_mem x y
    generalize pattern x y = t at h
    simp only [thirteen, List.mem_cons, List.not_mem_nil, or_false] at h
    rcases h with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;>
      decide
  · intro h
    simp only [four, List.mem_cons, List.not_mem_nil, or_false] at h
    rcases h with rfl | rfl | rfl | rfl
    · refine ⟨-1/2, -1/2, by simp [InCone]; grind, ?_⟩
      rw [root_of_pattern, pattern, sgn_neg (show (-1/2 : Rat) < 0 by grind),
        sgn_zero (show (-1/2 : Rat) - -1/2 = 0 by grind)]; rfl
    · refine ⟨1/2, 0, by simp [InCone]; grind, ?_⟩
      rw [root_of_pattern, pattern, sgn_pos (by grind), sgn_zero (by grind), sgn_pos (by grind)]; rfl
    · refine ⟨0, 1/2, by simp [InCone]; grind, ?_⟩
      rw [root_of_pattern, pattern, sgn_zero (by grind), sgn_pos (by grind), sgn_neg (by grind)]; rfl
    · refine ⟨0, 0, by simp [InCone]; grind, ?_⟩
      rw [root_of_pattern, pattern, sgn_zero (show (0 : Rat) - 0 = 0 by grind),
        sgn_zero (show (0 : Rat) = 0 by grind)]; rfl

/-! ## The note's one-parameter segment -/

/-- Division by a positive rational keeps the sign. -/
theorem sgn_div_pos (a d : Rat) (hd : 0 < d) : sgn (a / d) = sgn a := by
  have hi : 0 < d⁻¹ := Rat.inv_pos.mpr hd
  have e : a / d = d⁻¹ * a := by grind
  rw [e]
  simp only [sgn]
  by_cases a0 : a < 0
  · have : d⁻¹ * a < d⁻¹ * 0 := Rat.mul_lt_mul_of_pos_left a0 hi
    grind
  · by_cases a1 : a = 0
    · subst a1; grind
    · have : d⁻¹ * 0 < d⁻¹ * a := Rat.mul_lt_mul_of_pos_left (by grind) hi
      grind

def segX (p : Rat) : Rat := (5 * p - 1) / (1 + 18 * p)
def segY (p : Rat) : Rat := (p - 1) / (1 + 18 * p)

/-- The segment lies on the line `19x - 23y = 4`. -/
theorem segment_line (p : Rat) (hp : 0 ≤ p) : 19 * segX p - 23 * segY p = 4 := by
  have hd : 1 + 18 * p ≠ 0 := by grind
  simp only [segX, segY]; grind

theorem segment_pattern (p : Rat) (hp : 0 ≤ p) :
    pattern (segX p) (segY p) = (sgn (5 * p - 1), sgn (p - 1), sgn (4 * p)) := by
  have hd : 0 < 1 + 18 * p := by grind
  have e : segX p - segY p = (4 * p) / (1 + 18 * p) := by
    simp only [segX, segY]; grind
  simp only [pattern]; rw [e]; simp only [segX, segY, sgn_div_pos _ _ hd]

/-- The five local classes on the segment, at `p = 0`, `0 < p < 1/5`, `p = 1/5`,
`1/5 < p < 1`, `p = 1`. -/
theorem segment_classes (p : Rat) (h0 : 0 ≤ p) (h1 : p ≤ 1) :
    pattern (segX p) (segY p) =
      if p = 0 then (-1, -1, 0)
      else if p < 1/5 then (-1, -1, 1)
      else if p = 1/5 then (0, -1, 1)
      else if p < 1 then (1, -1, 1)
      else (1, 0, 1) := by
  rw [segment_pattern p h0]
  simp only [sgn]
  by_cases a : p = 0 <;> by_cases b : p < 1/5 <;> by_cases c : p = 1/5 <;> by_cases d : p < 1 <;>
    simp only [a, b, c, d, if_true, if_false] <;> grind

end Stab.Hexagon
