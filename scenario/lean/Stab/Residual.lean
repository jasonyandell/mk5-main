-- SPDX-FileCopyrightText: Jason Yandell
-- SPDX-License-Identifier: CC0-1.0

/-!
# Q2: the residual congruence of a decision interface
(Jason's note *A minimal comparison algebra for a declared 42 family*, 2026-10-21,
sections "Whole-ending family and interface" and "The exact minimum"; Kozen,
*Automata and Computability*, lectures 13-16)

A deterministic decision interface is a Moore machine: contexts `S`, block labels `B`,
outputs `O`, a total transition `δ` and an output map `out`. The 42 interface is the
totalization of a partial one (`Next`, `Ctx`): a block from a live decision leads to the
next decision, to HALT after the final tile, or to REJECT when it has no positive
probability; HALT and REJECT are sinks.

Mathlib has Myhill-Nerode for languages (`Mathlib/Computability/MyhillNerode.lean`), but
this project's Lean has no Mathlib, and the note's object is a Moore machine with a
general output, not a language. The argument is short, so it is proved here directly.

* `Res s t`: every block sequence gives equal outputs. It is an equivalence
  (`res_equivalence`), a right congruence preserving the output (`res_out`,
  `res_step`), the coarsest such (`res_coarsest`), and its bottom-up form is
  "equal output and equivalent labelled successors" (`res_iff`).
* Lower bound: an exact deterministic realization never merges inequivalent contexts
  (`realization_separates`), so pairwise inequivalent contexts land on distinct
  machine states (`realization_card`). The quotient machine attains it
  (`quotient_realizes`, `quotient_states`).
* Partition refinement: `Round k` is the round-`k` partition from the output partition.
  It is "agree on all block sequences of length at most `k`" (`Round_iff`), `Res` is
  their intersection (`res_iff_all`), and once a round repeats, it is `Res`
  (`refine_stable`): the bottom-up and the refinement constructions agree.
* The depth ladder (`ladder`): if a level function is zero exactly where the output
  already decides the class and each block lowers it, refinement is exact at depth
  `max level`. For 42 (`ladder_42`): with at most `n` own tiles in hand, refinement is
  exact after `n - 1` blocks. The final forced tile is level zero.
* The bound is attained (`one_block_not_enough`): three own tiles, two inequivalent
  roots that one block cannot separate. "The last tile is forced, so refinement
  stabilizes after one block" is false in general; the course's shape ladder
  (11, 462, 477, 477) needs the second block.
-/

namespace Stab.Residual

section Moore
variable {S B O : Type} (δ : S → B → S) (out : S → O)

/-- Feed a block sequence. -/
def run : S → List B → S
  | s, [] => s
  | s, b :: u => run (δ s b) u

/-- The residual equivalence: every block sequence gives equal outputs. -/
def Res (s t : S) : Prop := ∀ u, out (run δ s u) = out (run δ t u)

theorem res_equivalence : Equivalence (Res δ out) where
  refl _ _ := rfl
  symm h u := (h u).symm
  trans h h' u := (h u).trans (h' u)

theorem res_out {s t : S} (h : Res δ out s t) : out s = out t := h []

/-- A right congruence: equivalent contexts stay equivalent after any block. -/
theorem res_step {s t : S} (h : Res δ out s t) (b : B) : Res δ out (δ s b) (δ t b) :=
  fun u => h (b :: u)

/-- Bottom-up form: equal output and equivalent successors under every label. -/
theorem res_iff (s t : S) :
    Res δ out s t ↔ out s = out t ∧ ∀ b, Res δ out (δ s b) (δ t b) := by
  constructor
  · intro h; exact ⟨res_out δ out h, res_step δ out h⟩
  · rintro ⟨ho, hs⟩ u
    cases u with
    | nil => exact ho
    | cons b u => exact hs b u

/-- The coarsest: every right congruence that preserves the output refines `Res`. -/
theorem res_coarsest (r : S → S → Prop) (hout : ∀ s t, r s t → out s = out t)
    (hstep : ∀ s t b, r s t → r (δ s b) (δ t b)) : ∀ s t, r s t → Res δ out s t := by
  intro s t h u
  induction u generalizing s t with
  | nil => exact hout s t h
  | cons b u ih => exact ih (δ s b) (δ t b) (hstep s t b h)

/-! ## The lower bound and the quotient that attains it -/

/-- A deterministic machine with states `Q`, initialized by `ι`, realizes the interface
exactly when it gives the required output after every block sequence. -/
def Realizes {Q : Type} (δ' : Q → B → Q) (out' : Q → O) (ι : S → Q) : Prop :=
  ∀ s u, out' (run δ' (ι s) u) = out (run δ s u)

theorem realization_separates {Q : Type} (δ' : Q → B → Q) (out' : Q → O) (ι : S → Q)
    (h : Realizes δ out δ' out' ι) {s t : S} (he : ι s = ι t) : Res δ out s t := by
  intro u; rw [← h s u, ← h t u, he]

/-- Pairwise inequivalent contexts occupy pairwise distinct states of any exact
realization: the machine has at least as many states as the quotient has classes. -/
theorem realization_card {Q : Type} (δ' : Q → B → Q) (out' : Q → O) (ι : S → Q)
    (h : Realizes δ out δ' out' ι) (cs : List S)
    (hcs : cs.Pairwise (fun s t => ¬ Res δ out s t)) : (cs.map ι).Nodup := by
  unfold List.Nodup
  rw [List.pairwise_map]
  exact hcs.imp (fun hne he => hne (realization_separates δ out δ' out' ι h he))

def setoid : Setoid S := ⟨Res δ out, res_equivalence δ out⟩

/-- The quotient machine. -/
def qδ (q : Quotient (setoid δ out)) (b : B) : Quotient (setoid δ out) :=
  Quotient.lift (fun s => Quotient.mk (setoid δ out) (δ s b))
    (fun _ _ h => Quotient.sound (res_step δ out h b)) q

def qout (q : Quotient (setoid δ out)) : O :=
  Quotient.lift out (fun _ _ h => res_out δ out h) q

theorem q_run (s : S) (u : List B) :
    run (qδ δ out) (Quotient.mk (setoid δ out) s) u = Quotient.mk (setoid δ out) (run δ s u) := by
  induction u generalizing s with
  | nil => rfl
  | cons b u ih => exact ih (δ s b)

theorem quotient_realizes : Realizes δ out (qδ δ out) (qout δ out) (Quotient.mk (setoid δ out)) := by
  intro s u; rw [q_run]; rfl

/-- The quotient identifies exactly the equivalent contexts: it attains the bound. -/
theorem quotient_states (s t : S) :
    Quotient.mk (setoid δ out) s = Quotient.mk (setoid δ out) t ↔ Res δ out s t :=
  ⟨Quotient.exact, fun h => Quotient.sound h⟩

/-! ## Partition refinement and its rounds -/

/-- Round `k` of Moore refinement from the output partition. -/
def Round : Nat → S → S → Prop
  | 0, s, t => out s = out t
  | k + 1, s, t => Round k s t ∧ ∀ b, Round k (δ s b) (δ t b)

/-- Round `k` distinguishes exactly what block sequences of length at most `k` reveal. -/
theorem Round_iff : ∀ (k : Nat) (s t : S),
    Round δ out k s t ↔ ∀ u : List B, u.length ≤ k → out (run δ s u) = out (run δ t u)
  | 0, s, t => by
    constructor
    · intro h u hu
      cases u with
      | nil => exact h
      | cons _ _ => simp at hu
    · intro h; exact h [] (Nat.le_refl 0)
  | k + 1, s, t => by
    simp only [Round]
    constructor
    · rintro ⟨h, hb⟩ u hu
      cases u with
      | nil => exact (Round_iff k s t).mp h [] (Nat.zero_le _)
      | cons b u =>
        exact (Round_iff k (δ s b) (δ t b)).mp (hb b) u (by simp at hu; omega)
    · intro h
      refine ⟨(Round_iff k s t).mpr (fun u hu => h u (by omega)), fun b => ?_⟩
      exact (Round_iff k (δ s b) (δ t b)).mpr (fun u hu => h (b :: u) (by simp; omega))

theorem Round_of_res {s t : S} (h : Res δ out s t) (k : Nat) : Round δ out k s t :=
  (Round_iff δ out k s t).mpr (fun u _ => h u)

theorem Round_out : ∀ {k : Nat} {s t : S}, Round δ out k s t → out s = out t
  | 0, _, _, h => h
  | _ + 1, _, _, h => Round_out h.1

/-- `Res` is the intersection of the rounds. -/
theorem res_iff_all (s t : S) : Res δ out s t ↔ ∀ k, Round δ out k s t := by
  constructor
  · exact fun h k => Round_of_res δ out h k
  · intro h u
    exact (Round_iff δ out u.length s t).mp (h u.length) u (Nat.le_refl _)

/-- Once a round repeats, refinement has stopped and the round is `Res`. -/
theorem refine_stable (k : Nat) (hk : ∀ s t, Round δ out (k + 1) s t ↔ Round δ out k s t) :
    ∀ s t, Round δ out k s t ↔ Res δ out s t := by
  have up : ∀ j s t, Round δ out (k + j) s t ↔ Round δ out k s t := by
    intro j
    induction j with
    | zero => intro s t; exact Iff.rfl
    | succ j ih =>
      intro s t
      show (Round δ out (k + j) s t ∧ ∀ b, Round δ out (k + j) (δ s b) (δ t b)) ↔ _
      rw [ih s t]
      simp only [ih]
      exact hk s t
  intro s t
  constructor
  · intro h
    rw [res_iff_all]
    intro j
    by_cases hj : j ≤ k
    · exact (Round_iff δ out k s t).mp h |> fun h' => (Round_iff δ out j s t).mpr
        (fun u hu => h' u (Nat.le_trans hu hj))
    · have := (up (j - k) s t).mpr h
      rwa [show k + (j - k) = j by omega] at this
  · exact fun h => Round_of_res δ out h k

/-! ## The depth ladder -/

/-- If level zero means the output already decides the class, and every block lowers
the level by one (or keeps it at zero), round `k` is exact for contexts of level at
most `k`. -/
theorem ladder (ℓ : S → Nat)
    (h0 : ∀ s t, ℓ s = 0 → ℓ t = 0 → out s = out t → Res δ out s t)
    (hdown : ∀ s b, ℓ (δ s b) ≤ ℓ s - 1) :
    ∀ (k : Nat) (s t : S), ℓ s ≤ k → ℓ t ≤ k → Round δ out k s t → Res δ out s t := by
  intro k
  induction k with
  | zero => intro s t hs ht h; exact h0 s t (by omega) (by omega) h
  | succ k ih =>
    intro s t hs ht h
    rw [res_iff]
    refine ⟨Round_out δ out h, fun b => ?_⟩
    exact ih (δ s b) (δ t b) (by have := hdown s b; omega) (by have := hdown t b; omega) (h.2 b)

/-- With every level at most `L`, refinement is exact at round `L`. -/
theorem ladder_exact (ℓ : S → Nat) (L : Nat)
    (h0 : ∀ s t, ℓ s = 0 → ℓ t = 0 → out s = out t → Res δ out s t)
    (hdown : ∀ s b, ℓ (δ s b) ≤ ℓ s - 1) (hL : ∀ s, ℓ s ≤ L) (s t : S) :
    Round δ out L s t ↔ Res δ out s t :=
  ⟨ladder δ out ℓ h0 hdown L s t (hL s) (hL t), fun h => Round_of_res δ out h L⟩

end Moore

/-! ## The 42 interface: totalization, HALT and REJECT -/

/-- Where a block leads from a live decision. -/
inductive Next (S : Type) | to (s : S) | halt | reject

/-- A totalized context. -/
inductive Ctx (S : Type) | live (s : S) | halt | reject

/-- A totalized output: a live decision's output (legal menu, maximizing set), or a sink. -/
inductive Out (O : Type) | dec (o : O) | halt | reject

section FortyTwo
variable {S B O : Type} [DecidableEq B] (next : S → B → Next S) (dout : S → O)

def tδ : Ctx S → B → Ctx S
  | .live s, b => match next s b with
    | .to s' => .live s'
    | .halt => .halt
    | .reject => .reject
  | .halt, _ => .reject
  | .reject, _ => .reject

def tout : Ctx S → Out O
  | .live s => .dec (dout s)
  | .halt => .halt
  | .reject => .reject

/-- The 42 hypotheses, as data: `hand s` own tiles held at decision `s`; a live block
plays one of them; at the final tile (one in hand) the block that plays it leads to
HALT and every other block to REJECT, and which block that is can be read from the
output (the menu names the tile). -/
structure Hand (hand : S → Nat) (final : O → B) : Prop where
  pos : ∀ s, 1 ≤ hand s
  play : ∀ s b s', next s b = .to s' → hand s' + 1 = hand s
  last : ∀ s b, hand s = 1 → next s b = if b = final (dout s) then Next.halt else Next.reject

/-- The level of a context: own decisions still to come after this one, the final forced
tile being zero. -/
def level (hand : S → Nat) : Ctx S → Nat
  | .live s => hand s - 1
  | .halt => 0
  | .reject => 0

theorem level_down (hand : S → Nat) (final : O → B) (hh : Hand next dout hand final)
    (c : Ctx S) (b : B) : level hand (tδ next c b) ≤ level hand c - 1 := by
  cases c with
  | live s =>
    simp only [tδ]
    cases h : next s b with
    | to s' => simp only [level]; have := hh.play s b s' h; have := hh.pos s'; omega
    | halt => simp [level]
    | reject => simp [level]
  | halt => simp [tδ, level]
  | reject => simp [tδ, level]

theorem sink_res (c : Ctx S) (hc : c = .halt ∨ c = .reject) (u : List B) (hu : u ≠ []) :
    run (tδ next) c u = .reject := by
  induction u generalizing c with
  | nil => exact absurd rfl hu
  | cons b u ih =>
    rcases hc with rfl | rfl <;>
    · cases u with
      | nil => rfl
      | cons b' u' => exact ih .reject (Or.inr rfl) (List.cons_ne_nil _ _)

theorem level_zero (hand : S → Nat) (final : O → B) (hh : Hand next dout hand final)
    (c d : Ctx S) (hc : level hand c = 0) (hd : level hand d = 0)
    (ho : tout dout c = tout dout d) : Res (tδ next) (tout dout) c d := by
  -- Two sinks with the same output are the same sink; two final tiles with the same
  -- output have the same transitions.
  have same_step : ∀ s t, hand s = 1 → hand t = 1 → dout s = dout t →
      ∀ b, tδ next (.live s) b = tδ next (.live t) b := by
    intro s t hs ht he b
    simp only [tδ, hh.last s b hs, hh.last t b ht, he]
  cases c with
  | live s =>
    cases d with
    | live t =>
      simp only [tout, Out.dec.injEq] at ho
      have hs : hand s = 1 := by simp [level] at hc; have := hh.pos s; omega
      have ht : hand t = 1 := by simp [level] at hd; have := hh.pos t; omega
      rw [res_iff]
      refine ⟨by simp [tout, ho], fun b => ?_⟩
      rw [same_step s t hs ht ho b]
      exact (res_equivalence _ _).refl _
    | halt => simp [tout] at ho
    | reject => simp [tout] at ho
  | halt =>
    cases d with
    | live t => simp [tout] at ho
    | halt => exact (res_equivalence _ _).refl _
    | reject => simp [tout] at ho
  | reject =>
    cases d with
    | live t => simp [tout] at ho
    | halt => simp [tout] at ho
    | reject => exact (res_equivalence _ _).refl _

/-- The depth ladder for 42: with at most `n` own tiles in hand anywhere, refinement from
the output partition is exact after `n - 1` blocks. Three tiles (the Gus course): two
blocks. Four tiles (the note's ending): three. -/
theorem ladder_42 (hand : S → Nat) (final : O → B) (hh : Hand next dout hand final)
    (n : Nat) (hn : ∀ s, hand s ≤ n) (c d : Ctx S) :
    Round (tδ next) (tout dout) (n - 1) c d ↔ Res (tδ next) (tout dout) c d := by
  apply ladder_exact (tδ next) (tout dout) (level hand) (n - 1)
  · exact level_zero next dout hand final hh
  · exact level_down next dout hand final hh
  · intro c; cases c with
    | live s => simp only [level]; have := hn s; omega
    | halt => simp [level]
    | reject => simp [level]

end FortyTwo

/-! ## The bound is attained: one block does not suffice with three tiles -/

namespace Witness

/-- Two roots `r1, r2` (three tiles), two second decisions `a1, a2` (two tiles), one
final tile `f`. The roots look alike for one block; the second decisions differ in which
observation they accept, so the roots differ at the second block. -/
inductive St | r1 | r2 | a1 | a2 | f
  deriving DecidableEq

def hand : St → Nat
  | .r1 | .r2 => 3
  | .a1 | .a2 => 2
  | .f => 1

/-- Block labels: `p` plays and sees one thing, `q` plays and sees another. -/
inductive Blk | p | q
  deriving DecidableEq

/-- Outputs: the roots share one, the seconds share one, the final tile has its own. -/
def dout : St → Nat
  | .r1 | .r2 => 0
  | .a1 | .a2 => 1
  | .f => 2

def final (_ : Nat) : Blk := .p

def next : St → Blk → Next St
  | .r1, .p => .to .a1
  | .r2, .p => .to .a2
  | .a1, .p => .to .f
  | .a2, .q => .to .f
  | .f, .p => .halt
  | _, _ => .reject

theorem hand_ok : Hand next dout hand final where
  pos s := by cases s <;> decide
  play s b s' h := by cases s <;> cases b <;> cases s' <;> simp_all [next, hand]
  last s b hs := by cases s <;> cases b <;> simp_all [next, hand, final]

/-- The roots agree on every block sequence of length one but not on all of them. -/
theorem one_block_not_enough :
    Round (tδ next) (tout dout) 1 (.live .r1) (.live .r2) ∧
      ¬ Res (tδ next) (tout dout) (.live .r1) (.live .r2) := by
  constructor
  · refine ⟨rfl, fun b => ?_⟩
    cases b <;> rfl
  · intro h
    have := h [.p, .p]
    simp [run, tδ, next, tout] at this

end Witness

end Stab.Residual
