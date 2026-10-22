# LEAN report (research night 2026-10-12)

Assembled by the orchestrator from the Lean role's messages (the subagent report-file
guard refused the role's own writes of REPORT.md). Verified independently by the
orchestrator at 01:33 CDT: `lake build` 0.14 s warm (0.67 s clean), `lake env lean
Axioms.lean` prints only `propext`, `Classical.choice`, `Quot.sound` for every theorem;
`grep sorry Stab/*.lean` is empty.

Toolchain: `leanprover/lean4:v4.33.0-rc1`, no Mathlib; core `omega`, `simp`, `grind`.

```
cd /Users/jason/code/mk5-main/scratch/research-night/lean
rm -rf .lake/build && lake build      # 0.67 s
lake env lean Axioms.lean             # axiom check
```

Files: `Stab/Ladder.lean` (L1), `Stab/Rank.lean` (L2), `Stab/Coherence.lean` (L3),
`Stab/Sharp.lean` (the bound is attained), `Axioms.lean`.

## Two findings the prose glossed over

1. **The brief's L1 statement was off by one, and Lean proves it false**
   (`brief_indexing_off_by_one`). The response at `c` settles from iterate `h c + 1`,
   not `h c`. A coordinate with no dependencies still needs one application. Note 2 has
   it right because there `P^k = R^[k+1] U`, so `P^23` is the 24th application.
2. **The bound is reached** (`bound_attained`). An abstract local operator can keep the
   same root answer for `n` applications and change at application `n + 1`. This is
   note 2 section 2.3 in miniature: unchanged root moves certify nothing.

## L1: the stabilization theorem (note 2, section 2.1) — Stab/Ladder.lean

Coordinates `C` and actions `A` can be any types; neither needs to be finite.

```lean
def iter (R : (C → A) → (C → A)) : Nat → (C → A) → (C → A)
  | 0, p => p
  | k + 1, p => R (iter R k p)
def Local (dep : C → C → Prop) (R : (C → A) → (C → A)) : Prop :=
  ∀ (c : C) (p q : C → A), (∀ d, dep c d → p d = q d) → R p c = R q c
inductive Chain (dep : C → C → Prop) : C → Nat → Prop
  | nil (c : C) : Chain dep c 0
  | cons {c d : C} {n : Nat} : dep c d → Chain dep d n → Chain dep c (n + 1)
def IsHeight (dep : C → C → Prop) (h : C → Nat) : Prop := ∀ c d, dep c d → h d < h c
```

**Core theorem, proved.** Stated over chains, so the height is literally the longest
dependency chain from `c`.

```lean
theorem settle (hloc : Local dep R) :
    ∀ (m : Nat) (c : C), (∀ n, Chain dep c n → n ≤ m) →
      ∀ (p q : C → A) (k j : Nat), m < k → m < j → iter R k p c = iter R j q c
```

**Corollaries, proved.**

```lean
theorem chain_le_height (hh : IsHeight dep h) : ∀ {c n}, Chain dep c n → n ≤ h c
theorem stable_from_height (hloc : Local dep R) (hh : IsHeight dep h) (p : C → A) (c : C)
    (k : Nat) (hk : h c + 1 ≤ k) : iter R k p c = iter R (h c + 1) p c
theorem limit_indep ... (hk : h c + 1 ≤ k) : iter R k p c = iter R k q c
theorem fixed_point (hloc) (hh) (hH : ∀ c, h c ≤ H) (p : C → A) :
    R (iter R (H + 1) p) = iter R (H + 1) p
theorem fixed_point_unique (hloc) (hh) (hH : ∀ c, h c ≤ H) (p q : C → A) (hq : R q = q) :
    q = iter R (H + 1) p
```

`fixed_point_unique` goes beyond the brief: under a uniform height bound, `R` has
exactly one fixed point.

**The off-by-one, proved.** One coordinate with no dependencies, the constant-`true`
response, a `false` starting policy.

```lean
theorem brief_indexing_off_by_one :
    ¬ (∀ k, 0 ≤ k →
        iter (C := Unit) (A := Bool) (fun _ _ => true) k (fun _ => false) () =
        iter (C := Unit) (A := Bool) (fun _ _ => true) 0 (fun _ => false) ())
```

**Tightness, proved** (Stab/Sharp.lean). In `shiftR n`, coordinate `i < n` copies
coordinate `i+1`, and coordinates at or above `n` answer `true`. Local for the chain
`i → i+1` (`shiftR_local`) with height `n - i` (`shiftR_height`).

```lean
theorem bound_attained (n : Nat) :
    iter (shiftR n) n (fun _ => false) 0 = false ∧
    iter (shiftR n) (n + 1) (fun _ => false) 0 = true
```

The root holds one answer for n applications and flips at application n+1, so the
`h c + 1` bound is attained and a plateau of unchanged root moves certifies nothing.
Note 2's construction is a real game with payoffs; this is only an operator.

**Correspondence with the theory report's T2 (§3).** Lean's `Local dep R` is
theory's Loc; `IsHeight dep h` and `rank` are theory's Dec; `h`/`rank` is its rk.
Theory's base-independent `P* := R^[rk c + 1] P0 c` is `stable_from_height` together
with `limit_indep`; `fixed_point_unique` adds that P* is R's unique fixed point. The
two indexings agree on the +1.

## L2: rank bound and the 42 instance (note 2, section 2.2) — Stab/Rank.lean

Abstract part, proved:

```lean
theorem settled_by_rank (hloc : Local dep R)
    (hr : ∀ c d, dep c d → rank d < rank c) (hpos : ∀ c, 1 ≤ rank c)
    (p q : C → A) (c : C) (k : Nat) (hk : rank c ≤ k) :
    iter R k p c = iter R (rank c) q c
theorem fixed_point_by_rank (hloc : Local dep R)
    (hr : ∀ c d, dep c d → rank d < rank c) (hpos : ∀ c, 1 ≤ rank c)
    (hmax : ∀ c, rank c ≤ Rmax) (p : C → A) :
    R (iter R Rmax p) = iter R Rmax p ∧ ∀ q, R q = q → q = iter R Rmax p
```

(via `pred_rank_isHeight`: with all ranks ≥ 1, rank − 1 is a height function.)

42 tile counts, proved. `Counts := Fin 4 → Nat`; `rk n = (n0-1)+(n1-1)+(n2-1)+(n3-1)`
with truncated subtraction = note 2's R = Σ max(n_j − 1, 0).

```lean
theorem rk_opening : rk (fun _ => 7) = 24
theorem rk_play_le (n : Counts) (j : Fin 4) : rk (play n j) ≤ rk n
theorem rk_play_genuine (n : Counts) (j : Fin 4) (hj : 2 ≤ n j) : rk (play n j) + 1 = rk n
theorem rk_later_lt (n : Counts) (j : Fin 4) (hj : 2 ≤ n j) (js : List (Fin 4)) :
    rk (playAll (play n j) js) < rk n
theorem settled_42 {dep : C → C → Prop} {R : (C → A) → (C → A)}
    (hloc : Local dep R) (pos : C → Counts) (seat : C → Fin 4)
    (hgen : ∀ c, 2 ≤ pos c (seat c))
    (hlater : ∀ c d, dep c d → ∃ js, pos d = playAll (play (pos c) (seat c)) js)
    (p q : C → A) (c : C) (hc : pos c = fun _ => 7) (k : Nat) (hk : 24 ≤ k) :
    iter R k p c = iter R 24 q c
```

Reading: every play weakly lowers R; a genuine play lowers it by exactly one; every
position after a genuine decision has strictly smaller R. At the opening lead the
response is settled by the 24th application (= P^23 in note 2's indexing), from any
starting policy.

Two hypotheses carry the 42 content and are ASSUMED, not checked against
scenario/planner.py:

- `hgen`: every coordinate is a genuine decision (forced coordinates are dropped, as
  note 2 says).
- `hlater`: the response at c reads the lower rung only at decisions after c's own
  play. True for Walt under fixed beliefs (the plan from an information set contains
  only later decisions). No dominoes, suits, or legality are modeled.

## L3: finite exact best response and the coherence bound (note 1, sections 2-3) — Stab/Coherence.lean

Argmax, proved. `best v a l` scans the nonempty list `a :: l` and replaces the champion
only on a strict improvement, so ties keep the earlier element (the lowest-tile-id rule
when the list is sorted by id).

```lean
theorem best_mem (v : α → Int) : ∀ (a : α) (l : List α), best v a l ∈ a :: l
theorem best_max (v : α → Int) :
    ∀ (a : α) (l : List α) (x : α), x ∈ a :: l → v x ≤ v (best v a l)
theorem exists_argmax (v : α → Int) (a : α) (l : List α) :
    ∃ m, m ∈ a :: l ∧ ∀ x, x ∈ a :: l → v x ≤ v m
```

Coherence bound, proved in exact integers. A belief is an integer weight vector over a
list of worlds (probabilities over a common denominator D). Per-world values lie in
[0, M] (M = 1 is the make indicator). `V ws μ u = Σ_w μ w · u w`;
`L1 ws μ μ' = Σ_w |μ w − μ' w|`, so TV = L1 / (2D).

```lean
theorem value_shift (ws : List W) (μ μ' u : W → Int) (M : Int)
    (hmass : S ws μ = S ws μ') (hu : ∀ w, 0 ≤ u w ∧ u w ≤ M) :
    2 * iabs (V ws μ u - V ws μ' u) ≤ M * L1 ws μ μ'

theorem regret_le (ws : List W) (μ μ' : W → Int) (M : Int) (val : P → W → Int)
    (hmass : S ws μ = S ws μ') (hval : ∀ π w, 0 ≤ val π w ∧ val π w ≤ M)
    (a : P) (l : List P) (πstar : P) (hstar : πstar ∈ a :: l) :
    V ws μ (val πstar) - V ws μ (val (best (fun π => V ws μ' (val π)) a l)) ≤
      M * L1 ws μ μ'

theorem certificate (ws : List W) (μ μ' : W → Int) (M : Int) (ua ub : W → Int)
    (hmass : S ws μ = S ws μ') (ha : ∀ w, 0 ≤ ua w ∧ ua w ≤ M)
    (hb : ∀ w, 0 ≤ ub w ∧ ub w ≤ M)
    (hgap : V ws μ' ua - V ws μ' ub > M * L1 ws μ μ') :
    V ws μ ua > V ws μ ub
```

Divided through by D (by 2D for value_shift):

- |V_μ(π) − V_μ'(π)| ≤ M·TV for every policy.
- The policy chosen by argmax under the wrong belief μ' loses at most 2M·TV under the
  true belief μ. Note 1's 2δ, with no horizon factor.
- If the margin under μ' exceeds 2M·TV, the order is the same under μ. Note 1's action
  certificate Δ > 2δ.

One extra came out of the proof: the weights never need to be nonnegative. The bound
holds for any two signed weight vectors with equal totals.

Helper lemmas (all proved): `S_sub`, `S_sub_mul` (linearity of the list sum), `term_up`
(x(2v − M) ≤ |x|·M for 0 ≤ v ≤ M), `sum_bound`. The trick:
Σ(μ−μ')v = ½Σ(μ−μ')(2v − M) because the masses agree.

## L4: without locality the ladder need not settle (note 1, section 4) — Stab/Cycle.lean

Hidden matching pennies. Two coordinates (false = "0", true = "1"), actions Bool.
Coordinate 0 wants to match coordinate 1; coordinate 1 wants to mismatch coordinate 0.
Each reads the other's previous value.

```lean
def pennies (p : Bool → Bool) : Bool → Bool :=
  fun c => if c then !p false else p true

theorem pennies_no_fixed_point (q : Bool → Bool) : pennies q ≠ q
theorem pennies_never_settles (p : Bool → Bool) (k : Nat) :
    pennies (iter pennies k p) ≠ iter pennies k p
theorem pennies_moves (p : Bool → Bool) (k : Nat) :
    iter pennies (k + 1) p ≠ iter pennies k p
theorem pennies_period_four (p : Bool → Bool) (k : Nat) :
    iter pennies (k + 4) p = iter pennies k p
theorem pennies_local_needs_cycle (dep : Bool → Bool → Prop) (h : Local dep pennies) :
    dep false true ∧ dep true false
theorem pennies_no_height :
    ¬ ∃ (dep : Bool → Bool → Prop) (h : Bool → Nat), Local dep pennies ∧ IsHeight dep h
```

Reading: pennies has no fixed point at all, so no iterate from any start is one,
consecutive iterates always differ, and the ladder cycles with period 4
((a,b) → (b,¬a) → (¬a,¬b) → (¬b,a) → (a,b)). Any dep making pennies local must contain
both edges 0→1 and 1→0, so no height function exists and L1's hypotheses cannot be
met. This is the formal witness that a response whose beliefs are recomputed from the
rung below (so it reads earlier coordinates through the likelihoods) can oscillate
forever.

## L5: early-stopping certificate (note 2, section 2.4) — Stab/Cycle.lean

Values are integers (rationals over a common denominator).

```lean
theorem early_stop_strict (acts : List α) (Qk Qs ε : α → Int) (a : α)
    (hclose : ∀ b, Qk b - ε b ≤ Qs b ∧ Qs b ≤ Qk b + ε b)
    (hgap : ∀ b, b ∈ acts → b ≠ a → Qk a - ε a > Qk b + ε b) :
    ∀ b, b ∈ acts → b ≠ a → Qs b < Qs a

theorem early_stop_argmax (a0 : α) (l : List α) (Qk Qs ε : α → Int) (a : α)
    (ha : a ∈ a0 :: l)
    (hclose : ∀ b, Qk b - ε b ≤ Qs b ∧ Qs b ≤ Qk b + ε b)
    (hgap : ∀ b, b ∈ a0 :: l → b ≠ a → Qk a - ε a > Qk b + ε b) :
    best Qs a0 l = a ∧ ∀ m, m ∈ a0 :: l → (∀ x, x ∈ a0 :: l → Qs x ≤ Qs m) → m = a
```

Reading: a is the unique maximizer of Q⋆, and the deterministic scan under Q⋆ returns
it. Getting the ε_a is the hard part, and it is not formalized: note 2 says the
encounter-probability bound has to range over all own continuations starting with a.

## Build and axioms

```
cd /Users/jason/code/mk5-main/scratch/research-night/lean
rm -rf .lake/build && lake build      # 1.1 s wall, 6 modules, all built
lake env lean Axioms.lean             # #print axioms for 13 headline theorems
```

| Toolchain | Mathlib | Clean build | sorry/admit |
|---|---|---|---|
| leanprover/lean4:v4.33.0-rc1 | no | 1.1 s | 0 |

The axiom check prints only the standard three (propext, Classical.choice,
Quot.sound) and never sorryAx. Files: Stab/Ladder.lean (L1), Stab/Rank.lean (L2),
Stab/Coherence.lean (L3), Stab/Sharp.lean (tightness), Stab/Cycle.lean (L4, L5),
Axioms.lean. Nothing is admitted.

## What this says about the real Zeb

Take any response operator with fixed beliefs, exact evaluation and a deterministic tie
rule that reads the rung below only at strictly later genuine decisions. Its ladder
settles at the 42 opening lead by the 24th application (P^23). The settled ladder does
not depend on the starting policy, and it is the operator's unique fixed point. In an
abstract operator the bound h+1 can be reached, so agreement between adjacent rungs
proves nothing. On a finite policy list the deterministic argmax returns a true
maximizer. A belief off by TV δ costs at most 2δ in pmake under the true belief, and a
margin above 2δ certifies the move. A provisional value within ε of the eventual one,
with a gap above the ε's, certifies the eventual unique argmax.

## What it does not say

- **Sampling.** Finite N is not modeled. value_shift applies to a sample only with a TV
  bound already in hand, and it does not account for the selection made after looking.
- **The floor.** The size-16 floor swaps in the net partway down the tree, which
  changes R. Whether the floored operator is still local, and for which dep, is not
  modeled.
- **The learned field.** The net is trained on Zeb's own pour, so it moves with the
  operator. These theorems hold the operator fixed.
- **Recomputed beliefs.** L1/L2 hold only for fixed beliefs. With β^F recomputed from
  the rung below, the response at c also reads earlier decisions through the likelihood
  of the plays already seen. Then hlater fails, dependencies can point backward, and L4
  shows the iteration can cycle forever with no fixed point. This is the formal form of
  note 1 section 4. L4 only shows that settling is not guaranteed. It does not show
  that 42 with β^F cycles.
- **O2 and O3.** Everything is single-seat O1 against a fixed field. Partnership and
  team optimality are untouched.
- **The 42 model.** It is tile counts only. hgen and hlater are assumptions about Walt's
  operator and are not checked against scenario/planner.py.
- **"Optimal move in finite time."** For the fixed-belief ladder, finiteness is proved
  at 24 applications. O1 optimality of uniform-over-fiber is not proved. regret_le only
  bounds the loss from belief error, and the C1 counterexamples say that error is not
  zero.

LEAN DONE.

## Appendix: the hypotheses against scenario/planner.py, the floor, the tie rule, and one catch

Checked by the Lean role at ~01:40 CDT. Line references are `scenario/planner.py`.
New file `Stab/Floor.lean` (2 theorems); clean build of 7 modules 0.99 s, no sorry,
standard axioms only.

**(a) hlater holds when the tilt is none or dice, which includes shipped Zeb.**
Coordinate = (rung, seat, remaining hand, record). The cache key at line 320 is
(k, seat, own, rec.key()); `rec.key()` includes forbid (record.py:42). The sample seed
at line 304 is `_mix(config.seed, k, viewer, own, tape_key(rec), *forbid)`, a function
of the coordinate alone; bid and bidder_team are fixed for the hand. So `_modeled_move`
is a pure function of the coordinate. Q(a) starts at `root.advance(t)` (371-377).
Every read of the lower rung, via `_partition`→`_field_move` (421-431, 486-491) or
`_handed_down` (475-484), happens at a record reached after the viewer's root play.
Frontier playouts (502+) never read the profile. With `tilt="field"`, the live weights
read `_modeled_move(0, ...)` at PAST records (`tilt_weights`, 285-290), which breaks
hlater; a `belief=` callable can read anything. These are exactly L4's setting.

**(b) hgen holds.** Forced coordinates return their only tile before consulting any
rung: 323-325 in `_modeled_move`, 264-265 at the root, 495-496 in dice. So forced
coordinates are constant and never read p. Dropping them from dep is exact, and every
remaining coordinate holds 2 or more tiles.

**(c) The floor is clean.** `_handed_down` (483) = `_modeled_move(k-1, viewer, hand,
rec)`: the lower rung's value at the viewer's OWN later coordinate. Its "fresh sample"
is that coordinate's own seed (304): the lower rung's sample, not something the live
seat draws, and nothing earlier enters it. The root is never floored (`_floored` is
only checked inside `_value` and `_trace`, 396-398, which start at `root.advance(t)`).
The small catch: dep must include the viewer's own later coordinates, not only the
other seats'. Those are still later, so the rank still drops and L2's bound of 24 is
unchanged. Proved:

```lean
theorem floored_local (hloc : Local dep R) (floored : C → Prop) [DecidablePred floored]
    (f : C → C) (hf : ∀ c, floored c → dep c (f c)) :
    Local dep (fun p c => if floored c then p (f c) else R p c)
```

Whether a cell is floored depends on the cell size, which depends on modeled moves at
later coordinates. `floored_local` takes the floored predicate as fixed, so it does not
cover this directly; the operator is still a function of p at later coordinates, so
Local holds with the same dep. That last step is argued, not formalized.

**(d) The tie rule holds.** `_best` (336-341) sorts by tile id and returns the first
extremum, the lowest tile id: a deterministic function of the q dictionary, so the
operator is a function and not a relation. Equal floats are compared exactly, and the
Rust core is bit-identical. Lean's `best` keeps the earlier element on a sorted list,
the same rule.

**THE CATCH: rung-k operators are seeded by k.** L1 iterates ONE operator, but Walt's
rung k is R_k, with k mixed into the seed at line 304. The samples, and so the
decisions, differ by rung even when the Rung configs are identical. Proved:

```lean
theorem rung_seeded_no_settle (p : Unit → Bool) (k : Nat) :
    iterS flip (k + 2) p () ≠ iterS flip (k + 1) p ()
```

Here every `flip k` is Local for the EMPTY dep, so the height is 0 everywhere, yet the
iterates alternate forever. Consequences:

- L1 and L2 apply to exact Walt (the fiber fully enumerated; fiber.py does this when
  the fiber is no larger than the sample) and to a rung-independent seed.
- They do not apply to the sampled ladder as configured: the rungs differ by sampling
  noise at every height.
- Deployed Zeb is k0 over the net: one application of R_0 to the field. The ladder
  question is moot for it. What it inherits is locality (proved above, floor included)
  and the regret bound of L3.

**Closed 2026-10-12 (#112):** both engines now seed an invocation by its coordinate alone
(`_mix(config.seed, viewer, own, tape_key(rec), *forbid)`), so every rung with the same
Rung config is the same operator and L1/L2 describe the sampled ladder.

# The minimal comparison algebra (night 2026-10-21)

Jason's note *A minimal comparison algebra for a declared 42 family* (2026-10-21), made
the project's own. Same toolchain, no Mathlib; `lake build` 4.5 s clean; the 24 new
headline theorems in `Axioms.lean` print only the standard three; no sorry. Wiki:
`wiki/experiments/comparison-algebra-2026-10-21.md`.

- **`Stab/Hexagon.lean`, the local algebra.** With `A, B, C` the indicator vectors the
  six viewer orders of 3-1, 4-1, 5-1 collapse to, `x = E[A − B]`, `y = E[C − B]`, `β = E[B]`:
  `Q_formulas` (Q31 = β + max(x, 0), Q41 = β + max(y, 0), Q51 = β + max(x, y)),
  `next_comparison` (the second decisions compare x, y, x − y), `mixing_no_gain`
  (randomizing the six programs never beats the best tile), `no_scalar_statistic`
  (no linear scalar carries both x and y), `hexagon` (a six-type belief realizes (x, y)
  iff |x|, |y|, |x − y| ≤ 1; `cone_of_belief` for any atoms), `thirteen_patterns`
  (exactly 13 sign patterns of (x, y, x − y), all on the hexagon), `seq_classes`
  (the sequential output is the pattern), `four_root_orders`, and the note's segment
  (`segment_line` 19x − 23y = 4; `segment_classes`, five classes split at p = 1/5).
- **`Stab/Residual.lean`, the residual congruence.** `Res` (equal outputs after every
  block word) is an equivalence and a right congruence (`res_iff`), the coarsest one
  that keeps outputs (`res_coarsest`); an exact realization never merges inequivalent
  states (`realization_card`), and the quotient machine attains it (`quotient_realizes`,
  `quotient_states`). Refinement round k is agreement on words of length ≤ k, and a stable
  round is `Res` (`refine_stable`). With the HALT/REJECT totalization and at most n own
  tiles, refinement is exact after n − 1 blocks (`ladder_42`).
- **`Stab/Mixing.lean`, the mixing boundary.** For the threshold decision
  sgn((5p − 1)/(1 + 18p)): `mass_injective` (the observation's mass (1 + 18p)/560 keeps p),
  `mixing_forces_equality` (a relation that keeps the decision and every convex mixture
  is equality on [0, 1]; witness λ = 1/5, q = (1 − m)/4), `note_example` (0.08, 0.12
  mixed with 0.30 straddle 1/5).
- **`Stab/Plans.lean`, fixed plans in a one-parameter family** (2026-10-22). A fixed plan's
  value is a line in p and the optimum is their upper envelope: `optimal_between` (the
  parameters where a plan is optimal form an interval) and `optimal_on_piece` (where one
  plan is optimal across [u, w], any plan optimal at an interior point is too). So the
  fewest fixed plans optimal at every p is the number of affine pieces of the optimum.
  The note's ending has five (`silas/tests/controller.rs`), against 31 classes for the
  complete maximizing sets.

## Two findings against the night's brief

1. **13 is not the count of weak orders of (Q31, Q41, Q51).** That triple takes exactly 4
   (`four_root_orders`: Q51 is never the unique maximum). 13 is the count of weak orders of
   (E[A], E[B], E[C]), the sign patterns of (x, y, x − y), which are the sequential classes
   (`seq_classes`). The note says so; the brief's paraphrase did not.
2. **"The last tile is forced, so refinement stabilizes after one block" is false.** The bound
   is n − 1 blocks (`ladder_42`), and `Witness.one_block_not_enough` is a three-tile course
   that needs two. The Gus course's shape ladder 11, 462, 477, 477 uses both.

## What it does not prove

That the fewest-plans count equals the number of pieces is the two lemmas plus a two-line
argument in the module docstring, not a theorem with a `pieces` definition. The fixture
facts are computed, not proved: Silas rebuilds the 560-deal ending on its own rules and
reproduces the 767-state controller and its 1, 18, 31, 31 ladder, derives the segment's
gap (5p − 1)/(1 + 18p), and finds 0.08 and 0.12 in one cell (`silas/tests/controller.rs`).
Nothing here is about a learned net.
