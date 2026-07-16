---
title: The Belief/Policy/Value Algebra
kind: topic
first_seen: 2026-07-16
last_updated: 2026-07-16
status: active
---

## What this is

The formalization of the belief/policy/value recursion Jason named in the
2026-07-15→16 conversation ("your belief is a function of your policy; your
policy is based on what you believe; value is similar") — the same thread that
ratified the argument's-referees doctrine on [[count-fate-ledger]]. The algebra
is definitions plus Bayes; nothing here is a build commitment. Conclusions are
tiered by epistemic status below; the measurement program is
[issue #64](https://github.com/jasonyandell/mk5-main/issues/64). Per the
emergent-values intent ([[otis]], binding): this page's vocabulary (channel,
convention, signal) is analysis vocabulary, never a named feature.

## Objects

- Deal ω; public history h; a seat's information state s = (own hand x, h).
- Policy per seat: π(a|s) = 𝟙[a legal] · σ̃(a|s) — indicator physics ×
  discretionary choice. The field π = all four seats' policies.
- Return G: points this hand, marks through the race.

## The three relationships

**Belief is Bayes through the field.**

```
b(ω | s) ∝ p(ω | x) · Π_t σ_{j(t)}( a_t | s_{j(t)}(ω, h_<t) )      i.e.  b = B(π)
```

The likelihood of a world is the product, over observed actions, of how
probable the actor's policy makes that action in the info state ω would have
handed them. The indicator factors (follow-suit, voids) are policy-invariant
physics; all of the recursion enters through σ̃.

**Value is derived, not fundamental.** Given (π, b):

```
V^π(s) = E_{ω~b(·|s)} [ W^π(ω, h) ],    W^π = realized return, all seats play π
```

and likewise Q^π(s,a). The recursion is the **pair** (π, b); V is a cache of
an expectation they determine. The V-net exists because the cache is
expensive, not because value is a third degree of freedom.

**Policy closes the loop:** π′ = improve(Q^π, b). A fixed point — b* = B(π*),
π* rational against its own continuations under b* — is a
sequential-equilibrium-shaped object (Kreps–Wilson): beliefs consistent with
strategies, strategies rational given beliefs. [[jud]]'s "trained until
everything lined up" is this object's name.

## The tilt form

Let u(ω|h) ∝ p(ω|x)·𝟙[ω consistent with h] — the physics posterior, exactly
what [[expected-q-value|eq]] samples. Let g(ω) = Σ_t log σ̃(a_t | s_t(ω)) —
the discretionary log-likelihood of the observed history, world by world. Then

```
b(ω | h) ∝ u(ω | h) · e^{g(ω)}
```

**The true posterior is an exponential tilt of eq's uniform by g.**
Consistency-sampling is exactly the g ≡ 0 limit; everything meaning-shaped in
42 lives in g. D(b ‖ u) — in practice, the effective sample size of weights
e^g on u-samples — measures how much meaning a history carries beyond physics.
The [[gus]] belief head is an amortization of e^g. Measured once, at one
mid-game position, in [[jud]]'s 2026-06-14 demo sketch: ESS 128 → 10.5.

## The coupling theorem ("the tiger")

Perturb the field, log σ̃ → log σ̃ + ε·φ, and accumulate the direction along
the observed history: φ_h(ω) = Σ_t φ(a_t, s_t(ω)). Then

```
d/dε log b(ω) = φ_h(ω) − E_b[φ_h]                      (centered score)
D( b ‖ b_ε )  = (ε²/2) · Var_{ω~b}( φ_h ) + O(ε³)
```

The belief–policy coupling in direction φ is the posterior variance of the
change's accumulated log-likelihood. It vanishes iff **the change** is
world-nondiscriminating along h; forced actions contribute zero. The coupling
measures the discrimination of the *update direction*, not of the current
field.

## The information identity ("the treasure")

```
I(ω ; a_t | h_<t) = E_{ω~b} [ D( σ(·|s_t(ω)) ‖ σ̄ ) ],    σ̄ = E_b[σ]
Σ_t I(ω ; a_t | h_<t) = I(ω ; h)                          (chain rule)
```

Per-decision meaning is the belief-expected divergence of the actor's
world-conditional action distribution from its predictive mixture; the whole
hand's meaning is the sum. Tiger and treasure are **one functional** —
world-discrimination — evaluated at the update direction (tiger) vs the
current field (treasure). Corollary: the update directions that grow meaning
are exactly the belief-destabilizing ones, direction by direction.

## E[Q], located

```
Q_eq(s,a) = E_{ω~u(·|s)} [ Q°(ω, h, a) ]        Q° = all-four-clairvoyant value
```

eq is the honest object E_b[Q^π] with the field π deleted from both places it
appears: the belief (B(π) → u; drop the σ̃ product, keep the indicators) and
the continuation (π → Q°). The two deletions are not a bound in general — a
seat's own clairvoyance inflates its value, the opponents' deflates it;
measured net at the auction was inflation (the #26 over-bidder,
[[w42-champion-selfplay-fixed-point]]). This is [[strategy-fusion]]'s "eq is
not 42" in operator form.

## Exact decompositions

**The gap.**

```
V^π(s) − E_u[Q°]  =  (E_b − E_u)[Q°]  +  E_b[ W^π − Q° ]
```

Term 1: belief-reweighting of clairvoyant values — the [[champion-ladder]]
rung #24/#25 lever, real in accuracy, measured marks-neutral in play. Term 2:
honest-vs-clairvoyant continuation under belief — the insurance economy
(protection, information value, signaling).

**The retention slice** (first order):

```
Δ(keep guard g) ≈ P_b[threat held] · P[field's σ exploits it] · E[swing | exploited]
```

belief about cards × belief about policy × stakes.

**Claim-vs-cash** ([[count-fate-ledger]], The argument's referees):

```
unpaid claim = V̂ approximation error + field-model error + genuine disequilibrium
```

Three failure modes, identical at the table, different repairs (data; rollout
seats; the loop).

## The (ε, init) family

The book ([[w42]]) is a policy class Σ_book — hard rules plus judgment calls;
the gut feels are exactly where the book leaves σ̃ underdetermined, a
parameter vector θ. "Play the book better than a human can" = max_θ V(σ_θ,
B(field)): exact Bayes where the human has feel, calibrated thresholds where
the human has lore. Legibility is a trust region KL(σ ‖ σ_book-human) ≤ ε:
ε = 0 is a book engine (recognizable, "right," never weird); ε = ∞ with eq
init is the wall assault ([[the-wall]]); small ε is a book-anchored learner
with a dialed revelation rate. The strategic options differ by two knobs
(ε, initialization), not by machinery.

## Conclusions that CAN be drawn (mathematical, given the model)

1. **Tilt form**: b = u·e^g exactly; eq ≡ g = 0; "belief beyond physics" is
   one scalar function per world, amortizable by a net.
2. **Coupling theorem**: belief movement under a loop step is Var_b of the
   change's accumulated log-likelihood — zero for world-nondiscriminating
   updates, zero from forced actions. A loop that improves play without
   changing hand-conditional choice structure moves no beliefs.
3. **Information identity + chain rule**, and the tiger/treasure coincidence:
   meaning-growth and belief-churn are the same directions.
4. **V is (π,b)-determined**: convergence is diagnosed on (π, b) consistency;
   V-error is repairable by evaluation data alone.
5. **eq = π deleted twice**, and the deletions are not a bound in general.
6. **The exact gap decomposition** (reweighting + insurance) and the exact
   claim-vs-cash accounting.
7. **A zero-one-step-cost channel exists**: on value-tied action sets,
   informative tie-breaking has zero immediate EV cost by definition of tied
   (equilibrium cost can grow later, through opponents reading — that part is
   not free).

## Conclusions that MIGHT be drawn (conjectures, each with its deciding probe)

1. **Phase 1's retention tie is a docile-field result** — the middle factor
   P[field exploits] ≈ 0 — not evidence that protection is worthless.
   Consistent with [[otis-guard-premium]] (G3 real, G1 opposing); decided by
   **M6** (field docility).
2. **Accidental conventions already exist**: `lens:ev` breaks value-ties
   deterministically as a function of the hand, so the field writes
   hand-correlated signals on the free channel now, and the [[gus]] belief
   head may passively read them. Decided by **M4** (shuffle-tie-break).
3. **The remaining play-side edge is (almost) entirely term 2** — supported by
   the #24/#25 nulls, "almost entirely" unproven; sharpened by **M1**
   (meaning map).
4. **Conventions emerge first on the free channel** (the value-tied "meh"
   sloughs — cheap-talk-first, Lewis/Skyrms transfer to 42's common-interest
   partner channel). The same states carry no retention lesson
   ([[count-fate-ledger]] dispersion triage) AND maximal convention
   opportunity — different organs. Watched via **M3** (bandwidth) + **M5**
   over loop rounds.
5. **The docile field is a coordination failure, not a floor**: writer and
   reader each wait for the other; escape requires coordination — neutral
   drift + passive reading, parameter sharing across partner seats, or the
   book as a pre-agreed codebook. Corollary conjecture: **the book is an
   equilibrium-selection device**, not just knowledge — formal support for
   the "play the book better" path. Testable via book-seeded vs unseeded
   loops (unregistered).
6. **Two-timescale convergence** (fast belief clock, slow policy clock,
   damping scheduled against measured tiger) converges where simultaneous
   updates would orbit — the standard stochastic-approximation frame; its
   conditions are plausible here, unverified.
7. **The (ε, init) family** is a faithful map of the strategic options — a
   design frame, not a theorem.

## Measurement program

[Issue #64](https://github.com/jasonyandell/mk5-main/issues/64): M1 meaning
map · M2 tilt profile · M3 channel bandwidth · M4 accidental-convention
detector (highest surprise-per-flop; run first) · M5 realized tiger (the
loop's meaning-meter, partner to claim-vs-cash's honesty-meter) · M6 field
docility. All cheap on existing infrastructure; registered predictions before
any probe runs, per the #55 operating doctrine.

## Links

[[count-fate-ledger]] · [[strategy-fusion]] · [[expected-q-value]] · [[pimc]] ·
[[jud]] · [[otis]] · [[gus]] · [[the-wall]] · [[belief-conditioned-self-play]] ·
[[belief-bayes-ceiling]] · [[w42]] · [[champion-ladder]] ·
[[otis-guard-premium]] · [[w42-champion-selfplay-fixed-point]]
