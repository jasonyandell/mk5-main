# Texas 42: Play-Phase Specification Companion

This document is a **rules/spec companion** for the **play phase only** (tricks), intended to be unambiguous and directly checkable against implementations.

It is deliberately minimal and delegates algebraic details to:

- `docs/theory/SUIT_ALGEBRA.md` (led suit, following, trump power, trick ordering key)
- `docs/theory/PLAY_PHASE_ALGEBRA.md` (play-phase state model, signed-reward decomposition)

---

## §1. Scope

Fix a single hand’s play phase:

- 4 seats, 2 teams (0/2 vs 1/3)
- a fixed deal (assignment of 28 dominoes into 4 hands of 7)
- a fixed declaration $\delta$

This spec covers:

- legal moves (follow-if-possible)
- trick winner determination
- trick scoring (42 points total per hand)
- deterministic state progression through 7 tricks

This spec does **not** cover bidding, contract success/failure, marks, or any early termination rules.

---

## §2. Required Primitives

The play phase depends on the following primitives (defined in `docs/theory/SUIT_ALGEBRA.md`):

1. **Led suit**
   - $\ell = \mathsf{ledSuit}(d,\delta) \in \{0..7\}$ for a lead domino $d$
2. **Follow predicate**
   - $\mathsf{follows}(d,\ell,\delta) \in \{\mathsf{true},\mathsf{false}\}$
3. **Trick ordering key**
   - $\tau(d,\ell,\delta) \in \mathbb{N}$ where larger wins

The following property is required:

- **Unique winner**: for any legal trick, $\arg\max$ of $\tau$ over the 4 played dominoes is unique.

---

## §3. Legal Moves (Follow-If-Possible)

Given a state with current leader $L$ and trick prefix length $k$:

- If $k=0$ (leading), any domino in the current player’s hand is legal.
- If $k>0$, let $\ell$ be the led suit induced by the leader’s first domino $d_0$.
  - If the current player has at least one remaining domino $d$ with $\mathsf{follows}(d,\ell,\delta)$, they must play such a domino.
  - Otherwise, they may play any remaining domino.

This is the only legality constraint in the play phase.

---

## §4. Trick Resolution

When the 4th domino of a trick is played:

1. Compute led suit $\ell$ from the leader’s first domino.
2. Compute the trick winner seat:
   - $W = \arg\max\limits_{j\in\{0,1,2,3\}} \tau(d_j,\ell,\delta)$, where $d_j$ are the 4 played dominoes in seat order starting at leader.
3. The next trick leader becomes $W$.
4. The current trick resets (empty prefix).

---

## §5. Scoring

Define trick points:

$$\mathsf{pts}(d_0,d_1,d_2,d_3)=1+\sum_{j=0}^3 c(d_j)$$

where $c(d)\in\{0,5,10\}$ is the counting-domino value.

Required invariants:

- Across 7 tricks: $\sum \mathsf{pts} = 42$.
- Team points satisfy: $\text{Team0Points} + \text{Team1Points} = 42$.

For minimax analysis, `docs/theory/PLAY_PHASE_ALGEBRA.md` defines a signed reward emitted only on trick completion. That decomposition is admissible for play-phase evaluation because mid-trick moves have reward 0 and terminal value is 0.

---

## §6. Determinism and Checkable Properties

For any fixed (deal, $\delta$), the play phase is a finite deterministic game with perfect information:

- No randomness during play.
- Legal move set is a deterministic function of state.
- Trick resolution is deterministic given the 4 played dominoes.

Suggested check suite (implementation-agnostic):

- **Follow compliance**: no legal move allows sloughing when a follower exists.
- **Winner uniqueness**: each legal completed trick has exactly one winner.
- **Score conservation**: total points across 7 tricks is 42.
- **State progression**: exactly 28 plays, grouped into 7 tricks.

---

## §7. Code Correspondence (solver2)

The `scripts/solver2/` implementation realizes this spec as:

- Primitives: `scripts/solver2/tables.py` (`led_suit_for_lead_domino`, `can_follow`, `trick_rank`, `resolve_trick`)
- Legal-move enforcement: `scripts/solver2/context.py` (`SeedContext.LOCAL_FOLLOW`) + `scripts/solver2/expand.py`
- Trick completion + rewards: `scripts/solver2/context.py` (`TRICK_WINNER`, `TRICK_POINTS`, `TRICK_REWARD`)
- Minimax target: `scripts/solver2/solve.py` (signed rewards, terminal value 0)

