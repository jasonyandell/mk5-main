---
title: BookStrategyPlayer — multi-step strategy framework with recording
kind: entity
first_seen: local-2026-05-03
last_updated: local-2026-05-03
status: design (build pending)
---

## What it is

BookStrategyPlayer is a player architecture that plays via a library of named multi-step
**strategies** rather than a single per-decision utility. Each strategy is a small object
with an explicit lifecycle (recognition → commitment → execution → completion). Strategies
that don't apply at a given decision fall back to a baseline player ([[w42-lens-v1-utility-head-to-head|Lens]]
with the EV utility, the established strongest fixed-utility baseline).

The framework was motivated by [[w42-book-claim-synthesis-and-ai-directions]]'s "single-
decision blind spot" insight: most book claims are multi-step plans, but most W42 probes
test single-decision contrasts. A planner that can express plans is the natural next
abstraction level for testing the book.

## The Strategy protocol

Each strategy implements four lifecycle methods, all pure functions of `(game_state, plan_state)`:

```python
class Strategy(Protocol):
    name: str
    def applies_to(self, gs, ps) -> bool: ...
    def priority(self, gs, ps) -> float: ...
    def commit_recognition(self, gs, ps) -> dict: ...   # called once per plan instance
    def next_action(self, gs, ps) -> tuple[Action | None, dict]: ...
    def plan_done(self, gs, ps) -> bool: ...
    # Optional:
    def observe(self, opp_action, opp_seat, gs, ps) -> dict: ...   # opponent modeling hook
```

By convention: following through on an active plan returns priority ≥ 10.0; recognizing a
new opportunity returns 1.0–5.0. This monotonicity prevents plan-flipping mid-execution.

`PlanState` carries:

- `active_plans: dict[str, dict]` — namespaced per-strategy private state
- `facts: dict[str, Any]` — publishable cross-strategy facts (e.g., `void_suits`)

Strategies do not call each other's methods. They contribute facts (publishable) and
actions (arbitrated). Composition is the framework arbitrating among declared facts and
proposed actions, not strategies negotiating with each other.

## The player loop

Five phases per decision:

1. **Retire** completed/dead plans (each strategy's `plan_done` is consulted)
2. **Recognize** new applicable strategies; call `commit_recognition` for each fresh one
3. **Arbitrate** — highest-priority applicable strategy wins; fallback to Lens(ev) if none
4. **Apply state delta** — `next_action` returns `(action, plan_state_delta)`; the player applies the delta
5. **Bail or play** — `action=None` means strategy bailed at the last second; use fallback

`BookStrategyPlayer(strategies=[], fallback=Lens(ev))` is **identical** to Lens(ev). The
framework is a strict superset of the baseline. Any margin in head-to-head measurement
is exactly the strategy library's contribution.

## Five composition modes

The protocol's purity makes meta-strategies compose naturally:

1. **State-conditioned strategy selection** — `priority(gs, ps)` already takes game state.
   `TrumpPulling.priority` returns `5.0 - gs.trick_idx` (high early, fades); `ThrowawayLadder`
   does the inverse. The arbitration loop selects the right strategy per state. Free out of
   the box. Connects to [[w42|t42-nwuu]] (state-conditioned utility) — same idea, applied to
   strategies instead of utilities.

2. **Strategy chaining** — one strategy's plan publishes a `fact`; another strategy reads it
   via `applies_to`. Example: `SingletonLeadToVoid` publishes `facts.void_suits`;
   `CountSteeringIntoVoid` reads it. Requires formalizing publish/subscribe in `PlanState`
   (~30 LOC framework addition).

3. **Hierarchical strategies** — a Strategy's `next_action` can delegate to a sub-player
   with its own library. Recursive composition with no protocol changes. Maps directly to the
   book's chapter structure — Ch 3 (bidder play), Ch 5 (setter defense), Ch 8 (84-bid endgame)
   are each natural sub-libraries.

4. **Opponent-aware (counter-strategies)** — add an `observe(opp_action, ...)` hook to the
   protocol. Strategies accumulate opponent statistics in shared facts. Counter-strategies
   trigger when opponent patterns match a precondition. (~50 LOC framework addition.)

5. **Cross-hand match memory** — current architecture is per-hand. Lifting state to
   match-level (e.g. "watch opponent for 5 hands then switch to counter") requires extending
   the player with a `match_state` arg passed through lifecycle methods. Bigger lift; the
   parallel-hand simulator currently treats hands as independent.

## The book IS a meta-strategy

The deepest implication of the architecture: **the book's chapter structure is exactly mode
1 + mode 3.** Chapters partition state space (Ch 3 = bidder offense, Ch 5 = setter defense,
etc.) and within each chapter live multiple specific tactics as sub-strategies. A faithful
book-encoded player is:

```python
TheBook = BookStrategyPlayer(
    strategies=[
        Chapter3_BidderPlay(),       # composite of trump-pulling, etc.
        Chapter4_PartnerSupport(),
        Chapter5_SetterDefense(),    # composite of void-creation, low-trump-trap, etc.
        Chapter8_AllTricks84(),      # composite of throwaway-ladder, etc.
        Chapter12_HighBidPlay(),     # composite of pounce-window, etc.
    ],
    fallback=Lens("ev"),
)
```

`Lens(ev) vs TheBook` is then a single head-to-head match measuring how much the *entire
book* is worth in points-per-hand — the question the [[w42-book-validation-campaign]] has
been trying to answer all along, just at the wrong abstraction level until now.

## Recording → training pipeline

Per-decision records get logged structurally:

```python
@dataclass
class DecisionRecord:
    hand_id, trick_idx, decision_idx
    game_state_tensor                   # serialized GameStateTensor
    plan_state_snapshot                 # active_plans + facts at decision time
    legal_actions
    applicable_strategy_names           # which fired
    strategy_priorities                 # all returned scores, not just winner
    chosen_strategy                     # None = fallback
    chosen_action
    fallback_action                     # what Lens(ev) WOULD have chosen — free counterfactual
    immediate_trick_outcome             # filled at trick-end
    hand_point_margin                   # filled at hand-end
    plan_completion_status              # per-strategy completed/disrupted/abandoned
```

The free counterfactual (`fallback_action`) is the key. At every decision we record both
what the strategy player did *and* what Lens(ev) would have done. The difference is the
strategy's contribution exactly. Re-running the hand with the counterfactual action gives
ground-truth per-decision causal effect (paired-seed style); off-policy estimation gives a
cheaper approximation once a value model is trained.

Three trainable models, in increasing ambition:

- **Model A — strategy selector.** Replaces hand-crafted `priority(gs, ps)` with a learned
  function. Input: encoded game_state + plan_state + applicability mask. Output: per-
  strategy logits. Loss: regression on per-strategy expected hand_point_margin (Q-learning
  over macro-actions). The framework is unchanged; only the arbitration step gets a learned
  head.

- **Model B — plan-success predictor.** P(plan completes successfully) and E[points if it
  completes] per (state, strategy). Useful as input to Model A or as an early-bail signal.

- **Model C — end-to-end policy distillation.** Predict actions directly, bypassing the
  library at inference. The strategies become *training scaffolding*. Discovery side-effect:
  if the trained policy consistently picks an action where no encoded strategy applied,
  that region is a candidate for a new strategy nobody wrote down. **Strategy discovery
  bootstraps from gameplay data.**

## Where the parked models suddenly have roles

This pipeline naturally re-activates the three parked-or-experimental models:

- **[[gus]]** (belief-state encoder) → input encoder for Model A. Gus turns partial-
  observability into a clean belief tensor; the strategy selector consumes it. This is what
  Gus was originally designed for — feeding decision-time models. The decision-time model
  isn't [[burl]], it's the strategy selector.

- **[[burl]]** (slow experimental LLM) → candidate Model A. Strategies have human-readable
  names; an LLM is unusually well-suited to "given this game situation, which named
  strategy should fire?" because it can reason about strategy rationales in natural
  language. The structured action space (15-30 named strategies, not raw 7-domino choice)
  is what makes Burl tractable — small action space, narrative reasoning, STaR-style chain-
  of-thought to justify the pick. **This is the use case Burl was built for and we just
  hadn't found it yet.**

- **[[zeb]]** (parked AlphaZero variant) → either training-data generator or a backup Model
  C target. Zeb's policy head trained on strategy-labeled data instead of raw action labels
  has a much smaller, structured output space.

## Why this is the cleanest path past EV the campaign has identified

[[w42-lens-v1-utility-head-to-head]] established that EV is the ceiling for fixed
*pointwise* utilities at one-step lookahead. The other paths to "beat EV":

| path | status |
|---|---|
| Multi-utility heads | **dead** — Lens v1 killed it |
| Soft-cliff utilities (disaster, etc.) | **dead** — disaster confirmed EV is the ceiling for fixed pointwise tweaks |
| Generic MCTS (rung-2) | expensive build; doesn't directly use book wisdom |
| Lookahead-Lens K=2 | impractical — branching factor ~3-4 over 8-12 plies (4 players × K=2-3) |
| **Strategy-selector trained from book-strategy gameplay** | **uses book wisdom as structured action space; trains on cheap self-play with free counterfactuals; deployed model has planning capability without paying MCTS branching cost** |

The strategy-selector path was not on the original list because it required the framework
to exist first. Now that we're scoping the framework, training the selector is the natural
next layer — and the training data accumulates as a pure side effect of running matchups.

## Phased build plan

**Phase 1 (now, ~3-4h build):**
- Strategy framework (~400 LOC)
- 1-3 starter strategies (~150 LOC each); recommended first: `singleton_lead_to_void` —
  rehabilitates [[w42-bookval-v1-wave2-void-creation]] (which was contradicted at single-
  decision granularity)
- DecisionRecord serialization to parquet (~50 LOC; logged but not yet trained on)
- Head-to-head measurement vs Lens(ev) (~50 LOC, reuses `w42/lens_v1/parallel_match.py`)
- Publish/subscribe `facts` extension to PlanState (~30 LOC; mode 2 enabler)

Phase 1 deliverables are useful even if Phase 2/3 never happen: a measurement instrument
+ training pipeline as free side effect + the option to spend on Phase 2/3 only after
Phase 1 confirms strategies are worth measuring.

**Phase 2 (after ~5-10 strategies, ~50K labeled decisions):**
- Train Model A (learned strategy selector) on recorded data
- A/B test learned-selector vs hand-crafted-priority head-to-head
- Decide whether to invest in Burl-as-selector (LLM path) or smaller NN

**Phase 3 (research frontier, gated on Phase 2):**
- Strategy discovery via clustering of fallback-invoked decisions where the learned
  selector deviates from hand-crafted priorities
- End-to-end policy distillation (Model C) if Phase 2 lifts are real

## Composition limits worth knowing

Three things the framework can't do, no matter how clever the strategies:

1. **One domino per decision.** If two strategies both want different dominoes, priority
   arbitration picks one and the other's plan goes infeasible. The framework can detect
   and log the conflict but not resolve it — game constraint, not framework limit.
2. **No automatic plan negotiation.** Two active plans can rest on contradictory
   predictions of opponent behavior; the framework doesn't reason about plan compatibility.
   Strategy-author responsibility.
3. **Partnership coordination via assumed-mirror only.** A strategy that requires partner
   to play the same library can be modeled by having the partner-seat run a copy of the
   same player. Asymmetric self-play; the simulator already supports this.

## Status

- **Designed:** 2026-05-03 in conversation with the orchestrator. Architecture, lifecycle
  protocol, composition modes, recording format, and phased build plan all specified.
- **Bead pending:** Phase 1 build (framework + first strategy + measurement harness +
  recording) to be filed.
- **Built:** not yet.
- **Strategies encoded:** none yet.

## Links

- [[w42-book-claim-synthesis-and-ai-directions]] — the methodology insight that motivated
  this design
- [[w42-book-validation-campaign]] — the campaign this serves
- [[w42-lens-v1-utility-head-to-head]] — the EV-as-ceiling result that constrained the
  design space
- [[w42-bookval-v3-utility-argmax-divergence]] — Wave 4.0 measurement that gated the
  whole architecture-decision branch
- [[gus]] — input encoder candidate
- [[burl]] — Model A (strategy selector) candidate; the use case Burl was built for
- [[zeb]] — training-data generator / Model C target
- [[forge]] — the simulator the player runs against
