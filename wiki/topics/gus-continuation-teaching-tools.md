---
title: "Gus continuation teaching: solver and generation tools"
kind: topic
first_seen: 2026-10-21
last_updated: 2026-10-21
status: active
---
<!-- SPDX-FileCopyrightText: Jason Yandell -->
<!-- SPDX-License-Identifier: CC0-1.0 -->

## The question

Can [[gus-tiny-gpt]] learn stronger contingent play from a broader course of
consequential decisions? [[gus-contingent-maskfit-2026-10-20]] establishes
near-complete acquisition of its known lessons; [[gus-contingent-maskbreadth-2026-10-20]]
provides a favorable wide-versus-narrow finite contrast, but its combined fresh
gate failed. Neither establishes that data alone suffices. The current teacher
course has 128 physical TRAIN roots; the large imitation pours are a different
source of supervision.

The research loop is stopped at Jason's request. The authorized capability
inventory does not restart training or select a new experiment. The next useful
bridge is earlier information-consistent teaching under the same G0/Zeb field.
The conversation-derived teaching direction is recorded separately as the
IDEATED [[gus-historical-choice-wish]].

## Solver shortlist

The Texas-42 paths below are under its repository at source
`9d6a5a2e12958183d49d0cd95c8617c4ef7a12e4`. Its Walt work is explicitly
**exploratory implementation evidence**, not promoted mathematical authority.

| Need | Existing implementation | Fit and boundary |
|---|---|---|
| Exact best-response success probability | `walt/walt/src/solver/factor_belief.rs::response_success_mass` | Integer success mass over integer belief mass; full legal focal MAX and observation SUM. Accepts a declared deterministic `SlicePolicy` field. |
| An executable contingent teacher | Same module, `extract_success_policy` | Returns optimal mass and a root-local `ExtractedPolicy`; fixed-policy evaluation can reprice its decisions. Default ties choose the lowest tile. |
| Complete-policy evaluation | Same module, `viewer_success_mass` and `viewer_score_profile` | Exact make mass or all 43 final-score masses for one fixed policy; the profile is not a collection of separately optimized thresholds. |
| Counted hidden support and updates | `FactorBelief`, `ExactCoverOracle`, `FiberOracle`, `SupportOracle` | Uniform legal root support and deterministic-field conditioning; one shared learner decision per information state. |
| Bounded earlier teaching | Texas-42 `solver/focal_horizon.rs`, `focal_ladder.rs`, `grammar.rs` | Deterministic work budgets and executable lower policies; unresolved bounds are not exact labels. A later option when full exact solving is too costly. |
| Independent alternative solvers | Texas-42 `rob/crates/player/src/solver.rs`, `strat/hidden_scalar.rs` | Real information-set solvers, but built-in greedy or uniform-legal fields and different utilities require changes for G0/Zeb. |
| Four-tile historical performance engineering | [[walt]], [[walt-spec]], [[hoyt]], [[perf]] | Lawful information-set best response; optimized wave/table machinery is archived, old fields and timings do not establish G0/Zeb throughput. |
| Guaranteed make/set without a field model | [[silas]], `silas/src/outcome.rs::forces` | Quantifies over every legal play of all other seats, partner included. A guarantee test, not expected success against G0/Zeb. |

`SupportOracle` supports the evolving conditioned factors; the narrower
`FiberOracle` is not interchangeable for all later histories. Factorized counting
need not be the fastest execution backend: the bundled complete-world evaluator
(`solver/bundle.rs`) is another reusable fixed-policy route. Measure native work
and field calls before choosing a backend.

A fresh targeted native regression,
`solver_extraction::an_extracted_policy_reprices_to_the_extraction_optimum`,
passes on the existing six trick-five/six fixtures under their declared native
field. It checks extraction against optimization and independent fixed-policy
repricing. It does not test four-tile G0/Zeb throughput or the unbuilt bridge.

The Rust counted-belief route is closer to the current rational teacher than
reviving a floating-point Jud-specific implementation. Its callable field
interface is `SlicePolicy::choose(decl, remaining_hand, legal, public_record)` in
`solver/adaptive.rs`; `branch_masses` and `condition` classify possible acting
hands, not clairvoyant per-world best actions. It is not yet a Gus adapter.

The bridge must carry the **complete pre-root public history**, bidder and
original own-hand reconstruction: `RootPosition` retains aggregate pre-root
residue, while `PublicRecord.history` contains only post-root plays. It must
select G0 or pinned Zeb by acting seat, reproduce existing token/position
requests exactly, and bind the full context and policy identities in caches.
The extracted DAG is local to its root; its ID/table is not a global Gus policy.
It contains the selected policy's reachable undecided decisions, whereas the
current exporter also supplies all-action Q rows. Export scope, tie handling,
caps and input parity require explicit integration, not a silent replacement.

## Silas can generate examples

The native constructor is implemented on open
[PR #166](https://github.com/jasonyandell/mk5-main/pull/166), head
`c05b8869392b8783fe660704eaef533bf7cd9e33`; it is not merged into main.
Its [README](https://github.com/jasonyandell/mk5-main/blob/c05b8869392b8783fe660704eaef533bf7cd9e33/silas/README.md#constructing-worlds)
and [implementation](https://github.com/jasonyandell/mk5-main/blob/c05b8869392b8783fe660704eaef533bf7cd9e33/silas/src/construct.rs)
define three complementary tools:

- `PreparedFiber`: count and checked unranking of the legal hidden allocations
  for a fixed frame, reusing the allocation table.
- `WorldSpace`: condition a fixed frame on a query having an answer, count the
  complete accepted worlds, and sample them uniformly by integer rank. Multiple
  witnesses never multiply a world's weight.
- `OpeningSpace`: vary all four hands under a fixed legal auction/declaration,
  using a checked ownership-query fragment. The CLI `silas construct` emits
  replayable JSON with count, seed/rank, frame and holdings.

The existing regression fixture describes 1,843,717,075,200 accepted full deals
in 100 ownership cells, with 13,300 different viewer hands. This is a
four-fives ownership family, not a proved bidding or playing strategy. The PR's
eight construction tests pass on a fresh targeted run; the full historical
PR gate was not rerun. No new throughput or strength measurement follows.
The current Python bridge does not expose the constructor classes; use a Rust
bridge or explicit CLI interchange after resolving the PR integration.

## Generation is not the teacher's belief

`WorldSpace` holds own hand and record fixed: more hidden completions alone are
not more observable lessons. `OpeningSpace` can vary own hands but does not invent
legal midgame histories; generate deals and legally play them forward, or mine
recorded prefixes. Its admitted fragment excludes custom predicates and
hand-sensitive viewer predicates such as solved outcomes. Neither constructor
accepts a policy-weighted posterior.

A hidden ownership query may enrich the physical training population, but it
must not become information the actor or teacher assumes the actor knows. Under
the present protocol the teacher still starts from the full uniform
legal-consistent root support and carries it through future field observations.
`silas/src/dynamics.rs::step_belief_given` supplies exact likelihood conditioning;
resetting to uniform after every observed action would change the problem.

Generated shape does not prove a consequential choice. The continuation solve
must supply the differing action values and an executable observation-contingent
policy. Public/own-hand features can stratify lessons; hidden state stays in
referee/generator evidence. Natural held-out play remains separate from enriched
training. Preserve all 630,987 existing reservations and detect duplicate physical
deals across generators by canonical holdings, not by seed labels alone.

## Existing pours are usable source positions

[[lem-pour-a-2026-10-16]] supplies 250,000 full deals and 4,349,960 recorded
positions. The local raw pip and doubles/no-trump archives and manifests exist;
`scenario/atlas.py::Pour` retains all 28 actor-attributed plays per deal, original
hands, declaration and final score. `Pour.position` reconstructs a prefix, so
four-tile roots can be replayed without new self-play just to obtain histories.

The useful slice for the unchanged current task is the **pip-declaration**
portion (217,376 deals / 3,787,301 labeled positions), after checking training
provenance and exclusions. Doubles/no-trump coverage does not authorize expanding
the current seven-pip declaration contract. Older local 200k Zeb shards add
3,478,822 recorded positions, and a 50k pour has 864,604; those pools need the
same source and duplicate checks rather than naive concatenation. Current
reserved-deal exclusions still apply; a new protocol would have to name any
permitted reuse of earlier TRAIN sources explicitly. Evaluation reservations
remain unavailable for teaching.

The stored splats are sampled planning labels from their own historical fields;
they are not exact G0/Zeb contingent certificates. Reuse physical prefixes and
**relabel** under the selected teacher. Off-policy source histories are legitimate
candidate roots when their distribution is declared, but do not establish current
on-policy whole-hand performance. Existing training deals cannot become fresh
confirmation merely by cutting them at an earlier trick.

## Smallest useful next integration

1. Translate an existing tiny teacher root into the native solver, preserve
   G0/Zeb requests and full history, and compare exact values plus extracted-policy
   replay against the present rational reference.
2. Collect a bounded spread of genuine four-tile roots from existing lawful
   trajectories. Four tiles at trick start and four tiles acting last are distinct
   scopes: worst-case hidden allocations are 34,650 and 1,680 respectively.
3. Measure consequential choices, contingent witnesses, distinct observable
   coverage and generation cost before committing to a teaching-corpus size.
4. Use Silas's opening families to fill demonstrated coverage gaps, then compare
   broader teaching against the frozen recipe on untouched physical deals.

These are proposed steps, not registered experiments. Preserve clean inputs,
fixed G0/pinned Zeb, no deployed search, and the separation of teacher gain,
TRAIN acquisition, unseen finite transfer and actual whole-hand strength.
