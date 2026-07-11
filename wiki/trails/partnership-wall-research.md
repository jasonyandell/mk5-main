---
title: Partnership Wall Research
kind: trail
first_seen: local-2026-07-11
last_updated: local-2026-07-11
status: active
---

This trail reads the wiki as a cumulative experimental record around
[[the-wall]]. It does not begin from a preferred architecture. It separates
surviving hypotheses, negative results, confounded measurements, unbuilt
designs, and missing instruments, then routes them through
[[partnership-value]] and [[partnership-research-gates]].

## Baselines that must remain distinct

The pure-play baseline is [[expected-q-value|E[Q]]] n=10 consumed as
`lens:ev`. It remains the strongest measured play policy. The full-match
baseline is `margin:wp(head_8) + lens:ev`: [[jud]]'s realized-outcome bidder
over the same play policy. It beat the prior `net:wp + lens:ev` stack by
`+0.38` and `+0.42` marks/game on two reserved seed blocks
([[w42-plateau-probe]], [[w42-jud-v1]], [[champion]]).

The demonstrated full-match gain is therefore a bidding gain. No current
result demonstrates a play-side partnership gain. The production
[[forge]] selector also remains distinct from the validated baseline: its
`p_make` collapse was not changed to the stronger `lens:ev` collapse found by
[[w42-lens-v1-utility-head-to-head]].

## Evidence map

| workstream | surviving evidence | contrary evidence or seam |
|---|---|---|
| [[forge]] | Exact complete-world solving, [[expected-q-value|E[Q]]], and the analysis substrate are separate working artifacts. They provide a bootstrap, referee, and per-world values. | Perfect-information value does not establish hidden-information partnership strategy; [[oracle-vs-human-play]] keeps that boundary explicit. |
| [[expected-q-value|E[Q]]] | Legal actions are evaluated across sampled worlds consistent with public history. The full 85-bin action PDF survives for downstream use, and `lens:ev` remains the pure-play champion. | Consistency and void evidence do not weight worlds by the likelihood of voluntary actions. Signaling was explicitly deferred. A fresh scalar choice discards distribution shape and any persistent convention. |
| [[gus]] | Belief, policy, value, world-Q, and opponent-policy heads were built. Public play sequence carries signal; auction conditioning improved held-out belief accuracy by `+2.59pp` in [[w42-champion-auction-belief]]. | Gus reached the consistency-information ceiling early, every LAMIR/lookahead consumer lost to direct `pi_me`, and improved calibration did not improve downstream play ([[belief-bayes-ceiling]], [[belief-propagation-gap]]). Better belief is not yet better use of belief. |
| [[burl]] | Rules-as-tools, game-state inspection, belief trajectories, and an interactive microscope established useful diagnostic grammar. | Gemma silently could not see tool responses in every B2-B9 rollout. The 90% result, environment-shape tests, and candlewax-policy null are confounded; the clean post-fix check covered only five decisions. Burl has not established partnership strength ([[gemma-tool-response-shape]]). |
| [[w42]] | The book became detectors, legal-action tables, paired contrasts, branch atlases, and bounded claim statuses. Joined public tags reduced regret from `1.360` to `1.126`; sequence/seat was the dominant family ([[w42-phase3-joined-claim-row-model-table]]). | Tags predict action selection but do not prove the strategies behind them. Partner fit, action likelihoods, persistent plans, and auction partner/opponent values remain absent or design-only. |
| [[champion]] | The full marks-to-7 arena, bidder ladder, auction-conditioned belief, and self-play bridge were built. Realized-outcome bidding produced the first learned marks win over the prior champion. | The play arena is PIMC-versus-PIMC and cannot reward signaling or concealment. Belief-weighted and score-conditioned play nulls do not test an information-reactive partnership ([[champion-design-review]]). |
| [[jud]] | Policy-conditional realized value prices auctions honestly. Current-trick search recovered two-thirds of greedy Jud play's gap without an oracle. | Greedy play, loop rounds, more worlds, and better paper calibration did not reach `lens:ev`. Hand-level targets and capacity remain entangled explanations; [[w42-jud-v1]] did not discriminate between them. |
| [[the-book-enters|Winning 42]] | Claim-by-claim validation found a trustworthy hypothesis source. [[w42-book-second-pass]] exposed auction decoding, choice-derived inference, signaling, role/order conditionals, score conventions, and concrete multi-trick sequences missed by the first pass. | Book claims are hypotheses until the matching information regime and utility are tested. [[book-strategy-player]] was designed but never built, and several plan claims were evaluated with the wrong instrument. |

## Epistemic ledger

### Hard results

- [[champion]]'s score-conditioned play lost `-1.20` marks/game; changing
  play risk by match score is not a supported lever.
- [[w42-lens-v1-utility-head-to-head]] left EV as the top-scoring tested
  one-step collapse on the bid-30 slice. `p_make`, CVaR, robust quantile, and
  disaster-style policies did not establish a route past EV.
- [[w42-jud-v1]] found greedy Jud play to be a poor move ranker. Self-play
  rounds moved its play quality zero; doubling worlds and improving
  calibration did not close the residual search gap.
- [[gus]] found every tested LAMIR/lookahead variant worse than direct
  `pi_me`. Better hidden-state prediction alone did not improve its consumer.

### Confounded or mis-scoped results

- [[champion]]'s belief-weighted play null used opponents and partners that do
  not react to revelation. It is evidence against posterior reweighting in
  that harness, not against partnership inference.
- [[burl]]'s major pre-fix evaluations used rendered prompts that discarded
  tool responses. They cannot establish either a tool-policy ceiling or a
  clean negative for LLM reasoning.
- The high-bid pounce claim was graded with perfect-information oracle value
  although [[w42-book-second-pass]] states an imperfect-information hedge.
- Historical versus-random evaluation used a non-winning objective and had
  seat asymmetry ([[vs-random-eval-is-suspect]]).
- `WorldSamplerMRV` differs from exact enumeration by about 6.8 Q points in a
  tractable late-hand check. [[consumption-ledger]] records the unresolved
  threat to historical labels and evaluations.

### Unbuilt, not failed

- Voluntary-action likelihood models and action-conditioned posterior updates.
- Fixed-versus-shuffled partnership cohorts and a partner-interaction metric.
- Partners and opponents that update beliefs from actions inside rollouts.
- Persistent information-set plans with partner-visible intent.
- A context-licensed consumer of the full action PDF.
- A causal match-score auction ablation.
- [[book-strategy-player]], information-set resolving, and the proposed
  Jud-v2 continuation.

## The generic-plan correction

[[book-strategy-player]] proposed `recognize -> commit -> execute -> retire` as
a general answer to one-step play. [[champion-design-review]] corrects the
load-bearing overclaim: Forge Q already solves to the end of the hand inside
each exact world, so ordinary singleton, reentry, trump-pull, and count-cash
plans are already priced within-world.

Only information-set consistency, partner-visible intent, concealment, and
belief-dependent continuation remain plausible missing plan value. The
strip-the-protector, double-ahead-of-off, crisis re-plan, and two-defender 84
sequences in [[w42-book-second-pass]] remain hypotheses for those narrower
mechanisms, not evidence that a generic symbolic plan library will win.

## Stage 0 — measurement cleanup

1. Baselines are pinned by their actual consumers: `P0 = lens:ev` for pure
   play and `C0 = margin:wp(head_8) + lens:ev` for full matches.
2. `WorldSamplerMRV` is audited against exact enumeration on tractable late
   states. PDF error, action-rank error, and selected-action error accompany
   mean-Q error.
3. Full matches to seven marks become primary. Regret, belief calibration,
   points, make rate, and explanations remain diagnostics.
4. Every record fingerprints bidder, player, value head, corpus, sampler,
   utility, and partner assignment. [[w42-jud-v1]]'s policy-conditional pricing
   forbids mixing consumers without relabeling.
5. Game/deal-clustered intervals and common sampled worlds replace
   decision-row independence assumptions.
6. An information-sensitive harness makes partners and opponents react to
   observed actions.
7. Forced state injection covers 84, match point, rare plans, and partner
   signals that self-selected trajectories rarely reach.

The stage closes only when exact-enumeration disagreement is bounded, `C0`
reproduces on two seed blocks, and each result names one bidder, player,
partner, and measurement mechanism.

**Sampler-audit result:** [[world-sampler-mrv-audit]] falsifies the advertised
validity guarantee before any C0 reproduction. On the historical late state,
greedy MRV reaches a no-candidate branch with exact probability `1/3`; the live
code injects `00` outside the pool in `0.33449` of 100,000 samples and shifts an
action by as much as `4.619 Q`. No best action flips in the three-state panel,
so population decision harm remains unmeasured. Uniform rejection then passed
the three audit fixtures but failed a real JudSearch state: only `924` of
`17,153,136` labeled-seat partitions are valid, and 40,960 proposals produced
3 of 10 requested worlds. The surviving `uniform-completion-dp-v1` replacement
samples from exact suffix-completion counts and passes every exact support plus
that low-mass regression. A CUDA benchmark, state-level exposure audit, and
two-block C0 reproduction now precede the remaining Stage-0 instruments.

**Decision-record result:** [[partnership-decision-record-v1]] now replays
Arena hands into one row per actual play with separate public, actor-information,
auction/score-context, and offline-world identities. It fingerprints C0's
bidder, player, artifacts, sampler, utility, code state, roles, and static
partner assignment without pretending that action likelihood, fixed/shuffled
cohorts, plan state, or Q/PDF tensors were observed. This closes the shared
future-state identity seam; it does not close C0 reproduction or the reactive
partnership harness.

## Stage 1 — joined failure atlas

The first atlas contains a balanced, inspectable cohort of high-drama early
decisions plus matched low-drama controls across bidder, partner, left setter,
right setter, and trick order. Forced rare fixtures remain a separate stratum.
[[gus-drama-atlas]] supplies outcome variance, action fragility, and belief
sharpness; [[w42-phase2-seat-position-strategy-map]] supplies role/order and
public-evidence boundaries.

The join key combines deal, match score, auction, play prefix, actor, and legal
action. Each row records:

- role/order, partner/opponent identity, auction, score, and legal actions;
- `C0`, [[w42-jud-v1|JudSearch]], [[gus|Gus]], [[w42|W42]], and relevant
  book-conditional choices;
- E[Q] mean, full PDF, per-world Q, fragility, and sampling error;
- beliefs before and after each public action plus actor-policy likelihood;
- detector, plan, continuation, completion, and disruption labels;
- fixed/shuffled partner condition, points, marks, make/set outcome, and all
  policy fingerprints.

The atlas assigns failures only after controlling for ordinary move
difficulty: generic ranking, auction interpretation, action-derived belief,
role/order, partner interaction, plan continuation, distribution shape, or
score.

**v0 build result:** [[partnership-failure-atlas-v0]] joins 75,079 legal
actions across 28,000 W42 decisions with exact role/order and action-local
proxy identity. It also falsifies the stronger assumption that the retained
archive can already localize *Champion* failures: source policy, champion
action-state fingerprints, Gus drama identity, partner assignment, action
likelihood, persistent plan state, and match score do not join. The atlas is a
measurement spine, not yet the balanced champion-failure cohort specified
above.

## Stage 2 — discriminating causal microgames

| direction | supporting evidence | contrary evidence and prior attempt | missing instrument | falsifying experiment |
|---|---|---|---|---|
| Target granularity x capacity | [[w42-jud-v1]] shows a useful value head but weak move ranking; JudSearch recovered `+2.28` marks/game over greedy play. | [[gus|Gus]] often gained more from data than capacity. [[jud|Jud]] changed worlds, calibration, and input data, not target and capacity independently. | A factorial crossing hand-level versus per-move ranking targets with present versus modestly larger capacity on identical states and compute. | Per-move targets fail to improve held-out pairwise ranking or JudSearch marks. A capacity-only interaction selects capacity; a target-only interaction selects supervision. |
| Auction decoder x role/order x score | Auction-conditioned [[gus|Gus]] gained `+2.59pp`; realized-value bidding is the [[champion]]'s only demonstrated marks gain; [[w42-book-second-pass]] gives explicit bid-to-hand and score hypotheses. | Score-conditioned play was negative, the pass-model pilot was null, and book conventions may be population-specific. | Same-hand counterfactual auctions with partner/opponent, failed/winning bid, auction position, and scores `{0-0, 6-6, 6-0, 0-6}` plus masked controls. | Bid/role claims have no held-out information, or the decoder and score treatment fail to improve paired marks over masked and shuffled-auction controls. |
| Action likelihood x partner convention | [[the-book-enters|The book]] supplies choice-derived inference and signaling hypotheses; [[w42]]'s sequence/seat family dominates; [[w42-style-partnership-concept-buckets]] finds directional partner-legibility proxies. | Some support proxies are harmful; [[champion]]'s null was harness-blind; [[gus]]'s stronger opponent policy did not improve lookahead. No direct voluntary-action likelihood test ran. | Sender convention on/off x receiver decoder on/off x fixed/shuffled partner, with belief-updating opponents and permuted-role controls. | Action likelihood fails to improve held-out belief log loss, or improved belief produces no sender-by-decoder interaction in set rate or marks. |
| Information-set plan persistence | [[w42-book-second-pass]] supplies fully specified strip-the-protector, double-ahead-of-off, crisis, and two-defender-84 sequences. | [[forge|Forge]] Q already prices ordinary within-world plans; [[w42-jud-v1|JudSearch]] captures current-trick continuation; [[book-strategy-player|BookStrategyPlayer]] never ran. | Merged information sets requiring one common policy across worlds, with persistent/erased plan state and partner-visible/partner-blind variants. | Forge already selects and continues the plan in the full-information control, and persistent state adds no lift in merged-information or partner-visible arms. |
| Contextual distribution consumer | Full action PDFs exist; [[gus-drama-atlas]] localizes uncertain, fragile, high-impact opening decisions. | No tested fixed collapse beat EV, score-conditioned play lost, and [[burl]]'s distribution-policy result is confounded. | Matched-mean, different-shape fixtures with exact future-information or match utility, stratified by role and score. | Retaining PDF shape never changes the optimal continuation or improve held-out marks over EV. Distribution shape then remains diagnostic rather than causal. |

## Stage 3 — architecture selection

[[partnership-research-gates]] maps each causal result to an eligible build.
CFR, a larger network, an LLM, a symbolic strategy library, and [[jud|Jud]] v2 remain
unselected until their specific mechanism passes. Belief accuracy,
interpretability, book agreement, and E[Q] imitation do not satisfy a gate by
themselves.

## Stage 4 — build ladder

1. Land the joined schema, enumeration audit, and forced-microgame runner.
2. Reproduce `C0` with every diagnostic and partner assignment recorded.
3. Add the smallest mechanism that passes a Stage-2 falsifier as a narrow
   adapter over `C0`.
4. Run marks-to-7 matches with fixed and shuffled partners against
   information-reactive opponents.
5. Retrain policy-conditioned value only if the adapter changes the policy
   distribution.
6. Remove the claimed channel in a causal ablation.
7. Scale capacity, search, or self-play only after the mechanism survives.
8. Use W42 detectors and Burl narration afterward to explain the measured
   gain, not to define it.

## Success criterion

Success is a policy that beats `C0` on paired full-match marks on two held-out
seed blocks, gains more with matched than shuffled partners, and loses the
advantage when the proposed role/partnership mechanism is removed. That is a
strategic marks gain. Interpretability, E[Q] imitation, and book agreement are
instruments.

## Links

[[partnership-value]] [[partnership-research-gates]] [[the-wall]]
[[the-wall-biography]] [[consumption-ledger]] [[champion-design-review]]
[[w42-book-second-pass]] [[w42-jud-v1]]
