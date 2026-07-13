---
title: Partnership Wall Research
kind: trail
first_seen: bc4eb386
last_updated: b28fb55a
status: active
---

This trail reads the wiki as a cumulative experimental record around
[[the-wall]]. It does not begin from a preferred architecture. It separates
surviving hypotheses, falsified designs, adequately measured nulls, bounded
null observations, confounded measurements, unanswerable questions, unbuilt
designs, and missing instruments, then routes them through
[[partnership-value]] and [[partnership-research-gates]].

Scope: partnership/coordination is the family this program newly instruments
— the best-specified untested direction, not the presumptive one. The
measurement spine built here (repaired sampler, canonical decision records,
and a fingerprinted C0 reproduction contract) serves every registered
direction on [[the-wall]] equally:
[[jud]] v2's per-move-target ladder, CFR+ over distilled values
([[lamir1-ceiling]]), and the contextual distribution consumer
([[past-belief-future-direction]]) gate through the same
[[partnership-research-gates]] table as the coordination mechanisms.

## Current position

| surface | status | what is true now | boundary / next gate |
|---|---|---|---|
| Pure-play baseline | MEASURED + REPRODUCED | `P0 = lens:ev` n=10 remains the strongest measured play policy; on the repaired sampler the symmetry sanity is clean and `judsearch:n10(r4)` still loses `-1.42`/`-1.54` on both held-out blocks ([[stage-0-closure]]). | The next challenger is Lane B's per-move-target ladder. |
| Full-match baseline | MEASURED + REPRODUCED | `C0 = margin:wp(head_8) + lens:ev`; its demonstrated advantage is a bidding gain, reproduced on the repaired sampler: `+0.385 [+0.102,+0.668]` and `+0.486 [+0.199,+0.775]` on the two reserved blocks ([[stage-0-closure]]). | Lane grading is unblocked. |
| World sampler | BUILT + validated + measured | `uniform-completion-dp-v1` (with the `4123b2d5` MPS `where` fix) passes every per-device regression including CUDA; it is kernel-launch bound (~30 ms/call), so batch width, not device, is the throughput lever, and a C0 block costs ~2× legacy wall time on MPS. Historical exposure: `2.51%` of a 32k reconstructed late-state population carried nonzero legacy malformed mass. | Scan B decision-level harm (argmax flips/regret) reports under [[stage-0-closure]]. |
| Historical failure atlas | BUILT | [[partnership-failure-atlas-v0]] joins 75,079 actions / 28,000 decisions and inventories 114 live sources. | It proves archive insufficiency for Champion attribution; it does not measure a partnership null. |
| Future decision record | BUILT + integration-validated | [[partnership-decision-record-v1]] records public/info/context/world identities, eight separate mechanism sections, and exact policy/artifact/sampler provenance. | A clean C0 smoke retained 644 decisions; live Q/PDF, belief change, action likelihood, plan state, and fixed/shuffled cohort remain unavailable. |
| Partnership value | UNTESTED | No valid negative, null, or positive partnership result exists. | Requires fixed-versus-shuffled partners and actors that react to public actions. |
| Research directions | SELECTED 2026-07-13 | Competing explanations remain preserved below; [[research-lane-selection]] selects the [[auction-decoder]] and the [[jud]] target-granularity ladder as the primary experiments, with the convention factorial following. | Stage 0 closes before any lane result is graded; if both primary gates fail, selection returns to this ledger. |
| Successor architecture | INTENTIONALLY UNSELECTED | CFR, larger nets, LLMs, symbolic libraries, and Jud v2 remain candidates, not plans. | Only a mechanism that passes [[partnership-research-gates]] earns a build. |

## PR boundary — measurement readiness

This PR ends at the ability to make the next result trustworthy. It supplies:

- an exact-fixture-audited production sampler and explicit remaining exposure
  and performance boundaries;
- canonical decision identities and policy, artifact, sampler, role, score,
  and partner provenance;
- a joined historical atlas that says exactly which retrospective questions
  the surviving records cannot answer;
- a result vocabulary that keeps falsification, null, bounded observation,
  confound, and instrument insufficiency distinct; and
- one cumulative evidence map in which competing explanations remain visible.

It does not select the next mechanism, causal microgame, experiment, or
architecture. Benchmarking the sampler, measuring historical exposure, and
reproducing P0/C0 close the measurement baseline; they do not privilege a
research direction. The unlock is that a later choice can now be recorded,
joined, and falsified for the reason it claims.

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

## Result vocabulary

The program uses result labels narrowly so absence of evidence does not become
evidence of absence.

| label | meaning here | current example |
|---|---|---|
| Finding | A valid measurement establishes a mechanism or fact. | Legacy MRV can fabricate worlds; exact malformed mass is `1/3` on one fixture. |
| Negative result / falsification | A valid test makes a proposed mechanism, design, or useful effect fail its stated criterion. | `uniform-rejection-v1` is mathematically uniform when it returns but fails a real low-valid-mass state, so it is rejected as the production repair. |
| Null result | A valid, adequately powered test finds no useful effect within stated bounds. | No partnership null has been measured yet. |
| Bounded null observation | No effect appears in a small diagnostic panel whose scope is explicitly limited. | No action argmax flips in the three sampler-audit states. |
| Confounded result | More than the intended variable changed, so the effect cannot be attributed. | The historical `~6.8 Q` comparison used different world encodings and lacks action/N/RNG provenance. |
| Instrument insufficiency / inconclusive | Available records cannot answer the question. | The retained archive cannot attribute current-Champion partnership failures; the partnership hypothesis remains untested. |

## Evidence map

| workstream | surviving evidence | contrary evidence or seam |
|---|---|---|
| [[forge]] | Exact complete-world solving, [[expected-q-value|E[Q]]], and the analysis substrate are separate working artifacts. They provide a bootstrap, referee, and per-world values. | Perfect-information value does not establish hidden-information partnership strategy; [[oracle-vs-human-play]] keeps that boundary explicit. |
| [[expected-q-value|E[Q]]] | Legal actions are evaluated across sampled worlds consistent with public history. The full 85-bin action PDF survives for downstream use, and `lens:ev` remains the pure-play champion. | Consistency and void evidence do not weight worlds by the likelihood of voluntary actions. Signaling was explicitly deferred. A fresh scalar choice discards distribution shape and any persistent convention. |
| [[gus]] | Belief, policy, value, world-Q, and opponent-policy heads were built. Public play sequence carries signal; auction conditioning improved held-out belief accuracy by `+2.59pp` in [[w42-champion-auction-belief]]. A belief-sampled Q-mean second opinion modestly beat direct `pi_me`, and a learned router reduced regret further on 560 held-out decisions ([[gus-qmean-router]]). | Every tested multi-step LAMIR-1 rollout mode lost to direct `pi_me`; improved calibration did not improve downstream play ([[lamir1-ceiling]], [[belief-propagation-gap]]). The Q-mean router had only eight baseline blunders and no marks-to-7 validation. Better belief is not yet reliably better use of belief. |
| [[burl]] | Rules-as-tools, game-state inspection, belief trajectories, and an interactive microscope established useful diagnostic grammar. | Gemma silently could not see tool responses in every B2-B9 rollout. The 90% result, environment-shape tests, and candlewax-policy null are confounded; the clean post-fix check covered only five decisions. Burl has not established partnership strength ([[gemma-tool-response-shape]]). |
| [[w42]] | The book became detectors, legal-action tables, paired contrasts, branch atlases, and bounded claim statuses. Joined public tags reduced regret from `1.360` to `1.126`; sequence/seat was the dominant family ([[w42-phase3-joined-claim-row-model-table]]). | Tags predict action selection but do not prove the strategies behind them. Partner fit, action likelihoods, persistent plans, and auction partner/opponent values remain absent or design-only. |
| [[champion]] | The full marks-to-7 arena, bidder ladder, auction-conditioned belief, and self-play bridge were built. Realized-outcome bidding produced the first learned marks win over the prior champion. The belief-conditioned self-play loop converges to a stable fixed point ([[w42-champion-selfplay-fixed-point]]). | The play arena is PIMC-versus-PIMC and cannot reward signaling or concealment. Belief-weighted and score-conditioned play nulls do not test an information-reactive partnership ([[champion-design-review]]). The converged belief-conditioned bidder over-bids via double-dummy optimism ([[strategy-fusion]]) and loses on marks even after calibration. |
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
- [[gus]] found every tested multi-step LAMIR-1 rollout mode worse than direct
  `pi_me`. [[lamir1-ceiling]] carries the mechanism (Kubíček & Lisý): distilled
  scalar value noise flips argmax at decision boundaries, while a policy
  trained on argmax preserves ordering; a distilled leaf cannot support
  look-ahead without CFR+. The [[consumption-ledger]] records this warning as
  having pre-explained JS3 — the prior-sweep lesson.
- That is not a universal consumer negative. [[gus-qmean-router]] found a
  belief-sampled Q-mean second opinion modestly improved regret, and a learned
  router over direct `pi_me` versus Q-mean reduced it from `0.551` to roughly
  `0.42–0.43` while routing `5–7%` of decisions. The result is bounded by 560
  decisions, eight baseline blunders, and no full-match marks evaluation, but
  it preserves selective candidate consumption as positive evidence.
- [[w42-champion-selfplay-fixed-point]] found the belief-conditioned bidder
  reaches a stable self-play fixed point yet loses `-3.0` to `-3.8` marks/game
  raw and `-2.0` to `-2.6` calibrated against `net:wp`: double-dummy P(make)
  optimism ([[strategy-fusion]]) over-bids, and dampening it shrinks but does
  not close the gap. Any auction-decoder consumer must price realized, not
  double-dummy, outcomes.

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
- The historical `~6.8 Q` sampler comparison changed world encoding and lacks
  action, sample-count, and RNG provenance. [[world-sampler-mrv-audit]] retires
  it as a clean estimate; historical population exposure remains unanswered.

### Unbuilt, not failed

- Action-conditioned posterior updates that consume a voluntary-action
  likelihood. An opponent-action model exists — [[pi-opp-head]], 68.57%
  oracle top-1 — but nothing updates belief from it.
- Fixed-versus-shuffled partnership cohorts and a partner-interaction metric.
- Partners and opponents that update beliefs from actions inside rollouts.
- Persistent information-set plans with partner-visible intent.
- A context-licensed consumer of the full action PDF, including the
  mode/signal/hedge/gamble meta-strategy set of
  [[past-belief-future-direction]] whose training data already sits in the
  oracle's per-world tensor.
- A causal match-score auction ablation.
- CFR+ / multi-valued-states over distilled values — the one
  theoretically-sanctioned look-ahead path never walked ([[lamir1-ceiling]]).
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
that low-mass regression. Review of the shipped repair then found one more
defect in the same silent-bias class: MPS's int64 `gather` rounds the 62-bit
draws through float32, leaving every world valid while collapsing the
distribution (support 24/60, chi-square `1.4e5` at 60k samples on the bias
fixture). Fixed at `4123b2d5` with an exact `where` selection; the dead-end
and uniformity regressions now parameterize over every available device. A
CUDA benchmark, state-level exposure audit, and two-block C0 reproduction now
precede the remaining Stage-0 instruments.

**Decision-record result:** [[partnership-decision-record-v1]] now replays
Arena hands into one row per actual play with separate public, actor-information,
auction/score-context, and offline-world identities. It fingerprints C0's
bidder, player, artifacts, sampler, utility, code state, roles, and static
partner assignment without pretending that action likelihood, fixed/shuffled
cohorts, plan state, or Q/PDF tensors were observed. This closes the shared
future-state identity seam; it does not close C0 reproduction or the reactive
partnership harness.

## Stage 1 — joined failure atlas

The intended joined failure view distinguishes high-drama early decisions from
matched low-drama controls across bidder, partner, left setter, right setter,
and trick order. [[gus-drama-atlas]] supplies outcome variance, action
fragility, and belief sharpness; [[w42-phase2-seat-position-strategy-map]]
supplies role/order and public-evidence boundaries.

The complete record contract would join deal, match score, auction, play
prefix, actor, and legal action, with:

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
above. This is an instrument-sufficiency falsification, not a negative or null
result about partnership value.

## Stage 2 — competing explanation ledger, not a plan

| direction | supporting evidence | contrary evidence and prior attempt | unresolved seam |
|---|---|---|---|
| Target granularity x capacity | [[w42-jud-v1]] shows a useful value head but weak move ranking; JudSearch recovered `+2.28` marks/game over greedy play. [[lamir1-ceiling]] supplies a mechanism: distilled scalar value noise flips argmax while a policy trained on argmax preserves ordering. | [[gus|Gus]] often gained more from data than capacity. [[jud|Jud]] changed worlds, calibration, and input data rather than separating target from capacity. A leaf used for look-ahead may require CFR+ / multi-valued states rather than supervision alone. | Target quality, capacity, and look-ahead fitness remain entangled. |
| Auction decoder x role/order x score | Auction-conditioned [[gus|Gus]] gained `+2.59pp`; realized-value bidding is the [[champion]]'s only demonstrated marks gain; [[w42-book-second-pass]] supplies bid-to-hand and score hypotheses. | Score-conditioned play was negative, the pass-model pilot was null, and book conventions may be population-specific. [[w42-champion-selfplay-fixed-point]] found a converged belief-conditioned bidder still lost by pricing double-dummy P(make). | Information gain, role/order semantics, score use, and realized-outcome pricing have not been attributed separately. |
| Action-derived inference and partnership legibility | Table play naturally reveals information about holdings, priorities, and intent relative to a partner's known policy; [[w42-book-second-pass]] records choice-derived inference and explicit conventions. [[convention-aware-blueprint-search]] preserves the book as a possible coordinated sender/receiver initialization rather than waiting for unilateral search to invent a code. | Most informative actions are selected because they play well, not because they are deliberate messages. Intentional reliable signals are sparse; simple action-to-intent rules are noisy; book conventions may be incomplete or population-specific; no result measures their aggregate contribution to marks. | Natural policy legibility, explicit convention, partner-specific familiarity, generic good-play inference, and opponent decoding remain distinct and unmeasured. |
| Information-set plan persistence | [[w42-book-second-pass]] supplies concrete multi-trick sequences. | [[forge|Forge]] Q already prices ordinary within-world plans; [[w42-jud-v1|JudSearch]] captures current-trick continuation; [[book-strategy-player|BookStrategyPlayer]] never ran. | Ordinary continuation, cross-world policy consistency, and partner-visible intent remain separated in theory but not evidence. |
| Belief-weighted Jud MCTS | [[w42-jud-v1|JudSearch]] improved greedy Jud play by `+2.28` marks/game; its worlds sweep added only `+0.11`, so flat sample count was not the lever. [[gus-qmean-router]] keeps belief-sampled candidate generation in the positive ledger. | Zeb MCTS never beat E[Q] n=10 at pure play; every LAMIR-1 look-ahead mode lost to direct `pi_me`; a determinized MCTS tree retains strategy fusion. | Adaptive depth, information-set node identity, mid-tree belief updates, leaf fitness, and convention response remain separable and unmeasured. See [[belief-weighted-jud-mcts]]. |
| Contextual distribution consumer | Full action PDFs exist; [[gus-drama-atlas]] localizes uncertain, fragile, high-impact opening decisions; [[past-belief-future-direction]] describes a richer meta-strategy surface. [[gus-qmean-router]] is bounded positive evidence for selective consumption. | No tested fixed collapse beat EV, score-conditioned play lost, and [[burl]]'s distribution-policy result is confounded. | The project has not shown when distribution shape changes a valuable decision or full-match marks. |

[[research-lane-selection]] (2026-07-13) selects two rows as the primary
experiments — auction decoder and target granularity — with the convention
factorial following. The ledger's purpose is unchanged: it prevents the
selection from forgetting positive evidence, contrary evidence, or missing
instruments, and it is where selection returns if the primary gates fail.

Among the surviving architectures, [[convention-aware-blueprint-search]] is
unusually concrete: it joins action-derived inference, policy legibility, and
information-set continuation through an agreed blueprint, and it can consume
the measurement substrate already built by this program.

### Book-seeded convention overlay — why it survives

[[w42-book-second-pass|Winning 42]] provides the first concrete coordinated
policy initialization in the project record: install the same convention
overlay on sender and receiver, retain a learned policy as the complete
fallback, then let search operate inside an already shared codebook. This
escapes the requirement that unilateral search invent both sides of a signal
at once.

The direct double-dummy verdict is not the classifier. Oracle endorsement can
coexist with additional partner-decoding value when good technique is also
informative. The discriminating measurement is a sender-overlay x
partner-reader x opponent-reader factorial:

- sender on, readers off isolates direct technique or sacrifice;
- partner reader on isolates the partnership interaction;
- opponent readers on expose the price of becoming legible to the other team;
- paired full-match marks give the net only after those components remain
  separately attributable.

The book also supplies opponent action-likelihood hypotheses. Opponent
book-likeness can be inferred as a latent policy type, creating a four-seat
information model rather than only a private partnership code. The same public
action can coordinate a partner, reveal an opponent's likely hand, and expose
the sender to counter-inference; the factorial measures all three.

Top-unplayed-trump/donation, donate-highest, dump-to-inform, 84 keep priorities,
auction messages, and the separate Plunge/Splash variant provide an immediate
mining surface. The candidate remains alongside target granularity, auction
decoding, information-set planning, and distributional utility. It earns a
durable place in the ledger because it is structurally distinct, buildable from
existing pieces, and capable of producing mechanism-specific evidence even
when individual conventions vary.

### Belief-weighted Jud MCTS — why it survives

[[belief-weighted-jud-mcts]] is the natural search consumer joining Jud's
positive current-trick result to the information semantics above. JudSearch
already established that search can turn the realized-value leaf into a much
better move ranking (`+2.28` marks/game over greedy play). Doubling worlds did
almost nothing; MCTS changes adaptive branch allocation and continuation depth,
not merely sample count.

The decisive fork is node identity. Root-belief determinized MCTS is deeper
JudSearch and may add tactical value, but its separate world trees retain
strategy fusion. Information-set MCTS shares action statistics across worlds
the actor cannot distinguish and updates later-seat beliefs after simulated
public actions. That version can price information-set consistency, action-
derived inference, and partner response.

The resulting stack is compact:

`belief particles -> information-set MCTS -> blueprint policy -> Jud V_realized leaf`

The nested J0/J1/J2/J3/J4 decomposition on the topic page separates current
JudSearch, root belief weighting, adaptive determinized depth, information-set
belief updates, and the book/learned convention layer. Zeb and LAMIR remain
relevant prior evidence, but neither tested this combination. The idea earns
research-trail visibility because its strongest components have independent
project evidence and its structural increment has a clean ablation.

## Architecture and build remain withheld

[[partnership-research-gates]] records what evidence would make an architecture
eligible; it is not an implementation queue. CFR, a larger network, an LLM, a
symbolic strategy library, and [[jud|Jud]] v2 remain unselected as
architectures. [[research-lane-selection]] selects the next *experiments*;
promotion continues to gate through the table unchanged.

The general wall criterion is higher paired held-out full-match marks for a
demonstrated strategic reason, with the gain removed by the claimed mechanism's
ablation. A partnership-value claim additionally requires a matched-pair
advantage over shuffled partners. Interpretability, E[Q] imitation, and book
agreement remain instruments.

## Links

[[partnership-value]] [[partnership-research-gates]] [[the-wall]]
[[the-wall-biography]] [[consumption-ledger]] [[champion-design-review]]
[[w42-book-second-pass]] [[w42-jud-v1]] [[w42-champion-selfplay-fixed-point]]
[[lamir1-ceiling]] [[strategy-fusion]] [[past-belief-future-direction]]
[[pi-opp-head]] [[world-sampler-mrv-audit]]
[[convention-aware-blueprint-search]] [[belief-weighted-jud-mcts]]
