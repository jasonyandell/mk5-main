---
title: w42 Final Empirical Strategy Report
kind: experiment
first_seen: 2026-05-02
last_updated: 2026-07-06
status: superseded
---

**Superseded.** This closed the *initial* w42 survey (2026-05-02); it is not
final for the workstream, which continued through [[w42-phase4-final-claim-audit]]'s
64-claim closure, the [[w42-book-validation-campaign]] waves 1-4, and the
[[w42-jud-v1]] champion capstone. See [[w42-book-claim-synthesis-and-ai-directions]]
for the later synthesis.

## Summary

[[w42]] completed the initial Winning 42 strategy survey. The result is not "the
book failed to correlate." The result is:

- book-shaped strategy tags improve small supervised policy probes over raw
  public state;
- exact arithmetic, ruleset, and scoring claims can be supported now;
- most tactical and partnership recommendations are still proxy-only,
  context-limited, underpowered, or not-yet-tested;
- `E[Q] N=10` remains much stronger than the tiny w42 models;
- the next phase should be targeted, distribution-aware digging, not broader
  surveying.

The book is useful as a hypothesis generator. The current evidence supports
continuing the w42 line, but it does not justify broad book-claim conclusions.

## Method

The survey used four evidence modes:

| mode | examples | interpretation |
|---|---|---|
| deterministic enumeration / rules checks | [[w42-odds-ruleset-claim-validation]], [[w42-doubles-no-trump-claim-validation]] | can support exact arithmetic and ruleset substrate claims |
| deterministic scoring transforms | [[w42-scoring-objective-drift-claim-validation]] | can support objective-accounting claims |
| small supervised model probes | [[w42-raw-public-state-baseline]], [[w42-v0-strategy-tags-baseline]], [[w42-rich-tag-many-signal-probe]], [[w42-multi-seed-larger-eval-replication]] | tests whether tags help prediction/regret, not whether a book recommendation is true |
| proxy reports / concept buckets | [[w42-concept-bucket-regret]], [[w42-bidding-risk-budget-claim-validation]], [[w42-partner-support-claim-validation]], [[w42-setter-defense-claim-validation]], [[w42-84-claim-validation]], [[w42-belief-attention-concept-buckets]], [[w42-style-partnership-concept-buckets]] | identifies measurable surfaces and priorities, usually not final verdicts |

No new run was executed for this final report. It synthesizes the completed w42
survey artifacts.

## Model Results

The model-side result is promising but bounded.

The five-seed larger-eval replication on 2,800 held-out decisions reported:

| feature set | seeds | mean regret | 95% CI half-width | match | tail `>=5` |
|---|---:|---:|---:|---:|---:|
| raw public state | 5 | 1.967 | 0.044 | 59.94% | 13.05% |
| raw plus v0 tags | 5 | 1.606 | 0.063 | 64.20% | 10.79% |
| raw plus v0 plus rich tags | 5 | 1.515 | 0.050 | 64.89% | 10.38% |

Paired regret deltas:

| comparison | mean delta | 95% CI half-width | reading |
|---|---:|---:|---|
| v0 minus raw | -0.361 | 0.064 | strategy tags clearly helped versus raw |
| rich minus raw | -0.452 | 0.064 | rich tags clearly helped versus raw |
| rich minus v0 | -0.091 | 0.086 | modest positive signal, not a promotion verdict |

On the same replication slice, `E[Q] N=10` reached 0.175 mean regret. The tiny
models remain far behind the boss baseline.

## Distribution-Aware Refinement

Follow-up inspection in the E[Q] PDF browser visualizer showed why mean regret is
necessary but incomplete. Some decisions are not well summarized by their
expected value: the PDF can show separated shelves, threshold cliffs, and heavy
disaster tails. In those positions, the strategic question is not only "which
move has the best mean?" but "which branch am I entering, and can I mitigate the
bad branch before it hardens?"

This matters for the next w42 phase. Book concepts such as preserving the 84
stopper, safe donation, setter pounce, avoiding a broken off suit, or bracing
after a poisoned trick are often tail-risk and branch-management claims. Future
model/report beads should therefore record distribution features when E[Q] PDFs
or sampled worlds are available:

- make/set threshold mass;
- variance and quantiles;
- lower-tail or CVaR-style disaster risk;
- branch or shelf labels visible in the PDF;
- whether a legal move appears to mitigate, expose, or preserve future options.

The sampled-world data also supports a stronger diagnostic than a human table can
observe directly: hidden-domino threat attribution. The visual PDF may show two
high-odds shelves and two intermediate lumps; those modes are often driven by
which player holds a specific unseen domino or set of dominoes. Because the
generated training data records hidden ownership and outcome branches together,
w42 can estimate which unseen holdings have large impact magnitude even when a
live agent only has a belief distribution over them.

That suggests a separate evaluation axis for belief work:

- how much the outcome distribution changes when a specific hidden domino is
  assigned to each plausible holder;
- whether the model's belief mass is concentrated on high-impact hidden
  threats, not just calibrated on average ownership;
- whether a strategy tag or detector identifies plays that mitigate the
  high-impact hidden branches;
- whether the selector distinguishes harmless uncertainty from uncertainty that
  can decide make/set or swing a trick.

This refinement does not change any claim-ledger status. It changes the next
measurement target: do not ask only whether a tag improves average regret; ask
whether it helps identify and respond to branch-shaped tactical states and the
hidden-domino threats that create them.

## Claim Families

| family | current status | evidence | next test |
|---|---|---|---|
| rules and state accounting | supported for checked predicates | Chapter 1 and Chapter 13 deterministic checks passed | keep as regression fixtures |
| exact odds | supported for checked arithmetic | Chapter 16 hand-count, suit/void, double-count, modal-hand, and four-trump assignment enumeration | package as stable odds fixtures if promoted |
| scoring objective drift | supported for deterministic scoring algebra; context-limited for tournament speed | Chapter 10 terminal transforms and scoreboard examples | policy-population and tournament simulation |
| doubles/no-trump ruleset substrate | supported for regime membership/follow predicates; strategy choices underpowered | Chapter 9 deterministic checks and static hand slices | paired declaration/play rollouts |
| bidding risk budget | detector-supported / context-limited | exact static exposure and duplicate-count enumeration | auction-aware make/set or E[Q] counterfactuals |
| bidder sequencing | underpowered / context-limited | proxy buckets improve model alignment but no sequence counterfactual | trump-first/off-first paired rollout |
| partner support | underpowered with directional hints | safe donation and lead-away proxies, often unpaired | direct partner-intent/forcedness detectors |
| setter defense | mostly not-yet-tested / underpowered | missing setter-role and pounce-window detectors; naive count-to-opponent is high regret | implement direct setter-pounce labels |
| 84 bidder/defender | context-limited static support; no regret verdict | exact static weapon/ownership pools; 84 regret missing from model buckets | 84-specific state generator and paired last-trick tests |
| belief, attention, table discipline | design-ready, mostly not-yet-tested | legal public-state boundary mapped | leakage audit and Burl trace lint |
| style and partnership ecology | report-only / not-yet-tested | partner-legibility and overbid proxies are directional but weak | repeated-player or partner-shuffle data |

## Supported Claims

The survey supports only narrow claims where the evidence actually matches the
claim:

- seven-domino double-six hand count is 1,184,040;
- checked suit/void, double-count, modal-hand, and four-trump assignment odds;
- Chapter 1 point accounting and follow-suit predicates;
- Chapter 13 checked ruleset gates;
- Chapter 9 deterministic doubles/no-trump regime predicates;
- Chapter 10 deterministic scoring-objective drift claims.

These are mostly arithmetic, legality, and scoring-substrate claims. They are
real wins, but they are not the same as validating human tactical advice.

## Underpowered And Context-Limited Areas

The tactical material remains the rich digging field:

- "bid only enough" needs auction counterfactuals, not static hand risk.
- "pull trump unless..." needs sequence rollouts, not broad pressure buckets.
- partner donation needs forcedness, true partner state, and legal-public
  feature separation.
- setter pounce needs bidder-off windows and set-threshold accounting.
- 84 needs a dedicated generator and last-trick weapon state tests.
- belief and style need trace audits, repeated players, or controlled prompt/data
  interventions.

The survey did find many measurable surfaces. It did not finish adjudicating
them.

## Caveats

- Many reports use local w42 artifacts and existing Gus corpora.
- Several validation scripts are deterministic one-shot reports rather than W&B
  trajectory runs; the W&B series standard applies to future iterative runs.
- Hidden-hand labels are acceptable for evaluation but not for live strategy
  features.
- The tiny w42 models are probes. They are not production Gus replacements.
- Rich tags can help a model without proving that any specific book sentence is
  strategically correct.

## Artifacts

| artifact | location |
|---|---|
| w42 charter | [[w42]] |
| claim ledger schema | [[w42-claim-ledger]] |
| W&B project | `https://wandb.ai/jasonyandell-forge42/w42` |
| W&B dashboard page | [[w42-wandb-run-comparison-dashboard]] |
| W&B series standard | [[w42-wandb-series-logging-standard]] |
| multi-seed replication | [[w42-multi-seed-larger-eval-replication]] |
| HF artifact decision | [[w42-hugging-face-artifact-publishing]] |

HF links: not applicable. No artifact is mature enough for HF publication yet.

Claim-ledger impact: synthesis only. This page records the current ledger shape
but does not move central claim statuses.

## Next Questions

- Which direct detector should be implemented first: setter pounce, 84 weapon
  preservation, or auction bid margin?
- Does eight-epoch or larger-data training preserve the rich-over-v0 signal?
- Which strategy tags reduce tail-risk errors rather than only mean regret?
- Which PDF/distribution features expose "brace for disaster" or mitigation
  states better than scalar E[Q]?
- Which hidden dominoes and holders explain the main shelves or lumps in an
  E[Q] PDF, and can a belief model learn to weight those threats correctly?
- Can Burl traces expose where the model has strategy vocabulary but poor
  reasoning discipline?
- Which outputs deserve promotion from w42 research into Gus, Burl, forge, or HF?

## Provenance

| field | value |
|---|---|
| bead | `t42-csw6.28` |
| commands | `bd show t42-csw6 --json`; `bd show t42-csw6.28 --json`; wiki/report inspection; `git rev-parse HEAD` |
| configs | not applicable |
| data inputs | completed w42 wiki reports and w42 artifacts cited above |
| commit SHA at synthesis time | `326fc5092d31e94b67fa5b548e6596b6d5e1d3d2` |
| random seeds | not applicable for synthesis; see source reports |
| W&B links | existing project links cited above; no new run |
| HF links | not applicable |

## Links

[[w42]] | [[winning42-strategy-measurement]] | [[w42-claim-ledger]] |
[[w42-multi-seed-larger-eval-replication]] |
[[w42-next-model-decision]] | [[w42-promote-or-retire]]
