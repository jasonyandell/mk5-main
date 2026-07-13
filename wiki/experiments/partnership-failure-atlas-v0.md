---
title: Partnership failure atlas v0
kind: experiment
first_seen: bc4eb386
last_updated: a2bb0437
status: complete
---

The first build from [[partnership-wall-research]] asks a measurement question
before an architecture question: can the retained artifacts be joined closely
enough to attribute a current-champion failure to a partnership mechanism?

## Method

`w42/partnership_failure_atlas/run_atlas.py` treats
`decision key + candidate domino` as an identity only where uniqueness and set
equality can be proven. It joins the retained action tables from
[[w42-phase3-joined-claim-row-model-table]],
[[w42-phase3-sequence-seat-counterfactuals]], and
[[w42-phase2-seat-position-strategy-map]], plus
[[w42-phase4-sequence-handshape-tests]] and the explicitly confounded prior
attempt at [[w42-bookval-v1-wave1-cross-ai-agreement]]. Relevant [[gus-drama-atlas]],
[[arena]], and [[champion]] artifacts remain in a checksum inventory when their
state namespace or granularity does not support that join.

Every unavailable construct has a blank value and an explicit status. Oracle
values and `eval_only_labels` remain offline outcomes, not runtime features.
The full gzip is deterministic; the validator checks identity uniqueness,
missing-value contracts, row counts, field order, and artifact hashes.
It also re-hashes every manifest input and 114 inventoried live sources so a
stale but internally consistent build fails validation.

## Result

| surface | result |
|---|---:|
| legal-action rows | 75,079 |
| decision states | 28,000 |
| actual-action rows | 28,000 |
| exact sequence/seat joins | 75,079 |
| exact seat-position joins | 75,079 |
| exact handshape joins | 75,079 |
| exact cross-AI prior-attempt joins | 75,079 |
| action-local partnership-proxy rows | 26,310 |
| bidding-risk rows | 7,216 |
| `bidder_lead_plan` proxy rows | 14,257 |
| EV-best / threshold-best disjoint decisions | 2,173 |
| EV-best / safest-tail disjoint decisions | 1,996 |
| confounded dist-lens rows / decisions | 997 / 754 |
| inventoried compatible/incompatible sources | 114 |

The positive result is a reusable role/order action spine. It preserves oracle
mean, regret, threshold and lower-tail summaries; bidder/partner/setter role;
trick order and phase; the observed action; and public, action-local book
detectors.

The handshape join exposes genuine fixed-collapse disagreement: EV-best has no
overlap with threshold-mass-best on 2,173 decisions and with safest-tail on
1,996. This is a useful microgame-selection surface, not evidence for
distributional utility. `is_best_threshold` and `is_safest_tail` are fixed
scalar ranking proxies with no contextual transform or policy consumer. The
field historically named `gus_top_action` is only `is_best_threshold`, not a
Gus output; the atlas renames it. The old dist-lens field remains confounded
because it was merged from another corpus by positional indices without a
shared state identity.

The load-bearing result is an **instrument-sufficiency failure**, not a
partnership null: the atlas contains zero attributable current-champion
failures because the source trajectory policy is not fingerprinted and
Champion trajectories share no canonical action-state key with W42. Full
per-world Q, joinable Gus drama, partner identity, fixed/shuffled assignment,
actor-policy likelihood, action-conditioned posterior change, persistent plan
state, complete auction history, and pre-hand match score are also absent.
Every action row also has `bid_value=30`: declaration and static risk proxies
survive, but the atlas cannot estimate a bid-level or auction-policy effect.

An observed source action with regret therefore cannot be called a champion
error, and a partner-support detector firing cannot be called
[[partnership-value]]. The prior archive localizes role-sensitive situations;
it does not identify the mechanism of the current wall. The stronger claim
that the retained archive could already localize Champion partnership failures
is falsified. Whether partnership failure exists or matters remains untested.

## Decision

No successor architecture is selected. [[world-sampler-mrv-audit]] closes the
exact-fixture sampler-repair gate with completion-count sampling, while CUDA
performance, historical exposure, and C0 reproduction remain open. The
future-Arena state/policy identity, match-score, auction, and static
partner-assignment seams are now closed by [[partnership-decision-record-v1]].
Action likelihood, full uncertainty surfaces,
fixed/shuffled assignment, persistent plan state, and an information-reactive
outcome harness remain open. Only after those randomized arms exist can the
Stage-2 microgames in [[partnership-research-gates]] be populated honestly.

This result is contrary evidence for immediately training a classifier over the
joined table: it could learn role/order and book-proxy action choice, but it
could not be shown to repair the champion or create a partner interaction.

## Artifacts and reproduction

- `w42/partnership_failure_atlas/atlas_full.csv.gz`
- `w42/partnership_failure_atlas/atlas_sample.csv`
- `w42/partnership_failure_atlas/evidence_inventory.csv`
- `w42/partnership_failure_atlas/summary.json`
- `w42/partnership_failure_atlas/manifest.json`

```bash
python w42/partnership_failure_atlas/run_atlas.py
python w42/partnership_failure_atlas/validate_outputs.py
python -m unittest w42.partnership_failure_atlas.test_failure_atlas -v
```

## Links

[[partnership-wall-research]] [[partnership-value]]
[[partnership-research-gates]] [[w42]] [[champion]] [[gus]] [[arena]]
