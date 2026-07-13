---
title: w42 84 Claim Validation
kind: experiment
first_seen: 2026-05-02
last_updated: 2026-07-13
status: complete
---

**Complete for the static-math scope it covers** (exact double-six
enumeration and hypergeometric proxies are final). The dynamic side this page
explicitly says it cannot measure continued on three later pages — see
"Where This Stands" below.

## Where This Stands

This page froze its claim-ledger rows at
`underpowered`/`context-limited`/`not-yet-tested` on 2026-05-03. The 84 story
then moved in three steps:

- [[w42-phase2-84-weapon-preservation-probe]] built the first dynamic branch
  lab: six hand-picked bid-84 fixtures with schema-v2 E[Q] PDFs. First proxy
  contrasts — preserve over spend `+0.136` mean Q (13 pairs, deliberately
  thin), offense trump-pull over final-off `+4.01` (23 pairs). Fixture
  evidence only; no ledger promotion.
- [[w42-phase3-84-seed-mining-corpus]] replaced hand-picked fixtures with a
  mined natural-seed menu: 50,000 seeds scanned, 214,229 candidate 84 rows,
  256 recommended seeds. A powered data route, not an action-value proof.
- [[w42-phase4-84-dynamic-seed-tests]] ran 24 mined natural seed games (672
  decisions): preserving expendable/dead assets instead of spending live
  weapons or protectors is `+1.946` mean Q (62 pairs), dead-asset release
  `+0.504` (29 pairs), bidder trump-pull before final-off `+4.295` (76 pairs).

The verdict those pages support: 84 preservation is a real action surface —
on reached natural seed states, preserving live weapons/protectors beats
spending them, dead-asset release has positive value, and trump-pull before
final-off is strongly positive ([[w42-book-claim-synthesis-and-ai-directions]]
carries this as the campaign-level reading). The evidence remains policy-trace
evidence, not arbitrary late-state injection: final set attribution, full
throwaway-ladder bottlenecks, score 42-vs-84 terminal counterfactuals, and the
straight-off "set by good players" population claim stayed blocked in every
page. This page's frozen rows were never individually reconciled against that
later evidence; [[w42-phase4-final-claim-audit]] holds the 64-row closure
state.

## Summary

[[w42]] now has a report-local validation pass for Winning 42 Chapter 7/8 84
claims.

- report owner bead: `t42-csw6.21`
- report artifacts: `w42/eighty_four_claim_validation/`
- evidence mode: exact double-six static enumeration plus hypergeometric defender
  asset proxies
- primary W&B run:
  `https://wandb.ai/jasonyandell-forge42/w42/runs/vmta9zew`

The report validates what the current artifacts can count: bidder 84 shape buckets,
straight-off matching-double ownership odds, defender last-trick weapon inventory, and
same-suit pair availability. It does not validate make/set rates, oracle regret, score
gates, legal throwaway decisions, or dynamic abandonment quality.

Follow-up: [[w42-phase2-84-weapon-preservation-probe]] now includes a
`t42-0b4l.7` dynamic branch lab. That lab generates six explicit bid-84 hands
with schema-v2 E[Q] PDFs and branch-atlas hidden-threat rows, producing first
proxy labels for live doubles, same-suit pairs, pair protectors, dead-asset
release, and preserve/spend contrasts. It narrows the dynamic-data blocker but
does not promote the static claim statuses on this page.

## Key Question

Which bidder-side and defender-side 84 claims from [[winning42-ch07-taking-every-trick-84]]
and [[winning42-ch08-setting-84]] can be checked with current report-local artifacts,
and which still need detector implementation, replay labels, or oracle rollouts?

## Method

| field | value |
|---|---|
| report owner bead | `t42-csw6.21` |
| ruleset / score mode | straight 42, pip-trump declarations, 84 contract semantics from chapter/ruleset pages |
| evidence mode | exact double-six combinatorics plus report-only hypergeometric defender asset proxies |
| decision slice | 84 bidder and defender report buckets |
| train/eval split policy | not applicable |
| statistical test | exact counts and exact hypergeometric probabilities; no random sampling |
| success criterion | conservative claim-status table plus explicit cannot-measure list |

The analysis enumerates all `28 choose 7` bidder hands under seven pip-trump
declarations: 8,288,280 hand/declaration pairs. It distinguishes trump tiles,
non-trump doubles, and non-double offs because Chapter 7's "one off" 84 language treats
doubles as forcing/protection assets rather than ordinary offs.

## Data Manifest

| field | value |
|---|---|
| manifest path | `w42/eighty_four_claim_validation/summary.json` |
| dataset id/version | `w42-eighty-four-static-claim-validation` / `v1` |
| source corpora | not applicable |
| source wiki pages | [[winning42-ch07-taking-every-trick-84]], [[winning42-ch08-setting-84]], [[winning42-ch12-advanced-bidding-playing]], [[winning42-ch13-optional-variations]], [[w42-strategy-tags-v1-map]], [[w42-detector-tests]], [[w42-concept-bucket-regret]], [[w42-claim-ledger]] |
| generated by | `python w42/eighty_four_claim_validation/validate_84_claims.py --wandb-mode online --wandb-name t42-csw6.21-84-validation-corrected-s0042-f746b93` |
| split policy | not applicable |
| leakage exclusions checked | no hidden-owner live features were emitted; hidden ownership appears only as report-time probability/proxy |
| random seeds | analysis seed `42`; no sampling |

## Feature / Detector Set

| detector | scope | public-state safe? | concept bucket | source claim |
|---|---|---|---|---|
| `84_contract_regime` | report/rules context | yes | 84 | Ch07 all-seven-tricks contract; Ch13 high-bid scoring |
| `protected_one_off_84_shape` | opening-hand report bucket | own-hand safe | bidder 84 | Ch07 protected-off 84 shape |
| `straight_off_84_risk` | opening-hand / odds bucket | own-hand plus public prior | bidder 84 / odds | Ch07 straight-off risk |
| `three_trump_three_double_84` | opening-hand report bucket | own-hand safe | bidder 84 | Ch07 non-four-trump candidate |
| `two_off_same_suit_84` | opening-hand report bucket | own-hand safe | bidder 84 | Ch07 two-off ordering problem |
| `live_final_weapon_tiles` | defender report proxy | report-only for hidden owners | defender 84 | Ch08 last-trick weapons |
| `live_same_suit_pair` | defender report proxy | report-only for hidden owners | defender 84 | Ch08 double-ahead-off defense |
| `throwaway_ladder_rank` | designed only | own-action safe after implementation | defender 84 | Ch08 discard priority ladder |
| `84_set_attribution` | designed only | report-only | defender 84 | Ch08 set causes |

## Findings

| finding | evidence | metric | status |
|---|---|---:|---|
| Canonical one-off 84 candidate shapes are countable but rare under ungated all-hand enumeration. | `bidder_shape_summary.csv` | 180,180 / 8,288,280 = 2.174% | underpowered |
| Protected one-off shapes outnumber straight one-off shapes in the static candidate pool. | `bidder_shape_summary.csv` | protected 1.589%; straight 0.585% | underpowered |
| The three-trump / three-non-trump-double / one-off bucket is nonempty and exactly measurable. | `bidder_shape_summary.csv` | 73,500 / 8,288,280 = 0.887% | underpowered |
| Two-off same-suit shapes with a shared protecting double are measurable as an ordering bucket. | `bidder_shape_summary.csv` | 207,900 / 8,288,280 = 2.508% | underpowered |
| The book's two-to-one missing-double ownership claim is exact for one named matching double. | `defender_last_trick_weapon_distribution.csv` | opponent team owns a named missing double with probability 14/21 = 66.667% | context-limited |
| If either of two matching doubles can set a straight off, the opponent-team ownership probability is not two-to-one. | `defender_last_trick_weapon_distribution.csv` | at least one of two missing doubles: 90.000% | context-limited |
| Static defender last-trick weapon pools usually yield one to four weapons for a single defender, but not always. | `defender_last_trick_weapon_distribution.csv` | one-to-four probability ranges 56.667% to 95.320% by final-off pool size | context-limited |
| Against a protected off, a single defender starts with at least two same-suit pair tiles in 40.702% of static protected-off cases. | `defender_same_suit_pair_distribution.csv` | pool size 4; P(K>=2 from 7 of 21) = 40.702% | underpowered |
| Current raw/v0 model artifacts still cannot isolate 84 regret. | [[w42-concept-bucket-regret]] | 84 bucket listed as missing detector | not-yet-tested |

## Bidder Buckets

| bucket | n | percent of hand/declaration pairs | report interpretation |
|---|---:|---:|---|
| all hand/declaration pairs | 8,288,280 | 100.000% | denominator |
| one non-double off candidate shape | 180,180 | 2.174% | Chapter 7 canonical 84 search space |
| protected one off | 131,670 | 1.589% | bidder has a same-suit double ahead of the final off |
| straight one off | 48,510 | 0.585% | unprotected final-off risk bucket |
| three trump / at least three non-trump doubles / one off | 73,500 | 0.887% | non-four-trump candidate bucket |
| two offs same suit with shared protecting double | 207,900 | 2.508% | two-off ordering bucket |

These are shape and prior buckets only. They do not say the bid is good, made, set,
or score-correct; that needs [[forge]] oracle rollouts or match utility simulation.

## Defender Buckets

| asset pool | final-off cases | mean weapons for one defender | P(0) | P(1-4) | P(>4) |
|---|---:|---:|---:|---:|---:|
| 2 | 7 | 0.667 | 43.333% | 56.667% | 0.000% |
| 3 | 7 | 1.000 | 27.368% | 72.632% | 0.000% |
| 4 | 14 | 1.333 | 16.725% | 83.275% | 0.000% |
| 5 | 14 | 1.667 | 9.838% | 90.058% | 0.103% |
| 6 | 21 | 2.000 | 5.534% | 93.911% | 0.555% |
| 7 | 14 | 2.333 | 2.951% | 95.320% | 1.729% |
| 8 | 14 | 2.667 | 1.476% | 94.448% | 4.076% |
| 9 | 7 | 3.000 | 0.681% | 91.269% | 8.050% |
| 10 | 7 | 3.333 | 0.284% | 85.707% | 14.009% |

The "one to four possible last-trick weapons" claim is directionally supported as a
static inventory description but remains context-limited: high-pool final offs can
give a single defender more than four weapons, and low-pool final offs often give a
defender zero weapons.

## Last-Trick Weapons

The report separates three ideas that the prose can blur:

- A named missing matching double is exactly a two-to-one opponent-team ownership event
  because 14 of 21 unknown slots belong to opponents.
- A straight off with two relevant missing matching doubles is a 90.000% opponent-team
  ownership event for at least one matching double.
- Carrying a setting tile to the last trick is not an ownership event. It needs legal
  follow pressure, forced-break labels, partner interaction, and possibly oracle action
  values.

## Abandonment Triggers

| trigger | asset before | abandon when | current status |
|---|---|---|---|
| watched final-off suit exhausted | live double or same-suit tile that only beats a watched final off | all higher same-suit target tiles are public-played, held by bidder, or no longer plausible final offs | designed, not replay-measured |
| bidder double-ahead protection revealed/proved | single saved double in the protected suit | a lone double can no longer beat the bidder's final off because the bidder can force the suit with the double ahead | static proxy only |
| same-suit pair target killed | two-tile same-suit pair plus optional protectors | the lower/upper target role is impossible because target tiles are public-dead or the pair was forced apart | detector map exists; no replay labels |
| protector spent or no longer needed | side-suit protector guarding a live pair from forced follow | the pair is dead, the watched lead suit is exhausted, or legal follow pressure can no longer force the pair tile | not measured |
| partner kills target branch | defender asset kept for a final-off branch partner can publicly eliminate | partner play/discard makes the bidder final-off branch impossible under public evidence | requires belief/replay |

## What Current Artifacts Cannot Measure

- Make/set rates for protected-off, straight-off, three-trump, and two-off 84 hands.
- Regret or terminal win-rate for bidding 42 versus 84 near 250.
- Whether trumps-first / doubles-second / final-off-last sequencing is oracle-optimal.
- Whether a defender voluntarily broke a live pair or was forced to follow.
- Whether a throwaway ladder violation cost E[Q].
- Whether a set came from preserved double, preserved pair, partner rescue, bidder error,
  or accidental survival.
- Whether [[burl]] recognizes 84 mode or reasons from public evidence in traces.
- Whether [[gus]] belief quality degrades with 84 tracking load.

## Claim-Ledger Impact

| claim id | before | after | reason | evidence artifact |
|---|---|---|---|---|
| `ch07-84-contract-regime` | chapter harvest | context-limited | source/rules-backed, but no score replay in this bead | `claim_summary.csv` |
| `ch07-protected-one-off-84-shape` | chapter harvest | underpowered | shape count exists; no make/set rollout | `bidder_shape_summary.csv` |
| `ch07-straight-off-two-to-one-double` | chapter harvest | context-limited | exact ownership odds for named double; policy set rate untested | `defender_last_trick_weapon_distribution.csv` |
| `ch07-score-42-vs-84-gate` | chapter harvest | not-yet-tested | no score-state oracle or match utility simulation | `claim_summary.csv` |
| `ch08-one-to-four-last-trick-weapons` | chapter harvest | context-limited | static inventory mostly fits but has zero and >4 cases | `defender_last_trick_weapon_distribution.csv` |
| `ch08-double-ahead-needs-same-suit-pair` | chapter harvest | underpowered | same-suit pair availability counted; no final-two-trick proof | `defender_same_suit_pair_distribution.csv` |
| `ch08-abandon-dead-assets` | chapter harvest | not-yet-tested | trigger table only | `abandonment_trigger_table.csv` |
| `ch08-throwaway-priority-ladder` | chapter harvest | not-yet-tested | no legal discard/action regret labels | `claim_summary.csv` |

Claim ledger impact: no central claim-ledger change. The report writes a local delta at
`w42/eighty_four_claim_validation/claim_ledger_delta.json`.

## W&B / HF Links

| system | link |
|---|---|
| W&B primary corrected run | `https://wandb.ai/jasonyandell-forge42/w42/runs/vmta9zew` |
| W&B useful failed run | `https://wandb.ai/jasonyandell-forge42/w42/runs/oizyjty5` |
| W&B superseded completed run | `https://wandb.ai/jasonyandell-forge42/w42/runs/w4wupdih` |
| W&B artifact | `w42-84-claim-validation-f746b93` |
| HF dataset | not applicable |
| HF model | not applicable |
| HF artifact | not applicable |

The useful failed run captured a CSV schema bug after W&B init. The superseded completed
run used the wrong "off" semantics by counting non-trump doubles as ordinary offs; the
corrected run `vmta9zew` is the one this report uses.

## Caveats

- Static enumeration is not game play. It counts shapes and asset pools, not make/set
  results.
- Hypergeometric defender probabilities assume random ownership of the unknown tiles
  after a final-off case; they do not model bidding selection, partner choices, or forced
  follow pressure.
- Bidder candidate counts are over all hands and pip declarations, not filtered by actual
  auction policy or score context.
- The report uses hidden ownership only as report-time evaluation/proxy information, not
  as a live feature.
- No [[forge]] E[Q], [[gus]] model probe, or [[burl]] trace review ran in this bead.

## Artifact Manifest

| artifact | path / id | produced by | durable? |
|---|---|---|---|
| report | `wiki/experiments/w42-84-claim-validation.md` | manual synthesis from generated artifacts | yes |
| validation script | `w42/eighty_four_claim_validation/validate_84_claims.py` | manual w42 script | w42 |
| summary | `w42/eighty_four_claim_validation/summary.json` | validation script | w42 |
| bidder shape table | `w42/eighty_four_claim_validation/bidder_shape_summary.csv` | validation script | w42 |
| trump distribution | `w42/eighty_four_claim_validation/trump_count_distribution.csv` | validation script | w42 |
| example cases | `w42/eighty_four_claim_validation/example_cases.csv` | validation script | w42 |
| defender weapon table | `w42/eighty_four_claim_validation/defender_last_trick_weapon_distribution.csv` | validation script | w42 |
| defender pair table | `w42/eighty_four_claim_validation/defender_same_suit_pair_distribution.csv` | validation script | w42 |
| abandonment trigger table | `w42/eighty_four_claim_validation/abandonment_trigger_table.csv` | validation script | w42 |
| claim summary | `w42/eighty_four_claim_validation/claim_summary.csv` | validation script | w42 |
| claim ledger delta | `w42/eighty_four_claim_validation/claim_ledger_delta.json` | validation script | w42 |

## Exact Commands / Configs / Seeds

```bash
git -C /Users/jason/code/mk5-main worktree add -b w42/csw6-21 /Users/jason/code/mk5-main/.claude/worktrees/w42-csw6-21 forge
sed -n '1,220p' wiki/AGENTS.md
bd show t42-csw6.21
rg --files wiki w42 | rg -i 'w42|winning42|claim|ledger|detector|bucket|84|eighty|chapter|ch7|ch8|template|charter|lab'
sed -n '1,240p' wiki/entities/w42.md
sed -n '1,260p' wiki/experiments/w42-lab-infrastructure.md
sed -n '1,240p' w42/report_template.md
sed -n '1,260p' wiki/experiments/w42-claim-ledger.md
sed -n '1,280p' wiki/experiments/winning42-ch07-taking-every-trick-84.md
sed -n '1,300p' wiki/experiments/winning42-ch08-setting-84.md
sed -n '1,220p' wiki/experiments/w42-strategy-tags-v1-map.md
sed -n '1,220p' wiki/experiments/w42-detector-tests.md
sed -n '1,260p' wiki/experiments/w42-concept-bucket-regret.md
cat w42/concept_bucket_regret_report/manifest.json
sed -n '1,80p' w42/concept_bucket_regret_report/concept_bucket_regret.csv
sed -n '1,240p' w42/strategy_tags_v1_map/detector_map.json
rg -n "84|eighty|last trick|last-trick|protect|abandon|straight off|double-ahead|same-suit|Game bids|126|168" wiki/experiments/winning42-ch12-advanced-bidding-playing.md wiki/experiments/winning42-ch13-optional-variations.md wiki/experiments/winning42-ch14-history-tournaments.md wiki/experiments/winning42-strategy-measurement.md wiki/entities/forge-analysis.md wiki/entities/forge.md wiki/entities/gus.md wiki/experiments/gus-strategy-tags-probe.md
python w42/eighty_four_claim_validation/validate_84_claims.py --wandb-mode online
python w42/eighty_four_claim_validation/validate_84_claims.py --wandb-mode online --wandb-name t42-csw6.21-84-validation-corrected-s0042-f746b93
cat w42/eighty_four_claim_validation/summary.json
sed -n '1,80p' w42/eighty_four_claim_validation/bidder_shape_summary.csv
sed -n '1,80p' w42/eighty_four_claim_validation/defender_last_trick_weapon_distribution.csv
sed -n '1,80p' w42/eighty_four_claim_validation/defender_same_suit_pair_distribution.csv
```

Configs:

- `w42/eighty_four_claim_validation/summary.json`
- W&B entity/project: `jasonyandell-forge42/w42`
- W&B group: `w42-csw6-84-claim-validation`

Seeds:

- data generation: not applicable
- dataset shuffle: not applicable
- train: not applicable
- eval: not applicable
- oracle/world sampling: not applicable
- analysis seed: `42`

Commit SHA at corrected run:

- `f746b93f4ea0bacc697ed27b9dbf2fbd9ef8e744`

## Next Checks

- Implement replay labels for `live_final_weapon_tiles`, `live_same_suit_pair`,
  `pair_protector_status`, and `throwaway_ladder_rank`.
- Run 84-filtered [[forge]] oracle rollouts for protected-off, straight-off, three-trump,
  and two-off buckets.
- Add score-state simulation for `score_42_vs_84_gate`.
- Add [[burl]] trace cases that force public-evidence reasoning about 84 mode, watched
  final suits, and abandonment.

## Links

[[w42]] | [[w42-claim-ledger]] | [[w42-strategy-tags-v1-map]] |
[[w42-detector-tests]] | [[w42-concept-bucket-regret]] |
[[winning42-ch07-taking-every-trick-84]] | [[winning42-ch08-setting-84]]
