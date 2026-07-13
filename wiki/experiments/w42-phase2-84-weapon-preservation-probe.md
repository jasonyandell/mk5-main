---
title: w42 Phase2 84 Weapon Preservation Probe
kind: experiment
first_seen: local-2026-05-02
last_updated: afd4802
status: superseded
---

**Superseded** by [[w42-phase4-84-dynamic-seed-tests]], which reruns this
page's dynamic branch lab over 24 mined seeds (672 decisions) rather than 6
hand-picked fixtures — resolving the "hand-picked, not ecological" blocker
this page names, though final-set attribution and unreached last-trick
tableaux remain open in both pages.

## Summary

This page started as the first phase-2 pass for bead `t42-5m82.6`: a targeted
[[w42]] probe for 84 weapon preservation. It now also records the dynamic
follow-up for bead `t42-0b4l.7`.

The original result was a direct-label and fixture surface, not a make/set
verdict. The `t42-0b4l.7` follow-up adds generated 84 contract hands, schema-v2
E[Q] PDFs, branch-atlas hidden-threat rows, and proxy preserve/spend contrasts.
It is still not powered broad-corpus proof, but it is no longer static-only.

## Question

When an 84 hand is branch-shaped, which dominoes function as weapons or
stoppers, when should they be preserved or abandoned, and how should w42 measure
the difference between a voluntary weapon break and a forced loss?

This is deliberately distribution-aware. In 84, scalar E[Q] can be a poor
summary because one trick flips the contract. The important measurements are
make/set threshold mass, lower-tail risk, forcedness, and hidden-domino threat
attribution.

## Starting Evidence

The prior 84 validation bead produced exact static evidence:

| prior result | value | source |
|---|---:|---|
| protected one-off candidate shapes | 1.589% of hand/declaration pairs | `w42/eighty_four_claim_validation/summary.json` |
| straight one-off candidate shapes | 0.585% | `w42/eighty_four_claim_validation/summary.json` |
| three-trump / three-double / one-off bucket | 0.887% | `w42/eighty_four_claim_validation/summary.json` |
| two-off same-suit with protector bucket | 2.508% | `w42/eighty_four_claim_validation/summary.json` |
| opponent team owns a named missing matching double | 66.667% | `w42/eighty_four_claim_validation/summary.json` |
| opponent team owns at least one of two matching doubles | 90.000% | `w42/eighty_four_claim_validation/summary.json` |

Those are shape and ownership facts, not play-quality facts. They do not yet say
whether preserving, spending, or abandoning a weapon was correct in a legal
action state.

## Label Surface

The generated label spec is:

| label | scope | live-safe? | current status |
|---|---|---|---|
| `is_84_contract_decision` | state | yes | source-backed; needs engine fixture |
| `bidder_final_off_candidate` | bidder hand / report | own-hand for bidder; hidden for defenders | static shapes counted |
| `defender_live_double_weapon` | defender action | own-hand plus public history | static inventory proxy exists |
| `defender_live_same_suit_pair` | defender hand / action | own-hand plus public history | static pair availability counted |
| `pair_protector` | action-local | yes for holder | designed, not measured |
| `preservation_opportunity` | legal action set | yes | requires replay/action labels |
| `forced_spend` | legal action set | yes | requires engine legal-action replay |
| `voluntary_weapon_break` | action-local | yes | requires E[Q] or oracle counterfactual labels |
| `dead_asset_abandonment` | action-local | yes | trigger table exists; replay labels missing |
| `hidden_weapon_attribution` | offline evaluation | no live feature | needs saved-world E[Q] artifact |

The key split is forcedness. Breaking a weapon is only an error if a preserving
legal alternative existed.

## Fixture Cases

The artifact `fixture_cases.csv` defines five starting fixtures:

| fixture | purpose | status |
|---|---|---|
| `protected_one_off_bidder_shape` | protected final off where defenders may need same-suit pair pressure | static shape only |
| `straight_off_named_double_threat` | straight off where a named matching double is a high-impact hidden threat | ownership odds counted |
| `two_off_same_suit_ordering` | final-off ordering branch for two offs in one suit | static bucket counted |
| `defender_pair_protector_choice` | free-discard choice between protector and lower-value signal | designed fixture |
| `dead_double_release` | formerly live double whose target branch is public-dead | trigger designed |

These fixtures are meant to become deterministic tests before any model training.

## Measurement Axes

| axis | primary metric | reason |
|---|---|---|
| scalar value | paired mean E[Q] delta | useful baseline |
| threshold mass | `make84_mass_delta` / `set84_mass_delta` | 84 is contract-threshold dominated |
| tail risk | lower-tail mass or CVaR-style branch score | preservation often protects against disaster |
| hidden threat attribution | impact-weighted belief calibration | not all unknown dominoes matter equally |
| forcedness | voluntary break regret excluding forced spend | avoids punishing unavoidable losses |

## Leakage Boundary

Hidden ownership is allowed only as offline supervision, attribution, or
evaluation. A live model may use legal public state and learned beliefs, but not
the hidden truth. The probe should report hidden-weapon attribution separately
from live feature feasibility.

## Dynamic Branch Lab

The `t42-0b4l.7` dynamic run uses explicit hands instead of waiting for lucky
seed matches. It covers six fixture deals: a six-trump laydown control,
protected one-off `4-3`, straight-off named `4-4` stopper threat, two-off
same-suit ordering, defender same-suit pair/protector pressure, and a dead-asset
release control.

| measure | value |
|---|---:|
| fixture games | 6 |
| generated decision states | 168 |
| legal action rows | 584 |
| hidden-threat attribution rows | 4497 |
| dynamic proxy labels | 10 |
| preserve-vs-spend paired decisions | 13 |
| offense trump-vs-final-off paired decisions | 23 |

Headline proxy labels:

| label | actions | decision states | note |
|---|---:|---:|---|
| `laydown_84_proof_fixture` | 56 | 14 | six-trump laydown control only |
| `defender_live_double_weapon_proxy` | 10 | 10 | live double can beat a bidder off under full-deal reconstruction |
| `defender_live_same_suit_pair_proxy` | 21 | 9 | same-suit pair can beat a bidder final-off candidate |
| `pair_protector_proxy` | 4 | 3 | thin but present in generated play |
| `dead_asset_release_candidate_proxy` | 7 | 5 | release candidates from the dead-asset control |

The first preserve/spend contrast is deliberately conservative. Across 13
same-decision defender pairs, the lower-asset preserving alternative beats the
best live-asset spend by only `+0.136` mean Q, with `+0.005` threshold-mass delta
and `-0.0048` lower-tail-mass delta. That is useful because it says the fixture
lab can find the choice surface, not because it proves the book rule.

The offense comparison is clearer but still fixture-limited: across 23
same-decision pairs, trump-pull candidates beat final-off candidates by about
`+4.01` mean Q with nearly unchanged threshold/tail mass. That matches the book's
trumps-first shape on this constructed slice, but it is not yet a powered
protected-off make-rate estimate.

W&B run: [f7uzoo7f](https://wandb.ai/jasonyandell-forge42/w42/runs/f7uzoo7f).

## Claim Ledger

No central claim-ledger status changes. This bead sharpens labels and produces
the first dynamic examples for claims that remain underpowered or
context-limited in [[w42-84-claim-validation]].

| claim area | current reading |
|---|---|
| protected-off 84 | static shapes exist; make/set rollout still missing |
| straight-off 84 | ownership odds are exact for named doubles; carry-to-last-trick is unmeasured |
| defender same-suit pair | static pair availability counted; final-two-trick proof missing |
| abandonment | trigger table exists; replay/action regret labels missing |
| throwaway ladder | designed but not action-labeled |

The dynamic branch lab adds three blockers before any promotion:

- the six deals are hand-picked fixtures, not an ecological corpus;
- reached decisions depend on the current greedy E[Q] policy trace, so unreached
  last-trick tableaux still need state injection or seed mining;
- live/spend labels are asset-priority proxies, not final set-attribution proofs.

The phase-3 follow-up [[w42-phase3-84-seed-mining-corpus]] resolves the
"hand-picked fixtures only" blocker by mining 50000 generated seeds. It emits
214229 candidate 84 rows and 256 recommended natural seeds spanning protected
one-offs, straight one-offs, two-off same-suit structures, defender live doubles,
same-suit pairs, pair protectors, and dead-asset release controls. This is a
documented seed corpus, not an action-value proof; preserve/spend regret still
needs branch-atlas generation or true late-state injection.

## Artifacts

| artifact | path |
|---|---|
| build script | `w42/eighty_four_weapon_preservation_probe/build_probe.py` |
| label spec | `w42/eighty_four_weapon_preservation_probe/label_spec.csv` |
| fixture cases | `w42/eighty_four_weapon_preservation_probe/fixture_cases.csv` |
| measurement axes | `w42/eighty_four_weapon_preservation_probe/measurement_axes.csv` |
| summary | `w42/eighty_four_weapon_preservation_probe/summary.json` |
| manifest | `w42/eighty_four_weapon_preservation_probe/manifest.json` |
| dynamic runner | `w42/eighty_four_weapon_preservation_probe/run_dynamic_branch_lab.py` |
| dynamic summary | `w42/eighty_four_weapon_preservation_probe/dynamic_branch_lab/summary.json` |
| dynamic action labels | `w42/eighty_four_weapon_preservation_probe/dynamic_branch_lab/dynamic_84_action_labels.csv` |
| dynamic paired contrasts | `w42/eighty_four_weapon_preservation_probe/dynamic_branch_lab/dynamic_84_paired_contrasts.csv` |
| dynamic branch atlas | `w42/eighty_four_weapon_preservation_probe/dynamic_branch_lab/branch_atlas/` |
| phase-3 seed-mining corpus | `w42/eighty_four_seed_mining/` |

W&B for the original spec pass: not applicable. W&B for the dynamic branch lab:
`f7uzoo7f`.

HF: not applicable. This is a local report/spec surface, not a publishable
dataset or checkpoint.

## Provenance

| field | value |
|---|---|
| bead | `t42-5m82.6` |
| command | `python w42/eighty_four_weapon_preservation_probe/build_probe.py` |
| dynamic bead | `t42-0b4l.7` |
| dynamic command | `.venv/bin/python w42/eighty_four_weapon_preservation_probe/run_dynamic_branch_lab.py --samples 256 --wandb-mode online` |
| source artifacts | `w42/eighty_four_claim_validation/summary.json`; `example_cases.csv`; `abandonment_trigger_table.csv` |
| source pages | [[winning42-ch07-taking-every-trick-84]]; [[winning42-ch08-setting-84]]; [[w42-84-claim-validation]] |
| random seeds | fixture ids `840700` through `840705` |
| W&B | `f7uzoo7f` for the dynamic branch lab |
| HF | not applicable |

## Links

[[w42]] | [[w42-final-empirical-strategy-report]] |
[[w42-next-model-decision]] | [[w42-84-claim-validation]] |
[[winning42-ch07-taking-every-trick-84]] |
[[winning42-ch08-setting-84]]
