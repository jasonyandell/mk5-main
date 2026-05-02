---
title: w42 Phase2 84 Weapon Preservation Probe
kind: experiment
first_seen: local-2026-05-02
last_updated: local-2026-05-02
status: active
---

## Summary

This page completes the first phase-2 pass for bead `t42-5m82.6`: a targeted
[[w42]] probe for 84 weapon preservation.

The result is not a new make/set verdict. It is a direct-label and fixture
surface for the next empirical run. Prior [[w42-84-claim-validation]] artifacts
already count static 84 shapes and defender weapon pools; this bead turns that
into labels for preservation, forced spending, dead-asset abandonment, and
hidden-weapon attribution.

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

## Next Dynamic Run

The smallest useful next run is not a broad model. It is a constructed or
filtered 84 slice with:

- legal action states for defender free-discard decisions;
- current contract, bidder seat, trick index, and public play history;
- per-action E[Q] PDF or sampled-world labels where feasible;
- hidden-world ownership only in the label/evaluation payload;
- forced-spend versus voluntary-break annotation.

Success is a report that can say: this action spent a live weapon, a preserving
legal alternative existed, and the preserve/spend choice changed threshold mass
or tail risk.

## Claim Ledger

No central claim-ledger status changes. This bead sharpens labels for claims that
remain underpowered or context-limited in [[w42-84-claim-validation]].

| claim area | current reading |
|---|---|
| protected-off 84 | static shapes exist; make/set rollout still missing |
| straight-off 84 | ownership odds are exact for named doubles; carry-to-last-trick is unmeasured |
| defender same-suit pair | static pair availability counted; final-two-trick proof missing |
| abandonment | trigger table exists; replay/action regret labels missing |
| throwaway ladder | designed but not action-labeled |

## Artifacts

| artifact | path |
|---|---|
| build script | `w42/eighty_four_weapon_preservation_probe/build_probe.py` |
| label spec | `w42/eighty_four_weapon_preservation_probe/label_spec.csv` |
| fixture cases | `w42/eighty_four_weapon_preservation_probe/fixture_cases.csv` |
| measurement axes | `w42/eighty_four_weapon_preservation_probe/measurement_axes.csv` |
| summary | `w42/eighty_four_weapon_preservation_probe/summary.json` |
| manifest | `w42/eighty_four_weapon_preservation_probe/manifest.json` |

W&B: not applicable. No training or long iterative run occurred.

HF: not applicable. This is a local report/spec surface, not a publishable
dataset or checkpoint.

## Provenance

| field | value |
|---|---|
| bead | `t42-5m82.6` |
| command | `python w42/eighty_four_weapon_preservation_probe/build_probe.py` |
| source artifacts | `w42/eighty_four_claim_validation/summary.json`; `example_cases.csv`; `abandonment_trigger_table.csv` |
| source pages | [[winning42-ch07-taking-every-trick-84]]; [[winning42-ch08-setting-84]]; [[w42-84-claim-validation]] |
| random seeds | not applicable |
| W&B | not applicable |
| HF | not applicable |

## Links

[[w42]] | [[w42-final-empirical-strategy-report]] |
[[w42-next-model-decision]] | [[w42-84-claim-validation]] |
[[winning42-ch07-taking-every-trick-84]] |
[[winning42-ch08-setting-84]]
