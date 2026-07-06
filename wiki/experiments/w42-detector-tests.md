---
title: w42 Detector Tests
kind: experiment
first_seen: local-2026-05-02
last_updated: local-2026-05-02
status: active
---

## Summary

[[w42]] now has a deterministic detector check suite for `t42-csw6.9`:

- `w42/detector_tests.py`
- `w42/detector_tests/report.json`
- `w42/detector_tests/checks.csv`
- `w42/detector_tests/coverage.csv`

The suite validates existing [[w42-strategy-tags-v0]] semantics against literal
public-state fixtures and checks that the [[w42-strategy-tags-v1-map]] detector
enumeration remains internally covered. It does not invent new detector semantics,
run training, run W&B, publish HF artifacts, modify [[gus]] core paths, modify
[[burl]], or modify [[forge]] oracle behavior.

## Scope

The executable fixture checks cover the current cheap public/action detector surface:

| family | checks | positive | negative |
|---|---:|---:|---:|
| schema | 2 | 2 | 0 |
| rule_variant_gate | 2 | 1 | 1 |
| legal_action_summary | 4 | 2 | 2 |
| effective_walker_or_promoted_tile | 1 | 1 | 0 |
| safe_partner_count_donation | 6 | 2 | 4 |
| doubles_native_suit_removal | 4 | 3 | 1 |
| setter_pounce_window | 3 | 2 | 1 |
| no_trump_support_double_preservation | 2 | 0 | 2 |
| count_protection_throwaway | 2 | 1 | 1 |

The v1 map checks cover all 8 buckets and all 48 mapped detector names, including
all 12 high-priority detector families named for first implementation order:
`rule_variant_gate`, `score_mode_objective`, `84_contract_regime`,
`side_specific_off_risk`, `candidate_bid_loss_budget`, `unnecessary_bid_margin`,
`safe_partner_count_donation`, `setter_pounce_window`,
`count_protection_throwaway`, `final_walker_counter`,
`doubles_native_suit_removal`, and `no_trump_support_double_preservation`.

## Results

| metric | value |
|---|---:|
| total checks | 53 |
| passed | 53 |
| failed | 0 |
| fixture semantic checks | 26 |
| v1 map checks | 27 |
| v0 global tags checked for schema width | 68 |
| v0 action tags checked for schema width | 32 |
| v1 buckets covered | 8 |
| v1 detector names covered | 48 |
| v1 high-priority detector names covered | 12 |

The fixture cases include positive and negative examples for declaration/ruleset
boundary tags, legal follow restriction, partner donation windows, opponent/pounce
windows, no-trump versus doubles-trump double handling, and count-protection proxies.

## Enumeration Boundary

True engine enumeration was not run. The existing v1 artifact is a chapter-derived
design map, not an engine-integrated detector implementation, so full game-tree
enumeration would be too opaque for this bead and would risk inventing semantics
outside the current detector surface. This bead instead records a reproducible
lightweight suite: literal fixture hands plus v1 map coverage checks.

Future promotion beads can replace or extend this with exact enumeration once the
v1 detector predicates are implemented as callable code.

## Artifacts

| artifact | purpose |
|---|---|
| `w42/detector_tests.py` | deterministic fixture and map check runner |
| `w42/detector_tests/report.json` | machine-readable run summary and provenance |
| `w42/detector_tests/checks.csv` | one row per assertion |
| `w42/detector_tests/coverage.csv` | v1 bucket and detector coverage report |

Data inputs:

- `w42/strategy_tags_v0.py`
- `w42/strategy_tags_v1_map/detector_map.json`
- literal fixture hands in `w42/detector_tests.py`

## Reproducibility

Run commit at artifact generation:
`c603a0d9b19374414753e0953ca1535455f0d0a6`.

Exact detector command:

```bash
python w42/detector_tests.py
```

Exact checks run:

```bash
git status --short --branch
bd show t42-csw6.9
sed -n '1,220p' wiki/AGENTS.md
sed -n '1,260p' wiki/experiments/w42-strategy-tags-v0.md
sed -n '1,260p' wiki/experiments/w42-strategy-tags-v1-map.md
sed -n '1,320p' w42/strategy_tags_v0.py
sed -n '320,760p' w42/strategy_tags_v0.py
sed -n '1,260p' w42/strategy_tags_v1_map/detector_map.json
sed -n '1,240p' w42/strategy_tags_v1_map/README.md
sed -n '1,280p' gus/model/strategy_features.py
sed -n '280,620p' gus/model/strategy_features.py
python w42/detector_tests.py
```

Config:

| key | value |
|---|---|
| bead | `t42-csw6.9` |
| device | `cpu` |
| source mode | literal public-state fixtures plus v1 detector-map coverage |
| data input | `w42/strategy_tags_v0.py`; `w42/strategy_tags_v1_map/detector_map.json`; literal fixtures |
| output directory | `w42/detector_tests/` |
| checkpoint | `not applicable` |
| W&B links | `not applicable` |
| HF links | `not applicable` |
| claim ledger impact | `no claim-ledger change` |

Random seeds:

| seed | value |
|---|---:|
| torch | `42` |
| fixture generation | `not applicable` |
| engine enumeration | `not applicable` |
| train loader | `not applicable` |
| world sampling | `not applicable` |
| eval sampling | `not applicable` |

## Claim Ledger

no claim-ledger change

The suite supports detector surface correctness for the current v0 fixtures and
v1 map coverage only. It does not support or contradict any Winning 42 strategy
claim, because no enumeration, oracle rollout, Gus/w42 probe, or Burl trace review
was run.

## Links

[[w42]] | [[w42-strategy-tags-v0]] | [[w42-strategy-tags-v1-map]] |
[[w42-claim-ledger]] | [[winning42-strategy-measurement]] | [[gus]]
