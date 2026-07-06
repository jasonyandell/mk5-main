---
title: w42 Hidden Threat Legacy Mining
kind: experiment
first_seen: local-2026-05-03
last_updated: afd4802
status: complete
---

## Summary

[[w42]] uses the large legacy [[gus]] corpus to move hidden-threat attribution
from a branch-atlas pilot into a full-corpus diagnostic pass. The run streams
all 100 legacy seed-modulo chunks and emits compact aggregates rather than a
full action table.

This is the direct bead `t42-0b4l.4` result. It connects hidden domino/holder
impact labels to claim-detector surfaces, distribution-aware EV omissions, and
same-decision mitigation contrasts. It does not move central claim statuses.

## Data Slice

| field | value |
|---|---:|
| input files | 100 legacy Gus chunks |
| games | 10000 |
| decisions | 280000 |
| legal-action rows summarized | 748305 |
| declarations | 1000 games per declaration 0-9 |
| scalar-omission decisions | 137029 |
| hidden-downside decisions | 205760 |
| actual action not safest tail | 54778 |

W&B run:
`https://wandb.ai/jasonyandell-forge42/w42/runs/74vfet6o`.

The W&B run logs 20 progress points over `progress/files_processed`, including
decision/action coverage, scalar-omission decisions, hidden-downside decisions,
hidden-impact rates, and max/mean hidden-impact scores.

## Artifacts

| artifact | path |
|---|---|
| script | `w42/hidden_threat_legacy_mining/run_hidden_threat_legacy_mining.py` |
| summary | `w42/hidden_threat_legacy_mining/summary.json` |
| group metrics | `w42/hidden_threat_legacy_mining/group_metrics.csv` |
| hidden driver rollup | `w42/hidden_threat_legacy_mining/hidden_driver_rollup.csv` |
| mitigation contrasts | `w42/hidden_threat_legacy_mining/mitigation_contrasts.csv` |
| branch-impact claim markers | `w42/hidden_threat_legacy_mining/claim_branch_impact_evidence.csv` |
| examples | `w42/hidden_threat_legacy_mining/examples.json` |
| manifest | `w42/hidden_threat_legacy_mining/manifest.json` |

The artifact directory is about 2 MB because it intentionally avoids exporting
the 748305-row action table. Row-level detail is reproducible from the corpus
paths and the script.

## Findings

Hidden-holder impact is widespread in the legacy corpus. Across legal actions,
85.16% have a top hidden-impact score of at least `5.0`; the mean top hidden
impact score is `17.46`, and the max is `85.04`.

The best direct mitigation result is the close-mean contrast:

| contrast | paired decisions | mean delta | lower-tail delta | hidden-downside delta |
|---|---:|---:|---:|---:|
| low hidden-downside vs high hidden-downside, close mean | 4619 | -0.0214 | -0.0332 | -7.4919 |

The 95% CI for the mean delta crosses zero (`[-0.0536, 0.0093]`), while the
lower-tail and hidden-downside reductions are clearly negative. This is the
first strong W42 evidence for a claim-independent tactical idea: some actions
can reduce catastrophic hidden-holder branches while preserving scalar mean.

Other contrasts show the tradeoff surface:

| contrast | paired decisions | mean delta | threshold delta | lower-tail delta | hidden-downside delta |
|---|---:|---:|---:|---:|---:|
| safest tail vs top mean | 42536 | -2.5137 | -0.0307 | -0.0105 | -0.0727 |
| top threshold vs top mean | 40435 | -1.7646 | +0.0118 | +0.0191 | +0.2278 |
| actual vs safest tail | 42624 | +1.6856 | +0.0413 | +0.0221 | +0.3734 |

The selector often accepts lower-tail and hidden-downside exposure in exchange
for mean and threshold mass. That is not automatically wrong; it is the
distribution-aware surface the book-claim tests need to preserve instead of
hiding behind scalar E[Q].

## Driver Rollup

The hidden driver rollup is not a book-claim proof, but it names where branch
impact concentrates. Examples include called-suit doubles in matching pip
declarations, no-trump high count holders, and partner/defender ownership of
specific count-heavy dominoes. These rows are useful as belief-quality and
trace-review targets because they ask whether a public-state model attends to
the hidden facts that actually move the outcome branch.

## Claim Routing Impact

This pass gives branch-impact evidence to multiple downstream claim families:

- setter pounce and count pressure can now be sliced by hidden-downside exposure;
- partner donation and forcedness can separate mean value from disaster-tail
  mitigation;
- doubles/no-trump and no-trump control labels can ask whether support doubles
  reduce branch risk or merely improve mean;
- 84 and bid-margin claims still need generated regimes, but the same hidden
  impact metrics are ready to attach when those regimes exist.

The marker table
`w42/hidden_threat_legacy_mining/claim_branch_impact_evidence.csv` attaches
these evidence surfaces to the relevant claim IDs and families. It is a routing
artifact only.

No central claim ledger status changes. The result is diagnostic and routing
evidence.

## Leakage Boundary

`world_hands`, `q_per_world`, E[Q], hidden holders, hidden-impact scores, and
future branch outcomes are offline labels. They may train or evaluate learned
belief/threat targets. They must not become live hidden-truth policy features.

## Commands

```bash
python -m py_compile w42/hidden_threat_legacy_mining/run_hidden_threat_legacy_mining.py

python w42/hidden_threat_legacy_mining/run_hidden_threat_legacy_mining.py \
  --max-files 1 \
  --output-dir w42/hidden_threat_legacy_mining_smoke \
  --wandb-mode disabled \
  --bootstrap-samples 100 \
  --example-limit 6

python w42/hidden_threat_legacy_mining/run_hidden_threat_legacy_mining.py \
  --output-dir w42/hidden_threat_legacy_mining \
  --bootstrap-samples 1000 \
  --wandb-mode online \
  --wandb-group w42-hidden-threat-legacy-mining \
  --wandb-name t42-0b4l.4-hidden-threat-legacy-v0 \
  --log-every-files 5
```

Validation:

```bash
python - <<'PY'
import csv, json
base = "w42/hidden_threat_legacy_mining"
summary = json.load(open(base + "/summary.json"))
assert summary["coverage"]["input_files"] == 100
assert summary["coverage"]["decision_rows"] == 280000
assert summary["coverage"]["action_rows"] == 748305
rows = list(csv.DictReader(open(base + "/mitigation_contrasts.csv")))
assert len(rows) == 4
markers = list(csv.DictReader(open(base + "/claim_branch_impact_evidence.csv")))
assert len(markers) == 8
assert any(
    row["contrast_id"] == "low_hidden_downside_vs_high_hidden_downside_close_mean"
    and int(row["paired_decision_n"]) > 4000
    for row in rows
)
print("validated hidden threat legacy mining")
PY
```

## Links

[[w42]] | [[w42-claim-data-inventory]] |
[[w42-phase2-hidden-domino-threat-attribution]] |
[[w42-phase2-distribution-aware-ev-report]] |
[[w42-powered-branch-atlas-v1]] | [[w42-branch-atlas-scaled-v0]]
