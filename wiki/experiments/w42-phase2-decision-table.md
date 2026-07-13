---
title: w42 Phase 2 Decision Table
kind: experiment
first_seen: 2026-05-02
last_updated: 2026-07-06
status: superseded
---

**Superseded.** Every extension this page's Next Steps proposed (bid amount
and margin, joint-world ownership, direct setter-pounce/84-preservation
labels) was delivered by [[w42-powered-branch-atlas-v1]],
[[w42-branch-atlas-scaled-v0]], [[w42-gus-corpus-tactical-claim-deep-dive]],
and [[w42-phase2-84-weapon-preservation-probe]].

## Summary

[[w42]] now has a v0 bridge table for targeted strategy probes:
`w42/phase2_decision_table/decision_actions.csv` and
`w42/phase2_decision_table/decision_states.csv`.

The table joins the local [[eq-browser-visualizers]] sample to seat/role
context, public trick state, actor hand/action facts, distribution-aware E[Q]
labels, and explicit missing columns for direct tactical labels and
hidden-domino threat attribution.

This is not a training run and does not move the claim ledger. It is the first
shared table shape for the next [[w42-next-model-decision]] work package.

## Data Slice

The v0 table read
`forge/analysis/results/data/eq_pdf_v3_sample.jsonl` (a local, untracked
artifact; no longer present in the repo — sha256 recorded in
`w42/phase2_decision_table/manifest.json`).

| artifact | rows | grain |
|---|---:|---|
| `decision_actions.csv` | 346 | one legal action with an E[Q] PDF |
| `decision_states.csv` | 140 | one public decision state |
| `examples.json` | 12 | inspectable branch-shaped decisions |
| `schema.json` | 1 | feature groups and leakage boundary |

All PDF rows in this slice use 1000 samples.

## Schema

The action table separates four feature groups:

| group | examples | live status |
|---|---|---|
| public context | declaration, move/trick position, score before action, current trick winner, led suit, public count already played | online-safe |
| actor private context | actor remaining hand size, count points, called-suit count, doubles | legal for that actor's perspective |
| candidate action | candidate domino, count points, double/called-suit flags, follow/beat/current-trick flags | online-safe |
| offline labels | E[Q] mean/std, threshold mass, quantiles, CVaR, branch peaks, shelf gap, scalar-EV omission flag | training/eval only |

The v0 table also carries explicit missing/offline columns:

- bid amount and bid margin are not present in the visualizer export;
- saved joint-world hidden ownership is not present in this JSONL;
- top hidden threat holder/domino/impact are therefore blank in v0.

## Findings

The bridge table reproduces the earlier distribution-aware signal while adding
role and action context:

| finding | value |
|---|---:|
| decision states | 140 |
| legal action rows | 346 |
| scalar-EV omission decisions | 68 |
| decisions with at least one multi-peak PDF action | 118 |
| high-variance close-mean action rows | 87 |
| multi-peak PDF action rows | 309 |
| wide-shelf action rows | 259 |
| large lower-tail action rows | 177 |

The actual sampled action is the top-mean action in 133 of 140 decision states,
the top visualizer-threshold action in 128 of 140, and the safest lower-tail
action in 114 of 140. That is a useful sanity check: the sample mostly follows
the scalar selector, but the table now preserves where threshold and tail views
disagree.

## Strategy Surface

The detector vocabulary from [[w42-phase2-seat-position-strategy-map]] now has
action-level counts on the visualizer slice:

| detector | action rows |
|---|---:|
| `first_trick_public_belief_update` | 281 |
| `late_trick_threshold_closure` | 101 |
| `last_to_act_closure_policy` | 68 |
| `defender_damage_lead_class` | 46 |
| `bidder_first_lead_plan` | 35 |
| `setter_count_before_certainty` | 30 |
| `partner_forcedness_and_safety` | 13 |
| `partner_third_seat_safe_donation` | 13 |
| `first_setter_pounce_window` | 9 |

These are table tags, not verdicts. They make slices possible: setter pounce,
partner donation, last-to-act closure, late thresholds, and bidder first leads
can now be filtered against the same distribution-aware labels.

## Leakage Boundary

The table keeps online-safe features separate from offline labels.

Online-safe fields include public trick state, declaration, legal action facts,
and the actor's own hand. Offline-only fields include E[Q] PDFs, scalar E[Q],
future outcome ranks, threshold/tail labels, and future hidden-domino threat
attribution.

The v0 hidden-threat columns are intentionally blank because the source JSONL has
already collapsed sampled worlds into PDFs. The next data-generation pass needs
saved joint worlds before [[w42-phase2-hidden-domino-threat-attribution]] can
populate top holder/domino/impact columns.

## Commands

```bash
python w42/phase2_decision_table/build_phase2_decision_table.py \
  --input forge/analysis/results/data/eq_pdf_v3_sample.jsonl

python -m py_compile w42/phase2_decision_table/build_phase2_decision_table.py
```

The validation pass loads both CSVs, all JSON artifacts, and verifies the row
counts recorded in `summary.json`.

## W&B / HF

No W&B run, HuggingFace artifact, model checkpoint, or training dataset was
published. The table is a local research artifact and schema review target.

## Next Steps

The next useful pass is a generated slice that adds:

- bid amount and bid margin;
- joint-world ownership plus per-world outcomes;
- direct setter-pounce labels;
- direct 84 preservation labels;
- belief-impact attribution columns.

Once that exists, w42 can train raw/v0/rich/direct/distribution-aware variants
with W&B series and report tail-risk, threshold-mass, and claim-specific bucket
metrics.

## Links

[[w42]] | [[w42-next-model-decision]] |
[[w42-phase2-seat-position-strategy-map]] |
[[w42-phase2-distribution-aware-ev-report]] |
[[w42-phase2-hidden-domino-threat-attribution]] |
[[eq-browser-visualizers]]
