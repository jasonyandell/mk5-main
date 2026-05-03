---
title: w42 Claim Data Inventory
kind: experiment
first_seen: local-2026-05-03
last_updated: local-2026-05-03
status: active
---

## Summary

[[w42]] now has a compact inventory of the local [[gus]] corpora that can be
used for Winning 42 book-claim recovery. The inventory was created after the
session recovery found that the large corpus was not under `scratch/`, but under
`gus/data/`.

The full pass inspected one payload at a time. It did not copy `.pt` files or
emit full legal-action rows.

## Artifacts

| artifact | path |
|---|---|
| generator | `w42/claim_data_inventory/build_claim_data_inventory.py` |
| corpus file table | `w42/claim_data_inventory/corpus_files.csv` |
| claim-family route table | `w42/claim_data_inventory/claim_family_routes.csv` |
| summary | `w42/claim_data_inventory/summary.json` |
| manifest | `w42/claim_data_inventory/manifest.json` |

## Coverage

The inventory sees 114 `.pt` files in `gus/data/`.

| corpus family | files | GiB | inferred games | inferred decisions |
|---|---:|---:|---:|---:|
| legacy seed-modulo declaration chunks | 100 | 108.459 | 10000 | 280000 |
| legacy single 100-game file | 1 | 1.003 | not counted in main coverage | not counted in main coverage |
| schema-v2 all-declaration fixed-bid train | 10 | 0.719 | 1000 | 28000 |
| eval files | 2 | 0.345 | not counted in train coverage | not counted in train coverage |
| cache / other | 1 | 0.000 | not applicable | not applicable |

The full schema pass inspected 113 payloads: all 100 legacy chunks, the legacy
single file, all 10 v2 train files, and both eval files. The only uninspected
file is `_len_cache_train_10k.pt`.

The legacy chunks cover nominal seeds 0-9999. The declaration mapping is
`decl_id = seed % 10`, giving 1000 games per declaration, including declaration
7 (`doubles_trump`) and declaration 9 (`no_trump`). This is broad within-regime
data. It is not same-hand paired declaration evidence.

The v2 train files cover seeds 0-99 with all declarations 0-9 per seed at fixed
`bid_value=30`. This is useful paired-declaration smoke data, but much smaller
and not real auction or bid-margin data.

## Schema Findings

All inspected legacy chunks have `world_hands`, `q_per_world`, `e_q`,
`legal_mask`, `action_taken`, and `player` on decisions. They do not carry
observed `bid_value`, `oracle_softmax_per_seat`, `legal_mask_per_seat`, or
`voids_per_seat`.

All ten inspected v2 train files carry fixed `bid_value=30` and the schema-v2
per-seat fields. Their `q_per_world` sample count is 200 per decision.

Observed sample counts across inspected files are 200, 800, 1600, 2400, 3200,
4000, 4800, 5600, and 6400. The large legacy corpus therefore improves power
and hidden-world diagnostics, but it does not add the missing bid-margin,
auction, or 84-contract fields.

## Claim Routing

The inventory changes the recovery plan without moving any claim status.

| bead | family | route after inventory |
|---|---|---|
| `t42-0b4l.4` | hidden threat / belief impact | directly testable now as offline diagnostics over the large corpus |
| `t42-0b4l.5` | bidding risk / bid only enough | still requires auction and bid-margin generation for hard bidding claims |
| `t42-0b4l.6` | seat/position and bidder sequencing | filtered corpus rows can support descriptive/action-local labels; sequence counterfactuals still need generation |
| `t42-0b4l.7` | 84 stopper / weapon preservation | filtered corpus rows can rehearse generic preservation labels; true 84 contract claims still require generation |
| `t42-0b4l.8` | doubles-as-trump / no-trump tactics | within-regime tactical labels are directly mineable now; powered same-hand regime choice still needs paired generation |
| `t42-0b4l.9` | claim-tag model probes | large corpus is useful after direct detectors are sharpened |

The practical recovery priority is therefore Chapter 9 within-regime mining:
declaration 7 and declaration 9 have enough legacy data to look for low-double
sacrifice, support-double preservation, suit-count/walker, and no-trump defense
preservation labels. The same-hand "no-trump over doubles-trump" regime-choice
claim remains blocked by the legacy seed-modulo design, except for smaller v2
smoke checks.

## Leakage Boundary

Hidden truth remains offline-only. `world_hands`, `q_per_world`, E[Q], hidden
holders, and future outcomes can label reports or train belief targets. They
cannot become live policy features.

## Commands

```bash
python w42/claim_data_inventory/build_claim_data_inventory.py --full-legacy-scan

python -m py_compile w42/claim_data_inventory/build_claim_data_inventory.py

python - <<'PY'
import csv, json
rows = list(csv.DictReader(open("w42/claim_data_inventory/corpus_files.csv", newline="")))
routes = list(csv.DictReader(open("w42/claim_data_inventory/claim_family_routes.csv", newline="")))
summary = json.load(open("w42/claim_data_inventory/summary.json"))
assert len(rows) == summary["aggregate"]["file_count"] == 114
assert summary["aggregate"]["inferred_decisions_by_family"]["legacy_seed_mod_decl"] == 280000
assert any(row["bead_id"] == "t42-0b4l.8" for row in routes)
print("validated", len(rows), "files", len(routes), "routes")
PY
```

## Claim-Ledger Impact

No claim status changed. This is an evidence-availability and routing artifact.

## Links

[[w42]] | [[w42-phase2-claim-analysis-matrix]] |
[[w42-doubles-no-trump-claim-validation]] |
[[winning42-ch09-doubles-no-trump]]
