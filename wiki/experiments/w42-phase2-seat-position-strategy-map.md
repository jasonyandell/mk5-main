---
title: w42 Phase 2 Seat Position Strategy Map
kind: experiment
first_seen: local-2026-05-02
last_updated: local-2026-05-02
status: active
---

## Summary

[[w42]] phase 2 maps Winning 42 advice by seat, position, role, and hand phase.
It is a vocabulary map, not a verdict. It does not change claim-ledger status.

The map turns role-specific advice into detector targets: what P0/P1/P2/P3 care
about as bidder, setter, partner, defender, leader, follower, last-to-act, early
trick actor, late trick actor, 84 bidder/defender, doubles-trump player, and
no-trump player.

The durable artifact is
`w42/seat_position_strategy_map/position_strategy_matrix.csv`. Each row records
public-state prerequisites, the book concept, current v0/rich detector coverage,
the missing direct detector, a suggested metric, leakage risk, and dependencies
on phase-2 statistics or distribution labels.

## Artifact Shape

| artifact | role |
|---|---|
| `w42/seat_position_strategy_map/position_strategy_matrix.csv` | main role/phase detector matrix |
| `w42/seat_position_strategy_map/summary.json` | machine-readable row counts, coverage summary, and first-trick walkthrough |
| `w42/seat_position_strategy_map/manifest.json` | provenance, source list, leakage boundary, and validation contract |

The matrix has 17 rows across bidder, setter, partner, defender, 84,
doubles/no-trump, auction-last-seat, public-belief-update, and Burl trace-review
roles.

## Detector Coverage

Existing v0 tags already cover much of the mechanical public state:

- declaration and regime identity;
- trick phase, lead position, last-to-act, and legal/follow facts;
- current trick count/trump pressure;
- partner or opponent currently winning;
- action identity for count, trump, doubles, and donations;
- played/unseen count and visible pip summaries.

The richer v1 detector map already names several concept families:

- `candidate_bid_loss_budget`, `side_specific_off_risk`, and
  `unnecessary_bid_margin`;
- `setter_pounce_window` and `count_before_certainty`;
- `safe_partner_count_donation`;
- 84 contract/live-asset labels;
- doubles/no-trump regime comparison;
- `legal_inference_boundary` and `early_trick_belief_discovery`.

The missing phase-2 layer is direct role/position composition. A detector such as
`count_donation_to_partner` is not enough by itself; P2 supporting the bidder on
the first trick, P3 closing the trick as defender, and a setter donating count
before partner certainty are different strategic states even when the candidate
domino is the same count tile.

## First-Trick Walkthrough

The E[Q] PDF visualizer gives a concrete position map. The walkthrough uses the
local visualizer sample record documented by [[eq-browser-visualizers]] and tied
to `forge/analysis/scripts/export_eq_visualizer_data.py`, which exports
`eq_pdf_v3_sample.jsonl` and `27b_eq_per_game.jsonl` from the small E[Q] PDF
sample tensor when present. This is artifact provenance, not screenshot-only
evidence.

Record: Game 1/5, declaration blanks.

| seat | action | role lens | detector implication |
|---|---|---|---|
| P0 | lead `6-0` | bidder and first leader | opening-lead plan under blank trump; distinguish trump pull, planned off, count cash, and reentry preservation |
| P1 | play `0-0` | left setter and first follower | first setter response/pounce/follow state; `0-0` takes current control in blanks |
| P2 | play `4-2` | bidder partner and third-to-act | partner support cannot be generic count dumping; current winner, legal follow, and safety dominate |
| P3 | play `4-0` | right setter and last-to-act | last-to-act closure has maximal public trick information but cannot beat P1's blank double |

P1 wins trick 1. The same trick therefore exercises four distinct role lenses:
P0's lead-plan detector, P1's first-setter response detector, P2's partner-safety
detector, and P3's closure detector.

## Metrics

The phase-2 metric surface should stay distribution-aware:

- paired E[Q] delta or regret for the same public decision;
- make/set threshold mass rather than mean E[Q] alone;
- lower-tail or CVaR-style disaster risk;
- branch/shelf labels from E[Q] PDFs;
- forced-versus-voluntary action labels;
- live-asset survival for 84;
- belief calibration changes after public first-trick evidence;
- unsupported hidden-claim rate for Burl traces.

This matches the final w42 report's refinement: many book concepts are really
branch-management claims, not scalar expected-value claims.

## Leakage Boundary

Online-safe inputs are public auction/declaration/score, own hand, legal actions,
public trick history, current trick state, public count/suit/trump depletion, and
beliefs learned from legal public evidence.

Report-only labels include true hidden owners, partner/opponent private hands,
oracle E[Q], per-world outcome branches, completed-hand set attribution, future
outcomes, and hidden-truth labels used to audit trace leakage.

Any detector that requires a true holder, a future trick result, or full-deal
counterfactual ownership is marked eval-only/offline in the matrix.

## Completion Notes

No fresh model training, W&B run, or claim validation run was performed. The bead
creates a structured map for phase-2 detector work and preserves the conservative
w42 rule that detector existence does not move claim-ledger status.

Validation:

```bash
python - <<'PY'
import csv, json
from pathlib import Path
base = Path('w42/seat_position_strategy_map')
with (base / 'position_strategy_matrix.csv').open(newline='') as f:
    rows = list(csv.DictReader(f))
assert len(rows) == 17
assert set(rows[0]) == {
    'role',
    'phase',
    'public_state_prerequisites',
    'book_concept',
    'current_detector_coverage',
    'missing_detector',
    'suggested_metric',
    'leakage_risk',
    'phase2_stats_distribution_dependencies',
}
for name in ['summary.json', 'manifest.json']:
    json.loads((base / name).read_text())
print('validated', len(rows), 'rows')
PY
```

## Provenance

| field | value |
|---|---|
| bead | `t42-5m82.2` |
| data inputs | wiki chapter pages, `w42/strategy_tags_v0/tag_schema.json`, `w42/strategy_tags_v1_map/detector_map.json`, and E[Q] visualizer docs/exporter |
| evidence mode | map-only synthesis |
| claim-ledger impact | no claim-ledger change |
| W&B links | not applicable |
| HF links | not applicable |

## Links

[[w42]] | [[w42-final-empirical-strategy-report]] |
[[winning42-strategy-measurement]] | [[eq-browser-visualizers]]
