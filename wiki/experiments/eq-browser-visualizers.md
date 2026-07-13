---
title: E[Q] Browser Visualizers
kind: experiment
first_seen: 2026-05-02
last_updated: 2026-07-13
status: complete
---

## Summary

The E[Q] browser visualizers are live again from local, repo-relative data. They
make the imperfect-information E[Q] surface visible in three complementary ways:

- `eq_surface_3d.html` shows the aggregate domino-value surface over moves.
- `eq_pdf_discs.html` shows per-action outcome PDFs, win thresholds, means, and
  uncertainty for sampled games.
- `eq_game_journey.html` shows each player's domino-value trajectories through a
  sampled hand.

This is a visualization of the E[Q] marginalization surface, not proof that the
perfect-information oracle directly describes human Texas 42 play. It is useful
because it makes the "candlewax" shape inspectable: many positions have jagged,
uncertain, near-tied action surfaces rather than one obvious move.

## How To Run

From repo root:

```bash
python forge/analysis/scripts/export_eq_visualizer_data.py --limit 5
cd forge/analysis/results
python serve.py
```

Then open:

- `http://localhost:8000/web/eq_surface_3d.html`
- `http://localhost:8000/web/eq_pdf_discs.html`
- `http://localhost:8000/web/eq_game_journey.html`

The server is necessary because the HTML pages fetch JSONL data from
`../data/*.jsonl`; opening the files directly can fall back to dummy data or show
empty grids.

## Data Files

`forge/analysis/scripts/export_eq_visualizer_data.py` writes:

| file | source | visualizer |
|---|---|---|
| `forge/analysis/results/data/27a_eq_surface.jsonl` | `27a_eq_matrix.csv` and `27a_domino_order.csv` | aggregate 3D surface |
| `forge/analysis/results/data/27b_eq_per_game.jsonl` | `forge/data/eq_pdf_s9200-9201_d10_10s.pt` | game journey |
| `forge/analysis/results/data/eq_pdf_v3_sample.jsonl` | `forge/data/eq_pdf_s9200-9201_d10_10s.pt` | PDF discs |

The regenerated sample is intentionally small: one aggregate surface and five
sample games. It is enough for browser inspection without creating a large data
artifact.

## Browser Verification

Verified with the Codex in-app Browser Use harness on `2026-05-02`.

Observed:

- `eq_surface_3d.html` loaded `27a aggregate E[Q] surface` rather than dummy data.
- `eq_pdf_discs.html` loaded live game state, active hand dominoes, per-action
  PDFs, mean/std labels, and offense/defense win-threshold markers.
- `eq_game_journey.html` loaded five sample games with player hands and visible
  value trajectories.

## Interpretation

The visualizers support the current E[Q]/w42 discussion:

- E[Q] is already a bridge from perfect-information oracle values to
  imperfect-information play by marginalizing over sampled hidden worlds.
- The surface is often not cleanly argmax-shaped; many moves look close,
  volatile, or uncertainty-dominated.
- That makes w42-informed reranking testable in a narrow way: leave clear E[Q]
  decisions alone, and test book-derived detectors only inside near-tie or
  high-uncertainty windows.
- The PDF visualizer is the best place to look for "why did p_make pick this?"
  and "what would w42 have noticed that the mean did not?"

## Caveats

- The 27a surface is aggregate analysis data, not a single game.
- The PDF/journey sample uses a small local E[Q] PDF tensor
  (`forge/data/eq_pdf_s9200-9201_d10_10s.pt`, not checked into the repo) with
  low per-decision sample count; it is for visual inspection, not final
  empirical claims. Without that local file, the export script only rebuilds
  the 27a aggregate surface.
- Browser visual shape can guide hypotheses, but all policy changes still need
  paired game/regret tests.

## Provenance

| field | value |
|---|---|
| bead | `t42-2x64` |
| commands | `python forge/analysis/scripts/export_eq_visualizer_data.py --limit 5`; `cd forge/analysis/results && python serve.py`; Browser Use verification |
| data inputs | `forge/analysis/results/tables/27a_eq_matrix.csv`; `forge/analysis/results/tables/27a_domino_order.csv`; `forge/data/eq_pdf_s9200-9201_d10_10s.pt` |
| W&B links | not applicable |
| claim-ledger impact | no claim-ledger change |

## Links

[[expected-q-value]] | [[oracle-vs-human-play]] | [[candlewax]] |
[[w42]] | [[w42-eq-n10-comparison-slice]]
