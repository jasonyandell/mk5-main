---
title: w42 Phase3 84 Seed Mining Corpus
kind: experiment
first_seen: local-2026-05-03
last_updated: afd4802
status: superseded
---

**Superseded** by [[w42-phase4-84-dynamic-seed-tests]], which turns this
page's natural-84-seed inventory into a dynamic reached-state branch-atlas
artifact — the "future branch-atlas/state-injection" work this page names
landed the same archive commit.

## Summary

[[w42]] bead `t42-qtwb.2` takes the documented seed-mining route for phase-3
84/endgame work. The current engine path can generate full 84 branch-atlas
traces from explicit deals, but it does not yet expose a clean arbitrary
late-state injector. This corpus scans generated full deals for naturally
occurring 84 bidder structures and defender asset patterns, giving future
branch-atlas/state-injection runs a large seed menu instead of six hand-built
fixtures.

The run scans 50000 seeds, 4 seats, and 7 pip declarations: 1400000
seat/declaration checks. It emits 214229 candidate 84 rows, 43013 unique
candidate seeds, and a 256-row recommendation table. The mined surfaces include
protected one-off, straight one-off, two-off same-suit, defender live doubles,
same-suit pairs, pair protectors, dead-asset releases, and one natural all-trump
laydown.

Claim-ledger impact: no central claim promotion. This is a powered data route
and seed corpus, not an action-value proof. It upgrades 84 phase-3 from
"hand-picked fixtures only" to "documented natural seed mining with W&B/artifact
archival."

W&B: `https://wandb.ai/jasonyandell-forge42/w42/runs/f33g4fy1`.

## Method

| field | value |
|---|---|
| bead | `t42-qtwb.2` |
| artifact directory | `w42/eighty_four_seed_mining/` |
| miner | `w42/eighty_four_seed_mining/mine_84_seed_corpus.py` |
| validation | `w42/eighty_four_seed_mining/validate_outputs.py` |
| seed range | `0..49999` |
| checks | `50000 * 4 seats * 7 pip declarations = 1400000` |
| recommendation rows | 256 highest-scoring candidate rows |
| W&B run | `f33g4fy1` |

The miner classifies bidder structures:

- `laydown_all_trumps`
- `protected_one_off`
- `straight_one_off`
- `three_trump_three_double_one_off`
- `two_off_same_suit`

It then attaches full-deal defender asset proxies:

- `defender_live_double_weapon`
- `defender_live_same_suit_pair`
- `pair_protector_pressure`
- `dead_asset_release_control`

Rows are retained only when they have an 84-like bidder structure. Defender
assets are attached to those rows as follow-up labels, not mined as generic
standalone defense states.

## Results

| surface | candidate rows |
|---|---:|
| two-off same-suit | 183722 |
| protected one-off | 22176 |
| three-trump / three-double / one-off | 17935 |
| straight one-off | 8330 |
| laydown all trumps | 1 |
| defender live same-suit pair | 206213 |
| defender live double weapon | 140264 |
| pair protector pressure | 85753 |
| dead asset release control | 192522 |

The single natural laydown appears at seed `46198` in the scanned range. The top
recommendation rows include compact, human-readable deals such as seed `6`
(`bidder_seat=0`, fives, protected one-off `4-3`) and seed `16`
(`bidder_seat=2`, twos, straight one-off `6-1`) with opponent matching doubles,
same-suit pair pressure, and protector pressure already present in the full
deal.

## Artifacts

| artifact | content |
|---|---|
| `candidate_84_seed_rows.csv` | 214229 retained seed/seat/declaration/final-off rows. |
| `recommended_84_seed_rows.csv` | 256 highest-scoring rows for dynamic branch-atlas or state-injection follow-up. |
| `surface_summary.csv` | surface-level row counts and example seeds. |
| `family_decl_summary.csv` | surface-by-declaration counts and hit rates. |
| `seed_recommendations.csv` | seed-level recommendations with best rows embedded as JSON. |
| `examples.json` | compact examples by surface. |
| `summary.json` | run config, counts, W&B link, and scientific boundary. |

The artifact directory is about 68 MB, mostly the candidate-row CSV. It is
useful locally as a mining corpus; a future commit or public artifact should
either compress it or promote only the script, summaries, and recommendation
rows.

## Interpretation

The scan changes the phase-3 route in three ways:

- It proves the seed space is rich enough for natural 84 follow-up. The project
  no longer has to rely only on handcrafted six-fixture labs.
- It separates bidder structures from defender assets. A row can say "this is a
  protected one-off candidate and the opponents have a matching double plus a
  same-suit pair pressure surface" before any policy trace is generated.
- It gives `t42-qtwb.2` a reproducible menu for later branch-atlas jobs:
  choose rows by surface, declaration, seat, and defender asset mix, then run
  generated E[Q] games or true late-state injection.

What it does not do:

- It does not score legal action choices.
- It does not prove preserve/spend regret.
- It does not prove straight-off 84 set rate.
- It does not replace the need for a true late-hand state injector.

## Validation

```bash
python -m py_compile \
  w42/eighty_four_seed_mining/mine_84_seed_corpus.py \
  w42/eighty_four_seed_mining/validate_outputs.py

python w42/eighty_four_seed_mining/mine_84_seed_corpus.py \
  --out-dir w42/eighty_four_seed_mining \
  --seeds 50000 \
  --recommendations 256 \
  --smoke \
  --wandb-mode online \
  --wandb-group w42-84-seed-mining \
  --wandb-name t42-qtwb.2-84-seed-mining-v0

python w42/eighty_four_seed_mining/validate_outputs.py \
  --artifact-dir w42/eighty_four_seed_mining \
  --min-candidate-rows 200000 \
  --min-recommended-rows 256
```

## Links

[[w42]] | [[w42-phase2-84-weapon-preservation-probe]] |
[[w42-claim-analysis-synthesis-report]] |
[[winning42-ch07-taking-every-trick-84]] |
[[winning42-ch08-setting-84]] | [[forge]]

## Audit (2026-07-07)

Two-pass audit against code and artifacts; no issues found.

- Headline counts, per-surface table, laydown seed `46198`, and the seed 6 / seed 16 recommendation examples all match `summary.json`, `examples.json`, and `recommended_84_seed_rows.csv`; measured directory size is 67 MB (page's "about 68 MB" is within rounding).
- W&B run `f33g4fy1` is external and verified only as the URL recorded in `summary.json`.
- The page's compress-or-promote-summaries suggestion remains the cheap follow-up.
