---
title: HuggingFace Assets — the off-git shelf
kind: topic
first_seen: 2026-07-15
last_updated: 2026-10-14
status: active
---

Everything the project has published to HuggingFace under `jasonyandell`,
inventoried 2026-07-15 via the authenticated API. Everything is public
(all 90 remaining private repos flipped 2026-07-15 — nothing here is a
secret; it's the opposite). HF is the
durable home for anything too big for git — [[run-artifacts-policy]] governs
the split. The established upload path is `scripts/hf_publish/upload.py`.

## Datasets (17, inventoried 2026-10-14 by the Archive desk)

Sizes from the Hub's file metadata. Every dataset is public except
`plunge-backups`. The two superseded pour repos (`texas-42-zeb-pour`,
`texas-42-zeb-pour-200k`, 36 GB) were deleted 2026-10-14 on Jason's word; every
chunk is in the 250k pour, matched by SHA-256. The license column is the card's front matter. The four older repos that
declared none (walt-archive, mk5-run-evidence, mk5-fleet-payload, zeb-42-examples)
were dedicated CC0 on 2026-10-14 on Jason's word, each with a card and `LICENSE`
([[license-line]]).

| repo | size | license | what | hub |
|---|---|---|---|---|
| [texas-42-zeb-pour-250k](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-pour-250k) | 36.2 GB | CC0 | the Zeb pour, 5 × 50k deals, Parquet + raw + 3 nets | [[scenario-player]] |
| [texas-42-walt-pour](https://huggingface.co/datasets/jasonyandell/texas-42-walt-pour) | 0.76 GB | CC0 | the k1 Walt pour, margin utility | [[scenario-player]] |
| [texas-42-walt-pour-make](https://huggingface.co/datasets/jasonyandell/texas-42-walt-pour-make) | 0.72 GB | CC0 | the k1 Walt pour, make utility (recovered 2026-10-14) | [[scenario-player]] |
| [texas-42-zeb-research-2026-10](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10) | 1.4 GB | CC0 | October's studies, one directory each: `book/`, `zeb/`, `research-night/`, `posterior-arena/`, `endgames-o2/`, `tidy-arena/` | [[book-study-2026-10-12]], [[scenario-player]] |
| [texas-42-mk5-research-2026-04-07](https://huggingface.co/datasets/jasonyandell/texas-42-mk5-research-2026-04-07) | 4.9 GB | CC0 | the oracle and LLM era's evidence (Burl harvests and STaR runs, champion runs, jud, Gus adapters, burl-perf, w42, family hands, resume smoke, candlewax) | [[burl]], [[jud]], [[gus]], [[champion-ladder]] |
| [texas-42-walt-lanes-2026-08-09](https://huggingface.co/datasets/jasonyandell/texas-42-walt-lanes-2026-08-09) | 0.67 GB | CC0 | `~/data/texas-42`: the partnership gym, relational and policy-synthesis campaigns, the sunshine cycles, the Plunge evidence lanes | texas-42 `experiments/partnership`, `experiments/kiln` |
| [texas-42-walt-speed-2026-09](https://huggingface.co/datasets/jasonyandell/texas-42-walt-speed-2026-09) | 0.49 GB | CC0 | texas-42 `experiments/`: full-game-speed results and docs, the response ladder's inputs, the walt42x intake | texas-42 |
| [texas-42-walt-archive](https://huggingface.co/datasets/jasonyandell/texas-42-walt-archive) | 1.9 GB | CC0 | the walt-factory probe era (2026-08), the kiln snapshots, the tiny-net ladder tarballs | texas-42 `walt/ARCHIVE.md` |
| [texas-42-walt-response-ladder](https://huggingface.co/datasets/jasonyandell/texas-42-walt-response-ladder) | 0.24 GB | CC0 | the compiled-policy head-to-heads, curated | texas-42 |
| [texas-42-walt-ladder-labels](https://huggingface.co/datasets/jasonyandell/texas-42-walt-ladder-labels) | 2.2 GB | CC0 | the tiny-net ladder's label piles | texas-42 |
| [texas-42-tournament](https://huggingface.co/datasets/jasonyandell/texas-42-tournament), [texas-42-tournament-selftest](https://huggingface.co/datasets/jasonyandell/texas-42-tournament-selftest) | 1.6 MB | CC0 | the ladder's participants and faceoffs (live) | [[tournament]] |
| [texas-42-joint-world-corpus](https://huggingface.co/datasets/jasonyandell/texas-42-joint-world-corpus), [-v2](https://huggingface.co/datasets/jasonyandell/texas-42-joint-world-corpus-v2) | 118.7 + 118.1 GB | MIT | Gus's joint-world corpora (the local `gus/data` is a copy of v1) | [[joint-world-tensor]], [[otis-phase-r]] |
| [mk5-run-evidence](https://huggingface.co/datasets/jasonyandell/mk5-run-evidence) | 89 MB | CC0 | w42 tactical replication, otis models, table42, arena results, champion, parlor | [[run-artifacts-policy]] |
| [mk5-fleet-payload](https://huggingface.co/datasets/jasonyandell/mk5-fleet-payload) | 142 MB | CC0 | the tarball each rented box pulled for the July 2026 corpus regeneration (not the February Zeb fleet) | [[otis-phase-r]] |
| plunge-backups | 59 KB | private | age-encrypted D1 dumps of Plunge production; `RESTORE.md` inside | Plunge branch `claude/offsite-backup` |

## Models (106)

- [`zeb-42`](https://huggingface.co/jasonyandell/zeb-42) (MIT, 74 MB) — the
  February Zeb model, the old meaning of the name ([[zeb]]);
  [`zeb-42-examples`](https://huggingface.co/jasonyandell/zeb-42-examples)
  (CC0, 20.6 GB) — the 3,418 self-play example batches the Vast workers
  pushed while training it, including the full-teacher E[Q] run (`lb-v-eq-1920`).
- [`texas-42-walt-tiny-net-ladder`](https://huggingface.co/jasonyandell/texas-42-walt-tiny-net-ladder)
  (CC0, 265 MB) — the tiny-net ladder weights (texas-42).

**Adapter families** (99 repos, 81 to 268 MB each; each family keeps per-epoch
`-ep1..3` or per-iter siblings):

| Family | Repos | License | Context |
|---|---|---|---|
| `gemma-4-e2b-texas42-star-iter0..27` | 28 | none | the STaR loop ([[star-10-iterations]]) |
| `gemma-4-e2b-texas42-stage0*` (base, v3, v4-iter/ep, v5, v5-clean, kerry) | 19 | none | Stage 0 SFT ladder ([[stage-0-v1-training]], [[kerry-adapter]]) |
| `gemma-4-e2b-texas42-burl-*` (smoke, iter0–4) | 7 | none | Burl iterations ([[burl]], [[burl-iter0-eval]]) |
| `qwen3-1.7b-texas42-stage0-v5..v10-maskfix` | 33 | apache-2.0 | the Qwen pivot ([[base-model-pivot-qwen]], [[v5-adapter]]) |
| `qwen3-1.7b-texas42-rationalize-v1*` | 4 | apache-2.0 | the rationalize lane |
| `qwen3-14b-texas42-stage0-v9*` | 4 | apache-2.0 | the 14B probe |

**Not Texas 42** (listed for completeness): `gomoku-9x9`,
`gomoku-13x13`, `rapfi-arm64` — a separate gomoku project.

## What stayed local, and why (2026-10-14)

- **Never public:** Roberson's *Winning 42* in every form (`scratch/winning42*`,
  `~/code/winning42-archive`, `scratch/book/{BRIEF,CANDIDATES}.md`, the giveback
  draft, the book-audit catalogs); the claude.ai conversation export
  (`scratch/claudeai-archive`); the FlyWire connectome download
  (`scratch/fly/data`, third-party); session handoff notes that quote Jason.
- **Already on the Hub in another form:** the staged pour copies in
  `scratch/hf` (35 GB), `gus/data` (111 GB, corpus v1), the atlas pour chunks
  (15 GB, the 250k pour's `raw/`), the seven `~/data/texas-42/backups`
  snapshots (the walt-archive's `kiln/snapshots`), the walt-factory archive's
  curated set.
- **Held for Jason:** `~/data/texas-42/seed-survey` (8.6 GB; by its own header a
  deterministic, SHA-pinned function of committed inputs) and
  `~/data/texas-42/walt-response-ladder` (5.9 GB raw; the curated export is up).
- **Disposable:** cargo `target/` trees (full-game-speed 21 GB, walt, rob,
  response-ladder), `lean/.lake`, venvs, the tournament's pinned builds.
- **Lost before recovery** (12 of the 28 cited paths found gone; 13 had been
  staged for the Hub first, 3 are rebuildable): `scratch/stochastic-field`,
  `scratch/book/wave1/overlay-price.md`, the `otis-night2` scripts,
  `equiv-census/class_cfr_census.py`, `brainstorm/criticality.py`,
  `verify_sft_mask`, `lane-a/PREDICTIONS.md`, `PERF_GOAL.md`, `study141.rs`,
  `burl-perf-2/iter26` and `iter29_phase2`, `x.json`, the reason v1 report
  (`scratch/reason`; v2 survived). Recoverable elsewhere:
  `viz/pour_viz_5` (the export is in git and on the Hub), the archaeology
  evidence (from the private claude.ai export), the 250k pour notes (the card).
  The Archive desk's notebook (`links/archive-notebook` in the feed,
  [[standing-desks]]) keeps the full table.

## Moving evidence data

No custom tooling — the stock `hf` CLI does both directions. Run from the
repo root so paths mirror:

```bash
# up: a directory mirrors its relative path automatically
hf upload jasonyandell/mk5-run-evidence <dir> <dir> --repo-type dataset
# up: a single FILE defaults to root — pass the path twice (footgun)
hf upload jasonyandell/mk5-run-evidence <file> <file> --repo-type dataset
# pin: upload prints the commit URL; or name the revision with a tag
hf repo tag create jasonyandell/mk5-run-evidence <tag> --repo-type dataset
# down: restores into the tree through the shared cache (free per-worktree)
hf download jasonyandell/mk5-run-evidence --repo-type dataset \
  --include "<path>/**" --revision <tag-or-sha> --local-dir .
```

Wiki pages cite the tag- or sha-pinned URL the upload prints, never `main`.

## Referenced but absent

These names appear in wiki/code but no such repo exists on HF (never pushed,
or deleted) — treat citations of them as broken until resolved:

- `gus-42-worlds` — referenced in [[w42-lab-infrastructure]],
  [[gen-fleet]], and two `w42/*_claim_validation` scripts.
- `w42-strategy-corpus`, `w42-report-artifacts` — referenced in
  [[w42-lab-infrastructure]].

## Links

[[run-artifacts-policy]] [[joint-world-tensor]] [[otis-phase-r]] [[zeb]]
[[gus]] [[burl]] [[zeb-fleet-ops]] [[parlor]]
