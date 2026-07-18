---
title: HuggingFace Assets — the off-git shelf
kind: topic
first_seen: 2026-07-15
last_updated: 2026-07-18
status: active
---

Everything the project has published to HuggingFace under `jasonyandell`,
inventoried 2026-07-15 via the authenticated API. Everything is public
(all 90 remaining private repos flipped 2026-07-15 — nothing here is a
secret; it's the opposite). HF is the
durable home for anything too big for git — [[run-artifacts-policy]] governs
the split. The established upload path is `scripts/hf_publish/upload.py`.

## Datasets (4)

- [`mk5-run-evidence`](https://huggingface.co/datasets/jasonyandell/mk5-run-evidence)
  (public) — row-level run evidence mirrored out of git at repo-relative
  paths ([[run-artifacts-policy]]): per-game CSVs, the guard-premium ledger,
  `otis/models/*.pt` heads, all of `arena/results/`. Tag
  `otis-night2-2026-07-15` pins the migration. Tag
  [`table42-gamenight-1`](https://huggingface.co/datasets/jasonyandell/mk5-run-evidence/tree/table42-gamenight-1/table42)
  pins the [[table42]] salvage (2026-07-18): game night 1's host run dirs
  (`table42/host/run/*/` — log.jsonl, config, views), the v1-era game
  logs (`table42/logs/`), and the h2h rows (`table42/jud_vs_otis/*`,
  `table42/jud_vs_gus/n128/`; `jud_vs_burl` has only its runner script —
  no per-game CSVs ever existed on disk). Moved with the stock `hf`
  CLI (see below); public means the dataset viewer + `hf://` paths
  (DuckDB/pandas) read it with no token.

- [`texas-42-joint-world-corpus`](https://huggingface.co/datasets/jasonyandell/texas-42-joint-world-corpus)
  (public) — the original joint-world tensor corpus behind Gus belief training
  ([[joint-world-tensor]], [[gus]]). Superseded for new work by v2 after the
  contamination finding.
- [`texas-42-joint-world-corpus-v2`](https://huggingface.co/datasets/jasonyandell/texas-42-joint-world-corpus-v2)
  (public) — the clean regenerated deck from otis night 2: 1.04B worlds,
  0 invalid, Bayes ceiling 40.119% ([[otis-phase-r]],
  [[belief-bayes-ceiling]]). Written by `forge/eq/regen_corpus_v2.py`.
- [`mk5-fleet-payload`](https://huggingface.co/datasets/jasonyandell/mk5-fleet-payload)
  — the zeb fleet-ops payload bundle
  ([[zeb-fleet-ops]]).

## Models

- [`zeb-42`](https://huggingface.co/jasonyandell/zeb-42) — the released Zeb
  model ([[zeb]]).
- [`zeb-42-examples`](https://huggingface.co/jasonyandell/zeb-42-examples) —
  its worked-examples companion.

**Adapter families** (~90 repos; each family keeps per-epoch `-ep1..3`
or per-iter siblings):

| Family | Repos | Context |
|---|---|---|
| `gemma-4-e2b-texas42-star-iter0..27` | 28 | the STaR loop ([[star-10-iterations]]) |
| `gemma-4-e2b-texas42-stage0*` (base, v3, v4-iter/ep, v5, v5-clean, kerry) | ~20 | Stage 0 SFT ladder ([[stage-0-v1-training]], [[kerry-adapter]]) |
| `qwen3-1.7b-texas42-stage0-v5..v10-maskfix` | ~25 | the Qwen pivot ([[base-model-pivot-qwen]], [[v5-adapter]]) |
| `qwen3-14b-texas42-stage0-v9*` | 4 | the 14B probe |
| `qwen3-1.7b-texas42-rationalize-v1*` | 4 | the rationalize lane |
| `gemma-4-e2b-texas42-burl-*` (smoke, iter0–4) | 7 | Burl iterations ([[burl]], [[burl-iter0-eval]]) |

**Not Texas 42** (listed for completeness): `gomoku-9x9`,
`gomoku-13x13`, `rapfi-arm64` — a separate gomoku project.

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
[[gus]] [[burl]] [[zeb-fleet-ops]]
