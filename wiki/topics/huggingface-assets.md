---
title: HuggingFace Assets — the off-git shelf
kind: topic
first_seen: 2026-07-15
last_updated: 2026-07-15
status: active
---

Everything the project has published to HuggingFace under `jasonyandell`,
inventoried 2026-07-15 via the authenticated API (public + private). HF is the
durable home for anything too big for git — [[run-artifacts-policy]] governs
the split. The established upload path is `scripts/hf_publish/upload.py`.

## Datasets (3)

- [`texas-42-joint-world-corpus`](https://huggingface.co/datasets/jasonyandell/texas-42-joint-world-corpus)
  (public) — the original joint-world tensor corpus behind Gus belief training
  ([[joint-world-tensor]], [[gus]]). Superseded for new work by v2 after the
  contamination finding.
- [`texas-42-joint-world-corpus-v2`](https://huggingface.co/datasets/jasonyandell/texas-42-joint-world-corpus-v2)
  (public) — the clean regenerated deck from otis night 2: 1.04B worlds,
  0 invalid, Bayes ceiling 40.119% ([[otis-phase-r]],
  [[belief-bayes-ceiling]]). Written by `forge/eq/regen_corpus_v2.py`.
- `mk5-fleet-payload` (private) — the zeb fleet-ops payload bundle
  ([[zeb-fleet-ops]]).

## Models

**Public:**
- [`zeb-42`](https://huggingface.co/jasonyandell/zeb-42) — the released Zeb
  model ([[zeb]]).
- [`zeb-42-examples`](https://huggingface.co/jasonyandell/zeb-42-examples) —
  its worked-examples companion.

**Private adapter families** (~90 repos; each family keeps per-epoch `-ep1..3`
or per-iter siblings):

| Family | Repos | Context |
|---|---|---|
| `gemma-4-e2b-texas42-star-iter0..27` | 28 | the STaR loop ([[star-10-iterations]]) |
| `gemma-4-e2b-texas42-stage0*` (base, v3, v4-iter/ep, v5, v5-clean, kerry) | ~20 | Stage 0 SFT ladder ([[stage-0-v1-training]], [[kerry-adapter]]) |
| `qwen3-1.7b-texas42-stage0-v5..v10-maskfix` | ~25 | the Qwen pivot ([[base-model-pivot-qwen]], [[v5-adapter]]) |
| `qwen3-14b-texas42-stage0-v9*` | 4 | the 14B probe |
| `qwen3-1.7b-texas42-rationalize-v1*` | 4 | the rationalize lane |
| `gemma-4-e2b-texas42-burl-*` (smoke, iter0–4) | 7 | Burl iterations ([[burl]], [[burl-iter0-eval]]) |

**Not Texas 42** (public, listed for completeness): `gomoku-9x9`,
`gomoku-13x13`, `rapfi-arm64` — a separate gomoku project.

## Referenced but absent

These names appear in wiki/code but no such repo exists on HF (never pushed,
or deleted) — treat citations of them as broken until resolved:

- `gus-42-worlds` — referenced in [[w42-lab-infrastructure]],
  [[gen-fleet]], and two `w42/*_claim_validation` scripts.
- `w42-strategy-corpus`, `w42-report-artifacts` — referenced in
  [[w42-lab-infrastructure]].

## Not yet on HF

Run evidence untracked from git on 2026-07-15 (per-game CSVs, guard-premium
ledger, `otis/models/*.pt` heads) currently lives only on the M5's disk — the
proposed home is a private `mk5-run-evidence` dataset; see
[[run-artifacts-policy]].

## Links

[[run-artifacts-policy]] [[joint-world-tensor]] [[otis-phase-r]] [[zeb]]
[[gus]] [[burl]] [[zeb-fleet-ops]]
