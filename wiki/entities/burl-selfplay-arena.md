---
title: Burl Self-Play Arena (4-Claude orchestrator)
kind: entity
first_seen: 2026-04-19
last_updated: 2026-04-19
status: retired
---

## What it is

The Selfplay Arena is a single-process orchestrator in `burl/arena/` that runs a full
[[texas-42]] hand with four per-seat Claude Agent SDK sessions. One `commit_play` per turn,
all 28 plays resolved. Reuses `agent_runner_native`'s seat-filtered prompts and the
`haiku_spike/agent.py` Agent SDK pattern. (commit message @ 35c75ff)

## CLI

- `--seed` — determines the deal and declaration (from the seed's source dataset)
- `--tag` — appends a suffix to output stems for head-to-head runs without clobbering
  (added at 39aafaf)

## Outputs

Structured JSONL (per-turn events), human-readable markdown (trick-by-trick narration with
rationale snippets), run log.

## Runs at this frontier

| Run | Model | Result | Cost | Wall |
|---|---|---|---|---|
| Haiku seed 900010 | [[haiku-4-5]] × 4 | Bidder team 0-42 (shutout) | $0.84 | 7m22s |
| Opus seed 900010 | Opus 4.7 × 4 | Bidder team 7-35 | $5.58 | 8m50s |

Same deal, both teams set (bad deal for team 0). Opus salvages 7 points where Haiku was
shut out. Opening lead diverges: Haiku led low trump generically; Opus led double-ones
after 7 `eq_outcome_distribution` evaluations. See [[experiments/opus-vs-haiku-arena]].

Tool-use efficiency delta (same 28 decisions):
- `trump_declared`: Haiku 24× vs Opus 1× — Haiku re-queries what's in the system prompt
- `is_legal`: Haiku 92× vs Opus 27× — same pattern
- `eq_outcome_distribution`: both ~1.5/turn
- `conditional_outcome`: 0 across both full games (56 decisions)

(commit message @ 39aafaf)

## conditional_outcome structural finding

Across 145+ decisions spanning single-decision Haiku, single-decision Opus, Haiku full game,
and Opus full game, no model has ever called `conditional_outcome` zero-shot. This is the
4th observation of the pattern. See [[topics/conditional-outcome-structural-nonuse]].
(commit messages @ 35c75ff, 39aafaf)

## Status

The 2-run POC (Haiku shutout, Opus salvage) was never extended past these two games.
Last commit touching the arena is `39aafaf` (2026-04-19); dormant since, and superseded
along with the rest of [[burl]] by [[w42-jud-v1|jud]]'s pure-NN self-play line.
