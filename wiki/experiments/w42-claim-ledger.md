---
title: w42 Claim Ledger
kind: experiment
first_seen: 2026-05-02
last_updated: 2026-07-13
status: superseded
---

**Superseded as methodology.** The [[jud]] track replaced this status
vocabulary with registered predictions stated before measurement
([[w42-jud-v1]]). The populated 64-row ledger this schema produced was carried
to closure by [[w42-phase2-statistics-claims-ledger]] and audited by
[[w42-phase4-final-claim-audit]]; the canonical synthesis is
[[w42-book-claim-synthesis-and-ai-directions]]. The vocabulary below remains
the correct reference for every 2026-05 w42 page that cites it. Which clock is
this: `phase 1–4` here is the 2026-05-02→03 claim-ledger sweep, not the
bookval-v1 waves or the [[champion]]/[[jud]] rungs.

## Summary

The w42 claim ledger is the conservative evidence register for
[[winning42-strategy-measurement]]. It ties each Winning 42 chapter claim to a detector,
metric or test, data source, readiness state, evidence artifact, status, and caveats.
This schema/vocabulary was real and used across roughly six dozen w42 bead reports
through phase 4 ([[w42-phase4-final-claim-audit]]). On disk it exists as
`w42/claim_ledger.schema.json`, `w42/claim_ledger.template.json`, six scattered
per-bead delta/update files, and one populated central ledger:
`w42/statistics_claims_ledger/claims.csv` (64 rows, built by
`build_statistics_claims_ledger.py`, git sha `e55a6f8`), which
[[w42-phase2-statistics-claims-ledger]] assembled and phase 4 carried to closure
as the canonical claim ledger. The transition to [[jud]]'s
registered-prediction methodology was silent — no page recorded it until the
succession note above.

The ledger is schema-first. A detector definition, chapter harvest, or report bucket is not
empirical support by itself. A claim can move to `supported` or `contradicted` only after an
explicit enumeration, [[forge]] oracle rollout, [[gus]] or w42 probe, or [[burl]] trace review
has run on the stated slice.

Machine-readable schema:

- `w42/claim_ledger.schema.json`
- `w42/claim_ledger.template.json`

Claim ledger impact for `t42-csw6.5`: schema created, no empirical claim tested.

## Status Vocabulary

Use exactly these ledger statuses:

| status | meaning |
|---|---|
| `not-yet-tested` | A chapter/source claim and detector or test shape exists, but no evidence run has checked it. |
| `supported` | Evidence supports the claim on the stated slice. |
| `contradicted` | Evidence argues against the claim on the stated slice. |
| `context-limited` | Evidence or detector scope is valid only in a restricted ruleset, score mode, population, or state regime. |
| `underpowered` | The available sample, proxy, or detector is too weak for a verdict. |

Chapter pages may use older local phrases such as "supported-rules" or "untested." Ledger
entries should normalize those phrases to this vocabulary. Deterministic rule checks can be
`supported` only when the exact check has run and the evidence artifact is recorded.

## Required Fields

| field | purpose |
|---|---|
| `claim_id` | Stable lowercase id, prefixed by source where useful, for example `ch02-bid-only-enough`. |
| `chapter_source` | Chapter/page/source pointer, including line ranges or wiki page when available. |
| `claim` | Natural-language claim being tested, not the detector name. |
| `detector` | Detector, feature, bucket, proof predicate, trace audit, or report check that operationalizes the claim. |
| `metric_test` | Metric or test that would update the claim status. |
| `data_source` | Enumeration table, generated games, forge E[Q]/oracle output, Gus/w42 corpus, Burl traces, synthetic fixtures, or external/human logs. |
| `readiness` | One or more readiness labels: `enumeration`, `oracle`, `gus`, `w42`, `burl`, `ruleset`, `trace-audit`, `report-only`. |
| `evidence_artifact` | File, table, figure, run id, artifact id, or `not applicable` when no evidence exists yet. |
| `status` | One of the five vocabulary values above. |
| `caveats` | Why the status is narrow, weak, or still pending. |

Ledger entries should also include provenance fields for reproducibility:

- `commands`: exact commands/checks that produced the evidence.
- `configs`: config files, CLI flags, ruleset flags, model settings, or `not applicable`.
- `data_inputs`: exact input files, corpus shards, seed ranges, or `not applicable`.
- `commit_sha`: commit that produced the artifact or report.
- `random_seeds`: random seeds or `not applicable`.
- `wandb_links`: W&B run or artifact URLs, or `not applicable`.
- `hf_links`: HuggingFace dataset/model/artifact URLs, or `not applicable`.

## Coverage Notes From Chapter Sweep

The schema covers the recurring shapes in the Winning 42 chapter pages:

- Chapters 1, 9, 10, 11, and 13 need ruleset, legality, scoring, anti-leakage, and variant
  contamination checks. Their best first evidence artifacts are table-driven unit tests,
  legal-mask fixtures, scoring fixtures, and trace audits.
- Chapters 2, 3, 4, 5, 7, 8, and 12 need bid-risk, play-sequencing, partner-support,
  setter-pounce, 84, no-trump, and exception detectors. Their first empirical reports
  should use paired regret, tail regret, make/set rate, set attribution, held-count-never-used
  rate, unsafe donation tail loss, and bucket-shift tables.
- Chapters 6, 11, 14, and 15 need public-evidence and style boundaries. Their claims should
  record whether a detector uses only legal public state, public bids, legal play, failures
  to follow, or trace text; anything relying on private partner facts belongs in caveats or
  in an anti-leakage check, not as evidence.
- Chapter 16 has exact enumeration evidence for hand counts, void/double priors, modal
  hand shape, and four-trump 27-configuration claims. Future ledger entries for those
  claims may be `supported` only when they cite the exact enumeration artifact/command.
  Strategy recommendations derived from those odds still need oracle or model evidence.

## Entry Template

```json
{
  "claim_id": "ch02-bid-only-enough",
  "chapter_source": {
    "wiki_page": "wiki/experiments/winning42-ch02-bidding.md",
    "source_slice": "scratch/winning42/winning42.with_figures.md lines 735-1406"
  },
  "claim": "A bidder should bid only enough to win the auction when the same captured points score regardless of bid size.",
  "detector": "unnecessary_bid_margin",
  "metric_test": "Auction counterfactual: unnecessary bid margin versus set rate, paired regret, and tail loss.",
  "data_source": "Generated auction logs plus forge E[Q] or oracle rollouts.",
  "readiness": ["enumeration", "oracle", "gus", "burl"],
  "evidence_artifact": "not applicable",
  "status": "not-yet-tested",
  "caveats": "Detector shape exists from the chapter harvest, but no evidence run has checked this claim.",
  "provenance": {
    "commands": [],
    "configs": "not applicable",
    "data_inputs": "not applicable",
    "commit_sha": "not applicable",
    "random_seeds": "not applicable",
    "wandb_links": "not applicable",
    "hf_links": "not applicable"
  }
}
```

## Evidence Update Rules

1. Keep claims atomic. Split a book paragraph into separate ledger rows when rule legality,
   model regret, belief calibration, and trace faithfulness need different evidence.
2. Prefer the narrowest status that matches the evidence. A detector smoke test can support
   detector correctness without supporting the strategy recommendation.
3. Record data slice and ruleset gates in the entry. Straight 42, doubles-as-trump,
   no-trump, marks, 84, Nel-O, Sevens, Plunge/Splash, fallback-follow, and coaching modes
   should not be mixed silently.
4. Record whether evidence uses public state, perfect-information labels, oracle values,
   or trace text. Hidden-hand labels are acceptable for evaluation artifacts, but not for
   live strategy features.
5. W&B and HF links are required only when those systems produced a run, dataset,
   checkpoint, or artifact. Otherwise write `not applicable`.

## Checks Run For This Schema Bead

No empirical training, enumeration, oracle rollout, Gus/w42 probe, Burl trace review, W&B
run, or HF artifact was produced for this bead.

Exact commands/checks run:

```bash
git status --short --branch
bd show t42-csw6.5 --json
rg --files wiki/experiments | rg 'winning42-ch.*\.md$|winning42-strategy-measurement\.md$|gus-strategy-tags-probe\.md$'
sed -n '1,220p' wiki/AGENTS.md
sed -n '1,220p' wiki/entities/w42.md
sed -n '1,260p' wiki/experiments/winning42-strategy-measurement.md
sed -n '1,240p' wiki/experiments/gus-strategy-tags-probe.md
sed -n '1,220p' wiki/entities/forge-analysis.md
sed -n '1,220p' wiki/entities/forge.md
sed -n '1,220p' wiki/entities/gus.md
sed -n '1,220p' wiki/experiments/winning42-ch01-in-a-nutshell.md
sed -n '1,220p' wiki/experiments/winning42-ch02-bidding.md
sed -n '1,220p' wiki/experiments/winning42-ch03-bidder-play.md
sed -n '1,220p' wiki/experiments/winning42-ch04-partner-support.md
sed -n '1,220p' wiki/experiments/winning42-ch05-setter-defense.md
sed -n '1,220p' wiki/experiments/winning42-ch06-concentration-style.md
sed -n '1,220p' wiki/experiments/winning42-ch07-taking-every-trick-84.md
sed -n '1,220p' wiki/experiments/winning42-ch08-setting-84.md
sed -n '1,220p' wiki/experiments/winning42-ch09-doubles-no-trump.md
sed -n '1,220p' wiki/experiments/winning42-ch10-tournament-scoring.md
sed -n '1,220p' wiki/experiments/winning42-ch11-table-talk.md
sed -n '1,220p' wiki/experiments/winning42-ch12-advanced-bidding-playing.md
sed -n '1,220p' wiki/experiments/winning42-ch13-optional-variations.md
sed -n '1,220p' wiki/experiments/winning42-ch14-history-tournaments.md
sed -n '1,220p' wiki/experiments/winning42-ch15-celebrities-style.md
sed -n '1,260p' wiki/experiments/winning42-ch16-statistical-odds.md
find scratch -maxdepth 3 -type f | sort | rg 'w42|winning42|claim'
git rev-parse HEAD
date +%Y-%m-%d
```

Run/artifact fields for this bead:

| field | value |
|---|---|
| empirical run | not applicable |
| configs | not applicable |
| data inputs | wiki pages listed above |
| commit SHA at schema drafting | `489c1fd747e6eb6ec3a165ac119620373c03cb5a` |
| random seeds | not applicable |
| W&B links | not applicable |
| HF links | not applicable |
| claim ledger impact | schema created, no empirical claim tested |

## Links

[[w42]] | [[winning42-strategy-measurement]] | [[gus-strategy-tags-probe]] |
[[forge-analysis]] | [[forge]] | [[gus]] | [[burl]]
