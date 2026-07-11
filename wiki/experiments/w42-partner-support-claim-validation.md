---
title: w42 Partner Support Claim Validation
kind: experiment
first_seen: local-2026-05-02
last_updated: local-2026-05-02
status: active
---

## Summary

[[w42]] ran a report-only proxy validation for [[winning42-ch04-partner-support]]
using the held-out Gus joint-world eval corpus, v0 public-state/action-local strategy
tags, and oracle E[Q] labels. The report checks whether Chapter 4 partner-support
claims leave measurable signal in existing public-state tags before a richer
bidder-partner detector exists.

The follow-up [[w42-gus-corpus-tactical-claim-deep-dive]] adds a larger
role-gated direct slice from the Gus v2 all-declaration corpus. It keeps the
Chapter 4 conclusion conservative but strengthens the safety boundary: safe
partner count donation under current bidder-team control has a small positive
paired signal, while unsafe count donation into a defense-won trick is strongly
negative.

[[w42-tactical-claim-replication]] reruns that direct slice with full
legal-action rows and the reusable claim harness. It confirms safe partner count
at `+0.683` Q versus same-decision alternatives and unsafe partner count at
`-8.428`. The result still treats safe donation as context-limited because
current bidder-team control is not yet the same as guaranteed future trick
control.

Generated artifacts:

- `w42/partner_support_claim_validation/analyze_partner_support.py`
- `w42/partner_support_claim_validation/summary.json`
- `w42/partner_support_claim_validation/claim_proxy_stats.csv`

Evidence mode: oracle E[Q] report slice over existing public-state features. No model
training, W&B run, HF artifact, Burl trace review, or Gus core-path edit was performed.

Claim ledger impact: no claim-ledger change.

## Key Question

Do the existing v0 action-local tags provide enough empirical signal to validate Chapter
4 claims about safe count donation, lead capture, virtual boss tiles, non-disruptive
support, and off-suit/count-exposure inference?

## Method

The analyzer loads `gus/data/corpus_eval_20.pt` from the main checkout as a read-only
input because this independent worktree does not contain the large corpus bytes. It uses
`JointWorldFullDataset(..., include_strategy_features=True)` with eval seed `43`,
then computes oracle regret for every legal candidate action:

`regret(action) = max_legal E[Q] - E[Q](action)`.

For each Chapter 4 proxy, it reports:

- candidate-action mean regret with nonparametric bootstrap 95% confidence intervals;
- paired decision contrast when a decision exposes both preferred and alternative proxy
  actions: best preferred-proxy regret minus best alternative-proxy regret;
- oracle-best decision rate for decisions containing each proxy class.

Bootstrap seed: `20260502`. Bootstrap samples: `5000`.

## Data Manifest

| field | value |
|---|---|
| report owner bead | `t42-csw6.19` |
| ruleset / score mode | inherited from existing Gus eval corpus; no variant split available in this report |
| evidence mode | oracle E[Q] report slice over public-state strategy tags |
| decision slice | all 560 held-out eval decisions; partner-support proxies are action-local subsets |
| manifest path | `w42/partner_support_claim_validation/summary.json` |
| source corpus | `/Users/jason/code/mk5-main/gus/data/corpus_eval_20.pt` |
| source wiki pages | `wiki/experiments/winning42-ch04-partner-support.md`, `wiki/experiments/w42-strategy-tags-v0.md`, `wiki/experiments/w42-strategy-tags-v1-map.md` |
| split policy | eval-only seeds `900000-900019`, inherited from w42/Gus eval convention |
| leakage exclusions checked | oracle E[Q] used only as labels/metrics; no hidden hands, Burl traces, W&B, HF, or table-talk text used as features |
| sample size | 560 eval decisions; 1519 legal candidate actions |
| random seeds | eval dataset seed `43`; bootstrap seed `20260502`; data/train seeds not applicable |
| commit SHA | `c603a0d9b19374414753e0953ca1535455f0d0a6` |

## Proxy Definitions

| claim id | preferred proxy | alternative proxy | public-state safe? | caveat |
|---|---|---|---|---|
| `ch04-safe-partner-count-donation` | legal count action tagged `count_donation_to_partner` | legal count action tagged `count_donation_to_opponent` | yes | Does not prove bidder-partner intent or hidden future overtrump safety. |
| `ch04-low-trump-trap-against-count-dump` | partner count donation without `trump_in` | count action with `trump_in` | yes | Does not identify deliberate low-trump trap leads. |
| `ch04-lead-capture-for-support` | action tagged `beats_current` | legal action not tagged `beats_current` | yes | Measures trick capture generally, not partner-of-bidder support. |
| `ch04-effective-double-highest-remaining` | off non-double with no higher live same-suit tile | off non-double with higher same-suit tiles live | yes | A virtual-boss tag is not a full "lead this tile" proof. |
| `ch04-lead-away-from-count-damage` | lead-position off action with no live count in led suit | lead-position off action with live count in led suit | yes | Live-count presence is only a coarse count-exposure proxy. |
| `ch04-avoid-disruptive-partner-trump-lead` | lead-position off action | lead-position trump action | yes | Does not identify bidder's partner, bidder trump length, or exception windows. |

## Findings

| claim id | preferred n | alternative n | paired n | preferred mean regret, 95% CI | alternative mean regret, 95% CI | paired preferred-alt regret, 95% CI | report status |
|---|---:|---:|---:|---:|---:|---:|---|
| `ch04-safe-partner-count-donation` | 25 | 79 | 0 | 2.743 [0.925, 5.086] | 7.026 [5.087, 9.142] | not applicable | underpowered, directional support only |
| `ch04-low-trump-trap-against-count-dump` | 24 | 7 | 0 | 2.857 [0.982, 5.396] | 6.808 [1.322, 15.002] | not applicable | underpowered, directional support only |
| `ch04-lead-capture-for-support` | 56 | 1463 | 43 | 6.128 [3.980, 8.530] | 4.578 [4.205, 4.958] | 2.220 [-2.367, 6.660] | underpowered |
| `ch04-effective-double-highest-remaining` | 60 | 541 | 21 | 3.089 [1.797, 4.653] | 5.571 [4.993, 6.179] | 2.299 [0.161, 4.364] | proxy contradiction; no ledger move |
| `ch04-lead-away-from-count-damage` | 82 | 157 | 22 | 6.916 [5.632, 8.266] | 8.127 [7.124, 9.100] | -2.228 [-5.891, 0.808] | underpowered, directional support only |
| `ch04-avoid-disruptive-partner-trump-lead` | 239 | 59 | 36 | 7.711 [6.924, 8.515] | 6.790 [5.372, 8.266] | 0.362 [-1.985, 2.404] | underpowered |

## Interpretation

Safe donation has the cleanest directional signal: candidate actions tagged as donating
count to partner have lower unpaired oracle regret than candidate actions tagged as
donating count to an opponent-currently-winning trick. The sample is small and unpaired,
so this is not ledger-grade support.

[[w42-gus-corpus-tactical-claim-deep-dive]] improves that evidence shape. On
400 paired same-decision cases, safe partner count has a mean E[Q] delta of
`+0.683` with 95% CI `[+0.147, +1.232]` versus other actions. On 559 paired
cases, unsafe partner count has a mean delta of `-8.428` with 95% CI
`[-9.268, -7.613]`. This supports the "do not donate count into a losing
trick" boundary on the Gus v2 slice, while the positive safe-donation claim
remains context-limited until a true guaranteed-trick detector is added.

Low-liability off leads also lean in the book's direction, with lower regret than
count-exposing off leads, but the paired interval crosses zero. The existing `live_count`
tag is too coarse to decide the count-exposure claim.

The virtual-boss proxy is mixed: virtual-boss candidate actions have lower unpaired mean
regret than other off non-doubles, but in the 21 decisions where both proxy classes appear,
the virtual-boss candidate's best regret is higher. This argues that "highest remaining in
suit" is not sufficient as a recommendation without role, count, trick, and downstream lead
context.

The trump-lead and lead-capture tests are not partner-support tests yet. They are included
because they probe the surfaces Chapter 4 cares about, but the v0 tags do not identify
"current player is bidder's partner" or the bidder's support need.

## Claim-Ledger Impact

| claim id | before | after | reason | evidence artifact |
|---|---|---|---|---|
| `ch04-safe-partner-count-donation` | `underpowered` on chapter page | unchanged | Directional, unpaired proxy only; partner intent and guarantee strength are not isolated. | `w42/partner_support_claim_validation/claim_proxy_stats.csv` |
| `ch04-low-trump-trap-against-count-dump` | `underpowered` on chapter page | unchanged | Tiny alternative sample and no paired decisions. | `w42/partner_support_claim_validation/claim_proxy_stats.csv` |
| `ch04-lead-capture-for-support` | `underpowered` on chapter page | unchanged | Generic trick-capture proxy, not partner-support gated. | `w42/partner_support_claim_validation/claim_proxy_stats.csv` |
| `ch04-effective-double-highest-remaining` | `underpowered` on chapter page | unchanged | Proxy contradiction is useful negative evidence, but detector is too coarse for ledger status. | `w42/partner_support_claim_validation/claim_proxy_stats.csv` |
| `ch04-lead-away-from-count-damage` | `underpowered` on chapter page | unchanged | Directional paired effect with CI crossing zero. | `w42/partner_support_claim_validation/claim_proxy_stats.csv` |
| `ch04-avoid-disruptive-partner-trump-lead` | `underpowered` on chapter page | unchanged | Missing bidder-partner role gate and exception features. | `w42/partner_support_claim_validation/claim_proxy_stats.csv` |

Claim ledger impact: no claim-ledger change.

## W&B / HF Links

| system | link |
|---|---|
| W&B run | not applicable |
| W&B artifact | not applicable |
| HF dataset | not applicable |
| HF model | not applicable |
| HF artifact | not applicable |

## Caveats

- The evidence is oracle E[Q] over generated Gus eval decisions, not human gameplay.
- The report uses existing v0 tags; it does not implement the richer v1
  `partner_support_regime` role gate.
- The main corpus bytes were read from `/Users/jason/code/mk5-main/gus/data/` because the
  independent worktree did not contain them.
- Count donation and virtual-boss proxies are action-local and public-state safe, but they
  do not encode partner intent, bid margin, future lead value, or hidden overtrump facts.
- Sample sizes for actual donation windows are small: 25 partner-donation actions and 7
  trump-in count-dump actions on this eval slice.
- No FDR correction was applied because this is an exploratory Chapter 4 proxy report with
  no claim-ledger status updates.

## Exact Commands / Configs / Seeds

```bash
git -C /Users/jason/code/mk5-main worktree add -b w42/csw6-19 /Users/jason/code/mk5-main/.claude/worktrees/w42-csw6-19 forge
git rev-parse HEAD
bd show t42-csw6.19
sed -n '1,220p' wiki/AGENTS.md
sed -n '1,260p' wiki/entities/w42.md
sed -n '1,260p' wiki/experiments/w42-report-template.md
sed -n '1,260p' wiki/experiments/w42-claim-ledger.md
sed -n '1,320p' wiki/experiments/winning42-ch04-partner-support.md
sed -n '1,260p' wiki/experiments/w42-dataset-manifest.md
sed -n '1,260p' wiki/experiments/w42-strategy-tags-v0.md
sed -n '1,260p' wiki/experiments/w42-strategy-tags-v1-map.md
sed -n '1,220p' wiki/entities/forge.md
sed -n '1,220p' wiki/entities/gus.md
python w42/partner_support_claim_validation/analyze_partner_support.py --eval /Users/jason/code/mk5-main/gus/data/corpus_eval_20.pt --eval-seed 43 --bootstrap-seed 20260502 --bootstrap-samples 5000 --output-dir w42/partner_support_claim_validation
python -m py_compile w42/partner_support_claim_validation/analyze_partner_support.py
```

Configs:

- `w42/partner_support_claim_validation/analyze_partner_support.py`

Seeds:

- data generation: not applicable
- dataset shuffle: not applicable
- train: not applicable
- eval dataset seed: `43`
- bootstrap seed: `20260502`
- bootstrap samples: `5000`

Commit SHA:

- `c603a0d9b19374414753e0953ca1535455f0d0a6`

## Next Checks

- Implement a v1 `partner_support_regime` gate so Chapter 4 tests filter to
  current-player-is-bidder-partner positions.
- Add guarantee-strength labels for certain/probable/unsafe partner-won tricks.
- Replace the virtual-boss proxy with a role- and lead-context report that tests whether
  highest-remaining suit tiles actually recover lead or lower set risk.
- Run the same claim table on a larger eval-only slice with paired counterfactuals before
  moving any claim-ledger status.

## Links

[[w42]] | [[winning42-ch04-partner-support]] | [[w42-strategy-tags-v0]] |
[[w42-strategy-tags-v1-map]] | [[w42-claim-ledger]] |
[[w42-tactical-claim-replication]] | [[forge]] | [[gus]]

## Audit (2026-07-07)

Two-pass audit against code and artifacts; no issues found.

- The source corpus `gus/data/corpus_eval_20.pt` is referenced by machine-local absolute path; recording its sha256 in `summary.json` would make the report reproducible from other checkouts.
