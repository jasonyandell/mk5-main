---
title: w42 W&B Run Comparison Dashboard
kind: experiment
first_seen: local-2026-05-02
last_updated: afd4802
status: superseded
---

**Superseded.** The dashboard and its 10-run inventory are accurate as of
2026-05-02, but the practice it prescribes for future runs was abandoned:
W&B logging tapered off after phase3 and is absent entirely from the
book-validation/jud/champion era (zero W&B mentions in
[[w42-phase4-final-claim-audit]], [[w42-lens-v1-utility-head-to-head]],
[[w42-jud-v1]], or [[w42-champion-selfplay-fixed-point]]). No successor
dashboard exists because the workstream stopped using W&B as its evidence
notebook.

## Summary

[[w42]] now has a W&B comparison view definition for the baseline, rich-tag,
smoke, claim-validation, diagnostic, superseded, and failed runs in project
`jasonyandell-forge42/w42`.

Programmatic W&B run inspection works locally: `wandb.Api()` loaded credentials
from `/Users/jason/.netrc` and returned 10 runs from the project. Programmatic
workspace/report creation is not available in this environment because
`wandb_workspaces` is not installed, so the deliverable is a reusable view
definition plus a comparison table with direct run links.

Artifacts:

- `w42/wandb_run_comparison_dashboard/run_comparison.csv`
- `w42/wandb_run_comparison_dashboard/wandb_view_definition.json`
- `w42/wandb_run_comparison_dashboard/README.md`

W&B project: `https://wandb.ai/jasonyandell-forge42/w42`.

## View Definition

Recommended W&B view:

| field | value |
|---|---|
| entity/project | `jasonyandell-forge42/w42` |
| filter | `tags contains w42` |
| include states | `finished`, `failed`, `crashed` |
| group by | `group`, then `config.bead_id` |
| sort | `created_at asc` |
| key columns | name, id, state, group, tags, `config.bead_id`, `config.git_sha`, `summary.status`, `summary.best_mean_regret`, `summary.best_match_rate`, `summary.best_tail_regret_rate_ge_5`, `summary.ruleset_supported`, `summary.ruleset_checks/failed`, `summary.failure/type`, `summary.failure/message`, url |

Recommended panels:

- model comparison: `summary.best_mean_regret` by `config.bead_id`
- model comparison: `summary.best_match_rate` by `config.bead_id`
- claim-validation table filtered to `tags contains strategy-validation`
- failures/splats table filtered to failed/crashed state plus diagnostic and superseded runs

Machine-readable definition:
`w42/wandb_run_comparison_dashboard/wandb_view_definition.json`.

## Run Comparison

| category | bead | state | primary? | headline | W&B link |
|---|---|---|---|---|---|
| raw smoke | `t42-csw6.10` | finished | primary smoke | `best_mean_regret=2.1360` on a 4-row smoke eval | `https://wandb.ai/jasonyandell-forge42/w42/runs/wv7pkuco` |
| raw full baseline | `t42-csw6.10` | local-only | primary local | best mean regret `2.4711`, match `59.64%`, final mean regret `2.8336` on 560 eval rows | not applicable |
| v0 full baseline | `t42-csw6.11` | local-only | primary local | best mean regret `1.9997`, match `63.93%`, final mean regret `2.0991` on 560 eval rows | not applicable |
| rich full baseline | `t42-csw6.12` | finished | primary | best mean regret `1.9696`, match `65.54%`, tail regret `>=5` rate `11.43%` | `https://wandb.ai/jasonyandell-forge42/w42/runs/3xyy2dxr` |
| rich full baseline | `t42-csw6.12` | finished | superseded | same core metrics; superseded by tail-delta label correction | `https://wandb.ai/jasonyandell-forge42/w42/runs/c5x85xfx` |
| setter defense validation | `t42-csw6.20` | finished | primary | direct setter-pounce detectors missing; ungated count-to-opponent proxy is high-regret | `https://wandb.ai/jasonyandell-forge42/w42/runs/mc9bcj6n` |
| 84 validation | `t42-csw6.21` | failed | useful failure | CSV schema splat after W&B init: `live_weapon_pool` field mismatch | `https://wandb.ai/jasonyandell-forge42/w42/runs/oizyjty5` |
| 84 validation | `t42-csw6.21` | finished | superseded | completed with wrong off semantics | `https://wandb.ai/jasonyandell-forge42/w42/runs/w4wupdih` |
| 84 validation | `t42-csw6.21` | finished | primary | named missing matching double opponent-team ownership `66.67%`; either of two `90.00%` | `https://wandb.ai/jasonyandell-forge42/w42/runs/vmta9zew` |
| doubles/no-trump validation | `t42-csw6.22` | finished | diagnostic | first pass: 4 supported ruleset checks, 1 failed checker assumption | `https://wandb.ai/jasonyandell-forge42/w42/runs/fveoarwf` |
| doubles/no-trump validation | `t42-csw6.22` | finished | primary | corrected pass: 5 supported ruleset checks, 0 failed | `https://wandb.ai/jasonyandell-forge42/w42/runs/0kop3jhk` |
| scoring drift validation | `t42-csw6.23` | finished | primary | deterministic scoring transform; 6 Chapter 10 claims supported in local delta | `https://wandb.ai/jasonyandell-forge42/w42/runs/7keeve33` |

Full table: `w42/wandb_run_comparison_dashboard/run_comparison.csv`.

## Grouping And Tags

Current grouping is useful but inconsistent across beads:

| group | contents |
|---|---|
| `t42-csw6` | smoke, rich, setter-defense |
| `w42-csw6-84-claim-validation` | 84 failed, superseded, corrected |
| `w42-doubles-no-trump-claim-validation` | Chapter 9 diagnostic and corrected runs |
| `w42-csw6-scoring-objective-drift` | Chapter 10 scoring run |

Future w42 runs should keep the existing [[w42-lab-infrastructure]] tags and add
lineage tags that make dashboards easier:

- Always include: `w42`, `winning42`, `strategy-validation`, `promoted`, and
  `t42-csw6.N` for new runs. Historical runs may still carry the old `scratch`
  tag.
- Model runs should include feature tags: `raw-public-state`,
  `v0-strategy-tags`, `rich-tags`, `many-signal`.
- Claim runs should include bucket tags: `setter-defense`, `pounce-window`,
  `eighty-four`, `doubles`, `no-trump`, `scoring`, `tournament`,
  `trump-pressure`, `off-protection`.
- Lineage/status tags should be added where possible: `primary`, `superseded`,
  `diagnostic`, `failure`, `claim-validation`, `smoke`.

## W&B / HF Status

| system | status |
|---|---|
| W&B API | available; `wandb.Api()` returned 10 project runs |
| W&B project | `https://wandb.ai/jasonyandell-forge42/w42` |
| W&B hosted dashboard/report | not created; local programmatic report/workspace support is unavailable without `wandb_workspaces` |
| W&B reusable view definition | `w42/wandb_run_comparison_dashboard/wandb_view_definition.json` |
| HF dataset/model/artifact links | not applicable for this dashboard bead |

The lab-infrastructure page mentions live smoke run ids `5ychhiid`, `as3xy7oz`,
and `rwexij8m`; the current W&B API check could not fetch those ids. The live
project listing did return `wv7pkuco` as the visible success smoke run and
`oizyjty5` as the visible failed run useful for failure/splat inspection.

## Claim-Ledger Impact

No claim-ledger change.

This bead compares and organizes existing run evidence. It does not add a new
enumeration, oracle rollout, model training result, Burl trace audit, or central
claim-ledger status decision. Existing local claim-ledger deltas remain attached
to their source reports.

## Exact Commands

```bash
git -C /Users/jason/code/mk5-main worktree add -b w42/csw6-26 /Users/jason/code/mk5-main/.claude/worktrees/w42-csw6-26 forge
sed -n '1,240p' wiki/AGENTS.md
bd show t42-csw6.26
rg --files wiki scratch | rg 'w42|winning42|wandb|claim|scoring|rich|setter|doubles|trump|84'
sed -n '1,220p' wiki/experiments/w42-lab-infrastructure.md
sed -n '1,220p' wiki/experiments/w42-raw-public-state-baseline.md
sed -n '1,220p' wiki/experiments/w42-v0-strategy-tags-baseline.md
sed -n '1,260p' wiki/experiments/w42-rich-tag-many-signal-probe.md
sed -n '1,240p' wiki/experiments/w42-setter-defense-claim-validation.md
sed -n '1,240p' wiki/experiments/w42-84-claim-validation.md
sed -n '1,240p' wiki/experiments/w42-doubles-no-trump-claim-validation.md
sed -n '1,240p' wiki/experiments/w42-scoring-objective-drift-claim-validation.md
python -m wandb status
python - <<'PY'
import wandb
api = wandb.Api(timeout=20)
runs = list(api.runs("jasonyandell-forge42/w42", per_page=100))
print("api_status=ok")
print("run_count", len(runs))
for run in runs:
    print(run.id, run.name, run.state, run.group, run.url)
PY
python - <<'PY'
try:
    import wandb_workspaces
    print("wandb_workspaces=available")
except Exception as exc:
    print("wandb_workspaces=unavailable")
    print(type(exc).__name__, exc)
PY
```

Configs:

- `w42/wandb_run_comparison_dashboard/wandb_view_definition.json`
- `w42/wandb_run_comparison_dashboard/run_comparison.csv`

Seeds: not applicable; no run occurred.

Commit SHA during dashboard construction: `1830e5f`.

## Links

[[w42]] | [[w42-lab-infrastructure]] | [[w42-raw-public-state-baseline]] |
[[w42-v0-strategy-tags-baseline]] | [[w42-rich-tag-many-signal-probe]] |
[[w42-setter-defense-claim-validation]] | [[w42-84-claim-validation]] |
[[w42-doubles-no-trump-claim-validation]] |
[[w42-scoring-objective-drift-claim-validation]]
