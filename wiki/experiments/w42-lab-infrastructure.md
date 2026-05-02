---
title: w42 Lab Infrastructure
kind: experiment
first_seen: local-2026-05-02
last_updated: local-2026-05-02
status: active
---

## Summary

[[w42]] uses W&B and Hugging Face as optional lab infrastructure once training
runs, durable datasets, or promotable artifacts exist. This page standardizes the
names before those artifacts are created, so early work can remain local without
inventing new conventions each time.

Initial infrastructure checks found `wandb` unavailable on `PATH`, `hf`
installed but unauthenticated, and the older `huggingface-cli` command present
only as a deprecated shim. A later w42 infrastructure patch installed `wandb`
into the active Python and added optional W&B logging to the raw and v0-tagged
baseline scripts. W&B authentication is still pending, so runs can be created in
offline mode and synced after login.

## Local Setup

Secrets stay out of git. Local credentials, if added later, belong in the normal
tool locations or environment variables:

- W&B: login via `python -m wandb login` or `WANDB_API_KEY` in the caller's
  environment.
- Hugging Face: login via `hf auth login` or `HF_TOKEN` in the caller's
  environment.
- Modal/cloud secrets, if used later, should follow the existing Forge/LEM
  pattern of named secrets rather than committed token files.

Local default until configured:

```bash
export WANDB_ENTITY=jasonyandell-forge42
export WANDB_PROJECT=w42
export HF_NAMESPACE=jasonyandell
```

The active mise Python currently has both `wandb` and `huggingface_hub`
available. The W&B executable may not be on the shell `PATH`, so the reliable
local command form is:

```bash
python -m wandb status
python -m wandb login
```

Training scripts should accept explicit flags or environment overrides for all
remote destinations. If W&B is unavailable or disabled, runs should still write a
local manifest under `scratch/w42/` with the same run identity fields. If W&B is
enabled before login, w42 scripts use offline mode by default and record the
local sync command in their run metadata.

## W&B Conventions

Entity/project:

- Entity: `jasonyandell-forge42`
- Project: `w42`

Run group schema:

```text
w42-{bead_slug}-{experiment_slug}
```

Examples:

- `w42-csw6-lab-infrastructure`
- `w42-tags-v0-rich-strategy`
- `w42-probe-count-donation`

Run name schema:

```text
{model_slug}-{feature_set}-s{seed}-{short_sha}
```

Examples:

- `tiny-policy-raw-public-s0000-489c1fd`
- `tiny-policy-rich-tags-s0000-489c1fd`
- `bucket-probe-pounce-window-s0042-489c1fd`

Every W&B run should include tags:

- `w42`
- `winning42`
- `strategy-validation`
- `forge-eq`
- `gus-format` when it reuses Gus corpus shape
- `scratch` or `promoted`
- the relevant concept bucket, such as `bidding-risk`, `count-donation`,
  `trump-pressure`, `off-protection`, `pounce-window`, `eighty-four`,
  `walker-endgame`, or `belief-memory`

Required config keys:

- `bead_id`
- `git_sha`
- `data_manifest`
- `source_corpus`
- `dataset_name`
- `dataset_version`
- `ruleset`
- `label_source`
- `decision_slice`
- `feature_set`
- `concept_buckets`
- `model_family`
- `model_params`
- `random_seed`
- `train_seed`
- `split_seed`
- `eval_seed`
- `baseline_policy`
- `metrics`
- `claim_ledger_status_before`
- `local_artifact_path`
- `hf_repo_id`
- `wandb_group`

Implemented script flags:

```bash
--wandb / --no-wandb        # default: --wandb
--wandb-project w42
--wandb-entity jasonyandell-forge42
--wandb-group t42-csw6
--wandb-name <run-name>
--wandb-mode auto|online|offline|disabled
```

`--wandb-mode auto` chooses `online` when a W&B API key or login is present and
`offline` otherwise.

w42 policy: model/experiment scripts should use W&B by default. Use
`--no-wandb` only for deliberately boring local checks. Failed, splatted, or
interrupted experiment attempts are useful evidence and should create titled
runs when W&B can initialize.

Automatic run names use:

```text
{bead_id}-{feature_tag}-s{seed}-{short_sha}
```

Examples:

- `t42-csw6.10-raw-public-state-s0042-f1aac89`
- `t42-csw6.11-v0-strategy-tags-s0042-f1aac89`

Failure capture:

- uncaught exceptions after W&B init set summary `status=failed`
- summary captures `failure/type`, `failure/message`, and
  `failure/traceback_tail`
- W&B logs `status/failed=1`
- successful runs set summary `status=completed`

Successful smoke command:

```bash
python scratch/w42/raw_public_state_baseline.py \
  --train gus/data/corpus_train_100.pt \
  --eval gus/data/corpus_eval_20.pt \
  --train-limit 4 \
  --eval-limit 4 \
  --epochs 1 \
  --batch-size 8 \
  --prediction-sample-limit 2 \
  --output-dir scratch/w42/wandb_smoke/default_success \
  --wandb-mode offline
```

Successful smoke result: W&B created offline run id `rwexij8m`, logged
train/eval/final metrics, marked `status=completed`, and wrote local artifacts
under `scratch/w42/wandb_smoke/default_success/`. The cloud sync waits on login.

Failure smoke command:

```bash
python scratch/w42/raw_public_state_baseline.py \
  --train scratch/w42/does-not-exist.pt \
  --eval gus/data/corpus_eval_20.pt \
  --train-limit 4 \
  --eval-limit 4 \
  --epochs 1 \
  --batch-size 4 \
  --prediction-sample-limit 1 \
  --output-dir scratch/w42/wandb_smoke/default_failure \
  --wandb-mode offline
```

Failure smoke result: W&B created offline run id `as3xy7oz`, marked
`status=failed`, captured `failure/type=FileNotFoundError`, captured the missing
path in `failure/message`, and stored the traceback tail.

Resume policy:

- Resume an interrupted run only when the same checkpoint or local manifest stores
  the W&B run id.
- Use `resume="allow"` with the stored id for exact continuation.
- Start a new run id for a new hypothesis, new feature set, changed data
  manifest, or changed random seed.
- Do not rely on display name uniqueness for lineage. Names are human-readable;
  ids and manifests are the durable join keys.

Run links:

- W&B run: offline smoke ids `rwexij8m`, `as3xy7oz`; live links pending login
- W&B artifact: `not applicable`

## Hugging Face Conventions

Namespace:

- HF namespace: `jasonyandell`

Repository naming:

- Durable dataset repo: `jasonyandell/w42-strategy-corpus`
- Durable report artifact repo, if a report outgrows git/wiki tables:
  `jasonyandell/w42-report-artifacts`
- Promoted model repo: `jasonyandell/w42-{model_slug}`

Dataset naming inside manifests:

```text
w42-{slice_slug}-{feature_set}-v{major}
```

Examples:

- `w42-early-decisions-raw-public-v0`
- `w42-full-decisions-rich-tags-v0`
- `w42-pounce-window-buckets-v0`

Model naming:

```text
w42-{model_family}-{feature_set}-{data_slug}
```

Examples:

- `w42-tiny-policy-rich-tags-early-decisions`
- `w42-bucket-probe-count-donation-full-decisions`

Artifact file naming:

```text
{artifact_slug}-{git_sha}-s{seed}.{ext}
```

Examples:

- `metrics-489c1fd-s0000.json`
- `predictions-489c1fd-s0000.parquet`
- `claim-ledger-deltas-489c1fd-s0000.csv`

HF links:

- Dataset repo: `not applicable`
- Model repo: `not applicable`
- Artifact repo: `not applicable`

## What Stays Local

Keep these local under `scratch/w42/` until a later bead promotes them:

- one-off notebooks and first-pass scripts
- tiny corpora used only to validate feature shapes
- failed or underpowered runs
- raw generated games whose data card is not ready
- intermediate predictions, logits, and per-decision debug dumps
- private credentials, local cache paths, and machine-specific launch wrappers

Publish only when an artifact has a clear manifest, reproducible command, stable
schema, and report text that explains what was measured. Detector existence alone
does not justify publishing a dataset or moving a claim out of `not-yet-tested`.

## Commands And Checks

Required reading and prerequisite checks:

```bash
sed -n '1,240p' wiki/AGENTS.md
bd show t42-csw6.2 --json
sed -n '1,260p' wiki/entities/w42.md
sed -n '1,260p' wiki/experiments/winning42-strategy-measurement.md
sed -n '1,260p' wiki/experiments/gus-strategy-tags-probe.md
sed -n '1,280p' wiki/entities/forge-analysis.md
sed -n '1,280p' wiki/entities/forge.md
sed -n '1,280p' wiki/entities/gus.md
sed -n '281,620p' wiki/entities/gus.md
bd show t42-csw6.1 --json
git rev-parse --short HEAD
```

Local infrastructure checks:

```bash
rg -n "wandb|W&B|Weights|huggingface|Hugging Face|hf_|HF_|HF_HOME|WANDB" . \
  --glob '!**/.git/**' --glob '!**/__pycache__/**' --glob '!**/*.pt' \
  --glob '!**/*.ckpt' --glob '!**/*.parquet'
command -v wandb
wandb status
command -v huggingface-cli
huggingface-cli whoami
command -v hf
hf auth whoami
hf --version
git status --short
```

Observed results:

- Base commit for this bead: `489c1fd`
- `bd show t42-csw6.1 --json`: closed; charter exists at [[w42]]
- `command -v wandb`: no path returned
- `wandb status`: `zsh:1: command not found: wandb`
- Later install: `python -m pip install wandb` installed `wandb==0.26.1`
- Later status: `python -m wandb status` works; `api_key` is still `null`
- `command -v huggingface-cli`:
  `/Users/jason/.local/share/mise/installs/python/3.12/bin/huggingface-cli`
- `huggingface-cli whoami`: deprecated; directed callers to use `hf`
- `command -v hf`:
  `/Users/jason/.local/share/mise/installs/python/3.12/bin/hf`
- `hf auth whoami`: `Error: Not logged in`
- `hf --version`: `1.12.0`
- `git status --short`: clean before editing this page

No training run occurred. Random seeds are `not applicable`.

## Claim Ledger

no claim-ledger change

This bead defines lab conventions only. It creates no detector, model, dataset,
metric result, run, artifact, or claim status change.

## Links

[[w42]] | [[winning42-strategy-measurement]] |
[[gus-strategy-tags-probe]] | [[forge-analysis]] | [[forge]] | [[gus]]
