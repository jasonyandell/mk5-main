# Zeb Worker Saga

Date: 2026-02-15
Bead: `t42-bzsk`
Owner: Jason + Codex
Experiment: `lb-v-eq-3740`

## Scope and Intent

This document captures the real operational story of standing up `lb-v-eq-3740`:

- what the project is trying to do,
- the exact constraints from the bead,
- which runbooks/docs were used as source of truth,
- what actually happened in execution,
- what was fixed in code,
- where the run currently stands.

## Canonical Documentation Used

These are the primary docs used to execute and debug the rollout:

- `forge/zeb/EXPERIMENTS.md`
  - experiment contract: `create` and `learn`
  - Stage 0/1 learner commands
  - source separation expectations
- `forge/zeb/START_NEW_RUN.md`
  - namespace/W&B coupling model
  - warm-start behavior and bootstrap semantics
- `forge/zeb/vast/RUNBOOK.md`
  - fleet launch/monitoring guidance
  - hybrid self-play + eval-aux topology
- `forge/zeb/learner/RUNBOOK.md`
  - learner metric expectations and eval-aux observability

## Bead Context (`t42-bzsk`)

### Objective

Create and run a warm-start experiment:

- New namespace: `lb-v-eq-3740`
- Warm-start from `large-belief.pt` at HF step `3740` (`1,700,608` games)
- Topology: 1 learner + 2 self-play workers + 1 eval-aux worker

### Required ML Constraints

- Preserve objective separation:
  - `selfplay-mcts` -> policy + value + belief
  - `eval-eq-zeb` -> value + belief
- Keep `eval_aux_policy_weight=0.0` by default
- Keep eval staleness controls enabled
- Keep eval stream separate from self-play production stream

### Acceptance Criteria (condensed)

1. Bootstrap exists at `forge/zeb/checkpoints/lb-v-eq-3740-bootstrap.pt`
2. Learner runs with `--weights-name lb-v-eq-3740`
3. Hybrid topology live (2 self-play + 1 eval-aux)
4. Stage 1 metrics show non-zero eval mix with policy weight 0
5. Eval-aux lagging indicator metrics are present in W&B
6. Run is documented so it can be replayed

## Experiment Identity

```bash
export HF_WEIGHTS_REPO=jasonyandell/zeb-42
export HF_EXAMPLES_REPO=jasonyandell/zeb-42-examples

export EXP=lb-v-eq-3740
export FLEET_SELFPLAY=zeb-lb-v-eq-3740-selfplay
export FLEET_EVALAUX=zeb-lb-v-eq-3740-evalaux

export BOOTSTRAP=forge/zeb/checkpoints/lb-v-eq-3740-bootstrap.pt
export WANDB_PROJECT=zeb-mcts
export WANDB_RUN=lb-v-eq-3740
```

## What Was Executed

### 1) Bootstrap Creation

Created:

- `forge/zeb/checkpoints/lb-v-eq-3740-bootstrap.pt`

From:

- HF `large-belief.pt`
- provenance captured with source step/games in checkpoint payload

### 2) Learner Start (Stage 1)

Learner has been run with eval-aux enabled and policy weight pinned to zero.

Key flags:

- `--weights-name lb-v-eq-3740`
- `--eval-aux-enabled`
- `--eval-aux-batch-fraction 0.05`
- `--eval-aux-policy-weight 0.0`
- `--eval-aux-value-weight 1.0`
- `--eval-aux-belief-weight 0.5`

### 3) Fleet Start

- Self-play fleet launched via `go-belief.sh` (2 workers target)
- Eval-aux fleet launched via `go-experiment.sh` (1 worker target)

## Challenges Encountered

### A) Vast host startup instability

Symptoms:

- Instances remained `loading` for long periods
- Required repeated culling/replacement of duds

Impact:

- learner entered `No new examples... pausing training` intervals

### B) Eval-aux crash on worker start

Observed error on remote instance:

- `No module named forge.zeb.worker.run_eval_aux`

Root cause:

- active launch path used module mode only (`python -m ...run_eval_aux`)
- cloned runtime on some hosts did not resolve that module

Fix applied:

- Added fallback in `forge/zeb/vast/vast_monitor.sh`
- Added fallback in `forge/zeb/vast/vast_up.sh`

New behavior in eval-aux mode:

1. try `python -u -m forge.zeb.worker.run_eval_aux ...`
2. fallback to `python -u forge/zeb/worker/run_eval_aux.py ...`

### C) Process control/logging friction

- Some detached launches were not reliable in this session
- Stable control came from explicit monitor processes + PTY-attached learner
- Learner output currently attached to a PTY, not a file, unless relaunched with `tee`

## Code Changes Made During This Saga

- `forge/zeb/vast/vast_monitor.sh`
  - eval-aux launch fallback to script-path execution
- `forge/zeb/vast/vast_up.sh`
  - same fallback in non-monitor launcher path

## Operational Facts Learned

1. Namespace/W&B coupling is working as designed
- HF state `wandb_run_id` drives resume lineage
- new experiment namespace gave a clean lineage (`ubry8wiu`)

2. Self-play production recovered once hosts stabilized
- self-play examples resumed ingest and learner resumed cycle advancement

3. Eval-aux is the fragile leg right now
- worker often reaches `loading/running` but needs confirmation of sustained batch generation and upload

4. Launching eval-aux through `go-experiment.sh` is correct
- keeps curated GPU/DPH/oracle defaults through monitor path

## Monitoring and Triage

Single-pane monitor for all 3 groups:

```bash
watch -n 20 '
echo "=== LEARNER ===";
ps -eo pid,etime,cmd | rg "forge\.zeb\.learner\.run|PID" -N || true;
echo;
echo "=== SELFPLAY FLEET ===";
cd /home/jason/v2/mk5-tailwind/forge/zeb/vast && ./vast_status.sh zeb-lb-v-eq-3740-selfplay || true;
echo;
echo "=== EVALAUX FLEET ===";
cd /home/jason/v2/mk5-tailwind/forge/zeb/vast && ./vast_status.sh zeb-lb-v-eq-3740-evalaux || true;
'
```

Learner output (current attached session path may change):

```bash
pid=$(pgrep -f "forge.zeb.learner.run --weights-name lb-v-eq-3740" | head -n1)
readlink /proc/$pid/fd/1
# then tail -f that /dev/pts/N path
```

HF and W&B:

- HF weights: `https://huggingface.co/jasonyandell/zeb-42/tree/main`
- HF examples namespace: `https://huggingface.co/datasets/jasonyandell/zeb-42-examples/tree/main/lb-v-eq-3740`
- W&B run: `https://wandb.ai/jasonyandell-forge42/zeb-mcts/runs/ubry8wiu`

## Current Risk Register

1. Eval-aux worker may still flap during startup
- Mitigation: monitor for the first `batch` lines and periodic uploads

2. Loading-state duds on Vast can starve ingest
- Mitigation: destroy `loading > 10m`, rely on monitor replenish

3. Learner pauses when fresh examples are absent (`max-example-age`)
- Mitigation: keep at least one healthy self-play producer alive continuously

## Replay Checklist

1. Verify bootstrap exists: `forge/zeb/checkpoints/lb-v-eq-3740-bootstrap.pt`
2. Start learner with Stage 1 flags above
3. Start self-play: `go-belief.sh 2`
4. Start eval-aux: `go-experiment.sh 1`
5. Confirm:
   - self-play examples ingested
   - eval-aux examples ingested
   - cycle logs show non-zero `mix(self/eval)` and `eval_aux_policy_w=0`

