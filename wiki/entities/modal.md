---
title: Modal
kind: entity
first_seen: a8bccfa
last_updated: pending-this-ingest
status: active
---

## What it is

Modal is a serverless GPU compute platform used across the project's forge/burl/zeb
infrastructure for training and inference, independent of any single consumer project.
It was first wired up for [[lem]], which wrapped its training and inference paths in Modal
functions running on GPU instances provisioned on demand, with model weights cached across
invocations using Modal volumes (lem/gemma_star/modal_app.py @ a8bccfa). LEM went dormant in
mid-April 2026, but Modal usage continued independently (`forge/modal_app.py` @ 089e850 Feb
10; `burl/modal` @ 19ba103 Apr 28) — LEM was never the sole consumer.

## Usage in LEM

| Use | GPU | Notes |
|---|---|---|
| First-contact inference | L4 (22GB) | fp16, single-prompt sanity check |
| Stage 0 training | L4 (22GB) | bf16, gradient checkpointing, ~60 min, 1 epoch |
| Stage 0 inference | L4 (22GB) | Adapter loaded via PeftModel, merged and unloaded |

Training timeout is set to 4 hours to accommodate full runs. Eval is disabled for L4
training because the eval forward pass OOMs; 100% train accuracy by step 50 makes eval
redundant. (commit message @ df73c8d)

## Credits and cost

Modal provides $30 free credit. At this frontier the Stage 0 training run and both contact
experiments fit within free-tier usage. (lem/OVERVIEW.md @ 24ae55a)

## Volumes and secrets

- **Volume** `gemma-e2b-cache`: caches Gemma 4 E2B weights across invocations
  (`HF_HOME=/model-cache`).
- **Secret** `huggingface-secret`: provides the HF token for gated model access and
  adapter push.

(lem/gemma_star/modal_app.py @ a8bccfa)

## GPU selection

| GPU | VRAM | Used for |
|---|---|---|
| L4 | 22GB | Stage 0 fine-tune (bf16 + gradient checkpointing fills card); first- and second-contact inference |
| A100 | 40GB+ | Stage 1 standalone LoRA training (`train_star.py`) |
| B200 | — | Stage 1 full STaR loop (`star_loop.py`); 120 tok/s batch inference for [[gemma-4-e2b]] |

L4 is the proven Stage 0 recipe. A100 adopted for Stage 1 training at b99c64d. B200 adopted
for the full STaR loop at 26f5ddf, achieving 120 tok/s with the HF SDPA + torch.compile +
left-padded batching recipe (5-example iteration in 151s). (commit messages @ b99c64d, 26f5ddf)

## Related

[[experiments/first-gemma-contact]] and [[experiments/stage-0-v1-training]] both ran on
Modal L4. [[lora-unsloth]] is the fine-tuning library invoked inside Modal functions.

`forge/MODAL_ORIENTATION.md` and `forge/MODAL_MONITOR.md` are the live Modal ops
runbooks for [[forge]] oracle generation (job launch, fleet monitoring, cost math),
kept in-repo as operational docs.
