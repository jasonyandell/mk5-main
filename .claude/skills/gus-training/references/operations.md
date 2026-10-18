<!-- SPDX-FileCopyrightText: Jason Yandell -->
<!-- SPDX-License-Identifier: CC0-1.0 -->

# Gus execution and friction notes

These notes route the October 2026 workflow. Read current CLI help and the
registered experiment before launching; local paths and service configuration
can change. They do not authorize extra training or publication outside the
user's task.

## Entry points

Run from the chosen mk5 checkout using its `scratch/venv/bin/python -u`.
`scripts/check` creates the checkout-specific environment and native dependencies.

| Need | Entry point |
|---|---|
| Prepare lawful pours and pretrain | `python -u -m lem.observer.train --help`; repeat `--source`, declare `--deals`, `--steps`, device and fresh `--out` |
| Continue same run | Same arguments plus `--resume`, larger total `--steps`; same run ID |
| Branch a supervised continuation | `--init-from EXISTING_RUN --out NEW_RUN`; reads `latest.pt` and carries optimizer/sampler; see restrictions below |
| Whole-deal prediction intervals/comparison | `lem.observer.uncertainty`, `lem.observer.compare` |
| Rebuild worked-hand lesson | `lem.observer.inspect`; preserve the original lesson's checkpoint identity |
| Reward block / fixed panel | `python -u -m lem.observer.reward --help` (`block` or `panel`) |
| Registered depth campaign | `python -u -m lem.observer.campaign --help` (`init`, `prepare`, `run`) |
| Independent campaign replay/input audit | `python -u -m lem.observer.audit --root PATH` |
| Existing short integration pair | `lem.observer.h2h --games 2 --candidate PATH --opponent PATH_OR_SPEC --exclude-deals JSON --out FRESH_PATH` |
| Tournament registration/rated pilot | `python -u -m tournament --help`; follow current bid-30 division documentation |
| Publish completed campaign selection | `python -u -m lem.observer.release --root PATH`; generation publishing is part of campaign execution |
| Phone-readable progress plot | `python -u -m lem.observer.plot --root PATH` (Matplotlib environment) |

`train --init-from` requires matching data, model configuration and tokenizer,
plus seed, learning rate, batch, evaluation interval and device. It does not
select `best.pt` or provide a generic warm-start for a changed learning recipe.
Use it for a longer continuation; implement a distinct, recorded initialization
path when an ablation changes those settings.

**Campaign-specific code:** `campaign.py` and `release.py` currently encode the
2026-10-18 schedule, seed namespaces, W&B grouping and archive destination.
A new directory alone does not make a new independent experiment. Inspect and
version those choices before reuse. A completed state is terminal. For another
hypothesis, adapt the configuration explicitly; don't overwrite the old archive
or quietly reuse its confirmation panel.

## Shared-machine execution and recovery

- Heavy units go through `scripts/cpu-slot --threads T --timeout SECONDS -- …`.
  Set T to actual expected work, not total machine size. The normal cap is
  300 seconds. A longer uninterrupted trajectory needs a stated reason; the
  old pretraining exception is not permission for all future runs.
- The campaign controller already wraps its heavy units. Do not put another
  held CPU slot around the controller and make nested workers wait on it.
- A requested one-minute faceoff means an outer cap of 60 seconds. Players
  still think on deterministic worlds/nodes/work budgets. Preserve unfinished
  claims; do not convert a timeout into a loss, reroll it, or selectively omit
  slow pairings without reporting that coverage is incomplete.
- An exec timeout/yield is not process exit. Keep its session handle and check
  whether it is alive before any restart. Slot-file text can be stale: only an
  active `flock` means occupied. Do not wait for a quiet machine or kill peers.
- Bounded blocks write completion receipts atomically. Resume from verified
  state and existing receipts. An incomplete smoke directory is retained for
  inspection, not deleted to obtain a fresh favorable result.
- Checkpoints may record absolute participant paths. In a new checkout, fetch
  the pinned bytes and update an explicit new manifest; don't silently point
  an old identity at different weights.
- Keep the training environment stable. Use a separate small plotting env if
  optional graphics dependencies are missing; don't upgrade a live trainer.
- Large pours should use preallocated numeric arrays. The first scale-up risk
  was Python-object expansion, not the tiny transformer's memory footprint.

## W&B and readable progress

Use existing authentication from `docs/SECRETS.md`; never print keys. The verified
campaign used SDK 0.31, base `https://api.forge.coreweave.com`, entity
`jasonyandell-forge42`, projects `lem-observer` and `gus-population`. Inspect
current configuration rather than assuming public wandb.ai URLs. Save and send
the returned URL from `wandb-run.json` early.

Persist a run ID before recovery. The installed SDK lacked
`wandb.util.generate_id`; the controller uses `uuid.uuid4().hex[:8]`. Resume the
existing ID instead of creating a second run that resembles the first.

Plot update steps, hands seen and equivalent passes with explicit units. On a
phone, late loss changes disappear under the first steep drop: provide a late
window plus accuracy/impossible mass and a reference line, while retaining the
full curve. Separate collection time, optimizer time, validation, H2H, queue and
publishing. Most reward-generation wall time was collecting play, not gradients;
fresh timing is descriptive, not a strength budget.

Keep train return, development score, imitation retention and H2H smoke outcomes
in separate metric namespaces. A summary update is not new historical telemetry.
Read back publication/summary receipts. JSON float transport can round the last
bit; compare numeric readbacks with a tight declared tolerance, while hashes,
integers and identifiers remain exact.

## Publication and interpretation handoff

Existing model repo: `jasonyandell/gus`. Original tags: `pretrained-v0`, then
`gus-g001` through `gus-g020`; selected tag `campaign-selected-20261018`.
Research data: `jasonyandell/texas-42-zeb-research-2026-10`, original prefix
`lem/gus-depth-campaign-2026-10-18`. Tournament records have their own existing
`jasonyandell/texas-42-tournament` dataset. New runs need distinct identities.

Use cached Hub auth or the documented Keychain path, not credentials in files.
Publish through existing staged/hash-checked code. Verify immutable revisions
with fresh anonymous downloads and tensor/logit parity; listing a remote file
alone is not a full download check. Large-archive metadata verification and
sample download verification are different claims: report which was done.
Load checkpoints with `weights_only=True`; exported safetensors are available.
This is a custom PyTorch model, not an assumed `AutoModel` architecture.
The model release's `gus.pt` is an inference artifact: it omits optimizer and
sampler/RNG state. True recovery requires the full archived training checkpoint
and matching receipts/configuration. Do not present loading exported weights
with a fresh optimizer as resuming the original training trajectory.

For interp, hand over the exact tokenizer, model config, saved untrained twin,
formation series, lawful held-out examples and learning curves. Name which
checkpoint was selected and which fully trained layers changed. Native attention
or activation pictures are observations; they do not establish a unit's meaning.
Keep the original interactive lesson as a stable specimen when later models
arrive. Verify the actual delivery surface before claiming it renders on mobile.
