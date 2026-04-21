# Gus — Distributed Corpus Generation via Vast.ai Fleet

> Plan for cutting wall-clock on diverse-seed generation. Fleet out the
> embarrassingly-parallel joint-world tensor generation across cheap Vast.ai
> GPUs, push chunks to HuggingFace, let the local M5 Max pull and train.
> Reuses the `forge/zeb/vast/` pattern wholesale.

## Why this exists

Right now gen runs on one M5 Max. Each 100-game chunk takes 7-9 min
(contended) or ~7 min (solo). 10k games = ~10 hours wall. Diverse-seed gen
(10 declarations per seed, 10× more state diversity per unit compute) would
multiply that to ~100 hours — infeasible locally.

Vast.ai has interruptible 3090/4090 instances at ~$0.20-0.40/hr. A fleet of
8-16 cheap cards running in parallel would cut wall time from days to hours,
at total cost of $20-40 for a 10k-game diverse-seed corpus.

The workload is trivially parallel: each seed (or seed×decl pair) is
independent. No coordination needed between workers. Just a work queue and
a sink for results.

## Architecture

```
                     ┌──────────────────────┐
                     │  HuggingFace repo    │
                     │  jasonyandell/       │
                     │    gus-42-worlds     │
                     │  (chunks/, eval/)    │
                     └──────────┬───────────┘
                         pull   ▲   push
                                │
   ┌───────────┐     ┌──────────┴───────────┐     ┌───────────┐
   │ Vast #1   │────►│    Work assignment   │◄────│ Vast #N   │
   │ 3090 $.20 │     │  (seed/decl tuples)  │     │ 4090 $.35 │
   │           │     │                      │     │           │
   │ generates │     │  Simplest: each      │     │ generates │
   │ chunks    │     │  worker takes a      │     │ chunks    │
   │ [seed..]  │     │  disjoint seed range │     │ [seed..]  │
   └─────┬─────┘     │  (static partition)  │     └─────┬─────┘
         │           │                      │           │
         │ push      │  No learner needed;  │     push  │
         │ chunks    │  gen is stateless    │           │
         │           └──────────────────────┘           │
         ▼                                              ▼
   ┌─────────────────────────────────────────────────────────┐
   │             HuggingFace chunks/ uploads                 │
   └─────────────────────────────────────────────────────────┘
                                │
                                ▼ pull chunks incrementally
                    ┌──────────────────────┐
                    │   Local M5 Max       │
                    │   - training         │
                    │   - eval             │
                    │   - ad-hoc spikes    │
                    └──────────────────────┘
```

No learner loop. No replay buffer. Workers are pure sources; HF is the sink;
local is the consumer.

## Pieces to build

### 1. Generator hardening (prerequisite)

- **`--n-decl-per-seed N`** flag in `forge/eq/generate/cli.py`. Today's code
  does `decl_ids = [i % 10 for i in range(n_games)]` — one decl per seed,
  rotating. Change to: for each seed in `[start, start+n_seeds)`, emit
  `n_decl_per_seed` rows covering rotating declarations. Per-game output
  becomes `(seed, decl)` keyed instead of just `seed`.
- **Deterministic output paths**: `gus-worlds-seed{start}-{end}-decl{decl_start}-{decl_end}.pt`
  so workers can write without collision. If `--n-decl-per-seed 10`, a worker
  gets both a seed range and all 10 decls for it.
- **CUDA/MPS device auto-detect already in place** (current cli fallback works).

### 2. Work partitioning

**Simplest viable**: static range assignment. 10k seeds × 10 decls = 100k
(seed, decl) jobs. Worker N of M takes `seed in [N*10k/M, (N+1)*10k/M)` and
all 10 decls for it.

- No coordination needed — workers only read their range, write their chunks.
- Idempotent: if a worker dies mid-range, restart and skip existing chunks
  (same `if [ -f "$out" ]; then continue; fi` pattern as `gen_chunks.sh`).
- HF chunks are the source of truth for what's done. A worker that restarts
  should `hf_hub_list_repo_files` its range to skip completed ones.

### 3. Worker boot script

Analogous to `forge/zeb/vast/go.sh`. The worker:
```
1. Clone the repo
2. pip install deps (torch, transformers, forge reqs, huggingface_hub)
3. Download oracle checkpoint from HF
4. Run the chunked gen over assigned seed range
5. Push each chunk to HF as it completes
6. Exit when range is done (or idle if preemptible)
```

Pattern: mirror `forge/zeb/vast/go-experiment.sh` shape. HF_TOKEN via env var.

### 4. HuggingFace dataset repo

**`jasonyandell/gus-42-worlds`** — public dataset.

Layout:
```
chunks/
  seed_0000-0099_decl_0-9.pt         (~11 GB each at 10 decls/seed)
  ...
eval/
  seed_900000-900019_decl_0-9.pt
adapters/
  (optional, if we want to ship trained checkpoints alongside)
README.md
  - schema (the DecisionRecordGPU format)
  - oracle version
  - seed convention (0..899999 train, 900000+ eval)
  - how to load + iterate
```

Incremental uploads via `huggingface-cli upload` per chunk. No single-file
size issue (≤5GB per-file limit; our chunks are ~1-11 GB so might need to
split by decl within a seed range). Likely chunk on 100 seeds × 1 decl ≈
1.1 GB to stay safely under limits.

### 5. Fleet management

Lean on `forge/zeb/vast/vast_{up,monitor,status,down}.sh`. Rename/adapt
minimally:
- `ZEB_REPO_ID` → `GUS_REPO_ID`
- `go-experiment.sh` → `go-gus-gen.sh` (different entry point)
- Fleet name: `GUS_FLEET=gus-gen` for namespacing

### 6. Local consumer

Local script that periodically:
```
1. huggingface-cli download jasonyandell/gus-42-worlds --include "chunks/seed_*.pt"
2. Check which chunks are new
3. (Optional) Retrain student as corpus grows past milestones (5k, 10k, 20k)
```

The existing `JointWorldFullDataset` already accepts a list of paths or a
glob — so training just points at `gus/data/corpus_train_chunk_*.pt` as it
does now, and new chunks pulled from HF just slot in.

## Cost envelope

Generation throughput on M5 Max: ~14 games/min at SEM<0.5 adaptive. On a
3090 (roughly 2-3× M5 Max for MPS-shape transformers): 28-42 games/min.

For a 10k-game diverse-seed corpus (= 100k seed×decl jobs):
- 1 worker × 28 g/min = 60 hours
- 8 workers × 28 g/min = 7.5 hours
- 16 workers × 28 g/min = 3.75 hours

At $0.25/hr × 8 workers × 7.5 hours = **~$15** for an 8-worker fleet.

## Phasing

**Phase 0 (prerequisite, ~1 hour of code)**: add `--n-decl-per-seed` to the
generator. Sanity-check the output schema. Decide on chunk size (seed range
× decl range) that keeps files ≤5 GB.

**Phase 1 (fleet-of-1, ~2 hours of code)**: create `jasonyandell/gus-42-worlds`
HF repo, write `go-gus-gen.sh` worker script, launch ONE Vast instance
pointed at a small seed range (seeds 10000-10099, all 10 decls), watch it
complete and push chunks. Debug end-to-end.

**Phase 2 (fleet, ~1 hour)**: crank the worker count. Reuse
`vast_monitor.sh` for self-healing. Start with 4 workers on interruptibles,
scale up if stable.

**Phase 3 (operationalize, ~1 hour)**: local script that pulls new chunks
from HF into `gus/data/`, and a cron (or cron-like) that checks every hour
and retrains the student when corpus grows past a milestone.

## Division of concerns

- **Local M5 Max**: training, evaluation, ad-hoc spikes. Pulls chunks from HF
  as needed. Never generates except for one-off experiments.
- **Vast fleet**: stateless generation only. Reads oracle ckpt from HF,
  writes chunks to HF, exits. Preemption-safe — lost chunks just get re-run
  by the next worker or a restart.
- **HuggingFace**: durable storage + source of truth for which chunks exist.

## Risks and mitigations

- **Oracle ckpt bandwidth**: each worker downloads ~40 MB ckpt on boot. Fine.
- **HF push contention**: each worker pushes to a different chunk path, so
  no conflicts. Worst case: duplicate upload attempts; HF handles gracefully
  or a `skip-if-exists` flag on the CLI.
- **Interruption**: workers checkpoint at chunk granularity. Lost mid-chunk
  = lose that chunk only. `if [ -f "$out" ]` skips completed chunks on
  restart.
- **Cost runaway**: Vast.ai has per-instance max-bid and auto-shutdown.
  `vast_monitor.sh` patterns already handle this.
- **Oracle version skew**: if we update the oracle ckpt, old chunks are
  stale. Mitigate by versioning the HF repo path:
  `chunks/oracle_v1/seed_*.pt`.

## Related existing infra

- `forge/zeb/vast/` — fleet pattern we're mirroring. Docker, Vast scripts,
  HF push logic.
- `forge/eq/generate/cli.py` — current generator. Needs `--n-decl-per-seed`.
- `huggingface_hub` Python API — already used by forge.
- `forge/models/` — contains the oracle ckpt we need to ship to workers.

## Explicit not-goals

- **Not a learner loop.** Workers don't train; local does. No online RL.
- **Not a coordination service.** Static range partitioning is enough at
  this scale (~100k jobs, workers can cover their own ranges idempotently).
- **Not a training fleet.** Training happens on local or on a single larger
  card when we want speed. Distillation's data-efficient enough that one
  GPU for hours beats many GPUs for minutes.

## Open questions

- Do we want public or private HF repo? User said public is fine (public
  domain game, legitimate ML data). Public also sidesteps the 5 GB private-
  tier limit.
- Should we chunk at 100 seeds × 1 decl (~1.1 GB) or 100 seeds × 10 decls
  (~11 GB, needs file split)? Probably the former — simpler, fits in
  per-file limits cleanly.
- Should the oracle ckpt live on HF or be bundled in the worker docker image?
  HF pull is fine; keeps the image small.
- When we eventually care about distillation cost: could workers ALSO do
  tokenization offline, producing training-ready tensors instead of raw
  game records? Micro-optimization, defer.
