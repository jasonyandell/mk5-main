# Burl — STaR Run-3 Plan: filter-only on the 2000-decision strict pool

**Date**: 2026-04-25 (end of session)
**Status**: Plan, not yet executed. Handoff for next session.
**Branch**: `forge`. Working tree clean as of `54883b3`.

---

## TL;DR

The 2000-decision Burl harvest is in. The corpus is bucket-tagged. Pre-flight
training instrumentation is wired. The next session should run STaR filter-only
on the 1062-row strict pool with conservative hyperparameters (rank=8,
lr=3e-5, 1 epoch, val-loss + early-stop), evaluate on the held-out
sequential 560 baseline, and decide whether to advance to a rationalization
pass on `BURL_BREAKS_CONSENSUS`.

Expected wall: ~1–2 hours on the M5 Max (rank-16 scaling: ~30 min/100 rows
at lr=2e-5; rank-8 at lr=3e-5 is in the same ballpark on a 750-row training
split). Expected outcome: a Burl LoRA adapter that lowers regret on the
sequential-560 eval below the 0.517 deployable Q-mean baseline. If it
doesn't, this is a research negative result; iterate on the corpus filter
or hyperparameters before assuming the recipe is wrong.

---

## What's already done — the prerequisites

1. **Corpus harvested** (`harvest_batched_20260425_072910/`).
   2000 decisions on `D_required_first` variant, `max_tokens=2048`, batch=6,
   5h 46m wall, **0 quarantines, 0 illegal commits**. See
   [`wiki/experiments/burl-2000-harvest.md`](../wiki/experiments/burl-2000-harvest.md).

2. **Bucket tagging done.** `tag_corpus.py` ran the K=200 belief-sampled
   per-decision scoring against `gus/data/corpus_train_chunk_0-99.pt`,
   producing `corpus_index.jsonl` and `HARVEST_SUMMARY.md` in the harvest
   dir. Filter the index to `global_idx < 2000` for harvest-only counts.

3. **Strict pool: 1062 rows.** Of those, 202 are non-trivial gold
   (excludes ALL_AGREE_CORRECT). Sharpest loss bucket is BURL_BREAKS_CONSENSUS
   at 299 rows.

4. **Training instrumentation wired** (`a936fb2`).
   `burl/train/star_mlx.py` now supports `--val-corpus`, `--val-batches`,
   `--early-stop-val-rise FLOAT`, `--early-stop-patience N`, and snapshots
   the lowest-val-loss params so the saved adapter is the best checkpoint
   seen, not the final iteration. 21/21 tests pass.

5. **Min-char filter wired** (prior session).
   `scratch/belief_trajectory_rollout/star/build_filtered_corpus.py` has
   `--min-assistant-chars N` (default 0). At 300, strips ~40% of templated
   short rows uniformly across all buckets; 0 gold-bucket decisions are lost.
   Pre-built corpora exist at:
   - `scratch/belief_trajectory_rollout/star/corpus_strict_20260425/` (no min filter)
   - `scratch/belief_trajectory_rollout/star/corpus_strict_min300_20260425/` (min=300, recommended)

6. **Static review pages** at https://burl-42-review.pages.dev — engineer
   and family-audience versions of the bucket distribution + diagnostics.

---

## Why filter-only first, not rationalization

The prior 71-row STaR collapsed (loss 0.10, adapter looped on history strings,
0/3 eval). The post-mortem identified two memorization-friendly conditions
both present in that run:

- **Tiny corpus.** 71 rows is below the threshold where a rank-16 adapter
  can avoid memorizing per-row idiosyncrasies. 1062 strict + a 300-char
  filter is ~750–800 rows, still on the small side for rank-16 but
  comfortable at rank-8.
- **High lr × high rank × small data interaction.** mlx-lm has no gradient
  clipping; the prior run at lr=1e-4 rank=16 spiked.

Filter-only on the bigger corpus isolates one variable: does Burl learn
*good* reasoning when shown only Burl-already-correct traces? Rationalization
adds the second variable (does the model fabricate plausible reasoning when
the answer is provided?), which the postmortem flagged as poorly understood
in our pipeline. **Solve filter-only first, decide rationalization second.**

---

## Recipe — copy-paste

### Step 1: Build the filtered corpus

The min300 corpus from the 2000-decision batched harvest lives at
`scratch/belief_trajectory_rollout/star/corpus_strict_min300_FROM_HARVEST_BATCHED_20260425_072910/`.
If you need to rebuild against a different bucket selection or filter cutoff:

```bash
PYTHONPATH=. .venv/bin/python scratch/belief_trajectory_rollout/star/build_filtered_corpus.py \
    --harvest scratch/belief_trajectory_rollout/harvest_batched_20260425_072910 \
    --include-buckets ALL_AGREE_CORRECT BURL_ALONE_FIXES BOTH_FIX BURL_INDEPENDENT_RIGHT BURL_FOLLOWS_PI_RIGHT \
    --min-assistant-chars 300 \
    --out-dir scratch/belief_trajectory_rollout/star/corpus_strict_min300_FROM_HARVEST_BATCHED_20260425_072910
```

Output is split into `train.jsonl` and `val.jsonl` (default 80/20). At
min=300 across the 1062 strict-pool decisions that's 1025 surviving
decisions (37 emptied by the filter) → 2686 train rows + 665 val rows
(per-decision rows, not per-decision counts; multiple per-turn rows per
decision is what raises the count above the prior session's prediction).

### Step 1b: Verify before training (BLOCKING)

```bash
jq .harvest_dir scratch/belief_trajectory_rollout/star/corpus_strict_min300_FROM_HARVEST_BATCHED_20260425_072910/manifest.json
```

This must read `scratch/belief_trajectory_rollout/harvest_batched_20260425_072910`.
If it doesn't, stop and rebuild. The wiki has the lesson at
[[burl-2000-harvest]] under "Footgun caught (2026-04-25)" — the prior
session's pre-built corpus was inadvertently sourced from the held-out
eval harvest; manifest-check is the cheapest defense.

### Step 2: Train

**Launch with full session detachment** so a parent-shell death (e.g. `/remote-control`) doesn't kill the trainer:

```bash
TS=$(date +%Y%m%d_%H%M%S)
OUT="scratch/belief_trajectory_rollout/star/adapters/run3_${TS}"
mkdir -p "$OUT"
nohup /usr/bin/env bash -c "exec .venv/bin/python -u burl/train/star_mlx.py \
    --corpus scratch/belief_trajectory_rollout/star/corpus_strict_min300_FROM_HARVEST_BATCHED_20260425_072910/train.jsonl \
    --val-corpus scratch/belief_trajectory_rollout/star/corpus_strict_min300_FROM_HARVEST_BATCHED_20260425_072910/val.jsonl \
    --adapter-out ${OUT}/ \
    --rank 8 \
    --lr 3e-5 \
    --epochs 1 \
    --steps-per-eval 50 \
    --early-stop-val-rise 1.02 \
    --early-stop-patience 2 \
    > ${OUT}/train.log 2>&1" </dev/null >/dev/null 2>&1 &
disown
```

**Flag-name footguns** (caught at the run-3 launch — verify the trainer's argparse before edits):

- The flag is `--corpus`, not `--train-corpus`.
- The flag is `--adapter-out`, not `--out`.
- `--early-stop-val-rise` is **multiplicative**, not additive: `1.02` means "trip when val > 1.02 × best", not "trip on a 0.02 rise". The earlier draft of this plan said `0.02`, which would trip on every eval (val > 2% of best is always true).
- `python` ≠ `.venv/bin/python` on this machine. The mlx wheel is venv-only.

Hyperparameter rationale:

- `--rank 8`: prior 71-row collapse was at rank-16. Rank-8 is the safer
  setting on a sub-1000-row corpus and should still cover the
  reasoning-shape patterns in the strict pool. If the run completes cleanly
  without overfit, a rank-16 follow-up is justified for ablation.
- `--lr 3e-5`: prior collapse was at lr=1e-4. 3e-5 is conservative but
  still enough to land a useful adapter on a 750-row corpus in 1 epoch.
- `--epochs 1`: 1 epoch on 750 rows ≈ 750 grad steps. Enough for
  meaningful drift without overfit risk on rank-8.
- `--steps-per-eval 50`: every 50 grad steps recompute val loss. ~15 evals
  per epoch — sufficient resolution for early-stop without burning eval
  compute.
- `--early-stop-val-rise 1.02 --early-stop-patience 2`: trips when val_loss
  has stayed > best_val_loss × 1.02 for 2 consecutive evals. Conservative —
  catches genuine overfitting but tolerates a single noisy eval.

### Step 3: Eval

The held-out sequential 560 corpus is the strongest baseline because it
predates the harvest and uses `max_tokens=8192` so there's no
chat-template/truncation confound. Use the existing rollout harness with
`--adapter-path` pointing at the run-3 LoRA:

```bash
PYTHONPATH=. .venv/bin/python scratch/belief_trajectory_rollout/star/eval_adapter_smoke.py \
    --adapter-path scratch/belief_trajectory_rollout/star/adapters/run3_<timestamp>/ \
    --eval-corpus harvest_20260424_133611/  \
    --variant D_required_first \
    --max-tokens 2048 \
    --batch-size 6 \
    --out scratch/belief_trajectory_rollout/star/eval/run3_eval_seq560/
```

(`eval_adapter_smoke.py` already exists from a prior session; check its
flags before running and adapt as needed.)

Then re-tag against `per_decision_eval_k200.jsonl` and compare bucket
distributions to the unadapted 560 baseline.

---

## Success criteria

A clean win for run-3 looks like:

1. **Training survives.** Val loss goes down for several evals, then
   plateaus. No collapse to ≤0.10. Best-checkpoint snapshot fires at least
   once.
2. **Strict-pool eval matches.** The adapter, evaluated on the same strict
   pool it trained on, picks the gold-bucket play at >90% rate. (Sanity
   check; if this fails, training didn't take.)
3. **Held-out improvement.** On the sequential 560:
   - Mean Burl regret drops below 0.517 (the deployable Q-mean baseline)
     by a meaningful margin.
   - `matches_bot` rate increases over the unadapted 560 (currently ~52%).
   - `BURL_BREAKS_CONSENSUS` count drops vs unadapted — that's the bucket
     STaR is most directly trying to fix even without rationalization.
4. **No new failure modes.** Adapter doesn't introduce illegal commits
   (Phase A guards should keep this at 0; if not, something is wrong with
   the harness, not the adapter). Forced-commit rate stays ≤12%.

A partial win is also fine to publish: regret unchanged but
`BURL_BREAKS_CONSENSUS` shrinks, or matches_bot up but regret unchanged
(decision quality up at the median, tail unchanged). Negative results
should be filed as a wiki experiment page either way.

---

## Failure modes to watch for

- **Loss collapse.** If train loss drops below 0.15 in the first 100
  steps, abort. The corpus has memorizable boilerplate beyond what
  min=300 catches; investigate before retrying.
- **Val loss diverges from train loss early.** Rank-8 should be capacity-
  bounded for this corpus size. Divergence in the first 100 steps means
  lr is too high; halve to 1.5e-5 and retry.
- **Adapter loops on history strings.** Spot-check `eval_adapter_smoke.py`
  output by hand on 5 decisions; if the model emits literal substrings
  from train decisions, the rank/data balance is still off.
- **Held-out regret unchanged or worse.** Most informative outcome — the
  recipe runs cleanly but doesn't transfer. This is the case where
  rationalization on `BURL_BREAKS_CONSENSUS` becomes the next-most-promising
  intervention, since filter-only's signal didn't ladder out of the strict
  pool.

---

## Stretch goal — rationalization on BURL_BREAKS_CONSENSUS

Only attempt **after** filter-only shows a clean signal. Rationalization
on the 299-row sharpest-loss bucket is a much riskier pass: the model is
shown the correct play and asked to write the trajectory that would have
led there. The danger is fabrication — the model writes plausible
reasoning that doesn't connect to its actual policy, the trajectory is
trained in, and the adapter becomes confidently wrong on similar positions.

The candlewax workstream's `reasoning-coherence-verification` topic
(see `wiki/topics/reasoning-coherence-verification.md`) names this as
the open bottleneck. Don't pursue rationalization without a verifier in
the loop.

---

## Pointers

- **Wiki entry for the harvest:** [`wiki/experiments/burl-2000-harvest.md`](../wiki/experiments/burl-2000-harvest.md)
- **Wiki entry for STaR:** [`wiki/topics/star.md`](../wiki/topics/star.md) — has the "Burl 2000-decision corpus ready" section explaining recipe lessons
- **Wiki entry for the resilience layer:** [`wiki/topics/batched-harvest-resilience.md`](../wiki/topics/batched-harvest-resilience.md) — relevant if a re-harvest is needed
- **Decision page:** [`wiki/decisions/max-tokens-2048-floor.md`](../wiki/decisions/max-tokens-2048-floor.md) — locks the token-cap floor
- **Static review (engineer):** https://burl-42-review.pages.dev/HARVEST_REVIEW.html
- **Static review (family):** https://burl-42-review.pages.dev/
- **Trainer:** [`burl/train/star_mlx.py`](train/star_mlx.py)
- **Corpus builder:** `scratch/belief_trajectory_rollout/star/build_filtered_corpus.py`
- **Eval smoke:** `scratch/belief_trajectory_rollout/star/eval_adapter_smoke.py`
- **Postmortem of the 71-row collapse:** `scratch/belief_trajectory_rollout/star/POSTMORTEM_002918.md`
- **Corpus comparison (min0 vs min300):** `scratch/belief_trajectory_rollout/star/CORPUS_COMPARISON.md`

---

## Footguns

- **Pre-built corpora are claims, not facts.** The prior session's
  `corpus_strict_min300_20260425/` was built from
  `harvest_20260424_133611` — the 560-decision held-out eval set, not
  the 2000-decision batched harvest. Caught at scout time on 2026-04-25
  by reading `manifest.json`. Bad dir renamed
  `..._FROM_HELD_OUT_EVAL_DO_NOT_TRAIN`. Always verify the manifest's
  `harvest_dir` before training on a "pre-existing" corpus. See
  [[burl-2000-harvest]] "Footgun caught (2026-04-25)" and the recipe
  lesson on [[star]].

## Postmortem — run-3 first attempt (2026-04-25)

Three launch attempts, three failures, no adapter on disk. Full account
in [`wiki/experiments/burl-star-run3.md`](../wiki/experiments/burl-star-run3.md).
TL;DR:

- Attempt 1 died when the parent Claude Code shell was renamed (children
  killed). Detachment via `nohup setsid` is now in §Step 2 above.
- Attempt 2 used bare `python` (no mlx). Now also called out in §Step 2.
- Attempt 3 detached cleanly, ran 37 minutes, val loss 2.354 → 0.302
  over 9 evals (no collapse, no early-stop trip), then crashed at iter
  487/1343 with `RuntimeError: [metal::malloc] Resource limit (499000)
  exceeded` inside the cosine LR scheduler. The `train_mlx` exception
  handler only catches `EarlyStopRequested`, so the in-memory best
  snapshot was lost.

**Three blockers before run-3b:**

1. **Resumable checkpointing on `star_mlx.py`** — periodic on-disk
   adapter writes every N optimizer steps + `--resume-from <adapter-dir>`.
   This is the priority fix (per the resumability project memory) and
   gates everything else.
2. **Generic-exception catch in `train_mlx`** — serialize the in-memory
   best snapshot on any crash, not just `EarlyStopRequested`. Cheap;
   land alongside (1).
3. **Reduce peak-memory headroom** — drop `--max-seq-length 4096 → 2048`
   first (the corpus's p99 turn is ~600 tok per
   [[max-tokens-2048-floor]]); fall back to `--batch 1` if 2048 still
   OOMs.

## When you finish

1. Update [`wiki/topics/star.md`](../wiki/topics/star.md) with the run-3
   result section. New page goes under `wiki/experiments/burl-star-run3.md`
   if results are worth their own page.
2. Append an ingest entry to [`wiki/log.md`](../wiki/log.md) with the run-3
   verdict. Bump `wiki/index.md` if any new pages landed.
3. If the adapter ships, add an entity page under
   `wiki/entities/burl-star-run3-adapter.md` (or whatever name reflects
   the result).
4. If a re-harvest is needed, the resilience scaffold at
   `scratch/belief_trajectory_rollout/harvest_batched.py` is ready to
   reuse. Default-flag behaviour is byte-identical to the v2 launch
   command — change `--limit` and `--corpus-path` to scale.
