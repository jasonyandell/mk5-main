---
title: Continuous batching dispatcher — design
kind: topic
first_seen: 0310b12
last_updated: 0310b12
status: active
---

## What this page is

Design doc for the continuous-batching dispatcher referenced from
[[burl-perf-phase2]] and [[perf-on-the-table]].  Written during scribe-A's
GPU-contention pause so the design lands cleanly without depending on
specific wall-time measurements.  When the pause lifts, the same code
gets re-validated; this doc stays put.

## Problem

`burl/eval/run_move4_star_rollout_batched.py` and
`scratch/belief_trajectory_rollout/harvest_batched.py` both use a
sync-wave loop:

```
for wave in waves:
    while any(not s.done for s in wave):
        active = [s for s in wave if not s.done]
        completions = batch_generate(active)  # all of wave's active turn-K
        for s, c in zip(active, completions):
            apply_step(s, c)                  # CPU: tools + state
```

Two costs:

1. **Within a wave**, `batch_generate` waits for *every* active stream's
   turn-K to finish before any stream's turn-K+1 starts.  A 4-turn
   decision blocks at the GPU on the 8-turn decision's last 4 turns.
2. **Across waves**, the sync barrier between wave-N's last turn and
   wave-N+1's first turn idles the GPU during model-load + per-decision
   `_init_decision_state` setup for the next wave.

mlx-lm 0.31.2 ships `BatchGenerator` (`mlx_lm/generate.py:1486`) with
the primitives to remove both costs:

- `insert(prompts, max_tokens, ...)` adds streams to a deque
  (`_unprocessed_sequences`).
- `_next()` runs one prompt-step + one decode-step on the union of
  the two batches, automatically promoting prompt-finished streams to
  the generation batch and pulling unprocessed streams into the prompt
  batch.
- `next_generated()` returns finished tokens (with
  `finish_reason == "stop" | "length" | None` per token) so the caller
  knows when a stream is ready to be retired and replaced.
- `prefill_batch_size`, `completion_batch_size` cap the per-step
  width.

That's continuous batching.  We don't need to write a scheduler; we
need to *use* the one mlx-lm already exposes.

## API surface for Burl

A single class, `ContinuousDispatcher`, that owns the
`BatchGenerator` lifecycle and exposes a small, testable API:

```python
class ContinuousDispatcher:
    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        *,
        completion_batch_size: int = 8,
        prefill_batch_size: int = 8,    # batch>=14 broadcast bug guard
        max_tokens: int = 8192,
        sampler=None,
    ): ...

    def submit(self, stream_id: Hashable, prompt_text: str) -> int:
        """Tokenize prompt_text; insert into the generator;
        return the mlx-lm uid.  Caller maps stream_id <-> uid."""

    def pump(self, until: Literal["any", "all"] = "any") -> list[Finished]:
        """Drive next_generated() until at least one stream
        (or all streams) finishes.  Returns list of Finished records:

            Finished(stream_id, uid, completion_text, finish_reason).
        """

    def in_flight(self) -> int: ...  # for backpressure decisions

    def close(self) -> None: ...     # release wired_limit, sync stream
```

`Finished` is a small dataclass; the dispatcher hides mlx-lm's
`Response` shape from callers so we can change mlx-lm pinning without
breaking the harness.

The dispatcher does **not** know about `_DecisionState`, `apply_step`,
or any harness logic.  It is a pure inference primitive.  This keeps
[[batched-harvest-resilience]]'s OOM/quarantine layer disjoint and
gives the bench a stable inference interface.

## Bench harness loop

```python
dispatcher = ContinuousDispatcher(
    model.model, model.tokenizer,
    completion_batch_size=batch,
    prefill_batch_size=min(batch, 8),
    max_tokens=model.max_tokens,
    sampler=model._sampler,
)

# Submit initial turn for every decision in the run.
states_by_id = {st.gi: st for st in initial_states}
for st in initial_states:
    prompt = render_prompt(st)
    dispatcher.submit(stream_id=st.gi, prompt_text=prompt)

# Drive until empty.
while dispatcher.in_flight():
    for f in dispatcher.pump(until="any"):
        st = states_by_id[f.stream_id]
        apply_step(st, f.completion_text, ...)
        if not st.done:
            dispatcher.submit(stream_id=st.gi, prompt_text=render_prompt(st))

dispatcher.close()
```

Every decision flows through one dispatcher.  Inter-wave barriers
disappear; intra-wave straggler tail disappears.  The CPU work
(tools + state) is the *only* synchronous cost between turns of a
single decision, and it's serialized only against that decision —
other streams keep decoding on the GPU.

## Production harvest path

`harvest_batched.py` carries [[batched-harvest-resilience]]'s wave-based
plumbing:

- `wave_in_progress.txt` sentinel written before the first generate
  call of each wave; cleared after the wave's last generate; SIGKILL
  recovery quarantines whatever wave was active when the process died.
- `quarantine.jsonl` records every `gi_range` whose wave hit Metal
  OOM / `broadcast_shapes` / `Resource exhausted`.
- `--rerun-quarantined` retries quarantined gi's at half batch size.

In continuous mode, "wave" no longer exists as a temporal unit.  The
right primitive is a **cohort**: a fixed-size group of decisions that
enter the dispatcher together, share a sentinel, and quarantine
together if any of them trips an OOM.

```
cohort_size = batch    # dispatcher's pool size
for cohort_start in range(0, len(todo), cohort_size):
    cohort = todo[cohort_start : cohort_start + cohort_size]
    write_cohort_sentinel(cohort)
    for st in cohort:
        dispatcher.submit(st.gi, render_prompt(st))

    try:
        while dispatcher_in_flight_for_cohort(cohort) > 0:
            for f in dispatcher.pump():
                ... apply_step ... maybe submit next turn ...
    except (OOMError, MLXBroadcastError) as e:
        quarantine_cohort(cohort, e)
        dispatcher.close()
        dispatcher = ContinuousDispatcher(...)  # fresh state
        continue

    clear_cohort_sentinel(cohort)
```

This preserves the resilience contract while delivering the
intra-cohort straggler-tail savings.  The trade-off vs. fully-mixed
continuous batching: when one cohort's OOM aborts, in-flight streams
from other cohorts (if we allowed cross-cohort mixing) would be lost
too.  By fencing cohorts, we lose the cross-cohort straggler savings
in exchange for a clean retry semantic.

For Burl-scale harvests (560–2000 decisions, cohort ≈ 8), the
intra-cohort savings dominate: a cohort of 8 with turn-counts
{4,4,5,5,6,6,8,8} previously paid 8 × max(turn) × decode_time on the
GPU; under continuous, fast streams roll into early-finish slots and
the wall is closer to mean × decode_time.  Cross-cohort would shave
the inter-cohort gap (tens of ms per cohort), which is in the noise.

## OOM resilience checklist

The continuous version must preserve every property of
[[batched-harvest-resilience]]:

- [ ] `_is_oom_like(exc)` classifier is the same; cohort handler wraps
      `dispatcher.pump()` with the same try/except.
- [ ] On classify-True: append to `quarantine.jsonl`, mark every
      stream in the cohort `gen_failed=True`, drop the dispatcher,
      build a fresh one, advance to next cohort.
- [ ] On classify-False (unfamiliar `RuntimeError`): same as today —
      mark cohort gen_failed, break out, no quarantine record.
- [ ] Cohort sentinel write/clear identical to wave sentinel; SIGKILL
      recovery on resume reads the orphan sentinel and quarantines.
- [ ] `--rerun-quarantined --retry-batch-size 4` opens a smaller
      dispatcher pool and submits the quarantined gi's; same
      resolved/terminal accounting.

## MLX batch>=14 broadcast bug guard

User memory + [[batched-harvest-resilience]] note that mlx-lm 0.31.2's
chunked-prefill path crashes with a `broadcast_shapes` error at
`prefill_batch_size=8` (the default) when the batch is >=14 with
heterogeneous prompt lengths (>2300-token variance).  The workaround
is `prefill_batch_size=2` or `batch=8`.

The dispatcher pins `prefill_batch_size = min(cohort_size, 8)` by
default and exposes the knob.  At cohort sizes ≤ 8 (Burl's normal
shape), the bug is not reachable.  At larger cohorts, the dispatcher
either drops to `prefill_batch_size=2` or splits the cohort.

## Open questions

- How does prefill_batch_size=2 vs cohort_size=8 interact with the
  dispatcher's continuous re-injection?  When a stream finishes and
  a new one is submitted, is it queued into the prefill batch (size
  2) and decoded only after the first 2-stream prefill completes?
  If so, there's a hidden serial cost to re-injection at small
  prefill widths.  Verify on instrumented run.
- Does `BatchGenerator.close()` between cohorts release the model's
  KV pool cleanly, or does it leak?  Test with a smoke that creates
  + closes the dispatcher 100× and watches `mx.get_peak_memory()`.

## Current code vs this design

What's in `perf/batch` today (commit `29da3d2`): `run_bench_continuous`
in `burl/eval/bench_decision_latency.py` is an inline implementation
of the bench-harness loop.  It instantiates `BatchGenerator` directly,
manages `uid_to_state` and `uid_to_token_buf` dicts, and pumps
`gen.next_generated()` until the state map is empty.  The "submit /
pump / close" abstraction this page describes is **not yet extracted**
into a `ContinuousDispatcher` class.

That extraction is the right next step (separate concerns: dispatcher
owns inference, harness owns state machine + tools), but it's not
required to validate the lever.  The validation runs on the inline
code; the refactor is a simplification commit afterward.  The design
doc is here so the abstraction is settled before the refactor —
saves churn when production-harvest migration starts.

## Validation plan (post-pause)

1. Re-baseline `baseline-bf16-t0` clean (no parallel scribe load).
   Two consecutive runs to confirm <1% wall variance.
2. Re-run `prefix-cache` with the same uncontended window.  Expected
   delta direction: still negative on M5 Max for the reasons in
   "Lever 1 — root-cause writeup", but the magnitude needs the clean
   number to write down.
3. Re-run `continuous-batching` with the same uncontended window,
   three trials.  Expected delta direction: positive (1.3–2×); the
   precise number again needs a clean window.  K1 stability gate:
   ≥ 4/5 vs the clean baseline-t0 (recognising the kernel-noise
   widening at gi=72).
4. If continuous's clean win is ≥ 30%, declare Phase 2 done.  If
   the win is in the 10–30% band, document the partial win and
   defer the full harvest migration to a follow-up.
5. Phase-exit gate (full560) deferred to scribe-C / Phase 4 because
   it's a 2 hour run and the 5-row gate already discriminates the
   levers' direction.

## Links

[[burl-perf-phase2]] · [[perf-on-the-table]] ·
[[batched-harvest-resilience]] · [[batch-throughput-bench]] ·
[[mlx-lm]]
