---
title: Harvest cohort abstraction
kind: topic
first_seen: 1f48fda
last_updated: 1f48fda
status: active
---

## What this page is

A 1-page proposal for the **cohort** primitive that bridges
[[batched-harvest-resilience]]'s wave-sentinel + quarantine semantics
onto a continuous-batching dispatcher.  Sibling spec to the open
question filed at `wiki/questions/open.md` ("How does the [[batched-
harvest-resilience]] wave-sentinel + quarantine layer migrate to a
continuous-batching dispatcher?").  Companion to
[[continuous-batching-dispatcher-design]] — that page describes the
*inference primitive*; this page describes the *resilience contract*
that wraps it.

## Problem

The current `harvest_batched.py` resilience layer is **wave-aware**.
A wave is a temporal unit: ~6–8 decisions enter `model.step_batch()`
together, every active stream advances one turn at a time, all streams
in a wave finish before the next wave starts.  The three resilience
mechanisms hang off the wave boundary:

- **Quarantine ledger record** = `{wave_idx, gi_range, seeds, error_*}`.
- **SIGKILL sentinel** is written before the wave's first
  `step_batch`, cleared after the wave's last `step_batch`.
- **`--rerun-quarantined`** retries an entire wave's gi list at
  `--retry-batch-size`.

A continuous-batching dispatcher (per [[continuous-batching-dispatcher-design]])
removes the wave abstraction.  Decisions enter the
`BatchGenerator` pool independently, finish independently, re-submit
their next turn independently.  At any moment N streams are in flight
and the membership churns step-to-step.  There is no "wave" to write
into a quarantine record, no "wave start" boundary to write a sentinel
at, no "wave end" to clear it at.

If we just delete the wave abstraction, we lose the recovery contract.
The question is how to keep it.

## Proposal: cohort = a fenced group of decisions

A **cohort** is a fixed-size group of decisions that:

1. Enter the dispatcher pool together.
2. Share **one** quarantine sentinel record.
3. Quarantine **as a unit** if any stream trips a Metal OOM /
   `broadcast_shapes` / `Resource exhausted` while the cohort is in
   flight.
4. Cleared and replaced atomically — finished cohort → next cohort.

Cohort size = the dispatcher's pool width (`completion_batch_size`).
For Burl-scale harvests at batch=8, cohort_size=8.

### Lifecycle

```
for cohort_start in range(0, len(todo_gis), cohort_size):
    cohort_gis = todo_gis[cohort_start : cohort_start + cohort_size]
    cohort_idx = cohort_start // cohort_size

    # 1. Sentinel: same shape as today's wave_in_progress.txt.
    write_cohort_sentinel(harvest_dir, cohort_idx, cohort_gis, ...)

    # 2. Submit every cohort member's first turn.
    states = [init_decision_state(gi) for gi in cohort_gis]
    for st in states:
        dispatcher.submit(stream_id=st.gi, prompt_text=render_prompt(st))

    # 3. Pump until cohort is empty; quarantine the whole cohort on OOM.
    try:
        while dispatcher.in_flight() > 0:
            for f in dispatcher.pump():
                st = states_by_id[f.stream_id]
                apply_step(st, f.completion_text)
                if not st.done:
                    dispatcher.submit(st.gi, render_prompt(st))
    except (RuntimeError, MemoryError) as e:
        ok, kind = _is_oom_like(e)
        if ok:
            quarantine_cohort(harvest_dir, cohort_idx, cohort_gis,
                              error_kind=kind, error_message=str(e))
            for st in states:
                if not st.done:
                    st.done = True
                    st.result_meta["gen_failed"] = True
            # Drop the dispatcher's poisoned state, build a fresh one.
            dispatcher.close()
            dispatcher = ContinuousDispatcher(...)
        else:
            raise

    # 4. Finalize each decision's trace_summary.json (or skip on gen_failed).
    for st in states:
        finalize(st, oracle, wall_s)

    # 5. Clear cohort sentinel — cohort is durable on disk.
    clear_cohort_sentinel(harvest_dir, cohort_idx)
```

The four files from `batched-harvest-resilience` survive byte-for-byte:
`quarantine.jsonl`, `quarantine_resolved.jsonl`,
`quarantine_terminal.jsonl`, `wave_in_progress.txt` (renamed
`cohort_in_progress.txt`).  The orphan-sentinel-detection on resume
still works — same code path, replace `wave_idx` with `cohort_idx`.
`--rerun-quarantined` reads the same ledger and submits the
quarantined gi's at `--retry-batch-size`, which is now just a smaller
cohort.

## What changes vs the wave model

| Aspect | Wave model (today) | Cohort model |
|--------|-------------------|--------------|
| Group unit | Wave = `batch_size` decisions, all at the same turn | Cohort = `batch_size` decisions, possibly at different turns |
| GPU utilization | Idle on the straggler tail of every wave | Continuous; freed slot fills as soon as a stream finishes |
| Quarantine granularity | One wave = one record | One cohort = one record (same field shape) |
| Sentinel boundary | First/last step_batch of the wave | First submit / last finalize of the cohort |
| Cross-cohort GPU sharing | N/A — no cross-wave overlap | None, by design — keeps quarantine semantics clean |

The trade-off is explicit: **fenced cohorts give up cross-cohort
straggler savings to preserve a clean quarantine contract.** When
cohort-N's last 2 streams are finishing their tails, the GPU could in
principle be processing cohort-(N+1)'s first prefill, but that cross-
cohort mixing would entangle their failure modes — an OOM from a
cohort-(N+1) stream would now poison cohort-N's still-in-flight
streams too, and we'd quarantine both.  The cleaner contract wins.

For Burl-scale harvests (560–2000 decisions, cohort=8), the
intra-cohort straggler savings dominate.  A cohort of 8 with turn-counts
{4,4,5,5,6,6,8,8} previously paid `8 × max(turn) × decode_time` on
the GPU; the cohort version pays closer to `mean × decode_time`.  The
inter-cohort gap (model load + per-decision init for the next cohort,
~tens of ms) is in the noise.

## Failure-mode map

Every failure mode the wave model handles, the cohort model handles
identically:

| Failure | Wave behavior | Cohort behavior |
|---------|---------------|-----------------|
| Metal OOM mid-step | classify, quarantine wave, gen_failed all members, continue | classify, quarantine cohort, gen_failed all in-flight members, drop+rebuild dispatcher, continue |
| `broadcast_shapes` (mlx-lm 0.31.2) | same as Metal OOM | same as Metal OOM |
| `Resource exhausted` | same | same |
| `MemoryError` | same | same |
| Non-OOM `RuntimeError` | break wave; mark wave gen_failed; continue (no quarantine record) | break cohort; mark in-flight gen_failed; rebuild dispatcher; continue (no quarantine record) |
| OS SIGKILL | orphan sentinel detected on resume → quarantine that wave | orphan sentinel detected on resume → quarantine that cohort |
| Apply_step exception (unrelated to inference) | per-decision: bail one decision, continue wave | per-decision: bail one decision, cohort keeps decoding the rest |
| `--rerun-quarantined` retry | smaller batch (default 4) over the quarantined gi list | smaller cohort_size over the quarantined gi list — same code |

## Why "drop and rebuild" the dispatcher on OOM

`BatchGenerator` carries internal state — `_unprocessed_sequences`,
`_prompt_batch.prompt_cache`, `_generation_batch.prompt_cache`,
the wired-memory limit context.  After a Metal OOM mid-decode, all of
that is in an unknown state, and continuing to push streams into it
risks compounding the failure.  The wave model handles this by simply
exiting the wave loop body — `step_batch` returns control, and the
next wave starts with a fresh `batch_generate(...)` call (no
persistent state).  Under continuous batching the dispatcher is
long-lived, so we have to explicitly close + reopen it.

`BatchGenerator.close()` releases the wired-memory limit and syncs
the generation stream.  Building a fresh `BatchGenerator` is cheap
(~ms; the model weights are not reloaded).  The cost is paying the
prefill again for any streams that were mid-prefill at OOM time —
those rejoin the next cohort's pool with a fresh pre-decode state.
That cost is bounded by `cohort_size × prompt_tokens / prefill_tps`,
which on M5 Max is ~1 s.

## CLI surface (preserved)

```
--cohort-size N           Default 8.  Replaces --batch-size at the
                          dispatcher layer.
--rerun-quarantined       Same as today.
--retry-cohort-size N     Default 4.  Replaces --retry-batch-size.
--inject-oom-at-cohort N  Test only — replaces --inject-oom-at-wave N.
--resume <dir>            Same as today.  Reads cohort_in_progress.txt
                          + quarantine.jsonl just like wave today.
```

The flag rename is the only user-facing change.  All four ledger
file names stay (`quarantine.jsonl`, `quarantine_resolved.jsonl`,
`quarantine_terminal.jsonl`, `cohort_in_progress.txt`) — the sentinel
filename rename is the only on-disk break, and a one-time migration
script can rename `wave_in_progress.txt` → `cohort_in_progress.txt`
on resume of a pre-cohort harvest dir (defensive; in practice the
v2 [[burl-2000-harvest]] left zero orphan sentinels, so most resumes
won't hit this).

## Acceptance criteria

The migrated `harvest_batched.py` is correct iff:

1. `--inject-oom-at-cohort 3 --cohort-size 8 --limit 24` produces
   a quarantine record with `cohort_idx=3` and `gi_range` of length
   8, drops the dispatcher, rebuilds, and finishes cohorts 0..2 +
   4..n with `gen_failed=False`.
2. `--rerun-quarantined --retry-cohort-size 4` reads the
   quarantine.jsonl, retries the 8 quarantined gi's at cohort_size=4
   (two cohorts), appends 8 records to quarantine_resolved.jsonl
   (or terminal.jsonl on second failure).
3. SIGKILL during a cohort, then `--resume <dir>`, detects the
   orphan `cohort_in_progress.txt`, appends a quarantine record
   with `error_kind="SIGKILL_or_crash"`, clears the sentinel,
   continues from the next unfinished cohort.
4. v2 [[burl-2000-harvest]] reproducer (no OOM injection) runs
   end-to-end with zero quarantine records and **strictly faster**
   wall than the wave version (intra-cohort straggler savings) at
   matched K1.

## Open questions

- Does mlx-lm's `broadcast_shapes` bug fire differently when the
  cohort's prompt-length variance is mid-decode rather than mid-wave?
  Need a smoke that mixes a 200-token continuation prompt with a
  fresh 2400-token first-turn prompt in the same cohort — this is
  the contention pattern the bug originally lit up on at batch≥14.
- Is `BatchGenerator.close()` safe to call mid-`pump()` (e.g. inside
  the OOM `except` block while there are still in-flight streams)?
  Verify with an explicit `gen.remove(uids, return_prompt_caches=False)`
  drain step before close, in case mlx-lm's context manager assumes
  empty pool at close.
- Should cohort_idx be a global counter (durable across resume) or
  reset to 0 each invocation?  Wave model uses session-relative
  `wave_idx` which is fine because waves don't overlap across
  resumes — the orphan detection handles the boundary.  Cohorts
  should follow the same convention; cohort_idx is per-invocation.

## Implementation phasing

Suggested commit sequence on a `perf/harvest-cohort` worktree (not
this scribe's branch):

1. Extract `ContinuousDispatcher` from the inline
   `run_bench_continuous` per [[continuous-batching-dispatcher-design]]'s
   "Current code vs this design" note.  Test against the bench.
2. Port `harvest_batched.py`'s wave loop to a cohort loop using the
   dispatcher.  Preserve every quarantine code path.
3. Pass acceptance criteria 1–3 with `--inject-oom-at-cohort`.
4. Smoke acceptance criterion 4 with a 32–64-decision run; full v2
   reproducer is a separate ticket.

## Links

[[batched-harvest-resilience]] · [[continuous-batching-dispatcher-design]] ·
[[burl-perf-phase2]] · [[burl-2000-harvest]] · [[mlx-lm]] · [[burl]]
