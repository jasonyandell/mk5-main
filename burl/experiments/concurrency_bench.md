# T13 — Client-side concurrency for `run_move4_star_rollout`

## TL;DR

- `run_move4_star_rollout.py` is now async under the hood, with a new
  `--concurrency N` flag (default **1** — backwards compatible, byte-
  identical output for existing invocations).
- Local unit-test benchmark (mocked 0.1s/decision, N=8):
  **seq=0.809s, par4=0.207s → 3.92× speedup**.
- Modal benchmark is DEFERRED — do not fire while iter-3 rollouts (T11/T12)
  are live on the same endpoint. Plan below; team-lead fires manually
  after those finish.

## Design

### What changed

- **Single flag.** `--concurrency N` (int, default 1). At N=1 the code
  path degenerates to the original sequential loop; at N>1 it schedules
  up to N decisions in flight against the endpoint at once.
- **async all the way down.** `run_star_rollout` is now an async def;
  the CLI `_main()` wraps it in `asyncio.run(...)`.
- **Concurrency mechanism.** Each decision's work unit (`_run_one_rollout`
  for Phase A, `_gate_chain` for Phase B) is pushed onto a worker thread
  via `asyncio.to_thread(...)`. `asyncio.gather` fires up to N at a time.
  The blocking network call inside the Modal adapter (`server.generate_native.remote(...)`)
  releases the GIL during I/O, so N threads genuinely parallelize.
- **Batched scheduling.** We iterate the dataset in chunks of size N,
  `gather` the chunk, then write traces + check cost-cap before
  scheduling the next chunk. This preserves:
  - **Ordering.** `asyncio.gather` returns results in schedule order;
    we write to `rollout_traces.jsonl` in that same order, so trace
    ordering matches dataset ordering exactly.
  - **Cost-cap semantics.** The cap check fires after each batch.
    Once tripped, the outer loop breaks; no further work scheduled.
    Worst-case overshoot is (N − 1) extra decisions in the batch that
    tripped the cap (same as the existing "one decision" overshoot at
    concurrency=1, scaled by batch size).
- **Phase B symmetry.** The EQ-gate retry chain for a single loss is
  bundled into one awaitable (`_gate_chain`), so a loss with
  `max_gate_retries=3` doesn't consume three concurrency slots — it
  consumes one for the duration of its own internal retry sequence.
  This matches the semantic intent: concurrency is between decisions,
  not across a single decision's internal loop.

### What did NOT change

- `_run_one_rollout`, `_gate_one`: still sync. These are the per-decision
  work units; threading them in is mechanical, so no reason to cascade
  async through the rest of the harness.
- Output file formats: `rollout_traces.jsonl`, `gate_traces.jsonl`,
  `<corpus>_stats.json` are identical.
- `BurlTrace`, `_build_record`, `_classify`, `_lookup_decision`: untouched.
- Every existing CLI flag: unchanged defaults, unchanged semantics.

### Why `asyncio.to_thread` and not a native async model_fn

The Modal client's `.remote(...)` is sync at the adapter layer and
doesn't expose an async variant. `asyncio.to_thread` is one line of
glue, preserves the existing sync harness, and only costs one thread
per in-flight decision (at most `concurrency` threads alive at once).
For a workload bounded by a 4-slot vLLM batch on Modal, the thread-
count ceiling is 4. No reason to rewrite the harness.

## Local benchmark (mocked, no Modal)

Test: `burl/eval/test_rollout_concurrency.py::test_concurrency_speedup`.

Setup:
- N=8 synthetic decisions.
- `_run_one_rollout` monkey-patched to `time.sleep(0.1)` + return a
  minimal `BurlTrace` — simulates a 100 ms decision, matches the
  rough scale of the real Modal round-trip at Gemma 4 E2B.
- Same stubbed Modal app for both runs.

Results (from the passing test run):

| concurrency | wall time | per-decision | speedup |
|-------------|-----------|--------------|---------|
| 1           | 0.809s    | 0.101s       | 1.00×   |
| 4           | 0.207s    | 0.026s       | **3.92×** |

Theoretical ceiling at N=4: 4.0× (8 decisions in 2 waves of 0.1s).
Observed 3.92× reflects asyncio.run + thread-spawn overhead.

Other tests in the file:
- `test_concurrency_preserves_ordering`: at concurrency=4 over N=8,
  `rollout_traces.jsonl` lines come out in dataset order (seed 900000
  through 900007). PASSES.
- `test_concurrency_one_is_sequential`: at concurrency=1, behavior +
  trace ordering matches a plain sequential loop (verified on N=4).
  PASSES.
- `test_cost_cap_stops_scheduling`: with `cost_cap_usd=1e-9` and N=12
  at concurrency=4, only the first batch (4 traces) lands on disk;
  no subsequent batches scheduled. PASSES.

Full `pytest burl/ -q`: **105 passed**.

## Plan for the Modal benchmark (team-lead fires manually)

**Do not run while iter-3 (T11, T12) rollouts are live on the vLLM
endpoint** — they share `max_inputs=4` and would contend for GPU.

### Proposed bench

Once iter-3 rollouts finish, fire twice with different concurrencies on
a short subset:

```bash
# Baseline (sequential) — 5 decisions, primer on.
python -u -m burl.eval.run_move4_star_rollout \
  --dataset burl/eval/data/move4_decisions_n50.jsonl \
  --out-dir scratch/bench_seq \
  --corpus scratch/bench_seq_corpus.jsonl \
  --n 5 --max-turns 6 --max-retries 3 \
  --cost-cap-usd 0.20 --concurrency 1

# Concurrent — same subset, concurrency=4 (Modal max_inputs ceiling).
python -u -m burl.eval.run_move4_star_rollout \
  --dataset burl/eval/data/move4_decisions_n50.jsonl \
  --out-dir scratch/bench_par \
  --corpus scratch/bench_par_corpus.jsonl \
  --n 5 --max-turns 6 --max-retries 3 \
  --cost-cap-usd 0.20 --concurrency 4
```

### What to record

For each run, the stats.json + stdout give:
- `wall_time_seconds`
- `estimated_usd`
- `n_wins`, `n_legal_losses`, `n_illegal`, `n_exhausted`
- `tool_histogram_rollouts`

### Expected outcome

- **Wall-time ratio**: par4 ≈ seq / k, where k ≤ 4. Modal vLLM
  continuous batching plus vLLM's own scheduler mean the real-world
  k is typically 2.5-3.5× on a shared L4, not the theoretical 4×.
  Anything ≥ 2.5× is a win.
- **Correctness**: `n_wins + n_legal_losses + n_illegal + n_exhausted`
  must match between runs (same dataset slice), and tool histograms
  should be very similar (Gemma's greedy decode is deterministic at
  temperature=0, but we're at 0.6 so some sampling drift is expected).
  Flag any >10% divergence in bot-match rate as a correctness regression.
- **Cost**: estimated $0.08-0.10 per run (5 decisions × ~$0.016 on an L4
  at wall under 60s/run). Budget $0.25 total for the bench.

### Red flags to watch for

- If wall-time ratio is < 1.5×, something is wrong — either the endpoint
  is queuing (max_inputs really is the ceiling) or our `asyncio.to_thread`
  isn't actually parallelizing. Check: does the Modal dashboard show 4
  concurrent inputs during the `concurrency=4` run?
- If bot-match rate differs by > 10%, the concurrency path is subtly
  corrupting state somewhere. Compare `rollout_traces.jsonl` seed-by-seed
  between the two runs — the order must match (we write in dataset order
  regardless of completion order).

## Files

- `burl/eval/run_move4_star_rollout.py` — +~80 lines net (added async,
  `--concurrency`, batched gather; removed nothing except the inline
  sync for-loop bodies).
- `burl/eval/test_rollout_concurrency.py` — new, 4 tests, ~230 lines.
- `scratch/burl_p5_iter2_prep/concurrency_bench.md` — this file.
