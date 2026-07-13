# Arena perf pass — dispatch/sync reduction on the oracle decision path

**Result: 2.38× games/sec on MPS (0.56 → 1.34 games/s), decisions byte-identical.**
Branch `worktree-agent-a446fb068f6574e91` (worktree of mk5-main, based on
`forge@537b3e2`), commits `838f48d` (levers) + follow-up (dead-code cleanup +
this report). Merge = cherry-pick or merge the branch into `forge`.

## Profile breakdown (before)

cProfile, 8 games net:wp+lens:ev self-play, `--device cpu`, n_samples=10
(8.7 s wall, 1284 `LensPlay.choose` ticks):

| stage | share | per-tick | what it was |
|---|---|---|---|
| oracle forward | 62% | 4.2 ms | `torch._transformer_encoder_layer_fwd` — already batched, untouched |
| world sampling (MRV) | 12% | 0.84 ms | ~45 kernels/step × 21 steps, one dead `.any()` sync/step, `.item()` sync, per-game Python pool loop (1 sync/game) |
| tokenize prep | 7% | 0.46 ms | `build_remaining_bitmasks`: 4×7 Python loop ≈ 170 kernel launches/call |
| state→tensor | 2.4% | 0.16 ms | per-element device writes (hundreds/tick) + boolean-mask sync |
| net:wp bidder | 1.5% | — | confirmed irrelevant (candidate c skipped) |

On CPU the forward dominates; on MPS the forward shrinks and everything else
becomes kernel-dispatch + sync overhead — the measured "40% GPU". The per-tick
fixed cost was ~1,500–2,000 kernel launches and ~25–45 GPU→CPU syncs.

## Levers implemented (all decision-preserving)

The one invariant: **never touch the torch RNG stream** (order, shapes, and
count of `torch.rand` calls). Everything below is deterministic-math
restructuring around it.

1. **`forge/zeb/eq_player.py::zeb_states_to_game_state_tensor`** — assemble
   hands/played/history/trick arrays in numpy, one host→device transfer per
   tensor. Was: per-element device writes + a boolean-mask indexing sync.
2. **`forge/eq/generate/sampling.py::sample_worlds_batched`** — vectorized
   pool construction via sort trick, *byte-identical ascending order preserved*
   (the corpus rejection sampler is order-sensitive); maskless `scatter_add`
   for "my tiles out of pool" (duplicate-safe); `infer_voids_batched` rewritten
   as maskless scatter_add (kills `.any()` + `nonzero` syncs); per-device
   lookup-table cache.
3. **`forge/eq/sampling_mrv_gpu.py::sample_worlds_mrv_gpu`** —
   - removed the per-step `if not active.any(): break`: it can never fire (the
     largest-pool sample stays active through step `max_pool_size − 1` by
     construction), so it was 21 pure syncs per call;
   - optional `max_pool_size` argument (LensPlay supplies it from CPU-side
     ZebGameState) kills the `pool_sizes.max().item()` sync; exact-match
     fallback when not supplied;
   - popcount for MRV slack via 3-kernel bit expansion (was ~12-kernel
     parallel-bit-count) — same integers;
   - random-set-bit inlined with `argmax(cumsum > target)` selection —
     provably the same bit as the old `cumsum == target+1 & bits_set` argmax
     (cumsum increments only at set bits; argmax takes the first max);
   - hoisted per-step scalar-tensor construction; `scatter_add` need-decrement
     (identical values, fewer launches). ~45 → ~20 kernels/step.
4. **`forge/eq/generate/tokenization.py::build_remaining_bitmasks`** — gather +
   bit-pack, ~170 → ~6 kernel launches/call; kept int32 output dtype.
5. **`forge/eq/game_tensor.py`** — per-device cache for
   LED_SUIT/TRICK_RANK/CAN_FOLLOW (`.to(device)` was a fresh host→device copy
   per call, ~4/tick); memoized `current_player` (read ~5×/tick; instances are
   immutable — transitions build new instances via `apply_actions`).

Deleted now-dead `_random_set_bit_vectorized` (no-legacy rule).

## The lever NOT taken (and why): constant batch width / cross-half pooling

The task's candidate (a) — refill/interleave games so the lockstep batch stays
wide — is **impossible under the identical-actions constraint** with this
sampler: `sample_worlds_mrv_gpu` draws `torch.rand(total_samples)` from the
*global* torch generator once per MRV step, where both `total_samples` (batch
composition) and the step count (`max_pool_size`, a batch max) depend on which
games share a tick. Any refill or cross-half pooling changes every subsequent
world sample, hence E[Q], hence actions. Making sampling per-game-deterministic
would equally diverge from the baseline realization.

If byte-identity is ever relaxed (statistically-equivalent games accepted),
topping up the pool so batch width stays constant is the next lever — the
straggler tail (batch decays to width 1–2) is now a *larger fraction* of the
remaining wall since per-tick overhead shrank. Estimate ~1.3–1.5× more.

## Correctness gate (byte-identity)

`per_hand.csv` / `per_game.csv` compared with `cmp` (byte-for-byte), baseline
(`537b3e2` + untouched arena/forge) vs optimized, same seed/args:

- **CPU**: 12 games, seed 0 — identical (`scratch/perf/base12` vs `opt12`,
  re-verified post-cleanup as `opt12b`).
- **MPS**: 32 games, seed 0 — identical in *both* bench pairs
  (`bench_A1`≡`bench_B1`, `bench_A2`≡`bench_B2`; 353 hands each). Same device
  ⇒ same RNG stream ⇒ the optimized code reproduces the baseline actions
  exactly on the production device.

Tests (CPU, MPS hidden via plugin so the reserved GPU stayed untouched during
the corpus job): `arena` + `champion` **95 passed**, 5 MPS-gated tests re-run
on real MPS after GPU release: **5 passed** (3 skips = gitignored belief
adapters absent in the worktree, expected). `forge/eq`: **229 passed**; 7
failures are **pre-existing** — identical list on baseline
(`test_generate_gpu` posterior/enumeration, `test_per_sample_decl_id`
backward-compat; unrelated to this change).

## MPS bench (after GPU release)

32 games net:wp+lens:ev self-play, seed 0, n_samples 10, `--device mps`,
alternating A,B,A,B on an M5 Max (~3.5 min total, under the 10-min cap);
`elapsed_s` from summary.json (match loop only, excludes model load):

| run | code | wall | games/s |
|---|---|---|---|
| A1 | baseline | 56.6 s | 0.566 |
| B1 | optimized | 23.8 s | 1.347 |
| A2 | baseline | 57.0 s | 0.562 |
| B2 | optimized | 24.0 s | 1.333 |

**Speedup 2.38× (2.36–2.38 across pairs; baseline and optimized each <1%
run-to-run).** CPU wall also improved ~1.14× (9.6 → 8.4 s on the 12-game gate)
— consistent with the forward dominating on CPU and overhead dominating on MPS.

## Pass #2: fast batching (byte-identity relaxed by decision, 2026-07-06)

The lever above was taken once byte-identity was explicitly relaxed.
`arena.engine.run_paired` pools BOTH halves of the paired match into one
lockstep batch: deal seeds and opening auctions are per-game deterministic
(`hand_seed` + per-hand auction RNG never see batch composition), so the
paired-seed structure is intact; only realized play diverges. For a fixed
set of games, all-at-once pooling is tick-optimal — total ticks = the
longest game's ticks, which no capped-width refill queue can beat — so the
"refill" lever reduces to this single pooled loop.

Surface: `run_match(fast_batching=...)`, default **off** (the sequential
halves remain the exactly-reproducible regression path, re-verified
byte-identical vs `e04b5bf` on the 32-game MPS gate); `arena.cli
--fast-batching` default **on** (corpus/loop generation), `--no-fast-batching`
to opt out. Fast mode is itself run-to-run deterministic on a fixed device
(same seed → same stream → same regrouping; 32-game per_hand.csv identical
across repeats).

Bench (M5 Max MPS, net:wp+lens:ev self-play, n_samples 10, `elapsed_s`):

| bench | exact | fast | speedup |
|---|---|---|---|
| 32 games, seed 0 | 24.1 s (1.33 g/s) | 15.5 / 14.8 s (2.06–2.16 g/s) | **1.55–1.63×** |
| 128 games, seed 0 | 35.3 s (3.63 g/s) | 27.5 s (4.65 g/s) | 1.29× |
| 128 games, seed 1 | 39.3 s (3.26 g/s) | 27.9 s (4.59 g/s) | 1.41× |

The gain shrinks with n_games, as it must: the straggler tail is a fixed
number of ticks, a larger fraction of small runs. Distribution equivalence
(256 games/mode, seeds 0+1): made-rate 68.8% vs 69.2% (two-prop z, p=0.78),
mark margin −0.10 vs −0.24 (Welch, p=0.64), hands/game 11.00 vs 10.99
(p=0.91) — statistically indistinguishable. Tests: arena+champion suites
green (111 passed + 3 pre-existing worktree skips; 2 new fast-batching
tests).
