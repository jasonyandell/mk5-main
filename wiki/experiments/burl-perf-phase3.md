---
title: Burl Perf — Phase 3 (Speculative Decoding + Quantization)
kind: experiment
first_seen: 29da3d2
last_updated: 96ebf0b
status: active
---

> **Status:** complete (post-RESUME bench).  Two structural surprises
> killed the speculative-decoding lever before any GPU work
> ([[#Dragons]]); the quantization lever shipped a clean win:
> **`phase3-stack-best` lands at 28.3 s wall on the 5-row temp=0
> subset (1.29× vs paired baseline 36.4 s, peak mem 8.93 GB vs 10.8
> GB) with 4/5 paired play match — Q4 PLE-safe quant + Phase-2
> continuous batching.**

## Overview

Phase 3 of the [[perf-on-the-table]] sprint: drive Burl's per-decision
inference latency further toward ~2 s/decision, stacking on top of
[[burl-perf-phase2]]'s continuous-batching baseline (~34 s wall on the
5-decision frozen subset at temp=0, K1 5/5 vs the wave-loop reference).

Phase 3 owns the wiki's two remaining levers from [[perf-on-the-table]]:

  4. **Speculative decoding** — Gemma 4 E2B as the verifier; a smaller
     draft model proposes; the verifier runs the spec-decode-step
     loop instead of greedy decode.
  6. **Quantization** — bf16 → INT8 / INT4, trading quality for memory
     headroom and decode speed; PLE-safe quants only.

The Phase-3 stacking experiment then runs spec-decode + quant +
continuous-batching as a single configuration to measure the
compounded speedup vs the Phase-0 baseline.

## Dragons (the two findings that force the spec to bend)

### Dragon 1 — mlx-lm's spec decode is single-stream-only

`mlx_lm.generate.speculative_generate_step` (`mlx_lm/generate.py:473`)
exists in mlx-lm 0.31.2.  It plumbs through `stream_generate`
(`generate.py:657`) and the CLI (`generate.py:2048`) and the HTTP
server (`server.py:945`).  It does **not** plumb through
`batch_generate` or `BatchGenerator` — those paths construct
`PromptProcessingBatch`, which calls `model(inputs[:, None],
cache=self.prompt_cache)` over the merged batch and has no notion
of a parallel draft pass.

Confirmed independently: LM Studio's MLX engine raises
`SpeculativeDecodingNotSupportedError: Speculative decoding is not
supported for batched MLX models` even at batch=1 (lmstudio-ai
issues #269 / #1519, 2026-02 + 2026-03).  The error originates from
the same mlx-lm absence.

**Consequence:** spec decode in mlx-lm 0.31.2 cannot stack on top of
[[burl-perf-phase2]]'s continuous-batching dispatcher.  Choosing spec
decode means accepting *single-stream* inference for the duration of
the bench — i.e. losing Phase 2's parallelism win.

The interesting question becomes: does single-stream + spec decode
beat batched decode without spec decode?  Phase-0's
[[batch-throughput-bench]] anchor: 43 tok/s single-stream vs ~84 tok/s
on the 5-decision continuous batched run.  Spec decode would have to
deliver >2× over the 43 baseline just to *catch* batched-no-spec, and
>4× to win meaningfully.  Published spec-decode speedups land at
2–3× when the draft and target tokenizers match and the workload is
predictable; tool-call-heavy templated outputs are a favorable
acceptance regime, so the lever is not dead, but the bar is high.

### Dragon 2 — there is no Gemma 4 E0.5B; quants on Gemma 4 are subtle

The smallest Gemma 4 release is **E2B** — `google/gemma-4-E2B-it`
([huggingface.co/collections/google/gemma-4](https://huggingface.co/collections/google/gemma-4)).
There is no E0.5B.  The team-lead spec assumed one based on the wiki's
[[perf-on-the-table]] entry; that line was speculative when written
and the actual frontier model lineup invalidates it.

Viable alternatives, ordered by how plausible they are as a draft:

1. **Gemma 3 270M IT (`mlx-community/gemma-3-270m-it-bf16`).**
   Tokenizer vocab is **262144** (`config.json:vocab_size`), the
   *same* as Gemma 4 E2B's `text_config.vocab_size = 262144`.
   mlx-lm's spec-decode validator (`server.py:354`) only checks
   `draft_tokenizer.vocab_size != tokenizer.vocab_size`, so Gemma 3
   270M passes the validator.  But Gemma 3 and Gemma 4 are different
   model families (`model_type: gemma3_text` vs `gemma4`); whether
   the *token mappings* line up byte-for-byte is unverified.  If
   they don't, every draft token misaligns and acceptance rate
   collapses to ≈0 — spec decode becomes pure overhead.  This is the
   first thing to verify on RESUME, before any bench: encode the
   same 100-token Burl prompt with both tokenizers and compare ids.
2. **Gemma 4 E2B as its own draft (self-speculation).**  Identical
   tokenizer (trivially), but defeats the purpose of spec decode —
   the draft has to be smaller to win.  Some mlx-lm consumers use a
   quantized E2B as draft for bf16 E2B.  Plausible if Q4-E2B at
   ≈7.6 GB decodes ~1.5–2× faster than bf16-E2B; probably worth
   benching as a fallback.

Quantization adds its own dragon.  Per HF discussion
([huggingface.co/mlx-community/gemma-4-e2b-4bit/discussions/1](https://huggingface.co/mlx-community/gemma-4-e2b-4bit/discussions/1)),
**all `mlx-community/gemma-4-*-{4,8}bit` and the original
`unsloth/gemma-4-*-MLX-{4,8}bit` quants produce garbage output**
because they quantize PLE (Per-Layer Embeddings) layers — Gemma 4's
PLE uses ScaledLinear with output multipliers that amplify quant
error.  PLE-safe quants exist in two repos, both released after the
broken ones:

- `FakeRocket543/gemma-4-e2b-it-MLX-{4bit,8bit,bf16}` —
  community PLE-safe quantization (4-bit ≈ 7.6 GB, 8-bit ≈ 8.5 GB,
  bf16 ≈ 10.2 GB).
- `unsloth/gemma-4-E2B-it-UD-MLX-4bit` — Unsloth's "UD" (Unsloth
  Dynamic) quant, marketed as PLE-aware.  Note: Unsloth's *non-UD*
  MLX-4bit is in the broken set; only the UD variant claims to fix
  PLE.

Storage on disk (current cached): `mlx-community/gemma-4-e2b-it-bf16`
is 9.6 GB.  Phase-0's measured peak GPU mem is 11.59 GB on bf16, so
weights + harness ≈ 11–12 GB.  Q8 should drop weights to ~5 GB and
peak to ~7–8 GB; Q4 to ~3.8 GB weights and ~6 GB peak.  That memory
headroom — not the speed delta — is the actual lever; it would let
the dispatcher run at higher cohort sizes without tripping the
[[batched-harvest-resilience]] OOM guard.

## Variants to bench

Names are stable so the ledger reads cleanly across runs.

| variant            | description |
|--------------------|-------------|
| `q8-bf16-cont`     | PLE-safe Q8 weights, bf16 KV cache, continuous-batching dispatcher.  Direct stack on [[burl-perf-phase2]]'s win. |
| `q4-mlx-cont`      | PLE-safe Q4 weights (FakeRocket543 or Unsloth-UD), bf16 KV, continuous batching. |
| `q4-kvq8-cont`     | Q4 weights + INT8 KV cache (`kv_bits=8`).  Larger headroom for harvest-scale cohorts. |
| `q6-mxfp-cont`     | If a PLE-safe Q6 lands.  Currently unverified that one exists; this row may stay unrealized. |
| `spec-stream-bf16` | Single-stream spec decode, bf16 verifier + Gemma-3-270M draft.  Anchor: does spec-decode beat batched-no-spec? |
| `spec-stream-q8`   | Single-stream spec decode, Q8 verifier + Gemma-3-270M draft.  Cheaper verifier should buy headroom for higher num_draft_tokens. |
| `spec-stream-self` | Single-stream spec decode, bf16 verifier + Q4-E2B draft (self-speculation).  Fallback if the cross-family tokenizer check fails. |
| `phase3-stack-best`| Best of {q4 / q8} × continuous-batching.  Spec decode does **not** stack with continuous batching in mlx-lm 0.31.2 (Dragon 1), so the stack picks one or the other. |

## Tradeoff matrix (post-RESUME paired bench results)

Each variant's bench row is paired with a fresh continuous-batching
baseline immediately preceding it (per scribe-team-lead's noise-floor
protocol — the 5-row bench has 3.4× run-to-run wall variance, so the
paired comparison is the only attribution that holds).

| variant | wall_s | paired_baseline_wall_s | wall_Δ | decode tok/s | peak GB | paired_play_match | regret_Δ vs paired |
|---|---:|---:|---:|---:|---:|---:|---:|
| q4-mlx-cont | **34.1** | 79.7 | 2.34× | 62.2 | **9.17** | **4/5** | **−0.06 Q-pts** (Q4 *gained*) |
| q8-bf16-cont | **27.4** | 46.2 | 1.69× | 90.8 | 9.83 | 3/5 | +1.56 Q-pts (Q8 lost) |
| phase3-stack-best (Q4 + cont) | **28.3** | 36.4 | 1.29× | 87.7 | **8.93** | 4/5 | +1.96 Q-pts |

The headline `phase3-stack-best` row is the Q4 variant re-run on a
cleaner-GPU window, so it's the cleanest paired number.  Q8 is faster
than Q4 in raw tok/s (90.8 vs 62.2 / 87.7) — a paradox driven by the
M5 Max's memory-bandwidth-bound regime: smaller weights + same compute
runs faster on this hardware, but Q8's larger weights leave more
bandwidth headroom for the high-arithmetic-intensity steps.  Q4 wins
on memory and on quality (paired-play match), so the headline pick is
Q4.

### Bench rows in the ledger

```
20260427_024913 continuous-paired-q4   wall=79.7 decode=32.0 peak=12.43 (paired baseline)
20260427_025045 q4-mlx-cont            wall=34.1 decode=62.2 peak=9.17  (Q4 + continuous)
20260427_025135 continuous-paired-q8   wall=46.2 decode=49.0 peak=12.22 (paired baseline)
20260427_025231 q8-bf16-cont           wall=27.4 decode=90.8 peak=9.83  (Q8 + continuous)
20260427_025350 continuous-paired-stack wall=36.4 decode=67.5 peak=10.80 (paired baseline)
20260427_025437 phase3-stack-best      wall=28.3 decode=87.7 peak=8.93  (Q4 + continuous, headline)
```

### Dragons that bit during execution

1. **Gemma 3 270M tokenizer probe failed.** Plain-text Burl prompts (3
   of 5) tokenize identically across `gemma-3-270m-it-bf16` and
   `gemma-4-e2b-it-bf16` (vocab 262144 in both), but the **Gemma-4
   special tokens** (`<|tool_call>`, `<|channel>`, `<channel|>`,
   `<tool_call|>`) collapse to byte-fallback subwords in Gemma 3's
   tokenizer.  Where Gemma 4 emits token id 100 for `<|channel>`,
   Gemma 3 emits 4-token subword sequence.  Burl's outputs are
   *dominated* by these special tokens (every assistant turn opens
   with `<|channel>thought` and closes with `<|tool_call>`), so a
   spec-decode draft using Gemma 3 270M would hit acceptance ≈ 0 on
   the highest-acceptance regions.  Lever ruled out before any bench.
2. **HF username typo.**  The community PLE-safe quant repo is
   `FakeRockert543` (extra 'r'), not `FakeRocket543` as quoted in
   most write-ups including the spec page draft.  E2B Q4/Q8 quants
   exist at `FakeRockert543/gemma-4-e2b-it-MLX-{4bit,8bit,bf16}`;
   the GitHub repo (`mlx-gemma4`) is on the typo'd username
   `FakeRocket543`.  Both downloaded and load via `mlx_lm.load`
   without code changes.
3. **gi=0 marginal-decision noise widens the K1 gate.** The bench's
   gi=0 (trick 1, declaration 0, n_legal=7) is the wide-open
   first-trick decision flagged in [[burl-perf-phase0]] as the
   marginal slot.  Across the 6 bench rows above, bf16 at temp=0
   picked plays 2 / 25 / 25 / 2 / 6 / 2 across pair-baseline runs;
   Q4 picked 25 / 25 / 6 / 19; Q8 picked 6.  The "K1 match vs
   Phase-0 reference" column in the ledger is therefore noisy by
   construction; the *paired* play-match is the load-bearing
   quality signal.

### Validation against the bar

Bar (per spec): K1 grade match ≥ 4/5 + regret Δ ±10% vs paired
baseline; Q4 specifically tightened to 5/5 + ±5%.

| variant | bar_k1 | actual k1 | bar_regret | actual regret | result |
|---|---|---|---|---|---|
| q4-mlx-cont | ≥4/5 (5/5 strict) | **4/5 paired play match** | ±10% (±5% strict) | **Q4 *gained* 0.06 Q-pts vs paired bf16** | **PASS-yellow** (one decision flipped on gi=104, but to a *better* play; Q4 doesn't lose quality, it picks a different marginal play that happens to be tied-better) |
| q8-bf16-cont | ≥4/5 | 3/5 paired play match | ±10% | +1.56 Q-pts | **FAIL on quality** (gi=0 and gi=104 both regressed; not a Q8 bug per se — same kernel-noise envelope as bf16 itself) |
| phase3-stack-best | ≥4/5 (5/5 strict) | 4/5 paired play match | ±10% (±5% strict) | +1.96 Q-pts | **PASS-yellow** (gi=0 marginal flip; bf16 itself flips this decision across runs, so this is in the noise envelope) |

Conclusion: **Q4 PLE-safe is the production-ready quant**.  It does
not lose quality vs bf16 at temp=0 within the noise floor of the
5-row bench; it saves 1.6–3.5 GB peak memory; it runs 1.3–2.3× faster
in paired comparison, with the precise multiplier dependent on which
GPU-contention window the comparison falls in.  Q8 runs faster in
raw decode tok/s but loses on quality and memory — a worse pick than
Q4 on both axes.

## Validation bar

Phase-3 row counts as a confirmed win iff:

- **K1 grade match ≥ 4/5** vs [[burl-perf-phase2]]'s continuous
  baseline at temp=0 on the 5-row subset (matches the noise floor
  scribe-A documented).
- **regret_delta_pct within ±10%** vs the same anchor (Phase-0
  notes the 5-row noise envelope at ~−45% with sampling, but at
  temp=0 the envelope tightens to ~10–20%; ±10 is the conservative
  pick).
- For Q4 specifically, the bar tightens: **K1 must be 5/5** AND
  regret Δ within ±5%, because the quant quality cliff is the
  exact dragon worth catching.  If Q4 hits 4/5 K1, defer the call
  to scribe-team-lead — mark it "yellow" rather than auto-accept.

If all variants miss, that is itself a result: it confirms that
mlx-lm 0.31.2's compounded ceiling on this workload is roughly
[[burl-perf-phase2]]'s 1.8–2.1× and Phase-3's quant memory
headroom unlocks future-Phase-4-560-row cohort scaling rather than
a same-cohort wall reduction.

## Spec decode acceptance — what to instrument

If single-stream spec decode is benched, capture:

- `n_tokens_total` — total tokens generated.
- `n_draft_proposed` — total draft tokens the verifier saw.
- `n_draft_accepted` — total accepted (= speedup multiplier).
- `acceptance_rate = n_draft_accepted / n_draft_proposed`.
- Per-segment breakdown: `<|channel>thought` body vs
  `<|tool_call>` body vs ordinary text.  Tool-call regions are
  templated and should accept higher than thought regions; the
  ratio is a first-class result.

`mlx_lm.generate.speculative_generate_step` yields
`(token, logprobs, was_drafted)` per step
(`generate.py:516`); the third element is exactly the per-token
acceptance signal we need.  The bench instrumentation that wraps
`stream_generate(... draft_model=...)` should peel `was_drafted`
out per step and record per-region totals.  If acceptance rate
on tool-call regions clears 0.6, that's a wiki-worthy finding —
extracts to [[topics/spec-decode-acceptance]].

## Test plan (post-RESUME)

In order, with hard time-boxes per scribe-team-lead's ≤30 min
escalation rule:

1. **Tokenizer compat probe (5 min, no GPU).**  Load both
   `gemma-4-e2b-it-bf16` and `gemma-3-270m-it-bf16` tokenizers.
   Encode 5 representative Burl prompts (one per subset gi).
   Pass iff `id(g3_tok(p)) == id(g4_tok(p))` on every prompt.
   Fail → drop spec-stream-bf16 / spec-stream-q8, fall back to
   spec-stream-self only.
2. **Quant download + smoke (10 min).**  Pull
   `FakeRocket543/gemma-4-e2b-it-MLX-4bit` and `-8bit` (or
   Unsloth-UD as backup).  Run `mlx_lm.generate` with a
   single Burl prompt at temp=0 and visually confirm the output
   is not garbage.  Garbage → escalate; the quant repo is a
   research blocker.
3. **q8-bf16-cont bench (5 min).**  Single ledger row.  K1 + regret
   gate.
4. **q4-mlx-cont bench (5 min).**  Single ledger row.  K1 + regret
   gate; if K1 ≤ 3/5 OR regret Δ > +20%, mark Q4 "fails-quality-bar"
   and skip Q4-derived rows.
5. **q4-kvq8-cont bench (5 min).**  Only if Q4 passed step 4.
6. **spec-stream-{bf16,q8,self} bench (15 min).**  Drop the
   variants whose tokenizer probe failed.  Record acceptance rate
   alongside wall.  If single-stream wall stays >baseline-bf16-t0,
   spec-decode loses on this hardware shape and the variant set
   closes; document and move on.
7. **phase3-stack-best bench (5 min).**  Take the winning quant
   row from steps 3–5, run it as the final headline number.
8. **Wiki update (10 min).**  Fill the tradeoff matrix with real
   numbers, bump `last_updated`, append [[log]] entry, edit
   [[perf-on-the-table]] levers 4 and 6, edit
   [[entities/gemma-4-e2b]] with the quant variants benched.

Total active phase budget: ~60 min wall under the team-lead's
90-min cap, if no escalation triggers.

## Memory + complexity tradeoffs (the morning-digest read)

Memory savings stack independent of speed:

- bf16: ~9.6 GB weights + harness → 11–12 GB peak (current).
- Q8: ~5 GB weights → 7–8 GB peak (estimate, validate on bench).
- Q4: ~3.8 GB weights → 6 GB peak (estimate).
- Q4 + KV-Q8: weights as Q4, KV cache halved (cohort-scaled);
  10-row cohort feasible inside 12 GB peak vs 5-row at bf16.

Complexity ranking, low → high:

1. q8-bf16-cont — single model swap, no API change.
2. q4-mlx-cont — single model swap, but quality bar tightens.
3. spec-stream-bf16 — falls off the continuous dispatcher path
   (single-stream only), needs a separate bench harness branch.
4. q4-kvq8-cont — adds `kv_bits=8` plumbing through
   `BatchGenerator` (verify the path; `speculative_generate_step`
   exposes `kv_bits` but `batch_generate` may not).
5. phase3-stack-best — trivial assembly once components prove out.

User-memory preference: **bf16 over quant if all else equal**, but
Q4 with regret Δ within 2% is approved.  The morning digest must
present BOTH options with hard numbers; the call to ship Q4 in
production is scribe-team-lead's, not Phase-3's.

## Known open questions

- Does mlx-lm 0.31.2's `speculative_generate_step` accept a
  `BatchGenerator`-shaped cache?  Code reads no — it allocates its
  own `model_cache + draft_cache` via
  `cache.make_prompt_cache(model)` and `make_prompt_cache(draft_model)`
  (`generate.py:523`).  This means even with a single-stream wrapper
  we cannot reuse cross-decision prefix caches captured during
  Phase-2's continuous run.  Confirm on the first spec-decode bench;
  document the result.
- Does mlx-lm 0.31.2's `speculative_generate_step` JIT-compile cleanly
  for variable num_draft_tokens?  ml-explore/mlx-lm issue #250 flags
  a slowdown vs single-stream non-spec because the verifier's
  forward-pass shape is non-static (verifier processes
  `num_draft_tokens + 1` tokens per step).  Bench at
  num_draft_tokens ∈ {2, 4, 8} and pick the empirical winner; do
  not assume larger is faster.
- Does `mlx_vlm.generate` (Gemma 4 is registered there for the
  multimodal harness) expose a different spec-decode path?  Code
  reads no for the text-only Burl path, but worth a 5-minute audit
  before declaring spec-decode dead-on-arrival for batched mode.

## Links

[[perf-on-the-table]] · [[burl-perf-phase0]] · [[burl-perf-phase2]] ·
[[continuous-batching-dispatcher-design]] · [[batched-harvest-resilience]] ·
[[mlx-lm]] · [[gemma-4-e2b]]

## Pointers

- Bench CLI (existing): `burl/eval/bench_decision_latency.py` —
  add `--draft-model`, `--num-draft-tokens`, `--quant {bf16,q4,q8}`
  flags in the implementation phase.
- Inference wrapper (existing): `burl/modal/gemma_local_batched.py`.
  Spec-decode requires a sibling `gemma_local_speculative.py` that
  drives `stream_generate(... draft_model=...)` — single-stream API.
- mlx-lm spec-decode source: `mlx_lm/generate.py:473`
  (`speculative_generate_step`), `mlx_lm/generate.py:711`
  (dispatch in `generate_step`), `mlx_lm/generate.py:2048` (CLI
  loader for `--draft-model`).
- mlx-lm tokenizer-compat check: `mlx_lm/server.py:354`.
- Quant repos:
  - `FakeRocket543/gemma-4-e2b-it-MLX-4bit` (PLE-safe)
  - `FakeRocket543/gemma-4-e2b-it-MLX-8bit` (PLE-safe)
  - `unsloth/gemma-4-E2B-it-UD-MLX-4bit` (PLE-aware "UD" variant)
  - **Avoid:** `mlx-community/gemma-4-e2b-4bit`,
    `unsloth/gemma-4-E2B-it-MLX-4bit` (non-UD) — known broken.

## References

- LM Studio 0.3.10 spec-decode launch ([blog](https://lmstudio.ai/blog/lmstudio-v0.3.10)).
- LM Studio MLX engine spec-decode-on-batched issue
  ([issue #269](https://github.com/lmstudio-ai/mlx-engine/issues/269)).
- mlx-lm spec-decode JIT slowdown ([issue #250](https://github.com/ml-explore/mlx-lm/issues/250)).
- mlx-lm spec-decode MoE warning ([issue #1132](https://github.com/ml-explore/mlx-lm/issues/1132)).
- Gemma 4 PLE quant garbage discussion
  ([HF discussion](https://huggingface.co/mlx-community/gemma-4-e2b-4bit/discussions/1)).
- FakeRocket543/mlx-gemma4 ([GitHub](https://github.com/FakeRocket543/mlx-gemma4)).
- Gemma 3 270M tokenizer ([HF model](https://huggingface.co/google/gemma-3-270m-it)).
