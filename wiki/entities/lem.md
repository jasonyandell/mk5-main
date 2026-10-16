---
title: LEM — Little Expert Model
kind: entity
first_seen: 2026-04-09
last_updated: 2026-10-16
status: complete
superseded_by: burl
---

## What it is

**The name lives on.** Since 2026-10-15 "Lem" is also the project's generative player and
the subject of its interpretability program, [[lem-the-teacher]]; Jason reused the name on
purpose ("they're good names and it kinda echoes"). This page is the 2026-04 project, kept as
history. **Its code was removed from the tree at b4a3e5b7 (2026-10-16)**: `lem/gemma_star/`, `lem/narrate/`, `lem/rules/` and `lem/OVERVIEW.md` survive only in git
history and in the commit-pinned source digests this page cites; the adapters stay on Hugging Face
(`jasonyandell/qwen3-1.7b-texas42-stage0-v10-maskfix` and its line). `lem/` now holds the new Lem.

LEM (Little Expert Model) is an experiment in training a phone-class open model to play
[[texas-42]] via STaR-style self-taught reasoning layered on a backwards curriculum. The
base model is [[gemma-4-e2b]] — a model that has never heard of Texas 42 at the start of
the experiment. LEM replaces a bot mid-game and asks whether Gemma can match or beat the
bot's expected outcome. (lem/OVERVIEW.md @ a8bccfa)

## Tombstone

The old LEM is finished. This section is its epitaph: what it was, what it measured, why it
stopped, and what the new [[lem-the-teacher]] takes from it. The chronological record is under
"Progress" below.

### Dates and commits

| Event | Date | Commit |
|---|---|---|
| Born: narration generator, rules primer, first Gemma contact | 2026-04-09 | a8bccfa |
| Last substantive code commit: completion-only loss, 1.7B v10-maskfix | 2026-04-17 | be7efc4 |
| Last doc commit to `lem/`: "current working vision" appended to the overview | 2026-04-18 | e2d30362 |
| Retired: Burl introduced as the successor mechanism | 2026-04-18 | 8d26e0d |
| Code deleted under the no-legacy rule | 2026-10-15 (23:45 CDT, 2026-10-16 UTC) | b4a3e5b7 |

The `lem/` code history is `git log a8bccfa^..be7efc4 -- lem/` (40 commits touching `lem/`
between 2026-04-09 and 2026-04-17); `git log --all -- lem/` also lists the doc commits
(8d26e0d, e2d30362, e2171816) and the new Lem's own commits. (git log, lem/ @ b4a3e5b7)

### What it tried

Teach a phone-class open model to play [[texas-42]] with no prior knowledge of it: a rules
adapter first (Stage 0), then [[star]] on trick-6 decisions graded by [[k1-grading]], then a
[[backwards-curriculum]] one ply at a time. The base was [[gemma-4-e2b]], then
[[qwen3-1.7b]] from 2026-04-16 (and [[qwen3-14b]] as a capacity probe). The north star was
[[expected-q-value]] delta against the bot LEM replaced. (lem/OVERVIEW.md @ a8bccfa)

### What it found

Each number is as its source has it. No source reports an interval for any of them.

| Finding | Number | Source |
|---|---|---|
| First STaR chain, 15 iterations on Gemma | 30% pass at iter 0, plateau 36-42%, best 42% (iters 5 and 7); 200 or 300 examples per iteration | [[star-10-iterations]] |
| Kerry and v3 curriculum raised the floor | avg ~37% to ~43% to ~44%; peak 42% to 46% to 48%; illegal 33% to ~12% to ~13% | [[stage-0-adapter-line]] (8c1bb14) |
| Rigid grader hid real knowledge | `legal_moves` 0% under the rigid grader, 70% under the flexible one; v4 overall 67% | [[flexible-grader]], [[stage-0-v4-comprehension-eval]] |
| Base-model pivot | 100% on Qwen 3 1.7B vs 60% on Gemma, same corpus and eval | [[base-model-pivot-qwen]] (3465e29) |
| 14B capacity probe | 86% comprehension, 97/100 clean rationalizations vs ~68/100 on 1.7B | [[qwen-14b-capacity]] (0c7392f) |
| Completion-only loss | 1.7B comprehension 83% to 86%, equal to 14B v9 at about a third of the compute | [[sft-completion-only-loss]], [[v10-maskfix-breakthrough]] (be7efc4) |
| The plateau that did not move | bot-match 55/100 before and after the mask fix; 96/100 legal | [[v10-adapter]] |
| Total STaR compute | ~$25 on B200 for 15 iterations | [[star-10-iterations]] |

Caveats the sources state. The 14B run predates the mask fix and was never rerun with it, so
its 97/100 is confounded with the loss bug ([[sft-completion-only-loss]], scope note). The
rationalization numbers come from scouted decisions filtered by the
[[rationalization-verifier]], not from play. The "86%" is comprehension of engine-verified
facts, not strength at play.

### Why it ended

Superseded by [[burl]], a sibling mechanism rather than a continuation. The 55/100 bot-match
did not move with gradient reallocation or with capacity, which left the diagnosis that
open-ended play selection is a different skill from fact recall and that more flashcard-shaped
SFT might not move it. Burl moved the rules into the [[engine]] as tools, put beliefs behind
a tool call, trained on trajectories, and chose Gemma for agentic tool use, which reversed the
rationale of the Qwen pivot. ([[lem-to-burl-handoff]] sections 2, 3 and 5; [[burl]], "The bet")
Burl was in turn superseded by [[jud]] (burl, status note). None of LEM's open questions was
answered; the pivot closed them.

### Lessons learned

1. **Clairvoyant targets teach clairvoyant play.** LEM's Stage 1 games came from the E[Q]
   greedy policy, its keep rule compared `E[Q][gemma]` to `E[Q][bot]`, and its bot-match
   grade measured agreement with the bot (the E[Q] bot, as 8d26e0d's vocabulary names it). E[Q] is built from per-world solves of a
   player who sees all four hands, and [[the-wall]] records that every student of that teacher
   learned to play as if it saw them. LEM's measured plateau is not attributed to this in any
   source; the lesson is about target design. The new Lem trains on information-state targets
   only, Zeb's value for the seat and every seat's next play, never a per-world clairvoyant
   value, under the [[strategy-fusion-fence]]. (lem/OVERVIEW.md @ a8bccfa; [[lem-the-teacher]],
   "The fence")
2. **When SFT stalls, inspect the loss mask.** A ~50-token answer was diluted about 9 times by
   ~400 memorized prompt tokens; three stuck metrics moved 10-18 points once the loss covered
   only the completion. ([[sft-completion-only-loss]], be7efc4.) The new Lem's plan records no
   masking rule yet; the check is filed on its stage 3 epic (#178): the loss covers the targets only.
3. **A broken grader reads as a broken model.** A rigid grader scored `legal_moves` 0% and a
   five-category inline grader scored a 14-category eval 35% where the shared grader read 86%
   on the same responses. ([[flexible-grader]], [[v10-maskfix-breakthrough]].) The new Lem's
   stage 1 reports every share with its interval and against a shuffled-label baseline, and
   was pre-registered before it ran ([[lem-field-net-audit-2026-10-15]]).
4. **Compare capacity arms under one pipeline.** The 14B-over-1.7B gap was partly the loss
   bug, which only 1.7B had fixed, so "capacity helps" and "the mask helps" were entangled.
   ([[qwen-14b-capacity]].) The new Lem's scaling family runs every arm under one pipeline and one loss, and prices
   every spend first ([[lem-the-teacher]], "The nets" and "Budget").
5. **Training shape must match target behavior.** Flashcards, rationalizations and structured
   templates moved comprehension and left play selection flat at 55/100. The new Lem's net
   predicts the next play of every seat and the value of the position from the information
   state, which is the behavior itself. ([[lem-to-burl-handoff]], section 2; [[lem-the-teacher]],
   "The nets".)
6. **Pick the base model for the consumer's job.** Qwen won on comprehension and speed
   (throughput on B200 was the real fix for the underutilization bead, not kernel tuning);
   Gemma won on tool use for Burl. ([[base-model-pivot-qwen]]; [[lem-to-burl-handoff]], section 5.)
   The new Lem is not a language model: a small GPT over the information state, read back out
   by interpretability ([[lem-the-teacher]], "Prior work").
7. **Stages that end by pivot leave no answers.** The Stage 2 ratchet, the variance-filter
   thresholds, DPO versus rationalization, and STaR on the post-maskfix stack were all
   never run. ("Open questions" below; lem/OVERVIEW.md @ e2d30362 notes STaR had never run on
   the post-maskfix stack.) The new Lem's five stages each stand alone and ship their own
   result, and stage 1 states its decision rule in advance ([[lem-the-teacher]], "The five
   stages").

### Where everything lives now

- **Hugging Face, `jasonyandell/` (about 80 repos; adapters and per-epoch checkpoints, none deleted):**
  - Gemma 4 E2B Stage 0: `gemma-4-e2b-texas42-stage0`, `-stage0-kerry`, `-stage0-v3`,
    `-stage0-v4-*` (iter0-3, tiny10ep, full3ep and its epochs), `-stage0-v5` and `-stage0-v5-clean`
    (with `-ep1..3`).
  - Gemma STaR chain: `gemma-4-e2b-texas42-star-iter0` through `-iter27` (the wiki documents
    iter0-14).
  - Qwen 3 1.7B: `qwen3-1.7b-texas42-stage0-v5` through `-v10`, `-v10-maskfix` (each with
    `-ep1..3`), and `qwen3-1.7b-texas42-rationalize-v1`.
  - Qwen 3 14B: `qwen3-14b-texas42-stage0-v9` (with `-ep1..3`).
  - The best LEM adapter is `qwen3-1.7b-texas42-stage0-v10-maskfix` ([[v10-adapter]]).
  - `gemma-4-e2b-texas42-burl-*` belong to [[burl]], not to LEM ([[burl-line]]).
- **Weights & Biases:** `jasonyandell-forge42/lem-stage0` and `jasonyandell-forge42/lem-star`.
- **Git history:** `git show a8bccfa:lem/OVERVIEW.md` is the original plan;
  `git show be7efc4:lem/OVERVIEW.md` (897 lines) is the full working log;
  `git show e2d30362:lem/OVERVIEW.md` adds the 2026-04-18 "current working vision" (reasoning as
  exploring the distribution, claims phrased as `P(...)`, the 100%-legal gate, three next moves).
  `git show b4a3e5b7^:lem/` holds the code before deletion (`lem/OVERVIEW.md` there is already a
  stub pointing to this page). The narration and Q&A corpora (`lem/data/`, gitignored) are in neither git nor Hugging Face;
  no record says whether a copy survives on disk.
- **Wiki:** the adapter chain and its table are [[stage-0-adapter-line]]; the narrative is
  [[lem-to-burl-handoff]]; commit digests are in `wiki/sources/` (a8bccfa, 6bb8a40, 24ae55a,
  efad16e, 908773a, 8c1bb14, 3465e29, b857299, 0c7392f, be7efc4).

## North star

> **E[Q] delta vs. the bot it replaced**, measured on a held-out set of seeds, rising over
> training iterations.

[[expected-q-value]] delta is the sole success metric. Win rate is not tracked; luck does
not wash out in a single game but does in aggregate, and E[Q] is a cleaner signal.
(lem/OVERVIEW.md @ a8bccfa)

The Stage 0 adapter chain (stage-0 → kerry → v3 → v4 → v5 → v9 → v10) is summarized with
a comparative table on [[stage-0-adapter-line]]; per-adapter receipts live in
`experiments/`.

## Two-stage plan

### Stage 0 — Rules Adapter

Teaches Gemma what Texas 42 is before any reasoning is demanded.

- Manual rules primer (1–2 pages; see `lem/rules/primer.md`)
- Synthetic Q&A corpus generated by the game engine (legal moves, trick winners, count
  math, void inference)
- [[lora-unsloth]] fine-tune
- Gate: parsing sanity check on held-out [[narration]]s

The [[rules-adapter]] must pass the parsing gate before Stage 1 begins.
(lem/OVERVIEW.md @ a8bccfa)

### Stage 1 — STaR on Trick-6 Decisions

Teaches the rules-adapter model to reason about a single ply of lookahead.

- Generate games: all 10 declarations × many seeds × 4 seats via [[forge]]'s E[Q] greedy
  policy
- Narrate each game from the narrator's seat (see [[narration]])
- Filter to trick-6 decisions where `|legal| ≥ 2` and the E[Q] gap is non-trivial with
  bounded SEM (forced plays and coin-flip decisions carry no learnable signal)
- Upload filtered dataset to HuggingFace
- Inference pass: collect thinking traces from the rules-adapter model
- Grade via [[k1-grading]]: keep trace if `E[Q][gemma] ≥ E[Q][bot]`; on failure apply
  [[r1-rationalization]] (reveal bot's action, ask Gemma to justify it)
- LoRA step on kept traces
- Iterate until E[Q] delta plateaus

(lem/OVERVIEW.md @ a8bccfa)

### Stage 2+ — Backwards curriculum

Once Stage 1 plateaus, widen the curriculum backward: trick 5, then 4, then 3. Each stage
adds one ply of lookahead. The [[backwards-curriculum]] structure is explicit — difficulty
is a counted quantity, not a heuristic. (lem/OVERVIEW.md @ a8bccfa)

## Design commitments

| Choice | Value |
|---|---|
| Base model | [[gemma-4-e2b]] |
| Narrator framing | "You are taking over for a bot" |
| Game gen policy | E[Q] greedy, N=10 |
| Declaration coverage | All 10 per seed |
| Perspectives per game | All 4 (split train/val by seed, not narration) |
| Keep rule | [[k1-grading]]: keep iff `E[Q][gemma] ≥ E[Q][bot]` |
| Failure handling | [[r1-rationalization]] |
| Fine-tune method | [[lora-unsloth]] |
| Dataset home | HuggingFace |

(lem/OVERVIEW.md @ a8bccfa)

## Compute menu

- **3050 Ti (local)** — narration generation, parsing checks, interactive LoRA iteration
- **Vast.ai** — 3090/4090 for longer training runs
- **Modal** — burstable parallel work (mass Q&A generation, mass inference passes)
- **HuggingFace** — shared dataset and adapter store across all compute nodes

## Non-goals

- Beating the full-info Stage 1 oracle (impossible; it sees all four hands)
- Beating the existing bot in game outcomes (E[Q] delta is the metric, not win rate)
- Training a general-purpose Texas 42 explainer or rules tutor (the rules adapter is a
  means, not an end)
- Matching perfect play

(lem/OVERVIEW.md @ a8bccfa)

## First contact

See [[first-gemma-contact]]. At commit a8bccfa, [[gemma-4-e2b]] correctly identified
trump, led suit, void status, and score math in its first run against a narrated game.
Errors observed: confused initial hand with remaining hand; misclassified 6-4 as trump
under fives-trump. Both gaps are exactly what Stage 0 is intended to close.
(commit message @ a8bccfa)

## Repo layout

```
lem/
├── OVERVIEW.md
├── narrate/        ← Stage 1 data gen: seed → narration
├── rules/          ← Stage 0: primer + engine-generated Q&A corpus
└── gemma_star/     ← STaR harness: inference, grading, rationalization, LoRA
```

## Progress

### Stage 0 complete (2026-04-10, commits 6bb8a40–24ae55a)

[[stage-0-adapter]] trained and pushed to HuggingFace (`jasonyandell/gemma-4-e2b-texas42-stage0`).
Wandb run: `jasonyandell-forge42/lem-stage0`.

Key milestones:
- 55/55 primer facts verified against the engine (`lem/rules/verify_primer.py`).
- 3500-example Q&A corpus generated across 7 categories.
- Loss: 32 → 0.001 in 40 steps; 100% token accuracy by step 50 on [[modal]] L4.
- [[second-gemma-contact]]: hand-tracking fixed, final answer legal and correct;
  trump membership errors (4-4, 6-4 under fives-trump) remain.

Key insight: trump membership did not transfer from Q&A format to narration-context reasoning.
The expectation is that [[star]] Stage 1 closes this gap through play and [[k1-grading]]
correction, not further drilling. (lem/OVERVIEW.md @ 24ae55a)

Stage 1 is unblocked. Next: generate narration dataset in batch, filter to trick-6 decisions,
build STaR harness, run first E[Q]-delta measurement on held-out seeds.
See [[stage-0-v1-training]] for the full training record.

### Stage 1 infrastructure (2026-04-10, commit b99c64d)

- **Batch narration generator online** — 200 seeds → 3148 trick-6 training examples in 8.5 min
  on 3050 Ti. Output: `lem/data/narrations_train.jsonl` (seeds 0–199) and
  `lem/data/narrations_eval.jsonl` (seeds 900000–900049, 812 examples) —
  not preserved (predates the HF data strategy, see [[huggingface-assets]]).
- **Held-out eval seeds declared** — seeds 900000–909999 are permanently reserved for
  evaluation and must never appear in any training dataset. All [[expected-q-value]] delta
  measurements use this range. See [[eval-seed-holdout]].
  Training seeds: 0–899999; within training, `seed % 1000 < 950` → train, `>= 950` → val.
  (lem/OVERVIEW.md @ b99c64d)
- **Training moves to A100** — Stage 1 training runs on [[modal]] A100 (was L4 for Stage 0).

(commit message @ b99c64d)

### Stage 1 kickoff (2026-04-10, commits 7538016–8c5fbca)

- **[[star-harness]] built** — Modal batch function implementing the 6-step STaR flow:
  inference → parse → K1 grade → keep or rationalize → JSONL output.
- **5-example smoke test passed end-to-end** — 1 K1 pass (20%), 2 failures rationalized,
  2 illegals rationalized, all traces collected. See [[star-harness-5ex-smoke]].
- **Base-model K1 baseline measured** — 10 trick-6 decisions, base Gemma (no adapter),
  local llama.cpp: 60% K1 pass / 30% legal-suboptimal / 10% illegal / 0% parse-fail.
  Surprisingly high; suggests many trick-6 decisions are near-argmax-unanimous.
  This is the number Stage 0 adapter + STaR iterations must beat.
  See [[base-model-k1-baseline]].
- **Single-GPU continuous loop** (`star_loop.py`) — HF `model.generate()` batch inference
  + K1 grade + rationalize + LoRA train + HF push, all in one Modal function. (vLLM was
  originally used but abandoned — see infrastructure note below.)

(commit messages @ 7538016, f578bfa, 8c5fbca)

### Stage 1 iteration 0 (2026-04-10, commits 68a0416–576b694)

**Result**: 30% K1 pass / 40% illegal, 15 min on H100, ~$1.
The 40% illegal rate lands squarely in the [[discard-illegal-traces]] prediction
zone ("~40% = Stage 0 needs more rules work"). See [[star-iter-0]].

**Compute setup that works at this frontier:**
- H100 — full STaR loop (`star_loop.py`); ~15 min/iteration
- A100 — standalone LoRA training (`train_star.py`)
- CPU + Q4_K_M GGUF via llama.cpp — local interactive debug (`star_local.py`)

**Infrastructure: vLLM abandoned, HF batch replace.** vLLM bundles its own `transformers`
version, which conflicts with the [[gemma-4-e2b]] `Gemma4ClippableLinear` monkey-patch
required by PEFT. Attempts to make the patch conditional failed. vLLM is removed entirely;
`star_loop.py` now uses HF `model.generate()` with batched tokenization, padding, and
decode. This is also a simplification: it removes the ~5-minute vLLM engine cold-start
each iteration. Note: this may be revisited on newer hardware in later commits.
(commit messages @ 68a0416, 8724e93, d913932; lem/OVERVIEW.md @ 576b694)

### Stage 1 STaR — 10 iterations (2026-04-11, commits ff0d0d2–efad16e)

10 iterations chained as separate [[modal]] runs (avoids container staleness; limits blast
radius per iteration; each adapter independently inspectable).
Wandb: `jasonyandell-forge42/lem-star`.

**Results:**

| Iter | Pass rate | Notes |
|---|---|---|
| 0 | 30% | Baseline with Stage 0 adapter, 3148-ex pool |
| 1–4 | 34–38% | Gradual improvement |
| 5 | 42% | Pool expanded to 7409 ex (seeds 0–499) at this iter |
| 6 | 36% | |
| 7 | 42% | Best alongside iter 5 |
| 8–9 | 36–40% | Plateau |
| 10 | 39% | |
| 11 | 41% | |
| 12 | 40% | |
| 13 | 39% | |
| 14 | 38% | Plateau confirmed |

Average pass rate ~38–40%. Loss: 31.6 → 11.8 over 10 iters; stable at ~10 through iter 14.
Data pool at iter 5 grew from 3148 (seeds 0–199) to 7409 (seeds 0–499); pool expanded
further to 11,672 at 908773a (added seeds 500–799 via
`lem/data/narrations_train_500_799.jsonl`, not preserved — see [[huggingface-assets]]).
15 adapters pushed to HF:
`jasonyandell/gemma-4-e2b-texas42-star-iter{0..14}`. Total compute cost: ~$25 on B200.
(lem/OVERVIEW.md @ 908773a)

**Plateau hypothesis**: the ~40% ceiling likely reflects K1-without-fact-verification.
The model may be learning wrong game-facts that happen to produce correct plays ~40% of the
time, but cannot improve further because the underlying reasoning is polluted. Breaking the
plateau probably requires [[scratchpad-validation]] after the format is bootstrapped via SFT
— generate correct scratchpad examples from the engine, train one LoRA pass to teach the
format, then resume validated STaR. (lem/OVERVIEW.md @ 908773a)

**Compute setup (B200 all-in):**
- Full STaR iteration (5 examples): ~2.5 min on B200, ~$0.26
- LoRA training step: ~13s on B200
- Local debug: llama.cpp CPU, 11 tok/s, free
- Narration generation: local 3050 Ti, 0.4 seeds/s, free

**Architecture finding — LoRA adapter key mismatch resolved:** PEFT reported "missing keys
for `language_model.layers.{15..34}.self_attn.{k,v}_proj`" when merging adapters. This
is explained by [[gemma-4-e2b]]'s KV-sharing: layers 15–34 reuse KV projections from
layers 0–14 and have no k_proj/v_proj of their own. The adapter was complete all along;
the warning is architectural, not a bug. No training changes needed.
(lem/OVERVIEW.md @ efad16e)

### Narration v3 — public state block (2026-04-11, commit 7f1994e)

- **Narration v3** adds a public state block after every trick: all dominoes played so far
  (N/28), count domino status (taken by which team or still out), and the narrator's
  remaining hand. See [[public-state-block]]. Cost: ~60 tokens/trick, ~300 extra
  per prompt. Rationale: state that is visible at the real table should be given, not
  reconstructed from prose — a 2B model should not bear the bookkeeping burden that no
  human player bears either. (commit message @ 7f1994e)

### Stage 0 v2 (Kerry) — 2026-04-11, commits f8cdbe7–43009a4

New Stage 0 variant based on [[kerry-curriculum]] (Kerry Newberry's Learner's Guide,
engine-scaled). 15,000 examples in four sections (A: 15%, B: 20%, C: 40%, D: 25%);
section C weighted heaviest because following-suit under trump is the hardest concept for
both humans and the model. Adapter trained in 150 steps on [[modal]] B200.

- **Adapter published**: [[kerry-adapter]] (`jasonyandell/gemma-4-e2b-texas42-stage0-kerry`)
- **Results** (same prompt as all prior contacts): trump non-membership fixed ("no fives,
  no trump" for 6-2/6-1 correctly), trump membership still partially wrong (6-4 stubbornly
  called trump under fives — same narrowed error from contact 1), strategic reasoning
  dramatically deeper (evaluates both options), final answer legal and correct.
- See [[third-gemma-contact]]. (commit messages @ f8cdbe7, 43009a4)

### Kerry STaR and Stage 0 v3 — 2026-04-11, commits a2498e4–8c1bb14

**Kerry STaR iters 0-2** (Kerry adapter + public state narrations): 44% / 42% / 44% pass,
10.5–13% illegal. The illegal rate dropped from 33% (v1 baseline) to ~13% — Kerry's
section C weighting on following-suit exercises pays off directly.

**Stage 0 v3 adapter** ([[v3-adapter]]): Kerry 15k + 5k targeted trump-drill Q&A = 20k
examples, 200 steps on [[modal]] B200. Five STaR iterations: 44 / 42 / **48** / 47 / 38%.
Average ~44%. Peak 48% at iter 2 — new high water mark. Illegal rate ~13%.

**Stage 0 progression:**

| Stage 0 | Avg pass | Peak | Illegal |
|---|---|---|---|
| v1 (3.5k Q&A) | ~37% | 42% | 33% |
| Kerry (15k) | ~43% | 46% | 12% |
| v3 (20k = Kerry + trump drill) | ~44% | 48% | 13% |

Each curriculum round raises the floor. The plateau at ~40% from ingest 10 was a
Stage-0-quality ceiling, not a K1-grading ceiling. The "fact-verification is the only
path forward" hypothesis (from ingest 10) is weakened — better curriculum alone moves the
needle. Best adapter at this frontier: `star-iter2` on v3 base at 48%.
(commit messages @ a2498e4, 601f622, 8c1bb14)

### Stage 0 v4 (game-context) — 2026-04-13, commits 4729dad–2f11f32

Pivot from flashcard Q&A to game-context Q&A. Five question types grounded in actual game
records: `where_is` (track a domino across tricks), `count_status` (count capture by team),
`is_trump` (with pip-level reasoning), `what_beats` (ranking in context), `legal_moves`
(full derivation per domino in hand). Prompts are ~170 tokens vs ~2,400 in v3.

- **Corpus**: 31,830 train + 7,725 eval, engine-verified.
- **Training**: 3 epochs on [[modal]] B200, `batch_size=16` (no gradient checkpointing),
  1.4× faster than v3. Final: loss 0.06, 97.4% token accuracy. Per-epoch HF checkpoints.
- **Eval results** (flexible grader, 100 held-out examples): `is_trump` 100%, `where_is`
  90%, `legal_moves` 70%, `count_status` 60%, `what_beats` 15%. Overall **67%**.
- **Key finding**: two eval bugs (EOS token + left-pad slicing) were masking real model
  knowledge in earlier iterations. See [[flexible-grader]] and [[1d3e1b7]].
- **Behavioral change**: thinking mode disabled for inference — base model "I am reasoning
  about Bridge" hallucinations cease with adapter loaded and thinking off.
- See [[v4-adapter]] and [[stage-0-v4-comprehension-eval]].

(commit messages @ 4729dad, 1d3e1b7, 3c33e86, 2f11f32)

### Stage 0 v5: base model pivot to Qwen 3 1.7B — 2026-04-16, commit 3465e29

Gemma 4 E2B retired as the LEM base model. See [[base-model-pivot-qwen]].

- **Why**: Gemma was both slower (architectural — PLE, KV-sharing, no flash-attention-2)
  and less accurate (60% on comprehension_eval_v5 vs Qwen's 100%). The B200 underutilization
  bead (t42-hv08) was resolved by switching models, not GPU/kernel tuning.
- **New base**: [[qwen3-1.7b]] (1.7B params, Apache 2.0, FA2, Unsloth + xformers on B200).
  Throughput: 36K tok/s, ~19 min/training run on [[modal]] B200.
- **New adapter**: [[v5-adapter]] (`jasonyandell/qwen3-1.7b-texas42-stage0-v5`).
  Comprehension eval v5: **100%** (same corpus, same eval as v4's 60%).
- Gemma pipeline retained untouched; Qwen becomes the active base for all downstream Stage 1.

(commit message @ 3465e29)

### Stage 0 v7-v10 + capacity experiment + mask fix — 2026-04-17, commits b857299–be7efc4

**v7→v9 (b857299)**: Curriculum expands from 5 to 14 categories. New structured-reasoning-template
categories: `conditional_beat` (v7), `beaters_in_unseen`, `partner_response`,
`intervention_check` (v8), `visibility_audit`, `highest_unseen_in_suit` (v9). A
[[rationalization-verifier]] (6 engine checks) filters hallucinations out of
training data. [[v9-adapter]] hits **83% comprehension** overall. Rationalization plateaus
at ~68/100 on 1.7B. `visibility_audit` at 0% on all sizes — structural format problem
(long enumeration), not capacity. See [[stage-0-v9-14categories]].

**v10 + 14B (0c7392f)**: Two parallel experiments:
- [[v10-adapter]] joint-trains v9 comprehension + 331 upweighted (10×) clean rationalizations.
  Transition test: **55/100 bot-match**, 96/100 legal, visible reasoning. 83% comprehension preserved.
- [[qwen3-14b]] (same v9 data): **86% comprehension** (+3pp), **97/100 rationalization** (+43pp over 1.7B).
  Lucid multi-factor reasoning emerges. `partner_response` 48→75%, `beaters_in_unseen` 46→61%.
  See [[qwen-14b-capacity]].

**Mask fix (be7efc4)**: TRL's `SFTConfig` default computes loss over prompt+answer — ~50-token
answer diluted ~9× by memorized prompt tokens. Fix: `prompt`/`completion` format enables
completion-only loss. [[sft-completion-only-loss]]. 1.7B v10-maskfix: **86% comprehension**
(= 14B v9 at 1/3 cost). `intervention_check` 70→88%, `partner_response` 48→58%,
`beaters_in_unseen` 46→56%. Bot-match **55/100 unchanged** — confirming it is not a
gradient-allocation problem. Capacity or STaR is the next lever.

**End-of-LEM-replay state:**
- Best adapter: `jasonyandell/qwen3-1.7b-texas42-stage0-v10-maskfix` — 86% comprehension, 55/100 bot-match.
- 14B rationalization (97/100) is still the ceiling; mask fix does not close that gap.
- Open paths: compound 14B + joint training, or STaR on v10/14B.

See [[v10-maskfix-breakthrough]]. (commit messages @ b857299, 0c7392f, be7efc4)

## How it ended

LEM's replay ended at v10-maskfix (86% comprehension, 55/100 bot-match); the project
pivoted to [[burl]]'s tool-orchestration approach before any of the open questions below
(variance-filter thresholds, Stage 2 ratchet timing, DPO-vs-R1) were answered. See
[[lem-to-burl-handoff]] for the full pivot narrative — the next-day handoff is not
otherwise visible from this page. (lem/OVERVIEW.md @ be7efc4; trails/lem-to-burl-handoff @ 8d26e0d)

## Open questions (never answered — closed by pivot, not resolution)

- Exact variance filter thresholds (`δ`, `σ_max`) — to be determined from wandb
- Stage 0 corpus size and category mix — depends on parsing sanity check result
- Whether the rules primer stays in the system prompt or gets distilled into weights
- When to ratchet from Stage 1 to Stage 2
- Whether [[r1-rationalization]] suffices or DPO-style preference learning is needed
  for Stage 3+
- LoRA rank, LR, epochs — start from Unsloth Gemma 4 recipe defaults

(lem/OVERVIEW.md @ a8bccfa)
