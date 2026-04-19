# Burl — forward plan from iter-4+

**Status**: iter-3-rules is the current winning adapter. iter-4-thoughts produced a clean byte-identical null result. This document is the working plan for iter-5+ under the new constraints of an M5 Max Mac (local GPU, higher tok/sec, training data less of a concern).

Read this alongside `SPIKE_REPORT.md` (the scientific log through 2026-04-19) and `burl/experiments/` (write-ups for every experiment in the session).

---

## Where we actually are

### Adapters on the Pareto frontier

- **`jasonyandell/gemma-4-e2b-texas42-burl-iter3-rules`** — 90% bot-match on N=10, 0 retry-exhausted, 100% first-legal. Trained on 30-row rules-as-tools corpus. **The session's winner for robustness.** SFT reinforced the `trick_winner_if` habit (1.56 → 1.70 per-decision), validating the "tools replace memorization" thesis.
- **`jasonyandell/gemma-4-e2b-texas42-burl-iter1`** — 88.9% bot-match at `--max-retries 7`, mean_eq_delta **−0.16** (11× tighter than iter-3-rules). **Best single-decision quality.** Trained on 30-row trimmed-primer corpus. Structural reasoning idiom (`trump_declared` + `is_trump` + `is_legal`); barely calls `eq_outcome_distribution` (1/9 completed decisions).

These are genuinely different idioms, not two versions of the same thing. iter-3-rules calls `trick_winner_if` to simulate tricks; iter-1 uses trump structure to reason indirectly. Both match or beat base-Gemma-with-no-training.

### The structural gap that survives every experiment

`conditional_outcome` — the counterfactual-probe tool Burl was architecturally designed to use for E[Q] distribution disambiguation — is called **zero times across 145+ decisions** covering:

- Base Haiku (T3 smoke, T8 N=30, T16 full game arena)
- Base Opus 4.7 (T14 partial, T15 full game arena)
- Every trained Burl adapter (iter-0 through iter-4-thoughts)

This is the single sharpest signal of the session. Not a training-data gap — the ceiling models don't use it either. **This is a tool-surface design issue.** See [§Candlewax redesign](#candlewax-redesign) below.

### The LoRA-capacity ceiling

iter-4-thoughts was an A/B against iter-3-rules with only `preserve_thoughts=True` different (same corpus, same recipe, same rank 16, same 3 epochs). Result: byte-for-byte identical output on 42/42 turns across N=10. Either:
- (a) rank-16 × 30 rows × 3 epochs can't express thought-token gradient distinctly from tool-call gradient, or
- (b) Gemma's thinking reflex is pre-trained and can't be reshaped by SFT at any scale we can reach.

Disambiguating (a) vs (b) is the cheapest M5-Max-tuned experiment. See [§Disambiguate LoRA capacity](#disambiguate-lora-capacity).

---

## Toolkit ready to use (all committed, all tested)

### Training infrastructure
- `burl/train/star.py` with `preserve_thoughts: bool = False` kwarg — toggles the `formatting_func` path that bypasses `strip_thinking()`. Default off preserves iter-0..3 reproducibility.
- `burl/train/star_iter4_thoughts.py` — launcher using `preserve_thoughts=True` on iter-3-rules' corpus.
- `burl/train/star_iter2.py`, `burl/train/star_iter3_v2.py`, `burl/train/star_iter3_rules.py` — per-iteration launchers, all parameterized on `--corpus`, `--adapter-name`, `--rank`, `--lr`, `--epochs`, `--batch`, `--grad-accum`. Copy-paste template ready for iter-5+.
- `burl/train/test_iter2_loader.py` — 17 schema sanity checks (commit_play presence, special tokens atomic, chat-template round-trip, labels-not-all-masked).
- `burl/train/test_formatting_func.py` — 12 checks pinning the preserve-thoughts invariants.

### Rollout / eval infrastructure
- `burl/eval/run_move4_star_rollout.py` with EQ-gate Phase B wired and `--concurrency N` for async parallel rollouts (3.92× speedup at c=4). CLI flags: `--gate-variant {minimal,tool-nudge,social}`, `--max-gate-retries`, `--eq-epsilon`, `--no-primer`, `--enable-rules-tools`.
- `burl/eval/run_move4_spike.py` — adapter eval. CLI: `--adapter`, `--no-primer`, `--enable-rules-tools`, `--max-retries`, `--n`.
- `burl/harness/eq_gate.py` — pure-function gate logic, 27 pytests pinning feedback-prompt invariants (no bot_play leakage, no domino_id leakage, no per-play-E[Q] number leakage).

### Prompt-shape flag matrix
`render_native_messages(enable_primer=True, enable_rules_tools=False)`:
- default → trimmed primer + 42-framing (iter-1 winner)
- `enable_rules_tools=True` → rules-as-tools preamble + 42-framing (iter-3-rules winner)
- `enable_primer=False` → 42-framing only (spike-v2 shape; commit-discipline-fragile but produces eq-heavy traces)
- incoherent combo refused

### Tool surfaces
- **Engine tools** (`burl/tools/engine.py`): `is_legal`, `is_trump`, `unseen`, `void_audit`, `trump_declared`, `game_summary` (written, not registered).
- **E[Q] tools** (`burl/tools/eq_distribution.py`): `eq_outcome_distribution`, `conditional_outcome`. Returns 85-bin PDF + mean/stdev/p_make/percentiles.
- **Rules tools** (`burl/tools/rules.py`): `count_dominoes_remaining` (dead — 0 calls ever), `trick_winner_if` (winner — 17 calls on N=10 post-SFT), `what_beats_what`, `contract_progress`.

### Self-play arena
- `burl/selfplay/arena.py` — 4-model full-hand orchestrator, one independent SDK session per seat. CLI: `--seed`, `--tag`. Outputs structured JSONL + human-readable game markdown.
- Model-agnostic; swap the per-seat callable.
- Validated on seed 900010: Haiku 0-42, Opus 7-35 on the same bad-for-bidder deal.

### SDK lock fix for Claude Agent SDK
- `burl/haiku_spike/agent.py` top-of-file monkey-patch serializes `Query._handle_sdk_mcp_request` with an `asyncio.Lock`. Committed as `2830be0`. Necessary for Opus via in-process MCP; insufficient — second failure mode (lone-call stdio timeout) still exists.
- **HTTP transport alternative validated** in scratch (not committed). Productization ~60 min if needed for clean Opus ceiling data.

### Reference traces
- `burl/experiments/haiku_full_notes.md` — 30-decision Haiku reference at $0.78.
- `burl/experiments/opus_spike_notes.md` — partial Opus data (T14 killed at 6/30 with 50% MCP failure rate).
- `burl/experiments/arena_notes.md` — Haiku vs Opus full-hand head-to-head.

---

## Scientific open questions, prioritized

### 1. Disambiguate LoRA capacity

The iter-4-thoughts byte-identical result has two plausible causes. Test:

**iter-5-thoughts-scaled** — same corpus as iter-3-rules (30 rows), `preserve_thoughts=True`, but **rank 32 or 64** and **10+ epochs**. If the output diverges from iter-3-rules, interpretation (a) was right and more rank is the lever. If still byte-identical, interpretation (b) holds and SFT on winning thought-traces can't retarget the policy.

**On M5 Max**: this is cheap locally. rank-64 LoRA on ~30-50 rows at bf16 trains in under an hour on Apple Silicon unified memory (no PCIe bottleneck). Gives us the signal we need at effectively zero marginal cost.

**Decision rule**:
- If iter-5-thoughts-scaled diverges meaningfully → invest in the preserve-thoughts lineage at higher scale (iter-6 at N=100).
- If still byte-identical → abandon preserve_thoughts entirely, redirect to environment-shape levers.

### 2. Candlewax redesign — make `conditional_outcome` reachable

Five zero-shot models fail to call `conditional_outcome` across 145+ decisions. Working hypothesis: the tool's *return shape* makes the distribution shape illegible. 85 raw PDF bins is not a gestalt a model can parse into "this is bimodal, I should probe."

**Candidate redesigns**:

**A. Bimodality hint in `eq_outcome_distribution` return.**
```python
{
    "play": 0,
    "mean": 6.572, "stdev": 20.564, "p_make": 0.61,
    # NEW:
    "distribution_shape": "bimodal",
    "modes": [{"center": 22.9, "mass": 0.46}, {"center": -18.2, "mass": 0.15}],
    "gap_between_modes": 41.1,
    "suggested_counterfactuals": [
        {"player": "partner", "holds": 21, "rationale": "high trump — resolves positive mode"},
        {"player": "right_opp", "holds": 15, "rationale": "5-0 — resolves negative mode"},
    ],
}
```
This is the cleanest environment-shape lever. No authored reasoning in training data. The tool invites the counterfactual probe by suggesting specific useful ones. If models pick them up zero-shot, we've validated (B)-category bootstrap without any demo injection.

**B. Tool rename**: `conditional_outcome` → `eq_assuming(play, ...)`. The "question-shaped" name may be more legible.

**C. Meta-tool**: `what_would_change_my_mind(play) → list[assumption]`. Returns the set of unseen-world facts that most swing the E[Q] of a given play. Mechanical, no authored reasoning, directly answers the "when should I probe further?" question.

**Recommended order**: A first (highest signal, requires modifying eq_distribution.py). If it works, we don't need B or C. If it fails to change behavior, C probably does the job with a different slant.

### 3. Pareto frontier — can we land both eq-delta and robustness?

Currently iter-1@r7 (best eq-delta) and iter-3-rules (best robustness) sit at different points. Plausible recipes:

**iter-5-hybrid corpus** — re-harvest on the iter-3-rules prompt shape (rules-as-tools + primer), but explicitly reward rollouts that call `eq_outcome_distribution` in the gate's K1 filter. That's a grading change, not a demo injection — "kept if committed_play matches bot AND at least one non-`is_legal` tool call happened in the trace." It biases the corpus toward the pattern we want without ever telling the model what to think.

**M5-Max-tuned scale bump** — iter-5 at rank 32, N=100 rollouts. 4× the data, 2× the rank. If iter-3-rules's pattern scales gracefully, the combined adapter should close the eq-delta gap.

### 4. Opus ceiling data — HTTP transport productization

T14 partial data showed Opus does reach for `eq_outcome_distribution` naturally when the SDK doesn't break. A clean N=30 would give us:
- The "does the ceiling model ever reach for `conditional_outcome`?" answer definitively (currently 0/6 clean observations, too few).
- Opus's native reasoning shape across the same decision set, as a reference trace for comparison against Burl adapters.

**Effort**: ~60 min to productize the worktree scratch files (`mcp_http_server.py` etc.) into `burl/haiku_spike/agent_http.py` + equivalent runner. No further SDK fighting — HTTP transport is validated.

**Priority**: lower than 1 and 2. Do after iter-5 lands so we can compare iter-5 behavior to a real Opus ceiling.

---

## M5 Max-specific considerations

User just acquired an M5 Max Mac. Shifts in constraints:

### What gets cheaper
- **Local LoRA training**: rank-64 or even rank-128 becomes tractable. Apple Silicon unified memory means no PCIe bottleneck; effective VRAM for bf16 is large for a 5B-effective Gemma 4 E2B (PLE + KV sharing).
- **Iteration speed**: no Modal container cold-start per experiment. Local iteration is minutes not 10-minute cycles.
- **Rollout cost**: if we can run the base model locally for rollouts via MLX or llama.cpp + a patched tool-use path, iter-5 corpus generation stops costing $0.70 per 50 decisions. (Caveat: the existing gemma_serve_native pattern is Modal-bound; local rollout infra is new work, ~half a day.)

### What stays capped
- **Anthropic API / Agent SDK**: still rate-limited by Max 20x weekly quota (generous, not free; `cost_usd` values are subscription-quota, not billing). Claude Code sessions still authenticated.
- **The E[Q] oracle**: `forge/eq/` + Stage-1 Q-value oracle tables are the grading bottleneck. Not GPU-bound; CPU-bound on table lookups. Already fast.
- **Modal for B200 training**: still the sharpest training path for the existing iter-0..4 recipes until someone invests in a local training pipeline. M5 Max is better for small experiments; B200 is better for final iter-N runs that need clean hyperparameter sweeps.

### Changed experiment strategy

With the M5 Max:
- **Run disambiguation experiments locally** (rank-64, longer epochs, sweeping ratio/rank/epochs on the same corpus) — can burn 10+ local training runs in a day.
- **Keep final iter-N runs on Modal B200** — reproducibility and rapid eval.
- **Self-play arena games** — can run locally if we port run_decision_haiku to MLX or use a small local judge model. Currently 7-9 min wall time per game via Claude Agent SDK; local could be similar with less coordination overhead.

---

## Experiment queue (in priority order)

### Tier 1 — do next

**E1: iter-5-thoughts-scaled disambiguation.** rank-64, 10 epochs, same iter-3-rules corpus. Use M5 Max local. Expected wall time ~30 min. Success signal: output diverges from iter-3-rules on any of the 10 N=10 eval decisions.

**E2: Candlewax-aware `eq_outcome_distribution`.** Modify the tool's return shape per candidate A above. Run rollout against base Gemma on 20 decisions with the new tool. Success signal: `conditional_outcome` call rate > 0. Doesn't require training.

### Tier 2 — do after Tier 1 answers

**E3: iter-5-rules-scaled corpus sweep.** If E1 says LoRA capacity matters, run rank-32 + N=100 rollouts with rules-as-tools shape + EQ-gate rationalization. Target: close the eq-delta gap to iter-1@r7 while keeping iter-3-rules's robustness.

**E4: EQ-gate with richer feedback.** Current `tool-nudge` variant says "use your tools to compare." Try `eq-nudge-explicit` that says "call `eq_outcome_distribution` on each candidate before committing" — more directive, risks yes-bias but tests whether the gate can *add* tool-use instead of merely reinforcing existing patterns.

### Tier 3 — long-tail / when curious

**E5: Opus N=30 ceiling via HTTP transport.** Only if we want comparative data for iter-5 vs ceiling.

**E6: Arena with iter-3-rules as all four seats.** Self-play with our own adapter. Does the tool-heavy idiom survive a full 7 tricks when the partner is also using it? Does commit discipline hold?

**E7: Drop `count_dominoes_remaining`.** 0 calls across 145+ decisions. Delete, retrain to ensure nothing implicitly depended on it, see if the streamlined menu changes tool-use patterns.

---

## What we've decided not to do (and why)

**Seed the corpus with Haiku-authored demonstrations.** User is philosophically against this — it's "learning from a stranger's diary" rather than bootstrapping. The reasoning we want has to come from Gemma's own rollouts (with environment shaping), not from imitating Haiku's narrative style.

**Full fine-tuning instead of LoRA.** Scale-up is on the table via rank increase, but moving to full fine-tuning risks overfitting at our N and removes the cleanliness of the LoRA-vs-base comparison. Revisit only if rank-64 still shows byte-identical output at N=100.

**Push harder on spike-v2 shape (no primer).** The iter-3-v2 data showed 32% retry-exhaustion at rollout, 20% at eval. The primer is doing real commit-discipline work; removing it loses more than it gains. Rules-as-tools preamble is the right substitute.

**Chase the `ThinkingBlock.thinking=""` SDK bug.** Separate from the parallel-tool-use bug, Opus's extended thinking output tokens are billed but the extracted thinking string is empty. Likely an SDK issue unrelated to our experiments; flagged but not in scope.

---

## Reading list for a returning session

In order of "pick up context":

1. **`SPIKE_REPORT.md`** — the full scientific log. TL;DR is current as of 2026-04-19.
2. **This file (`ITER4_PLAN.md`)** — the forward plan.
3. **`burl/OVERVIEW.md`** — architectural overview with iter-3-rules noted as current winner.
4. **`burl/experiments/iter3_rules_eval_writeup.md`** — the winner adapter's evaluation in depth.
5. **`burl/experiments/iter4_thoughts_eval_writeup.md`** — the clean-negative result that sets up the LoRA-capacity question.
6. **`burl/experiments/opus_spike_notes.md`** — the `conditional_outcome=0` structural finding.
7. **`burl/experiments/arena_notes.md`** — Haiku vs Opus on a full 7-trick hand; cleanest "what the ceiling looks like" artifact.

All committed in the session ending 2026-04-19. Start with E1 on M5 Max.
