# Open Questions

Questions raised during replay that the sources at that time do not yet answer. Each entry is tagged with the sha that raised it. When a later ingest resolves a question, move it to `resolved.md` with the resolving sha.

Format:

```
- **Q:** <question>
  - Raised: `<shortsha>` ([[source-page]])
  - Context: <one line>
```

---

- **Q:** Does the rules primer stay in the system prompt forever or get distilled into the model's weights?
  - Raised: `a8bccfa` ([[sources/a8bccfa]])
  - Context: The primer is prepended via `--with-primer`. Whether Stage 0 training removes the need for it in-prompt is an explicit open gap.

- **Q:** Is R1 rationalization sufficient, or will later stages need DPO-style preference learning?
  - Raised: `a8bccfa` ([[sources/a8bccfa]])
  - Context: STaR's [[topics/r1-rationalization]] is simpler than DPO. The frontier leaves room to revisit.

- **Q:** What LoRA rank, learning rate, and epoch count should LEM use? *(Stage 0 partially resolved @ 24ae55a: 1 epoch sufficient, bf16 required; Stage 1 still open.)*
  - Raised: `a8bccfa` ([[sources/a8bccfa]])
  - Context: Defaults from Unsloth's Gemma 4 recipe; "adjust based on what wandb shows." Stage 0 recipe confirmed; Stage 1 hyperparameters remain to be determined.

- **Q:** What variance filter thresholds (`δ`, `σ_max`) should Stage 1 use?
  - Raised: `a8bccfa` ([[sources/a8bccfa]])
  - Context: "Start permissive, tighten from wandb."

- **Q:** What is the right Stage 0 corpus size and category mix?
  - Raised: `a8bccfa` ([[sources/a8bccfa]])
  - Context: "Depends on the parsing sanity check result."

- **Q:** When does Stage 1 plateau? (Ratchet trigger for Stage 2.)
  - Raised: `a8bccfa` ([[sources/a8bccfa]])
  - Context: The frontier commits to ratcheting only after Stage 1 plateaus, but plateau criteria are not specified.

- **Q:** What is the actual K1 ceiling for trick-6 decisions?
  - Raised: `f578bfa` ([[sources/f578bfa]])
  - Context: Base model passes K1 at 60% on 10 decisions. Many trick-6 positions may be near-unanimous-argmax, putting the achievable ceiling well below 100%.

- **Q:** How much of the 30→42% K1 gain is strategy improvement vs rules internalization?
  - Raised: `efad16e` ([[sources/efad16e]])
  - Context: illegal_rate was not tracked per-iteration in the 10-iter run; the diagnostic that would distinguish strategy from rules progress is missing for this dataset.

- **Q:** Is what_beats (15% on v4 eval) under-trained, or is ranking fundamentally harder than membership for this model?
  - Raised: `2f11f32` ([[sources/2f11f32]])
  - Context: All other v4 eval categories are ≥60%; what_beats is isolated at 15%. Unclear whether more training data or a qualitatively different approach is needed.

- **Q:** Does 14B's 97/100 rationalization survive the SFT mask fix?
  - Raised: `be7efc4` ([[sources/be7efc4]])
  - Context: 14B was trained without the mask fix; mask fix changed 1.7B bot-match not at all. Whether 14B's advantage shrinks under correct gradient allocation is unknown.

- **Q:** What breaks the 55/100 bot-match ceiling — capacity or STaR iteration?
  - Raised: `be7efc4` ([[sources/be7efc4]])
  - Context: Mask fix confirmed bot-match is not a gradient-allocation problem. Next candidates are 14B capacity or returning to STaR iteration with the new foundation.

- **Q:** Does the model use conditional_outcome when tool responses are actually visible (post chat-template fix)?
  - Raised: `54f7776` ([[sources/54f7776]])
  - Context: The original 0/145 calls finding was a chat-template confound — tool responses were silently dropped. Whether the model calls conditional_outcome in a correctly wired environment is now genuinely unknown.

- **Q:** Does v2's residual −2.4pp gap on `BURL_BREAKS_CONSENSUS` (vs sequential 560) reflect real batched-mode policy drift or sampling noise?
  - Raised: `063fcac` ([[sources/063fcac]])
  - Context: v1's truncation bug inflated that bucket; v2 dropped from 17.3% → 14.9% in the right direction. The 560 decisions in the sequential pilot align with v2 by `(seed, declaration, narrator_seat, legal_plays)` tuples, admitting a paired McNemar test on bucket flips. Not yet run. If the test fails-to-reject, batched-mode parity is settled; if it rejects, there's a residual systematic shift to characterize before treating the v2 corpus as a drop-in replacement for sequential. See [[experiments/burl-2000-harvest]].

- **Q:** What sets the perf-subset-5 wall floor at 3.4× run-to-run variance on identical config?
  - Raised: `160ed1c` ([[experiments/burl-perf-phase1]])
  - Context: Same `bench_decision_latency.py --variant baseline-bf16 --subset 5 --temperature 0.6` reproduces wall at 40-134 s on M5 Max with no concurrent MLX processes detected (`ps aux` empty of competing python). decode_tok_s tracks the wall: 47-184 tok/s. Likely candidates are OS scheduler contention, Metal compiler-cache warmth, MPS shared-memory pressure, or thermal state — none probed yet. Practical implication: Phase 1's lever wall deltas (which would have been ~10-20% under the original "reduce 8192 → 2048" assumption) are unattributable at this floor; Phase 4's 560-row run is the cleaner measurement.

- **Q:** Does Gemma 4's parallel-tool-call shape return under a future Burl SFT round?
  - Raised: `160ed1c` ([[experiments/burl-perf-phase1]])
  - Context: 297/297 sampled turns under the [[entities/wax-museum]] gate emit one tool call per turn — the post-trained chat-template envelope can carry many, but the gate-state instructions train the model into a single-call rhythm. A future Burl SFT round that includes parallel-tool-call traces (e.g. `belief_trajectory()` + `explore_game(X)` in one assistant turn before any tool response is fed back) would unlock Lever 2 of [[topics/perf-on-the-table]]. Not currently planned; recorded as a precondition.


