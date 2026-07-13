# Resolved Questions

Questions that were open and have since been answered by a later ingest. Each entry retains the original raising sha and adds the resolving sha.

Format:

```
- **Q:** <question>
  - Raised: `<shortsha>` ([[source-page]])
  - Resolved: `<shortsha>` ([[source-page]])
  - Answer: <one line>
```

---

- **Q:** Is the `WorldSamplerMRV` `~6.8 Q` discrepancy a real sampler defect
  corrupting historical Burl evaluation and Forge E[Q] data?
  - Raised: `afd4802` ([[batch-throughput-bench]], [[7321952]])
  - Resolved: `bc4eb386` ([[world-sampler-mrv-audit]])
  - Answer: Yes there is a real defect, but not the one the number described.
    The `~6.8 Q` figure is retired as confounded (mixed hand encodings, no
    action/N/RNG provenance). The exact audit falsifies the sampler's validity
    guarantee instead: one historical late state emits malformed worlds with
    probability `1/3` and shifts action value by up to `4.619 Q`, with no
    argmax flip in the three-state panel. The original "by how much,
    corpus-wide" half was not answered; it continues as the narrower exposure
    question in `open.md`.

- **Q:** Will vLLM return once Gemma 4's transformers incompatibility is resolved, or is HF `model.generate()` the permanent simpler choice?
  - Raised: `8724e93` ([[8724e93]])
  - Resolved: `26f5ddf` ([[26f5ddf]])
  - Answer: HF generate + SDPA is the committed path. The incompatibility is structural — Gemma4ForConditionalGeneration's multimodal weight layout is unsupported by vLLM 0.19.0's LoRA path — not a transient version issue.

- **Q:** Why does the LoRA adapter report missing keys for Gemma 4 E2B layers 15-34?
  - Raised: implicitly at `2c2b851` (star-harness inference quirks)
  - Resolved: `efad16e` ([[efad16e]])
  - Answer: KV-sharing architecture — layers 15-34 have no k/v projections by design. The warning is expected and benign; the adapter is complete.

- **Q:** How much trick-state should the narration restate after each trick?
  - Raised: implicitly at `a8bccfa` (open-voice notes in narration design; never formally logged in open.md)
  - Resolved: `7f1994e` ([[7f1994e]])
  - Answer: Full public-state block after every trick (~60 tok): dominoes played, count status, remaining hand. State visible at the real table belongs to the narrator, not the model.

- **Q:** Can the Stage 1 plateau (36-42%) be broken by more data diversity, scratchpad-validation, or a larger base model?
  - Raised: `efad16e` ([[efad16e]])
  - Resolved: `8c1bb14` ([[8c1bb14]]) — partial
  - Answer: The "better Stage 0 curriculum" path (not in the original list) breaks the plateau: v3 STaR peaks at 48% vs v1's 42%. Original options (a) more data diversity and (c) larger base model remain untested; (b) scratchpad-validation still deferred pending format-bootstrap.

- **Q:** Why does 6-4 stay stubbornly misidentified as trump under fives, even after Kerry curriculum Stage 0 training?
  - Raised: `43009a4` ([[43009a4]])
  - Resolved: `3c33e86` ([[3c33e86]])
  - Answer: Resolved by v4 game-context Q&A training. is_trump scores 100% on 100-example held-out eval — the 6-4 error is gone.

- **Q:** Is fact-verification the only way forward past the Stage 1 plateau?
  - Raised: implicitly at `908773a` (ingest 10 K1-ceiling hypothesis: plateau named as "K1 without fact-verification ceiling")
  - Resolved: `3465e29` ([[3465e29]])
  - Answer: No. Better base model (Qwen 3 1.7B) + better curriculum lifts comprehension from 60% to 100% without fact-verification. The plateau was Stage-0-quality bound, not K1-grading bound. Scratchpad validation remains an option for future work but is not proven necessary.

- **Q:** When will the model have learned the scratchpad format well enough to enable fact-validation? What mechanism will teach the format first?
  - Raised: `78ba940` ([[78ba940]])
  - Resolved: `0c7392f` ([[0c7392f]])
  - Answer: Resolved via a different mechanism than expected. v10 joint training bootstrapped the RATIONALIZATION format (not scratchpad per se) by upweighting rationalization examples in the SFT mix. The format-bootstrap principle worked; scratchpad validation specifically remains shelved.

- **Q:** Will tool-use let a 2B-class model play competently without the comprehension curriculum LEM required?
  - Raised: `8d26e0d` ([[8d26e0d]])
  - Resolved: `3781dce` ([[3781dce]]) — partial
  - Answer: Yes on 10-decision eval: 70% K1 base (XML), 88.9% K1 native. Burl premise survives first contact. Needs larger eval before declaring full competency.

- **Q:** Is retry-on-illegal cheap enough in practice to be Burl's error-correction strategy?
  - Raised: `8d26e0d` ([[8d26e0d]])
  - Resolved: `4b3ba3d` ([[4b3ba3d]])
  - Answer: Yes. Move 3 had 0 illegal moves on the 10-decision eval; retry overhead was zero.

- **Q:** Does the model know WHEN to call which tool — engine vs Zeb vs neither?
  - Raised: `8d26e0d` ([[8d26e0d]])
  - Resolved: `3781dce` ([[3781dce]]) — split answer
  - Answer: No for XML format (only ever calls is_legal). Yes for native format (uses full surface: eq_outcome_distribution 15×, trump_declared 9×). Format is the determining factor.

- **Q:** Does the primer tradeoff (commit discipline vs eq-shy) have a clean resolution?
  - Raised: implicitly at `09b841e` (B4 iter-1 mixed — commit discipline lost with trimmed primer)
  - Resolved: `dbadb5f` ([[dbadb5f]])
  - Answer: Yes. Rules-as-tools + no primer → 90% bot-match and 0 retry-exhausted. Tools cover rules; primer is not needed and actively harmful. Full primer → 70%.

- **Q:** Will native tool-use + rules-as-tools hit the Burl performance target?
  - Raised: implicitly at `b8116b5` (B3 iter-0 pipeline — target was to exceed the 88.9% spike baseline systematically)
  - Resolved: `dbadb5f` ([[dbadb5f]])
  - Answer: Yes on 10-decision eval: 90% bot-match, 0 retry-exhausted, 100% first-legal. Needs larger eval to confirm at scale.

- **Q:** When do `SystemSet` / `AdvertisedSet` Move kinds land in burl-lab's `core/transcript.py`, and does `state.json` get fully dropped at that point or does any read path linger?
  - Raised: 2026-05-02 (burl-lab server end-to-end milestone)
  - Resolved: 2026-05-02 (burl-lab journal-canonical milestone, same day)
  - Answer: Resolved fully. `SystemSet`, `AdvertisedSet`, `ToolAdded`, `ToolRemoved` (and `UserText`) are journaled by phase handlers directly via `transcript.append`. `state.json` is gone from the runtime; `server/app.py:_load_state` is `fold(replay(session_dir))` with no snapshot read or write. Verified by `tests/test_server_smoke.py::test_journal_is_canonical_no_state_json` and `test_transcript_roundtrip.py::test_state_journal_only_no_state_json_needed`. See [[burl-lab]] Status section.

- **Q:** What fraction of historical Forge/Burl/Champion states had nonzero `WorldSamplerMRV` malformed-world or valid-world bias, and does the uniform repair change C0 action ranks or paired marks?
  - Raised: `bc4eb386` ([[world-sampler-mrv-audit]])
  - Resolved: 2026-07-13 ([[stage-0-closure]], on a reconstructed proxy population — random legal playouts, not the literal historical stream)
  - Answer: 2.51% of 32,000 tractable late states carried nonzero legacy malformed mass (nonzero median 0.19, max 0.83). Decision harm concentrates in the tail: 20 argmax flips in the 200 worst-mass states, exact regret up to 7.37 Q, biggest per-action shifts (27 Q) mostly cancelling in the argmax. The repair does NOT change paired marks conclusions: C0 reproduces on both reserved blocks (+0.385/+0.486 vs original +0.38/+0.42) and judsearch's deficit reproduces (−1.42/−1.54 vs −1.39). The repaired sampler costs ~2× C0 wall time on MPS.

- **Q:** Does the rules primer stay in the system prompt forever or get distilled into the model's weights?
  - Raised: `a8bccfa` ([[a8bccfa]])
  - Context: The primer is prepended via `--with-primer`. Whether Stage 0 training removes the need for it in-prompt is an explicit open gap.
  - Resolved: 2026-07-13 — closed by pivot (never answered); LEM's two-stage plan never progressed to distilling the primer out, [[rules-as-tools]] replaced textual primers on the Burl line, and the base pivoted Gemma→Qwen.

- **Q:** Is R1 rationalization sufficient, or will later stages need DPO-style preference learning?
  - Raised: `a8bccfa` ([[a8bccfa]])
  - Context: STaR's [[r1-rationalization]] is simpler than DPO. The frontier leaves room to revisit.
  - Resolved: 2026-07-13 — closed by pivot (never answered); Stage 3+ never ran, LEM plateaued in Stage 1 and pivoted to [[burl]] before preference learning was ever needed (see [[r1-rationalization]] "Open questions").

- **Q:** What LoRA rank, learning rate, and epoch count should LEM use? *(Stage 0 partially resolved @ 24ae55a: 1 epoch sufficient, bf16 required; Stage 1 still open.)*
  - Raised: `a8bccfa` ([[a8bccfa]])
  - Context: Defaults from Unsloth's Gemma 4 recipe; "adjust based on what wandb shows." Stage 0 recipe confirmed; Stage 1 hyperparameters remain to be determined.
  - Resolved: 2026-07-13 — closed by pivot (never answered); the Stage 0 recipe was confirmed, but Stage 1 hyperparameter tuning was abandoned when the base model pivoted Gemma→Qwen ([[base-model-pivot-qwen]]).

- **Q:** What variance filter thresholds (`δ`, `σ_max`) should Stage 1 use?
  - Raised: `a8bccfa` ([[a8bccfa]])
  - Context: "Start permissive, tighten from wandb."
  - Resolved: 2026-07-13 — closed by pivot (never answered); the δ/σ_max variance filter was never tuned; LEM's STaR line ended before Stage 1 hardened.

- **Q:** What is the right Stage 0 corpus size and category mix?
  - Raised: `a8bccfa` ([[a8bccfa]])
  - Context: "Depends on the parsing sanity check result."
  - Resolved: 2026-07-13 — closed by pivot (never answered); the corpus grew across curricula (kerry/v3/v4/v9/v10) but the "right size/mix" question dissolved at the Gemma→Qwen pivot rather than settling on an answer.

- **Q:** When does Stage 1 plateau? (Ratchet trigger for Stage 2.)
  - Raised: `a8bccfa` ([[a8bccfa]])
  - Context: The frontier commits to ratcheting only after Stage 1 plateaus, but plateau criteria are not specified.
  - Resolved: 2026-07-13 — closed by pivot (never answered); Stage 1 plateaued (~40%, curriculum-bounded per [[stage-0-progression-star]]) but the Stage 2 ratchet it gated never ran; the project pivoted base model, then to Burl.

- **Q:** What is the actual K1 ceiling for trick-6 decisions?
  - Raised: `f578bfa` ([[f578bfa]])
  - Context: Base model passes K1 at 60% on 10 decisions. Many trick-6 positions may be near-unanimous-argmax, putting the achievable ceiling well below 100%.
  - Resolved: 2026-07-13 — closed by pivot (never answered); the trick-6 K1 ceiling was never characterized; the K1-grading line ended with the LEM base-model pivot.

- **Q:** How much of the 30→42% K1 gain is strategy improvement vs rules internalization?
  - Raised: `efad16e` ([[efad16e]])
  - Context: illegal_rate was not tracked per-iteration in the 10-iter run; the diagnostic that would distinguish strategy from rules progress is missing for this dataset.
  - Resolved: 2026-07-13 — closed by pivot (never answered); the per-iteration illegal_rate that would separate strategy from rules progress was never back-filled for that run, and the STaR line was superseded before re-instrumentation.

- **Q:** Is what_beats (15% on v4 eval) under-trained, or is ranking fundamentally harder than membership for this model?
  - Raised: `2f11f32` ([[2f11f32]])
  - Context: All other v4 eval categories are ≥60%; what_beats is isolated at 15%. Unclear whether more training data or a qualitatively different approach is needed.
  - Resolved: 2026-07-13 — closed by pivot (never answered); what_beats never got a dedicated dig; LEM comprehension work ended before it was retested at scale.

- **Q:** Does 14B's 97/100 rationalization survive the SFT mask fix?
  - Raised: `be7efc4` ([[be7efc4]])
  - Context: 14B was trained without the mask fix; mask fix changed 1.7B bot-match not at all. Whether 14B's advantage shrinks under correct gradient allocation is unknown.
  - Resolved: 2026-07-13 — closed by pivot (never answered); no 14B mask-fix run was ever executed (see [[sft-completion-only-loss]] "Scope at this frontier"); LEM ended the next ingest.

- **Q:** What breaks the 55/100 bot-match ceiling — capacity or STaR iteration?
  - Raised: `be7efc4` ([[be7efc4]])
  - Context: Mask fix confirmed bot-match is not a gradient-allocation problem. Next candidates are 14B capacity or returning to STaR iteration with the new foundation.
  - Resolved: 2026-07-13 — closed by pivot (never answered); neither the capacity nor the further-STaR path was pursued on the 55/100 ceiling; LEM plateaued and pivoted to [[burl]].

- **Q:** Does the model use conditional_outcome when tool responses are actually visible (post chat-template fix)?
  - Raised: `54f7776` ([[54f7776]])
  - Context: The original 0/145 calls finding was a chat-template confound — tool responses were silently dropped. Whether the model calls conditional_outcome in a correctly wired environment is now genuinely unknown.
  - Resolved: 2026-07-13 — closed by pivot (never answered); Burl's tool-orchestration line was superseded by [[jud]]'s pure-NN direction, so the corrected-harness re-test never ran (see [[conditional-outcome-structural-nonuse]]).

- **Q:** Does v2's residual −2.4pp gap on `BURL_BREAKS_CONSENSUS` (vs sequential 560) reflect real batched-mode policy drift or sampling noise?
  - Raised: `063fcac` ([[063fcac]])
  - Context: v1's truncation bug inflated that bucket; v2 dropped from 17.3% → 14.9% in the right direction. The 560 decisions in the sequential pilot align with v2 by `(seed, declaration, narrator_seat, legal_plays)` tuples, admitting a paired McNemar test on bucket flips. Not yet run. If the test fails-to-reject, batched-mode parity is settled; if it rejects, there's a residual systematic shift to characterize before treating the v2 corpus as a drop-in replacement for sequential. See [[burl-2000-harvest]].
  - Resolved: 2026-07-13 — closed by pivot (never answered); the paired McNemar test was never run; the Burl batched-harvest corpus was superseded by [[jud]] before parity was settled.

- **Q:** How does the [[batched-harvest-resilience]] wave-sentinel + quarantine layer migrate to a continuous-batching dispatcher?
  - Raised: `c002075` ([[burl-perf-phase2]])
  - Context: Phase 2 lever 2 confirmed continuous batching is a 1.8–2.1× wall win at the bench layer, but the production harvest's resilience plumbing assumes a wave abstraction. Forward path: define a "cohort" that fences a logical group of decisions into the dispatcher pool with a shared sentinel, so quarantine semantics ("this cohort failed") survive. Open: whether mlx-lm 0.31.2's broadcast-shapes bug fires differently under continuous mid-flight prefill vs synchronous wave prefill.
  - Resolved: 2026-07-13 — closed by pivot (never answered); the continuous-batching dispatcher migration never happened; Burl harvest infrastructure was retired at the [[jud]] pivot.

- **Q:** Will burl-lab's phase markers stay harness-private as a post-commit-Q&A adapter co-trains, or get tokenized?
  - Raised: 2026-05-02 (burl-lab platform spec)
  - Context: SPEC.md is explicit that phase identifiers and transitions are not surfaced to the model, since retraining is not a goal of the experimentation platform. Once a co-trained post-commit-Q&A adapter is on the table, the question reopens — phase boundaries are exactly the kind of structure-aware signal a multi-task adapter could exploit, and tokenizing them changes the model's view from "messages a phase produces" to "messages tagged with the phase that produced them." See [[burl-lab]], [[post-commit-q-and-a]], [[play-adapter-lock-in]].
  - Resolved: 2026-07-13 — closed by pivot (never answered); the co-trained post-commit-Q&A adapter was never built; burl-lab was retired with the Burl line.

- **Q:** Does burl-lab's HATEOAS `next_tools` advertisement actually shift Burl's tool selection, or does the model still defer to the system-prompt protocol-text even when the prior tool result names the next move?
  - Raised: 2026-05-02 (burl-lab platform spec)
  - Context: The [[improvised-tools]] adoption asymmetry finding (`play_brief` registered but never called because the protocol section named only `explore_game(play=X)` literally) drove two structural fixes in burl-lab: rendered protocol text from active ToolSpecs, and HATEOAS `next_tools` on every tool result. The first is by-construction; the second is empirical. If `next_tools` advertisement does not move adoption, the rendered-protocol-text lever is the only one that does — and the platform's surface area shrinks to "edit the active set, watch the protocol text re-render, see what the model does." See [[burl-lab]], [[improvised-tools]], [[burl-tool-wishlist]].
  - Resolved: 2026-07-13 — closed by pivot (never answered); the `next_tools` adoption experiment never ran to conclusion; burl-lab was retired with the Burl line.

- **Q:** What additional prompt/tool-response framing lets Gemma choose `19` on Burl microscope case `global_idx=1` without oracle/original-play leakage or human steering?
  - Raised: `local-2026-05-07` ([[burl-microscope]])
  - Context: `board_snapshot()` is a strong first-read surface, but fair no-reference `snapshot-first` and `legal-brief` runs both committed `25` on the `BURL_BREAKS_CONSENSUS` case where oracle/pi/qmean prefer `19`.
  - Resolved: 2026-07-13 — closed by pivot (never answered); the Burl microscope steering line ended at the [[jud]] pivot; no further framing experiment ran.
