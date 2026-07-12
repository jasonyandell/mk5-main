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

- **Q:** How does the [[batched-harvest-resilience]] wave-sentinel + quarantine layer migrate to a continuous-batching dispatcher?
  - Raised: `c002075` ([[burl-perf-phase2]])
  - Context: Phase 2 lever 2 confirmed continuous batching is a 1.8–2.1× wall win at the bench layer, but the production harvest's resilience plumbing assumes a wave abstraction. Forward path: define a "cohort" that fences a logical group of decisions into the dispatcher pool with a shared sentinel, so quarantine semantics ("this cohort failed") survive. Open: whether mlx-lm 0.31.2's broadcast-shapes bug fires differently under continuous mid-flight prefill vs synchronous wave prefill.

- **Q:** Will burl-lab's phase markers stay harness-private as a post-commit-Q&A adapter co-trains, or get tokenized?
  - Raised: 2026-05-02 (burl-lab platform spec)
  - Context: SPEC.md is explicit that phase identifiers and transitions are not surfaced to the model, since retraining is not a goal of the experimentation platform. Once a co-trained post-commit-Q&A adapter is on the table, the question reopens — phase boundaries are exactly the kind of structure-aware signal a multi-task adapter could exploit, and tokenizing them changes the model's view from "messages a phase produces" to "messages tagged with the phase that produced them." See [[burl-lab]], [[post-commit-q-and-a]], [[play-adapter-lock-in]].

- **Q:** Does burl-lab's HATEOAS `next_tools` advertisement actually shift Burl's tool selection, or does the model still defer to the system-prompt protocol-text even when the prior tool result names the next move?
  - Raised: 2026-05-02 (burl-lab platform spec)
  - Context: The [[improvised-tools]] adoption asymmetry finding (`play_brief` registered but never called because the protocol section named only `explore_game(play=X)` literally) drove two structural fixes in burl-lab: rendered protocol text from active ToolSpecs, and HATEOAS `next_tools` on every tool result. The first is by-construction; the second is empirical. If `next_tools` advertisement does not move adoption, the rendered-protocol-text lever is the only one that does — and the platform's surface area shrinks to "edit the active set, watch the protocol text re-render, see what the model does." See [[burl-lab]], [[improvised-tools]], [[burl-tool-wishlist]].


- **Q:** What additional prompt/tool-response framing lets Gemma choose `19` on Burl microscope case `global_idx=1` without oracle/original-play leakage or human steering?
  - Raised: `local-2026-05-07` ([[burl-microscope]])
  - Context: `board_snapshot()` is a strong first-read surface, but fair no-reference `snapshot-first` and `legal-brief` runs both committed `25` on the `BURL_BREAKS_CONSENSUS` case where oracle/pi/qmean prefer `19`.

- **Q:** What fraction of historical Forge/Burl/Champion states had nonzero
  `WorldSamplerMRV` malformed-world or valid-world bias, and does the uniform
  repair change C0 action ranks or paired marks?
  - Raised: `bc4eb386` ([[world-sampler-mrv-audit]])
  - Context: the causal panel proves exact `1/3` malformed mass on one late
    state and valid-only TVD `0.0333` on another, but finds no argmax flip in
    three states. A state-level exposure scan and two-block C0 reproduction are
    required before revising historical promotion claims.

- **Q:** Does switching forge's production `select_actions` from Lens(p_make) to ev-argmax raise the Zeb-Large win rate above the current 55.7% (E[Q] N=100)?
  - Raised: `afd4802` ([[w42-lens-v1-utility-head-to-head]])
  - Context: `forge/eq/generate/actions.py::select_actions` hardcodes p_make-argmax-with-EV-tiebreak — effectively Lens(p_make), the **worst** of the four utilities in the Lens v1 round-robin (Lens(ev) beats Lens(p_make) by +5.42 pts/hand). The recommended one-line switch to ev-argmax is **NOT applied** as of `afd4802` (verified against the current file plus the Zeb-protocol play path in `forge/zeb/eq_player.py`) — open two months on.

- **Q:** Does jud v2 — a bigger leaf on per-move targets (E[Q] distilled as a bootstrap value) plus opponents-in-rollout — close the play gap the v1 hand-level MLP could not?
  - Raised: `afd4802` ([[w42-jud-v1]], [[jud]])
  - Context: jud v1 held the unification at the auction but was mechanism-limited at play. `judsearch` recovered two-thirds of the play gap oracle-free (JS1 PASS +2.28) but neither more worlds (JS2 below band) nor a better-calibrated head (JS3 falsified) closed the rest. v1's diagnosis: the wall is per-move discrimination — a 470k MLP on hand-level Monte-Carlo labels cannot out-rank E[Q] n=10's per-move oracle. v2's cue follows directly from that diagnosis; unbuilt.

- **Q:** Which era-5 gestation designs (the IDEATED generation — Harl, LLem, walker, and the rest) are worth resurrecting, and which did the built LEM/Burl/Gus/jud line already subsume?
  - Raised: `afd4802` ([[the-gestation]], [[ideated-not-built]])
  - Context: The 52-day zero-commit gestation designed a whole generation out loud that mostly never shipped. The archaeology catalogued the names (classified IDEATED, never claimed as artifacts per anti-rot rule 4) but did not adjudicate which remain live options vs. which the built line already answered.

- **Q:** Does the `contradicted` verdict on high-bid pounce dissolve under an imperfect-information defender (belief/PIMC, no hand visibility), scoped to bid ≥ 35 and bidder ≤ 2 offs?
  - Raised: `5e3f3245` ([[w42-book-second-pass]] §4, [[w42-bookval-v1-wave2-pounce-high-bid]] caveat 0)
  - Context: The book's clause (p. 112, "regardless of whether you know who will win the trick") is an imperfect-information hedge; the probe's E[Q] arms evaluate under sampled-complete-worlds where the oracle always knows the trick winner — a regime in which "pounce regardless" is worse by construction. Prediction: the contradiction dissolves toward neutral-or-positive. Every other `contradicted`/`context-limited` row where the book hedges against *not knowing* deserves the same audit.

- **Q:** Do the book's bid→hand posteriors hold empirically (35 ⇒ two offs/one five-count; 31 ⇒ ≥1 double; shuffler-last-at-30/31 uninformative), and does [[jud]]'s head_8 respect the {30,31,35,36} bid lattice ("if you can bid 32, then you can bid 35")?
  - Raised: `5e3f3245` ([[w42-book-second-pass]] §1, proposed experiment 1)
  - Context: The auction decoder is the highest-value missed cluster — bids as messages, not risk budgets. Testable from bid-aware corpora as P(hand features | bid, seat, who-raised); the lattice is a dominance claim over the whole bid action space with one stated exception (last bidder raising a standing 31).

- **Q:** What does the missing book text say — pages 181–182 and 185–186 are absent from the OCR, the ch 16 four-trump table is truncated (37%/52% conditionals cut mid-row), and worked hands 1–14 are image-only diagrams with no prose reconstruction?
  - Raised: `5e3f3245` ([[w42-book-second-pass]] §Proposed experiments, item 9)
  - Context: Re-scan via the Kindle CDP pipeline (`scratch/winning42/`); the hand diagrams need prose reconstruction to be replayable as probes.

- **Q:** Can an explicit [[w42-book-second-pass|Winning 42]] convention overlay plus a learned fallback become a complete, calibrated blueprint whose partner-decoding gain remains positive after opponent decoding and full-match evaluation?
  - Raised: `1a4482fe` ([[convention-aware-blueprint-search]])
  - Context: Installing the same book overlay on sender and receiver supplies coordinated initialization before unilateral search. The proposed attribution is sender overlay x partner reader x opponent reader; double-dummy value isolates direct technique but does not classify away convention value. Latent opponent book-likeness extends the same design into four-seat action-derived inference. No contrary experiment has been run.
