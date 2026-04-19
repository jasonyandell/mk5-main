# Burl experiments — session record

This directory holds the durable write-ups from experiment batches. Each file
was originally drafted as a working note under `scratch/burl_p5_iter2_prep/`
during an active session and promoted here when the experiment closed.

Files reference `scratch/.../traces.jsonl` and similar paths for the raw
artifacts — those are gitignored and may no longer exist. The writeups here
are the persistent scientific record; traces can be regenerated from the
published HuggingFace adapter names if needed.

## Session 2026-04-19 — iter-2 / iter-3 / iter-4 chapter

### Burl adapter trail
- `iter2_eval_writeup.md` — coverage-only iter-2 on 118-row blend; regressed. Introduced the "corpus-quality dilution" hypothesis.
- `iter3_v2_eval_writeup.md` — spike-v2 prompt shape (no primer) at N=18. Quality held on completed decisions (87.5%) but 20% retry-exhausted. **Surfaced the "primer is load-bearing for commit discipline" finding.**
- `iter3_rules_eval_writeup.md` — rules-as-tools prompt shape at N=30. **The session's winning adapter: 90% bot-match, 0 retry-exhausted, 100% first-legal. `trick_winner_if` per-decision usage went UP after SFT (1.56 → 1.70).**
- `iter4_thoughts_design.md` + `iter4_thoughts_eval_writeup.md` — A/B against iter-3-rules with `preserve_thoughts=True`. **Byte-for-byte identical output**. LoRA capacity ceiling finding.

### Infrastructure design docs
- `corpus_blend_design.md` — LS-Mixture-style shortener + 33% blend ratio. Paired with the `strip_thinking()` schema surprise that reshaped the intervention's meaning.
- `eq_gate_design.md` — rejection-sampling gate protocol for STaR Phase B. Feedback-prompt invariants (no bot-play leakage), classifier branches, yes-bias mitigation.
- `rules_as_tools_design.md` — 1,549-word primer → 645-byte preamble + four tools (`count_dominoes_remaining`, `trick_winner_if`, `what_beats_what`, `contract_progress`).
- `launch_iter2.md` — training launcher + the schema surprise (`strip_thinking()` drops thought blocks pre-tokenization).
- `concurrency_bench.md` — async client-side concurrency for rollouts, 3.92× local benchmark.

### Reference-model spikes
- `haiku_spike_notes.md` — T3, 3-decision smoke via Claude Agent SDK.
- `haiku_full_notes.md` — T8, 30 decisions at $0.78. **First `conditional_outcome=0` observation.**
- `opus_spike_notes.md` — T14 partial (killed). Lock-fix-necessary-but-insufficient + second MCP failure mode + tool-surface finding (`conditional_outcome=0` across 86+ total decisions).
- `arena_notes.md` — T15/T16, Haiku vs Opus full-hand head-to-head. Same bad deal, Opus salvages 7 pts vs Haiku's shutout; `trump_declared` re-query ratio 24:1.
