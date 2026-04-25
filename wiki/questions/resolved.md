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

- **Q:** Will vLLM return once Gemma 4's transformers incompatibility is resolved, or is HF `model.generate()` the permanent simpler choice?
  - Raised: `8724e93` ([[sources/8724e93]])
  - Resolved: `26f5ddf` ([[sources/26f5ddf]])
  - Answer: HF generate + SDPA is the committed path. The incompatibility is structural — Gemma4ForConditionalGeneration's multimodal weight layout is unsupported by vLLM 0.19.0's LoRA path — not a transient version issue.

- **Q:** Why does the LoRA adapter report missing keys for Gemma 4 E2B layers 15-34?
  - Raised: implicitly at `2c2b851` (star-harness inference quirks)
  - Resolved: `efad16e` ([[sources/efad16e]])
  - Answer: KV-sharing architecture — layers 15-34 have no k/v projections by design. The warning is expected and benign; the adapter is complete.

- **Q:** How much trick-state should the narration restate after each trick?
  - Raised: implicitly at `a8bccfa` (open-voice notes in narration design; never formally logged in open.md)
  - Resolved: `7f1994e` ([[sources/7f1994e]])
  - Answer: Full public-state block after every trick (~60 tok): dominoes played, count status, remaining hand. State visible at the real table belongs to the narrator, not the model.

- **Q:** Can the Stage 1 plateau (36-42%) be broken by more data diversity, scratchpad-validation, or a larger base model?
  - Raised: `efad16e` ([[sources/efad16e]])
  - Resolved: `8c1bb14` ([[sources/8c1bb14]]) — partial
  - Answer: The "better Stage 0 curriculum" path (not in the original list) breaks the plateau: v3 STaR peaks at 48% vs v1's 42%. Original options (a) more data diversity and (c) larger base model remain untested; (b) scratchpad-validation still deferred pending format-bootstrap.

- **Q:** Why does 6-4 stay stubbornly misidentified as trump under fives, even after Kerry curriculum Stage 0 training?
  - Raised: `43009a4` ([[sources/43009a4]])
  - Resolved: `3c33e86` ([[sources/3c33e86]])
  - Answer: Resolved by v4 game-context Q&A training. is_trump scores 100% on 100-example held-out eval — the 6-4 error is gone.

- **Q:** Is fact-verification the only way forward past the Stage 1 plateau?
  - Raised: implicitly at `908773a` (ingest 10 K1-ceiling hypothesis: plateau named as "K1 without fact-verification ceiling")
  - Resolved: `3465e29` ([[sources/3465e29]])
  - Answer: No. Better base model (Qwen 3 1.7B) + better curriculum lifts comprehension from 60% to 100% without fact-verification. The plateau was Stage-0-quality bound, not K1-grading bound. Scratchpad validation remains an option for future work but is not proven necessary.

- **Q:** When will the model have learned the scratchpad format well enough to enable fact-validation? What mechanism will teach the format first?
  - Raised: `78ba940` ([[sources/78ba940]])
  - Resolved: `0c7392f` ([[sources/0c7392f]])
  - Answer: Resolved via a different mechanism than expected. v10 joint training bootstrapped the RATIONALIZATION format (not scratchpad per se) by upweighting rationalization examples in the SFT mix. The format-bootstrap principle worked; scratchpad validation specifically remains shelved.

- **Q:** Will tool-use let a 2B-class model play competently without the comprehension curriculum LEM required?
  - Raised: `8d26e0d` ([[sources/8d26e0d]])
  - Resolved: `3781dce` ([[sources/3781dce]]) — partial
  - Answer: Yes on 10-decision eval: 70% K1 base (XML), 88.9% K1 native. Burl premise survives first contact. Needs larger eval before declaring full competency.

- **Q:** Is retry-on-illegal cheap enough in practice to be Burl's error-correction strategy?
  - Raised: `8d26e0d` ([[sources/8d26e0d]])
  - Resolved: `4b3ba3d` ([[sources/4b3ba3d]])
  - Answer: Yes. Move 3 had 0 illegal moves on the 10-decision eval; retry overhead was zero.

- **Q:** Does the model know WHEN to call which tool — engine vs Zeb vs neither?
  - Raised: `8d26e0d` ([[sources/8d26e0d]])
  - Resolved: `3781dce` ([[sources/3781dce]]) — split answer
  - Answer: No for XML format (only ever calls is_legal). Yes for native format (uses full surface: eq_outcome_distribution 15×, trump_declared 9×). Format is the determining factor.

- **Q:** Does the primer tradeoff (commit discipline vs eq-shy) have a clean resolution?
  - Raised: implicitly at `09b841e` (B4 iter-1 mixed — commit discipline lost with trimmed primer)
  - Resolved: `dbadb5f` ([[sources/dbadb5f]])
  - Answer: Yes. Rules-as-tools + no primer → 90% bot-match and 0 retry-exhausted. Tools cover rules; primer is not needed and actively harmful. Full primer → 70%.

- **Q:** Will native tool-use + rules-as-tools hit the Burl performance target?
  - Raised: implicitly at `b8116b5` (B3 iter-0 pipeline — target was to exceed the 88.9% spike baseline systematically)
  - Resolved: `dbadb5f` ([[sources/dbadb5f]])
  - Answer: Yes on 10-decision eval: 90% bot-match, 0 retry-exhausted, 100% first-legal. Needs larger eval to confirm at scale.
