# Index

The catalog of every page in the wiki. Each line: `[[page]] — one-line hook (status)`. Updated every ingest.

## Entities

### Shared infrastructure

- [[entities/texas-42|texas-42]] — the game: 28 dominoes, 2 partnerships, 7 tricks, 42 points per hand (active)
- [[entities/forge|forge]] — pipeline with three components: solver, E[Q] framework, E[Q] bot; shared by LEM and Burl (active)
- [[entities/engine|engine]] — TS game engine (src/core/); authoritative on rules, move legality, state transitions (active)
- [[entities/modal|modal]] — Modal serverless compute platform; L4 for Stage 0, A100 for training, B200 for inference (active)

### LEM — STaR curriculum project

- [[entities/lem|lem]] — the Little Expert Model project: can a small (1.7B-2B) open model learn to play Texas 42 via STaR on a backwards curriculum — 16 ingests from Gemma 4 E2B through Qwen 3 1.7B v10-maskfix (active)
- [[entities/gemma-4-e2b|gemma-4-e2b]] — former LEM base, 2.3B params; retired-as-lem-base; active in Burl for agentic reasons (retired-as-base)
- [[entities/qwen3-1.7b|qwen3-1.7b]] — current LEM base model; 100% comprehension eval; 36K tok/s on B200 (active)
- [[entities/qwen3-14b|qwen3-14b]] — capacity experiment; 97/100 rationalization; visibility_audit 0% structural gap; not adopted as production base (active)
- [[entities/star-harness|star-harness]] — STaR harness: Modal harness, local llama.cpp runner, single-GPU HF generate loop (vLLM abandoned) (active)
- [[entities/stage-0-adapter|stage-0-adapter]] — Stage 0 v1 adapter; superseded by kerry-adapter (superseded)
- [[entities/kerry-adapter|kerry-adapter]] — Stage 0 v2; Kerry curriculum 15k examples; superseded by v3-adapter (superseded)
- [[entities/v3-adapter|v3-adapter]] — Stage 0 v3; Kerry + trump-drill 20k; peak STaR 48%; superseded by v4-adapter (superseded)
- [[entities/v4-adapter|v4-adapter]] — Stage 0 v4; game-context Q&A, Gemma base; 67% eval; superseded by v5-adapter (superseded)
- [[entities/v5-adapter|v5-adapter]] — Stage 0 v5; Qwen base, 100% comprehension; superseded by v9-adapter (superseded)
- [[entities/v9-adapter|v9-adapter]] — Stage 0 v9; 14 categories + rationalization verifier; superseded by v10-adapter (superseded)
- [[entities/v10-adapter|v10-adapter]] — active Stage 0 adapter: Qwen 3 1.7B, joint training, mask fix; 86% comprehension, 55/100 bot-match (active)

### Burl — tool-using agent project

- [[entities/burl|burl]] — tool-using Texas 42 agent; engine authority on rules, Zeb on beliefs, Burl reasons between (active)
- [[entities/zeb|zeb]] — 3.3M-param belief model; parked then superseded by Gus (parked)
- [[entities/burl-iter0-adapter|burl-iter0-adapter]] — first Burl STaR LoRA; baked eq-shy pathology from Layer-1 corpus; 60% bot-match (active)
- [[entities/burl-iter1-adapter|burl-iter1-adapter]] — iter-1 LoRA; trimmed primer (500 vs 1549 words); 5/10 retry-exhausted; 80% bot-match on 5 completed (active)
- [[entities/haiku-4-5|haiku-4-5]] — Claude Haiku 4.5; Burl reference-trace teacher; N=30 run: 72.4% bot-match; zero conditional_outcome usage (active)
- [[entities/iter3-rules-adapter|iter3-rules-adapter]] — winning Burl adapter: rules-as-tools + no primer; 90% bot-match, 0 retry-exhausted, 100% first-legal (active)
- [[entities/selfplay-arena|selfplay-arena]] — game-level eval harness; Burl vs Burl or Burl vs baseline; ships with B7 (active)
- [[entities/mlx-lm|mlx-lm]] — local Apple Silicon inference path; 1.86× wall vs Modal; batch ceiling 43→1334 tok/s (16×) after B9 bench (active)
- [[entities/candlewax-spike|candlewax-spike]] — E2E spike: Qwen 3.6-35B-A3B via mlx-vlm; reasoning-coherence verifier; pivots away from LLM-as-reasoner (active)
- [[entities/wax-museum|wax-museum]] — hard-gated HATEOAS harness with three extension hooks: system_prompt_transform, preload_tool_calls, menu_override (active)
- [[entities/belief-trajectory|belief-trajectory]] — tool wiring Gus's calibrated belief head into Burl; replaces static E[Q] PDF primitive (active)

### Gus — oracle distillation project

- [[entities/gus|gus]] — third sibling project; distills forge's E[Q] oracle via multi-head transformer; skips reasoning channel; 5-head LAMIR-ready (active)
- [[entities/joint-world-tensor|joint-world-tensor]] — packed game-state representation enabling look-ahead without oracle calls; core Gus data structure (active)
- [[entities/gen-fleet|gen-fleet]] — Vast.ai distributed corpus generation plan; 5 pre-launch fixes identified (bid=30 bias, schema v2, lazy dataset done); not yet launched (plan)

## Topics

- [[topics/star|star]] — Self-Taught Reasoner; LEM's Stage 1 training paradigm (active)
- [[topics/backwards-curriculum|backwards-curriculum]] — start at the end of the game, ratchet backward one ply per stage (active)
- [[topics/rules-adapter|rules-adapter]] — Stage 0: 3500-example Q&A corpus across 7 categories, primer 55/55 verified; Q&A drilling does not transfer to narration (active)
- [[topics/narration|narration]] — Texas 42 game → second-person prose from one player's seat (active)
- [[topics/k1-grading|k1-grading]] — "beat the bot it replaced": STaR keep rule, `E[Q][gemma] ≥ E[Q][bot]` (active)
- [[topics/r1-rationalization|r1-rationalization]] — STaR failure branch: reveal the bot's action, train on the model's rationalization of it (active)
- [[topics/expected-q-value|expected-q-value]] — E[Q]: scalar measuring position value, supplied by forge's Q-value checkpoint (active)
- [[topics/lora-unsloth|lora-unsloth]] — parameter-efficient fine-tuning; full reproducible Stage 0 recipe: ClippableLinear patch, bf16, gradient checkpointing, eval disabled, 1 epoch (active)
- [[topics/learned-by-playing|learned-by-playing]] — key insight: Q&A drilling transfers hand-tracking but not trump membership; rules learn by playing (active)
- [[topics/scratchpad-validation|scratchpad-validation]] — attempted structured output + fact-validation in one step; 64.5% invalid; format-bootstrap lesson; code parked for later (retired)
- [[topics/kerry-curriculum|kerry-curriculum]] — A/B/C/D curriculum from Kerry Newberry's Learner's Guide; 15k examples (A 15%, B 20%, C 40%, D 25%) used in Stage 0 v2 (active)
- [[topics/trump-drilling|trump-drilling]] — 5 drill types (is_trump/list_trumps/which_trumps/trump_or_follow/count_trump); 5k examples added in Stage 0 v3 (active)
- [[topics/game-context-qa|game-context-qa]] — 5-type game-context Q&A from real game records, ~170 tok prompts; expanded to 14 categories in v9 (active)
- [[topics/rationalization-verifier|rationalization-verifier]] — 6 engine checks verifying rationalizations against ground truth; introduced in v9 (active)
- [[topics/single-fact-enumeration|single-fact-enumeration]] — evaluation method: one fact per Q, enables precise category-level accuracy tracking (active)
- [[topics/tool-orchestration|tool-orchestration]] — Burl's core philosophy: engine authority on rules, Zeb on beliefs, Burl reasons between and commits (active)
- [[topics/eq-gate-star|eq-gate-star]] — staged iter-2/3 workstream: gate STaR keep on E[Q] delta, not just K1 match (active)
- [[topics/ls-mixture|ls-mixture]] — staged workstream: mix legal-but-suboptimal traces into SFT to train commit discipline (active)
- [[topics/rules-as-tools|rules-as-tools]] — validated: rules-as-tools + no primer → iter-3 winner at 90% bot-match; trick_winner_if usage UP post-SFT confirms tools-replace-memorization (active)
- [[topics/reference-trace-distillation|reference-trace-distillation]] — staged workstream: distill Haiku 4.5 reference traces into Gemma via SFT (active)
- [[topics/preserve-thoughts|preserve-thoughts]] — confirmed at N=560: thought-block emission 0% → ~95% (phase change), AND a play-quality win on paired n=130 (regret 2.255 vs 3.016, −25%, identical 32.3% FC rate) (confirmed)
- [[topics/commit-discipline-collapse|commit-discipline-collapse]] — strict-pool training disrupts when-to-commit timing without disrupting what-to-commit quality; harness's force-fallback substitutes for missing commit signal (forced regret 1.60 < not-forced 2.46); cost is yield, not quality (active)
- [[topics/iter-without-regression|iter-without-regression]] — milestone: harvest-2 + run-4 is the first Burl iteration with no new bug, pathology, or regression; every prior iter introduced one (eq-shy, retry-exhausted, rank-collapse, loss-collapse, three launch failures); foundation for real iter-N comparisons (active)
- [[topics/conditional-outcome-structural-nonuse|conditional-outcome-structural-nonuse]] — reframed: 0/145 calls was chat-template confound (tool responses invisible); open question is whether model uses it when visible (active)
- [[topics/candlewax|candlewax]] — reasoning-coherence verification approach: multimodal model checks reasoning traces against game state; bypasses LLM-as-reasoner bottleneck (active)
- [[topics/reasoning-coherence-verification|reasoning-coherence-verification]] — identified bottleneck in Burl STaR loop; model produces syntactically valid traces that are semantically incoherent (active)
- [[topics/student-distillation|student-distillation]] — Gus training paradigm: distill forge's E[Q] oracle into a small transformer via supervised learning on oracle outputs (active)
- [[topics/lamir1|lamir1]] — 5 rollout modes + lamir1-piopp; direct 0.551 beats all 8 look-ahead variants; Q_head OOD at depleted leaves is root cause; Bug 6 + Fix 6 documented (active)
- [[topics/dense-q-supervision|dense-q-supervision]] — 3400× per-decision signal: Q supervision regularizes the shared encoder; both training-time regularizer and inference-time LAMIR primitive (active)
- [[topics/pimc|pimc]] — Perfect-Information Monte Carlo inference variants; direct π_me beats single-step PIMC because policy head already is the marginalized policy (active)
- [[topics/regret-eval|regret-eval]] — primary Gus quality metric; bimodal: 73% perfect, 6% blunder tail drives all mean regret; near-tie rate 70-75%; ported to Burl 2026-04-26: paired in-distribution n=180 has run-3c at 1.92 regret vs naked-Burl 3.13 (−39%) (active)
- [[topics/v-pi-decoupling|v-pi-decoupling]] — V_head correctly predicts +26 while π_me picks −0.4 play; heads decouple because training objectives only couple them indirectly (active)
- [[topics/consistency-regularizer|consistency-regularizer]] — v3 at 10k: 0.551 regret (first Gus adapter under 1.0); gap vs v2 widens from tied at 3k to −33% at 10k (active)
- [[topics/probe-analysis|probe-analysis]] — 6-probe interpretability receipt on v3-10k; counterfactual V tracks oracle within 0.5 Q-pts; embedding structure + attention patterns confirm real game learning (active)
- [[topics/qmae-plateau|qmae-plateau]] — Q_head scaling wall: 3k→10k improves qMAE only 7% vs regret −60%; structural cause is one-world-per-forward-pass; fix paths documented (active)
- [[topics/shine-analysis|shine-analysis]] — 73% perfect-decision rate: 59% dead-ties, 25.4% sharp-and-perfect; zero-inference routing heuristic (legal_count ≤ 2 OR dec ≥ 22) covers 80% of decisions (active)
- [[topics/belief-propagation-gap|belief-propagation-gap]] — calibration fine-tune improves KL 21% but downstream Q/PIMC regress; co-training the full {belief, world_encoder, Q_head} cluster is prerequisite (active)
- [[topics/lazy-iterable-dataset|lazy-iterable-dataset]] — streaming IterableDataset bounds memory to one chunk + buffer; unlocks 10k+ game corpora; --lazy flag on training scripts (active)
- [[topics/blunder-detector|blunder-detector]] — GBM classifier predicting student blunders (regret > 8 Q-pt); v1 oracle AUC 0.926, v2 student-only AUC 0.839; top feature: pi_peak (active)
- [[topics/detect-and-route|detect-and-route]] — blunder-gated inference wrapper; oracle fallback 0.49 regret at 25% flag; PIMC-Q and next-best-adapter both hurt (active)
- [[topics/router-reality-check|router-reality-check]] — honest PoC conclusion: oracle routing works, every non-oracle fallback hurts; prerequisite for no-oracle path is Q_head multi-world variance regularization (active)
- [[topics/pi-opp-head|pi-opp-head]] — 1,879-param PiOppHead trained on oracle softmax; 68.6% accuracy vs ~55% rotated π_me; real side product of LAMIR-1 work (active)
- [[topics/lamir1-ceiling|lamir1-ceiling]] — direct 0.551 beats all 8 look-ahead modes; q-bootstrap-belief 0.655 closest look-ahead; scalar V/Q noise vs T×T matrix; four pivot options (active)
- [[topics/q-head-augmentation|q-head-augmentation]] — path (a) closed: random depletion ≠ structured causal depletion; 2.216 regret — augmented Q_head worse than pre-fix rollouts (retired)
- [[topics/belief-bayes-ceiling|belief-bayes-ceiling]] — top-1 accuracy solved at 39.184%; early-game information scarcity is the cause; remaining lever is posterior shape (active)
- [[topics/belief-co-train|belief-co-train]] — co-train falsified (KL −20%, regret worse); q-bootstrap-belief 0.655 unexpected win — belief-sampled worlds beat corpus worlds (active)
- [[topics/past-belief-future-direction|past-belief-future-direction]] — §22: acting well under unresolvable uncertainty; meta-strategy distribution as richer student output; analytics extractable from q_per_world (active)
- [[topics/batched-harvest-resilience|batched-harvest-resilience]] — three-part pattern: OOM classifier + quarantine ledger + SIGKILL sentinel; turns multi-hour batched harvest from one-failure-or-restart into one-failure-or-retry-six-decisions (active)
- [[topics/batched-eval-resilience|batched-eval-resilience]] — eval-side port of the harvest pattern: atomic write + per-wave summary roll-up + in-wave per-decision OOM fallback + --resume-dir; turns 4h evals into "lose at most 3 min on any failure" (active)
- [[topics/perf-on-the-table|perf-on-the-table]] — Gemma 4 E2B benches at 1334 tok/s; harness runs at ~70 tok/s — six identified levers compound to ~7-10× on M5 Max alone, no model changes (active)

## Experiments

- [[experiments/first-gemma-contact|first-gemma-contact]] — a8bccfa: first Gemma 4 E2B inference pass, no adapter, showed coherent reasoning with state/trump-membership gaps (active)
- [[experiments/stage-0-v1-training|stage-0-v1-training]] — Modal L4, 1 epoch, 208 steps, loss 32→0.001; adapter published to HuggingFace (active)
- [[experiments/second-gemma-contact|second-gemma-contact]] — adapter-loaded re-run: hand-tracking fixed, trump membership still broken (active)
- [[experiments/star-harness-5ex-smoke|star-harness-5ex-smoke]] — 5-example Modal smoke test: 1 pass, 2 rationalized, 2 illegal-rationalized (active)
- [[experiments/base-model-k1-baseline|base-model-k1-baseline]] — surprising 60% K1 pass on 10 trick-6 decisions with base model (no adapter) (active)
- [[experiments/star-iter-0|star-iter-0]] — Stage 1 iteration 0: 30% K1 pass, 40% illegal, 15 min on H100, ~$1 (active)
- [[experiments/scratchpad-v2-iter0|scratchpad-v2-iter0]] — scratchpad validation attempt: 64.5% invalid on 5 traces; ran once before revert (retired)
- [[experiments/star-10-iterations|star-10-iterations]] — STaR Stage 1: 15 iterations complete; 38-42% plateau confirmed; K1-without-fact-verification ceiling hypothesis; 15 adapters, ~$25 total (active)
- [[experiments/third-gemma-contact|third-gemma-contact]] — Kerry adapter eval: trump non-membership fixed, 6-4 case narrowed, strategic depth dramatically improved (active)
- [[experiments/stage-0-progression-star|stage-0-progression-star]] — v1/Kerry/v3 STaR progression table: avg 37/43/44, peak 42/46/48, illegal 33/12/13; each curriculum raises the floor (active)
- [[experiments/stage-0-v4-comprehension-eval|stage-0-v4-comprehension-eval]] — 100-example held-out eval: 67% overall, is_trump 100%, where_is 90%, legal_moves 70%, count_status 60%, what_beats 15% (active)
- [[experiments/stage-0-v9-14categories|stage-0-v9-14categories]] — v9 14-category eval; rationalization verifier introduced (active)
- [[experiments/qwen-14b-capacity|qwen-14b-capacity]] — 14B capacity experiment: 97/100 rationalization; visibility_audit 0% on both 1.7B and 14B — structural gap (active)
- [[experiments/v10-maskfix-breakthrough|v10-maskfix-breakthrough]] — mask fix: TRL SFTConfig was diluting answer gradient ~9×; prompt/completion format fixes it; 86% comprehension, 55/100 bot-match (active)
- [[experiments/zeb-calibration-eval|zeb-calibration-eval]] — Zeb advertised 72% top-1 → actual 39% hidden-only; calibration gap caused parking decision (active)
- [[experiments/burl-move3-base|burl-move3-base]] — base Gemma on XML harness: 100% legal, 70% K1, 60% bot-match; only calls is_legal (active)
- [[experiments/burl-move4-native-spike|burl-move4-native-spike]] — native tool-use format: 88.9% K1 + 88.9% bot-match; full tool surface used (eq_outcome_distribution 15×, trump_declared 9×) (active)
- [[experiments/burl-phase1-primer|burl-phase1-primer]] — Phase 1: primer corpus build; eq-shy pathology identified in Layer-1 training data (active)
- [[experiments/burl-phase2-starcorpus|burl-phase2-starcorpus]] — Phase 2: STaR corpus generation for iter-0; vLLM-LoRA blocker resolved via hf_overrides (active)
- [[experiments/burl-iter0-eval|burl-iter0-eval]] — iter-0 end-to-end eval: 60% bot-match, regressed 10pp from Layer-1 baseline; eq-shy baked into weights (active)
- [[experiments/burl-iter1-mixed|burl-iter1-mixed]] — iter-1 trimmed-primer eval: 5/10 retry-exhausted; 80% bot-match + -0.76 eq_delta on 5 completed; commit discipline was load-bearing (active)
- [[experiments/iter3-comparison|iter3-comparison]] — iter-3 conditions compared: rules-as-tools + no primer wins at 90% bot-match; full primer at 70%; null iter-4 preserves score (active)
- [[experiments/iter4-null-preserve-thoughts|iter4-null-preserve-thoughts]] — preserve_thoughts A/B: reframed as SFTConfig truncation artifact (max_seq_length=1024 clipped thought-bearing rows); LoRA capacity hypothesis retired (retired)
- [[experiments/opus-vs-haiku-arena|opus-vs-haiku-arena]] — selfplay arena: Opus 7 / Haiku 0 on seed 900010; Opus 1× trump_declared vs Haiku 24× — major tool-economy delta (active)
- [[experiments/iter5-e1-rank-sweep|iter5-e1-rank-sweep]] — iter-5 E1: rank-16 with preserve_thoughts + truncation fix; 70% bot-match, -2.83 eq_delta (active)
- [[experiments/iter5-e2-candlewax-null|iter5-e2-candlewax-null]] — iter-5 E2: candlewax-aware null; bimodality at tool surface doesn't change behavior (active)
- [[experiments/batch-throughput-bench|batch-throughput-bench]] — MLX batch bench: 43 → 1334 tok/s (16×) after batching fix (active)
- [[experiments/candlewax-spike-e2e|candlewax-spike-e2e]] — candlewax E2E spike: Qwen 3.6-35B-A3B via mlx-vlm; reasoning-coherence verifier works end-to-end (active)
- [[experiments/chat-template-fix-validation|chat-template-fix-validation]] — post-fix base Gemma: 5/5 bot-match on N=5 held-out; faithfully quotes numeric tool responses (active)
- [[experiments/gus-joint-world-tire-kick|gus-joint-world-tire-kick]] — first Gus smoke test: joint-world tensor shape validation and forward-pass sanity check (active)
- [[experiments/gus-v0-v1-belief|gus-v0-v1-belief]] — v0 MLP 34.6% overfit → v1 transformer 37.5% data-bound; late-game 75% confirms architecture works (active)
- [[experiments/gus-4head-baseline|gus-4head-baseline]] — full 4-head student on 100g: π_me 57.9% with no overfit; dense Q supervision is the regularizer (active)
- [[experiments/gus-v2-voids-1000g|gus-v2-voids-1000g]] — 1000g scaling + explicit voids: π_me 66% flat, belief +1.4pp; transformer already inferred voids attentionally (active)
- [[experiments/gus-lamir-primitive-eval|gus-lamir-primitive-eval]] — direct π_me 65.4% beats PIMC K=1 (62.1%) and K=50 (61.8%); multi-step LAMIR deferred until π_opp trained (active)
- [[experiments/gus-scaling-ladder|gus-scaling-ladder]] — adapter ladder 100g→3000g: best v2_voids_3000g_big at 1.39 Q-pt regret; data dominates capacity; scaling continues without plateau (active)
- [[experiments/gus-arena-pilot|gus-arena-pilot]] — first game-level eval: 50% contracts made vs 80% all-bot baseline; 1.39 Q-pt regret compounds to ~30pp game gap; V/π decoupling surfaced (active)
- [[experiments/gus-shine-analysis|gus-shine-analysis]] — characterizes 410/560 perfect decisions; 104 sharp-and-perfect prove genuine mid-game capability; two-stage routing heuristic derived (active)
- [[experiments/gus-belief-calibration-diagnostic|gus-belief-calibration-diagnostic]] — frozen-trunk belief fine-tune: KL −21%, PIMC regressed 1pp; receipt 15; co-train prerequisite (active)
- [[experiments/gus-blunder-detector|gus-blunder-detector]] — oracle-feature (AUC 0.926) and student-feature (AUC 0.839) blunder classifiers; 20% flag rate → 1.13→0.49 regret; router > ensemble (active)
- [[experiments/gus-router-pilot|gus-router-pilot]] — detect-and-route end-to-end validation; oracle fallback works (0.49 regret at 25%); PIMC-Q-K50 hurts due to Q_head signal noise (active)
- [[experiments/gus-v3-consistency-full-run|gus-v3-consistency-full-run]] — first sub-1.0 regret: v3 at 10k → 0.551; v2-3k→v3-10k total −60%; consistency loss rides forward into LAMIR-1 (active)
- [[experiments/gus-probe|gus-probe]] — interpretability probes on v3-10k; counterfactual V oracle agreement ±0.5 Q-pts; 6-6 impact trumpness-gated; game structure internalized (active)
- [[experiments/gus-lamir1-pilot|gus-lamir1-pilot]] — first LAMIR-1 rollout: 2.384 regret (vs 0.551 direct); adversarial leaf states from argmax π_me as π_opp identified as root cause (active)
- [[experiments/gus-lamir1-mode-comparison|gus-lamir1-mode-comparison]] — 4 modes compared; v-bootstrap isolates V_head distribution shift (not opp sim); Bug 5 + Fix 2 found; q-bootstrap/qleaf added (active)
- [[experiments/gus-pi-opp-training|gus-pi-opp-training]] — frozen trunk + PiOppHead; 68.6% oracle accuracy; NaN bug (0×−∞) found and fixed; Schema v2 loader extended (active)
- [[experiments/gus-lamir1-piopp|gus-lamir1-piopp]] — full 8-mode ladder; lamir1-piopp worse than lamir1-qleaf; Bug 6 + depletion OOD root cause; four §20 pivot options (active)
- [[experiments/gus-q-head-augmentation|gus-q-head-augmentation]] — path (a) postmortem: aug Q_head 2.216 regret, worse than baseline; random ≠ causal depletion; path closed (active)
- [[experiments/gus-belief-co-train|gus-belief-co-train]] — §21: Bayes ceiling 39.184% confirmed; co-train falsified; q-bootstrap-belief 0.655 best look-ahead; §22 future direction (active)
- [[experiments/burl-2000-harvest|burl-2000-harvest]] — Burl 2000-decision batched harvest on D_required_first; 5h 46m, 0 quarantines, 0 illegal commits; strict pool 1062, BURL_BREAKS_CONSENSUS 299; v1 truncation bug caught and fixed (active)
- [[experiments/burl-star-run3|burl-star-run3]] — filter-only STaR on the 1062-row strict pool; preserve-thoughts is load-bearing (run-3c thoughts ~95%, run-3b 0%); paired in-distribution n=180 confirms run-3c beats naked-Burl on oracle regret by **−39%** (1.92 vs 3.13); FORCED_COMMIT 12%→34% is decision-shape not play-quality cost (active)
- [[experiments/burl-harvest-2|burl-harvest-2]] — first STaR self-sharpening test: harvest-2 with run-3c-as-rollout + filter-only run-4. Strategic distribution unchanged from harvest-1 (strict pool 1062→1075); play quality plateaued (run-4 regret 2.97 ≈ base 3.13, regresses ~1.05 vs run-3c 1.92); FC repaired (33.9%→9.4%); earlier ILLEGAL=28.6% read corrected to a tagger denominator artifact (active)
- [[experiments/burl-perf-phase0|burl-perf-phase0]] — Phase 0 perf bench: 5-row hermetic subset (gi=0/36/72/104/136 across declarations 0..4 + trick positions 1/3/5/6/7); baseline-bf16 at 79.5s/5dec, 87 decode tok/s, 11.6 GB; bench drives the production batched eval path and tracks ledger CSV + per-run JSON under `burl/eval/results/` (active)

## Decisions

- [[decisions/eval-seed-holdout|eval-seed-holdout]] — seeds 900000–909999 permanently held out for eval; never used in training (active)
- [[decisions/discard-illegal-traces|discard-illegal-traces]] — illegal and parse-fail traces are discarded, not rationalized; illegal_rate becomes a diagnostic metric (active)
- [[decisions/public-state-block|public-state-block]] — post-trick public state block (dominoes played, count, hand) appended to narration; state visible at the table belongs to the narrator, not the model (active)
- [[decisions/flexible-grader|flexible-grader]] — grade by fact-extraction from free-form responses, not rigid pattern match; moved legal_moves 0% → 70% on same responses (active)
- [[decisions/base-model-pivot-qwen|base-model-pivot-qwen]] — pivot base from Gemma 4 E2B to Qwen 3 1.7B; dual rationale: accuracy (100% vs 60%) + throughput (36K tok/s); t42-hv08 resolved (active)
- [[decisions/sft-completion-only-loss|sft-completion-only-loss]] — use prompt/completion dataset format to enable completion-only loss; fixes ~9× gradient dilution from memorized prompt tokens (active)
- [[decisions/zeb-parked-eq-primitive|zeb-parked-eq-primitive]] — Zeb parked after calibration gap discovered; E[Q] N=10 PDF is new Burl belief primitive (active)
- [[decisions/native-tool-use-format|native-tool-use-format]] — harness bends to Gemma's post-training grammar; native tool-use format unlocks full tool surface vs XML-only is_legal (active)
- [[decisions/primer-tradeoff|primer-tradeoff]] — primer teaches rules but also bakes eq-shy behavior; trimmed primer is B4's planned fix (active)
- [[decisions/commit-discipline|commit-discipline]] — commit discipline (willingness to play without retrying) is load-bearing; dropping primer entirely kills it (active)
- [[decisions/sft-max-seq-length|sft-max-seq-length]] — set SFTConfig max_seq_length=4096; TRL default 1024 was clipping thought-bearing rows (median 2054, max 4210); parallel to sft-completion-only-loss trap (active)
- [[decisions/gemma-tool-response-shape|gemma-tool-response-shape]] — Gemma 4 Jinja chat-template silently drops role="tool" messages; fix: wrap tool response as user turn with numeric content (active)
- [[decisions/max-tokens-2048-floor|max-tokens-2048-floor]] — batched Burl harvest minimum per-turn cap; sequential p99 1770 chars / max 2639 ≈ 900 tok; 1024 truncates ~1% of turns mid-thought (active)
- [[decisions/resumable-checkpointing|resumable-checkpointing]] — Burl STaR trainer writes periodic on-disk adapters, accepts `--resume`, and persists the in-memory best on any crash; closes the run-3 OOM data-loss footgun (active)

## Sources

- [[sources/a8bccfa|a8bccfa]] — 2026-04-09: narration generator, rules primer, first Gemma contact
- [[sources/6bb8a40|6bb8a40]] — 2026-04-09: Stage 0 infra (primer verification, Q&A generator, training script)
- [[sources/9571a7b|9571a7b]] — 2026-04-09: PEFT fix (Gemma4ClippableLinear → nn.Linear, bf16)
- [[sources/df73c8d|df73c8d]] — 2026-04-10: Stage 0 complete (training success, second contact)
- [[sources/24ae55a|24ae55a]] — 2026-04-10: OVERVIEW progress log + Stage 1 plan
- [[sources/b99c64d|b99c64d]] — 2026-04-10: batch narration generator, eval seed holdout, A100 switch
- [[sources/7538016|7538016]] — 2026-04-10: STaR harness commit
- [[sources/f578bfa|f578bfa]] — 2026-04-10: local runner + K1 baseline measurement
- [[sources/8c5fbca|8c5fbca]] — 2026-04-10: single-GPU loop (vLLM; later abandoned)
- [[sources/6e71df9|6e71df9]] — 2026-04-10: trivial Modal GPU fix
- [[sources/fb47ab3|fb47ab3]] — 2026-04-10: discard-illegal-traces policy; ClaudeAI credited for insight
- [[sources/68a0416|68a0416]] — 2026-04-10: conditional ClippableLinear patch (vLLM clash workaround)
- [[sources/8724e93|8724e93]] — 2026-04-10: vLLM replaced with HF batch generate
- [[sources/d913932|d913932]] — 2026-04-10: last vLLM reference removed
- [[sources/576b694|576b694]] — 2026-04-10: OVERVIEW progress-log addition; compute recipe documented
- [[sources/2c2b851|2c2b851]] — 2026-04-10: vLLM migration attempt on B200 (lifespan: 80 min)
- [[sources/26f5ddf|26f5ddf]] — 2026-04-10: vLLM out again; SDPA + torch.compile + batching settles at 120 tok/s
- [[sources/380f3fa|380f3fa]] — 2026-04-11: introduce scratchpad validation
- [[sources/c88582c|c88582c]] — 2026-04-11: chain script for scratchpad
- [[sources/34775ca|34775ca]] — 2026-04-11: pass_rate → valid_pass_rate rename
- [[sources/b12fcec|b12fcec]] — 2026-04-11: relax scratchpad validation to hand-only
- [[sources/78ba940|78ba940]] — 2026-04-11: revert to simple K1
- [[sources/5946c94|5946c94]] — 2026-04-11: iterate.sh back to v1; scratchpad saga closes
- [[sources/ff0d0d2|ff0d0d2]] — 2026-04-11: combined dataset 3148 → 7409 examples
- [[sources/efad16e|efad16e]] — 2026-04-11: OVERVIEW update with full 10-iter results and next-steps
- [[sources/908773a|908773a]] — 2026-04-11: iters 10-14 added; plateau confirmed; ceiling hypothesis stated
- [[sources/7f1994e|7f1994e]] — 2026-04-11: public state block after every trick; narration v3 format
- [[sources/f8cdbe7|f8cdbe7]] — 2026-04-11: Kerry Q&A corpus commit (15k examples, A/B/C/D curriculum)
- [[sources/43009a4|43009a4]] — 2026-04-11: Kerry adapter eval; third-contact results
- [[sources/a2498e4|a2498e4]] — 2026-04-11: v3 trump-drill corpus + adapter training
- [[sources/601f622|601f622]] — 2026-04-11: v3 STaR iterations; progression table
- [[sources/8c1bb14|8c1bb14]] — 2026-04-11: OVERVIEW update; plateau-as-curriculum-bound reframing
- [[sources/4729dad|4729dad]] — 2026-04-13: game-context Q&A corpus + v4 adapter training
- [[sources/1d3e1b7|1d3e1b7]] — 2026-04-13: eval bugs fixed (EOS token, left-pad slicing); flexible grader introduced
- [[sources/3c33e86|3c33e86]] — 2026-04-13: 100-example held-out eval; thinking-mode-off finding
- [[sources/2f11f32|2f11f32]] — 2026-04-13: OVERVIEW update; v4 results and what_beats gap noted
- [[sources/3465e29|3465e29]] — 2026-04-16: Qwen 3 1.7B pivot; v5-adapter; 100% comprehension
- [[sources/b857299|b857299]] — 2026-04-17: v9 14-category corpus + rationalization verifier
- [[sources/0c7392f|0c7392f]] — 2026-04-17: 14B capacity experiment + v10 joint training
- [[sources/be7efc4|be7efc4]] — 2026-04-17: mask fix + v10-maskfix eval; LEM finale OVERVIEW
- [[sources/8d26e0d|8d26e0d]] — 2026-04-18: Burl introduction; vocabulary cleanup; LEM→Burl handoff trail
- [[sources/d9baf3b|d9baf3b]] — 2026-04-18: Burl harness scaffold; Zeb calibration eval
- [[sources/4b3ba3d|4b3ba3d]] — 2026-04-18/19: Move 3 base eval (XML); Zeb parking decision
- [[sources/3781dce|3781dce]] — 2026-04-19: Move 4 native tool-use spike; 88.9% K1
- [[sources/b8116b5|b8116b5]] — 2026-04-19: Phase 1 primer corpus; eq-shy pathology identified
- [[sources/fd6032b|fd6032b]] — 2026-04-19: Phase 2 STaR corpus generation; vLLM-LoRA blocker resolved
- [[sources/0168210|0168210]] — 2026-04-19: Phase 3-4 training + primer-tradeoff decision
- [[sources/789e14d|789e14d]] — 2026-04-19: iter-0 eval; 60% bot-match; regression diagnosed
- [[sources/09b841e|09b841e]] — 2026-04-19: iter-1 trimmed primer; 80% on 5 completed; commit-discipline lesson
- [[sources/f164796|f164796]] — 2026-04-19: eq-gate-star workstream scaffold
- [[sources/3414507|3414507]] — 2026-04-19: ls-mixture workstream scaffold
- [[sources/b3a27e2|b3a27e2]] — 2026-04-19: rules-as-tools workstream scaffold
- [[sources/1f13f92|1f13f92]] — 2026-04-19: reference-trace-distillation scaffold; Haiku 4.5 introduced
- [[sources/761587c|761587c]] — 2026-04-19: Haiku N=30 reference run; 72.4% bot-match; zero conditional_outcome
- [[sources/80704f0|80704f0]] — 2026-04-19: iter-2 prep infrastructure wiring
- [[sources/eebcae5|eebcae5]] — 2026-04-19: iter-2 prep continuation
- [[sources/b5d05de|b5d05de]] — 2026-04-19: iter-2 prep cluster close; four workstreams staged
- [[sources/c698091|c698091]] — 2026-04-19: enable_primer three-mode flag (off/trim/full)
- [[sources/abb1b3d|abb1b3d]] — 2026-04-19/20: enable_rules_tools threaded
- [[sources/65c749c|65c749c]] — 2026-04-20: async STaR rollout concurrency prep
- [[sources/faefca7|faefca7]] — 2026-04-20: iter-3 prep continuation
- [[sources/c2aa3a7|c2aa3a7]] — 2026-04-20: iter-3 prep cluster close
- [[sources/35c75ff|35c75ff]] — 2026-04-19: selfplay arena scaffold
- [[sources/2830be0|2830be0]] — 2026-04-19: Opus lock for arena
- [[sources/20f4fa2|20f4fa2]] — 2026-04-19: iter-4 preserve_thoughts A/B
- [[sources/39aafaf|39aafaf]] — 2026-04-19: arena --tag Opus vs Haiku run
- [[sources/dbadb5f|dbadb5f]] — 2026-04-19: session docs; $20.50/$40 cumulative cost noted
- [[sources/6fea6ab|6fea6ab]] — 2026-04-19: MLX-LM local inference path for Apple Silicon
- [[sources/edf86e9|edf86e9]] — 2026-04-19: SFTConfig max_seq_length=4096 fix; iter-4 null reframed
- [[sources/1efb9c5|1efb9c5]] — 2026-04-19: iter-5 E1 rank-sweep; preserve_thoughts with truncation fix
- [[sources/ceca203|ceca203]] — 2026-04-19: candlewax concept + reasoning-coherence-verification framing
- [[sources/7321952|7321952]] — 2026-04-19: iter-5 E2 candlewax-null eval
- [[sources/ed3cfc3|ed3cfc3]] — 2026-04-19: MLX batch throughput bench; 43→1334 tok/s
- [[sources/b0952a2|b0952a2]] — 2026-04-19: candlewax E2E spike scaffold
- [[sources/6a97d55|6a97d55]] — 2026-04-19/20: candlewax spike run; Qwen 3.6-35B-A3B via mlx-vlm
- [[sources/aeafe22|aeafe22]] — 2026-04-20: PRACTICALITIES.md split; 8 practicalities captured
- [[sources/0545342|0545342]] — 2026-04-20: session close; candlewax/iter-5 cluster docs
- [[sources/54f7776|54f7776]] — 2026-04-20: chat-template bug fix; Gemma 4 role="tool" silently dropped
- [[sources/d858781|d858781]] — 2026-04-20: wax_museum harness + belief_trajectory tool
- [[sources/1bf1885|1bf1885]] — 2026-04-23: Gus integration; Burl finale OVERVIEW
- [[sources/42a7535|42a7535]] — 2026-04-23: Gus kickoff doc; LEM/Burl/Gus sibling framing
- [[sources/31e10ef|31e10ef]] — 2026-04-23: joint-world tensor + BUILD_PLAN; 5-head LAMIR-ready plan
- [[sources/c04bda3|c04bda3]] — 2026-04-20: Gus v0 scaffolding — belief head + dataset + trainer + eval
- [[sources/8dbf7f3|8dbf7f3]] — 2026-04-20: Gus v1 transformer belief student; architecture works, data-bound
- [[sources/da21f52|da21f52]] — 2026-04-20: full 4-head student; dense Q supervision regularizes encoder
- [[sources/2e4f586|2e4f586]] — 2026-04-20: glob-expand fix enabling chunked corpus loading for 1000g
- [[sources/3c02d10|3c02d10]] — 2026-04-21: Gus v2 explicit void features; +1.4pp belief, π_me flat
- [[sources/5a4c9b9|5a4c9b9]] — 2026-04-21: LAMIR-primitive eval; direct π_me vs PIMC variants
- [[sources/2a09050|2a09050]] — 2026-04-21: regret-based eval introduced; reframes 35% bot-mismatches as near-ties
- [[sources/0472125|0472125]] — 2026-04-21: MORNING2_STATUS — 2000g scaling results; 1.60 Q-pt regret; scaling lessons
- [[sources/a50c9ef|a50c9ef]] — 2026-04-21: decision-hardness analyzer; high-regret = high oracle E[Q] spread; honest mistakes
- [[sources/5cdec8a|5cdec8a]] — 2026-04-21: MORNING3_STATUS — full adapter ladder; data dominates capacity; ceiling named
- [[sources/fdcd654|fdcd654]] — 2026-04-21: 120-epoch ceiling confirmation; all 2000g runs converge on 1.60–1.65 regret
- [[sources/286eb23|286eb23]] — 2026-04-21: 3000g NEW BEST 1.39 Q-pt regret; data scaling not yet plateaued
- [[sources/f0139a3|f0139a3]] — 2026-04-21: PRACTICALITIES 10 receipts; bimodal regret (73% perfect, 6% blunder tail)
- [[sources/a8bc35a|a8bc35a]] — 2026-04-21: GEN_FLEET plan; Vast.ai distributed gen, ~$15/10k games
- [[sources/1a2f67f|1a2f67f]] — 2026-04-21: arena + visualizer; 50% vs 80% contracts made; V/π decoupling surfaced
- [[sources/b007cf3|b007cf3]] — 2026-04-21: v3 consistency regularizer smoke-tested; full run pending
- [[sources/f90682c|f90682c]] — 2026-04-21: oracle-feature blunder detector (AUC 0.926) + ensemble analysis; router > ensemble
- [[sources/5373223|5373223]] — 2026-04-21: student-feature blunder detector (AUC 0.839); pi_peak top feature; deployable
- [[sources/109f9e1|109f9e1]] — 2026-04-21: PRACTICALITIES receipts 11-13; arena gap, ensemble hurts, detect-and-route summary
- [[sources/eba5103|eba5103]] — 2026-04-21: detect-and-route inference wrapper; oracle 0.49 regret; PIMC-Q and next-adapter both hurt
- [[sources/a09ef43|a09ef43]] — 2026-04-21: PRACTICALITIES receipt 14; router PoC reality-check; no-oracle prerequisite named
- [[sources/695f2ef|695f2ef]] — 2026-04-21: explanation sketcher; template NL rationalization over head outputs; V/π disagreement flagged in prose
- [[sources/7a9c720|7a9c720]] — 2026-04-21: shine analysis; 25.4% sharp-and-perfect; zero-inference routing heuristic
- [[sources/137a8e7|137a8e7]] — 2026-04-21: PRACTICALITIES receipt 15; belief calibration doesn't propagate; co-train prerequisite
- [[sources/f138069|f138069]] — 2026-04-21: lazy IterableDataset; RSS bounded at 3.4 GB regardless of corpus size; unblocks 10k+ training
- [[sources/b4040c5|b4040c5]] — 2026-04-21: PRACTICALITIES §§16-17; first sub-1.0 regret announced; oracle utility is p_make (cliff-shaped)
- [[sources/41fdb3c|41fdb3c]] — 2026-04-21: PRACTICALITIES §18; qMAE plateau — structural, not data/capacity; LAMIR-1 not blocked
- [[sources/a14200f|a14200f]] — 2026-04-21: GEN_FLEET pre-launch fix list; bid=30 bias + schema v2 must land before fleet run
- [[sources/31f0ec3|31f0ec3]] — 2026-04-21: v2-10k-big lands (0.818); v3 wins by 33%; consistency loss into LAMIR-1
- [[sources/245918d|245918d]] — 2026-04-21: probe writeup §19; six probes; V tracks oracle ±0.5 Q-pts; counterfactual tool promotable
- [[sources/581bf1f|581bf1f]] — 2026-04-22: LAMIR-1 first rollout; 2.384 regret; adversarial leaf states identified
- [[sources/7d2af99|7d2af99]] — 2026-04-22: direct + v-bootstrap modes; per-trick-pos table; V_head distribution shift isolated
- [[sources/e4e6862|e4e6862]] — 2026-04-22: Bug 5 fix — per-world game_hands for opp simulation
- [[sources/566bc4d|566bc4d]] — 2026-04-22: Fix 2 — sign-flip V_head by leaf-player team parity
- [[sources/8544fbe|8544fbe]] — 2026-04-22: q-bootstrap mode added; world-conditioned Q_head at depth-1
- [[sources/fb03970|fb03970]] — 2026-04-22: lamir1-qleaf mode added; full rollout + Q_head leaf; four-mode harness complete
- [[sources/93859a0|93859a0]] — 2026-04-22: π_opp training skeleton; frozen v3 trunk + PiOppHead; Schema v2 corpus
- [[sources/dcd9365|dcd9365]] — 2026-04-22: Schema v2 dataset loader; oracle_softmax_per_seat + legal_mask + voids exposed
- [[sources/1a1a324|1a1a324]] — 2026-04-22: lamir1-piopp mode wired; trained PiOppHead replaces rotated π_me for opp steps
- [[sources/b4e8ecd|b4e8ecd]] — 2026-04-22: NaN fix — zero illegal log_probs before dot-product (0×−∞ trap)
- [[sources/2c380a6|2c380a6]] — 2026-04-22: Bug 6 fix — zero depleted dominoes from world_assign at Q_head leaf
- [[sources/8106f01|8106f01]] — 2026-04-22: PRACTICALITIES §20 initial; full ladder + Q_head OOD root cause
- [[sources/b42669a|b42669a]] — 2026-04-22: §20 final; 8-mode ladder complete; scalar noise vs T×T matrix; four pivot options
- [[sources/a9fa0c6|a9fa0c6]] — 2026-04-22: Q_head augmentation fine-tuner; path (a) implementation
- [[sources/5f390fb|5f390fb]] — 2026-04-22: path (a) postmortem; aug Q_head 2.216 regret; random ≠ causal depletion
- [[sources/548d32a|548d32a]] — 2026-04-22: §21 belief ceiling; Bayes-optimal 39.184%; belief_ceiling.py diagnostic
- [[sources/cf8ff79|cf8ff79]] — 2026-04-22: co-train falsified + q-bootstrap-belief 0.655 unexpected win; sample_worlds.py finally used
- [[sources/94d8646|94d8646]] — 2026-04-22: §22 past belief future direction; meta-strategy distribution; no code
- [[sources/063fcac|063fcac]] — 2026-04-24: Phase A guards on wax_museum (turn-budget extension on reject + forced-commit fallback); enables 2000-decision harvest

## Trails

- [[trails/lem-to-burl-handoff|lem-to-burl-handoff]] — thematic walkthrough of the LEM→Burl transition: shared infrastructure, diverging philosophies, open questions inherited (active)

## Open questions

See [[questions/open]].
