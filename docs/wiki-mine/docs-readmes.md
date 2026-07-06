# Pre-Wiki Knowledge Mining: docs/, READMEs, module lore, retired beads

**Scope:** everything in `/Users/jason/code/mk5-main` EXCEPT `forge/` (sibling agent) and `wiki/` itself (sibling auditor). Read-only sweep.
**Date of sweep:** 2026-07-06.
**Historian's headline:** The wiki is an ML-era document (LEM/Burl/Gus/forge/champion, ~April–June 2026). The project's **founding half (Dec 2025 – Feb 2026)** — the DP solver, the suit algebra, the τ-encoding saga, the PI-oracle/strategy-fusion methodology, and the argmax accuracy ceiling — lives almost entirely in `docs/`, and the wiki's source digests cite `docs/` **only twice** (`docs/rules.md`, `docs/arena-perf`). A fresh model reading only the wiki inherits the *conclusions* of that era (E[Q], pimc, rank-vs-price) without the *derivations, ceilings, and dead-ends* that make them load-bearing.

**Scope note (2026-07-06):** The `.beads/issues.jsonl` mining was reassigned to a dedicated sibling agent (**mine-beads**) with a deeper pass. This report is now the **docs/README** sweep only. Where a finding's primary evidence lives in a retired bead, I re-grounded it on the `docs/` source and flag the bead detail as **[→ mine-beads owns]**.

---

## 1. Inventory Table

Wiki status legend: **ABSENT** = no wiki page covers it; **PARTIAL** = a wiki page touches the topic but omits the founding content/derivation; **DIGESTED** = a `wiki/sources/*` digest or dedicated page absorbs it.

### docs/ — founding architecture + ML

| Path | Date | Subject | Wiki status |
|------|------|---------|-------------|
| docs/VISION.md | Nov 2025 | Engine north star: determinism, capability multiplayer, AI-as-peers, key bets | PARTIAL (entities/engine.md is a stub; vision/invariants ABSENT) |
| docs/CONCEPTS.md | Nov 2025 (mk8) | Full engine implementation reference: Layers, GameRules 18-method interface, capability filtering, event sourcing | PARTIAL (engine.md names the pieces; the 18-method contract + invariants ABSENT) |
| docs/ARCHITECTURE_PRINCIPLES.md | 2026-04-19 | Design philosophy / mental models | ABSENT |
| docs/ORIENTATION.md | 2026-04-19 | Developer onboarding/navigation | ABSENT (wiki has its own entrypoints) |
| docs/theory/SUIT_ALGEBRA.md | (theory) | **The founding math**: 8-suit covering, called/power sets κ/π, three-tier τ trick-rank, unique-winner proof, S₇ symmetry, GPU table layout (<1KB) | **ABSENT** |
| docs/theory/SUIT_ALGEBRA_PURE.md | (theory) | Pure/condensed variant of the above | ABSENT |
| docs/theory/PLAY_PHASE_ALGEBRA.md | (theory) | Play-phase formalism for fixed (deal, declaration); score-free packed state = per-trick transition rewards (matches solver2) | ABSENT |
| docs/theory/PLAY_PHASE_SPEC.md | (theory) | Play-phase spec companion | ABSENT |
| docs/random-tiebreaker-ceiling.md | 2026-04-19 | **73.96% argmax accuracy ceiling**; 44.3% of states have tied-optimal actions; the 1-in-10M zero-agency state | **ABSENT** |
| docs/oracle-state-space-analysis.md | 2026-04-19 | 180GB / 1000 seeds; per-declaration branching (doubles-trump biggest at 33.8M avg); seed 434 OOM at ~190M | PARTIAL (forge-analysis.md exists; these numbers ABSENT) |
| docs/solver2-data.md | 2026-04-19 | 8.4B states / 48.8GB; **41-bit packed state encoding**; local-index-per-player seed-specific mapping; V/Q semantics | PARTIAL (entities/engine.md references tables.py only) |
| docs/claudeai-mlp.md | 2025-12-28 | **The original pipeline vision**: DP → Value MLP → Fast PIMC → Transformer; τ-encoding cross-seed fix | ABSENT (its descendant rank-vs-price exists; lineage untraced) |
| docs/EQ_STAGE2_TRAINING.md | 2026-04-19 | Research-grade Stage 2 spec: posterior-weighted info-set E[Q], world sampling, behavior model | PARTIAL (expected-q-value topic is thin; the spec/knobs ABSENT) |
| docs/INTERMEDIATE_AI.md | 2026-04-19 | TS-side Monte Carlo + constraint tracker; the 4-0-is-trump void-inference bug; "never re-derive rules" | ABSENT |
| docs/analysis-draft.md | 2026-04-19 | "Structural Analysis Survey" — 200GB oracle data, goal is *understanding not prediction*; symmetry/topology probes | ABSENT |
| docs/zeb-worker-saga.md | 2026-02-15 | Operational story of `lb-v-eq-3740` warm-start run; eval-aux fragility | PARTIAL (zeb parked per decisions/zeb-parked-eq-primitive) |
| docs/CONCEPTS/CAPABILITY_SYSTEM/MULTIPLAYER/TESTING_PATTERNS/CLIENT_* | Nov 2025 | Engine subsystem references | ABSENT (engine-side, out of ML wiki scope) |
| docs/rules.md, docs/rules-gherkin.md, docs/rules-tournament.md | — | Official Texas 42 rules | DIGESTED (cited by 2 source digests; entities/texas-42.md) |
| docs/adrs/ADR-20250111-system-authority.md | Jan 2025* | Scripted-action system authority | ABSENT |
| docs/adrs/ADR-2025111x (×5) | Nov 2025 | Single composition point, protocol leaks, connection.reply, URL replay, onehand terminal | PARTIAL (decisions/ has ML decisions, not these engine ADRs) |
| docs/adrs/ADR-20251124-layer-unification.md | Nov 2025 | Unified Layer arch (rules + action-gen in one surface) | PARTIAL (CONCEPTS describes result; ADR rationale ABSENT) |
| docs/EQ_STAGE2 refs to docs/EQ_MVP.md | — | **Dangling**: EQ_MVP.md no longer exists (redirect target gone) | n/a |
| docs/research/answer.md | 2026-01-05 | **PI-oracle definitive synthesis**: evaluator-not-teacher, fix-trump-before-aggregation, label-every-completion-model | ABSENT (concept in rank-vs-price; operational rules ABSENT) |
| docs/research/{question,opus45*,chatgpt52*,gemini3*}.md | Jan 2026 | The multi-AI research inputs that produced answer.md | ABSENT |

\* ADR-20250111 header says 2025-01-11 but sits among Nov-2025 ADRs; the "2025-01-12"/"2025-01-24" dates inside two others are near-certainly typos for 2025-11.

### Module READMEs / OVERVIEWs

| Path | Date | Subject | Wiki status |
|------|------|---------|-------------|
| lem/OVERVIEW.md | 2026-04-19 | LEM pipeline overview | **DIGESTED** (cited 82× across wiki sources — the single most-digested doc) |
| burl/OVERVIEW.md | 2026-04-20 | Burl agent overview | DIGESTED (32× citations) |
| gus/OVERVIEW.md + MORNING*/BUILD_PLAN/PRACTICALITIES | Apr 2026 | Gus build story | DIGESTED (gus entity + experiments) |
| lem/rules/primer.md, lem/narrate/OVERVIEW.md | Apr 2026 | Rules primer, narration | DIGESTED (13× / 9×) |
| champion/README.md | 2026-06-13 | Champion ladder | DIGESTED (entities/champion.md) |
| champion/evidence/champion-one-organ-theory-2026-06-14.md | 2026-06-14 | "One organ" synthesis over Fable's framework | PARTIAL/DIGESTED (topics/champion-design-review.md has Fable verbatim; this synthesis partly) |
| arena/README.md | 2026-06-12 | Arena harness | DIGESTED (entities/arena.md) |
| w42/README.md + w42/*/README | May 2026 | Winning-42 validation program | DIGESTED (huge w42-* experiment coverage) |
| burl/experiments/*.md (~20 writeups) | Apr–May 2026 | Iter writeups, benches | DIGESTED (burl-* experiments + perf playbooks) |
| SPIKE_REPORT.md (top-level) | 2026-04-19 | Burl move3/4 spike log | DIGESTED (experiments/burl-move4-native-spike, decisions/native-tool-use-format) |
| MORNING_DIGEST.md (top-level) | 2026-04-27 | Burl perf sprint digest | DIGESTED (playbooks/perf-sprint-*, burl-perf-phase*) |
| AGENTS.md, CLAUDE.md | Apr–May 2026 | Operating instructions | n/a (instructions, not knowledge) |

### scratch/ design notes (curated skim — not exhaustive)

| Path | Subject | Wiki status |
|------|---------|-------------|
| scratch/winning42/strategy_measurement_breakdown.md, winning42.full.md | Roberson's *Winning 42* extraction + strategy-measurement scaffolding | DIGESTED (winning42-ch01..16 experiments) |
| scratch/champion-run/*handoff*, MISSION.md, epic29_body.md | Champion epic handoffs | DIGESTED (champion entity/experiments) |
| scratch/jud-v0/*, jud-v1/* | Jud build reports | DIGESTED (experiments/w42-jud-v0, w42-jud-v1) |
| scratch/candlewax_spike/mlx_lora_research.md | MLX LoRA research for candlewax spike | PARTIAL (candlewax-spike entity) |
| scratch/BLUNDER_FORENSICS.md, SHINE_ANALYSIS.md | Gus blunder/shine analyses | DIGESTED (topics/blunder-detector, shine-analysis) |
| scratch/lamir_paper_notes.md | LAMIR paper notes feeding lamir1 topic | PARTIAL (topics/lamir1) |
| scratch/belief_trajectory_rollout/* | Belief-trajectory harvest ops | DIGESTED (belief-trajectory entity) |

---

## 2. Findings Ledger (insights not, or not fully, in the wiki)

### F1 — The 73.96% argmax accuracy ceiling (ABSENT)
**Claim:** Any argmax-over-Q policy is capped at ~74% action-match accuracy, because 44.3% of oracle states have tied-optimal actions. This is a hard ceiling every policy-accuracy number must be read against.
> "**Ceiling** | **0.7396** (73.96%) … Only 55.7% of states have a unique optimal action. The remaining 44.3% have ties, reducing the theoretical maximum accuracy achievable by any argmax-based policy." — `docs/random-tiebreaker-ceiling.md` (2026-04-19)

Tie distribution: 1-way 55.69%, 2-way 22.73%, 3-way 18.21%, 4+ 3.4%.
**Wiki status:** ABSENT. The wiki's many "ceiling" mentions (`base-model-k1-baseline`, `log.md`) are about *K1/trick-6 grading* ceilings, a different and unquantified notion. The 73.96% number and its cause appear nowhere in `wiki/`.

### F2 — The zero-agency "spectator to their own defeat" state (ABSENT)
**Claim:** Exactly one state in 10M has all 7 actions tied — the player has literally zero agency; the game is fully determined before they act. A vivid teaching artifact for "the game is often decided earlier than the decision point."
> "P2, dealt no trumps at all in a sixes game, is a spectator to their own defeat." — `docs/random-tiebreaker-ceiling.md`
**Wiki status:** ABSENT (`grep` for "spectator"/"zero agency" in wiki: no hits).

### F3 — Raw domino IDs don't generalize; τ (power-rank) encoding is seed-invariant (lineage ABSENT)
**Claim:** The founding encoding lesson. Raw domino IDs memorize per-seed; power-rank (τ) transfers. This is the seed of what the wiki now calls **rank-vs-price**, but the wiki never traces it to its origin.
> "'Player 0 has domino 14' … Meaning changes per seed. 'Player 0 has 3rd-highest trump' … Same meaning … Seed-invariant! … The model learned 'when player 0 has domino 14 and player 1 has domino 22, value is +8' — but that's **seed-specific memorization**, not game understanding." — `docs/claudeai-mlp.md` §7.5 (2025-12-28)
> "**Key Insight:** 'Has boss trump' transfers across seeds. 'Has domino 14' doesn't." — ibid.
The doc names the exact fix and function: τ-encoding via `trick_rank(domino_id, led_suit, decl_id)` from `tables.py`.
**Wiki status:** ABSENT as lineage. `topics/rank-vs-price.md` is the mature descendant; `entities/engine.md`/`forge.md` mention `trick_rank` only as a narration helper. **[→ mine-beads owns the τ-encoding ticket arc (t42-wzsq → t42-74vy).]**

### F4 — The MLP→transformer pivot: why a value MLP was abandoned (PARTIAL)
**Claim:** The original plan (F10) put a **Value MLP** between the DP solver and PIMC. `docs/claudeai-mlp.md` already documents the fatal crack: the MLP fails to generalize across seeds (val MSE 0.022 same-seeds vs 0.040 held-out — "The model learns seed-specific patterns, not generalizable game understanding," §7.5), and even after the τ-encoding fix, move-ordering (not MSE) is what actually matters for PIMC (§7.4: "MSE might not capture move ordering"). The doc is the last MLP-era artifact; the pipeline that shipped is transformer-based.
**Wiki status:** ABSENT — the wiki takes the transformer as given and never records that a cheaper flat-MLP value head was tried and abandoned (a re-proposal risk). **[→ mine-beads owns the punchline "MLP can't represent relational structure; attention can" (t42-1d1g) — the explicit architectural verdict lives in that ticket, not in docs/.]**

### F5 — The PI-oracle / strategy-fusion founding methodology (concept PARTIAL, operational rules ABSENT)
**Claim:** The methodology that still governs the E[Q] pipeline was settled Jan 2026 in `docs/research/answer.md`, a synthesis of 6 AI research passes (opus45cc, chatgpt52max, gemini3cli + 2 meta + a lit review). Its load-bearing rules:
> "1. **Use it as an evaluator, not a policy teacher.** … never imitate oracle actions. 2. **Fix trump before aggregation.** Compute `max_trump(E[V])` not `E[max_trump(V)]`. This eliminates Strategy Fusion at the trump-selection layer." — `docs/research/answer.md` §Executive Summary
The doc formalizes the trap as **Max(Average) vs Average(Max)**, grounds it in Frank & Basin (1998) and Long et al. (2010), and estimates Texas 42 is PIMC-friendly (leaf correlation ~0.8–1.0). Its blunt warning is the part the wiki most lacks:
> "PIMC output is an **optimistic upper bound**, not the true achievable probability. … Without these safeguards, it's self-deception."
And the labeling discipline: *"Never emit unlabeled `P(make)`. Always emit `P_M(make)` with method tag."*
**Wiki status:** PARTIAL. Strategy fusion is well-covered as a *late* realization (`rank-vs-price`, `pimc`, champion fixed-point), but the founding operational rules — especially **"fix trump before aggregation / max E[V] not E[max V]"** — return **zero** wiki hits. EQ_STAGE2's line "This is not strategy fusion" only parses against answer.md's rules.

### F6 — The suit algebra: the crystal-palace math foundation (ABSENT)
**Claim:** The entire rule system is one algebra — an 8-suit covering with called set κ(δ) and power set π(δ), a three-tier trick-rank τ, a proven-unique winner, and an S₇ symmetry that makes all pip-trump declarations *isomorphic as legality structures but not as games*. The GPU solver is a direct transcription: the whole rule system fits in <1KB / ~2.2KB of tables, branchless.
> "Pip-trump declarations are isomorphic as legality structures, not as games." — `docs/theory/SUIT_ALGEBRA.md` §9
> "The algebra doesn't just clarify the rules—it makes them *fast*." — §Appendix
**Wiki status:** ABSENT (`grep` for S_7 / effective suit / called set / 8-suit in wiki: no hits). This is the founding "beautiful + correct" artifact and it is invisible to the wiki.

### F7 — "Counts are the boulders": the structural-survey research posture (ABSENT)
**Claim:** `docs/analysis-draft.md` is the founding *research posture* over the oracle data — a 12-part survey (entropy decomposition, symmetry orbits, persistent homology, DFA/Hurst, spectral/diffusion maps, count-basin analysis) whose stated goal is understanding, not prediction, and whose central hypothesis is that the **count dominoes determine V and everything else is residual tactical structure**.
> "The goal isn't prediction. It's understanding. If structure exists, we find it. If not, we know." — `docs/analysis-draft.md` §Overview
> "If within-basin variance is tiny, counts determine V. If … large, suit/trump structure matters. **The residual variance IS the tactical structure, isolated.**" — §6.2 Basin Analysis
It also names the game's exact symmetry group (Team reflection Z₂ negating V; seat rotation; non-trump suit permutation S₆) and the "decided at declaration" question (does V collapse to a low-dimensional function of count-capture?).
**Wiki status:** ABSENT. `forge-analysis/` exists but the *survey program* and the counts-as-sufficient-statistic hypothesis are not in the wiki. **[→ mine-beads owns the tickets that ran individual probes, e.g. t42-xp0p manifold / t42-l79w partner-MI.]**

### F8 — Never re-derive game rules (the constraint-tracker bug) (ABSENT)
**Claim:** A load-bearing engineering lesson from the TS Monte Carlo AI: a trump domino cannot follow its off-suit, so re-deriving void inference from `dominoContainsSuit` produces contradictory constraints and sampling fails. The fix mirrors the engine's exact `getValidPlaysBase`.
> "Never re-derive game rules — The constraint tracker's bug came from not using the exact same follow-suit logic as the game engine." + "Trump changes everything — 4-0 is a trump when 4s are trump, not a 0." — `docs/INTERMEDIATE_AI.md`
Also states the sampling invariant reused everywhere since: "A valid distribution MUST always exist (the real game state is one)."
**Wiki status:** ABSENT (engine-side; the world-sampling invariant recurs in EQ_STAGE2 §4.2 but the lesson's origin/rationale is not in wiki).

### F9 — The 41-bit packed state + local-index-per-player mapping (PARTIAL)
**Claim:** The solver's state is 41 bits with **local** (0–6) hand indices whose mapping to global domino IDs is *seed-specific* via `deal_from_seed()`. This local/global split is the direct cause of F3 (why raw IDs fail) and a persistent footgun in every downstream tokenizer.
> "The bitmasks use *local* indices (0-6) into each player's sorted hand, not global domino IDs. The mapping from local index to global domino ID is seed-specific." — `docs/solver2-data.md`
**Wiki status:** PARTIAL. EQ_STAGE2 (forge, sibling scope) leans on it heavily; no wiki page states the encoding.

### F10 — The original 4-stage pipeline vision + "compression" framing (ABSENT)
**Claim:** The whole thing was conceived (Dec 2025) as **DP Solver → Value MLP → Fast PIMC → Transformer**, where the MLP "compresses 65GB of perfect answers into ~2MB." The confidence-ladder methodology (one-seed → cross-seed → spot-check → PIMC-integration) is the founding validation ritual.
> "The MLP compresses your 65GB of solved positions into ~2MB of learned function. That's the bridge from 'I have the answers' to 'I can use them fast enough.'" — `docs/claudeai-mlp.md`
**Wiki status:** ABSENT. The MLP leg was abandoned (F4), so the wiki starts mid-pipeline; the original plan and its intent are lost.

---

## 3. Retired beads — handed to mine-beads

The founding ML arc (Dec 2025 – Jan 2026) predates every wiki entity, and its authoritative record is `.beads/issues.jsonl`. **That mining is owned by the sibling agent `mine-beads` (deeper pass)** — this report does not duplicate it. The findings above tag relevant tickets with **[→ mine-beads owns]** so the synthesizer can join the two reports. Two doc-existence facts I surfaced that mine-beads should corroborate against ticket text:

- `docs/SOLVER_REFINED.md` (the DP-solver spec the origin ticket built from) and `docs/EQ_MVP.md` (the redirect target EQ_STAGE2 still names) **no longer exist** — dangling founding references.
- `scripts/solver2/` — the path across every theory doc and `docs/solver2-data.md` — was migrated to `forge/`. That rename is why old docs point at vanished paths.

**Careful-historian note on dates:** many docs carry a 2026-04-19 mtime (a bulk checkout/copy), so I dated by *content* where possible (e.g., claudeai-mlp "Created: 2025-12-28"; zeb-saga "Date: 2026-02-15"; answer.md sources dated Jan 2026). Treat a 2026-04-19 mtime as "no later than," not authored-on.

---

## 4. Top 10 Things the Wiki Doesn't Know (ranked by how much a fresh model would be misled)

All ten are grounded in `docs/`; the beads corroboration lives in the mine-beads report.

1. **The 73.96% argmax accuracy ceiling (F1).** A fresh model reads policy-match numbers like 74–88% as "mediocre → push to 100%," not knowing ~74% is the *theoretical max* for argmax because 44% of states are tied. Mis-frames every Burl/Gus/jud accuracy result. Highest-impact gap.
2. **The strategy-fusion operational rules — evaluator-not-teacher, fix-trump-before-aggregation, label-the-completion-model (F5).** The wiki has the *concept*, not the *rules*. A fresh model could compute `E[max_trump V]` (let the oracle pick best-trump-per-world) and silently reintroduce the exact bias `answer.md` eliminated — the difference between "actionable decision support" and, in the doc's own word, "self-deception."
3. **τ / rank-vs-price *lineage* (F3).** rank-vs-price reads as a champion-era insight; it is the Dec-2025 founding encoding lesson ("has boss trump transfers; has domino 14 doesn't"). A fresh model treats bedrock as a recent hypothesis.
4. **The Value-MLP was tried and abandoned (F4 + F10).** `claudeai-mlp.md` shows the flat MLP failing to generalize across seeds (val 0.022 → test 0.040) and that move-ordering, not MSE, is what PIMC needs. A fresh model, not knowing this, may re-propose a cheap flat-MLP value head and burn a cycle rediscovering the gap.
5. **The suit algebra (F6).** The crystal-palace math the whole solver transcribes — κ/π, three-tier τ, unique-winner proof, S₇ symmetry, <1KB branchless tables. A fresh model touching rule/table code has no map of the elegance it must preserve, and can't exploit the symmetry (all pip-trumps share one legality structure).
6. **The 41-bit local-index-per-player state encoding (F9).** Every tokenizer footgun ("why do raw domino IDs leak seed identity?") traces here; it's also *why* F3 is true. Absent this, encoding bugs look novel each time.
7. **The zero-agency "spectator to their own defeat" state (F2).** The project's sharpest teaching artifact — a proven 1-in-10M state with no agency at all. Pure loss to the wiki; a fresh model has no vivid handle on "the game is often decided before the decision."
8. **"Counts are the boulders" — the structural-survey posture (F7).** `analysis-draft.md`'s hypothesis that count dominoes determine V and the rest is residual tactical structure, plus the stance "the goal isn't prediction, it's understanding." A fresh model inherits the prediction machinery without the curiosity that built it.
9. **"Never re-derive game rules" + the world-sampling invariant (F8).** A concrete, expensive bug (trump-follows-off-suit → contradictory voids → sampling fails) and the invariant EQ_STAGE2 still relies on ("a valid distribution always exists — the real deal is one").
10. **The original 4-stage pipeline + "compression" framing (F10).** Explains *why forge exists* — compress 65GB of ground truth into fast evaluation — and why the MLP stage is missing from the shipped stack. Without it the pipeline looks like it started at the transformer.

---

## Most valuable single document

**`docs/random-tiebreaker-ceiling.md`** — it is short, quantitative, self-contained, and it silently governs the interpretation of every policy-accuracy number in the entire wiki (the 73.96% ceiling), while also carrying the project's best single teaching artifact (the 1-in-10M zero-agency "spectator to their own defeat" state). If only one pre-wiki doc were promoted into the wiki, this is the one: it changes how a fresh model reads the numbers it *does* have.

Runner-up: **`docs/research/answer.md`** (the strategy-fusion methodology bible) for the same "prevents a specific silent mistake" reason; and **`docs/theory/SUIT_ALGEBRA.md`** as the irreplaceable founding-math artifact.
