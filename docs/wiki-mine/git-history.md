# Pre-Wiki Git History — Texas 42 (mk5-main)

Mined read-only from `git log`/`git show`. Scope: the pre-wiki era, everything
before the **wiki epoch = `b1de970` (2026-04-24)**, "docs(wiki): promote replay
wiki from history/ to tracked wiki/". Repo bounds: `f989134` (2025-07-28 "green")
→ `bdebb82` (2026-07-06). 1539 commits total; **1161 of them pre-wiki.**

Key structural fact established up front, because it reframes everything below:
**every single `wiki/sources/` digest is dated 2026-04-09 or later.** The wiki was
born looking forward from the LEM/Burl/Gus era. The entire *founding* of the ML
project — the GPU solver, Crystal Forge, and the whole E[Q] pipeline — predates
the first digest by three to four months and is **undigested**. Git commit
messages are the only diary for that period.

---

## 1. Era map

| Era | Dates | ~Commits | Theme |
|-----|-------|----------|-------|
| **A. The web game** | 2025-07-28 → 2025-11 | ~217 | Building the Texas 42 web implementation itself. TDD cadence — commit messages are literally "green", "draft", "mobiledraft". Pure-functional event-sourced engine, layer system, multiplayer. No ML yet. |
| **B. Rules algebra + solver birth** | 2025-12 | 213 | Christmas 2025 spent on the *factored algebraic* rules model (`SUIT_ALGEBRA.md`, τ-encoding, `CALLED` rename, doubles-as-trump). Then **the GPU tablebase solver lands Dec 27** (`6f82f9f`), and **Crystal Forge — the Lightning-first ML pipeline — Dec 30** (`2559818`). The pivot from "web game" to "ML project." |
| **C. The January explosion (E[Q] + analytics)** | 2026-01 | **508** | The single densest month in repo history. Two parallel blitzes: (1) the **E[Q] imperfect-info pipeline** born and iterated end-to-end (MVP plan → marginalization → posterior-weighting → GPU-native → exact world enumeration, all in 12 days); (2) the **analytics "module" notebooks** (25x/26x) systematically testing 42 folk wisdom against oracle data — several land as *documented negative results*. |
| **D. Zeb evaluation** | 2026-02 | 100 | Turning E[Q] into a *player* and grading it. E[Q] player eval framework, Bradley-Terry Elo + W&B, Vast.ai fleet. Closes with the **full-teacher E[Q] capacity-ceiling finding** (Feb 16, `6081420`). |
| **E. Dormancy** | 2026-03 | **0** | Zero commits. A full-month gap. Worth noting as a hard seam. |
| **F. LEM / Burl / Gus / candlewax** | 2026-04 (to epoch 04-24) | ~120 pre-epoch | Project revives on a new axis: LLM reasoning. **LEM** (narration + first Gemma contact Apr 9; Gemma→Qwen pivot Apr 16). **Burl** born Apr 18. **candlewax** spike Apr 19–20 (bimodal legibility + the LLM-as-reasoner retirement). **Gus** kickoff Apr 20. Wiki epoch Apr 24. This era *is* covered by digests. |

Post-epoch (for orientation only, not in scope): May = Lens; June = the Champion
arc #20–#31 (belief/self-play, mostly nulls); July = jud (value-native bidder).

---

## 2. Annotated chronology — the significant pre-wiki commits

Ordered chronologically. "Digest" = does `wiki/sources/<sha>.md` exist.

| sha | date | essence | why it matters | digest |
|-----|------|---------|----------------|--------|
| `6f82f9f` | 2025-12-27 | **GPU backward-induction solver**, 95 tests. Solves seed=100 in 22s / 10.3M states. Park-Miller LCG matches the TS engine; 47-bit packed state. | The oracle's foundation. Every E[Q] label ultimately traces to this tablebase. Known 1-pt discrepancy vs TS minimax flagged at birth (t42-1a6e). | **NO** |
| `b541a4b` | 2025-12-27 | Python solver for complete regret tables. | The CPU reference the GPU solver validates against. | **NO** |
| `2559818` | 2025-12-30 | **Crystal Forge** — Lightning-first ML pipeline. oracle/ + ml/ + cli/ (tokenize/train/eval), first train/val/test splits. | The birth of the ML project as a *pipeline*, not a script. solver2 archived here. | **NO** |
| `189da59` | 2026-01-02 | Marginalized Q-value training pipeline for imperfect-info play. | First articulation of the imperfect-info problem that E[Q] exists to solve. | **NO** |
| `a81fe27` | 2026-01-05 | PI oracle bidding research synthesis (t42-g8wt). | Early bidding-side thinking — the axis the whole project eventually converges back onto (jud). | **NO** |
| `e6f4152` | 2026-01-07 | **"Coverage vs trump count — SURPRISING NEGATIVE RESULT."** Oracle data: coverage_score *hurts* E[V] (β=−0.29). | The clearest of the folk-wisdom refutations; the analytics era's signature finding. Directly contradicts canonical (Roberson) strategy. | **NO** |
| `851f776` | 2026-01-07 | "25k: Information value — knowing opponent hands." | Quantifies the value of hidden information — the exact quantity jud's referee later re-derives. | **NO** |
| `0ca12d4` | 2026-01-10 | **E[Q] training MVP plan.** Two-stage: perfect-info model as oracle → train imperfect-info policy on E[logits] labels. | The founding design doc of E[Q]. States the whole strategy in five lines. | **NO** |
| `ece6dcf` | 2026-01-10 | E[Q] data generation pipeline + backtracking sampler. | First working data generator. | **NO** |
| `341dd53` | 2026-01-11 | **Proper E[Q] marginalization over hypothetical worlds** — reconstruct hypothetical initial hands (initial = remaining + played_by) rather than using the true deal. | The correctness fix that makes E[Q] actually marginalize over opponent uncertainty instead of cheating with ground truth. | **NO** |
| `becb59f` | 2026-01-17 | Posterior-weighted E[Q] with ESS mitigation. | Moves from uniform to posterior world-weighting — the same importance-weighting idea Champion #25 revisits (and finds null) five months later. | **NO** |
| `6968b63` | 2026-01-19 | GPU-native E[Q] pipeline + self-describing datasets. | Performance/scale rewrite. | **NO** |
| `098a047` | 2026-01-22 | **Exact E[Q] via world enumeration.** Observed plays collapse the world set: decision 0 = 399M worlds → decision 27 = 1 world (exact). | The key epistemic insight of the founding era: late-game E[Q] is *exact*, not sampled. The 237,543× / 199M× reductions are load-bearing intuition. | **NO** |
| `09343fa` | 2026-01-22 | Deprecate CPU pipeline (RuntimeError guards — "known E[Q] collapse bugs") + beautiful visualize.ipynb (belief-cloud uncertainty-collapse GIFs). | Marks GPU as the one true path; the belief-cloud viz is the visual ancestor of the "belief" thread. | **NO** |
| `0d8f336` | 2026-02-05 | E[Q] player evaluation framework (Zeb). | E[Q] stops being data and becomes a *player* to grade. | **NO** |
| `3a77bb6` | 2026-02-09 | Bradley-Terry Elo + W&B logging for eval matrix. | The scoring harness the whole later arena inherits. | **NO** |
| `6081420` | 2026-02-16 | **Close full-teacher E[Q] experiment.** 1059 cycles; finding: E[Q] signal doesn't beat ~74% vs random at 3.3M params — **a capacity ceiling.** | The founding era's terminal negative result — a small policy net can't absorb the oracle. Motivates the pivot to LLMs (LEM). | **NO** |
| — | **2026-03** | *(silence — zero commits)* | The seam between the small-net era and the LLM era. | — |
| `a8bccfa` | 2026-04-09 | LEM: narration generator, rules primer, first Gemma contact. | The LLM-reasoning era begins. First digest exists. | yes |
| `7538016` | 2026-04-10 | LEM STaR harness — inference, K1 grading, R1 rationalization. | STaR loop scaffolding. | yes |
| `3465e29` | 2026-04-16 | **LEM Stage 0 v5 — pivot base Gemma→Qwen 3 1.7B (100% vs 60%).** "the real answer was model choice, not GPU tuning." | Cleanly-reasoned base-model pivot; kills the B200-underutilization bead by re-diagnosing it. | yes |
| `b857299` | 2026-04-17 | LEM Stage 0 v7→v9 — structured reasoning templates + verifier. | Reasoning-quality scaffolding. | yes |
| `8d26e0d` / `d9baf3b` | 2026-04-18 | **Burl** introduced as LEM's sibling + foundational tool harness. | Burl (tool-use reasoner) born. | yes |
| `1efb9c5` `b0952a2` `ceca203` | 2026-04-19 | **candlewax** landing: bimodal-aware `eq_outcome_distribution`, empirical spike-driver dominoes, iter-5 writeups (truncation reframe + candlewax null). | Makes bimodality legible at the tool surface; documents that the *tool* is correct but Gemma's policy won't depth-probe. | yes |
| `0545342` | 2026-04-20 | **candlewax spike closes** — multimodal PDFs + engine fact-checker + MLX LoRA STaR end-to-end. **Pivots away from LLM-as-reasoner** (verifier is a multi-week subproject). | The era's defining retirement (see §3). Also: "image-as-alignment rescues small models"; training collapse past loss ~0.5. | yes |
| `42a7535` | 2026-04-20 | **Gus** kickoff — neural policy+value+belief, skips reasoning channel, E[Q] as variance-free reward. | Gus born; the belief-brain lineage begins here. | yes |

---

## 3. Pivot ledger (verbatim reasons)

The "we already tried that" list. Reasons quoted from commit bodies.

1. **CPU E[Q] pipeline → GPU-only** (`09343fa`, 2026-01-22)
   > "Add RuntimeError guards to all cpu_deprecated modules to prevent accidental imports (CPU pipeline has known E[Q] collapse bugs)."

2. **Random world sampling → exact enumeration** (`098a047`, 2026-01-22)
   > "Replace random world sampling with exact enumeration of consistent worlds… observed opponent plays massively constrain possible worlds: Decision 0: 399M worlds; Decision 25: 2 worlds; Decision 27: 1 world (EXACT)."

3. **Full-teacher E[Q] small-net policy — CAPACITY CEILING** (`6081420`, 2026-02-16)
   > "E[Q] policy signal doesn't improve play beyond ~74% vs random at 3.3M params — appears to be a capacity ceiling for this approach." *(This is the negative result that motivates leaving small nets for LLMs.)*

4. **Coverage-score folk wisdom — REFUTED** (`e6f4152`, 2026-01-07)
   > "Folk wisdom claims '2 trumps + perfect coverage beats 4 trumps + naked lows' but oracle data shows coverage_score actually HURTS expected value." (Companions: `67a7d23` "Coverage protects — folk wisdom NOT confirmed"; `8b18d36` "Naked lows hurt — NOT confirmed"; `d562250` "Voiding is active — NOT confirmed"; `25g` "Partner synergy — NO significant interaction.")

5. **Base model Gemma → Qwen 3 1.7B** (`3465e29`, 2026-04-16)
   > "Gemma 4 E2B was both slower (architectural: PLE, KV-sharing, no FA2) and less accurate (60%) than Qwen 3 1.7B (100%)… the real answer was model choice, not GPU/kernel tuning."

6. **LLM-as-reasoner — RETIRED (candlewax)** (`0545342`, 2026-04-20) — *the best-documented pivot in git; see final note.*
   > "Pivots away from LLM-as-reasoner — reasoning-coherence verification is the bottleneck and requires a multi-week verifier subproject, not a weekend… STaR iter 2 plateaus without reasoning verifier; confirms the limit."
   > Training-collapse receipt: "any fit past loss ~0.5 on this corpus shape destroys output generation (repetition loops). LR=1e-5 always collapses."

7. **candlewax return-shape — validated tool, NULL live signal** (`ceca203`/`1efb9c5`, 2026-04-19)
   > "The environment-shape lever is validated at its firing site. The blocker is Gemma's policy preferring 'try another play' over 'probe this play's uncertainty.'" (E3: 98 eq calls, 74 non-unimodal, **0** conditional_outcome calls.)

Post-epoch pivots (context, not in scope but they close threads that *started* pre-wiki):
- **Belief-weighted play — DEAD LEVER** (`f919f2c`, 2026-06-14): "Both play-side levers now measured dead… it holds for the auction, not play." (This retroactively closes the posterior-weighting idea from `becb59f`, Jan 17.)
- **Champion #31 magnitude channel — measured dead**; **#26 self-play converges to a calibratable over-bidder** (double-dummy/strategy-fusion optimism, ~0.25 high).

---

## 4. Gap list — significant pre-wiki commits with NO wiki/sources digest

Ranked by insight value. **These are the founding-era memories a fresh model
cannot recover from the wiki.** The cut is stark: the digest set begins at
2026-04-09; *nothing before it is digested.*

1. **`098a047` (2026-01-22) — exact E[Q] via world enumeration.** Highest value. The single sharpest epistemic insight of the founding era (observed play collapses 399M worlds → 1). Nothing in the wiki captures that late-game E[Q] is exact.
2. **`6081420` (2026-02-16) — full-teacher E[Q] capacity-ceiling null.** The negative result that *ended* the small-net era and motivated LLMs. A fresh model would re-attempt small-net distillation without it.
3. **`0ca12d4` (2026-01-10) — E[Q] two-stage MVP plan.** The founding design doc; the whole oracle-distillation strategy in five lines.
4. **`341dd53` (2026-01-11) — proper marginalization (hypothetical hands).** The correctness principle (don't peek at the true deal) that everything downstream depends on; easy to silently get wrong again.
5. **`6f82f9f` / `2559818` (2025-12-27/30) — GPU solver + Crystal Forge birth.** The literal foundation. The wiki treats the oracle as a given; its origin, the Park-Miller/TS parity, and the 1-pt discrepancy are undocumented.
6. **`e6f4152` + the 25x/26x notebook blitz (2026-01-07) — folk-wisdom refutations.** Coverage hurts E[V]; naked lows don't hurt; voiding active; no partner synergy. Directly contradicts the canonical (family/Roberson) strategy the project treats as ground truth elsewhere. High collision risk with human intuition.
7. **`becb59f` (2026-01-17) — posterior-weighted E[Q] w/ ESS.** The importance-weighting idea that Champion #25 later re-derives and finds null; the wiki has the null but not the founding attempt.
8. **`0d8f336` / `3a77bb6` (2026-02) — Zeb eval framework + Bradley-Terry Elo.** Origin of the scoring harness the arena still uses.
9. **`189da59` (2026-01-02) / `a81fe27` (2026-01-05) — imperfect-info pipeline + PI bidding synthesis.** Earliest framing of both the imperfect-info problem and the bidding axis the project eventually returns to.

Suggested digest-writing priority: 1 → 2 → 3 → 4 → 5, then the folk-wisdom cluster (6) as a single synthesis page since the notebooks share one theme.

---

### Final summary

- **Wiki epoch is `b1de970` (2026-04-24); every source digest postdates 2026-04-09**, so the founding ML era is entirely undigested — git messages are its only record.
- **Six eras:** web game (Jul–Nov 2025) → solver/Forge birth (Dec 2025) → the 508-commit **January E[Q] + analytics explosion** → Zeb eval (Feb) → a **dead-silent March** → the LEM/Burl/Gus/candlewax revival (April, where the wiki picks up).
- **The founding arc is a clean three-beat story:** build the exact oracle (Dec) → distill it into imperfect-info E[Q], discovering late-game E[Q] is *exact* not sampled (Jan) → hit a small-net capacity ceiling (Feb), which forces the pivot to LLMs (April).
- **The richest undigested seam** is the January E[Q] pipeline (`098a047`, `341dd53`, `0ca12d4`) plus the February capacity-ceiling null (`6081420`) — the "why we left small nets" that the wiki never states.
- **The pivot ledger's throughline:** the project repeatedly finds that *play* is near-oracle and the leverage is in *bidding/hidden-information* — a lesson first hinted at in January's information-value notebook and only named explicitly in the July jud work.

**Pivot best-documented in git but least-documented in the wiki:** the **capacity-ceiling retirement of the full-teacher E[Q] small-net policy** (`6081420`, 2026-02-16). Its git message states the experiment scale (1059 cycles across 25%/95% E[Q] mix regimes), the metric (~74% vs random), the parameter budget (3.3M), and the verbatim conclusion ("appears to be a capacity ceiling for this approach") — a complete, decisive negative result. Yet it has no digest and sits in the dead-silent Feb→March seam, so nothing in the wiki explains *why* the project abandoned small-net distillation and pivoted to LLM reasoners in April. A fresh model reading only the wiki would see LEM/Burl/Gus appear with no stated cause. (The candlewax retirement `0545342` is comparably well-documented in git *and* has a digest, so it is less of a gap even though its message is richer.)
