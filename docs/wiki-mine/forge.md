# Forge: Pre-Wiki Lore (Founding Era, Dec 2025 → Apr 2026)

Historian mining of `forge/` — the oldest, most load-bearing subsystem. Focus:
the exact solver, the oracle→net distillation, E[Q]'s deliberate anti-strategy-fusion
design, and the first quantified contact with the "candlewax" wall. Wiki status of each
claim checked by grep against `wiki/` before labeling.

Citations are `path:line` or `<sha>`. Dates from `git log --diff-filter=A` (file-add) and
commit dates.

---

## 1. Timeline (dated, cited)

| Date | Event | Citation |
|---|---|---|
| **pre-2025-12-30** | A GPU tablebase solver named **`solver2`** already existed under `scripts/solver2/` (context.py, expand.py, campaign.py, declarations.py). It is the *direct ancestor* of `forge/oracle/`. | `git log` f1f634b, 3069158; moved by 2559818 |
| **2025-12-30** | **Crystal Forge founded.** Commit 2559818 "Crystal Forge: Lightning-first ML pipeline" promotes solver2 → `forge/oracle/` (GPU tablebase, backward induction) and adds `forge/ml/` (LightningModule + DataModule) to *distill the solver into a net*. `solve.py` and `schema.py` are born here. Message: "forge/oracle/: GPU tablebase solver (from solver2/)". | 2559818 (`--date=short`) |
| 2025-12-30 | `q0-q6` naming settled (was `mv0-mv6`); V-semantics consolidated ("value-to-go", not score-so-far). | 77f3823, later 68450a9 "consolidate oracle V semantics" |
| **2025-12-31** | Distillation training begins in earnest: **"Normalize value targets to [-1,1] range for balanced loss"** — the Q-value MSE regression setup. | 585ee66 |
| 2025-12-31 | Cloud training scaffolding (Lambda Labs A100), numpy<2 pin. | 628c75e, 0a8dbb5 |
| **2026-01-06** | **First quantified candlewax contact.** Analysis reports `00_executive_summary.md` and `11_imperfect_info.md` land. Report 11 measures marginalized oracle data: "The same P0 hand can swing from -42 to +40 depending on who holds what. Only 11% of hands are 'stable'." Variance decomposition: **53% of oracle outcome variance is within-hand** (opponent cards), 47% between-hand. | report/11 `git log` b4be946; report/00 7591f46 |
| **2026-01-10** | **E[Q] pipeline is born.** `forge/eq/oracle.py` added: "E[Q] data generation pipeline with backtracking sampler" (t42-zqbo). Stage-2 imperfect-info marginalization begins. | ece6dcf |
| **2026-01-18** | **The strategy-fusion spec is written.** `docs/EQ_STAGE2_TRAINING.md` (schema v2.1) lands with §3 "What We Are Learning: Information-Set Q, Not Perfect-Info Q" — the explicit "this is *not* strategy fusion" care. Generator modularized (reduction.py, outcomes.py). | cc59c61, d88e966 |
| **2026-01-22 → 24** | **The candlewax visual artifact.** E[Q] 3D surface, per-game journey, and the **PDF "disc" histograms** (`eq_pdf_discs.html`) are built. 01-24: "store full E[Q] PDF (85 bins per action)". These *are* the "melted candlewax discs" — named that only months later. | f635cfa, 0d4f63e, ef199b0 |
| Jan–Feb 2026 | E[Q] pipeline hardened: posterior-weighted E[Q] with ESS mitigation (becb59f), adaptive convergence sampling ("drunken master"), GPU-native rewrite claiming **12,325x** speedup (0db7750), exact E[Q] via full world enumeration (098a047). | see §3 |
| Feb–Mar 2026 | **Zeb**: AlphaZero self-play (557K–3.3M net), distributed Vast.ai fleet, HF Hub as sole broker. "1.5M self-play games... 70% win rate in 5 days." Marginalized-Q generation mode added to counter fragile single-deal strategies. | forge/README.md:120-135; a7d6b5b, 62e3b53 |
| **2026-04-19/20** | **The name "candlewax" is coined** — but for a *Burl tool-surface concept*, in the reasoning-coherence-verification spike, not credited to the founding analysis. `candlewax.md` wiki page `first_seen: 1efb9c5`. | wiki/topics/candlewax.md; wiki/index.md:370 |
| months later | Wiki (`wiki/`) created. The founding-era understanding above is **mostly not in it** (see §5). | — |

Key structural fact a fresh reader needs: **`forge/oracle/` is `solver2` promoted**, and the
whole `forge/ml/` Lightning stack exists for one purpose — imitation-distill the exact
solver's Q-values into a `DominoTransformer` (6 layers, 8 heads, 256-dim, 3.3M params;
`forge/models/README.md`). The net is not learned from self-play at Stage 1; it is a
**supervised copy of a solved tablebase**.

---

## 2. Findings ledger

Each: {claim · evidence · date · wiki status}.

**F1 — The oracle is an exact backward-induction tablebase, not a heuristic or an MCTS.**
"Enumerate all reachable game states (~50k) ... Solve backwards ... V = max(Q) for Team 0's
turn, min(Q) for Team 1's turn" (`forge/ORIENTATION.md:280-292`; algorithm in
`forge/oracle/solve.py::enumerate_gpu`, `depth in range(28,0,-1)`). Date 2025-12-30 (2559818).
**Wiki status: PARTIAL** — `entities/forge.md` names "Perfect-information solver ... solves by
backward induction," but no fresh reader learns it is `solver2` promoted or how the enum/solve
split works.

**F2 — V is value-to-go in Team-0 frame, in points [-42,+42], and cannot encode score-so-far.**
"the packed `state` does not encode 'score so far' or trick history, so V cannot represent an
accumulated score; it must be value-to-go" (`forge/oracle/schema.py:28-31`). 2025-12-30.
**Wiki status: PARTIAL** — `expected-q-value.md` describes E[Q] loosely as "averaged over game
outcomes" but the exact team-frame / value-to-go semantics live only in the schema docstring.

**F3 — E[Q] targets are computed as an expectation *per action*, deliberately NOT argmax-per-world-then-average. This is the anti-strategy-fusion care.**
Verbatim: "This is not 'strategy fusion': we are not selecting the best action inside each world
and then averaging. We compute an expectation **per action** and let the downstream policy choose
`argmax` on μ(a)." (`docs/EQ_STAGE2_TRAINING.md:101`). 2026-01-18.
**Wiki status: ABSENT** — the wiki's entire strategy-fusion discussion (`pimc.md`,
`rank-vs-price.md`, `gus-probe.md`) is April+ and treats strategy fusion as a *rediscovered PIMC
inference flaw*, never citing that the founding E[Q] spec was explicitly architected around it in
January. See §3.

**F4 — The "6-6 vs 2-2 slam dunk": single-deal training teaches fragile strategies; marginalizing over N opponent distributions is the fix.**
"With a trump-heavy hand ... 6-6 (double-six) is **always** the best lead ... 2-2 **might** work
if opponents can't beat it, but often fails. Training on single-deal data causes the model to learn
fragile strategies." Fix: "generate N shards with different opponent distributions ... implicitly
learns to prefer **robust** moves (high Q across all samples) over **fragile** ones."
(`forge/ORIENTATION.md:628-653`; validation TODO "Slam Dunk Test" `forge/eq/README.md:640`,
t42-xtu1). Cites arXiv 2407.05876: "~3 samples per position is sufficient" (`ORIENTATION.md:673`).
Date ~Jan–Feb 2026. **Wiki status: ABSENT** — no wiki page names the 6-6-vs-2-2 canonical example
or the "3 samples is enough" marginalization result.

**F5 — Candlewax, quantified at founding: the same hand's outcome swings −42→+40; only 11% of hands are stable; risk is unpredictable from your own hand.**
"Mean V spread 34.8 points ... Only 11% of hands are 'stable' (spread < 10)"
(`forge/analysis/report/11_imperfect_info.md`). "Risk is fundamentally unpredictable from your
hand. The uncertainty in 42 comes from opponent hands, not your own." (`forge/analysis/CLAUDE.md`,
13b; σ(V) model CV R² = **-0.34**, "worse than mean prediction"). 2026-01-06.
**Wiki status: PARTIAL/CONTRADICTED-ON-DATE** — `candlewax.md` describes the *phenomenon* well but
dates `first_seen: 1efb9c5` (April 2026) and frames it as a Burl tool field. The founding
January measurement (and that the mean-is-a-lie insight predates the name by ~3.5 months) is absent.

**F6 — The napkin formula: only n_doubles and trump_count survive; everything else is noise.**
"Oracle E[V] ≈ 14 + 6×(n_doubles) + 3×(trump_count)" — "only two features survive multivariate
analysis" with bootstrap CIs excluding zero; 2-feature model CV-generalizes better than 10-feature
(`report/00_executive_summary.md:85-102`). 2026-01-06.
**Wiki status: PRESENT** — `entities/forge-analysis.md:55` records it as "the napkin formula."

**F7 — Inverse risk-return: good hands are *safer* hands (r = -0.381), opposite of markets.**
`report/00:126` and `forge/analysis/CLAUDE.md` (12a). 2026-01-06.
**Wiki status: PRESENT** — `topics/risk-return-inverse.md`, `forge-analysis.md:54`.

**F8 — Six pieces of Texas-42 folk wisdom were tested against the oracle; NONE confirmed; "coverage beats trumps" was REFUTED-INVERTED.**
"Coverage *hurts* E[V] (β = -0.288, p=0.0001). Voids enable trumping ... 4 trumps + voids beats 2
trumps + coverage." (`forge/README.md:186-198`). Also: threshold cliffs at 30/35 NOT confirmed
(largest cliff is 38→39). 2026 (README). **Wiki status: ABSENT** — no wiki page carries the
folk-wisdom refutation table or the coverage-inversion result. High misleading-risk (a fresh model
will cite folk wisdom as strategy).

**F9 — Game phase structure: "order → chaos → resolution." Endgame (depth ≤4) is 100% deterministic; opening 40% consistent; mid-game 22% (most chaotic).**
`report/00:138-148`, `forge/analysis/CLAUDE.md` (15d). 2026-01-06.
**Wiki status: ABSENT** — not found in wiki grep.

**F10 — Adaptive "drunken master" sampling: game outcomes are the WRONG metric for E[Q] label quality; confident labels across diverse positions are what matter.**
"Game outcomes are NOT the right metric - tiny E[Q] differences cascade into different games. What
matters for training: confident labels across diverse positions ('drunken master' technique)."
Adaptive achieves 8.4× better SEM (0.077 vs 0.649) (`forge/ORIENTATION.md:856-860`; README.md:105).
~Jan 2026. **Wiki status: ABSENT** — the "drunken master" principle and the "outcomes are the wrong
metric" warning are not in the wiki. (Note: `regret-eval` in wiki independently rediscovered a
related "mean regret is misleading" point for Burl — see §5.)

**F11 — Bidding uses Monte-Carlo simulation, NOT the value head, because bid thresholds are a cliff landscape MSE can't fit.**
"The value head predicts smooth game values, but bidding thresholds (30, 31, 32, 36, 42) create a
'cliff' landscape that doesn't suit MSE regression. Simulation naturally handles this."
(`forge/ORIENTATION.md:565`). **Wiki status: ABSENT/PARTIAL** — the wiki's `rank-vs-price` reaches a
*related* conclusion ("bids consume prices ... read cardinally") in July 2026 without noting the
founding docs already knew bidding is a cliff/threshold problem unsuited to smooth value regression.

**F12 — E[Q] outputs are POINTS, not logits — do NOT softmax them (a repeated, emphatic warning).**
"`e_q_mean` contains Q-values in **POINTS** ... Do NOT apply softmax" (`forge/eq/README.md:385`,
repeated 5+ times; schema field `metadata["schema"]["q_semantics"]=="minimax_value_to_go"` exists
"to make it difficult to accidentally treat `e_q_mean` as policy logits"
`docs/EQ_STAGE2_TRAINING.md:345`). 2026-01. **Wiki status: ABSENT** — a fresh model wiring E[Q] into
a policy will softmax it unless warned.

**F13 — Marginalized oracle data is a distinct, built dataset (201 base seeds × 3 opp configs, `deal_with_fixed_p0`).**
`forge/analysis/CLAUDE.md` (Marginalized shards section); `report/11`. It is the empirical bridge
between perfect-info oracle and imperfect-info reality. **Wiki status: PARTIAL** — wiki mentions
"marginalized data is a partial bridge" (log.md:1563) but does not document the dataset's
construction or that it is the founding candlewax evidence.

---

## 3. The strategy-fusion story (reconstructed)

The founding team cared about strategy fusion **twice**, at two depths, and wrote both down.

**Layer 1 — the training-data care (Stage 1 marginalization).**
The Stage-1 oracle solves *one specific deal* with god-view. A net trained on single-deal Q-values
learns strategies that only work when opponents happen to hold the cards that deal assumed — the
6-6-vs-2-2 case (F4). The fix was the **marginalized shard** dataset: same P0 hand, N opponent
shuffles (`generate_continuous --marginalized`, `deal_with_fixed_p0`). Gradient descent over N
worlds implicitly rewards moves with high Q *across* worlds (robust) and punishes moves high in only
some (fragile). Grounded in arXiv 2407.05876's "~3 samples suffices" (`ORIENTATION.md:624-673`).

**Layer 2 — the label-construction care (Stage 2 E[Q]), and the "and I mean it".**
This is the sharp one, `docs/EQ_STAGE2_TRAINING.md:91-101`. The naive imperfect-info target is
"sample worlds, in each pick the best action, average the picks." That is **strategy fusion**: it
assumes the actor will *know which world it is in* at decision time, which it won't. The spec
refuses it explicitly:

> "This is not 'strategy fusion': we are not selecting the best action inside each world and then
> averaging. We compute an expectation **per action** and let the downstream policy choose `argmax`
> on μ(a)."  (`docs/EQ_STAGE2_TRAINING.md:101`)

Concretely (`§4.4`): for each legal action a, μ(a) = (1/N) Σ_i Q_i(a) — average the *same action's*
value across worlds, THEN argmax over a. The order of operations (average-then-max, never
max-then-average) *is* the anti-strategy-fusion invariant. The spec also builds **posterior
weighting** (§4.5) so worlds are reweighted by transcript likelihood (a Bayes-shaped importance
sampler), and **variance σ²(a)** is emitted as a first-class uncertainty signal — the team wanted
the model to *see* the spread, not just the mean (§4.5, §5).

**What the wiki did with this later.** By April–July 2026 the wiki rediscovered "strategy fusion" —
but as a *flaw of inference-time PIMC*, framed fresh:
- `pimc.md`: "direct π_me beats both single-step PIMC variants ... π_me is itself the marginalized
  policy ... Adding worlds at inference introduces variance without providing new marginalization."
  (5a4c9b9) — this is *exactly* the founding insight (the net already carries the average), but
  presented as a new empirical finding, uncited to the January spec.
- `rank-vs-price.md` (July 2026): "PIMC's strategy-fusion optimism — per-world double-dummy values
  assume the player will *know the world* when acting later — is a distribution-shape error." This
  is a *beautiful, correct* restatement of the same hazard the EQ spec's §3 was avoiding — arrived
  at independently, 6 months later, with no link back.
- `gus-probe.md`: an early "strategy fusion" probe diagnosis was **retracted** as measurement error.

The through-line the wiki lacks: **the E[Q] framework was designed from the start (Jan 18, 2026) to
compute a marginalized, per-action, posterior-weighted expectation specifically so it would NOT
commit strategy fusion.** The "average-per-action-then-argmax" order is load-bearing and documented.
A fresh model that "improves" Gus by sampling worlds and taking argmax-per-world-then-averaging would
be reintroducing the exact error a spec section was written to forbid.

---

## 4. The candlewax first-contact

"Candlewax" as a *word* is April 2026 (Burl era). The *phenomenon* — bimodal / high-spread outcome
distributions where the mean lies — was measured, visualized, and reasoned-about at founding.

**Earliest quantified evidence (2026-01-06, `report/11_imperfect_info.md`):**
- Mean V spread across opponent configs = **34.8 points**; max spread **82**; 38% of hands spread
  >40; **only 11% stable** (spread <10).
- "Opponent hands matter enormously. The same P0 hand can swing from -42 to +40 depending on who
  holds what."
- Variance decomposition (`report/00:106-118`): **53% of oracle outcome variance is within-hand**
  (opponent cards), only 47% between-hand. Even with perfect play, the majority of your outcome is
  not about your hand.
- Risk is unpredictable from your own hand: σ(V) model **CV R² = -0.34** (`CLAUDE.md` 13b/14b).

**The visual artifact (2026-01-22→24):** the E[Q] pipeline gained a full outcome **PDF** ("store
full E[Q] PDF (85 bins per action)", ef199b0) and the browser visualizers, including the per-play
outcome histograms later nicknamed the **"melted candlewax" discs** (`eq_pdf_discs.html`). The
naming is retro-applied in `lem/OVERVIEW.md:969`: "`eq_pdf_discs.html` (per-play outcome histograms,
the 'melted candlewax' discs)." So the shape the name refers to was rendered in January; the name
attached in April.

**Why this matters as a wall:** if 53% of variance and the whole outcome bimodality come from hidden
opponent cards, then a scalar mean E[Q] is a *lossy* summary of a decision — the founding data proves
it before any LLM or Burl work began. Every downstream design tension ("mean is a lie," teach the
model the *shape*, reasoning-coherence over point estimates) traces to this January measurement.

---

## 5. Top 10 things the wiki doesn't know (ranked by misleading-potential)

1. **E[Q] was deliberately built to avoid strategy fusion — average-per-action-then-argmax, never
   max-per-world-then-average (`EQ_STAGE2_TRAINING.md:101`).** The wiki's strategy-fusion content is
   all a *later rediscovery* framed as a PIMC flaw. A fresh model reading only the wiki will not know
   the founding order-of-operations invariant and can silently reintroduce the error — this is the
   exact "fresh sessions rebuild PIMC" failure the mining brief warns about, in its purest form.

2. **`forge/oracle/` is `solver2` promoted, and Stage 1 is a supervised distillation of an exact
   tablebase — not self-play, not MCTS.** (2559818, "GPU tablebase solver (from solver2/)".) The wiki
   `entities/forge.md` says "backward induction" but nothing about the solver2 lineage or that the
   net is an imitation copy. Fresh readers conflate the exact solver with the learned net.

3. **The candlewax phenomenon was measured and named-in-spirit in January 2026, ~3.5 months before
   the word.** Wiki `candlewax.md` dates it `first_seen: 1efb9c5` (April) as a Burl tool field. The
   founding evidence (V spread 34.8, −42→+40, 53% within-hand variance) and its origin in
   `report/11` are absent — the wall looks newer and shallower than it is.

4. **Six folk-wisdom rules were oracle-tested and NONE survived; "coverage beats trumps" was
   inverted (coverage *hurts* E[V]).** (`README.md:186-198`.) Entirely absent from the wiki. A fresh
   model will otherwise cite coverage/voiding/threshold-cliff folk wisdom as if true.

5. **The 6-6-vs-2-2 "slam dunk" canonical example + "~3 opponent samples suffices" marginalization
   result.** (`ORIENTATION.md:628-673`, arXiv 2407.05876.) The wiki has the abstract idea that π_me
   "is the marginalized policy" but not the worked example or the sample-count evidence that makes it
   teachable.

6. **E[Q] values are POINTS, never logits — do not softmax (F12).** Repeated emphatically in
   founding docs, absent in wiki. A concrete foot-gun for anyone wiring E[Q] into a policy head.

7. **"Drunken master": game outcomes are the WRONG metric for label quality; confident labels across
   diverse positions are the goal (adaptive SEM sampling, 8.4× better).** (`ORIENTATION.md:856`.)
   Absent. The wiki's Burl-side `regret-eval` independently found "mean regret is misleading" but
   never connects to the founding E[Q] version of the same lesson.

8. **Bidding is a cliff/threshold landscape unsuited to smooth MSE value regression — hence Monte
   Carlo P(make).** (`ORIENTATION.md:565`.) The wiki's July `rank-vs-price` reaches a cousin of this
   ("bids consume prices, read cardinally") with no link to the founding reason.

9. **Posterior-weighted E[Q] with ESS/variance diagnostics was first-class from the Stage-2 spec** —
   worlds reweighted by transcript likelihood, σ²(a) emitted as an uncertainty signal
   (`EQ_STAGE2_TRAINING.md:§4.5, §5`). The wiki mentions Gus-belief reweighting later but not that
   the Bayes-shaped importance sampler and uncertainty head were designed in January.

10. **Quantitative oracle structure facts that are strategy-relevant and absent from wiki:** endgame
    (depth ≤4) is 100% deterministic / mid-game is the chaos peak (22% consistency) — "order → chaos
    → resolution" (F9); count-capture explains ~92% of late-game V variance, R²>0.99 at depth ≤12
    (`report/00:40-50`); 5-5 is the single strongest domino (2.8× enriched in winners), 6-0 the worst
    (3× in losers) (F8 neighbors). These are the empirical spine of "what the oracle actually knows."

---

## Summary (5 lines)

Forge's founding era (Dec 30 2025 → Jan 2026) built an **exact backward-induction tablebase**
(`solver2` promoted to `forge/oracle/`), distilled it into a 3.3M `DominoTransformer`, then in
January designed the **E[Q] framework to deliberately avoid strategy fusion** — averaging per action
across sampled worlds, never argmax-per-world (`EQ_STAGE2_TRAINING.md:101`, "and I mean it"). The
**candlewax wall** was measured on 2026-01-06 (report 11: same hand swings −42→+40, 53% of variance
is opponents' cards) and rendered as 85-bin PDF "discs" by 2026-01-24 — ~3.5 months before the word
"candlewax" was coined for a Burl tool. The wiki's strategy-fusion pages are all an April–July
*rediscovery* framed as a PIMC inference flaw, uncited to the founding spec that engineered around
it. Six folk-wisdom rules were oracle-refuted (coverage *inverts*); none of that reached the wiki.

**Most heartbreaking omission:** `docs/EQ_STAGE2_TRAINING.md:101` — "This is not 'strategy fusion':
we are not selecting the best action inside each world and then averaging. We compute an expectation
per action." The entire E[Q] framework was architected in January 2026 to *not* commit strategy
fusion, and wrote down the exact average-then-argmax invariant that guarantees it. Six months later
the wiki rediscovered strategy fusion from scratch as "PIMC optimism" (`rank-vs-price`, July) with no
memory that it had already been named, understood, and designed against at the foundation — the
precise mechanism by which a fresh strong model, reading only the wiki, rebuilds the PIMC solver the
founders already knew they didn't need.
