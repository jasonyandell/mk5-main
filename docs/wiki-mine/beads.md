# Beads Archive Mining — Texas 42 Founding Era (Nov 2025 → May 2026)

Source: `.beads/issues.jsonl` (650 issues + 2 memory records), `.beads/interactions.jsonl` (238 records).
Method: parsed JSONL directly with `forge/venv/bin/python` (bd uninstalled). READ-ONLY.
Wiki cross-check: `grep -rliE` over `wiki/`.

**Headline framing:** The beads tracker predates the wiki by ~5 months. The wiki was born
**2026-04-24** (`git log` first commit `b1de970`). **496 of 650 issues (76%) were created before
the wiki existed.** Those pre-wiki issues carry the entire pre-E[Q] design ladder — MCCFR, PIMC,
the factored engine algebra, the value-head failure, strategy fusion — almost none of which any
wiki page absorbed. The wiki effectively begins its story at E[Q]/candlewax and treats everything
before as prehistory.

---

## 1. Census

### Totals
- **650 issues** (plus 2 `_type:memory` records — DoltHub backup config, hooks-uninstalled note).
- Date range: **2025-11-16 → 2026-05-25**.
- Status: `closed` 625, `open` 20, `in_progress` 4, `blocked` 1.
- Type: `task` 491, `feature` 77, `epic` 28, `bug` 26, `chore` 28.
- Priority: p0=31, p1=219, p2=328, p3=69, p4=3.

### By month created
| Month | Created | Closed | Era |
|-------|--------:|-------:|-----|
| 2025-11 | 144 | 121 | **pre-wiki** (founding) |
| 2025-12 | 148 | 117 | **pre-wiki** (founding) |
| 2026-01 | 204 | 194 | **pre-wiki** (founding) |
| 2026-02 | 0 | 0 | *gap — no issues created* |
| 2026-03 | 0 | 0 | *gap — no issues created* |
| 2026-04 | 5 | 2 | wiki born 04-24 |
| 2026-05 | 149 | 191 | post-wiki (w42 workstream) |

**Two distinct epochs separated by a Feb–Mar 2026 dead zone.** The founding burst (Nov–Jan, 496
issues) built the engine, solver, oracle, and first NN/PIMC/E[Q] work. After a two-month silence,
a second burst (May, 149 issues) is the wiki-native w42 book-validation campaign — already
well-documented in `wiki/experiments/`.

### Close-reason quality
Of 625 closed: **434 closed with an empty reason, 55 auto-closed as stale (>30d, on 2026-05-03),
only 136 carry a real written reason.** Critically, **essentially all 136 real close-reasons are
from May 2026** (post-wiki, w42). The founding era closed issues terse/empty — so **the founding-era
treasure lives in issue *descriptions*, not close reasons.** Descriptions are rich: 341 issues
have >500-char descriptions, 176 >1000 char, 58 >2000 char.

### interactions.jsonl
238 records, all `kind: field_change` (status/priority/description transitions) by "Jason Yandell".
**Purely mechanical audit log** — the only content is the `reason` field, which duplicates close
reasons already in issues.jsonl. No standalone design content. (Note: the brief estimated 84k lines;
the actual file is 238 lines.)

---

## 2. Theme Map (cluster / count / date-range / era note)

| Cluster | Count | Range | Pre-wiki density |
|---------|------:|-------|------------------|
| LEM/burl | 194 | Nov–May | spread; heaviest Dec (65) + May (47) |
| solve/oracle | 157 | Nov–May | **founding-dense**: Dec 64, Jan 36 |
| bidding | 150 | Nov–May | **founding-dense**: Nov 29, Dec 36; May 56 |
| infra | 90 | Nov–May | Dec 43 (pipeline build-out) |
| lens/utility | 72 | Nov–May | May 30 (utility-lens work), Jan 15 |
| belief/gus | 66 | Apr–May | **entirely post-wiki** (May 65) — already in wiki |
| eq/expected-Q | 63 | Dec–May | Jan 19 founding + May 36 |
| arena | 40 | Nov–May | Dec 15, Nov 8 |
| strategy-fusion | 23 | Dec–May | **founding-concentrated: Dec 2025 (19)** |
| null/abandon/pivot | 14 | Nov–May | scattered |
| candlewax/bimodal | 10 | Nov–May | thin everywhere; **"candlewax" is a later wiki coinage, barely in beads** |

**Reading:** the belief/Gus cluster is 100% post-wiki and already synthesized. The genuine
pre-wiki treasure clusters are **solve/oracle, bidding, and strategy-fusion (all peaking Dec 2025)** —
the months when the exact-solver → distillation → imperfect-info pipeline was invented and argued out.
The word "candlewax" almost never appears in beads; the underlying bimodal/distribution-lens insight
is present but was named later, in the wiki era.

---

## 3. Findings Ledger

Each: {id · date · claim · verbatim quote · wiki status}. Status = ABSENT / PARTIAL(page) / PRESENT(page).

### F1 — The "depressed android" defect that forced exact per-world solving
- **t42-9ed · 2025-12-14 · task**
- Claim: PIMC's *heuristic rollouts* made defeatist plays when losing ("dump count, we're losing
  anyway"), because greedy per-trick rollout can't see that fighting still has win lines. The fix —
  replace heuristic rollouts with **full minimax to hand completion** — is the origin of the
  exact-solve-each-sampled-world architecture the whole project now rests on.
- Quote: *"Heuristic rollout is greedy... This can't see future tricks. When losing: Option A (fight):
  avg 18 pts, lose 80%. Option B (give up): avg 15 pts, lose 95%. Both look similarly bad → AI might
  pick 'give up'. Solution: Replace heuristic rollouts with full minimax to hand completion... No
  heuristic evaluation function - searches to terminal state."*
- **Wiki status: ABSENT.** No wiki page contains "depressed android" or the heuristic-rollout→minimax
  pivot. The wiki treats exact per-world solving as a given, never as a bug fix.

### F2 — Strategy Fusion: the formal Max(Average) vs Average(Max) methodology
- **t42-g8wt · 2026-01-05 · task** (research synthesis → `docs/research/answer.md`)
- Claim: The central methodological hazard of PI-oracle-for-II play is **Strategy Fusion**
  (Frank & Basin 1998): `E[max_play(outcome)] ≥ max_play(E[outcome])`, always ≥ 0, causing systematic
  *optimistic bias*. Texas 42 has a *second* fusion site — trump selection. The fix is to fix the
  decision **before** aggregating: `max_trump E_M[V(d,trump)]`, never `E_M[max_trump V(d,trump)]`.
- Quote: *"WRONG: E_M[ max_trump V(d, trump) ] ...the oracle picks the optimal trump for each sampled
  deal. But you don't know which deal you're in. RIGHT: max_trump E_M[ V(d, trump) ]. This is
  Max(Average) vs Average(Max)—the central methodological fix."*
- **Wiki status: PARTIAL / effectively ABSENT.** `wiki/entities/champion.md` names the *phrase*
  "strategy fusion" twice as a residual PIMC flaw, but **no wiki page cites `docs/research/answer.md`,
  states the inequality, names Frank & Basin, or gives the "fix trump before aggregation" rule.**
  The formal analysis that guided the E[Q] generator design is unreferenced by the wiki (0 citations).

### F3 — Percentile-25 pessimistic aggregation (the ancestor of `robust_q25`)
- **t42-eiod / t42-tgke · 2026-01-03 · feature**
- Claim: Before the mean-marginalization that became E[Q], the team built a **pessimistic
  percentile-25 Q aggregation** across opponent distributions, to punish situationally-fragile moves
  ("2-2 works only when opponents lack trumps"). This is the direct Jan-2026 ancestor of the
  `robust_q25` utility that Lens v1 tests in May.
- Quote: *"aggregating Q-values across opponent distributions using percentile_25 (pessimistic/robust
  estimation)... the model learns fragile strategies because it can't distinguish universally optimal
  moves (6-6 always good) from situationally optimal moves (2-2 works only when opponents lack trumps)."*
- **Wiki status: ABSENT.** `percentile`/`robust_q25` appears in May wiki pages as a utility name, but
  the wiki never records that pessimistic aggregation was *built and tried as a training-data
  transform in Jan 2026*, nor connects `robust_q25` back to it. The lineage is severed.

### F4 — Marginalized-Q genesis: the 6-6-vs-2-2 fragility and the "3 samples" result
- **t42-elle · 2026-01-02 · feature**
- Claim: The founding rationale for E[Q]. Perfect-info Q leaks: model preferred 2-2 over 6-6 when a
  specific deal showed both at Q=+42, though 6-6 is universally optimal. Fix = run oracle 3× per P0
  hand with different opponent distributions; implicit averaging teaches robustness. Grounded in
  arxiv 2407.05876 ("~3 samples per position is sufficient").
- Quote: *"Training from perfect-info oracle Q-values causes the model to learn fragile strategies
  (e.g., preferring 2-2 over 6-6 when both show Q=+42 in a specific deal, but 6-6 is universally
  optimal)... Key finding: ~3 samples per position is sufficient."*
- **Wiki status: PARTIAL.** "marginaliz" appears in 12 wiki files as a mechanism, but the founding
  *rationale* — the concrete 6-6/2-2 fragility example, the arxiv reference, the 3-samples result — is
  ABSENT. The wiki has the *what*, not the *why it was invented*.

### F5 — Value head failed at 7.4 pts MAE → pivot to game-simulation bidding
- **t42-6m0l · 2025-12-31 · task**
- Claim: A direct value head to predict hand strength **plateaued at 7.4 points MAE (too noisy)**.
  Because the *policy* model was 97.8% accurate at move choice, the team pivoted: let the policy play
  complete games and *count actual points*. This simulation-based bidder is the direct ancestor of the
  forge E[Q] bidding evaluator. Also fixes the objective: bidding is **P(make) threshold crossing, not
  E[points]** — `mark_swing = (2·P(make) − 1)·marks_at_stake`.
- Quote: *"We tried a value head to directly predict hand strength. It plateaued at 7.4 points MAE -
  too noisy. But our policy model is 97.8% accurate at picking moves. So: let it play complete games
  and count actual points... Game awards marks based on thresholds, not raw points... So we compute
  P(make), not E[points]."*
- **Wiki status: value-head-failure ABSENT (0 wiki hits for "7.4"/"value head plateau"); P(make)-not-
  E[points] PARTIAL** (`champion.md` says "marks-to-7 win probability... not raw points"; log.md §16-17
  argues p_make-argmax). The wiki has the conclusion but not the failed-value-head origin story.

### F6 — The Factored Algebraic Model: absorption ⊥ power, S₇ symmetry, "suit 7"
- **t42-9xy3 · 2025-12-26 (imported from a ClaudeAI conversation) + t42-vwnt · 2025-12-26**
- Claim: The current engine's core algebra. Trump conflates two *independent* operations —
  **Absorption** (which dominoes belong to which suit) and **Power** (which beats which). Nello proves
  independence (same absorption as doubles-trump, different power). All 7 pip-trumps are isomorphic
  under S₇; therefore *all absorbed dominoes lead "suit 7"*, not the trump pip value. Four small
  constant tables (828 entries) replace all conditional game logic and are GPU-ready.
- Quote (design): *"Trump conflates two independent operations: Absorption... and Power... Nello proves
  they're independent."* Quote (user decision, t42-vwnt): *"User explicitly chose: All absorbed
  dominoes use suit 7, rejecting the old model where trump pip value was reused. Quote: 'this confusion
  via incidental value alignment has cost us time and time again.'"*
- **Wiki status: ABSENT.** No wiki page mentions absorption/power factoring, S₇ symmetry, "suit 7", or
  the factored tables. This is the mathematical foundation of `domino-tables.ts`/the solver, and the
  wiki is silent on it. (Some content lives in `docs/theory/SUIT_ALGEBRA.md`, but the wiki doesn't
  point there.)

### F7 — "Decided at declaration" manifold hypothesis (intrinsic-dim ≈ 5)
- **t42-xp0p · 2026-01-06 · task** (+ epic **t42-q0be** Imperfect Information Analysis Suite, 26 tasks)
- Claim: A structural hypothesis that the game is "decided at declaration" — all paths from one deal
  should lie on a low-dimensional manifold (intrinsic dim ≈ 5, one per count), diverging only at a few
  genuine decision points. Predicted bimodal intrinsic dimension distinguishing "easy" vs "contested"
  deals — an early articulation of the bimodality that later became "candlewax".
- Quote: *"If the game is 'decided at declaration,' all paths from the same deal should cluster tightly
  in some embedding space, diverging only at the few genuine decision points... Is intrinsic dimension
  ≈ 5 (one per count)?... Some deals are 'contested' → Bimodal intrinsic dimension (easy vs hard deals)."*
- **Wiki status: ABSENT.** No wiki page carries the manifold / "decided at declaration" / intrinsic-
  dimension framing, nor the 26-task Imperfect Information Analysis Suite (`t42-q0be`) it anchored.

### F8 — MCCFR abandoned: count-centric abstraction too lossy
- **t42-tgr · 2025-12-14 · task** (+ retirement of `src/game/ai/cfr/`)
- Claim: An entire earlier research line — MCCFR (Monte Carlo Counterfactual Regret Minimization) — was
  built (172MB trained strategy) and *deleted*. The count-centric abstraction couldn't learn
  suit-specific play (e.g., "don't lead 5-0 when treys are trump").
- Quote: *"MCCFR was explored but the count-centric abstraction proved too lossy. The strategy couldn't
  learn suit-specific play... CFR is punted. 'Boring and competent' isn't worth the squeeze when we
  could get that with fixed MCTS, and neural nets offer more upside for fun play."*
- **Wiki status: ABSENT.** Zero wiki mention of MCCFR/CFR. A whole abandoned approach — and the
  *reason* it failed — is invisible to a wiki-only reader, who might re-propose it.

### F9 — PIMC blunder-washout empirics (the tolerance that justified NN distillation)
- **t42-k54h · 2025-12-29 · task**
- Claim: The empirical check that PIMC *averaging washes out* the transformer's ~4.5% blunder rate:
  soft-vote at 50 samples cut mean regret 0.58→0.19 and blunders 4.5%→0.52%. Also states the correct
  objective ("regret on the TRUE deal", not "agreement with DP optimal"). This is the evidence base for
  trusting a distilled NN inside PIMC.
- Quote: *"Wrong objective: 'Does PIMC agree with DP optimal?' Right objective: 'What regret does PIMC
  incur on the TRUE deal?'... DP optimal knows hidden hands. PIMC optimizes expected value over
  plausible hands. These are different problems!"*
- **Wiki status: ABSENT.** The regret-washout curve and the "measure regret on the true deal" framing
  are not in the wiki.

### F10 — MCTS-bidding replaced miscalibrated threshold logic (Nov 2025)
- **t42-oqd · 2025-11-20 · task**
- Claim: The *first* bidding system was lexicographic hand-scoring vs `BID_THRESHOLDS` that was so
  miscalibrated the AI always bid 30 and never passed. Replaced with MCTS simulation. This is the
  earliest documented "measure by simulation, not by hand-scored heuristic" decision — the seed of the
  entire later philosophy.
- Quote: *"Thresholds completely miscalibrated → AI always bids 30, never passes. This code was never
  properly hooked up and is a dead end... Use Monte Carlo simulation for bidding decisions."*
- **Wiki status: ABSENT.**

### F11 (PRESENT — noted to prevent false "absent" claims) — Lens v1: EV wins, p_make is the WORST utility, production still ships p_make
- **t42-4ouu · 2026-05-04 (closed) + t42-10yj · open** — post-wiki
- Claim: Head-to-head, EV-greedy beats p_make-greedy by **+5.42 pts/hand**, total order
  `ev > robust_q25 ≳ cvar_10 > p_make`, all CIs exclude zero. Yet production
  `forge.eq.generate.actions.select_actions` is essentially Lens(p_make) — **the worst utility.**
  `t42-10yj` (still OPEN) is the one-line ev-argmax fix.
- **Wiki status: PRESENT.** Well-covered in `wiki/log.md` §2461–2478 and
  `wiki/experiments/w42-book-claim-synthesis-and-ai-directions.md` (§765–766, 841). Listed here only so
  the ledger is not mistaken for claiming it absent. **Note the internal tension the wiki does hold:**
  log.md §16-17 (b4040c5) argues *"oracle utility is p_make (cliff-shaped), not E[Q]; U = E[Q] + C·p_make
  was wrong (picks guaranteed-loss over sliver-of-hope). Correct: pure p_make argmax"* — which sits
  against Lens v1's "EV wins." Both are in the wiki; the reconciliation (generation vs head-to-head
  play; the still-open `t42-10yj`) is worth a wiki decision page.

---

## 4. Decision / Pivot Ledger (verbatim "tried/decided/abandoned because…")

| id · date | Decision | Verbatim |
|-----------|----------|----------|
| **t42-tgr** · 2025-12-14 | **Abandon MCCFR** | *"count-centric abstraction proved too lossy... couldn't learn suit-specific play... CFR is punted. 'Boring and competent' isn't worth the squeeze when we could get that with fixed MCTS, and neural nets offer more upside for fun play."* |
| **t42-9ed** · 2025-12-14 | **Heuristic rollouts → minimax-to-completion** | *"When losing... Both look similarly bad → AI might pick 'give up'. Solution: PIMC + Minimax... searches to terminal state."* |
| **t42-oqd** · 2025-11-20 | **Threshold bidding → MCTS bidding** | *"Thresholds completely miscalibrated → AI always bids 30, never passes... This code was never properly hooked up and is a dead end."* |
| **t42-6m0l** · 2025-12-31 | **Value head → game simulation** | *"We tried a value head... It plateaued at 7.4 points MAE - too noisy... So: let it play complete games and count actual points."* |
| **t42-vwnt** · 2025-12-26 | **All absorbed dominoes = "suit 7"** (user decision) | *"User explicitly chose: All absorbed dominoes use suit 7, rejecting the old model where trump pip value was reused. Quote: 'this confusion via incidental value alignment has cost us time and time again.'"* |
| **t42-elle** · 2026-01-02 | **Adopt marginalized Q (3 opp-samples)** | *"Training from perfect-info oracle Q-values causes the model to learn fragile strategies... ~3 samples per position is sufficient."* |
| **t42-eiod** · 2026-01-03 | **Add percentile-25 pessimistic aggregation** | *"aggregating Q-values across opponent distributions using percentile_25 (pessimistic/robust estimation)."* |
| **t42-g8wt** · 2026-01-05 | **Fix decisions before aggregating (anti-fusion)** | *"WRONG: E_M[ max_trump V ]... RIGHT: max_trump E_M[ V ]. Max(Average) vs Average(Max)—the central methodological fix."* |
| **t42-4ouu** · 2026-05-04 | **EV beats p_make head-to-head** | *"EV WINS, total ordering ev > robust_q25 ≳ cvar_10 > p_make, all CIs exclude zero, ev beats p_make by +5.42 pts/hand. Wave 4.0 reading inverted."* |
| **t42-10yj** · 2026-05-03 (OPEN) | **Production selector is the worst utility** | *"select_actions is essentially Lens(p_make)... the current production E[Q] action selector is the WORST of the four utilities tested."* |

---

## 5. Top 10 Things the Wiki Doesn't Know

Ranked by how badly a fresh strong model reading only the wiki would be misled or waste effort.

1. **The entire pre-E[Q] design ladder is missing.** MCCFR → threshold-bidding → MCTS/PIMC-heuristic →
   PIMC-minimax → transformer distillation → marginalized-Q → percentile-25 → E[Q]. The wiki opens at
   E[Q]/candlewax and treats the rest as prehistory. A fresh model can't see *why* each rung was
   climbed or abandoned, and could re-propose a dead approach (esp. #2, #6). **This is the single most
   consequential gap.**

2. **MCCFR was built and deleted, and *why* (F8).** 172MB of trained CFR strategy exists in git
   history. The wiki never mentions CFR. A model asked "should we try CFR/regret minimization?" would
   have no idea it was tried and failed on the count-centric abstraction.

3. **The factored engine algebra: absorption ⊥ power, S₇ symmetry, "suit 7" (F6).** This is the
   mathematical foundation of the whole engine and solver, decided by the user personally
   ("incidental value alignment has cost us time and time again"). The wiki is completely silent;
   a model refactoring the rules could reintroduce exactly the trump-pip aliasing bug the design
   eliminated.

4. **Strategy Fusion, formally (F2).** The wiki uses the phrase but never states the
   Max(Average)-vs-Average(Max) inequality, cites Frank & Basin, or records the "fix trump before
   aggregation" rule that shaped the E[Q] generator. `docs/research/answer.md` holds it and is cited by
   **zero** wiki pages. A model reasoning about the E[Q] estimator's bias would miss the known theory.

5. **The value head failed at 7.4 MAE — that's *why* bidding is by simulation (F5).** The wiki presents
   simulation-based/P(make) bidding as the design, never as the survivor of a failed direct-regression
   approach. A model might "simplify" back to a value-head regressor, re-walking the plateau.

6. **percentile-25 pessimistic aggregation is `robust_q25`'s ancestor (F3).** The wiki tests
   `robust_q25` in May with no memory that pessimistic aggregation was built as a *training-data
   transform* in January. The lineage — and the reason pessimism was ever on the table — is severed.

7. **The "depressed android" defect (F1)** — the concrete failure (defeatist count-dumping) that
   justified exact per-world minimax. The wiki assumes exact solving; it never records the bug that
   made it necessary. Reintroducing heuristic rollouts for speed would silently resurrect the defect.

8. **The "decided at declaration" manifold hypothesis and the 26-task Imperfect-Information Analysis
   Suite (F7).** intrinsic-dim ≈ 5, bimodal easy/contested deals — an early, structural articulation of
   what later got the "candlewax" name. The wiki's candlewax framing has no memory of this geometric
   origin or the analysis suite that probed it.

9. **PIMC blunder-washout empirics and the "regret on the TRUE deal" objective (F9).** The evidence
   that averaging tolerates a ~4.5% NN blunder rate (→0.5% at 50 samples), plus the sharp warning not
   to measure "agreement with DP optimal." A model tuning sample counts or eval metrics would redo this.

10. **The p_make-vs-EV production tension is *half* in the wiki (F11).** Both facts are there (log.md:
    "p_make argmax is correct" §16-17, and "EV wins, p_make is worst" §2461-2478) but **no wiki decision
    page reconciles them**, and the load-bearing consequence — `t42-10yj`, the one-line ev-argmax
    production fix, still OPEN — is only a log entry. A model could ship either direction confidently
    and be wrong about the other context.

---

## Summary (5 lines)

- The beads tracker is **76% pre-wiki** (496/650 issues, Nov 2025–Jan 2026); the wiki was born
  2026-04-24 and starts its story at E[Q]/candlewax, so the founding design ladder is largely unrecorded.
- Founding-era treasure lives in **descriptions, not close reasons** (434/625 closes were empty; all 136
  real reasons are post-wiki May); `interactions.jsonl` is mechanical status-change noise, not design.
- The wiki is genuinely missing, verbatim and verified at 0 wiki hits: MCCFR's build-and-abandon, the
  factored absorption/power/S₇ engine algebra, the "depressed android" minimax pivot, the 7.4-MAE
  value-head failure, and the "decided at declaration" manifold hypothesis.
- Strategy Fusion is named in the wiki but the formal Max(Average)-vs-Average(Max) methodology
  (`docs/research/answer.md`, Frank & Basin 1998) is cited by **zero** wiki pages.
- The Lens-v1 "EV wins / p_make is worst" finding **is** well-documented (don't re-mine it), but its
  reconciliation with the earlier "p_make-argmax is correct" ruling and the still-open production fix
  (`t42-10yj`) has no wiki decision page.

**Single best-documented decision the wiki never absorbed:** *t42-vwnt / t42-9xy3 — the factored
algebraic engine model.* It is fully specified (four constant tables, absorption ⊥ power, S₇ symmetry,
the Nello independence proof), carries a **direct user decision with a quotable rationale** ("All
absorbed dominoes use suit 7... this confusion via incidental value alignment has cost us time and
time again"), and underpins every solver/engine line in the repo — yet no wiki page mentions absorption,
power, S₇, or "suit 7." It is the highest-value, most-quotable, most-load-bearing decision the wiki
has completely forgotten.
