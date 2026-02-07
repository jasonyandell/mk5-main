# Perfect-Information Oracles for Imperfect-Information Bidding in Texas 42 — Final Synthesis (Codex, xhigh)

**Status:** Final synthesis (integrated response to `docs/research/question.md`)

**Inputs:**
- `docs/research/chatgpt52max-synthesis.md`
- `docs/research/opus45cc-synthesis.md`
- `docs/research/gemini3cli-synthesis.md`

**Scope (as in question.md):** Bidding evaluation only; seat 0 opening bidder; no opponent/partner bidding behavior model; no play-phase policy design beyond what is needed to interpret/label oracle-based estimates.

---

## Part I — Evaluation of the Three Synthesis Responses

### 1) `docs/research/chatgpt52max-synthesis.md` (ChatGPT 5.2 Max)

**What it gets right (high confidence)**
- **Correct framing:** distinguishes “oracle as evaluator” vs “oracle as policy teacher,” and anchors the problem in information sets (Kuhn, 1953) and determinization critiques (Frank & Basin, 1998).
- **Explicit estimator semantics:** consistently labels outputs as completion-model-dependent (`P_M(make | …)`), and treats PIMC aggregation as a generally optimistic estimate.
- **Decision-layer fusion fix:** clearly states the Texas 42-specific trap (letting the evaluator pick trump per completion) and fixes it with `Max(Average)` per contract.
- **Engineering-ready pipeline:** provides a concrete Path A pipeline (sampling, DP evaluation, binomial CI, score quantiles), plus an error decomposition and a per-cell output schema.
- **Compressed-oracle safety:** emphasizes “97% argmax agreement” ≠ value accuracy and pushes targeted audits + deferral rather than random checks.

**Gaps / tighten-ups**
- **Quantification of the fusion gap:** it characterizes when determinization can work but does not carry a concrete “order of magnitude” number for analogous trick-taking domains (question Q5).
- **Robust bound wording:** the “worst-case over samples” (`min_i make_i`) is useful as a stress-test flag, but it is a degenerate estimator (often 0/1) and should be labeled as such.
- **References need verification:** it explicitly labels references as “starting set,” which is correct; bibliographic details should still be verified when implementing.

**Net:** best methodological spine; use it as the backbone for definitions, labeling discipline, pipeline, and output schema.

### 2) `docs/research/opus45cc-synthesis.md` (Claude Opus 4.5)

**What it gets right (high confidence)**
- **Excellent coverage of Q1–Q14:** mirrors the question’s structure and touches every acceptance criterion.
- **Useful empirical intuition:** pulls in the “PIMC-friendly” parameters (leaf correlation, disambiguation) from Long et al. (2010) to argue Texas 42 is structurally closer to Bridge/Skat than to poker-like failures.
- **Practical bias-mitigation menu:** lists multiple ways to reduce/diagnose strategy fusion (consistent-action simulation, delayed resolution / EPIMC, variance penalization, bracketing).
- **Concrete risk communication:** pushes quantiles + a “robustness gap” style metric and emphasizes tail risk for close-to-threshold bids.

**Gaps / tighten-ups**
- **Hard numbers need cautious phrasing:** the ~0.1 points/game and related figures are presented as fairly crisp; without checking the original tables/conditions, they should be treated as indicative (parameter- and game-dependent), not as a universal correction.
- **“Apply 5–10% bias correction” is too specific as a default:** better to *measure and expose* the gap (bracketing / stress tests) than to bake in a fixed discount.
- **Some heuristics are uncalibrated:** e.g., “>30% lead variation” as a fusion flag may be useful, but it is not a literature-standard threshold; treat as an internal diagnostic pending validation.

**Net:** keep its breadth, Q-by-Q completeness, and the structural argument for when determinization is “close enough,” but downgrade any fixed numeric correction into “verify/measure in-domain.”

### 3) `docs/research/gemini3cli-synthesis.md` (Gemini 3 CLI)

**What it gets right (high confidence)**
- **Clear distillation warning:** states the “marginalized training” failure cleanly (oracle actions are conditioned on hidden hands).
- **Concrete adversarial-audit idea:** “killer layouts” via local swaps/hill-climbing is an implementable stress test that directly matches your need for targeted auditing.
- **Simple pipeline sketch:** gives an end-to-end picture from sampling to fixed-trump evaluation to risk summaries.

**Gaps / tighten-ups**
- **Non-standard vocabulary:** “Deterministic Mirage,” “Bead architecture,” and “SmartStack” are not standard literature terms; the question explicitly asks for standard vocabulary and avoids novel proposals unless sourced.
- **Coverage gaps:** less explicit about confidence intervals, completion-model validation, and the separation of modeling error vs sampling error vs fusion bias vs approximator error.

**Net:** keep the adversarial search / hill-climb audit mental model; avoid its non-standard terms unless backed by a source.

### Combined takeaways (what survives cross-checking)

1. **Oracle-as-evaluator is valid; oracle-as-policy-teacher is not** for imperfect-information decisions (information-set mismatch).
2. **All reported “P(make)” must be labeled with a completion model** `M`; “P(make)” without `M` is not well-defined.
3. **Fix the avoidable fusion at trump selection:** compute `Max(Average)`, not `Average(Max)`.
4. **Treat PIMC-style aggregation as an optimistic estimate;** bracket it and report tail risk and sampling uncertainty.
5. **Compressed oracle requires targeted/adversarial auditing and deferral**; random spot checks are insufficient when errors cluster.

---

## Part II — Final Synthesized Answer to `docs/research/question.md`

### 0) Thesis (one paragraph)

Use your perfect-information DP solver as a *deal evaluator* on sampled completions, not as a move teacher. The resulting contract × bid-level matrix is inherently **completion-model-dependent** and (if it relies on perfect-information play per completion) generally **optimistic** due to determinization / strategy fusion. In Texas 42 you can eliminate one major *avoidable* fusion artifact by evaluating each trump (contract) separately across the same completion set and only then choosing the best (`Max(Average)`). The remaining optimism must be handled by explicit labeling, uncertainty reporting (CI + quantiles), and a safety layer: bracketing with a non-clairvoyant evaluator plus targeted/adversarial DP audits for the compressed oracle.

### 1) What you are estimating (and the minimum correct labeling)

#### Notation

- `I`: information at bid time (here: our 7 tiles; no auction history).
- `D(I)`: all fully specified deals consistent with `I`.
- `M`: a completion model (distribution over `D(I)`).
- `c`: a contract (in this scope, primarily “fixed trump choice”; any extra contract variants are out of scope).
- `k`: bid level (points threshold to “make”).
- `V_PI(d, c)`: DP oracle value on completed deal `d` under contract `c` in the perfect-information game (0–42).
- `Make_PI(d, c, k) = 1[V_PI(d, c) ≥ k]`.

#### What Path A actually estimates

Given samples `d_i ~ M`:

- `P̂_upper,M(make | c, k) = (1/N) Σ_i Make_PI(d_i, c, k)`

This is **model-labeled** (depends on `M`) and typically an **upper / optimistic estimate** of real make probability, because `V_PI` assumes play optimized with knowledge of `d_i` that is not available at decision time (Frank & Basin, 1998).

So the minimum defensible output is not `P(make)` but something like:

- `P̂_upper,uniform_consistent(make | hand H, contract c, level k) ± CI`

with explicit tags: `{completion_model, evaluator_type=PI_upper_bound}`.

### 2) Valid vs invalid uses of the perfect-information oracle (Q1 + “marginalized training”)

#### Valid uses (sound within scope)

- **Value evaluator on a completed deal:** compute `V_PI(d,c)` for a fully specified deal `d` and fixed contract `c`. This is ground truth for that *perfect-information* instance.
- **Outcome summarizer over a completion model:** compute `E_M[V_PI(d,c)]`, `P_M(V_PI(d,c) ≥ k)`, and quantiles of `V_PI(d,c)` under explicitly stated `M`.
- **Arbitration/deferral:** when the fast approximator is uncertain or the decision is high leverage, fall back to DP on those cases.

#### Invalid uses (unsound for imperfect information)

- **Policy distillation from oracle actions:** training a bidder/player directly on oracle-recommended actions from completed deals leaks hidden information. The learned mapping is not conditioned on information sets and will reproduce “clairvoyant” move choices that are not implementable at test time. This is the core “marginalized training problem” you observed and is exactly what information-set theory warns about (Kuhn, 1953; Frank & Basin, 1998).

### 3) Completion weighting without opponent models (Q8)

Without a likelihood model for bidding behavior, completion weighting is a design choice that must be stated and (ideally) sensitivity-tested.

| Completion model | Definition | Assumptions | Failure mode | Use |
|---|---|---|---|---|
| **M1: Uniform over consistent deals** | `d ~ Uniform(D(I))` | No extra evidence beyond `I` | Miscalibrated if you later add auction info, selection bias, or non-random deal generation | Baseline for seat-0 opening |
| **M2: Worst-case / adversarial** | `min_{d∈D(I)} V(d,c)` (or approx via adversarial search) | Nature is hostile | Overly conservative; often “never bid” | Safety analysis only |
| **M3: Risk-sensitive tails** | summarize lower tail: `q_α`, `CVaR_α` under M1 | Decision-maker is risk averse | Can underbid if used alone | “Safety level” view |
| **M4: Prior-weighted by auction actions** | `P(d|bids)` | Requires behavior model (out of scope here) | “Smuggling” assumptions if done ad hoc | Post-auction only |
| **M5: Learned belief model** | NN estimates `P(tile∈hand)` | Requires data + careful calibration | Distribution shift; hard to audit | Play-phase, future work |

**Important nuance:** If deals are truly uniformly random and `I` is only your hand, then M1 is also the *correct conditional distribution* of the remaining tiles. The question still correctly insists you label it, because the moment you condition on *anything else* (auction, dataset selection, etc.) the distribution changes.

### 4) Strategy fusion / determinization pathologies and the Texas 42-specific fix (Q5–Q7)

#### 4.1 The core inequality and why “oracle averaging” is optimistic

Determinization evaluates each completion as if you can choose the best contingent line for that hidden world. This creates the classic gap:

- `E[ max (world-conditioned value) ] ≥ max ( E[value] )`

Frank & Basin (1998) describe this under **strategy fusion** and **non-locality** as central determinization pathologies.

In practice, the optimism is not constant: it is typically **largest near thresholds** (where small clairvoyant improvements flip make/set) and **smaller on “lock” hands** (where all reasonable policies win).

#### 4.2 Avoidable fusion in Texas 42: trump selection after the auction

Texas 42’s “choose trump after winning the bid” creates an extra, avoidable fusion channel if you let the evaluator choose trump per completion.

Do **not** compute:

```
Average(Max_trump V_PI(d, trump))      # wrong: Average(Max)
```

Instead compute, for each candidate trump `T`:

```
μ(T) = Average_d [ V_PI(d, T) ]
choose T* = argmax_T μ(T)              # right: Max(Average)
```

This ensures you compare *commitment strategies* (“if I choose Fives, what happens on average?”), rather than letting the oracle pick the “right” trump for each hidden world.

#### 4.3 How big is the remaining gap in trick-taking-like domains?

Long et al. (2010) study when perfect-information Monte Carlo sampling works well and emphasize structural predictors such as **leaf correlation** and **disambiguation factor**. In trick-taking-like settings (high leaf correlation; information revealed over time), determinization often performs surprisingly well, with reported average losses on the order of **tenths of a point per game** in some synthetic/analogous settings. This is consistent with why PIMC-style approaches have been competitive in games like Bridge and Skat (Ginsberg, 2001; Long et al., 2010), while failing in poker-like games where disambiguation is near zero (Frank & Basin, 1998).

For Texas 42, the safe stance is:

- expect **hand-dependent optimism**, largest near bid thresholds;
- treat PIMC results as actionable *decision support* but not literal probabilities;
- **measure the gap in-domain** via bracketing and stress tests rather than assuming a fixed correction factor.

#### 4.4 Techniques to estimate/bound/correct the gap (menu)

- **Decision-layer correction:** `Max(Average)` for trump selection (eliminates a pure artifact).
- **Consistent-action / information-set constraints:** force the same action across completions at indistinguishable decision points; conceptually related to information-set search and ISMCTS (Cowling et al., 2012).
- **Delayed perfect-information resolution (EPIMC-style ideas):** postpone perfect-information solving to deeper nodes; reduces early fusion at the cost of computation (as referenced in the syntheses; verify exact citations before relying on them).
- **Variance / stability diagnostics:** prefer actions/contracts whose value is stable across completions over “coups” that depend on precise hidden layouts.
- **Bracketing:** compute both an optimistic (PI oracle) and pessimistic (non-clairvoyant policy) estimate to expose the “clairvoyance gap” directly.

### 5) Baseline pipeline for contract × bid-level matrices (Path A)

This is the minimum defensible approach under your scope, producing **model-labeled optimistic estimates** with statistical uncertainty.

1. **Declare completion model `M`** (start with uniform-consistent for seat 0).
2. **Sample `N` completions** `d_1..d_N ~ M`.
   Practical ranges in analogous systems are typically tens to a few hundred samples; increase `N` until the CI is decision-useful.
3. **Evaluate each contract separately** (fixed trump) on the *same* completion set:
   compute `v_i,c = V_PI(d_i,c)` for all `c`.
4. **Convert to bid outcomes:** for each bid level `k`, `make_i,c,k = 1[v_i,c ≥ k]`.
5. **Aggregate + uncertainty:**
   - `P̂_upper,M(make | c,k) = mean_i make_i,c,k`
   - report a binomial CI (Wilson score is a good default near 0/1).
   - report score summaries: mean and quantiles (e.g., 5/50/95).
6. **Select contract with the criterion you intend to optimize:**
   - maximize `P̂(make|c,k)` for a fixed `k`, or
   - maximize expected value, or
   - maximize a risk-sensitive objective (e.g., `q05` or `CVaR_05`).

**Variance reduction tip:** reuse the same sampled completions across contracts (common random numbers) so contract comparisons have lower Monte Carlo noise.

### 6) Robust/conservative bounds and bracketing (Paths C and D)

Because `P̂_upper` can be optimistic, add at least one conservative lens.

#### Path C: Robust summaries under the same completion model

- **Tail quantile:** `q05(V | c)` as a “safety level.”
- **CVaR:** average of the worst α-fraction of outcomes to measure fragility.
- **Sample worst-case flag (stress test):** “did any sampled completion fail?”
  Useful as an alarm, but not a true worst-case bound over `D(I)`.

#### Path D: Pessimistic non-clairvoyant evaluator (bracketing)

Define a policy/evaluator `V_II(d,c)` that does *not* condition on hidden hands (rule-based, greedy, or a learned policy trained without privileged information). Then compute:

- upper bracket: `V_PI(d,c)`
- lower bracket: `V_II(d,c)`

Aggregate both under the same `M` to publish:

- `P̂_upper,M(make | c,k)` and `P̂_lower,M(make | c,k)`
- a gap metric, e.g. `Δ(c,k) = P̂_upper,M − P̂_lower,M`

This directly operationalizes the “don’t self-deceive” requirement: when `Δ` is large, the hand is fragile or requires clairvoyant lines, so a high `P̂_upper` should not be treated as a reliable make probability.

### 7) Using the compressed oracle safely (Path B) (Q10–Q12)

**Problem:** “97% argmax agreement” is not a guarantee of value accuracy, and bidding is thresholded (small value errors flip make/set around `k`). Errors that cluster by structure can evade random audits.

#### Safe integration pattern

1. **Use the compressed oracle for bulk scoring** across many completions/contracts.
2. **Identify decision-critical regions to audit with DP**, e.g.:
   - near threshold: `|E[v̂] − k|` small,
   - high variance across completions,
   - “locks” or near-locks (where a single failure completion is decisive),
   - large disagreement across seeds/ensembles (if available).
3. **Targeted DP audits** on those regions; do not rely on uniform random sampling.
4. **Selective deferral:** if the approximator is in an unvalidated region or low confidence, defer to DP or label “low confidence.”

#### Credible adversarial auditing schemes (not random) (Q11)

Two practical options that fit your setting (hand fixed; completion space large):

- **Adversarial completion search (“killer layouts”):** keep your hand fixed; perform local swaps of hidden tiles between other players to maximize a discrepancy objective, e.g. `|V̂_NN(d,c) − V_PI(d,c)|` or “classification flip” near `k`. This is a hill-climb / multi-start search version of adaptive stress testing.
- **Stratified stress tests:** deliberately oversample structurally risky completion families (e.g., partner has/doesn’t have key doubles; opponents concentrated in trump) and measure error rates separately.

#### Handling systematic (not random) approximator error (Q12)

- **Stratify and report error by structural features** (point margins, trump concentration, presence/absence of key tiles).
- **Calibrate or abstain:** if errors correlate with features, gate the approximator by those features.
- **Audit the tails:** the 1–3% of cases that matter most for “locks” and threshold bids are exactly where clustered errors can dominate decision quality.

### 8) Output design and confidence reporting (Q13–Q14)

For Texas 42 bidding, the most useful default is:

- **Primary metric:** `P̂(make|c,k)` with CI (model-labeled).
- **Risk metric:** `q05` (or `CVaR_05`) of points for `c`.
- **Bracketing (if available):** `[P̂_lower, P̂_upper]` and/or `[q05, mean, q95]`.
- **Diagnostic flags:**
  - high fragility: large robustness gap (`mean − q05`) or large bracket gap (`upper − lower`)
  - near-threshold: `|mean − k|` small
  - high fusion susceptibility: large dispersion of `V_PI(d,c)` across completions

A per-cell schema that meets the question’s “no implicit assumptions” constraint:

- `completion_model` (explicit id)
- `evaluator_type` (`PI_upper_bound` vs `II_lower_bound` etc.)
- `N`, seed
- `P̂_make` + CI method
- score quantiles
- tail metric(s)
- audit metadata if using a compressed oracle (DP-audited fraction, worst observed discrepancy near threshold)

### 9) When is PIMC “close enough” for Texas 42? (Q2 + Q5)

Given Texas 42’s trick-taking structure (information revealed each trick; finite branching; partnership), it plausibly sits in the regime where determinization is often useful in practice (as in Bridge/Skat-like domains), but it is not guaranteed. The correct operational answer is:

- **Use PIMC-style estimates as decision support,** especially for ranking contracts and identifying “obvious” bids.
- **Treat near-threshold bids as high-risk**: require tighter CIs, tail reporting, and/or bracketing.
- **Measure in-domain optimism** by comparing PI-upper estimates against a non-clairvoyant lower bracket and by running adversarial completion searches. If the gap is small for most hands and concentrated in rare edge cases, PIMC is a good pragmatic baseline for bidding evaluation.

---

## Acceptance Criteria Mapping (from `docs/research/question.md`)

| Criterion | How this synthesis satisfies it |
|---|---|
| Cite sources for hard claims; flag modeling choices | Citations to Frank & Basin (1998), Long et al. (2010), Ginsberg (2001), Kuhn (1953), Cowling et al. (2012); completion model `M` treated as explicit choice |
| Menu of completion models w/ assumptions/failure modes | Table in §3 |
| Baseline pipeline w/ uncertainty | Path A in §5 + CI guidance |
| Credible adversarial auditing scheme | §7: killer-layout search + stratified stress tests + deferral |
| Quantify/characterize fusion gap | §4.3: structural predictors + “order of magnitude” characterization + explicit “measure in-domain” guidance |
| Avoid rabbit holes | Focuses on evaluator-based bidding matrices; play-policy design only as bracketing/bias diagnosis |
| Evaluator vs policy-teacher distinction | §2 with information-set grounding |

---

## References (starting set; verify bibliographic details when implementing)

- Frank, I., & Basin, D. (1998). *Search in Games with Incomplete Information.* Artificial Intelligence. (Determinization pathologies: strategy fusion, non-locality.)
- Long, J. R., Sturtevant, N. R., Buro, M., & Furtak, T. (2010). *Understanding the Success of Perfect Information Monte Carlo Sampling in Game Tree Search.* AAAI. (Structural predictors of when PIMC works well.)
- Ginsberg, M. L. (2001). *GIB: Imperfect Information in a Computationally Challenging Game.* JAIR. (Bridge: sampling + search with double-dummy components.)
- Kuhn, H. W. (1953). *Extensive Games and the Problem of Information.* (Information sets.)
- Cowling, P. I., Powley, E. J., & Whitehouse, D. (2012). *Information Set Monte Carlo Tree Search.* IEEE TCIAIG. (Information-set search alternative to determinization.)
- Zinkevich, M., Johanson, M., Bowling, M., & Piccione, C. (2007). *Regret Minimization in Games with Incomplete Information.* NeurIPS. (CFR framework; oracle as evaluator at leaves.)
- Silver, D., & Veness, J. (2010). *Monte-Carlo Planning in Large POMDPs.* (POMCP; belief-state planning contrast to determinization.)
