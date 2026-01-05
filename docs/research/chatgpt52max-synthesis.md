# Perfect-Information Oracles for Imperfect-Information Bidding in Texas 42 — Synthesis

**Status:** Research synthesis (answer to `docs/research/question.md`)

**Scope:** Bidding evaluation only; seat 0 opening bidder; no opponent/partner bidding behavior model; no play-phase policy design beyond what is needed to interpret oracle-based estimates.

**Core warning:** Any “sample completions + perfect-information solve + aggregate” result is a *model-labeled, generally optimistic* estimate because it evaluates play with information you do not have (determinization / strategy fusion pathologies).

---

## Executive Summary

You can use an exact perfect-information DP solver (“double-dummy” style oracle) to support imperfect-information bidding, but only if you:

1. **Treat it as an evaluator, not a policy teacher.** Distilling actions from a perfect-information oracle into an imperfect-information policy is fundamentally mismatched to information sets.
2. **Make the completion model explicit and label outputs.** “P(make)” is undefined without a distribution over hidden deals.
3. **Eliminate *avoidable* strategy fusion at the decision layer.** In Texas 42, do *not* let the evaluator “pick the right trump for each completion.” Evaluate each contract separately across completions, then compare.
4. **Report uncertainty and robustness, not just a point estimate.** Provide confidence intervals (sampling error), quantiles / CVaR-style tail summaries (risk), and at least one pessimistic bracket (lower bound) to expose “clairvoyance gap.”
5. **Use the compressed oracle only with targeted DP auditing and deferral.** High argmax agreement does not certify value accuracy, and clustered errors require adversarial/structured testing, not random spot checks.

---

## 1) Problem Restatement (What You’re Actually Estimating)

Observed state at the bid: **only our 7 dominoes** (seat 0 opening; no auction history). Hidden state: the remaining 21 dominoes partitioned into three 7-tile hands.

You want a **contract × bid-level matrix** where each cell reports something like:

- `P(make | our hand, contract c, bid level k, completion model M)` and/or a distribution of achievable points.

Two critical clarifications:

1. **The probability depends on the completion model** `M` (a distribution over hidden deals consistent with what is observed).
2. **The value depends on the play policy class.** If you evaluate each completion with a perfect-information oracle, you are estimating an *optimistic bound* that assumes play conditioned on the full deal.

This is the determinization / strategy fusion issue formalized in early critiques of determinization in imperfect-information games (Frank & Basin, 1998).

---

## 2) Taxonomy: Valid vs. Invalid Uses of a Perfect-Information Oracle

### Valid (sound) uses

- **Leaf/value evaluator:** Given a fully specified deal `d` and contract `c`, compute the exact game-theoretic value under perfect information: `V_PI(d, c)`. This is ground truth for that *perfect-information* game instance.
- **Upper-bound feature extractor:** Use `V_PI(d, c)` over sampled deals to produce *optimistic* summaries: `E_M[V_PI(d, c)]`, `P_M(V_PI(d, c) ≥ k)`, quantiles of `V_PI`.
- **Certification / arbitration:** When a fast approximator is uncertain or high leverage, defer to the exact oracle for those cases.

### Invalid (unsound) uses

- **Policy distillation from oracle actions:** Training a bidder/player directly on oracle-recommended actions for completed deals leaks hidden information into the target and learns a mapping that cannot be conditioned correctly at test time (information set mismatch). This is exactly why “oracle as policy teacher” fails in imperfect-information settings.

Conceptually, the “unit of decision” in imperfect-information game theory is the **information set** (Kuhn, 1953). Any method that implicitly chooses different actions for indistinguishable states is committing strategy fusion (Frank & Basin, 1998).

---

## 3) Completion Weighting Without an Opponent/Partner Behavior Model

Because auction inference is out of scope, you do not have a likelihood model for bids/passes. Therefore, completion weighting must be stated as a **design choice**, and all reported probabilities must be labeled with it.

Let `D(I)` be the set of deals consistent with information `I` (here, `I = our hand`).

### Model M1: Uniform over consistent deals (baseline)

- **Definition:** Sample `d ~ Uniform(D(I))`.
- **Assumption:** With no other evidence, treat all consistent hidden deals as equally likely.
- **Failure mode:** If the real deal distribution is biased (non-uniform dealing protocol; non-random hand construction in data; later, auction constraints), estimates will be miscalibrated.
- **When appropriate:** Exactly your stated scope (seat 0 opening, no auction history).

### Model M2: Worst-case / adversarial completions (robust bound)

- **Definition:** Evaluate the **minimum** over `D(I)` (or an approximation via minimum over a sampled subset): `min_d V(d, c)` or `min_d 1[V(d, c) ≥ k]`.
- **Assumption:** Nature (or a hostile sampler) selects the most harmful completion consistent with what you know.
- **Failure mode:** Often overly conservative; may recommend never bidding.
- **Use:** Safety certification and “how bad can it get?” diagnostics, not as the only decision criterion.

### Model M3: Risk-sensitive tails (quantiles / CVaR-style)

Instead of only using means or raw success probability, summarize the **lower tail** of the score distribution:

- `q_α(c) = α-quantile of V(d, c)` under `M`
- `CVaR_α(c) = E[V(d, c) | V(d, c) ≤ q_α(c)]` under `M`

These are standard risk measures in decision-making: they are not “new inference,” they are **different aggregations** of the same sampled distribution.

### Practical requirement (from your prompt)

Every output must include the completion model label:

> `P_M(make | ...)` not `P(make | ...)`.

---

## 4) Strategy Fusion / Determinization Pathologies: Where the Bias Enters

### 4.1 The core inequality (why PIMC is optimistic)

Determinization evaluates each completion as if you can act optimally *given that completion*.

This creates the classic gap:

- **Clairvoyant aggregation (what oracle averaging approximates):** `E_M[ max_{clairvoyant play} payoff ]`
- **Achievable under uncertainty:** `max_{information-set-consistent play} E_M[ payoff ]`

And in general: `E[max] ≥ max[E]` (Frank & Basin, 1998). Intuitively: averaging over worlds where you “know the answer” gives you credit for adapting to hidden state.

### 4.2 Texas 42-specific “avoidable fusion”: trump/contract choice

Your bidding output is “contract × bid level.” The decision is to pick a single contract `c` (trump choice) *without* seeing the completion.

So you must not compute:

`E_M[ max_c V_PI(d, c) ]`  (this fuses different contracts across completions)

Instead compute per-contract summaries:

`for each c:  E_M[ V_PI(d, c) ]` and/or `P_M(V_PI(d, c) ≥ k)`  
then choose:

`c* = argmax_c E_M[V_PI(d, c)]` (or the risk-adjusted criterion you prefer).

This “Max(Average), not Average(Max)” correction removes a *purely methodological* fusion artifact at the contract-selection layer. It does **not** eliminate within-play fusion (the oracle still plays with full information), but it prevents the evaluator from hallucinating that you can pick a different trump for each hidden world.

### 4.3 When is PIMC “good enough” in analogous domains?

The literature documents both success and failure:

- **Success cases:** Trick-taking card games have often tolerated determinization surprisingly well in practice (e.g., Bridge programs like GIB used sampling + perfect-information search as a core component; Ginsberg, 2001).
- **Failure cases:** Games where hidden information remains hidden until the end (poker-like) or where deception/non-locality dominates are poor fits for naive determinization (Frank & Basin, 1998).

Long et al. (2010) provide an empirical and theoretical characterization of when perfect-information Monte Carlo sampling succeeds, emphasizing game-structural properties (e.g., the extent to which outcomes across possible worlds are correlated and the rate at which information is revealed). The practical takeaway for Texas 42 is not “bias is small,” but:

- **You should expect the gap to be hand-dependent and largest near thresholds.**
- **You should measure and expose it** via bracketing and targeted evaluation, not assume it away.

Sources: Frank & Basin (1998); Long et al. (2010); Ginsberg (2001).

---

## 5) Baseline Pipeline (Path A) for Contract × Bid-Level Matrices

This pipeline produces a **model-labeled optimistic estimate** plus uncertainty, which is the minimum defensible baseline under your scope constraints.

### Step 1: Choose and record completion model `M`

For seat-0 opening with no auction info, start with:

- `M = Uniform(D(I))` where `I = our hand`.

Record the label in outputs.

### Step 2: Sample `N` completed deals

Sample completions `d_1, …, d_N ~ M`.

If you want a general-purpose estimator for many contracts/bid levels, use **one shared sample set** per hand and reuse it across contracts to reduce variance in comparisons (common random numbers).

### Step 3: Evaluate each completion with the DP oracle

For each completion `d_i` and each contract `c`:

- compute `v_i,c = V_PI(d_i, c)` (0–42 points).

### Step 4: Convert oracle values into bidding outcomes

For each bid level `k`:

- define `make_i,c,k = 1[v_i,c ≥ k]`.

Then estimate:

- `P̂_upper,M(make | c, k) = (1/N) Σ_i make_i,c,k`

This is an **upper bound / optimistic estimate** because it uses perfect-information play on each completion.

### Step 5: Quantify sampling uncertainty

For `P̂` (a binomial proportion), report a confidence interval using a well-behaved method (e.g., Wilson score interval) rather than a normal approximation, especially near 0 or 1.

Also report basic score distribution summaries per contract:

- mean of `v_i,c`
- quantiles (e.g., 10/50/90)
- a lower-tail metric (e.g., 5th percentile) as a “safety level” proxy under model `M`.

### Step 6: Sequential sampling (optional but recommended)

Instead of fixing `N`, sample until:

- confidence intervals are sufficiently narrow for the decision, or
- the ranking between top contracts/bid levels is stable.

This is standard Monte Carlo practice and directly addresses “variance reduction / CI” needs.

---

## 6) Robust/Conservative Bounds and Bracketing (Paths C and D)

Because `P̂_upper,M` can be materially optimistic, you need at least one conservative lens to satisfy the “don’t self-deceive” requirement.

### 6.1 Worst-case over sampled completions (cheap robust proxy)

For each `(c, k)`:

- `P̂_worst,S(make | c, k) = min_i make_i,c,k` over your sampled set.

This is not a true worst-case over `D(I)`, but it is a **stress-test bound** that answers: “Did we see any sampled deal where this fails?”

More informative for scores:

- `q_α(v | c)` for small `α` (e.g., 5%) under `M`.

### 6.2 Bracketing with a pessimistic imperfect-information evaluator

To explicitly expose the strategy-fusion gap, compute a lower bracket using an evaluator that **does not condition on hidden hands** (or conditions only on what would be observable during play).

One defensible bracketing pattern is:

- **Upper bracket:** `V_PI(d, c)` (oracle; clairvoyant)
- **Lower bracket:** `V_heur(d, c)` where `heur` is a fixed, imperfect-information policy (greedy, rule-based, or a learned policy trained without privileged information)

Aggregate both across completions under the same `M`:

- `P̂_upper,M(make | c,k)` and `P̂_lower,M(make | c,k)`

Then define a diagnostic gap:

- `Δ(c,k) = P̂_upper,M − P̂_lower,M` (or the analogous score-gap in points).

This “gap” is not a single standardized metric name across all fields, but it corresponds closely to:

- determinization optimism / strategy fusion vulnerability (Frank & Basin, 1998)
- “value of information” style reasoning (difference between clairvoyant and non-clairvoyant performance)
- robust optimization’s “price of robustness” framing (nominal vs robust objective).

The key is not the label; it is **publishing the bracket** so decisions can be made with eyes open.

---

## 7) Compressed Oracle + Selective DP Audit (Path B)

Your compressed oracle is an accelerator, not ground truth. “97% argmax agreement” is insufficient for bidding certification because:

- bidding is thresholded (small value errors flip make/set near `k`)
- errors can be clustered (not i.i.d. noise)
- sampling hides tail failures unless you target them

### 7.1 Safe integration pattern

1. **Use the compressed oracle for bulk scoring** over many sampled completions/contracts.
2. **Identify high-leverage cells / situations** that must be audited:
   - near-threshold: `|E[v] − k|` small
   - high-variance: large dispersion of `v_i,c`
   - high stakes: aggressive bid levels where a small `P(make)` misestimate changes the decision
3. **Targeted DP audits** on those cases, not uniform random audits.
4. **Deferral / abstention rule:** if the compressed oracle is uncertain in a region you have not validated, defer to DP or label as low confidence.

### 7.2 What “adversarial auditing” means here (without assuming a bidding model)

Even with no auction inference, you can stress-test the approximator by searching for *completions consistent with our hand* that maximize its error.

A generic, source-aligned approach (in spirit of red-teaming / stress testing) is:

- hold our hand fixed
- search the space of completions by local moves (e.g., swap hidden tiles between opponents/partner)
- maximize a discrepancy objective like `|V̂_NN(d,c) − V_PI(d,c)|` or “classification flip” near a bid threshold

This targets clustered failure modes far more effectively than random spot checks.

---

## 8) Error Decomposition (What Can Go Wrong, and How to Measure It)

Your final matrix has multiple error sources; treating them separately prevents “accuracy theater.”

1. **Completion-model error (modeling):** `M` is a choice.  
   - Measure via sensitivity analysis across plausible `M` (uniform vs stress-test mixtures; risk-averse summaries).
2. **Monte Carlo sampling error (statistics):** finite `N`.  
   - Measure via confidence intervals and sequential stopping.
3. **Strategy fusion bias (method):** `P̂_upper,M` assumes clairvoyant play.  
   - Measure/expose via bracketing (`upper` vs `lower`) and via fusion susceptibility diagnostics (e.g., large dispersion / action instability across completions).
4. **Approximation error (compressed oracle):** `V̂_NN` deviates from `V_PI`.  
   - Measure via targeted audits stratified by decision-critical regions (near thresholds, structural clusters).
5. **Implementation/solver correctness:** DP oracle is ground truth only if correct.  
   - Measure via unit tests on small subgames, consistency checks, and cross-validation against known locks/edge cases.

---

## 9) Recommended Output Schema (Per Contract × Bid-Level Cell)

To meet your acceptance criteria and avoid implicit assumptions, each matrix cell `(c,k)` should report:

- `completion_model`: explicit identifier (e.g., `uniform_consistent`)
- `N`: sample size and (optionally) RNG seed
- `P̂_upper`: `P̂_upper,M(make | c,k)` with CI
- `score_quantiles`: e.g., q10/q50/q90 of `V_PI(d,c)` under `M`
- `tail_metric`: e.g., q05 or CVaR-style lower-tail mean
- `lower_bracket` (if available): pessimistic evaluator estimate + CI
- `gap`: bracket difference (diagnostic), plus a warning flag if large
- `audit_info` (if using compressed oracle): DP-audited fraction and worst observed discrepancy near the threshold

Critically: never emit an unlabeled `P(make)`; always emit `P_M(make)` plus the method tag (upper-bound/oracle vs lower-bound/heuristic).

---

## References (starting set; verify exact bibliographic details as you implement)

- Frank, I., & Basin, D. (1998). *Search in Games with Incomplete Information.* Artificial Intelligence. (Introduces/clarifies strategy fusion and non-locality critiques of determinization.)
- Long, J. R., Sturtevant, N. R., Buro, M., & Furtak, T. (2010). *Understanding the Success of Perfect Information Monte Carlo Sampling in Game Tree Search.* AAAI. https://webdocs.cs.ualberta.ca/~nathanst/papers/pimc.pdf
- Ginsberg, M. L. (2001). *GIB: Imperfect Information in a Computationally Challenging Game.* Journal of Artificial Intelligence Research. (Bridge system using sampling + search.)
- Cowling, P. I., Powley, E. J., & Whitehouse, D. (2012). *Information Set Monte Carlo Tree Search.* IEEE Transactions on Computational Intelligence and AI in Games. (ISMCTS.)
- Zinkevich, M., Johanson, M., Bowling, M., & Piccione, C. (2007). *Regret Minimization in Games with Incomplete Information.* NeurIPS. (CFR.)
- Kuhn, H. W. (1953). *Extensive Games and the Problem of Information.* (Formalizes information sets.)
- Silver, D., & Veness, J. (2010). *Monte-Carlo Planning in Large POMDPs.* (POMCP; belief-state planning alternative to determinization.)
- Schmid, M., et al. (2023). *Student of Games.* Science. (Modern decomposition/value-based approaches for imperfect-information games; emphasizes values vs direct policy imitation in the presence of hidden information.)

