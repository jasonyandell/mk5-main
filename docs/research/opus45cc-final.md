# Final Synthesis: Perfect-Information Oracles for Imperfect-Information Bidding

**Status:** Final synthesis integrating three deep research responses

**Sources evaluated:**
- `opus45cc-synthesis.md` (Claude Opus 4.5)
- `chatgpt52max-synthesis.md` (ChatGPT o3 5.2 Max)
- `gemini3cli-synthesis.md` (Gemini 3 CLI)

---

## Meta-Analysis: Evaluating the Three Syntheses

### Comparative Assessment

| Criterion | opus45cc | chatgpt52max | gemini3cli |
|-----------|----------|--------------|------------|
| **Empirical grounding** | ★★★★★ | ★★★★☆ | ★★★☆☆ |
| **Methodological rigor** | ★★★★☆ | ★★★★★ | ★★★☆☆ |
| **Practical actionability** | ★★★★★ | ★★★★☆ | ★★★★☆ |
| **Epistemic honesty** | ★★★★☆ | ★★★★★ | ★★★☆☆ |
| **Completeness** | ★★★★★ | ★★★★★ | ★★★☆☆ |
| **Conciseness** | ★★★☆☆ | ★★★☆☆ | ★★★★★ |

### Individual Evaluations

**opus45cc-synthesis.md** is the most empirically grounded, providing specific measurements from Long et al. (2010): ~0.1 points/game strategy fusion gap, ~0.063 tournament points/deal in Skat, ~15% of games affected. It offers the most complete coverage of the research questions and provides clear actionable steps. However, it may be slightly overconfident about the gap being "small enough"—the empirical measurements are from Skat and synthetic games, not Texas 42 specifically.

**chatgpt52max-synthesis.md** is the most methodologically rigorous, with the strongest emphasis on error decomposition (five distinct sources), explicit labeling requirements, and uncertainty quantification. Its output schema is the most complete. It's appropriately cautious, noting the gap is "hand-dependent and largest near thresholds" rather than claiming a fixed magnitude. However, it's verbose and lacks some of the specific empirical numbers.

**gemini3cli-synthesis.md** is the most concise and introduces useful terminology ("Trump Fit Hallucination," "Bead Architecture") and the SmartStack completion model concept not found in the others. However, it lacks depth in citations, error decomposition, and nuance around when PIMC fails.

### Points of High Confidence (All Three Agree)

These conclusions are robust—all three syntheses converge:

1. **Oracle as evaluator, not policy teacher.** Direct policy distillation from oracle actions leaks hidden information and fails under imperfect information.

2. **Max(Average) not Average(Max) for trump selection.** This is THE critical methodological fix. Evaluate each candidate trump across all completions, then compare averages. Never let the oracle choose different trumps for different completions.

3. **Strategy Fusion is real but manageable in trick-taking games.** The E[max] ≥ max[E] inequality creates systematic optimistic bias, but empirical evidence from analogous games (Bridge, Skat, Hearts) shows this is often small.

4. **Uniform sampling is the appropriate baseline for Seat 0.** With no auction history, all consistent deals are equally likely—there's no information to condition on.

5. **Explicit completion model labeling is mandatory.** Report `P_M(make)` not `P(make)`. The completion model is a design choice, not ground truth.

6. **Adversarial auditing over random sampling for the compressed oracle.** Random spot-checks confirm aggregate accuracy while missing clustered failures. Targeted testing is required.

7. **Bracketed estimates are essential.** Report both optimistic (oracle-derived) and pessimistic (heuristic-derived) bounds.

### Points of Moderate Confidence (Synthesis Required)

These require reconciling different framings or filling gaps:

1. **Magnitude of Strategy Fusion gap.** opus45cc claims ~0.1 points/game based on Long et al. chatgpt52max is more cautious, noting it's "hand-dependent." The truth: the gap depends on game structure parameters (leaf correlation, disambiguation factor), and Texas 42 likely falls in the favorable regime—but this should be empirically validated, not assumed.

2. **Sample size requirements.** opus45cc suggests 100-200 completions, citing GIB (50) and Kermit (saturation at 160). chatgpt52max recommends sequential sampling until CIs narrow. Both are valid—use sequential sampling but expect convergence around N=150.

3. **Completion models beyond uniform.** Only gemini3cli mentions "SmartStack" (constructing completions satisfying count/double thresholds). This is relevant for post-bidding scenarios with passes, but potentially smuggles assumptions. For Seat 0 opening, stick with uniform.

### Points of Lower Confidence (Gaps or Disagreements)

These need empirical validation for Texas 42:

1. **Quantitative Strategy Fusion gap for Texas 42 specifically.** No synthesis provides empirical measurements for 42 itself. The ~0.1 point figure is from Skat and synthetic games. Recommendation: measure this empirically before trusting it.

2. **Appropriate pessimistic evaluator for bracketing.** All three mention using a "heuristic" or "greedy" policy for lower bounds, but none specifies what this should be. We need to define and implement a concrete imperfect-information baseline.

3. **When PIMC fails catastrophically in 42.** The syntheses note PIMC fails in poker-like games, but don't identify specific 42 scenarios where it might fail. Are there hands where hidden information is uniquely critical?

---

## Consolidated Recommendations

### Core Methodology

#### The Central Fix: Postponed Trump Aggregation

All three syntheses agree this is critical. To avoid "Trump Fit Hallucination":

```
WRONG:  E_M[ max_trump V(d, trump) ]
        Oracle picks best trump per completion, then average

RIGHT:  max_trump E_M[ V(d, trump) ]
        Evaluate each trump across all completions, then pick best average
```

This is not an approximation—it's the correct formulation of the bidding question: "Which trump choice maximizes my expected score across all possible hidden deals?"

#### Valid Oracle Uses

| Use | Status | Rationale |
|-----|--------|-----------|
| Evaluate `V(d,c)` for fully specified deal | ✓ Valid | Ground truth for that PI game |
| Aggregate `E_M[V(d,c)]` over sampled deals | ✓ Valid | Optimistic upper bound |
| Compute `P_M(V(d,c) ≥ k)` | ✓ Valid | Upper bound on make probability |
| Audit compressed oracle on specific deals | ✓ Valid | Certification |
| Train policy to imitate oracle moves | ✗ Invalid | Information set mismatch |
| Let oracle choose trump per completion | ✗ Invalid | Strategy Fusion |

### Pipeline: Contract × Bid-Level Matrix

Synthesizing the best elements from all three:

#### Step 1: Sample Completions

- Generate N completions from `Uniform(D(I))` where I = our 7 dominoes
- Use common random numbers (same completions for all trump candidates)
- Start with N=100, extend via sequential sampling until CI width is acceptable
- Expect saturation around N=150-200

#### Step 2: Evaluate Each Trump Candidate Separately

For each candidate trump T ∈ {Blanks, Ones, Twos, Threes, Fours, Fives, Sixes, Doubles}:
- Fix trump BEFORE oracle evaluation (prevents Strategy Fusion)
- Compute `V_i = Oracle(completion_i, T)` for all i ∈ [1,N]
- Store full distribution of outcomes

#### Step 3: Aggregate with Uncertainty

For each (trump, bid_level) pair:

**Primary metrics:**
- `P̂_upper(make | T, k)` = fraction of completions where `V_i ≥ k`
- `CI` = Wilson score interval (handles near 0/1 better than normal)

**Risk metrics:**
- `μ` = mean(V_i) — expected score
- `q_5` = 5th percentile — "bad break" safety level
- `ΔR` = μ - q_5 — Robustness Gap

**Strategy stability:**
- If optimal opening lead varies across >30% of completions, flag "Strategy Fusion susceptible"

#### Step 4: Generate Lower Bracket (Pessimistic Bound)

Evaluate completions with an imperfect-information heuristic policy:
- Define a deterministic greedy/rule-based player that doesn't condition on hidden hands
- Compute `V_heur_i` for each completion
- `P̂_lower(make | T, k)` = fraction where `V_heur_i ≥ k`

**Gap metric:**
- `Δ_fusion = P̂_upper - P̂_lower`
- Large gap indicates high "clairvoyance value"—the oracle is exploiting hidden info heavily

#### Step 5: Audit Compressed Oracle (If Using)

For bulk evaluation with compressed oracle:
1. Identify high-leverage regions:
   - Near-threshold: |μ - k| small
   - High-variance: large std(V_i)
   - Strategy-unstable: action varies across completions
2. Run DP oracle on flagged cases
3. Implement selective deferral when confidence < threshold
4. Track and report audit coverage and max observed discrepancy

#### Step 6: Output with Full Labeling

Each matrix cell (trump, bid_level) reports:

```
{
  completion_model: "uniform_consistent_seat0",
  N: 150,
  P_upper: 0.73,
  P_upper_CI: [0.65, 0.80],  // Wilson score
  P_lower: 0.61,             // heuristic evaluator
  fusion_gap: 0.12,          // P_upper - P_lower
  score_mean: 34.2,
  score_q5: 28,              // 5th percentile
  score_q50: 35,             // median
  score_q95: 40,             // 95th percentile
  robustness_gap: 6.2,       // mean - q5
  strategy_stable: true,     // <30% lead variance
  audit_coverage: 0.15,      // if using compressed oracle
  flags: []                  // warnings if any
}
```

---

## Completion Models: When to Use What

| Model | Definition | When to Use | Failure Mode |
|-------|------------|-------------|--------------|
| **Uniform** | All consistent deals equally likely | Seat 0, no auction history | Ignores any structural priors |
| **Worst-case** | min over consistent deals | Safety certification only | Too conservative for decisions |
| **SmartStack** | Filter by count/double thresholds | Post-bidding with passes | Smuggles assumptions about "weak" |
| **Quantile/CVaR** | Focus on lower tail | Risk-averse decisions | Not a distribution, just summary |

**For this project (Seat 0 opening bidder):** Use Uniform as the primary model. Report worst-case/quantiles as supplementary risk metrics, not as the decision criterion.

---

## Adversarial Auditing Framework

The compressed oracle's ~97% argmax agreement is insufficient because:
1. Errors cluster in structural regions (not i.i.d.)
2. Bidding is thresholded—small value errors flip make/set
3. Random sampling confirms aggregate accuracy while missing targeted failures

### Recommended Approach: Adaptive Stress Testing (AST)

1. **Hold our hand fixed**
2. **Search completion space** by local moves (swap tiles between hidden hands)
3. **Maximize discrepancy**: |V̂_NN(d,c) - V_DP(d,c)| or "classification flip" near threshold
4. **Characterize failure regions** by structural features
5. **Implement selective deferral** in those regions

### Deferral Strategy

- Identify "Danger Zones": tight trump fits, near-threshold scores, high action variance
- If hand falls in Danger Zone and using compressed oracle, defer to DP
- The ~3% disagreement rate bounds maximum deferral frequency

---

## Error Decomposition

Understanding what can go wrong, and how to measure it (from chatgpt52max, the best treatment):

| Error Source | Type | Measurement |
|--------------|------|-------------|
| **Completion model** | Modeling choice | Sensitivity analysis across models |
| **Monte Carlo sampling** | Statistical | Confidence intervals, sequential stopping |
| **Strategy Fusion** | Method | Upper/lower bracket gap |
| **Approximator bias** | Compressed oracle | Targeted audits, stratified evaluation |
| **Solver correctness** | Implementation | Unit tests on known cases |

**Critical insight:** These errors compound but are measurable independently. Don't conflate them.

---

## Addressing Original Research Questions

### Literature and Analogous Domains (Q1-Q4)

**Consensus answer:** PIMC (sample → solve → aggregate) is the established method, used successfully in Bridge (GIB), Skat (Kermit), Hearts. Failures occur in poker-like games where information stays hidden until showdown.

**Key vocabulary:**
- Strategy Fusion — choosing different strategies for indistinguishable states (Frank & Basin 1998)
- Determinization pathologies — umbrella term (Long et al. 2010)
- Double-dummy analysis — Bridge term for PI evaluation
- Information set — states indistinguishable to a player (Kuhn 1953)

### Strategy Fusion Gap (Q5-Q7)

**Consensus answer:** In trick-taking games with high leaf correlation (~0.8-1.0) and moderate disambiguation factor (~0.6), the gap is small. Long et al. report ~0.1 points/game in Skat-like conditions.

**Confidence level:** Moderate. These numbers are from analogous games, not Texas 42. Empirical validation needed.

**Mitigation:** Max(Average) for trump selection; bracketing; flagging high-variance hands.

### Completion Modeling (Q8-Q9)

**Consensus answer:** Without a behavioral model, use Uniform over consistent deals. Label all outputs. Validate by plotting variance vs N (expect saturation ~150).

### Approximation and Auditing (Q10-Q12)

**Consensus answer:** Use compressed oracle for bulk evaluation; defer to DP in high-leverage regions. Adversarial (not random) testing targets failure modes. AST and selective prediction are established techniques.

### Output Design (Q13-Q14)

**Consensus answer:** Report P(make) with CI as primary; 5th percentile as safety bound; robustness gap as confidence indicator; bracketed estimates when gap is large. Use Wilson score intervals near 0/1.

---

## What Remains Unaddressed

The three syntheses leave these gaps:

1. **Empirical Strategy Fusion measurement for Texas 42.** All estimates extrapolate from Skat/Bridge. We should measure the actual gap on sampled 42 hands.

2. **Concrete pessimistic evaluator.** All mention "heuristic" or "greedy" policy for lower bounds, but none specifies the implementation. We need to define a deterministic baseline player.

3. **Doubles-as-suit handling.** The syntheses focus on numeric trump suits but don't address whether Doubles-as-suit has different Strategy Fusion characteristics.

4. **Lock detection.** All mention "locks" (guaranteed wins) but don't provide algorithms to detect them efficiently.

5. **Sensitivity to partner play quality.** Oracle assumes perfect partner play. Real partners may be imperfect. This could create a second "clairvoyance gap" beyond opponent information.

---

## Acceptance Criteria Verification

| Criterion | Status | Evidence |
|-----------|--------|----------|
| Sources cited for hard claims | ✓ | Frank & Basin (1998), Long et al. (2010), Ginsberg (2001), Cowling et al. (2012) |
| Completion models enumerated with assumptions/failures | ✓ | Four models with table |
| Baseline pipeline for contract × level matrices | ✓ | 6-step pipeline with output schema |
| Adversarial auditing scheme (not random) | ✓ | AST framework + deferral strategy |
| Strategy Fusion gap characterized | ✓ | ~0.1 points/game in analogous domains; needs 42-specific validation |
| Evaluator vs policy-teacher distinction | ✓ | Valid/invalid use table; explicit warning |
| Domain-appropriate (not rabbit holes) | ✓ | Texas 42 as testbed, methods domain-general |

---

## Recommended Next Steps

### Immediate (validation)

1. **Measure Strategy Fusion gap on 42 hands.** Sample 1000 hands, compute P̂_upper and P̂_lower (with a simple greedy player), measure average gap.

2. **Implement baseline greedy player.** Define rules: "lead highest trump," "follow with lowest," etc. This becomes the pessimistic evaluator.

3. **Run sample size saturation analysis.** Plot CI width vs N for representative hands. Confirm saturation around 150.

### Medium-term (pipeline)

4. **Build contract × level matrix generator.** Implement the 6-step pipeline with full output schema.

5. **Implement compressed oracle deferral.** Identify structural features that predict disagreement. Set threshold.

6. **AST probing.** Build local search over completions that maximizes oracle disagreement.

### Long-term (extensions)

7. **Post-bidding completion models.** When passes provide information, implement SmartStack or Bayesian updating.

8. **Play-phase evaluation.** Extend to mid-game decisions (different information set structure).

---

## Conclusion

The three research syntheses converge on a clear methodology: use the perfect-information oracle as an **outcome evaluator** over sampled completions, **fix trump before aggregation** to eliminate avoidable Strategy Fusion, **report labeled bracketed estimates** with confidence intervals, and **audit the compressed oracle adversarially** rather than randomly.

The Strategy Fusion gap is real but empirically manageable in trick-taking games with Texas 42's structural properties. PIMC-derived estimates are optimistic upper bounds, but actionable when properly qualified.

The critical warning: **never emit unlabeled probabilities**. Every P(make) must specify its completion model and acknowledge it represents oracle-optimal play, not achievable human performance.

The pipeline is ready for implementation. Empirical validation on Texas 42 specifically should precede deployment—trust the structural arguments, but verify the numbers.

---

## References (Consolidated)

- Frank, I., & Basin, D. (1998). Search in Games with Incomplete Information. *Artificial Intelligence*.
- Long, J. R., Sturtevant, N. R., Buro, M., & Furtak, T. (2010). Understanding the Success of Perfect Information Monte Carlo Sampling in Game Tree Search. *AAAI*.
- Ginsberg, M. L. (2001). GIB: Imperfect Information in a Computationally Challenging Game. *JAIR*.
- Cowling, P. I., Powley, E. J., & Whitehouse, D. (2012). Information Set Monte Carlo Tree Search. *IEEE TCIAIG*.
- Zinkevich, M., Johanson, M., Bowling, M., & Piccione, C. (2007). Regret Minimization in Games with Incomplete Information. *NeurIPS*.
- Kuhn, H. W. (1953). Extensive Games and the Problem of Information.
- Arjonilla, F. J., et al. (2024). Extended Perfect Information Monte Carlo. (EPIMC)
- Cazenave, T., & Ventos, V. (2021). αμ Search.
- Schmid, M., et al. (2023). Student of Games. *Science*.
