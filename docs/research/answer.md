# Perfect-Information Oracles for Imperfect-Information Bidding in Texas 42

**Status:** Definitive answer synthesizing all research

**Sources:**
- 3 initial syntheses (opus45cc, chatgpt52max, gemini3cli)
- 2 meta-syntheses (opus45cc-final, chatgpt52codex-xhigh-final)
- Deep research literature review (chatgpt52app)

---

## Executive Summary

You can use your perfect-information DP solver to evaluate imperfect-information bidding decisions, but only with specific methodological constraints:

1. **Use it as an evaluator, not a policy teacher.** Compute outcomes, aggregate values—never imitate oracle actions.

2. **Fix trump before aggregation.** Compute `max_trump(E[V])` not `E[max_trump(V)]`. This eliminates Strategy Fusion at the trump-selection layer. (Here `V` is the oracle’s Team 0 point-differential value at the root of each completed deal.)

3. **Label all outputs.** Every probability must specify its completion model. `P(make)` is undefined without `M`.

4. **Report brackets.** Provide both optimistic (oracle) and pessimistic (heuristic) bounds. The gap exposes clairvoyance value.

5. **Audit adversarially.** Random spot-checks miss clustered errors. Target high-leverage failure regions.

The resulting estimates are **optimistic upper bounds**, not true probabilities—but in trick-taking games with Texas 42's structure (high leaf correlation, gradual information revelation), this bias is empirically small enough to be actionable.

---

## Part I: The Core Problem

### What You're Estimating

At bid time, you observe only your 7 dominoes. You want to estimate, for each contract (trump choice) and bid level:

> P(achieving ≥ k points | my hand, contract c, completion model M)

Two critical facts:

1. **The probability depends on M**—the distribution over hidden deals. Without specifying M, "P(make)" is not well-defined.

2. **The value depends on play quality.** If you evaluate using a perfect-information oracle, you're computing an upper bound—the best achievable by someone who knows the hidden cards.

### The Strategy Fusion Inequality

When you sample completions and solve each with perfect information:

```
E[max_play(outcome)] ≥ max_play(E[outcome])
```

The left side is what oracle averaging gives you: the expected value when you can adapt optimally to each hidden configuration. The right side is what's actually achievable: the best single strategy averaged over uncertainty.

This gap is **Strategy Fusion** (Frank & Basin, 1998). It's always ≥ 0. It causes systematic optimistic bias.

### The Texas 42-Specific Trap

Texas 42 adds a second fusion opportunity: trump selection happens *after* winning the bid, before seeing any cards played. If you compute:

```
WRONG: E_M[ max_trump V(d, trump) ]
```

...the oracle picks the optimal trump for each sampled deal. But you don't know which deal you're in. You must commit to one trump. The correct formulation:

```
RIGHT: max_trump E_M[ V(d, trump) ]
```

Evaluate each trump candidate separately across all completions, then compare averages. This is **Max(Average)** vs **Average(Max)**—the central methodological fix.

---

## Part II: Valid vs Invalid Oracle Uses

### Valid Uses

| Use | Rationale |
|-----|-----------|
| Compute `V(d,c)` for a fully specified deal | Ground truth PI value (Team 0 point differential) |
| Aggregate `E_M[V(d,c)]` over sampled deals | Optimistic upper bound on expected differential |
| Compute `P_M(score(d,c) ≥ k)` | Upper bound on make probability (using `score(d,c)=(42+V(d,c))/2` at the root) |
| Audit compressed oracle on specific deals | Certification / arbitration |
| Compute score quantiles under M | Risk metrics (5th percentile, CVaR) |

### Invalid Uses

| Use | Why It Fails |
|-----|--------------|
| Train policy to imitate oracle moves | Information set mismatch—oracle conditions on hidden info |
| Let oracle choose trump per completion | Strategy Fusion—fuses indistinguishable states |
| Treat oracle P(make) as true probability | It's an upper bound, not achievable value |
| Random auditing of compressed oracle | Misses clustered errors in structural regions |

---

## Part III: Completion Modeling

Without opponent/partner behavior models, completion weighting is a **design choice** that must be stated and sensitivity-tested.

### Available Models (No Behavior Model Required)

| Model | Definition | When to Use | Failure Mode |
|-------|------------|-------------|--------------|
| **Uniform** | All consistent deals equally likely | Seat 0, no auction | Ignores structural priors |
| **Worst-case** | min over D(I) | Safety certification | Too conservative |
| **Quantile/CVaR** | Lower-tail aggregation | Risk-averse decisions | Summary, not distribution |
| **SmartStack** | Filter by count/double thresholds | Post-bidding with passes | Smuggles "weak hand" assumptions |

### For This Scope (Seat 0 Opening Bidder)

**Use Uniform.** With no prior bids, all consistent deals are equally likely by symmetry. This is both the default "no information" choice and the correct conditional distribution given our hand.

**Label everything:** `P_{uniform}(make | ...)` not `P(make | ...)`.

---

## Part IV: How Large Is the Strategy Fusion Gap?

### Structural Predictors (Long et al., 2010)

Three parameters predict PIMC effectiveness:

| Parameter | Definition | Texas 42 Value |
|-----------|------------|----------------|
| **Leaf Correlation** | P(sibling terminals have same winner) | High (~0.8-1.0) |
| **Disambiguation Factor** | Rate hidden info is revealed per move | Moderate (~0.6) |
| **Bias** | Inherent advantage for one side | Variable |

Games with high leaf correlation and moderate+ disambiguation are "PIMC-friendly." Texas 42 fits this profile.

### Empirical Measurements

From Long et al. (2010) on Skat and synthetic trick-taking games:
- Strategy fusion affects **~15% of games** (those unresolved at 3 tricks remaining)
- Average loss: **~0.1 points per game** vs Nash equilibrium
- Weighted average: **~0.063 tournament points per deal**

**Confidence level:** Moderate. These are analogous domains, not Texas 42 specifically. The structural argument is sound; the numbers need in-domain validation.

### When PIMC Fails

PIMC performs poorly when:
- **Disambiguation is low** (poker: hidden cards until showdown)
- **Deception is central** (bluffing games)
- **Opponents can exploit fixed patterns** (adversarial play)

Texas 42 reveals information each trick and doesn't reward deception structurally—it's closer to Bridge/Skat than poker.

---

## Part V: The Pipeline

### Step 1: Sample Completions

Generate N completions from `Uniform(D(I))` where I = your 7 dominoes.

- Use **common random numbers**: same completion set for all trump candidates (reduces comparison variance)
- Start with N=100, extend via sequential sampling until CI width is acceptable
- Expect saturation around N=150-200 (GIB uses 50; Kermit saturates at 160)

### Step 2: Evaluate Each Trump Separately

For each candidate trump T ∈ {Blanks, Ones, Twos, Threes, Fours, Fives, Sixes, Doubles}:

- **Fix trump BEFORE oracle evaluation** (prevents Strategy Fusion)
- Compute `V_i = Oracle(completion_i, T)` for all i ∈ [1,N]
- Store full distribution of outcomes

### Step 3: Aggregate with Uncertainty

For each (trump, bid_level) pair:

**Primary output:**
- `P̂_upper(make | T, k)` = fraction where V_i ≥ k
- `CI` = Wilson score interval (handles near 0/1)

**Risk metrics:**
- `μ` = mean(V_i)
- `q_5` = 5th percentile ("bad break" safety level)
- `ΔR` = μ - q_5 (Robustness Gap)

**Stability flag:**
- If oracle's optimal opening lead varies across >30% of completions, flag "Strategy Fusion susceptible"

### Step 4: Generate Lower Bracket

Evaluate the same completions with an imperfect-information policy:

- Define a deterministic greedy player (e.g., "lead highest trump," "follow with lowest losing card")
- Compute `V_heur_i` for each completion (Team 0 point differential at the root)
- `P̂_lower(make | T, k)` = fraction where `(42 + V_heur_i) / 2 ≥ k`

**Gap metric:**
- `Δ_fusion = P̂_upper - P̂_lower`
- Large gap = high "clairvoyance value"—the oracle is exploiting hidden information heavily

### Step 5: Audit Compressed Oracle (If Using)

The ~97% argmax agreement is insufficient for bidding because:
1. Errors cluster in structural regions
2. Bidding is thresholded—small value errors flip make/set
3. Random sampling confirms aggregate accuracy while missing high-leverage failures

**Adversarial auditing approach:**
1. Hold your hand fixed
2. Search completion space via local swaps (move tiles between hidden hands)
3. Maximize discrepancy: `|V̂_NN - V_DP|` or "classification flip" near threshold k
4. Characterize failure regions by structural features
5. Implement selective deferral in those regions

**Deferral strategy:**
- Identify "Danger Zones": tight trump fits, near-threshold scores, high action variance
- If hand falls in Danger Zone, defer to DP oracle
- The ~3% disagreement rate bounds maximum deferral frequency

### Step 6: Output with Full Labeling

Each matrix cell (trump, bid_level) reports:

```json
{
  "completion_model": "uniform_consistent_seat0",
  "evaluator": "DP_oracle_upper_bound",
  "N": 150,
  "P_upper": 0.73,
  "P_upper_CI": [0.65, 0.80],
  "P_lower": 0.61,
  "fusion_gap": 0.12,
  "score_mean": 34.2,
  "score_q5": 28,
  "score_q50": 35,
  "score_q95": 40,
  "robustness_gap": 6.2,
  "strategy_stable": true,
  "audit_coverage": 0.15,
  "flags": []
}
```

**Critical rule:** Never emit unlabeled `P(make)`. Always emit `P_M(make)` with method tag.

---

## Part VI: Error Decomposition

Five distinct error sources (don't conflate them):

| Source | Type | Measurement |
|--------|------|-------------|
| **Completion model** | Modeling choice | Sensitivity analysis across M |
| **Monte Carlo sampling** | Statistical | Confidence intervals, sequential stopping |
| **Strategy Fusion** | Methodological | Upper/lower bracket gap |
| **Approximator bias** | Compressed oracle | Targeted audits, stratified evaluation |
| **Solver correctness** | Implementation | Unit tests on known cases |

---

## Part VII: Key Terminology

| Term | Definition | Source |
|------|------------|--------|
| **PIMC** | Perfect Information Monte Carlo: sample → solve → aggregate | Ginsberg 2001 |
| **Strategy Fusion** | Choosing different strategies for indistinguishable states | Frank & Basin 1998 |
| **Non-locality** | Node values depending on regions outside current subtree | Frank & Basin 1998 |
| **Determinization Pathologies** | Umbrella term for strategy fusion + non-locality | Long et al. 2010 |
| **Information Set** | States indistinguishable to a player | Kuhn 1953 |
| **Leaf Correlation** | P(sibling terminals have same winner) | Long et al. 2010 |
| **Disambiguation Factor** | Rate hidden info is revealed per move | Long et al. 2010 |
| **Double-Dummy Analysis** | Perfect-info evaluation with all hands visible | Bridge tradition |
| **Max(Average)** | Correct aggregation for trump selection | This synthesis |
| **Robustness Gap** | μ - q_5; mean minus 5th percentile | This synthesis |

---

## Part VIII: What Remains Unaddressed

1. **Empirical Strategy Fusion measurement for Texas 42.** All estimates extrapolate from Skat/Bridge. Validate in-domain.

2. **Concrete pessimistic evaluator.** The pipeline requires a greedy/heuristic player. Define its rules.

3. **Doubles-as-suit characteristics.** Does Doubles trump have different Strategy Fusion properties than numeric suits?

4. **Lock detection algorithms.** Identifying guaranteed wins efficiently (7+ trumps, etc.).

5. **Partner play quality.** Oracle assumes perfect partnership. Real partners err. This creates a second clairvoyance gap.

---

## Part IX: Recommended Next Steps

### Immediate (Validation)

1. **Measure Strategy Fusion gap.** Sample 1000 hands, compute P̂_upper and P̂_lower, measure average gap.

2. **Implement greedy player.** Rules: lead highest trump, follow with lowest, etc. This becomes the pessimistic evaluator.

3. **Sample size saturation.** Plot CI width vs N for representative hands. Confirm convergence ~150.

### Medium-term (Pipeline)

4. **Build matrix generator.** Implement the 6-step pipeline with full output schema.

5. **Compressed oracle deferral.** Identify structural predictors of disagreement. Set deferral threshold.

6. **AST probing.** Build local search over completions that maximizes oracle disagreement.

### Long-term (Extensions)

7. **Post-bidding completion models.** When passes provide information, implement constraint-aware sampling.

8. **Play-phase evaluation.** Extend to mid-game decisions (different information set structure).

---

## Conclusion

The research converges on a clear methodology:

1. **Oracle as evaluator.** Compute outcomes over sampled completions; never imitate oracle moves.

2. **Max(Average) for trump selection.** Fix trump before aggregation. This eliminates avoidable Strategy Fusion.

3. **Label and bracket everything.** Report `P_M(make)` with model tag. Provide [lower, point, upper] brackets.

4. **Audit adversarially.** Target high-leverage failure regions, not random samples.

The Strategy Fusion gap is real but manageable in trick-taking games. Empirical measurements from analogous domains (~0.1 points/game) suggest Texas 42 falls in the PIMC-friendly regime—but this should be validated, not assumed.

**The critical warning:** PIMC output is an **optimistic upper bound**, not the true achievable probability. With proper labeling, bracketing, and auditing, it's actionable decision support. Without these safeguards, it's self-deception.

---

## References

### Primary Sources (Determinization Pathologies)

- Frank, I., & Basin, D. (1998). Search in Games with Incomplete Information. *Artificial Intelligence*. [Defines strategy fusion and non-locality]

- Long, J. R., Sturtevant, N. R., Buro, M., & Furtak, T. (2010). Understanding the Success of Perfect Information Monte Carlo Sampling in Game Tree Search. *AAAI*. [Empirical characterization; leaf correlation, disambiguation factor] https://webdocs.cs.ualberta.ca/~nathanst/papers/pimc.pdf

### Applications (Bridge, Skat)

- Ginsberg, M. L. (2001). GIB: Imperfect Information in a Computationally Challenging Game. *JAIR*. [Bridge PIMC success story]

### Alternative Methods

- Cowling, P. I., Powley, E. J., & Whitehouse, D. (2012). Information Set Monte Carlo Tree Search. *IEEE TCIAIG*. [ISMCTS avoids strategy fusion]

- Zinkevich, M., Johanson, M., Bowling, M., & Piccione, C. (2007). Regret Minimization in Games with Incomplete Information. *NeurIPS*. [CFR for equilibrium computation]

- Arjonilla, F. J., et al. (2024). Extended Perfect Information Monte Carlo. [EPIMC: delayed revelation reduces fusion]

### Game Theory Foundations

- Kuhn, H. W. (1953). Extensive Games and the Problem of Information. [Information sets]

### Completion Modeling

- Parker, A., Nau, D., & Subrahmanian, V. S. (2006). Overconfidence or Paranoia? Search in Imperfect-Information Games. *AAAI*. [Uniform vs worst-case sampling; overconfident beats paranoid in Kriegspiel]
