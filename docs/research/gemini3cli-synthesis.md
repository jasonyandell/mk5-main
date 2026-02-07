# Synthesis: Perfect-Information Oracles for Imperfect-Information Bidding in Texas 42

This document provides a research-grounded synthesis of how to correctly employ a perfect-information (PI) solver to evaluate imperfect-information bidding decisions in Texas 42. It addresses the fundamental pathologies of determinization, solves the "marginalized training" problem, and provides a pipeline for producing robust contract × bid-level matrices.

---

## 1. Thesis: From Oracle to Evaluator

The primary shift in methodology is moving the Perfect-Information Oracle from a **policy teacher** to an **outcome evaluator**. We avoid the "Deterministic Mirage" by using the oracle to evaluate fixed strategies across sampled worlds, rather than allowing the oracle to choose the strategy for us in each world.

---

## 2. Solving the Strategy Fusion Problem (The "Trump Fit Hallucination")

The central limitation of Perfect Information Monte Carlo (PIMC) is **Strategy Fusion**: the algorithm assumes it can choose different optimal actions (e.g., different trump suits) for different hidden worlds that are actually indistinguishable to the player.

### The Mitigation: Max-of-Averages (Postponed Reasoning)
To neutralize the bias where the oracle "fuses" strategies, we must invert the standard PIMC aggregation:

*   **Naive PIMC (Invalid):** $E[\max_{trump}(Oracle(deal, trump))]$. This allows the oracle to peek at the cards to pick the best trump for every sample, leading to "Trump Fit Hallucinations."
*   **Postponed PIMC (Validated):** $\max_{trump}(E[Oracle(deal, trump)])$. We force the evaluation of a *fixed* trump suit across all sampled worlds. The agent's "value" for a hand is the maximum expected value achievable by a single, robust trump choice.

**Empirical Characterization:**
In trick-taking games like Skat and Hearts, the strategy fusion gap is typically small (~0.1 points per game) because these games have high **leaf correlation** (terminal outcomes are similar across worlds) and a high **disambiguation factor** (information is revealed quickly). Texas 42 shares these "PIMC-friendly" traits, meaning that while PIMC results are optimistic upper bounds, they are actionable if the Trump selection is fixed during evaluation.

---

## 3. Solving the Marginalized Training Problem

Training a model to imitate oracle move recommendations (e.g., "lead the 6-4") is **invalid** for imperfect-information games. The oracle's moves are conditioned on hidden information (data leakage), causing the learner to fail when that information is removed.

**The Correct Paradigm:**
*   **Oracle as Evaluator:** Use the oracle to compute terminal values or point totals for completions.
*   **Policy from Reasoning:** The bidding policy should emerge from aggregating these values (marginalizing over uncertainty), not from imitating the oracle's specific move choices.

---

## 4. Completion Weighting: Enumerate and Label

Since opponent/partner behavior models are out of scope, we must use structural completion models. Any reported probability **must** be labeled with its model.

| Model | Assumption | Failure Mode |
| :--- | :--- | :--- |
| **Uniform Random** | All consistent deals are equally likely. | Ignores "negative info" (e.g., a "Pass" implies a weak hand). |
| **SmartStack (Count-Class)** | Hands are constructed to satisfy specific count/double thresholds. | "Smuggles" assumptions about what constitutes a "weak" hand. |
| **Minimax / Adversarial** | The hidden tiles are distributed in the worst possible way for the bidder. | Overly pessimistic; may prevent valid bids on "likely" wins. |

**Recommendation:** Use **Uniform Random over Consistent Deals** as the baseline for Seat 0. If history exists, use **SmartStack** to filter for hands that would have realistically passed.

---

## 5. Viable Pipeline: The "Bead" Architecture

To produce a **Contract × Bid-Level Matrix**, follow this 5-stage pipeline:

1.  **Sampling:** Generate $N$ (100–200) completions consistent with the observable hand.
2.  **Fixed-Trump Simulation:** For each candidate trump suit (Sixes, Fives, Doubles, etc.):
    *   Evaluate every sample with the DP Oracle, forcing the specific trump.
    *   Result: A distribution of outcomes for each *fixed* strategy.
3.  **Robust Aggregation:**
    *   **Mean ($\mu$):** The optimistic expected value.
    *   **Safety Level ($L$):** The 5th percentile outcome (captures "bad breaks").
    *   **Robustness Gap:** $\Delta R = \mu - L$.
4.  **Adversarial Audit:** For "lock" configurations or threshold decisions, search for "Killer Layouts" that break the contract using local hill-climbing on tile swaps.
5.  **Final Matrix:** Report the probability of making the bid level $k$ under the trump suit that maximized the average.

---

## 6. Adversarial Auditing for the Compressed Oracle

A 97% argmax agreement with the DP Oracle is insufficient for certification because errors cluster in high-leverage structural regions (e.g., context-conditioning failures).

*   **Adaptive Stress Testing (AST):** Instead of random checks, use a search algorithm to find hand configurations where the Compressed Oracle deviates from the DP Oracle.
*   **Selective Deferral:** Implement a confidence threshold (or identify "Danger Zones" like tight trump fits). If the hand is in a Danger Zone, **abstain** from the neural prediction and run the exact DP Oracle.

---

## 7. Output Design & Confidence Reporting

The system should never report a single "P(make)" without context. Recommended output:

1.  **Bracketed Estimate:** `[Robust_Min, Expected_Avg, Optimistic_Max]`
2.  **P(Make):** Fraction of worlds where the fixed-trump strategy succeeds.
3.  **Risk Flag:** High **Robustness Gap** ($\Delta R$) indicates the hand is "Fragile" and relies on partner fit or lucky breaks.
4.  **Completion Model Label:** (e.g., "Assumes Uniform Random consistent with Seat 0").

---

## 8. Vocabulary & Canonical References

*   **PIMC (Perfect Information Monte Carlo):** Sampling hidden states and solving each as a perfect-information game.
*   **Strategy Fusion:** The pathology of choosing different strategies for indistinguishable states. (*Frank & Basin, 1998*)
*   **Leaf Correlation:** Probability that sibling terminals have the same winner; high in 42, making it PIMC-friendly. (*Long et al., 2010*)
*   **Disambiguation Factor:** Rate at which hidden info is revealed; $\approx 0.6$ for 42.
*   **Robustness Gap:** The difference between average-case oracle results and realizable outcomes under uncertainty.
