# Bead: Perfect-Information Oracles for Imperfect-Information Bidding in Texas 42



**Status:** Research bead (deep literature synthesis prompt)

**Scope:** Bidding evaluation only. Play-phase decisions are future work.

**Primary risks:** (1) Strategy Fusion bias—treating PIMC upper bounds as true probabilities. (2) Smuggling unstated assumptions about completion weighting. (3) Trusting compressed oracle without adversarial auditing.



---



## Thesis



We have an exact perfect-information solver for fully specified deals. We want to use it correctly to evaluate imperfect-information bidding decisions—without solving the wrong problem through hidden assumptions or invalid aggregation.



---



## Scope



### In scope

- **Single-hand bidding evaluation**: given our 7 dominoes, estimate success probability for each contract (trump choice) at each bid level

- **Seat 0, opening bidder**: no prior bids to condition on

- **Partnership fixed by rules**: seat 2 is always partner



### Out of scope (explicitly deferred)

- Opponent/partner bidding behavior models (no auction inference)

- Play-phase decisions under hidden information (voids, signaling, trick history)

- Variant contracts (nello, doubles-as-suit)

- Implementation details (GPU, PyTorch, performance)



---



## Definitions



**Fully specified deal:** A concrete assignment of all 28 dominoes to 4 players (7 each).



**Perfect-information oracle (DP solver):** Given a fully specified deal and a contract, computes the exact achievable points for Team0 under perfect play. Correct by construction for that deal. Output is points (0–42).



**"Make" a bid:** For bid level k, a contract makes on a completed deal if Team0's oracle value ≥ k.



**Compressed oracle:** A neural network trained to approximate DP oracle outputs. Fast but imperfect. Not ground truth.



---



## What We Have



### 1. DP Oracle (Ground Truth)



Enumerates reachable states, computes minimax values via backward induction. Provides exact V(s) and Q(s,a) for any fully specified deal.



**Limitations:** Computationally expensive. Cannot be run on every completion for every decision.



### 2. Compressed Oracle (Heuristic Accelerator)



~97% argmax agreement with DP oracle on held-out positions.



**What this means:** On 97% of positions, the model's top move matches the oracle's.



**What this does not mean:**

- Value estimates are accurate

- Errors are random rather than systematic

- The 3% errors are uniformly distributed across position types



**Known failure mode:** Systematic context-conditioning errors—the model learns patterns (e.g., "this is a good lead") without properly conditioning on contract. This is not noise; it's systematic failure to contextualize. Errors cluster in specific structural situations, which motivates adversarial auditing over random sampling.



**Empirical observation:** Hands that are theoretical locks (100% win) sometimes evaluate to ~99% when assessed via compressed oracle over sampled completions. The DP oracle confirms these are true locks. The error is in the compressed oracle's evaluation of specific completions.



**Role:** Triage and exploration. Rough comparisons. Not certification. Not trustworthy on tail events (locks, threshold decisions) without DP audit.



### 3. Preliminary Bidding Artifact



Contract × bid-level heatmaps of estimated P(make) using compressed oracle as evaluator.



**Status:** Demonstrates that oracle-driven aggregation can extract interpretable structure. Does not demonstrate correctness. Known to understate guarantees.



---



## Critical Discovery: The Marginalized Training Problem



We trained a model directly on DP oracle move recommendations. High accuracy on held-out positions. Subtle but fatal failure mode.



**The problem:** Oracle moves are conditioned on the actual hidden hands. "Play 6-4" means "6-4 is optimal against *these specific* opponent holdings." The model receives only observable state but learns to imitate these recommendations.



**The consequence:** The model learned locally optimal responses to specific hidden configurations, not robust responses to the distribution of possible configurations. This is data leakage—perfect-information ground truth poisoning imperfect-information learning.



**The key distinction:**

- Oracle as **evaluator**: "Given this completed deal, what's the value?" → Valid use

- Oracle as **policy teacher**: "Learn to play like the oracle" → Invalid for imperfect-information decisions



This means direct policy distillation from oracle outputs is suspect. Decision guidance must come from aggregating evaluations over completions, not from imitating oracle moves.



---



## Critical Limitation: The Strategy Fusion Problem



Even valid aggregation over completions has a fundamental bias.



**The problem:** When you average oracle results across completions, you compute E[max(score)]—the expected value when you play optimally against each specific hidden configuration. But in actual play, you don't know which configuration you're in. Real play requires max[E(score)]—the best strategy given uncertainty about hidden hands.



**The math:** E[max] ≥ max[E]. Always. Averaging perfect-information outcomes is an *upper bound* on achievable value, not the true value.



**The mechanism:** In each sampled completion, the oracle knows exactly where dangerous dominoes are. It never takes a losing finesse. It never plays for a split that doesn't exist. When you average these omniscient results, you get credit for navigating hazards that are actually invisible to you.



**The consequence:** Path A (sample + evaluate + aggregate) will systematically overestimate success probability. If this approach says "95% make," reality may be 80%. The gap depends on how much hidden information affects optimal play.



**What this means for outputs:**

- Results from oracle averaging are **optimistic bounds**, not true probabilities

- The gap between E[max] and max[E] must be acknowledged, and ideally quantified

- This is a known phenomenon in game AI literature (PIMC / Perfect Information Monte Carlo / "determinization pathologies")



**What research must provide:**

- How large is this gap in analogous domains (bridge, poker)?

- What techniques exist to estimate or bound the gap?

- When is PIMC "good enough" despite the bias?



---



## The Central Unknown: Completion Weighting



We do not know how to weight completions of the hidden hands.



This is THE modeling choice. Research must treat it as explicit, not default.



**Required from literature synthesis:**

- Enumerate completion models that are valid without opponent behavior models (uniform over consistent deals, worst-case/robust)

- For each: what it assumes, what it breaks, how outputs should be labeled

- No novel proposals unless documented in prior work



**Scope constraint:** Since opponent/partner bidding behavior is out of scope, Bayesian posteriors over completions are not available—they require a likelihood function from a behavior model. Uniform over consistent deals is a candidate baseline (and common in the literature), but the synthesis must treat it as a modeling choice requiring sensitivity analysis, not as canonical truth.



**Constraint:** Any reported probability must be labeled with its completion model. "P(make)" without stating the distribution over hidden deals is not a valid output.



---



## What We Want



### Primary deliverable

A research-grounded method to produce a **contract × bid-level matrix**:

- For each contract c and bid level k: estimated P(make) and/or distribution of achievable points

- Under imperfect information (we know only our hand)

- With explicit completion model labeling

- With explicit acknowledgment that PIMC-derived values are optimistic upper bounds



### Secondary deliverables

- Taxonomy of valid vs invalid uses of perfect-info evaluators under hidden information

- Characterization of Strategy Fusion gap (how optimistic is the bound?)

- Error decomposition: sampling variance, completion model assumptions, PIMC bias, approximator error

- Adversarial auditing framework for using compressed oracle without self-deception



---



## Viable Paths



### Path A: Sample Completions + DP Oracle Evaluation + Aggregation



1. Generate completions consistent with our hand (under declared completion model)

2. Evaluate each with DP oracle under each contract

3. Aggregate into P(make) / distributions



**Critical caveat:** Due to Strategy Fusion, this produces an **optimistic upper bound**, not a true probability. The oracle plays perfectly against each known completion; you cannot.



**What research should provide:**

- How this is done in analogous domains (bridge double-dummy, contract bidding games)

- Known estimator issues, variance reduction, confidence intervals

- Standard term for the core operation and its failure modes

- How practitioners characterize or bound the E[max] vs max[E] gap



### Path B: Compressed Oracle + Selective DP Audit



Use compressed oracle for bulk evaluation, DP oracle for certification.



**Critical constraint:** Random auditing will confirm aggregate accuracy while missing clustered failures. If errors concentrate in high-leverage positions (which the 2-2 failure mode suggests), random sampling won't catch them. Auditing must be adversarial/targeted, not random.



**What research should provide:**

- Adversarial sampling designs that target likely failure regions (not random spot-checks)

- Calibration and selective prediction methods

- How to bound decision error when approximator has systematic (not random) bias

- "Abstain" strategies for low-confidence regions



### Path C: Robust/Conservative Bounds



Instead of expected P(make), compute:

- Worst-case over completions

- Lower confidence bounds

- Quantile guarantees



**What research should provide:**

- Established robust-decision frameworks with deterministic inner evaluators

- When to prefer bounds over point estimates



### Path D: Heuristic Lower Bound (Bracketing)



Instead of perfect play, evaluate completions using a deterministic imperfect heuristic (greedy play, rule-based policy).



**Role:** Provides a pessimistic lower bound. The true achievable value lies between Path A (oracle, optimistic) and Path D (heuristic, pessimistic).



**Key metric:** The "Robustness Gap" = Oracle Value − Heuristic Value. Large gaps indicate hands requiring precise, non-obvious play to win—high risk of human error even if the oracle says "make."



**What research should provide:**

- Is bracketing between optimistic and pessimistic evaluators a standard technique?

- How is the robustness gap used in bidding safety assessments?



---



## Paths That Will Not Work



1. **Implicit completion weighting:** Any P(make) without stated distribution assumption

2. **Treating PIMC output as true probability:** Strategy Fusion means oracle averaging gives E[max], an upper bound, not max[E], the achievable value

3. **Treating compressed oracle as ground truth:** Especially near thresholds where small errors flip make/fail

4. **Direct policy training on oracle moves:** Marginalized training problem—learns responses to specific hidden worlds, not robust responses to uncertainty

5. **Ignoring systematic approximator error:** 97% argmax agreement does not mean 3% random noise

6. **Random auditing of compressed oracle:** Errors cluster in structural failure modes; random sampling confirms aggregate accuracy while missing high-leverage failures



---



## Research Questions



### Literature and analogous domains

1. What methods exist for using perfect-information evaluators to make decisions under hidden information?

2. Where is "sample completions + evaluate + aggregate" (PIMC) used successfully? What are the documented failure modes?

3. What is the standard vocabulary for these operations and pitfalls (e.g., "determinization pathologies," "strategy fusion")?

4. What analogous domains beyond poker have faced this structure? (Planning under partial observability, robust optimization)



### Strategy Fusion / PIMC bias

5. How large is the gap between E[max] (PIMC output) and max[E] (achievable value) in trick-taking games?

6. What techniques exist to estimate, bound, or correct for this gap?

7. How do systems combine optimistic evaluators (PIMC/oracle) and pessimistic evaluators (heuristics) to bracket the true value? Is the "Robustness Gap" a standard metric for bidding safety?



### Completion modeling

8. What completion-weighting assumptions are used in practice when no opponent model is available?

9. How do systems justify or validate their choice of completion model (including uniform-random)? What sensitivity analyses are standard?



### Approximation and auditing

10. How do practitioners safely substitute learned approximators for exact evaluators?

11. What adversarial auditing approaches target likely failure regions rather than random sampling?

12. How should systematic (not random) approximator error be handled?



### Output design

13. When do systems prefer P(make) vs expected value vs quantiles vs robust bounds?

14. What confidence reporting is standard?



---



## Acceptance Criteria



A successful synthesis must:

- Cite sources for all hard claims; flag modeling choices as choices

- Provide a menu of completion models (valid without behavior models) with assumptions and failure modes

- Describe at least one baseline pipeline for producing contract × level matrices with uncertainty

- Describe at least one credible adversarial auditing scheme for compressed oracle use (not random sampling)

- Quantify or characterize the Strategy Fusion gap (E[max] vs max[E]) in analogous domains—when is PIMC "close enough"?

- Avoid domain rabbit holes: Texas 42 is the testbed, not the point

- Address the evaluator-vs-policy-teacher distinction with relevant literature



---



## Context for Researchers



Texas 42 is a four-player partnership trick-taking game with dominoes. 28 tiles, 7 per player, ~10 tricks per hand. Small enough to solve individual deals exactly, large enough that exhaustive analysis of all information sets is infeasible.



The game has structure that may be exploitable: partnership play, trick-taking mechanics similar to Bridge/Spades/Hearts, trump selection as discrete choice, certain guaranteed-win configurations ("locks"). Crucially, the game is zero-sum on points (0–42). Small optimizations by the oracle (e.g., sniping a 1-count domino) accumulate. This makes Strategy Fusion bias systematic (point inflation) rather than just variance.



We are not relying on a learned model to infer uncertainty implicitly. Uncertainty is represented explicitly as a completion model over hidden deals, and exact reasoning is applied to each completion.