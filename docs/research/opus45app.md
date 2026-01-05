# Using Perfect-Information Oracles for Imperfect-Information Bidding

Perfect-information oracles can legitimately inform imperfect-information bidding decisions, but only as **evaluators of outcomes**—never as policy teachers. The critical E[max] ≥ max[E] inequality (strategy fusion bias) means sampling-based approaches produce systematically optimistic estimates. However, empirical research on trick-taking games demonstrates this bias is often small enough to be practically acceptable: **~0.1 points per game** in Skat/Hearts conditions (high leaf correlation, gradual information revelation). The key is understanding when PIMC works, when it fails, and how to use oracle values correctly within game-theoretic algorithms that properly marginalize over uncertainty.

## The strategy fusion phenomenon creates predictable optimistic bias

**Strategy fusion** occurs when determinization-based algorithms assume a player can perform different optimal actions in different hidden worlds that are actually indistinguishable. Frank & Basin (1998) first formalized this in "Search in Games with Incomplete Information," establishing that PIMC is "prone to two distinct types of errors, irrespective of the number of hypothetical worlds examined."

The mathematical foundation is straightforward: PIMC computes **E[max(score)]**—expected value when playing optimally against each sampled configuration—while optimal play under uncertainty requires **max[E(score)]**—the best strategy given the belief distribution. Jensen's inequality guarantees E[max] ≥ max[E], creating systematic overconfidence. Your observation that "guaranteed-win configurations cause systematic point inflation" is exactly this phenomenon.

Long, Sturtevant, Buro & Furtak (2010) provided the critical empirical characterization in "Understanding the Success of Perfect Information Monte Carlo Sampling in Game Tree Search." They identified **three game parameters** predicting PIMC effectiveness:

- **Leaf correlation** (probability sibling terminals have same winner): Trick-taking games typically show **0.8–1.0** correlation
- **Disambiguation factor** (rate hidden information is revealed): Hearts/Skat/Bridge have df ≈ **0.6** as cards are revealed each trick
- **Bias** (probability one player dominates): Variable, but extreme bias helps PIMC

In synthetic games matching trick-taking parameters (correlation 0.8+, df 0.6), PIMC loses only **~0.1 points per game** versus Nash equilibrium while gaining **~0.4 points** over random play. In actual Skat, the strategy fusion gap affects only ~15% of games (those unresolved at 3 tricks remaining), with weighted average loss of **0.063 tournament points per deal**—statistically insignificant over tournament scales.

**When PIMC fails catastrophically**: poker-like games with no disambiguation (df ≈ 0), phantom games where actions don't reveal information, and games heavily rewarding information hiding. Texas 42's structure (cards revealed per trick, finite action spaces, high leaf correlation) places it firmly in the "PIMC-friendly" regime.

## Oracles must provide values, not action recommendations

The phenomenon you discovered—that training on oracle move recommendations produces "locally optimal responses to specific hidden configurations"—is well-documented under multiple names: **strategy fusion**, **information leakage**, and **conditional versus marginalized training**.

Cowling, Powley & Whitehouse (2012) formalized this precisely: "Strategy fusion occurs when performing simulations of a determinized perfect information game which relies on the assumption that a player can perform distinct actions on multiple game states" that are actually indistinguishable. When an oracle recommends "play 6-4" for one hidden configuration and "play 5-5" for another, a learner cannot distinguish these situations—imitating both recommendations produces a suboptimal mixed strategy.

The correct paradigm, established by the CFR literature and modern poker AI, separates roles clearly:

| Valid Oracle Use | Invalid Oracle Use |
|------------------|-------------------|
| Compute terminal/leaf **values** | Provide **action** recommendations |
| Evaluate counterfactual outcomes | Serve as imitation target |
| Estimate expected payoff given completion | Train policy directly on oracle moves |

The Schmid et al. (2023) Science paper "Student of Games" states this explicitly: "Summarizing the policies below depth d by a set of **values**, which can be used to reconstruct policies at depth d and beyond, is the basis of decomposition in imperfect information games." The oracle's role is evaluation; the **policy** emerges from game-theoretic reasoning (CFR, regret matching, belief-state planning) that properly marginalizes over uncertainty.

For your Texas 42 bidding application, this means: use the DP oracle to evaluate **contract outcomes** across sampled completions, then aggregate these values to estimate P(make). Never train a bidding model by imitating oracle recommendations on specific completions.

## Five completion-weighting schemes with distinct assumptions

When sampling hidden hand configurations without an opponent behavior model, five approaches appear in the literature:

**Uniform random over consistent deals** is the baseline. Sample deals satisfying hard constraints (cards played, voids shown) with equal probability. Assumptions: all consistent worlds equally likely; no informative signals from bidding. Failure mode: ignores information encoded in opponent bids/passes. Used by: basic PIMC implementations, Information Set UCT baselines. For Texas 42 bidding where no opponent actions have occurred, this is the natural starting point.

**Prior-probability weighted (bidding inference)** constrains samples using observed bids. Kermit (Furtak & Buro, 2013) uses table-based procedures incorporating opponent declarations, tracking void suits and learned histograms. GIB uses Bayesian updating: if analysis indicates West should play a card 80% of the time when holding it, and West doesn't, GIB adjusts probabilities accordingly. Failure mode: requires accurate behavioral model; model mismatch degrades performance. For pre-bidding Texas 42 evaluation, this reduces to uniform sampling.

**Neural network-based weighting** trains card-location predictors on human gameplay (AAAI 2019, "Improving Search with Supervised Learning in Skat"). The network predicts P(card in each hand) given observed sequence, providing "substantial increase in cardplay strength." Failure mode: requires substantial training data; distribution shift when opponents differ from training population. Relevant for play-phase decisions but less applicable to bidding without opponent actions.

**Minimax/worst-case robust** weights completions adversarially. Frank & Basin formalized the "best defence model" establishing equilibrium strategies exist. In practice, this is rarely used because it's overly pessimistic in trick-taking games—PIMC already performs well due to favorable game structure. However, for conservative bidding bounds (discussed below), worst-case analysis over completions provides meaningful lower bounds.

**Information-theoretic weighting** (entropy reduction, information gain) has theoretical appeal but limited practical adoption. More relevant for play-phase move selection than world sampling.

**Validation approaches** include Target Sampling Success Rate (TSSR)—measuring how well the sampling distribution covers the true world—and Perfect Information Post Mortem Analysis (PIPMA)—comparing moves to optimal perfect-information play. Sensitivity analysis varies sample size until performance saturates (typically **50–200 samples** for trick-taking games; GIB uses 50, Kermit saturates at 160).

## A baseline pipeline for contract × bid-level evaluation

To produce P(make | hand, contract, bid_level) matrices with proper uncertainty quantification:

**Step 1: Define the completion space.** Given your 7 dominoes, enumerate or sample configurations of the remaining 21 tiles among 3 opponents and partner. With 28 dominoes and 7 per player, the combinatorial space is large but tractable with sampling. Apply dealing constraints (if any exist in Texas 42's dealing protocol).

**Step 2: Sample N completions uniformly** (absent behavior model). Empirically determine N by plotting evaluation variance versus sample count; expect saturation around 100–200 samples based on trick-taking game literature. For initial exploration, N=200 provides reasonable precision.

**Step 3: For each completion, evaluate with DP oracle.** Run your perfect-information solver on the fully-specified deal for each candidate contract. Record binary outcome (make/set) and point differential.

**Step 4: Aggregate with explicit uncertainty quantification.** Compute:
- **Point estimate**: P̂(make) = (successes)/N
- **95% confidence interval**: Wilson score interval or Agresti-Coull interval (better behaved near 0/1 than normal approximation)
- **Strategy fusion warning flag**: If oracle recommends different opening leads in >30% of completions, flag potential strategy fusion inflation

**Step 5: Apply conservative adjustment for strategy fusion.** Based on Long et al. empirical measurements:
- For states with high leaf correlation and df ≈ 0.6 (typical trick-taking), expect **~5–10% optimistic bias** in close decisions
- For "lock" configurations (guaranteed wins under all completions), oracle estimate is exact—no strategy fusion possible
- For marginal contracts, report **bracketed estimates**: [pessimistic_bound, point_estimate, optimistic_oracle]

**Variance reduction**: Importance sampling (weight by hand-strength strata), stratified sampling (ensure coverage of extreme distributions), and early termination when confidence interval sufficiently narrow. The OCBA (Optimal Computing Budget Allocation) framework concentrates samples on close decisions—stop evaluating clearly dominated contracts early.

**Output**: For each contract × bid-level combination, report P̂(make), 95% CI, strategy-fusion susceptibility score, and sample standard error. Downstream bidding decisions should incorporate the confidence interval, not just the point estimate.

## Adversarial auditing must target structural failure regions

Your compressed oracle (neural network with ~97% argmax agreement, clustered errors) requires structured adversarial probing, not random spot-checks. Three approaches from the literature:

**Adaptive Stress Testing (AST)** uses reinforcement learning to learn failure-inducing input distributions (Corso et al., 2019; Lipkis & Agogino, 2024). The key insight: RL efficiently explores the space of "hard cases" by learning correlations between input features and failure. For your domino game context, this means training a failure-finder that learns which hand configurations and game states cause approximator errors. AST finds rare failures with far greater sample efficiency than Monte Carlo and can identify multiple disparate failure modes.

**Selective prediction with deferral** implements "abstain when uncertain." The framework (Geifman & El-Yaniv, 2017) adds a gating mechanism that rejects predictions below confidence thresholds. For your approximator:

1. Train a confidence estimator (or use softmax temperature calibration)
2. Identify threshold where deferred predictions would have been wrong at high rate
3. In deployment, use exact DP oracle when approximator confidence falls below threshold
4. The ~3% disagreement rate bounds the maximum deferral frequency

**SPTD (Selective Prediction via Training Dynamics)** analyzes SGD trajectory instability to detect aleatoric uncertainty—high disagreement across training checkpoints indicates unreliable predictions. This requires access to intermediate training checkpoints but provides strong uncertainty signals without architecture modification.

**Systematic bias detection** for ensemble and neural approximators: residual analysis checking whether errors sum to zero (they won't for systematic bias), stratified evaluation across structural subgroups (hand strengths, suit distributions, point margins), and Empirical Distribution Matching to correct identified biases post-hoc.

For your specific clustered errors ("context-conditioning failures"), the auditing strategy should:

1. **Identify structural features** of hands/states where errors cluster (point margins near cutoffs, unusual suit distributions, certain trump configurations)
2. **Oversample these regions** using AST or stratified adversarial sampling
3. **Implement selective deferral** where confidence correlates with these structural features
4. **Quantify decision error bounds**: if approximator systematically over-/under-estimates values in identified regions, compute worst-case decision error given the bias magnitude

## Robust bounds provide safety margins for bidding decisions

Rather than trusting point estimates of P(make), robust/conservative approaches provide safety margins:

**Worst-case over completions** computes min_{completions} oracle_value. For bidding, this asks: "What's the worst distribution of hidden tiles I might face?" This is typically too conservative (you'd never bid), but provides a meaningful lower bound. The **Robustness Gap** = P̂(make)_mean - P(make)_worst-case quantifies bid safety.

**CVaR (Conditional Value-at-Risk)** objectives offer a middle ground. Chow et al. (NeurIPS 2015) proved CVaR minimization is equivalent to worst-case expected cost under bounded model perturbations. For bidding: instead of optimizing expected points, optimize the expected points in the worst α% of completions. This captures "how bad does it get if tiles are distributed against us?"

**Hurwicz criterion** explicitly balances pessimism and optimism: value = α×(worst outcome) + (1-α)×(best outcome). Setting α near 1 produces conservative bids; α near 0 produces aggressive bids. For tournament play with different bid incentives (slam bonuses, etc.), α can be tuned to match strategic objectives.

**Bracketed estimates** report: [pessimistic_bound, point_estimate_with_CI, optimistic_oracle]. The pessimistic bound might use a handcrafted heuristic (weaker than oracle), the optimistic uses raw oracle averages. The gap between these bounds indicates decision confidence—wide gaps suggest deferring to more conservative bids.

**Bertsimas-Sim approach** from robust optimization: parameterize "budget of uncertainty" Γ controlling how many tiles can be adversarially placed. Γ=0 gives nominal (expected) evaluation; Γ=max gives full worst-case. Intermediate Γ provides probabilistic guarantees with reduced conservatism.

## Canonical vocabulary and essential references

**Strategy Fusion**: The error where PIMC incorrectly assumes different strategies can be used for indistinguishable states. Canonical: Frank & Basin (1998), *Artificial Intelligence* 100(1-2).

**Determinization Pathologies**: Umbrella term for strategy fusion and non-locality problems in PIMC. Canonical: Long et al. (2010), *AAAI*.

**Non-locality**: Node values depending on tree regions outside the current subtree (opponents using private information to steer play). Canonical: Frank & Basin (1998).

**PIMC (Perfect Information Monte Carlo)**: Sampling hidden states, solving each with perfect-info oracle, aggregating results. Canonical: Ginsberg (2001), *JAIR* 14 (GIB system).

**Information Set**: Collection of game states indistinguishable to a player; the fundamental unit of imperfect-information game theory. Canonical: Kuhn (1953), von Neumann & Morgenstern (1944).

**ISMCTS (Information Set MCTS)**: MCTS variant operating on information sets to avoid strategy fusion. Canonical: Cowling et al. (2012), *IEEE Trans. CIAIG*.

**CFR (Counterfactual Regret Minimization)**: The algorithm for computing Nash equilibria in extensive-form games with imperfect information. Canonical: Zinkevich et al. (2007), *NIPS*.

**Counterfactual Values**: Expected utility if player "tried to reach" an information set, properly marginalized over hidden states. Canonical: Neller & Lanctot (2013), tutorial.

**Double-Dummy Analysis**: Bridge terminology for perfect-information solving with all hands visible. Standard benchmark for bridge AI evaluation.

**αμ (Alpha-Mu) Search**: Algorithm designed to repair strategy fusion and non-locality in card games. Canonical: Cazenave & Ventos (2021).

**Recursive PIMC / IIMC**: Using imperfect-information playouts within PIMC to mitigate strategy fusion. Canonical: Furtak & Buro (2013), *IEEE CIG*.

**Belief-State Planning**: Planning over probability distributions over hidden states (POMDP framework). Canonical: Silver & Veness (2010), POMCP algorithm.

## The core insight for your Texas 42 application

Texas 42's game structure—high leaf correlation, gradual information revelation (df ≈ 0.6), finite action spaces, tractable endgames—places it in the regime where PIMC sampling with a DP oracle provides reliable bidding estimates despite strategy fusion bias. The empirically measured gap in comparable games (Skat, Hearts) is small enough (~0.1 points/game) that PIMC-based P(make) estimates are actionable.

The critical constraints: use the oracle as an **evaluator** (compute contract outcomes across completions, aggregate values), never as a **policy teacher** (don't train on oracle move recommendations). Report **bracketed estimates** with confidence intervals, flagging hands where oracle recommendations vary substantially across completions. For your compressed oracle, implement **selective deferral** to the exact DP solver in low-confidence regions identified through structured adversarial probing.

The pipeline—sample completions uniformly (pre-bidding, no behavior model), evaluate with DP oracle, aggregate with uncertainty quantification, apply conservative adjustments—provides the methodologically sound foundation for your contract × bid-level matrices.