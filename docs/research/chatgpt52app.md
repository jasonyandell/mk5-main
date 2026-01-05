Using Perfect-Information Evaluators in Hidden-Information Decision Making
Introduction
In many games and decision problems with hidden information, AI systems have leveraged perfect-information evaluators – “oracle” methods that assume all hidden variables are known – to guide decisions. The classic example is determinization, or Perfect Information Monte Carlo (PIMC) simulation, where we sample possible completions of unknown information and evaluate each as if it were a perfect-information scenario
webdocs.cs.ualberta.ca
webdocs.cs.ualberta.ca
. This approach has achieved notable successes in domains like Bridge and Skat
webdocs.cs.ualberta.ca
. However, it also introduces well-known biases and pathologies. As Russell and Norvig famously noted, “averaging over clairvoyance” can suggest moves “that no sane person would follow”
webdocs.cs.ualberta.ca
, because the evaluator overestimates strategies by assuming knowledge that the real decision-maker does not actually have. This report synthesizes research on how to support decision-making under hidden information using perfect-information oracles while mitigating their pitfalls. We organize the discussion into clear topics:
Strategy Fusion and PIMC Bias: Definitions of the key pathologies (strategy fusion, non-locality) and methods to estimate, bound, or correct the bias (the gap between E[max(score)] and max(E(score))).
Broader Applications: Analogous issues and solutions beyond the test domain (Texas 42), including trick-taking card games (Bridge, etc.), poker, planning under partial observability (POMDPs), and robust optimization.
Completion Modeling without Opponent Models: Strategies for sampling hidden information or opponent behavior when no reliable model exists – from uniform random sampling to worst-case (conservative) completions.
Safe Use of Approximate Evaluators: Techniques for safely using learned or approximate evaluators instead of exact oracles, including adversarial auditing, abstention, and focused sampling to avoid high-impact errors.
Bracketing and Robustness Bounds: “Bracketing” approaches that use both optimistic (perfect-info) and pessimistic evaluations to bound true performance, and whether a so-called “Robustness Gap” is a standard metric.
Terminology and Reporting: Standard vocabulary for these concepts (e.g. determinization pathologies like strategy fusion) and best practices for outputting recommendations with confidence metrics (e.g. probability of success, expected value, quantiles, robust bounds).
Throughout, we draw insights from both academic literature and industry research (DeepMind, Meta AI, etc.), focusing on broadly applicable methods rather than domain-specific tricks. The goal is to highlight general principles to harness powerful perfect-information evaluators safely in hidden-information settings.
Key Terminology and Determinization Pathologies
Before diving deeper, we define important terms and known problems that arise when using perfect-information evaluation in hidden-information contexts:
Determinization / Perfect Information Monte Carlo (PIMC): Sampling hidden information (e.g. unknown cards or states) to create one or more deterministic worlds, then solving each as a perfect-information game (often via search or simulation). The results are averaged to guide action choice
webdocs.cs.ualberta.ca
. PIMC is essentially “averaging over clairvoyance”
webdocs.cs.ualberta.ca
 – treating unknowns as if they were known for evaluation purposes.
Strategy Fusion: A pathology identified by Frank and Basin (1998) in which a decision procedure improperly assumes it can execute different optimal strategies in different worlds
webdocs.cs.ualberta.ca
. In reality, the player must choose a single strategy without knowing which world (state of hidden information) they are in. PIMC “incorrectly believes it can use a different strategy in each world”
webdocs.cs.ualberta.ca
, effectively overestimating the value of a move by cherry-picking the best response for each possible scenario. This leads to an optimistic bias: the evaluated value $E[\max(\text{score})]$ (the expectation of achieving the max in each scenario) exceeds the attainable $ \max(E[\text{score}])$ (the best single strategy’s expected score). Strategy fusion is a major source of PIMC bias.
Non-Locality: A second determinization pathology from Frank & Basin’s analysis
webdocs.cs.ualberta.ca
. In perfect-information games, the value of a node in the game tree depends only on its subtree. In imperfect-information games, however, a node’s value can depend on other parts of the game tree because opponents may use hidden information to steer play into favorable regions
webdocs.cs.ualberta.ca
. PIMC evaluation ignores these non-local dependencies. For example, a plan might look safe in each sampled world independently, but an informed opponent could deduce information from your plan and counter it (a dependency PIMC fails to account for)
webdocs.cs.ualberta.ca
. Non-locality can cause the PIMC evaluator to misestimate values since it cannot model opponents’ information-based strategy shifts
webdocs.cs.ualberta.ca
.
PIMC Bias (Determinization Bias): A general term for the error introduced by using a perfect-information evaluator under hidden information. It often manifests as over-optimism due to strategy fusion – the evaluator’s tendency to overestimate a move’s value by assuming world-by-world optimal adaptation. Formally, it’s the gap between $E[\max(\text{outcome})]$ and $\max(E[\text{outcome}])$, which is always $\ge 0$ by Jensen’s inequality (and is exactly the expected value of perfect information, a concept from decision theory). A larger gap means the evaluator is more overly optimistic about a move’s prospects. Non-locality can further worsen this bias by failing to account for adversarial exploitation of one’s strategy.
Information Set / Infostate: The set of all states (worlds) that are possible given what a player has observed. In hidden-information games, the true state is uncertain, and the player makes decisions based on an information set. A fundamental constraint is that a single decision must apply to the whole information set – you cannot choose different actions for different states within the same info set. Determinization methods often violate this by effectively choosing different actions per sampled state, hence introducing strategy fusion.
Determinization Pathologies: An umbrella term for the kinds of errors above (strategy fusion, non-locality, and related issues) that plague naive use of determinized simulations
webdocs.cs.ualberta.ca
webdocs.cs.ualberta.ca
. These pathologies explain why a straightforward perfect-information approach can mislead decision-making under uncertainty, and motivate the corrective techniques discussed in this report.
Having established the vocabulary, we now explore methods to quantify and mitigate PIMC bias and related issues.
Estimating and Correcting Strategy Fusion Bias
Quantifying the Bias: Directly detecting strategy fusion or non-locality in complex games is difficult because solving the true imperfect-information game exactly is usually intractable
webdocs.cs.ualberta.ca
. Researchers have proposed surrogate metrics to estimate how prone a given game or scenario is to PIMC error. Long et al. (2010) introduced measures like leaf correlation and disambiguation factor
webdocs.cs.ualberta.ca
webdocs.cs.ualberta.ca
:
Leaf Correlation (lc): The probability that all terminal outcomes under a given node are identical
webdocs.cs.ualberta.ca
. High leaf correlation means that no matter how uncertainty resolves, the outcome is essentially the same – a property that makes a game friendly to determinization. Low leaf correlation (outcomes diverge widely across worlds) indicates opportunities for a player to drastically alter payoffs late in the game, which is fertile ground for strategy fusion issues
webdocs.cs.ualberta.ca
. When lc is low, PIMC is more likely to overestimate a strategy by assuming an ability to “pick the winning world.”
Bias (b): A measure of inherent advantage in the game (how often one player tends to win)
webdocs.cs.ualberta.ca
. If one side is heavily favored (very high or low bias), large homogeneous regions of the game tree exist where that player’s strategy is obvious
webdocs.cs.ualberta.ca
. In such cases, even a flawed search might perform well by steering into those dominant regions. Moderate bias games (balanced outcomes) might be more prone to PIMC errors if the search fails to identify critical swing scenarios.
Disambiguation Factor (df): How quickly hidden information gets revealed as the game progresses
webdocs.cs.ualberta.ca
. In trick-taking card games (bridge, Skat, 42, etc.), each played card uncovers some private info, so uncertainty reduces with each trick – a high disambiguation factor
webdocs.cs.ualberta.ca
. In contrast, in poker no private info is revealed until showdown (low disambiguation until the very end)
webdocs.cs.ualberta.ca
. The longer uncertainty persists, the more PIMC has to rely on hypothetical worlds without ever getting feedback, which exacerbates strategy fusion opportunities. High df means by the time decisions get critical, much uncertainty is resolved, making determinization more reliable
webdocs.cs.ualberta.ca
.
These properties can be measured in actual game trees and used as indicators of PIMC’s likely accuracy
webdocs.cs.ualberta.ca
webdocs.cs.ualberta.ca
. For example, Bridge and trick-taking games have a reasonably high disambiguation factor (cards gradually revealed), which helps explain why PIMC-based methods can perform decently there
webdocs.cs.ualberta.ca
. Poker, with very low information revelation, is far more problematic – naive determinization would severely mislead, aligning with the fact that game-theoretic methods outperform PIMC in poker (discussed later). Another way to quantify the bias is computing the Expected Value of Perfect Information (EVPI) in decision-theoretic terms – essentially the $E[\max] - \max(E)$ gap for a decision. EVPI gives an upper bound on how much better an oracle could do by knowing all secrets. If EVPI is large in a domain, a perfect-info evaluator has a lot of “extra” potential to appear strong by tailoring strategies to each world, meaning strategy fusion bias could be large. In practice, calculating EVPI exactly can be as hard as solving the game, but it conceptually frames the maximum possible bias. Bounding or Reducing the Bias: Several techniques attempt to rein in strategy fusion and produce more realistic evaluations:
Consistent Action Simulation: The simplest correction is to enforce that the same action is taken across all sampled determinizations for a given decision point, rather than allowing each world to choose its own best move. In other words, evaluate each candidate action by fixing that action in all worlds and averaging the result. This is in fact how many PIMC implementations choose moves at the root: by computing the average score of each action across random deals and selecting the highest average (which yields $\max(E[\text{score}])$). However, deeper in the lookahead, many implementations still allow branches to diverge per world. A stricter approach is to constrain the search such that whenever the information set is the same, the algorithm considers one action for all worlds in that set. This idea underlies Information Set Monte Carlo Tree Search (ISMCTS)
arxiv.org
. ISMCTS, introduced by Cowling et al. (2012), treats the decision node as an information set and selects one action per simulation that is applied consistently across a sampled world. Over many simulations, it biases towards actions that perform well on average for the info set, rather than optimistically per-world. ISMCTS thereby avoids explicit strategy fusion and has been used successfully in games like Dou Di Zhu and others
arxiv.org
arxiv.org
.
Delayed Revealing / Partially Clairvoyant Search: A recent innovation is to postpone the point at which perfect-information evaluation is applied. Extended PIMC (EPIMC) by Arjonilla et al. (2024) delays the “oracle” resolution until a deeper depth in the search tree instead of at the root
arxiv.org
arxiv.org
. For example, rather than solving each full deal from the start, EPIMC might simulate moves under uncertainty for a few rounds and only then, at depth d, reveal the remaining unknowns and evaluate terminal outcomes with perfect information. By doing so, the strategy must remain coherent for the first d moves (across all sampled worlds), reducing strategy fusion opportunities in those initial moves
arxiv.org
arxiv.org
. The authors prove that increasing the delay depth d never worsens strategy fusion bias, and in fact there exists some depth that eliminates strategy fusion entirely in finite games
arxiv.org
. Intuitively, if you delay until the game is almost over, you’ve essentially forced a single strategy through most of the game, approaching a proper information-set strategy. EPIMC demonstrated improved performance in games specifically designed to exacerbate strategy fusion, confirming that judiciously “postponing clairvoyance” makes the evaluation more realistic
arxiv.org
arxiv.org
.
Heuristic Bias Correction: Some systems explicitly adjust the oracle’s evaluation to account for hidden information. For instance, the Bridge-playing program GIB augmented its double-dummy (perfect info) solver with an alpha-beta search over lattices to represent sets of possible deals
webdocs.cs.ualberta.ca
. This technique partially accounted for the uncertainty by evaluating regions of deal space (lattices) rather than one deal at a time, thus capturing some information constraints. GIB reportedly gained a small but measurable improvement (about 0.1 International Match Point per deal) from this correction
webdocs.cs.ualberta.ca
. Other heuristics include penalizing strategies that have high variance across determinized worlds (on the theory that high variance might signal a strategy-fusion-driven gamble). By favoring moves that are consistently good across most sampled worlds, the evaluator leans toward robust choices (at the cost of sometimes missing a brilliant “double dummy” coup that only works in a few cases).
Multiple World Models (Bracketing – see later section): Another approach is to run two evaluations – one optimistic (assume perfect information or best-case responses in each world) and one pessimistic (assume worst-case or enforce uniform actions) – to bound the true value. We will discuss this bracketing strategy in detail later, but note here that it provides an estimate of the bias. If the oracle (optimistic) says a bid’s success probability is, say, 80%, but a conservative evaluation says only 50%, the gap (30%) essentially reflects potential strategy fusion or adversarial exploitation. Recognizing a large gap can signal low confidence in the recommendation, prompting perhaps a more cautious strategy.
In summary, while pure PIMC can be dangerously optimistic in many imperfect-information settings, researchers have developed both analytical tools to predict when it will fail and algorithmic tweaks to mitigate its worst flaws. In many real games, these adjustments – or alternatives like ISMCTS – ensure that perfect-information evaluators remain useful guides rather than misleading oracles.
Applications Beyond Texas 42: Bridge, Poker, Planning, and Robust Decisions
The strategy fusion problem and its solutions are broadly relevant across domains with hidden information. We now survey analogous challenges and techniques in a few key areas: Trick-Taking Card Games (Bridge, Skat, etc.): These games (including Texas 42, a domino trick-taking game) have been fertile ground for PIMC-based AI, starting with Ginsberg’s Bridge program GIB in the 1990s. Bridge is a quintessential imperfect-information game: each player only sees their own hand and the bidding provides some clues, but during play one does not know the distribution of the remaining cards. GIB’s breakthrough was to sample dozens of random deals consistent with the bidding and use a double-dummy solver (a perfect-info search) to play out each deal
webdocs.cs.ualberta.ca
. By averaging the results, it chose plays that maximized the probability of making the contract. Despite the theoretical critiques, this worked startlingly well – GIB achieved world-champion-level play by 2001
webdocs.cs.ualberta.ca
. The “mystery” of PIMC’s success in Bridge was later addressed by Long et al. (2010), who noted Bridge has properties (fairly high leaf correlation and rapid information revelation through trick play) that make PIMC’s assumptions less harmful
webdocs.cs.ualberta.ca
webdocs.cs.ualberta.ca
. Nonetheless, expert Bridge AIs still add domain-specific fixes: e.g. GIB’s lattice search to slightly account for uncertainty
webdocs.cs.ualberta.ca
, or more recent algorithms like α–μ search which try to combine sampling with inference. In the related card game Skat, a PIMC-based agent by Buro et al. also achieved expert performance
webdocs.cs.ualberta.ca
, suggesting that when opponents are reasonably “average” and information gets revealed as play progresses, determinization can be surprisingly competitive. However, as soon as multiple players collude or play adversarially, the limitations become evident – a determinized AI may be exploited by those who recognize its fixed pattern across deals. Poker and Game-Theoretic Approaches: In sharp contrast to Bridge, deterministic oracle approaches fared poorly in competitive poker. Poker’s hidden information (opponents’ private cards) stays hidden until showdown (low disambiguation factor
webdocs.cs.ualberta.ca
), and strong opponents actively randomize and deceive. A PIMC approach to poker would involve sampling opponents’ hands and computing the best counter-strategy for each – essentially assuming you could see their cards and respond accordingly. This is a textbook case of strategy fusion: if you could play a different strategy for every possible opponent hand, you’d do far better than any single mixed strategy. The bias would be enormous and the resulting strategy deeply flawed against real opponents. Modern poker AIs instead rely on game-theoretic algorithms like Counterfactual Regret Minimization (CFR) to compute mixed strategies that are robust (Nash equilibrium approximations) and cannot be exploited by an opponent 
arxiv.org
. Notably, Brown et al. (2020) point out that algorithms like AlphaZero’s MCTS do not directly apply to imperfect information because “the value of an action may depend on the probability that it will be chosen”
frontiersin.org
 – in other words, you can’t evaluate a bluff in isolation; its success depends on how often you bluff (an information-sensitive consideration). This insight underlies why poker solutions integrate opponent modeling or equilibrium finding rather than naive sampling. That said, there have been hybrid approaches. Facebook’s ReBel (2020) and DeepMind’s Player of Games (2021) combine search with learned value networks and explicit belief modeling
highstakesdb.com
. They do sample hidden states during search, but maintain a common strategy across them by incorporating game-theoretic reasoning. For example, ReBel builds a search tree in belief-state space (states include probability distributions of opponent hands) rather than in concrete states, thus avoiding strategy fusion by design. These advances show that even in poker, one can incorporate simulation-based lookahead as long as one properly accounts for uncertainty and does not allow branching strategies per determinization. The result is an AI that effectively brackets the performance of pure search and pure equilibrium: Player of Games, for instance, wasn’t as strong as a specialized poker solver or as AlphaZero in chess, but it achieved high performance in both domains
highstakesdb.com
 by blending MCTS with CFR. The takeaway for imperfect-information domains is that game-theoretic consistency (ensuring a single strategy across possible worlds) is key to avoiding illusions of grandeur that a perfect-info oracle might introduce. Planning Under Partial Observability (POMDPs): Outside of games, the determinization dilemma appears in AI planning and robotics. Faced with a Partially Observable Markov Decision Process, one approach is to determinize by assuming some fixed scenario – often the most likely scenario – and then plan as if fully observable. This was the idea behind algorithms like FF-Replan in classical planning competitions: they treat unknown outcomes as if a particular outcome will occur, plan accordingly, and re-plan if reality differs. Such planners are fast and often effective, but they can exhibit catastrophic failures due to strategy fusion-like reasoning. Essentially, the plan is optimized for the assumed world and might not even make sense if a different hidden state is true. For example, a robot might plan a path assuming an unobstructed corridor (most likely), but if the corridor is actually blocked, the plan has no provision to recover – analogous to an AI assuming the “easy deal” in Bridge and failing if the cards lie differently. Researchers have noted that “most-likely outcome” determinization cannot handle contingencies that require preparing for less-likely events
lis.csail.mit.edu
isrr-2011.org
. This is precisely the flaw of strategy fusion: overcommitting to one world. Advanced POMDP solvers instead keep track of a belief state (a distribution over states) and compute actions that maximize expected utility over that belief or maximize worst-case guarantees. Monte Carlo planning algorithms like POMCP (Partially Observable Monte Carlo Planning) avoid full determinization; they sample not only state trajectories but also observation outcomes, gradually refining a policy that works well in expectation without assuming clairvoyance. POMCP’s particles represent possible states but the algorithm’s choice at a belief node is unified across those particles, akin to ISMCTS in games. On the other end, robust planning techniques solve for actions that guarantee success if possible regardless of hidden state. This maps to paranoid worst-case reasoning (discussed next), which can be overly conservative. In practice, planning under uncertainty often combines optimistic and pessimistic models to find a sweet spot – e.g., contingent planning where you prepare branches for different revelations, or minimax regret criteria where you minimize the maximum loss relative to an ideal strategy with full info. These methods explicitly address the gap between $E[\max]$ and $\max(E)$ by either narrowing it (hedging strategies that perform consistently across states) or by treating the hidden state as an adversary to ensure $\max(E)$ is achieved under worst-case. Robust Optimization: In fields like operations research and finance, we encounter a similar dichotomy. An optimizer with perfect information about future uncertainties would choose a different solution for each scenario (analogous to per-world optimal plays). Without that, one can either optimize for the expected outcome (analog of PIMC’s average) or for the worst-case outcome (analog of paranoid strategy). The difference in result or performance is often called the price of robustness. For example, a supply chain plan that minimizes average cost may assume “likely” demand scenarios, whereas a robust plan hedges against the worst demand spike. The robust plan guarantees feasibility in all cases but usually incurs higher cost on average – reflecting the robustness gap. In our context, the Robustness Gap could be seen as the performance difference between a strategy optimized for average hidden information and one optimized for worst-case. While “Robustness Gap” isn’t a standard metric name in game AI literature, the concept appears in various forms. In poker, it corresponds to exploitability: the difference between your performance against a typical opponent and against a worst-case, fully exploiting opponent. In robust decision-making, it’s analogous to how much expected value you sacrifice to protect against adversarial scenarios. We will revisit this when discussing bracketing. For now, note that robust optimization frames hidden information as an adversary and thereby completely avoids strategy fusion – at the cost of potential under-performance if the adversary model is too pessimistic. Across these domains, the core lesson is consistent: a strategy must be evaluated on its ability to handle all possible states consistent with what is known, not just each state in isolation. Methods that respect that (equilibrium finding, belief-state search, contingency planning) generally avoid the false allure that a perfect-information evaluator may project. Domains with natural feedback of info (trick-taking games) tend to be more forgiving, whereas domains with lingering uncertainty (poker, Kriegspiel chess, etc.) demand more careful approaches.
Completion Modeling Without Opponent Models
A practical challenge is how to model the unknown information – e.g. opponent’s hand or behavior – when we lack a reliable prior or model. This affects what distribution of worlds a perfect-info evaluator samples, which in turn influences the decision. Three broad strategies are commonly considered:
Uniform Random Sampling: If no better knowledge is available, assume all hidden possibilities consistent with observations are equally likely. This is a straightforward application of the principle of insufficient reason. In games, this often means assuming opponents’ private cards or moves are random (or drawn from a default distribution). Uniform sampling is the default in many PIMC implementations (e.g., dealing random cards for the unseen hands in Bridge/Texas 42). The justification is that if you truly have no information about the opponent, the best you can do is maximize your average outcome over all possibilities. An interesting result by Parker et al. (2006) supports this approach: they compared a so-called “overconfident” strategy (which assumes the opponent plays uniformly at random – essentially an average-case assumption) versus a “paranoid” strategy (which assumes the opponent will act optimally to minimize our payoff, i.e. worst-case) in Kriegspiel (a partially observable chess variant). The overconfident model consistently outperformed the paranoid model in their experiments
academia.edu
. In other words, assuming opponents will just play a mix of moves (rather than perfect counter-strategy) led to better decisions and outcomes in that domain. The intuition is that overly pessimistic models squander opportunities – if the opponent isn’t actually an omniscient adversary, a paranoid strategy forfeits too much value. Uniform sampling effectively treats the opponent as a baseline stochastic actor and often yields a reasonably strong best-response to an average opponent. This works well if opponents are not extremely deceptive or if hidden information is truly random (like dealing of cards).
Worst-Case (Adversarial) Completions: In high-stakes or safety-critical settings, one might instead assume the hidden information will turn out in the worst possible way for us – essentially treating nature or the opponent as an adversary. This leads to a minimax or maximin strategy: choose the action whose guarantee (over all possible completions) is best. Game AI literature refers to this as the paranoid algorithm in multi-player or imperfect-info contexts
dl.acm.org
. For example, a paranoid Bridge player would assume the opponents’ cards lie in the most unfavorable configuration and that they will always make the best counter-play given any clue. The upside of this approach is a strong guarantee – if you can find a strategy that succeeds even in the worst case, you are safe no matter what. The downside is that it can be overly conservative. As noted above, paranoid strategies can fare poorly if the worst-case assumptions rarely materialize
academia.edu
. They might miss out on chances to score higher against less-than-perfect opponents. In robust optimization terms, a worst-case plan often has a “robustness tax” – it trades away performance in typical scenarios for security in extreme ones. Nonetheless, worst-case modeling is appropriate when failure is not an option (e.g., security, safety domains) or when facing a truly adversarial opponent (e.g., solving a zero-sum game theoretically). In absence of opponent models, it provides a clear, if pessimistic, baseline.
Conservative or Robust Sampling Techniques: Between uniform randomness and pure worst-case, there are intermediate approaches. One idea is biased sampling – sample hidden states not uniformly, but weighted towards those that are plausible and potentially dangerous. For instance, if some hidden configurations would lead to a very bad outcome for our action, we might overweight those in the evaluation to see if the action still holds up. This can be seen as a form of risk-averse expectation: not quite worst-case, but skewed to punish high-risk scenarios. Another approach is minimax regret sampling – evaluate actions by the regret if a certain state holds, and choose the action that minimizes the maximum regret. This often results in a more balanced decision than pure minimax, allowing some optimism if the worst cases are very improbable. In practical terms, one might do a mix of samples: e.g., 70% drawn uniformly and 30% drawn adversarially (or specifically include the known extreme cases). This “stress-testing” ensures that the chosen action is not catastrophically bad in any single scenario, while still prioritizing overall expected value.
When opponent behavior models are unavailable, the justification for uniform sampling often comes from equilibrium theory (if opponents are rational but you have no info, the Bayesian optimal prior might be uniform) and from empirical results like Parker’s showing that playing against an average-case assumption can outperform always bracing for the worst
academia.edu
. On the other hand, worst-case reasoning is justified by prudence in adversarial domains – if you cannot afford to be wrong even once, you must consider the worst-case completions. The key is to align the modeling strategy with the nature of the domain: e.g., for a casual trick-taking AI playing unknown humans, assuming a reasonable distribution of hands and moves (perhaps even a slight bias towards simpler or more common strategies) will yield better performance and enjoyment. In contrast, for a cybersecurity AI trying to detect intrusions (an adversarial setting), assuming an intelligent adversary (worst-case) is prudent. Hybrid and Adaptive Strategies: Some research has explored dynamically adjusting the opponent model on the fly. For example, a system might start by assuming a uniform model, but if an opponent’s play starts to appear optimally geared against our strategy (signaling a strong adversary), the system can shift to a more paranoid mode. Conversely, if the opponent seems to make random or suboptimal moves, the system could remain in average-case mode to exploit that. In the absence of any opponent model, the initial uniform assumption is a neutral starting point, and observations can inform updates (akin to Bayesian belief updates on opponent type). In summary, uniform sampling is typically the default “no-model” approach for hidden information, offering a solid average-case strategy, whereas worst-case completions provide a safety net at the cost of potential underperformance. Other robust sampling techniques aim for a middle ground, ensuring that a strategy isn’t brittle against unlikely but severe scenarios without fully sacrificing expected value. Empirically, as Parker et al. showed in Kriegspiel, the usual perfect-information search assumption (opponent moves randomly unless forced) tends to outplay an overly pessimistic approach
academia.edu
 – hence many AI systems lean toward optimism unless there’s evidence to do otherwise.
Safe Use of Approximators Instead of Exact Oracles
Perfect-information evaluators (e.g. deep search or combinatorial solvers) can be extremely slow or even intractable in complex games. Thus, practitioners often use approximate evaluators – for instance, a neural network that estimates the value of a position, or a shallow heuristic simulation – in place of exact computation. However, introducing an approximator raises concerns: it might be biased or exploitable in ways the exact oracle isn’t, especially under hidden information. To safely integrate approximators, researchers employ several strategies:
Adversarial Auditing and Stress-Testing: Before deploying an approximate evaluator in critical decision-making, it’s tested against adversarial scenarios to find weaknesses. In a game context, this can mean searching for a hidden-information scenario where the approximator’s evaluation is drastically wrong, then seeing if an opponent (or a different algorithm) can steer the game into that scenario. This is analogous to red-teaming an AI model. For example, suppose we have a neural network that evaluates bridge contracts. We might feed it deals where a certain bid is very borderline and see if we can tweak the deal (within what’s consistent with bidding) to confuse the network – perhaps giving it a distribution of cards where its preferred line of play actually fails. By identifying such adversarial deals, developers can either retrain the evaluator to fix those errors or incorporate logic to handle them. In essence, adversarial auditing treats the evaluator as a system to be attacked: we simulate a clever opponent that chooses hidden information or actions to maximize our loss, given we follow the evaluator. This approach connects to the concept of exploitability in games: a safe strategy should have low exploitability (no easy way for an adversary to force a bad outcome). Audit processes try to approximate this by testing the AI against tough sequences of events. DeepMind’s AlphaZero for perfect games didn’t need this (the search itself mitigated evaluation errors), but in imperfect games, Facebook’s ReBel and DeepMind’s Player of Games explicitly measure exploitability as part of training
highstakesdb.com
. Similarly, we can evaluate how an approximator might be exploited and use that as a yardstick for safety.
Abstain and Escalation Strategies: A reliable decision support system should know its limits. When an approximator is unsure or operating outside its validated zone, the system can abstain (defer the decision or choose a safe default) or escalate to a more robust method (e.g., do a deeper search or ask for human input). For example, an AI bidding assistant for Texas 42 might refrain from giving a confident high bid recommendation if its evaluations of that bid are extremely close to the threshold or highly volatile across deal samples. Instead, it might say “Bid cautiously; my confidence in making the bid is low.” This is analogous to classification systems that have a “don’t know” option. In games, one could design the AI to play a simple, non-disastrous strategy when its neural net evaluation is uncertain, thereby avoiding blunders. Another form of abstention is fallback to exact calculation for critical points. For instance, if a chess engine’s neural evaluator is uncertain in a sharp endgame, it may switch to a precise endgame tablebase. In partial information games, if an approximated PIMC value for a move is within a tiny margin of another move (i.e., essentially a tie with possible swing), the system might call for more simulations or just choose a lower-variance play.
Targeted Sampling for High-Leverage Errors: Combining approximators with selective exact evaluation can yield the best of both. The idea is to use the fast approximator generally, but devote extra computation to scenarios where an error would be very costly. High-leverage situations include those late in the game (where a wrong move loses the match), or rare branches where the approximator’s training data was sparse. One implementation is to monitor the variance or uncertainty of the approximator’s output. For example, if a neural network outputs not just a point estimate but some confidence measure (or if an ensemble of networks disagree), that flags a state as needing more scrutiny. The system can then spawn additional simulations or deeper searches for that state. In Monte Carlo Tree Search, this happens naturally: the algorithm will explore moves with uncertain outcomes more (using UCB1 or similar exploration bonuses) – effectively allocating more samples to places where the heuristic might be unreliable. Another approach is online refinement: methods like DeepStack (for poker) would solve a small lookahead game exactly (via linear programming) whenever the AI faced a particularly tricky situation that the neural net value might mis-evaluate
arxiv.org
arxiv.org
. This ensured that critical betting decisions were backed by a sound game-theoretic calculation if needed. We can generalize this idea: use the approximator for breadth, but zoom in with more accurate tools on the critical junctures identified either by domain knowledge or by the approximator’s own signals.
Adversarial Training and Robustification: To the extent possible, one can train the approximator itself to be robust against tricky hidden-info scenarios. This is akin to adversarial training in supervised learning – e.g., augment training data with scenarios that were found to fool the evaluator, so it learns not to be fooled. In games, self-play training often naturally includes some adversarial element, as the opponent (even if initially weak) gets better and finds holes in the strategy, which the learning then patches. Meta AI’s ReBel, for example, integrates search and training such that the final policy/value network is consistent with Nash play, closing off exploitative loopholes
arxiv.org
arxiv.org
. From an engineering perspective, if we have a black-box oracle (like a simulator) that’s too slow to query everywhere, we might train a surrogate model for it but continuously validate and correct that surrogate on a targeted set of test cases (especially on boundary cases where the oracle’s output sharply changes). This reduces the risk of blind spots in the approximator.
In summary, replacing an exact evaluator with an approximation introduces new “hidden information” – namely, the uncertainty about the evaluator’s correctness. The strategies above treat that uncertainty carefully: by testing for worst-case errors (adversarial audit), giving the system an option to say “not sure” or fall back (abstain/escalate), and focusing precision on the most consequential decisions (targeted sampling). Industrial game AIs often combine neural networks with traditional search precisely to harness the strengths of both – the neural net provides a heuristic global view, and search verifies and refines decisions in local tactical spots. This approach was key to success in Go and has been extended to imperfect information games with additional care to ensure the search remains sound (e.g., Pluribus poker AI performed limited-depth lookahead to resolve situations beyond the neural strategy). The result of safe approximator use is that the AI’s overall decision policy is more robust than the raw approximator alone. For instance, AlphaGo’s policy network alone might suggest a risky move, but MCTS would typically catch if that move leads to an obvious refutation. In a hidden-information setting, one might similarly use a belief-state tree search to double-check a neural recommendation in cases where a human or adversary could exploit a naive play.
Bracketing Evaluations and the “Robustness Gap”
One compelling technique for decision support under uncertainty is bracketing: use two different evaluators – one optimistic, one pessimistic – to get upper and lower bounds on an action’s true value. This approach explicitly acknowledges uncertainty. For example, an AI bidding system might use a perfect-information solver to estimate the best-case expected score for a bid (assuming everything falls favorably and we could adjust play to the actual layout), and simultaneously use a minimax or very conservative evaluator to estimate the worst-case outcome (assuming the least favorable layout and optimal defense by opponents). Suppose for a certain bid, the oracle says P(make) ≈ 80% (if you could always choose the right line of play for the actual distribution), but the pessimistic model says P(make) ≈ 50% (if the opponents have a devious lie of cards or act optimally against you). Then the truth, with best play under uncertainty, likely lies in between – perhaps 65%. The gap between 50% and 80% is a measure of uncertainty or robustness gap. Is the term “Robustness Gap” standard? Not precisely by that name. However, related concepts exist: in game theory we speak of exploitability gap between a strategy and an equilibrium (how much worse it does against a perfect adversary than against a standard opponent). In robust optimization, we quantify the cost of hedging by the difference between robust and nominal objective values (often called the price of robustness). And as noted earlier, decision analysts use EVPI to measure the value of having perfect information – which is essentially the optimistic evaluation minus the actual optimal value without that info. All these are conceptually robustness gaps. So while you might not find “Robustness Gap” defined in a textbook, the idea of an upper-lower bound difference as a metric of uncertainty is well-established. Using Brackets in Practice: By presenting both an optimistic and pessimistic evaluation, the system can communicate the range of possible outcomes. For instance, a planning tool might say: “Action A yields an expected reward of 100 in the best case, but could be as low as 60 in the worst case; Action B ranges from 90 to 85.” A decision-maker seeing this knows Action B is more robust (narrow range) whereas Action A has higher upside but larger downside. In game-playing AI, bracketing can be internal: the AI chooses the move whose lower-bound value is highest if it’s being risk-averse, or maybe uses the upper-bound to push for wins when needed. Some bridge programs effectively do this when deciding whether to bid a thin game contract – they consider the double-dummy tricks (upper bound) and a more conservative estimate. If the gap is too large, they might avoid the bid unless the match situation demands a swing. Bounds with Perfect vs Imperfect Evaluators: One can obtain a lower bound by restricting the evaluator to not use hidden info. For example, evaluate the action assuming you do not know the hidden cards and must fix a single strategy. This imperfect-information evaluation might be done via a simplified game model or even self-play simulations. It’s guaranteed not to overestimate the value, so it’s a plausible lower bound. The perfect-information evaluation provides an upper bound (often an unattainable one, but useful for reference). If these two are close, it means the decision is robust: even with full information, you wouldn’t do much better than the strategy you have to choose now, so PIMC bias is low. If they diverge widely, it flags that the decision is information-sensitive. Frank & Basin’s work can be seen in this light: they effectively constructed scenarios where perfect-info value and actual value diverge greatly, to illustrate the pathology
webdocs.cs.ualberta.ca
webdocs.cs.ualberta.ca
. Reporting the Gap: In some cases, the system may explicitly report something like a “robustness index” or confidence interval. For example, quantiles can be reported: “Under plausible deals, the 25% worst-case outcome for this bid is -20 points, the median is +10, and the 90% best-case is +50.” Such reporting corresponds to bracketing the distribution of outcomes, not just expectation. It gives the user an idea of risk. A large interquantile range would signal high uncertainty. If a user is risk-averse, they might focus on the lower quantile (a kind of robust bound), whereas a risk-seeker might eye the upper quantile. As for whether the “Robustness Gap” is a standard metric, we can conclude: it’s not typically a single named metric in literature, but it is a natural derivative of doing bracketing analysis. In research papers, one might present both the optimistic and pessimistic performance of an algorithm to demonstrate its reliability. For example, a planner might be evaluated on worst-case scenario regret in addition to average reward, effectively highlighting a robustness gap. In game competitions, especially in imperfect info games, strategies are often evaluated by their worst-case opponent outcome (safety) and their performance against a field of typical opponents (average-case). The difference is essentially what we’re calling the robustness gap. So while you won’t see a formula labeled “RobustnessGap = X”, the concept is pervasive. Standard Metrics Related to Robustness: Just to connect terminology: Exploitability (in zero-sum games) is one standard measure – it measures how much worse a strategy is against a perfect adversary than against a best-response to itself (zero exploitability means fully robust, like a Nash equilibrium). Maximin value vs expected value is another – the maximin is the guaranteed payoff (worst-case), and the difference between expected payoff (against some distribution) and maximin is essentially how much you rely on the opponent not playing the worst case. In partial info planning, probability of success (P(make)) vs confidence level trade-offs serve a similar role – e.g., “I can achieve an 80% success plan if I accept 10% risk of failure, but if I want 99% success guarantee, my plan’s value drops (robustness cost).” Bracketing techniques explicitly put these considerations at the forefront, making the system safer and more transparent. By always considering both a optimistic lens (often easier to compute with an oracle) and a pessimistic lens (which may come from a simpler heuristic or assumption of adversary), the decision-maker or AI can avoid being blindsided by the oracle’s rose-colored glasses. Instead, the truth is understood to lie somewhere between the two evaluations, and one can drive that gap down with better strategies or by acquiring more information (which, interestingly, is the only real way to eliminate it – hence the value of information notion).
Output Formatting and Confidence Reporting
Finally, we consider how a system should report its decisions or evaluations in a way that reflects the uncertainties and the analysis above. Clear output formatting and confidence measures are crucial for trust and effective use of AI recommendations, especially in imperfect-information contexts where no answer can be absolute. Probability of Success (P(make)): In many partial-information decision problems (like bidding a contract, or choosing a strategy with a success/failure outcome), the most straightforward metric to report is the probability of success. For example, an AI bidding assistant could say “I estimate a 72% chance of making this contract.” This single number is intuitive and allows the human to weigh risk vs reward. It is essentially the expected value in binary success terms. However, it’s important that the system also communicate assumptions behind that probability (e.g., “assuming average card distribution among the opponents”). If bracketing was done, the system might add “...worst-case success probability 40% if cards lie poorly.” Including context with P(make) prevents the user from treating it as a guarantee. Expected Value (Score): In scenarios where outcomes have a range of values (not just success/fail), the expected score or payoff is often reported. For instance, a poker bot might evaluate a bet and conclude it has an +5 chip expected value. Expected value (EV) is the criterion most rational decision frameworks optimize, but by itself it hides variability. Two actions can have the same EV, one being a sure thing and the other a gamble. Thus, expected value is best paired with some measure of dispersion. Quantiles and Ranges: To communicate the risk or variance, systems can report quantiles of the outcome distribution. For example: “If you take this action, in 90% of cases you will score at least 0 (break even), but there’s a 10% chance of losing 50 points; on the upside, 10% of the time you’d gain 50 or more.” Such statements give a fuller picture. Sometimes a simplified range is given: “Outcome will be between -50 and +100 in likely scenarios.” One must be careful not to overwhelm the user, but key percentiles (like 10th, median, 90th) or a confidence interval can greatly aid understanding. In decision support tools (e.g., business decision analysis software), it’s standard to show a distribution or at least a high-low error bar for each option. Robust Bounds / Worst-Case: In high-confidence settings, the system should state robust guarantees if available: “Guaranteed at least 3 tricks no matter how the cards lie” or “Worst-case outcome: you lose 20 points.” This appeals to risk-averse users and is important when the user might want to minimize downside. Even if the system doesn’t optimize for worst-case, knowing the worst-case outcome of its recommendation is valuable. If the worst-case is unacceptable to the user, they might override the suggestion. Providing this info aligns with safe AI principles – it’s a form of transparency about what happens if Murphy’s Law strikes. When to Report What: Not every situation warrants an exhaustive statistical report. The system could adapt its output detail to the decision criticality and the robustness gap. For routine decisions where the optimistic and pessimistic evaluations coincide (low gap), a single metric (like expected value or win probability) might suffice – the system is essentially confident. When there is a large gap or high uncertainty, the system should reveal more: e.g., “I’m less certain here: by one analysis your chances are 70%, but under a tough defense it could be only 50%.” This conditional reporting ensures the user isn’t lulled by a precise number that belies high variance. Examples in Practice: In chess engines (perfect information, but analogous reporting), modern UIs often show win probability rather than just a raw score, because win probability is easier to interpret by humans. In bridge software or human bridge play, players talk about percentages (“this line of play makes 60% of the time”). A future AI assistant might explicitly output, “Bid 4♠ – roughly 65% chance to make, with an average score of +400; if the trumps split badly (worst 10% case), you could go down 1.” This kind of explanation not only gives the recommendation but also educates the user on the risk factors (here, bad trump split). From an academic perspective, researchers like to report confidence bounds on their AI’s performance: e.g., an AI might report its move along with a confidence level (some systems using MCTS can output an intrinsic confidence, like how convincingly one move outranked others in simulations). If a system is unsure (say two moves had nearly equal evaluations), it might communicate that: “This is a close call – two options are effectively tied in my analysis.” That is much better than arbitrarily picking one and conveying false certainty. In summary, good output formatting for hidden-information decisions should include: (a) a recommended action, (b) a primary metric (probability of success or expected value) for that action, and (c) at least one form of confidence or uncertainty measure (range, quantile, or a comparison of optimistic vs pessimistic outcomes). By reporting these, the system aligns with how humans think about uncertainty (we like to know best-case, worst-case, and likelihood of success). It also mirrors the internal analysis done via the methods discussed: if we went to the trouble of bounding and robustifying evaluations, we should expose that to the end-user so they can make an informed decision or trust the AI appropriately.
Sources: The insights above draw from a range of literature on game AI and decision theory. Frank & Basin’s seminal critique of determinization
webdocs.cs.ualberta.ca
webdocs.cs.ualberta.ca
 defined the strategy fusion and non-locality issues, while Sturtevant et al. explored why PIMC sometimes succeeds despite these issues
webdocs.cs.ualberta.ca
webdocs.cs.ualberta.ca
. Techniques like ISMCTS
arxiv.org
 and delayed revelation (EPIMC)
arxiv.org
arxiv.org
 were developed to address these pathologies. Parker et al.’s work on overconfidence vs paranoia in gameplay provides empirical evidence for the efficacy of average-case modeling over worst-case in many scenarios
academia.edu
. The rise of game-theoretic algorithms in poker and their integration with search (DeepStack, ReBeL, Player of Games) demonstrate modern methods to maintain consistency across hidden information
frontiersin.org
highstakesdb.com
. Finally, concepts from decision analysis (like EVPI and quantile reporting) inform how we interpret and communicate the gap between optimistic and robust evaluations. These sources collectively underline the central message: Imperfect-information decision making benefits from a careful balance – using powerful perfect-information oracles to push performance, while anchoring them with reality checks, robust policies, and clear communication of uncertainty.
Citations
Understanding the Success of Perfect Information Monte Carlo Sampling in Game Tree Search

https://webdocs.cs.ualberta.ca/~nathanst/papers/pimc.pdf
Understanding the Success of Perfect Information Monte Carlo Sampling in Game Tree Search

https://webdocs.cs.ualberta.ca/~nathanst/papers/pimc.pdf
Understanding the Success of Perfect Information Monte Carlo Sampling in Game Tree Search

https://webdocs.cs.ualberta.ca/~nathanst/papers/pimc.pdf
Understanding the Success of Perfect Information Monte Carlo Sampling in Game Tree Search

https://webdocs.cs.ualberta.ca/~nathanst/papers/pimc.pdf
Understanding the Success of Perfect Information Monte Carlo Sampling in Game Tree Search

https://webdocs.cs.ualberta.ca/~nathanst/papers/pimc.pdf
Understanding the Success of Perfect Information Monte Carlo Sampling in Game Tree Search

https://webdocs.cs.ualberta.ca/~nathanst/papers/pimc.pdf
Understanding the Success of Perfect Information Monte Carlo Sampling in Game Tree Search

https://webdocs.cs.ualberta.ca/~nathanst/papers/pimc.pdf
Understanding the Success of Perfect Information Monte Carlo Sampling in Game Tree Search

https://webdocs.cs.ualberta.ca/~nathanst/papers/pimc.pdf
Understanding the Success of Perfect Information Monte Carlo Sampling in Game Tree Search

https://webdocs.cs.ualberta.ca/~nathanst/papers/pimc.pdf
Understanding the Success of Perfect Information Monte Carlo Sampling in Game Tree Search

https://webdocs.cs.ualberta.ca/~nathanst/papers/pimc.pdf
Understanding the Success of Perfect Information Monte Carlo Sampling in Game Tree Search

https://webdocs.cs.ualberta.ca/~nathanst/papers/pimc.pdf
Understanding the Success of Perfect Information Monte Carlo Sampling in Game Tree Search

https://webdocs.cs.ualberta.ca/~nathanst/papers/pimc.pdf
Understanding the Success of Perfect Information Monte Carlo Sampling in Game Tree Search

https://webdocs.cs.ualberta.ca/~nathanst/papers/pimc.pdf
Understanding the Success of Perfect Information Monte Carlo Sampling in Game Tree Search

https://webdocs.cs.ualberta.ca/~nathanst/papers/pimc.pdf
Understanding the Success of Perfect Information Monte Carlo Sampling in Game Tree Search

https://webdocs.cs.ualberta.ca/~nathanst/papers/pimc.pdf
Understanding the Success of Perfect Information Monte Carlo Sampling in Game Tree Search

https://webdocs.cs.ualberta.ca/~nathanst/papers/pimc.pdf
Understanding the Success of Perfect Information Monte Carlo Sampling in Game Tree Search

https://webdocs.cs.ualberta.ca/~nathanst/papers/pimc.pdf
Understanding the Success of Perfect Information Monte Carlo Sampling in Game Tree Search

https://webdocs.cs.ualberta.ca/~nathanst/papers/pimc.pdf
Understanding the Success of Perfect Information Monte Carlo Sampling in Game Tree Search

https://webdocs.cs.ualberta.ca/~nathanst/papers/pimc.pdf
Understanding the Success of Perfect Information Monte Carlo Sampling in Game Tree Search

https://webdocs.cs.ualberta.ca/~nathanst/papers/pimc.pdf
Understanding the Success of Perfect Information Monte Carlo Sampling in Game Tree Search

https://webdocs.cs.ualberta.ca/~nathanst/papers/pimc.pdf

https://www.arxiv.org/pdf/2408.02380

https://www.arxiv.org/pdf/2408.02380

https://www.arxiv.org/pdf/2408.02380

https://www.arxiv.org/pdf/2408.02380

https://www.arxiv.org/pdf/2408.02380

https://www.arxiv.org/pdf/2408.02380

https://www.arxiv.org/pdf/2408.02380

https://www.arxiv.org/pdf/2408.02380
Understanding the Success of Perfect Information Monte Carlo Sampling in Game Tree Search

https://webdocs.cs.ualberta.ca/~nathanst/papers/pimc.pdf
Understanding the Success of Perfect Information Monte Carlo Sampling in Game Tree Search

https://webdocs.cs.ualberta.ca/~nathanst/papers/pimc.pdf
Understanding the Success of Perfect Information Monte Carlo Sampling in Game Tree Search

https://webdocs.cs.ualberta.ca/~nathanst/papers/pimc.pdf

https://www.arxiv.org/pdf/2408.02380

Frontiers | AlphaZe∗∗: AlphaZero-like baselines for imperfect information games are surprisingly strong

https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2023.1014561/full

HighStakesDB - Google’s “Player of Games” Ready to Conquer All-Comers

https://highstakesdb.com/news/high-stakes-reports/google-s-player-of-games-ready-to-conquer-all-comers

HighStakesDB - Google’s “Player of Games” Ready to Conquer All-Comers

https://highstakesdb.com/news/high-stakes-reports/google-s-player-of-games-ready-to-conquer-all-comers
[PDF] Integrated task and motion planning in belief space

https://lis.csail.mit.edu/pubs/tlp/IJRRBelFinal.pdf
[PDF] Pre-image backchaining in belief space for mobile manipulation

http://www.isrr-2011.org/ISRR-2011/Program_files/Papers/Kaelbling-ISRR-2011.pdf

(PDF) kriegspiel program - Academia.edu

https://www.academia.edu/157915/kriegspiel_program

Overconfidence or paranoia? search in imperfect-information games

https://dl.acm.org/doi/10.5555/1597348.1597355

https://www.arxiv.org/pdf/2408.02380

https://www.arxiv.org/pdf/2408.02380

[2007.13544] Combining Deep Reinforcement Learning and Search for Imperfect-Information Games

https://arxiv.org/abs/2007.13544

[2007.13544] Combining Deep Reinforcement Learning and Search for Imperfect-Information Games

https://arxiv.org/abs/2007.13544
Understanding the Success of Perfect Information Monte Carlo Sampling in Game Tree Search

https://webdocs.cs.ualberta.ca/~nathanst/papers/pimc.pdf
All Sources
webdocs.cs.ualberta

arxiv

fro