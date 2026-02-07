# Stage 2 Training: Posterior-Weighted Information-Set E[Q] from a Perfect-Info Q Oracle

> **Quick start**: For essential commands and directory overview, see the [E[Q] section in forge/ORIENTATION.md](../forge/ORIENTATION.md#eq-training-pipeline-stage-2).

This document is the **research-grade Stage 2 training-data spec** for `forge/eq/*` (dataset schema `2.1`). The intended audience is ML researchers working on imperfect-information decision making and value learning.

## 0. Executive Summary

We want a deployable **Stage 2** model that plays Texas 42 well under imperfect information by learning an **information-set action-value function**:

\[
f_\theta(I_t)\; \approx\; \mu_t(a) = \mathbb{E}_{w \sim p(w \mid I_t)}\left[Q^\star(w, a)\right]\quad\text{for each legal move }a
\]

Where:
- \(I_t\) is the public transcript + the acting player’s current hand (i.e., the player’s information set in tournament 4-player 42).
- \(w\) is a completion of hidden hands (a “world”).
- \(Q^\star(w,a)\) is the **perfect-information** team-frame action value from the frozen Stage 1 oracle at the current decision.

We generate supervised targets by sampling many worlds consistent with public constraints (voids + remaining tiles), querying Stage 1 to get \(Q\) in each world, and aggregating into:
- **Mean** \(\mu(a)=\mathbb{E}_w[Q(a)]\)
- **Variance** \(\sigma^2(a)=\mathrm{Var}_w[Q(a)]\) (uncertainty signal)

Important semantic pin (Stage 1 / oracle):
- \(Q^\star\) is a **minimax value-to-go** in *points* (remaining Team0 − Team1 point differential) in \([-42, 42]\), as defined by the solver/oracle schema (`forge/oracle/schema.py`). It is not a probability and it is not “score so far”.

Critically, we support **posterior-weighted** aggregation: worlds are reweighted by how likely they make the recent transcript under an explicit behavior model (a Boltzmann/advantage policy derived from Stage 1, with always-on uniform mixing for robustness). This upgrades “uniformly average over consistent worlds” into a Bayes-shaped importance sampler over hidden hands.

Primary code:
- **GPU generation (recommended)**: `forge/eq/generate_gpu.py`, `forge/eq/generate_pipelined.py`, `forge/cli/generate_eq_continuous.py`
- **CPU generation (debug/reference)**: `forge/eq/generate.py`, `forge/eq/generate_dataset.py`
- World sampling: `forge/eq/sampling.py`, `forge/eq/sampling_gpu.py`, `forge/eq/voids.py`
- Stage 1 oracle wrapper: `forge/eq/oracle.py`
- Stage 2 (public) tokenizer: `forge/eq/transcript_tokenize.py`, `forge/eq/tokenize_gpu.py`
- Posterior weighting: `forge/eq/posterior.py`, `forge/eq/posterior_gpu.py`
- Debugging: `forge/eq/viewer.py`, `scripts/analyze_eq_predictions.py`, `scripts/validate_eq_computation.py`

## 1. Evolution from MVP

`docs/EQ_MVP.md` (now a redirect to this document) was the original feasibility validation. This spec differs in several material ways:

1. **Targets are point-valued E[Q], not E[logits].**
   - We now use Q-value models (see `forge/models/README.md`) so outputs are interpretable in points.
2. **Stage 2 labels include uncertainty.**
   - We emit \(\mu(a)\) and \(\sigma^2(a)\) per move, plus state-level scalars \(U_{\text{mean}}, U_{\text{max}}\).
3. **Posterior weighting is first-class.**
   - Worlds are reweighted by transcript likelihood under an explicit behavior model (importance sampling / particle filtering).
4. **Transcript generation is intentionally stochastic.**
   - We generate data from a mixed exploration policy so Stage 2 learns robust play under non-oracle transcripts.

The rest of this doc describes the implemented approach and the knobs we can tune to improve data quality.

## 2. Stage Definitions (for this project)

**Stage 1 (exists):** perfect-information action-value oracle.
- Input: all 4 hands + current trick state
- Output: \(Q(a)\) for the acting player’s 7 initial-hand slots
- Semantics: expected remaining point differential **in the acting player’s team frame** (see §4.3).
- Wrapper: `forge/eq/oracle.py::Stage1Oracle`
- Recommended checkpoint: `forge/models/domino-qval-large-3.3M-qgap0.071-qmae0.94.ckpt`

**Stage 2 (this dataset):** imperfect-information value/policy model.
- Input: public transcript + current hand (Stage 2 tokenizer)
- Output: \(\mu(a)\) (and optionally \(\sigma^2(a)\)) for each action slot
- Deployment: single forward pass + legal mask + `argmax`

**Stage 3 (planned):** explicit cross-hand / signaling / partner modeling.
- Not covered here; Stage 2 aims to be a strong “human-facing” baseline player first.

### 2.1 The Stage 2 Information Set \(I_t\): Token Contract and Completeness

In this repo, Stage 2 is trained on *public information + the acting player’s private hand* **at play time** (no bidding/marks/scoring context is modeled in `forge/eq/*` today).

Concretely, the implemented \(I_t\) representation is the output of:
- `forge/eq/transcript_tokenize.py::tokenize_transcript(my_hand, plays, decl_id, current_player)`

It contains:
- `decl_id` (public)
- the acting player’s **current remaining hand** as a set/sequence of domino ids (private)
- the **ordered** play transcript so far as `(abs_player, domino_id)` pairs (public), converted into a relative player id `rel_player = (abs_player - current_player) mod 4`

Token fields (8 features per token) are described in `forge/eq/transcript_tokenize.py`:
- domino pips + count value (public)
- relative player id (public, perspective-dependent)
- `is_in_hand` and `token_type` (structure)

**Completeness caveat (critical):** this representation is sufficient *only if the downstream Stage 2 model is order-aware* (positional encodings or explicit position features). Trick boundaries, current trick prefix, “who led”, and most void evidence are encoded via **sequence order** (e.g., `len(plays) % 4` determines the current trick length).

If we ever train a permutation-invariant encoder over these tokens, the effective \(I_t\) will be under-specified and the labels will appear irreducibly noisy.

## 3. What We Are Learning: Information-Set Q, Not Perfect-Info Q

Texas 42 is imperfect information: opponent hands are hidden, but actions reveal constraints (voids, forced follows, etc.). A model trained only on perfect-information oracle trajectories will overfit to “god-view” and fail to learn the correct decision rule under uncertainty.

The Stage 2 training target is the information-set action-value vector:

\[
\mu_t(a) = \mathbb{E}_{w \sim p(w \mid I_t)} \left[ Q^\star(w, a) \right]
\]

This is not “strategy fusion”: we are not selecting the best action inside each world and then averaging. We compute an expectation **per action** and let the downstream policy choose `argmax` on \(\mu(a)\).

## 4. Data Generation Algorithm

Canonical entrypoint:
- `python -m forge.eq.generate_dataset ...` (`forge/eq/generate_dataset.py`)

Each generated game yields 28 play-decisions (4 players × 7 tricks). Each decision record is created at runtime inside:
- `forge/eq/generate.py::generate_eq_game`

### 4.1 Game Loop and Decision Points

At each decision in a simulated game:

1. Maintain public constraints:
   - Void suits inferred online from failed follows (`can_follow(...)`).
   - Which dominoes have been played.
2. Sample hidden-hand completions (“worlds”) consistent with constraints.
3. Reconstruct a full hypothetical **initial** deal per world (Stage 1 expects fixed 7-slot hands).
4. Query the Stage 1 oracle for per-world \(Q\) at the **current** decision.
5. Compute weights:
   - Uniform: \(w_i = 1/N\)
   - Posterior-weighted: \(w_i \propto \exp(\log p(\text{recent transcript} \mid w_i))\)
6. Aggregate into \(\mu(a)\) and \(\sigma^2(a)\) for each remaining domino.
7. Encode Stage 2 input tokens from public info.
8. Choose the next action (greedy or stochastic exploration policy) to advance the transcript.
9. Record the example.

Note on “decision record” vs “learning signal”:
- Many positions are *forced* (`|A_legal| = 1`). The current pipeline records them (they are valid supervised pairs), but they are low-information for learning action selection. Whether to filter or downweight forced examples is a Stage 2 training design choice (see §5.1 and §6.4).

### 4.2 World Sampling: Hard Constraints First

World sampling is **not** rejection sampling; it is backtracking with MRV (minimum-remaining-values) heuristic:
- `forge/eq/sampling.py::sample_consistent_worlds`

Inputs:
- `my_player`, `my_hand`: the acting player’s private hand at this step
- `played`: already-played domino ids
- `hand_sizes`: remaining hand sizes (shrinking through the hand)
- `voids`: hard void suits inferred from transcript so far
- `decl_id`, `n_samples`

Output:
- `remaining_worlds`: list of N worlds, each world is `[hand0, hand1, hand2, hand3]` containing **remaining** domino ids for each seat.

This guarantees a feasible assignment whenever constraints are consistent (the real game state is a witness), and fails loudly if void tracking becomes inconsistent.

### 4.3 Reconstructing “Initial Hands” and Local-Index Semantics

Stage 1’s state space is built on **fixed 7-slot local indices** per player (0–6), matching the oracle parquet schema (`forge/oracle/schema.py`).

But `sample_consistent_worlds(...)` produces *remaining* hands only. To query Stage 1 we reconstruct, for each world and player \(p\):

\[
\text{initial\_hand}_p \;=\; \mathrm{sort}\left(\text{remaining\_hand}_p \;\cup\; \text{played\_by}_p\right)
\]

Implementation:
- `forge/eq/generate.py::_build_hypothetical_worlds_batched`

We also construct `remaining` bitmasks in the oracle’s local-index space:
- `remaining[world, p]` is a 7-bit mask indicating which local slots are still in-hand at the current step.

**Actor-frame contract (critical):** Stage 1 was trained to output \(Q\) in the **acting player’s team frame** (Team0 players maximize \(Q\); Team1 players maximize \(-Q\)). This is consistent with Stage 1 tokenization/training (`forge/ml/tokenize.py` flips targets by `team = current_player % 2`) and Stage 1 loss (`forge/ml/module.py` flips Q for team1).

Operationally:
- `oracle.query_batch(..., current_player=actor)` must return a vector where `argmax` is the actor’s preferred action.

### 4.4 Uniform E[Q] (Baseline)

Given per-world oracle outputs \(Q_i(\cdot)\), we compute:

\[
\mu(a) = \frac{1}{N} \sum_{i=1}^N Q_i(a)
\qquad
\sigma^2(a) = \frac{1}{N}\sum_{i=1}^N Q_i(a)^2 - \mu(a)^2
\]

This is implemented in `forge/eq/generate.py` by mapping each world’s 7-slot outputs back to the current remaining-hand domino ids and aggregating.

### 4.5 Posterior-Weighted E[Q]

Uniform averaging treats all constraint-satisfying worlds as equally likely. This is maximally ignorant once the transcript contains signal. We support importance weights derived from transcript likelihood:

\[
w_i \;\propto\; \exp\left(\sum_{j \in \mathcal{W}} \log \pi_{\text{beh}}(a_j \mid I_j(w_i)) \right)
\]

Where \(\mathcal{W}\) is a sliding window of the last \(K\) plays and \(I_j(w_i)\) is the **actor’s information set at time \(j\)** under world \(w_i\).

Implementation:
- `forge/eq/generate.py::compute_posterior_weights`
- `forge/eq/generate.py::_score_step_likelihood`

With normalized weights \(w_i\) (summing to 1), we aggregate targets as:
\[
\mu(a) = \sum_i w_i Q_i(a)
\qquad
\sigma^2(a) = \sum_i w_i Q_i(a)^2 - \mu(a)^2
\]
(i.e., mean and variance are computed under the same posterior weights).

#### 4.5.0 Prefix Reconstruction Contract (Likelihood Scoring)

Posterior weighting is only meaningful if, for each scored step \(j\), we reconstruct the actor’s information set at time \(j\) correctly under each world \(w_i\).

Given the public prefix `play_history[:j]`, `_score_step_likelihood(...)` deterministically reconstructs:
- `played_before_j`: set of domino ids played before \(j\)
- `trick_plays_j`: the current trick prefix (plays since the most recent lead)
- `leader_j`: the leader for this trick (actor if this step is the lead)
- per-world `remaining_masks_j` in Stage 1’s local-index space (initial-hand slots not yet played)

Two invariants prevent “explaining the transcript via bugs”:
- Observed domino not in actor’s reconstructed initial hand ⇒ the world is inconsistent ⇒ assign log-prob `-inf`.
- Observed domino in-hand but illegal ⇒ a reconstruction/rules bug (should be loud in dev via `strict_integrity=True`).

Performance + correctness invariant (batching):
- Public trick plays are represented as `(player, domino_id)` (world-invariant public ids), never `(player, local_idx)` (world-dependent). This is required both to avoid mapping drift and to preserve batched oracle queries across worlds.

#### 4.5.1 Behavior Model: Advantage-Softmax + Always-On Uniform Mixing

Because we have no human logs, the “posterior” is only defined relative to an assumed behavior model. We use a bounded-rational policy derived from Stage 1:

1. For step \(j\), query Stage 1 for the actor’s legal action values \(Q(\cdot)\).
2. Convert to advantage-like logits:
   \[
   A(a) = Q(a) - \mathrm{mean}_{a' \in \mathcal{A}_{\text{legal}}} Q(a')
   \]
3. Softmax with temperature \(\tau\):
   \[
   p_{\text{soft}}(a) = \mathrm{softmax}(A/\tau)
   \]
4. Robust mixture with uniform over legal moves (model-mismatch tolerance):
   \[
   \pi_{\text{beh}}(a) = (1-\beta)p_{\text{soft}}(a) + \beta \cdot \mathrm{Unif}(\mathcal{A}_{\text{legal}})
   \]

Parameters:
- \(\tau\) controls sharpness (higher → more human-like stochasticity; lower → “oracle-like” determinism).
- \(\beta\) prevents brittle posteriors when the behavior model is mis-specified.

Numerical stability:
- We use a small `+1e-30` inside `log()` to avoid \(\log(0)\); \(\beta\) is the semantic robustness knob.

#### 4.5.2 Sliding Window (K) and Weight Stabilization (Δ)

We compute log weights over a recent window of size \(K\) (default 8):
- Controls both compute and posterior sharpness.
- Avoids “posterior collapse” from multiplying many small probabilities.

We stabilize log weights per decision:
- subtract `max(logw)`
- clip to `[-Δ, 0]` (default Δ=30)
- normalize via `softmax(logw)`

This turns extreme likelihood ratios into a bounded concentration level.

#### 4.5.3 Mapping Integrity Checks (Non-Negotiable)

Per-step, per-world:
- If observed domino is not in the actor’s (reconstructed) initial hand → that world is inconsistent with public transcript; set log-prob to \(-\infty\).
- If observed domino is in-hand but not legal given reconstructed trick state → this indicates a rules/state reconstruction error; the code can raise (`strict_integrity=True`) or count and continue.

This avoids silently “explaining” the transcript by bugs in local-index mapping or trick reconstruction.

### 4.6 Stage 2 Input Encoding (Public Information Only)

Tokenizer:
- `forge/eq/transcript_tokenize.py::tokenize_transcript`

Sequence structure:
- 1 declaration token
- up to 7 current-hand tokens (domino features + `is_in_hand=1`)
- all play-history tokens so far (domino features + relative player id + `is_in_hand=0`)

Features (8 dims):
- high pip, low pip, is_double, count_value, relative_player, is_in_hand, decl_id, token_type

The key constraint: Stage 2 never sees opponent hands.

### 4.7 Transcript Generation Policy (Exploration) and Distribution Design

The dataset distribution is shaped by how we generate transcripts. If we always play greedy `argmax(E[Q])`, we will:
- undersample “off-policy” lines that humans take,
- reduce diversity of information patterns,
- potentially bias posterior weighting toward overly-optimal histories.

We therefore support a stochastic exploration policy:
- `forge/eq/generate.py::ExplorationPolicy`

Mechanisms (applied in order):
1. **Blunder** (bounded regret): occasionally choose a suboptimal move within `blunder_max_regret` points.
2. **Epsilon**: choose a random legal move with probability `epsilon`.
3. **Boltzmann**: sample from `softmax(E[Q]/temperature)` if enabled; otherwise greedy.

We log exploration metadata per decision:
- mode (`greedy|boltzmann|epsilon|blunder`)
- `q_gap = Q_greedy - Q_taken`
- action entropy of the (temperature=1) softmax over legal moves

Known interaction (important when posterior weighting is enabled):
- Transcript generation intentionally includes off-greedy/off-oracle actions (exploration). The posterior likelihood model is derived from Stage 1’s action preferences. Without robustness (β and logw clipping), the posterior can “explain” exploration noise by shifting hidden hands. This is expected; tune β/τ/K/Δ with that coupling in mind.

## 5. What Goes Into the Saved Dataset

Generator:
- `forge/eq/generate_dataset.py`

Output is a PyTorch `.pt` dict (CPU tensors) with keys:
- `transcript_tokens`: `(N, MAX_TOKENS, 8)` int tensor (padded)
- `transcript_lengths`: `(N,)` lengths before padding
- `e_q_mean`: `(N, 7)` float tensor; padded with `-inf` beyond current hand size
- `legal_mask`: `(N, 7)` bool tensor; padded `False` beyond current hand size
- `action_taken`: `(N,)` int tensor; index into the **current hand order** at that step
- `game_idx`, `decision_idx`: indices for grouping/analysis
- `train_mask`: per-example split mask (train vs val), consistent within a game

Uncertainty:
- `e_q_var`: `(N, 7)` per-action variance (0 for padded slots)
- `u_mean`, `u_max`: `(N,)` state-level uncertainty scalars computed over legal actions

Posterior diagnostics (persisted subset):
- `ess`: `(N,)` effective sample size
- `max_w`: `(N,)` max particle weight
Additional diagnostics are computed during generation (e.g., entropy / effective world count, window NLL, invalid/illegal counts, mitigation flags) but are not yet fully persisted as dataset fields.

Exploration metadata (persisted):
- `exploration_mode`: `(N,)` int8 encoding of selection mode
- `q_gap`: `(N,)` regret of chosen action vs greedy baseline

Global metadata:
- `metadata`: versioned config + summary stats (Q range, ESS distribution, exploration rates, checkpoint path, etc.)

### 5.0 Dataset Semantics (Schema v2.1)

The saved dataset includes explicit, machine-readable semantics:

- `metadata["version"] == "2.1"`
- `metadata["schema"]["q_semantics"] == "minimax_value_to_go"`
- `metadata["schema"]["q_units"] == "points"` and `metadata["schema"]["q_normalization"] == "raw"`
- `metadata["schema"]["tokenizer_version"] == "transcript_v1"`
- `metadata["posterior"]` and `metadata["exploration"]` record generation-time knobs

These fields are intended to make it difficult to accidentally treat `e_q_mean` as policy logits (e.g., by applying softmax).

### 5.1 Stage 2 Training Contract (Consumer Guidance)

This repository does not yet contain the Stage 2 model implementation, but the dataset is designed to support a clean supervised learning contract:

**Inputs**
- `transcript_tokens` + `transcript_lengths` define a variable-length public-information sequence from the acting player’s perspective.

**Targets**
- `e_q_mean` is the per-slot \(\mu(a)=\mathbb{E}[Q(a)]\) in **points** (actor-team frame).
- `legal_mask` identifies which action slots are legal at that state.
- Optional: `e_q_var`, `u_mean`, `u_max` expose \(\sigma^2(a)\) and aggregate uncertainty scalars.

**Recommended baseline loss (mean-only)**
\[
\mathcal{L}_{\mu} = \frac{1}{|\mathcal{A}_{\text{legal}}|}\sum_{a} \mathbf{1}[a \in \mathcal{A}_{\text{legal}}]\;\left(\hat{\mu}(a) - \mu(a)\right)^2
\]

Notes:
- Masking to legal moves is the conservative choice: Stage 1/solver semantics only define \(Q\) on legal actions. (The tensor contains values for in-hand-but-illegal slots, but those are not decision-relevant at that state.)
- Normalize targets to a stable range for training (e.g., divide by 42.0 to map to roughly \([-1,1]\)), mirroring Stage 1’s q-value loss scaling in `forge/ml/module.py`.

**Optional uncertainty-aware training (mean + variance)**
- Treat `e_q_var` as a *predictable property of the information set* (epistemic variability across hidden hands), not merely noise.
- A simple approach is a second head predicting \(\log \hat{\sigma}^2(a)\) with a Gaussian NLL on legal moves:
\[
\mathcal{L} = \sum_{a \in \mathcal{A}_{\text{legal}}}\left(\frac{(\hat{\mu}(a)-\mu(a))^2}{\exp(\log \hat{\sigma}^2(a))} + \log \hat{\sigma}^2(a)\right)
\]
- Alternatively, train \(\hat{\sigma}^2(a)\) directly via MSE to `e_q_var` and keep the mean loss separate. (This is less principled but often stable.)

**Policy head (optional)**
- `action_taken` can supervise an auxiliary policy head (behavior cloning from the exploration policy), but the primary objective remains learning \(\mu(a)\) for downstream `argmax` play.

### 5.2 Stage 2 Model Design Considerations (Not Decided Yet)

We have not selected a Stage 2 architecture yet. However, the dataset design implies non-negotiable modeling requirements and a few clear tradeoffs:

- **Order-awareness is required** unless we enrich tokens: trick boundaries, legality context, and void evidence are encoded via transcript order.
- **Output structure**: the cleanest interface is to predict per-hand-slot \(\hat{\mu}(a)\) (and optionally \(\log \hat{\sigma}^2(a)\)) by reading the embeddings of the 7 hand tokens.
- **Fixed canvas vs variable length**: current storage pads to a fixed `MAX_TOKENS`, but tokenization is variable-length; we can choose an architecture that uses explicit masks, or redesign the tokenizer into a fixed-layout canvas (decl + 7 hand slots + 28 play slots) with position features.
- **Forced-move handling**: forced positions can be included but downweighted (e.g., weight proportional to `(|A_legal|-1)`), or filtered at training time; this affects sample efficiency and calibration.

## 6. Tuning Knobs (“Levers”) and Expected Effects

The knobs below govern both statistical quality and compute cost.

### 6.1 Monte Carlo Resolution

- `n_samples` (N worlds per decision): reduces Monte Carlo noise in \(\mu,\sigma^2\) at the cost of oracle queries and sampling time.
  - Typical: 100 (fast), 200–1000 (research-grade / offline generation).

### 6.2 Posterior Weighting (Belief Shaping)

Key parameters (see `forge/eq/generate.py::PosteriorConfig` and `forge/eq/generate_dataset.py` CLI):

- `tau` (τ): temperature of advantage-softmax.
  - Higher τ → flatter behavior model → weights closer to uniform → less confident posterior.
  - Lower τ → sharper behavior model → more confident posterior; higher risk of “confident wrongness” if behavior model is mis-specified.

- `beta` (β): uniform mixture coefficient (always-on robustness).
  - Higher β → less sensitivity to behavior model; higher ESS; labels closer to uniform averaging.
  - Lower β → tighter posterior; more reliance on Stage 1 as behavior model.

- `window_k` (K): number of recent plays used for likelihood weighting.
  - Larger K → more transcript conditioning but greater weight degeneracy risk.
  - Smaller K → closer to uniform; cheaper.

- `delta` (Δ): log-weight clipping window.
  - Smaller Δ → bounded likelihood ratios → higher ESS but less posterior movement.
  - Larger Δ → allow sharper posteriors; increases collapse risk.

Practical default phase schedule (stability → sharpness), assuming no human logs:
- Phase A: τ=10, β=0.10, K=8, Δ=30
- Phase B: τ=8,  β=0.07, K=12, Δ=30
- Phase C: τ=6,  β=0.05, K=16, Δ=30

Advance phases only if:
- posterior recovery on synthetic rollouts is stable, and
- downstream decision benefit improves relative to uniform averaging.

### 6.3 Degeneracy Mitigation and Particle Health

We log (at least) `ess` and `max_w`. Interpretation:
- ESS near N: weights ~ uniform; transcript provides little additional information under the behavior model.
- ESS \(\ll N\): a few worlds dominate; label variance may increase; risk of biased training distribution if frequent.

Degeneracy mitigation exists in code:
- **Enabled by default** (when posterior weighting is enabled): ESS-triggered uniform re-mixing (a second-line safeguard on top of logw clipping + β).
- **Disabled by default**: adaptive window sizing and resample+rejuvenate. Treat these as research features until validated.

### 6.4 Transcript Distribution (Exploration)

Exploration controls (see `forge/eq/generate.py::ExplorationPolicy` and CLI flags in `forge/eq/generate_dataset.py`):

- `temperature` (Boltzmann exploration): increases action diversity smoothly.
- `epsilon`: injects uniform random actions; high variance but coverage improvement.
- `blunder_rate` + `blunder_max_regret`: models bounded human error without destroying learning signal.

General guidance:
- Keep exploration mild enough that the policy still wins; the goal is coverage, not chaos.
- Track `q_gap` distribution; constrain heavy tails (catastrophic blunders) unless intentionally studying robustness.

## 7. Validation and Debugging Workflow

### 7.1 Invariants / Sanity Checks

- No NaN/Inf in legal `e_q_mean`.
- Values are within plausible bounds (oracle Q is in \([-42, 42]\) by construction).
- `legal_mask` matches rules from `GameState.legal_actions()`.
- Posterior health: look at `ESS` distribution and `max_w` spikes.
- Uncertainty: `e_q_var >= 0`, and `u_mean/u_max` behave sensibly (increase in ambiguous mid-game positions).

The dataset generator prints summary statistics and checks for common failures:
- `forge/eq/generate_dataset.py`

### 7.2 Tools

- Viewer (interactive browsing): `python -m forge.eq.viewer path/to/dataset.pt`
  - Shows \(\mu\) and (if present) \(\sigma\) per legal move.
- Fresh recomputation spot-check: `scripts/validate_eq_computation.py`
  - Verifies dataset E[Q] aligns with fresh world sampling/oracle queries.
- Outcome correlation/cursor analysis: `scripts/analyze_eq_predictions.py`
  - Compares predicted E[Q] to realized hand outcomes (coarse but useful sanity).

### 7.3 Failure Modes Worth Watching

- **Mapping bugs**: local-index alignment, remaining masks, trick reconstruction. These can look like “posterior collapse” but are actually logic errors.
- **Confident wrongness**: ESS looks healthy, but behavior model is wrong. (We partially mitigate via β; richer diagnostics can be persisted if needed.)
- **Dataset bias**: if filtering/mitigation preferentially removes hard positions, Stage 2 trains on easy states and becomes overconfident in real play.

## 8. Relationship to Stage 3

Stage 2 targets are **single-agent information-set values** (conditioned on the actor’s private hand, not partner modeling). This is sufficient to produce a strong baseline player vs humans.

Stage 3 can build on Stage 2 by explicitly modeling:
- partner signaling value,
- belief updates conditioned on partner strategy,
- cross-hand coordination and conventions.

The dataset already emits uncertainty signals that should be useful for deciding when to infer vs when to force-play, and later for gating “signal-seeking” behavior.

## 9. Code Map (Quick Reference)

```
forge/eq/
├── Generation (GPU - recommended, ~100x faster)
│   ├── generate_gpu.py          # Batched game generation
│   ├── generate_pipelined.py    # Async pipelined generation
│   ├── generate_continuous.py   # Gap-filling continuous mode
│   └── collate.py               # GPU records → training examples
│
├── Generation (CPU - debug/reference)
│   ├── generate.py              # Single-game generation with labels
│   ├── generate_dataset.py      # Batch dataset CLI
│   └── generate_game.py         # Game simulation logic
│
├── GPU Pipeline
│   ├── game_tensor.py           # Tensorized game state
│   ├── sampling_gpu.py          # GPU world sampling
│   ├── sampling_mrv_gpu.py      # MRV heuristic on GPU
│   ├── posterior_gpu.py         # GPU posterior weighting
│   └── tokenize_gpu.py          # GPU transcript tokenization
│
├── Sampling & Posterior (CPU)
│   ├── sampling.py              # Backtracking sampler (MRV heuristic)
│   ├── posterior.py             # Posterior weighting
│   ├── voids.py                 # Void inference from play history
│   └── worlds.py                # World reconstruction utilities
│
├── Tokenization & Types
│   ├── transcript_tokenize.py   # Stage 2 tokenizer (public info only)
│   ├── types.py                 # Type definitions
│   ├── exploration.py           # Exploration policy definitions
│   └── reduction.py             # E[Q] aggregation
│
├── Support
│   ├── oracle.py                # Stage1Oracle wrapper for batch queries
│   ├── game.py                  # GameState for simulation
│   ├── outcomes.py              # Game outcome tracking
│   └── rejuvenation.py          # Particle rejuvenation (experimental)
│
└── Analysis & Debugging
    ├── viewer.py                # Interactive debug viewer
    ├── analyze_eq_dataset.py    # Dataset statistics
    └── analyze_eq_v2.py         # Advanced analysis

forge/cli/
└── generate_eq_continuous.py    # Production CLI entry point

scripts/
├── analyze_eq_predictions.py    # Outcome correlation analysis
└── validate_eq_computation.py   # Fresh recomputation spot-check
```
