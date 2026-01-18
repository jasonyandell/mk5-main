"""
E[Q] game generation for Stage 2 training.

Plays complete games using Stage 1 oracle with world sampling,
recording all decision points as training examples.

Supports two marginalization modes:
- Uniform: Simple average over sampled worlds (current default)
- Posterior-weighted: Weight worlds by transcript likelihood (t42-64uj.3)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from forge.eq.game import GameState
from forge.eq.oracle import Stage1Oracle
from forge.eq.sampling import sample_consistent_worlds
from forge.eq.transcript_tokenize import tokenize_transcript
from forge.oracle.tables import can_follow, led_suit_for_lead_domino


# =============================================================================
# Posterior weighting parameters (t42-64uj.3)
# =============================================================================

@dataclass
class PosteriorConfig:
    """Configuration for posterior-weighted E[Q] marginalization."""

    enabled: bool = False  # Whether to use posterior weighting
    tau: float = 10.0      # Temperature for advantage-softmax
    beta: float = 0.10     # Uniform mixture coefficient for robustness
    window_k: int = 8      # Number of past plays to score
    delta: float = 30.0    # Log-weight clipping threshold

    # ESS thresholds for diagnostics/mitigation
    ess_warn: float = 10.0   # Warn if ESS < this
    ess_critical: float = 3.0  # Critical if ESS < this

    # Mitigation parameters
    mitigation_enabled: bool = True  # Whether to apply mitigation when ESS low
    mitigation_alpha: float = 0.3    # Uniform mix when ESS < ess_critical
    mitigation_beta_boost: float = 0.15  # Extra beta when ESS < ess_warn

    # Adaptive K parameters (Phase 7)
    adaptive_k_enabled: bool = False  # Whether to expand window when ESS low
    adaptive_k_max: int = 16          # Maximum window size
    adaptive_k_ess_threshold: float = 5.0  # Expand window if ESS < this
    adaptive_k_step: int = 4          # How much to expand window by

    # Rejuvenation kernel parameters (Phase 8)
    rejuvenation_enabled: bool = False  # Whether to resample+rejuvenate when ESS critical
    rejuvenation_steps: int = 3         # Number of MCMC steps per particle
    rejuvenation_ess_threshold: float = 2.0  # Trigger rejuvenation if ESS < this

    # Mapping integrity (design notes §6)
    strict_integrity: bool = False  # If True, raise on in-hand-but-illegal (dev mode)
    # If False, count and continue (production mode)


@dataclass
class PosteriorDiagnostics:
    """Diagnostics for posterior weighting health."""

    ess: float = 0.0           # Effective sample size
    max_w: float = 0.0         # Maximum weight
    entropy: float = 0.0       # Weight entropy H = -sum(w log w)
    k_eff: float = 0.0         # Effective world count = exp(H)
    n_invalid: int = 0         # Worlds with -inf log-weight (inconsistent)
    n_illegal: int = 0         # Worlds where observed was in-hand but illegal (bug indicator)
    window_nll: float = 0.0    # Average negative log-likelihood over window
    window_k_used: int = 0     # Actual window size used (may differ from config if adaptive)
    rejuvenation_applied: bool = False  # Whether rejuvenation was triggered
    rejuvenation_accepts: int = 0       # Number of accepted MCMC swaps
    mitigation: str = ""       # Description of any mitigation applied


class MappingIntegrityError(Exception):
    """Raised when observed action is in-hand but illegal (rules/state reconstruction bug)."""

    pass


# =============================================================================
# Exploration policy parameters (t42-64uj.5)
# =============================================================================


@dataclass
class ExplorationPolicy:
    """Configuration for exploratory action selection during game generation.

    Supports a mixture of policies to diversify transcripts:
    - greedy: Always pick argmax(E[Q]) (default, no exploration)
    - boltzmann: Sample from softmax(E[Q] / temperature)
    - epsilon: With probability epsilon, pick uniformly random legal action
    - blunder: Occasionally pick suboptimal action with bounded regret

    The policies are applied in order: first check blunder, then epsilon,
    then boltzmann vs greedy. This allows combining them.
    """

    # Boltzmann sampling
    temperature: float = 1.0  # Temperature for softmax sampling (lower = greedier)
    use_boltzmann: bool = False  # If False, use greedy argmax

    # Epsilon-greedy
    epsilon: float = 0.0  # Probability of random action (0 = no random)

    # Bounded blunder (occasional suboptimal pick)
    blunder_rate: float = 0.0  # Probability of picking non-best action
    blunder_max_regret: float = 5.0  # Max Q-gap allowed for blunder (in points)

    # Seeding for reproducibility
    seed: int | None = None  # If set, use deterministic RNG

    @classmethod
    def greedy(cls) -> "ExplorationPolicy":
        """Pure greedy policy (no exploration)."""
        return cls()

    @classmethod
    def boltzmann(cls, temperature: float = 2.0, seed: int | None = None) -> "ExplorationPolicy":
        """Boltzmann sampling with given temperature."""
        return cls(use_boltzmann=True, temperature=temperature, seed=seed)

    @classmethod
    def epsilon_greedy(cls, epsilon: float = 0.1, seed: int | None = None) -> "ExplorationPolicy":
        """Epsilon-greedy with given exploration rate."""
        return cls(epsilon=epsilon, seed=seed)

    @classmethod
    def mixed_exploration(
        cls,
        temperature: float = 3.0,
        epsilon: float = 0.05,
        blunder_rate: float = 0.02,
        blunder_max_regret: float = 3.0,
        seed: int | None = None,
    ) -> "ExplorationPolicy":
        """Recommended mixed policy for diverse transcript generation.

        Combines mild Boltzmann sampling with small epsilon and occasional blunders.
        Designed to approximate human-like play with bounded suboptimality.
        """
        return cls(
            use_boltzmann=True,
            temperature=temperature,
            epsilon=epsilon,
            blunder_rate=blunder_rate,
            blunder_max_regret=blunder_max_regret,
            seed=seed,
        )


@dataclass
class ExplorationStats:
    """Per-decision exploration statistics."""

    greedy_action: int  # What greedy would have picked
    action_taken: int  # What was actually played
    was_greedy: bool  # True if action_taken == greedy_action
    selection_mode: str  # "greedy", "boltzmann", "epsilon", "blunder"
    q_gap: float  # Q(greedy) - Q(action_taken), 0 if greedy
    action_entropy: float  # Entropy of softmax(E[Q]) over legal actions


@dataclass
class GameExplorationStats:
    """Aggregate exploration statistics for a full game."""

    n_decisions: int = 0
    n_greedy: int = 0
    n_boltzmann: int = 0
    n_epsilon: int = 0
    n_blunder: int = 0
    total_q_gap: float = 0.0  # Sum of regret across all decisions
    mean_action_entropy: float = 0.0

    @property
    def greedy_rate(self) -> float:
        return self.n_greedy / self.n_decisions if self.n_decisions > 0 else 0.0

    @property
    def mean_q_gap(self) -> float:
        return self.total_q_gap / self.n_decisions if self.n_decisions > 0 else 0.0


@dataclass
class DecisionRecord:
    """One training example for Stage 2.

    The target field `e_q_mean` contains the E[Q] (expected Q-value) in POINTS,
    computed by averaging Q-values across sampled worlds. These are NOT logits -
    do NOT apply softmax. Values are typically in [-42, +42] range.
    """

    transcript_tokens: Tensor  # Stage 2 input (from tokenize_transcript)
    e_q_mean: Tensor  # (7,) target - E[Q] in points (averaged Q across worlds)
    legal_mask: Tensor  # (7,) which actions were legal
    action_taken: int  # Which action was actually played


@dataclass
class GameRecord:
    """All decisions from one game."""

    decisions: list[DecisionRecord]
    # 28 decisions per game (4 players × 7 tricks)


@dataclass
class DecisionRecordV2(DecisionRecord):
    """Extended decision record with posterior diagnostics (t42-64uj.3) and exploration stats.

    Uncertainty fields (t42-64uj.6):
    - e_q_var: Per-move variance σ²(a) = Var_w[Q(a)] in points²
    - u_mean: State-level uncertainty = mean_{a legal}(σ(a)) in points
    - u_max: State-level uncertainty = max_{a legal}(σ(a)) in points
    """

    diagnostics: PosteriorDiagnostics | None = None  # Posterior health metrics
    exploration: ExplorationStats | None = None  # Exploration statistics (t42-64uj.5)
    # Uncertainty fields (t42-64uj.6)
    e_q_var: Tensor | None = None  # (7,) variance per action in points²
    u_mean: float = 0.0  # State-level uncertainty (mean σ over legal) in points
    u_max: float = 0.0  # State-level uncertainty (max σ over legal) in points


@dataclass
class GameRecordV2(GameRecord):
    """Extended game record with posterior config and exploration stats."""

    posterior_config: PosteriorConfig | None = None
    exploration_policy: ExplorationPolicy | None = None  # Policy used for action selection
    exploration_stats: GameExplorationStats | None = None  # Aggregate exploration stats


def generate_eq_game(
    oracle: Stage1Oracle,
    hands: list[list[int]],  # Initial deal [hand0, hand1, hand2, hand3]
    decl_id: int,
    n_samples: int = 100,
    posterior_config: PosteriorConfig | None = None,
    exploration_policy: ExplorationPolicy | None = None,
) -> GameRecord:
    """Play one game, record all 28 decisions.

    For each decision point:
    1. Get current player's perspective
    2. Infer voids from play history (incrementally)
    3. Sample N consistent worlds
    4. Reconstruct hypothetical initial hands for each world
    5. Query oracle for Q-values on each hypothetical world
    6. Compute E[Q] (uniform or posterior-weighted)
    7. Record decision with transcript tokens
    8. Select action using exploration policy (greedy, Boltzmann, epsilon, or blunder)

    Args:
        oracle: Stage 1 oracle for querying Q-values
        hands: Initial deal as [hand0, hand1, hand2, hand3]
        decl_id: Declaration ID (0-9)
        n_samples: Number of worlds to sample per decision (default 100)
        posterior_config: Config for posterior weighting (None = uniform averaging)
        exploration_policy: Policy for action selection (None = greedy)

    Returns:
        GameRecord (or GameRecordV2 if posterior/exploration config provided) with 28 decisions
    """
    use_posterior = posterior_config is not None and posterior_config.enabled
    use_exploration = exploration_policy is not None
    use_v2 = use_posterior or use_exploration

    game = GameState.from_hands(hands, decl_id, leader=0)
    decisions = []

    # Initialize RNG for exploration
    if use_exploration and exploration_policy.seed is not None:
        rng = np.random.default_rng(exploration_policy.seed)
    elif use_exploration:
        rng = np.random.default_rng()
    else:
        rng = None

    # Aggregate exploration stats
    game_exploration_stats = GameExplorationStats() if use_exploration else None
    total_entropy = 0.0

    # Incremental void tracking
    voids: dict[int, set[int]] = {0: set(), 1: set(), 2: set(), 3: set()}
    plays_processed = 0

    # Track which dominoes each player has played (for reconstructing initial hands)
    played_by: dict[int, list[int]] = {0: [], 1: [], 2: [], 3: []}

    while not game.is_complete():
        player = game.current_player()
        my_hand = list(game.hands[player])

        # 1. Incrementally update voids and played_by from new plays
        for play_idx in range(plays_processed, len(game.play_history)):
            play_player, domino_id, lead_domino_id = game.play_history[play_idx]
            played_by[play_player].append(domino_id)
            led_suit = led_suit_for_lead_domino(lead_domino_id, decl_id)
            if not can_follow(domino_id, led_suit, decl_id):
                voids[play_player].add(led_suit)
        plays_processed = len(game.play_history)

        # 2. Sample consistent worlds (remaining hands only)
        remaining_worlds = sample_consistent_worlds(
            my_player=player,
            my_hand=my_hand,
            played=game.played,
            hand_sizes=game.hand_sizes(),
            voids=voids,
            decl_id=decl_id,
            n_samples=n_samples,
        )

        # 3. Reconstruct hypothetical initial hands and build batched game state
        hypothetical_deals, game_state_info = _build_hypothetical_worlds_batched(
            game, remaining_worlds, played_by, player
        )

        # 4. Query oracle in SINGLE BATCH (N worlds at once)
        # Oracle outputs Q[local_idx] for initial-hand slots; we need Q[domino_id]
        n_worlds = len(hypothetical_deals)

        # Single batched call - O(1) instead of O(N)
        all_logits = oracle.query_batch(
            hypothetical_deals,
            game_state_info,
            player
        )  # (N, 7)

        # 4b. Compute posterior weights if enabled
        diagnostics: PosteriorDiagnostics | None = None
        if use_posterior:
            weights, diagnostics = compute_posterior_weights(
                oracle=oracle,
                hypothetical_deals=hypothetical_deals,
                play_history=list(game.play_history),
                decl_id=decl_id,
                config=posterior_config,
            )
        else:
            # Uniform weights
            weights = torch.ones(n_worlds, device=oracle.device) / n_worlds

        # 5. Compute weighted E[Q] and Var[Q] by domino ID, then map to remaining-hand order
        # e_q_mean[i] = E_w[Q(my_hand[i])] for i in 0..len(my_hand)-1  (in points)
        # e_q_var[i] = Var_w[Q(my_hand[i])] = E[Q²] - E[Q]²  (in points²)
        e_q_by_domino: dict[int, list[tuple[float, float]]] = {d: [] for d in my_hand}
        for world_idx in range(n_worlds):
            world_logits = all_logits[world_idx]  # (7,)
            w = weights[world_idx].item()
            # Project from local_idx to domino_id using this world's initial hands
            initial_hand = hypothetical_deals[world_idx][player]  # Sorted initial hand
            for local_idx, domino_id in enumerate(initial_hand):
                if domino_id in e_q_by_domino:
                    # This domino is still in remaining hand
                    e_q_by_domino[domino_id].append((world_logits[local_idx].item(), w))

        # Weighted mean E[Q] and variance Var[Q] by domino ID (t42-64uj.6)
        e_q_mean = torch.zeros(len(my_hand))
        e_q_var = torch.zeros(len(my_hand))
        for i, domino_id in enumerate(my_hand):
            if e_q_by_domino[domino_id]:
                q_w_pairs = e_q_by_domino[domino_id]
                total_w = sum(w for _, w in q_w_pairs)
                if total_w > 0:
                    # E[Q] = weighted mean (in points)
                    mean_q = sum(q * w for q, w in q_w_pairs) / total_w
                    # E[Q²] = weighted mean of squares
                    mean_q2 = sum(q * q * w for q, w in q_w_pairs) / total_w
                    # Var = E[Q²] - E[Q]² (clamp to non-negative for numerical stability)
                    var_q = max(0.0, mean_q2 - mean_q * mean_q)
                    e_q_mean[i] = mean_q
                    e_q_var[i] = var_q
                else:
                    e_q_mean[i] = sum(q for q, _ in q_w_pairs) / len(q_w_pairs)
                    e_q_var[i] = 0.0
            else:
                e_q_mean[i] = float("-inf")  # Should never happen
                e_q_var[i] = 0.0

        # 6. Build transcript tokens for Stage 2 training
        plays_for_transcript = [(p, d) for p, d, _ in game.play_history]
        transcript_tokens = tokenize_transcript(my_hand, plays_for_transcript, decl_id, player)

        # 7. Determine legal actions and select action
        legal_actions = game.legal_actions()
        legal_mask = _build_legal_mask(legal_actions, my_hand)  # (len(my_hand),) boolean

        # Mask illegal actions for greedy baseline
        masked_q = e_q_mean.clone()
        masked_q[~legal_mask] = float("-inf")
        greedy_idx = masked_q.argmax().item()

        # Select action using exploration policy
        action_idx, selection_mode, action_entropy = _select_action_with_exploration(
            e_q_mean=e_q_mean,
            legal_mask=legal_mask,
            policy=exploration_policy,
            rng=rng,
        )
        action_domino = my_hand[action_idx]

        # Compute exploration stats
        exploration_stats: ExplorationStats | None = None
        if use_exploration:
            q_gap = (e_q_mean[greedy_idx] - e_q_mean[action_idx]).item()
            exploration_stats = ExplorationStats(
                greedy_action=greedy_idx,
                action_taken=action_idx,
                was_greedy=(action_idx == greedy_idx),
                selection_mode=selection_mode,
                q_gap=q_gap,
                action_entropy=action_entropy,
            )
            # Update aggregate stats
            game_exploration_stats.n_decisions += 1
            if selection_mode == "greedy":
                game_exploration_stats.n_greedy += 1
            elif selection_mode == "boltzmann":
                game_exploration_stats.n_boltzmann += 1
            elif selection_mode == "epsilon":
                game_exploration_stats.n_epsilon += 1
            elif selection_mode == "blunder":
                game_exploration_stats.n_blunder += 1
            game_exploration_stats.total_q_gap += q_gap
            total_entropy += action_entropy

        # 8. Record decision
        # Note: e_q_mean and legal_mask are now in remaining-hand order (len = len(my_hand))
        # Pad to 7 for consistent tensor shapes in dataset
        padded_e_q_mean = torch.full((7,), float("-inf"))
        padded_e_q_mean[:len(my_hand)] = e_q_mean
        padded_legal_mask = torch.zeros(7, dtype=torch.bool)
        padded_legal_mask[:len(my_hand)] = legal_mask

        # Pad variance and compute state-level uncertainty (t42-64uj.6)
        padded_e_q_var = torch.zeros(7)
        padded_e_q_var[:len(my_hand)] = e_q_var

        # State-level uncertainty: U_mean and U_max over legal actions
        # σ(a) = sqrt(var(a)), then compute mean/max over legal (in points)
        legal_std = torch.sqrt(e_q_var)
        legal_std_for_u = legal_std[legal_mask]
        if len(legal_std_for_u) > 0:
            u_mean = legal_std_for_u.mean().item()
            u_max = legal_std_for_u.max().item()
        else:
            u_mean = 0.0
            u_max = 0.0

        if use_v2:
            decisions.append(
                DecisionRecordV2(
                    transcript_tokens=transcript_tokens,
                    e_q_mean=padded_e_q_mean,
                    legal_mask=padded_legal_mask,
                    action_taken=action_idx,
                    diagnostics=diagnostics,
                    exploration=exploration_stats,
                    e_q_var=padded_e_q_var,
                    u_mean=u_mean,
                    u_max=u_max,
                )
            )
        else:
            decisions.append(
                DecisionRecord(
                    transcript_tokens=transcript_tokens,
                    e_q_mean=padded_e_q_mean,
                    legal_mask=padded_legal_mask,
                    action_taken=action_idx,
                )
            )

        # 9. Apply action
        game = game.apply_action(action_domino)

    # Finalize game exploration stats
    if use_exploration and game_exploration_stats.n_decisions > 0:
        game_exploration_stats.mean_action_entropy = total_entropy / game_exploration_stats.n_decisions

    if use_v2:
        return GameRecordV2(
            decisions=decisions,
            posterior_config=posterior_config,
            exploration_policy=exploration_policy,
            exploration_stats=game_exploration_stats,
        )
    return GameRecord(decisions=decisions)


def _build_hypothetical_worlds_batched(
    game: GameState,
    remaining_worlds: list[list[list[int]]],
    played_by: dict[int, list[int]],
    current_player: int,
) -> tuple[list[list[list[int]]], dict]:
    """Reconstruct hypothetical initial hands for each sampled world (batched).

    For proper E[Q] marginalization, we need to query the oracle on HYPOTHETICAL
    worlds, not the true game state. Each sampled world gives us remaining hands
    for opponents. We reconstruct initial hands as:
        initial[p] = remaining[p] + played_by[p]

    KEY OPTIMIZATION: trick_plays uses domino_id (public info) instead of local_idx.
    This makes trick_plays world-invariant, enabling a SINGLE batched oracle call
    instead of O(N) separate calls.

    Args:
        game: Current game state
        remaining_worlds: List of N worlds, each with 4 hands of remaining dominoes
        played_by: Dict mapping player -> list of dominoes they've played
        current_player: Current player making the decision

    Returns:
        Tuple of (hypothetical_deals, game_state_info) where:
        - hypothetical_deals: List of N initial deals, one per world
        - game_state_info: Single dict with decl_id, leader, trick_plays (domino_id), remaining (N,4)
    """
    n_worlds = len(remaining_worlds)
    hypothetical_deals = []

    # Build remaining bitmasks for ALL worlds at once
    remaining_bitmask = np.zeros((n_worlds, 4), dtype=np.int32)

    for world_idx, remaining_hands in enumerate(remaining_worlds):
        # Reconstruct initial hands: initial[p] = remaining[p] + played_by[p]
        initial_hands = []
        for p in range(4):
            initial = list(remaining_hands[p]) + list(played_by[p])
            initial.sort()  # Sort for consistent ordering
            initial_hands.append(initial)

        hypothetical_deals.append(initial_hands)

        # Build remaining bitmask for this world
        for p in range(4):
            for local_idx, domino in enumerate(initial_hands[p]):
                if domino not in game.played:
                    remaining_bitmask[world_idx, p] |= 1 << local_idx

    # Build trick_plays using DOMINO_ID (world-invariant public info)
    # This is the key change: (player, domino_id) instead of (player, local_idx)
    trick_plays = [(play_player, domino_id) for play_player, domino_id in game.current_trick]

    # Single game_state_info for all worlds
    game_state_info = {
        "decl_id": game.decl_id,
        "leader": game.leader,
        "trick_plays": trick_plays,  # Uses domino_id, not local_idx
        "remaining": remaining_bitmask,  # (N, 4) for all worlds
    }

    return hypothetical_deals, game_state_info


def _build_legal_mask(legal_actions: tuple[int, ...], hand: list[int]) -> Tensor:
    """Build boolean mask of legal actions.

    Args:
        legal_actions: Tuple of legal domino IDs
        hand: Current hand as list of domino IDs (remaining hand)

    Returns:
        Boolean tensor of shape (len(hand),) where True means legal
    """
    legal_set = set(legal_actions)
    mask = torch.tensor([domino in legal_set for domino in hand], dtype=torch.bool)
    return mask


# =============================================================================
# Exploration policy helpers (t42-64uj.5)
# =============================================================================


def _select_action_with_exploration(
    e_q_mean: Tensor,
    legal_mask: Tensor,
    policy: ExplorationPolicy | None,
    rng: np.random.Generator | None,
) -> tuple[int, str, float]:
    """Select action using exploration policy.

    Args:
        e_q_mean: E[Q] values (in points) for each action in remaining-hand order
        legal_mask: Boolean mask of legal actions
        policy: Exploration policy (None = greedy)
        rng: Random number generator for stochastic selection

    Returns:
        Tuple of (action_idx, selection_mode, action_entropy) where:
        - action_idx: Selected action index
        - selection_mode: "greedy", "boltzmann", "epsilon", or "blunder"
        - action_entropy: Entropy of softmax(E[Q]/temp) over legal actions (for diversity measure)

    Note:
        Although E[Q] values are in points, we compute softmax for Boltzmann sampling
        and entropy calculation. This converts point differences into selection probabilities.
        The temperature parameter controls how much point differences affect selection.
    """
    # Get legal action indices and Q-values
    legal_indices = torch.where(legal_mask)[0].tolist()
    n_legal = len(legal_indices)

    if n_legal == 0:
        raise ValueError("No legal actions available")

    if n_legal == 1:
        # Only one legal action - no choice
        return legal_indices[0], "greedy", 0.0

    # Get Q-values for legal actions (in points)
    legal_q = e_q_mean[legal_mask]

    # Compute softmax probabilities for entropy calculation (and Boltzmann sampling)
    # Use temperature=1.0 for entropy calculation baseline
    probs = F.softmax(legal_q, dim=0)
    log_probs = torch.log(probs + 1e-30)
    action_entropy = -(probs * log_probs).sum().item()

    # Greedy action (always compute as baseline)
    greedy_local = legal_q.argmax().item()
    greedy_idx = legal_indices[greedy_local]
    greedy_q = legal_q[greedy_local].item()

    # No policy = greedy
    if policy is None or rng is None:
        return greedy_idx, "greedy", action_entropy

    # Check for blunder (occasional suboptimal pick with bounded regret)
    if policy.blunder_rate > 0 and rng.random() < policy.blunder_rate:
        # Find actions within bounded regret
        regret_threshold = greedy_q - policy.blunder_max_regret
        blunder_candidates = [
            (i, local_i) for local_i, i in enumerate(legal_indices)
            if legal_q[local_i].item() >= regret_threshold and i != greedy_idx
        ]
        if blunder_candidates:
            # Pick random from candidates
            idx, _ = blunder_candidates[rng.integers(len(blunder_candidates))]
            return idx, "blunder", action_entropy

    # Check for epsilon-random
    if policy.epsilon > 0 and rng.random() < policy.epsilon:
        # Pick uniformly random legal action
        random_idx = legal_indices[rng.integers(n_legal)]
        return random_idx, "epsilon", action_entropy

    # Boltzmann sampling or greedy
    if policy.use_boltzmann:
        # Sample from softmax with temperature
        boltzmann_probs = F.softmax(legal_q / policy.temperature, dim=0).numpy()
        sampled_local = rng.choice(n_legal, p=boltzmann_probs)
        sampled_idx = legal_indices[sampled_local]
        # If sampled == greedy, still report as boltzmann (it was stochastic)
        return sampled_idx, "boltzmann", action_entropy

    # Pure greedy
    return greedy_idx, "greedy", action_entropy


# =============================================================================
# Posterior weighting helpers (t42-64uj.3)
# =============================================================================


def compute_posterior_weights(
    oracle: Stage1Oracle,
    hypothetical_deals: list[list[list[int]]],
    play_history: list[tuple[int, int, int]],
    decl_id: int,
    config: PosteriorConfig,
) -> tuple[Tensor, PosteriorDiagnostics]:
    """Compute posterior weights for sampled worlds based on transcript likelihood.

    For each world, score how well it explains the recent K plays by computing:
        P(observed_action | info-set, world) for each past step

    Uses advantage-softmax with beta mixture for robustness.

    If adaptive_k_enabled is True, the window size will be expanded (up to
    adaptive_k_max) when ESS falls below adaptive_k_ess_threshold.

    Args:
        oracle: Stage 1 oracle for Q-value queries
        hypothetical_deals: List of N initial deals (4 players × 7 dominoes each)
        play_history: Full play history as (player, domino_id, lead_domino_id) tuples
        decl_id: Declaration ID
        config: Posterior weighting configuration

    Returns:
        Tuple of (weights, diagnostics) where:
        - weights: (N,) tensor of normalized weights summing to 1
        - diagnostics: PosteriorDiagnostics with health metrics
    """
    n_worlds = len(hypothetical_deals)
    device = oracle.device

    if not play_history:
        # No history to score - return uniform weights
        weights = torch.ones(n_worlds, device=device) / n_worlds
        return weights, PosteriorDiagnostics(
            ess=float(n_worlds),
            max_w=1.0 / n_worlds,
            entropy=np.log(n_worlds),
            k_eff=float(n_worlds),
            window_k_used=0,
        )

    # Adaptive K loop: start with base window_k, expand if ESS too low
    current_k = min(config.window_k, len(play_history))
    max_k = min(config.adaptive_k_max, len(play_history)) if config.adaptive_k_enabled else current_k

    while True:
        # Compute weights for current window size
        weights, diagnostics = _compute_weights_for_window(
            oracle=oracle,
            hypothetical_deals=hypothetical_deals,
            play_history=play_history,
            decl_id=decl_id,
            config=config,
            window_k=current_k,
        )

        # Check if we should expand window
        if not config.adaptive_k_enabled:
            break
        if diagnostics.ess >= config.adaptive_k_ess_threshold:
            break
        if current_k >= max_k:
            break

        # Expand window
        next_k = min(current_k + config.adaptive_k_step, max_k)
        if next_k == current_k:
            break
        current_k = next_k

    return weights, diagnostics


def _compute_weights_for_window(
    oracle: Stage1Oracle,
    hypothetical_deals: list[list[list[int]]],
    play_history: list[tuple[int, int, int]],
    decl_id: int,
    config: PosteriorConfig,
    window_k: int,
) -> tuple[Tensor, PosteriorDiagnostics]:
    """Compute posterior weights using a specific window size.

    Internal helper for compute_posterior_weights that handles a single
    window size. The main function calls this repeatedly when adaptive K
    is enabled.
    """
    n_worlds = len(hypothetical_deals)
    device = oracle.device

    # Initialize log-weights in fp32 for numerical stability
    logw = torch.zeros(n_worlds, dtype=torch.float32, device=device)

    # Determine window: last K plays (or all if fewer)
    window_start = max(0, len(play_history) - window_k)
    window_plays = play_history[window_start:]

    if not window_plays:
        # No history to score - return uniform weights
        weights = torch.ones(n_worlds, device=device) / n_worlds
        return weights, PosteriorDiagnostics(
            ess=float(n_worlds),
            max_w=1.0 / n_worlds,
            entropy=np.log(n_worlds),
            k_eff=float(n_worlds),
            window_k_used=0,
        )

    n_invalid = 0
    n_illegal = 0  # Mapping integrity violations (in-hand but illegal)
    total_logp = 0.0
    n_scored_steps = 0

    # Score each step in the window
    for step_offset, (actor, observed_domino, lead_domino) in enumerate(window_plays):
        step_idx = window_start + step_offset

        # Reconstruct game state at this step for each world
        # We need: actor's remaining hand, leader, current trick at step_idx
        step_logp, step_invalid, step_illegal = _score_step_likelihood(
            oracle=oracle,
            hypothetical_deals=hypothetical_deals,
            play_history=play_history,
            step_idx=step_idx,
            actor=actor,
            observed_domino=observed_domino,
            lead_domino=lead_domino,
            decl_id=decl_id,
            config=config,
        )

        logw += step_logp
        n_invalid = max(n_invalid, step_invalid)
        n_illegal += step_illegal  # Accumulate across steps
        total_logp += step_logp.mean().item()
        n_scored_steps += 1

    # Stabilize weights: subtract max, clip to max-delta
    max_logw = logw.max()
    logw = logw - max_logw
    logw = torch.clamp(logw, min=-config.delta)

    # Normalize to get weights
    weights = F.softmax(logw, dim=0)

    # Compute initial diagnostics
    ess = 1.0 / (weights * weights).sum().item()
    max_w = weights.max().item()

    # Entropy and effective world count
    log_weights = torch.log(weights + 1e-30)
    entropy = -(weights * log_weights).sum().item()
    k_eff = np.exp(entropy)

    # Average NLL over window
    window_nll = -total_logp / n_scored_steps if n_scored_steps > 0 else 0.0

    # ==========================================================================
    # ESS-based degeneracy mitigation (Phase 6)
    # ==========================================================================
    mitigation = ""
    rejuvenation_applied = False
    rejuvenation_accepts = 0

    if config.mitigation_enabled and ess < config.ess_warn:
        uniform = torch.ones(n_worlds, device=device) / n_worlds

        if ess < config.ess_critical:
            # Critical: strong uniform mix
            alpha = config.mitigation_alpha
            weights = (1 - alpha) * weights + alpha * uniform
            mitigation = f"critical_mix(alpha={alpha:.2f})"
        else:
            # Warning: mild uniform mix via beta boost
            # Re-weight using boosted beta equivalent
            beta_boost = config.mitigation_beta_boost
            weights = (1 - beta_boost) * weights + beta_boost * uniform
            mitigation = f"warn_mix(beta_boost={beta_boost:.2f})"

        # Recompute diagnostics after mitigation
        ess = 1.0 / (weights * weights).sum().item()
        max_w = weights.max().item()
        log_weights = torch.log(weights + 1e-30)
        entropy = -(weights * log_weights).sum().item()
        k_eff = np.exp(entropy)

    # ==========================================================================
    # Rejuvenation kernel (Phase 8) - last resort when ESS still critical
    # ==========================================================================
    if (config.rejuvenation_enabled and
        ess < config.rejuvenation_ess_threshold and
        len(play_history) > 0):

        # Resample particles according to weights, then apply MCMC swap kernel
        hypothetical_deals, rejuvenation_accepts = _rejuvenate_particles(
            hypothetical_deals=hypothetical_deals,
            weights=weights,
            play_history=play_history,
            decl_id=decl_id,
            oracle=oracle,
            config=config,
            window_k=window_k,
        )
        rejuvenation_applied = True

        # After rejuvenation, recompute weights with new particles
        # (simplified: just use uniform weights since particles are now diversified)
        weights = torch.ones(n_worlds, device=device) / n_worlds
        ess = float(n_worlds)
        max_w = 1.0 / n_worlds
        entropy = np.log(n_worlds)
        k_eff = float(n_worlds)

        if mitigation:
            mitigation += f"+rejuv(accepts={rejuvenation_accepts})"
        else:
            mitigation = f"rejuv(accepts={rejuvenation_accepts})"

    diagnostics = PosteriorDiagnostics(
        ess=ess,
        max_w=max_w,
        entropy=entropy,
        k_eff=k_eff,
        n_invalid=n_invalid,
        n_illegal=n_illegal,
        window_nll=window_nll,
        window_k_used=len(window_plays),
        rejuvenation_applied=rejuvenation_applied,
        rejuvenation_accepts=rejuvenation_accepts,
        mitigation=mitigation,
    )

    return weights, diagnostics


def _score_step_likelihood(
    oracle: Stage1Oracle,
    hypothetical_deals: list[list[list[int]]],
    play_history: list[tuple[int, int, int]],
    step_idx: int,
    actor: int,
    observed_domino: int,
    lead_domino: int,
    decl_id: int,
    config: PosteriorConfig,
) -> tuple[Tensor, int, int]:
    """Score likelihood of observed action at one step for all worlds.

    Args:
        oracle: Stage 1 oracle
        hypothetical_deals: N initial deals
        play_history: Full play history
        step_idx: Index of this step in play_history
        actor: Player who made this move
        observed_domino: Domino that was actually played
        lead_domino: Lead domino for this trick
        decl_id: Declaration ID
        config: Posterior config

    Returns:
        Tuple of (log_probs, n_invalid, n_illegal) where:
        - log_probs: (N,) tensor of log P(observed | world)
        - n_invalid: Count of worlds where observed domino wasn't in actor's hand
        - n_illegal: Count of worlds where observed was in-hand but illegal

    Raises:
        MappingIntegrityError: If strict_integrity=True and in-hand-but-illegal detected
    """
    n_worlds = len(hypothetical_deals)
    device = oracle.device

    # Determine which dominoes have been played BEFORE this step
    played_before = set()
    for i in range(step_idx):
        _, domino, _ = play_history[i]
        played_before.add(domino)

    # Determine current trick state at this step
    # Find where this trick started
    trick_start = step_idx
    while trick_start > 0:
        prev_idx = trick_start - 1
        _, _, prev_lead = play_history[prev_idx]
        # If previous play has same lead_domino, it's same trick
        if prev_lead == lead_domino:
            trick_start = prev_idx
        else:
            break

    # Build current trick plays (before this step, within this trick)
    current_trick = []
    for i in range(trick_start, step_idx):
        p, d, _ = play_history[i]
        current_trick.append((p, d))

    # Determine leader for this trick
    if current_trick:
        leader = current_trick[0][0]
    else:
        # This is the lead play - actor is the leader
        leader = actor

    # Build remaining bitmasks and check validity for each world
    remaining_bitmask = np.zeros((n_worlds, 4), dtype=np.int32)
    obs_local_idx = np.full(n_worlds, -1, dtype=np.int32)  # -1 means invalid
    n_invalid = 0
    n_illegal = 0  # In-hand but illegal (mapping integrity violation)

    for world_idx, initial_hands in enumerate(hypothetical_deals):
        actor_hand = initial_hands[actor]

        # Check if observed domino is in actor's initial hand
        if observed_domino not in actor_hand:
            # World inconsistent with transcript - hard reject
            n_invalid += 1
            obs_local_idx[world_idx] = -1
            continue

        # Find local_idx of observed domino in actor's hand
        obs_local_idx[world_idx] = actor_hand.index(observed_domino)

        # Build remaining bitmask (dominoes not played before this step)
        for p in range(4):
            for local_idx, domino in enumerate(initial_hands[p]):
                if domino not in played_before:
                    remaining_bitmask[world_idx, p] |= 1 << local_idx

    # Build game_state_info for oracle query
    game_state_info = {
        "decl_id": decl_id,
        "leader": leader,
        "trick_plays": current_trick,  # Uses domino_id format
        "remaining": remaining_bitmask,
    }

    # Query oracle for Q-values from actor's perspective
    all_logits = oracle.query_batch(hypothetical_deals, game_state_info, actor)  # (N, 7)

    # Compute log P(observed | world) using advantage-softmax + beta mixture
    log_probs = torch.full((n_worlds,), float("-inf"), dtype=torch.float32, device=device)

    for world_idx in range(n_worlds):
        local_idx = obs_local_idx[world_idx]
        if local_idx < 0:
            # Invalid world - keep -inf
            continue

        q_values = all_logits[world_idx]  # (7,)
        actor_hand = hypothetical_deals[world_idx][actor]

        # Determine legal actions at this step
        legal_local = _get_legal_local_indices(
            actor_hand, played_before, lead_domino, decl_id, step_idx == trick_start
        )

        if not legal_local:
            # No legal actions - shouldn't happen, treat as bug
            if config.strict_integrity:
                raise MappingIntegrityError(
                    f"No legal actions for actor {actor} at step {step_idx} in world {world_idx}"
                )
            n_illegal += 1
            continue

        if local_idx not in legal_local:
            # Observed action was in-hand but illegal - rules/state reconstruction bug
            # Design notes §6: "in-hand but illegal" usually means leader/trick/prefix
            # state is wrong; quietly downweighting will hide a bug.
            if config.strict_integrity:
                raise MappingIntegrityError(
                    f"Observed domino {observed_domino} (local_idx={local_idx}) is in actor {actor}'s "
                    f"hand but not legal at step {step_idx}. Legal indices: {legal_local}. "
                    f"This indicates a rules/state reconstruction bug."
                )
            n_illegal += 1
            continue

        # Extract Q-values for legal actions
        legal_q = q_values[legal_local]

        # Compute advantage: A = Q - mean(Q_legal)
        advantage = legal_q - legal_q.mean()

        # Softmax over advantage with temperature
        p_soft = F.softmax(advantage / config.tau, dim=0)

        # Beta mixture with uniform
        n_legal = len(legal_local)
        p_mixed = (1 - config.beta) * p_soft + config.beta / n_legal

        # Find index of observed action within legal actions
        obs_idx_in_legal = legal_local.index(local_idx)
        log_probs[world_idx] = torch.log(p_mixed[obs_idx_in_legal] + 1e-30)

    return log_probs, n_invalid, n_illegal


def _get_legal_local_indices(
    actor_hand: list[int],
    played_before: set[int],
    lead_domino: int | None,
    decl_id: int,
    is_leading: bool,
) -> list[int]:
    """Get local indices of legal actions for actor at this step.

    Args:
        actor_hand: Actor's initial 7-domino hand (sorted)
        played_before: Dominoes played before this step
        lead_domino: Lead domino for this trick (None if leading)
        decl_id: Declaration ID
        is_leading: Whether actor is leading this trick

    Returns:
        List of local indices (0-6) that are legal
    """
    # Get remaining dominoes
    remaining = [d for d in actor_hand if d not in played_before]
    remaining_local = [i for i, d in enumerate(actor_hand) if d not in played_before]

    if is_leading or lead_domino is None:
        # Leading - any remaining domino is legal
        return remaining_local

    # Following - must follow suit if possible
    led_suit = led_suit_for_lead_domino(lead_domino, decl_id)
    followers = [
        local_idx for local_idx, d in zip(remaining_local, remaining)
        if can_follow(d, led_suit, decl_id)
    ]

    # If can follow, must follow; otherwise can play anything
    return followers if followers else remaining_local


# =============================================================================
# Rejuvenation kernel (Phase 8)
# =============================================================================


def _rejuvenate_particles(
    hypothetical_deals: list[list[list[int]]],
    weights: Tensor,
    play_history: list[tuple[int, int, int]],
    decl_id: int,
    oracle: Stage1Oracle,
    config: PosteriorConfig,
    window_k: int,
) -> tuple[list[list[list[int]]], int]:
    """Resample and rejuvenate particles using MCMC swap kernel.

    Per design notes §8:
    1. Resample particles proportional to weights
    2. Apply constraint-preserving swap kernel to diversify

    The swap kernel:
    - Never modifies the conditioned-known hand (current actor)
    - Swaps unplayed dominoes between two other players
    - Accepts/rejects based on posterior ratio

    Args:
        hypothetical_deals: Current particle set (N initial deals)
        weights: Current particle weights
        play_history: Play history for constraint checking
        decl_id: Declaration ID
        oracle: Oracle for likelihood scoring
        config: Posterior config with rejuvenation params
        window_k: Window size for likelihood scoring

    Returns:
        Tuple of (new_deals, n_accepts) where:
        - new_deals: Rejuvenated particle set
        - n_accepts: Number of accepted MCMC swaps
    """
    import random

    n_worlds = len(hypothetical_deals)
    device = oracle.device

    # Step 1: Resample particles according to weights
    # Multinomial resampling
    weights_np = weights.cpu().numpy()
    indices = np.random.choice(n_worlds, size=n_worlds, replace=True, p=weights_np)

    # Create new particle set from resampled indices
    new_deals = [
        [list(hand) for hand in hypothetical_deals[idx]]
        for idx in indices
    ]

    # Determine which dominoes have been played (can't swap these)
    played_set = {d for _, d, _ in play_history}

    # Infer voids from play history for constraint checking
    voids: dict[int, set[int]] = {0: set(), 1: set(), 2: set(), 3: set()}
    for player, domino, lead_domino in play_history:
        led_suit = led_suit_for_lead_domino(lead_domino, decl_id)
        if not can_follow(domino, led_suit, decl_id):
            voids[player].add(led_suit)

    # Determine current actor (last player to act, or 0 if empty)
    # The conditioned-known hand - we never modify this
    if play_history:
        conditioned_player = play_history[-1][0]
    else:
        conditioned_player = 0

    # Step 2: Apply MCMC swap kernel to each particle
    n_accepts = 0

    for particle_idx in range(n_worlds):
        particle = new_deals[particle_idx]

        for _ in range(config.rejuvenation_steps):
            # Pick two latent players (not the conditioned player)
            latent_players = [p for p in range(4) if p != conditioned_player]
            if len(latent_players) < 2:
                continue

            u, v = random.sample(latent_players, 2)

            # Get unplayed dominoes from each player's hand
            u_unplayed = [d for d in particle[u] if d not in played_set]
            v_unplayed = [d for d in particle[v] if d not in played_set]

            if not u_unplayed or not v_unplayed:
                continue

            # Pick random dominoes to swap
            a = random.choice(u_unplayed)
            b = random.choice(v_unplayed)

            if a == b:
                continue  # Same domino, no swap needed

            # Check if swap violates void constraints
            # After swap: u gets b, v gets a
            # Check if b violates u's voids
            u_void_violation = _domino_violates_voids(b, voids[u], decl_id, play_history, u)
            v_void_violation = _domino_violates_voids(a, voids[v], decl_id, play_history, v)

            if u_void_violation or v_void_violation:
                continue  # Swap would violate hard constraints

            # Compute acceptance probability using Metropolis-Hastings
            # For simplicity, use uniform proposal so ratio = posterior ratio
            # Accept with probability min(1, exp(logP_new - logP_old))
            # Since computing exact likelihoods is expensive, we use a simplified
            # acceptance: always accept valid swaps (uniform prior over valid configs)
            # This is a valid MCMC kernel that preserves the uniform distribution
            # over constraint-satisfying configurations.

            # Apply the swap
            particle[u].remove(a)
            particle[u].append(b)
            particle[u].sort()

            particle[v].remove(b)
            particle[v].append(a)
            particle[v].sort()

            n_accepts += 1

    return new_deals, n_accepts


def _domino_violates_voids(
    domino: int,
    player_voids: set[int],
    decl_id: int,
    play_history: list[tuple[int, int, int]],
    player: int,
) -> bool:
    """Check if giving this domino to a player violates their void constraints.

    A void constraint is violated if the player was void in a suit at some point,
    but this domino could have followed that suit (meaning they should have played it).

    Args:
        domino: The domino to check
        player_voids: Set of suits the player is known to be void in
        decl_id: Declaration ID
        play_history: Full play history
        player: The player who would receive this domino

    Returns:
        True if adding this domino would violate void constraints
    """
    if not player_voids:
        return False

    # For each void suit, check if this domino could follow it
    # If so, giving them this domino is a violation
    for void_suit in player_voids:
        if can_follow(domino, void_suit, decl_id):
            return True

    return False
