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
    mitigation: str = ""       # Description of any mitigation applied


class MappingIntegrityError(Exception):
    """Raised when observed action is in-hand but illegal (rules/state reconstruction bug)."""

    pass


@dataclass
class DecisionRecord:
    """One training example for Stage 2."""

    transcript_tokens: Tensor  # Stage 2 input (from tokenize_transcript)
    e_logits: Tensor  # (7,) target - averaged logits from oracle
    legal_mask: Tensor  # (7,) which actions were legal
    action_taken: int  # Which action was actually played


@dataclass
class GameRecord:
    """All decisions from one game."""

    decisions: list[DecisionRecord]
    # 28 decisions per game (4 players × 7 tricks)


@dataclass
class DecisionRecordV2(DecisionRecord):
    """Extended decision record with posterior diagnostics (t42-64uj.3)."""

    diagnostics: PosteriorDiagnostics | None = None  # Posterior health metrics


@dataclass
class GameRecordV2(GameRecord):
    """Extended game record with posterior config."""

    posterior_config: PosteriorConfig | None = None


def generate_eq_game(
    oracle: Stage1Oracle,
    hands: list[list[int]],  # Initial deal [hand0, hand1, hand2, hand3]
    decl_id: int,
    n_samples: int = 100,
    posterior_config: PosteriorConfig | None = None,
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
    8. Play the best action (argmax of E[Q] over legal actions)

    Args:
        oracle: Stage 1 oracle for querying Q-values
        hands: Initial deal as [hand0, hand1, hand2, hand3]
        decl_id: Declaration ID (0-9)
        n_samples: Number of worlds to sample per decision (default 100)
        posterior_config: Config for posterior weighting (None = uniform averaging)

    Returns:
        GameRecord (or GameRecordV2 if posterior_config provided) with 28 decisions
    """
    use_posterior = posterior_config is not None and posterior_config.enabled
    game = GameState.from_hands(hands, decl_id, leader=0)
    decisions = []

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

        # 5. Compute weighted E[Q] by domino ID, then map to remaining-hand order
        # e_logits[i] = E_w[Q(my_hand[i])] for i in 0..len(my_hand)-1
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

        # Weighted average E[Q] by domino ID
        e_logits = torch.zeros(len(my_hand))
        for i, domino_id in enumerate(my_hand):
            if e_q_by_domino[domino_id]:
                q_w_pairs = e_q_by_domino[domino_id]
                total_w = sum(w for _, w in q_w_pairs)
                if total_w > 0:
                    e_logits[i] = sum(q * w for q, w in q_w_pairs) / total_w
                else:
                    e_logits[i] = sum(q for q, _ in q_w_pairs) / len(q_w_pairs)
            else:
                e_logits[i] = float("-inf")  # Should never happen

        # 6. Build transcript tokens for Stage 2 training
        plays_for_transcript = [(p, d) for p, d, _ in game.play_history]
        transcript_tokens = tokenize_transcript(my_hand, plays_for_transcript, decl_id, player)

        # 7. Determine legal actions and select best
        legal_actions = game.legal_actions()
        legal_mask = _build_legal_mask(legal_actions, my_hand)  # (len(my_hand),) boolean

        # Mask illegal actions and select
        masked_logits = e_logits.clone()
        masked_logits[~legal_mask] = float("-inf")
        action_idx = masked_logits.argmax().item()
        action_domino = my_hand[action_idx]

        # 8. Record decision
        # Note: e_logits and legal_mask are now in remaining-hand order (len = len(my_hand))
        # Pad to 7 for consistent tensor shapes in dataset
        padded_e_logits = torch.full((7,), float("-inf"))
        padded_e_logits[:len(my_hand)] = e_logits
        padded_legal_mask = torch.zeros(7, dtype=torch.bool)
        padded_legal_mask[:len(my_hand)] = legal_mask

        if use_posterior:
            decisions.append(
                DecisionRecordV2(
                    transcript_tokens=transcript_tokens,
                    e_logits=padded_e_logits,
                    legal_mask=padded_legal_mask,
                    action_taken=action_idx,
                    diagnostics=diagnostics,
                )
            )
        else:
            decisions.append(
                DecisionRecord(
                    transcript_tokens=transcript_tokens,
                    e_logits=padded_e_logits,
                    legal_mask=padded_legal_mask,
                    action_taken=action_idx,
                )
            )

        # 9. Apply action
        game = game.apply_action(action_domino)

    if use_posterior:
        return GameRecordV2(decisions=decisions, posterior_config=posterior_config)
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

    diagnostics = PosteriorDiagnostics(
        ess=ess,
        max_w=max_w,
        entropy=entropy,
        k_eff=k_eff,
        n_invalid=n_invalid,
        n_illegal=n_illegal,
        window_nll=window_nll,
        window_k_used=len(window_plays),
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
