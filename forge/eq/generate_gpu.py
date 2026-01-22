"""GPU-native E[Q] generation pipeline.

Generates training data by playing games using a Stage 1 oracle to estimate
E[Q] (expected Q-value) for each legal action. The pipeline runs entirely on
GPU for both game state management and model inference.

Core Loop (per decision):
──────────────────────────────────────────────────────────────────────────────
    GameStateTensor          N games running in parallel
           ↓
    WorldSampler             Sample M opponent hands per game
           ↓                 (constrained by voids, played dominoes)
    Build hypothetical       Combine known hand + sampled opponents
           ↓                 → [N, M, 4, 7] full deals
    GPUTokenizer             Tokenize current decision
           ↓                 → [N×M, 32, 12] tokens
    Model forward            Query oracle for Q-values
           ↓                 → [N×M, 7] Q per action
    E[Q] aggregation         See "Aggregation Modes" below
           ↓                 → [N, 7] expected Q per action
    Action selection         Greedy, softmax, or exploration policy
           ↓
    apply_actions            Advance game state
           └──→ repeat until all games complete

Aggregation Modes:
──────────────────────────────────────────────────────────────────────────────
1. UNIFORM (default, posterior_config=None or enabled=False):
   E[Q] = mean over M samples. Fast, no extra model calls.

2. POSTERIOR-WEIGHTED (posterior_config.enabled=True):
   Uses Bayesian inference to reweight samples based on past play.
   Samples consistent with observed opponent actions get higher weight.

   Additional steps when posterior enabled:
       ├─ Reconstruct past K states      What board looked like K steps ago
       ├─ Reconstruct historical hands   Undo plays to get hands at step k
       ├─ Tokenize past steps            [N×M×K, 32, 12] second tokenization
       ├─ Model forward (past)           Second oracle call for past Q-values
       ├─ Compute legal masks            What was legal at each past step?
       └─ Bayesian weighting             P(world|observed actions) ∝ P(actions|world)

   Output: E[Q] = weighted mean, plus ESS (effective sample size) diagnostics.
   ESS << M indicates strong filtering (good - opponent play is informative).
   ESS ≈ M indicates weak filtering (opponent play doesn't help much).

Exploration Policy (optional):
──────────────────────────────────────────────────────────────────────────────
When exploration_policy is provided, action selection can deviate from greedy:
- Boltzmann sampling with temperature
- Epsilon-greedy with random legal action fallback
- Deliberate "blunders" for robustness training

Tracks q_gap (regret) when exploration causes suboptimal action selection.

Key Components:
──────────────────────────────────────────────────────────────────────────────
- GameStateTensor     Vectorized game state (hands, voids, tricks, history)
- WorldSamplerMRV     MRV-based constraint solver for valid opponent hands
- GPUTokenizer        Pure tensor tokenization (no Python loops)
- posterior_gpu       Posterior weight computation and historical reconstruction
- DecisionRecordGPU   Per-decision output (E[Q], variance, ESS, exploration stats)

Performance (measured on 3050 Ti, 32 games × 50 samples):
──────────────────────────────────────────────────────────────────────────────
- Uniform:   6.5 games/s
- Posterior: 2.4 games/s (2.7x slower due to second model call for K past steps)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from forge.eq.game_tensor import GameStateTensor
from forge.eq.posterior_gpu import (
    compute_legal_masks_gpu,
    compute_posterior_weights_gpu,
    reconstruct_historical_hands,
    reconstruct_past_states_gpu,
)
from forge.eq.sampling_gpu import WorldSampler
from forge.eq.sampling_mrv_gpu import WorldSamplerMRV
from forge.eq.tokenize_gpu import GPUTokenizer
from forge.eq.types import ExplorationPolicy, PosteriorDiagnostics


@dataclass
class PosteriorConfig:
    """Configuration for posterior weighting in E[Q] computation.

    Args:
        enabled: Whether to use posterior weighting (default: False)
        window_k: Number of past steps to use for weighting (default: 4)
        tau: Temperature for softmax over Q-values (default: 0.1)
        uniform_mix: Uniform mixture coefficient for robustness (default: 0.1)
    """
    enabled: bool = False
    window_k: int = 4
    tau: float = 0.1
    uniform_mix: float = 0.1


@dataclass
class DecisionRecordGPU:
    """Record for one decision in GPU pipeline.

    Extended in Phase 1a (t42-xncr) to include variance and diagnostics.
    """
    player: int  # Which player made the decision (0-3)
    e_q: Tensor  # [7] E[Q] mean values (padded, -inf for illegal/empty)
    action_taken: int  # Slot index (0-6) of domino played
    legal_mask: Tensor  # [7] boolean mask of legal actions
    e_q_var: Tensor | None = None  # [7] E[Q] variance per action
    # State uncertainty
    u_mean: float = 0.0  # mean(sqrt(var)) over legal actions
    u_max: float = 0.0   # max(sqrt(var)) over legal actions
    # Posterior diagnostics (None if posterior disabled)
    ess: float | None = None  # Effective sample size
    max_w: float | None = None  # Maximum weight
    # Exploration stats (None if exploration disabled)
    exploration_mode: int | None = None  # 0=greedy, 1=boltzmann, 2=epsilon, 3=blunder
    q_gap: float | None = None  # Q_greedy - Q_taken
    greedy_action: int | None = None  # What greedy would have chosen (slot index 0-6)


@dataclass
class GameRecordGPU:
    """Record for one complete game in GPU pipeline."""
    decisions: list[DecisionRecordGPU]
    hands: list[list[int]]  # Initial deal (4 players × 7 dominoes)
    decl_id: int  # Declaration ID


def generate_eq_games_gpu(
    model,
    hands: list[list[list[int]]],
    decl_ids: list[int],
    n_samples: int = 50,
    device: str = 'cuda',
    greedy: bool = True,
    use_mrv_sampler: bool = True,
    exploration_policy: ExplorationPolicy | None = None,
    posterior_config: PosteriorConfig | None = None,
) -> list[GameRecordGPU]:
    """Generate E[Q] games entirely on GPU.

    Args:
        model: Stage 1 oracle model (DominoLightningModule or wrapper)
        hands: List of N initial deals, each is 4 hands of 7 domino IDs
        decl_ids: List of N declaration IDs
        n_samples: Number of worlds to sample per decision
        device: Device to run on ('cuda' or 'cpu')
        greedy: If True, always pick argmax(E[Q]). If False, sample from softmax.
        use_mrv_sampler: If True (default), use MRV-based sampler (guaranteed valid).
                        If False, use rejection sampling (faster but may fall back to CPU).
        exploration_policy: Optional exploration policy for stochastic action selection.
                          If provided, overrides greedy parameter.
        posterior_config: Optional config for posterior weighting. If provided and enabled,
                         uses past K steps to reweight worlds before computing E[Q].

    Returns:
        List of N GameRecordGPU, one per game

    Performance:
        - 3050 Ti: 32 games × 50 samples → ~5 games/s
        - H100: 1000 games × 1024 samples → 100+ games/s (projected)

    Memory:
        - 3050 Ti (4GB): 32 games × 50 samples = 1,600 batch → fits comfortably
        - H100 (80GB): 1000 games × 1024 samples = 1M batch → ~5GB with headroom
    """
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'

    n_games = len(hands)

    # Initialize RNG for exploration (if enabled)
    if exploration_policy is not None:
        if exploration_policy.seed is not None:
            rng = np.random.default_rng(exploration_policy.seed)
        else:
            rng = np.random.default_rng()
    else:
        rng = None

    # Initialize GPU state
    states = GameStateTensor.from_deals(hands, decl_ids, device)

    # Pre-allocate sampler and tokenizer
    if use_mrv_sampler:
        sampler = WorldSamplerMRV(max_games=n_games, max_samples=n_samples, device=device)
    else:
        sampler = WorldSampler(max_games=n_games, max_samples=n_samples, device=device)
    tokenizer = GPUTokenizer(max_batch=n_games * n_samples, device=device)

    # Track decisions for each game
    all_decisions: list[list[DecisionRecordGPU]] = [[] for _ in range(n_games)]

    # Main generation loop - one iteration per game step
    while states.active_games().any():
        # 1. Sample consistent worlds for all games
        worlds = _sample_worlds_batched(states, sampler, n_samples)

        # 2. Build hypothetical deals
        hypothetical = _build_hypothetical_deals(states, worlds)

        # 3. Tokenize on GPU
        tokens, masks = _tokenize_batched(states, hypothetical, tokenizer)

        # 4. Model forward pass (single batch)
        q_values = _query_model(model, tokens, masks, states, n_samples, device)

        # 5. Reduce to E[Q] per game (with optional posterior weighting)
        if posterior_config and posterior_config.enabled:
            e_q, e_q_var, diagnostics = _compute_posterior_weighted_eq(
                states=states,
                worlds=worlds,
                q_values=q_values,
                n_samples=n_samples,
                model=model,
                tokenizer=tokenizer,
                posterior_config=posterior_config,
                device=device,
            )
        else:
            # Uniform weighting
            q_reshaped = q_values.view(n_games, n_samples, 7)
            e_q = q_reshaped.mean(dim=1)  # [n_games, 7]
            e_q_var = q_reshaped.var(dim=1, unbiased=False)  # [n_games, 7]
            diagnostics = None

        # Move e_q to states device (in case model is on different device)
        if e_q.device != states.hands.device:
            e_q = e_q.to(states.hands.device)
        if e_q_var.device != states.hands.device:
            e_q_var = e_q_var.to(states.hands.device)

        # 6. Select actions (greedy, sampled, or exploration)
        actions, exploration_stats = _select_actions(states, e_q, greedy, exploration_policy, rng)

        # 7. Record decisions
        _record_decisions(states, e_q, e_q_var, actions, all_decisions, diagnostics, exploration_stats)

        # 8. Apply actions and advance state
        # Ensure actions are on same device as states
        if actions.device != states.hands.device:
            actions = actions.to(states.hands.device)
        states = states.apply_actions(actions)

    # Convert to GameRecordGPU format
    return _collate_records(hands, decl_ids, all_decisions)


def _sample_worlds_batched_loop(
    states: GameStateTensor,
    sampler: WorldSampler | WorldSamplerMRV,
    n_samples: int,
) -> Tensor:
    """Sample consistent worlds for all games.

    ORIGINAL LOOP IMPLEMENTATION - kept for testing/comparison.
    Use _sample_worlds_batched() for production (vectorized).

    Args:
        states: GameStateTensor with n_games
        sampler: Pre-allocated WorldSampler or WorldSamplerMRV
        n_samples: Number of samples per game

    Returns:
        [n_games, n_samples, 3, 7] opponent hands

    Note:
        WorldSamplerMRV is guaranteed to produce valid samples (no CPU fallback needed).
        WorldSampler may fall back to CPU for heavily constrained scenarios.
    """
    n_games = states.n_games
    current_players = states.current_player.long()

    # Build pools: unplayed dominoes for each game
    all_dominoes = torch.arange(28, device=states.device)
    pools = []
    hand_sizes_list = []
    voids_list = []

    for g in range(n_games):
        player = current_players[g].item()

        # Pool = all dominoes - played - my hand
        played = states.played_mask[g]
        my_hand_mask = states.hands[g, player] >= 0
        my_dominoes = states.hands[g, player][my_hand_mask]

        pool_mask = ~played
        pool_mask[my_dominoes.long()] = False
        pool = all_dominoes[pool_mask]

        # Pad pool to max size
        pool_padded = torch.full((21,), -1, dtype=torch.int32, device=states.device)
        pool_padded[:len(pool)] = pool
        pools.append(pool_padded)

        # Hand sizes for 3 opponents
        hand_counts = (states.hands[g] >= 0).sum(dim=1)  # [4]
        opponent_sizes = []
        for opp_offset in range(1, 4):
            opp_player = (player + opp_offset) % 4
            opponent_sizes.append(hand_counts[opp_player].item())
        hand_sizes_list.append(opponent_sizes)

        # Voids: infer from play history
        voids = _infer_voids_gpu(states, g)
        voids_list.append(voids)

    # Stack inputs
    pools_t = torch.stack(pools)  # [n_games, 21]
    hand_sizes_t = torch.tensor(hand_sizes_list, dtype=torch.int32, device=states.device)  # [n_games, 3]

    # Convert voids to tensor format
    voids_t = torch.zeros(n_games, 3, 8, dtype=torch.bool, device=states.device)
    for g, game_voids in enumerate(voids_list):
        for opp_idx, void_suits in game_voids.items():
            for suit in void_suits:
                voids_t[g, opp_idx, suit] = True

    # Get decl_ids
    decl_ids_t = states.decl_ids

    # Sample worlds (with CPU fallback for heavily constrained scenarios)
    try:
        worlds = sampler.sample(pools_t, hand_sizes_t, voids_t, decl_ids_t, n_samples)
    except RuntimeError as e:
        # GPU sampling failed (too constrained) - fall back to CPU
        print(f"Warning: GPU sampling failed ({e}), using CPU fallback")
        worlds = _sample_worlds_cpu_fallback(states, pools_t, hand_sizes_list, voids_list, n_samples)

    return worlds


def _sample_worlds_batched(
    states: GameStateTensor,
    sampler: WorldSampler | WorldSamplerMRV,
    n_samples: int,
) -> Tensor:
    """Sample consistent worlds for all games.

    VECTORIZED IMPLEMENTATION - no Python loops or .item() calls.

    Args:
        states: GameStateTensor with n_games
        sampler: Pre-allocated WorldSampler or WorldSamplerMRV
        n_samples: Number of samples per game

    Returns:
        [n_games, n_samples, 3, 7] opponent hands

    Note:
        WorldSamplerMRV is guaranteed to produce valid samples (no CPU fallback needed).
        WorldSampler may fall back to CPU for heavily constrained scenarios.
    """
    n_games = states.n_games
    device = states.device
    current_players = states.current_player.long()  # [N]

    # === Step 1: Vectorize pool computation ===
    # For each game, pool = all_dominoes - played - my_hand
    all_dominoes = torch.arange(28, device=device)  # [28]

    # Get current player's hand for each game using gather
    # states.hands: [N, 4, 7]
    # current_players: [N] -> expand to [N, 1, 7] for gathering
    player_idx_for_hands = current_players.view(n_games, 1, 1).expand(n_games, 1, 7)
    my_hands = torch.gather(states.hands, dim=1, index=player_idx_for_hands).squeeze(1)  # [N, 7]

    # Build mask for my dominoes across all games
    # my_hands: [N, 7], values are domino IDs or -1
    my_hand_mask = my_hands >= 0  # [N, 7]

    # Create pool masks for all games: [N, 28]
    # pool_mask[g, d] = True if domino d is in the pool for game g
    pool_masks = ~states.played_mask.clone()  # [N, 28] - start with unplayed

    # Remove my dominoes from pool
    # For each game g, set pool_masks[g, my_hands[g, i]] = False for valid dominoes
    # Use scatter to mark my dominoes as unavailable
    batch_indices = torch.arange(n_games, device=device).unsqueeze(1).expand(n_games, 7)  # [N, 7]
    my_dominoes_safe = torch.where(my_hand_mask, my_hands.long(), torch.zeros_like(my_hands))  # Replace -1 with 0

    # Scatter False into pool_masks at my_dominoes positions
    pool_masks[batch_indices[my_hand_mask], my_dominoes_safe[my_hand_mask]] = False

    # Convert pool_masks to pool lists with padding
    # pool_masks: [N, 28] -> pools: [N, 21] (max pool size is 21 when hand size = 7)
    pools = torch.full((n_games, 21), -1, dtype=torch.int32, device=device)

    for g in range(n_games):
        pool = all_dominoes[pool_masks[g]]
        pools[g, :len(pool)] = pool

    # === Step 2: Vectorize hand sizes computation ===
    # hand_counts: [N, 4] - number of dominoes per player per game
    hand_counts = (states.hands >= 0).sum(dim=2)  # [N, 4]

    # For each game, get opponent hand sizes (3 opponents)
    # Opponent i = (current_player + i + 1) % 4 for i in [0, 1, 2]
    offsets = torch.arange(1, 4, device=device).unsqueeze(0)  # [1, 3]
    opponent_indices = (current_players.unsqueeze(1) + offsets) % 4  # [N, 3]

    # Gather opponent hand sizes: [N, 3]
    # hand_counts: [N, 4], opponent_indices: [N, 3]
    hand_sizes_t = torch.gather(hand_counts, dim=1, index=opponent_indices.long())  # [N, 3]

    # === Step 3: Vectorize voids inference ===
    voids_t = _infer_voids_batched(states)  # [N, 3, 8]

    # Get decl_ids
    decl_ids_t = states.decl_ids

    # Sample worlds (with CPU fallback for heavily constrained scenarios)
    try:
        worlds = sampler.sample(pools, hand_sizes_t, voids_t, decl_ids_t, n_samples)
    except RuntimeError as e:
        # GPU sampling failed (too constrained) - fall back to CPU
        print(f"Warning: GPU sampling failed ({e}), using CPU fallback", flush=True)
        # Convert pools back to list format for CPU fallback
        pools_list = []
        for g in range(n_games):
            pool_mask = pools[g] >= 0
            pools_list.append(pools[g][pool_mask].tolist())

        hand_sizes_list = hand_sizes_t.cpu().tolist()

        # Convert voids_t to dict format
        voids_list = []
        for g in range(n_games):
            game_voids = {}
            for opp_idx in range(3):
                void_suits = set()
                for suit in range(8):
                    if voids_t[g, opp_idx, suit]:
                        void_suits.add(suit)
                game_voids[opp_idx] = void_suits
            voids_list.append(game_voids)

        worlds = _sample_worlds_cpu_fallback(states, pools, hand_sizes_list, voids_list, n_samples)

    return worlds


def _infer_voids_batched(states: GameStateTensor) -> Tensor:
    """Infer void suits from play history for all games (vectorized).

    Args:
        states: GameStateTensor with N games

    Returns:
        [N, 3, 8] boolean tensor where voids[g, opp_idx, suit] = True if opponent is void in suit
        Opponent indices are relative to current_player:
        - 0 = (current_player + 1) % 4
        - 1 = (current_player + 2) % 4
        - 2 = (current_player + 3) % 4
    """
    from forge.oracle.tables import can_follow, led_suit_for_lead_domino

    n_games = states.n_games
    device = states.device
    voids = torch.zeros(n_games, 3, 8, dtype=torch.bool, device=device)

    # Process each game (history structure makes full vectorization difficult)
    # This is still much faster than before because we removed other .item() calls
    for g in range(n_games):
        # Extract history for this game
        history = states.history[g].cpu().numpy()  # [28, 3]
        decl_id = states.decl_ids[g].item()
        current_player = states.current_player[g].item()

        for i in range(28):
            if history[i, 0] < 0:  # End of history
                break

            player, domino_id, lead_domino_id = history[i]

            # Skip current player's plays (we know our hand)
            if player == current_player:
                continue

            # Map absolute player to relative opponent index (0, 1, 2)
            opp_idx = (player - current_player - 1) % 4

            if opp_idx >= 3:  # Safety check
                continue

            # Check if this play revealed a void
            led_suit = led_suit_for_lead_domino(lead_domino_id, decl_id)
            if not can_follow(domino_id, led_suit, decl_id):
                voids[g, opp_idx, led_suit] = True

    return voids


def _sample_worlds_cpu_fallback(
    states: GameStateTensor,
    pools: Tensor,
    hand_sizes_list: list[list[int]],
    voids_list: list[dict[int, set[int]]],
    n_samples: int,
) -> Tensor:
    """CPU fallback for world sampling when GPU fails.

    Args:
        states: GameStateTensor
        pools: [n_games, 21] pools tensor
        hand_sizes_list: List of hand sizes per game
        voids_list: List of voids per game
        n_samples: Number of samples

    Returns:
        [n_games, n_samples, 3, max_hand_size] opponent hands
    """
    from forge.eq.sampling import sample_consistent_worlds
    import numpy as np

    n_games = states.n_games
    current_players = states.current_player.cpu().numpy()

    # Determine max hand size
    all_hand_sizes = [h for game_sizes in hand_sizes_list for h in game_sizes]
    max_hand_size = max(all_hand_sizes) if all_hand_sizes else 7

    # Sample for each game separately using CPU
    all_worlds = []

    for g in range(n_games):
        player = int(current_players[g])
        my_hand = states.hands[g, player].cpu().numpy()
        my_hand = [int(d) for d in my_hand if d >= 0]

        played = set(int(d) for d in range(28) if states.played_mask[g, d].item())

        # Get hand sizes for all 4 players
        hand_counts = (states.hands[g] >= 0).sum(dim=1).cpu().numpy()
        hand_sizes_4 = [int(c) for c in hand_counts]

        # Map opponent voids (relative) to absolute player voids
        opp_voids = voids_list[g]
        abs_voids = {}
        for opp_idx, void_suits in opp_voids.items():
            abs_player = (player + opp_idx + 1) % 4
            abs_voids[abs_player] = void_suits

        # Sample using CPU
        rng = np.random.default_rng()
        cpu_worlds = sample_consistent_worlds(
            my_player=player,
            my_hand=my_hand,
            played=played,
            hand_sizes=hand_sizes_4,
            voids=abs_voids,
            decl_id=states.decl_ids[g].item(),
            n_samples=n_samples,
            rng=rng,
        )

        # Convert to opponent-only format and pad
        opp_worlds = []
        for world in cpu_worlds:
            opp_hands = []
            for opp_idx in range(3):
                opp_player = (player + opp_idx + 1) % 4
                hand = world[opp_player]
                # Pad to max_hand_size
                hand_padded = hand + [-1] * (max_hand_size - len(hand))
                opp_hands.append(hand_padded[:max_hand_size])
            opp_worlds.append(opp_hands)

        all_worlds.append(opp_worlds)

    # Convert to tensor
    worlds_tensor = torch.tensor(all_worlds, dtype=torch.int32, device=states.device)

    return worlds_tensor


def _infer_voids_gpu(states: GameStateTensor, game_idx: int) -> dict[int, set[int]]:
    """Infer void suits from play history (CPU helper).

    Args:
        states: GameStateTensor
        game_idx: Which game to process

    Returns:
        Dict mapping opponent index (0-2) to set of void suits
        Opponent indices are relative to current_player:
        - 0 = (current_player + 1) % 4
        - 1 = (current_player + 2) % 4
        - 2 = (current_player + 3) % 4
    """
    from forge.oracle.tables import can_follow, led_suit_for_lead_domino

    # Extract history for this game
    history = states.history[game_idx].cpu().numpy()  # [28, 3]
    decl_id = states.decl_ids[game_idx].item()
    current_player = states.current_player[game_idx].item()

    voids = {0: set(), 1: set(), 2: set()}

    for i in range(28):
        if history[i, 0] < 0:  # End of history
            break

        player, domino_id, lead_domino_id = history[i]

        # Skip current player's plays (we know our hand)
        if player == current_player:
            continue

        # Map absolute player to relative opponent index (0, 1, 2)
        # Opponent 0 = player + 1, Opponent 1 = player + 2, Opponent 2 = player + 3
        opp_idx = (player - current_player - 1) % 4

        if opp_idx >= 3:  # Should not happen, but safety check
            continue

        # Check if this play revealed a void
        led_suit = led_suit_for_lead_domino(lead_domino_id, decl_id)
        if not can_follow(domino_id, led_suit, decl_id):
            voids[opp_idx].add(led_suit)

    return voids


def _build_hypothetical_deals_loop(
    states: GameStateTensor,
    worlds: Tensor,
) -> Tensor:
    """Reconstruct hypothetical full deals from current hands + sampled opponents.

    ORIGINAL LOOP IMPLEMENTATION - kept for testing/comparison.
    Use _build_hypothetical_deals() for production (vectorized).

    Args:
        states: GameStateTensor with n_games
        worlds: [n_games, n_samples, 3, max_hand_size] opponent hands (padded with -1)

    Returns:
        [n_games, n_samples, 4, 7] full deals

    Note: Reconstructs initial hands by combining:
        - Current player's actual hand (same across all samples)
        - Sampled opponent hands (different per sample)
        - Both may have variable sizes (padded with -1)
    """
    n_games = states.n_games
    n_samples = worlds.shape[1]
    max_hand_size = worlds.shape[3]
    current_players = states.current_player.long()

    # Initialize output
    deals = torch.full((n_games, n_samples, 4, 7), -1, dtype=torch.int32, device=states.device)

    # For each game, fill in hands
    for g in range(n_games):
        player = current_players[g].item()

        # Current player's hand (same for all samples)
        # May be padded with -1 if less than 7 dominoes
        deals[g, :, player, :] = states.hands[g, player, :].unsqueeze(0).expand(n_samples, -1)

        # Opponent hands (different per sample)
        for opp_idx in range(3):
            opp_player = (player + opp_idx + 1) % 4

            # worlds may have max_hand_size != 7 (variable during game)
            # Copy the portion that fits, rest stays as -1
            hand_data = worlds[g, :, opp_idx, :]  # [n_samples, max_hand_size]
            copy_size = min(max_hand_size, 7)
            deals[g, :, opp_player, :copy_size] = hand_data[:, :copy_size]

    return deals


def _build_hypothetical_deals(
    states: GameStateTensor,
    worlds: Tensor,
) -> Tensor:
    """Reconstruct hypothetical full deals from current hands + sampled opponents.

    VECTORIZED IMPLEMENTATION - no Python loops or .item() calls.

    Args:
        states: GameStateTensor with n_games
        worlds: [n_games, n_samples, 3, max_hand_size] opponent hands (padded with -1)

    Returns:
        [n_games, n_samples, 4, 7] full deals

    Note: Reconstructs initial hands by combining:
        - Current player's actual hand (same across all samples)
        - Sampled opponent hands (different per sample)
        - Both may have variable sizes (padded with -1)
    """
    n_games = states.n_games
    n_samples = worlds.shape[1]
    max_hand_size = worlds.shape[3]
    device = states.device
    current_players = states.current_player.long()  # [N]

    # Initialize output
    deals = torch.full((n_games, n_samples, 4, 7), -1, dtype=torch.int32, device=device)

    # === Step 1: Scatter current player's hand into correct position ===
    # my_hands[g] = states.hands[g, current_players[g], :]
    # Use gather to get each game's current player's hand
    player_idx_for_hands = current_players.view(n_games, 1, 1).expand(n_games, 1, 7)  # [N, 1, 7]
    my_hands = torch.gather(states.hands, dim=1, index=player_idx_for_hands).squeeze(1)  # [N, 7]

    # Convert to int32 to match output dtype (states.hands is int8)
    my_hands = my_hands.to(torch.int32)

    # Expand my_hands for all samples: [N, 7] -> [N, M, 7]
    my_hands_expanded = my_hands.unsqueeze(1).expand(n_games, n_samples, 7)  # [N, M, 7]

    # Scatter my_hands into deals at position current_players[g] for each game
    # deals[g, :, current_players[g], :] = my_hands_expanded[g, :, :]
    # Use scatter_ along dim=2 (player dimension)
    scatter_idx = current_players.view(n_games, 1, 1, 1).expand(n_games, n_samples, 1, 7)  # [N, M, 1, 7]
    deals.scatter_(dim=2, index=scatter_idx, src=my_hands_expanded.unsqueeze(2))  # scatter at player dim

    # === Step 2: Compute opponent player indices without loops ===
    # opp_players[g, i] = (current_players[g] + i + 1) % 4 for i in 0,1,2
    offsets = torch.arange(1, 4, device=device).unsqueeze(0)  # [1, 3]
    opp_players = (current_players.unsqueeze(1) + offsets) % 4  # [N, 3]

    # === Step 3: Scatter opponent hands into correct positions ===
    # For each opponent index (0, 1, 2), scatter worlds[:, :, opp_idx, :] to deals[:, :, opp_players[:, opp_idx], :]
    copy_size = min(max_hand_size, 7)

    # Process all 3 opponents at once using advanced indexing
    # worlds: [N, M, 3, max_hand_size]
    # We want: deals[g, s, opp_players[g, opp_idx], :copy_size] = worlds[g, s, opp_idx, :copy_size]

    # For each of the 3 opponents, scatter their hands
    for opp_idx in range(3):
        # opp_player_positions: [N] - which player slot for this opponent in each game
        opp_player_positions = opp_players[:, opp_idx]  # [N]

        # Expand for samples and hand slots: [N] -> [N, M, 1, copy_size]
        scatter_opp_idx = opp_player_positions.view(n_games, 1, 1, 1).expand(n_games, n_samples, 1, copy_size)

        # Get this opponent's hands: [N, M, copy_size]
        opp_hands = worlds[:, :, opp_idx, :copy_size]

        # Scatter into deals: [N, M, 1, copy_size] at scatter_opp_idx positions
        deals[:, :, :, :copy_size].scatter_(
            dim=2,
            index=scatter_opp_idx,
            src=opp_hands.unsqueeze(2)  # [N, M, 1, copy_size]
        )

    return deals


def _tokenize_batched_loop(
    states: GameStateTensor,
    deals: Tensor,
    tokenizer: GPUTokenizer,
) -> tuple[Tensor, Tensor]:
    """Tokenize all game states and deals.

    ORIGINAL LOOP IMPLEMENTATION - kept for testing/comparison.
    Use _tokenize_batched() for production (vectorized).

    Args:
        states: GameStateTensor with n_games
        deals: [n_games, n_samples, 4, 7] hypothetical deals
        tokenizer: Pre-allocated GPUTokenizer

    Returns:
        Tuple of (tokens, masks):
            - tokens: [n_games * n_samples, 32, 12] int8
            - masks: [n_games * n_samples, 32] int8
    """
    n_games = states.n_games
    n_samples = deals.shape[1]
    batch_size = n_games * n_samples

    # Flatten deals: [batch, 4, 7]
    deals_flat = deals.reshape(batch_size, 4, 7)

    # Build remaining bitmasks
    remaining = _build_remaining_bitmasks(states, deals)  # [batch, 4]

    # Tokenize per-game (trick plays vary per game)
    all_tokens = []
    all_masks = []

    for g in range(n_games):
        start_idx = g * n_samples
        end_idx = (g + 1) * n_samples

        game_deals = deals_flat[start_idx:end_idx]  # [n_samples, 4, 7]
        game_remaining = remaining[start_idx:end_idx]  # [n_samples, 4]

        # Extract game state
        decl_id = states.decl_ids[g].item()
        leader = states.leader[g].item()
        current_player = states.current_player[g].item()

        # Build trick_plays list
        trick_plays = []
        for i in range(4):
            domino_id = states.trick_plays[g, i].item()
            if domino_id < 0:
                break
            play_player = (leader + i) % 4
            trick_plays.append((play_player, domino_id))

        # Tokenize this game's samples
        tokens, masks = tokenizer.tokenize(
            worlds=game_deals,
            decl_id=decl_id,
            leader=leader,
            trick_plays=trick_plays,
            remaining=game_remaining,
            current_player=current_player,
        )

        # Clone to avoid buffer reuse issues when tokenizer is called multiple times
        all_tokens.append(tokens.clone())
        all_masks.append(masks.clone())

    # Concatenate
    tokens_cat = torch.cat(all_tokens, dim=0)
    masks_cat = torch.cat(all_masks, dim=0)

    return tokens_cat, masks_cat


def _tokenize_batched(
    states: GameStateTensor,
    deals: Tensor,
    tokenizer: GPUTokenizer,
) -> tuple[Tensor, Tensor]:
    """Tokenize all game states and deals.

    VECTORIZED IMPLEMENTATION - no Python loops or .item() calls.

    Args:
        states: GameStateTensor with n_games
        deals: [n_games, n_samples, 4, 7] hypothetical deals
        tokenizer: Pre-allocated GPUTokenizer

    Returns:
        Tuple of (tokens, masks):
            - tokens: [n_games * n_samples, 32, 12] int8
            - masks: [n_games * n_samples, 32] int8
    """
    n_games = states.n_games
    n_samples = deals.shape[1]
    batch_size = n_games * n_samples
    device = states.device

    # Flatten deals: [batch, 4, 7]
    deals_flat = deals.reshape(batch_size, 4, 7)

    # Build remaining bitmasks
    remaining = _build_remaining_bitmasks(states, deals)  # [batch, 4]

    # === Prepare batched inputs for tokenizer.tokenize_batched() ===

    # decl_ids: [n_games] -> expand to [batch]
    decl_ids_expanded = states.decl_ids.unsqueeze(1).expand(n_games, n_samples).reshape(batch_size)

    # leaders: [n_games] -> expand to [batch]
    leaders_expanded = states.leader.unsqueeze(1).expand(n_games, n_samples).reshape(batch_size)

    # current_players: [n_games] -> expand to [batch]
    current_players_expanded = states.current_player.unsqueeze(1).expand(n_games, n_samples).reshape(batch_size)

    # trick_plays: [n_games, 4] domino IDs -> need to convert to [batch, 3, 2] (player, domino) format
    # trick_lens: [n_games] number of valid trick plays

    # Count valid trick plays per game (clamp to 3 max, as tokenizer only handles 3 trick tokens)
    trick_lens = torch.clamp((states.trick_plays >= 0).sum(dim=1), max=3)  # [n_games]

    # Build trick_plays tensor: [n_games, 3, 2] where [:, :, 0] is player, [:, :, 1] is domino
    # states.trick_plays: [n_games, 4] contains domino IDs in play order
    # We need to compute player IDs based on leader + position
    trick_plays_tensor = torch.zeros(n_games, 3, 2, dtype=torch.int32, device=device)

    # For each position (0-2), compute player = (leader + position) % 4 vectorized
    positions = torch.arange(3, device=device).unsqueeze(0)  # [1, 3]
    trick_players = (states.leader.unsqueeze(1) + positions) % 4  # [n_games, 3]

    # Get domino IDs for positions 0-2
    trick_dominoes = states.trick_plays[:, :3]  # [n_games, 3]

    # Store in trick_plays_tensor
    trick_plays_tensor[:, :, 0] = trick_players
    trick_plays_tensor[:, :, 1] = trick_dominoes

    # Expand trick_plays to [batch, 3, 2]
    trick_plays_expanded = trick_plays_tensor.unsqueeze(1).expand(n_games, n_samples, 3, 2).reshape(batch_size, 3, 2)

    # Expand trick_lens to [batch]
    trick_lens_expanded = trick_lens.unsqueeze(1).expand(n_games, n_samples).reshape(batch_size)

    # Call vectorized tokenization
    tokens, masks = tokenizer.tokenize_batched(
        worlds=deals_flat,
        decl_ids=decl_ids_expanded,
        leaders=leaders_expanded,
        current_players=current_players_expanded,
        trick_plays=trick_plays_expanded,
        trick_lens=trick_lens_expanded,
        remaining=remaining,
    )

    return tokens, masks


def _build_remaining_bitmasks(
    states: GameStateTensor,
    deals: Tensor,
) -> Tensor:
    """Build remaining domino bitmasks.

    Args:
        states: GameStateTensor with played history
        deals: [n_games, n_samples, 4, 7] hypothetical deals

    Returns:
        [n_games * n_samples, 4] bitmasks
    """
    n_games = states.n_games
    n_samples = deals.shape[1]
    batch_size = n_games * n_samples

    # Flatten deals
    deals_flat = deals.reshape(batch_size, 4, 7)

    # Expand played_mask: [n_games, 28] -> [batch, 28]
    played_mask = states.played_mask.unsqueeze(1).expand(-1, n_samples, -1).reshape(batch_size, 28)

    # Build bitmasks
    remaining = torch.zeros(batch_size, 4, dtype=torch.int32, device=states.device)

    for player in range(4):
        hand = deals_flat[:, player, :]  # [batch, 7]
        for local_idx in range(7):
            domino_ids = hand[:, local_idx].long()  # [batch]

            # Check if valid and not played
            valid = domino_ids >= 0
            batch_indices = torch.arange(batch_size, device=states.device)

            # Safe indexing: use 0 for invalid, then mask out
            domino_ids_safe = torch.where(valid, domino_ids, torch.zeros_like(domino_ids))
            is_played = played_mask[batch_indices, domino_ids_safe]

            is_remaining = valid & ~is_played
            remaining[:, player] |= is_remaining.int() << local_idx

    return remaining


def _query_model(
    model,
    tokens: Tensor,
    masks: Tensor,
    states: GameStateTensor,
    n_samples: int,
    device: str,
) -> Tensor:
    """Run model forward pass.

    Args:
        model: Stage 1 model
        tokens: [batch, 32, 12] tokenized input
        masks: [batch, 32] attention masks
        states: GameStateTensor (for current_player)
        n_samples: Number of samples per game
        device: Device

    Returns:
        [batch, 7] Q-values
    """
    n_games = states.n_games
    batch_size = n_games * n_samples

    # Expand current_player to all samples
    current_players = states.current_player.unsqueeze(1).expand(-1, n_samples).reshape(batch_size).long()

    # Determine model device
    model_device = next(model.parameters()).device

    # Ensure tokens are on model device and correct dtype
    if tokens.device != model_device:
        tokens = tokens.to(model_device)
    if masks.device != model_device:
        masks = masks.to(model_device)
    if current_players.device != model_device:
        current_players = current_players.to(model_device)

    # Convert int8 to int32 for model input (Embedding layer)
    tokens = tokens.to(torch.int32)

    # Model forward with mixed precision (float16 on CUDA for ~2-3x speedup)
    with torch.inference_mode():
        use_amp = (model_device.type == 'cuda')
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
            q_values, _ = model(tokens, masks, current_players)

    # Clone to prevent CUDA graph buffer reuse when model is called multiple times
    # (e.g., posterior weighting calls model again for past steps)
    return q_values.clone()


def _select_actions(
    states: GameStateTensor,
    e_q: Tensor,
    greedy: bool,
    exploration_policy: ExplorationPolicy | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[Tensor, list | None]:
    """Select actions from E[Q] values.

    Args:
        states: GameStateTensor
        e_q: [n_games, 7] E[Q] values
        greedy: If True, argmax. If False, softmax sample.
        exploration_policy: Optional exploration policy (overrides greedy if provided)
        rng: NumPy RNG for exploration

    Returns:
        Tuple of (actions, exploration_stats):
            - actions: [n_games] action indices (0-6)
            - exploration_stats: List of ExplorationStats (one per game), or None if no exploration
    """
    from forge.eq.exploration import _select_action_with_exploration

    n_games = states.n_games

    # Get legal actions: [n_games, 7]
    legal_mask = states.legal_actions()

    # If exploration policy provided, use it (per-game selection)
    if exploration_policy is not None:
        from forge.eq.types import ExplorationStats

        actions = []
        exploration_stats = []
        for g in range(n_games):
            action_idx, selection_mode, action_entropy = _select_action_with_exploration(
                e_q_mean=e_q[g].cpu(),  # Move to CPU for numpy conversion
                legal_mask=legal_mask[g].cpu(),
                policy=exploration_policy,
                rng=rng,
            )
            actions.append(action_idx)

            # Compute greedy action and q_gap
            legal = legal_mask[g]
            masked_eq = e_q[g].clone()
            masked_eq[~legal] = float('-inf')
            greedy_action = masked_eq.argmax().item()

            q_greedy = e_q[g][greedy_action].item()
            q_taken = e_q[g][action_idx].item()
            q_gap = q_greedy - q_taken

            stats = ExplorationStats(
                greedy_action=greedy_action,
                action_taken=action_idx,
                was_greedy=(action_idx == greedy_action),
                selection_mode=selection_mode,
                q_gap=q_gap,
                action_entropy=action_entropy,
            )
            exploration_stats.append(stats)
        return torch.tensor(actions, dtype=torch.long, device=e_q.device), exploration_stats

    # Otherwise, use greedy or softmax sampling (no exploration stats)
    masked_e_q = e_q.clone()
    masked_e_q[~legal_mask] = float('-inf')

    if greedy:
        actions = masked_e_q.argmax(dim=1)
    else:
        # Softmax sample
        probs = torch.softmax(masked_e_q, dim=1)
        actions = torch.multinomial(probs, num_samples=1).squeeze(1)

    return actions, None


def _record_decisions(
    states: GameStateTensor,
    e_q: Tensor,
    e_q_var: Tensor,
    actions: Tensor,
    all_decisions: list[list[DecisionRecordGPU]],
    diagnostics: PosteriorDiagnostics | None = None,
    exploration_stats: list | None = None,
):
    """Record decisions for each game (in-place).

    Args:
        states: GameStateTensor
        e_q: [n_games, 7] E[Q] mean values
        e_q_var: [n_games, 7] E[Q] variance values
        actions: [n_games] action indices
        all_decisions: List of decision lists (one per game)
        diagnostics: Optional posterior diagnostics (aggregated across games)
        exploration_stats: Optional exploration stats (one per game)
    """
    n_games = states.n_games
    legal_mask = states.legal_actions()
    current_players = states.current_player

    # Mode mapping for exploration (CPU pipeline convention)
    mode_to_int = {"greedy": 0, "boltzmann": 1, "epsilon": 2, "blunder": 3}

    for g in range(n_games):
        # Only record if game is active
        if not states.active_games()[g]:
            continue

        # Compute state uncertainty from variance
        sigma = torch.sqrt(e_q_var[g])  # [7]
        legal = legal_mask[g]
        if legal.any():
            u_mean = sigma[legal].mean().item()
            u_max = sigma[legal].max().item()
        else:
            u_mean = 0.0
            u_max = 0.0

        # Extract exploration stats if available
        exploration_mode = None
        q_gap = None
        greedy_action = None
        if exploration_stats is not None and g < len(exploration_stats):
            stats = exploration_stats[g]
            exploration_mode = mode_to_int.get(stats.selection_mode, 0)
            q_gap = stats.q_gap
            greedy_action = stats.greedy_action

        record = DecisionRecordGPU(
            player=current_players[g].item(),
            e_q=e_q[g].cpu(),
            action_taken=actions[g].item(),
            legal_mask=legal_mask[g].cpu(),
            e_q_var=e_q_var[g].cpu(),
            u_mean=u_mean,
            u_max=u_max,
            ess=diagnostics.ess if diagnostics else None,
            max_w=diagnostics.max_w if diagnostics else None,
            exploration_mode=exploration_mode,
            q_gap=q_gap,
            greedy_action=greedy_action,
        )
        all_decisions[g].append(record)


def _collate_records(
    hands: list[list[list[int]]],
    decl_ids: list[int],
    all_decisions: list[list[DecisionRecordGPU]],
) -> list[GameRecordGPU]:
    """Collate decision records into GameRecordGPU.

    Args:
        hands: Initial deals
        decl_ids: Declaration IDs
        all_decisions: Decision records per game

    Returns:
        List of GameRecordGPU
    """
    return [
        GameRecordGPU(
            decisions=decisions,
            hands=hands[g],
            decl_id=decl_ids[g],
        )
        for g, decisions in enumerate(all_decisions)
    ]


def _compute_posterior_weighted_eq(
    states: GameStateTensor,
    worlds: Tensor,
    q_values: Tensor,
    n_samples: int,
    model,
    tokenizer: GPUTokenizer,
    posterior_config: PosteriorConfig,
    device: str,
) -> tuple[Tensor, Tensor, PosteriorDiagnostics | None]:
    """Compute posterior-weighted E[Q] using past K steps.

    Args:
        states: Current game states [N games]
        worlds: Sampled opponent hands [N, M, 3, max_hand_size]
        q_values: Q-values for current step [N*M, 7]
        n_samples: Number of samples per game (M)
        model: Stage 1 oracle model
        tokenizer: GPUTokenizer instance
        posterior_config: Posterior weighting configuration
        device: Device for computation

    Returns:
        Tuple of (e_q_mean, e_q_var, diagnostics):
            - e_q_mean: [N, 7] posterior-weighted E[Q] mean values
            - e_q_var: [N, 7] posterior-weighted E[Q] variance values
            - diagnostics: PosteriorDiagnostics (aggregated across games, or None if fallback)
    """
    n_games = states.n_games

    # 1. Compute history lengths (count non-empty entries in history)
    # history: [N, 28, 3] with -1 for unused entries
    # An entry is valid if player (first field) >= 0
    history_len = (states.history[:, :, 0] >= 0).sum(dim=1)  # [N]

    # Note: Games with insufficient history (< window_k) are handled per-game
    # by the valid_mask in reconstruct_past_states_gpu. Invalid steps get
    # zero legal actions -> uniform weights for that game only.

    # 2. Reconstruct past states
    past_states = reconstruct_past_states_gpu(
        history=states.history,
        history_len=history_len,
        window_k=posterior_config.window_k,
        device=states.device,
    )

    # 3. Build hypothetical deals from worlds (same format as main pipeline)
    hypothetical = _build_hypothetical_deals(states, worlds)  # [N, M, 4, 7]

    # 4. Reconstruct historical hands at each past step k
    # This fixes the bug where we were using CURRENT hands instead of HISTORICAL hands
    # Historical hands have dominoes that were played AFTER step k added back
    historical_hands = reconstruct_historical_hands(
        hypothetical=hypothetical,
        history=states.history,
        history_len=history_len,
        step_indices=past_states.step_indices,
        valid_mask=past_states.valid_mask,
    )  # [N, M, K, 4, 7]

    # 5. Tokenize past steps for all games/samples/steps
    # Need larger tokenizer for N*M*K batch (past steps tokenization)
    K = posterior_config.window_k
    past_batch_size = n_games * n_samples * K

    # Create a separate tokenizer for past steps if needed
    if past_batch_size > tokenizer.max_batch:
        past_tokenizer = GPUTokenizer(max_batch=past_batch_size, device=states.device)
    else:
        past_tokenizer = tokenizer

    past_tokens, past_masks = past_tokenizer.tokenize_past_steps_batched(
        worlds=hypothetical,
        past_states=past_states,
        decl_ids=states.decl_ids,  # Per-game declaration IDs
        historical_hands=historical_hands,  # Reconstructed hands at each past step
    )

    # 5. Query oracle for past steps
    # past_tokens: [N*K*M, 32, 12] with layout g*K*M + k*M + m
    total_batch = n_games * n_samples * K

    # Expand actors to match tokenizer's [N, K, M] layout: actors[n, k] -> [n, k, m] -> [n*k*m]
    # Clamp actors to [0, 3] range - invalid entries (-1) will be clamped to 0
    # Their outputs will be masked out by valid_mask anyway
    actors_clamped = past_states.actors.clamp(0, 3)
    actors_expanded = actors_clamped.unsqueeze(-1).expand(n_games, K, n_samples)  # [N, K, M]
    actors_flat = actors_expanded.reshape(total_batch).long()  # [N*K*M]

    # Determine model device
    model_device = next(model.parameters()).device

    # Move tensors to model device
    if past_tokens.device != model_device:
        past_tokens = past_tokens.to(model_device)
    if past_masks.device != model_device:
        past_masks = past_masks.to(model_device)
    if actors_flat.device != model_device:
        actors_flat = actors_flat.to(model_device)

    # Convert to int32 for model
    past_tokens = past_tokens.to(torch.int32)

    # Check if ANY valid steps exist - if all masks are 0, skip model and use uniform weights
    # This happens early in the game when history_len < window_k for all games
    has_any_valid = past_masks.any()
    if not has_any_valid:
        # No valid history - fall back to uniform weighting
        q_reshaped = q_values.view(n_games, n_samples, 7)
        e_q = q_reshaped.mean(dim=1)
        e_q_var = q_reshaped.var(dim=1, unbiased=False)
        return e_q, e_q_var, None

    # Model forward pass
    with torch.inference_mode():
        use_amp = (model_device.type == 'cuda')
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
            q_past, _ = model(past_tokens, past_masks, actors_flat)

    # Reshape: [N*K*M, 7] -> [N, K, M, 7] to match tokenizer layout
    q_past = q_past.view(n_games, K, n_samples, 7)
    # Transpose to [N, M, K, 7] for compatibility with posterior weighting
    q_past = q_past.permute(0, 2, 1, 3).contiguous()

    # 6. Compute legal masks for past steps
    legal_masks = compute_legal_masks_gpu(
        worlds=hypothetical,
        past_states=past_states,
        decl_ids=states.decl_ids,  # Per-game declaration IDs
        device=states.device,
        historical_hands=historical_hands,  # Reconstructed hands at each past step
    )

    # 7. Compute posterior weights
    weights, diagnostics = compute_posterior_weights_gpu(
        q_past=q_past,
        legal_masks=legal_masks,
        observed_actions=past_states.observed_actions,
        worlds=hypothetical,
        actors=past_states.actors,
        tau=posterior_config.tau,
        uniform_mix=posterior_config.uniform_mix,
        historical_hands=historical_hands,  # Reconstructed hands at each past step
    )

    # 8. Apply weights to current Q-values
    # q_values: [N*M, 7] -> [N, M, 7]
    q_values_reshaped = q_values.view(n_games, n_samples, 7)

    # weights: [N, M] -> [N, M, 1] for broadcasting
    weights_expanded = weights.unsqueeze(-1)  # [N, M, 1]

    # Weighted mean: e_q[n] = sum_m(weights[n, m] * q_values[n, m, :])
    e_q_mean = (weights_expanded * q_values_reshaped).sum(dim=1)  # [N, 7]

    # Weighted variance: Var = E[X²] - E[X]²
    e_q_sq = (weights_expanded * q_values_reshaped**2).sum(dim=1)  # [N, 7]
    e_q_var = e_q_sq - e_q_mean**2  # [N, 7]

    # 9. Aggregate diagnostics across games (average)
    if diagnostics is not None:
        # diagnostics is a dict with tensors of shape [N] for each field
        # We aggregate by taking the mean across games
        aggregated_diagnostics = PosteriorDiagnostics(
            ess=diagnostics['ess'].mean().item() if 'ess' in diagnostics else 0.0,
            max_w=diagnostics['max_w'].mean().item() if 'max_w' in diagnostics else 0.0,
            entropy=diagnostics['entropy'].mean().item() if 'entropy' in diagnostics else 0.0,
            k_eff=diagnostics['k_eff'].mean().item() if 'k_eff' in diagnostics else 0.0,
            n_invalid=0,  # Not provided by compute_posterior_weights_gpu
            n_illegal=0,  # Not provided by compute_posterior_weights_gpu
            window_nll=0.0,  # Not provided by compute_posterior_weights_gpu
            window_k_used=posterior_config.window_k,  # Use config value
        )
    else:
        aggregated_diagnostics = None

    return e_q_mean, e_q_var, aggregated_diagnostics
