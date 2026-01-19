"""GPU-native E[Q] generation pipeline.

Phase 4 of 4-phase GPU pipeline (t42-k7ny):
- Phase 1: GameStateTensor - GPU game state representation ✓
- Phase 2: WorldSampler - GPU world sampling ✓
- Phase 3: GPUTokenizer - GPU tokenization ✓
- Phase 4: Full GPU pipeline integration (this file)

Architecture:
    GameStateTensor (N games on GPU)
        ↓
    WorldSampler (N × M samples)
        ↓
    GPUTokenizer (pure tensor ops)
        ↓
    Model forward (single batch)
        ↓
    E[Q] aggregation → best actions
        ↓
    apply_actions (vectorized)
        └──→ repeat until games done

Performance targets:
    - 3050 Ti (4GB): 32 games × 50 samples → 5+ games/s
    - H100 (80GB): 1000 games × 1024 samples → 100+ games/s
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
    reconstruct_past_states_gpu,
)
from forge.eq.sampling_gpu import WorldSampler
from forge.eq.sampling_mrv_gpu import WorldSamplerMRV
from forge.eq.tokenize_gpu import GPUTokenizer
from forge.eq.types import ExplorationPolicy


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

    Minimal record focusing on E[Q] values and actions.
    For full posterior/exploration features, use CPU pipeline.
    """
    player: int  # Which player made the decision (0-3)
    e_q: Tensor  # [7] E[Q] values (padded, -inf for illegal/empty)
    action_taken: int  # Slot index (0-6) of domino played
    legal_mask: Tensor  # [7] boolean mask of legal actions


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
            e_q = _compute_posterior_weighted_eq(
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
            e_q = q_values.view(n_games, n_samples, 7).mean(dim=1)  # [n_games, 7]

        # Move e_q to states device (in case model is on different device)
        if e_q.device != states.hands.device:
            e_q = e_q.to(states.hands.device)

        # 6. Select actions (greedy, sampled, or exploration)
        actions = _select_actions(states, e_q, greedy, exploration_policy, rng)

        # 7. Record decisions
        _record_decisions(states, e_q, actions, all_decisions)

        # 8. Apply actions and advance state
        # Ensure actions are on same device as states
        if actions.device != states.hands.device:
            actions = actions.to(states.hands.device)
        states = states.apply_actions(actions)

    # Convert to GameRecordGPU format
    return _collate_records(hands, decl_ids, all_decisions)


def _sample_worlds_batched(
    states: GameStateTensor,
    sampler: WorldSampler | WorldSamplerMRV,
    n_samples: int,
) -> Tensor:
    """Sample consistent worlds for all games.

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


def _tokenize_batched(
    states: GameStateTensor,
    deals: Tensor,
    tokenizer: GPUTokenizer,
) -> tuple[Tensor, Tensor]:
    """Tokenize all game states and deals.

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

        all_tokens.append(tokens)
        all_masks.append(masks)

    # Concatenate
    tokens_cat = torch.cat(all_tokens, dim=0)
    masks_cat = torch.cat(all_masks, dim=0)

    return tokens_cat, masks_cat


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

    return q_values


def _select_actions(
    states: GameStateTensor,
    e_q: Tensor,
    greedy: bool,
    exploration_policy: ExplorationPolicy | None = None,
    rng: np.random.Generator | None = None,
) -> Tensor:
    """Select actions from E[Q] values.

    Args:
        states: GameStateTensor
        e_q: [n_games, 7] E[Q] values
        greedy: If True, argmax. If False, softmax sample.
        exploration_policy: Optional exploration policy (overrides greedy if provided)
        rng: NumPy RNG for exploration

    Returns:
        [n_games] action indices (0-6)
    """
    from forge.eq.exploration import _select_action_with_exploration

    n_games = states.n_games

    # Get legal actions: [n_games, 7]
    legal_mask = states.legal_actions()

    # If exploration policy provided, use it (per-game selection)
    if exploration_policy is not None:
        actions = []
        for g in range(n_games):
            action_idx, _, _ = _select_action_with_exploration(
                e_q_mean=e_q[g].cpu(),  # Move to CPU for numpy conversion
                legal_mask=legal_mask[g].cpu(),
                policy=exploration_policy,
                rng=rng,
            )
            actions.append(action_idx)
        return torch.tensor(actions, dtype=torch.long, device=e_q.device)

    # Otherwise, use greedy or softmax sampling
    masked_e_q = e_q.clone()
    masked_e_q[~legal_mask] = float('-inf')

    if greedy:
        actions = masked_e_q.argmax(dim=1)
    else:
        # Softmax sample
        probs = torch.softmax(masked_e_q, dim=1)
        actions = torch.multinomial(probs, num_samples=1).squeeze(1)

    return actions


def _record_decisions(
    states: GameStateTensor,
    e_q: Tensor,
    actions: Tensor,
    all_decisions: list[list[DecisionRecordGPU]],
):
    """Record decisions for each game (in-place).

    Args:
        states: GameStateTensor
        e_q: [n_games, 7] E[Q] values
        actions: [n_games] action indices
        all_decisions: List of decision lists (one per game)
    """
    n_games = states.n_games
    legal_mask = states.legal_actions()
    current_players = states.current_player

    for g in range(n_games):
        # Only record if game is active
        if not states.active_games()[g]:
            continue

        record = DecisionRecordGPU(
            player=current_players[g].item(),
            e_q=e_q[g].cpu(),
            action_taken=actions[g].item(),
            legal_mask=legal_mask[g].cpu(),
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
) -> Tensor:
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
        e_q: [N, 7] posterior-weighted E[Q] values
    """
    n_games = states.n_games

    # 1. Compute history lengths (count non-empty entries in history)
    # history: [N, 28, 3] with -1 for unused entries
    # An entry is valid if player (first field) >= 0
    history_len = (states.history[:, :, 0] >= 0).sum(dim=1)  # [N]

    # If there's insufficient history (less than K steps for any game), fall back to uniform
    if (history_len < posterior_config.window_k).any():
        # Not enough history for posterior weighting, use uniform mean
        return q_values.view(n_games, n_samples, 7).mean(dim=1)  # [n_games, 7]

    # 2. Reconstruct past states
    past_states = reconstruct_past_states_gpu(
        history=states.history,
        history_len=history_len,
        window_k=posterior_config.window_k,
        device=states.device,
    )

    # 3. Build hypothetical deals from worlds (same format as main pipeline)
    hypothetical = _build_hypothetical_deals(states, worlds)  # [N, M, 4, 7]

    # 4. Tokenize past steps for all games/samples/steps
    # Need larger tokenizer for N*M*K batch (past steps tokenization)
    K = posterior_config.window_k
    past_batch_size = n_games * n_samples * K

    # Create a separate tokenizer for past steps if needed
    if past_batch_size > tokenizer.max_batch:
        past_tokenizer = GPUTokenizer(max_batch=past_batch_size, device=states.device)
    else:
        past_tokenizer = tokenizer

    past_tokens, past_masks = past_tokenizer.tokenize_past_steps(
        worlds=hypothetical,
        past_states=past_states,
        decl_id=states.decl_ids[0].item(),  # Assume single decl_id for batch
    )

    # 5. Query oracle for past steps
    # past_tokens: [N*M*K, 32, 12], we need to expand current_player info
    total_batch = n_games * n_samples * K

    # Expand actors to all samples: actors[n, k] -> [n, m, k] -> [n*m*k]
    actors_expanded = past_states.actors.unsqueeze(1).expand(n_games, n_samples, K)  # [N, M, K]
    actors_flat = actors_expanded.reshape(total_batch).long()  # [N*M*K]

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

    # Model forward pass
    with torch.inference_mode():
        use_amp = (model_device.type == 'cuda')
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
            q_past, _ = model(past_tokens, past_masks, actors_flat)

    # Reshape: [N*M*K, 7] -> [N, M, K, 7]
    q_past = q_past.view(n_games, n_samples, K, 7)

    # 6. Compute legal masks for past steps
    legal_masks = compute_legal_masks_gpu(
        worlds=hypothetical,
        past_states=past_states,
        decl_id=states.decl_ids[0].item(),
        device=states.device,
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
    )

    # 8. Apply weights to current Q-values
    # q_values: [N*M, 7] -> [N, M, 7]
    q_values_reshaped = q_values.view(n_games, n_samples, 7)

    # weights: [N, M] -> [N, M, 1] for broadcasting
    weights_expanded = weights.unsqueeze(-1)  # [N, M, 1]

    # Weighted sum: e_q[n] = sum_m(weights[n, m] * q_values[n, m, :])
    e_q = (weights_expanded * q_values_reshaped).sum(dim=1)  # [N, 7]

    return e_q
