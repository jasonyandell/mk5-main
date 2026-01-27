"""GPU-native E[Q] generation pipeline.

Generates training data by playing games using a Stage 1 oracle to estimate
E[Q] (expected Q-value) for each legal action. The pipeline runs entirely on
GPU for both game state management and model inference.

Core Loop (per decision):
    GameStateTensor          N games running in parallel
           |
    WorldSampler             Sample M opponent hands per game
           |                 (constrained by voids, played dominoes)
    Build hypothetical       Combine known hand + sampled opponents
           |                 -> [N, M, 4, 7] full deals
    GPUTokenizer             Tokenize current decision
           |                 -> [N*M, 32, 12] tokens
    Model forward            Query oracle for Q-values
           |                 -> [N*M, 7] Q per action
    E[Q] aggregation         Mean or posterior-weighted
           |                 -> [N, 7] expected Q per action
    Action selection         Greedy, softmax, or exploration policy
           |
    apply_actions            Advance game state
           -> repeat until all games complete

Performance (measured on 3050 Ti, 32 games x 50 samples):
- Uniform:   6.5 games/s
- Posterior: 2.4 games/s (2.7x slower due to second model call for K past steps)
"""

from __future__ import annotations

import numpy as np
import torch

from forge.eq.enumeration_gpu import WorldEnumeratorGPU
from forge.eq.game_tensor import GameStateTensor
from forge.eq.sampling_gpu import WorldSampler
from forge.eq.sampling_mrv_gpu import WorldSamplerMRV
from forge.eq.tokenize_gpu import GPUTokenizer
from forge.eq.types import ExplorationPolicy

from .actions import collate_records, record_decisions, select_actions
from .adaptive import sample_until_convergence, sample_until_convergence_posterior
from .deals import build_hypothetical_deals
from .enumeration import enumerate_or_sample_worlds
from .eq_compute import compute_eq_pdf, compute_eq_with_counts
from .model import query_model
from .posterior import compute_posterior_weighted_eq
from .sampling import sample_worlds_batched
from .tokenization import tokenize_batched
from .types import AdaptiveConfig, DecisionRecordGPU, GameRecordGPU, PosteriorConfig


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
    use_enumeration: bool = False,
    enumeration_threshold: int = 100_000,
    adaptive_config: AdaptiveConfig | None = None,
    seeds: list[int] | None = None,
) -> list[GameRecordGPU]:
    """Generate E[Q] games entirely on GPU.

    Args:
        model: Stage 1 oracle model (DominoLightningModule or wrapper)
        hands: List of N initial deals, each is 4 hands of 7 domino IDs
        decl_ids: List of N declaration IDs
        n_samples: Number of worlds to sample per decision (ignored if adaptive_config
                   is enabled, or if use_enumeration=True and world count is below
                   enumeration_threshold)
        device: Device to run on ('cuda')
        greedy: If True, always pick argmax(E[Q]). If False, sample from softmax.
        use_mrv_sampler: If True (default), use MRV-based sampler (guaranteed valid).
                        If False, use rejection sampling.
        exploration_policy: Optional exploration policy for stochastic action selection.
                          If provided, overrides greedy parameter.
        posterior_config: Optional config for posterior weighting. If provided and enabled,
                         uses past K steps to reweight worlds before computing E[Q].
        use_enumeration: If True, enumerate ALL valid worlds instead of sampling when
                        the estimated world count is below enumeration_threshold.
                        This gives exact E[Q] for late-game positions.
        enumeration_threshold: Maximum worlds to enumerate per game. Above this,
                              falls back to sampling. Default 100,000.
        adaptive_config: Optional config for adaptive convergence-based sampling.
                        If provided and enabled, samples in batches until E[Q] converges
                        (max SEM < threshold) or max_samples is reached.

    Returns:
        List of N GameRecordGPU, one per game

    Performance:
        - 3050 Ti: 32 games x 50 samples -> ~5 games/s
        - H100: 1000 games x 1024 samples -> 100+ games/s (projected)

    Memory:
        - 3050 Ti (4GB): 32 games x 50 samples = 1,600 batch -> fits comfortably
        - H100 (80GB): 1000 games x 1024 samples = 1M batch -> ~5GB with headroom
    """
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError("CUDA not available. GPU-only pipeline requires CUDA.")

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

    # Determine sample allocation for adaptive vs fixed sampling
    use_adaptive = adaptive_config is not None and adaptive_config.enabled
    if use_adaptive:
        # For adaptive: allocate for batch_size (we sample incrementally)
        sampler_max_samples = adaptive_config.batch_size
        tokenizer_samples = adaptive_config.batch_size
    else:
        sampler_max_samples = n_samples
        tokenizer_samples = n_samples

    # Pre-allocate sampler and tokenizer
    if use_mrv_sampler:
        sampler = WorldSamplerMRV(max_games=n_games, max_samples=sampler_max_samples, device=device)
    else:
        sampler = WorldSampler(max_games=n_games, max_samples=sampler_max_samples, device=device)

    # Pre-allocate enumerator if enabled
    enumerator = None
    if use_enumeration:
        enumerator = WorldEnumeratorGPU(
            max_games=n_games,
            max_worlds=enumeration_threshold,
            device=device,
        )

    # Tokenizer batch size: start with batch_size, will recreate if needed
    # Don't pre-allocate for full enumeration_threshold (causes OOM on small GPUs)
    tokenizer = GPUTokenizer(max_batch=n_games * tokenizer_samples, device=device)

    # Track decisions for each game
    all_decisions: list[list[DecisionRecordGPU]] = [[] for _ in range(n_games)]

    # Decision index counter for logging
    decision_idx = 0

    # Main generation loop - one iteration per game step
    while states.active_games().any():
        # Track samples used and convergence (for adaptive mode)
        n_samples_used = None
        did_converge = None

        # Hybrid mode: decide between enumeration vs adaptive/fixed based on game phase
        # - Late game (history >= 12): try enumeration if enabled
        # - Early/mid game: use adaptive if enabled, else fixed sampling
        history_len = (states.history[:, :, 0] >= 0).sum(dim=1).min().item()
        should_enumerate = use_enumeration and history_len >= 12

        use_posterior = posterior_config is not None and posterior_config.enabled

        if use_adaptive and not should_enumerate:
            # Adaptive convergence-based sampling
            if use_posterior:
                # Adaptive + posterior: use weighted statistics
                e_q, e_q_var, e_q_pdf, diagnostics, n_samples_used, did_converge = sample_until_convergence_posterior(
                    states=states,
                    sampler=sampler,
                    tokenizer=tokenizer,
                    model=model,
                    adaptive_config=adaptive_config,
                    posterior_config=posterior_config,
                    device=device,
                    decision_idx=decision_idx,
                    seeds=seeds,
                )
            else:
                # Adaptive without posterior: simple statistics
                e_q, e_q_var, e_q_pdf, diagnostics, n_samples_used, did_converge = sample_until_convergence(
                    states=states,
                    sampler=sampler,
                    tokenizer=tokenizer,
                    model=model,
                    adaptive_config=adaptive_config,
                    device=device,
                    decision_idx=decision_idx,
                    seeds=seeds,
                )
        else:
            # Fixed sampling (original behavior)
            # 1. Sample or enumerate consistent worlds for all games
            if use_enumeration:
                worlds, world_counts, actual_n_samples = enumerate_or_sample_worlds(
                    states, enumerator, sampler, n_samples, enumeration_threshold
                )
            else:
                worlds = sample_worlds_batched(states, sampler, n_samples)
                world_counts = None
                actual_n_samples = n_samples

            # 2. Build hypothetical deals
            hypothetical = build_hypothetical_deals(states, worlds)

            # 3. Tokenize on GPU (recreate tokenizer if batch exceeds capacity)
            # Use worlds.shape[1] (not actual_n_samples) because the tensor may be padded
            n_worlds_padded = worlds.shape[1]
            batch_size = n_games * n_worlds_padded
            if batch_size > tokenizer.max_batch:
                tokenizer = GPUTokenizer(max_batch=batch_size, device=device)
            tokens, masks = tokenize_batched(states, hypothetical, tokenizer)

            # 4. Model forward pass (single batch)
            q_values = query_model(model, tokens, masks, states, n_worlds_padded, device)

            # 5. Reduce to E[Q] per game (with optional posterior weighting)
            q_reshaped = q_values.view(n_games, n_worlds_padded, 7)

            if posterior_config and posterior_config.enabled:
                e_q, e_q_var, e_q_pdf, diagnostics = compute_posterior_weighted_eq(
                    states=states,
                    worlds=worlds,
                    q_values=q_values,
                    n_samples=n_worlds_padded,
                    model=model,
                    tokenizer=tokenizer,
                    posterior_config=posterior_config,
                    device=device,
                )
            else:
                # Uniform weighting (handles variable world counts from enumeration)
                if world_counts is not None:
                    # Enumeration: use world_counts for proper averaging
                    e_q, e_q_var = compute_eq_with_counts(q_reshaped, world_counts)
                    e_q_pdf = compute_eq_pdf(q_reshaped, weights=None, world_counts=world_counts)
                else:
                    # Sampling: all games have same sample count
                    e_q = q_reshaped.mean(dim=1)  # [n_games, 7]
                    e_q_var = q_reshaped.var(dim=1, unbiased=False)  # [n_games, 7]
                    e_q_pdf = compute_eq_pdf(q_reshaped)
                diagnostics = None

        # Move tensors to states device (in case model is on different device)
        if e_q.device != states.hands.device:
            e_q = e_q.to(states.hands.device)
        if e_q_var.device != states.hands.device:
            e_q_var = e_q_var.to(states.hands.device)
        if e_q_pdf.device != states.hands.device:
            e_q_pdf = e_q_pdf.to(states.hands.device)

        # 6. Select actions by p_make (greedy, sampled, or exploration)
        actions, exploration_stats = select_actions(states, e_q, e_q_pdf, greedy, exploration_policy, rng)

        # 7. Record decisions
        record_decisions(states, e_q, e_q_var, e_q_pdf, actions, all_decisions, diagnostics, exploration_stats, n_samples_used, did_converge)

        # 8. Apply actions and advance state
        # Ensure actions are on same device as states
        if actions.device != states.hands.device:
            actions = actions.to(states.hands.device)
        states = states.apply_actions(actions)

        # Increment decision counter
        decision_idx += 1

    # Convert to GameRecordGPU format
    return collate_records(hands, decl_ids, all_decisions)
