"""Posterior-weighted world marginalization for E[Q] generation (t42-64uj.3)."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from forge.eq.oracle import Stage1Oracle
from forge.eq.rejuvenation import _rejuvenate_particles
from forge.eq.types import MappingIntegrityError, PosteriorConfig, PosteriorDiagnostics
from forge.oracle.tables import can_follow, led_suit_for_lead_domino


def compute_posterior_weights(
    oracle: Stage1Oracle,
    hypothetical_deals: list[list[list[int]]],
    play_history: list[tuple[int, int, int]],
    decl_id: int,
    config: PosteriorConfig,
    rng: np.random.Generator | None = None,
) -> tuple[Tensor, PosteriorDiagnostics]:
    """Compute posterior weights for sampled worlds based on transcript likelihood."""
    n_worlds = len(hypothetical_deals)
    device = oracle.device
    if rng is None:
        rng = np.random.default_rng()

    if not play_history:
        weights = torch.ones(n_worlds, device=device) / n_worlds
        return weights, PosteriorDiagnostics(
            ess=float(n_worlds),
            max_w=1.0 / n_worlds,
            entropy=np.log(n_worlds),
            k_eff=float(n_worlds),
            window_k_used=0,
        )

    current_k = min(config.window_k, len(play_history))
    max_k = (
        min(config.adaptive_k_max, len(play_history))
        if config.adaptive_k_enabled
        else current_k
    )

    while True:
        weights, diagnostics = _compute_weights_for_window(
            oracle=oracle,
            hypothetical_deals=hypothetical_deals,
            play_history=play_history,
            decl_id=decl_id,
            config=config,
            window_k=current_k,
            rng=rng,
        )

        if not config.adaptive_k_enabled:
            break
        if diagnostics.ess >= config.adaptive_k_ess_threshold:
            break
        if current_k >= max_k:
            break

        next_k = min(current_k + config.adaptive_k_step, max_k)
        if next_k == current_k:
            break
        current_k = next_k

    return weights, diagnostics


def compute_posterior_weights_many(
    *,
    oracle: Stage1Oracle,
    hypothetical_deals_list: list[list[list[list[int]]]],
    play_histories: list[list[tuple[int, int, int]]],
    decl_ids: list[int],
    config: PosteriorConfig,
    rngs: list[np.random.Generator | None] | None = None,
) -> tuple[list[Tensor], list[PosteriorDiagnostics]]:
    """Compute posterior weights for multiple games, batching oracle calls by decl_id."""
    n_games = len(hypothetical_deals_list)
    if len(play_histories) != n_games or len(decl_ids) != n_games:
        raise ValueError(
            "hypothetical_deals_list, play_histories, and decl_ids must have same length"
        )

    if rngs is None:
        rngs = [None] * n_games
    if len(rngs) != n_games:
        raise ValueError("rngs must have same length as hypothetical_deals_list")

    if config.adaptive_k_enabled or config.rejuvenation_enabled:
        weights_out: list[Tensor] = []
        diags_out: list[PosteriorDiagnostics] = []
        for deals, history, decl_id, rng in zip(
            hypothetical_deals_list, play_histories, decl_ids, rngs
        ):
            w, d = compute_posterior_weights(
                oracle=oracle,
                hypothetical_deals=deals,
                play_history=history,
                decl_id=decl_id,
                config=config,
                rng=rng,
            )
            weights_out.append(w)
            diags_out.append(d)
        return weights_out, diags_out

    device = oracle.device
    weights_out: list[Tensor | None] = [None] * n_games
    diags_out: list[PosteriorDiagnostics | None] = [None] * n_games

    to_score: list[int] = []
    for i in range(n_games):
        n_worlds = len(hypothetical_deals_list[i])
        if not play_histories[i]:
            w = torch.ones(n_worlds, device=device) / n_worlds
            d = PosteriorDiagnostics(
                ess=float(n_worlds),
                max_w=1.0 / n_worlds,
                entropy=np.log(n_worlds),
                k_eff=float(n_worlds),
                window_k_used=0,
            )
            weights_out[i] = w
            diags_out[i] = d
        else:
            to_score.append(i)

    if not to_score:
        if any(w is None for w in weights_out) or any(d is None for d in diags_out):
            raise AssertionError(
                "Internal error: missing weights/diagnostics for some games"
            )
        return [w for w in weights_out if w is not None], [
            d for d in diags_out if d is not None
        ]

    by_decl: dict[int, list[int]] = defaultdict(list)
    for idx in to_score:
        by_decl[decl_ids[idx]].append(idx)

    for decl_id, game_indices in by_decl.items():
        worlds_all: list[list[list[int]]] = []
        trick_plays_all: list[list[tuple[int, int]]] = []

        actors_chunks: list[np.ndarray] = []
        leaders_chunks: list[np.ndarray] = []
        remaining_chunks: list[np.ndarray] = []

        metas: list[dict] = []
        cursor = 0

        for game_i in game_indices:
            hypothetical_deals = hypothetical_deals_list[game_i]
            play_history = play_histories[game_i]
            n_worlds = len(hypothetical_deals)

            window_k = min(config.window_k, len(play_history))
            window_start = max(0, len(play_history) - window_k)
            window_plays = play_history[window_start:]
            n_steps = len(window_plays)

            if n_steps == 0:
                w = torch.ones(n_worlds, device=device) / n_worlds
                d = PosteriorDiagnostics(
                    ess=float(n_worlds),
                    max_w=1.0 / n_worlds,
                    entropy=np.log(n_worlds),
                    k_eff=float(n_worlds),
                    window_k_used=0,
                )
                weights_out[game_i] = w
                diags_out[game_i] = d
                continue

            step_infos = []
            for step_offset, (actor, observed_domino, lead_domino) in enumerate(
                window_plays
            ):
                step_idx = window_start + step_offset

                played_before = {domino for _, domino, _ in play_history[:step_idx]}

                trick_start = step_idx
                while trick_start > 0:
                    prev_idx = trick_start - 1
                    _, _, prev_lead = play_history[prev_idx]
                    if prev_lead == lead_domino:
                        trick_start = prev_idx
                    else:
                        break

                current_trick = [(p, d) for p, d, _ in play_history[trick_start:step_idx]]
                leader = current_trick[0][0] if current_trick else actor
                is_leading = step_idx == trick_start

                step_infos.append(
                    {
                        "actor": actor,
                        "observed_domino": observed_domino,
                        "lead_domino": lead_domino,
                        "played_before": played_before,
                        "current_trick": current_trick,
                        "leader": leader,
                        "is_leading": is_leading,
                        "step_idx": step_idx,
                    }
                )

            total_samples = n_steps * n_worlds
            worlds_all.extend(hypothetical_deals * n_steps)

            actors = np.zeros(total_samples, dtype=np.int32)
            leaders = np.zeros(total_samples, dtype=np.int32)
            remaining = np.zeros((total_samples, 4), dtype=np.int32)

            obs_local_idx = np.full(total_samples, -1, dtype=np.int16)
            legal_mask = np.zeros((total_samples, 7), dtype=np.bool_)
            step_n_invalid = np.zeros(n_steps, dtype=np.int32)
            step_n_illegal = np.zeros(n_steps, dtype=np.int32)

            local_trick_plays: list[list[tuple[int, int]]] = []
            for step_offset, info in enumerate(step_infos):
                base_idx = step_offset * n_worlds
                actor = info["actor"]
                played_before = info["played_before"]
                lead_domino = info["lead_domino"]
                is_leading = info["is_leading"]
                observed_domino = info["observed_domino"]

                for world_idx, initial_hands in enumerate(hypothetical_deals):
                    sample_idx = base_idx + world_idx
                    actor_hand = initial_hands[actor]

                    actors[sample_idx] = actor
                    leaders[sample_idx] = info["leader"]
                    local_trick_plays.append(info["current_trick"])

                    if observed_domino not in actor_hand:
                        step_n_invalid[step_offset] += 1
                    else:
                        obs_local_idx[sample_idx] = actor_hand.index(observed_domino)

                    legal_local = _get_legal_local_indices(
                        actor_hand, played_before, lead_domino, decl_id, is_leading
                    )
                    if not legal_local:
                        if config.strict_integrity:
                            raise MappingIntegrityError(
                                f"No legal actions for actor {actor} at step {info['step_idx']} in world {world_idx}"
                            )
                        step_n_illegal[step_offset] += 1
                    else:
                        for local_idx in legal_local:
                            legal_mask[sample_idx, local_idx] = True
                        local_idx_obs = int(obs_local_idx[sample_idx])
                        if local_idx_obs >= 0 and not legal_mask[sample_idx, local_idx_obs]:
                            if config.strict_integrity:
                                raise MappingIntegrityError(
                                    f"Observed domino {observed_domino} (local_idx={local_idx_obs}) is in actor {actor}'s "
                                    f"hand but not legal at step {info['step_idx']}. Legal indices: {legal_local}."
                                )
                            step_n_illegal[step_offset] += 1

                    for p in range(4):
                        for local_idx, domino in enumerate(initial_hands[p]):
                            if domino not in played_before:
                                remaining[sample_idx, p] |= 1 << local_idx

            trick_plays_all.extend(local_trick_plays)
            actors_chunks.append(actors)
            leaders_chunks.append(leaders)
            remaining_chunks.append(remaining)

            metas.append(
                {
                    "game_i": game_i,
                    "n_worlds": n_worlds,
                    "n_steps": n_steps,
                    "window_k_used": n_steps,
                    "step_n_invalid": step_n_invalid,
                    "step_n_illegal": step_n_illegal,
                    "obs_local_idx": obs_local_idx,
                    "legal_mask": legal_mask,
                    "slice": slice(cursor, cursor + total_samples),
                }
            )
            cursor += total_samples

        if not metas:
            continue

        actors_all = np.concatenate(actors_chunks, axis=0)
        leaders_all = np.concatenate(leaders_chunks, axis=0)
        remaining_all = np.concatenate(remaining_chunks, axis=0)

        if hasattr(oracle, "query_batch_multi_state"):
            q_all = oracle.query_batch_multi_state(
                worlds=worlds_all,
                decl_id=decl_id,
                actors=actors_all,
                leaders=leaders_all,
                trick_plays_list=trick_plays_all,
                remaining=remaining_all,
            )
        else:
            # Slow fallback for test doubles.
            q_chunks: list[Tensor] = []
            for si, world in enumerate(worlds_all):
                game_state_info = {
                    "decl_id": decl_id,
                    "leader": int(leaders_all[si]),
                    "trick_plays": trick_plays_all[si],
                    "remaining": remaining_all[si : si + 1],
                }
                q_chunks.append(oracle.query_batch([world], game_state_info, int(actors_all[si])))
            q_all = torch.cat(q_chunks, dim=0)

        q_device = q_all.device
        q = q_all.to(dtype=torch.float32)

        for meta in metas:
            sl = meta["slice"]
            n_worlds = meta["n_worlds"]
            n_steps = meta["n_steps"]

            obs_local_idx = meta["obs_local_idx"]
            legal_mask = meta["legal_mask"]

            obs_idx_t = torch.from_numpy(obs_local_idx.astype(np.int64)).to(device=q_device)
            legal_mask_t = torch.from_numpy(legal_mask).to(device=q_device, dtype=torch.bool)

            legal_f = legal_mask_t.to(dtype=torch.float32)
            legal_count = legal_f.sum(dim=1)
            has_legal = legal_count > 0

            denom = legal_count.clamp(min=1.0)
            mean_q_legal = (q[sl] * legal_f).sum(dim=1) / denom
            advantage = (q[sl] - mean_q_legal.unsqueeze(1)) / config.tau
            advantage = advantage.masked_fill(~legal_mask_t, float("-inf"))

            p_soft = F.softmax(advantage, dim=1)
            uniform = legal_f / denom.unsqueeze(1)
            p_mixed = (1 - config.beta) * p_soft + config.beta * uniform

            safe_obs = obs_idx_t.clamp(min=0)
            obs_prob = p_mixed.gather(dim=1, index=safe_obs.unsqueeze(1)).squeeze(1)
            invalid_obs = obs_idx_t < 0
            log_prob = torch.log(obs_prob + 1e-30)
            log_prob = torch.where(
                invalid_obs | (~has_legal),
                torch.full_like(log_prob, -1e9),
                log_prob,
            )

            log_prob = log_prob.view(n_steps, n_worlds)
            logw = log_prob.sum(dim=0)

            total_logp = logw.mean().item()
            max_logw = logw.max()
            logw_stable = torch.clamp(logw - max_logw, min=-config.delta)
            weights = F.softmax(logw_stable, dim=0)

            ess = 1.0 / (weights * weights).sum().item()
            max_w = weights.max().item()

            log_weights = torch.log(weights + 1e-30)
            entropy = -(weights * log_weights).sum().item()
            k_eff = float(np.exp(entropy))

            window_nll = -total_logp / n_steps if n_steps > 0 else 0.0

            mitigation = ""
            rejuvenation_applied = False
            rejuvenation_accepts = 0

            if config.mitigation_enabled and ess < config.ess_warn:
                uniform_w = torch.ones(n_worlds, device=q_device) / n_worlds
                if ess < config.ess_critical:
                    alpha = config.mitigation_alpha
                    weights = (1 - alpha) * weights + alpha * uniform_w
                    mitigation = f"critical_mix(alpha={alpha:.2f})"
                else:
                    beta_boost = config.mitigation_beta_boost
                    weights = (1 - beta_boost) * weights + beta_boost * uniform_w
                    mitigation = f"warn_mix(beta_boost={beta_boost:.2f})"

                ess = 1.0 / (weights * weights).sum().item()
                max_w = weights.max().item()
                log_weights = torch.log(weights + 1e-30)
                entropy = -(weights * log_weights).sum().item()
                k_eff = float(np.exp(entropy))

            diagnostics = PosteriorDiagnostics(
                ess=ess,
                max_w=max_w,
                entropy=entropy,
                k_eff=k_eff,
                n_invalid=int(meta["step_n_invalid"].max())
                if len(meta["step_n_invalid"]) > 0
                else 0,
                n_illegal=int(meta["step_n_illegal"].sum())
                if len(meta["step_n_illegal"]) > 0
                else 0,
                window_nll=window_nll,
                window_k_used=n_steps,
                rejuvenation_applied=rejuvenation_applied,
                rejuvenation_accepts=rejuvenation_accepts,
                mitigation=mitigation,
            )

            weights_out[meta["game_i"]] = weights
            diags_out[meta["game_i"]] = diagnostics

    if any(w is None for w in weights_out) or any(d is None for d in diags_out):
        raise AssertionError("Internal error: missing weights/diagnostics for some games")
    return [w for w in weights_out if w is not None], [d for d in diags_out if d is not None]


def _compute_weights_for_window(
    oracle: Stage1Oracle,
    hypothetical_deals: list[list[list[int]]],
    play_history: list[tuple[int, int, int]],
    decl_id: int,
    config: PosteriorConfig,
    window_k: int,
    rng: np.random.Generator,
) -> tuple[Tensor, PosteriorDiagnostics]:
    """Compute posterior weights using a specific window size."""
    n_worlds = len(hypothetical_deals)
    device = oracle.device

    window_start = max(0, len(play_history) - window_k)
    window_plays = play_history[window_start:]

    if not window_plays:
        weights = torch.ones(n_worlds, device=device) / n_worlds
        return weights, PosteriorDiagnostics(
            ess=float(n_worlds),
            max_w=1.0 / n_worlds,
            entropy=np.log(n_worlds),
            k_eff=float(n_worlds),
            window_k_used=0,
        )

    logw, n_invalid, n_illegal = _score_all_steps_batched(
        oracle=oracle,
        hypothetical_deals=hypothetical_deals,
        play_history=play_history,
        window_plays=window_plays,
        window_start=window_start,
        decl_id=decl_id,
        config=config,
    )

    total_logp = logw.mean().item()
    n_scored_steps = len(window_plays)

    max_logw = logw.max()
    logw = logw - max_logw
    logw = torch.clamp(logw, min=-config.delta)

    weights = F.softmax(logw, dim=0)

    ess = 1.0 / (weights * weights).sum().item()
    max_w = weights.max().item()

    log_weights = torch.log(weights + 1e-30)
    entropy = -(weights * log_weights).sum().item()
    k_eff = np.exp(entropy)

    window_nll = -total_logp / n_scored_steps if n_scored_steps > 0 else 0.0

    mitigation = ""
    rejuvenation_applied = False
    rejuvenation_accepts = 0

    if config.mitigation_enabled and ess < config.ess_warn:
        uniform = torch.ones(n_worlds, device=device) / n_worlds

        if ess < config.ess_critical:
            alpha = config.mitigation_alpha
            weights = (1 - alpha) * weights + alpha * uniform
            mitigation = f"critical_mix(alpha={alpha:.2f})"
        else:
            beta_boost = config.mitigation_beta_boost
            weights = (1 - beta_boost) * weights + beta_boost * uniform
            mitigation = f"warn_mix(beta_boost={beta_boost:.2f})"

        ess = 1.0 / (weights * weights).sum().item()
        max_w = weights.max().item()
        log_weights = torch.log(weights + 1e-30)
        entropy = -(weights * log_weights).sum().item()
        k_eff = np.exp(entropy)

    if (
        config.rejuvenation_enabled
        and ess < config.rejuvenation_ess_threshold
        and len(play_history) > 0
    ):
        hypothetical_deals, rejuvenation_accepts = _rejuvenate_particles(
            hypothetical_deals=hypothetical_deals,
            weights=weights,
            play_history=play_history,
            decl_id=decl_id,
            oracle=oracle,
            config=config,
            window_k=window_k,
            rng=rng,
        )
        rejuvenation_applied = True

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
    """Score likelihood of observed action at one step for all worlds."""
    n_worlds = len(hypothetical_deals)
    device = oracle.device

    played_before = set()
    for i in range(step_idx):
        _, domino, _ = play_history[i]
        played_before.add(domino)

    trick_start = step_idx
    while trick_start > 0:
        prev_idx = trick_start - 1
        _, _, prev_lead = play_history[prev_idx]
        if prev_lead == lead_domino:
            trick_start = prev_idx
        else:
            break

    current_trick = []
    for i in range(trick_start, step_idx):
        p, d, _ = play_history[i]
        current_trick.append((p, d))

    leader = current_trick[0][0] if current_trick else actor
    is_leading = step_idx == trick_start

    actors = np.full(n_worlds, actor, dtype=np.int32)
    leaders = np.full(n_worlds, leader, dtype=np.int32)
    trick_plays_list = [current_trick] * n_worlds
    remaining = np.zeros((n_worlds, 4), dtype=np.int32)

    obs_local_idx = np.full(n_worlds, -1, dtype=np.int16)
    legal_mask = np.zeros((n_worlds, 7), dtype=np.bool_)

    n_invalid = 0
    n_illegal = 0

    for world_idx, initial_hands in enumerate(hypothetical_deals):
        actor_hand = initial_hands[actor]

        if observed_domino not in actor_hand:
            n_invalid += 1
            obs_local_idx[world_idx] = -1
        else:
            obs_local_idx[world_idx] = actor_hand.index(observed_domino)

        legal_local = _get_legal_local_indices(
            actor_hand, played_before, lead_domino, decl_id, is_leading
        )
        if not legal_local:
            if config.strict_integrity:
                raise MappingIntegrityError(
                    f"No legal actions for actor {actor} at step {step_idx} in world {world_idx}"
                )
            n_illegal += 1
        else:
            for local_idx in legal_local:
                legal_mask[world_idx, local_idx] = True

            local_idx_obs = int(obs_local_idx[world_idx])
            if local_idx_obs >= 0 and not legal_mask[world_idx, local_idx_obs]:
                if config.strict_integrity:
                    raise MappingIntegrityError(
                        f"Observed domino {observed_domino} (local_idx={local_idx_obs}) is in actor {actor}'s "
                        f"hand but not legal at step {step_idx}. Legal indices: {legal_local}."
                    )
                n_illegal += 1

        for p in range(4):
            for local_idx, domino in enumerate(initial_hands[p]):
                if domino not in played_before:
                    remaining[world_idx, p] |= 1 << local_idx

    if hasattr(oracle, "query_batch_multi_state"):
        all_q_values = oracle.query_batch_multi_state(
            worlds=hypothetical_deals,
            decl_id=decl_id,
            actors=actors,
            leaders=leaders,
            trick_plays_list=trick_plays_list,
            remaining=remaining,
        )
    else:
        q_chunks: list[Tensor] = []
        for si, world in enumerate(hypothetical_deals):
            game_state_info = {
                "decl_id": decl_id,
                "leader": int(leaders[si]),
                "trick_plays": trick_plays_list[si],
                "remaining": remaining[si : si + 1],
            }
            q_chunks.append(oracle.query_batch([world], game_state_info, int(actors[si])))
        all_q_values = torch.cat(q_chunks, dim=0)

    obs_idx_t = torch.from_numpy(obs_local_idx.astype(np.int64)).to(device=device)
    legal_mask_t = torch.from_numpy(legal_mask).to(device=device, dtype=torch.bool)

    q = all_q_values.to(dtype=torch.float32)
    legal_f = legal_mask_t.to(dtype=torch.float32)
    legal_count = legal_f.sum(dim=1)
    has_legal = legal_count > 0

    denom = legal_count.clamp(min=1.0)
    mean_q_legal = (q * legal_f).sum(dim=1) / denom
    advantage = (q - mean_q_legal.unsqueeze(1)) / config.tau
    advantage = advantage.masked_fill(~legal_mask_t, float("-inf"))

    p_soft = F.softmax(advantage, dim=1)
    uniform = legal_f / denom.unsqueeze(1)
    p_mixed = (1 - config.beta) * p_soft + config.beta * uniform

    safe_obs = obs_idx_t.clamp(min=0)
    obs_prob = p_mixed.gather(dim=1, index=safe_obs.unsqueeze(1)).squeeze(1)
    invalid_obs = obs_idx_t < 0
    log_prob = torch.log(obs_prob + 1e-30)

    log_prob = torch.where(
        invalid_obs | (~has_legal),
        torch.full_like(log_prob, -1e9),
        log_prob,
    )

    return log_prob, n_invalid, n_illegal


def _get_legal_local_indices(
    actor_hand: list[int],
    played_before: set[int],
    lead_domino: int | None,
    decl_id: int,
    is_leading: bool,
) -> list[int]:
    """Get local indices of legal actions for actor at this step."""
    remaining = [d for d in actor_hand if d not in played_before]
    remaining_local = [i for i, d in enumerate(actor_hand) if d not in played_before]

    if is_leading or lead_domino is None:
        return remaining_local

    led_suit = led_suit_for_lead_domino(lead_domino, decl_id)
    followers = [
        local_idx
        for local_idx, d in zip(remaining_local, remaining)
        if can_follow(d, led_suit, decl_id)
    ]
    return followers if followers else remaining_local


def _score_all_steps_batched(
    oracle: Stage1Oracle,
    hypothetical_deals: list[list[list[int]]],
    play_history: list[tuple[int, int, int]],
    window_plays: list[tuple[int, int, int]],
    window_start: int,
    decl_id: int,
    config: PosteriorConfig,
) -> tuple[Tensor, int, int]:
    """Score all K window steps in a SINGLE batched oracle call."""
    n_worlds = len(hypothetical_deals)
    n_steps = len(window_plays)
    device = oracle.device

    if n_steps == 0:
        return torch.zeros(n_worlds, device=device), 0, 0

    step_infos = []

    for step_offset, (actor, observed_domino, lead_domino) in enumerate(window_plays):
        step_idx = window_start + step_offset

        played_before = set()
        for i in range(step_idx):
            _, domino, _ = play_history[i]
            played_before.add(domino)

        trick_start = step_idx
        while trick_start > 0:
            prev_idx = trick_start - 1
            _, _, prev_lead = play_history[prev_idx]
            if prev_lead == lead_domino:
                trick_start = prev_idx
            else:
                break

        current_trick = []
        for i in range(trick_start, step_idx):
            p, d, _ = play_history[i]
            current_trick.append((p, d))

        leader = current_trick[0][0] if current_trick else actor
        is_leading = step_idx == trick_start

        step_infos.append(
            {
                "actor": actor,
                "observed_domino": observed_domino,
                "lead_domino": lead_domino,
                "played_before": played_before,
                "current_trick": current_trick,
                "leader": leader,
                "is_leading": is_leading,
                "step_idx": step_idx,
            }
        )

    total_samples = n_steps * n_worlds
    expanded_worlds = hypothetical_deals * n_steps

    actors = np.zeros(total_samples, dtype=np.int32)
    leaders = np.zeros(total_samples, dtype=np.int32)
    trick_plays_list: list[list[tuple[int, int]]] = []
    remaining = np.zeros((total_samples, 4), dtype=np.int32)

    obs_local_idx = np.full(total_samples, -1, dtype=np.int16)
    legal_mask = np.zeros((total_samples, 7), dtype=np.bool_)
    step_n_invalid = np.zeros(n_steps, dtype=np.int32)
    step_n_illegal = np.zeros(n_steps, dtype=np.int32)

    for step_offset, info in enumerate(step_infos):
        base_idx = step_offset * n_worlds
        actor = info["actor"]
        played_before = info["played_before"]
        current_trick = info["current_trick"]
        leader = info["leader"]
        observed_domino = info["observed_domino"]
        lead_domino = info["lead_domino"]
        is_leading = info["is_leading"]

        for world_idx, initial_hands in enumerate(hypothetical_deals):
            sample_idx = base_idx + world_idx
            actor_hand = initial_hands[actor]

            actors[sample_idx] = actor
            leaders[sample_idx] = leader
            trick_plays_list.append(current_trick)

            if observed_domino not in actor_hand:
                step_n_invalid[step_offset] += 1
                obs_local_idx[sample_idx] = -1
            else:
                obs_local_idx[sample_idx] = actor_hand.index(observed_domino)

            legal_local = _get_legal_local_indices(
                actor_hand, played_before, lead_domino, decl_id, is_leading
            )
            if not legal_local:
                if config.strict_integrity:
                    raise MappingIntegrityError(
                        f"No legal actions for actor {actor} at step {info['step_idx']} in world {world_idx}"
                    )
                step_n_illegal[step_offset] += 1
            else:
                for local_idx in legal_local:
                    legal_mask[sample_idx, local_idx] = True

                local_idx_obs = int(obs_local_idx[sample_idx])
                if local_idx_obs >= 0 and not legal_mask[sample_idx, local_idx_obs]:
                    if config.strict_integrity:
                        raise MappingIntegrityError(
                            f"Observed domino {observed_domino} (local_idx={local_idx_obs}) is in actor {actor}'s "
                            f"hand but not legal at step {info['step_idx']}. Legal indices: {legal_local}."
                        )
                    step_n_illegal[step_offset] += 1

            for p in range(4):
                for local_idx, domino in enumerate(initial_hands[p]):
                    if domino not in played_before:
                        remaining[sample_idx, p] |= 1 << local_idx

    if hasattr(oracle, "query_batch_multi_state"):
        all_q_values = oracle.query_batch_multi_state(
            worlds=expanded_worlds,
            decl_id=decl_id,
            actors=actors,
            leaders=leaders,
            trick_plays_list=trick_plays_list,
            remaining=remaining,
        )
    else:
        q_chunks: list[Tensor] = []
        for sample_idx, world in enumerate(expanded_worlds):
            game_state_info = {
                "decl_id": decl_id,
                "leader": int(leaders[sample_idx]),
                "trick_plays": trick_plays_list[sample_idx],
                "remaining": remaining[sample_idx : sample_idx + 1],
            }
            q_chunks.append(oracle.query_batch([world], game_state_info, int(actors[sample_idx])))
        all_q_values = torch.cat(q_chunks, dim=0)

    obs_idx_t = torch.from_numpy(obs_local_idx.astype(np.int64)).to(device=device)
    legal_mask_t = torch.from_numpy(legal_mask).to(device=device, dtype=torch.bool)

    q = all_q_values.to(dtype=torch.float32)
    legal_f = legal_mask_t.to(dtype=torch.float32)
    legal_count = legal_f.sum(dim=1)
    has_legal = legal_count > 0

    denom = legal_count.clamp(min=1.0)
    mean_q_legal = (q * legal_f).sum(dim=1) / denom
    advantage = (q - mean_q_legal.unsqueeze(1)) / config.tau
    advantage = advantage.masked_fill(~legal_mask_t, float("-inf"))

    p_soft = F.softmax(advantage, dim=1)
    uniform = legal_f / denom.unsqueeze(1)
    p_mixed = (1 - config.beta) * p_soft + config.beta * uniform

    safe_obs = obs_idx_t.clamp(min=0)
    obs_prob = p_mixed.gather(dim=1, index=safe_obs.unsqueeze(1)).squeeze(1)
    invalid_obs = obs_idx_t < 0
    log_prob = torch.log(obs_prob + 1e-30)

    log_prob = torch.where(
        invalid_obs | (~has_legal),
        torch.full_like(log_prob, -1e9),
        log_prob,
    )

    log_prob = log_prob.view(n_steps, n_worlds)
    log_probs_sum = log_prob.sum(dim=0)

    total_n_invalid = int(step_n_invalid.max()) if len(step_n_invalid) > 0 else 0
    total_n_illegal = int(step_n_illegal.sum()) if len(step_n_illegal) > 0 else 0
    return log_probs_sum, total_n_invalid, total_n_illegal

