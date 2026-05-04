"""Parallel-hand simulator for Lens-vs-Lens head-to-head matches.

Mirrors the structure of `forge.zeb.eq_player._run_eq_vs_eq_batched` but
with utility-aware action selection on each side. The forge eq batched path
always uses p_make-with-EV-tiebreak inside `select_actions`; we cannot
cleanly extend it to parameterize the utility (the utility is hard-coded in
that helper). So we replicate the action-loop here and inject our own
`argmax_under_utility` from `lens.py`.

The match is paired-seed: the SAME shuffle seed produces the SAME initial
deal+bidder+decl for both team-assignment halves. We rotate seats so team A
plays seats {0,2} for half the games and {1,3} for the other half.

Output per match:
  - `team_a_score`, `team_b_score` per hand
  - `seed`, `a_seats`, `bidder`, `decl_id`, `bid_value`
  - aggregated mean point margin (Team_A − Team_B)
"""
from __future__ import annotations

import random as stdlib_random
import time
from dataclasses import dataclass

import torch

from forge.eq.generate.deals import build_hypothetical_deals
from forge.eq.generate.eq_compute import compute_eq_pdf
from forge.eq.generate.model import query_model
from forge.eq.generate.sampling import sample_worlds_batched
from forge.eq.generate.tokenization import tokenize_batched
from forge.eq.sampling_mrv_gpu import WorldSamplerMRV
from forge.eq.tokenize_gpu import GPUTokenizer
from forge.zeb.eq_player import zeb_states_to_game_state_tensor
from forge.zeb.evaluate import _get_current_player
from forge.zeb.game import apply_action, game_seed, is_terminal, legal_actions, new_game
from forge.zeb.types import BidState, GamePhase, ZebGameState

from .lens import argmax_under_utility


# ---------------------------------------------------------------------------
# Per-hand record + match result
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HandRecord:
    """One paired hand."""
    seed: int
    a_team: int  # 0 or 1
    bidder: int
    decl_id: int
    bid_value: int
    team_a_pts: int
    team_b_pts: int
    n_decisions_a: int
    n_decisions_b: int

    @property
    def margin(self) -> int:
        return self.team_a_pts - self.team_b_pts


@dataclass
class LensMatchResult:
    utility_a: str
    utility_b: str
    n_samples: int
    n_hands: int
    fp_dtype: str
    elapsed_s: float
    hands: list[HandRecord]

    @property
    def mean_margin(self) -> float:
        if not self.hands:
            return 0.0
        return sum(h.margin for h in self.hands) / len(self.hands)

    @property
    def a_wins(self) -> int:
        return sum(1 for h in self.hands if h.team_a_pts > h.team_b_pts)

    @property
    def decisive(self) -> int:
        return sum(1 for h in self.hands if h.margin != 0)


# ---------------------------------------------------------------------------
# Force-bid-30 helper (Wave 4.0 corpus is bid=30 only)
# ---------------------------------------------------------------------------

def _force_bid_30(state: ZebGameState) -> ZebGameState:
    """Override the random bid in `new_game` so every hand is bid=30."""
    bid_state = BidState(
        bids=(30, 0, 0, 0),
        high_bidder=state.bidder,
        high_bid=30,
    )
    return ZebGameState(
        hands=state.hands,
        dealer=state.dealer,
        phase=state.phase,
        bid_state=bid_state,
        decl_id=state.decl_id,
        bidder=state.bidder,
        played=state.played,
        play_history=state.play_history,
        current_trick=state.current_trick,
        trick_leader=state.trick_leader,
        team_points=state.team_points,
    )


# ---------------------------------------------------------------------------
# Core: utility-aware batched action selection
# ---------------------------------------------------------------------------

def _apply_utility_actions(
    model,
    indices: list[int],
    game_states: list[ZebGameState],
    utility: str,
    n_samples: int,
    sampler: WorldSamplerMRV,
    tokenizer: GPUTokenizer,
    states: list[ZebGameState],
    active: list[bool],
    device: str,
    decision_counter: dict[int, int],
    bid_value: int = 30,
) -> None:
    """Batched utility-greedy decision + action application. Mutates states/active."""
    n_eq = len(indices)
    gst = zeb_states_to_game_state_tensor(game_states, device)

    with torch.no_grad():
        worlds = sample_worlds_batched(gst, sampler, n_samples)
        deals = build_hypothetical_deals(gst, worlds)

        batch_needed = n_eq * n_samples
        if batch_needed > tokenizer.max_batch:
            tokenizer = GPUTokenizer(max_batch=batch_needed, device=device)

        tokens, masks = tokenize_batched(gst, deals, tokenizer)
        q_values = query_model(model, tokens, masks, gst, n_samples, device)

        q_reshaped = q_values.view(n_eq, n_samples, 7)
        e_q = q_reshaped.mean(dim=1)
        e_q_pdf = compute_eq_pdf(q_reshaped)

        # bid_values per game from gst (may be unset; fall back to bid_value
        # arg). All hands in this run are bid=30 so a flat list is fine.
        bid_values = [bid_value] * n_eq

        actions = argmax_under_utility(
            utility=utility,
            e_q=e_q,
            e_q_pdf=e_q_pdf,
            bidder=gst.bidder.long(),
            current_players=gst.current_player.long(),
            legal_mask=gst.legal_actions(),
            bid_values=bid_values,
        )

    for idx, game_idx in enumerate(indices):
        action = int(actions[idx].item())
        states[game_idx] = apply_action(states[game_idx], action)
        decision_counter[game_idx] = decision_counter.get(game_idx, 0) + 1
        if is_terminal(states[game_idx]):
            active[game_idx] = False


# ---------------------------------------------------------------------------
# Run one half (A as one team) — paired seeds with the other half
# ---------------------------------------------------------------------------

def _run_half(
    model,
    *,
    utility_a: str,
    utility_b: str,
    n_hands: int,
    n_samples_a: int,
    n_samples_b: int,
    a_team: int,            # 0 or 1
    base_seed: int,
    device: str,
    bid_value: int = 30,
) -> tuple[list[HandRecord], dict[int, int], dict[int, int]]:
    """Run n_hands hands in parallel with team A as a_team."""
    states = [
        _force_bid_30(new_game(seed=game_seed(base_seed, i)))
        for i in range(n_hands)
    ]
    seeds = [game_seed(base_seed, i) for i in range(n_hands)]
    active = [True] * n_hands
    decisions_a: dict[int, int] = {}
    decisions_b: dict[int, int] = {}

    # GPU resources sized to worst-case batch
    sampler_a = WorldSamplerMRV(max_games=n_hands, max_samples=n_samples_a, device=device)
    tokenizer_a = GPUTokenizer(max_batch=n_hands * n_samples_a, device=device)
    sampler_b = WorldSamplerMRV(max_games=n_hands, max_samples=n_samples_b, device=device)
    tokenizer_b = GPUTokenizer(max_batch=n_hands * n_samples_b, device=device)

    model.eval()

    while any(active):
        a_indices, a_states = [], []
        b_indices, b_states = [], []
        for i, (state, is_active) in enumerate(zip(states, active)):
            if not is_active:
                continue
            player = _get_current_player(state)
            if player % 2 == a_team:
                a_indices.append(i)
                a_states.append(state)
            else:
                b_indices.append(i)
                b_states.append(state)

        if a_indices:
            _apply_utility_actions(
                model, a_indices, a_states, utility_a, n_samples_a,
                sampler_a, tokenizer_a, states, active, device,
                decisions_a, bid_value,
            )
        if b_indices:
            _apply_utility_actions(
                model, b_indices, b_states, utility_b, n_samples_b,
                sampler_b, tokenizer_b, states, active, device,
                decisions_b, bid_value,
            )

    records: list[HandRecord] = []
    for i, state in enumerate(states):
        team0_pts, team1_pts = state.team_points
        a_pts = team0_pts if a_team == 0 else team1_pts
        b_pts = team1_pts if a_team == 0 else team0_pts
        records.append(HandRecord(
            seed=seeds[i],
            a_team=a_team,
            bidder=state.bidder,
            decl_id=state.decl_id,
            bid_value=bid_value,
            team_a_pts=int(a_pts),
            team_b_pts=int(b_pts),
            n_decisions_a=decisions_a.get(i, 0),
            n_decisions_b=decisions_b.get(i, 0),
        ))
    return records, decisions_a, decisions_b


# ---------------------------------------------------------------------------
# Public match driver: paired-seed, both team assignments
# ---------------------------------------------------------------------------

def run_lens_match(
    model,
    *,
    utility_a: str,
    utility_b: str,
    n_hands: int,
    n_samples: int = 10,
    device: str = "mps",
    base_seed: int = 0,
    bid_value: int = 30,
    verbose: bool = False,
) -> LensMatchResult:
    """Run a Lens(utility_a) vs Lens(utility_b) match.

    Half the hands have team A play seats {0, 2}; the other half play
    seats {1, 3}. The same shuffle seeds are reused across both halves so
    the matchup is paired-difference (collapses card-luck variance).
    """
    half = n_hands // 2

    t0 = time.time()
    if verbose:
        print(f"  [Lens] {utility_a} vs {utility_b}  N={n_samples}  hands={n_hands}", flush=True)
        print(f"    half 1: A=team0 ({half} hands)", flush=True)
    h0, dec_a0, dec_b0 = _run_half(
        model,
        utility_a=utility_a, utility_b=utility_b,
        n_hands=half, n_samples_a=n_samples, n_samples_b=n_samples,
        a_team=0, base_seed=base_seed, device=device, bid_value=bid_value,
    )
    if verbose:
        m0 = sum(r.margin for r in h0) / max(len(h0), 1)
        print(f"      half-1 mean margin: {m0:+.2f}", flush=True)
        print(f"    half 2: A=team1 ({half} hands, paired seeds)", flush=True)
    h1, dec_a1, dec_b1 = _run_half(
        model,
        utility_a=utility_a, utility_b=utility_b,
        n_hands=half, n_samples_a=n_samples, n_samples_b=n_samples,
        a_team=1, base_seed=base_seed, device=device, bid_value=bid_value,
    )
    if verbose:
        m1 = sum(r.margin for r in h1) / max(len(h1), 1)
        print(f"      half-2 mean margin: {m1:+.2f}", flush=True)

    elapsed = time.time() - t0
    return LensMatchResult(
        utility_a=utility_a,
        utility_b=utility_b,
        n_samples=n_samples,
        n_hands=2 * half,
        fp_dtype="fp32",
        elapsed_s=elapsed,
        hands=h0 + h1,
    )
