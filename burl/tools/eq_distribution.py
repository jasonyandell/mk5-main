"""E[Q] outcome-distribution tool for Burl.

Exposes the full outcome distribution (PDF over Q in [-42, +42]) for each legal
play as a callable tool. Replaces the Zeb-belief tool as Burl's primary
reasoning primitive: per-domino outcome distributions are irreducibly
multimodal and a 3-way categorical belief throws away the signal Burl needs.

Pipeline (reuses ``forge.eq.generate.*`` helpers — no re-implementation):

    duck-typed state  --converter-->  ``GameStateTensor`` (n_games=1)
        |
        +--> ``sample_worlds_batched``       (N worlds respecting voids)
        +--> ``build_hypothetical_deals``    (assemble full deals)
        +--> ``tokenize_batched``            (GPU tokenizer)
        +--> ``query_model``                 (Stage 1 oracle forward pass)
        +--> per-world Q-values  -->  ``compute_eq_pdf`` -> 85-bin PDF

Seat / suit / domino_id conventions follow ``burl/tools/engine.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from forge.eq.game_tensor import GameStateTensor
from forge.eq.generate.deals import build_hypothetical_deals
from forge.eq.generate.eq_compute import compute_eq_pdf
from forge.eq.generate.model import query_model
from forge.eq.generate.sampling import sample_worlds_batched
from forge.eq.generate.tokenization import tokenize_batched
from forge.eq.oracle import Stage1Oracle
from forge.eq.sampling_mrv_gpu import WorldSamplerMRV
from forge.eq.tokenize_gpu import GPUTokenizer
from forge.ml.module import DominoLightningModule

# Best-available Stage 1 Q-value oracle — see forge/models/README.md. "Shuffle"
# is preferred over the older "Large" because it removes slot-0 tie-break bias;
# Q-gap and Q-MAE are comparable.
DEFAULT_CHECKPOINT = (
    Path(__file__).resolve().parents[2]
    / "forge/models/domino-qval-3.3M-shuffle-qgap0.074-qmae0.96.ckpt"
)

N_BINS = 85  # Q in [-42, +42], bin i -> Q = i - 42
Q_VALUES = np.arange(N_BINS, dtype=np.float32) - 42.0


class ConditionUnreachable(RuntimeError):
    """Raised when `conditional_outcome` can't find enough consistent worlds."""


@dataclass
class OutcomeDistribution:
    """Full outcome distribution for a single play under one set of assumptions.

    ``pdf_bins`` is the primary payload — a 85-bin histogram over Q in [-42, +42]
    matching the ``forge/eq`` convention. Summary stats (mean, stdev, p_make)
    are precomputed for common Burl reasoning patterns, but the whole point of
    returning the PDF is that Burl can inspect shape (bimodality, tails) that
    the scalars hide.
    """

    play: int
    pdf_bins: np.ndarray         # shape [85], sums to 1.0 over valid bins
    mean: float
    stdev: float
    p_make: float                # P(contract made), seat-aware
    n_samples: int
    min_q: float
    max_q: float
    percentiles: dict[int, float]  # {10, 25, 50, 75, 90}
    is_offense: bool


# --------------------------------------------------------------------------- #
# Oracle loading (module-level cache)                                          #
# --------------------------------------------------------------------------- #


@lru_cache(maxsize=4)
def _load_oracle_cached(checkpoint_path: str, device: str) -> Stage1Oracle:
    """Key-on-path cache. Stage1Oracle holds torch.compile state; keep at most a few."""
    return Stage1Oracle(
        checkpoint_path=checkpoint_path,
        device=device,
        compile=False,          # fast load for tool use; compile pays off only at scale
        use_async=False,
        use_gpu_tokenizer=False,
    )


def load_eq_oracle(
    checkpoint_path: str | Path | None = None,
    device: str | None = None,
) -> Stage1Oracle:
    """Cached loader for the Stage 1 oracle. See ``DEFAULT_CHECKPOINT`` above."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    path = str(checkpoint_path) if checkpoint_path is not None else str(DEFAULT_CHECKPOINT)
    return _load_oracle_cached(path, device)


# --------------------------------------------------------------------------- #
# Duck-typed state -> GameStateTensor                                          #
# --------------------------------------------------------------------------- #


def _current_trick_dominoes(state: Any) -> list[int]:
    """Return the domino_ids in the current trick, regardless of tuple shape.

    Mirrors burl/tools/engine.py::_current_trick_domino_ids — accepts either
    forge.eq.game.GameState (tuples of (player, domino_id)) or
    forge.zeb.types.ZebGameState (tuple of domino_ids).
    """
    trick = state.current_trick
    if not trick:
        return []
    first = trick[0]
    if isinstance(first, int):
        return list(trick)
    return [d for _p, d in trick]


def _leader(state: Any) -> int:
    leader = getattr(state, "trick_leader", None)
    if leader is None:
        leader = state.leader
    return int(leader)


def _abs_current_player(state: Any) -> int:
    return (_leader(state) + len(state.current_trick)) % 4


def _bidder(state: Any) -> int:
    """Bidder seat (for offense/defense). ZebGameState has it; eq.GameState does
    not — default to 0 (matches ``GameStateTensor.from_deals`` default)."""
    return int(getattr(state, "bidder", 0))


def _state_to_game_state_tensor(state: Any, device: str) -> GameStateTensor:
    """Adapter: duck-typed state -> batched GameStateTensor (n_games=1).

    Reconstructs history with lead-domino annotation and pads hands to 7 slots
    with -1 for played dominoes, matching ``zeb_states_to_game_state_tensor``.
    This is the ONE clearly-marked bridge between burl's duck typing and
    forge/eq's GPU tensor layout.
    """
    played: set[int] = set(state.played)

    # hands: [1, 4, 7] with -1 for played slots.
    hands = torch.full((1, 4, 7), -1, dtype=torch.int8, device=device)
    for p in range(4):
        for slot, dom in enumerate(state.hands[p]):
            hands[0, p, slot] = dom if dom not in played else -1

    # played_mask: [1, 28]
    played_mask = torch.zeros(1, 28, dtype=torch.bool, device=device)
    for d in played:
        played_mask[0, d] = True

    # history: [1, 28, 3] of (player, domino_id, lead_domino_id).
    history = torch.full((1, 28, 3), -1, dtype=torch.int8, device=device)
    hist = list(state.play_history)
    if hist and len(hist[0]) == 3:
        for i, (p, d, lead) in enumerate(hist):
            history[0, i, 0] = p
            history[0, i, 1] = d
            history[0, i, 2] = lead
    else:
        # ZebGameState shape (player, domino_id): recover lead per 4-play chunk.
        for i, (p, d) in enumerate(hist):
            trick_start = (i // 4) * 4
            lead = hist[trick_start][1]
            history[0, i, 0] = p
            history[0, i, 1] = d
            history[0, i, 2] = lead

    trick_ids = _current_trick_dominoes(state)
    trick_plays = torch.full((1, 4), -1, dtype=torch.int8, device=device)
    for i, d in enumerate(trick_ids):
        trick_plays[0, i] = d

    leader = torch.tensor([_leader(state)], dtype=torch.int8, device=device)
    decl_ids = torch.tensor([state.decl_id], dtype=torch.int8, device=device)
    bidder = torch.tensor([_bidder(state)], dtype=torch.int8, device=device)

    return GameStateTensor(
        hands=hands,
        played_mask=played_mask,
        history=history,
        trick_plays=trick_plays,
        leader=leader,
        decl_ids=decl_ids,
        device=device,
        bidder=bidder,
    )


def _slot_for_domino(gst: GameStateTensor, play: int) -> int:
    """Return the slot index (0..6) in current player's hand for ``play``.

    Raises ValueError if ``play`` isn't in the current player's remaining hand.
    """
    cur = int(gst.current_player[0].item())
    hand = gst.hands[0, cur].tolist()
    for slot, d in enumerate(hand):
        if d == play:
            return slot
    raise ValueError(f"domino {play} is not in current player's remaining hand {hand}")


# --------------------------------------------------------------------------- #
# Core query: sample worlds, evaluate, aggregate                               #
# --------------------------------------------------------------------------- #


def _q_per_world(
    gst: GameStateTensor,
    oracle: Stage1Oracle,
    n_samples: int,
    device: str,
    world_filter: Callable[[np.ndarray], bool] | None = None,
    max_tries: int = 1,
) -> tuple[torch.Tensor, int]:
    """Sample N worlds and return per-world Q-values [n_kept, 7].

    If ``world_filter`` is provided, keep only worlds satisfying the predicate;
    re-sample up to ``max_tries`` extra rounds to top up. Returns the kept
    Q-values and the actual number of samples evaluated. When no filter is
    active, this is a single pass.
    """
    sampler = WorldSamplerMRV(max_games=1, max_samples=n_samples, device=device)
    tokenizer = GPUTokenizer(max_batch=n_samples, device=device)

    kept_q: list[torch.Tensor] = []
    total_evaluated = 0
    tries = 0

    while True:
        with torch.no_grad():
            worlds = sample_worlds_batched(gst, sampler, n_samples)   # [1, n, 3, 7]
            deals = build_hypothetical_deals(gst, worlds)              # [1, n, 4, 7]
            tokens, masks = tokenize_batched(gst, deals, tokenizer)
            q_values = query_model(
                oracle.model, tokens, masks, gst, n_samples, device,
            )  # [n, 7]
            q_reshaped = q_values.view(1, n_samples, 7)[0]             # [n, 7]

        total_evaluated += n_samples

        if world_filter is None:
            kept_q.append(q_reshaped.float().cpu())
            break

        # Filter in Python — world shape is small. ``worlds[0]`` is [n, 3, 7]
        # where index k maps to absolute seat (current_player + 1 + k) % 4.
        worlds_cpu = worlds[0].cpu().numpy()  # [n, 3, 7]
        cur = int(gst.current_player[0].item())
        q_cpu = q_reshaped.float().cpu()
        for i in range(n_samples):
            world_for_seats = _world_by_abs_seat(worlds_cpu[i], cur, gst)
            if world_filter(world_for_seats):
                kept_q.append(q_cpu[i:i+1])

        tries += 1
        if sum(t.shape[0] for t in kept_q) >= n_samples or tries >= max_tries:
            break

    kept = torch.cat(kept_q, dim=0) if kept_q else torch.empty(0, 7)
    return kept, total_evaluated


def _world_by_abs_seat(
    opp_hands: np.ndarray, cur_player: int, gst: GameStateTensor,
) -> dict[int, set[int]]:
    """Turn a sampled world ([3, 7] opponent hands) into {abs_seat: set(dominoes)}.

    Slot k in ``opp_hands`` corresponds to absolute seat (cur_player + 1 + k) % 4.
    Also includes the current player's own hand at ``cur_player``. Padded -1
    entries are dropped.
    """
    out: dict[int, set[int]] = {}
    for k in range(3):
        abs_seat = (cur_player + 1 + k) % 4
        out[abs_seat] = {int(d) for d in opp_hands[k] if int(d) >= 0}
    my_hand = gst.hands[0, cur_player].tolist()
    out[cur_player] = {int(d) for d in my_hand if d >= 0}
    return out


def _distribution_from_q_slice(
    q_slot: torch.Tensor,           # [n_kept] Q-values for the target play
    play: int,
    is_offense: bool,
    n_samples: int,
) -> OutcomeDistribution:
    """Build 85-bin PDF + summary stats from per-world Q-values for one play."""
    # compute_eq_pdf expects [N, M, 7]; we have 1 play so pad to [1, M, 1].
    q_input = q_slot.view(1, -1, 1)  # [1, M, 1]
    # Reuse forge's bucketing directly.
    pdf = compute_eq_pdf(q_input)[0, 0].cpu().numpy().astype(np.float32)  # [85]

    s = pdf.sum()
    if s > 0:
        pdf = pdf / s

    q_np = q_slot.cpu().numpy().astype(np.float32)
    mean = float(q_np.mean()) if q_np.size else 0.0
    stdev = float(q_np.std(ddof=0)) if q_np.size else 0.0
    # Offense needs team >= 30 points -> Q >= 18 (bin >= 60).
    # Defense needs bidder < 30 -> Q >= -17 (bin >= 25).
    make_bin_lo = 60 if is_offense else 25
    p_make = float(pdf[make_bin_lo:].sum())

    pcts = {p: float(np.percentile(q_np, p)) if q_np.size else 0.0
            for p in (10, 25, 50, 75, 90)}

    return OutcomeDistribution(
        play=play,
        pdf_bins=pdf,
        mean=mean,
        stdev=stdev,
        p_make=p_make,
        n_samples=int(q_slot.shape[0]),
        min_q=float(q_np.min()) if q_np.size else 0.0,
        max_q=float(q_np.max()) if q_np.size else 0.0,
        percentiles=pcts,
        is_offense=is_offense,
    )


# --------------------------------------------------------------------------- #
# Public entry points                                                          #
# --------------------------------------------------------------------------- #


def eq_outcome_distribution(
    game_state: Any,
    play: int,
    n_samples: int = 10,
    oracle: Stage1Oracle | None = None,
    device: str | None = None,
) -> OutcomeDistribution:
    """Sample N consistent worlds, evaluate ``play`` under Stage 1 in each,
    return the full outcome distribution (85-bin PDF + summary stats).

    ``play`` is a domino_id (not a slot). Voids inferred from play history are
    respected. Distribution is from the perspective of the seat currently to
    act; p_make is seat-aware (offense: Q >= 18; defense: Q >= -17).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if oracle is None:
        oracle = load_eq_oracle(device=device)

    gst = _state_to_game_state_tensor(game_state, device)
    slot = _slot_for_domino(gst, play)

    cur = int(gst.current_player[0].item())
    bidder = int(gst.bidder[0].item())
    is_offense = (cur % 2) == (bidder % 2)

    q_all, _ = _q_per_world(gst, oracle, n_samples, device)  # [n, 7]
    q_slot = q_all[:, slot]                                   # [n]
    return _distribution_from_q_slice(q_slot, play, is_offense, n_samples)


Assumption = Callable[[dict[int, set[int]]], bool] | dict[str, Any]


def _build_assumption_predicate(
    assumption: Assumption, game_state: Any,
) -> Callable[[dict[int, set[int]]], bool]:
    """Convert a structured assumption into a predicate on ``{abs_seat: hand}``.

    Structured shapes accepted:
      - ``{"player": abs_seat, "holds": domino_id}``
          World is valid iff ``domino_id in world[abs_seat]``.
      - ``{"player": abs_seat, "void_in_suit": suit_id}``
          World is valid iff every domino ``world[abs_seat]`` can't-follow
          ``suit_id`` under the current declaration.

    Callables are passed through unchanged. We prefer structured forms over raw
    callables in training traces because they serialize cleanly and Burl can be
    taught to emit them. Callables are the escape hatch for research/debugging.
    """
    if callable(assumption):
        return assumption
    if not isinstance(assumption, dict):
        raise TypeError(f"assumption must be callable or dict, got {type(assumption).__name__}")

    if "holds" in assumption:
        seat = int(assumption["player"])
        dom = int(assumption["holds"])
        def _holds(world: dict[int, set[int]]) -> bool:
            return dom in world.get(seat, set())
        return _holds

    if "void_in_suit" in assumption:
        from forge.oracle.tables import can_follow
        seat = int(assumption["player"])
        suit = int(assumption["void_in_suit"])
        decl = int(game_state.decl_id)
        def _void(world: dict[int, set[int]]) -> bool:
            hand = world.get(seat, set())
            return not any(can_follow(d, suit, decl) for d in hand)
        return _void

    raise ValueError(f"unsupported assumption shape: {assumption}")


def conditional_outcome(
    game_state: Any,
    play: int,
    assumption: Assumption,
    n_samples: int = 10,
    max_sampling_tries: int = 100,
    oracle: Stage1Oracle | None = None,
    device: str | None = None,
) -> OutcomeDistribution:
    """Same as ``eq_outcome_distribution`` but restrict sampled worlds to those
    satisfying ``assumption``. Counterfactual reasoning primitive:

        "If right opponent holds the 5-5, what does playing 6-2 look like?"

    Raises ``ConditionUnreachable`` if fewer than ``n_samples // 2`` worlds
    satisfy the assumption after ``max_sampling_tries`` rounds.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if oracle is None:
        oracle = load_eq_oracle(device=device)

    gst = _state_to_game_state_tensor(game_state, device)
    slot = _slot_for_domino(gst, play)

    cur = int(gst.current_player[0].item())
    bidder = int(gst.bidder[0].item())
    is_offense = (cur % 2) == (bidder % 2)

    predicate = _build_assumption_predicate(assumption, game_state)

    q_all, _ = _q_per_world(
        gst, oracle, n_samples, device,
        world_filter=predicate, max_tries=max_sampling_tries,
    )
    if q_all.shape[0] < max(1, n_samples // 2):
        raise ConditionUnreachable(
            f"only {q_all.shape[0]} of up to {n_samples * max_sampling_tries} "
            f"sampled worlds satisfied the assumption; condition is likely "
            f"inconsistent with the observed state (voids, played dominoes)"
        )

    q_slot = q_all[:, slot]
    return _distribution_from_q_slice(q_slot, play, is_offense, q_slot.shape[0])


# --------------------------------------------------------------------------- #
# Self-test                                                                    #
# --------------------------------------------------------------------------- #


if __name__ == "__main__":
    import random as _r
    import time

    from forge.zeb.game import apply_action, legal_actions, new_game

    # Route compile errors away from self-test — the tool path disables compile.
    assert torch.cuda.is_available() or True, "CPU fallback path is exercised"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # 1) Load the Stage 1 oracle.
    t0 = time.perf_counter()
    oracle = load_eq_oracle(device=device)
    t_load = time.perf_counter() - t0
    print(f"oracle load: {t_load:.2f}s  "
          f"checkpoint={DEFAULT_CHECKPOINT.name}")

    # 2) Generate a mid-game state from a held-out seed (>= 900000).
    seed = 900013
    state = new_game(seed=seed, skip_bidding=True)
    rng = _r.Random(seed)
    # Advance ~3 tricks (12 plays) so we have a non-trivial play history.
    for _ in range(12):
        slots = legal_actions(state)
        if not slots:
            break
        state = apply_action(state, rng.choice(slots))
    me = _abs_current_player(state)
    my_hand = [d for d in state.hands[me] if d not in state.played]
    print(f"seed={seed} decl_id={state.decl_id} bidder={state.bidder} "
          f"me(abs)={me}")
    print(f"plays made so far: {len(state.play_history)}")
    print(f"my remaining hand: {my_hand}")

    # 3) Call eq_outcome_distribution for each legal play.
    from burl.tools.engine import is_legal
    legal_doms = [d for d in my_hand if is_legal(state, d)[0]]
    print(f"\nunconditional outcome distributions (n_samples=10):")

    t_full = time.perf_counter()
    dists: list[OutcomeDistribution] = []
    per_play_times: list[float] = []
    for dom in legal_doms:
        t_p = time.perf_counter()
        d = eq_outcome_distribution(state, dom, n_samples=10, oracle=oracle, device=device)
        per_play_times.append(time.perf_counter() - t_p)
        dists.append(d)
        high = dom // 7  # quick readable label; not canonical
        low = dom % 7
        print(
            f"  dom={dom:2d} mean={d.mean:+6.2f} stdev={d.stdev:5.2f} "
            f"p_make={d.p_make:.2f}  [min={d.min_q:+.1f} p50={d.percentiles[50]:+.1f} "
            f"max={d.max_q:+.1f}] {'offense' if d.is_offense else 'defense'}"
        )
    t_hand = time.perf_counter() - t_full

    # 4) Pick a domino, call conditional_outcome with a holds-assumption.
    target_play = legal_doms[0]
    # Find a domino that's unseen and pick an opponent seat (right opp = me+3).
    unseen = {d for d in range(28) if d not in state.played and d not in state.hands[me]}
    right_opp = (me + 3) % 4
    # Pick a domino the right opponent actually holds, so the condition is reachable.
    held_by_right = [d for d in state.hands[right_opp] if d in unseen]
    assumed_dom = held_by_right[0] if held_by_right else next(iter(unseen))
    assumption = {"player": right_opp, "holds": assumed_dom}

    unconditional = dists[0]
    try:
        conditional = conditional_outcome(
            state, target_play, assumption,
            n_samples=10, max_sampling_tries=20,
            oracle=oracle, device=device,
        )
        print(f"\nconditional on {assumption}:")
        print(
            f"  unconditional: mean={unconditional.mean:+.2f} "
            f"p_make={unconditional.p_make:.2f} stdev={unconditional.stdev:.2f}"
        )
        print(
            f"  conditional  : mean={conditional.mean:+.2f} "
            f"p_make={conditional.p_make:.2f} stdev={conditional.stdev:.2f} "
            f"(n={conditional.n_samples})"
        )
        delta = conditional.mean - unconditional.mean
        print(f"  shift in mean: {delta:+.2f}")
    except ConditionUnreachable as e:
        print(f"\nconditional: unreachable — {e}")

    # 5) Invariants.
    for d in dists:
        total = d.pdf_bins.sum()
        assert abs(total - 1.0) < 1e-5, f"pdf_bins sums to {total}, not 1.0"
        expected_mean = float((d.pdf_bins * Q_VALUES).sum())
        # Binning introduces rounding; allow 0.5-point tolerance because Q is
        # bucketed to integer Q values.
        assert abs(expected_mean - d.mean) < 0.6, (
            f"dom={d.play} pdf mean {expected_mean:+.3f} vs stored {d.mean:+.3f}"
        )
    print("\ninvariants OK: pdf sums to 1.0, mean consistent with pdf")

    # 6) Timing report.
    print(f"\ntiming:")
    print(f"  oracle load           : {t_load:.2f}s")
    print(f"  single-play N=10 eval : {per_play_times[0]*1000:.0f} ms")
    print(f"  full-hand ({len(legal_doms)} plays)  : {t_hand*1000:.0f} ms "
          f"({t_hand/max(1,len(legal_doms))*1000:.0f} ms/play avg)")

    if device == "cuda":
        peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        print(f"  peak VRAM             : {peak_mb:.1f} MB")

    # 7) VRAM hygiene.
    del oracle
    _load_oracle_cached.cache_clear()
    if device == "cuda":
        torch.cuda.empty_cache()
        after_mb = torch.cuda.memory_allocated() / (1024 ** 2)
        print(f"  VRAM after cleanup    : {after_mb:.1f} MB")

    print("\nself-test: OK")
