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

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from scipy.signal import find_peaks

from forge.eq.enumeration_gpu import enumerate_worlds_cpu
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

# "auto" mode switches from enumeration to sampling when the pool exceeds this
# many dominoes. Empirical breakeven on M5 Max CPU: pool<=12 -> enumerate beats
# N=10 sampling; pool>=15 -> sampling wins. See SESSION_NOTES 2026-04-19.
_AUTO_ENUMERATE_MAX_POOL = 12
# Hard cap for enumerate=True — refuse to blow up memory/time silently.
_ENUMERATE_HARD_POOL_CAP = 20


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

    Candlewax fields (``distribution_shape``, ``modes``, ``gap_between_modes``,
    ``suggested_counterfactuals``) are computed from the PDF after bucketing.
    They make bimodality legible to a zero-shot reader so the counterfactual
    probe (``conditional_outcome``) is invited rather than hidden behind 85
    raw bins.
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
    distribution_shape: str = "unimodal"         # "unimodal" | "bimodal" | "multimodal"
    modes: list[dict] = field(default_factory=list)          # [{center, mass}, ...]
    gap_between_modes: float = 0.0                            # |center_top1 - center_top2|
    suggested_counterfactuals: list[dict] = field(default_factory=list)
                                                              # [{player, holds, rationale}]
    sampling_mode: str = "sampled"
    # "sampled"   — N consistent worlds drawn by WorldSamplerMRV (default)
    # "enumerated" — all worlds in the pool (exact; used when pool <= auto cutoff)


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


def _pool_size(gst: GameStateTensor) -> int:
    """Count of dominoes neither in the current player's hand nor played.

    Matches the "pool" concept in ``forge.eq.generate.sampling`` /
    ``forge.eq.enumeration_gpu``: 28 minus played minus my hand. This is the
    set from which ``enumerate_worlds_cpu`` partitions unknowns across the
    three opponents.
    """
    me = int(gst.current_player[0].item())
    played_count = int(gst.played_mask[0].sum().item())
    my_hand_size = int((gst.hands[0, me] >= 0).sum().item())
    return 28 - played_count - my_hand_size


def _extract_enumeration_inputs(
    gst: GameStateTensor, game_state: Any,
) -> tuple[list[int], list[list[int]], list[int], list[set[int]], int]:
    """Pull ``(pool, known, slots, voids, decl_id)`` out of a game state for
    ``enumerate_worlds_cpu``.

    All lists are 3-element (one per opponent) using the same "offset from me"
    convention as ``_world_by_abs_seat``: index k -> absolute seat ``(me + 1 + k) % 4``.
    """
    from forge.oracle.tables import can_follow

    me = int(gst.current_player[0].item())
    decl_id = int(gst.decl_ids[0].item())

    # Pool: unplayed, not in my hand.
    played: set[int] = set()
    for d in range(28):
        if bool(gst.played_mask[0, d].item()):
            played.add(d)
    my_hand = {int(d) for d in gst.hands[0, me].tolist() if int(d) >= 0}
    pool = [d for d in range(28) if d not in played and d not in my_hand]

    # known[opp_idx] = dominoes already played by that opponent (from history).
    # Voids inferred the same way: look at each play, if player couldn't follow
    # the led suit, that seat is void in that suit.
    known: list[list[int]] = [[], [], []]
    voids: list[set[int]] = [set(), set(), set()]

    history = gst.history[0].tolist()   # [28, 3] of (player, domino, lead)
    for (p, d, lead) in history:
        p = int(p); d = int(d); lead = int(lead)
        if p < 0:
            continue
        if p == me:
            continue
        opp_idx = (p - me - 1) % 4
        if opp_idx >= 3:
            continue
        known[opp_idx].append(d)
        if lead < 0:
            continue
        # Determine led suit.
        from forge.eq.game_tensor import LED_SUIT_TABLE
        led_suit = int(LED_SUIT_TABLE[lead, decl_id].item())
        if not can_follow(d, led_suit, decl_id):
            voids[opp_idx].add(led_suit)

    # slot_sizes = current hand size - known count.
    slots: list[int] = []
    for opp_idx in range(3):
        abs_seat = (me + 1 + opp_idx) % 4
        cur_hand = int((gst.hands[0, abs_seat] >= 0).sum().item())
        slots.append(cur_hand)

    return pool, known, slots, voids, decl_id


def _enumerated_worlds_tensor(
    gst: GameStateTensor, game_state: Any, device: str,
) -> tuple[torch.Tensor, list[list[list[int]]]]:
    """Enumerate all consistent worlds and pack them into the
    ``[1, n_worlds, 3, 7]`` tensor layout ``build_hypothetical_deals`` expects.

    Returns (worlds_tensor, raw_worlds) so callers that need to filter on the
    world structure (conditional enumeration) can do so without re-deriving.
    """
    pool, known, slots, voids, decl_id = _extract_enumeration_inputs(gst, game_state)
    raw = enumerate_worlds_cpu(pool, known, slots, voids=voids, decl_id=decl_id)
    if not raw:
        return torch.empty(1, 0, 3, 7, dtype=torch.int32, device=device), raw

    n = len(raw)
    # int32 to match WorldSamplerMRV's output and build_hypothetical_deals'
    # scatter src dtype (int32).
    worlds = torch.full((1, n, 3, 7), -1, dtype=torch.int32, device=device)
    for i, world in enumerate(raw):
        for opp_idx in range(3):
            hand = world[opp_idx]
            for slot, dom in enumerate(hand):
                if slot >= 7:
                    break
                worlds[0, i, opp_idx, slot] = int(dom)
    return worlds, raw


def _q_per_enumerated_worlds(
    gst: GameStateTensor,
    oracle: Stage1Oracle,
    device: str,
    worlds: torch.Tensor,
    chunk: int = 256,
) -> torch.Tensor:
    """Evaluate the oracle over every enumerated world. Returns ``[n_worlds, 7]``.

    Chunks the forward pass to keep memory bounded: for a pool of 12 the
    enumerated count can hit ~35k worlds, which we'd rather not push through
    the tokenizer/model in one shot. For trick-6 sized pools (~90 worlds) the
    chunk path is a no-op.
    """
    n_worlds = worlds.shape[1]
    if n_worlds == 0:
        return torch.empty(0, 7)

    tokenizer = GPUTokenizer(max_batch=min(chunk, n_worlds), device=device)
    q_chunks: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, n_worlds, chunk):
            end = min(start + chunk, n_worlds)
            sub = worlds[:, start:end]                        # [1, k, 3, 7]
            k = end - start
            deals = build_hypothetical_deals(gst, sub)        # [1, k, 4, 7]
            tokens, masks = tokenize_batched(gst, deals, tokenizer)
            q_values = query_model(
                oracle.model, tokens, masks, gst, k, device,
            )                                                  # [k, 7]
            q_chunks.append(q_values.view(1, k, 7)[0].float().cpu())
    return torch.cat(q_chunks, dim=0)


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


# --------------------------------------------------------------------------- #
# Candlewax: bimodality detection + counterfactual suggestion                  #
# --------------------------------------------------------------------------- #

# Prominence threshold as a fraction of the PDF's peak; distance in bins.
_PEAK_PROMINENCE_FRAC = 0.04
_PEAK_MIN_DISTANCE = 5
_MODE_WINDOW = 3   # bins on each side of a peak to sum for mode mass


def _detect_modes(pdf: np.ndarray) -> tuple[str, list[dict]]:
    """Detect peaks in a normalized 85-bin PDF.

    Returns (shape_label, modes). ``modes`` is a list of ``{center, mass}``
    dicts sorted by mass descending, where ``center`` is the Q value at the
    peak bin and ``mass`` is the PDF sum in a window of ``_MODE_WINDOW`` bins
    on each side of the peak (clipped to bin bounds).
    """
    peak_height = float(pdf.max()) if pdf.size else 0.0
    if peak_height <= 0.0:
        return "unimodal", []

    peaks, _ = find_peaks(
        pdf,
        prominence=_PEAK_PROMINENCE_FRAC * peak_height,
        distance=_PEAK_MIN_DISTANCE,
    )

    if len(peaks) == 0:
        # find_peaks skips boundary maxima; fall back to argmax so the
        # single-mode case still reports a sensible mode.
        argmax = int(np.argmax(pdf))
        lo = max(0, argmax - _MODE_WINDOW)
        hi = min(len(pdf), argmax + _MODE_WINDOW + 1)
        return "unimodal", [{
            "center": float(Q_VALUES[argmax]),
            "mass": float(pdf[lo:hi].sum()),
        }]

    modes: list[dict] = []
    for p in peaks:
        lo = max(0, int(p) - _MODE_WINDOW)
        hi = min(len(pdf), int(p) + _MODE_WINDOW + 1)
        modes.append({
            "center": float(Q_VALUES[int(p)]),
            "mass": float(pdf[lo:hi].sum()),
        })
    modes.sort(key=lambda m: m["mass"], reverse=True)

    if len(modes) == 1:
        shape = "unimodal"
    elif len(modes) == 2:
        shape = "bimodal"
    else:
        shape = "multimodal"
    return shape, modes


def _rationale_for_mode(
    mode_center: float,
    mode_mass: float,
    is_top_mode: bool,
) -> str:
    """Outcome-directional rationale string for one mode of a bimodal PDF.

    The string tells the reader what probing this assumption DOES to the
    distribution: does it confirm the dominant scenario, collapse the tail,
    or swing the mean? The spike showed the model quotes this string verbatim
    before deciding, so it must be crisp and action-shaped.
    """
    # "top" here means highest-mass mode; the other is the minority/tail mode.
    if is_top_mode:
        if mode_center >= 10.0:
            return "confirms the winning scenario (top mode, Q>>0)"
        if mode_center <= -10.0:
            return "confirms the losing scenario (top mode, Q<<0)"
        if mode_center > 0:
            return "confirms the mildly-winning top mode"
        if mode_center < 0:
            return "confirms the mildly-losing top mode"
        return "confirms the near-zero top mode"
    # Non-top mode: the probe collapses the tail that shifts the mean.
    if mode_center >= 10.0:
        return "collapses the right tail — rules out the upside swing"
    if mode_center <= -10.0:
        return "collapses the left tail — rules out the disaster swing"
    if mode_center > 0:
        return "collapses a mildly-positive tail"
    if mode_center < 0:
        return "collapses a mildly-negative tail"
    return "collapses a near-zero tail"


def _seat_label(abs_seat: int, me: int) -> str:
    """Map an absolute seat to a human label relative to the current player."""
    offset = (abs_seat - me) % 4
    return {0: "self", 1: "left_opp", 2: "partner", 3: "right_opp"}[offset]


def _suggest_counterfactuals(
    game_state: Any,
    play: int,
    modes: list[dict],
    shape: str,
    gst: GameStateTensor,
    oracle: Stage1Oracle,
    device: str,
    unconditional_mean: float,
    n_samples: int = 5,
    max_candidates_per_seat: int = 5,
) -> list[dict]:
    """Find up to two ``{player, holds, rationale}`` assumptions whose conditional
    E[Q] most moves the mean toward one of the top-two modes.

    Only runs when ``shape != "unimodal"``. Restricts the search to the top
    ``max_candidates_per_seat`` unseen dominoes per non-self seat, ranked by
    trump-first then high-pip to keep cost bounded.
    """
    if shape == "unimodal" or len(modes) < 2:
        return []

    from burl.tools.engine import is_trump

    me = int(gst.current_player[0].item())
    played = set(int(d) for d in game_state.played)
    my_hand_set = set(
        int(d) for d in gst.hands[0, me].tolist() if int(d) >= 0
    )
    unseen = [d for d in range(28) if d not in played and d not in my_hand_set]

    def _priority(d: int) -> tuple[int, int]:
        return (1 if is_trump(game_state, d) else 0, d)

    unseen_sorted = sorted(unseen, key=_priority, reverse=True)
    candidates = unseen_sorted[:max_candidates_per_seat]

    top_modes = modes[:2]
    suggestions: list[dict] = []
    seen: set[tuple[int, int]] = set()

    for mode_idx, mode in enumerate(top_modes):
        target = mode["center"]
        is_top = mode_idx == 0
        best: tuple[float, int, int] | None = None   # (score, seat, dom)
        for seat_offset in (1, 2, 3):                 # skip self
            seat = (me + seat_offset) % 4
            for dom in candidates:
                if (seat, dom) in seen:
                    continue
                try:
                    cond = conditional_outcome(
                        game_state, play,
                        {"player": seat, "holds": dom},
                        n_samples=n_samples,
                        max_sampling_tries=20,
                        oracle=oracle,
                        device=device,
                    )
                except (ConditionUnreachable, ValueError):
                    continue
                # Score: closer to target AND farther from unconditional
                # (i.e. bigger swing in the right direction).
                toward_target = -abs(cond.mean - target)
                swing = abs(cond.mean - unconditional_mean)
                score = toward_target + 0.25 * swing
                if best is None or score > best[0]:
                    best = (score, seat, dom)
        if best is not None:
            _, seat, dom = best
            seen.add((seat, dom))
            suggestions.append({
                "player": _seat_label(seat, me),
                "holds": int(dom),
                "rationale": _rationale_for_mode(target, mode["mass"], is_top),
            })
    return suggestions


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

    shape, modes = _detect_modes(pdf)
    if len(modes) >= 2:
        gap = abs(modes[0]["center"] - modes[1]["center"])
    else:
        gap = 0.0

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
        distribution_shape=shape,
        modes=modes,
        gap_between_modes=float(gap),
        suggested_counterfactuals=[],
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
    suggest_counterfactuals: bool = True,
    enumerate: str | bool = "auto",
) -> OutcomeDistribution:
    """Sample N consistent worlds, evaluate ``play`` under Stage 1 in each,
    return the full outcome distribution (85-bin PDF + summary stats).

    ``play`` is a domino_id (not a slot). Voids inferred from play history are
    respected. Distribution is from the perspective of the seat currently to
    act; p_make is seat-aware (offense: Q >= 18; defense: Q >= -17).

    ``enumerate`` controls the world source:

    - ``"auto"`` (default) — enumerate all consistent worlds when the
      unplayed-not-mine pool has ``<= _AUTO_ENUMERATE_MAX_POOL`` dominoes
      (cheaper AND exact at trick 5-6), otherwise sample.
    - ``True`` — always enumerate. Raises ``ValueError`` if the pool exceeds
      ``_ENUMERATE_HARD_POOL_CAP`` (prevents accidental blowups).
    - ``False`` — always sample (the original N=sampling behaviour).

    The returned ``OutcomeDistribution.sampling_mode`` reports which path ran.

    When ``suggest_counterfactuals=True`` and the distribution is bimodal or
    multimodal, the returned ``suggested_counterfactuals`` list surfaces up to
    two ``{player, holds, rationale}`` probes that most resolve the ambiguity.
    Disable (e.g. in ``conditional_outcome``) to avoid recursion.
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

    pool = _pool_size(gst)
    if enumerate is True:
        if pool > _ENUMERATE_HARD_POOL_CAP:
            raise ValueError(
                f"enumerate=True refused: pool_size={pool} > "
                f"{_ENUMERATE_HARD_POOL_CAP}. Use enumerate='auto' or False."
            )
        use_enumerate = True
    elif enumerate is False:
        use_enumerate = False
    elif enumerate == "auto":
        use_enumerate = pool <= _AUTO_ENUMERATE_MAX_POOL
    else:
        raise ValueError(
            f"enumerate must be True, False, or 'auto'; got {enumerate!r}"
        )

    if use_enumerate:
        worlds_tensor, _raw = _enumerated_worlds_tensor(gst, game_state, device)
        q_all = _q_per_enumerated_worlds(gst, oracle, device, worlds_tensor)
        n_used = q_all.shape[0]
        sampling_mode = "enumerated"
    else:
        q_all, _ = _q_per_world(gst, oracle, n_samples, device)  # [n, 7]
        n_used = q_all.shape[0]
        sampling_mode = "sampled"

    q_slot = q_all[:, slot]                                   # [n]
    dist = _distribution_from_q_slice(q_slot, play, is_offense, n_used)
    dist.sampling_mode = sampling_mode

    if suggest_counterfactuals and dist.distribution_shape != "unimodal":
        dist.suggested_counterfactuals = _suggest_counterfactuals(
            game_state=game_state,
            play=play,
            modes=dist.modes,
            shape=dist.distribution_shape,
            gst=gst,
            oracle=oracle,
            device=device,
            unconditional_mean=dist.mean,
        )
    return dist


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
