"""Audit ``WorldSamplerMRV`` without conflating sampling and model encoding.

The audit has four deliberately separate layers:

1. enumerate every public-information-consistent remaining-hand world;
2. calculate the exact distribution induced by MRV's sequential choices;
3. verify the live tensor sampler against that induced distribution; and
4. reweight one canonical set of Stage-1 Q rows under uniform and MRV weights.

Late-hand fixtures keep exact enumeration and the analytic MRV recursion small.
The MRV recursion is important: a large empirical sample can reveal a mismatch,
but it cannot by itself distinguish structural bias from Monte Carlo error.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any, Mapping, Sequence

import torch

from forge.eq.enumeration_gpu import enumerate_worlds_cpu
from forge.eq.game_tensor import GameStateTensor
from forge.eq.generate.sampling import infer_voids_batched, sample_worlds_batched
from forge.eq.sampling_mrv_gpu import WorldSamplerMRV
from forge.oracle.tables import DOMINO_HIGH, DOMINO_LOW, can_follow
from forge.zeb.game import apply_action, legal_actions, new_game
from forge.zeb.types import ZebGameState


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
DEFAULT_MANIFEST = HERE / "manifest.json"

WorldKey = tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]


@dataclass(frozen=True)
class FixtureSpec:
    """A deterministic public-state fixture."""

    name: str
    deal_seed: int
    n_plays: int
    rollout_seed: int
    provenance: str = "audit panel"


@dataclass(frozen=True)
class SamplerInputs:
    """The exact boundary shared by enumeration and MRV sampling."""

    pool: tuple[int, ...]
    hand_sizes: tuple[int, int, int]
    voids: tuple[frozenset[int], frozenset[int], frozenset[int]]
    decl_id: int
    current_player: int


def domino_token(domino_id: int) -> str:
    """Return the repo's compact high-pip/low-pip domino token."""

    return f"{DOMINO_HIGH[domino_id]}{DOMINO_LOW[domino_id]}"


def load_manifest(path: Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def fixture_specs(manifest: Mapping[str, Any]) -> list[FixtureSpec]:
    return [FixtureSpec(**row) for row in manifest["fixtures"]]


def advance_fixture(spec: FixtureSpec) -> ZebGameState:
    """Replay the historical fixture construction exactly.

    The original Burl test used ``new_game(seed=seed, skip_bidding=True)`` and
    an independent ``random.Random(seed)`` to pick legal slots.  Keeping the
    rollout seed explicit makes that provenance inspectable.
    """

    state = new_game(seed=spec.deal_seed, skip_bidding=True)
    rng = random.Random(spec.rollout_seed)
    for step in range(spec.n_plays):
        actions = legal_actions(state)
        if not actions:
            raise RuntimeError(
                f"{spec.name}: no legal action before requested play {step}"
            )
        state = apply_action(state, rng.choice(actions))
    return state


def state_snapshot(state: ZebGameState) -> dict[str, Any]:
    """Convert a Zeb state into the public ``GameStateTensor`` snapshot schema."""

    played = set(state.played)
    hands = [
        [domino_id if domino_id not in played else -1 for domino_id in hand]
        for hand in state.hands
    ]
    played_mask = [domino_id in played for domino_id in range(28)]

    history: list[list[int]] = []
    for index, (player, domino_id) in enumerate(state.play_history):
        trick_start = (index // 4) * 4
        lead_domino_id = state.play_history[trick_start][1]
        history.append([player, domino_id, lead_domino_id])
    history.extend([[-1, -1, -1]] * (28 - len(history)))

    trick_plays = list(state.current_trick) + [-1] * (4 - len(state.current_trick))
    return {
        "schema_version": "forge.eq.snapshot.v1",
        "decl_id": state.decl_id,
        "bid_value": state.bid_state.high_bid,
        "bidder": state.bidder,
        "hands": hands,
        "played_mask": played_mask,
        "history": history,
        "trick_plays": trick_plays,
        "leader": state.trick_leader,
    }


def state_tensor(state: ZebGameState, device: str | torch.device = "cpu") -> GameStateTensor:
    return GameStateTensor.from_snapshot([state_snapshot(state)], device=device)


def sampler_inputs(state: ZebGameState) -> SamplerInputs:
    """Extract the same pool, hand sizes, and voids used by production sampling."""

    gst = state_tensor(state, "cpu")
    current_player = int(gst.current_player[0].item())
    my_remaining = {
        int(domino_id)
        for domino_id in gst.hands[0, current_player].tolist()
        if int(domino_id) >= 0
    }
    pool = tuple(
        domino_id
        for domino_id in range(28)
        if not bool(gst.played_mask[0, domino_id].item())
        and domino_id not in my_remaining
    )

    hand_counts = (gst.hands >= 0).sum(dim=2)
    hand_sizes = tuple(
        int(hand_counts[0, (current_player + offset) % 4].item())
        for offset in range(1, 4)
    )
    if sum(hand_sizes) != len(pool):
        raise AssertionError(
            f"opponent hand sizes {hand_sizes} do not partition pool {pool}"
        )

    void_tensor = infer_voids_batched(gst)[0]
    voids = tuple(
        frozenset(suit for suit in range(8) if bool(void_tensor[seat, suit].item()))
        for seat in range(3)
    )
    return SamplerInputs(
        pool=pool,
        hand_sizes=hand_sizes,  # type: ignore[arg-type]
        voids=voids,  # type: ignore[arg-type]
        decl_id=state.decl_id,
        current_player=current_player,
    )


def canonical_world(world: Sequence[Sequence[int]]) -> WorldKey:
    """Canonicalize remaining hands; padding and slot order carry no information."""

    hands = tuple(
        tuple(sorted(int(domino_id) for domino_id in hand if int(domino_id) >= 0))
        for hand in world
    )
    if len(hands) != 3:
        raise ValueError(f"world must contain three relative opponent hands: {world!r}")
    return hands  # type: ignore[return-value]


def enumerate_exact_worlds(inputs: SamplerInputs) -> tuple[WorldKey, ...]:
    """Enumerate the exact uniform remaining-hand world set."""

    raw = enumerate_worlds_cpu(
        pool=list(inputs.pool),
        known=[[], [], []],
        slots=list(inputs.hand_sizes),
        voids=[set(suits) for suits in inputs.voids],
        decl_id=inputs.decl_id,
    )
    worlds = tuple(sorted(canonical_world(world) for world in raw))
    if len(worlds) != len(set(worlds)):
        raise AssertionError("exact enumerator emitted duplicate worlds")
    return worlds


def candidate_masks(inputs: SamplerInputs) -> tuple[int, int, int]:
    """Build MRV's per-seat candidate masks from public void constraints."""

    masks: list[int] = []
    for void_suits in inputs.voids:
        mask = 0
        for domino_id in inputs.pool:
            violates = any(
                can_follow(domino_id, suit, inputs.decl_id)
                for suit in void_suits
            )
            if not violates:
                mask |= 1 << domino_id
        masks.append(mask)
    return tuple(masks)  # type: ignore[return-value]


def _bit_ids(mask: int) -> tuple[int, ...]:
    return tuple(index for index in range(28) if mask & (1 << index))


def analytic_mrv_distribution(
    pool: Sequence[int],
    hand_sizes: Sequence[int],
    candidates: Sequence[int],
) -> tuple[dict[WorldKey, float], float]:
    """Calculate the live MRV tensor algorithm's exact output distribution.

    MRV chooses the first seat with minimum ``candidate_count - need`` and
    then chooses uniformly among that seat's currently available candidates.
    Uniform local choices are not generally uniform over completed worlds:
    candidate choices can have different numbers of legal completions.

    The second return value is the probability that a path reaches an active
    state with no candidate for the selected seat.  The live tensor code does
    not reject or backtrack there: ``argmax`` over an all-false bit row returns
    domino 0, so domino 0 is injected even when it is not in the pool.  This
    function deliberately emulates that fallback and retains its malformed
    terminal output.  Comparing the output keys with exact enumeration then
    measures the invalid mass directly.
    """

    if len(hand_sizes) != 3 or len(candidates) != 3:
        raise ValueError("MRV audit expects exactly three opponent seats")
    pool_mask = sum(1 << int(domino_id) for domino_id in pool)
    expected = tuple(int(size) for size in hand_sizes)
    if sum(expected) != len(pool):
        raise ValueError("hand sizes must partition the pool")

    # state = (available, needs, hand_masks).  Aggregating equal states avoids
    # enumerating every ordering of choices separately.
    start = (pool_mask, expected, (0, 0, 0), False)
    frontier: dict[
        tuple[int, tuple[int, int, int], tuple[int, int, int], bool], float
    ] = {
        start: 1.0
    }

    for _step in range(len(pool)):
        next_frontier: defaultdict[
            tuple[int, tuple[int, int, int], tuple[int, int, int], bool], float
        ] = defaultdict(float)
        for (available, needs, hands, had_dead_end), state_probability in frontier.items():
            if available == 0:
                next_frontier[(available, needs, hands, had_dead_end)] += state_probability
                continue
            slack: list[int] = []
            for seat in range(3):
                if needs[seat] == 0:
                    slack.append(1000)
                else:
                    slack.append((candidates[seat] & available).bit_count() - needs[seat])
            selected_seat = min(range(3), key=lambda seat: (slack[seat], seat))
            choices = _bit_ids(candidates[selected_seat] & available)
            if not choices:
                # This is the current tensor implementation's all-false
                # ``argmax`` behavior, not a proposed repair.
                choices = (0,)
                had_dead_end = True

            branch_probability = state_probability / len(choices)
            for domino_id in choices:
                new_needs = list(needs)
                new_needs[selected_seat] -= 1
                new_hands = list(hands)
                new_hands[selected_seat] |= 1 << domino_id
                next_frontier[
                    (
                        available & ~(1 << domino_id),
                        tuple(new_needs),  # type: ignore[arg-type]
                        tuple(new_hands),  # type: ignore[arg-type]
                        had_dead_end,
                    )
                ] += branch_probability
        frontier = dict(next_frontier)

    distribution: defaultdict[WorldKey, float] = defaultdict(float)
    dead_end_mass = 0.0
    for (_available, _needs, hands, had_dead_end), probability in frontier.items():
        key = canonical_world([_bit_ids(mask) for mask in hands])
        distribution[key] += probability
        if had_dead_end:
            dead_end_mass += probability

    total = sum(distribution.values())
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-10):
        raise AssertionError(f"analytic MRV probability mass is {total}, expected 1")
    return dict(distribution), dead_end_mass


def sample_empirical_distribution(
    state: ZebGameState,
    inputs: SamplerInputs,
    n_samples: int,
    seed: int,
) -> tuple[dict[WorldKey, float], tuple[WorldKey, ...]]:
    """Run the live CPU tensor sampler with a deterministic torch seed."""

    torch.manual_seed(seed)
    gst = state_tensor(state, "cpu")
    sampler = WorldSamplerMRV(max_games=1, max_samples=n_samples, device="cpu")
    sampled = sample_worlds_batched(
        gst,
        sampler,
        n_samples,
        max_pool_size=len(inputs.pool),
    )[0]
    keys = tuple(canonical_world(world.tolist()) for world in sampled)
    counts = Counter(keys)
    return ({key: count / n_samples for key, count in counts.items()}, keys)


def distribution_tvd(
    left: Mapping[WorldKey, float],
    right: Mapping[WorldKey, float],
    left_invalid: float = 0.0,
    right_invalid: float = 0.0,
) -> float:
    keys = set(left) | set(right)
    return 0.5 * (
        sum(abs(left.get(key, 0.0) - right.get(key, 0.0)) for key in keys)
        + abs(left_invalid - right_invalid)
    )


def marginal_probabilities(distribution: Mapping[WorldKey, float]) -> dict[tuple[int, int], float]:
    marginals: defaultdict[tuple[int, int], float] = defaultdict(float)
    for world, probability in distribution.items():
        for seat, hand in enumerate(world):
            for domino_id in hand:
                marginals[(seat, domino_id)] += probability
    return dict(marginals)


def top_marginal_errors(
    exact: Mapping[WorldKey, float],
    observed: Mapping[WorldKey, float],
    current_player: int,
    limit: int = 10,
) -> list[dict[str, Any]]:
    exact_marginals = marginal_probabilities(exact)
    observed_marginals = marginal_probabilities(observed)
    keys = set(exact_marginals) | set(observed_marginals)
    rows = []
    for seat, domino_id in keys:
        exact_probability = exact_marginals.get((seat, domino_id), 0.0)
        observed_probability = observed_marginals.get((seat, domino_id), 0.0)
        rows.append(
            {
                "relative_opponent": seat,
                "absolute_seat": (current_player + seat + 1) % 4,
                "domino_id": domino_id,
                "domino": domino_token(domino_id),
                "uniform_probability": exact_probability,
                "mrv_probability": observed_probability,
                "delta": observed_probability - exact_probability,
            }
        )
    rows.sort(key=lambda row: (-abs(row["delta"]), row["relative_opponent"], row["domino_id"]))
    return rows[:limit]


def implementation_conformance(
    analytic: Mapping[WorldKey, float],
    analytic_invalid: float,
    empirical: Mapping[WorldKey, float],
    empirical_invalid: float,
    n_samples: int,
) -> dict[str, Any]:
    """Compare live frequencies with the analytic distribution in z units."""

    cells = [
        (key, analytic.get(key, 0.0))
        for key in set(analytic) | set(empirical)
    ]
    cells.append((None, analytic_invalid))
    max_z = 0.0
    worst_cell: WorldKey | None = None
    for key, probability in cells:
        observed = empirical_invalid if key is None else empirical.get(key, 0.0)
        variance = max(probability * (1.0 - probability), 1.0 / n_samples)
        standard_error = math.sqrt(variance / n_samples)
        z = abs(observed - probability) / standard_error
        if z > max_z:
            max_z = z
            worst_cell = key
    return {
        "max_abs_standardized_residual": max_z,
        "threshold": 7.0,
        "passes": max_z <= 7.0,
        "worst_world": world_json(worst_cell) if worst_cell is not None else "invalid",
        "empirical_vs_analytic_tvd": distribution_tvd(
            empirical,
            analytic,
            empirical_invalid,
            analytic_invalid,
        ),
    }


def world_json(world: WorldKey | None) -> list[list[int]] | None:
    if world is None:
        return None
    return [list(hand) for hand in world]


def played_by_absolute_seat(state: ZebGameState) -> tuple[tuple[int, ...], ...]:
    rows: list[list[int]] = [[], [], [], []]
    for player, domino_id in state.play_history:
        rows[player].append(domino_id)
    return tuple(tuple(sorted(row)) for row in rows)


def packed_deal_for_output(
    state: ZebGameState,
    world: WorldKey,
    *,
    require_valid: bool,
) -> tuple[tuple[int, ...], ...]:
    """Pack a sampler output into a controlled 4x7 oracle input.

    This explicit reconstruction is the confound control.  Every Q row is
    evaluated once from the same full deal and then reweighted; sampled and
    enumerated paths never receive differently shaped hand encodings.  Invalid
    MRV outputs are padded if necessary and retained when ``require_valid`` is
    false, allowing their downstream effect to be measured without pretending
    they are legal deals.
    """

    current_player = (state.trick_leader + len(state.current_trick)) % 4
    played_by = played_by_absolute_seat(state)
    deal: list[tuple[int, ...] | None] = [None, None, None, None]
    deal[current_player] = tuple(state.hands[current_player])
    for relative_seat, remaining_hand in enumerate(world):
        absolute_seat = (current_player + relative_seat + 1) % 4
        raw_hand = tuple(sorted((*remaining_hand, *played_by[absolute_seat])))
        if len(raw_hand) > 7:
            raise AssertionError(
                f"seat {absolute_seat}: reconstructed {len(raw_hand)} tiles, maximum is 7"
            )
        full_hand = raw_hand + (-1,) * (7 - len(raw_hand))
        deal[absolute_seat] = full_hand
    packed = tuple(hand for hand in deal if hand is not None)
    valid_dominoes = sorted(domino for hand in packed for domino in hand if domino >= 0)
    if require_valid and (len(packed) != 4 or valid_dominoes != list(range(28))):
        raise AssertionError("reconstructed world is not a complete 28-domino deal")
    return packed


def full_deal_for_world(state: ZebGameState, world: WorldKey) -> tuple[tuple[int, ...], ...]:
    """Pack one exact remaining-hand world into a valid 4x7 initial deal."""

    return packed_deal_for_output(state, world, require_valid=True)


def checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def remaining_worlds_tensor(
    worlds: Sequence[WorldKey],
    device: str,
) -> torch.Tensor:
    """Pack canonical remaining hands in production's sampler tensor layout."""

    packed = torch.full(
        (1, len(worlds), 3, 7),
        -1,
        dtype=torch.int32,
        device=device,
    )
    for world_index, world in enumerate(worlds):
        for seat, hand in enumerate(world):
            if len(hand) > 7:
                raise ValueError(f"sampler output hand is too long: {hand}")
            if hand:
                packed[0, world_index, seat, : len(hand)] = torch.tensor(
                    hand,
                    dtype=torch.int32,
                    device=device,
                )
    return packed


def query_world_q_by_encoding(
    state: ZebGameState,
    worlds: Sequence[WorldKey],
    checkpoint: Path,
    device: str,
    chunk_size: int = 256,
) -> dict[str, dict[WorldKey, list[float]]]:
    """Evaluate sampler outputs under production and sensitivity encodings.

    ``remaining_only_canonical`` is primary: it calls production
    ``build_hypothetical_deals`` with the current actor hand and remaining-only
    opponent worlds.  A single ascending slot order ensures that only world
    weights change between uniform and MRV expectations.

    ``full_initial_control`` reconstructs played-plus-remaining initial deals.
    It is an encoding-sensitivity control, not the current production format.
    """

    from forge.eq.generate.deals import build_hypothetical_deals
    from forge.eq.generate.model import query_model
    from forge.eq.generate.tokenization import tokenize_batched
    from forge.eq.oracle import Stage1Oracle
    from forge.eq.tokenize_gpu import GPUTokenizer

    if not checkpoint.exists():
        raise FileNotFoundError(f"Stage-1 checkpoint is absent: {checkpoint}")
    oracle = Stage1Oracle(
        checkpoint_path=checkpoint,
        device=device,
        compile=False,
        use_async=False,
        use_gpu_tokenizer=False,
    )
    gst = state_tensor(state, device)
    q_by_encoding: dict[str, dict[WorldKey, list[float]]] = {
        "remaining_only_canonical": {},
        "full_initial_control": {},
    }
    exact_inputs = set(enumerate_exact_worlds(sampler_inputs(state)))
    for start in range(0, len(worlds), chunk_size):
        batch_worlds = worlds[start : start + chunk_size]
        remaining_deals = build_hypothetical_deals(
            gst,
            remaining_worlds_tensor(batch_worlds, device),
        )
        full_deals = torch.tensor(
            [[
                packed_deal_for_output(
                    state,
                    world,
                    require_valid=world in exact_inputs,
                )
                for world in batch_worlds
            ]],
            dtype=torch.int32,
            device=device,
        )
        for encoding, deals in (
            ("remaining_only_canonical", remaining_deals),
            ("full_initial_control", full_deals),
        ):
            tokenizer = GPUTokenizer(max_batch=len(batch_worlds), device=device)
            tokens, masks = tokenize_batched(gst, deals, tokenizer)
            q_values = query_model(
                oracle.model,
                tokens,
                masks,
                gst,
                n_samples=len(batch_worlds),
                device=device,
                chunk_size=chunk_size,
            )
            for world, q_row in zip(
                batch_worlds,
                q_values.float().cpu().tolist(),
                strict=True,
            ):
                q_by_encoding[encoding][world] = q_row
    return q_by_encoding


def rank_actions(
    state: ZebGameState,
    exact_worlds: Sequence[WorldKey],
    q_rows: Mapping[WorldKey, Sequence[float]],
    exact_distribution: Mapping[WorldKey, float],
    mrv_distribution: Mapping[WorldKey, float],
    empirical_distribution: Mapping[WorldKey, float],
) -> dict[str, Any]:
    """Measure expected-Q and legal-action rank effects of world weighting."""

    required_worlds = set(exact_distribution) | set(mrv_distribution) | set(empirical_distribution)
    missing = required_worlds - set(q_rows)
    if missing:
        raise ValueError(f"Q rows missing for {len(missing)} sampler outputs")
    legal_slots = tuple(int(slot) for slot in legal_actions(state))
    current_player = (state.trick_leader + len(state.current_trick)) % 4

    def expected(slot: int, distribution: Mapping[WorldKey, float]) -> float:
        return sum(
            probability * float(q_rows[world][slot])
            for world, probability in distribution.items()
        )

    mrv_valid_mass = sum(mrv_distribution.get(world, 0.0) for world in exact_worlds)
    empirical_valid_mass = sum(empirical_distribution.get(world, 0.0) for world in exact_worlds)
    mrv_valid = {
        world: mrv_distribution.get(world, 0.0) / mrv_valid_mass
        for world in exact_worlds
        if mrv_distribution.get(world, 0.0) > 0.0
    }
    empirical_valid = {
        world: empirical_distribution.get(world, 0.0) / empirical_valid_mass
        for world in exact_worlds
        if empirical_distribution.get(world, 0.0) > 0.0
    }

    rows: list[dict[str, Any]] = []
    for slot in legal_slots:
        uniform_value = expected(slot, exact_distribution)
        mrv_value = expected(slot, mrv_distribution)
        empirical_value = expected(slot, empirical_distribution)
        mrv_valid_value = expected(slot, mrv_valid)
        empirical_valid_value = expected(slot, empirical_valid)
        domino_id = state.hands[current_player][slot]
        rows.append(
            {
                "slot": slot,
                "domino_id": domino_id,
                "domino": domino_token(domino_id),
                "uniform_eq": uniform_value,
                "analytic_mrv_eq": mrv_value,
                "empirical_mrv_eq": empirical_value,
                "analytic_shift_q": mrv_value - uniform_value,
                "empirical_shift_q": empirical_value - uniform_value,
                "analytic_valid_conditional_eq": mrv_valid_value,
                "analytic_valid_conditional_shift_q": mrv_valid_value - uniform_value,
                "empirical_valid_conditional_eq": empirical_valid_value,
            }
        )

    uniform_rank = sorted(rows, key=lambda row: (-row["uniform_eq"], row["slot"]))
    mrv_rank = sorted(rows, key=lambda row: (-row["analytic_mrv_eq"], row["slot"]))
    empirical_rank = sorted(rows, key=lambda row: (-row["empirical_mrv_eq"], row["slot"]))
    valid_rank = sorted(
        rows,
        key=lambda row: (-row["analytic_valid_conditional_eq"], row["slot"]),
    )
    uniform_best = uniform_rank[0]
    mrv_best = mrv_rank[0]
    uniform_by_slot = {row["slot"]: row["uniform_eq"] for row in rows}
    return {
        "legal_action_count": len(rows),
        "actions": sorted(rows, key=lambda row: row["slot"]),
        "uniform_rank_slots": [row["slot"] for row in uniform_rank],
        "analytic_mrv_rank_slots": [row["slot"] for row in mrv_rank],
        "empirical_mrv_rank_slots": [row["slot"] for row in empirical_rank],
        "analytic_valid_conditional_rank_slots": [row["slot"] for row in valid_rank],
        "uniform_best_slot": uniform_best["slot"],
        "analytic_mrv_best_slot": mrv_best["slot"],
        "argmax_flipped": uniform_best["slot"] != mrv_best["slot"],
        "valid_conditional_argmax_flipped": uniform_best["slot"] != valid_rank[0]["slot"],
        "exact_regret_of_mrv_choice_q": (
            uniform_best["uniform_eq"] - uniform_by_slot[mrv_best["slot"]]
        ),
        "max_abs_action_shift_q": max(abs(row["analytic_shift_q"]) for row in rows),
        "max_abs_valid_conditional_action_shift_q": max(
            abs(row["analytic_valid_conditional_shift_q"]) for row in rows
        ),
        "analytic_valid_mass": mrv_valid_mass,
        "empirical_valid_mass": empirical_valid_mass,
    }


def analyze_fixture(
    spec: FixtureSpec,
    n_samples: int,
    empirical_seed: int,
    q_by_encoding: Mapping[str, Mapping[WorldKey, Sequence[float]]] | None = None,
) -> tuple[dict[str, Any], ZebGameState, tuple[WorldKey, ...], dict[WorldKey, float]]:
    state = advance_fixture(spec)
    inputs = sampler_inputs(state)
    exact_worlds = enumerate_exact_worlds(inputs)
    if not exact_worlds:
        raise RuntimeError(f"{spec.name}: exact enumeration found no consistent worlds")
    exact_distribution = {world: 1.0 / len(exact_worlds) for world in exact_worlds}
    analytic_distribution_all, analytic_dead_end = analytic_mrv_distribution(
        inputs.pool,
        inputs.hand_sizes,
        candidate_masks(inputs),
    )
    empirical_distribution_all, sampled_keys = sample_empirical_distribution(
        state,
        inputs,
        n_samples=n_samples,
        seed=empirical_seed,
    )
    exact_set = set(exact_worlds)
    empirical_invalid = sum(
        probability
        for world, probability in empirical_distribution_all.items()
        if world not in exact_set
    )
    empirical_distribution = {
        world: probability
        for world, probability in empirical_distribution_all.items()
        if world in exact_set
    }
    analytic_unexpected = set(analytic_distribution_all) - exact_set
    analytic_invalid = sum(analytic_distribution_all[world] for world in analytic_unexpected)
    analytic_distribution = {
        world: probability
        for world, probability in analytic_distribution_all.items()
        if world in exact_set
    }
    analytic_valid_mass = sum(analytic_distribution.values())
    empirical_valid_mass = sum(empirical_distribution.values())
    analytic_valid_conditional = {
        world: probability / analytic_valid_mass
        for world, probability in analytic_distribution.items()
    }
    empirical_valid_conditional = {
        world: probability / empirical_valid_mass
        for world, probability in empirical_distribution.items()
    }

    current_player = inputs.current_player
    legacy_mrv_conformance = implementation_conformance(
        analytic_distribution_all,
        0.0,
        empirical_distribution_all,
        0.0,
        n_samples,
    )
    uniform_conformance = implementation_conformance(
        exact_distribution,
        0.0,
        empirical_distribution_all,
        0.0,
        n_samples,
    )
    top_errors = top_marginal_errors(
        exact_distribution,
        analytic_distribution_all,
        current_player,
    )
    result: dict[str, Any] = {
        "fixture": asdict(spec),
        "state": {
            "decl_id": state.decl_id,
            "bidder": state.bidder,
            "current_player": current_player,
            "trick_leader": state.trick_leader,
            "current_trick": list(state.current_trick),
            "legal_slots": list(legal_actions(state)),
            "legal_dominoes": [
                domino_token(state.hands[current_player][slot])
                for slot in legal_actions(state)
            ],
        },
        "constraints": {
            "pool": list(inputs.pool),
            "pool_dominoes": [domino_token(domino_id) for domino_id in inputs.pool],
            "hand_sizes": list(inputs.hand_sizes),
            "void_suits_by_relative_opponent": [sorted(suits) for suits in inputs.voids],
        },
        "exact_world_count": len(exact_worlds),
        "sample_count": n_samples,
        "sample_unique_world_count": len(set(sampled_keys)),
        "analytic_mrv_dead_end_probability": analytic_dead_end,
        "analytic_mrv_invalid_mass": analytic_invalid,
        "empirical_invalid_fraction": empirical_invalid,
        "uniform_vs_analytic_mrv_tvd": distribution_tvd(
            exact_distribution,
            analytic_distribution_all,
        ),
        "uniform_vs_empirical_mrv_tvd": distribution_tvd(
            exact_distribution,
            empirical_distribution_all,
        ),
        "uniform_vs_analytic_mrv_valid_conditional_tvd": distribution_tvd(
            exact_distribution,
            analytic_valid_conditional,
        ),
        "uniform_vs_empirical_mrv_valid_conditional_tvd": distribution_tvd(
            exact_distribution,
            empirical_valid_conditional,
        ),
        "analytic_valid_mass": analytic_valid_mass,
        "empirical_valid_mass": empirical_valid_mass,
        "max_abs_marginal_error": abs(top_errors[0]["delta"]) if top_errors else 0.0,
        "top_marginal_errors": top_errors,
        "top_valid_conditional_marginal_errors": top_marginal_errors(
            exact_distribution,
            analytic_valid_conditional,
            current_player,
        ),
        "live_sampler_algorithm": getattr(
            WorldSamplerMRV, "algorithm", "legacy-greedy-mrv"
        ),
        # Keep the historical key for old result readers. At current HEAD it
        # intentionally fails once the repaired sampler diverges from the
        # analytically reconstructed legacy implementation.
        "implementation_conformance": legacy_mrv_conformance,
        "legacy_mrv_conformance": legacy_mrv_conformance,
        "uniform_conformance": uniform_conformance,
        "downstream_q": None,
    }
    if q_by_encoding is not None:
        by_encoding = {
            encoding: rank_actions(
                state,
                exact_worlds,
                q_rows,
                exact_distribution,
                analytic_distribution_all,
                empirical_distribution_all,
            )
            for encoding, q_rows in q_by_encoding.items()
        }
        primary = by_encoding["remaining_only_canonical"]
        control = by_encoding["full_initial_control"]
        result["downstream_q"] = {
            "status": "complete",
            "primary_encoding": "remaining_only_canonical",
            "primary_encoding_note": (
                "production remaining-only representation with one canonical opponent "
                "slot order; malformed MRV outputs are retained"
            ),
            "control_encoding": "full_initial_control",
            "control_encoding_note": (
                "played-plus-remaining 4x7 sensitivity control; not current production"
            ),
            "by_encoding": by_encoding,
            "encoding_sensitivity": {
                "argmax_flip_agrees": primary["argmax_flipped"] == control["argmax_flipped"],
                "primary_max_abs_action_shift_q": primary["max_abs_action_shift_q"],
                "control_max_abs_action_shift_q": control["max_abs_action_shift_q"],
                "max_shift_difference_q": (
                    primary["max_abs_action_shift_q"] - control["max_abs_action_shift_q"]
                ),
            },
        }
    return result, state, exact_worlds, exact_distribution


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def aggregate_summary(
    manifest: Mapping[str, Any],
    fixture_results: Sequence[Mapping[str, Any]],
    checkpoint: Path,
    checkpoint_hash: str | None,
    device: str,
    oracle_enabled: bool,
) -> dict[str, Any]:
    downstream = [
        row["downstream_q"]
        for row in fixture_results
        if row.get("downstream_q") and row["downstream_q"].get("status") == "complete"
    ]
    primary_downstream = [
        row["by_encoding"]["remaining_only_canonical"] for row in downstream
    ]
    max_shift = max(
        (float(row["max_abs_action_shift_q"]) for row in primary_downstream),
        default=None,
    )
    flips = sum(bool(row["argmax_flipped"]) for row in primary_downstream)
    valid_conditional_flips = sum(
        bool(row["valid_conditional_argmax_flipped"])
        for row in primary_downstream
    )
    legacy_conformant = all(
        row["legacy_mrv_conformance"]["passes"] for row in fixture_results
    )
    uniform_conformant = all(
        row["uniform_conformance"]["passes"] for row in fixture_results
    )
    live_valid = all(
        row["empirical_invalid_fraction"] == 0.0
        for row in fixture_results
    )
    legacy_valid = all(
        row["analytic_mrv_invalid_mass"] <= 1e-12
        for row in fixture_results
    )
    biased = any(row["uniform_vs_analytic_mrv_tvd"] > 1e-9 for row in fixture_results)
    valid_world_biased = any(
        row["uniform_vs_analytic_mrv_valid_conditional_tvd"] > 1e-9
        for row in fixture_results
    )
    dead_end_detected = any(
        row["analytic_mrv_dead_end_probability"] > 1e-12
        for row in fixture_results
    )
    return {
        "schema_version": manifest["schema_version"],
        "status": "complete" if oracle_enabled and len(downstream) == len(fixture_results) else "partial",
        "run": {
            "device": device,
            "checkpoint": str(checkpoint.relative_to(REPO_ROOT)) if checkpoint.is_relative_to(REPO_ROOT) else str(checkpoint),
            "checkpoint_sha256": checkpoint_hash,
            "oracle_enabled": oracle_enabled,
            "fixture_count": len(fixture_results),
            "sample_count_per_fixture": fixture_results[0]["sample_count"] if fixture_results else 0,
        },
        "findings": {
            "analytic_structural_bias_detected": biased,
            "dead_end_fallback_detected": dead_end_detected,
            "valid_world_bias_detected": valid_world_biased,
            "live_sampler_algorithm": (
                fixture_results[0]["live_sampler_algorithm"]
                if fixture_results else "unknown"
            ),
            "live_sampler_matches_analytic_mrv": legacy_conformant,
            "live_sampler_matches_uniform": uniform_conformant,
            "all_live_sampled_worlds_valid": live_valid,
            "legacy_mrv_all_outputs_valid": legacy_valid,
            "max_analytic_invalid_mass": max(
                row["analytic_mrv_invalid_mass"] for row in fixture_results
            ),
            "max_uniform_vs_analytic_mrv_tvd": max(
                row["uniform_vs_analytic_mrv_tvd"] for row in fixture_results
            ),
            "max_uniform_vs_analytic_mrv_valid_conditional_tvd": max(
                row["uniform_vs_analytic_mrv_valid_conditional_tvd"]
                for row in fixture_results
            ),
            "max_abs_marginal_error": max(
                row["max_abs_marginal_error"] for row in fixture_results
            ),
            "downstream_q_fixture_count": len(downstream),
            "downstream_argmax_flip_count": flips,
            "downstream_valid_conditional_argmax_flip_count": valid_conditional_flips,
            "max_abs_action_shift_q": max_shift,
            "max_abs_valid_conditional_action_shift_q": max(
                (
                    float(row["max_abs_valid_conditional_action_shift_q"])
                    for row in primary_downstream
                ),
                default=None,
            ),
        },
        "historical_claim": {
            **manifest["historical_claim"],
            "clean_reproduction_status": "not_directly_comparable",
            "reason": (
                "The surviving note omits the selected action, sample count, RNG seed, "
                "and hand-encoding parity. The audit reports the clean historical-fixture "
                "effect without treating numeric proximity as replication."
            ),
        },
        "instrument_boundary": manifest["instrument_boundary"],
        "fixtures": list(fixture_results),
    }


def run_audit(
    manifest: Mapping[str, Any],
    n_samples: int,
    empirical_seed: int,
    device: str,
    with_oracle: bool,
) -> dict[str, Any]:
    specs = fixture_specs(manifest)
    checkpoint = REPO_ROOT / manifest["checkpoint"]
    checkpoint_hash = checkpoint_sha256(checkpoint) if checkpoint.exists() else None
    results: list[dict[str, Any]] = []

    # Load/evaluate per fixture rather than retaining one model in the pure
    # analysis function.  This keeps tests free of Lightning/model imports.
    for index, spec in enumerate(specs):
        result, state, exact_worlds, _ = analyze_fixture(
            spec,
            n_samples=n_samples,
            empirical_seed=empirical_seed + index,
        )
        if with_oracle:
            inputs = sampler_inputs(state)
            analytic_outputs, _dead_end_mass = analytic_mrv_distribution(
                inputs.pool,
                inputs.hand_sizes,
                candidate_masks(inputs),
            )
            outputs_to_query = tuple(sorted(set(exact_worlds) | set(analytic_outputs)))
            q_by_encoding = query_world_q_by_encoding(
                state,
                outputs_to_query,
                checkpoint=checkpoint,
                device=device,
                chunk_size=int(manifest.get("oracle_chunk_size", 256)),
            )
            # Reuse the already sampled deterministic result by rerunning the
            # cheap analysis with Q rows.  This guarantees one code path for
            # downstream weighting and validation.
            result, _, _, _ = analyze_fixture(
                spec,
                n_samples=n_samples,
                empirical_seed=empirical_seed + index,
                q_by_encoding=q_by_encoding,
            )
        results.append(result)

    return aggregate_summary(
        manifest,
        results,
        checkpoint,
        checkpoint_hash,
        device,
        with_oracle,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--empirical-seed", type=int, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument(
        "--skip-oracle",
        action="store_true",
        help="Run distribution/marginal audit only; downstream Q status is partial.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = load_manifest(args.manifest)
    n_samples = int(args.samples or manifest["sample_count"])
    empirical_seed = int(args.empirical_seed or manifest["empirical_seed"])
    if n_samples <= 0:
        raise ValueError("--samples must be positive")
    device = resolve_device(args.device)
    result = run_audit(
        manifest,
        n_samples=n_samples,
        empirical_seed=empirical_seed,
        device=device,
        with_oracle=not args.skip_oracle,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
