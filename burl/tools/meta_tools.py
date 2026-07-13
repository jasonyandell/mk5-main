"""Meta-tools for Burl — decision-aid tools that operate over other tools.

``what_would_change_my_mind(play)`` is the first and (for now) only entry in
this module. It ranks unseen-world assumptions by how much they move the
E[Q] of a given play, so Burl can see which probes are worth running BEFORE
it has to read a (possibly bimodal) PDF. This short-circuits the
breadth-first "evaluate every alternative play" escape hatch the live eval
revealed in the candlewax session.

Shape is patterned after ``eq_distribution._suggest_counterfactuals`` but
does NOT reuse its private helpers — that function is a sibling, not a
dependency. We only import public names (``eq_outcome_distribution``,
``conditional_outcome``, ``ConditionUnreachable``, ``load_eq_oracle``) from
``burl.tools.eq_distribution``. Any duplication is intentional: the
candlewax file is owned by a concurrent editor.
"""

from __future__ import annotations

from typing import Any

import torch

from burl.tools.engine import is_trump, unseen
from burl.tools.eq_distribution import (
    ConditionUnreachable,
    Stage1Oracle,
    conditional_outcome,
    eq_outcome_distribution,
    load_eq_oracle,
)


# Start simple: if the unseen pool is small enough, enumerate it all. Above
# this threshold we trim to top-``_MAX_DOMINOS_PER_SEAT`` by the same
# trumps-first heuristic _suggest_counterfactuals uses, so cost stays
# bounded at ~3 * k * conditional-call overhead.
_SMALL_POOL_CUTOFF = 15
_MAX_DOMINOS_PER_SEAT = 5

# Qualitative tags applied to the rationale text. Thresholds are Q units,
# tuned to match the language the _rationale_for_mode helper uses on the
# candlewax side so the model reads consistent signal across both tools.
_BIG_SHIFT_THRESHOLD = 6.0       # |shift| >= this -> "big shift"
_CONFIRM_THRESHOLD = 1.5         # |shift| < this  -> "confirms current outlook"
_DISASTER_FLAG_CEILING = -8.0    # conditional mean <= this, shift negative -> "disaster flag"


def _seat_label(abs_seat: int, me: int) -> str:
    """Relative-seat human label. Duplicated from ``eq_distribution._seat_label``
    to keep this module importable without reaching into a private name."""
    offset = (abs_seat - me) % 4
    return {0: "self", 1: "left_opp", 2: "partner", 3: "right_opp"}[offset]


def _me_abs(game_state: Any) -> int:
    """Absolute seat for the player to act (current player).

    Duplicates the very small ``_abs_current_player`` helper in
    ``burl.tools.engine`` — public in spirit but private in name. Inlined
    here so this file is self-contained.
    """
    leader = getattr(game_state, "trick_leader", None)
    if leader is None:
        leader = game_state.leader
    return (int(leader) + len(game_state.current_trick)) % 4


def _priority_for_seat(game_state: Any, d: int) -> tuple[int, int]:
    """Sort key: trumps first, then high-pip id. Matches the candlewax
    candidate ranking."""
    return (1 if is_trump(game_state, d) else 0, int(d))


def _qualitative_tag(
    unconditional_mean: float,
    conditional_mean: float,
    shift: float,
) -> str:
    """Short label for the rationale string.

    - ``disaster flag``: conditional mean is deeply negative AND worse than
      unconditional (the probe reveals a hidden catastrophe).
    - ``big shift``: |shift| is large in either direction (mean moves a lot).
    - ``confirms current outlook``: |shift| is tiny (probe doesn't move mean).
    - ``moderate shift``: everything in between.
    """
    abs_shift = abs(shift)
    if (
        conditional_mean <= _DISASTER_FLAG_CEILING
        and shift <= -_BIG_SHIFT_THRESHOLD
    ):
        return "disaster flag"
    if abs_shift >= _BIG_SHIFT_THRESHOLD:
        return "big shift"
    if abs_shift < _CONFIRM_THRESHOLD:
        return "confirms current outlook"
    return "moderate shift"


def _format_rationale(
    unconditional_mean: float,
    conditional_mean: float,
    shift: float,
) -> str:
    """Action-shaped, mechanical rationale string.

    Format: ``"<tag>: flips mean from <uncond> to <cond> (<shift>)"``.

    The leading tag is qualitative; the numeric tail is the ground truth.
    Tested to be short (<80 chars) so it renders cleanly in tool-observation
    blocks.
    """
    tag = _qualitative_tag(unconditional_mean, conditional_mean, shift)
    return (
        f"{tag}: flips mean from {unconditional_mean:+.1f} "
        f"to {conditional_mean:+.1f} ({shift:+.1f})"
    )


def _candidate_dominos(
    game_state: Any,
    small_pool_cutoff: int = _SMALL_POOL_CUTOFF,
    max_per_seat: int = _MAX_DOMINOS_PER_SEAT,
) -> list[int]:
    """Pick the unseen dominoes to probe.

    If the unseen pool is small (<= cutoff) we probe all of it — the cost
    is bounded and we'd rather give an exact answer. If it's larger, we
    trim to the top ``max_per_seat`` by the trumps-first / high-pip
    heuristic to keep total calls <= 3 * max_per_seat.
    """
    unseen_set = unseen(game_state)
    if len(unseen_set) <= small_pool_cutoff:
        return sorted(unseen_set)
    ranked = sorted(
        unseen_set,
        key=lambda d: _priority_for_seat(game_state, d),
        reverse=True,
    )
    return ranked[:max_per_seat]


def what_would_change_my_mind(
    game_state: Any,
    play: int,
    n_samples_per_probe: int = 5,
    top_k: int = 5,
    oracle: Stage1Oracle | None = None,
    device: str | None = None,
) -> dict:
    """For a given legal play, rank unseen-world assumptions by how much
    they shift E[Q] of that play. Mechanical, no authored reasoning.

    Returns:
      {
        "play": int,
        "unconditional_mean": float,
        "assumptions": [
            {"player": <seat>, "holds": <domino_id>,
             "conditional_mean": float, "shift": float,
             "rationale": "flips mean from X to Y (+Z)"},
            ...  # top_k entries, sorted by |shift| desc
        ],
      }

    Seats in the returned ``assumptions`` are RELATIVE labels (``self`` is
    never present; non-self = ``left_opp``, ``partner``, ``right_opp``) to
    match the candlewax ``suggested_counterfactuals`` shape. This is the
    shape the model is already primed to read.

    The unconditional baseline and every conditional probe set
    ``suggest_counterfactuals=False`` to avoid recursion: this tool
    BUILDS the counterfactual list itself.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if oracle is None:
        oracle = load_eq_oracle(device=device)

    # Baseline: mean of play's outcome with no conditioning.
    baseline = eq_outcome_distribution(
        game_state,
        int(play),
        n_samples=n_samples_per_probe,
        oracle=oracle,
        device=device,
        suggest_counterfactuals=False,
    )
    uncond_mean = float(baseline.mean)

    me = _me_abs(game_state)
    candidates = _candidate_dominos(game_state)

    probes: list[dict] = []
    for seat_offset in (1, 2, 3):            # skip self
        seat_abs = (me + seat_offset) % 4
        for dom in candidates:
            assumption = {"player": seat_abs, "holds": int(dom)}
            try:
                cond = conditional_outcome(
                    game_state,
                    int(play),
                    assumption,
                    n_samples=n_samples_per_probe,
                    max_sampling_tries=20,
                    oracle=oracle,
                    device=device,
                )
            except (ConditionUnreachable, ValueError):
                # Inconsistent with voids / already-played dominoes — skip.
                continue

            cond_mean = float(cond.mean)
            shift = cond_mean - uncond_mean
            probes.append({
                "player": _seat_label(seat_abs, me),
                "holds": int(dom),
                "conditional_mean": round(cond_mean, 3),
                "shift": round(shift, 3),
                "rationale": _format_rationale(uncond_mean, cond_mean, shift),
            })

    probes.sort(key=lambda p: abs(p["shift"]), reverse=True)
    top = probes[:max(0, int(top_k))]

    return {
        "play": int(play),
        "unconditional_mean": round(uncond_mean, 3),
        "assumptions": top,
    }


# --------------------------------------------------------------------------- #
# Self-test                                                                    #
# --------------------------------------------------------------------------- #


if __name__ == "__main__":
    import random as _r

    from forge.zeb.game import apply_action, legal_actions, new_game

    from burl.tools.engine import is_legal

    seed = 900013
    state = new_game(seed=seed, skip_bidding=True)
    rng = _r.Random(seed)
    for _ in range(12):
        slots = legal_actions(state)
        if not slots:
            break
        state = apply_action(state, rng.choice(slots))
    me = _me_abs(state)
    my_hand = [d for d in state.hands[me] if d not in state.played]
    legal = [d for d in my_hand if is_legal(state, d)[0]]
    play = legal[0]

    print(f"seed={seed} me(abs)={me} play={play}")
    result = what_would_change_my_mind(state, play, n_samples_per_probe=5, top_k=5)
    print(f"unconditional_mean: {result['unconditional_mean']:+.2f}")
    print(f"top {len(result['assumptions'])} assumptions (sorted by |shift|):")
    for a in result["assumptions"]:
        print(
            f"  {a['player']:>9s} holds {a['holds']:2d}  "
            f"cond={a['conditional_mean']:+6.2f}  "
            f"shift={a['shift']:+6.2f}  "
            f"{a['rationale']}"
        )
    print("self-test: OK")
