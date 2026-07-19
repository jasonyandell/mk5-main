"""hoyt/br.py — exact best response vs a frozen profile, net-free.

`br_solve(subgame, profile, payoff43, hero)` per CONTRACTS.md. Value is the
hero-optimal expectation of `payoff43[final declaring points]` in DECLARING
orientation (never sign-flipped); hero maximizes sign*value with
sign = +1 iff hero_team == bid_team. Tie rule: hero moves ascending, first
strict sign*v > sign*best improvement wins (walt parity, root and interior).

Three execution shapes:

- **SigmaTable + hero=root.me** (net-free walt): the table carries the
  captured walk structure, so the solve is leaf-weighting + walt's backward
  pass only — no tree walk at all. `rewalk=True` forces the generic engine
  through the table positionally instead (self-consistency / honest
  "re-walk" timing).
- **StochasticProfile + hero=root.me**: full-width expectimax — profile
  slots replicate over the support with weight *= prob, same SoA wave
  machinery. If a wave blows the slot budget the solve falls back to one
  tree per legal root move (root_values still exact and complete).
- **StochasticProfile + hidden hero**: hero's hand is world-determined, so
  the BR separates exactly by hero's root hand — one hero-full-width solve
  per distinct hand group (hero's remaining hand is then node-determined
  within a group), values summed over groups. best_move/root_values are
  None/{} (the subgame root is root.me's decision, and root.me is then a
  profile seat); strategy keys are (hero_hand_mask, path).

`strategy` maps hero info sets REACHABLE WHILE HERO FOLLOWS IT (profile
seats free) to moves; hero=me keys are public tile paths from the subgame
root, root entry ().
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from hoyt.profiles import (
    SigmaTable,
    StochasticProfile,
    _DictProfileProvider,
    _TableProvider,
    _UniformProvider,
)
from hoyt.subgame import (
    DEFAULT_SLOT_BUDGET,
    KernelMemoryError,
    Subgame,
    backward,
    extract_strategy,
    run_engine,
)


def payoff_points() -> np.ndarray:
    """E[declaring points]: payoff43 = arange(43) (walt payoff='points')."""
    return np.arange(43, dtype=np.float64)


def payoff_make(bid_value: int) -> np.ndarray:
    """P(make): step at the bid line (walt payoff='make')."""
    return (np.arange(43) >= int(bid_value)).astype(np.float64)


@dataclass
class BRResult:
    value: float               # declaring orientation, /= total weight
    best_move: int | None      # None when hero doesn't act at the root
    root_values: dict          # move -> value, EVERY legal root move
    strategy: dict             # hero info set -> move (see module doc)
    n_nodes: int = 0
    meta: dict = field(default_factory=dict)


def _bits_asc(mask: int) -> list[int]:
    out = []
    m = int(mask)
    while m:
        b = m & -m
        out.append(b.bit_length() - 1)
        m ^= b
    return out


def _validate_table(sub: Subgame, table: SigmaTable) -> None:
    if table.root_key != sub.root_key:
        raise ValueError("SigmaTable was compiled for a different root")
    if table.worlds_digest != sub.worlds_digest:
        raise ValueError(
            "SigmaTable was compiled for a different world set "
            "(weights may differ freely; the worlds themselves may not)")


def _root_pick(sub: Subgame, waves, vals, sign):
    """walt's root tie rule verbatim: moves ascending, first strict
    sign*v > sign*best wins. Returns (value, best_move, root_values)."""
    t1 = waves[1]["tile"]
    best_val = None
    best_move = None
    root_values: dict[int, float] = {}
    for i in range(len(t1)):
        v = float(vals[i])
        root_values[int(t1[i])] = v / sub.total_w
        if best_val is None or sign * v > sign * best_val:
            best_val = v
            best_move = int(t1[i])
    return best_val / sub.total_w, int(best_move), root_values


def _leaf_vals(res: dict, payoff43: np.ndarray) -> np.ndarray:
    leaf = res["leaf"]
    wsum = np.bincount(leaf["snode"], weights=leaf["swt"],
                       minlength=leaf["M"])
    return payoff43[leaf["ptsvd"]] * wsum


def _finish_me(sub, waves, vals, sign, want_strategy, n_nodes, meta):
    vals, choice = backward(waves, vals, sign, collect_choice=want_strategy)
    value, best_move, root_values = _root_pick(sub, waves, vals, sign)
    strategy = extract_strategy(waves, choice, best_move, True) \
        if want_strategy else {}
    return BRResult(value=value, best_move=best_move,
                    root_values=root_values, strategy=strategy,
                    n_nodes=n_nodes, meta=meta)


def _provider_for(profile: StochasticProfile):
    if len(profile):
        return _DictProfileProvider(profile), True
    if not profile.uniform_fallback:
        raise ValueError("empty StochasticProfile without uniform_fallback "
                         "defines no behavior")
    return _UniformProvider(), False


def br_solve(subgame: Subgame, profile, payoff43, hero=None, *,
             rewalk: bool = False, want_strategy: bool = True,
             slot_budget: int = DEFAULT_SLOT_BUDGET) -> BRResult:
    """Exact best response of ``hero`` (default root.me); every non-hero
    seat plays ``profile``. Zero torch in the loop."""
    sub = subgame
    payoff43 = np.ascontiguousarray(payoff43, dtype=np.float64)
    if payoff43.shape != (43,):
        raise ValueError("payoff43 must be a length-43 leaf table")
    hero = sub.me if hero is None else int(hero)
    sign = 1.0 if hero % 2 == sub.bid_team else -1.0
    t0 = time.perf_counter()

    if isinstance(profile, SigmaTable):
        if hero != sub.me:
            raise NotImplementedError(
                "SigmaTable is domain-bound to the hero=root.me walk; a "
                "hidden seat's BR needs a StochasticProfile covering all "
                "four seats (CFR export)")
        _validate_table(sub, profile)
        if rewalk:
            res = run_engine(sub, _TableProvider(profile), hero=hero,
                             slot_budget=slot_budget)
            out = _finish_me(sub, res["waves"], _leaf_vals(res, payoff43),
                             sign, want_strategy, res["n_nodes"],
                             {"mode": "table-rewalk"})
        else:
            wsum = np.bincount(profile.leaf_node,
                               weights=sub.weights[profile.leaf_world],
                               minlength=profile.n_leaf_nodes)
            vals = payoff43[profile.leaf_pts] * wsum
            out = _finish_me(sub, profile.waves, vals, sign, want_strategy,
                             profile.n_nodes, {"mode": "table-fast"})
        out.meta["wall_ms"] = (time.perf_counter() - t0) * 1e3
        return out

    if not isinstance(profile, StochasticProfile):
        raise TypeError(f"unknown profile type {type(profile).__name__}")
    provider, needs_ids = _provider_for(profile)

    if hero == sub.me:
        try:
            res = run_engine(sub, provider, hero=hero,
                             need_path_ids=needs_ids,
                             slot_budget=slot_budget)
        except KernelMemoryError:
            out = _br_me_chunked(sub, provider, needs_ids, payoff43, sign,
                                 want_strategy, slot_budget)
            out.meta["wall_ms"] = (time.perf_counter() - t0) * 1e3
            return out
        out = _finish_me(sub, res["waves"], _leaf_vals(res, payoff43),
                         sign, want_strategy, res["n_nodes"],
                         {"mode": "stochastic"})
        out.meta["wall_ms"] = (time.perf_counter() - t0) * 1e3
        return out

    # ---- hidden hero: exact separation by hero's root hand --------------
    col = int(sub.col_arr[hero])
    if col < 0:
        raise ValueError(f"hero must be a seat 0..3, got {hero}")
    hands = sub.worlds[:, col]
    total = 0.0
    strategy: dict = {}
    n_nodes = 0
    n_groups = 0
    for hv in np.unique(hands):
        m = hands == hv
        res = run_engine(sub, provider, hero=hero,
                         worlds=sub.worlds_i64[m], weights=sub.weights[m],
                         heromask0=int(hv), need_path_ids=needs_ids,
                         slot_budget=slot_budget)
        vals, choice = backward(res["waves"], _leaf_vals(res, payoff43),
                                sign, collect_choice=want_strategy)
        total += float(vals.sum())          # root = profile node: sum
        if want_strategy:
            for k, v in extract_strategy(res["waves"], choice, None,
                                         False).items():
                strategy[(int(hv), k)] = v
        n_nodes += res["n_nodes"]
        n_groups += 1
    return BRResult(value=total / sub.total_w, best_move=None,
                    root_values={}, strategy=strategy, n_nodes=n_nodes,
                    meta={"mode": "hidden-hero", "n_groups": n_groups,
                          "wall_ms": (time.perf_counter() - t0) * 1e3})


def _br_me_chunked(sub, provider, needs_ids, payoff43, sign, want_strategy,
                   slot_budget) -> BRResult:
    """Stochastic-mode fallback when one tree blows the slot budget: solve
    a separate tree per legal root move (memory / #moves), keep the exact
    per-move values and the winner's strategy."""
    if sub.led0 >= 0:
        fb = int(sub.CFB[sub.led0])
        lm = sub.my0 & fb or sub.my0
    else:
        lm = sub.my0
    best_val = None
    best_move = None
    root_values: dict[int, float] = {}
    strategy: dict = {}
    n_nodes = 0
    for mv in _bits_asc(lm):
        res = run_engine(sub, provider, hero=sub.me,
                         need_path_ids=needs_ids, root_moves=1 << mv,
                         slot_budget=slot_budget)
        vals, choice = backward(res["waves"], _leaf_vals(res, payoff43),
                                sign, collect_choice=want_strategy)
        v = float(vals[0])                  # single wave-1 node: this move
        root_values[mv] = v / sub.total_w
        strat_mv = extract_strategy(res["waves"], choice, mv, True) \
            if want_strategy else {}
        n_nodes += res["n_nodes"]
        if best_val is None or sign * v > sign * best_val:
            best_val, best_move, strategy = v, mv, strat_mv
    return BRResult(value=best_val / sub.total_w, best_move=int(best_move),
                    root_values=root_values, strategy=strategy,
                    n_nodes=n_nodes, meta={"mode": "stochastic-chunked"})


def profile_value(subgame: Subgame, profile: StochasticProfile,
                  payoff43, slot_budget: int = DEFAULT_SLOT_BUDGET) -> float:
    """Expected payoff with ALL FOUR seats on ``profile`` (no best
    responder) — the CFR lane's baseline for gap = BR - value. Declaring
    orientation, same normalization as br_solve. ``slot_budget`` mirrors
    br_solve's knob: a root whose cfr_solve needed a raised budget needs
    the same headroom to value its exported profile."""
    sub = subgame
    payoff43 = np.ascontiguousarray(payoff43, dtype=np.float64)
    provider, needs_ids = _provider_for(profile)
    res = run_engine(sub, provider, hero=-1, need_path_ids=needs_ids,
                     slot_budget=slot_budget)
    vals, _ = backward(res["waves"], _leaf_vals(res, payoff43), 1.0)
    return float(vals.sum()) / sub.total_w
