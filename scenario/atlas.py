"""The atlas: fully known deals, their perfect-information game trees, and
Walt's honest imperfect-information evaluation written on their positions.

The mold is the shape; the pour is the labels.

The mold is the tree's shape — how many distinct positions each layer holds
(layer d = d tiles played), every legal play from every seat's real hand
followed, with transposition on the whole public record (`Record.key()`:
played, leader, open trick, banked, forbid). The terminal layer is left out
by default: it holds only scored ends, nothing to label. This plain search
is the referee for the compiled `CompiledWalt.mold`, which counts the
middle layers' hundreds of millions of positions.

The pour fills the mold along the line Walt actually plays. Walt takes all
four chairs (honest self-play: every seat samples its own scenarios at
every decision), and at each planned position the whole splat is kept —
for every legal move, the leaf value each of the decider's scenarios
reaches under the plan — not just the tile chosen. Deals advance in
lockstep waves, one play per wave, every planned decision of a wave in one
batch on all cores; a forced move (one legal tile) is applied unplanned
and labels nothing. The record of a position is its deal's first
`n_plays` plays, so the play sequences are the spine of the dataset and
every label hangs off one by a single integer.

Every leaf is the bidder team's final margin, an integer in [-42, 42],
under either utility — which is how `outcomes` is stored. The utility
shapes only the plan: `q` is in the planner's units (`scenario.planner`;
a make probability under `make`, a mean margin under `points`), and any
bid's make probability is a tail of the splat. The helpers below read a
position from the mover's side where a policy would: an advantage is
positive when the move is good for the seat to play.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field, fields
from functools import cached_property
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from forge.zeb.types import BidState, GamePhase, ZebGameState

from scenario.planner import HAND_TOTAL, Analysis
from scenario.record import Record, empty_record
from scenario.rules import ALL_TILES, mask_of, rules_for, tiles_of

LAST_LAYER = 27  # the final play's layer; layer 28 is the finished hand
N_PLAYS = 28

Deal = tuple[int, int, int, int]        # dealt hands, absolute seats
Plays = tuple[tuple[int, int], ...]     # (seat, tile) in play order


def mold(hands: Sequence[int], decl_id: int, leader: int, max_layer: int | None = None) -> list[int]:
    """Positions per layer, layers 0..=max_layer, for the deal `hands` (four
    tile masks, absolute seats) with `leader` to lead under `decl_id`."""
    _check_deal(hands)
    if not 0 <= leader < 4:
        raise ValueError("the leader is a seat in 0..4")
    last = LAST_LAYER if max_layer is None else max_layer
    if not 0 <= last <= 28:
        raise ValueError("a hand has layers 0..=28")
    rules = rules_for(decl_id)
    root = empty_record(leader)
    layer = {root.key(): root}
    counts = []
    for d in range(last + 1):
        counts.append(len(layer))
        if d == last:
            break
        nxt = {}
        for rec in layer.values():
            for t in tiles_of(rules.legal(hands[rec.seat] & ~rec.played, rec.led)):
                child = rec.advance(t, rules)
                nxt[child.key()] = child
        layer = nxt
    return counts


def _check_deal(hands: Sequence[int]) -> None:
    dealt = len(hands) == 4 and hands[0] | hands[1] | hands[2] | hands[3] == ALL_TILES
    if not dealt or any(h.bit_count() != 7 for h in hands):
        raise ValueError("a deal is four disjoint seven-tile hands")


# -- deals ---------------------------------------------------------------------------


def deal(seed: int) -> Deal:
    """The seeded deal: `random.Random(seed)` shuffles the 28 tiles, seven
    to a seat in seat order."""
    return _hands(_shuffled(seed))


def fiber(seed: int, n: int, seat: int = 0) -> list[Deal]:
    """`n` deals from one hand's fiber: `seat` keeps its hand from
    `deal(seed)`, and the other 21 tiles are reshuffled among the other
    seats for each deal (`random.Random(31·seed)`, one stream for all n)."""
    held = deal(seed)[seat]
    rest = [t for t in _shuffled(seed) if not (held >> t) & 1]
    rng = random.Random(seed * 31)
    out = []
    for _ in range(n):
        o = rest[:]
        rng.shuffle(o)
        out.append(_hands(o[:7 * seat] + tiles_of(held) + o[7 * seat:]))
    return out


def _shuffled(seed: int) -> list[int]:
    tiles = list(range(28))
    random.Random(seed).shuffle(tiles)
    return tiles


def _hands(tiles: Sequence[int]) -> Deal:
    m = [mask_of(tiles[7 * s:7 * s + 7]) for s in range(4)]
    return (m[0], m[1], m[2], m[3])


# -- the wave engine -----------------------------------------------------------------


@dataclass(frozen=True)
class Decision:
    """A planned position of one deal, opened up by the planner."""

    deal: int
    plays: Plays                 # the plays before it; the record is their replay
    analysis: Analysis


@dataclass(frozen=True)
class Hand:
    """One deal played out."""

    plays: Plays                 # all 28, in play order
    banked: tuple[int, int]      # (team 0, team 1) final points


def play_waves(
    deals: Sequence[Sequence[int]], bidder: int, bid: int, decl_ids: Sequence[int], walt,
    on_wave: Callable[[int, list[Decision]], None] | None = None,
) -> list[Hand]:
    """Play every deal to the end with `walt` (a `CompiledWalt`) in all four
    chairs, the bidder leading, in lockstep: wave w makes play w of every
    deal. A forced move is applied without asking; the rest of a wave is
    one `decide_batch` — or, when `on_wave` wants the splats, one
    `analyze_batch`, handed to `on_wave(w, decisions)`. Deterministic: each
    decision is a pure function of Walt's config and the position."""
    if len(deals) != len(decl_ids):
        raise ValueError("one declaration per deal")
    for d in deals:
        _check_deal(d)
    held = [tuple(tuple(tiles_of(h)) for h in d) for d in deals]
    rules = [rules_for(x) for x in decl_ids]
    recs = [empty_record(bidder)] * len(deals)
    plays: list[Plays] = [()] * len(deals)
    for wave in range(N_PLAYS):
        asks: list[int] = []
        for i, rec in enumerate(recs):
            legal = tiles_of(rules[i].legal(deals[i][rec.seat] & ~rec.played, rec.led))
            if len(legal) == 1:
                plays[i] += ((rec.seat, legal[0]),)
                recs[i] = rec.advance(legal[0], rules[i])
            else:
                asks.append(i)
        if not asks:
            continue
        states = [_state(held[i], bidder, bid, decl_ids[i], plays[i], recs[i]) for i in asks]
        if on_wave is None:
            tiles = [t for t, _ in walt.decide_batch(states, [bid] * len(asks))]
        else:
            analyses = walt.analyze_batch(states, [bid] * len(asks))
            on_wave(wave, [Decision(i, plays[i], a) for i, a in zip(asks, analyses)])
            tiles = [a.tile for a in analyses]
        for i, t in zip(asks, tiles):
            plays[i] += ((recs[i].seat, t),)
            recs[i] = recs[i].advance(t, rules[i])
    return [Hand(p, r.banked) for p, r in zip(plays, recs)]


def _state(hands, bidder: int, bid: int, decl_id: int, plays: Plays, rec: Record):
    """The engine's PLAYING state at a record, for the planner's interface."""
    return ZebGameState(
        hands=hands, dealer=(bidder + 3) % 4, phase=GamePhase.PLAYING,
        bid_state=BidState(tuple(bid if s == bidder else 0 for s in range(4)), bidder, bid),
        decl_id=decl_id, bidder=bidder, played=frozenset(t for _, t in plays),
        play_history=plays, current_trick=rec.trick, trick_leader=rec.leader,
        team_points=rec.banked,
    )


# -- the pour ------------------------------------------------------------------------


@dataclass(frozen=True)
class Position:
    """One planned position of a pour, with its whole splat. Arrays are
    indexed by action (in `actions` order, ascending tile id) and by
    scenario (the decider's sample order); `q[a]` is the planner's value,
    Σ_r weights[r]·u(outcomes[a, r]) / Σ_r weights[r] with u `in_plan_units`."""

    deal: int
    plays: Plays                 # the plays before this position
    hands: Deal                  # the real deal, which the decider did not see
    decl_id: int
    bidder: int
    bid: int
    utility: str
    actions: tuple[int, ...]
    q: np.ndarray                # [A] float64
    outcomes: np.ndarray         # [A, W] int8, bidder-team final margin
    worlds: np.ndarray           # [W, 4] uint32, remaining hands per scenario
    weights: np.ndarray          # [W] float64, the belief, unnormalized
    chosen: int                  # the tile Walt played here
    banked: tuple[int, int]      # how the deal really ended

    @property
    def viewer(self) -> int:
        return self.record.seat

    @property
    def own(self) -> int:
        return self.hands[self.viewer] & ~self.record.played

    @cached_property
    def record(self) -> Record:
        rules = rules_for(self.decl_id)
        rec = empty_record(self.bidder)
        for _, t in self.plays:
            rec = rec.advance(t, rules)
        return rec

    @property
    def sign(self) -> int:
        """+1 when the decider is on the bidding side (it maximizes the
        bidder-team value), -1 for a defender."""
        return 1 if self.viewer % 2 == self.bidder % 2 else -1


@dataclass(frozen=True)
class Pour:
    """The advantage dataset: columns, ragged by position through counts.

    Per deal (D): `hands`, `decl`, `plays` [D, 28, 2] (seat, tile), the real
    ending `banked`. Per position (P): its `deal`, `n_plays` (its record is
    that deal's first n_plays plays, and play n_plays is its chosen tile),
    `n_actions`, `n_worlds`. Then flat: `actions` and `q` (Σ n_actions),
    `worlds` and `weights` (Σ n_worlds), `outcomes` (Σ n_actions·n_worlds,
    action-major within a position)."""

    bidder: int
    bid: int
    utility: str
    hands: np.ndarray = field(repr=False)       # [D, 4] uint32
    decl: np.ndarray = field(repr=False)        # [D] int8
    plays: np.ndarray = field(repr=False)       # [D, 28, 2] int8
    banked: np.ndarray = field(repr=False)      # [D, 2] int8
    deal: np.ndarray = field(repr=False)        # [P] int32
    n_plays: np.ndarray = field(repr=False)     # [P] int8
    n_actions: np.ndarray = field(repr=False)   # [P] int8
    n_worlds: np.ndarray = field(repr=False)    # [P] int32
    actions: np.ndarray = field(repr=False)     # [A] int8
    q: np.ndarray = field(repr=False)           # [A] float64
    worlds: np.ndarray = field(repr=False)      # [W, 4] uint32
    weights: np.ndarray = field(repr=False)     # [W] float64
    outcomes: np.ndarray = field(repr=False)    # [O] int8

    def __len__(self) -> int:
        return len(self.deal)

    @cached_property
    def _offsets(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        a, w = self.n_actions.astype(np.int64), self.n_worlds.astype(np.int64)
        return tuple(np.concatenate(([0], np.cumsum(x))) for x in (a, w, a * w))

    def position(self, p: int) -> Position:
        ao, wo, oo = (int(o[p]) for o in self._offsets)
        na, nw, d, k = int(self.n_actions[p]), int(self.n_worlds[p]), int(self.deal[p]), int(self.n_plays[p])
        plays = tuple((int(s), int(t)) for s, t in self.plays[d])
        return Position(
            deal=d, plays=plays[:k], hands=tuple(int(h) for h in self.hands[d]),  # type: ignore[arg-type]
            decl_id=int(self.decl[d]), bidder=self.bidder, bid=self.bid, utility=self.utility,
            actions=tuple(int(t) for t in self.actions[ao:ao + na]), q=self.q[ao:ao + na],
            outcomes=self.outcomes[oo:oo + na * nw].reshape(na, nw), worlds=self.worlds[wo:wo + nw],
            weights=self.weights[wo:wo + nw], chosen=plays[k][1],
            banked=(int(self.banked[d, 0]), int(self.banked[d, 1])),
        )

    def positions(self):
        return (self.position(p) for p in range(len(self)))


_COLUMNS = tuple(f.name for f in fields(Pour) if f.name not in ("bidder", "bid", "utility"))


def pour(
    deals: Sequence[Sequence[int]], bidder: int, bid: int, declare: Callable[[int], int], walt,
    progress: Callable[[int, int], None] | None = None,
) -> Pour:
    """Play `deals` with `play_waves` and keep every planned position's
    splat. `declare(hand) -> decl_id` is the bidder's declaring rule (the
    arena's is `best_trump`; the scenario package does not import the
    arena, so the caller passes it). `progress(wave, positions so far)` is
    called after each planned wave."""
    decls = [declare(d[bidder]) for d in deals]
    at: list[tuple[int, int]] = []
    acts: list[int] = []
    qs: list[float] = []
    nas: list[int] = []
    nws: list[int] = []
    worlds: list[Sequence[int]] = []
    weights: list[float] = []
    outs: list[np.ndarray] = []

    def keep(wave: int, decisions: list[Decision]) -> None:
        for dec in decisions:
            a = dec.analysis
            order = sorted(a.q)
            at.append((dec.deal, len(dec.plays)))
            acts.extend(order)
            qs.extend(a.q[t] for t in order)
            nas.append(len(order))
            nws.append(len(a.worlds))
            worlds.extend(a.worlds)
            weights.extend(a.weights)
            outs.append(_integral(np.array([a.outcomes[t] for t in order], dtype=np.float64)))
        if progress is not None:
            progress(wave, len(at))

    hands = play_waves(deals, bidder, bid, decls, walt, keep)
    return Pour(
        bidder=bidder, bid=bid, utility=walt.config.utility,
        hands=np.array(deals, dtype=np.uint32).reshape(-1, 4),
        decl=np.array(decls, dtype=np.int8),
        plays=np.array([h.plays for h in hands], dtype=np.int8).reshape(-1, N_PLAYS, 2),
        banked=np.array([h.banked for h in hands], dtype=np.int8).reshape(-1, 2),
        deal=np.array([d for d, _ in at], dtype=np.int32),
        n_plays=np.array([k for _, k in at], dtype=np.int8),
        n_actions=np.array(nas, dtype=np.int8), n_worlds=np.array(nws, dtype=np.int32),
        actions=np.array(acts, dtype=np.int8), q=np.array(qs, dtype=np.float64),
        worlds=np.array(worlds, dtype=np.uint32).reshape(-1, 4),
        weights=np.array(weights, dtype=np.float64),
        outcomes=np.concatenate([o.ravel() for o in outs]) if outs else np.zeros(0, np.int8),
    )


def _integral(values: np.ndarray) -> np.ndarray:
    """Leaf margins as int8."""
    out = np.rint(values)
    if not np.array_equal(out, values) or np.abs(out).max(initial=0) > HAND_TOTAL:
        raise AssertionError("a leaf value is not an integer in [-42, 42]")
    return out.astype(np.int8)


def save(p: Pour, out: str | Path, **about) -> None:
    """`out/pour.npz` (the columns, compressed) and `out/manifest.json`
    (the scalars the columns are read under, the totals, and `about`)."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "pour.npz", **{c: getattr(p, c) for c in _COLUMNS})
    manifest = {
        "bidder": p.bidder, "bid": p.bid, "utility": p.utility,
        "deals": len(p.hands), "positions": len(p), "forced": N_PLAYS * len(p.hands) - len(p),
        "actions": len(p.actions), "worlds": len(p.worlds), **about,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def load(out: str | Path) -> Pour:
    out = Path(out)
    m = json.loads((out / "manifest.json").read_text())
    with np.load(out / "pour.npz") as z:
        return Pour(m["bidder"], m["bid"], m["utility"], **{c: z[c] for c in _COLUMNS})


# -- reading a position --------------------------------------------------------------


def advantages(pos: Position) -> dict[int, float]:
    """Each move's value minus the mean over the legal moves (the group), in
    the mover's units: positive is good for the seat to play. In the plan's
    units: make probability under `make`, margin under `points`."""
    return dict(zip(pos.actions, (pos.sign * (pos.q - pos.q.mean())).tolist()))


def paired_advantages(pos: Position) -> np.ndarray:
    """[A, W]: in each scenario, each move's leaf minus the moves' mean leaf,
    in the mover's units. Its belief-weighted mean over scenarios is
    `advantages`; the pairing removes the deal's luck, which every move in
    a scenario shares."""
    o = pos.outcomes.astype(np.float64)
    return pos.sign * (o - o.mean(axis=0))


def in_plan_units(pos: Position, margins: np.ndarray) -> np.ndarray:
    """The utility the plan scored these margins with: the make indicator
    under `make`, the margin itself under `points`. `q` is the belief-weighted
    mean of this over each action's splat."""
    if pos.utility == "make":
        return (margins >= 2 * pos.bid - HAND_TOTAL).astype(np.float64)
    return margins.astype(np.float64)


def makes(pos: Position, action: int, bid: int | None = None) -> np.ndarray:
    """[W] bool: the scenarios in which `action` makes `bid` (the contract
    by default) for the bidding side — a tail of the margin splat."""
    o = pos.outcomes[pos.actions.index(action)]
    return o >= 2 * (pos.bid if bid is None else bid) - HAND_TOTAL


def success_set(pos: Position, action: int, bid: int | None = None) -> np.ndarray:
    """Indices of the scenarios in which `action` makes the bid."""
    return np.flatnonzero(makes(pos, action, bid))


def holding_predictors(
    pos: Position, subset: Sequence[int],
) -> tuple[float, dict[tuple[int, int], tuple[float, float]]]:
    """What must be true for `subset` (scenario indices, e.g. a success
    set): the base rate P(subset), and for each (seat, tile) hidden from the
    decider that some scenario puts there, (P(subset | seat holds tile),
    P(seat holds tile)) — all belief-weighted."""
    w = pos.weights
    inside = np.zeros(len(w), dtype=bool)
    inside[np.asarray(subset, dtype=np.int64)] = True
    total = w.sum()
    hidden = ALL_TILES & ~pos.record.played & ~pos.own
    out = {}
    for seat in range(4):
        if seat == pos.viewer:
            continue
        for t in tiles_of(hidden):
            holds = (pos.worlds[:, seat] >> np.uint32(t)) & 1 == 1
            mass = w[holds].sum()
            if mass > 0:
                out[(seat, t)] = (float(w[holds & inside].sum() / mass), float(mass / total))
    return float(w[inside].sum() / total), out


@dataclass(frozen=True)
class Calibration:
    """The bidder's opening decision, forecast against the real endings."""

    n: int
    forecast_margin: float           # mean belief-weighted margin of the chosen lead's splat
    real_margin: float               # mean bidder-team final margin
    forecast_make: float             # mean belief mass in which the chosen lead makes
    real_make: float                 # share of deals the bidding side made


def calibration(p: Pour) -> Calibration:
    opening = [pos for pos in p.positions() if not pos.plays]
    if not opening:
        raise ValueError("the pour holds no opening decisions")
    bt = p.bidder % 2
    fm = [float(pos.weights @ pos.outcomes[pos.actions.index(pos.chosen)] / pos.weights.sum())
          for pos in opening]
    fk = [float(pos.weights[makes(pos, pos.chosen)].sum() / pos.weights.sum()) for pos in opening]
    banked = p.banked.astype(np.int64)
    margin = banked[:, bt] - banked[:, 1 - bt]
    return Calibration(
        n=len(opening),
        forecast_margin=float(np.mean(fm)),
        real_margin=float(margin.mean()),
        forecast_make=float(np.mean(fk)),
        real_make=float(np.mean(banked[:, bt] >= p.bid)),
    )


def calibration_table(c: Calibration) -> str:
    fm = f"{c.forecast_margin:+.1f}"
    return "\n".join([
        f"calibration, the bidder's opening lead ({c.n} deals)",
        "                 forecast     real",
        f"  mean margin    {fm:>8}  {c.real_margin:+7.1f}",
        f"  make share     {c.forecast_make:>8.0%}  {c.real_make:>7.0%}",
    ])


# -- the command line ----------------------------------------------------------------


def _main(argv: Sequence[str] | None = None) -> None:
    import argparse
    import threading
    import time

    from arena.walt_play import _declared_trump, parse_walt_spec
    from scenario.compiled import CompiledWalt
    from scenario.planner import Config, Rung

    ap = argparse.ArgumentParser(prog="python -u -m scenario.atlas")
    sub = ap.add_subparsers(dest="cmd", required=True)
    pp = sub.add_parser("pour", help="Walt self-play, every planned position's splat kept")
    pp.add_argument("--deals", type=int, required=True)
    pp.add_argument("--spec", default="k1,n512,m8", help="a walt spec, as the arena reads it")
    pp.add_argument("--bid", type=int, default=30)
    pp.add_argument("--bidder", type=int, default=0)
    pp.add_argument("--seed", type=int, default=1, help="deal i is deal(seed + i)")
    pp.add_argument("--fiber", action="store_true",
                    help="hold the bidder's hand of deal(seed); reshuffle the other 21")
    pp.add_argument("--out", required=True)
    mp = sub.add_parser("mold", help="positions per layer of deal(seed)")
    mp.add_argument("--seed", type=int, default=1)
    mp.add_argument("--bidder", type=int, default=0)
    mp.add_argument("--max-layer", type=int, default=None)
    args = ap.parse_args(argv)

    t0 = time.time()
    if args.cmd == "mold":
        hands = deal(args.seed)
        decl = _declared_trump(hands[args.bidder])
        counts = CompiledWalt(Config((Rung(1, None),))).mold(hands, decl, args.bidder, args.max_layer)
        print(f"mold of deal({args.seed}), seat {args.bidder} leads on decl {decl}")
        for d, c in enumerate(counts):
            print(f"  layer {d:2d}: {c:>14,}")
        print(f"  total   : {sum(counts):>14,}   ({time.time() - t0:.1f}s)")
        return

    deals = (fiber(args.seed, args.deals, args.bidder) if args.fiber
             else [deal(args.seed + i) for i in range(args.deals)])
    walt = parse_walt_spec(args.spec).walt
    if not isinstance(walt, CompiledWalt):
        raise SystemExit("the pour plays in batches; it needs the compiled core (drop `py`)")
    print(f"pour: {args.deals} {'fiber ' if args.fiber else ''}deals from seed {args.seed}, "
          f"walt {args.spec} ({walt.config.utility}), seat {args.bidder} bids {args.bid}", flush=True)
    status = {"wave": -1, "positions": 0}
    done = threading.Event()

    def heartbeat() -> None:
        while not done.wait(30.0):
            print(f"  ... wave {status['wave'] + 1} planning, {status['positions']} positions kept, "
                  f"{time.time() - t0:.0f}s", flush=True)

    def progress(wave: int, positions: int) -> None:
        print(f"  wave {wave:2d}: {positions - status['positions']:4d} planned, "
              f"{positions} kept, {time.time() - t0:.0f}s", flush=True)
        status.update(wave=wave, positions=positions)

    threading.Thread(target=heartbeat, daemon=True).start()
    try:
        p = pour(deals, args.bidder, args.bid, _declared_trump, walt, progress)
    finally:
        done.set()
    wall = time.time() - t0
    save(p, args.out, spec=args.spec, seed=args.seed, fiber=args.fiber, commit=_git_head(),
         deal_seeds=None if args.fiber else [args.seed + i for i in range(args.deals)],
         wall_seconds=round(wall, 1))
    size = (Path(args.out) / "pour.npz").stat().st_size
    print(f"poured {len(p)} positions ({N_PLAYS * len(p.hands) - len(p)} forced) from "
          f"{len(p.hands)} deals in {wall:.0f}s -> {args.out}/pour.npz ({size / 1e6:.1f} MB)")
    print(calibration_table(calibration(p)))


def _git_head() -> str:
    """The commit the code is at, `-dirty` when tracked files differ from it."""
    import subprocess

    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=Path(__file__).parent, capture_output=True,
                              text=True, check=True).stdout.strip()

    return git("rev-parse", "HEAD") + ("-dirty" if git("status", "--porcelain", "--untracked-files=no") else "")


if __name__ == "__main__":
    _main()
