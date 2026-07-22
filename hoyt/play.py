"""Play a late-hand continuation against bounded full-Metal Hoyt.

This is deliberately a *player experiment*, not a new reference claim.  Each
seat independently re-solves from its own information set: its remaining hand,
the public auction, and the public play history.  The real hidden deal is used
only by the zeb referee to apply actions; it never enters ``EndgameRoot`` or the
solver seed.

The default table starts at H5 after Jud has supplied a deterministic opening
prefix.  From there a human controls the current leader and the other three
seats use sparse external-sampling CFR on Metal.  A JSON session in ``scratch``
makes the table resumable and lets a conversation drive one human move at a
time without exposing the other hands.

Honesty line: action intervals below cover independent finite-sample
evaluation of the fixed candidate only.  They do not cover CFR optimization
shortfall, do not certify equilibrium, and do not make the physical-uniform
belief behavior-conditioned on earlier actions.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Callable, Sequence

from arena.play import PlayPolicy, RandomPlay
from forge.oracle.declarations import DECL_ID_TO_NAME
from forge.oracle.tables import DOMINOES, resolve_trick
from forge.zeb.game import (
    apply_action,
    current_player,
    is_terminal,
    legal_actions,
    new_game,
)
from forge.zeb.types import BidState, GamePhase, ZebGameState
from hoyt.br import payoff_make, payoff_points
from hoyt.sampled_br import SampledCFR
from walt.contracts import EndgameRoot

__all__ = [
    "HoytDecision",
    "HoytPlay",
    "SolveBudget",
    "TableSession",
    "advance_to_horizon",
    "format_domino",
    "load_session",
    "render_position",
    "root_from_state",
    "save_session",
]

_SEAT_NAMES = ("North", "East", "South", "West")
_SESSION_VERSION = 1


@dataclass(frozen=True)
class SolveBudget:
    """One registered amount of work for an acting-seat re-solve."""

    epochs: int
    train_batch: int
    eval_batches: int
    eval_batch: int
    capacity: int = 1 << 20


_DEFAULT_BUDGETS = {
    # H5/H6 are the measured full-Metal shapes.  Smaller horizons retain the
    # H5 budget but become rapidly cheaper as the remaining tree contracts.
    6: SolveBudget(epochs=50, train_batch=32,
                   eval_batches=8, eval_batch=128),
    5: SolveBudget(epochs=100, train_batch=64,
                   eval_batches=8, eval_batch=256),
    4: SolveBudget(epochs=100, train_batch=64,
                   eval_batches=8, eval_batch=256),
    3: SolveBudget(epochs=120, train_batch=64,
                   eval_batches=8, eval_batch=256),
    2: SolveBudget(epochs=120, train_batch=64,
                   eval_batches=8, eval_batch=256),
    1: SolveBudget(epochs=0, train_batch=1,
                   eval_batches=1, eval_batch=2),
}


@dataclass(frozen=True)
class HoytDecision:
    """Public receipt for one acting-seat decision.

    ``margin`` and ``margin_error`` are in the selected utility's units:
    declaring points for ``points`` and probability for ``make``.  The error
    is the sum of the simultaneous empirical-Bernstein radii on the top two
    candidate actions.  ``separated`` therefore means the fixed-candidate
    evaluation bands do not overlap; it is not an optimization certificate.
    """

    seat: int
    horizon: int
    action: int                 # zeb slot index
    domino: int                 # public domino id
    utility: str
    value: float | None
    value_error: float | None
    margin: float | None
    margin_error: float | None
    separated: bool | None
    n_worlds: int | None
    samples: int
    rows: int
    peak_frontier: int
    wall_seconds: float
    forced: bool = False


def _remaining(state: ZebGameState, seat: int) -> tuple[int, ...]:
    return tuple(d for d in state.hands[seat] if d not in state.played)


def root_from_state(state: ZebGameState) -> EndgameRoot:
    """Project a referee state to exactly what the acting seat may know."""
    if state.phase != GamePhase.PLAYING:
        raise ValueError("Hoyt can only solve a playing state")
    me = current_player(state)
    return EndgameRoot(
        decl_id=int(state.decl_id),
        bidder=int(state.bidder),
        bid_value=int(state.bid_state.high_bid),
        bids=tuple(int(x) for x in state.bid_state.bids),
        dealer=int(state.dealer),
        me=me,
        my_hand=tuple(sorted(_remaining(state, me))),
        play_history=tuple((int(s), int(d)) for s, d in state.play_history),
        trick_leader=int(state.trick_leader),
        current_trick=tuple(int(d) for d in state.current_trick),
        team_points=tuple(int(x) for x in state.team_points),
    )


def _decision_seed(root: EndgameRoot, base_seed: int) -> int:
    """Stable seed derived only from the acting seat's information state."""
    payload = repr((int(base_seed), root)).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(),
                          "little") & 0x7FFF_FFFF


class HoytPlay(PlayPolicy):
    """Arena-shaped late-hand player backed by a fresh sampled-CFR re-solve."""

    def __init__(
        self,
        *,
        utility: str = "make",
        seed: int = 0,
        budgets: dict[int, SolveBudget] | None = None,
    ):
        if utility not in ("make", "points"):
            raise ValueError("utility must be 'make' or 'points'")
        self.utility = utility
        self.seed = int(seed)
        self.budgets = dict(_DEFAULT_BUDGETS if budgets is None else budgets)
        self.last_decisions: list[HoytDecision] = []

    def _budget(self, horizon: int) -> SolveBudget:
        if horizon not in self.budgets:
            raise ValueError(
                f"no playable full-Metal budget for H{horizon}; "
                "start the table at H5 or H6")
        return self.budgets[horizon]

    def _choose_one(self, state: ZebGameState) -> HoytDecision:
        legal = tuple(int(a) for a in legal_actions(state))
        seat = current_player(state)
        horizon = len(_remaining(state, seat))
        if len(legal) == 1:
            action = legal[0]
            return HoytDecision(
                seat=seat, horizon=horizon, action=action,
                domino=int(state.hands[seat][action]), utility=self.utility,
                value=None, value_error=None, margin=None, margin_error=None,
                separated=None, n_worlds=None, samples=0, rows=0,
                peak_frontier=0, wall_seconds=0.0, forced=True)

        budget = self._budget(horizon)
        root = root_from_state(state)
        payoff = (payoff_make(root.bid_value) if self.utility == "make"
                  else payoff_points())
        solve_seed = _decision_seed(root, self.seed)
        t0 = time.perf_counter()
        solver = SampledCFR(root, payoff, capacity=budget.capacity)
        solver.train(budget.epochs, budget.train_batch, seed=solve_seed)
        result = solver.evaluate(
            budget.eval_batches, budget.eval_batch,
            seed=solve_seed + 10_000_000,
        )
        wall = time.perf_counter() - t0
        if result.best_move is None:
            raise RuntimeError("sampled CFR returned no root move")

        domino = int(result.best_move)
        try:
            action = tuple(state.hands[seat]).index(domino)
        except ValueError as exc:  # pragma: no cover - kernel invariant
            raise AssertionError("Hoyt returned a tile outside the acting hand") \
                from exc
        if action not in legal:  # pragma: no cover - kernel invariant
            raise AssertionError("Hoyt returned a follow-suit-illegal tile")

        sign = 1.0 if seat % 2 == state.bidder % 2 else -1.0
        ranked = sorted(
            ((sign * float(v), int(move))
             for move, v in result.root_values.items()),
            key=lambda pair: (-pair[0], pair[1]),
        )
        best_oriented, best_move = ranked[0]
        if best_move != domino:
            raise AssertionError("root value ranking disagrees with best_move")
        value = float(result.root_values[domino])
        value_error = float(result.root_value_error[domino])
        if len(ranked) > 1:
            second_oriented, second_move = ranked[1]
            margin = best_oriented - second_oriented
            margin_error = value_error \
                + float(result.root_value_error[second_move])
            separated = bool(margin > margin_error)
        else:  # legal length >1 above, retained as a defensive invariant
            margin = None
            margin_error = None
            separated = None

        return HoytDecision(
            seat=seat, horizon=horizon, action=action, domino=domino,
            utility=self.utility, value=value, value_error=value_error,
            margin=margin, margin_error=margin_error, separated=separated,
            n_worlds=int(solver.sampler.n_worlds), samples=int(result.samples),
            rows=int(result.occupied_buckets),
            peak_frontier=int(result.peak_frontier), wall_seconds=wall)

    def choose(
        self,
        states: Sequence[ZebGameState],
        bid_values: Sequence[int],
        marks: Sequence[tuple[int, int]] | None = None,
        marks_to_win: int = 7,
    ) -> list[int]:
        del marks, marks_to_win
        if len(states) != len(bid_values):
            raise ValueError("states and bid_values must have equal length")
        decisions = [self._choose_one(state) for state in states]
        self.last_decisions = decisions
        return [d.action for d in decisions]

    def __repr__(self) -> str:
        return f"HoytPlay(utility={self.utility!r})"


@dataclass
class TableSession:
    state: ZebGameState
    human_seat: int
    seed: int
    start_horizon: int
    prefix: str
    utility: str = "make"
    bot_seed: int = 42


def advance_to_horizon(
    seed: int,
    horizon: int,
    prefix_policy: PlayPolicy,
) -> ZebGameState:
    """Play a public prefix until the next mover holds ``horizon`` tiles."""
    if horizon not in (5, 6):
        raise ValueError("the playable entry horizon must be H5 or H6")
    state = new_game(int(seed))
    while not is_terminal(state):
        mover = current_player(state)
        if len(_remaining(state, mover)) <= horizon:
            return state
        action = prefix_policy.choose(
            [state], [int(state.bid_state.high_bid)])[0]
        state = apply_action(state, int(action))
    raise RuntimeError("hand ended before the requested entry horizon")


def _state_to_dict(state: ZebGameState) -> dict:
    return {
        "hands": [list(map(int, h)) for h in state.hands],
        "dealer": int(state.dealer),
        "phase": int(state.phase),
        "bid_state": {
            "bids": list(map(int, state.bid_state.bids)),
            "high_bidder": int(state.bid_state.high_bidder),
            "high_bid": int(state.bid_state.high_bid),
        },
        "decl_id": int(state.decl_id),
        "bidder": int(state.bidder),
        "played": sorted(map(int, state.played)),
        "play_history": [list(map(int, x)) for x in state.play_history],
        "current_trick": list(map(int, state.current_trick)),
        "trick_leader": int(state.trick_leader),
        "team_points": list(map(int, state.team_points)),
    }


def _state_from_dict(data: dict) -> ZebGameState:
    bid = data["bid_state"]
    return ZebGameState(
        hands=tuple(tuple(map(int, h)) for h in data["hands"]),
        dealer=int(data["dealer"]),
        phase=GamePhase(int(data["phase"])),
        bid_state=BidState(
            bids=tuple(map(int, bid["bids"])),
            high_bidder=int(bid["high_bidder"]),
            high_bid=int(bid["high_bid"]),
        ),
        decl_id=int(data["decl_id"]),
        bidder=int(data["bidder"]),
        played=frozenset(map(int, data["played"])),
        play_history=tuple(tuple(map(int, x))
                           for x in data["play_history"]),
        current_trick=tuple(map(int, data["current_trick"])),
        trick_leader=int(data["trick_leader"]),
        team_points=tuple(map(int, data["team_points"])),
    )


def save_session(session: TableSession, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": _SESSION_VERSION,
        "human_seat": int(session.human_seat),
        "seed": int(session.seed),
        "start_horizon": int(session.start_horizon),
        "prefix": session.prefix,
        "utility": session.utility,
        "bot_seed": int(session.bot_seed),
        "state": _state_to_dict(session.state),
    }
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    tmp.replace(target)


def load_session(path: str | Path) -> TableSession:
    payload = json.loads(Path(path).read_text())
    if int(payload.get("format_version", -1)) != _SESSION_VERSION:
        raise ValueError("unsupported full-Metal table session version")
    return TableSession(
        state=_state_from_dict(payload["state"]),
        human_seat=int(payload["human_seat"]),
        seed=int(payload["seed"]),
        start_horizon=int(payload["start_horizon"]),
        prefix=str(payload["prefix"]),
        utility=str(payload.get("utility", "make")),
        bot_seed=int(payload.get("bot_seed", 42)),
    )


def format_domino(domino: int) -> str:
    high, low = DOMINOES[int(domino)]
    return f"{high}-{low}"


def _parse_domino(text: str) -> tuple[int, int]:
    clean = text.strip().lower().replace("domino", "").strip()
    if clean.isdigit() and len(clean) == 1:
        return -1, int(clean)  # displayed choice number, resolved by caller
    clean = clean.replace("/", "-").replace(" ", "-")
    if "-" not in clean and len(clean) == 2 and clean.isdigit():
        clean = f"{clean[0]}-{clean[1]}"
    parts = [p for p in clean.split("-") if p]
    if len(parts) != 2 or not all(p.isdigit() for p in parts):
        raise ValueError("enter a choice number or a tile such as 6-5")
    a, b = map(int, parts)
    if not (0 <= a <= 6 and 0 <= b <= 6):
        raise ValueError("domino pips must be between 0 and 6")
    high, low = max(a, b), min(a, b)
    return high, low


def _action_from_text(state: ZebGameState, text: str) -> int:
    legal = tuple(map(int, legal_actions(state)))
    high, low = _parse_domino(text)
    if high == -1:
        choice = low
        if not 1 <= choice <= len(legal):
            raise ValueError(f"choice must be between 1 and {len(legal)}")
        return legal[choice - 1]
    wanted = (high, low)
    seat = current_player(state)
    for action in legal:
        if DOMINOES[state.hands[seat][action]] == wanted:
            return action
    raise ValueError(f"{high}-{low} is not a legal tile here")


def _trick_lines(state: ZebGameState) -> list[str]:
    lines = []
    history = list(state.play_history)
    for off in range(0, len(history), 4):
        plays = history[off:off + 4]
        labels = "  ".join(
            f"{_SEAT_NAMES[seat]} {format_domino(tile)}"
            for seat, tile in plays)
        if len(plays) == 4:
            leader = int(plays[0][0])
            tiles = tuple(int(tile) for _, tile in plays)
            outcome = resolve_trick(tiles[0], tiles, state.decl_id)
            winner = (leader + int(outcome.winner_offset)) % 4
            lines.append(
                f"Trick {off // 4 + 1}: {labels}  -> "
                f"{_SEAT_NAMES[winner]}, {int(outcome.points)} point(s)")
        else:
            lines.append(f"Current trick: {labels}")
    return lines


def render_position(session: TableSession) -> str:
    """Render only public state plus the human seat's remaining hand."""
    state = session.state
    human = session.human_seat
    bidder = int(state.bidder)
    declaring = bidder % 2
    role = "declaring team" if human % 2 == declaring else "defending team"
    partner = (human + 2) % 4
    lines = [
        f"Full-Metal Hoyt table — seed {session.seed}",
        f"You are {_SEAT_NAMES[human]} ({role}); "
        f"your partner is {_SEAT_NAMES[partner]}.",
        f"{_SEAT_NAMES[bidder]} bid {state.bid_state.high_bid}; "
        f"trump is {DECL_ID_TO_NAME[state.decl_id]}.",
        f"Banked points: team North/South {state.team_points[0]}, "
        f"team East/West {state.team_points[1]}.",
    ]
    trick_lines = _trick_lines(state)
    if trick_lines:
        lines.extend(["", *trick_lines])
    if is_terminal(state):
        made = state.team_points[declaring] >= state.bid_state.high_bid
        lines.extend([
            "",
            f"Final: North/South {state.team_points[0]}, "
            f"East/West {state.team_points[1]}. "
            f"The bid was {'made' if made else 'set'}.",
        ])
        return "\n".join(lines)
    if current_player(state) != human:
        lines.extend(["", f"Waiting for {_SEAT_NAMES[current_player(state)]}."])
        return "\n".join(lines)

    remaining = _remaining(state, human)
    legal = tuple(map(int, legal_actions(state)))
    lines.extend(["", "Your remaining hand: "
                  + "  ".join(format_domino(d) for d in remaining),
                  "Your legal plays:"])
    for i, action in enumerate(legal, 1):
        lines.append(f"  {i}. {format_domino(state.hands[human][action])}")
    return "\n".join(lines)


def _format_decision(decision: HoytDecision) -> str:
    seat = _SEAT_NAMES[decision.seat]
    tile = format_domino(decision.domino)
    if decision.forced:
        return f"{seat} plays {tile} (forced)."
    confidence = "separated" if decision.separated else "close"
    if decision.utility == "make":
        edge = 100.0 * float(decision.margin or 0.0)
        error = 100.0 * float(decision.margin_error or 0.0)
        measure = f"estimated edge {edge:.1f} pp, band ±{error:.1f} pp"
    else:
        edge = float(decision.margin or 0.0)
        error = float(decision.margin_error or 0.0)
        measure = f"estimated edge {edge:.2f} Q, band ±{error:.2f} Q"
    return (f"{seat} plays {tile} — {confidence} ({measure}; "
            f"{decision.wall_seconds:.1f}s, "
            f"{decision.n_worlds:,} physical worlds).")


def _prefix_policy(name: str, seed: int) -> PlayPolicy:
    if name == "random":
        return RandomPlay(seed ^ 0x5052_4546)
    if name != "jud":
        raise ValueError("prefix must be 'jud' or 'random'")
    import torch
    from arena.jud_play import JudPlay
    from champion.jud_net import load_jud_net

    torch.set_num_threads(1)
    return JudPlay(load_jud_net("champion/jud_net.pt", device="cpu"))


def _new_session(args) -> TableSession:
    policy = _prefix_policy(args.prefix, args.seed)
    state = advance_to_horizon(args.seed, args.horizon, policy)
    return TableSession(
        state=state,
        human_seat=current_player(state),
        seed=args.seed,
        start_horizon=args.horizon,
        prefix=args.prefix,
        utility=args.utility,
        bot_seed=args.bot_seed,
    )


def _advance_bots(
    session: TableSession,
    bot: HoytPlay,
    save: Callable[[], None],
) -> list[HoytDecision]:
    decisions = []
    while not is_terminal(session.state) \
            and current_player(session.state) != session.human_seat:
        state = session.state
        action = bot.choose([state], [state.bid_state.high_bid])[0]
        decision = bot.last_decisions[0]
        session.state = apply_action(state, action)
        save()
        decisions.append(decision)
    return decisions


def _hint(session: TableSession, bot: HoytPlay) -> HoytDecision:
    if is_terminal(session.state):
        raise ValueError("the hand is over")
    if current_player(session.state) != session.human_seat:
        raise ValueError("it is not the human seat's turn")
    bot.choose([session.state], [session.state.bid_state.high_bid])
    return bot.last_decisions[0]


def _interactive(session: TableSession, path: Path, bot: HoytPlay) -> int:
    while True:
        print("\n" + render_position(session), flush=True)
        if is_terminal(session.state):
            return 0
        try:
            text = input("\nPlay a number/tile, ask for 'hint', or 'quit': ").strip()
        except EOFError:
            print()
            return 0
        if text.lower() in ("q", "quit", "exit"):
            return 0
        if text.lower() in ("h", "hint", "?"):
            print("Hoyt is solving your seat...", flush=True)
            print(_format_decision(_hint(session, bot)), flush=True)
            continue
        try:
            action = _action_from_text(session.state, text)
        except ValueError as exc:
            print(f"No play made: {exc}", flush=True)
            continue
        session.state = apply_action(session.state, action)
        save_session(session, path)
        for decision in _advance_bots(
                session, bot, lambda: save_session(session, path)):
            print(_format_decision(decision), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Play an H5/H6 continuation against full-Metal Hoyt")
    parser.add_argument("--state", type=Path,
                        default=Path("scratch/full-metal-table/session.json"))
    parser.add_argument("--new", action="store_true",
                        help="replace the named scratch session")
    parser.add_argument("--seed", type=int, default=910000)
    parser.add_argument("--horizon", type=int, choices=(5, 6), default=5)
    parser.add_argument("--prefix", choices=("jud", "random"), default="jud")
    parser.add_argument("--utility", choices=("make", "points"), default="make")
    parser.add_argument("--bot-seed", type=int, default=42)
    parser.add_argument("--move", type=str,
                        help="make one human play, then run to the next human turn")
    parser.add_argument("--hint", action="store_true",
                        help="solve the current human decision without playing it")
    parser.add_argument("--interactive", action="store_true",
                        help="stay at the table until quit or hand end")
    args = parser.parse_args()

    if args.new:
        session = _new_session(args)
        save_session(session, args.state)
    else:
        session = load_session(args.state)
    bot = HoytPlay(utility=session.utility, seed=session.bot_seed)

    if args.hint:
        print(render_position(session), flush=True)
        print("\nHoyt is solving your seat...", flush=True)
        print(_format_decision(_hint(session, bot)), flush=True)
        return 0
    if args.move is not None:
        if is_terminal(session.state):
            raise ValueError("the hand is already over")
        if current_player(session.state) != session.human_seat:
            raise ValueError("the saved table is waiting for a bot seat")
        action = _action_from_text(session.state, args.move)
        session.state = apply_action(session.state, action)
        save_session(session, args.state)
        for decision in _advance_bots(
                session, bot, lambda: save_session(session, args.state)):
            print(_format_decision(decision), flush=True)
        print("\n" + render_position(session), flush=True)
        return 0
    if args.interactive:
        return _interactive(session, args.state, bot)
    print(render_position(session))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
