"""Full-game arena engine: deal → auction → play → marks, first to 7.

Each game is a race to `marks_to_win` marks between absolute team 0 (seats
0/2) and team 1 (seats 1/3); `a_team` says which absolute team player A
occupies, so paired halves rotate the seats under identical deal seeds.
Auctions run on CPU at hand boundaries; play decisions run in lockstep
across all live games so GPU play policies see one batch per tick.

The zeb engine owns the play phase (forge/zeb/game.py); the arena owns
everything around it. A hand enters the engine as a fully-formed PLAYING
state carrying the auction's bidder, declaration, and bid — the same
construction as the historical forced-bid-30 evals, with the force removed.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass

from forge.oracle.rng import deal_from_seed
from forge.zeb.game import apply_action, current_player, is_terminal
from forge.zeb.types import BidState, GamePhase, ZebGameState

from .auction import AuctionResult, BidPolicy, run_auction, score_hand
from .play import PlayPolicy


@dataclass(frozen=True)
class ArenaConfig:
    marks_to_win: int = 7
    max_redeals: int = 3  # pass-out reshakes before the shaker is forced to 30
    base_seed: int = 0


def hand_seed(base_seed: int, game_idx: int, hand_idx: int, redeal: int) -> int:
    """Decorrelated deal seed; independent of team assignment, so paired
    halves replay identical deals."""
    return hash((base_seed, game_idx, hand_idx, redeal)) & 0xFFFFFFFF


@dataclass(frozen=True)
class HandRecord:
    """One completed hand within a game."""

    game_idx: int
    hand_idx: int
    seed: int
    a_team: int
    dealer: int
    redeals: int
    forced: bool
    hands: tuple[tuple[int, ...], ...]  # initial deal, seat order (4 x 7 domino ids)
    bids: tuple[int, int, int, int]
    bidder: int
    bid_value: int
    decl_id: int
    team_points: tuple[int, int]  # absolute (team0, team1)
    made: bool
    marks_delta: tuple[int, int]
    marks_after: tuple[int, int]

    @property
    def bidder_team(self) -> int:
        return self.bidder % 2


@dataclass(frozen=True)
class GameRecord:
    """One completed game (race to marks_to_win)."""

    game_idx: int
    a_team: int
    marks: tuple[int, int]  # absolute (team0, team1)
    hands: tuple[HandRecord, ...]

    @property
    def winner_team(self) -> int:
        return 0 if self.marks[0] > self.marks[1] else 1

    @property
    def a_won(self) -> bool:
        return self.winner_team == self.a_team

    @property
    def a_marks(self) -> int:
        return self.marks[self.a_team]

    @property
    def b_marks(self) -> int:
        return self.marks[1 - self.a_team]


class _LiveGame:
    """Mutable per-game tracker driven by the lockstep loop."""

    def __init__(
        self,
        game_idx: int,
        a_team: int,
        cfg: ArenaConfig,
        bid_policies: tuple[BidPolicy, BidPolicy, BidPolicy, BidPolicy],
    ):
        self.game_idx = game_idx
        self.a_team = a_team
        self.cfg = cfg
        self.bid_policies = bid_policies
        self.marks = [0, 0]
        self.dealer = 0
        self.hand_idx = 0
        self.hands: list[HandRecord] = []
        self.state: ZebGameState | None = None
        self._auction: AuctionResult | None = None
        self._hands: tuple[tuple[int, ...], ...] = ()
        self._seed = -1
        self._redeals = 0
        self.done = False
        self._deal_and_bid()

    def _deal_and_bid(self) -> None:
        """Deal (reshaking through pass-outs) and run the auction."""
        redeals = 0
        while True:
            seed = hand_seed(self.cfg.base_seed, self.game_idx, self.hand_idx, redeals)
            hands = tuple(tuple(h) for h in deal_from_seed(seed))
            # int-only seed: str hashes are salted per-process, int hashes are not
            rng = random.Random(hash((seed, 0xA0C7)))
            result = run_auction(
                hands, self.dealer, self.bid_policies, rng,
                force_shaker=redeals >= self.cfg.max_redeals,
                marks=(self.marks[0], self.marks[1]),
                marks_to_win=self.cfg.marks_to_win,
            )
            if result is not None:
                break
            redeals += 1
            self.dealer = (self.dealer + 1) % 4

        self._auction = result
        self._hands = hands
        self._seed = seed
        self._redeals = redeals
        self.state = ZebGameState(
            hands=hands,
            dealer=self.dealer,
            phase=GamePhase.PLAYING,
            bid_state=BidState(
                bids=result.bids,
                high_bidder=result.winner,
                high_bid=result.high_bid,
            ),
            decl_id=result.decl_id,
            bidder=result.winner,
            played=frozenset(),
            play_history=(),
            current_trick=(),
            trick_leader=result.winner,
            team_points=(0, 0),
        )

    @property
    def bid_value(self) -> int:
        return self._auction.high_bid

    def apply(self, action: int) -> None:
        """Apply one play action; on hand completion, score marks and either
        finish the game or shake the next hand."""
        self.state = apply_action(self.state, action)
        if not is_terminal(self.state):
            return

        auction = self._auction
        score = score_hand(
            auction.high_bid, auction.winner % 2, self.state.team_points,
        )
        self.marks[0] += score.marks[0]
        self.marks[1] += score.marks[1]
        self.hands.append(HandRecord(
            game_idx=self.game_idx,
            hand_idx=self.hand_idx,
            seed=self._seed,
            a_team=self.a_team,
            dealer=self.dealer,
            redeals=self._redeals,
            forced=auction.forced,
            hands=self._hands,
            bids=auction.bids,
            bidder=auction.winner,
            bid_value=auction.high_bid,
            decl_id=auction.decl_id,
            team_points=self.state.team_points,
            made=score.made,
            marks_delta=score.marks,
            marks_after=(self.marks[0], self.marks[1]),
        ))

        if max(self.marks) >= self.cfg.marks_to_win:
            self.done = True
            self.state = None
            return

        self.hand_idx += 1
        self.dealer = (self.dealer + 1) % 4
        self._deal_and_bid()

    def record(self) -> GameRecord:
        assert self.done
        return GameRecord(
            game_idx=self.game_idx,
            a_team=self.a_team,
            marks=(self.marks[0], self.marks[1]),
            hands=tuple(self.hands),
        )


def _seat_policies(
    a_team: int, bid_a: BidPolicy, bid_b: BidPolicy,
) -> tuple[BidPolicy, BidPolicy, BidPolicy, BidPolicy]:
    return tuple((bid_a if seat % 2 == a_team else bid_b) for seat in range(4))


def _run_lockstep(
    games: list[_LiveGame],
    play_a: PlayPolicy,
    play_b: PlayPolicy,
    marks_to_win: int,
    log_every_s: float | None = None,
) -> None:
    """Drive every game to completion in lockstep.

    Every iteration routes each live game's current decision to the owning
    side's play policy, one batched call per side per tick. Games may mix
    a_team assignments — routing is per game. With log_every_s, prints a
    progress heartbeat so long runs are never silent.
    """
    t0 = last_log = time.time()
    while True:
        live = [g for g in games if not g.done]
        if not live:
            break
        if log_every_s is not None and time.time() - last_log >= log_every_s:
            last_log = time.time()
            hands = sum(len(g.hands) for g in games)
            print(
                f"    t={last_log - t0:5.0f}s  live {len(live)}/{len(games)}  "
                f"hands {hands}",
                flush=True,
            )
        a_games, b_games = [], []
        for g in live:
            side = a_games if current_player(g.state) % 2 == g.a_team else b_games
            side.append(g)
        for side_games, policy in ((a_games, play_a), (b_games, play_b)):
            if not side_games:
                continue
            actions = policy.choose(
                [g.state for g in side_games],
                [g.bid_value for g in side_games],
                [(g.marks[0], g.marks[1]) for g in side_games],
                marks_to_win,
            )
            for g, action in zip(side_games, actions):
                g.apply(action)


def run_half(
    *,
    n_games: int,
    a_team: int,
    cfg: ArenaConfig,
    bid_a: BidPolicy,
    bid_b: BidPolicy,
    play_a: PlayPolicy,
    play_b: PlayPolicy,
    log_every_s: float | None = None,
) -> list[GameRecord]:
    """Run n_games full games with player A as absolute team `a_team`."""
    seats = _seat_policies(a_team, bid_a, bid_b)
    games = [_LiveGame(i, a_team, cfg, seats) for i in range(n_games)]
    _run_lockstep(games, play_a, play_b, cfg.marks_to_win, log_every_s)
    return [g.record() for g in games]


def run_paired(
    *,
    half: int,
    cfg: ArenaConfig,
    bid_a: BidPolicy,
    bid_b: BidPolicy,
    play_a: PlayPolicy,
    play_b: PlayPolicy,
    log_every_s: float | None = None,
) -> list[GameRecord]:
    """Both halves of a paired match in ONE lockstep pool (fast batching).

    Deal seeds and opening auctions match the sequential halves exactly —
    they depend only on (base_seed, game_idx, hand_idx), never on batch
    composition. Pooling doubles the lockstep batch width and pays the
    straggler tail once instead of twice; for a fixed set of games this is
    tick-optimal (total ticks = the longest game's ticks), so no refill
    queue is needed. The cost: batch composition feeds the world-sampling
    RNG, so realized play diverges from the sequential path — games are
    statistically equivalent, not byte-identical (see
    docs/arena-perf-2026-07-06.md). Records return half 1 then half 2.
    """
    games = [
        _LiveGame(i, a_team, cfg, _seat_policies(a_team, bid_a, bid_b))
        for a_team in (0, 1)
        for i in range(half)
    ]
    _run_lockstep(games, play_a, play_b, cfg.marks_to_win, log_every_s)
    return [g.record() for g in games]
