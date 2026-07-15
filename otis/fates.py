"""Otis fate parser — per-count-tile fate ledger over a replayed Texas 42 game.

The neutral input (:class:`NeutralGame`) is the initial 4x7 deal, the declaration,
the bidder, the bid value, and the ordered list of plays. The parser replays the
seven tricks and, for each of the five count tiles (5-5, 6-4, 5-0, 4-1, 3-2), emits
a :class:`TileFate`:

  * ``holder_seat``     — seat that was dealt the tile (0-3, absolute)
  * ``trick_idx``       — 0..6, which trick the tile was played in
  * ``played_mode``     — how the HOLDER played the tile in {led, followed,
                          trumped_in, sloughed}
  * ``winner_seat``     — seat that won that trick
  * ``capture_side``    — {holder_team, opp_team}: did the tile's points go to the
                          holder's team or the opponents?
  * ``won_by_trump``    — did the winning domino win by trump power?

Plus per-team trick counts and per-team points.

Suit algebra (led suit, following, doubles, trump ranking, trick resolution) is
**reused** from ``forge.oracle.tables`` / ``forge.oracle.declarations`` — the exact
CPU helpers that back the GPU engine ``forge.eq.game_tensor.GameStateTensor``. We do
not reimplement the rules; we drive the same lookups the engine does. See
``wiki/topics/rules-of-42.md`` and ``wiki/topics/suit-algebra-spec.md``.

played_mode semantics (a count tile is one specific domino, so its mode is a pure
function of the tile, its trick's led suit, and the declaration):

  * ``led``        — the tile was the first play of its trick.
  * ``followed``   — the tile follows the led suit per 42 rules (``can_follow``).
  * ``trumped_in`` — the tile did NOT follow (holder was void in the led suit) and
                     the tile is a trump with power.
  * ``sloughed``   — the tile did NOT follow, and the tile is not a power trump
                     (an off-suit / no-power discard while void).

THE EXACT IDENTITY (P1, stop-the-line): for every game,
``team0_count + team0_tricks + team1_count + team1_tricks == 42`` and each team's
``points == count + tricks``. These are asserted in the parser itself, not only in
tests. Any failure is a parser/replay bug — fix it, never widen tolerance.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from forge.oracle.declarations import (
    DECL_ID_TO_NAME,
    N_DECLS,
    has_trump_power,
)
from forge.oracle.tables import (
    DOMINO_COUNT_POINTS,
    DOMINO_HIGH,
    DOMINO_LOW,
    DOMINOES,
    can_follow,
    is_in_called_suit,
    led_suit_for_lead_domino,
    trick_rank,
)

# --------------------------------------------------------------------------- #
# Domino / declaration encoding helpers
# --------------------------------------------------------------------------- #

# Reverse map (high, low) -> domino id. DOMINOES is ordered high-then-low.
_PIPS_TO_ID: dict[tuple[int, int], int] = {pips: i for i, pips in enumerate(DOMINOES)}


def domino_id_to_pips(domino_id: int) -> str:
    """Return the canonical ``"hi-lo"`` pip string for a domino id (e.g. ``"6-4"``)."""
    if not (0 <= domino_id < 28):
        raise ValueError(f"domino_id out of range 0..27: {domino_id}")
    return f"{DOMINO_HIGH[domino_id]}-{DOMINO_LOW[domino_id]}"


def pips_to_domino_id(pips: str) -> int:
    """Parse a ``"hi-lo"`` (or ``"lo-hi"``) pip string into a domino id."""
    parts = pips.split("-")
    if len(parts) != 2:
        raise ValueError(f"bad domino pip string: {pips!r}")
    a, b = int(parts[0]), int(parts[1])
    hi, lo = (a, b) if a >= b else (b, a)
    key = (hi, lo)
    if key not in _PIPS_TO_ID:
        raise ValueError(f"not a valid domino: {pips!r}")
    return _PIPS_TO_ID[key]


def decl_name(decl_id: int) -> str:
    """Resolve a forge declaration id to its canonical name (e.g. 5 -> ``"fives"``)."""
    if decl_id not in DECL_ID_TO_NAME:
        raise ValueError(f"unknown decl_id: {decl_id} (valid 0..{N_DECLS - 1})")
    return DECL_ID_TO_NAME[decl_id]


# The five count tiles, by domino id. 5-5 and 6-4 are worth 10; 5-0, 4-1, 3-2 are 5.
COUNT_TILE_PIPS: tuple[str, ...] = ("5-5", "6-4", "5-0", "4-1", "3-2")
COUNT_TILE_IDS: tuple[int, ...] = tuple(pips_to_domino_id(p) for p in COUNT_TILE_PIPS)

# Sanity: every count tile carries points, and the five of them total 35.
assert all(DOMINO_COUNT_POINTS[i] > 0 for i in COUNT_TILE_IDS)
assert sum(DOMINO_COUNT_POINTS[i] for i in COUNT_TILE_IDS) == 35

# played_mode labels
LED = "led"
FOLLOWED = "followed"
TRUMPED_IN = "trumped_in"
SLOUGHED = "sloughed"
PLAYED_MODES = (LED, FOLLOWED, TRUMPED_IN, SLOUGHED)


def _team_of(seat: int) -> int:
    """Absolute team of a seat: {0,2} -> 0, {1,3} -> 1."""
    return seat % 2


def _played_mode(tile_id: int, led_suit: int, decl_id: int, is_lead: bool) -> str:
    """Classify how the holder played ``tile_id`` given the trick's led suit.

    Assumes the play is legal (which it is for a real replayed game): a non-following
    play therefore implies the holder was void in the led suit.
    """
    if is_lead:
        return LED
    if can_follow(tile_id, led_suit, decl_id):
        return FOLLOWED
    # Did not follow -> holder was void in the led suit.
    if has_trump_power(decl_id) and is_in_called_suit(tile_id, decl_id):
        return TRUMPED_IN
    return SLOUGHED


# --------------------------------------------------------------------------- #
# Neutral game representation and parser outputs
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class NeutralGame:
    """Engine-neutral replayed game — the parser's only input.

    Attributes:
        game_id: Stable identifier for the game.
        hands: 4 lists of 7 domino ids, absolute seat order (the initial deal).
        decl_id: forge declaration id (0..9).
        bidder: seat 0-3 that won the auction and leads trick 1.
        bid_value: bid amount (30..42, or a mark value).
        plays: (seat, domino_id) in exact play order, length 28.
    """

    game_id: str
    hands: list[list[int]]
    decl_id: int
    bidder: int
    bid_value: int
    plays: list[tuple[int, int]]


@dataclass(frozen=True)
class TileFate:
    """Fate of one count tile within a single game."""

    tile: str  # pip string, e.g. "5-5"
    tile_id: int
    holder_seat: int
    trick_idx: int
    played_mode: str
    winner_seat: int
    capture_side: str  # "holder_team" | "opp_team"
    won_by_trump: bool


@dataclass(frozen=True)
class GameFates:
    """Full fate ledger for one game."""

    game_id: str
    decl_id: int
    decl_name: str
    bidder: int
    bid_value: int
    tiles: list[TileFate] = field(default_factory=list)
    team0_count: int = 0
    team0_tricks: int = 0
    team1_count: int = 0
    team1_tricks: int = 0
    team0_points: int = 0
    team1_points: int = 0


def _validate_neutral(game: NeutralGame) -> dict[int, int]:
    """Structural validation. Returns holder_seat-by-domino-id map."""
    if len(game.hands) != 4:
        raise ValueError(f"{game.game_id}: expected 4 hands, got {len(game.hands)}")
    holder_of: dict[int, int] = {}
    for seat, hand in enumerate(game.hands):
        if len(hand) != 7:
            raise ValueError(
                f"{game.game_id}: seat {seat} has {len(hand)} dominoes, expected 7"
            )
        for did in hand:
            if not (0 <= did < 28):
                raise ValueError(f"{game.game_id}: bad domino id {did} in seat {seat}")
            if did in holder_of:
                raise ValueError(f"{game.game_id}: domino {did} dealt twice")
            holder_of[did] = seat
    if len(holder_of) != 28:
        raise ValueError(f"{game.game_id}: deal covers {len(holder_of)} dominoes, need 28")
    if not (0 <= game.decl_id < N_DECLS):
        raise ValueError(f"{game.game_id}: decl_id {game.decl_id} out of range")
    if not (0 <= game.bidder < 4):
        raise ValueError(f"{game.game_id}: bidder {game.bidder} out of range")
    if len(game.plays) != 28:
        raise ValueError(f"{game.game_id}: expected 28 plays, got {len(game.plays)}")
    return holder_of


def parse_game_fates(game: NeutralGame) -> GameFates:
    """Replay ``game`` and emit its count-tile fate ledger.

    Reuses ``forge.oracle.tables`` for all suit algebra. Asserts the P1 identity
    (points sum to 42; each team's points == count + tricks) before returning.
    """
    holder_of = _validate_neutral(game)
    decl_id = game.decl_id

    # Per-tile fate accumulator (filled as tiles are played).
    tile_fate: dict[int, TileFate] = {}

    # Per-team tallies.
    team_count = [0, 0]
    team_tricks = [0, 0]

    # The bid winner leads trick 1; each trick's winner leads the next.
    leader = game.bidder

    played_ids: set[int] = set()

    for trick_idx in range(7):
        trick = game.plays[trick_idx * 4 : trick_idx * 4 + 4]

        # The recorded leader must match the trick's first player. This cross-checks
        # our replay against the source engine's own turn sequencing.
        lead_seat, lead_id = trick[0]
        if lead_seat != leader:
            raise ValueError(
                f"{game.game_id}: trick {trick_idx} led by seat {lead_seat}, "
                f"expected winner-derived leader {leader}"
            )

        led_suit = led_suit_for_lead_domino(lead_id, decl_id)

        # Resolve the trick. Rank each play; highest rank wins (ties -> first played).
        best_offset = 0
        best_rank = trick_rank(trick[0][1], led_suit, decl_id)
        trick_points = 1  # one point for winning the trick
        for offset, (seat, did) in enumerate(trick):
            # Every domino is unique and unplayed until now.
            if did in played_ids:
                raise ValueError(f"{game.game_id}: domino {did} played twice")
            played_ids.add(did)
            trick_points += DOMINO_COUNT_POINTS[did]
            if offset > 0:
                r = trick_rank(did, led_suit, decl_id)
                if r > best_rank:
                    best_rank = r
                    best_offset = offset

        winner_seat, winner_id = trick[best_offset]
        winner_team = _team_of(winner_seat)
        won_by_trump = has_trump_power(decl_id) and is_in_called_suit(winner_id, decl_id)

        team_tricks[winner_team] += 1
        team_count[winner_team] += trick_points - 1  # subtract the trick's own point

        # Record fate for any count tile played this trick.
        for offset, (seat, did) in enumerate(trick):
            if DOMINO_COUNT_POINTS[did] == 0:
                continue
            holder_seat = holder_of[did]
            if holder_seat != seat:
                raise ValueError(
                    f"{game.game_id}: {domino_id_to_pips(did)} dealt to seat "
                    f"{holder_seat} but played by seat {seat}"
                )
            mode = _played_mode(did, led_suit, decl_id, is_lead=(offset == 0))
            capture_side = (
                "holder_team" if _team_of(holder_seat) == winner_team else "opp_team"
            )
            tile_fate[did] = TileFate(
                tile=domino_id_to_pips(did),
                tile_id=did,
                holder_seat=holder_seat,
                trick_idx=trick_idx,
                played_mode=mode,
                winner_seat=winner_seat,
                capture_side=capture_side,
                won_by_trump=won_by_trump,
            )

        leader = winner_seat  # trick winner leads the next trick

    # Every count tile must have been seen exactly once.
    tiles: list[TileFate] = []
    for tid in COUNT_TILE_IDS:
        if tid not in tile_fate:
            raise ValueError(
                f"{game.game_id}: count tile {domino_id_to_pips(tid)} never played"
            )
        tiles.append(tile_fate[tid])

    team0_points = team_count[0] + team_tricks[0]
    team1_points = team_count[1] + team_tricks[1]

    # ---- P1 EXACT IDENTITY (stop-the-line) --------------------------------- #
    total = team_count[0] + team_tricks[0] + team_count[1] + team_tricks[1]
    assert total == 42, (
        f"{game.game_id}: point identity violated — "
        f"team0(count {team_count[0]} + tricks {team_tricks[0]}) + "
        f"team1(count {team_count[1]} + tricks {team_tricks[1]}) = {total} != 42"
    )
    assert team_tricks[0] + team_tricks[1] == 7, (
        f"{game.game_id}: tricks sum {team_tricks[0] + team_tricks[1]} != 7"
    )
    assert team_count[0] + team_count[1] == 35, (
        f"{game.game_id}: count sum {team_count[0] + team_count[1]} != 35"
    )
    assert team0_points == team_count[0] + team_tricks[0]
    assert team1_points == team_count[1] + team_tricks[1]

    return GameFates(
        game_id=game.game_id,
        decl_id=decl_id,
        decl_name=decl_name(decl_id),
        bidder=game.bidder,
        bid_value=game.bid_value,
        tiles=tiles,
        team0_count=team_count[0],
        team0_tricks=team_tricks[0],
        team1_count=team_count[1],
        team1_tricks=team_tricks[1],
        team0_points=team0_points,
        team1_points=team1_points,
    )
