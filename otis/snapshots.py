"""Adapter: arena per-hand snapshot JSON -> :class:`otis.fates.NeutralGame`.

The arena writes a snapshot payload (``arena/match.py:213`` ``snapshot_rows``,
row dataclass ``arena/engine.py:41`` ``HandRecord``) as a JSON dict
``{"snapshots": [...], "metadata": {...}}``. Each snapshot row is ONE completed
HAND, with:

  * ``hands``            — the initial 4x7 deal, seat order, forge domino ids
                          (``deal_from_seed`` in ``forge.oracle.rng`` — the SAME id
                          scheme ``otis.fates`` / ``forge.oracle.tables`` uses;
                          id 27 == 6-6, and each hand's union covers 0..27).
  * ``plays``           — 28 ``[seat, domino_id]`` pairs in play order. The bid
                          winner leads trick 1; each trick's winner leads next.
  * ``decl_id``         — winning declaration id (0..9).
  * ``bidder``          — winning seat; leads trick 1. ``plays[0][0] == bidder``.
  * ``bid_value``       — winning bid amount.
  * join / label keys carried as metadata (NOT into the parser input):
    ``seed``, ``game_idx``, ``hand_idx``, ``a_team``, ``dealer``, ``bids``,
    and the hand's REALIZED outcome ``bidder_team_pts`` / ``opp_team_pts`` /
    ``made``. ``(a_team, game_idx, hand_idx)`` uniquely identifies a hand across
    a paired match (``arena/match.py:229``).

The recorded ``bidder_team_pts`` / ``opp_team_pts`` give us a FREE SECOND REFEREE:
the parser replays the same play sequence independently, and its per-team points
must equal the arena's recorded points exactly. :func:`cross_check_points` asserts
this on every hand (raises on mismatch — never widen tolerance).

This module needs only ``json`` (pure stdlib); the parser (``otis.fates``) does
not need torch. Everything here is CPU. otis does not own the GPU.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from otis.fates import GameFates, NeutralGame

# Fields we require on every snapshot row.
_REQUIRED_FIELDS: tuple[str, ...] = (
    "a_team",
    "game_idx",
    "hand_idx",
    "seed",
    "dealer",
    "hands",
    "decl_id",
    "bids",
    "bidder",
    "bid_value",
    "plays",
    "bidder_team_pts",
    "opp_team_pts",
    "made",
)


@dataclass(frozen=True)
class SnapshotMeta:
    """Join keys + realized outcome for one arena hand, alongside its NeutralGame.

    Not part of the parser input — carried so downstream training joins can key
    on ``(a_team, game_idx, hand_idx)`` and supervise against the realized
    ``bidder_team_pts`` / ``opp_team_pts`` / ``made`` stamp.
    """

    seed: int
    game_idx: int
    hand_idx: int
    a_team: int
    dealer: int
    bidder: int
    bid_value: int
    decl_id: int
    bids: tuple[int, int, int, int]
    bidder_team_pts: int
    opp_team_pts: int
    made: int


@dataclass(frozen=True)
class SnapshotHand:
    """One arena hand: the parser-neutral game plus its join/label metadata."""

    game: NeutralGame
    meta: SnapshotMeta


def _row_to_hand(row: dict, game_id: str) -> SnapshotHand:
    """Convert one snapshot row into a :class:`SnapshotHand`."""
    missing = [f for f in _REQUIRED_FIELDS if f not in row]
    if missing:
        raise ValueError(f"{game_id}: snapshot row missing fields {missing}")

    hands: list[list[int]] = [[int(d) for d in hand] for hand in row["hands"]]
    plays: list[tuple[int, int]] = []
    for p in row["plays"]:
        if len(p) != 2:
            raise ValueError(f"{game_id}: play entry {p!r} is not [seat, domino_id]")
        plays.append((int(p[0]), int(p[1])))

    bids_raw = list(row["bids"])
    if len(bids_raw) != 4:
        raise ValueError(f"{game_id}: bids {bids_raw!r} is not a 4-tuple")

    game = NeutralGame(
        game_id=game_id,
        hands=hands,
        decl_id=int(row["decl_id"]),
        bidder=int(row["bidder"]),
        bid_value=int(row["bid_value"]),
        plays=plays,
    )
    meta = SnapshotMeta(
        seed=int(row["seed"]),
        game_idx=int(row["game_idx"]),
        hand_idx=int(row["hand_idx"]),
        a_team=int(row["a_team"]),
        dealer=int(row["dealer"]),
        bidder=int(row["bidder"]),
        bid_value=int(row["bid_value"]),
        decl_id=int(row["decl_id"]),
        bids=(int(bids_raw[0]), int(bids_raw[1]), int(bids_raw[2]), int(bids_raw[3])),
        bidder_team_pts=int(row["bidder_team_pts"]),
        opp_team_pts=int(row["opp_team_pts"]),
        made=int(row["made"]),
    )
    return SnapshotHand(game=game, meta=meta)


def cross_check_points(fates: GameFates, meta: SnapshotMeta) -> None:
    """Free second referee: parser-computed per-team points must equal the arena's.

    The bidder's absolute team is ``bidder % 2``; the arena records the bidder
    team's share as ``bidder_team_pts`` and the opponents' as ``opp_team_pts``
    (``arena/match.py:258``). The parser replays independently, so these must
    match EXACTLY. Raises ``ValueError`` on any mismatch — never widen tolerance.
    """
    bidder_team = meta.bidder % 2
    parsed = (fates.team0_points, fates.team1_points)
    parsed_bidder = parsed[bidder_team]
    parsed_opp = parsed[1 - bidder_team]
    if parsed_bidder != meta.bidder_team_pts or parsed_opp != meta.opp_team_pts:
        raise ValueError(
            f"{fates.game_id}: point cross-check FAILED — "
            f"parser bidder_team={parsed_bidder} opp={parsed_opp} "
            f"(team0={parsed[0]} team1={parsed[1]}, bidder seat {meta.bidder} "
            f"team {bidder_team}) vs arena bidder_team_pts={meta.bidder_team_pts} "
            f"opp_team_pts={meta.opp_team_pts}"
        )


def load_snapshot_hands(
    snapshot_path: str | Path,
    limit: int | None = None,
    game_id_prefix: str | None = None,
) -> Iterator[SnapshotHand]:
    """Yield :class:`SnapshotHand` records from one arena snapshot JSON file.

    Args:
        snapshot_path: Path to a ``*.snapshots.json`` file
            (``{"snapshots": [...], "metadata": {...}}``).
        limit: Max hands to yield (``None`` = all).
        game_id_prefix: Prefix for synthesised game ids (defaults to the file stem,
            with any trailing ``.snapshots`` removed).

    Yields:
        SnapshotHand, one per hand row, in file order.
    """
    snapshot_path = Path(snapshot_path)
    if game_id_prefix is None:
        stem = snapshot_path.stem
        if stem.endswith(".snapshots"):
            stem = stem[: -len(".snapshots")]
        game_id_prefix = stem

    with snapshot_path.open() as fh:
        payload = json.load(fh)
    if "snapshots" not in payload:
        raise ValueError(f"{snapshot_path}: payload has no 'snapshots' key")
    rows = payload["snapshots"]

    n = len(rows) if limit is None else min(limit, len(rows))
    for i in range(n):
        row = rows[i]
        # A stable, join-friendly id: (a_team, game_idx, hand_idx) is unique per match.
        game_id = (
            f"{game_id_prefix}:a{int(row['a_team'])}"
            f":g{int(row['game_idx'])}:h{int(row['hand_idx'])}"
        )
        yield _row_to_hand(row, game_id)
