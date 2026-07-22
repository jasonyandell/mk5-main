"""atlas/coordinate.py — the exact situation coordinate.

A situation has an exact coordinate: the knowns known, the unknowns not
vague but delimited.  ``CoordinateV1`` is a viewer's entire epistemic
situation in a play-phase points game — public state plus the viewer's own
hand — as a frozen dataclass with a canonical, fixed-width packed encoding.
``address()`` (blake2b-128 of the packed bytes) is THE address, stable
forever; the endgame census's no-hidden-isomorphisms verdict means this
naive address is already canonical.

Play is node deletion: ``transition`` applies one play from the viewer's
information (own plays always knowable; others' plays given as
observations), updating the played masks, the trick, banked points on trick
completion, and the voids matrix from observed follow-failures.

Team points are stored declaring-team-first (index 0 = the bidder's team),
so the coordinate is orientation-canonical.  Special contracts
(nello/splash/plunge/sevens) are out of v1 scope and raise rather than
approximate; the auction phase is a declared stub (``CoordinateV0Auction``).
"""
from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass

from forge.oracle.tables import can_follow, led_suit_for_lead_domino

from atlas.algebra import (
    N_DECLS,
    N_DOMINOES,
    get_algebra,
    legal_from_hand,
    tiles_of,
    trick_points,
    trick_winner_offset,
)

VERSION = 1
_EMPTY_TILE = 0xFF  # current-trick padding sentinel in the packed form
_PACK_FMT = "<9BI4I4B4B16s"
_PACK_SIZE = struct.calcsize(_PACK_FMT)


def _bit(t: int) -> int:
    return 1 << int(t)


@dataclass(frozen=True)
class CoordinateV1:
    """A viewer's exact situation in a play-phase points game.

    version   the encoding version (=1)
    decl_id   declaration 0..9 (8 = doubles-suit, flagged non-game)
    viewer    the seat whose information this is (0..3)
    viewer_hand   uint32 mask of tiles the viewer still holds (unplayed)
    played    per-seat uint32 masks of every tile that seat has played
    trick_leader  leader of the current (possibly empty) trick
    current_trick tile ids on the current trick, in play order (0..3 tiles;
                  seat of the k-th is (trick_leader + k) % 4)
    team_points   banked (declaring team, defending team)
    bid_value, bidder, dealer   the auction facts play conditions on
    voids     per-seat uint8: bit ls set == seat observed void in led-suit
              ls (0..6 pips, 7 = called) — every follow-failure all seats saw
    """

    version: int
    decl_id: int
    viewer: int
    viewer_hand: int
    played: tuple  # (uint32, uint32, uint32, uint32)
    trick_leader: int
    current_trick: tuple  # tile ids, len 0..3
    team_points: tuple  # (declaring, defending)
    bid_value: int
    bidder: int
    dealer: int
    voids: tuple  # (uint8, uint8, uint8, uint8)

    # ---- packing ------------------------------------------------------- #

    def pack(self) -> bytes:
        """The canonical address bytes: fixed-width, little-endian, versioned.
        A v1 extension slot (history_digest) is present and zeroed."""
        ct = list(self.current_trick) + [_EMPTY_TILE] * (3 - len(self.current_trick))
        return struct.pack(
            _PACK_FMT,
            self.version, self.decl_id, self.viewer, self.bidder, self.dealer,
            self.trick_leader, self.bid_value,
            self.team_points[0], self.team_points[1],
            int(self.viewer_hand),
            int(self.played[0]), int(self.played[1]),
            int(self.played[2]), int(self.played[3]),
            self.voids[0], self.voids[1], self.voids[2], self.voids[3],
            len(self.current_trick), ct[0], ct[1], ct[2],
            b"\x00" * 16,
        )

    @staticmethod
    def unpack(data: bytes) -> "CoordinateV1":
        if len(data) != _PACK_SIZE:
            raise ValueError(f"packed coordinate must be {_PACK_SIZE} bytes, "
                             f"got {len(data)}")
        f = struct.unpack(_PACK_FMT, data)
        version = f[0]
        if version != VERSION:
            raise ValueError(f"unsupported coordinate version {version}")
        n_ct = f[18]
        current_trick = tuple(int(x) for x in f[19:19 + n_ct])
        return CoordinateV1(
            version=version, decl_id=f[1], viewer=f[2], bidder=f[3], dealer=f[4],
            trick_leader=f[5], bid_value=f[6],
            team_points=(f[7], f[8]),
            viewer_hand=int(f[9]),
            played=(int(f[10]), int(f[11]), int(f[12]), int(f[13])),
            voids=(f[14], f[15], f[16], f[17]),
            current_trick=current_trick,
        )

    def address(self) -> bytes:
        """blake2b-128 of the packed bytes — THE address."""
        return hashlib.blake2b(self.pack(), digest_size=16).digest()

    # ---- structure ----------------------------------------------------- #

    @property
    def acting_seat(self) -> int:
        """The seat to act now (leader + tiles already on the trick)."""
        return (self.trick_leader + len(self.current_trick)) % 4

    @property
    def led_tile(self) -> int | None:
        return self.current_trick[0] if self.current_trick else None


def _validate_points_game(decl_id: int) -> None:
    if not (0 <= decl_id < N_DECLS):
        raise NotImplementedError(
            f"coordinate v1 covers play-phase points games (decl 0..{N_DECLS - 1}); "
            f"special contracts are out of scope, got decl_id={decl_id}")


def _voids_from_history(play_history, decl_id: int) -> list[int]:
    """Per-seat void bitmask (bit ls == seat failed to follow led-suit ls),
    accumulated over every observed play — the public follow-failure record.
    Mirrors walt.worlds._void_forbidden's grouping (chunks of 4, the trailing
    chunk the partial current trick)."""
    voids = [0, 0, 0, 0]
    for i in range(0, len(play_history), 4):
        chunk = play_history[i:i + 4]
        lead_tile = chunk[0][1]
        ls = led_suit_for_lead_domino(lead_tile, decl_id)
        for seat, tile in chunk[1:]:
            if not can_follow(tile, ls, decl_id):
                voids[seat] |= _bit(ls)
    return voids


def from_engine(state, viewer: int) -> CoordinateV1:
    """Build a coordinate from a forge.zeb.game ZebGameState for ``viewer``."""
    _validate_points_game(state.decl_id)
    if not (0 <= viewer < 4):
        raise ValueError(f"viewer must be 0..3, got {viewer}")

    played = [0, 0, 0, 0]
    for seat, dom in state.play_history:
        played[seat] |= _bit(dom)

    viewer_hand = 0
    for dom in state.hands[viewer]:
        if dom not in state.played:
            viewer_hand |= _bit(dom)

    bt = state.bidder % 2
    tp = state.team_points
    team_points = (tp[bt], tp[1 - bt])

    voids = _voids_from_history(state.play_history, state.decl_id)

    return CoordinateV1(
        version=VERSION,
        decl_id=state.decl_id,
        viewer=viewer,
        viewer_hand=viewer_hand,
        played=(played[0], played[1], played[2], played[3]),
        trick_leader=state.trick_leader,
        current_trick=tuple(state.current_trick),
        team_points=team_points,
        bid_value=state.bid_state.high_bid,
        bidder=state.bidder,
        dealer=state.dealer,
        voids=(voids[0], voids[1], voids[2], voids[3]),
    )


def legal(coord: CoordinateV1) -> int:
    """uint32 legal-move mask for the seat to act — defined only when that
    seat is the viewer (a hidden seat's legal set is world-dependent and the
    fiber module owns it)."""
    if coord.acting_seat != coord.viewer:
        raise ValueError(
            "legal(coord) is defined only when the viewer is to act; the "
            "acting seat's legal set is per-world (see atlas.fiber)")
    return legal_from_hand(coord.viewer_hand, coord.led_tile, coord.decl_id)


def transition(coord: CoordinateV1, tile: int) -> CoordinateV1:
    """Node deletion: apply one play of ``tile`` by the seat to act, from the
    viewer's information. The viewer's own plays must be legal; another seat's
    play is taken as an observation (updating masks and voids)."""
    tile = int(tile)
    seat = coord.acting_seat
    decl = coord.decl_id
    tbit = _bit(tile)

    if tbit & coord.played[seat] or any(tbit & coord.played[s] for s in range(4)):
        raise ValueError(f"tile {tile} has already been played")

    if seat == coord.viewer:
        if not (tbit & coord.viewer_hand):
            raise ValueError(f"viewer does not hold tile {tile}")
        if not (tbit & legal(coord)):
            raise ValueError(f"tile {tile} is not a legal play for the viewer")

    # observed follow-failure -> void (public; recorded for any seat)
    voids = list(coord.voids)
    if coord.current_trick:
        ls = led_suit_for_lead_domino(coord.current_trick[0], decl)
        if not can_follow(tile, ls, decl):
            voids[seat] |= _bit(ls)

    played = list(coord.played)
    played[seat] |= tbit
    viewer_hand = coord.viewer_hand & ~tbit if seat == coord.viewer else coord.viewer_hand
    new_trick = coord.current_trick + (tile,)

    if len(new_trick) < 4:
        return CoordinateV1(
            version=coord.version, decl_id=decl, viewer=coord.viewer,
            viewer_hand=viewer_hand, played=tuple(played),
            trick_leader=coord.trick_leader, current_trick=new_trick,
            team_points=coord.team_points, bid_value=coord.bid_value,
            bidder=coord.bidder, dealer=coord.dealer, voids=tuple(voids),
        )

    # trick complete: resolve, bank, hand the lead to the winner
    offset = trick_winner_offset(new_trick[0], new_trick, decl)
    winner = (coord.trick_leader + offset) % 4
    pts = trick_points(new_trick)
    idx = 0 if winner % 2 == coord.bidder % 2 else 1
    team_points = list(coord.team_points)
    team_points[idx] += pts

    return CoordinateV1(
        version=coord.version, decl_id=decl, viewer=coord.viewer,
        viewer_hand=viewer_hand, played=tuple(played),
        trick_leader=winner, current_trick=(),
        team_points=(team_points[0], team_points[1]),
        bid_value=coord.bid_value, bidder=coord.bidder, dealer=coord.dealer,
        voids=tuple(voids),
    )


# --------------------------------------------------------------------------- #
#  the one symmetry — transporting a coordinate through the 2<->3 arrow         #
# --------------------------------------------------------------------------- #

def _permute_mask(mask: int, perm, width: int = N_DOMINOES) -> int:
    out = 0
    m = int(mask)
    for t in range(width):
        if (m >> t) & 1:
            out |= 1 << int(perm[t])
    return out


def transport(coord: CoordinateV1) -> CoordinateV1:
    """Transport a coordinate through the game's one symmetry — the pips
    2<->3 arrow (tiles + declaration + led-suit domain), which carries the
    twos-game onto the threes-game (gate R6). Seats, banked points and the
    auction facts are arrow-invariant. An involution; a game isomorphism only
    for decl 2 and 3, where it commutes with transition (gate C6)."""
    alg = get_algebra()
    at, ad, als = alg.arrow_tile, alg.arrow_decl, alg.arrow_led
    voids = tuple(
        _permute_mask(v, als, width=8) if v else 0 for v in coord.voids)
    return CoordinateV1(
        version=coord.version,
        decl_id=int(ad[coord.decl_id]),
        viewer=coord.viewer,
        viewer_hand=_permute_mask(coord.viewer_hand, at),
        played=tuple(_permute_mask(p, at) for p in coord.played),
        trick_leader=coord.trick_leader,
        current_trick=tuple(int(at[t]) for t in coord.current_trick),
        team_points=coord.team_points,
        bid_value=coord.bid_value, bidder=coord.bidder, dealer=coord.dealer,
        voids=voids,
    )


# --------------------------------------------------------------------------- #
#  auction-phase coordinate — declared, stubbed in v1 (pack/unpack only)       #
# --------------------------------------------------------------------------- #

_AUCTION_FMT = "<2BIB7b"
_AUCTION_SIZE = struct.calcsize(_AUCTION_FMT)


@dataclass(frozen=True)
class CoordinateV0Auction:
    """The auction-phase situation: the viewer's hand, the bid history, and
    the dealer. Declared for the auction lane (#42); v1 stubs it with a
    canonical pack/unpack only — no transitions, no legality."""

    version: int
    viewer: int
    viewer_hand: int
    dealer: int
    bids: tuple  # seat-indexed bid history, -1 = not yet / pass sentinel

    def pack(self) -> bytes:
        bids = tuple(self.bids) + (-1,) * (7 - len(self.bids))
        return struct.pack(_AUCTION_FMT, self.version, self.viewer,
                           int(self.viewer_hand), self.dealer, *bids[:7])

    @staticmethod
    def unpack(data: bytes) -> "CoordinateV0Auction":
        if len(data) != _AUCTION_SIZE:
            raise ValueError(f"packed auction coordinate must be "
                             f"{_AUCTION_SIZE} bytes, got {len(data)}")
        f = struct.unpack(_AUCTION_FMT, data)
        if f[0] != VERSION:
            raise ValueError(f"unsupported auction coordinate version {f[0]}")
        bids = tuple(b for b in f[4:11] if b != -1)
        return CoordinateV0Auction(version=f[0], viewer=f[1],
                                   viewer_hand=int(f[2]), dealer=f[3], bids=bids)

    def address(self) -> bytes:
        return hashlib.blake2b(self.pack(), digest_size=16).digest()


def _tiles(mask: int):
    """Ascending tile ids in a mask (re-exported for callers/tests)."""
    return tiles_of(mask)
