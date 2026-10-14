# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0

"""The public record — what every scenario in a cell shares.

A cell of the search holds scenarios that agree on everything the deciding
player has observed. In 42 the whole of that is public: which tiles have
been played, the current trick, who leads, what each team has banked, and
which suits each seat has shown void in. The record carries exactly that,
immutably; `advance` applies one play and resolves a completed trick.
"""
from __future__ import annotations

from dataclasses import dataclass

from forge.zeb.types import ZebGameState

from scenario.rules import ALL_TILES, Rules, bit, mask_of, rules_for

HAND_SIZE = 7


@dataclass(frozen=True, slots=True)
class Record:
    played: int                             # bitmask of all tiles played
    leader: int                             # seat that led the current trick
    trick: tuple[int, ...]                  # tiles of the current trick, play order
    banked: tuple[int, int]                 # (team 0, team 1) points so far
    forbid: tuple[int, int, int, int]       # per seat: tiles it provably cannot hold

    @property
    def seat(self) -> int:
        """The seat to play next."""
        return (self.leader + len(self.trick)) % 4

    @property
    def led(self) -> int | None:
        return self.trick[0] if self.trick else None

    @property
    def finished(self) -> bool:
        return self.played == ALL_TILES

    def key(self) -> tuple:
        return (self.played, self.leader, self.trick, self.banked, self.forbid)

    def plays_so_far(self) -> int:
        return self.played.bit_count()

    def hand_size(self, seat: int) -> int:
        """Tiles seat still holds, implied by the record alone."""
        completed = (self.plays_so_far() - len(self.trick)) // 4
        in_trick = 1 if (seat - self.leader) % 4 < len(self.trick) else 0
        return HAND_SIZE - completed - in_trick

    def advance(self, tile: int, rules: Rules) -> "Record":
        """The record after the seat to play puts down `tile`."""
        seat = self.seat
        forbid = self.forbid
        led = self.led
        if led is not None and not rules.follows(tile, led):
            # Failing to follow proves the seat holds no tile of the led suit.
            shown = rules.follow_bits[rules.led_suit[led]]
            f = list(forbid)
            f[seat] |= shown
            forbid = (f[0], f[1], f[2], f[3])
        played = self.played | bit(tile)
        trick = self.trick + (tile,)
        if len(trick) < 4:
            return Record(played, self.leader, trick, self.banked, forbid)
        winner = (self.leader + rules.winner_offset(trick)) % 4
        points = rules.trick_points(trick)
        banked = list(self.banked)
        banked[winner % 2] += points
        return Record(played, winner, (), (banked[0], banked[1]), forbid)


def empty_record(first_leader: int) -> Record:
    return Record(0, first_leader, (), (0, 0), (0, 0, 0, 0))


def record_from_state(state: ZebGameState) -> Record:
    """Replay a PLAYING state's public history into a Record.

    The replay goes through `advance`, so the resulting played set, leader,
    trick, and banked totals are checked against the engine's own — every
    conversion doubles as a rules-conformance check.
    """
    rules = rules_for(state.decl_id)
    first_leader = state.play_history[0][0] if state.play_history else state.trick_leader
    rec = empty_record(first_leader)
    for seat, tile in state.play_history:
        if rec.seat != seat:
            raise AssertionError(f"record seat {rec.seat} != engine seat {seat}")
        rec = rec.advance(tile, rules)
    if rec.played != mask_of(state.played):
        raise AssertionError("record played set diverges from engine")
    if rec.leader != state.trick_leader or rec.trick != tuple(state.current_trick):
        raise AssertionError("record trick/leader diverges from engine")
    if rec.banked != tuple(state.team_points):
        raise AssertionError(f"record banked {rec.banked} != engine {state.team_points}")
    return rec


def remaining_hand(state: ZebGameState, seat: int) -> int:
    return mask_of(d for d in state.hands[seat] if d not in state.played)
