"""
Shared game-state replayer for oracle-greedy corpus mining.

Reconstructs mid-game position from GameRecordGPU decisions so filters can
be applied at each decision point.  Returns snapshot dicts conforming to
forge.eq.snapshot.v1.

BIDDER assumption: In the legacy corpus, player 0 always wins the auction
(the corpus was generated with player 0 as bidder in all games).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Any, Iterator

import torch

# Ensure project root on path before any forge imports
PROJECT_ROOT = "/Users/jason/code/mk5-main"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from forge.eq.generate.types import GameRecordGPU, DecisionRecordGPU
from forge.eq.game_tensor import SNAPSHOT_SCHEMA_VERSION
from forge.oracle.declarations import PIP_TRUMP_IDS, DOUBLES_TRUMP, DOUBLES_SUIT, NOTRUMP
from forge.oracle.tables import (
    DOMINO_HIGH,
    DOMINO_LOW,
    DOMINO_IS_DOUBLE,
    DOMINO_COUNT_POINTS,
    led_suit_for_lead_domino,
    can_follow,
)

BIDDER = 0  # always player 0 in legacy corpus


def get_trump_set(decl_id: int) -> set[int]:
    """Return set of domino IDs that are trump for a given declaration."""
    return {
        did for did in range(28)
        if led_suit_for_lead_domino(did, decl_id) == 7
    }


def get_suit_for_domino(domino_id: int, decl_id: int) -> int:
    """Return the suit a domino belongs to (7 = trump, 0-6 = pip suit by high pip)."""
    return led_suit_for_lead_domino(domino_id, decl_id)


@dataclass
class HandState:
    """Mutable per-player hand state during replay."""
    held: list[int]  # domino_ids still in hand (-1 slots removed during replay)
    original: list[int]  # initial 7-domino hand (immutable reference)


@dataclass
class GameState:
    """Full mutable game state during step-by-step replay."""
    decl_id: int
    bidder: int  # always 0 in legacy corpus
    trump_set: set[int]

    # Hands as 4 x 7 arrays with -1 padding (mirrors snapshot format)
    hands: list[list[int]]  # [player][slot] = domino_id or -1

    # Played mask (length 28)
    played_mask: list[bool]

    # History: list of [player, domino_id, lead_domino_id], padded to 28 with [-1,-1,-1]
    history: list[list[int]]
    history_count: int  # actual played dominoes in history

    # Current trick
    trick_plays: list[int]  # length 4, -1 for empty
    leader: int  # player who leads current trick
    trick_lead_domino: int  # domino_id that led current trick (-1 if not started)

    # Decision index (which decision in game.decisions we are at)
    decision_idx: int = 0

    @classmethod
    def initial(cls, game: GameRecordGPU) -> "GameState":
        """Build initial state from a GameRecordGPU before any plays."""
        # hands: deep copy as 4 x 7
        hands = [list(game.hands[p]) for p in range(4)]
        return cls(
            decl_id=game.decl_id,
            bidder=BIDDER,
            trump_set=get_trump_set(game.decl_id),
            hands=hands,
            played_mask=[False] * 28,
            history=[[-1, -1, -1]] * 28,
            history_count=0,
            trick_plays=[-1, -1, -1, -1],
            leader=0,  # bidder leads first
            trick_lead_domino=-1,
            decision_idx=0,
        )

    def current_player(self) -> int:
        n_trick = sum(1 for d in self.trick_plays if d >= 0)
        return (self.leader + n_trick) % 4

    def n_trick_plays(self) -> int:
        return sum(1 for d in self.trick_plays if d >= 0)

    def trick_number(self) -> int:
        """0-indexed trick number (0..6)."""
        return self.history_count // 4

    def to_snapshot(self, bid_value: int = 30) -> dict[str, Any]:
        """Emit a snapshot dict in forge.eq.snapshot.v1 format."""
        return {
            "schema_version": SNAPSHOT_SCHEMA_VERSION,
            "decl_id": self.decl_id,
            "bid_value": bid_value,
            "bidder": self.bidder,
            "hands": [list(h) for h in self.hands],
            "played_mask": list(self.played_mask),
            "history": [list(e) for e in self.history],
            "trick_plays": list(self.trick_plays),
            "leader": self.leader,
        }

    def apply_play(self, player: int, domino_id: int) -> None:
        """Apply one play and advance state."""
        n_trick = self.n_trick_plays()
        is_lead = (n_trick == 0)

        if is_lead:
            self.trick_lead_domino = domino_id
            lead_domino_for_history = domino_id
        else:
            lead_domino_for_history = self.trick_lead_domino

        # Record in history
        self.history[self.history_count] = [player, domino_id, lead_domino_for_history]
        self.history_count += 1

        # Mark played
        self.played_mask[domino_id] = True

        # Remove from hand (find slot)
        hand = self.hands[player]
        for slot_idx in range(7):
            if hand[slot_idx] == domino_id:
                hand[slot_idx] = -1
                break

        # Place in trick
        self.trick_plays[n_trick] = domino_id

        # If trick complete, resolve
        if n_trick + 1 == 4:
            self._resolve_trick()

    def _resolve_trick(self) -> None:
        """Determine trick winner and advance leader."""
        led_suit = led_suit_for_lead_domino(self.trick_lead_domino, self.decl_id)
        # Build trick plays with their players in order: leader, leader+1, leader+2, leader+3
        best_player = self.leader
        best_domino = self.trick_plays[0]
        best_rank = self._trick_rank(best_domino, led_suit)

        for i in range(1, 4):
            domino = self.trick_plays[i]
            rank = self._trick_rank(domino, led_suit)
            if rank > best_rank:
                best_rank = rank
                best_domino = domino
                best_player = (self.leader + i) % 4

        self.leader = best_player
        self.trick_plays = [-1, -1, -1, -1]
        self.trick_lead_domino = -1

    def _trick_rank(self, domino_id: int, led_suit: int) -> int:
        """
        Higher rank = wins the trick.
        Uses the same 6-bit (tier<<4 | rank) logic as game_tensor.TRICK_RANK_TABLE.

        Tier 2 = trump (called suit with trump power in decl 0-8)
        Tier 1 = follows led suit but not trump
        Tier 0 = does not follow (cannot win)

        Within tier 2 (trump):
          - doubles rank as 14 (highest)
          - non-doubles rank as high + low
        Within tier 1 (led suit):
          - doubles in their pip-suit rank as 14
          - non-doubles rank as high + low
          - in DOUBLES_TRUMP (decl 7) doubles-suit: rank by high pip
        """
        from forge.oracle.declarations import has_trump_power
        decl_id = self.decl_id
        high = DOMINO_HIGH[domino_id]
        low = DOMINO_LOW[domino_id]
        is_double = DOMINO_IS_DOUBLE[domino_id]
        domino_sum = high + low
        domino_suit = led_suit_for_lead_domino(domino_id, decl_id)
        is_trump = (domino_suit == 7)

        rank_in_pip_suit = 14 if is_double else domino_sum

        if is_trump and has_trump_power(decl_id):
            # Trump: tier 2
            if decl_id in PIP_TRUMP_IDS:
                rank = rank_in_pip_suit  # double=14 beats non-doubles
            elif decl_id == DOUBLES_TRUMP:
                rank = high  # ranked by high pip (6-6 > 5-5 > ...)
            else:
                rank = rank_in_pip_suit
            return (2 << 4) + rank
        else:
            # Check if can follow led suit
            can_follow_led: bool
            if led_suit == 7:
                can_follow_led = is_trump
            else:
                has_pip = (led_suit == high) or (led_suit == low)
                can_follow_led = has_pip and not is_trump

            if can_follow_led:
                if led_suit == 7:
                    # Following trump-as-led (NOTRUMP or fallback)
                    rank = high
                else:
                    rank = rank_in_pip_suit
                return (1 << 4) + rank
            else:
                return 0  # does not follow

    def clone(self) -> "GameState":
        """Deep copy for snapshot capture before applying a play."""
        return GameState(
            decl_id=self.decl_id,
            bidder=self.bidder,
            trump_set=set(self.trump_set),
            hands=[list(h) for h in self.hands],
            played_mask=list(self.played_mask),
            history=[list(e) for e in self.history],
            history_count=self.history_count,
            trick_plays=list(self.trick_plays),
            leader=self.leader,
            trick_lead_domino=self.trick_lead_domino,
            decision_idx=self.decision_idx,
        )


# --------------------------------------------------------------------------
# Suit utilities (pip-suit level, not trump-aware)
# --------------------------------------------------------------------------

def pip_suit_of(domino_id: int) -> int:
    """High-pip suit of a domino (0-6), ignoring trump context."""
    return DOMINO_HIGH[domino_id]


def low_suit_of(domino_id: int) -> int:
    return DOMINO_LOW[domino_id]


def distinct_off_suits_held(state: GameState, player: int) -> set[int]:
    """
    Return the set of non-trump suits (by high-pip) the player holds.
    A suit is 'off' if it's not the trump suit (led_suit != 7).
    """
    suits = set()
    for did in state.hands[player]:
        if did < 0:
            continue
        domino_suit = led_suit_for_lead_domino(did, state.decl_id)
        if domino_suit != 7:  # not trump
            suits.add(domino_suit)
    return suits


def trumps_held(state: GameState, player: int) -> list[int]:
    """Return domino IDs of trump dominoes still in player's hand."""
    return [
        did for did in state.hands[player]
        if did >= 0 and led_suit_for_lead_domino(did, state.decl_id) == 7
    ]


def suits_held(state: GameState, player: int) -> dict[int, list[int]]:
    """Return {suit: [domino_ids]} for all dominoes in player's hand."""
    result: dict[int, list[int]] = {}
    for did in state.hands[player]:
        if did < 0:
            continue
        s = led_suit_for_lead_domino(did, state.decl_id)
        result.setdefault(s, []).append(did)
    return result


def dominoes_remaining_in_suit(state: GameState, suit: int) -> list[int]:
    """Return unplayed domino IDs of a given suit (for any player)."""
    return [
        did for did in range(28)
        if not state.played_mask[did]
        and led_suit_for_lead_domino(did, state.decl_id) == suit
    ]


def dominant_trump(state: GameState) -> int:
    """Return the highest-ranking unplayed trump domino ID."""
    unplayed_trumps = [
        did for did in range(28)
        if not state.played_mask[did]
        and led_suit_for_lead_domino(did, state.decl_id) == 7
    ]
    if not unplayed_trumps:
        return -1
    # rank: double=114, else 100+sum
    def rank(did: int) -> int:
        if DOMINO_IS_DOUBLE[did]:
            return 114
        return 100 + DOMINO_HIGH[did] + DOMINO_LOW[did]
    return max(unplayed_trumps, key=rank)


# --------------------------------------------------------------------------
# Chunk loading
# --------------------------------------------------------------------------

def load_chunk(path: str) -> list[GameRecordGPU]:
    """Load a corpus .pt chunk; return list[GameRecordGPU]."""
    data = torch.load(path, map_location="cpu", weights_only=False)
    return data["results"]


# --------------------------------------------------------------------------
# Decision-level iterator
# --------------------------------------------------------------------------

@dataclass
class DecisionPoint:
    """A single decision point with full state context."""
    game_idx: int           # index in chunk
    decision_idx: int       # index in game.decisions
    player: int             # whose turn
    trick_number: int       # 0-indexed trick
    state_before: GameState # state BEFORE this play
    decision: DecisionRecordGPU
    game: GameRecordGPU

    def domino_played(self) -> int:
        """The domino ID actually played (greedy oracle choice)."""
        player = self.player
        slot = self.decision.action_taken
        return self.game.hands[player][slot]

    def greedy_domino(self) -> int:
        """The greedy oracle choice (max E[Q])."""
        player = self.player
        legal_mask = self.decision.legal_mask.tolist()
        e_q = self.decision.e_q.tolist()
        # For team 0 players (bidder side), max E[Q]. For team 1, min E[Q].
        team = player % 2  # 0 or 1; team 0 = bidder side
        best_slot = None
        best_val = None
        for slot, (legal, q) in enumerate(zip(legal_mask, e_q)):
            if not legal:
                continue
            val = q if team == 0 else -q
            if best_val is None or val > best_val:
                best_val = val
                best_slot = slot
        return self.game.hands[player][best_slot]


def iter_decisions(chunk_path: str) -> Iterator[DecisionPoint]:
    """
    Iterate over all decision points in a corpus chunk,
    yielding DecisionPoint with reconstructed game state before each play.
    """
    games = load_chunk(chunk_path)
    for game_idx, game in enumerate(games):
        state = GameState.initial(game)
        for dec_idx, decision in enumerate(game.decisions):
            player = decision.player
            trick_num = state.trick_number()
            n_trick = state.n_trick_plays()

            # Verify player matches expectation
            expected_player = state.current_player()
            if player != expected_player:
                # State diverged (edge case in corpus); skip this game
                break

            # Yield decision point with state snapshot BEFORE play
            dp = DecisionPoint(
                game_idx=game_idx,
                decision_idx=dec_idx,
                player=player,
                trick_number=trick_num,
                state_before=state.clone(),
                decision=decision,
                game=game,
            )
            yield dp

            # Apply the actual play to advance state
            domino_id = dp.domino_played()
            state.apply_play(player, domino_id)
            state.decision_idx = dec_idx + 1
