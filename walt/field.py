"""walt/field.py (builder B2) — the jud field oracle.

The `jud` play net as a deterministic, batched, memoized field σ for walt's
information-set solver. See walt/DESIGN.md (§Algorithm, §Beliefs) and
walt/contracts.py (§walt/field.py) for the frozen interface.

Everything here speaks DOMINO IDS (0..27). Game rules are never reimplemented;
led-suit / follow legality and trick scoring come from `forge.oracle.tables`.

Featurization contract (the load-bearing invariant): the 350-dim vector this
module builds is bit-identical (`torch.equal`) to
``champion.jud_net.featurize_state(state, seat=mover)`` on the CHILD (post-move)
state. The first 91 dims (hand ⊕ canonical auction) are child-invariant and are
computed ONCE per query with the reference torch functions; only the 259-dim
play block varies per legal move and is rebuilt in numpy, incrementally from the
parent's block (one 9-dim domino slice + the 7-dim global tail change).

Decision rule mirrors `arena.jud_play.JudPlay` exactly: value = mean_points of
the child logits, sign = +1 if mover is on the bidding team else -1, argmax of
sign*value, ties broken toward the lowest slot in the engine's hand ordering =
the lowest domino id among the seat's remaining sorted hand (np.argmax /
torch.argmax first-max semantics).
"""
from __future__ import annotations

from dataclasses import dataclass, field as _dc_field
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
import torch

from champion.bid_net import featurize_hand
from champion.jud_net import load_jud_net
from champion.margin_net import CANON_BID, mean_points
from forge.oracle.tables import (
    can_follow,
    led_suit_for_lead_domino,
    resolve_trick,
)
from gus.model.auction import auction_feature_vector
from champion.margin_net import canonical_auction

from walt.tables import get_luts, legal_moves_mask

# --------------------------------------------------------------------- #
#  Layout constants (mirror champion.jud_net)                            #
# --------------------------------------------------------------------- #

N_DOMINOES = 28
PER_DOMINO = 9
GLOBAL_DIM = 7
PLAY_DIM = N_DOMINOES * PER_DOMINO + GLOBAL_DIM   # 259
HAND_DIM = 63
AUCTION_DIM = 28
FEATURE_DIM = HAND_DIM + AUCTION_DIM + PLAY_DIM   # 350
N_TRICKS = 7
_GLOBAL_OFF = N_DOMINOES * PER_DOMINO             # 252

_DEFAULT_NET = Path(
    "/Users/jason/code/mk5-main/.claude/worktrees/walt/champion/jud_net.pt"
)


# --------------------------------------------------------------------- #
#  Play-block featurization (numpy, bit-parity with jud_net.featurize_play)
# --------------------------------------------------------------------- #

def _featurize_play_np(
    plays: Sequence[tuple[int, int]], seat: int, bidder: int, decl_id: int,
) -> tuple[np.ndarray, tuple[int, int]]:
    """[259] play block for POV ``seat`` + the (declaring, defending) banked
    points after complete tricks. Scalar tail values are written as Python
    float64 → float32 (the exact double-rounding torch does on tensor store), so
    the result is bit-identical to `champion.jud_net.featurize_play`."""
    feat = np.zeros(PLAY_DIM, dtype=np.float32)
    pts = [0, 0]  # (declaring team, defending team)
    bid_team = int(bidder) % 2
    for k, (player, domino) in enumerate(plays):
        base = PER_DOMINO * int(domino)
        feat[base + (int(player) - int(seat)) % 4] = 1.0
        feat[base + 4 + k % 4] = 1.0
        feat[base + 8] = (k // 4) / (N_TRICKS - 1)
        if k % 4 == 3:
            trick = tuple(int(d) for _, d in plays[k - 3:k + 1])
            out = resolve_trick(trick[0], trick, int(decl_id))
            winner = (int(plays[k - 3][0]) + out.winner_offset) % 4
            pts[0 if winner % 2 == bid_team else 1] += out.points
    n = len(plays)
    g = _GLOBAL_OFF
    feat[g + 0] = pts[0] / 42.0
    feat[g + 1] = pts[1] / 42.0
    feat[g + 2] = n / float(N_DOMINOES)
    feat[g + 3 + n % 4] = 1.0
    return feat, (pts[0], pts[1])


def _child_play_block(
    parent_block: np.ndarray,
    parent_pts: tuple[int, int],
    parent_hist: Sequence[tuple[int, int]],
    seat: int,
    move: int,
    bidder: int,
    decl_id: int,
) -> np.ndarray:
    """Child play block = parent block with the mover's play at ``move`` folded
    in. Identical (bit-for-bit) to ``_featurize_play_np(parent_hist + [(seat,
    move)], seat, ...)`` but O(1) instead of O(len(hist))."""
    child = parent_block.copy()
    k = len(parent_hist)          # index of the new play
    base = PER_DOMINO * int(move)
    child[base + 0] = 1.0         # (seat - seat) % 4 == 0 (mover is POV)
    child[base + 4 + k % 4] = 1.0
    child[base + 8] = (k // 4) / (N_TRICKS - 1)

    g = _GLOBAL_OFF
    n0, n = k, k + 1
    child[g + 3 + n0 % 4] = 0.0   # clear the parent's trick-fill one-hot
    child[g + 3 + n % 4] = 1.0
    if k % 4 == 3:                # this move completes a trick → rescore it
        trick = tuple(int(d) for _, d in parent_hist[k - 3:k]) + (int(move),)
        out = resolve_trick(trick[0], trick, int(decl_id))
        winner = (int(parent_hist[k - 3][0]) + out.winner_offset) % 4
        pts = [parent_pts[0], parent_pts[1]]
        pts[0 if winner % 2 == int(bidder) % 2 else 1] += out.points
        child[g + 0] = pts[0] / 42.0
        child[g + 1] = pts[1] / 42.0
    child[g + 2] = n / float(N_DOMINOES)
    return child


def _hand_auction_91(
    seat: int, orig_hand: Sequence[int], bids, bidder: int, dealer: int, decl_id: int,
) -> np.ndarray:
    """[91] child-invariant hand (63) ⊕ canonical auction (28), via the exact
    reference torch functions → float32 numpy (bit-identical for free)."""
    hand = featurize_hand(tuple(int(d) for d in orig_hand))          # [63] torch
    canon = canonical_auction(bids, int(bidder), int(dealer))
    auc = auction_feature_vector(
        canon, bidder=int(bidder), bid_value=CANON_BID,
        decl_id=int(decl_id), current_player=int(seat),
    )                                                                # [28] torch
    return torch.cat([hand, auc]).numpy()


def featurize(
    seat: int, orig_hand: Sequence[int], bids, bidder: int, dealer: int,
    decl_id: int, plays: Sequence[tuple[int, int]],
) -> np.ndarray:
    """Full [350] info-state feature for ``seat`` — the non-incremental
    reference-equivalent of `champion.jud_net.featurize_state(state, seat)`.

    ``orig_hand`` is the seat's ORIGINAL (pre-play) hand; which tiles are gone
    lives in ``plays`` (the play block), exactly as the engine featurizer does."""
    h91 = _hand_auction_91(seat, orig_hand, bids, bidder, dealer, decl_id)
    pblock, _ = _featurize_play_np(plays, int(seat), int(bidder), int(decl_id))
    return np.concatenate([h91, pblock])


# --------------------------------------------------------------------- #
#  LUT-fast incremental node context (the solver's hot path)            #
# --------------------------------------------------------------------- #

_INV1 = np.zeros(1, dtype=np.int64)   # unique-inverse for size-1 fast path


def _bits_list(mask: int) -> list[int]:
    """Domino ids set in a bitmask, ascending."""
    out = []
    m = int(mask)
    while m:
        b = m & -m
        out.append(b.bit_length() - 1)
        m ^= b
    return out


def resolve_lut(trick4, luts) -> tuple[int, int]:
    """(winner_offset, points) for a completed 4-tile trick, via walt LUTs.

    Identical to forge.oracle.tables.resolve_trick (first-max wins ties;
    points = 1 + riding count) — parity guaranteed because the LUTs are
    tabulated from trick_rank/count themselves (tables.py gates)."""
    led_suit = int(luts.led_suit[trick4[0]])
    r = luts.rank[led_suit]
    best_off, best_rank, pts = 0, int(r[trick4[0]]), 1 + int(luts.count[trick4[0]])
    for off in (1, 2, 3):
        t = trick4[off]
        pts += int(luts.count[t])
        rk = int(r[t])
        if rk > best_rank:
            best_off, best_rank = off, rk
    return best_off, pts


class NodeCtx:
    """Everything about a public history the field needs, carried
    incrementally down the solver's recursion instead of replayed per node:

    - ``hist``: the play history tuple (memo-key identity),
    - ``blocks``: (4, 259) float32 — the jud play block for each POV seat,
      bit-identical to `_featurize_play_np(hist, pov, ...)` row-by-row,
    - ``pts``: banked (declaring, defending) points from complete tricks,
    - ``played``: per-seat bitmask of tiles that seat has played.

    ``advance`` is O(1) and LAZY: it records a delta; the (4, 259) block
    array only materializes (nearest materialized ancestor + replayed
    deltas) when a memo-missing field query actually needs feature rows.
    Memo-hit-heavy subtrees never touch the arrays at all. Parity with the
    reference featurizer is covered by test_field gate1 through the
    `featurize()`/`decisions()` path and by the shared-memo equivalence of
    `decisions_at`.
    """

    __slots__ = ("hist", "pts", "played", "_blocks", "_parent", "_delta")

    def __init__(self, hist, pts, played, blocks=None, parent=None, delta=None):
        self.hist = hist
        self.pts = pts
        self.played = played
        self._blocks = blocks
        self._parent = parent
        self._delta = delta

    @classmethod
    def from_history(cls, plays, bidder: int, luts) -> "NodeCtx":
        hist = tuple((int(p), int(d)) for p, d in plays)
        blocks = np.zeros((4, PLAY_DIM), dtype=np.float32)
        pts = [0, 0]
        bid_team = int(bidder) % 2
        played = [0, 0, 0, 0]
        for k, (player, domino) in enumerate(hist):
            base = PER_DOMINO * domino
            for pov in range(4):
                blocks[pov, base + (player - pov) % 4] = 1.0
            blocks[:, base + 4 + k % 4] = 1.0
            blocks[:, base + 8] = np.float32((k // 4) / (N_TRICKS - 1))
            played[player] |= 1 << domino
            if k % 4 == 3:
                trick = tuple(d for _, d in hist[k - 3:k + 1])
                off, p = resolve_lut(trick, luts)
                winner = (hist[k - 3][0] + off) % 4
                pts[0 if winner % 2 == bid_team else 1] += p
        n = len(hist)
        g = _GLOBAL_OFF
        blocks[:, g + 0] = np.float32(pts[0] / 42.0)
        blocks[:, g + 1] = np.float32(pts[1] / 42.0)
        blocks[:, g + 2] = np.float32(n / float(N_DOMINOES))
        blocks[:, g + 3 + n % 4] = 1.0
        return cls(hist, (pts[0], pts[1]), tuple(played), blocks=blocks)

    def advance(self, seat: int, tile: int, bidder: int, luts) -> "NodeCtx":
        k = len(self.hist)
        pts = self.pts
        pts_after = None
        if k % 4 == 3:
            trick = tuple(d for _, d in self.hist[k - 3:]) + (tile,)
            off, p = resolve_lut(trick, luts)
            winner = (self.hist[k - 3][0] + off) % 4
            if winner % 2 == int(bidder) % 2:
                pts = (pts[0] + p, pts[1])
            else:
                pts = (pts[0], pts[1] + p)
            pts_after = pts
        played = list(self.played)
        played[seat] |= 1 << tile
        return NodeCtx(
            self.hist + ((seat, tile),), pts, tuple(played),
            parent=self, delta=(seat, tile, k, pts_after),
        )

    @property
    def blocks(self) -> np.ndarray:
        if self._blocks is not None:
            return self._blocks
        chain = []
        node = self
        while node._blocks is None:
            chain.append(node)
            node = node._parent
        blocks = node._blocks.copy()
        g = _GLOBAL_OFF
        for c in reversed(chain):
            seat, tile, k, pts_after = c._delta
            base = PER_DOMINO * tile
            for pov in range(4):
                blocks[pov, base + (seat - pov) % 4] = 1.0
            blocks[:, base + 4 + k % 4] = 1.0
            blocks[:, base + 8] = np.float32((k // 4) / (N_TRICKS - 1))
            blocks[:, g + 3 + k % 4] = 0.0
            blocks[:, g + 3 + (k + 1) % 4] = 1.0
            blocks[:, g + 2] = np.float32((k + 1) / float(N_DOMINOES))
            if pts_after is not None:
                blocks[:, g + 0] = np.float32(pts_after[0] / 42.0)
                blocks[:, g + 1] = np.float32(pts_after[1] / 42.0)
        self._blocks = blocks       # cache at self; drop the lazy chain refs
        self._parent = None
        self._delta = None
        return blocks


# --------------------------------------------------------------------- #
#  Public state a field decision conditions on                          #
# --------------------------------------------------------------------- #

def _mask_of(ids) -> int:
    m = 0
    for d in ids:
        m |= 1 << int(d)
    return m


def _replay_public(
    play_history: Sequence[tuple[int, int]], bidder: int, decl_id: int,
) -> tuple[tuple, int, tuple[int, int]]:
    """Derive (current_trick, trick_leader, team_points) from a play prefix.

    Used to rebuild a `PubState` from a bare history (the σ-consistency
    replay); the engine keeps these fields live, so `PubState.from_state`
    reads them directly instead."""
    leader = int(bidder)             # bidder leads trick 1
    pts = [0, 0]
    bid_team = int(bidder) % 2
    trick: list[int] = []
    trick_start_leader = leader
    for k, (player, domino) in enumerate(play_history):
        if not trick:
            trick_start_leader = int(player)
        trick.append(int(domino))
        if len(trick) == 4:
            out = resolve_trick(trick[0], tuple(trick), int(decl_id))
            winner = (trick_start_leader + out.winner_offset) % 4
            pts[0 if winner % 2 == bid_team else 1] += out.points
            leader = winner
            trick = []
    current_trick = tuple(trick)
    trick_leader = trick_start_leader if current_trick else leader
    return current_trick, trick_leader, (pts[0], pts[1])


@dataclass(frozen=True)
class PubState:
    """Public state a field decision conditions on. ``key`` is a precomputed
    hashable identity sufficient to determine the decision together with the
    acting seat and its hand: everything else (current trick, banked points) is
    a function of ``play_history``."""

    decl_id: int
    bidder: int
    bids: tuple
    dealer: int
    play_history: tuple
    trick_leader: int
    current_trick: tuple
    team_points: tuple
    key: tuple = _dc_field(default=(), compare=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "key",
            (int(self.decl_id), int(self.bidder), tuple(int(b) for b in self.bids),
             int(self.dealer), tuple((int(p), int(d)) for p, d in self.play_history)),
        )

    @classmethod
    def from_state(cls, state) -> "PubState":
        return cls(
            decl_id=int(state.decl_id),
            bidder=int(state.bidder),
            bids=tuple(int(b) for b in state.bid_state.bids),
            dealer=int(state.dealer),
            play_history=tuple((int(p), int(d)) for p, d in state.play_history),
            trick_leader=int(state.trick_leader),
            current_trick=tuple(int(d) for d in state.current_trick),
            team_points=tuple(int(x) for x in state.team_points),
        )

    @classmethod
    def from_history(
        cls, decl_id: int, bidder: int, bids, dealer: int,
        play_history: Sequence[tuple[int, int]],
    ) -> "PubState":
        hist = tuple((int(p), int(d)) for p, d in play_history)
        cur, leader, pts = _replay_public(hist, int(bidder), int(decl_id))
        return cls(
            decl_id=int(decl_id), bidder=int(bidder),
            bids=tuple(int(b) for b in bids), dealer=int(dealer),
            play_history=hist, trick_leader=leader,
            current_trick=cur, team_points=pts,
        )


# --------------------------------------------------------------------- #
#  The field oracle                                                     #
# --------------------------------------------------------------------- #

class FieldOracle:
    """jud argmax as a deterministic field, batched + memoized."""

    def __init__(self, net_path: str | Path = _DEFAULT_NET, device: str = "cpu"):
        self.device = device
        self.model = load_jud_net(net_path, device=device)  # returns model.eval()
        self.model.eval()
        # functional forward: the same Linear/ReLU/Linear/ReLU/Linear +
        # softmax·arange ops as JudNet.forward + mean_points (bit-identical —
        # nn.Linear IS F.linear), minus per-call Module dispatch overhead.
        L = self.model.net
        self._w = [(L[0].weight, L[0].bias), (L[2].weight, L[2].bias),
                   (L[4].weight, L[4].bias)]
        self._pts = torch.arange(43, dtype=torch.float32, device=device)
        # memo: (seat, hand_mask, pub.key) -> domino id jud plays
        self.memo: dict[tuple, int] = {}
        # (seat, orig_hand_mask, bids, bidder, dealer, decl_id) -> [91] float32
        self._h91_cache: dict[tuple, np.ndarray] = {}
        self.n_forward = 0
        self.n_rows = 0

    # -- child-invariant 91-dim block, cached ------------------------------
    def _h91(self, seat: int, orig_mask: int, bids, bidder: int, dealer: int,
             decl_id: int) -> np.ndarray:
        key = (seat, orig_mask, bids, bidder, dealer, decl_id)
        got = self._h91_cache.get(key)
        if got is None:
            if len(self._h91_cache) > 500_000:   # ~180MB cap; clear-all is fine
                self._h91_cache.clear()
            orig_hand = [d for d in range(N_DOMINOES) if (orig_mask >> d) & 1]
            got = _hand_auction_91(seat, orig_hand, bids, bidder, dealer, decl_id)
            self._h91_cache[key] = got
        return got

    def _forget_if_huge(self) -> None:
        if len(self.memo) > 4_000_000:
            self.memo.clear()

    def _ev(self, X: np.ndarray) -> np.ndarray:
        """[B] E[declaring pts] — functional JudNet.forward + mean_points."""
        F = torch.nn.functional
        with torch.no_grad():
            x = torch.from_numpy(X)
            if self.device != "cpu":
                x = x.to(self.device)
            h = F.relu(F.linear(x, *self._w[0]))
            h = F.relu(F.linear(h, *self._w[1]))
            probs = torch.softmax(F.linear(h, *self._w[2]), dim=-1)
            ev = (probs * self._pts).sum(dim=-1)
        return ev.cpu().numpy() if self.device != "cpu" else ev.numpy()

    # -- vectorized single-node decisions (the solver's hot path) ----------
    def decisions_at(self, seat: int, ctx: NodeCtx, hand_masks,
                     current_trick, auction, luts) -> np.ndarray:
        """jud's move for every hand mask in ``hand_masks``, all sharing the
        same acting ``seat`` and public context ``ctx``.

        ``auction`` = (decl_id, bidder, bids_tuple, dealer). Worlds sharing
        the acting seat's hand are deduped (np.unique); memo keys are
        content-identical to `decisions()`'s so both paths share one memo.
        Returns an int64 array aligned with ``hand_masks``."""
        hm = np.asarray(hand_masks, dtype=np.uint32).reshape(-1)
        if hm.size == 0:
            return np.empty(0, dtype=np.int64)
        decl_id, bidder, bids, dealer = auction
        # dedup in plain Python — node batches are small; np.unique overhead
        # dominates at this size
        vals = hm.tolist()
        if len(vals) == 1:
            uniq_list, inv = vals, None
        else:
            index: dict[int, int] = {}
            inv_l = []
            uniq_list = []
            for v in vals:
                i = index.get(v)
                if i is None:
                    i = len(uniq_list)
                    index[v] = i
                    uniq_list.append(v)
                inv_l.append(i)
            inv = None if len(uniq_list) == len(vals) and len(uniq_list) == 1 \
                else np.asarray(inv_l, dtype=np.int64)
        pubkey = (decl_id, bidder, bids, dealer, ctx.hist)
        out = np.empty(len(uniq_list), dtype=np.int64)

        if current_trick:
            fb = int(luts.can_follow_bits[luts.led_suit[int(current_trick[0])]])
        else:
            fb = None
        sign = 1.0 if seat % 2 == bidder % 2 else -1.0
        parent_block = None          # ctx.blocks materializes on first miss
        k = len(ctx.hist)
        completes = (k % 4 == 3)
        if completes:
            trick3 = tuple(d for _, d in ctx.hist[k - 3:])
            t3_leader = ctx.hist[k - 3][0]
        g = _GLOBAL_OFF

        rows: list[np.ndarray] = []
        misses: list[tuple[int, tuple, list[int], int]] = []
        for i, h in enumerate(uniq_list):
            key = (seat, h, pubkey)
            got = self.memo.get(key)
            if got is not None:
                out[i] = got
                continue
            if fb is None:
                lmask = h
            else:
                followers = h & fb
                lmask = followers if followers else h
            legal = _bits_list(lmask)
            if len(legal) == 1:
                self.memo[key] = legal[0]
                out[i] = legal[0]
                continue
            orig = h | ctx.played[seat]
            h91 = self._h91(seat, orig, bids, bidder, dealer, decl_id)
            if parent_block is None:
                parent_block = ctx.blocks[seat]
            start = len(rows)
            for a in legal:
                row = np.empty(FEATURE_DIM, dtype=np.float32)
                row[:HAND_DIM + AUCTION_DIM] = h91
                child = row[HAND_DIM + AUCTION_DIM:]
                child[:] = parent_block
                base = PER_DOMINO * a
                child[base + 0] = 1.0            # mover is POV
                child[base + 4 + k % 4] = 1.0
                child[base + 8] = np.float32((k // 4) / (N_TRICKS - 1))
                child[g + 3 + k % 4] = 0.0
                child[g + 3 + (k + 1) % 4] = 1.0
                child[g + 2] = np.float32((k + 1) / float(N_DOMINOES))
                if completes:
                    off, p = resolve_lut(trick3 + (a,), luts)
                    winner = (t3_leader + off) % 4
                    pts = list(ctx.pts)
                    pts[0 if winner % 2 == bidder % 2 else 1] += p
                    child[g + 0] = np.float32(pts[0] / 42.0)
                    child[g + 1] = np.float32(pts[1] / 42.0)
                rows.append(row)
            misses.append((i, key, legal, start))

        if rows:
            ev = self._ev(np.stack(rows))
            self.n_forward += 1
            self.n_rows += len(rows)
            for i, key, legal, start in misses:
                best = int(np.argmax(sign * ev[start:start + len(legal)]))
                move = legal[best]
                self.memo[key] = move
                out[i] = move
            self._forget_if_huge()
        return out if inv is None else out[inv]

    # -- legality (rules tabulated from forge.oracle.tables) --------------
    def _legal(self, hand_mask: int, pub: PubState) -> list[int]:
        hand = [d for d in range(N_DOMINOES) if (hand_mask >> d) & 1]  # ascending
        if not pub.current_trick:
            return hand
        led = int(pub.current_trick[0])
        led_suit = led_suit_for_lead_domino(led, int(pub.decl_id))
        followers = [d for d in hand if can_follow(d, led_suit, int(pub.decl_id))]
        return followers if followers else hand

    def decisions(self, queries: Sequence[tuple[int, int, PubState]]) -> list[int]:
        """For each (seat, hand_mask, pub) return the domino id jud plays.

        Memoized on (seat, hand_mask, pub.key). All memo-miss queries have every
        legal child featurized in numpy and scored in ONE torch forward."""
        results: list[Optional[int]] = [None] * len(queries)
        rows: list[np.ndarray] = []
        # per miss: (result_index, memo_key, legal, sign, row_slice_start)
        misses: list[tuple[int, tuple, list[int], float, int]] = []

        for qi, (seat, hand_mask, pub) in enumerate(queries):
            seat = int(seat)
            hand_mask = int(hand_mask)
            key = (seat, hand_mask, pub.key)
            memoed = self.memo.get(key)
            if memoed is not None:
                results[qi] = memoed
                continue
            legal = self._legal(hand_mask, pub)
            if len(legal) == 1:
                self.memo[key] = legal[0]
                results[qi] = legal[0]
                continue
            orig_mask = hand_mask | _played_by(pub.play_history, seat)
            h91 = self._h91(
                seat, orig_mask, pub.bids, pub.bidder, pub.dealer, pub.decl_id
            )
            pblock, ppts = _featurize_play_np(
                pub.play_history, seat, pub.bidder, pub.decl_id
            )
            start = len(rows)
            for a in legal:
                cblock = _child_play_block(
                    pblock, ppts, pub.play_history, seat, a, pub.bidder, pub.decl_id
                )
                rows.append(np.concatenate([h91, cblock]))
            sign = 1.0 if seat % 2 == int(pub.bidder) % 2 else -1.0
            misses.append((qi, key, legal, sign, start))

        if rows:
            ev = self._ev(np.stack(rows))
            self.n_forward += 1
            self.n_rows += len(rows)
            for qi, key, legal, sign, start in misses:
                vals = sign * ev[start:start + len(legal)]
                best = int(np.argmax(vals))     # first-max == lowest domino id
                move = legal[best]
                self.memo[key] = move
                results[qi] = move
        return results  # type: ignore[return-value]


def _played_by(play_history: Sequence[tuple[int, int]], seat: int) -> int:
    m = 0
    for p, d in play_history:
        if int(p) == int(seat):
            m |= 1 << int(d)
    return m


# --------------------------------------------------------------------- #
#  Exact B(σ) filter for a deterministic field                          #
# --------------------------------------------------------------------- #

def sigma_consistent(
    root,
    worlds: np.ndarray,
    oracle: FieldOracle,
    moves_filter: Callable[[int, int], bool],
) -> np.ndarray:
    """Boolean mask over ``worlds`` (rows): keep a world iff σ, given that
    world's hypothetical hands, would have produced every observed non-``me``
    move at a step ``moves_filter(seat, k)`` selects True.

    ``worlds`` is (N, 3) uint32 CURRENT-remaining hand masks for the three
    non-me seats in ascending absolute seat order (walt/worlds.py contract).
    Each filtered history step batches all still-alive worlds into ONE oracle
    call (which is itself one forward)."""
    me = int(root.me)
    hidden = [s for s in range(4) if s != me]          # ascending
    col = {s: i for i, s in enumerate(hidden)}
    hist = tuple((int(p), int(d)) for p, d in root.play_history)
    N = int(worlds.shape[0])
    if N == 0:
        return np.zeros(0, dtype=bool)

    decl_id = int(root.decl_id)
    bidder = int(root.bidder)
    bids = tuple(int(b) for b in root.bids)
    dealer = int(root.dealer)
    auction = (decl_id, bidder, bids, dealer)
    luts = get_luts(decl_id)

    # original hand mask per world per hidden seat = current remaining | all
    # the seat's plays across the observed history (vectorized uint32).
    worlds = np.asarray(worlds, dtype=np.uint32).reshape(N, 3)
    orig = {
        s: worlds[:, col[s]] | np.uint32(_played_by(hist, s)) for s in hidden
    }

    alive = np.ones(N, dtype=bool)
    played_before = {s: 0 for s in range(4)}           # cumulative, per seat
    ctx = NodeCtx.from_history((), bidder, luts)
    trick: list[int] = []
    for k, (seat, tile) in enumerate(hist):
        if (seat != me) and bool(moves_filter(seat, k)):
            live_idx = np.nonzero(alive)[0]
            if live_idx.size:
                not_before = np.uint32((~played_before[seat]) & 0xFFFFFFFF)
                hands_k = orig[seat][live_idx] & not_before
                moves = oracle.decisions_at(
                    seat, ctx, hands_k, tuple(trick), auction, luts
                )
                alive[live_idx] &= moves == tile
        # advance the observed play
        played_before[seat] |= 1 << int(tile)
        trick.append(int(tile))
        if len(trick) == 4:
            trick = []
        ctx = ctx.advance(seat, int(tile), bidder, luts)
    return alive
