"""Sparse external-sampling regret solving on Metal.

This is the bounded-memory vertical slice for full-metal Hoyt. A traversal
samples one exact hidden world, enumerates every action of the updating seat,
and samples one current-policy action at the other seats. The public tree is
never materialized: each microbatch holds only its current frontier and the
parent maps required for one backward pass.

The same traversal substrate supports two modes. ``SampledCFR`` alternates the
updating seat through a shared sparse regret table to produce a four-seat
candidate policy. ``SampledBR`` either starts from uniform opponents or forks
that table, resets one seat, and learns a candidate best response while the
other three seats stay frozen. Information sets live in a sorted sparse table
keyed by seat plus a 62-bit fingerprint, so there are no direct-hash bucket
collisions; the table also has a hard capacity and fails closed. The table is
grown between frozen-policy microbatches; this first slice indexes sparse
update records on the host before the reduced regret update returns to Metal.
Moving that sort/merge onto the GPU is an explicit remaining production step.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path

import numpy as np

from hoyt.subgame import build_subgame
from hoyt.worldsample import WorldSampler

try:
    import mlx.core as mx
except ImportError as exc:  # pragma: no cover
    mx = None
    _MLX_ERROR = exc
else:
    _MLX_ERROR = None

__all__ = [
    "SampledBR", "SampledBRResult", "SampledCFR", "SampledGapResult",
    "SparsePolicy", "audit_sampled_gap",
]

_STRIDE = 12
_PH0 = np.uint64(0xCBF29CE484222325)
_PH_PRIME = 0x100000001B3
_MASK64 = (1 << 64) - 1
_MASK62 = (1 << 62) - 1


def _mix64(value: int) -> int:
    value &= _MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _MASK64
    return value ^ (value >> 31)


def _info_key(seat: int, hand: int, path) -> np.uint64:
    ph = int(_PH0)
    for tile in path:
        ph = (ph * _PH_PRIME + int(tile) + 1) & _MASK64
    fingerprint = _mix64(ph ^ (int(hand) << 17)) & _MASK62
    return np.uint64((int(seat) << 62) | fingerprint)


def _root_signature(root) -> str:
    return hashlib.sha256(repr(root).encode("utf-8")).hexdigest()


def _bounded_mean_error(samples, low: float, high: float,
                        alpha: float) -> float:
    """Two-sided empirical-Bernstein radius for bounded iid observations."""
    x = np.asarray(samples, dtype=np.float64).reshape(-1)
    if len(x) < 2:
        raise ValueError("bounded mean interval needs at least two samples")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between zero and one")
    width = float(high) - float(low)
    if width < 0.0 or not np.isfinite(width):
        raise ValueError("payoff bounds must be finite and ordered")
    logterm = math.log(2.0 / alpha)
    variance = float(x.var(ddof=1))
    return math.sqrt(2.0 * variance * logterm / len(x)) \
        + 7.0 * width * logterm / (3.0 * (len(x) - 1))


_COMMON = r"""
inline ulong mix64(ulong z) {
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ul;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBul;
    return z ^ (z >> 31);
}

inline uint tile_at_rank(uint mask, uint rank) {
    for (uint t = 0; t < 28; ++t) {
        if (mask & (1u << t)) {
            if (rank == 0u) return t;
            rank -= 1u;
        }
    }
    return 31u;
}

inline ulong path_of(const device uint* state, uint i) {
    return ((ulong)state[i * 12u + 11u] << 32)
         | (ulong)state[i * 12u + 10u];
}

inline ulong info_key(const device uint* state, uint i, uint actor) {
    uint hand = state[i * 12u + actor];
    ulong fingerprint = mix64(path_of(state, i) ^ ((ulong)hand << 17));
    return ((ulong)actor << 62)
         | (fingerprint & 0x3FFFFFFFFFFFFFFFul);
}

inline uint legal_of(const device uint* state, uint i, uint actor,
                     const device uint* cfb) {
    uint hand = state[i * 12u + actor];
    uint ledp = state[i * 12u + 5u];
    if (ledp == 0u) return hand;
    uint follow = hand & cfb[ledp - 1u];
    return follow ? follow : hand;
}
"""

_COUNT_SOURCE = r"""
    uint i = thread_position_in_grid.x;
    uint n = state_shape[0] / 12u;
    if (i >= n) return;
    uint leader = state[i * 12u + 4u];
    uint actor = (leader + pos[0]) & 3u;
    uint legal = legal_of(state, i, actor, cfb);
    count[i] = actor == updater[0] ? popcount(legal) : 1u;
"""

_EXPAND_SOURCE = r"""
    uint i = thread_position_in_grid.x;
    uint n = state_shape[0] / 12u;
    if (i >= n) return;
    uint leader = state[i * 12u + 4u];
    uint actor = (leader + pos[0]) & 3u;
    uint legal = legal_of(state, i, actor, cfb);
    uint nlegal = popcount(legal);
    uint first = offset[i];
    uint nchild = offset[i + 1u] - first;
    uint chosen = 0u;
    ulong ikey = info_key(state, i, actor);
    uint lo = 0u;
    uint hi = nkeys[0];
    while (lo < hi) {
        uint mid = lo + ((hi - lo) >> 1);
        ulong probe = key_table[mid];
        if (probe < ikey) lo = mid + 1u;
        else hi = mid;
    }
    uint row = lo < nkeys[0] && key_table[lo] == ikey
               ? lo : 0xFFFFFFFFu;

    float total_pos = 0.0f;
    if (row != 0xFFFFFFFFu) {
        for (uint r = 0; r < nlegal; ++r)
            total_pos += max(regret[row * 7u + r], 0.0f);
    }

    if (actor != updater[0]) {
        // Independent counter stream per frontier parent. Path/hand keeps
        // branches reproducible; i prevents a whole microbatch at the same
        // information set from sharing one opponent draw.
        ulong z = mix64(info_key(state, i, actor) ^ seed[0]
                        ^ ((ulong)depth[0] << 48)
                        ^ ((ulong)i * 0xD1B54A32D192ED03ul));
        if (total_pos > 0.0f) {
            float draw = (float)((z >> 40) & 0xFFFFFFul)
                       * (total_pos / 16777216.0f);
            float cumulative = 0.0f;
            chosen = nlegal - 1u;
            for (uint r = 0; r < nlegal; ++r) {
                cumulative += max(regret[row * 7u + r], 0.0f);
                if (draw < cumulative) {
                    chosen = r;
                    break;
                }
            }
        } else {
            chosen = (uint)(z % (ulong)nlegal);
        }
    }

    for (uint q = 0; q < nchild; ++q) {
        uint rank = actor == updater[0] ? q : chosen;
        uint tile = tile_at_rank(legal, rank);
        uint ci = first + q;
        for (uint x = 0; x < 12u; ++x)
            child[ci * 12u + x] = state[i * 12u + x];
        child[ci * 12u + actor] &= ~(1u << tile);

        uint ledp = state[i * 12u + 5u];
        uint brankp = state[i * 12u + 6u];
        uint bseatp = state[i * 12u + 7u];
        uint tcnt = state[i * 12u + 8u];
        uint points = state[i * 12u + 9u];
        if (pos[0] == 0u) {
            uint led = led_suit[tile];
            child[ci * 12u + 5u] = led + 1u;
            child[ci * 12u + 6u] = rank_lut[led * 28u + tile] + 1u;
            child[ci * 12u + 7u] = actor + 1u;
            child[ci * 12u + 8u] = count_lut[tile];
        } else if (pos[0] < 3u) {
            uint led = ledp - 1u;
            uint rr = rank_lut[led * 28u + tile];
            if (rr + 1u > brankp) {
                child[ci * 12u + 6u] = rr + 1u;
                child[ci * 12u + 7u] = actor + 1u;
            }
            child[ci * 12u + 8u] = tcnt + count_lut[tile];
        } else {
            uint led = ledp - 1u;
            uint rr = rank_lut[led * 28u + tile];
            uint winner = rr + 1u > brankp ? actor : bseatp - 1u;
            uint award = tcnt + count_lut[tile] + 1u;
            if ((winner & 1u) == bid_team[0]) points += award;
            child[ci * 12u + 4u] = winner;
            child[ci * 12u + 5u] = 0u;
            child[ci * 12u + 6u] = 0u;
            child[ci * 12u + 7u] = 0u;
            child[ci * 12u + 8u] = 0u;
            child[ci * 12u + 9u] = points;
        }

        ulong ph = path_of(state, i) * 0x100000001B3ul + (ulong)tile + 1ul;
        child[ci * 12u + 10u] = (uint)ph;
        child[ci * 12u + 11u] = (uint)(ph >> 32);
        parent[ci] = i;
        action_rank[ci] = rank;
        if (actor == updater[0]) {
            float rp = row == 0xFFFFFFFFu ? 0.0f
                       : max(regret[row * 7u + rank], 0.0f);
            probability[ci] = total_pos > 0.0f ? rp / total_pos
                                                : 1.0f / (float)nlegal;
        } else {
            probability[ci] = 1.0f;
        }
    }
"""

_BACKWARD_SOURCE = r"""
    uint i = thread_position_in_grid.x;
    uint n = state_shape[0] / 12u;
    if (i >= n) return;
    uint leader = state[i * 12u + 4u];
    uint actor = (leader + pos[0]) & 3u;
    uint a = offset[i];
    uint b = offset[i + 1u];
    float value = 0.0f;
    for (uint k = a; k < b; ++k) value += probability[k] * vchild[k];
    vparent[i] = value;

    uint e0 = i * 7u;
    for (uint r = 0; r < 7u; ++r) {
        event_index[e0 + r] = 0u;
        event_delta[e0 + r] = 0.0f;
        event_key_lo[e0 + r] = 0u;
        event_key_hi[e0 + r] = 0u;
    }
    if (actor == updater[0]) {
        ulong key = info_key(state, i, actor);
        for (uint k = a; k < b; ++k) {
            uint r = k - a;
            event_index[e0 + r] = r;
            event_delta[e0 + r] = sign[0] * (vchild[k] - value);
            event_key_lo[e0 + r] = (uint)key;
            event_key_hi[e0 + r] = (uint)(key >> 32);
        }
    }
"""

_REDUCE_SOURCE = r"""
    uint i = thread_position_in_grid.x;
    if (i >= event_delta_shape[0]) return;
    float z = event_delta[i];
    if (z != 0.0f)
        atomic_fetch_add_explicit(&dense[event_index[i]], z,
                                  memory_order_relaxed);
"""

_LEAF_SOURCE = r"""
    uint i = thread_position_in_grid.x;
    uint n = state_shape[0] / 12u;
    if (i >= n) return;
    value[i] = payoff[state[i * 12u + 9u]];
"""


@dataclass
class SampledBRResult:
    value: float
    value_se: float
    value_error: float
    root_values: dict[int, float]
    root_value_se: dict[int, float]
    root_value_error: dict[int, float]
    best_move: int | None
    epochs: int
    samples: int
    table_capacity: int
    occupied_buckets: int
    key_collisions: int
    peak_frontier: int


@dataclass(frozen=True)
class SparsePolicy:
    """A frozen current-policy snapshot for one root.

    Positive cumulative regrets define the action probabilities; rows with no
    positive regret use the uniform fallback. Keys reserve their top two bits
    for the acting seat and use a 62-bit information fingerprint below it.
    """

    root: object
    keys: np.ndarray
    regret: np.ndarray

    def dist(self, seat: int, hand: int, path, legal_moves):
        """Return a sparse regret-matched distribution, uniform if unseen."""
        moves = np.sort(np.asarray(legal_moves, dtype=np.int64).reshape(-1))
        if not len(moves):
            raise ValueError("legal_moves must be nonempty")
        key = _info_key(seat, hand, path)
        row = int(np.searchsorted(self.keys, key))
        if row < len(self.keys) and self.keys[row] == key:
            positive = np.maximum(self.regret[row, :len(moves)], 0.0) \
                .astype(np.float64)
            total = float(positive.sum())
            if total > 0.0:
                return moves, positive / total
        return moves, np.full(len(moves), 1.0 / len(moves))

    def save(self, path) -> None:
        """Write a compact candidate-policy artifact for this exact root."""
        np.savez_compressed(
            Path(path), keys=self.keys, regret=self.regret,
            root_signature=np.asarray(_root_signature(self.root)),
            format_version=np.asarray(1, dtype=np.int64))

    @classmethod
    def load(cls, root, path):
        with np.load(Path(path), allow_pickle=False) as data:
            if int(data["format_version"].item()) != 1:
                raise ValueError("unsupported sparse policy artifact version")
            signature = str(data["root_signature"].item())
            if signature != _root_signature(root):
                raise ValueError("sparse policy artifact belongs to another root")
            return cls(root, data["keys"].copy(), data["regret"].copy())


@dataclass
class SampledGapResult:
    profile_value: float
    profile_value_se: float
    candidate_gains: dict[int, float]
    candidate_gain_se: dict[int, float]
    candidate_gain_error: dict[int, float]
    gap_lower: float
    candidate_gap_upper: float
    gap_upper: float
    shortfall_upper: float | None
    candidate_coverage: float
    gap_coverage: float | None
    verdict: str


class SampledBR:
    """Candidate best response to uniform or frozen sparse opponents."""

    def __init__(self, root, payoff43, capacity: int = 1 << 20,
                 updater: int | None = None,
                 frozen_policy: SparsePolicy | None = None):
        if mx is None:
            raise RuntimeError("SampledBR requires MLX") from _MLX_ERROR
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.root = root
        self.sampler = WorldSampler.from_root(root)
        self.capacity = int(capacity)
        payoff = np.asarray(payoff43, dtype=np.float32).reshape(-1)
        if len(payoff) != 43:
            raise ValueError("payoff43 must have 43 entries")
        self.payoff_np = payoff.copy()
        self.payoff = mx.array(payoff)
        # A one-world subgame is enough to parse the root and rule tables.
        sub = build_subgame(root, self.sampler.sample(1), np.ones(1))
        self.sub = sub
        self.updater = int(root.me if updater is None else updater)
        if not 0 <= self.updater < 4:
            raise ValueError("updater must be a seat in 0..3")
        self.sign = 1 if self.updater % 2 == sub.bid_team else -1
        if frozen_policy is None:
            keys = np.empty(0, dtype=np.uint64)
            regret = np.empty((0, 7), dtype=np.float32)
        else:
            if frozen_policy.root != root:
                raise ValueError("frozen policy belongs to a different root")
            keys = np.asarray(frozen_policy.keys, dtype=np.uint64).reshape(-1)
            regret = np.asarray(frozen_policy.regret, dtype=np.float32)
            if regret.shape != (len(keys), 7):
                raise ValueError("frozen policy regret must have shape (N, 7)")
            if len(keys) > self.capacity:
                raise MemoryError(
                    f"frozen policy has {len(keys):,} rows > capacity "
                    f"{self.capacity:,}")
            if len(keys) > 1 and (keys[1:] <= keys[:-1]).any():
                raise ValueError("frozen policy keys must be strictly sorted")
            regret = regret.copy()
            # This seat is the new BR candidate; the other three rows remain
            # the frozen opponent policy inherited from the CFR snapshot.
            regret[(keys >> np.uint64(62)) == self.updater] = 0.0
        self.keys_np = keys.copy()
        self.key_table = mx.array(
            self.keys_np if len(self.keys_np) else np.zeros(1, np.uint64))
        self.nkeys = mx.array([len(self.keys_np)], dtype=mx.uint32)
        # Only the occupied prefix is meaningful. Seven elements let the
        # first previously unseen information set follow the same growth path.
        initial = regret.reshape(-1) if len(regret) else np.zeros(7, np.float32)
        self.regret = mx.array(initial)
        self._scalar = {
            "bid_team": mx.array([sub.bid_team], dtype=mx.uint32),
        }
        self._updater = {u: mx.array([u], dtype=mx.uint32) for u in range(4)}
        self._sign = {
            u: mx.array([1.0 if u % 2 == sub.bid_team else -1.0],
                        dtype=mx.float32)
            for u in range(4)
        }
        self.cfb = mx.array(sub.luts.can_follow_bits.astype(np.uint32))
        self.led_suit = mx.array(sub.luts.led_suit.astype(np.uint32))
        self.rank = mx.array(sub.luts.rank.astype(np.uint32).reshape(-1))
        self.count = mx.array(sub.luts.count.astype(np.uint32))
        self._count_kernel = mx.fast.metal_kernel(
            name="hoyt_es_count", input_names=["state", "cfb", "pos",
            "updater"], output_names=["count"], source=_COUNT_SOURCE,
            header=_COMMON)
        self._expand_kernel = mx.fast.metal_kernel(
            name="hoyt_es_expand", input_names=["state", "offset", "regret",
            "cfb", "led_suit", "rank_lut", "count_lut", "pos", "updater",
            "key_table", "nkeys", "seed", "depth", "bid_team"],
            output_names=["child", "parent", "probability", "action_rank"],
            source=_EXPAND_SOURCE, header=_COMMON)
        self._backward_kernel = mx.fast.metal_kernel(
            name="hoyt_es_backward", input_names=["state", "offset",
            "probability", "vchild", "pos", "updater", "sign"],
            output_names=["vparent", "event_index", "event_delta",
            "event_key_lo", "event_key_hi"], source=_BACKWARD_SOURCE,
            header=_COMMON)
        self._reduce_kernel = mx.fast.metal_kernel(
            name="hoyt_es_reduce", input_names=["event_index", "event_delta"],
            output_names=["dense"], source=_REDUCE_SOURCE,
            atomic_outputs=True)
        self._leaf_kernel = mx.fast.metal_kernel(
            name="hoyt_es_leaf", input_names=["state", "payoff"],
            output_names=["value"], source=_LEAF_SOURCE)
        self.epochs = 0
        self.samples = 0
        self.peak_frontier = 0
        # This reports direct-table collisions, which the sorted table cannot
        # have. The 62-bit information fingerprint itself is probabilistic.
        self.key_collisions = 0

    @staticmethod
    def _call(kernel, inputs, n, shapes, dtypes, **kw):
        return kernel(inputs=inputs, grid=(n, 1, 1),
                      threadgroup=(min(256, n), 1, 1),
                      output_shapes=shapes, output_dtypes=dtypes, **kw)

    def _initial_state(self, worlds):
        worlds = np.asarray(worlds, dtype=np.uint32).reshape(-1, 3)
        n = len(worlds)
        state = np.zeros((n, _STRIDE), dtype=np.uint32)
        state[:, int(self.root.me)] = np.uint32(self.sub.my0)
        for col, seat in enumerate(self.sampler.seats):
            state[:, seat] = worlds[:, col]
        state[:, 4] = np.uint32(self.sub.leader0)
        state[:, 5] = np.uint32(self.sub.led0 + 1)
        state[:, 6] = np.uint32(self.sub.brank0 + 1)
        state[:, 7] = np.uint32(self.sub.bseat0 + 1)
        state[:, 8] = np.uint32(self.sub.cnt0)
        state[:, 9] = np.uint32(self.sub.ptsv0)
        state[:, 10] = np.uint32(int(_PH0) & 0xFFFFFFFF)
        state[:, 11] = np.uint32(int(_PH0) >> 32)
        return mx.array(state.reshape(-1))

    def _traverse(self, worlds, seed: int, updater: int | None = None):
        updater = self.updater if updater is None else int(updater)
        if not 0 <= updater < 4:
            raise ValueError("updater must be a seat in 0..3")
        state = self._initial_state(worlds)
        states = []
        transitions = []
        depth_n = 28 - self.sub.p0
        for depth in range(depth_n):
            n = int(state.shape[0]) // _STRIDE
            self.peak_frontier = max(self.peak_frontier, n)
            pos = mx.array([(self.sub.p0 + depth) & 3], dtype=mx.uint32)
            count, = self._call(
                self._count_kernel,
                [state, self.cfb, pos, self._updater[updater]], n,
                [(n,)], [mx.uint32])
            offset = mx.concatenate(
                [mx.zeros((1,), dtype=mx.uint32), mx.cumsum(count)])
            total = int(offset[-1].item())
            child, _parent, probability, _rank = self._call(
                self._expand_kernel,
                [state, offset, self.regret, self.cfb, self.led_suit,
                 self.rank, self.count, pos, self._updater[updater],
                 self.key_table, self.nkeys,
                 mx.array([seed & ((1 << 64) - 1)], dtype=mx.uint64),
                 mx.array([depth], dtype=mx.uint32),
                 self._scalar["bid_team"]],
                n, [(total * _STRIDE,), (total,), (total,), (total,)],
                [mx.uint32, mx.uint32, mx.float32, mx.uint32])
            states.append(state)
            transitions.append((offset, probability, pos))
            state = child

        nleaf = int(state.shape[0]) // _STRIDE
        value, = self._call(self._leaf_kernel, [state, self.payoff], nleaf,
                            [(nleaf,)], [mx.float32])
        event_i = []
        event_d = []
        event_lo = []
        event_hi = []
        root_q = None
        for depth in range(depth_n - 1, -1, -1):
            pstate = states[depth]
            offset, probability, pos = transitions[depth]
            n = int(pstate.shape[0]) // _STRIDE
            if depth == 0:
                root_q = value
            value, ei, ed, klo, khi = self._call(
                self._backward_kernel,
                [pstate, offset, probability, value, pos,
                 self._updater[updater], self._sign[updater]],
                n, [(n,), (n * 7,), (n * 7,), (n * 7,), (n * 7,)],
                [mx.float32, mx.uint32, mx.float32, mx.uint32, mx.uint32])
            event_i.append(ei)
            event_d.append(ed)
            event_lo.append(klo)
            event_hi.append(khi)
        return value, root_q, mx.concatenate(event_i), \
            mx.concatenate(event_d), mx.concatenate(event_lo), \
            mx.concatenate(event_hi)

    def _apply_events(self, action, delta, key_lo, key_hi, batch_size):
        mx.eval(action, delta, key_lo, key_hi)
        act = np.asarray(action, dtype=np.uint32)
        d = np.asarray(delta, dtype=np.float32)
        lo = np.asarray(key_lo, dtype=np.uint32)
        hi = np.asarray(key_hi, dtype=np.uint32)
        valid = d != 0.0
        if not valid.any():
            return
        act = act[valid]
        d = d[valid]
        keys = lo[valid].astype(np.uint64) \
            | (hi[valid].astype(np.uint64) << np.uint64(32))
        merged = np.union1d(self.keys_np, np.unique(keys))
        if len(merged) > self.capacity:
            raise MemoryError(
                f"sampled information table needs {len(merged):,} rows "
                f"> capacity {self.capacity:,}; reschedule with a larger slab")
        if len(merged) != len(self.keys_np):
            old = np.asarray(self.regret, dtype=np.float32) \
                .reshape(-1, 7)[:len(self.keys_np)]
            grown = np.zeros((max(1, len(merged)), 7), dtype=np.float32)
            if len(self.keys_np):
                grown[np.searchsorted(merged, self.keys_np)] = old
            self.keys_np = merged
            self.key_table = mx.array(
                merged if len(merged) else np.zeros(1, dtype=np.uint64))
            self.nkeys = mx.array([len(merged)], dtype=mx.uint32)
            self.regret = mx.array(grown.reshape(-1))
        row = np.searchsorted(self.keys_np, keys).astype(np.uint32)
        flat = row * np.uint32(7) + act
        ne = len(d)
        dense, = self._reduce_kernel(
            inputs=[mx.array(flat), mx.array(d)],
            grid=(ne, 1, 1), threadgroup=(min(256, ne), 1, 1),
            output_shapes=[(max(1, len(self.keys_np)) * 7,)],
            output_dtypes=[mx.float32], init_value=0.0)
        self.regret = self.regret + dense / np.float32(batch_size)
        mx.eval(self.regret)

    def snapshot(self) -> SparsePolicy:
        """Freeze the current regret-matched policy on the host boundary."""
        mx.eval(self.regret)
        regret = np.asarray(self.regret, dtype=np.float32) \
            .reshape(-1, 7)[:len(self.keys_np)].copy()
        return SparsePolicy(self.root, self.keys_np.copy(), regret)

    def train(self, epochs: int, batch_size: int, seed: int = 0):
        """Run frozen-policy external-sampling microbatches."""
        if epochs < 0 or batch_size <= 0:
            raise ValueError("epochs must be nonnegative and batch_size positive")
        for e in range(epochs):
            worlds = self.sampler.sample_metal(batch_size,
                                               seed + self.epochs + e)
            _v, _q, event_i, event_d, key_lo, key_hi = self._traverse(
                worlds, seed + self.epochs + e)
            self._apply_events(event_i, event_d, key_lo, key_hi, batch_size)
            self.samples += batch_size
        self.epochs += epochs
        return self

    def evaluate(self, batches: int, batch_size: int, seed: int = 1_000_000,
                 alpha: float = 0.05):
        if batches <= 0 or batch_size <= 0 or batches * batch_size < 2:
            raise ValueError("evaluation needs at least two samples")
        values = []
        qrows = []
        for b in range(batches):
            worlds = self.sampler.sample_metal(batch_size, seed + b)
            value, root_q, _ei, _ed, _lo, _hi = self._traverse(
                worlds, seed + b)
            mx.eval(value, root_q)
            values.append(np.asarray(value, dtype=np.float64))
            if self.updater == int(self.root.me):
                nlegal = int(root_q.shape[0]) // batch_size
                qrows.append(np.asarray(root_q, dtype=np.float64)
                             .reshape(batch_size, nlegal))
        val = np.concatenate(values)
        vm = float(val.mean())
        vse = float(val.std(ddof=1) / np.sqrt(len(val)))
        verr = _bounded_mean_error(
            val, float(self.payoff_np.min()), float(self.payoff_np.max()),
            alpha)
        root_values = {}
        root_value_se = {}
        root_value_error = {}
        best_move = None
        if qrows:
            q = np.concatenate(qrows)
            qm = q.mean(axis=0)
            qse = q.std(axis=0, ddof=1) / np.sqrt(len(q))
            legal = []
            mask = int(self.sub.my0)
            led = self.sub.led0
            if led >= 0:
                follow = mask & int(self.sub.luts.can_follow_bits[led])
                if follow:
                    mask = follow
            while mask:
                bit = mask & -mask
                legal.append(bit.bit_length() - 1)
                mask ^= bit
            if len(legal) != len(qm):
                raise AssertionError("root action/value layout mismatch")
            orient = self.sign * qm
            best = int(np.argmax(orient))
            root_values = {m: float(qm[i]) for i, m in enumerate(legal)}
            root_value_se = {m: float(qse[i]) for i, m in enumerate(legal)}
            root_value_error = {
                m: _bounded_mean_error(
                    q[:, i], float(self.payoff_np.min()),
                    float(self.payoff_np.max()), alpha / len(legal))
                for i, m in enumerate(legal)
            }
            best_move = legal[best]
        return SampledBRResult(
            value=vm,
            value_se=vse,
            value_error=verr,
            root_values=root_values,
            root_value_se=root_value_se,
            root_value_error=root_value_error,
            best_move=best_move,
            epochs=self.epochs,
            samples=self.samples,
            table_capacity=self.capacity,
            occupied_buckets=len(self.keys_np),
            key_collisions=self.key_collisions,
            peak_frontier=self.peak_frontier,
        )


class SampledCFR(SampledBR):
    """Shared-table four-seat external-sampling CFR candidate learner.

    One epoch performs one frozen-policy microbatch for each updating seat in
    seat order. The exported policy is the current ordinary-regret-matching
    policy. It is a candidate whose exploitability must be audited; no
    last-iterate equilibrium theorem is claimed.
    """

    def train(self, epochs: int, batch_size: int, seed: int = 0):
        if epochs < 0 or batch_size <= 0:
            raise ValueError("epochs must be nonnegative and batch_size positive")
        for e in range(epochs):
            epoch = self.epochs + e
            for updater in range(4):
                stream = seed + epoch * 4 + updater
                worlds = self.sampler.sample_metal(batch_size, stream)
                _v, _q, event_i, event_d, key_lo, key_hi = self._traverse(
                    worlds, stream, updater=updater)
                self._apply_events(event_i, event_d, key_lo, key_hi,
                                   batch_size)
                self.samples += batch_size
        self.epochs += epochs
        return self

    def fork_best_response(self, updater: int,
                           capacity: int | None = None) -> SampledBR:
        """Freeze this candidate and reset one seat into a BR learner."""
        return SampledBR(
            self.root, self.payoff_np,
            capacity=self.capacity if capacity is None else capacity,
            updater=updater, frozen_policy=self.snapshot())


def audit_sampled_gap(cfr: SampledCFR, br_epochs: int, train_batch: int,
                      eval_batches: int, eval_batch: int,
                      seed: int = 2_000_000,
                      shortfall_upper: float | None = None,
                      shortfall_alpha: float | None = None,
                      target_gap: float = 0.05,
                      alpha: float = 0.05) -> SampledGapResult:
    """Train independent per-seat candidate BRs and price a sampled gap.

    Without a calibrated ``shortfall_upper`` the candidate BRs still prove a
    lower bound on exploitability, but cannot prove convergence; the returned
    upper bound is therefore infinite and the verdict is ``unresolved`` unless
    the lower bound already proves ``not_converged``. With a finite shortfall,
    ``shortfall_alpha`` is mandatory so candidate and calibration miscoverage
    can be union-bounded to the requested total ``alpha``.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between zero and one")
    if shortfall_upper is None:
        if shortfall_alpha is not None:
            raise ValueError("shortfall_alpha requires shortfall_upper")
        candidate_alpha = alpha
        gap_coverage = None
    else:
        if shortfall_alpha is None or not 0.0 < shortfall_alpha < alpha:
            raise ValueError(
                "finite shortfall needs 0 < shortfall_alpha < alpha")
        candidate_alpha = alpha - shortfall_alpha
        gap_coverage = 1.0 - alpha

    # Five mean estimates (one baseline, four candidate BRs) split the
    # candidate budget. A second union bound combines that simultaneous band
    # with the independently calibrated optimization-shortfall budget.
    mean_alpha = candidate_alpha / 5.0
    baseline = cfr.evaluate(eval_batches, eval_batch, seed=seed,
                            alpha=mean_alpha)
    gains = {}
    ses = {}
    errors = {}
    bid_team = int(cfr.sub.bid_team)
    for updater in range(4):
        br = cfr.fork_best_response(updater)
        br.train(br_epochs, train_batch, seed=seed + 10_000 * (updater + 1))
        got = br.evaluate(eval_batches, eval_batch,
                          seed=seed + 100_000 + 10_000 * updater,
                          alpha=mean_alpha)
        sign = 1.0 if updater % 2 == bid_team else -1.0
        gains[updater] = sign * (got.value - baseline.value)
        ses[updater] = float(np.hypot(got.value_se, baseline.value_se))
        errors[updater] = got.value_error + baseline.value_error

    lower = max(0.0, max(gains[u] - errors[u] for u in range(4)))
    candidate_upper = max(0.0,
                          max(gains[u] + errors[u] for u in range(4)))
    if shortfall_upper is None:
        upper = float("inf")
    else:
        if shortfall_upper < 0 or not np.isfinite(shortfall_upper):
            raise ValueError("shortfall_upper must be finite and nonnegative")
        upper = candidate_upper + float(shortfall_upper)
    if lower > target_gap:
        verdict = "not_converged"
    elif upper <= target_gap:
        verdict = "converged"
    else:
        verdict = "unresolved"
    return SampledGapResult(
        profile_value=baseline.value,
        profile_value_se=baseline.value_se,
        candidate_gains=gains,
        candidate_gain_se=ses,
        candidate_gain_error=errors,
        gap_lower=lower,
        candidate_gap_upper=candidate_upper,
        gap_upper=upper,
        shortfall_upper=shortfall_upper,
        candidate_coverage=1.0 - candidate_alpha,
        gap_coverage=gap_coverage,
        verdict=verdict,
    )
