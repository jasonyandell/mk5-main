"""Exact bounded-memory sampling of physical hidden worlds.

``walt.worlds.enumerate_worlds`` is the audit oracle.  This module represents
the same conservation and public void constraints with a tiny continuation-
count dynamic program, then samples assignments without materializing the
world population.  The table is at most ``22 * 8^3`` uint32 values, including
an H7 root.

``sample_metal`` executes one independent sampler per Metal thread.  Its
counter-based random stream is keyed by ``(seed, sample, tile, retry)`` and
uses rejection before modulo, so every legal continuation is selected with
the same pseudorandom probability.  No rejection over whole deals is used.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from walt.tables import N_DOMINOES
from walt.worlds import _void_forbidden, seat_order

__all__ = ["WorldSampler"]

_DP_SHAPE = (22, 8, 8, 8)


_SAMPLE_SOURCE = r"""
    uint i = thread_position_in_grid.x;

    uint cap[3] = {counts[0], counts[1], counts[2]};
    uint hand[3] = {0u, 0u, 0u};
    ulong base_seed = seed[0] ^ ((ulong)i * 0x9E3779B97F4A7C15ul);

    for (uint k = 0; k < ntiles[0]; ++k) {
        uint weight[3] = {0u, 0u, 0u};
        uint total = 0u;
        uint am = allowed[k];
        for (uint s = 0; s < 3; ++s) {
            if ((am & (1u << s)) && cap[s] > 0u) {
                uint c0 = cap[0] - (s == 0u);
                uint c1 = cap[1] - (s == 1u);
                uint c2 = cap[2] - (s == 2u);
                uint at = (k + 1u) * 512u + c0 * 64u + c1 * 8u + c2;
                weight[s] = dp[at];
                total += weight[s];
            }
        }

        ulong x = base_seed ^ ((ulong)k * 0xD1B54A32D192ED03ul);
        ulong threshold = (0ul - (ulong)total) % (ulong)total;
        ulong z;
        uint retry = 0u;
        do {
            x += 0x9E3779B97F4A7C15ul + (ulong)retry++;
            z = x;
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ul;
            z = (z ^ (z >> 27)) * 0x94D049BB133111EBul;
            z ^= z >> 31;
        } while (z < threshold);
        uint draw = (uint)(z % (ulong)total);

        uint owner = 0u;
        while (draw >= weight[owner]) {
            draw -= weight[owner];
            owner += 1u;
        }
        cap[owner] -= 1u;
        hand[owner] |= 1u << tiles[k];
    }
    worlds[i * 3u] = hand[0];
    worlds[i * 3u + 1u] = hand[1];
    worlds[i * 3u + 2u] = hand[2];
"""


@dataclass
class WorldSampler:
    seats: tuple[int, int, int]
    tiles: np.ndarray
    allowed: np.ndarray
    counts: np.ndarray
    dp: np.ndarray

    @classmethod
    def from_root(cls, root):
        seats = seat_order(root)
        played = {int(d) for _, d in root.play_history}
        mine = {int(t) for t in root.my_hand}
        tiles = np.asarray(sorted(set(range(N_DOMINOES)) - played - mine),
                           dtype=np.uint8)
        made = {s: 0 for s in seats}
        for seat, _ in root.play_history:
            if seat in made:
                made[seat] += 1
        counts = np.asarray([7 - made[s] for s in seats], dtype=np.uint8)
        if int(counts.sum()) != len(tiles):
            raise ValueError("root violates hidden-tile conservation")

        forbidden = _void_forbidden(root)
        allowed = np.zeros(len(tiles), dtype=np.uint8)
        for k, tile in enumerate(tiles):
            bit = np.uint32(1) << np.uint32(tile)
            for col, seat in enumerate(seats):
                if not (forbidden[seat] & bit):
                    allowed[k] |= np.uint8(1 << col)
            if allowed[k] == 0:
                raise ValueError(f"tile {int(tile)} has no legal hidden owner")

        dp = np.zeros(_DP_SHAPE, dtype=np.uint32)
        dp[len(tiles), 0, 0, 0] = 1
        for k in range(len(tiles) - 1, -1, -1):
            am = int(allowed[k])
            remaining = len(tiles) - k
            for c0 in range(min(7, remaining) + 1):
                for c1 in range(min(7, remaining - c0) + 1):
                    c2 = remaining - c0 - c1
                    if not 0 <= c2 <= 7:
                        continue
                    total = 0
                    if am & 1 and c0:
                        total += int(dp[k + 1, c0 - 1, c1, c2])
                    if am & 2 and c1:
                        total += int(dp[k + 1, c0, c1 - 1, c2])
                    if am & 4 and c2:
                        total += int(dp[k + 1, c0, c1, c2 - 1])
                    dp[k, c0, c1, c2] = np.uint32(total)
        out = cls(seats=seats, tiles=tiles, allowed=allowed,
                  counts=counts, dp=dp)
        if out.n_worlds == 0:
            raise ValueError("root has no physical hidden worlds")
        return out

    @property
    def n_worlds(self) -> int:
        c0, c1, c2 = map(int, self.counts)
        return int(self.dp[0, c0, c1, c2])

    def sample(self, n: int, seed: int = 0) -> np.ndarray:
        """Sample ``n`` uniform legal worlds on the host, without enumeration."""
        if n < 0:
            raise ValueError("n must be nonnegative")
        rng = np.random.default_rng(seed)
        out = np.zeros((n, 3), dtype=np.uint32)
        for i in range(n):
            cap = self.counts.astype(np.int16)
            for k, tile in enumerate(self.tiles):
                weights = np.zeros(3, dtype=np.uint32)
                for s in range(3):
                    if int(self.allowed[k]) & (1 << s) and cap[s] > 0:
                        nxt = cap.copy()
                        nxt[s] -= 1
                        weights[s] = self.dp[k + 1, *map(int, nxt)]
                total = int(weights.sum())
                draw = int(rng.integers(total))
                owner = 0
                while draw >= int(weights[owner]):
                    draw -= int(weights[owner])
                    owner += 1
                cap[owner] -= 1
                out[i, owner] |= np.uint32(1) << np.uint32(tile)
        return out

    def sample_metal(self, n: int, seed: int = 0) -> np.ndarray:
        """Sample ``n`` uniform legal worlds with one Metal thread per deal."""
        if n < 0:
            raise ValueError("n must be nonnegative")
        if n == 0:
            return np.zeros((0, 3), dtype=np.uint32)
        try:
            import mlx.core as mx
        except ImportError as exc:
            raise RuntimeError("sample_metal requires MLX") from exc
        if "gpu" not in str(mx.default_device()).lower():
            raise RuntimeError("sample_metal requires the MLX Metal GPU")
        kernel = mx.fast.metal_kernel(
            name="hoyt_world_sample",
            input_names=["tiles", "allowed", "counts", "dp", "ntiles",
                         "seed"],
            output_names=["worlds"], source=_SAMPLE_SOURCE)
        worlds, = kernel(
            inputs=[mx.array(self.tiles.astype(np.uint32)),
                    mx.array(self.allowed.astype(np.uint32)),
                    mx.array(self.counts.astype(np.uint32)),
                    mx.array(self.dp.reshape(-1)),
                    mx.array([len(self.tiles)], dtype=mx.uint32),
                    mx.array([seed & ((1 << 64) - 1)], dtype=mx.uint64)],
            grid=(n, 1, 1), threadgroup=(min(256, n), 1, 1),
            output_shapes=[(n * 3,)], output_dtypes=[mx.uint32])
        return np.asarray(worlds, dtype=np.uint32).reshape(n, 3)
