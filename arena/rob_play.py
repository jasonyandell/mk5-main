"""rob — the texas-42 exact imperfect-information solver — as a PlayPolicy.

rob lives in a separate repository (jason/code/texas-42) with its own
certified Rust engine. This adapter takes no build dependency in either
direction: it talks to the prebuilt `rob_bridge` binary over a line
protocol of plain integers (one decision per line), and both projects
already share the canonical triangular domino encoding ((0,0)=0, (1,0)=1,
…, (6,6)=27), seat order, and team = seat % 2.

The bridge replies with rob's independently derived trick leader and team
point totals alongside the chosen domino, and `choose` asserts they match
the zeb engine's — so every decision doubles as a rules-conformance check
between the two rule implementations (trick resolution and the
1 + count-points trick score).

rob is bid-value- and score-blind here: it plays the exact-fiber Points
lens (signed point differential), the analytical counterpart of the
oracle players' E[pts] utility. Batch entries fan out across a small pool
of bridge processes so concurrent games overlap their solves.
"""
from __future__ import annotations

import os
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Sequence

from forge.zeb.game import current_player, legal_actions
from forge.zeb.types import ZebGameState

DEFAULT_BRIDGE = os.environ.get(
    "ROB_BRIDGE",
    str(
        Path.home()
        / "code/texas-42/.claude/worktrees/hierarchical-fibers-rung1"
        / "rob/target/release/rob_bridge"
    ),
)


class _Bridge:
    """One rob_bridge subprocess: a lock and a blocking ask()."""

    def __init__(self, binary: str):
        self.proc = subprocess.Popen(
            [binary],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self.lock = threading.Lock()

    def ask(self, request: str) -> tuple[int, int, int, int]:
        with self.lock:
            assert self.proc.stdin is not None and self.proc.stdout is not None
            self.proc.stdin.write(request + "\n")
            self.proc.stdin.flush()
            reply = self.proc.stdout.readline()
            if not reply:
                raise RuntimeError("rob_bridge died (empty reply)")
            domino, leader, p0, p1 = (int(t) for t in reply.split())
            return domino, leader, p0, p1


class RobPlay:
    """Exact-solver play via the rob_bridge subprocess pool."""

    def __init__(self, binary: str = DEFAULT_BRIDGE, workers: int = 8):
        if not Path(binary).is_file():
            raise FileNotFoundError(
                f"rob_bridge binary not found at {binary}; build it in the "
                "texas-42 repo (cargo build --release --bin rob_bridge) or "
                "set ROB_BRIDGE"
            )
        self._binary = binary
        self._bridges = [_Bridge(binary) for _ in range(workers)]
        self._pool = ThreadPoolExecutor(max_workers=workers)

    def choose(
        self,
        states: Sequence[ZebGameState],
        bid_values: Sequence[int],
        marks: Sequence[tuple[int, int]] | None = None,
        marks_to_win: int = 7,
    ) -> list[int]:
        futures = [
            self._pool.submit(self._decide, state, self._bridges[i % len(self._bridges)])
            for i, state in enumerate(states)
        ]
        return [f.result() for f in futures]

    def _decide(self, state: ZebGameState, bridge: _Bridge) -> int:
        player = current_player(state)
        hand = state.hands[player]
        request = [player, state.decl_id, state.bidder, *hand, len(state.play_history)]
        for actor, domino in state.play_history:
            request.append(actor)
            request.append(domino)
        chosen, leader, p0, p1 = bridge.ask(" ".join(str(v) for v in request))

        # Rules-conformance cross-check: rob replayed the same history under
        # its own certified rules; its derived leader and points must agree.
        if leader != state.trick_leader or (p0, p1) != state.team_points:
            raise RuntimeError(
                f"rob/zeb rules divergence: rob says leader={leader} "
                f"points=({p0},{p1}), zeb says leader={state.trick_leader} "
                f"points={state.team_points} after {state.play_history}"
            )
        slot = hand.index(chosen)
        if slot not in legal_actions(state):
            raise RuntimeError(
                f"rob chose domino {chosen} (slot {slot}) but zeb legal slots "
                f"are {legal_actions(state)} in {state}"
            )
        return slot

    def __repr__(self) -> str:
        return f"RobPlay(bridge={self._binary}, workers={len(self._bridges)})"
