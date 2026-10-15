# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0

# STRATEGY-FUSION FENCE.
# The Demon: a fenced tournament opponent; measuring stick only.
# Classic PIMC (Walt's sampler, Silas's double-dummy solver per world, averaged)
# seated so Walt and Zeb can be rated against strategy fusion head-on, the use
# wiki/topics/strategy-fusion-fence.md § Sanctioned uses names ("Baselines").
# Its games rate it; nothing reads its values, and nothing in Zeb, Walt or Silas
# may import this module. The Demon's page: wiki/topics/the-demon.md.
"""The Demon's seat: the arena's heuristic bidder, and play by the `demon` binary.

The binary (`tournament/demon/`, Rust) is built from the pinned source trees,
never from whatever the checkout holds: `build` extracts each pinned git tree
(`git archive <tree>`), checks it is the tree the pinned commit has at that
path when the commit is present, and runs `cargo build --release --locked`.
A seat is one binary process per team per game, sent its own tiles and the
public record only.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import shutil
import subprocess
import tarfile
from pathlib import Path
from typing import Sequence

from .participants import Participant

ROOT = Path(__file__).resolve().parent.parent
BINARY = "demon"


def _git(*args: str, binary: bool = False):
    out = subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=not binary)
    return out.stdout


def _has(obj: str) -> bool:
    return subprocess.run(["git", "cat-file", "-e", obj], cwd=ROOT, capture_output=True).returncode == 0


def build(p: Participant, out: Path, target_dir: Path) -> None:
    """Extract the pinned trees into `out/src` and build `out/demon` from them."""
    commit, trees = p.pin["commit"], p.pin["trees"]
    if not all(_has(t) for t in trees.values()):
        # A shallow clone: fetch the pinned commit (GitHub serves any reachable sha).
        subprocess.run(["git", "fetch", "--quiet", "origin", commit], cwd=ROOT, check=True)
    if _has(commit):
        for path, tree in trees.items():
            have = _git("rev-parse", f"{commit}:{path}").strip()
            if have != tree:
                raise RuntimeError(f"{p.id}: {commit}:{path} is tree {have}, the pin says {tree}")
    src = out / "src"
    for path, tree in trees.items():
        (src / path).mkdir(parents=True, exist_ok=True)
        with tarfile.open(fileobj=io.BytesIO(_git("archive", "--format=tar", tree, binary=True))) as t:
            t.extractall(src / path, filter="data")
    subprocess.run(["cargo", "build", "--release", "--locked", "--quiet",
                    "--manifest-path", str(src / "tournament/demon/Cargo.toml"),
                    "--target-dir", str(target_dir)], check=True)
    shutil.copy2(target_dir / "release" / BINARY, out / BINARY)


def decision_seed(game: str, hand_no: int, seat: int, n_plays: int) -> int:
    """The decision's seed: a function of where it is in the faceoff, never of the clock."""
    h = hashlib.sha256(f"demon/{game}/{hand_no}/{seat}/{n_plays}".encode()).hexdigest()
    return int(h[:15], 16)


class DemonSeat:
    """BidPolicy and PlayPolicy for one team in one game."""

    def __init__(self, p: Participant, artifacts: Path, game: str):
        from arena.cli import parse_bidder

        cfg = p.config
        self.bid_spec = cfg["bidder"]
        if self.bid_spec.startswith("pimc-over"):
            parts = self.bid_spec.split(":")
            if len(parts) != 4 or parts[0] != "pimc-over":
                raise ValueError("pimc-over needs worlds:opening_cutoff:opponent_cutoff")
            if not 1 <= int(parts[1]) <= 4096 or any(
                not math.isfinite(float(x)) or not 0 <= float(x) <= 1 for x in parts[2:]
            ):
                raise ValueError("pimc-over needs worlds 1..4096 and finite cutoffs in 0..1")
        self.declarations = tuple(cfg.get("declarations", range(10)))
        if not self.declarations or len(set(self.declarations)) != len(self.declarations) or any(
            not isinstance(d, int) or not 0 <= d < 10 for d in self.declarations
        ):
            raise ValueError("declarations must be distinct ids in 0..9")
        self.bidder = parse_bidder("heuristic" if self.bid_spec.startswith("pimc") else self.bid_spec,
                                   device="cpu", gus_adapter=None)
        self._bid_seats = {}
        self._forecasts = {}
        self.game = game
        self.hand_no = 1
        args = [str(artifacts / BINARY), str(cfg["worlds"]), "1" if cfg["margin_tiebreak"] else "0"]
        if "declaration_weight" in cfg:
            args.append(str(cfg["declaration_weight"]))
        self.proc = subprocess.Popen(
            args,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        self.decisions: list[dict] = []  # per decision: worlds, ms (for cost reports)

    # BidPolicy
    def bid(self, ctx, rng):
        self._bid_seats[tuple(ctx.hand)] = ctx.seat
        if self.bid_spec.startswith(("pimc:", "pimc-over:")):
            # A whole-game experiment: minimum point raise, partner pass,
            # and a stated threshold on the optimistic DD forecast.
            target = min((b for b in ctx.legal if b <= 42), default=None)
            if target is None or (ctx.high_seat >= 0 and ctx.high_seat % 2 == ctx.seat % 2):
                return 0
            decl, reply = self._forecast(ctx.hand, ctx.seat, target)
            cutoff = float(self.bid_spec.split(":")[2])
            if self.bid_spec.startswith("pimc-over:") and ctx.high_seat >= 0:
                # The partner case returned above. The opponent already holds
                # the auction, so passing is not the same option as opening.
                # This is an explicit experimental DD threshold, not a
                # calibrated claim about the chance of making the contract.
                cutoff = float(self.bid_spec.split(":")[3])
            return target if reply["makes"][decl] / reply["worlds"] >= cutoff else 0
        return self.bidder.bid(ctx, rng)

    def declare(self, hand, bid, rng):
        if self.bid_spec.startswith("pimc"):
            return self._forecast(hand, self._bid_seats[tuple(hand)], bid)[0]
        return self.bidder.declare(hand, bid, rng)

    def _forecast(self, hand, seat, bid):
        key = (self.hand_no, tuple(hand), seat, bid)
        if key not in self._forecasts:
            n = int(self.bid_spec.split(":")[1])
            reply = self._ask({"op":"forecast", "hand_mask":sum(1 << t for t in hand),
                               "seat":seat, "bid":bid, "worlds":n,
                               "declarations":list(self.declarations),
                               "seed":decision_seed(self.game, self.hand_no, seat, -1)})
            heuristic = self._ask({"op":"declaration_model", "hand_mask":sum(1 << t for t in hand)})["decl"]
            decl = max(self.declarations, key=lambda d:(reply["makes"][d], d == heuristic, -d))
            self._forecasts[key] = decl, reply
        return self._forecasts[key]

    # PlayPolicy
    def choose(self, states, bid_values: Sequence[int], marks=None, marks_to_win: int = 7) -> list[int]:
        from forge.zeb.game import current_player

        slots = []
        for s, bid in zip(states, bid_values):
            seat = current_player(s)
            dealt = list(s.hands[seat])
            req = {"op": "play", "seat": seat, "hand": [t for t in dealt if t not in s.played],
                   "decl": s.decl_id, "bid": int(bid), "bidder": s.bidder,
                   "plays": [list(p) for p in s.play_history],
                   "leader": s.trick_leader, "points": list(s.team_points),
                   "seed": decision_seed(self.game, self.hand_no, seat, len(s.play_history))}
            reply = self._ask(req)
            self.decisions.append({k: reply[k] for k in ("worlds", "ms", "proposals", "mass") if k in reply})
            slots.append(dealt.index(int(reply["tile"])))
        return slots

    def _ask(self, req: dict) -> dict:
        self.proc.stdin.write(json.dumps(req) + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError(f"the demon exited (code {self.proc.poll()})")
        reply = json.loads(line)
        if "error" in reply:
            raise RuntimeError(f"the demon refused: {reply['error']}")
        return reply

    def close(self) -> None:
        if self.proc.poll() is None:
            self.proc.stdin.close()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
