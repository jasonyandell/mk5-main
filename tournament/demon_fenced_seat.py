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


def required_makes(cutoff: float, worlds: int) -> int:
    """Smallest count whose floating ratio clears the existing cutoff rule.

    ceil(cutoff * worlds) can round up incorrectly (0.07 * 100 > 7).
    Integer bisection uses the exact comparison made by the original seat.
    """
    if worlds < 1 or not math.isfinite(cutoff) or not 0 <= cutoff <= 1:
        raise ValueError("a positive world count and finite cutoff in 0..1 are required")
    lo, hi = 0, worlds
    while lo < hi:
        mid = (lo + hi) // 2
        if mid / worlds >= cutoff:
            hi = mid
        else:
            lo = mid + 1
    return lo


class DemonSeat:
    """BidPolicy and PlayPolicy for one team in one game."""

    def __init__(self, p: Participant, artifacts: Path, game: str):
        from arena.cli import parse_bidder

        cfg = p.config
        self.bid_spec = cfg["bidder"]
        if self.bid_spec.startswith(("pimc-over", "pimc-jump", "pimc-bait")):
            parts = self.bid_spec.split(":")
            if len(parts) != 4 or parts[0] not in ("pimc-over", "pimc-jump", "pimc-bait"):
                raise ValueError("pimc-over / pimc-jump / pimc-bait needs worlds:opening_cutoff:opponent_cutoff")
            if not 1 <= int(parts[1]) <= 4096 or any(
                not math.isfinite(float(x)) or not 0 <= float(x) <= 1 for x in parts[2:]
            ):
                raise ValueError("pimc-over / pimc-jump / pimc-bait needs worlds 1..4096 and finite cutoffs in 0..1")
        self.bait_bid = cfg.get("bait_bid", 30)
        if (type(self.bait_bid) is not int or not 30 <= self.bait_bid <= 40
                or (self.bait_bid != 30 and not self.bid_spec.startswith("pimc-bait:"))):
            raise ValueError("bait_bid must be an integer 30..40 and requires a pimc-bait bidder")
        self.bounded_forecast = cfg.get("bounded_forecast", False)
        if not isinstance(self.bounded_forecast, bool):
            raise ValueError("bounded_forecast must be a boolean")
        self.support_forecast = cfg.get("support_forecast", False)
        if not isinstance(self.support_forecast, bool) or (self.support_forecast and not self.bounded_forecast):
            raise ValueError("support_forecast requires bounded_forecast and a boolean")
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
        self.cost = None  # set by close() (#173)

    # BidPolicy
    def bid(self, ctx, rng):
        self._bid_seats[tuple(ctx.hand)] = ctx.seat
        if self.bid_spec.startswith(("pimc:", "pimc-over:", "pimc-jump:", "pimc-bait:")):
            points = sorted(b for b in ctx.legal if b <= 42)
            if not points or (ctx.high_seat >= 0 and ctx.high_seat % 2 == ctx.seat % 2):
                return 0
            # With at least two seats still to speak, an opponent can overcall
            # before the final bidder. This experimental policy opens its
            # configured legal point bid even with a weak hand; the normal
            # cutoff applies later. Forty leaves a point overcall below 42.
            if (self.bid_spec.startswith("pimc-bait:") and ctx.high_seat < 0
                    and sum(b < 0 for b in ctx.bids) - 1 >= 2):
                return self.bait_bid if self.bait_bid in points else 0
            cutoff = float(self.bid_spec.split(":")[2])
            if self.bid_spec.startswith(("pimc-over:", "pimc-jump:", "pimc-bait:")) and ctx.high_seat >= 0:
                # The partner case returned above. The opponent already holds
                # the auction, so passing is not the same option as opening.
                # This is an explicit experimental DD threshold, not a
                # calibrated claim about the chance of making the contract.
                cutoff = float(self.bid_spec.split(":")[3])
            jump = self.bid_spec.startswith("pimc-jump:")
            def supported(target):
                if self.support_forecast:
                    n = int(self.bid_spec.split(":")[1])
                    reply = self._ask({"op": "forecast_support", "hand_mask": sum(1 << t for t in ctx.hand),
                                       "seat": ctx.seat, "bid": target, "worlds": n,
                                       "required": required_makes(cutoff, n),
                                       "declarations": list(self.declarations),
                                       "seed": decision_seed(self.game, self.hand_no, ctx.seat, -1)})
                    if reply["worlds"] != n or not isinstance(reply["supported"], bool):
                        raise RuntimeError("support forecast returned an invalid predicate")
                    return reply["supported"]
                decl, reply = self._forecast(ctx.hand, ctx.seat, target)
                return reply["makes"][decl] / reply["worlds"] >= cutoff

            if not supported(points[0]):
                return 0
            if not jump:
                return points[0]
            # Every target uses the same sampled worlds. Exact DD make counts
            # are nonincreasing in the target, including after maximizing over
            # declarations. Bisect to the highest supported legal point bid.
            lo, hi = 0, len(points)
            while lo + 1 < hi:
                mid = (lo + hi) // 2
                if supported(points[mid]):
                    lo = mid
                else:
                    hi = mid
            return points[lo]
        return self.bidder.bid(ctx, rng)

    def declare(self, hand, bid, rng):
        if self.bid_spec.startswith("pimc"):
            return self._forecast(hand, self._bid_seats[tuple(hand)], bid)[0]
        return self.bidder.declare(hand, bid, rng)

    def _forecast(self, hand, seat, bid):
        key = (self.hand_no, tuple(hand), seat, bid)
        if key not in self._forecasts:
            n = int(self.bid_spec.split(":")[1])
            reply = self._ask({"op":"forecast_best" if self.bounded_forecast else "forecast",
                               "hand_mask":sum(1 << t for t in hand),
                               "seat":seat, "bid":bid, "worlds":n,
                               "declarations":list(self.declarations),
                               "seed":decision_seed(self.game, self.hand_no, seat, -1)})
            if self.bounded_forecast:
                decl = reply["best_decl"]
                if decl not in self.declarations or reply["makes"][decl] is None:
                    raise RuntimeError("bounded forecast returned an invalid winning declaration")
            else:
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
        from .seats import close_and_cost

        self.cost = close_and_cost(self.proc)
