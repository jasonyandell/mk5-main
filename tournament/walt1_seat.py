# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0
"""An external `walt1 play` binary as a card-play seat of the bid30 ladder.

The 42 Bench: several clean-room Rust builds of level-1 Walt's `walt1 play`
endpoint (~/code/local/walt-corpus/ENDPOINT.md), plus the real walt-wasm
compiled natively as the reference. Each reads ONE request on stdin and prints
one JSON line. A participant pins the binary's sha256, the build's git commit
(or none, with a digest of its source files) and the tournament sources; every
mismatch fails closed, and every reply is checked against the referee: an
illegal choice, a wrong leader or points, a malformed or error reply raises.

Bidding is the ladder's common Bid30Bidder (faceoff.py), so only card play is
compared. Thinking is a fixed sample budget (n, n0), never a clock.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path

from .gus_seat import LADDER, ROOT, source_hashes
from .participants import Participant

KIND = "walt1-binary"
SEED_RULE = "walt1-bench-v1"
# A hung binary is a failure, not a budget: no decision of a working build takes minutes.
HANG_S = 600
MODULE = "tournament/walt1_seat.py"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_digest(build: Path) -> str:
    """sha256 over (path, sha256) of the build's src/** and Cargo manifests, sorted."""
    files = sorted([p for p in (build / "src").rglob("*") if p.is_file()]
                   + [build / f for f in ("Cargo.toml", "Cargo.lock") if (build / f).exists()])
    h = hashlib.sha256()
    for f in files:
        h.update(f"{f.relative_to(build)}\0{_sha256(f)}\n".encode())
    return h.hexdigest()


def _git(repo: Path, *a) -> str | None:
    r = subprocess.run(["git", "-C", str(repo), *a], capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else None


def git_state(repo: Path) -> dict:
    commit = _git(repo, "rev-parse", "--verify", "-q", "HEAD")
    if commit is None:
        return {"git_commit": None}
    return {"git_commit": commit, "git_dirty": bool(_git(repo, "status", "--porcelain"))}


def subtree_state(repo: Path, sub: str) -> dict:
    """The git tree at HEAD:sub and whether sub has local changes (the commit is informational)."""
    return {"tree": _git(repo, "rev-parse", f"HEAD:{sub}"),
            "dirty": bool(_git(repo, "status", "--porcelain", "--", sub))}


def tournament_sources() -> dict:
    s = source_hashes()
    for name in (MODULE, "tournament/faceoff.py", "tournament/seats.py"):
        s[name] = _sha256(ROOT / name)
    return s


def participant(name: str, binary: Path, build: Path, *, source_repo: Path | None = None,
                source_sub: str = "",
                n: int = 40, n0: int = 8, note: str = "") -> Participant:
    """`build` holds src/ and Cargo files. `source_repo`/`source_sub`: a git subtree holding
    the player's source when the build only wraps it (walt-ref wraps texas-42's walt/)."""
    binary, build = binary.resolve(), build.resolve()
    sha = _sha256(binary)
    if not re.fullmatch(r"[a-z0-9-]+", name):
        raise ValueError("name: lowercase letters, digits and dashes")
    pin = {"binary": str(binary), "binary_sha256": sha, "build": str(build),
           "source_sha256": source_digest(build), **git_state(build),
           "protocol": "walt1 play (ENDPOINT.md)", "source_files": tournament_sources()}
    if source_repo is not None:
        pin["source_repo"] = str(source_repo.resolve())
        pin["source_sub"] = source_sub
        pin["source_commit"] = _git(source_repo, "rev-parse", "HEAD")
        pin["source_subtree"] = subtree_state(source_repo, source_sub)
    return Participant(
        f"walt1-{name}-{sha[:8]}", KIND, pin,
        {"ladders": [LADDER], "bidder": "Bid30Bidder", "n": n, "n0": n0, "seed_rule": SEED_RULE,
         "budget": f"n {n} outer worlds, n0 {n0} per level-0 mind; one process per decision"},
        ("torch",),
        note or f"42 Bench build `{name}`: a walt1 play binary, level-1 Walt card play at "
                f"n {n}, n0 {n0}; the common Bid30Bidder bids.")


def verify(p: Participant) -> Path:
    pin = p.pin
    binary = Path(pin["binary"])
    if not binary.is_file() or _sha256(binary) != pin["binary_sha256"]:
        raise RuntimeError(f"{p.id}: {binary} is missing or not the pinned binary")
    build = Path(pin["build"])
    if source_digest(build) != pin["source_sha256"]:
        raise RuntimeError(f"{p.id}: {build} sources changed since the pin")
    want = {k: pin[k] for k in ("git_commit", "git_dirty") if k in pin}
    if git_state(build) != want:
        raise RuntimeError(f"{p.id}: {build} git state {git_state(build)} != pinned {want}")
    if "source_repo" in pin and subtree_state(Path(pin["source_repo"]), pin["source_sub"]) != pin["source_subtree"]:
        raise RuntimeError(f"{p.id}: {pin['source_repo']}:{pin['source_sub']} changed since the pin")
    for path, h in pin["source_files"].items():
        if _sha256(ROOT / path) != h:
            raise RuntimeError(f"{p.id}: tournament source {path} changed; use the pinned commit")
    return binary


def decision_seed(game: str, hand_no: int, seat: int, n_plays: int) -> int:
    """walt1-bench-v1: a function of the faceoff seed, pair, side, hand and decision;
    never of participant names or claim time, so a fixed opponent draws alike."""
    m = re.fullmatch(r".*-([0-9a-f]{16})/(\d+)/([01])", game)
    if not m:
        raise ValueError(f"walt1 seed needs a tournament game identity, not {game!r}")
    seed, pair, side = m.groups()
    key = f"{SEED_RULE}/{seed}/{int(pair)}/{side}/{hand_no}/{seat}/{n_plays}"
    return int(hashlib.sha256(key.encode()).hexdigest()[:16], 16)


def request(state, bid: int, seat: int, n: int, n0: int, seed: int) -> str:
    plays = " ".join(f"{a} {t}" for a, t in state.play_history)
    return "\n".join([
        "walt1 play", f"decl {state.decl_id}", f"bid {bid}", f"seat {seat}",
        f"bidder {state.bidder}", "hand " + " ".join(map(str, sorted(state.hands[seat]))),
        f"plays {plays}".rstrip() if plays else "plays", f"n {n}", f"n0 {n0}", f"seed {seed}", ""])


class Walt1Error(RuntimeError):
    pass


def check_reply(line: str, legal: tuple[int, ...], leader: int, points: list[int]) -> int:
    """The chosen tile, or Walt1Error naming exactly what is wrong."""
    try:
        r = json.loads(line)
    except json.JSONDecodeError:
        raise Walt1Error(f"malformed reply: {line[:200]!r}")
    if not isinstance(r, dict) or r.get("v") != 1:
        raise Walt1Error(f"not a v1 reply: {line[:200]!r}")
    if "error" in r:
        raise Walt1Error(f"error reply: {r['error']}")
    if r.get("kind") != "play" or not isinstance(r.get("choice"), int) or isinstance(r.get("choice"), bool):
        raise Walt1Error(f"no integer choice: {line[:200]!r}")
    if r["choice"] not in legal:
        raise Walt1Error(f"illegal choice {r['choice']} (legal {sorted(legal)})")
    if r.get("leader") != leader or r.get("points") != list(points):
        raise Walt1Error(f"leader/points {r.get('leader')}/{r.get('points')} != table {leader}/{list(points)}")
    if r.get("forced") is not (len(legal) == 1):
        raise Walt1Error(f"forced={r.get('forced')} with {len(legal)} legal tiles")
    return r["choice"]


class Walt1Seat:
    """PlayPolicy (and the ladder's Bid30Bidder) for one team in one game."""

    def __init__(self, p: Participant, game: str):
        from arena.bidders import Bid30Bidder

        self.binary = verify(p)
        self.id, self.n, self.n0 = p.id, p.config["n"], p.config["n0"]
        if p.config.get("seed_rule") != SEED_RULE:
            raise ValueError(f"{p.id}: unknown seed rule {p.config.get('seed_rule')!r}")
        self.bidder = Bid30Bidder()
        self.game = game
        self.hand_no = 1
        self.cpu_s = 0.0
        self.decisions = 0

    def bid(self, ctx, rng):
        return self.bidder.bid(ctx, rng)

    def declare(self, hand, bid, rng):
        return self.bidder.declare(hand, bid, rng)

    def choose(self, states, bid_values, marks=None, marks_to_win: int = 7) -> list[int]:
        import resource

        from forge.zeb.game import current_player, legal_actions

        slots = []
        for s, bid in zip(states, bid_values):
            seat = current_player(s)
            seed = decision_seed(self.game, self.hand_no, seat, len(s.play_history))
            req = request(s, int(bid), seat, self.n, self.n0, seed)
            before = resource.getrusage(resource.RUSAGE_CHILDREN)
            try:
                out = subprocess.run([str(self.binary)], input=req, capture_output=True, text=True,
                                     timeout=HANG_S)
            except subprocess.TimeoutExpired:
                raise Walt1Error(f"{self.id}: no reply in {HANG_S} s to\n{req}")
            after = resource.getrusage(resource.RUSAGE_CHILDREN)
            self.cpu_s += (after.ru_utime - before.ru_utime) + (after.ru_stime - before.ru_stime)
            self.decisions += 1
            lines = out.stdout.strip().splitlines()
            if out.returncode != 0 or len(lines) != 1:
                raise Walt1Error(f"{self.id}: exit {out.returncode}, {len(lines)} lines, "
                                 f"stderr {out.stderr[-300:]!r} on\n{req}")
            legal = tuple(s.hands[seat][i] for i in legal_actions(s))  # slots -> tile ids
            try:
                tile = check_reply(lines[0], legal, s.trick_leader, list(s.team_points))
            except Walt1Error as e:
                raise Walt1Error(f"{self.id}: {e} on\n{req}")
            slots.append(list(s.hands[seat]).index(tile))
        return slots

    def close(self) -> None:
        pass

    @property
    def cost(self) -> dict:
        return {"cpu_s": round(self.cpu_s, 3), "decisions": self.decisions}
