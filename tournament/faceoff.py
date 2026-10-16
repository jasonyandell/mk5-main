# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0

"""One faceoff: `pairs` duplicate deal pairs between two participants.

A duplicate pair is two games on the same deals with the seats swapped: in
game (p, 0) participant A sits at seats 0/2, in game (p, 1) at seats 1/3, and
every hand of both games is dealt from the same seed (the arena's hand_seed,
which depends on the pair's base seed and the hand number, never on who
sits where). Card luck cancels inside a pair; auctions that diverge are the
players' own doing.

A game is resumable at every hand boundary. After each hand the game process
rewrites its progress file (atomically) with every finished hand; a restarted
game reads it, restores marks, points, dealer and hand number, and deals the
next hand from the same seed it would have used. So a kill loses at most the
hand in flight, and the finished faceoff is the same faceoff.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from arena import cost

LADDERS = {
    "marks7": {"marks_to_win": 7},  # the tournament game: first to 7 marks
    "points250": {"points_to_win": 250},  # the book's game: first to 250 points
}


@dataclass(frozen=True)
class FaceoffSpec:
    ladder: str
    a: str
    b: str
    seed: int  # 64-bit, random at claim time: the faceoff's identity
    pairs: int
    claimed_utc: str  # e.g. 20261014T183005Z

    @property
    def id(self) -> str:
        return f"{self.claimed_utc}-{self.a}--{self.b}-{self.seed:016x}"

    @property
    def path(self) -> str:
        return f"faceoffs/{self.ladder}/{self.id}.json"

    def games(self) -> list[tuple[int, int]]:
        return [(p, t) for p in range(self.pairs) for t in (0, 1)]

    def base_seed(self, pair: int) -> int:
        h = hashlib.sha256(f"{self.seed:016x}/{pair}".encode()).hexdigest()
        return int(h[:15], 16)


def _atomic_write(path: Path, obj) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, sort_keys=True))
    os.replace(tmp, path)


def _hand_dict(h) -> dict:
    d = asdict(h)
    for k in ("game_idx", "a_team"):
        d.pop(k)
    return {k: (list(map(list, v)) if k in ("hands", "plays") else list(v) if isinstance(v, tuple) else v)
            for k, v in d.items()}


def play_game(spec_dict: dict, pair: int, a_team: int, progress_path: str,
              participants: dict, device: str | None = None) -> dict:
    """Play (or finish) one game; returns its progress dict with done=True."""
    from arena.engine import ArenaConfig, _LiveGame, _run_lockstep, _seat_policies

    from .participants import Participant
    from .seats import seat

    spec = FaceoffSpec(**spec_dict)
    progress = Path(progress_path)
    state = json.loads(progress.read_text()) if progress.exists() else \
        {"pair": pair, "a_team": a_team, "hands": [], "done": False, "elapsed_s": 0.0}
    if state["done"]:
        return state

    cfg = ArenaConfig(base_seed=spec.base_seed(pair), **LADDERS[spec.ladder])
    game_key = f"{spec.id}/{pair}/{a_team}"
    pa = Participant.from_record(participants[spec.a])
    pb = Participant.from_record(participants[spec.b])
    sa, sb = seat(pa, game_key, device), seat(pb, game_key, device)
    seats = (sa, sb)
    start = cost.snapshot()
    t0 = time.time()
    elapsed0 = state["elapsed_s"]

    class Game(_LiveGame):
        def __init__(self):
            done = state["hands"]
            # Restore the game at the boundary after its last finished hand.
            self._restore = done[-1] if done else None
            super().__init__(pair, a_team, cfg, _seat_policies(a_team, sa, sb))

        def _deal_and_bid(self):
            r = getattr(self, "_restore", None)
            if r is not None:
                self._restore = None
                self.marks = list(r["marks_after"])
                self.points = list(r["points_after"])
                self.hand_idx = len(state["hands"])
                self.dealer = (r["dealer"] + 1) % 4
            for s in seats:
                s.hand_no = self.hand_idx + 1
            super()._deal_and_bid()

        def apply(self, action):
            before = len(self.hands)
            super().apply(action)
            if len(self.hands) > before:
                state["hands"].append(_hand_dict(self.hands[-1]))
                state["elapsed_s"] = elapsed0 + (time.time() - t0)
                if self.done:
                    state.update(done=True, winner_team=self.winner,
                                 marks=list(self.marks), points=list(self.points))
                _atomic_write(progress, state)

    try:
        g = Game()
        _run_lockstep([g], sa, sb, cfg.marks_to_win)
    finally:
        sa.close()
        sb.close()
    # What this run of the game cost beside its wall time (#173), summed over resumes.
    spent = cost.since(start)
    add = {"referee_cpu_s": spent["cpu_self_s"]}
    if "instructions" in spent:
        add["referee_instructions"] = spent["instructions"]
    for side, s in (("a", sa), ("b", sb)):
        for k, v in (s.cost or {}).items():
            if k != "exit":
                add[f"{side}_{k}"] = v
    total = state.setdefault("cost", {})
    for k, v in add.items():
        total[k] = round(total.get(k, 0) + v, 3)
    _atomic_write(progress, state)
    return state


def assemble(spec: FaceoffSpec, participants: dict, games: list[dict], harness: dict,
             finished_utc: str) -> dict:
    """The faceoff's one immutable record."""
    out_games = []
    for g in sorted(games, key=lambda g: (g["pair"], g["a_team"])):
        a = g["a_team"]
        out_games.append({
            "pair": g["pair"], "a_team": a, "a_won": g["winner_team"] == a,
            "winner_team": g["winner_team"], "marks": g["marks"], "points": g["points"],
            "a_marks": g["marks"][a], "b_marks": g["marks"][1 - a],
            "n_hands": len(g["hands"]), "elapsed_s": round(g["elapsed_s"], 1), "hands": g["hands"],
            **({"cost": g["cost"]} if "cost" in g else {}),
        })
    a_wins = sum(g["a_won"] for g in out_games)
    return {
        "schema": "texas42-tournament-faceoff/1",
        "id": spec.id,
        "ladder": spec.ladder,
        "rules": LADDERS[spec.ladder],
        "seed": f"{spec.seed:016x}",
        "pairs": spec.pairs,
        "a": spec.a,
        "b": spec.b,
        "participants": {spec.a: participants[spec.a], spec.b: participants[spec.b]},
        "claimed_utc": spec.claimed_utc,
        "finished_utc": finished_utc,
        "harness": harness,
        # A participant that thinks on a wall clock plays a little differently on a busy
        # machine; its record says so unless cpu-slot gave the run the whole machine (#173).
        "noisy_timed": any(participants[p].get("config", {}).get("clock") == "real" for p in (spec.a, spec.b))
                       and harness.get("cpu_slot_exclusive") != "1",
        "summary": {"games": len(out_games), "a_wins": a_wins, "b_wins": len(out_games) - a_wins,
                    "a_marks": sum(g["a_marks"] for g in out_games),
                    "b_marks": sum(g["b_marks"] for g in out_games)},
        "games": out_games,
    }
