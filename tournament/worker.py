"""The worker loop: read → rate → choose → play → append → repeat.

    1. Finish any claim this machine left behind (a faceoff it started).
    2. Read every record; derive the ratings (tournament/rating.py).
    3. Choose the most informative pair this machine can seat (schedule.py).
    4. Claim it: a local directory named by the faceoff id, holding the spec
       (a fresh 64-bit random seed) and, as games run, each game's progress.
    5. Play its games, all at once, each resumable at every hand boundary.
    6. Assemble the record ONCE, save it in the claim, append it to the store
       (one file, one commit), then delete the claim.

A kill at any point loses at most the hands in flight:
- before step 4: nothing was started;
- during step 5: the claim holds every finished hand, and step 1 resumes it
  with the same seed, so the faceoff that lands is the one that was claimed;
- between saving the record and the append: step 1 appends the saved bytes;
- between the append and deleting the claim: step 1 appends the same bytes
  again, the store sees an identical file at the same path and does nothing.
No step can make one faceoff count twice: a faceoff is one path, and a path
is written once.

A claim is local on purpose. Two machines never coordinate; each one's
faceoff has its own random seed and its own path, so nothing they do can
collide. A claim lost with its machine (a cloud container that is gone) is a
faceoff never recorded, which biases nothing: whether it finishes does not
depend on how it was going.
"""
from __future__ import annotations

import json
import os
import platform
import random
import secrets
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from .faceoff import LADDERS, FaceoffSpec, assemble, play_game
from .participants import ANCHOR, Participant
from .rating import Game, fit
from .schedule import next_pair
from .store import dumps

ROOT = Path(__file__).resolve().parent.parent


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def log(msg: str) -> None:
    print(f"[{datetime.now(timezone.utc):%H:%M:%S}] {msg}", flush=True)


# ------------------------------------------------------------- reading

def read_state(store, ladder: str):
    participants = {}
    for _, data in store.read_all("participants"):
        r = json.loads(data)
        participants[r["id"]] = r
    records = [json.loads(d) for _, d in store.read_all(f"faceoffs/{ladder}")]
    return participants, records


def games_of(records) -> list[Game]:
    out = []
    for r in records:
        for g in r["games"]:
            out.append(Game(r["a"], r["b"], bool(g["a_won"]), f"{r['id']}/{g['pair']}"))
    return out


def ratings(participants: dict, records: list[dict], anchor: str = ANCHOR):
    return fit(games_of(records), list(participants), anchor)


# ------------------------------------------------------------- capabilities

def capabilities() -> set[str]:
    caps = set()
    if shutil.which("node"):
        caps.add("node")
    try:
        import torch

        caps.add("torch")
        # Forge is GPU-only (CLAUDE.md): the oracle's seat needs MPS or CUDA.
        if torch.backends.mps.is_available() or torch.cuda.is_available():
            caps.add("gpu")
        if (ROOT / "forge/models/domino-qval-large-3.3M-qgap0.071-qmae0.94.ckpt").exists():
            caps.add("eq-oracle")
    except ImportError:
        pass
    return caps


def runnable(participants: dict, caps: set[str], only: set[str] | None = None) -> list[str]:
    return sorted(pid for pid, r in participants.items()
                  if set(r.get("needs", ())) <= caps and (only is None or pid in only))


def harness() -> dict:
    def out(*cmd):
        try:
            return subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT).stdout.strip()
        except OSError:
            return ""
    return {"mk5_commit": out("git", "rev-parse", "HEAD"),
            "mk5_dirty": bool(out("git", "status", "--porcelain", "--", "tournament", "arena", "forge")),
            "python": platform.python_version(), "node": out("node", "--version"),
            "platform": f"{platform.system()}-{platform.machine()}",
            "worker": os.environ.get("TOURNAMENT_WORKER", "local")}


# ------------------------------------------------------------- claims

class Claims:
    """This machine's unfinished faceoffs. A worker holds an exclusive flock on
    each claim it is running, so two workers sharing the directory never run
    one claim twice; the kernel drops the lock when its process dies, which is
    exactly when the claim becomes resumable."""

    def __init__(self, root: Path):
        self.root = root
        self._held: dict[Path, object] = {}

    def open(self) -> list[Path]:
        if not self.root.exists():
            return []
        return sorted(p for p in self.root.iterdir() if (p / "spec.json").exists())

    def try_lock(self, d: Path) -> bool:
        import fcntl

        fh = open(d / ".lock", "a")
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            fh.close()
            return False
        self._held[d] = fh
        return True

    def release(self, d: Path) -> None:
        fh = self._held.pop(d, None)
        if fh is not None:
            fh.close()

    def create(self, spec: FaceoffSpec, participants: dict) -> Path:
        d = self.root / spec.id
        d.mkdir(parents=True, exist_ok=False)
        if not self.try_lock(d):  # locked before spec.json exists, so no one can resume it early
            raise RuntimeError(f"fresh claim {d} is locked")
        tmp = d / "spec.json.tmp"
        tmp.write_text(json.dumps({"spec": spec.__dict__, "participants": participants}, sort_keys=True))
        os.replace(tmp, d / "spec.json")
        return d


# ------------------------------------------------------------- one faceoff

def run_claim(claim: Path, store, *, parallel: bool = True, device: str | None = None) -> str:
    """Finish the claimed faceoff and append its record. Returns its path."""
    meta = json.loads((claim / "spec.json").read_text())
    spec = FaceoffSpec(**meta["spec"])
    participants = meta["participants"]
    record_file = claim / "record.json"
    if not record_file.exists():
        jobs = [(spec.__dict__, p, t, str(claim / f"game-{p}-{t}.json"), participants, device)
                for p, t in spec.games()]
        if parallel and len(jobs) > 1:
            with ProcessPoolExecutor(max_workers=len(jobs)) as pool:
                games = list(pool.map(play_game, *zip(*jobs)))
        else:
            games = [play_game(*j) for j in jobs]
        record = assemble(spec, participants, games, harness(), utc_stamp())
        tmp = record_file.with_suffix(".tmp")
        tmp.write_bytes(dumps(record))
        os.replace(tmp, record_file)
    data = record_file.read_bytes()
    record = json.loads(data)
    s = record["summary"]
    added = store.append(spec.path, data,
                         f"faceoff {spec.ladder}: {spec.a} {s['a_wins']}-{s['b_wins']} {spec.b} "
                         f"(marks {s['a_marks']}-{s['b_marks']}, {spec.pairs} duplicate pairs)")
    log(f"{'appended' if added else 'already recorded'} {spec.path}")
    shutil.rmtree(claim)
    return spec.path


# ------------------------------------------------------------- the loop

def work(store, *, ladder: str = "marks7", pairs: int = 1, budget_s: float = 240.0,
         max_faceoffs: int | None = None, only: set[str] | None = None,
         claims_dir: Path | None = None, parallel: bool = True, device: str | None = None,
         anchor: str = ANCHOR) -> int:
    """Run faceoffs until the budget (wall seconds) or max_faceoffs is spent.

    The budget bounds when a NEW faceoff may start; a claim in progress is
    always continued. Returns the number of records appended or confirmed.
    """
    if ladder not in LADDERS:
        raise ValueError(f"unknown ladder {ladder!r}; ladders: {', '.join(LADDERS)}")
    claims = Claims(claims_dir or ROOT / "scratch" / "tournament" / "claims")
    t0 = time.time()
    done = 0
    for claim in claims.open():
        if not claims.try_lock(claim):
            log(f"claim {claim.name} is being run by another worker; leaving it")
            continue
        log(f"resuming claim {claim.name}")
        try:
            run_claim(claim, store, parallel=parallel, device=device)
        finally:
            claims.release(claim)
        done += 1
    caps = capabilities()
    while (max_faceoffs is None or done < max_faceoffs) and time.time() - t0 < budget_s:
        participants, records = read_state(store, ladder)
        allowed = runnable(participants, caps, only)
        if len(allowed) < 2:
            log(f"fewer than two participants this machine can seat ({caps}); stopping")
            break
        f = ratings(participants, records, anchor)
        choice = next_pair(f, 2 * pairs, allowed, rng=random.SystemRandom())
        spec = FaceoffSpec(ladder, choice.a, choice.b, secrets.randbits(64), pairs, utc_stamp())
        log(f"claim {spec.id} (gain {choice.gain:.3f}, P(a wins) {f.p_win(choice.a, choice.b):.2f})")
        claim = claims.create(spec, {k: participants[k] for k in (spec.a, spec.b)})
        try:
            run_claim(claim, store, parallel=parallel, device=device)
        finally:
            claims.release(claim)
        done += 1
    return done


def seed_roster(store, roster) -> None:
    """Append each roster participant not yet in the store (idempotent)."""
    for p in roster:
        if store.append(p.path(), dumps(p.to_record()), f"participant: {p.id}"):
            log(f"added participant {p.id}")


def table(store, ladder: str = "marks7", anchor: str = ANCHOR) -> str:
    """The derived rating table, as Markdown."""
    from .rating import paired_margin

    participants, records = read_state(store, ladder)
    if anchor not in participants:
        return "(no participants yet)"
    f = ratings(participants, records, anchor)
    lines = [f"Ladder `{ladder}`: {len(records)} faceoffs, {sum(len(r['games']) for r in records)} games. "
             f"Bradley–Terry, anchor `{anchor}` = 1500, 95% cluster-robust intervals.", "",
             "| participant | Elo | 95% CI | games | wins |", "|---|---|---|---|---|"]
    for r in f.ratings:
        ci = "anchor" if r.player == anchor else f"[{r.lo:.0f}, {r.hi:.0f}]"
        lines.append(f"| `{r.player}` | {r.elo:.0f} | {ci} | {r.games} | {r.wins} |")
    pairs: dict[tuple[str, str], list[float]] = {}
    wins: dict[tuple[str, str], list[int]] = {}
    for rec in records:
        a, b = rec["a"], rec["b"]
        key, sign = ((a, b), 1) if a < b else ((b, a), -1)
        by_pair: dict[int, list[dict]] = {}
        for g in rec["games"]:
            by_pair.setdefault(g["pair"], []).append(g)
        for gs in by_pair.values():
            pairs.setdefault(key, []).append(sign * sum(g["a_marks"] - g["b_marks"] for g in gs) / len(gs))
        w = wins.setdefault(key, [0, 0])
        for g in rec["games"]:
            w[0 if (g["a_won"] == (sign == 1)) else 1] += 1
    if pairs:
        lines += ["", "| pair (first − second) | games | wins | marks/game | 95% CI |", "|---|---|---|---|---|"]
        for (x, y), diffs in sorted(pairs.items()):
            m, lo, hi = paired_margin(diffs)
            ci = "n/a" if lo != lo else f"[{lo:+.2f}, {hi:+.2f}]"
            lines.append(f"| `{x}` vs `{y}` | {sum(wins[(x, y)])} | {wins[(x, y)][0]}–{wins[(x, y)][1]} "
                         f"| {m:+.2f} | {ci} |")
    return "\n".join(lines)
