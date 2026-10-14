"""The worker loop is interruptible anywhere and never counts a faceoff twice.

Players here are the arena's cheap ones (bid30/heuristic + random play), so a
faceoff takes a second; the loop, the claims and the store are the real ones.
"""
from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from tournament import faceoff as F
from tournament import worker as W
from tournament.participants import Participant
from tournament.store import GitStore

ROOT = Path(__file__).resolve().parents[2]
CHEAP = [Participant("rand-a", "arena", {"spec": "bid30+random"}),
         Participant("rand-b", "arena", {"spec": "heuristic+random"}),
         Participant("rand-c", "arena", {"spec": "heuristic:3,1+random"})]


@pytest.fixture(autouse=True)
def git_identity(tmp_path_factory, monkeypatch):
    cfg = tmp_path_factory.mktemp("gitcfg") / "config"
    cfg.write_text("[user]\n\tname = tournament test\n\temail = test@example.invalid\n"
                   "[init]\n\tdefaultBranch = main\n[commit]\n\tgpgsign = false\n")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(cfg))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")


@pytest.fixture
def store(tmp_path):
    repo = tmp_path / "records"
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    s = GitStore(repo)
    W.seed_roster(s, CHEAP)
    return s


def records(store):
    return sorted(p for p, _ in store.read_all("faceoffs"))


def commits(store):
    out = subprocess.run(["git", "-C", str(store.root), "log", "--format=tformat:@", "--name-only", "--", "faceoffs"],
                         capture_output=True, text=True, check=True).stdout
    return [c.split() for c in out.split("@") if c.strip()]


def run(store, claims, **kw):
    kw.setdefault("pairs", 2)
    return W.work(store, ladder="marks7", parallel=False, claims_dir=claims, anchor="rand-a", **kw)


def test_each_faceoff_is_one_file_in_one_commit(store, tmp_path):
    assert run(store, tmp_path / "claims", max_faceoffs=3) == 3
    assert len(records(store)) == 3
    assert all(len(c) == 1 for c in commits(store)) and len(commits(store)) == 3
    rec = json.loads(next(d for _, d in store.read_all("faceoffs")))
    assert rec["pairs"] == 2 and len(rec["games"]) == 4
    assert {(g["pair"], g["a_team"]) for g in rec["games"]} == {(0, 0), (0, 1), (1, 0), (1, 1)}
    # duplicate deals: both games of a pair open on the same dealt hands
    by_pair = {}
    for g in rec["games"]:
        by_pair.setdefault(g["pair"], []).append(g["hands"][0]["hands"])
    assert all(a == b for a, b in by_pair.values())
    assert "rand-a" in W.table(store, anchor="rand-a")


KILLER = textwrap.dedent("""
    import os, sys
    sys.path.insert(0, {root!r})
    import tournament.faceoff as F
    real, n = F._atomic_write, [0]
    def write_then_die(path, obj):
        real(path, obj)
        n[0] += 1
        if n[0] >= {k}:
            os._exit(9)  # no cleanup, no finally: a kill
    F._atomic_write = write_then_die
    from tournament.store import GitStore
    from tournament.worker import work
    work(GitStore({repo!r}), pairs=1, max_faceoffs=1, parallel=False, claims_dir=__import__('pathlib').Path({claims!r}),
         anchor="rand-a")
""")


def test_killed_mid_faceoff_resumes_the_same_faceoff_once(store, tmp_path):
    claims = tmp_path / "claims"
    code = KILLER.format(root=str(ROOT), k=3, repo=str(store.root), claims=str(claims))
    assert subprocess.run([sys.executable, "-c", code]).returncode == 9
    (claim,) = W.Claims(claims).open()
    before = json.loads((claim / "game-0-0.json").read_text())
    assert len(before["hands"]) == 3 and not before["done"] and records(store) == []

    assert run(store, claims, max_faceoffs=1) == 1  # the resume is the one faceoff
    (path,) = records(store)
    rec = json.loads((store.root / path).read_bytes())
    assert rec["id"] == claim.name and not claim.exists()
    game = next(g for g in rec["games"] if (g["pair"], g["a_team"]) == (0, 0))
    assert game["hands"][:3] == before["hands"]  # the hands played before the kill are kept as played
    # and the resumed hands are dealt exactly as an uninterrupted game deals them
    meta = rec
    replay = F.play_game({"ladder": "marks7", "a": meta["a"], "b": meta["b"], "seed": int(meta["seed"], 16),
                          "pairs": 1, "claimed_utc": meta["claimed_utc"]}, 0, 0, str(tmp_path / "replay.json"),
                         meta["participants"])
    n = min(len(replay["hands"]), len(game["hands"]))
    assert [h["seed"] for h in replay["hands"][:n]] == [h["seed"] for h in game["hands"][:n]]
    assert [h["hands"] for h in replay["hands"][:n]] == [h["hands"] for h in game["hands"][:n]]


def test_a_claim_another_worker_holds_is_left_alone(store, tmp_path):
    claims = tmp_path / "claims"
    other = W.Claims(claims)  # another worker, mid-faceoff, holding its claim's lock
    participants, _ = W.read_state(store, "marks7")
    spec = F.FaceoffSpec("marks7", "rand-a", "rand-b", 12345, 1, "20261014T000000Z")
    claim = other.create(spec, {k: participants[k] for k in ("rand-a", "rand-b")})
    assert run(store, claims, max_faceoffs=0) == 0 and claim.exists() and records(store) == []
    other.release(claim)  # that worker died: the kernel drops its lock
    assert run(store, claims, max_faceoffs=0) == 1 and not claim.exists()
    assert [Path(p).stem for p in records(store)] == [spec.id]


def test_killed_after_the_append_does_not_append_again(store, tmp_path, monkeypatch):
    claims = tmp_path / "claims"
    real = W.shutil.rmtree

    def die(path, *a, **k):
        raise KeyboardInterrupt("killed between the commit and the cleanup")
    monkeypatch.setattr(W.shutil, "rmtree", die)
    with pytest.raises(KeyboardInterrupt):
        run(store, claims, max_faceoffs=1)
    monkeypatch.setattr(W.shutil, "rmtree", real)
    assert len(records(store)) == 1 and len(W.Claims(claims).open()) == 1

    assert run(store, claims, max_faceoffs=1) == 1  # resumes, finds it recorded, cleans up
    assert len(records(store)) == 1 and len(commits(store)) == 1 and W.Claims(claims).open() == []


def test_killed_between_the_record_and_the_append_appends_those_bytes(store, tmp_path, monkeypatch):
    claims = tmp_path / "claims"
    real = store.append

    def die(path, data, message):
        if path.startswith("faceoffs/"):
            raise KeyboardInterrupt("killed before the commit")
        return real(path, data, message)
    monkeypatch.setattr(store, "append", die)
    with pytest.raises(KeyboardInterrupt):
        run(store, claims, max_faceoffs=1)
    (claim,) = W.Claims(claims).open()
    saved = (claim / "record.json").read_bytes()
    monkeypatch.setattr(store, "append", real)

    run(store, claims, max_faceoffs=1)
    (path,) = records(store)
    assert (store.root / path).read_bytes() == saved
