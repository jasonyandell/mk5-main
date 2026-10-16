# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0

"""The no-conflict property, played out with real git: separate clones, a shared
bare remote, concurrent writers racing their pushes.

Claim under test: when every write adds a new file at a path no other writer
can choose, every interleaving of commits and pushes lands every record, one
commit each, with no merge and no conflict. The control shows the claim is
not vacuous: two writers that edit one shared index file do conflict.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import random
import subprocess
import time
from pathlib import Path

import pytest

from tournament.store import AppendConflict, GitStore, HfStore, dumps


def git(cwd, *args, check=True):
    return subprocess.run(["git", "-C", str(cwd), *args], capture_output=True, text=True, check=check)


@pytest.fixture(autouse=True)
def git_identity(tmp_path_factory, monkeypatch):
    cfg = tmp_path_factory.mktemp("gitcfg") / "config"
    cfg.write_text("[user]\n\tname = tournament test\n\temail = test@example.invalid\n"
                   "[init]\n\tdefaultBranch = main\n[commit]\n\tgpgsign = false\n"
                   "[pull]\n\trebase = true\n")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(cfg))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")


def make_remote(tmp: Path) -> Path:
    remote = tmp / "remote.git"
    subprocess.run(["git", "init", "-q", "--bare", str(remote)], check=True)
    seed = tmp / "seed"
    subprocess.run(["git", "clone", "-q", str(remote), str(seed)], check=True, capture_output=True)
    (seed / "README.md").write_text("tournament records\n")
    git(seed, "add", "README.md")
    git(seed, "commit", "-q", "-m", "init")
    git(seed, "push", "-q", "origin", "HEAD:main")
    return remote


def clone(remote: Path, where: Path) -> Path:
    subprocess.run(["git", "clone", "-q", str(remote), str(where)], check=True, capture_output=True)
    return where


def _writer(remote: str, root: str, name: str, n: int, seed: int, out) -> None:
    rng = random.Random(seed)
    store = GitStore(clone(Path(remote), Path(root)))
    try:
        for k in range(n):
            time.sleep(rng.random() * 0.05)
            path = f"faceoffs/marks7/{name}-{k:02d}-{rng.getrandbits(64):016x}.json"
            store.append(path, dumps({"writer": name, "k": k}), f"faceoff {name} {k}")
        out.put((name, store.rejected, None))
    except Exception as e:  # reported to the parent, which fails the test
        out.put((name, store.rejected, repr(e)))


def test_concurrent_writers_never_conflict_and_lose_nothing(tmp_path):
    remote = make_remote(tmp_path)
    writers, each = 5, 8
    ctx = mp.get_context("spawn")
    out = ctx.Queue()
    env_cfg = os.environ["GIT_CONFIG_GLOBAL"]
    procs = [ctx.Process(target=_writer, args=(str(remote), str(tmp_path / f"w{i}"), f"w{i}", each, i, out))
             for i in range(writers)]
    assert env_cfg  # children inherit the test identity through the environment
    for p in procs:
        p.start()
    results = [out.get(timeout=120) for _ in procs]
    for p in procs:
        p.join(timeout=60)
    assert [err for _, _, err in results if err] == []

    check = clone(remote, tmp_path / "check")
    files = sorted(f.name for f in (check / "faceoffs" / "marks7").iterdir())
    assert len(files) == writers * each  # zero lost records
    assert git(check, "rev-list", "--merges", "HEAD").stdout.strip() == ""  # zero merges
    log = git(check, "log", "--format=@%H", "--name-only", "--", "faceoffs").stdout.split("@")
    commits = [c.split() for c in log if c.strip()]
    assert len(commits) == writers * each and all(len(c) == 2 for c in commits)  # one file per commit
    # The race really happened: some pushes were rejected and replayed by rebase.
    print(f"rejected pushes, each rebased and replayed: {sum(r for _, r, _ in results)}")
    assert sum(r for _, r, _ in results) > 0


def test_a_rejected_push_rebases_without_conflict(tmp_path):
    remote = make_remote(tmp_path)
    a = GitStore(clone(remote, tmp_path / "a"), push=False)
    b = GitStore(clone(remote, tmp_path / "b"))
    a.append("faceoffs/marks7/a.json", dumps({"a": 1}), "a")  # committed, not pushed
    b.append("faceoffs/marks7/b.json", dumps({"b": 1}), "b")  # lands first
    a.push = True
    a._push()  # rejected once, rebased, pushed
    assert a.rejected == 1
    check = clone(remote, tmp_path / "check")
    assert sorted(p.name for p in (check / "faceoffs" / "marks7").iterdir()) == ["a.json", "b.json"]


def test_two_writers_adding_the_same_participant_is_one_add(tmp_path):
    remote = make_remote(tmp_path)
    a = GitStore(clone(remote, tmp_path / "a"), push=False)
    b = GitStore(clone(remote, tmp_path / "b"))
    data = dumps({"id": "zeb-9.9.9"})
    a.append("participants/zeb-9.9.9.json", data, "participant")
    b.append("participants/zeb-9.9.9.json", data, "participant")
    a.push = True
    a._push()  # its commit becomes empty in the rebase and is dropped
    check = clone(remote, tmp_path / "check")
    assert (check / "participants" / "zeb-9.9.9.json").read_bytes() == data


def test_append_is_idempotent_and_never_edits(tmp_path):
    remote = make_remote(tmp_path)
    s = GitStore(clone(remote, tmp_path / "a"))
    assert s.append("faceoffs/marks7/x.json", dumps({"x": 1}), "x") is True
    assert s.append("faceoffs/marks7/x.json", dumps({"x": 1}), "x") is False
    with pytest.raises(AppendConflict):
        s.append("faceoffs/marks7/x.json", dumps({"x": 2}), "x")
    with pytest.raises(ValueError):
        s.append("../escape.json", b"{}", "no")
    assert git(s.root, "rev-list", "--count", "HEAD").stdout.strip() == "2"


def test_control_a_shared_index_file_does_conflict(tmp_path):
    """The design the store avoids: every writer edits one manifest."""
    remote = make_remote(tmp_path)
    a, b = clone(remote, tmp_path / "a"), clone(remote, tmp_path / "b")
    for who, root in (("a", a), ("b", b)):
        (root / "index.json").write_text(f'["{who}"]\n')
        git(root, "add", "index.json")
        git(root, "commit", "-q", "-m", who)
    git(b, "push", "-q")
    assert git(a, "push", "-q", check=False).returncode != 0
    rebase = git(a, "pull", "-q", "--rebase", check=False)
    assert rebase.returncode != 0 and "CONFLICT" in (rebase.stdout + rebase.stderr)
    git(a, "rebase", "--abort", check=False)


# ---------------------------------------------------------------- HfStore (incremental reads)

class FakeHub:
    """A directory standing in for the Hub: counts listing calls and file resolves."""

    def __init__(self, root: Path):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        self.listings = 0
        self.resolves = 0

    # the HfApi surface HfStore uses
    def list_repo_tree(self, repo_id, path_in_repo, recursive, repo_type):
        self.listings += 1
        base = self.root / path_in_repo
        return [type("E", (), {"path": f.relative_to(self.root).as_posix()})
                for f in base.rglob("*") if f.is_file()]

    def create_commit(self, repo_id, repo_type, operations, commit_message):
        for op in operations:
            f = self.root / op.path_in_repo
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_bytes(op.path_or_fileobj)

    def file_exists(self, repo_id, path, repo_type):
        return (self.root / path).is_file()


def hf_store(hub: FakeHub, cache: Path) -> HfStore:
    store = HfStore("x/y", cache=cache, api=hub)

    def fetch(path):
        hub.resolves += 1
        target = store._root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(f"{target.name}.{os.getpid()}.{time.monotonic_ns()}.tmp")
        tmp.write_bytes((hub.root / path).read_bytes())
        os.replace(tmp, target)

    store._fetch = fetch
    return store


def test_hf_reads_resolve_only_files_the_cache_lacks(tmp_path):
    hub = FakeHub(tmp_path / "hub")
    for i in range(30):
        hub.create_commit("", "", [type("Op", (), {"path_in_repo": f"faceoffs/l/{i:02d}.json",
                                                    "path_or_fileobj": dumps({"i": i})})], "m")
    store = hf_store(hub, tmp_path / "cache")
    assert len(list(store.read_all("faceoffs/l"))) == 30
    assert (hub.listings, hub.resolves) == (1, 30)
    for _ in range(5):  # every later read is one listing and no resolves
        assert len(list(store.read_all("faceoffs/l"))) == 30
    assert (hub.listings, hub.resolves) == (6, 30)
    # a record added elsewhere costs exactly one resolve
    hub.create_commit("", "", [type("Op", (), {"path_in_repo": "faceoffs/l/zz.json",
                                                "path_or_fileobj": dumps({"i": 99})})], "m")
    assert len(list(store.read_all("faceoffs/l"))) == 31
    assert hub.resolves == 31
    # our own appends are cached on the way out: no resolve to read them back
    assert store.append("faceoffs/l/mine.json", dumps({"i": 7}), "m") is True
    assert len(list(store.read_all("faceoffs/l"))) == 32
    assert hub.resolves == 31
    # append-only holds through the cache
    assert store.append("faceoffs/l/mine.json", dumps({"i": 7}), "m") is False
    with pytest.raises(AppendConflict):
        store.append("faceoffs/l/mine.json", dumps({"i": 8}), "m")


def test_hf_concurrent_workers_share_one_cache(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    hub = FakeHub(tmp_path / "hub")
    cache = tmp_path / "cache"

    def worker(w):
        store = hf_store(hub, cache)
        for i in range(10):
            assert store.append(f"faceoffs/l/w{w}-{i}.json", dumps({"w": w, "i": i}), "m")
            list(store.read_all("faceoffs/l"))
        return store

    with ThreadPoolExecutor(4) as ex:
        list(ex.map(worker, range(4)))
    final = hf_store(hub, cache)
    got = dict(final.read_all("faceoffs/l"))
    assert len(got) == 40
    assert all(got[f"faceoffs/l/w{w}-{i}.json"] == dumps({"w": w, "i": i})
               for w in range(4) for i in range(10))
