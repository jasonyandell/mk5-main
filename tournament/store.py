# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0

"""The record store: append-only files in a git repository, one commit each.

The tournament's whole state is a directory of immutable JSON files:

    participants/<participant id>.json
    faceoffs/<ladder>/<faceoff id>.json

A writer only ever ADDS a file whose path no other writer can choose (the
faceoff id carries a 64-bit random seed and the claim's UTC time; a
participant id is the participant). It never edits or deletes one, and there
is no shared index: every table is derived by reading the directory. Two
commits that each add a different new path touch disjoint trees, so any
interleaving of concurrent writers rebases cleanly. That is the whole
no-conflict argument; `tests/test_store.py` plays it out with real clones.

Appending is idempotent: the same bytes at the same path are a no-op, so a
writer that was killed after its commit landed but before it noticed can
retry safely. Different bytes at an existing path are refused (append-only).

Two backends with one interface:

- `GitStore` — a local clone of any git remote. Commit, then push; a rejected
  push is `pull --rebase` and push again (the rebase cannot conflict).
- `HfStore` — a Hugging Face dataset repo (itself git) through the Hub's
  commit API: each append is one `create_commit` adding one file with no
  `parent_commit`, which the Hub applies on top of whatever head it has.
  The token comes from huggingface_hub's own store or HF_TOKEN in the
  environment (docs/SECRETS.md); it never touches a file here.
"""
from __future__ import annotations

import json
import os
import random
import subprocess
import time
from pathlib import Path
from typing import Iterator, Protocol


class AppendConflict(RuntimeError):
    """Different bytes already live at a path: records are never edited."""


class Store(Protocol):
    def append(self, path: str, data: bytes, message: str) -> bool:
        """Add one file in one commit. False if the identical file was already there."""
        ...

    def read_all(self, prefix: str) -> Iterator[tuple[str, bytes]]:
        """Every file under `prefix`, (path, bytes), in path order."""
        ...

    def exists(self, path: str) -> bool:
        ...


def dumps(obj) -> bytes:
    """The one serialization of a record: canonical JSON, so equal records are equal bytes."""
    return (json.dumps(obj, sort_keys=True, indent=1, ensure_ascii=False) + "\n").encode()


def _check_path(path: str) -> None:
    p = Path(path)
    if p.is_absolute() or ".." in p.parts or not path.endswith(".json"):
        raise ValueError(f"not a record path: {path!r}")


class GitStore:
    """A clone of a git remote. Every append is one commit, pushed at once."""

    def __init__(self, root: str | os.PathLike, *, push: bool = True, retries: int = 60):
        self.root = Path(root)
        self.push = push
        self.retries = retries
        self.rejected = 0  # pushes another writer beat (each one rebased and retried)

    def _git(self, *args: str, check: bool = True) -> subprocess.CompletedProcess:
        return subprocess.run(["git", "-C", str(self.root), *args], capture_output=True,
                              text=True, check=check)

    def exists(self, path: str) -> bool:
        return (self.root / path).exists()

    def append(self, path: str, data: bytes, message: str) -> bool:
        _check_path(path)
        self.sync()
        target = self.root / path
        if target.exists():
            if target.read_bytes() == data:
                self._push()  # a previous run may have committed without pushing
                return False
            raise AppendConflict(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(".tmp")
        tmp.write_bytes(data)
        os.replace(tmp, target)
        self._git("add", "--", path)
        self._git("commit", "-q", "-m", message, "--", path)
        self._push()
        return True

    def sync(self) -> None:
        if self.push and self._git("remote", check=False).stdout.strip():
            self._git("pull", "-q", "--rebase")

    def _push(self) -> None:
        if not self.push or not self._git("remote", check=False).stdout.strip():
            return
        for attempt in range(self.retries):
            if self._git("push", "-q", check=False).returncode == 0:
                return
            self.rejected += 1
            # Back off first, with jitter, so rival writers fall out of step; then
            # rebase and push at once, before the fresh base goes stale.
            time.sleep(random.uniform(0, 0.05 * 2 ** min(attempt, 5)))
            # Someone else pushed first. Their commits add other files, so the
            # rebase replays ours on top without a conflict, by construction.
            r = self._git("pull", "-q", "--rebase", check=False)
            if r.returncode != 0:
                self._git("rebase", "--abort", check=False)
                raise RuntimeError(f"rebase failed (should be impossible): {r.stderr.strip()}")
        raise RuntimeError(f"push did not land after {self.retries} attempts")

    def read_all(self, prefix: str) -> Iterator[tuple[str, bytes]]:
        base = self.root / prefix
        if not base.exists():
            return
        for f in sorted(base.rglob("*.json")):
            yield f.relative_to(self.root).as_posix(), f.read_bytes()


class HfStore:
    """A Hugging Face dataset repo, appended through the Hub's commit API.

    Reads are incremental. Records are immutable, so a file already in the
    local cache is never asked for again: `read_all` makes ONE listing call
    (the Hub's tree API) and resolves only the paths the cache lacks. A chunk
    of the always-on ladder therefore costs a handful of requests, not one
    per record (the full re-fetch hit the Hub's 5000-resolvers-per-5-minutes
    quota with two workers). Workers may share a cache directory: a file is
    complete or absent, and its bytes are the same whoever fetched it.
    """

    def __init__(self, repo_id: str, *, cache: str | os.PathLike, retries: int = 8, api=None):
        from huggingface_hub import HfApi

        self.repo_id = repo_id
        self.api = api or HfApi()
        self.cache = Path(cache)
        self.retries = retries
        self.retried = 0

    @property
    def _root(self) -> Path:
        return self.cache / "snapshot"

    def _fetch(self, path: str) -> None:
        """Resolve one record into the cache (one resolver request)."""
        from huggingface_hub import hf_hub_download

        hf_hub_download(self.repo_id, path, repo_type="dataset", local_dir=self._root)

    def _cached(self, path: str) -> bytes | None:
        f = self._root / path
        return f.read_bytes() if f.is_file() else None

    def exists(self, path: str) -> bool:
        return (self._root / path).is_file() or self.api.file_exists(
            self.repo_id, path, repo_type="dataset")

    def append(self, path: str, data: bytes, message: str) -> bool:
        from huggingface_hub import CommitOperationAdd

        from huggingface_hub.errors import HfHubHTTPError

        _check_path(path)
        for attempt in range(self.retries):
            if self.exists(path):
                have = self._cached(path)
                if have is None:
                    self._fetch(path)
                    have = self._cached(path)
                if have == data:
                    return False  # landed already (perhaps by an attempt whose reply was lost)
                raise AppendConflict(path)
            try:
                # No parent_commit: the Hub commits on top of its current head, so a
                # concurrent writer's commit (another path) never makes this one fail.
                self.api.create_commit(
                    self.repo_id, repo_type="dataset",
                    operations=[CommitOperationAdd(path_in_repo=path, path_or_fileobj=data)],
                    commit_message=message,
                )
                # Our own record is already in hand; no need to resolve it back.
                target = self._root / path
                target.parent.mkdir(parents=True, exist_ok=True)
                tmp = target.with_name(f"{target.name}.{os.getpid()}.tmp")
                tmp.write_bytes(data)
                os.replace(tmp, target)
                return True
            except HfHubHTTPError as e:  # a rate limit or a transient error: the retry is idempotent
                self.retried += 1
                if attempt == self.retries - 1:
                    raise
                time.sleep(min(60.0, 2.0 ** attempt))
        raise AssertionError("unreachable")

    def read_all(self, prefix: str) -> Iterator[tuple[str, bytes]]:
        listed = sorted(e.path for e in self.api.list_repo_tree(
            self.repo_id, path_in_repo=prefix, recursive=True, repo_type="dataset")
            if e.path.endswith(".json"))
        for path in listed:
            data = self._cached(path)
            if data is None:
                self._fetch(path)
                data = self._cached(path)
            yield path, data
