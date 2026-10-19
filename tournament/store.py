# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0

"""The record store: append-only records, with durable batched Hub publication.

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
- `HfStore` — a durable local outbox plus an immutable Hub snapshot. Appends
  are immediately visible locally; `flush` publishes many records in one
  parent-guarded commit. Workers sharing a cache share its lock and cooldown.
  The Hub token stays in the SDK credential store or environment.
"""
from __future__ import annotations

import json
import fcntl
from contextlib import contextmanager
from email.utils import parsedate_to_datetime
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
        """Durably record a file. False if identical bytes were already recorded."""
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


def _atomic_record(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with tmp.open("wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class HfStore:
    """Durable local records; at most one Hub commit per five minutes/cache.

    `append` acknowledges the fsynced outbox, not remote delivery. Reads merge
    that outbox with the immutable remote snapshot, so ratings need not wait.
    `flush` serializes publishers, guards the remote parent, and persists its
    cooldown before any attempt. An interrupted/uncertain commit keeps its
    outbox; a later flush reconciles identical remote bytes without duplication.
    Separate machines/caches have separate cadence; use one cache per repo per
    host when running multiple workers.
    """

    def __init__(self, repo_id: str, *, cache: str | os.PathLike, api=None):
        from huggingface_hub import HfApi
        self.repo_id = repo_id
        self.api = api or HfApi()
        self.cache = Path(cache)

    @property
    def _root(self) -> Path:
        return self.cache / "snapshot"

    @property
    def _pending(self) -> Path:
        return self.cache / "outbox"

    @contextmanager
    def _lock(self):
        self.cache.mkdir(parents=True, exist_ok=True)
        with (self.cache / "publish.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            yield

    def _fetch(self, path: str, revision: str | None = None) -> None:
        from huggingface_hub import hf_hub_download
        hf_hub_download(self.repo_id, path, repo_type="dataset", local_dir=self._root,
                        revision=revision, force_download=revision is not None)

    def _cached(self, path: str) -> bytes | None:
        f = self._root / path
        return f.read_bytes() if f.is_file() else None

    def exists(self, path: str) -> bool:
        _check_path(path)
        return ((self._pending / path).is_file() or (self._root / path).is_file()
                or self.api.file_exists(self.repo_id, path, repo_type="dataset"))

    def append(self, path: str, data: bytes, message: str) -> bool:
        _check_path(path)
        with self._lock():
            pending = self._pending / path
            if pending.is_file():
                have = pending.read_bytes()
            elif self.exists(path):
                have = self._cached(path)
                if have is None:
                    self._fetch(path)
                    have = self._cached(path)
            else:
                _atomic_record(pending, data)
                return True
            if have != data:
                raise AppendConflict(path)
            return False

    def _state(self):
        path = self.cache / "publication.json"
        return json.loads(path.read_text()) if path.exists() else {}

    def publication_status(self):
        with self._lock():
            return {**self._state(), "pending_records": len(list(self._pending.rglob("*.json")))}

    def flush(self) -> dict:
        """Publish up to 500 queued files, respecting cadence and Retry-After.

        No force override: restarts, CLI sync and parallel workers obey the same
        persisted deadline. Remote conflicts are fatal; quota deferrals are
        returned explicitly and leave all local evidence available.
        """
        from huggingface_hub import CommitOperationAdd
        from huggingface_hub.errors import HfHubHTTPError

        with self._lock():
            files = sorted(self._pending.rglob("*.json"))
            state = self._state()
            now = time.time()
            if not files:
                return {**state, "pending_records": 0}
            if now < state.get("next_attempt_unix", 0):
                return {**state, "pending_records": len(files)}
            state = {"state": "attempting", "next_attempt_unix": now + 300}
            receipt = self.cache / "publication.json"
            _atomic_record(receipt, dumps(state))
            batch = files[:500]
            try:
                parent = self.api.repo_info(self.repo_id, repo_type="dataset").sha
                existing = set(self.api.list_repo_files(self.repo_id, repo_type="dataset", revision=parent))
                operations = []
                for f in batch:
                    path = f.relative_to(self._pending).as_posix()
                    if path in existing:
                        self._fetch(path, revision=parent)
                        if self._cached(path) != f.read_bytes():
                            raise AppendConflict(path)
                    else:
                        operations.append(CommitOperationAdd(path_in_repo=path, path_or_fileobj=f.read_bytes()))
                revision = self.api.create_commit(
                    self.repo_id, repo_type="dataset", parent_commit=parent,
                    operations=operations,
                    commit_message=f"tournament: batch {len(operations)} immutable records",
                ).oid if operations else parent
                # Cache before removing each durable outbox item. A crash at any
                # boundary leaves either its outbox or its published snapshot.
                for f in batch:
                    _atomic_record(self._root / f.relative_to(self._pending), f.read_bytes())
                    f.unlink()
                state.update(state="published", revision=revision,
                             published_records=len(batch), added_records=len(operations))
            except HfHubHTTPError as error:
                response = error.response
                code = response.status_code if response is not None else None
                if code not in (409, 429, 500, 502, 503, 504):
                    raise
                if code == 429:
                    retry = response.headers.get("retry-after", "")
                    try:
                        deadline = time.time() + float(retry)
                    except ValueError:
                        try:
                            deadline = parsedate_to_datetime(retry).timestamp()
                        except (TypeError, ValueError, OverflowError):
                            deadline = time.time() + 3600
                    state["next_attempt_unix"] = max(state["next_attempt_unix"], deadline)
                state.update(state="deferred", error=str(error), status_code=code)
            _atomic_record(receipt, dumps(state))
            return {**state, "pending_records": len(list(self._pending.rglob("*.json")))}

    def read_all(self, prefix: str) -> Iterator[tuple[str, bytes]]:
        listed = {e.path for e in self.api.list_repo_tree(
            self.repo_id, path_in_repo=prefix, recursive=True, repo_type="dataset")
            if e.path.endswith(".json")}
        with self._lock():
            pending = {p.relative_to(self._pending).as_posix(): p.read_bytes()
                       for p in (self._pending / prefix).rglob("*.json")}
            cached = {p.relative_to(self._root).as_posix()
                      for p in (self._root / prefix).rglob("*.json")}
            rows = []
            for path in sorted(listed | pending.keys() | cached):
                data = self._cached(path)
                if path in listed and data is None:
                    self._fetch(path)
                    data = self._cached(path)
                if path in pending:
                    if data is not None and data != pending[path]:
                        raise AppendConflict(path)
                    data = pending[path]
                rows.append((path, data))
        yield from rows
