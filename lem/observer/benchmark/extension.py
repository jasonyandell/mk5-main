# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0
"""Immutable lazy records: the unit is (recipe, frozen row, sample index).

Completed units survive interruption; request order and batch size are not
identities. Timings belong in logs, never inside reproducible evidence bytes.
"""
import fcntl
import json
import os
from pathlib import Path
import tempfile

from .build import canonical, digest, identity


def atomic(path, value):
    path = Path(path)
    data = (canonical(value) + '\n').encode()
    fd, name = tempfile.mkstemp(prefix='pending-', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(data); stream.flush(); os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def sample_seed(coupling, row, sample, stream):
    """Named counter-based seeds: learner identity deliberately excluded."""
    if sample < 0 or not stream:
        raise ValueError('nonnegative sample and named stream required')
    return int(identity([coupling, row['id'], identity(row['question']), sample, stream])[:16], 16)


class Extension:
    def __init__(self, root, recipe):
        self.root = Path(root); self.root.mkdir(parents=True, exist_ok=True)
        self.recipe = recipe; self.id = identity(recipe)
        self.records = self.root / 'records'; self.records.mkdir(exist_ok=True)
        self.locks = self.root / '.locks'; self.locks.mkdir(exist_ok=True)
        with (self.locks / 'recipe').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            path = self.root / 'recipe.json'
            expected = {'id': self.id, 'recipe': recipe}
            if path.exists():
                if path.read_bytes() != (canonical(expected) + '\n').encode():
                    raise ValueError('extension recipe changed; use a new directory')
            else:
                atomic(path, expected)

    def key(self, row, sample):
        if sample < 0:
            raise ValueError('sample must be nonnegative')
        return {'extension': self.id, 'row_id': row['id'],
                'question_sha256': identity(row['question']), 'sample': sample}

    def read(self, path, key=None):
        record = json.loads(path.read_bytes())
        if (set(record) != {'key', 'result', 'sha256'}
                or record['sha256'] != identity([record['key'], record['result']])
                or record['key']['extension'] != self.id
                or path.stem != identity(record['key'])
                or path.read_bytes() != (canonical(record) + '\n').encode()
                or (key is not None and record['key'] != key)):
            raise ValueError('extension record checksum or identity mismatch')
        return record

    def ensure(self, row, sample, produce, recheck=False):
        key = self.key(row, sample); name = identity(key)
        path = self.records / (name + '.json')
        with (self.locks / name).open('a') as lock:
            # A busy unit is retried by the next invocation, not a long wait.
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return None
            existing = self.read(path, key) if path.exists() else None
            if existing and not recheck:
                return existing
            result = produce(row, sample)
            record = {'key': key, 'result': result, 'sha256': identity([key, result])}
            if existing:
                if record != existing:
                    raise ValueError('recomputation changed immutable evidence')
            else:
                atomic(path, record)
            return record

    def snapshot(self):
        """A sorted immutable export; later appends never rewrite an old one."""
        files = {}
        for path in sorted(self.records.glob('*.json')):
            self.read(path)
            files[path.name] = digest(path)
        snapshot = {'extension': self.id, 'records': files}
        directory = self.root / 'snapshots'; directory.mkdir(exist_ok=True)
        target = directory / (identity(snapshot) + '.json')
        if not target.exists():
            atomic(target, snapshot)
        elif target.read_bytes() != (canonical(snapshot) + '\n').encode():
            raise ValueError('snapshot checksum mismatch')
        return target
