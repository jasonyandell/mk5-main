# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0

"""Materialize a participant's pinned bytes into a local cache, checking every hash.

Nothing is trusted by name: the npm tarball must match its sha512 integrity,
each Plunge file its sha256, the Demon's and native Zeb's source each pinned git
tree (native Zeb's shipped files and binary by sha256 too). A
mismatch is an error, never a fallback.
"""
from __future__ import annotations

import base64
import hashlib
import os
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request
from pathlib import Path

from .participants import Participant

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CACHE = ROOT / "scratch" / "tournament" / "artifacts"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _integrity(data: bytes) -> str:
    return "sha512-" + base64.b64encode(hashlib.sha512(data).digest()).decode()


def materialize(p: Participant, cache: Path = DEFAULT_CACHE) -> Path:
    """The directory holding p's verified artifacts (created once, then reused)."""
    target = cache / p.id
    done = target / ".verified"
    if done.exists():
        return target
    tmp = Path(tempfile.mkdtemp(prefix=f".{p.id}-", dir=_ensure(cache)))
    try:
        if p.kind == "zeb-npm":
            _zeb(p, tmp)
        elif p.kind == "plunge-walt":
            _plunge(p, tmp)
        elif p.kind == "arena":
            _arena(p)
        elif p.kind == "demon":
            # STRATEGY-FUSION FENCE. The Demon: a fenced tournament opponent; measuring stick only.
            from .demon_fenced_seat import build
            build(p, tmp, cache / ".demon-target")
        elif p.kind == "zeb-native":
            from .zeb_native_seat import build
            build(p, tmp, cache / ".zeb-native-target")
        else:
            raise ValueError(f"unknown participant kind {p.kind!r}")
        (tmp / ".verified").write_text(p.id + "\n")
        if target.exists():
            shutil.rmtree(target)
        os.replace(tmp, target)
    finally:
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)
    return target


def _ensure(d: Path) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    return d


def _zeb(p: Participant, out: Path) -> None:
    spec = f"{p.pin['package']}@{p.pin['version']}"
    subprocess.run(["npm", "pack", spec, "--silent", "--pack-destination", str(out)],
                   check=True, capture_output=True, text=True)
    (tgz,) = out.glob("*.tgz")
    data = tgz.read_bytes()
    if _integrity(data) != p.pin["integrity"]:
        raise RuntimeError(f"{spec}: tarball integrity {_integrity(data)} != pinned {p.pin['integrity']}")
    with tarfile.open(tgz) as t:
        t.extractall(out, filter="data")
    tgz.unlink()


def _plunge(p: Participant, out: Path) -> None:
    for path, want in p.pin["files"].items():
        url = f"https://raw.githubusercontent.com/{p.pin['repo']}/{p.pin['commit']}/{path}"
        with urllib.request.urlopen(url, timeout=60) as r:
            data = r.read()
        if _sha256(data) != want:
            raise RuntimeError(f"{url}: sha256 {_sha256(data)} != pinned {want}")
        (out / Path(path).name).write_bytes(data)


def _arena(p: Participant) -> None:
    ckpt = p.pin.get("checkpoint")
    if ckpt:
        data = (ROOT / ckpt).read_bytes()
        if _sha256(data) != p.pin["checkpoint_sha256"]:
            raise RuntimeError(f"{ckpt}: sha256 does not match the pin")
