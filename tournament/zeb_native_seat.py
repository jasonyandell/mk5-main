# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0

"""The native Zeb seat's pinned build: a Zeb release's shipped JavaScript bidder
over the identical Rust player compiled natively.

Nothing here trusts a name. `build` checks that the pinned commit has the pinned
git tree at each Rust source directory and the pinned sha256 of each shipped
JavaScript file; takes the wasm bindings from the integrity-checked 0.4.0 npm
tarball (the wasm's sha256 is pinned; the native host never runs it, but the
shipped `index.js` imports its bindings); builds the player from the pinned
trees plus the JSON-lines wrapper in `tournament/zeb_native/` (an rlib output,
the JSON dependency, a CLI; nothing in the search or policy); and checks the
binary's sha256 against the pin.

0.4.1 changes no Rust (its pinned trees are 0.4.0's), so both releases are built
from `NATIVE_BUILD_COMMIT`, the 0.4.0 release, and share one binary.
"""
from __future__ import annotations

import base64
import hashlib
import io
import shutil
import subprocess
import tarfile
from pathlib import Path

from .participants import Participant

ROOT = Path(__file__).resolve().parent.parent
PATCH = Path(__file__).resolve().parent / "zeb_native" / "zeb-native.patch"
NATIVE_BUILD_COMMIT = "29f519953467f6aacd78bd17700f3c67fda6b4eb"  # zeb-v0.4.0
BUILD_FILES = ("zeb/Cargo.toml", "zeb/Cargo.lock", "zeb/build.rs", "zeb/rust-toolchain.toml",
               "zeb/LICENSE")
NPM_PACKAGE = "@jasonyandell/zeb@0.4.0"
NPM_INTEGRITY = "sha512-7DhOCStdZbGEkh+RBwdDWOYgptwS+OYEPoGLF3BVJwCs/A0rmMNdq+X3fBrgQ2C55WWRXb3dNVmFY9jucFDKDA=="
BINDINGS_SHA256 = "5805b03ddc757c019809d7957ace6082b2f5c1718c84e6cb1434a00b2ca32555"  # pkg/zeb.js
PATCH_SHA256 = "b284493f0be6be719f130eee79dd1a9ab56ea8b4dadba5d6ec25e6d070d20839"
BINARY = "zeb-native"


def _git(*args: str, binary: bool = False):
    out = subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=not binary)
    return out.stdout


def _has(obj: str) -> bool:
    return subprocess.run(["git", "cat-file", "-e", obj], cwd=ROOT, capture_output=True).returncode == 0


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _ensure_commits(*commits: str) -> None:
    for c in commits:
        if not _has(c):  # a shallow clone: GitHub serves any reachable sha
            subprocess.run(["git", "fetch", "--quiet", "origin", c], cwd=ROOT, check=True)


def _shipped(pin: dict, name: str) -> bytes:
    return _git("show", f"{pin['commit']}:zeb/{name.removeprefix('package/')}", binary=True)


def check_pin(p: Participant) -> None:
    """The pinned commit holds the pinned trees and files; the build commit holds the trees too."""
    pin = p.pin
    _ensure_commits(pin["commit"], NATIVE_BUILD_COMMIT)
    for commit in (pin["commit"], NATIVE_BUILD_COMMIT):
        for path, tree in pin["trees"].items():
            have = _git("rev-parse", f"{commit}:{path}").strip()
            if have != tree:
                raise RuntimeError(f"{p.id}: {commit}:{path} is tree {have}, the pin says {tree}")
    for name, want in pin["files"].items():
        got = _sha256(_shipped(pin, name))
        if got != want:
            raise RuntimeError(f"{p.id}: zeb/{name} sha256 {got} != pinned {want}")


def build(p: Participant, out: Path, target_dir: Path) -> None:
    """Fill `out` with package/ (the shipped JavaScript and bindings) and the native binary."""
    check_pin(p)
    if _sha256(PATCH.read_bytes()) != PATCH_SHA256:
        raise RuntimeError(f"{PATCH} is not the wrapper the trials built")
    pin = p.pin
    pkg = out / "package"
    (pkg / "pkg").mkdir(parents=True)
    for name in pin["files"]:
        (pkg / Path(name).name).write_bytes(_shipped(pin, name))
    # wasm bindings from the registry's 0.4.0 tarball, integrity-checked
    packed = out / "npm"
    packed.mkdir()
    subprocess.run(["npm", "pack", NPM_PACKAGE, "--silent", "--pack-destination", str(packed)],
                   check=True, capture_output=True, text=True)
    (tgz,) = packed.glob("*.tgz")
    got = "sha512-" + base64.b64encode(hashlib.sha512(tgz.read_bytes()).digest()).decode()
    if got != NPM_INTEGRITY:
        raise RuntimeError(f"{NPM_PACKAGE}: tarball integrity {got} != pinned {NPM_INTEGRITY}")
    with tarfile.open(tgz) as t:
        t.extractall(packed, filter="data")
    wasm = (packed / "package/pkg/zeb_bg.wasm").read_bytes()
    bindings = (packed / "package/pkg/zeb.js").read_bytes()
    if _sha256(wasm) != pin["wasm_sha256"]:
        raise RuntimeError(f"{p.id}: wasm sha256 {_sha256(wasm)} != pinned {pin['wasm_sha256']}")
    if _sha256(bindings) != BINDINGS_SHA256:
        raise RuntimeError(f"{p.id}: zeb.js sha256 {_sha256(bindings)} != {BINDINGS_SHA256}")
    (pkg / "pkg" / "zeb_bg.wasm").write_bytes(wasm)
    (pkg / "pkg" / "zeb.js").write_bytes(bindings)
    shutil.rmtree(packed)
    # the player, from the pinned trees at the build commit plus the wrapper patch
    src = out / "src"
    for path, tree in pin["trees"].items():
        (src / path).mkdir(parents=True, exist_ok=True)
        with tarfile.open(fileobj=io.BytesIO(_git("archive", "--format=tar", tree, binary=True))) as t:
            t.extractall(src / path, filter="data")
    for f in BUILD_FILES:
        (src / f).write_bytes(_git("show", f"{NATIVE_BUILD_COMMIT}:{f}", binary=True))
    subprocess.run(["patch", "-p1", "-s", "-i", str(PATCH)], cwd=src, check=True)
    subprocess.run(["cargo", "build", "--release", "--locked", "--quiet", "--bin", BINARY,
                    "--manifest-path", str(src / "zeb/Cargo.toml"),
                    "--target-dir", str(target_dir)], check=True)
    binary = target_dir / "release" / BINARY
    # The pinned binary hash names the binary the trials measured. A rebuild embeds its own
    # build paths, so its bytes differ from it; what the rebuild must match is every input
    # (trees, files, wrapper, toolchain, lockfile). Its own hash is recorded and checked on load.
    (out / (BINARY + ".sha256")).write_text(_sha256(binary.read_bytes()) + "\n")
    shutil.copy2(binary, out / BINARY)
    shutil.rmtree(src)


def verify_binary(artifacts: Path) -> Path:
    """The materialized binary is the one this machine's build recorded."""
    binary = artifacts / BINARY
    if _sha256(binary.read_bytes()) != (artifacts / (BINARY + ".sha256")).read_text().strip():
        raise RuntimeError(f"{binary} is not the binary that was built")
    return binary
