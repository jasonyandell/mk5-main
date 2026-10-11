#!/usr/bin/env bash
# From a checkout to pkg/: the .wasm, its JavaScript glue and their types.
#
# Deterministic: rust-toolchain.toml pins the compiler, Cargo.lock (--locked)
# pins every crate, wasm-pack fetches the wasm-bindgen that matches the lock
# and its own pinned wasm-opt, and the source paths that panic messages
# carry are remapped, so no local path reaches the bytes. A checkout builds
# the same bytes every time. A checkout at another path builds an equivalent
# .wasm with different bytes, because Cargo hashes the absolute path of
# ../scenario/core (outside this crate's workspace) into symbol names.
# Needs rustup and wasm-pack >= 0.13.
set -euo pipefail
cd "$(dirname "$0")/.."
repo="$(cd .. && pwd)"
cargo_home="${CARGO_HOME:-$HOME/.cargo}"
export CARGO_ENCODED_RUSTFLAGS="--remap-path-prefix=$repo=mk5-main"$'\x1f'"--remap-path-prefix=$cargo_home=cargo"
rm -rf pkg
wasm-pack build --target web --release --no-pack --out-dir pkg -- --locked
# wasm-pack writes a catch-all .gitignore into pkg/, which npm would honor
# and so leave the build out of the tarball.
rm pkg/.gitignore
