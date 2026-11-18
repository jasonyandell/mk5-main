#!/usr/bin/env bash
# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0

# From a checkout to pkg/: the .wasm, its JavaScript glue and their types.
#
# rust-toolchain.toml pins the compiler, Cargo.lock (--locked) pins every crate, wasm-pack
# fetches the wasm-bindgen that matches the lock and its own pinned wasm-opt, and the source
# paths that panic messages carry are remapped, so no local path reaches the bytes.
# Needs rustup and wasm-pack 0.15.0 exactly, as zeb/scripts/build.sh does.
set -euo pipefail
cd "$(dirname "$0")/.."
want="wasm-pack 0.15.0"
have="$(wasm-pack --version)"
if [ "$have" != "$want" ]; then
  echo "build.sh: need $want, have $have (cargo install wasm-pack --version 0.15.0 --locked)" >&2
  exit 1
fi
repo="$(cd .. && pwd)"
cargo_home="${CARGO_HOME:-$HOME/.cargo}"
export CARGO_ENCODED_RUSTFLAGS="--remap-path-prefix=$repo=mk5-main"$'\x1f'"--remap-path-prefix=$cargo_home=cargo"
rm -rf pkg
wasm-pack build --target web --release --no-pack --out-dir pkg -- --locked
# wasm-pack writes a catch-all .gitignore into pkg/, which npm would honor
# and so leave the build out of the tarball.
rm pkg/.gitignore
