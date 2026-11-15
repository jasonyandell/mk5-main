// SPDX-FileCopyrightText: Jason Yandell
// SPDX-License-Identifier: CC0-1.0

//! The WebAssembly surface: the `walt1 play` request as text, the response as a JSON line.
//! `index.js` wraps it in objects.

use wasm_bindgen::prelude::*;

/// One `walt1 play` request in, one JSON line out (WALT.md §6).
#[wasm_bindgen]
pub fn handle(request: &str) -> String {
    crate::handle(request)
}

/// The package version.
#[wasm_bindgen]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}
