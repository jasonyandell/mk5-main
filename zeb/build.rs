//! Embeds the field net: writes `$OUT_DIR/net.rs` with the blob's bytes and
//! its sha256, so the id `netId()` reports is computed from the very bytes
//! that ship and cannot drift from them.

use std::path::Path;
use std::{env, fs};

use sha2::{Digest, Sha256};

/// The net the package plays, as `zeb/scripts/export_net.py` wrote it.
const NET: &str = "net/margin_d250k_h128x64.fnet";

fn main() {
    println!("cargo::rerun-if-changed={NET}");
    let path = Path::new(&env::var("CARGO_MANIFEST_DIR").expect("cargo sets CARGO_MANIFEST_DIR")).join(NET);
    let blob = fs::read(&path).unwrap_or_else(|e| panic!("reading {}: {e}", path.display()));
    let id: String = Sha256::digest(&blob).iter().map(|b| format!("{b:02x}")).collect();
    let out = Path::new(&env::var("OUT_DIR").expect("cargo sets OUT_DIR")).join("net.rs");
    let source = format!(
        "/// The field net's blob (`FieldNet.to_bytes()`).\n\
         const NET: &[u8] = include_bytes!({path:?});\n\
         /// The blob's sha256, lowercase hex.\n\
         const NET_ID: &str = {id:?};\n"
    );
    fs::write(&out, source).unwrap_or_else(|e| panic!("writing {}: {e}", out.display()));
}
