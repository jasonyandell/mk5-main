# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: MIT

"""Export a field net to the blob the zeb package embeds.

    python -u -m zeb.scripts.export_net NET.npz [OUT.fnet]

Reads a net as `scenario.field_train` saves it, writes `FieldNet.to_bytes()`
(default: zeb/net/<name>.fnet), reads it back to check the round trip, and
prints its sha256 — the id `netId()` reports once the package embeds it.
Deterministic: the same net always writes the same bytes.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np

from scenario.field import FieldNet

NET_DIR = Path(__file__).resolve().parents[1] / "net"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("net", type=Path, help="a field net .npz")
    ap.add_argument("out", type=Path, nargs="?", help="the blob to write (default: zeb/net/<name>.fnet)")
    args = ap.parse_args()
    net = FieldNet.load(args.net)
    blob = net.to_bytes()
    back = FieldNet.from_bytes(blob)
    for k in ("w1", "b1", "w2", "b2", "w3", "b3", "s1", "s2"):
        assert np.array_equal(getattr(back, k), getattr(net, k)), f"{k} did not survive the round trip"
    out = args.out or NET_DIR / f"{args.net.stem}.fnet"
    out.write_bytes(blob)
    print(f"{out}: {len(blob):,} bytes, h1={net.b1.shape[0]} h2={net.b2.shape[0]}, "
          f"sha256 {hashlib.sha256(blob).hexdigest()}")


if __name__ == "__main__":
    main()
