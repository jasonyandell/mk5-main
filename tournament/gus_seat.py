# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0
"""Frozen Gus releases in the existing tournament, under their bid-30 contract."""
from __future__ import annotations

import hashlib
from pathlib import Path
import re

from .participants import Participant, ZEB_041_CONTESTED

ROOT = Path(__file__).resolve().parents[1]
LADDER = "marks7-bid30"
ANCHOR = "zeb-0.4.1-contested-bid30"
# These include the complete bidder, public encoder, forward pass and adapter.
SOURCE_FILES = ("tournament/gus_seat.py", "lem/observer/play.py", "lem/observer/model.py",
                "lem/observer/data.py", "arena/bidders.py", "arena/hand_metrics.py", "arena/auction.py",
                "scenario/rules.py", "scenario/record.py", "forge/zeb/game.py",
                "forge/zeb/types.py", "forge/oracle/tables.py")


def source_hashes():
    return {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in SOURCE_FILES}


def verify_sources(pin):
    for path, want in pin["source_files"].items():
        if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != want:
            raise RuntimeError(f"Gus bid30 source changed: {path}; use the pinned source commit")


def gus_participant(generation, revision, sha256, commit):
    if not re.fullmatch(r"g\d{3}", generation) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("a generation and immutable Hub commit are required")
    return Participant(
        f"gus-{generation}-bid30-{sha256[:8]}", "gus-observer",
        {"repo": "jasonyandell/gus", "revision": revision, "file": "gus.pt",
         "sha256": sha256, "source_commit": commit, "source_files": source_hashes()},
        {"ladders": [LADDER], "bidder": "Bid30Bidder", "declarations": list(range(7)),
         "device": "cpu", "dtype": "float32", "torch_threads": 1,
         "budget": "one forward pass; legal argmax; lowest tile id breaks exact ties"},
        ("torch", "gus-observer"),
        "Frozen clean observer; original own tiles and public bid30/pip/play prefix only. "
        "Two copies per partnership; fixed common Bid30Bidder. No learned bidding.")


def zeb_bid30(commit):
    p = ZEB_041_CONTESTED
    return Participant(ANCHOR, p.kind,
        {**p.pin, "bid30_source_commit": commit, "source_files": source_hashes()},
        {**p.config, "ladders": [LADDER], "bidder": "Bid30Bidder"},
        (*p.needs, "gus-observer"),
        "Pinned Zeb 0.4.1 native play, 160 worlds and seed0, with the common "
        "Bid30Bidder replacing its auction. Anchor only of the separate bid30 ladder.")


def download(p, out):
    from huggingface_hub import hf_hub_download
    path = Path(hf_hub_download(p.pin["repo"], p.pin["file"], revision=p.pin["revision"]))
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != p.pin["sha256"]:
        raise RuntimeError("Gus checkpoint does not match its immutable pin")
    (out / "gus.pt").write_bytes(data)


def make_seat(p, artifacts):
    import torch
    from arena.bidders import Bid30Bidder
    from lem.observer.play import ObserverPlay
    from .seats import ArenaSeat

    verify_sources(p.pin)
    checkpoint = artifacts / "gus.pt"
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != p.pin["sha256"]:
        raise RuntimeError("cached Gus checkpoint does not match its immutable pin")
    if (p.config["device"], p.config["dtype"], p.config["torch_threads"]) != ("cpu", "float32", 1):
        raise ValueError("the registered Gus adapter uses CPU float32 and one thread")
    torch.set_num_threads(1)
    return ArenaSeat(Bid30Bidder(), ObserverPlay(checkpoint, "cpu"))
