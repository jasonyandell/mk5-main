# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0

"""Gus's neurons, read with the probe kit (#194 on Gus; #203's split: Codex trains, the Lem desk
reads). Read-only against `lem/observer`: its own `encode` builds the input and its own
`Observer.forward(tokens, inspect=True)` returns each block's MLP pre-activations. Nothing in
`lem/observer` is changed or hooked.

A sample row is a decision: the seat to move, at a recorded position of a pour. Gus's input is that
seat's original seven tiles, the declaration, the bidder, bid 30 and the public plays so far, then
the mover's own seat token. The model scores the next tile at that last token, and the units read
are both blocks' 256 GELU pre-activations there. A pole is a neuron's pre-activation (`+`) or its
negation (`−`), as for the field net. "On" means pre > 0, the GELU's open side.

Only rows from Gus's held-out **test** deals are read (`lem.observer.data.partition`, the same
hash of the four hands that split its training corpus).

    python -u -m lem.probe gus-sample SAMPLE.npz LABELS.npz OUT_DIR       # test-split rows, their tokens and labels
    python -u -m lem.probe gus-audit  OUT_DIR CKPT AUDIT_DIR [--untrained SEED]
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from lem.probe.net import Activations

BATCH = 4096


def _local_deals(sample) -> tuple[list[str], np.ndarray]:
    """Each row's deal number inside its own pour (the merged sample renumbers later pours)."""
    from scenario.atlas import load
    sources = [str(s) for s in sample.source]
    local = np.empty(len(sample), np.int64)
    for k, src in enumerate(sources):
        m = sample.pour == k
        local[m] = load(src).deal[sample.index[m].astype(np.int64)]
    return sources, local


def build(sample_path: str | Path, labels_path: str | Path, out_dir: str | Path, log=print) -> dict:
    """Write Gus's audit rows: the sample's rows on Gus's test deals, with tokens, lengths and labels."""
    from lem.observer.data import SEAT, encode, partition
    from lem.probe.sample import COLUMNS, Sample
    s = Sample(sample_path)
    sources, local = _local_deals(s)
    keep = np.zeros(len(s), bool)
    hands_of = []
    for k, src in enumerate(sources):
        with np.load(Path(src) / "pour.npz") as z:
            hands = z["hands"]
        hands_of.append(hands)
        m = np.flatnonzero(s.pour == k)
        part = {int(d): partition(hands[d].tolist()) for d in np.unique(local[m])}
        keep[m] = [part[int(d)] == "test" for d in local[m]]
    rows = np.flatnonzero(keep)
    seqs = []
    for r in rows:
        k, v = int(s.pour[r]), int(s.viewer[r])
        hand = int(hands_of[k][local[r], v])
        plays = [(int(a), int(b)) for a, b in s.plays[r, :int(s.n_plays[r])]]
        seqs.append(np.append(encode(hand, v, int(s.bidder[r]), int(s.decl[r]), plays), SEAT))
    lengths = np.array([len(q) for q in seqs], np.int64)
    tokens = np.zeros((len(seqs), int(lengths.max())), np.int64)
    for i, q in enumerate(seqs):
        tokens[i, :len(q)] = q
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cols = {c: getattr(s, c)[rows] for c in COLUMNS}
    np.savez_compressed(out / "sample.npz", **cols, source=np.array(sources), layout=np.array(s.layout))
    with np.load(labels_path) as z:
        np.savez_compressed(out / "labels.npz", labels=z["labels"][rows], names=z["names"], identity=z["identity"])
    np.savez_compressed(out / "tokens.npz", tokens=tokens, lengths=lengths, rows=rows)
    info = {"sample": str(sample_path), "labels": str(labels_path), "rows": int(len(rows)),
            "of": int(len(s)), "deals": int(len(np.unique(s.pour[rows].astype(np.int64) * 10_000_000 + local[rows]))),
            "split": "test (lem.observer.data.partition)"}
    (out / "rows.json").write_text(json.dumps(info, indent=1))
    log(f"{out}: {len(rows):,} of {len(s):,} rows on Gus's test deals")
    return info


def load_model(ckpt: str | Path | None, untrained_seed: int | None = None, config: dict | None = None):
    """A saved Gus checkpoint, or an untrained twin: `Observer` built fresh under a torch seed."""
    import torch
    from lem.observer.data import VERSION
    from lem.observer.model import Config, Observer
    if untrained_seed is not None:
        torch.manual_seed(untrained_seed)
        m = Observer(Config(**config)).eval()
        return m, {"untrained_seed": untrained_seed, "config": config}
    saved = torch.load(ckpt, map_location="cpu", weights_only=True)
    if saved["tokenizer"] != VERSION:
        raise ValueError("checkpoint has a different input encoding")
    m = Observer(Config(**saved["config"])).eval()
    m.load_state_dict(saved["model"])
    return m, {"path": str(ckpt), "sha256": hashlib.sha256(Path(ckpt).read_bytes()).hexdigest(),
               "step": int(saved["step"]), "config": saved["config"], "data_sha256": saved["data_sha256"]}


def forward(model, tokens: np.ndarray, lengths: np.ndarray) -> Activations:
    """Both blocks' MLP pre-activations and the tile logits at each row's last token. A causal model
    ignores the right padding. Two-block models only (the kit's two-layer `Activations`)."""
    import torch
    if len(model.blocks) != 2:
        raise ValueError("the kit's Activations holds two layers; this model has %d" % len(model.blocks))
    pre = [[], []]
    logits = []
    with torch.inference_mode():
        for at in range(0, len(tokens), BATCH):
            t = torch.from_numpy(tokens[at:at + BATCH])
            last = torch.from_numpy(lengths[at:at + BATCH] - 1)
            out, traces = model(t, inspect=True)
            idx = torch.arange(len(t))
            for b in range(2):
                pre[b].append(traces[b]["pre"][idx, last].float().numpy())
            logits.append(out[idx, last].float().numpy())
    z1, z2 = np.concatenate(pre[0]), np.concatenate(pre[1])
    return Activations(z1, z1, z2, z2, np.concatenate(logits))
