# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0
"""Compile weighted splats without reading hidden-world arrays into memory.

Per-source caches are numeric, resumable at source boundaries and memory mapped.
Model tokens come exclusively from the original observer encoder/referee.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

import numpy as np

from ..data import QUERIES, VERSION, load_deals, prepare

SCHEMA = "gus-weighted-splat-cache-v1"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def histograms(outcomes, weights, n_actions, n_worlds):
    """One bounded block; exact weighted marginal, checked before float32 cast."""
    na, nw = np.asarray(n_actions, np.int64), np.asarray(n_worlds, np.int64)
    if (na < 1).any() or (na > 7).any() or (nw < 1).any():
        raise ValueError("invalid action/world counts")
    if len(weights) != nw.sum() or len(outcomes) != (na * nw).sum():
        raise ValueError("misaligned splat arrays")
    if not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("invalid scenario weights")
    m = np.asarray(outcomes, np.int16)
    if (np.abs(m) > 42).any() or ((m + 42) % 2).any():
        raise ValueError("outcome is not a lawful integral points margin")
    lengths = np.repeat(nw, na)
    action = np.repeat(np.arange(len(lengths)), lengths)
    starts = np.r_[0, np.cumsum(lengths)[:-1]]
    world_starts = np.repeat(np.r_[0, np.cumsum(nw)[:-1]], na)
    wi = np.repeat(world_starts - starts, lengths) + np.arange(len(m))
    h = np.bincount(action * 43 + (m + 42) // 2, weights=weights[wi],
                    minlength=len(lengths) * 43).reshape(-1, 43)
    totals = h.sum(-1, keepdims=True)
    if (totals <= 0).any():
        raise ValueError("zero total scenario weight")
    return h / totals


def build(source, out):
    source, out = Path(source), Path(out)
    raw = (source / "manifest.json").read_bytes()
    manifest = json.loads(raw)
    if (manifest["bidder"], manifest["bid"], manifest["utility"]) != (0, 30, "make"):
        raise ValueError("requires lawful seat-0 bid30 make-utility pour")
    if out.exists():
        raise ValueError("cache already exists; use a fresh output directory")
    out.mkdir(parents=True)
    n = manifest["deals"]
    # Preserve original source order; split by physical hands before any views.
    records, provenance = load_deals([source], n)
    by_seed = {r["seed"]: r for r in records}
    records = [by_seed[s] for s in manifest["deal_seeds"]]
    arrays = {}
    for name, shape, dtype in (("x", (4*n, 68), "uint8"), ("targets", (4*n,28), "uint8"),
            ("own_positions", (4*n,7), "int64"), ("split", (4*n,), "uint8"),
            ("possible", (4*n,28), "uint32"), ("legal", (n,28), "uint32"),
            ("lookup", (4*n,7,7), "int32"), ("seed", (n,), "int64"),
            ("hands", (n,4), "uint32"), ("settled", (n,28), "bool")):
        arrays[name] = np.lib.format.open_memmap(out / f"{name}.npy", mode="w+", dtype=dtype, shape=shape)
    arrays["lookup"][:] = 0
    bits = 1 << np.arange(28)
    from scenario.record import empty_record
    from scenario.rules import rules_for
    for lo in range(0,n,256):
        chunk = records[lo:lo+256]
        d = prepare(chunk)
        sl = slice(4*lo,4*(lo+len(chunk)))
        arrays["x"][sl], arrays["targets"][sl] = d["x"], d["targets"]
        arrays["own_positions"][sl] = np.broadcast_to(QUERIES, d["own"].shape)[d["own"]].reshape(-1,7)
        arrays["possible"][sl] = (d["possible"] * bits).sum(-1)
        arrays["legal"][lo:lo+len(chunk)] = (d["legal"][::4] * bits).sum(-1)
        arrays["split"][sl] = np.select([d["split"]=="validation",d["split"]=="test"],[1,2],default=0)
        for i,r in enumerate(chunk,lo):
            arrays["seed"][i], arrays["hands"][i] = r["seed"], r["hands"]
            rec, rules = empty_record(0), rules_for(r["decl"])
            for k,(_,tile) in enumerate(r["plays"]):
                arrays["settled"][i,k] = rec.banked[0] >= 30 or rec.banked[1] > 12
                rec = rec.advance(tile,rules)
    print(json.dumps({"source":str(source),"tokens_ready":n}),flush=True)
    del records, by_seed
    # Extract only label arrays; worlds.npy is deliberately never opened.
    temporary = out / "raw-labels"
    temporary.mkdir()
    with zipfile.ZipFile(source / "pour.npz") as z:
        columns = {}
        for name in ("deal","n_plays","n_actions","n_worlds","actions","q","weights","outcomes","plays"):
            z.extract(f"{name}.npy", temporary)
            columns[name] = np.load(temporary / f"{name}.npy", mmap_mode="r", allow_pickle=False)
    na,nw = columns["n_actions"].astype(np.int64),columns["n_worlds"].astype(np.int64)
    ao,wo,oo = (np.r_[0,np.cumsum(v)] for v in (na,nw,na*nw))
    a = int(ao[-1])
    h = np.lib.format.open_memmap(out / "histograms.npy", mode="w+", dtype="float32",shape=(a+1,43))
    tiles = np.lib.format.open_memmap(out / "candidates.npy",mode="w+",dtype="int64",shape=(a+1,))
    h[0], tiles[0] = 0,0
    tiles[1:] = columns["actions"]
    actors = columns["plays"][:,:,0]
    ordinal = np.zeros((n,28),np.int8)
    for seat in range(4):
        match = actors == seat
        ordinal[match] = (np.cumsum(match,axis=1)-1)[match]
    max_error = 0.
    seen = np.zeros((n,28),bool)
    for lo in range(0,len(na),256):
        hi = min(lo+256,len(na))
        block = histograms(columns["outcomes"][oo[lo]:oo[hi]], columns["weights"][wo[lo]:wo[hi]],na[lo:hi],nw[lo:hi])
        error = float(np.max(np.abs(block[:,30:].sum(-1)-columns["q"][ao[lo]:ao[hi]])))
        if error > 1e-7:
            raise ValueError(f"histogram make tail disagrees with teacher q: {error}")
        max_error = max(max_error,error)
        h[ao[lo]+1:ao[hi]+1] = block
        for p in range(lo,hi):
            d,k = int(columns["deal"][p]),int(columns["n_plays"][p])
            actor = int(actors[d,k])
            ids = columns["actions"][ao[p]:ao[p+1]].astype(np.int64)
            if seen[d,k] or len(set(ids.tolist())) != len(ids) or int((1 << ids).sum()) != arrays["legal"][d,k]:
                raise ValueError("duplicate position or target candidate set differs from referee")
            seen[d,k] = True
            arrays["lookup"][4*d+actor,ordinal[d,k],:na[p]] = np.arange(ao[p]+1,ao[p+1]+1)
    for v in [*arrays.values(),h,tiles]:
        v.flush()
    del columns
    shutil.rmtree(temporary)
    hashes = {p.name:digest(p) for p in sorted(out.glob("*.npy"))}
    receipt = {"schema":SCHEMA,"tokenizer":VERSION,"source":provenance[0],"deals":n,
               "positions":len(na),"actions":a,"max_make_tail_error":max_error,"files":hashes,
               "split_deals":{s:int((arrays["split"][::4]==i).sum()) for i,s in enumerate(("train","validation","test"))}}
    (out / "manifest.json").write_text(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps(receipt),flush=True)


class Corpus:
    def __init__(self, roots):
        self.roots = [Path(p) for p in roots]
        self.manifests = [json.loads((p/"manifest.json").read_text()) for p in self.roots]
        if any(m["schema"] != SCHEMA or m["tokenizer"] != VERSION for m in self.manifests):
            raise ValueError("cache schema mismatch")
        for root,manifest in zip(self.roots,self.manifests):
            for name,want in manifest["files"].items():
                if digest(root/name) != want:
                    raise ValueError(f"cache checksum mismatch: {root/name}")
        self.shards = [{k:np.load(p/f"{k}.npy",mmap_mode="r",allow_pickle=False) for k in
                       ("x","targets","own_positions","split","possible","legal","lookup","histograms","candidates","seed","hands","settled")}
                       for p in self.roots]
        hands = np.concatenate([s["hands"] for s in self.shards])
        if len(np.unique(hands,axis=0)) != len(hands):
            raise ValueError("physical deal overlap between sources")
        seeds=np.concatenate([s["seed"] for s in self.shards])
        if len(np.unique(seeds)) != len(seeds):
            raise ValueError("overlapping source seeds")
        self.offsets = np.r_[0,np.cumsum([len(s["x"]) for s in self.shards])]
        self.split = np.concatenate([s["split"] for s in self.shards])
        self.sha256 = hashlib.sha256(json.dumps(self.manifests,sort_keys=True).encode()).hexdigest()

    def batch(self, indices):
        indices = np.asarray(indices)
        result = {"x":np.empty((len(indices),68),np.int64),"targets":np.empty((len(indices),28),np.int64),
                  "own_positions":np.empty((len(indices),7),np.int64),"histograms":np.empty((len(indices),7,7,43),np.float32),
                  "candidates":np.empty((len(indices),7,7),np.int64)}
        for j,s in enumerate(self.shards):
            mask = (indices >= self.offsets[j]) & (indices < self.offsets[j+1])
            if not mask.any():
                continue
            ix = indices[mask]-self.offsets[j]
            for k in ("x","targets","own_positions"):
                result[k][mask] = s[k][ix]
            lookup = s["lookup"][ix]
            result["histograms"][mask] = s["histograms"][lookup]
            result["candidates"][mask] = s["candidates"][lookup]
        return result


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",required=True)
    parser.add_argument("--out",required=True)
    a=parser.parse_args()
    build(a.source,a.out)
