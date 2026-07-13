"""Build MANIFEST.json for gus/data corpus before HF upload.

Records sha256 + size + mtime + log-tail for every .pt and .log in gus/data/.
Also opens each .pt with torch.load to confirm it's a well-formed corpus blob.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import torch

DATA_DIR = Path(__file__).resolve().parents[2] / "gus" / "data"
OUT = DATA_DIR / "MANIFEST.json"


def sha256_file(path: Path, buf: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(buf):
            h.update(chunk)
    return h.hexdigest()


def classify(name: str) -> str:
    if name.startswith("corpus_v2_train_"):
        return "v2_train"
    if name.startswith("corpus_v2_eval"):
        return "v2_eval"
    if name.startswith("corpus_train_chunk_"):
        return "v1_train"
    if name.startswith("corpus_eval_"):
        return "v1_eval"
    if name == "corpus_train_100.pt":
        return "v1_train_pilot"
    if name.startswith("_len_cache_"):
        return "sidecar"
    return "other"


def inspect_pt(path: Path) -> dict:
    blob = torch.load(str(path), weights_only=False, map_location="cpu")
    info: dict = {"keys": sorted(list(blob.keys()))}
    if "results" in blob:
        info["n_games"] = len(blob["results"])
        if blob["results"]:
            g0 = blob["results"][0]
            info["n_decisions_game0"] = len(getattr(g0, "decisions", []))
            decs = getattr(g0, "decisions", [])
            jw_count = sum(
                1 for d in decs if getattr(d, "world_hands", None) is not None
            )
            info["jw_decisions_game0"] = jw_count
    if "seeds" in blob:
        seeds = blob["seeds"]
        info["n_seeds"] = len(seeds)
        if seeds:
            info["seed_min"] = int(min(seeds))
            info["seed_max"] = int(max(seeds))
    return info


def main() -> int:
    files = sorted(DATA_DIR.iterdir())
    pt_files = [p for p in files if p.suffix == ".pt"]
    log_files = {p.stem: p for p in files if p.suffix == ".log"}

    print(f"manifest target: {OUT}")
    print(f"found {len(pt_files)} .pt files; {len(log_files)} .log files", flush=True)

    entries = []
    total_bytes = 0
    t0 = time.time()
    for i, p in enumerate(pt_files, 1):
        size = p.stat().st_size
        total_bytes += size
        kind = classify(p.name)
        print(f"[{i}/{len(pt_files)}] {p.name} ({size/1e9:.2f} GB) [{kind}]", flush=True)

        entry: dict = {
            "name": p.name,
            "kind": kind,
            "bytes": size,
            "mtime": p.stat().st_mtime,
        }
        # Skip torch.load + sha256 if user passes --skip-verify (cheap manifest).
        if "--skip-verify" not in sys.argv:
            try:
                entry["torch_load"] = inspect_pt(p)
            except Exception as exc:
                entry["torch_load_error"] = repr(exc)
            entry["sha256"] = sha256_file(p)

        # Attach paired log if present.
        log = log_files.get(p.stem)
        if log:
            entry["log_name"] = log.name
            entry["log_text"] = log.read_text()
            entry["log_sha256"] = sha256_file(log)

        entries.append(entry)

    # Standalone .log files with no matching .pt.
    pt_stems = {p.stem for p in pt_files}
    orphan_logs = [log for stem, log in log_files.items() if stem not in pt_stems]
    for log in orphan_logs:
        entries.append({
            "name": log.name,
            "kind": "orphan_log",
            "bytes": log.stat().st_size,
            "mtime": log.stat().st_mtime,
            "sha256": sha256_file(log),
            "log_text": log.read_text(),
        })

    elapsed = time.time() - t0
    manifest = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "data_dir": str(DATA_DIR),
        "n_files": len(entries),
        "total_bytes": total_bytes,
        "elapsed_seconds": round(elapsed, 1),
        "files": entries,
    }
    OUT.write_text(json.dumps(manifest, indent=2))
    print(
        f"\nwrote {OUT} ({OUT.stat().st_size/1e6:.2f} MB)\n"
        f"total payload: {total_bytes/1e9:.2f} GB across {len(entries)} files\n"
        f"elapsed: {elapsed:.1f}s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
