"""Chunked, resumable regeneration of the joint-world corpus (otis Phase R).

Regenerates the April eq corpus on the repaired sampler with write-time
validity assertions (issues #51/#52/#55) and pushes each chunk to the
Hugging Face dataset repo `jasonyandell/texas-42-joint-world-corpus-v2` as it
lands. Designed to run unattended on a Vast.ai 4090 (or any CUDA host).

Per chunk: skip-if-on-HF -> generate -> independent referee scan
(scripts/referee_worlds.py; abort on DIRTY) -> sha256 -> upload .pt + .log ->
update MANIFEST.json on the repo -> delete the local .pt (keeps disk small).
A heartbeat thread logs every 45 s so silence is a bug, never a mystery.

Plan (mirrors the original repo's layout, decl-8 purged):
- corpus_eval_20.pt         seeds 900000-900019, adaptive, schema v1
- corpus_v2_eval.pt         seeds 900000-900019, adaptive, schema v2
- corpus_v2_train_{s}-{s+9}_9d.pt x10   seeds 0-99, 9 decls/seed, 200 samples,
                                        schema v2, world weights recorded
- corpus_train_chunk_{s}-{s+99}.pt x100 seeds 0-9999, adaptive, schema v1

Usage (on the GPU box, from the repo root):
    HF_TOKEN=... python -u -m forge.eq.regen_corpus_v2 \
        --subset all --device cuda 2>&1 | tee regen.log

`--subset evals|v2|v1` and `--v1-range START END` split work across
instances; every instance is independently resumable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ID = "jasonyandell/texas-42-joint-world-corpus-v2"
ADAPTIVE = [
    "--adaptive",
    "--min-samples", "100",
    "--max-samples", "50000",
    "--batch-size", "200",
    "--sem-threshold", "0.5",
]


def build_plan(subset: str, v1_range: tuple[int, int]) -> list[dict]:
    plan: list[dict] = []
    if subset in ("all", "evals"):
        plan.append(
            {
                "name": "corpus_eval_20.pt",
                "args": ["--start-seed", "900000", "--n-games", "20", *ADAPTIVE,
                         "--save-joint-worlds"],
            }
        )
        plan.append(
            {
                "name": "corpus_v2_eval.pt",
                "args": ["--start-seed", "900000", "--n-games", "20", *ADAPTIVE,
                         "--save-joint-worlds", "--schema", "v2"],
            }
        )
    if subset in ("all", "v2"):
        for s in range(0, 100, 10):
            plan.append(
                {
                    "name": f"corpus_v2_train_{s}-{s + 9}_9d.pt",
                    "args": ["--start-seed", str(s), "--n-games", "90",
                             "--n-decl-per-seed", "9", "--samples", "200",
                             "--schema", "v2", "--save-joint-worlds",
                             "--record-world-weights"],
                }
            )
    if subset in ("all", "v1"):
        lo, hi = v1_range
        for s in range(lo, hi, 100):
            plan.append(
                {
                    "name": f"corpus_train_chunk_{s}-{s + 99}.pt",
                    "args": ["--start-seed", str(s), "--n-games", "100",
                             *ADAPTIVE, "--save-joint-worlds"],
                }
            )
    return plan


class Heartbeat:
    def __init__(self, interval: float = 45.0):
        self.interval = interval
        self.status = "starting"
        self.t0 = time.time()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.wait(self.interval):
            print(
                f"[heartbeat] alive t+{time.time() - self.t0:.0f}s status={self.status}",
                flush=True,
            )

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", choices=["all", "evals", "v2", "v1"], default="all")
    ap.add_argument("--v1-range", nargs=2, type=int, default=[0, 10000],
                    metavar=("START", "END"))
    ap.add_argument("--out-dir", default="gus/data_regen")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--repo", default=REPO_ID)
    ap.add_argument("--keep-local", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN")
    if not token and not args.dry_run:
        print("ERROR: HF_TOKEN not set", flush=True)
        return 1
    api = HfApi(token=token) if not args.dry_run else None

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plan = build_plan(args.subset, tuple(args.v1_range))
    print(f"Plan: {len(plan)} files, subset={args.subset}, device={args.device}", flush=True)
    if args.dry_run:
        for item in plan:
            print(f"  {item['name']}: {' '.join(item['args'])}", flush=True)
        return 0

    existing = set(api.list_repo_files(args.repo, repo_type="dataset"))
    py = sys.executable

    with Heartbeat() as hb:
        for i, item in enumerate(plan):
            name = item["name"]
            if name in existing:
                print(f"[{i + 1}/{len(plan)}] SKIP {name} (already on HF)", flush=True)
                continue
            pt = out_dir / name
            log = out_dir / (name.removesuffix(".pt") + ".log")
            ref_json = out_dir / (name.removesuffix(".pt") + ".referee.json")

            # 1. Generate
            hb.status = f"generate {name}"
            cmd = [py, "-u", "-m", "forge.eq.generate", *item["args"],
                   "--device", args.device, "-o", str(pt)]
            print(f"[{i + 1}/{len(plan)}] GEN {name}: {' '.join(cmd)}", flush=True)
            t0 = time.time()
            with open(log, "w") as lf:
                proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT)
            if proc.returncode != 0:
                print(f"FATAL: generation failed for {name} "
                      f"(rc={proc.returncode}); see {log}", flush=True)
                print(Path(log).read_text()[-2000:], flush=True)
                return 2
            gen_s = time.time() - t0

            # 2. Independent referee (abort on DIRTY — stop the line)
            hb.status = f"referee {name}"
            proc = subprocess.run(
                [py, "-u", "scripts/referee_worlds.py", str(pt), "--json", str(ref_json)],
            )
            if proc.returncode != 0:
                print(f"FATAL: referee graded {name} DIRTY — stopping the line "
                      f"(issue #52 must not recur). Local file kept for autopsy: {pt}",
                      flush=True)
                return 3
            report = json.loads(ref_json.read_text())[0]

            # 3. Upload chunk + log + manifest in ONE commit (HF has a
            # 128-commits/hour ceiling — the zeb fleet lesson).
            hb.status = f"upload {name}"
            manifest_entry = {
                "sha256": report["sha256"],
                "bytes": report["bytes"],
                "games": report["games"],
                "decisions_with_worlds": report["decisions_with_worlds"],
                "worlds": report["worlds"],
                "invalid_worlds": report["invalid_worlds"],
                "weights_decisions": report["weights_decisions"],
                "recipe": " ".join(item["args"]),
                "device": args.device,
                "generation_seconds": round(gen_s, 1),
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }
            _upload_chunk(api, args.repo, pt, log, name, manifest_entry)
            existing.add(name)

            print(f"[{i + 1}/{len(plan)}] DONE {name}: {report['games']} games "
                  f"in {gen_s:.0f}s, {report['worlds']} worlds, 0 invalid, "
                  f"sha={report['sha256'][:16]}, uploaded", flush=True)

            if not args.keep_local:
                pt.unlink()

    print("ALL_CHUNKS_COMPLETE", flush=True)
    return 0


def _upload_chunk(api, repo: str, pt: Path, log: Path, name: str, entry: dict) -> None:
    from huggingface_hub import CommitOperationAdd, hf_hub_download

    manifest: dict = {}
    try:
        p = hf_hub_download(repo, "MANIFEST.json", repo_type="dataset",
                            force_download=True)
        manifest = json.loads(Path(p).read_text())
    except Exception:
        pass
    manifest[name] = entry
    tmp = Path("/tmp/regen_manifest.json")
    tmp.write_text(json.dumps(manifest, indent=2))

    for attempt in range(4):
        try:
            api.create_commit(
                repo_id=repo,
                repo_type="dataset",
                operations=[
                    CommitOperationAdd(path_in_repo=name, path_or_fileobj=str(pt)),
                    CommitOperationAdd(path_in_repo=log.name, path_or_fileobj=str(log)),
                    CommitOperationAdd(path_in_repo="MANIFEST.json", path_or_fileobj=str(tmp)),
                ],
                commit_message=f"regen: {name} (clean; otis phase R)",
            )
            return
        except Exception as exc:
            wait = 30 * (attempt + 1)
            print(f"[upload] attempt {attempt + 1} failed for {name}: {exc}; "
                  f"retrying in {wait}s", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"upload failed after retries: {name}")


if __name__ == "__main__":
    sys.exit(main())
