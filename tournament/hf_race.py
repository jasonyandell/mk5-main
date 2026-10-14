# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0

"""The no-conflict property on the real backend: writers racing commits to one HF dataset.

    python -u -m tournament.hf_race --repo jasonyandell/texas-42-tournament-selftest

W processes start together (a barrier) and each appends N files through
HfStore, one commit each. Afterwards the repo must hold every file, one
commit per file, and each re-append must be a no-op. The offline twin with
plain git clones is tests/test_store.py; this one checks the Hub itself.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import secrets
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tournament.store import AppendConflict, HfStore, dumps  # noqa: E402


def _writer(repo, run, name, n, barrier, out):
    store = HfStore(repo, cache=ROOT / "scratch" / "tournament" / "hf-race" / name)
    barrier.wait()
    paths, err = [], None
    try:
        for k in range(n):
            path = f"race/{run}/{name}-{k:02d}-{secrets.token_hex(8)}.json"
            store.append(path, dumps({"run": run, "writer": name, "k": k}), f"race {run} {name} {k}")
            paths.append(path)
        again = store.append(paths[0], dumps({"run": run, "writer": name, "k": 0}), "again")
        try:
            store.append(paths[0], dumps({"edited": True}), "edit")
            err = "an edit was accepted"
        except AppendConflict:
            pass
        if again:
            err = "a re-append was not a no-op"
    except Exception as e:
        err = repr(e)
    out.put((name, paths, store.retried, err))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--writers", type=int, default=6)
    ap.add_argument("--each", type=int, default=4)
    args = ap.parse_args()
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(args.repo, repo_type="dataset", exist_ok=True)
    before = {c.commit_id for c in api.list_repo_commits(args.repo, repo_type="dataset")}
    run = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    ctx = mp.get_context("spawn")
    barrier, out = ctx.Barrier(args.writers), ctx.Queue()
    procs = [ctx.Process(target=_writer, args=(args.repo, run, f"w{i}", args.each, barrier, out))
             for i in range(args.writers)]
    t0 = time.time()
    for p in procs:
        p.start()
    results = [out.get(timeout=600) for _ in procs]
    for p in procs:
        p.join()
    wall = time.time() - t0
    errors = [(n, e) for n, _, _, e in results if e]
    written = sorted(p for _, ps, _, _ in results for p in ps)
    files = {f for f in api.list_repo_files(args.repo, repo_type="dataset") if f.startswith(f"race/{run}/")}
    new = [c for c in api.list_repo_commits(args.repo, repo_type="dataset") if c.commit_id not in before]
    print(f"run {run}: {args.writers} writers x {args.each} appends in {wall:.1f}s")
    print(f"  files written {len(written)}, files on the Hub {len(files)}, missing {len(set(written) - files)}")
    print(f"  new commits {len(new)} (expected {len(written)}), retried calls {sum(r for _, _, r, _ in results)}")
    print(f"  errors {errors or 'none'}")
    ok = not errors and set(written) == files and len(new) == len(written) == args.writers * args.each
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
