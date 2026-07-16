"""Mirror run evidence between the repo tree and the public HF dataset.

Run-artifacts policy (wiki/decisions/run-artifacts-policy.md): git holds claims
and aggregate receipts; row-level data and trained heads live on HuggingFace at
the same relative paths. One HF commit per upload; revision-pinned URLs are
printed for wiki citation.

  python -u scripts/hf_publish/evidence.py up PATH [PATH...] [--tag TAG]
  python -u scripts/hf_publish/evidence.py get PATH [PATH...] [--revision REV]

PATHs are files or directories, repo-relative (or absolute inside the repo).
`up` mirrors them to hf://datasets/jasonyandell/mk5-run-evidence; `get`
restores them into the working tree. Downloads go through the shared HF cache
(~/.cache/huggingface), so repeated gets across worktrees are free.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

REPO_ID = "jasonyandell/mk5-run-evidence"
ROOT = Path(__file__).resolve().parents[2]


def _rel(path_str: str) -> Path:
    """Resolve a CLI path to a repo-relative Path, or die."""
    p = Path(path_str)
    p = p.resolve() if p.is_absolute() else (ROOT / p).resolve()
    try:
        return p.relative_to(ROOT)
    except ValueError:
        sys.exit(f"error: {path_str} is not inside the repo ({ROOT})")


def _expand(rel: Path) -> list[Path]:
    """A file stays a file; a directory becomes every file under it."""
    abs_path = ROOT / rel
    if abs_path.is_file():
        return [rel]
    if abs_path.is_dir():
        return sorted(
            f.relative_to(ROOT) for f in abs_path.rglob("*") if f.is_file()
        )
    sys.exit(f"error: {rel} does not exist")


def up(paths: list[str], tag: str | None) -> None:
    api = HfApi()
    api.create_repo(REPO_ID, repo_type="dataset", private=False, exist_ok=True)
    rels = [f for p in paths for f in _expand(_rel(p))]
    ops = [
        CommitOperationAdd(path_in_repo=str(r), path_or_fileobj=str(ROOT / r))
        for r in rels
    ]
    info = api.create_commit(
        repo_id=REPO_ID,
        repo_type="dataset",
        operations=ops,
        commit_message=f"up: {', '.join(paths)}",
    )
    sha = info.oid
    print(f"uploaded {len(rels)} file(s) at revision {sha}", flush=True)
    if tag:
        api.create_tag(REPO_ID, repo_type="dataset", tag=tag, revision=sha)
        sha = tag
        print(f"tagged: {tag}", flush=True)
    for r in rels:
        print(f"  https://huggingface.co/datasets/{REPO_ID}/blob/{sha}/{r}", flush=True)


def get(paths: list[str], revision: str) -> None:
    api = HfApi()
    listing = api.list_repo_files(REPO_ID, repo_type="dataset", revision=revision)
    for p in paths:
        rel = str(_rel(p))
        matches = [f for f in listing if f == rel or f.startswith(rel + "/")]
        if not matches:
            sys.exit(f"error: {rel} not found in {REPO_ID}@{revision}")
        for f in matches:
            cached = hf_hub_download(
                REPO_ID, f, repo_type="dataset", revision=revision
            )
            dest = ROOT / f
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(cached, dest)
            print(f"  {f}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="verb", required=True)
    p_up = sub.add_parser("up", help="mirror paths to the HF dataset")
    p_up.add_argument("paths", nargs="+")
    p_up.add_argument("--tag", help="tag the resulting revision (e.g. otis-night2-2026-07-15)")
    p_get = sub.add_parser("get", help="restore paths into the working tree")
    p_get.add_argument("paths", nargs="+")
    p_get.add_argument("--revision", default="main")
    args = parser.parse_args()
    if args.verb == "up":
        up(args.paths, args.tag)
    else:
        get(args.paths, args.revision)
    return 0


if __name__ == "__main__":
    sys.exit(main())
