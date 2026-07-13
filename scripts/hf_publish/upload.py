"""Upload gus/data/ to HuggingFace as a public dataset.

Three phases:
  --phase smoke   : eval corpora + v2 train (~1 GB) — proves auth + repo layout
  --phase v1      : 108 GB v1 train chunks — bulk push (resumable)
  --phase finalize: README + MANIFEST + len-cache + v0 pilot
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from huggingface_hub import HfApi, create_repo, upload_large_folder

REPO_ID_DEFAULT = "jasonyandell/texas-42-joint-world-corpus"
DATA_DIR = Path(__file__).resolve().parents[2] / "gus" / "data"
DOC_DIR = Path(__file__).resolve().parents[2] / "gus" / "data"


PATTERNS = {
    "smoke": [
        "corpus_eval_20.pt",
        "corpus_eval_20.log",
        "corpus_v2_eval.pt",
        "corpus_v2_train_*.pt",
        "corpus_v2_train_*.log",
    ],
    "v1": [
        "corpus_train_chunk_*.pt",
        "corpus_train_chunk_*.log",
    ],
    "finalize": [
        "README.md",
        "MANIFEST.json",
        "_len_cache_train_10k.pt",
        "corpus_train_100.pt",
        "corpus_train_100.log",
    ],
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=list(PATTERNS))
    parser.add_argument("--repo-id", default=REPO_ID_DEFAULT)
    parser.add_argument("--private", action="store_true",
                        help="Create as private (default: public)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    api = HfApi()
    print(f"target: hf://datasets/{args.repo_id}", flush=True)
    print(f"phase:  {args.phase}", flush=True)

    if not args.dry_run:
        create_repo(
            args.repo_id,
            repo_type="dataset",
            private=args.private,
            exist_ok=True,
        )

    patterns = PATTERNS[args.phase]
    matches = []
    for pat in patterns:
        matches.extend(sorted(DATA_DIR.glob(pat)))
    matches = sorted(set(matches))

    total_bytes = sum(p.stat().st_size for p in matches if p.exists())
    print(f"matched {len(matches)} files, {total_bytes/1e9:.2f} GB", flush=True)
    for p in matches[:8]:
        print(f"  {p.name} ({p.stat().st_size/1e6:.1f} MB)", flush=True)
    if len(matches) > 8:
        print(f"  ... +{len(matches)-8} more", flush=True)

    if args.dry_run:
        print("(dry-run, no upload)", flush=True)
        return 0

    # upload_large_folder is resumable + chunked + uses LFS automatically.
    # We pass --allow-patterns matching only this phase so it ignores the rest.
    upload_large_folder(
        folder_path=str(DATA_DIR),
        repo_id=args.repo_id,
        repo_type="dataset",
        allow_patterns=patterns,
        print_report=True,
    )
    print("upload complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
