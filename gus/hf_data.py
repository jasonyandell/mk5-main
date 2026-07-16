"""Resolve data paths: local disk first, HuggingFace on a miss.

The data strategy ([[huggingface-assets]], [[run-artifacts-policy]] in the
wiki): git holds claims and aggregate receipts; bulk data lives on public HF
datasets. This resolver makes HF the default for every loader — a path that
exists locally is returned untouched (zero hot-path overhead); a missing one
is fetched once into the shared HF cache (~/.cache/huggingface, shared by
every worktree) and the cached path is returned.

    from gus.hf_data import resolve
    corpus = torch.load(resolve("gus/data/corpus_eval_20.pt"))
"""
from __future__ import annotations

from pathlib import Path

CORPUS_V1 = "jasonyandell/texas-42-joint-world-corpus"
CORPUS_V2 = "jasonyandell/texas-42-joint-world-corpus-v2"
EVIDENCE = "jasonyandell/mk5-run-evidence"

_ROOT = Path(__file__).resolve().parents[1]


def hf_location(rel: str) -> tuple[str, str]:
    """Map a repo-relative data path to (hf_dataset_repo, path_in_repo).

    The corpus datasets are flat (basenames at repo root); everything else
    mirrors its repo-relative path in the evidence dataset.
    """
    if rel.startswith("gus/data/"):
        name = rel[len("gus/data/"):]
        repo = CORPUS_V2 if name.startswith("corpus_v2_") else CORPUS_V1
        return repo, name
    return EVIDENCE, rel


def resolve(path: str | Path) -> Path:
    """Local path if it exists; otherwise fetch from HF and return the cached copy."""
    p = Path(path)
    local = p if p.is_absolute() else _ROOT / p
    if local.exists():
        return local
    try:
        rel = str(local.relative_to(_ROOT))
    except ValueError:
        raise FileNotFoundError(
            f"{local} does not exist and is outside the repo, so it has no HF mapping"
        ) from None
    repo, name = hf_location(rel)
    from huggingface_hub import hf_hub_download  # deferred: never imported on a local hit

    print(f"[hf_data] {rel} not on disk; fetching from hf://datasets/{repo}", flush=True)
    return Path(hf_hub_download(repo, name, repo_type="dataset"))
