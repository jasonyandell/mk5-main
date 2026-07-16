"""hf_data resolver: mapping is pure, local hits never touch the network."""
from pathlib import Path

import pytest

from gus.hf_data import CORPUS_V1, CORPUS_V2, EVIDENCE, _ROOT, hf_location, resolve


def test_corpus_v1_maps_to_flat_basename():
    assert hf_location("gus/data/corpus_eval_20.pt") == (CORPUS_V1, "corpus_eval_20.pt")
    assert hf_location("gus/data/corpus_train_chunk_0-99.pt") == (
        CORPUS_V1,
        "corpus_train_chunk_0-99.pt",
    )


def test_corpus_v2_prefix_routes_to_v2_repo():
    assert hf_location("gus/data/corpus_v2_eval.pt") == (CORPUS_V2, "corpus_v2_eval.pt")


def test_everything_else_mirrors_into_evidence():
    assert hf_location("otis/models/otis_play_v0.pt") == (
        EVIDENCE,
        "otis/models/otis_play_v0.pt",
    )
    assert hf_location("arena/results/per_game.csv") == (
        EVIDENCE,
        "arena/results/per_game.csv",
    )


def test_resolve_returns_existing_absolute_path_untouched(tmp_path):
    f = tmp_path / "x.pt"
    f.write_bytes(b"1")
    assert resolve(f) == f


def test_resolve_returns_existing_relative_path_under_root():
    assert resolve("gus/hf_data.py") == _ROOT / "gus/hf_data.py"


def test_missing_path_outside_repo_raises_without_network(tmp_path):
    with pytest.raises(FileNotFoundError):
        resolve(tmp_path / "nope.pt")
