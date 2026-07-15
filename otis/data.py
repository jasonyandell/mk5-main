"""otis/data.py — margin-style feature rows joined to offline fate labels.

Conditioning is INFO-STATE ONLY: the 91-dim (declarer-hand ⊕ canonical-auction)
row from ``champion.margin_net`` (imported verbatim). Targets are the OFFLINE
labels from the W2 master parquets:

  * pricing  : ``bidder_team_pts`` (0..42)          — control + treatment
  * trick    : ``bidder_tricks``   (0..7)           — treatment
  * fate     : per-tile 8-class    (capture_bidding × played_mode) — treatment

The declarer HAND lives only in the on-policy snapshot chunks
(``scratch/otis-night/corpus/chunk_<source>_*.snapshots.json``); the parquets hold
the labels + the deal-hash split. Rows are joined on the W2 key
``(source, seed, game_idx, hand_idx, a_team)`` (w2_corpus.md §7). The snapshot's
own recorded ``bidder_team_pts`` is cross-checked against the parquet on the way in.

Row convention follows ``MarginDataset``: one declarer-POV row per hand, y taken
from the bidding team's realized points, so the fate "capture" axis reading
``bidding_team`` == "my team captures". Unlike ``MarginDataset`` no cross-half
dedup is applied — the two paired ``a_team`` halves are distinct outcome rows
(they share an info-state but realize different fates), and the parquet keys them
separately; both arms consume the identical row set, so the arm comparison stays
fair. Base order is a deterministic sort on the join key; the training seed only
governs DataLoader shuffling.
"""
from __future__ import annotations

import glob
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from champion.margin_net import FEATURE_DIM, featurize_snapshot
from otis.analysis.empirical_ledger import _source_tag
from otis.model import CAPTURE_IDX, MODE_IDX, TILE_PIPS

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PARQUET_DIR = ROOT / "scratch" / "otis-night"
DEFAULT_CORPUS_GLOB = str(ROOT / "scratch" / "otis-night" / "corpus" / "chunk_*.snapshots.json")
DEFAULT_CACHE_DIR = ROOT / "scratch" / "otis-night" / "data_cache"

JOIN_KEYS = ("source", "seed", "game_idx", "hand_idx", "a_team")


# --------------------------------------------------------------------------- #
# Label table (parquet join)                                                    #
# --------------------------------------------------------------------------- #


def _build_label_table(parquet_dir: Path) -> pd.DataFrame:
    """One row per hand keyed by ``JOIN_KEYS`` with targets + split.

    Columns: JOIN_KEYS, split, y_price (bidder_team_pts), y_trick (bidder_tricks),
    and y_fate_0..4 (8-class labels ordered by ``TILE_PIPS``).
    """
    hands = pd.read_parquet(parquet_dir / "master_hands.parquet")
    fates = pd.read_parquet(parquet_dir / "master_fates.parquet")

    # 8-class fate label per fate row, then pivot tile → column.
    fates = fates.copy()
    fates["cls"] = (
        fates["capture_bidding"].map(CAPTURE_IDX) * 4 + fates["played_mode"].map(MODE_IDX)
    ).astype(np.int64)
    fate_wide = (
        fates.pivot_table(index=list(JOIN_KEYS), columns="tile", values="cls", aggfunc="first")
        .reindex(columns=list(TILE_PIPS))
    )
    fate_wide.columns = [f"y_fate_{i}" for i in range(len(TILE_PIPS))]
    fate_wide = fate_wide.reset_index()

    tab = hands[list(JOIN_KEYS) + ["split", "bidder_team_pts", "bidder_tricks"]].merge(
        fate_wide, on=list(JOIN_KEYS), how="inner", validate="one_to_one"
    )
    tab = tab.rename(columns={"bidder_team_pts": "y_price", "bidder_tricks": "y_trick"})
    # No hand should be missing a tile fate.
    fate_cols = [f"y_fate_{i}" for i in range(len(TILE_PIPS))]
    if tab[fate_cols].isna().any().any():
        raise ValueError("fate label table has missing tiles after pivot/join")
    for c in fate_cols:
        tab[c] = tab[c].astype(np.int64)
    return tab


# --------------------------------------------------------------------------- #
# Dataset                                                                       #
# --------------------------------------------------------------------------- #


@dataclass
class OtisSplit:
    """Materialized tensors for one (split, source) selection.

    ``X`` [N,91], ``y_price`` [N], ``y_trick`` [N], ``y_fate`` [N,5] (all long
    except X float32), ``groups`` [N,2] int64 = (seed, hand_idx) deal-cluster key.
    """

    X: Tensor
    y_price: Tensor
    y_trick: Tensor
    y_fate: Tensor
    groups: Tensor
    keys: list[tuple]

    def __len__(self) -> int:
        return int(self.X.shape[0])


class OtisDataset(torch.utils.data.Dataset):
    """(x, y_price, y_trick, y_fate) samples for one split. Both arms use the
    identical tensor set; control simply ignores y_trick/y_fate downstream."""

    def __init__(self, data: OtisSplit) -> None:
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, i: int):
        d = self.data
        return d.X[i], d.y_price[i], d.y_trick[i], d.y_fate[i]


def _cache_path(cache_dir: Path, split: str, source: str) -> Path:
    return cache_dir / f"otis_{source}_{split}.pt"


def build_split(
    split: str,
    source: str = "selfplay",
    *,
    parquet_dir: Path = DEFAULT_PARQUET_DIR,
    corpus_glob: str = DEFAULT_CORPUS_GLOB,
    cache_dir: Path | None = DEFAULT_CACHE_DIR,
    verbose: bool = True,
) -> OtisSplit:
    """Materialize one split's tensors, joining snapshot hands to parquet labels.

    ``source`` ∈ {"selfplay","netwp","random","all"}; ``split`` ∈ {"train","val",
    "test","all"}. Deterministic row order (sorted by join key). Cached to
    ``cache_dir`` keyed by (source, split); delete the cache to rebuild.
    """
    if cache_dir is not None:
        cp = _cache_path(cache_dir, split, source)
        if cp.exists():
            blob = torch.load(cp, weights_only=False)
            if verbose:
                print(f"[data] loaded cache {cp} (N={blob['X'].shape[0]})", flush=True)
            return OtisSplit(
                X=blob["X"], y_price=blob["y_price"], y_trick=blob["y_trick"],
                y_fate=blob["y_fate"], groups=blob["groups"], keys=blob["keys"],
            )

    tab = _build_label_table(parquet_dir)
    if source != "all":
        tab = tab[tab["source"] == source]
    if split != "all":
        tab = tab[tab["split"] == split]
    # key -> target tuple
    label: dict[tuple, tuple] = {}
    for r in tab.itertuples(index=False):
        key = (r.source, int(r.seed), int(r.game_idx), int(r.hand_idx), int(r.a_team))
        label[key] = (
            int(r.y_price), int(r.y_trick),
            (int(r.y_fate_0), int(r.y_fate_1), int(r.y_fate_2), int(r.y_fate_3), int(r.y_fate_4)),
        )
    want_keys = set(label)
    if verbose:
        print(f"[data] {split}/{source}: {len(want_keys)} target hands", flush=True)

    chunks = sorted(glob.glob(corpus_glob))
    rows: dict[tuple, dict] = {}
    n_seen = 0
    for path in chunks:
        src = _source_tag(Path(path))
        if source != "all" and src != source:
            continue
        payload = json.loads(Path(path).read_text())
        snaps = payload["snapshots"] if isinstance(payload, dict) else payload
        for snap in snaps:
            key = (src, int(snap["seed"]), int(snap["game_idx"]),
                   int(snap["hand_idx"]), int(snap["a_team"]))
            if key not in want_keys or key in rows:
                continue
            price, trick, fate5 = label[key]
            if int(snap["bidder_team_pts"]) != price:
                raise ValueError(
                    f"{key}: snapshot bidder_team_pts {snap['bidder_team_pts']} "
                    f"!= parquet y_price {price}"
                )
            rows[key] = {
                "x": featurize_snapshot(snap),
                "price": price, "trick": trick, "fate": fate5,
            }
            n_seen += 1
            if verbose and n_seen % 20000 == 0:
                print(f"[data]   featurized {n_seen}/{len(want_keys)}", flush=True)

    missing = want_keys - set(rows)
    if missing:
        raise ValueError(f"{len(missing)} target hands had no snapshot (e.g. {next(iter(missing))})")

    ordered = sorted(rows)  # deterministic base order by join key
    X = torch.stack([rows[k]["x"] for k in ordered]).to(torch.float32)
    y_price = torch.tensor([rows[k]["price"] for k in ordered], dtype=torch.long)
    y_trick = torch.tensor([rows[k]["trick"] for k in ordered], dtype=torch.long)
    y_fate = torch.tensor([rows[k]["fate"] for k in ordered], dtype=torch.long)
    groups = torch.tensor([(k[1], k[3]) for k in ordered], dtype=torch.long)  # (seed, hand_idx)
    out = OtisSplit(X=X, y_price=y_price, y_trick=y_trick, y_fate=y_fate,
                    groups=groups, keys=ordered)

    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cp = _cache_path(cache_dir, split, source)
        torch.save(
            {"X": X, "y_price": y_price, "y_trick": y_trick, "y_fate": y_fate,
             "groups": groups, "keys": ordered}, cp,
        )
        if verbose:
            print(f"[data] cached → {cp} (N={len(ordered)})", flush=True)
    return out
