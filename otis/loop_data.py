"""otis/loop_data.py — W4 self-improvement loop data machinery.

Two reusable primitives the round-N loop stands on:

  (a) :func:`parse_round_snapshot` — parse ONE arena round snapshot
      (``r{n}_<arm>_snap.json``) through the exact W2 machinery
      (:func:`otis.analysis.empirical_ledger.build_tables`, which asserts the P1
      identity, the recorded-points referee, and the ledger identity on every
      hand) into a pair of per-round parquets under the loop dir:
      ``r{n}_<arm>_hands.parquet`` / ``r{n}_<arm>_fates.parquet``. Every round row
      is training data (``split == "train"``) — round self-play carries no held-out
      val; the fixed W2 val split remains the early-stop signal.

  (b) :func:`build_cumulative_split` — an arm's CUMULATIVE training set =
      the W2 selfplay master (train split, from the existing W3 data cache) +
      ALL of that arm's round parquets so far (rounds 1..N). Returns an
      :class:`otis.data.OtisSplit` ready to hand to :func:`otis.train.train` as
      the injected ``tr``. Val is the untouched W2 selfplay val split.

Featurization is INFO-STATE ONLY and imported verbatim from
``champion.margin_net`` (via :func:`otis.data.build_split`) — this module never
re-derives the 91-dim row, and the round labels are OFFLINE labels, exactly as W2.

Pure CPU. otis does not own the GPU here.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import torch

from champion.margin_net import split_of
from otis.analysis.empirical_ledger import add_split, build_tables
from otis.data import OtisSplit, build_split

ROOT = Path(__file__).resolve().parent.parent
LOOP_DIR = ROOT / "scratch" / "otis-night" / "loop"


def _chunk_link(snap_path: Path, round_num: int, arm: str, loop_dir: Path) -> Path:
    """Materialize a ``chunk_selfplay_*.snapshots.json`` name for the round snapshot.

    ``build_tables`` / ``build_split`` derive the source tag from the filename
    (``_source_tag``: ``chunk_<tag>_<n>.snapshots.json`` → tag). The round snapshot
    is named ``r{n}_<arm>_snap.json``, so we mirror it under a canonical selfplay
    chunk name (a copy, not a symlink — robust across tools) that both the parser
    and the featurizer will accept and tag ``selfplay``.
    """
    dst = loop_dir / f"chunk_selfplay_r{round_num}{arm}.snapshots.json"
    if not dst.exists() or dst.stat().st_mtime < snap_path.stat().st_mtime:
        shutil.copyfile(snap_path, dst)
    return dst


def parse_round_snapshot(
    snap_path: str | Path,
    round_num: int,
    arm: str,
    *,
    loop_dir: Path = LOOP_DIR,
) -> dict:
    """Parse one round snapshot → per-round hands/fates parquets. Returns a summary.

    All referees (P1 identity, recorded-points cross-check, ledger identity) run
    inside :func:`build_tables` and raise on any mismatch — never widen tolerance.
    Every parsed hand is tagged ``split == "train"`` (round self-play is 100%
    training data).
    """
    snap_path = Path(snap_path)
    loop_dir = Path(loop_dir)
    loop_dir.mkdir(parents=True, exist_ok=True)

    chunk = _chunk_link(snap_path, round_num, arm, loop_dir)
    hands, fates, _junk, n_cc = build_tables([chunk])
    if len(hands) == 0:
        raise ValueError(f"{snap_path}: parsed zero hands")

    # Round self-play is entirely training data; keep a real deal-hash split col
    # for provenance parity, then force train (the loop consumes all of it).
    hands = add_split(hands, split_of)
    fates = add_split(fates, split_of)
    hands["split"] = "train"
    fates["split"] = "train"

    hands_pq = loop_dir / f"r{round_num}_{arm}_hands.parquet"
    fates_pq = loop_dir / f"r{round_num}_{arm}_fates.parquet"
    hands.to_parquet(hands_pq, index=False)
    fates.to_parquet(fates_pq, index=False)

    return {
        "round": round_num,
        "arm": arm,
        "snapshot": str(snap_path),
        "chunk": str(chunk),
        "n_hands": int(len(hands)),
        "n_fate_rows": int(len(fates)),
        "crosscheck_ok": int(n_cc),
        "hands_parquet": str(hands_pq),
        "fates_parquet": str(fates_pq),
    }


def _round_split(round_num: int, arm: str, loop_dir: Path) -> OtisSplit:
    """Materialize one round's featurized OtisSplit (all rows, split=train).

    Reuses :func:`otis.data.build_split` against a per-round parquet dir + the
    round's canonical selfplay chunk, with a dedicated cache so rounds never
    collide with each other or with the W2 master cache.
    """
    round_dir = loop_dir / f"r{round_num}_{arm}_pq"
    round_dir.mkdir(parents=True, exist_ok=True)
    # build_split reads master_hands/master_fates from parquet_dir.
    for name in ("hands", "fates"):
        src = loop_dir / f"r{round_num}_{arm}_{name}.parquet"
        dst = round_dir / f"master_{name}.parquet"
        if not dst.exists() or dst.stat().st_mtime < src.stat().st_mtime:
            shutil.copyfile(src, dst)
    corpus_glob = str(loop_dir / f"chunk_selfplay_r{round_num}{arm}.snapshots.json")
    cache_dir = loop_dir / f"r{round_num}_{arm}_cache"
    return build_split(
        "train", "selfplay",
        parquet_dir=round_dir,
        corpus_glob=corpus_glob,
        cache_dir=cache_dir,
        verbose=True,
    )


def _concat_splits(parts: list[OtisSplit]) -> OtisSplit:
    keys: list[tuple] = []
    for p in parts:
        keys.extend(p.keys)
    return OtisSplit(
        X=torch.cat([p.X for p in parts], dim=0),
        y_price=torch.cat([p.y_price for p in parts], dim=0),
        y_trick=torch.cat([p.y_trick for p in parts], dim=0),
        y_fate=torch.cat([p.y_fate for p in parts], dim=0),
        groups=torch.cat([p.groups for p in parts], dim=0),
        keys=keys,
    )


def build_cumulative_split(
    arm: str,
    round_num: int,
    *,
    loop_dir: Path = LOOP_DIR,
) -> tuple[OtisSplit, OtisSplit]:
    """Cumulative (train, val) for ``arm`` at ``round_num``.

    train = W2 selfplay master (train split, from the W3 data cache) + ALL of
    this arm's round parquets 1..round_num. val = the untouched W2 selfplay val.
    """
    loop_dir = Path(loop_dir)
    w2_train = build_split("train", "selfplay")   # W2 master train (cached)
    w2_val = build_split("val", "selfplay")       # fixed early-stop signal
    parts = [w2_train]
    for r in range(1, round_num + 1):
        parts.append(_round_split(r, arm, loop_dir))
    cumulative = _concat_splits(parts)
    print(
        f"[loop_data] arm={arm} round={round_num} cumulative train N={len(cumulative)} "
        f"(W2 {len(w2_train)} + rounds {len(cumulative) - len(w2_train)}), "
        f"val N={len(w2_val)}",
        flush=True,
    )
    return cumulative, w2_val
