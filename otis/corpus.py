"""Adapter: eq-corpus ``GameRecordGPU`` payloads -> :class:`otis.fates.NeutralGame`.

The eq corpus (``gus/data/corpus_train_chunk_*.pt``) is a torch-saved dict:
``{"results": list[GameRecordGPU], ...}`` (schema in ``forge/eq/generate/types.py:54``,
payload build at ``forge/eq/generate/cli.py:551``). Each :class:`GameRecordGPU`:

  * ``hands``       — the true initial 4x7 deal (list[list[int]] of domino ids).
  * ``decisions``   — one :class:`DecisionRecordGPU` per play, IN PLAY ORDER. Each has
                      ``player`` (seat 0-3, = current player at that tick, recorded at
                      ``forge/eq/generate/actions.py:412``) and ``action_taken`` (slot
                      index 0-6 into that seat's 7-slot hand). Slot positions are stable
                      across the game — played slots become -1 but do not shift — so the
                      played domino id is ``hands[player][action_taken]``.
  * ``decl_id``     — forge declaration id (0..9).
  * ``bid_value``   — schema-v2 bid amount, else ``None`` (v1 corpus is bid-30 fixed).
  * ``bidder``      — winning seat when a real auction produced the deal, else ``None``.
                      When absent, seat 0 is the declarer/leader (from_deals default,
                      ``forge/eq/game_tensor.py:262``), and ``decisions[0].player`` is
                      exactly that leader — we use it as the bidder.

This module needs torch (to read ``.pt``); the parser (``otis.fates``) does not.
Everything here is CPU (``map_location="cpu"``); otis does not own the GPU.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import torch

from otis.fates import NeutralGame

DEFAULT_BID_VALUE = 30  # v1 eq corpus is bid-30 fixed


def _record_to_neutral(rec, game_id: str) -> NeutralGame:
    """Convert one ``GameRecordGPU`` into a :class:`NeutralGame`."""
    hands: list[list[int]] = [[int(d) for d in hand] for hand in rec.hands]

    decisions = rec.decisions
    if len(decisions) != 28:
        raise ValueError(
            f"{game_id}: expected 28 decisions, got {len(decisions)} "
            f"(only fully-played games are parseable)"
        )

    plays: list[tuple[int, int]] = []
    for d in decisions:
        seat = int(d.player)
        slot = int(d.action_taken)
        if not (0 <= seat < 4):
            raise ValueError(f"{game_id}: decision seat {seat} out of range")
        if not (0 <= slot < 7):
            raise ValueError(f"{game_id}: decision slot {slot} out of range")
        did = hands[seat][slot]
        if did < 0:
            raise ValueError(
                f"{game_id}: seat {seat} slot {slot} already played (id {did})"
            )
        plays.append((seat, did))

    bidder = int(rec.bidder) if getattr(rec, "bidder", None) is not None else int(decisions[0].player)
    bid_value = int(rec.bid_value) if getattr(rec, "bid_value", None) is not None else DEFAULT_BID_VALUE

    return NeutralGame(
        game_id=game_id,
        hands=hands,
        decl_id=int(rec.decl_id),
        bidder=bidder,
        bid_value=bid_value,
        plays=plays,
    )


def load_corpus_games(
    chunk_path: str | Path,
    limit: int | None = None,
    game_id_prefix: str | None = None,
) -> Iterator[NeutralGame]:
    """Yield :class:`NeutralGame` records from one eq-corpus chunk file.

    Args:
        chunk_path: Path to a ``corpus_*_.pt`` file.
        limit: Max games to yield (``None`` = all in the chunk).
        game_id_prefix: Prefix for synthesised game ids (defaults to the file stem).

    Yields:
        NeutralGame, one per game, in file order.
    """
    chunk_path = Path(chunk_path)
    if game_id_prefix is None:
        game_id_prefix = chunk_path.stem

    payload = torch.load(chunk_path, map_location="cpu", weights_only=False)
    results = payload["results"]

    n = len(results) if limit is None else min(limit, len(results))
    for i in range(n):
        yield _record_to_neutral(results[i], f"{game_id_prefix}:{i}")
