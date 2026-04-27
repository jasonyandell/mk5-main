"""Freeze the 5-decision perf-bench subset to a hermetic JSONL.

The bench (``burl/eval/bench_decision_latency.py``) reads this file to
resolve which ``global_idx``-es to evaluate against the gus eval corpus.
It is one-shot: we do not regenerate it on every bench run, otherwise
"perf delta vs prior run" stops meaning anything.

Subset rationale (defended in the wiki @ ``burl-perf-phase0``):

  * **Trick-position spread**: dec_in_game = global_idx % 28, and trick
    index = dec_in_game // 4.  We pick decisions at trick indices
    0, 2, 4, 5, 6 (i.e. trick positions 1, 3, 5, 6, 7 in 1-indexed
    terms).  This covers the early-game wide-open hand (n_legal=7) all
    the way to the trivially forced last play (n_legal=1).

  * **Declaration spread**: ``corpus_eval_20.pt`` puts a different
    declaration on each game.  We pick from games 0..4 so the subset
    spans declarations 0..4 — five distinct contracts means the
    bench's prefill cost is averaged over distinct rules-tool primer
    payloads, not biased by one declaration's prompt shape.

The picked set is gi = 0, 36, 72, 104, 136 — first decision of the
target trick within games 0..4 respectively.

Hermeticity:
  * The JSONL records the corpus.pt path + a SHA256 of its bytes so
    the bench fails fast if the corpus changes underneath it.
  * For each row we capture the metadata that ``BurlDecision`` exposes
    plus the narrator's initial 7-domino hand (``slot_to_dom``).  The
    bench does NOT use this metadata to bypass replay — it asserts
    that ``build_burl_decision(corpus, gi)`` round-trips to the same
    fingerprint before running.

Usage::

    PYTHONPATH=. .venv/bin/python -u burl/eval/data/freeze_perf_subset.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))


SUBSET_GIS = [0, 36, 72, 104, 136]
DECISIONS_PER_GAME = 28
PLAYERS_PER_TRICK = 4
OUT_PATH = Path(__file__).resolve().parent / "perf_subset_5.jsonl"
DEFAULT_CORPUS = Path("/Users/jason/code/mk5-main/gus/data/corpus_eval_20.pt")


def sha256_of(path: Path, chunk_bytes: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fid:
        while chunk := fid.read(chunk_bytes):
            h.update(chunk)
    return h.hexdigest()


def fingerprint_decision(bd) -> str:
    """Stable hash of the load-bearing fields of a BurlDecision."""
    payload = {
        "seed": int(bd.seed),
        "declaration": int(bd.declaration),
        "narrator_seat": int(bd.narrator_seat),
        "trick_idx": int(bd.trick_idx),
        "legal_plays": sorted(int(x) for x in bd.legal_plays),
        "bot_play": int(bd.bot_play),
        "bot_eq": round(float(bd.bot_eq), 6),
        "per_play_eq": {
            str(k): round(float(v), 6)
            for k, v in sorted(bd.per_play_eq.items())
        },
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


def main() -> int:
    from burl.eval.gus_eval_bridge import build_burl_decision, load_corpus

    corpus_sha = sha256_of(DEFAULT_CORPUS)
    print(f"corpus_eval_20.pt SHA256 = {corpus_sha}")

    corpus = load_corpus(DEFAULT_CORPUS)
    n_games = len(corpus["results"])
    rows: list[dict] = []
    for gi in SUBSET_GIS:
        bd = build_burl_decision(corpus, gi)
        game_idx = gi // DECISIONS_PER_GAME
        dec_in_game = gi % DECISIONS_PER_GAME
        trick_idx = dec_in_game // PLAYERS_PER_TRICK
        narrator_initial_hand = [
            int(d) for d in corpus["results"][game_idx].hands[bd.narrator_seat]
        ]
        row = {
            "global_idx": int(gi),
            "game_idx": int(game_idx),
            "dec_in_game": int(dec_in_game),
            "trick_idx_zero_based": int(trick_idx),
            "trick_position_one_based": int(trick_idx + 1),
            "seed": int(bd.seed),
            "declaration": int(bd.declaration),
            "narrator_seat": int(bd.narrator_seat),
            "n_legal": len(bd.legal_plays),
            "legal_plays": [int(x) for x in bd.legal_plays],
            "bot_play": int(bd.bot_play),
            "bot_eq": round(float(bd.bot_eq), 6),
            "eq_gap": round(float(bd.eq_gap), 6),
            "per_play_eq": {
                str(int(k)): round(float(v), 6)
                for k, v in sorted(bd.per_play_eq.items())
            },
            "narrator_initial_hand": narrator_initial_hand,
            "fingerprint": fingerprint_decision(bd),
        }
        rows.append(row)
        print(
            f"  gi={gi:3d}  game={game_idx} decl={bd.declaration} "
            f"seat={bd.narrator_seat} trick={trick_idx + 1}/7 "
            f"n_legal={len(bd.legal_plays)} bot={bd.bot_play} "
            f"fp={row['fingerprint'][:12]}"
        )

    header = {
        "_meta": True,
        "subset_name": "perf_subset_5",
        "n_decisions": len(rows),
        "corpus_path": str(DEFAULT_CORPUS),
        "corpus_sha256": corpus_sha,
        "n_games_in_corpus": int(n_games),
        "decisions_per_game": int(DECISIONS_PER_GAME),
        "players_per_trick": int(PLAYERS_PER_TRICK),
        "trick_positions_one_based_covered": sorted(
            {r["trick_position_one_based"] for r in rows}
        ),
        "declarations_covered": sorted({r["declaration"] for r in rows}),
        "rationale": (
            "5 decisions covering trick positions 1/3/5/6/7 across "
            "declarations 0..4 (one per game so prefill cost averages over "
            "distinct rules-tool primer payloads, not biased to one decl)."
        ),
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w") as fid:
        fid.write(json.dumps(header) + "\n")
        for row in rows:
            fid.write(json.dumps(row) + "\n")

    print(f"\nwrote {OUT_PATH} ({len(rows)} decisions + header)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
