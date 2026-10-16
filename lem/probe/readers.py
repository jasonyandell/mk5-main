# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0

"""Haiku readers: the cheap label, explained then scored on held-out cards.

Two passes per pole, each one fresh `claude -p --model haiku` call with
nothing in context but the cards (no activations, no Silas label, no
other pole):

1. **Explain.** The reader sees `EXPLAIN` cards from the pole's top set and
   names, in at most fifteen words, the one thing they share, or `none`.
2. **Score.** The reader sees the description and `SCORE` held-out cards,
   half from the top set and half drawn at random from all rows, shuffled,
   and marks each card Y or N. The *detection accuracy* is the mean of the
   hit rate on top cards and the miss rate on random cards (balanced), with
   its Wilson interval; chance is 0.5. The second null scores the same cards
   against another pole's description (`shifted`): a description that
   fits every card scores there too.

Cards are built once (`cards.py`); this module only talks to the reader.
Calls run in a small pool; every result is cached as JSON per pole so a
rerun costs nothing.
"""
from __future__ import annotations

import json
import random
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from lem.probe.audit import wilson

EXPLAIN = 12
SCORE = 10            # per side: SCORE top cards and SCORE random cards
MODEL = "haiku"
WORKERS = 6
SEED = 20261015

EXPLAIN_PROMPT = """You are reading positions from the game Texas 42 (dominoes, four seats, two teams, seven tricks; the bidder's side needs its bid in points, count tiles are 5-5, 6-4 worth ten and 5-0, 4-1, 3-2 worth five, each trick is one more point).
Below are {n} positions, each written from one player's seat with only what that player can see. All of them strongly activate the same unit of a small neural network that plays 42.
Name the ONE thing these positions share, in at most fifteen words, as a 42 player would say it (for example "holds the top trump", "partner is winning the trick", "void in the suit led with a trump in hand"). If they share nothing you can name, answer exactly: none.
Answer with the description only.

{cards}"""

SCORE_PROMPT = """You are reading positions from the game Texas 42 (dominoes, four seats, two teams, seven tricks; count tiles are 5-5, 6-4 worth ten and 5-0, 4-1, 3-2 worth five).
A description of a kind of position: "{description}"
For each of the {n} positions below, written from one player's seat, decide whether the description holds.
Answer with exactly {n} letters, one per position in order, Y if it holds and N if it does not, nothing else.

{cards}"""


def ask(prompt: str, model: str = MODEL) -> str:
    out = subprocess.run(["claude", "-p", "--model", model, prompt], capture_output=True, text=True, timeout=300)
    if out.returncode != 0:
        raise RuntimeError(out.stderr.strip()[-400:])
    return out.stdout.strip()


def numbered(cards: list[str]) -> str:
    return "\n\n".join(f"Position {i + 1}:\n{c}" for i, c in enumerate(cards))


def parse_marks(text: str, n: int) -> list[bool] | None:
    letters = [ch for ch in text.upper() if ch in "YN"]
    if len(letters) < n:
        return None
    return [ch == "Y" for ch in letters[:n]]


def read_pole(pole: dict, out_dir: Path, shift_description: str | None = None, model: str = MODEL) -> dict:
    """pole: {"name", "explain": [cards], "score_top": [cards], "score_random": [cards]}."""
    path = out_dir / f"{pole['name']}.json"
    if path.exists():
        return json.loads(path.read_text())
    rng = random.Random(f"{SEED}:{pole['name']}")
    description = ask(EXPLAIN_PROMPT.format(n=len(pole["explain"]), cards=numbered(pole["explain"])), model)
    description = description.splitlines()[0].strip().strip('"') if description else "none"
    cards = [(c, True) for c in pole["score_top"]] + [(c, False) for c in pole["score_random"]]
    rng.shuffle(cards)
    result = {"name": pole["name"], "description": description, "order": [t for _, t in cards]}
    for key, desc in (("own", description), ("shifted", shift_description)):
        if desc is None or desc.lower() == "none":
            result[key] = None
            continue
        text = ask(SCORE_PROMPT.format(description=desc, n=len(cards), cards=numbered([c for c, _ in cards])), model)
        marks = parse_marks(text, len(cards))
        if marks is None:
            result[key] = {"raw": text[:200], "marks": None}
            continue
        hits = sum(m for m, (_, t) in zip(marks, cards) if t)
        misses = sum(not m for m, (_, t) in zip(marks, cards) if not t)
        n_top = sum(t for _, t in cards)
        n_rand = len(cards) - n_top
        acc = 0.5 * (hits / max(1, n_top) + misses / max(1, n_rand))
        lo, hi = wilson(hits + misses, len(cards))
        result[key] = {"marks": marks, "hits": hits, "misses": misses, "accuracy": acc, "lo": lo, "hi": hi,
                       "description": desc}
    path.write_text(json.dumps(result, indent=1))
    return result


def run(cards_path: str | Path, out_dir: str | Path, workers: int = WORKERS, model: str = MODEL, log=print) -> list[dict]:
    poles = json.loads(Path(cards_path).read_text())
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Explain first (so the shifted null can borrow a neighbour's description), then score.
    results: dict[str, dict] = {}

    def explain_only(pole):
        return read_pole({**pole, "score_top": [], "score_random": []}, out_dir / "explain", None, model)

    (out_dir / "explain").mkdir(exist_ok=True)
    with ThreadPoolExecutor(workers) as ex:
        descs = list(ex.map(explain_only, poles))
    names = [p["name"] for p in poles]
    desc_of = {d["name"]: d["description"] for d in descs}
    log(f"explained {len(descs)} poles; {sum(d['description'].lower() == 'none' for d in descs)} said none")

    def score(i):
        pole = poles[i]
        shifted = desc_of[names[(i + len(names) // 2) % len(names)]]
        full = {**pole}
        cached = out_dir / f"{pole['name']}.json"
        if not cached.exists():
            # reuse the explanation: write it so read_pole scores without re-asking
            r = read_pole_with(full, out_dir, desc_of[pole["name"]], shifted, model)
            return r
        return json.loads(cached.read_text())

    with ThreadPoolExecutor(workers) as ex:
        for i, r in enumerate(ex.map(score, range(len(poles)))):
            results[r["name"]] = r
            if i % 32 == 0:
                log(f"  scored {i + 1}/{len(poles)}")
    return [results[n] for n in names]


def read_pole_with(pole: dict, out_dir: Path, description: str, shift_description: str | None, model: str) -> dict:
    """`read_pole` with the explanation already in hand."""
    path = out_dir / f"{pole['name']}.json"
    rng = random.Random(f"{SEED}:{pole['name']}")
    cards = [(c, True) for c in pole["score_top"]] + [(c, False) for c in pole["score_random"]]
    rng.shuffle(cards)
    result = {"name": pole["name"], "description": description, "order": [t for _, t in cards]}
    for key, desc in (("own", description), ("shifted", shift_description)):
        if desc is None or desc.lower() == "none":
            result[key] = None
            continue
        text = ask(SCORE_PROMPT.format(description=desc, n=len(cards), cards=numbered([c for c, _ in cards])), model)
        marks = parse_marks(text, len(cards))
        if marks is None:
            result[key] = {"raw": text[:200], "marks": None}
            continue
        hits = sum(m for m, (_, t) in zip(marks, cards) if t)
        misses = sum(not m for m, (_, t) in zip(marks, cards) if not t)
        n_top = sum(t for _, t in cards)
        n_rand = len(cards) - n_top
        acc = 0.5 * (hits / max(1, n_top) + misses / max(1, n_rand))
        lo, hi = wilson(hits + misses, len(cards))
        result[key] = {"marks": marks, "hits": hits, "misses": misses, "accuracy": acc, "lo": lo, "hi": hi,
                       "description": desc}
    path.write_text(json.dumps(result, indent=1))
    return result


def summarize(results: list[dict]) -> dict:
    own = [r["own"]["accuracy"] for r in results if r.get("own") and r["own"].get("marks") is not None]
    sh = [r["shifted"]["accuracy"] for r in results if r.get("shifted") and r["shifted"].get("marks") is not None]
    none = sum(r["description"].lower() == "none" for r in results)
    detect = sum(r.get("own") is not None and r["own"].get("lo", 0) > 0.5 for r in results)
    return {"poles": len(results), "none": none, "scored": len(own),
            "accuracy_mean": sum(own) / max(1, len(own)), "shifted_mean": sum(sh) / max(1, len(sh)),
            "detected": detect, "detected_share": detect / max(1, len(results))}
