# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0

"""The concept list: what a neuron could mean, written in Silas.

Each concept is one yes/no Silas query over the viewer's frame, named and
glossed. The list is the vocabulary the audit labels poles with, and it is
fixed before any pole is scored (the pre-registration names its hash).

Every query is evaluated on a belief that resolves no hidden tile
(`SilasBelief.uniform_resolving(frame, 0, 1)`): a viewer-only query answers
exactly, 0 or 1, and a query that reads a hidden world is refused by Silas
with "needs the unresolved tiles". That refusal is the fence: no label here
can read a hand the viewer cannot see (wiki/topics/strategy-fusion-fence.md).

The book's vocabulary (`silas/lib/book.silas`, brought in by `(include
book)`) supplies the game's words: bidder, setter, trump-tile, boss-trump,
trump-out, count-on-trick, partner-winning, can-take-it, off, walker,
sure-trump, safe-trick.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from lem.probe.sample import Sample

BOOK = "(include book)\n"


@dataclass(frozen=True)
class Concept:
    name: str
    gloss: str       # one line, in the game's words
    query: str       # the Silas text, a yes/no query


def _c(name: str, gloss: str, body: str, book: bool = True) -> Concept:
    return Concept(name, gloss, (BOOK if book else "") + f"(query {body})")


CONCEPTS: tuple[Concept, ...] = (
    # -- where in the hand and the trick
    _c("lead", "the viewer leads the trick", "(viewer me) (on-lead me)"),
    _c("second", "the viewer plays second to the trick", "(= 1 (card (roles c d) (trick-play c d)))"),
    _c("third", "the viewer plays third to the trick", "(= 2 (card (roles c d) (trick-play c d)))"),
    _c("last", "the viewer plays last to the trick", "(= 3 (card (roles c d) (trick-play c d)))"),
    _c("trick-1", "the first trick", "(trick-number 1)", book=False),
    _c("trick-2", "the second trick", "(trick-number 2)", book=False),
    _c("late", "the fifth trick or later", "(<= 5 (trick-number))", book=False),
    _c("trick-7", "the last trick", "(trick-number 7)", book=False),
    _c("hand-le-3", "three tiles or fewer left in the viewer's hand", "(viewer me) (<= (quota me) 3)", book=False),
    _c("forced", "the viewer has one legal play", "(= 1 (card (roles d) (own-legal d)))", book=False),
    # -- sides
    _c("bidding-side", "the viewer is the bidder or his partner", "(viewer me) (or (bidder me) (bidders-partner me))"),
    _c("is-bidder", "the viewer is the bidder", "(viewer me) (bidder me)"),
    _c("bidders-lead", "the bidder led the trick, or leads next", "(bidders-lead)"),
    # -- trump
    _c("holds-trump", "the viewer holds a trump", "(mine d) (trump-tile d)"),
    _c("no-trump-held", "the viewer holds no trump", "(none (roles d) (mine d) (trump-tile d))"),
    _c("trumps-3plus", "the viewer holds three trumps or more", "(<= 3 (card (roles d) (mine d) (trump-tile d)))"),
    _c("holds-boss-trump", "the viewer holds the highest live trump", "(mine d) (boss-trump d)"),
    _c("holds-sure-trump", "the viewer holds a trump no live trump outside his hand beats", "(mine d) (sure-trump d)"),
    _c("trump-out", "a trump outside the viewer's hand is still live", "(trump-out)"),
    _c("trump-led", "the trick was led in trump", "(trump-led)"),
    _c("partner-out-of-trump", "the viewer's partner has shown void in trump",
       "(viewer me) (partner me p) (shown-out-of-trump p)"),
    _c("setter-out-of-trump", "an opponent has shown void in trump",
       "(viewer me) (opponent me s) (shown-out-of-trump s)"),
    _c("can-ruff", "the viewer cannot follow the suit led and holds a trump, trump not led",
       "(led-context q) (not (trump q)) (none (roles d) (own-legal d) (in d q)) (mine t) (trump-tile t)"),
    # -- control
    _c("holds-master", "the viewer holds a tile no live tile beats when led", "(mine d) (master d)"),
    _c("masters-2plus", "the viewer holds two masters or more", "(<= 2 (card (roles d) (mine d) (master d)))"),
    _c("holds-walker", "the viewer holds a walker: not trump, cannot lose when led", "(mine d) (walker d)"),
    _c("holds-off", "the viewer holds an off: not trump, can lose when led", "(mine d) (off d)"),
    _c("holds-boss-of-led", "the viewer holds the boss of the suit led", "(led-context q) (mine d) (boss d q)"),
    _c("can-take-it", "a legal play of the viewer's beats the tile taking the trick", "(can-take-it)"),
    _c("void-in-led", "the viewer cannot follow the suit led",
       "(led-context q) (none (roles d) (own-legal d) (in d q))"),
    _c("partner-winning", "the viewer's partner is taking the trick so far", "(partner-winning)"),
    _c("bidder-winning", "the bidder is taking the trick so far", "(bidder-winning)"),
    _c("safe-trick", "partner is taking it and the viewer plays last or the winner cannot be beaten", "(safe-trick)"),
    _c("winner-cannot-be-beaten", "the tile taking the trick cannot be beaten by a setter still to play",
       "(winner-cannot-be-beaten)"),
    # -- count
    _c("count-on-trick", "a count tile is on the trick", "(count-on-trick)"),
    _c("ten-on-trick", "a ten-count tile is on the trick", "(trick-play c d) (count d 10)", book=False),
    _c("holds-count", "the viewer holds count", "(mine d) (count-tile d)"),
    _c("holds-ten", "the viewer holds a ten-count tile", "(mine d) (count d 10)", book=False),
    _c("holds-five", "the viewer holds a five-count tile", "(mine d) (count d 5)", book=False),
    _c("count-out", "a count tile outside the viewer's hand is still live", "(live d) (not (mine d)) (count-tile d)"),
    _c("count-out-20plus", "twenty count or more is live outside the viewer's hand",
       "(<= 20 (points (roles (domino d)) (live d) (not (mine d))))"),
    # -- the contract
    _c("made", "the bidding side has made its bid", "(to-make 0)", book=False),
    _c("set", "the bidding side is set", "(to-set 0)", book=False),
    _c("to-make-le-10", "the bidding side needs ten points or fewer", "(<= (to-make) 10)", book=False),
    _c("to-set-le-5", "the setters need five points or fewer", "(<= (to-set) 5)", book=False),
    _c("to-make-ge-20", "the bidding side needs twenty points or more", "(<= 20 (to-make))", book=False),
    # -- hand shape
    _c("holds-double", "the viewer holds a double", "(mine d) (double d)", book=False),
    _c("doubles-2plus", "the viewer holds two doubles or more", "(<= 2 (card (roles d) (mine d) (double d)))", book=False),
)

NAMES = tuple(c.name for c in CONCEPTS)
assert len(set(NAMES)) == len(NAMES), "concept names are unique"

# -- v2 (#185): what the first audit found missing. The declaration itself (Silas names the
# trump context q*, so a pip is told by a non-double tile of that pip being trump; doubles
# trump by two doubles being trump; no trump by no trump context), the readers' conjunctions
# of trick position with what is happening, and the voids a seat has shown outside trump.
PIPS = ("blanks", "ones", "twos", "threes", "fours", "fives", "sixes")
def _two_tiles(q: int) -> str:
    """Two non-double tiles of pip q with different other pips: both are trump
    exactly when q is the trump pip (one tile alone is also trump under its
    other pip's declaration; under doubles trump neither is)."""
    return f"(trump-tile {q}-{(q + 1) % 7}) (trump-tile {q}-{(q + 2) % 7})"


V2_ONLY: tuple[Concept, ...] = tuple(
    _c(f"trump-is-{PIPS[q]}", f"the declaration is {PIPS[q]} (two non-double {PIPS[q]} tiles are trump)", _two_tiles(q))
    for q in range(7)
) + (
    _c("trump-is-doubles", "doubles are trump", "(trump-tile 6-6) (trump-tile 0-0)"),
    _c("no-trump-game", "the hand is played with no trump", "(none (roles t) (trump t))"),
    _c("lead-no-trump", "the viewer leads holding no trump", "(viewer me) (on-lead me) (none (roles d) (mine d) (trump-tile d))"),
    _c("lead-boss-trump", "the viewer leads holding the boss trump", "(viewer me) (on-lead me) (mine d) (boss-trump d)"),
    _c("lead-made", "the viewer leads with the bid already made", "(viewer me) (on-lead me) (to-make 0)"),
    _c("lead-set", "the viewer leads with the bid already set", "(viewer me) (on-lead me) (to-set 0)"),
    _c("last-partner-winning", "the viewer plays last and partner is winning", "(last-to-play) (partner-winning)"),
    _c("last-setter-winning", "the viewer plays last and an opponent is winning",
       "(last-to-play) (viewer me) (opponent me w) (current-winner w)"),
    _c("last-count-on-trick", "the viewer plays last and count is on the trick", "(last-to-play) (count-on-trick)"),
    _c("last-trump-led", "the viewer plays last to a trump lead", "(last-to-play) (trump-led)"),
    _c("third-partner-winning", "the viewer plays third and partner is winning",
       "(= 2 (card (roles c d) (trick-play c d))) (partner-winning)"),
    _c("second-trump-led", "the viewer plays second to a trump lead", "(= 1 (card (roles c d) (trick-play c d))) (trump-led)"),
    _c("partner-void-off-suit", "the viewer's partner has shown void in a suit that is not trump",
       "(viewer me) (partner me p) (some (roles q) (not (trump q)) (shown-void p q))"),
    _c("setter-void-off-suit", "an opponent has shown void in a suit that is not trump",
       "(viewer me) (opponent me s) (some (roles q) (not (trump q)) (shown-void s q))"),
    _c("setter-void-in-led", "an opponent still to play has shown void in the suit led",
       "(viewer me) (opponent me s) (led-context q) (none (roles y) (trick-play s y)) (shown-void s q)"),
)
CONCEPTS_V2: tuple[Concept, ...] = CONCEPTS + V2_ONLY
NAMES_V2 = tuple(c.name for c in CONCEPTS_V2)
assert len(set(NAMES_V2)) == len(NAMES_V2), "v2 concept names are unique"


def identity(concepts: tuple[Concept, ...] = CONCEPTS) -> str:
    """The sha256 of a concept list's text: a pre-registration names it."""
    text = "\n".join(f"{c.name}\t{c.gloss}\t{c.query}" for c in concepts)
    return hashlib.sha256(text.encode()).hexdigest()


def compile_all(concepts: tuple[Concept, ...] = CONCEPTS):
    from scenario_core import SilasQuery
    return [SilasQuery(c.query) for c in concepts]


def evaluate(sample: Sample, start: int = 0, stop: int | None = None, budget: int = 10_000_000,
             log=print) -> np.ndarray:
    """[rows, C] bool: each concept at each row, exactly."""
    from scenario_core import SilasBelief, SilasFrame
    queries = compile_all()
    stop = len(sample) if stop is None else min(stop, len(sample))
    out = np.zeros((stop - start, len(queries)), bool)
    for r in range(start, stop):
        frame = SilasFrame(*sample.frame_args(r))
        belief = SilasBelief.uniform_resolving(frame, 0, 1)
        for k, q in enumerate(queries):
            e = q.summarize(belief, budget).event_probability
            if e.authority.label != "exact":
                raise AssertionError(f"{NAMES[k]} is not exact at row {r}: {e.authority.label}")
            out[r - start, k] = e.value == 1
        if (r - start) % 20_000 == 0 and log is not None:
            log(f"  concepts: row {r:,} of {stop:,}")
    return out


def save(path: str | Path, labels: np.ndarray) -> None:
    np.savez_compressed(path, labels=labels, names=np.array(NAMES), identity=np.array(identity()))


def write_list(path: str | Path) -> None:
    """The concept list as JSON, for the record and the readers."""
    Path(path).write_text(json.dumps(
        {"identity": identity(), "concepts": [c.__dict__ for c in CONCEPTS]}, indent=1))
