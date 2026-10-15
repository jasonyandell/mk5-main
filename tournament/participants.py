# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0

"""Participants: immutable records, each pinned to bytes that cannot move.

A participant is a JSON file `participants/<id>.json` in the record store,
added once and never edited. Adding one (Zeb 0.3.0 the day it is on npm) is
appending a file, not changing code: `python -m tournament add-zeb 0.3.0`
reads the registry's integrity hash and writes the record.

Kinds, and what pins each:

- `zeb-npm`      the npm tarball's sha512 integrity (and its sha1 shasum).
- `plunge-walt`  the Plunge commit that is deployed, plus the sha256 of each
                 file the seat loads (the Walt WASM, the bid book, its
                 manifest), fetched from that commit and checked on load.
- `arena`        an mk5 arena spec run by this checkout, plus the sha256 of
                 any weights it loads; the faceoff records the mk5 commit
                 that ran it.
- `demon`        the Demon (STRATEGY-FUSION FENCE: a fenced tournament
                 opponent; measuring stick only): an mk5 commit and the git
                 tree of each source directory its binary is built from,
                 plus its parameters (worlds, tiebreak, bidder).

`needs` names what a worker must have to seat it (`node`, `torch`, `gpu`, `eq-oracle`).
"""
from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass, field

SCHEMA = "texas42-tournament-participant/1"


@dataclass(frozen=True)
class Participant:
    id: str
    kind: str
    pin: dict
    config: dict = field(default_factory=dict)
    needs: tuple[str, ...] = ()
    note: str = ""

    def to_record(self) -> dict:
        """A function of the participant alone (no timestamp: the commit has
        one), so two writers adding the same participant write the same bytes
        and the store sees one add, not a conflict."""
        return {"schema": SCHEMA, "id": self.id, "kind": self.kind, "pin": self.pin,
                "config": self.config, "needs": list(self.needs), "note": self.note}

    @staticmethod
    def from_record(r: dict) -> "Participant":
        if r.get("schema") != SCHEMA:
            raise ValueError(f"not a participant record: {r.get('schema')!r}")
        return Participant(r["id"], r["kind"], r["pin"], r.get("config", {}),
                           tuple(r.get("needs", ())), r.get("note", ""))

    def path(self) -> str:
        return f"participants/{self.id}.json"


def zeb_npm(version: str, integrity: str, shasum: str, *, worlds: int = 160,
            cutoff: float = 0.8) -> Participant:
    return Participant(
        id=f"zeb-{version}",
        kind="zeb-npm",
        pin={"package": "@jasonyandell/zeb", "version": version,
             "integrity": integrity, "shasum": shasum},
        config={"worlds": worlds, "cutoff": cutoff},
        needs=("node",),
        note=(f"@jasonyandell/zeb {version} from the npm registry, integrity-checked; "
              f"the package's own bidder (forecast + chooseBid, cutoff {cutoff}) behind "
              f"Plunge's partner-pass convention; {worlds} worlds (the package default)."),
    )


PLUNGE_COMMIT = "26043bf93d2d9ac68db393ff4dfb8843a11ef1ef"

# The initial roster (2026-10-14). The anchor is production Walt: the player
# Jason's family plays every day, so its rating is 1500 by definition.
WALT = Participant(
    id="walt-plunge-26043bf9",
    kind="plunge-walt",
    pin={
        "repo": "jasonyandell/plunge",
        "commit": PLUNGE_COMMIT,
        "deployed": "https://plunge.texas42.workers.dev/version.json reported this build on 2026-10-14",
        "files": {
            "src/ai/phone/walt-player.wasm": "40d7a1eea658627f7bab37d3c1888ad3bf00216fb23a5d2ceeb7d2d1f4219119",
            "src/ai/book/played.json": "3e2bf7ae00170cd98aaaff74a174a4e94c3d416b5fda1c2a5d2fd3f65550a642",
            "src/ai/phone/manifest.json": "e15bc53ef6fa923ef090f3518e17097adc48d451653cea93a657798c99420860",
        },
        "player": "walt-table-v2",
        "source": {"repo": "jasonyandell/texas-42",
                   "commit": "cb1ef3b23072e4c268f31f625f2b61d5facc1929"},
    },
    config={"difficulty": "native-partner", "auction_pool": 4, "clock": "real"},
    needs=("node",),
    note=("Plunge's production Walt (difficulty native-partner, the default): the deployed "
          "walt-player.wasm, played as src/ai/native.ts livePlayerCall plays it (160 worlds on "
          "the bidder's opening lead, else 40 with the partner check); bids by the frozen book "
          "on a catalogue hand and otherwise by the shipped live survey (src/ai/auction.ts, "
          "a 4-worker pool, 20 s). Tournament deals are random, so the live survey is what "
          "bids almost every hand; on Plunge itself every deal is a catalogue deal."),
)

ZEB_010 = zeb_npm("0.1.0",
                  "sha512-GyEDG0FESKp31IjsvI1uGZkKI+eLhbO92DF+vfUy0bGT5Z8PoUSGNX/nqg+qMWv1cHz4QgUD3ssZzOQ2HwJOdQ==",
                  "30aee96882e6f2f72b186a850a8ac56205a2b6fb")
ZEB_020 = zeb_npm("0.2.0",
                  "sha512-YnlKLquBrPxOZwKemdNpEahsIFnus8Zl5Q/ukQTtmSQxxXdBlEQyORtb9N/dUfIiaoILpIr7wuKJ4NxxLdW3Uw==",
                  "127634861351003b29034caa0af44e212bd44817")

# STRATEGY-FUSION FENCE: the E[Q] n=10 participant is the oracle era's champion,
# seated only as a fenced measuring stick (wiki/topics/strategy-fusion-fence.md).
# Its seat is tournament/eq_fenced_seat.py; nothing in Zeb reads it.
EQ_N10 = Participant(
    id="eq-n10-fenced",
    kind="arena",
    pin={"spec": "heuristic+lens:ev", "n_samples": 10,
         "checkpoint": "forge/models/domino-qval-large-3.3M-qgap0.071-qmae0.94.ckpt",
         "checkpoint_sha256": "5f3f00de6215a28bcff76fc80844b53b595f4e98e6bc6ef4c53928bd73b325c1"},
    config={},
    needs=("torch", "gpu", "eq-oracle"),
    note=("STRATEGY-FUSION FENCE: E[Q] n=10 (arena lens:ev, the oracle era's champion) with "
          "the arena's heuristic (Roberson risk-budget) bidder. A fenced measuring stick: "
          "per-world clairvoyant Q averaged. Never a foundation."),
)


# STRATEGY-FUSION FENCE. The Demon: a fenced tournament opponent; measuring stick only.
# Classic PIMC on the CPU, seated so Walt and Zeb face strategy fusion head-on
# (wiki/topics/the-demon.md). Its seat is tournament/demon_fenced_seat.py.
DEMON_SOURCE = ("silas", "scenario/core", "tournament/demon")


def demon(commit: str, trees: dict[str, str], *, worlds: int, margin_tiebreak: bool = True,
          bidder: str = "heuristic", sampler_note: str = "Walt's exact-uniform sampler",
          declaration_weight: int = 1) -> Participant:
    # The source pin selects the sampler; this describes it without changing
    # the bytes of already-published participant records.
    tb = "" if margin_tiebreak else "-notb"
    if not 1 <= declaration_weight <= 1024:
        raise ValueError("declaration_weight must be 1..=1024")
    suffix = f"-read{declaration_weight}" if declaration_weight != 1 else ""
    if bidder != "heuristic":
        suffix += "-bid" + hashlib.sha256(bidder.encode()).hexdigest()[:8]
    extra = {"declaration_weight": declaration_weight} if declaration_weight != 1 else {}
    belief_note = (f" Declaration evidence uses a soft heuristic model: matching original "
                   f"hands have likelihood {declaration_weight}, others 1; posterior draws are IID "
                   f"by rejection, and small fibers use exact weighted sums."
                   if declaration_weight != 1 else "")
    bid_note = (f"Bids with the arena's `{bidder}` bidder, as E[Q] n=10 does. "
                if bidder == "heuristic" else
                f"Bidder `{bidder}`: common-world double-dummy contract forecasts; "
                f"minimum point raise and partner pass in full bidding mode. ")
    return Participant(
        id=f"demon-pimc-n{worlds}{tb}{suffix}-{commit[:8]}",
        kind="demon",
        pin={"repo": "jasonyandell/mk5-main", "commit": commit, "trees": dict(trees)},
        config={"worlds": worlds, "margin_tiebreak": margin_tiebreak, "bidder": bidder, **extra},
        needs=("cargo",),
        note=(f"STRATEGY-FUSION FENCE: the Demon, classic PIMC. At each play it draws {worlds} "
              f"worlds from its fiber ({sampler_note}; the whole fiber when "
              f"smaller), solves each double dummy (Silas's ClairvoyantSolver, make-or-set of the "
              f"contract), and plays the tile that makes or sets in the most worlds"
              f"{', then the best summed double-dummy margin' if margin_tiebreak else ''}, then the "
              f"lowest tile id. {bid_note}"
              f"Built on the CPU from the pinned git trees. A fenced measuring stick, never a "
              f"foundation.{belief_note}"),
    )


# The Demon as first seated (2026-10-14): 100 worlds, margin tiebreak, heuristic bidder.
# 100 worlds cost a mean 32 ms a decision on an M-series core (worst opening lead ~3 s).
DEMON_N100 = demon("b88c5432e16bbc5d32ad8d1a71e010e7144da24c",
                   {"silas": "75b67c11f21916344960fb5e9b7818197b04d234",
                    "scenario/core": "11735eb06e3f89c5c9c2d86ece0295a077c2cde8",
                    "tournament/demon": "40738d38be3880339f0ee49e4da1ab4ad813b791"},
                   worlds=100)

# Exact move/value parity with smaller per-world search tables. The original
# pin remains a baseline; 400 worlds is a challenger, not a strength verdict.
_DEMON_COMPACT_COMMIT = "18ac096d4c56506f8115f53daf2c95d6151ee0ea"
_DEMON_COMPACT_TREES = {"silas": "c10cb8b362c00b0ff7efad58d4e77e85219f346f",
                        "scenario/core": "11735eb06e3f89c5c9c2d86ece0295a077c2cde8",
                        "tournament/demon": "2431c1f1a62cbf2f4dc53d9273564f54bf285af5"}
DEMON_FAST_N100 = demon(_DEMON_COMPACT_COMMIT, _DEMON_COMPACT_TREES, worlds=100)
DEMON_N400 = demon(_DEMON_COMPACT_COMMIT, _DEMON_COMPACT_TREES, worlds=400)

# Compact ownership keys, equivalent-play pruning, exact threshold margins,
# and uniform samples without replacement. The code pin fixes the algorithm.
_DEMON_POWER_COMMIT = "228812ff5348868213f7f6ef1eda762f52a52429"
_DEMON_POWER_TREES = {"silas": "9dff8ca1be3bfbd511ecf84dc27934c71aab65cb",
                      "scenario/core": "66a68a2aba792c870d8be1692833669d34652658",
                      "tournament/demon": "35495a961ee9cc4fd24f03b1dff0809a4e08951f"}
_DISTINCT_SAMPLER = "uniform without replacement through exact completion-count ranks"
DEMON_DISTINCT_N100 = demon(_DEMON_POWER_COMMIT, _DEMON_POWER_TREES, worlds=100,
                            sampler_note=_DISTINCT_SAMPLER)
DEMON_DISTINCT_N1600 = demon(_DEMON_POWER_COMMIT, _DEMON_POWER_TREES, worlds=1600,
                             sampler_note=_DISTINCT_SAMPLER)

ROSTER = (WALT, ZEB_010, ZEB_020, EQ_N10, DEMON_N100, DEMON_FAST_N100, DEMON_N400,
          DEMON_DISTINCT_N100, DEMON_DISTINCT_N1600)
ANCHOR = WALT.id


def load_all(store) -> dict[str, Participant]:
    out = {}
    for _, data in store.read_all("participants"):
        p = Participant.from_record(json.loads(data))
        out[p.id] = p
    return out
