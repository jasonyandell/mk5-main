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

`needs` names what a worker must have to seat it (`node`, `torch`, `gpu`, `eq-oracle`).
"""
from __future__ import annotations

import json
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

ROSTER = (WALT, ZEB_010, ZEB_020, EQ_N10)
ANCHOR = WALT.id


def load_all(store) -> dict[str, Participant]:
    out = {}
    for _, data in store.read_all("participants"):
        p = Participant.from_record(json.loads(data))
        out[p.id] = p
    return out
