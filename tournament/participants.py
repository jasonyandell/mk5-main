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
- `zeb-native`   a Zeb release's shipped JavaScript bidder driving its own Rust
                 player compiled natively: the release commit, the git tree of
                 each Rust source directory, the sha256 of each shipped file
                 and of the native binary. A research seat: `needs` names a
                 capability no standing worker advertises.
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
STRAIGHT_DECLARATIONS = (0, 1, 2, 3, 4, 5, 6, 7, 9)


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


# Zeb as the Demon trials seated it: the release's own JavaScript bidding rule
# and the identical Rust player compiled natively (wiki/experiments/demon-vs-zeb-2026-10-15.md).
# 0.4.1 changes no Rust (the trees below are 0.4.0's), so one native binary seats both.
NATIVE_ZEB_BINARY_SHA256 = "c76303d02f82b3bf7b1ca8856cda0e14e5e1eb17df59002ef04191af92d8661a"
NATIVE_ZEB_TREES = {"zeb/src": "4d77a3ba2db5f7eb7ac953d4be562070c760dc5c",
                    "zeb/net": "f25de62037be0317882c3c51fbed5b236d3caf0b",
                    "scenario/core": "61cd6a5225f664be24bcd2f27d74b6744c71e769"}
ZEB_WASM_SHA256 = "bebfdfed14c92ce92586b0d2f35520f70cadc50a70f838200e84073c801e9154"


def zeb_native(version: str, commit: str, files: dict, *, overcall: bool, worlds: int = 160,
               cutoff: float = 0.8) -> Participant:
    """Zeb `version` match-aware, pip declarations as measured, native play; `overcall` is 0.4.1's."""
    tag = "contested" if overcall else "match"
    return Participant(
        id=f"zeb-{version}-{tag}",
        kind="zeb-native",
        pin={"repo": "jasonyandell/mk5-main", "commit": commit, "tag": f"zeb-v{version}",
             "trees": dict(NATIVE_ZEB_TREES), "files": dict(files),
             "wasm_sha256": ZEB_WASM_SHA256, "native_binary_sha256": NATIVE_ZEB_BINARY_SHA256},
        config={"worlds": worlds, "cutoff": cutoff, "seed": 0, "matchAware": True,
                "overcall": overcall, "legalDeclarations": [0, 1, 2, 3, 4, 5, 6], "reading": True},
        needs=("node", "zeb-native"),
        note=(f"@jasonyandell/zeb {version} as tag zeb-v{version} ships it: chooseBid with `match`"
              f"{' and `overcall: true` (the contested constants)' if overcall else ''}, "
              f"the seven pip declarations as measured, {worlds} worlds, cutoff {cutoff}, seed 0, "
              f"behind the partner-pass convention; plays through the identical Rust player "
              f"compiled natively (no search or policy edit). The package's JavaScript is read "
              f"from the tagged source because the npm tarball may not exist yet; its wasm is "
              f"0.4.0's, unchanged."),
    )


PLUNGE_COMMIT ="26043bf93d2d9ac68db393ff4dfb8843a11ef1ef"

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

# FAIR CLOCKS (#181): the same player on a virtual clock. The pinned wasm, book and manifest are
# production Walt's, byte for byte; only the host differs: it hands the wasm a now_us that
# advances `clock_step_us` per read, so the 14 s, 20 s and 500 ms budgets are counts of the
# solver's own work and a game replays bit for bit on any machine at any load. The step is
# 1e6 / (production's pooled now_us reads per real second on the registered panel, one quiet
# Apple-silicon Mac): wiki/experiments/fair-walt-2026-10-16.md.
FAIR_WALT_STEP_US = 0.839
FAIR_WALT = Participant(
    id="walt-plunge-26043bf9-fair",
    kind="plunge-walt",
    pin=json.loads(json.dumps(WALT.pin)),
    config={"difficulty": "native-partner", "auction_pool": 4, "clock": "virtual",
            "clock_step_us": FAIR_WALT_STEP_US},
    needs=("node",),
    note=("Production Walt on a virtual clock (FAIR CLOCKS, #181): the identical pinned wasm, book "
          "and manifest, driven by the same host with now_us a counter (the n-th read of a call "
          f"returns floor(n * {FAIR_WALT_STEP_US}) microseconds), so every deadline is a count of "
          "work and the auction survey's four-worker race is replayed as a schedule in virtual "
          "time. It is the ladder's anchor (1500) and replaces production Walt, withdrawn from "
          "scheduling with its records kept."),
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
          declaration_weight: int = 1, bounded_forecast: bool = False, support_forecast: bool = False,
          bait_bid: int = 30) -> Participant:
    # The source pin selects the sampler; this describes it without changing
    # the bytes of already-published participant records.
    tb = "" if margin_tiebreak else "-notb"
    if not 1 <= declaration_weight <= 1024:
        raise ValueError("declaration_weight must be 1..=1024")
    suffix = f"-read{declaration_weight}" if declaration_weight != 1 else ""
    if bidder != "heuristic":
        suffix += "-bid" + hashlib.sha256(bidder.encode()).hexdigest()[:8]
    extra = {"declaration_weight": declaration_weight} if declaration_weight != 1 else {}
    if bidder.startswith("pimc"):
        suffix += "-straight"
        extra["declarations"] = list(STRAIGHT_DECLARATIONS)
    if not isinstance(bounded_forecast, bool) or (bounded_forecast and not bidder.startswith("pimc")):
        raise ValueError("bounded_forecast requires a boolean and a PIMC bidder")
    if bounded_forecast:
        suffix += "-bounds"
        extra["bounded_forecast"] = True
    if not isinstance(support_forecast, bool) or (support_forecast and not bounded_forecast):
        raise ValueError("support_forecast requires bounded_forecast and a boolean")
    if support_forecast:
        suffix += "-support"
        extra["support_forecast"] = True
    if (type(bait_bid) is not int or not 30 <= bait_bid <= 40
            or (bait_bid != 30 and not bidder.startswith("pimc-bait:"))):
        raise ValueError("bait_bid must be an integer 30..40 and requires a pimc-bait bidder")
    if bait_bid != 30:
        suffix += f"-bait{bait_bid}"
        extra["bait_bid"] = bait_bid
    belief_note = (f" Declaration evidence uses a soft heuristic model: matching original "
                   f"hands have likelihood {declaration_weight}, others 1; posterior draws are IID "
                   f"by rejection, and small fibers use exact weighted sums."
                   if declaration_weight != 1 else "")
    raise_note = ("highest supported legal point bid by monotone bisection"
                  if bidder.startswith("pimc-jump:") else "minimum point raise")
    bid_note = (f"Bids with the arena's `{bidder}` bidder, as E[Q] n=10 does. "
                if bidder == "heuristic" else
                f"Bidder `{bidder}`: common-world double-dummy contract forecasts; "
                f"{raise_note} and partner pass in full bidding mode; "
                f"the nine straight declarations only (not nello's declaration 8). ")
    if bidder.startswith(("pimc-over:", "pimc-jump:", "pimc-bait:")):
        bid_note += (f"Uses DD cutoff {bidder.split(':')[3]} over an opponent's point bid "
                     f"and {bidder.split(':')[2]} when opening; no mark overcalls. ")
    if bidder.startswith("pimc-bait:"):
        opening = "minimum point bid" if bait_bid == 30 else f"point bid {bait_bid}"
        bid_note += ("With no standing bid and at least two seats still to speak, opens the "
                     f"{opening} regardless of its forecast. ")
    if support_forecast:
        bid_note += ("Bid decisions use exact sample support bounds; only a won auction computes "
                     "the best declaration and its winning count. ")
    if bounded_forecast:
        bid_note += ("Exact incumbent bounds skip losing declaration searches while preserving "
                     "the winning declaration and its count on the same sampled worlds. ")
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

# Straight-42 overcalling form: the exact frozen record confirmed against both
# native Zeb 0.4.0 bidders (wiki/experiments/demon-overcalling-2026-10-15.md).
DEMON_OVERCALL_N100 = demon("2e233d29387c2aea0404adffadcaffac3a845a38",
                            {"silas": "2330f0f2e2458f966629830dd7aa091b44872fbb",
                             "scenario/core": "61cd6a5225f664be24bcd2f27d74b6744c71e769",
                             "tournament/demon": "9ddb6a5856af05bada8da8ce1ba6db869386f3ee"},
                            worlds=100, bidder="pimc-over:64:0.875:0.5")

# The overcall's rematch against the Demon (wiki/experiments/rematch-zeb-041-vs-demon-2026-10-15.md):
# 0.4.1 as shipped, and 0.4.0 match-aware as its equal-deals reference.
ZEB_040_MATCH = zeb_native(
    "0.4.0", "29f519953467f6aacd78bd17700f3c67fda6b4eb",
    {"package/index.js": "c36422b30f31d2a9546906ee76ae95e621d5a7d18180a6aef1163d568f762933",
     "package/match.js": "4adb52884209db4f82dfa2a918a3c5253786027f97fb5242114da116d49ccc9a"},
    overcall=False)
ZEB_041_CONTESTED = zeb_native(
    "0.4.1", "0b57a8f5dff9fc7b5249c01be88927b7c8c7ecd0",
    {"package/index.js": "0ee47e61b5c02c0627df0f328747615e723e46cf568bf7eab7a1a93ecce02e3f",
     "package/match.js": "85aaa9b7242f00386cef77134fd332484228c5aac690280f7f3ec7107e292e61"},
    overcall=True)

# Opponent-specific early-40 form, independently confirmed against native
# Zeb 0.4.1 contested (wiki/experiments/demon-ascent-2026-10-16.md).
# This targets that bidder's overcall policy; it is not a general strength claim.
DEMON_BAIT40_N100 = demon("a561552d9007283f51e67fba8d85d4ed528589c1",
                          {"silas": "fad0fc47d19b57c796d182657fb9cf404984a438",
                           "scenario/core": "84714a99f9ff1d1ba532f9983e74f73c83cafe8e",
                           "tournament/demon": "3e7cab13b2819903c7616a480542905c0cd1919c"},
                          worlds=100, bidder="pimc-bait:64:0.875:0.875",
                          sampler_note=_DISTINCT_SAMPLER, bounded_forecast=True,
                          support_forecast=True, bait_bid=40)

# Same early-40 policy, with exact root equivalence and cached immutable rules.
# Instruction-primary native replay: wiki/experiments/demon-instructions-2026-10-16.md.
DEMON_BAIT40_REUSE_N100 = demon("d48909c485dc647ac6bada3577000b5602b95f64",
                                {"silas": "fad0fc47d19b57c796d182657fb9cf404984a438",
                                 "scenario/core": "84714a99f9ff1d1ba532f9983e74f73c83cafe8e",
                                 "tournament/demon": "26dd409acd3e71ac4bcab367fcd46906ffdea4f3"},
                                worlds=100, bidder="pimc-bait:64:0.875:0.875",
                                sampler_note="Walt's completion-count DP, distinct uniform worlds without replacement",
                                bounded_forecast=True, support_forecast=True, bait_bid=40)

ROSTER = (WALT, FAIR_WALT, ZEB_010, ZEB_020, EQ_N10, DEMON_N100, DEMON_FAST_N100, DEMON_N400,
          DEMON_DISTINCT_N100, DEMON_DISTINCT_N1600, DEMON_OVERCALL_N100,
          ZEB_040_MATCH, ZEB_041_CONTESTED, DEMON_BAIT40_N100, DEMON_BAIT40_REUSE_N100)
ANCHOR = FAIR_WALT.id

# Withdrawn from scheduling, records kept: their faceoffs still count in the rating fit, so the
# scale stays connected, but no worker starts a new one. Production Walt thinks on a wall clock
# (config "clock": "real"), which FAIR CLOCKS forbids at the table.
WITHDRAWN = {WALT.id: "wall-clock budgets (FAIR CLOCKS, #181); replaced by " + FAIR_WALT.id}


def supports_straight_games(record: dict) -> bool:
    """All live ladders play straight 42; a broader research policy cannot join.

    STRATEGY-FUSION FENCE: the first PIMC bidder's immutable record covered all
    ten physics, including nello's declaration 8. Keep that historical record
    readable, but never schedule it in the straight tournament.
    """
    cfg = record.get("config", {})
    if record.get("kind") == "demon" and cfg.get("bidder", "").startswith("pimc"):
        allowed = cfg.get("declarations", range(10))
        return bool(allowed) and set(allowed) <= set(STRAIGHT_DECLARATIONS)
    return True


def deterministic_budget(record: dict) -> bool:
    """FAIR CLOCKS: a player thinks on work, never on wall time (CLAUDE.md, #181).

    A record whose config says `"clock": "real"` is refused. A Walt seat must declare the
    virtual clock and its step (the wasm reads a clock; the host must be told which). Every
    other kind decides by worlds, nodes or seeds in its config and has no clock to declare.
    """
    cfg = record.get("config", {})
    if cfg.get("clock") == "real":
        return False
    if record.get("kind") == "plunge-walt":
        return cfg.get("clock") == "virtual" and cfg.get("clock_step_us", 0) > 0
    return True


def schedulable(record: dict) -> bool:
    """May a worker start a new faceoff with this participant?"""
    return (supports_straight_games(record) and record.get("id") not in WITHDRAWN
            and deterministic_budget(record))


def load_all(store) -> dict[str, Participant]:
    out = {}
    for _, data in store.read_all("participants"):
        p = Participant.from_record(json.loads(data))
        out[p.id] = p
    return out
