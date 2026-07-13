#!/usr/bin/env python3
"""Inventory Gus corpora for the W42 book-claim recovery.

The large legacy Gus corpus is about 111 GB, so this script separates cheap
filesystem coverage from bounded torch-object inspection. By default it inspects
representative legacy chunks plus all small v2/eval files. Pass
``--full-legacy-scan`` when a full one-chunk-at-a-time schema/count audit is
worth the wall time.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "gus" / "data"
OUT_DIR = ROOT / "w42" / "claim_data_inventory"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

LEGACY_RE = re.compile(r"corpus_train_chunk_(\d+)-(\d+)\.pt$")
LEGACY_SINGLE_RE = re.compile(r"corpus_train_100\.pt$")
V2_RE = re.compile(r"corpus_v2_train_(\d+)-(\d+)_d0-9\.pt$")
EVAL_RE = re.compile(r"corpus(?:_v2)?_eval(?:_\d+)?\.pt$")


DECL_NAMES = {
    0: "blanks",
    1: "aces",
    2: "deuces",
    3: "treys",
    4: "fours",
    5: "fives",
    6: "sixes",
    7: "doubles_trump",
    8: "variant_decl_8",
    9: "no_trump",
}


CSV_FILES_FIELDS = [
    "path",
    "corpus_family",
    "bytes",
    "size_gib",
    "start_seed",
    "end_seed",
    "seed_count",
    "inferred_decl_policy",
    "inspected",
    "game_count",
    "decision_count",
    "decl_counts_json",
    "sample_counts_json",
    "schema",
    "has_bid_value",
    "bid_values_json",
    "has_oracle_softmax_per_seat",
    "has_legal_mask_per_seat",
    "has_voids_per_seat",
    "blob_keys_json",
    "game_attrs_json",
    "decision_attrs_json",
    "load_seconds",
]


ROUTE_FIELDS = [
    "bead_id",
    "claim_family",
    "book_chapters",
    "matrix_previous_route",
    "legacy_111gb_support",
    "v2_support",
    "classification_after_inventory",
    "directly_testable_now",
    "filtered_rows_possible",
    "still_requires_generation",
    "reason",
    "next_data_action",
]


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def classify_file(path: Path) -> dict[str, Any]:
    name = path.name
    stat = path.stat()
    row: dict[str, Any] = {
        "path": rel(path),
        "bytes": stat.st_size,
        "size_gib": round(stat.st_size / (1024**3), 4),
        "start_seed": "",
        "end_seed": "",
        "seed_count": "",
        "corpus_family": "other",
        "inferred_decl_policy": "",
    }
    if m := LEGACY_RE.match(name):
        start = int(m.group(1))
        end = int(m.group(2))
        row.update(
            {
                "corpus_family": "legacy_seed_mod_decl",
                "start_seed": start,
                "end_seed": end,
                "seed_count": end - start + 1,
                "inferred_decl_policy": "decl_id = seed % 10",
            }
        )
    elif LEGACY_SINGLE_RE.match(name):
        row.update(
            {
                "corpus_family": "legacy_single_100",
                "start_seed": 0,
                "end_seed": 99,
                "seed_count": 100,
                "inferred_decl_policy": "decl_id = seed % 10; likely predecessor/duplicate of first chunk",
            }
        )
    elif m := V2_RE.match(name):
        start = int(m.group(1))
        end = int(m.group(2))
        row.update(
            {
                "corpus_family": "v2_all_decl_fixed_bid30",
                "start_seed": start,
                "end_seed": end,
                "seed_count": end - start + 1,
                "inferred_decl_policy": "all decl_ids 0-9 per seed",
            }
        )
    elif EVAL_RE.match(name):
        row.update({"corpus_family": "eval", "inferred_decl_policy": "from payload"})
    return row


def representative_legacy_files(paths: list[Path], n: int) -> set[Path]:
    if not paths:
        return set()
    if n <= 0 or n >= len(paths):
        return set(paths)
    indexes = sorted({round(i * (len(paths) - 1) / max(1, n - 1)) for i in range(n)})
    return {paths[i] for i in indexes}


def attr_names(obj: Any) -> list[str]:
    names = set(getattr(obj, "__dict__", {}).keys())
    if not names:
        names.update(name for name in dir(obj) if not name.startswith("_"))
    return sorted(names)


def scalar_or_none(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return value.item()
        return f"tensor{tuple(value.shape)}"
    if isinstance(value, (int, float, str, bool)):
        return value
    return str(value)


def inspect_payload(path: Path) -> dict[str, Any]:
    started = time.perf_counter()
    blob = torch.load(str(path), weights_only=False, map_location="cpu")
    load_seconds = time.perf_counter() - started

    games = list(blob.get("results", []))
    decl_counts: Counter[int] = Counter()
    decision_count = 0
    sample_counts: Counter[int] = Counter()
    game_attrs: set[str] = set()
    decision_attrs: set[str] = set()
    bid_values: Counter[str] = Counter()
    has_oracle_softmax = False
    has_legal_per_seat = False
    has_voids_per_seat = False
    has_bid_value = False

    for game in games:
        game_attrs.update(attr_names(game))
        decl_id = scalar_or_none(getattr(game, "decl_id", None))
        if decl_id is not None:
            decl_counts[int(decl_id)] += 1
        game_bid = scalar_or_none(getattr(game, "bid_value", None))
        if game_bid is not None:
            has_bid_value = True
            bid_values[str(game_bid)] += 1
        for decision in getattr(game, "decisions", []):
            decision_count += 1
            decision_attrs.update(attr_names(decision))
            q_per_world = getattr(decision, "q_per_world", None)
            if q_per_world is not None:
                sample_counts[int(q_per_world.shape[0])] += 1
            dec_bid = scalar_or_none(getattr(decision, "bid_value", None))
            if dec_bid is not None:
                has_bid_value = True
                bid_values[str(dec_bid)] += 1
            has_oracle_softmax = has_oracle_softmax or getattr(decision, "oracle_softmax_per_seat", None) is not None
            has_legal_per_seat = has_legal_per_seat or getattr(decision, "legal_mask_per_seat", None) is not None
            has_voids_per_seat = has_voids_per_seat or getattr(decision, "voids_per_seat", None) is not None

    return {
        "inspected": True,
        "game_count": len(games),
        "decision_count": decision_count,
        "decl_counts_json": json.dumps(dict(sorted(decl_counts.items())), sort_keys=True),
        "sample_counts_json": json.dumps(dict(sorted(sample_counts.items())), sort_keys=True),
        "schema": str(blob.get("schema", "legacy")),
        "has_bid_value": has_bid_value,
        "bid_values_json": json.dumps(dict(sorted(bid_values.items())), sort_keys=True),
        "has_oracle_softmax_per_seat": has_oracle_softmax,
        "has_legal_mask_per_seat": has_legal_per_seat,
        "has_voids_per_seat": has_voids_per_seat,
        "blob_keys_json": json.dumps(sorted(blob.keys())),
        "game_attrs_json": json.dumps(sorted(game_attrs)),
        "decision_attrs_json": json.dumps(sorted(decision_attrs)),
        "load_seconds": round(load_seconds, 3),
    }


def infer_aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    family_counts = Counter(row["corpus_family"] for row in rows)
    bytes_by_family: Counter[str] = Counter()
    inferred_games_by_family: Counter[str] = Counter()
    inferred_decisions_by_family: Counter[str] = Counter()
    seed_ranges: dict[str, list[list[int]]] = defaultdict(list)
    decl_coverage_by_family: dict[str, Counter[str]] = defaultdict(Counter)

    for row in rows:
        family = row["corpus_family"]
        bytes_by_family[family] += int(row["bytes"])
        if row["start_seed"] != "" and row["end_seed"] != "":
            seed_ranges[family].append([int(row["start_seed"]), int(row["end_seed"])])
        if family == "legacy_seed_mod_decl":
            seed_count = int(row["seed_count"])
            inferred_games_by_family[family] += seed_count
            inferred_decisions_by_family[family] += seed_count * 28
            for decl_id in range(10):
                decl_coverage_by_family[family][DECL_NAMES[decl_id]] += seed_count // 10
        elif family == "v2_all_decl_fixed_bid30":
            seed_count = int(row["seed_count"])
            games = seed_count * 10
            inferred_games_by_family[family] += games
            inferred_decisions_by_family[family] += games * 28
            for decl_id in range(10):
                decl_coverage_by_family[family][DECL_NAMES[decl_id]] += seed_count

    return {
        "file_count": len(rows),
        "family_counts": dict(sorted(family_counts.items())),
        "bytes_by_family": dict(sorted(bytes_by_family.items())),
        "gib_by_family": {k: round(v / (1024**3), 3) for k, v in sorted(bytes_by_family.items())},
        "total_gib": round(sum(bytes_by_family.values()) / (1024**3), 3),
        "seed_ranges": dict(seed_ranges),
        "inferred_games_by_family": dict(sorted(inferred_games_by_family.items())),
        "inferred_decisions_by_family": dict(sorted(inferred_decisions_by_family.items())),
        "decl_coverage_by_family": {
            family: dict(sorted(counter.items())) for family, counter in sorted(decl_coverage_by_family.items())
        },
    }


def claim_family_routes() -> list[dict[str, str]]:
    return [
        {
            "bead_id": "t42-0b4l.4",
            "claim_family": "hidden threat / belief impact",
            "book_chapters": "cross-cutting W42; supports Ch04/Ch05/Ch07/Ch08/Ch09 diagnostics",
            "matrix_previous_route": "branch-atlas diagnostic amplifier",
            "legacy_111gb_support": "Strong: 10000 games / 280000 decisions inferred, adaptive high-M q_per_world/world_hands available in sampled legacy chunks.",
            "v2_support": "Useful but smaller: 1000 games / 28000 decisions, schema v2 extras, fixed bid 30.",
            "classification_after_inventory": "directly testable now as offline diagnostics",
            "directly_testable_now": "yes",
            "filtered_rows_possible": "yes",
            "still_requires_generation": "no for broad hidden-holder impact; yes for hand-picked rare regimes",
            "reason": "The old corpus has exactly the offline labels needed for hidden-domino impact: world_hands, q_per_world, e_q, legal_mask, and action_taken.",
            "next_data_action": "Run a streaming hidden-threat summary over legacy chunks by declaration/action family; avoid exporting full action rows.",
        },
        {
            "bead_id": "t42-0b4l.5",
            "claim_family": "bidding risk / bid-only-enough / bid margin",
            "book_chapters": "Ch02 Bidding; Ch12 advanced bidding; Ch16 four-trump policy",
            "matrix_previous_route": "needs auction/bid-margin generation",
            "legacy_111gb_support": "Weak: inspected legacy chunks have no bid_value and no auction/margin fields.",
            "v2_support": "Limited: fixed bid_value=30 exists, but no real auction pressure or bid margin.",
            "classification_after_inventory": "still requires generation for hard bidding claims",
            "directly_testable_now": "no for bid-only-enough; partial for static hand-risk buckets",
            "filtered_rows_possible": "yes for play-after-declaration risk diagnostics, not for auction advice",
            "still_requires_generation": "yes",
            "reason": "The book's bidding claims depend on current high bid, minimum winning bid, bid ceiling, and counterfactual bid values; the big corpus is seed/declaration play data, not auction data.",
            "next_data_action": "Generate auction-aware candidate bid rows with bid amount, current high bid, bid margin, score, and make/set or E[Q] labels.",
        },
        {
            "bead_id": "t42-0b4l.6",
            "claim_family": "seat/position and bidder sequencing",
            "book_chapters": "Ch03 Bidder Play; Ch04/Ch05 role interactions",
            "matrix_previous_route": "needs sequence counterfactuals or model probes",
            "legacy_111gb_support": "Moderate: many legal play decisions by declaration/seat, but no explicit sequence counterfactual rows.",
            "v2_support": "Moderate: paired all-declaration v2 rows plus schema extras on a smaller slice.",
            "classification_after_inventory": "testable with filtered corpus rows for descriptive/action-local labels; generation needed for sequence counterfactuals",
            "directly_testable_now": "partial",
            "filtered_rows_possible": "yes",
            "still_requires_generation": "yes for trump-first/off-first counterfactual claims",
            "reason": "Large play data can power role and action-class slices, but it does not replay the same state under alternate planned sequences.",
            "next_data_action": "Mine legacy rows for bidder action-class frequencies and regret buckets, then generate matched sequence counterfactuals for claims that survive the filter.",
        },
        {
            "bead_id": "t42-0b4l.7",
            "claim_family": "84 stopper / weapon preservation",
            "book_chapters": "Ch07 Taking Every Trick / 84; Ch08 Setting 84",
            "matrix_previous_route": "needs dynamic 84 generation",
            "legacy_111gb_support": "Limited: q_per_world/world_hands can mine generic stopper-preservation-like states, but inspected legacy chunks lack bid_value or an 84 contract flag.",
            "v2_support": "Limited: fixed bid_value=30 is not an 84 contract; schema extras help detectors but not true 84 scoring.",
            "classification_after_inventory": "filtered corpus rows can prototype labels; true 84 claims still require generation",
            "directly_testable_now": "no for true 84 contract claims; partial for label fixtures and generic preservation diagnostics",
            "filtered_rows_possible": "yes for action-state discovery and detector rehearsal",
            "still_requires_generation": "yes",
            "reason": "The book's 84 claims change the objective to winning every trick; the available corpora do not encode actual 84 contracts or 84-specific set/make semantics.",
            "next_data_action": "Use legacy corpus to find candidate final-weapon/public-state patterns, then generate bid_value=84 or 84-regime branch-atlas rows for direct tests.",
        },
        {
            "bead_id": "t42-0b4l.8",
            "claim_family": "doubles-as-trump / no-trump tactics",
            "book_chapters": "Ch09 Doubles As Trump and No Trump",
            "matrix_previous_route": "needs paired regime generation",
            "legacy_111gb_support": "Good for within-regime play: inferred 1000 games / 28000 decisions each for decl 7, decl 8, and decl 9. Not same-hand paired across regimes.",
            "v2_support": "Good for paired same-seed declaration slices: 100 seeds x all 10 declarations, fixed bid 30, lower M=200.",
            "classification_after_inventory": "directly testable now for within-regime tactical labels; paired regime-choice claims still require generation or v2-limited analysis",
            "directly_testable_now": "partial",
            "filtered_rows_possible": "yes",
            "still_requires_generation": "yes for powered same-hand no-trump vs doubles-trump declaration choice",
            "reason": "Seed-modulo legacy data gives power inside declarations 7/9, but not the same hand evaluated under both declarations. The v2 slice has same-seed all-decl structure but is much smaller and fixed-bid.",
            "next_data_action": "Run decl 7/9 legacy row mining for low-double sacrifice, support-double preservation, suit-count/walker labels; separately generate paired declaration rows for regime-choice claims.",
        },
        {
            "bead_id": "t42-0b4l.9",
            "claim_family": "claim-tag model probes",
            "book_chapters": "all claim families after detector hardening",
            "matrix_previous_route": "after direct labels exist",
            "legacy_111gb_support": "Strong as a future training/eval substrate if compact labels are generated on the fly or cached narrowly.",
            "v2_support": "Useful for schema-v2 head targets and smaller smoke runs.",
            "classification_after_inventory": "defer until direct detectors are sharpened",
            "directly_testable_now": "partial for smoke probes",
            "filtered_rows_possible": "yes",
            "still_requires_generation": "no for generic tag probes; yes for families blocked on missing fields",
            "reason": "The big corpus can feed model probes, but model work should follow detector definitions and family evidence boundaries.",
            "next_data_action": "Do not start broad model ablations until .5/.7/.8 route-specific labels are settled.",
        },
    ]


def build_inventory(args: argparse.Namespace) -> dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_pt = sorted(DATA_DIR.glob("*.pt"))
    legacy = sorted([p for p in all_pt if LEGACY_RE.match(p.name)], key=lambda p: int(LEGACY_RE.match(p.name).group(1)))  # type: ignore[union-attr]
    v2 = sorted([p for p in all_pt if V2_RE.match(p.name)], key=lambda p: int(V2_RE.match(p.name).group(1)))  # type: ignore[union-attr]
    eval_files = sorted([p for p in all_pt if EVAL_RE.match(p.name)])

    inspect_set: set[Path] = set()
    inspect_set.update(representative_legacy_files(legacy, args.representative_legacy_files))
    if args.full_legacy_scan:
        inspect_set.update(legacy)
        inspect_set.update(path for path in all_pt if LEGACY_SINGLE_RE.match(path.name))
    inspect_set.update(v2)
    inspect_set.update(eval_files)
    if args.max_inspected_files:
        inspect_set = set(sorted(inspect_set)[: args.max_inspected_files])

    rows: list[dict[str, Any]] = []
    inspection_errors: dict[str, str] = {}
    for path in all_pt:
        row = classify_file(path)
        row.update(
            {
                "inspected": False,
                "game_count": "",
                "decision_count": "",
                "decl_counts_json": "",
                "sample_counts_json": "",
                "schema": "",
                "has_bid_value": "",
                "bid_values_json": "",
                "has_oracle_softmax_per_seat": "",
                "has_legal_mask_per_seat": "",
                "has_voids_per_seat": "",
                "blob_keys_json": "",
                "game_attrs_json": "",
                "decision_attrs_json": "",
                "load_seconds": "",
            }
        )
        if path in inspect_set:
            try:
                row.update(inspect_payload(path))
            except Exception as exc:  # pragma: no cover - inventory should record partial failures.
                inspection_errors[rel(path)] = repr(exc)
        rows.append(row)

    routes = claim_family_routes()
    aggregate = infer_aggregate(rows)
    inspected_rows = [row for row in rows if row["inspected"]]
    observed_schema_fields = {
        "files_inspected": len(inspected_rows),
        "schemas": sorted({row["schema"] for row in inspected_rows if row["schema"]}),
        "has_bid_value_files": sum(1 for row in inspected_rows if row["has_bid_value"] is True),
        "has_oracle_softmax_per_seat_files": sum(1 for row in inspected_rows if row["has_oracle_softmax_per_seat"] is True),
        "has_legal_mask_per_seat_files": sum(1 for row in inspected_rows if row["has_legal_mask_per_seat"] is True),
        "has_voids_per_seat_files": sum(1 for row in inspected_rows if row["has_voids_per_seat"] is True),
        "sample_counts": sorted(
            {
                int(k)
                for row in inspected_rows
                if row["sample_counts_json"]
                for k in json.loads(row["sample_counts_json"]).keys()
            }
        ),
        "game_attrs": sorted(
            {
                attr
                for row in inspected_rows
                if row["game_attrs_json"]
                for attr in json.loads(row["game_attrs_json"])
            }
        ),
        "decision_attrs": sorted(
            {
                attr
                for row in inspected_rows
                if row["decision_attrs_json"]
                for attr in json.loads(row["decision_attrs_json"])
            }
        ),
    }

    summary = {
        "schema_version": "w42.claim_data_inventory.v1",
        "bead_id": "t42-0b4l.11",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repo_commit": git_sha(),
        "data_dir": rel(DATA_DIR),
        "inventory_mode": "full_legacy_scan" if args.full_legacy_scan else "representative_legacy_scan",
        "aggregate": aggregate,
        "observed_schema_fields": observed_schema_fields,
        "inspection_errors": inspection_errors,
        "interpretation": {
            "legacy_111gb": "Large seed-modulo declaration corpus. Strong for broad play/action/belief diagnostics; not paired same-hand declaration evidence; no observed real bid margin.",
            "v2_fixed_bid30": "Small all-declaration paired-seed corpus. Useful for replication and paired declaration smoke; fixed bid_value=30 blocks true bid-margin and 84-contract claims.",
            "claim_ledger_impact": "No claim status changes. This inventory reroutes evidence plans and marks field availability.",
            "leakage_boundary": "world_hands, q_per_world, E[Q], hidden holders, and future outcomes remain offline labels/diagnostics only.",
        },
        "outputs": {
            "corpus_files": "w42/claim_data_inventory/corpus_files.csv",
            "claim_family_routes": "w42/claim_data_inventory/claim_family_routes.csv",
            "summary": "w42/claim_data_inventory/summary.json",
            "manifest": "w42/claim_data_inventory/manifest.json",
        },
    }
    manifest = {
        "schema_version": "w42.artifact_manifest.v1",
        "bead_id": "t42-0b4l.11",
        "artifact": "w42 claim data inventory",
        "created_at_utc": summary["created_at_utc"],
        "repo_commit": summary["repo_commit"],
        "command": " ".join(sys.argv),
        "source_corpora": [rel(path) for path in all_pt],
        "large_binary_policy": "Source .pt files are not copied into w42 artifacts; this directory contains compact CSV/JSON summaries only.",
        "claim_ledger_impact": summary["interpretation"]["claim_ledger_impact"],
        "leakage_exclusions": [
            "No hidden-world truth is promoted to a live feature.",
            "No claim status changes are made by this inventory.",
            "No full action-row dump is emitted.",
        ],
    }

    write_csv(OUT_DIR / "corpus_files.csv", rows, CSV_FILES_FIELDS)
    write_csv(OUT_DIR / "claim_family_routes.csv", routes, ROUTE_FIELDS)
    write_json(OUT_DIR / "summary.json", summary)
    write_json(OUT_DIR / "manifest.json", manifest)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--representative-legacy-files", type=int, default=5)
    parser.add_argument("--full-legacy-scan", action="store_true")
    parser.add_argument("--max-inspected-files", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    summary = build_inventory(parse_args())
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
