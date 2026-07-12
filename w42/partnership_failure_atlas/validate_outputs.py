#!/usr/bin/env python3
"""Validate generated partnership-failure-atlas artifacts."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any, Iterator


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SCHEMA_VERSION = "w42.partnership_failure_atlas.v2"

try:
    from .run_atlas import source_metadata, threshold_q
except ImportError:
    from run_atlas import source_metadata, threshold_q

REQUIRED_STATUS_FIELDS = [
    "uncertainty_status",
    "world_q_vector_status",
    "sampled_world_count_status",
    "world_sampler_status",
    "world_sampler_version_status",
    "world_weighting_status",
    "distribution_calibration_status",
    "gus_drama_join_status",
    "role_order_status",
    "partner_coordination_status",
    "partner_identity_status",
    "partner_assignment_status",
    "action_observation_status",
    "action_likelihood_status",
    "action_posterior_status",
    "plan_persistence_status",
    "distributional_utility_status",
    "auction_contract_status",
    "auction_history_status",
    "match_score_status",
    "source_trajectory_policy_status",
    "champion_action_status",
    "threshold_q_status",
    "book_detector_status",
    "threshold_utility_top_action_proxy_status",
    "dist_lens_top_action_status",
]

MISSING_VALUE_CONTRACTS = {
    "world_q_vector": "world_q_vector_status",
    "sampled_world_count": "sampled_world_count_status",
    "world_sampler": "world_sampler_status",
    "world_sampler_version": "world_sampler_version_status",
    "world_weighting": "world_weighting_status",
    "distribution_calibration": "distribution_calibration_status",
    "gus_outcome_variance": "gus_drama_join_status",
    "gus_action_fragility": "gus_drama_join_status",
    "gus_belief_sharpness": "gus_drama_join_status",
    "partner_identity": "partner_identity_status",
    "partner_assignment_condition": "partner_assignment_status",
    "action_likelihood": "action_likelihood_status",
    "action_posterior_delta": "action_posterior_status",
    "plan_id": "plan_persistence_status",
    "plan_step": "plan_persistence_status",
    "auction_history": "auction_history_status",
    "match_score_our": "match_score_status",
    "match_score_their": "match_score_status",
    "source_trajectory_policy": "source_trajectory_policy_status",
    "champion_action_candidate": "champion_action_status",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=HERE)
    parser.add_argument("--skip-hashes", action="store_true")
    return parser.parse_args()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_atlas(path: Path) -> Iterator[dict[str, str]]:
    with gzip.open(path, "rt", newline="", encoding="utf-8") as handle:
        yield from csv.DictReader(handle)


def validate(output_dir: Path, *, check_hashes: bool = True) -> dict[str, Any]:
    manifest_path = output_dir / "manifest.json"
    summary_path = output_dir / "summary.json"
    atlas_path = output_dir / "atlas_full.csv.gz"
    sample_path = output_dir / "atlas_sample.csv"
    inventory_path = output_dir / "evidence_inventory.csv"
    for path in [manifest_path, summary_path, atlas_path, sample_path, inventory_path]:
        if not path.is_file():
            raise AssertionError(f"missing artifact: {path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    full_build = int(manifest.get("build_options", {}).get("max_rows", 0)) == 0
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise AssertionError("manifest schema version mismatch")
    if summary.get("schema_version") != SCHEMA_VERSION:
        raise AssertionError("summary schema version mismatch")

    if check_hashes:
        for name, metadata in manifest["artifacts"].items():
            path = output_dir / name
            if path.stat().st_size != metadata["bytes"]:
                raise AssertionError(f"artifact byte count mismatch: {name}")
            if sha256_path(path) != metadata["sha256"]:
                raise AssertionError(f"artifact hash mismatch: {name}")
        for source_name, metadata in manifest["inputs"].items():
            path = ROOT / metadata["path"]
            if not path.is_file():
                raise AssertionError(f"manifest input is missing: {source_name}: {path}")
            if sha256_path(path) != metadata["sha256"]:
                raise AssertionError(f"manifest input hash mismatch: {source_name}")

    row_ids: set[str] = set()
    action_ids: set[tuple[str, str]] = set()
    decisions: set[str] = set()
    actual_count = 0
    book_endorsed_count = 0
    book_anti_count = 0
    threshold_proxy_count = 0
    safest_tail_count = 0
    dist_lens_count = 0
    dist_lens_decisions: set[str] = set()
    bid_values: set[int] = set()
    atlas_fields: list[str] | None = None
    for row in read_atlas(atlas_path):
        if atlas_fields is None:
            atlas_fields = list(row)
        if row["atlas_schema_version"] != SCHEMA_VERSION:
            raise AssertionError("atlas row schema version mismatch")
        row_id = row["atlas_row_id"]
        action_id = (row["decision_key"], row["candidate_domino"])
        if row_id in row_ids or action_id in action_ids:
            raise AssertionError(f"duplicate atlas action: {action_id}")
        row_ids.add(row_id)
        action_ids.add(action_id)
        decisions.add(row["decision_key"])
        if row["is_actual_action"] == "true":
            actual_count += 1
        if row["book_detector_endorsed"] == "true":
            book_endorsed_count += 1
        if row["book_detector_anti_endorsed"] == "true":
            book_anti_count += 1
        if row["threshold_utility_top_action_proxy"] == "true":
            threshold_proxy_count += 1
        if row["is_safest_tail"] == "true":
            safest_tail_count += 1
        if row["dist_lens_top_action_confounded"]:
            dist_lens_count += 1
            dist_lens_decisions.add(row["decision_key"])
            if row["dist_lens_top_action_status"] != "confounded:cross-corpus-positional-join":
                raise AssertionError("nonblank dist-lens values must remain confounded")
        elif not row["dist_lens_top_action_status"].startswith("unavailable:"):
            raise AssertionError("blank dist-lens values need an unavailable status")
        if row["bid_value"]:
            bid_values.add(int(float(row["bid_value"])))
        for field in REQUIRED_STATUS_FIELDS:
            if not row.get(field):
                raise AssertionError(f"missing status {field} for {action_id}")
        for value_field, status_field in MISSING_VALUE_CONTRACTS.items():
            if row.get(value_field):
                raise AssertionError(f"{value_field} must stay blank while {status_field} says unavailable")
        for join_field in [
            "join_joined_claim_status",
            "join_sequence_status",
            "join_seat_position_status",
            "join_handshape_status",
            "join_cross_ai_source_picks_status",
        ]:
            if row[join_field] != "available:exact-action-key":
                raise AssertionError(f"non-exact compatible join in {join_field}")
        if full_build and row["handshape_legal_action_n"] != row["legal_action_count"]:
            raise AssertionError(f"handshape legal count mismatch for {action_id}")
        if row["threshold_utility_top_action_proxy"] != row["is_best_threshold"]:
            raise AssertionError(f"threshold proxy differs from is_best_threshold for {action_id}")
        if row["cross_ai_ev_top_action"] != row["is_best_mean"]:
            raise AssertionError(f"source-picks EV flag differs from is_best_mean for {action_id}")
        if row["threshold_utility_top_action_proxy_status"] != (
            "proxy-only:is-best-threshold-not-actual-gus-output"
        ):
            raise AssertionError("threshold top-action proxy lost its provenance qualifier")
        if row["uncertainty_status"] != "partial:q-outcome-summaries-without-worlds-weights-or-calibration":
            raise AssertionError("uncertainty and distributional-utility status were conflated")
        if row["distributional_utility_status"] != (
            "proxy-only:fixed-threshold-and-tail-rankings-without-transform-consumer-or-context"
        ):
            raise AssertionError("distributional-utility proxy status mismatch")
        if row["lower_tail_q_cutoff"] != "-18" or row["lower_tail_mass_semantics"] != "P(Q<=-18)":
            raise AssertionError("lower-tail semantics mismatch")
        if not row["threshold_q"] or "role-and-bid-dependent" not in row["threshold_mass_semantics"]:
            raise AssertionError("threshold semantics are missing")
        expected_threshold = threshold_q(row["team"], row["bid_value"])
        if expected_threshold is None or row["threshold_q"] != format(expected_threshold, ".12g"):
            raise AssertionError(f"role/bid threshold mismatch for {action_id}")
        if row["auction_contract_status"] != (
            "fixed-bid30:declaration-and-risk-proxies-only; no auction choices/history"
        ):
            raise AssertionError("bidding surface must remain fixed-bid30")

    coverage = summary["coverage"]
    if len(row_ids) != coverage["action_rows"]:
        raise AssertionError("summary action row count mismatch")
    if len(decisions) != coverage["decision_rows"]:
        raise AssertionError("summary decision row count mismatch")
    if actual_count != coverage["actual_action_rows"]:
        raise AssertionError("summary actual-action row count mismatch")
    expected_counts = {
        "book_detector_endorsed_action_rows": book_endorsed_count,
        "book_detector_anti_endorsed_action_rows": book_anti_count,
        "threshold_utility_top_action_proxy_rows": threshold_proxy_count,
        "safest_tail_action_rows": safest_tail_count,
        "dist_lens_confounded_action_rows": dist_lens_count,
        "dist_lens_confounded_decisions": len(dist_lens_decisions),
    }
    for field, actual in expected_counts.items():
        if coverage[field] != actual:
            raise AssertionError(f"summary coverage mismatch: {field}")
    expected_bidding = {
        "bid_values": sorted(bid_values),
        "bid_value_cardinality": len(bid_values),
        "status": "fixed-bid30:cannot-estimate-bidding-or-bid-level-effects",
    }
    if summary.get("bidding_surface") != expected_bidding:
        raise AssertionError("summary bidding surface mismatch")
    if manifest.get("bidding_surface") != expected_bidding:
        raise AssertionError("manifest bidding surface mismatch")
    if expected_bidding["bid_values"] != [30]:
        raise AssertionError("compatible corpus is expected to be fixed at bid=30")
    if atlas_fields != manifest["atlas_fields"]:
        raise AssertionError("atlas header differs from manifest field order")

    with sample_path.open(newline="", encoding="utf-8") as handle:
        sample_rows = list(csv.DictReader(handle))
    if len(sample_rows) != coverage["sample_action_rows"]:
        raise AssertionError("sample row count mismatch")
    if not {row["atlas_row_id"] for row in sample_rows}.issubset(row_ids):
        raise AssertionError("sample contains a row outside the full atlas")

    with inventory_path.open(newline="", encoding="utf-8") as handle:
        inventory = list(csv.DictReader(handle))
    statuses = {row["atlas_join_status"].split(":", 1)[0] for row in inventory}
    if not {"available", "unjoinable"}.issubset(statuses):
        raise AssertionError("evidence inventory must contain both joined and unjoinable sources")
    if any(not row["join_reason"] for row in inventory):
        raise AssertionError("every evidence source needs an explicit join reason")
    if any(not row["scientific_status"] or not row["confound_status"] for row in inventory):
        raise AssertionError("every evidence source needs scientific and confound status")
    inventory_by_path = {row["path"]: row for row in inventory}
    required_gus_surfaces = {
        "gus/analysis/drama_atlas.parquet",
        "gus/analysis/drama_atlas_v2.parquet",
    }
    if not required_gus_surfaces.issubset(inventory_by_path):
        raise AssertionError("full Gus parquet surfaces are missing from evidence inventory")
    for source_path, row in inventory_by_path.items():
        path = ROOT / source_path
        if not path.is_file():
            raise AssertionError(f"inventoried source is missing: {source_path}")
        if check_hashes and sha256_path(path) != row["sha256"]:
            raise AssertionError(f"evidence inventory hash mismatch: {source_path}")
        live_count, live_count_status, live_fields = source_metadata(path)
        if live_count_status.startswith("available:"):
            if row["row_count"] != live_count:
                raise AssertionError(f"evidence inventory row count mismatch: {source_path}")
            if row["schema_fields"] != "|".join(live_fields):
                raise AssertionError(f"evidence inventory schema mismatch: {source_path}")
        if source_path in required_gus_surfaces:
            if row["row_count"] != "280560" or not row["atlas_join_status"].startswith("unjoinable:"):
                raise AssertionError(f"Gus parquet inventory boundary mismatch: {source_path}")
    cross_ai = inventory_by_path[
        "w42/book_validation_v1/wave1/t42-m2i7_cross_ai_agreement/per_action_source_picks.csv"
    ]
    if "confounded" not in cross_ai["scientific_status"] or "cross-corpus" not in cross_ai["confound_status"]:
        raise AssertionError("cross-AI prior attempt lost its confound status")

    return {
        "action_rows": len(row_ids),
        "decision_rows": len(decisions),
        "sample_rows": len(sample_rows),
        "inventory_rows": len(inventory),
        "hashes_checked": check_hashes,
        "live_sources_checked": len(inventory),
    }


def main() -> None:
    args = parse_args()
    result = validate(args.output_dir.resolve(), check_hashes=not args.skip_hashes)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
