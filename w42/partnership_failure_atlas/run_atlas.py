#!/usr/bin/env python3
"""Build an architecture-neutral partnership failure atlas from retained evidence.

Five W42 action tables share an exact (decision key, candidate domino)
identity. Gus drama tables, arena records, and champion A/B summaries do not.
This runner joins only the exact action rows and records every other source in
an evidence inventory with an explicit non-join reason. Fields whose original
names overstate their provenance are renamed and qualified rather than dropped.

No model, GPU, or project package import is required.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import heapq
import io
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator, TextIO


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent

SCHEMA_VERSION = "w42.partnership_failure_atlas.v2"
DEFAULT_JOINED = ROOT / "w42/joined_claim_row_model_table/joined_claim_action_rows.csv"
DEFAULT_SEQUENCE = ROOT / "w42/sequence_seat_counterfactuals/labeled_sequence_action_rows.csv"
DEFAULT_SEAT = ROOT / "w42/seat_position_claim_tests/labeled_action_rows.csv"
DEFAULT_SOURCE_PICKS = (
    ROOT / "w42/book_validation_v1/wave1/t42-m2i7_cross_ai_agreement/per_action_source_picks.csv"
)
DEFAULT_HANDSHAPE = ROOT / "w42/phase4_sequence_handshape_tests/labeled_handshape_action_rows.csv"

STATUS_AVAILABLE = "available:exact-action-key"
STATUS_PROXY_PARTNER = "proxy-only:public-role-and-action-local-labels"
STATUS_UNAVAILABLE_SOURCE = "unavailable:not-present-in-retained-action-tables"
STATUS_UNJOINABLE_GUS = "unjoinable:gus-corpus-state-namespace"
STATUS_UNJOINABLE_MATCH = "unjoinable:arena-or-champion-experiment-namespace"
STATUS_THRESHOLD_PROXY = "proxy-only:is-best-threshold-not-actual-gus-output"
STATUS_DIST_CONFOUNDED = "confounded:cross-corpus-positional-join"

ATLAS_FIELDS = [
    "atlas_schema_version",
    "atlas_row_id",
    "decision_key",
    "candidate_domino",
    "source_file",
    "source_seed",
    "source_game_idx",
    "source_decision_idx",
    "decl_name",
    "bid_value",
    "actor",
    "seat_role",
    "role_family",
    "team",
    "trick_idx",
    "trick_position",
    "position_family",
    "phase",
    "candidate_count_points",
    "candidate_is_called_suit",
    "candidate_is_double",
    "current_winner_team_before",
    "current_trick_count_before",
    "candidate_beats_current",
    "candidate_would_win_trick_now",
    "is_actual_action",
    "is_best_mean",
    "oracle_mean_q",
    "oracle_mean_regret",
    "oracle_threshold_mass",
    "oracle_lower_tail_mass",
    "threshold_q",
    "threshold_q_status",
    "threshold_mass_semantics",
    "lower_tail_q_cutoff",
    "lower_tail_mass_semantics",
    "is_best_threshold",
    "is_safest_tail",
    "threshold_gap",
    "handshape_legal_action_n",
    "called_suit_legal_n",
    "off_suit_legal_n",
    "count_legal_n",
    "double_legal_n",
    "beater_legal_n",
    "handshape_labels",
    "cross_ai_ev_top_action",
    "book_detector_endorsed",
    "book_detector_anti_endorsed",
    "book_detector_labels",
    "book_detector_status",
    "threshold_utility_top_action_proxy",
    "threshold_utility_top_action_proxy_status",
    "dist_lens_top_action_confounded",
    "dist_lens_top_action_status",
    "legal_action_count",
    "decision_mean_q_min",
    "decision_mean_q_max",
    "decision_mean_q_span",
    "decision_best_second_gap",
    "decision_actual_action_count",
    "actual_action_candidate",
    "actual_action_mean_regret",
    "actual_action_is_best_mean",
    "source_trajectory_policy",
    "source_trajectory_policy_status",
    "champion_action_candidate",
    "champion_action_status",
    "join_joined_claim_status",
    "join_sequence_status",
    "join_seat_position_status",
    "join_handshape_status",
    "join_cross_ai_source_picks_status",
    "uncertainty_status",
    "world_q_vector",
    "world_q_vector_status",
    "sampled_world_count",
    "sampled_world_count_status",
    "world_sampler",
    "world_sampler_status",
    "world_sampler_version",
    "world_sampler_version_status",
    "world_weighting",
    "world_weighting_status",
    "distribution_calibration",
    "distribution_calibration_status",
    "gus_outcome_variance",
    "gus_action_fragility",
    "gus_belief_sharpness",
    "gus_drama_join_status",
    "role_order_status",
    "role_order_labels",
    "partner_coordination_proxy_labels",
    "partner_coordination_status",
    "partner_identity",
    "partner_identity_status",
    "partner_assignment_condition",
    "partner_assignment_status",
    "action_observation_status",
    "action_likelihood",
    "action_likelihood_status",
    "action_posterior_delta",
    "action_posterior_status",
    "plan_proxy_labels",
    "plan_id",
    "plan_step",
    "plan_persistence_status",
    "distributional_utility_status",
    "bidding_risk_labels",
    "auction_contract_status",
    "auction_history",
    "auction_history_status",
    "match_score_our",
    "match_score_their",
    "match_score_status",
    "doubles_no_trump_labels",
    "eighty_four_public_labels",
    "hidden_public_proxy_labels",
    "model_feature_labels",
    "sequence_labels",
    "seat_position_labels",
    "eval_only_labels",
    "leakage_boundary",
]

SOURCE_INVENTORY_FIELDS = [
    "source_id",
    "path",
    "granularity",
    "row_count",
    "key_fields",
    "atlas_join_status",
    "join_reason",
    "dimensions_available",
    "scientific_status",
    "confound_status",
    "row_count_status",
    "schema_fields",
    "sha256",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--joined", type=Path, default=DEFAULT_JOINED)
    parser.add_argument("--sequence", type=Path, default=DEFAULT_SEQUENCE)
    parser.add_argument("--seat", type=Path, default=DEFAULT_SEAT)
    parser.add_argument("--source-picks", type=Path, default=DEFAULT_SOURCE_PICKS)
    parser.add_argument("--handshape", type=Path, default=DEFAULT_HANDSHAPE)
    parser.add_argument("--output-dir", type=Path, default=HERE)
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="Limit joined action rows for a smoke build; 0 means all rows.",
    )
    parser.add_argument("--sample-rows", type=int, default=256)
    return parser.parse_args()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_row_id(decision_key: str, candidate_domino: str) -> str:
    return hashlib.sha256(f"{decision_key}\x1f{candidate_domino}".encode()).hexdigest()[:24]


def split_labels(value: Any) -> tuple[str, ...]:
    return tuple(sorted({part.strip() for part in str(value or "").split("|") if part.strip()}))


def labels_with_prefix(labels: Iterable[str], prefixes: tuple[str, ...]) -> str:
    return "|".join(label for label in labels if label.startswith(prefixes))


def normalized_bool(value: Any) -> str:
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return "true"
    if text in {"0", "false", "no", "n"}:
        return "false"
    return ""


def finite_float(value: Any) -> float | None:
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def display_number(value: float | None) -> str:
    if value is None:
        return ""
    return format(value, ".12g")


def threshold_q(team: str, bid_value: Any) -> float | None:
    """Reproduce the role/bid threshold used by the retained W42 PDF labels."""

    bid = finite_float(bid_value)
    if bid is None or team not in {"offense", "defense"}:
        return None
    contract_points = 42.0 if int(bid) == 84 else bid
    if team == "offense":
        return 2.0 * contract_points - 42.0
    return 43.0 - 2.0 * contract_points


def load_action_index(path: Path, fields: list[str]) -> tuple[dict[tuple[str, str], dict[str, str]], list[str]]:
    index: dict[tuple[str, str], dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        header = list(reader.fieldnames or [])
        required = {"key", "candidate_domino", *fields}
        missing = sorted(required - set(header))
        if missing:
            raise ValueError(f"{path}: missing columns {missing}")
        for row in reader:
            key = (row["key"], row["candidate_domino"])
            if key in index:
                raise ValueError(f"{path}: duplicate action identity {key}")
            index[key] = {field: row.get(field, "") for field in fields}
    return index, header


def read_joined_groups(path: Path, max_rows: int) -> Iterator[list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {
            "key",
            "candidate_domino",
            "mean",
            "mean_regret",
            "threshold_mass",
            "lower_tail_mass",
            "feature_labels",
            "is_actual_action",
            "is_best_mean",
        }
        missing = sorted(required - set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"{path}: missing columns {missing}")
        group: list[dict[str, str]] = []
        current_key: str | None = None
        read_n = 0
        for row in reader:
            if max_rows and read_n >= max_rows:
                break
            read_n += 1
            decision_key = row["key"]
            if current_key is not None and decision_key != current_key:
                yield group
                group = []
            current_key = decision_key
            group.append(row)
        if group:
            yield group


def decision_summary(group: list[dict[str, str]]) -> dict[str, str]:
    means = [finite_float(row.get("mean")) for row in group]
    finite_means = sorted((value for value in means if value is not None), reverse=True)
    best = finite_means[0] if finite_means else None
    second = finite_means[1] if len(finite_means) > 1 else best
    actual = [row for row in group if normalized_bool(row.get("is_actual_action")) == "true"]
    actual_row = actual[0] if len(actual) == 1 else None
    minimum = min(finite_means) if finite_means else None
    maximum = max(finite_means) if finite_means else None
    return {
        "legal_action_count": str(len(group)),
        "decision_mean_q_min": display_number(minimum),
        "decision_mean_q_max": display_number(maximum),
        "decision_mean_q_span": display_number(None if minimum is None or maximum is None else maximum - minimum),
        "decision_best_second_gap": display_number(None if best is None or second is None else best - second),
        "decision_actual_action_count": str(len(actual)),
        "actual_action_candidate": actual_row.get("candidate_domino", "") if actual_row else "",
        "actual_action_mean_regret": actual_row.get("mean_regret", "") if actual_row else "",
        "actual_action_is_best_mean": normalized_bool(actual_row.get("is_best_mean")) if actual_row else "",
    }


def enrich_action(
    base: dict[str, str],
    sequence: dict[str, str],
    seat: dict[str, str],
    handshape: dict[str, str],
    source_picks: dict[str, str],
    decision: dict[str, str],
) -> dict[str, str]:
    feature_labels = split_labels(base.get("feature_labels"))
    sequence_labels = split_labels(sequence.get("derived_labels"))
    seat_labels = split_labels(seat.get("derived_labels"))
    handshape_labels = split_labels(handshape.get("labels"))
    book_labels = split_labels(source_picks.get("labels"))
    partner_labels = labels_with_prefix(
        tuple(sorted(set(feature_labels) | set(handshape_labels))),
        ("partner_", "ch04_partner_", "follower_partner_"),
    )
    role_labels = labels_with_prefix(
        feature_labels,
        ("role_", "seat_pos_", "sequence_", "phase_", "closure_", "follower_", "setter_"),
    )
    plan_labels = "|".join(
        label
        for label in tuple(sorted(set(feature_labels) | set(handshape_labels)))
        if label in {"bidder_lead_plan", "ch03_bidder_lead_plan"}
    )
    bidrisk_labels = labels_with_prefix(feature_labels, ("bidrisk_",))
    threshold = threshold_q(base.get("team", ""), base.get("bid_value"))
    dist_lens_value = normalized_bool(source_picks.get("dist_lens_top_action"))
    dist_lens_status = STATUS_DIST_CONFOUNDED if dist_lens_value else "unavailable:no-positional-cross-corpus-match"
    full_row = {
        "atlas_schema_version": SCHEMA_VERSION,
        "atlas_row_id": stable_row_id(base["key"], base["candidate_domino"]),
        "decision_key": base["key"],
        "candidate_domino": base["candidate_domino"],
        "source_file": base.get("source_file", ""),
        "source_seed": base.get("seed", ""),
        "source_game_idx": base.get("game_idx", ""),
        "source_decision_idx": base.get("decision_idx", ""),
        "decl_name": base.get("decl_name", ""),
        "bid_value": base.get("bid_value", ""),
        "actor": base.get("actor", ""),
        "seat_role": base.get("seat_role", ""),
        "role_family": base.get("role_family", ""),
        "team": base.get("team", ""),
        "trick_idx": base.get("trick_idx", ""),
        "trick_position": base.get("trick_position", ""),
        "position_family": sequence.get("position_family", ""),
        "phase": sequence.get("phase", ""),
        "candidate_count_points": base.get("candidate_count_points", ""),
        "candidate_is_called_suit": normalized_bool(base.get("candidate_is_called_suit")),
        "candidate_is_double": normalized_bool(base.get("candidate_is_double")),
        "current_winner_team_before": base.get("current_winner_team_before", ""),
        "current_trick_count_before": handshape.get("current_trick_count_before", ""),
        "candidate_beats_current": normalized_bool(sequence.get("candidate_beats_current")),
        "candidate_would_win_trick_now": normalized_bool(sequence.get("candidate_would_win_trick_now")),
        "is_actual_action": normalized_bool(base.get("is_actual_action")),
        "is_best_mean": normalized_bool(base.get("is_best_mean")),
        "oracle_mean_q": base.get("mean", ""),
        "oracle_mean_regret": base.get("mean_regret", ""),
        "oracle_threshold_mass": base.get("threshold_mass", ""),
        "oracle_lower_tail_mass": base.get("lower_tail_mass", ""),
        "threshold_q": display_number(threshold),
        "threshold_q_status": "available:derived-from-role-and-bid-using-retained-w42-semantics",
        "threshold_mass_semantics": "P(Q>=threshold_q); threshold_q is role-and-bid-dependent",
        "lower_tail_q_cutoff": "-18",
        "lower_tail_mass_semantics": "P(Q<=-18)",
        "is_best_threshold": normalized_bool(handshape.get("is_best_threshold")),
        "is_safest_tail": normalized_bool(handshape.get("is_safest_tail")),
        "threshold_gap": handshape.get("threshold_gap", ""),
        "handshape_legal_action_n": handshape.get("legal_action_n", ""),
        "called_suit_legal_n": handshape.get("called_suit_legal_n", ""),
        "off_suit_legal_n": handshape.get("off_suit_legal_n", ""),
        "count_legal_n": handshape.get("count_legal_n", ""),
        "double_legal_n": handshape.get("double_legal_n", ""),
        "beater_legal_n": handshape.get("beater_legal_n", ""),
        "handshape_labels": "|".join(handshape_labels),
        "cross_ai_ev_top_action": normalized_bool(source_picks.get("ev_top_action")),
        "book_detector_endorsed": normalized_bool(source_picks.get("detector_endorsed")),
        "book_detector_anti_endorsed": normalized_bool(source_picks.get("detector_anti_endorsed")),
        "book_detector_labels": "|".join(book_labels),
        "book_detector_status": "available:phase4-positive-and-negative-detector-sets",
        "threshold_utility_top_action_proxy": normalized_bool(source_picks.get("gus_top_action")),
        "threshold_utility_top_action_proxy_status": STATUS_THRESHOLD_PROXY,
        "dist_lens_top_action_confounded": dist_lens_value,
        "dist_lens_top_action_status": dist_lens_status,
        **decision,
        "source_trajectory_policy": "",
        "source_trajectory_policy_status": "unavailable:policy-not-fingerprinted-in-retained-action-tables",
        "champion_action_candidate": "",
        "champion_action_status": "unjoinable:no-shared-champion-action-state-fingerprint",
        "join_joined_claim_status": STATUS_AVAILABLE,
        "join_sequence_status": STATUS_AVAILABLE,
        "join_seat_position_status": STATUS_AVAILABLE,
        "join_handshape_status": STATUS_AVAILABLE,
        "join_cross_ai_source_picks_status": STATUS_AVAILABLE,
        "uncertainty_status": "partial:q-outcome-summaries-without-worlds-weights-or-calibration",
        "world_q_vector": "",
        "world_q_vector_status": STATUS_UNAVAILABLE_SOURCE,
        "sampled_world_count": "",
        "sampled_world_count_status": "unavailable:not-retained-with-compatible-action-rows",
        "world_sampler": "",
        "world_sampler_status": "unavailable:not-retained-with-compatible-action-rows",
        "world_sampler_version": "",
        "world_sampler_version_status": "unavailable:not-retained-with-compatible-action-rows",
        "world_weighting": "",
        "world_weighting_status": "unavailable:not-retained-with-compatible-action-rows",
        "distribution_calibration": "",
        "distribution_calibration_status": "unavailable:not-retained-with-compatible-action-rows",
        "gus_outcome_variance": "",
        "gus_action_fragility": "",
        "gus_belief_sharpness": "",
        "gus_drama_join_status": STATUS_UNJOINABLE_GUS,
        "role_order_status": "available:public-role-seat-order-and-phase",
        "role_order_labels": role_labels,
        "partner_coordination_proxy_labels": partner_labels,
        "partner_coordination_status": STATUS_PROXY_PARTNER,
        "partner_identity": "",
        "partner_identity_status": STATUS_UNAVAILABLE_SOURCE,
        "partner_assignment_condition": "",
        "partner_assignment_status": STATUS_UNAVAILABLE_SOURCE,
        "action_observation_status": "available:actual-action-indicator-without-policy-probability",
        "action_likelihood": "",
        "action_likelihood_status": STATUS_UNAVAILABLE_SOURCE,
        "action_posterior_delta": "",
        "action_posterior_status": STATUS_UNAVAILABLE_SOURCE,
        "plan_proxy_labels": plan_labels,
        "plan_id": "",
        "plan_step": "",
        "plan_persistence_status": "unavailable:no-cross-decision-plan-state-or-intervention",
        "distributional_utility_status": (
            "proxy-only:fixed-threshold-and-tail-rankings-without-transform-consumer-or-context"
        ),
        "bidding_risk_labels": bidrisk_labels,
        "auction_contract_status": "fixed-bid30:declaration-and-risk-proxies-only; no auction choices/history",
        "auction_history": "",
        "auction_history_status": STATUS_UNAVAILABLE_SOURCE,
        "match_score_our": "",
        "match_score_their": "",
        "match_score_status": STATUS_UNAVAILABLE_SOURCE,
        "doubles_no_trump_labels": labels_with_prefix(feature_labels, ("dnt_",)),
        "eighty_four_public_labels": labels_with_prefix(feature_labels, ("e84_public_",)),
        "hidden_public_proxy_labels": labels_with_prefix(feature_labels, ("hidden_proxy_",)),
        "model_feature_labels": "|".join(feature_labels),
        "sequence_labels": "|".join(sequence_labels),
        "seat_position_labels": "|".join(seat_labels),
        "eval_only_labels": "|".join(split_labels(base.get("eval_only_labels"))),
        "leakage_boundary": (
            "oracle/distribution columns are offline labels; role/order and proxy labels are public/action-local; "
            "eval_only_labels are never asserted as runtime features"
        ),
    }
    return {field: full_row.get(field, "") for field in ATLAS_FIELDS}


class DeterministicSample:
    """Keep rows with the lexicographically smallest stable row hashes."""

    def __init__(self, limit: int):
        self.limit = max(0, limit)
        self._heap: list[tuple[int, dict[str, str]]] = []

    def add(self, row: dict[str, str]) -> None:
        if not self.limit:
            return
        rank = int(hashlib.sha256(row["atlas_row_id"].encode()).hexdigest(), 16)
        item = (-rank, row)
        if len(self._heap) < self.limit:
            heapq.heappush(self._heap, item)
        elif rank < -self._heap[0][0]:
            heapq.heapreplace(self._heap, item)

    def rows(self) -> list[dict[str, str]]:
        return sorted((row for _, row in self._heap), key=lambda row: row["atlas_row_id"])


def open_deterministic_gzip_text(path: Path) -> tuple[TextIO, Any]:
    raw = path.open("wb")
    gz = gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0)
    text = io.TextIOWrapper(gz, encoding="utf-8", newline="")
    return text, raw


def source_metadata(path: Path) -> tuple[str, str, list[str]]:
    """Return row count, count provenance, and schema without reading data columns."""

    if path.suffix == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            return str(sum(1 for _ in reader)), "available:csv-row-scan", list(reader.fieldnames or [])
    if path.suffix == ".json":
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return "", "unavailable:invalid-json", []
        count = len(value) if isinstance(value, list) else 1
        fields = sorted(value) if isinstance(value, dict) else []
        return str(count), "available:json-parse", fields
    if path.suffix == ".parquet":
        try:
            import pyarrow.parquet as parquet  # type: ignore[import-not-found]
        except ImportError:
            return "", "unavailable:optional-pyarrow-not-installed", []
        parquet_file = parquet.ParquetFile(path)
        return (
            str(parquet_file.metadata.num_rows),
            "available:parquet-footer-via-pyarrow",
            list(parquet_file.schema_arrow.names),
        )
    if path.suffix in {".md", ".txt"}:
        with path.open(encoding="utf-8") as handle:
            return str(sum(1 for _ in handle)), "available:text-line-count", []
    return "", "unavailable:unsupported-file-type", []


def inventory_row(
    path: Path,
    *,
    source_id: str,
    granularity: str,
    key_fields: str,
    join_status: str,
    join_reason: str,
    dimensions: str,
    scientific_status: str,
    confound_status: str,
) -> dict[str, str]:
    row_count, row_count_status, schema_fields = source_metadata(path)
    return {
        "source_id": source_id,
        "path": path.relative_to(ROOT).as_posix(),
        "granularity": granularity,
        "row_count": row_count,
        "key_fields": key_fields,
        "atlas_join_status": join_status,
        "join_reason": join_reason,
        "dimensions_available": dimensions,
        "scientific_status": scientific_status,
        "confound_status": confound_status,
        "row_count_status": row_count_status,
        "schema_fields": "|".join(schema_fields),
        "sha256": sha256_path(path),
    }


def build_evidence_inventory(
    joined: Path,
    sequence: Path,
    seat: Path,
    handshape: Path,
    source_picks: Path,
) -> list[dict[str, str]]:
    rows = [
        inventory_row(
            joined,
            source_id="w42_joined_claim_actions",
            granularity="legal_action",
            key_fields="key+candidate_domino",
            join_status=STATUS_AVAILABLE,
            join_reason="exact unique identity shared by all five W42 action tables",
            dimensions="uncertainty_summaries|bidding_proxies|distributional_summaries|action_choice",
            scientific_status=(
                "offline action labels and public/action-local features; not a causal partnership estimate"
            ),
            confound_status="source trajectory policy is not fingerprinted",
        ),
        inventory_row(
            sequence,
            source_id="w42_sequence_seat_actions",
            granularity="legal_action",
            key_fields="key+candidate_domino",
            join_status=STATUS_AVAILABLE,
            join_reason="exact unique identity shared by all five W42 action tables",
            dimensions="role_order|partner_action_proxies|sequence|phase",
            scientific_status="public/action-local sequence detector evidence",
            confound_status="detector labels do not establish persistent plans or partner response",
        ),
        inventory_row(
            seat,
            source_id="w42_seat_position_actions",
            granularity="legal_action",
            key_fields="key+candidate_domino",
            join_status=STATUS_AVAILABLE,
            join_reason="exact unique identity shared by all five W42 action tables",
            dimensions="role_order|seat_position|partner_action_proxies",
            scientific_status="public/action-local seat detector evidence",
            confound_status="role and seat slices are not a fixed-versus-shuffled partnership intervention",
        ),
        inventory_row(
            handshape,
            source_id="w42_phase4_sequence_handshape_actions",
            granularity="legal_action",
            key_fields="key+candidate_domino",
            join_status=STATUS_AVAILABLE,
            join_reason="exact unique identity shared by all five W42 action tables",
            dimensions="threshold_tail_rankings|legal_action_shape|book_labels|current_trick_count",
            scientific_status="public/action-local and legal-candidate handshape proxy evidence",
            confound_status="private remaining hands and multi-trick causal continuations are absent",
        ),
        inventory_row(
            source_picks,
            source_id="w42_cross_ai_source_picks",
            granularity="legal_action",
            key_fields="key+candidate_domino",
            join_status=STATUS_AVAILABLE,
            join_reason="exact identity with atlas rows; unique fields retain their original provenance qualifiers",
            dimensions="book_detector_endorsement|threshold_top_proxy|cross_corpus_dist_lens_attempt",
            scientific_status="confounded diagnostic prior attempt, not clean cross-AI agreement evidence",
            confound_status=(
                "gus_top_action is is_best_threshold rather than Gus output; dist_lens_top_action is a "
                "cross-corpus positional merge"
            ),
        ),
    ]

    for path in sorted((ROOT / "gus/analysis").glob("drama_atlas*.parquet")):
        v2 = path.stem.endswith("_v2")
        rows.append(
            inventory_row(
                path,
                source_id=f"gus_{path.stem}",
                granularity="decision",
                key_fields="split+seed+game_idx+d_idx+current_player",
                join_status=STATUS_UNJOINABLE_GUS,
                join_reason=(
                    "apparent partial integer collisions were tested, but state, E[Q], and legal-action counts "
                    "diverge; there is no shared W42 action-state identity"
                ),
                dimensions=(
                    "outcome_variance|action_fragility|belief_sharpness|drama|real_drama"
                    if v2
                    else "outcome_variance|action_fragility|belief_sharpness|drama"
                ),
                scientific_status="full Gus decision-level drama surface retained for a future fingerprinted join",
                confound_status="different corpus/state namespace; integer collisions are false matches",
            )
        )

    for path in sorted((ROOT / "gus/analysis/tables").glob("*.csv")):
        top_rows = path.name.startswith("top20_drama")
        rows.append(
            inventory_row(
                path,
                source_id=f"gus_{path.stem}",
                granularity="selected_decision" if top_rows else "aggregate_slice",
                key_fields="seed+d_idx+current_player" if top_rows else "aggregate labels only",
                join_status=STATUS_UNJOINABLE_GUS,
                join_reason=(
                    "Gus decision indices and seeds refer to a different corpus/state encoding; "
                    "overlapping integers are not shared identities"
                ),
                dimensions="outcome_variance|action_fragility|belief_sharpness|drama",
                scientific_status="Gus drama summary or selected-example evidence",
                confound_status="different corpus/state namespace; cannot be attached to W42 actions",
            )
        )

    for path in sorted((ROOT / "arena/results").glob("**/*")):
        if not path.is_file() or path.suffix not in {".csv", ".json"}:
            continue
        if path.name == "per_hand.csv":
            granularity, keys = "match_hand", "experiment_path+game_idx+hand_idx+seed"
            dimensions = "bidding|contract_outcome|match_score_after_hand"
        elif path.name == "per_game.csv":
            granularity, keys = "match", "experiment_path+game_idx+a_team"
            dimensions = "match_score|marks|win"
        elif path.name == "summary.json":
            granularity, keys = "experiment", "experiment_path"
            dimensions = "policy_labels|marks|confidence_interval|bidding|match_score"
        else:
            continue
        rows.append(
            inventory_row(
                path,
                source_id=(
                    "arena_"
                    + path.relative_to(ROOT / "arena/results").as_posix().replace("/", "_").replace(".", "_")
                ),
                granularity=granularity,
                key_fields=keys,
                join_status=STATUS_UNJOINABLE_MATCH,
                join_reason="arena deals are not the W42 corpus decisions and carry no shared action-state fingerprint",
                dimensions=dimensions,
                scientific_status="arena outcome evidence at hand, match, or experiment granularity",
                confound_status="no shared W42 action-state fingerprint",
            )
        )

    champion_root = ROOT / "champion/evidence"
    champion_patterns = [
        "**/*summary.json",
        "**/*metrics.json",
        "**/manifest.json",
        "**/*report.md",
        "**/*predictions.md",
        "README.md",
        "champion-one-organ-theory-*.md",
        "handoff-*.md",
        "run26_selfplay/RESULTS.txt",
        "run26_selfplay/round*/ab.json",
        "run26_selfplay/round*/kl.txt",
    ]
    champion_paths = sorted(
        {
            path
            for pattern in champion_patterns
            for path in champion_root.glob(pattern)
            if path.is_file()
        }
    )
    for path in champion_paths:
        relative = path.relative_to(champion_root)
        if path.name.endswith("summary.json") or path.name == "ab.json":
            granularity = "experiment"
            dimensions = "policy_labels|marks|confidence_interval|bidding|match_score"
        elif path.name.endswith("metrics.json") or path.name == "kl.txt":
            granularity = "training_round_metrics"
            dimensions = "training_calibration|held_out_metrics|selfplay_convergence"
        elif path.name == "manifest.json":
            granularity = "artifact_manifest"
            dimensions = "corpus_provenance|artifact_scope"
        else:
            granularity = "evidence_narrative"
            dimensions = "registered_predictions|mechanism_interpretation|negative_results"
        rows.append(
            inventory_row(
                path,
                source_id=(
                    "champion_"
                    + relative.as_posix().replace("/", "_").replace(".", "_")
                ),
                granularity=granularity,
                key_fields="experiment_path",
                join_status=STATUS_UNJOINABLE_MATCH,
                join_reason="champion evidence is aggregate A/B evidence without W42 decision fingerprints",
                dimensions=dimensions,
                scientific_status="retained Champion/Jud experiment, metric, manifest, or narrative evidence",
                confound_status="experiment-level evidence cannot identify an action-row failure",
            )
        )
    return rows


def write_csv(path: Path, rows: Iterable[dict[str, str]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    full_path = output_dir / "atlas_full.csv.gz"
    sample_path = output_dir / "atlas_sample.csv"
    inventory_path = output_dir / "evidence_inventory.csv"
    summary_path = output_dir / "summary.json"
    manifest_path = output_dir / "manifest.json"

    sequence_fields = [
        "position_family",
        "phase",
        "candidate_beats_current",
        "candidate_would_win_trick_now",
        "derived_labels",
    ]
    seat_fields = ["position_family", "phase", "derived_labels"]
    handshape_fields = [
        "current_trick_count_before",
        "is_best_threshold",
        "is_safest_tail",
        "threshold_gap",
        "legal_action_n",
        "called_suit_legal_n",
        "off_suit_legal_n",
        "count_legal_n",
        "double_legal_n",
        "beater_legal_n",
        "labels",
    ]
    source_pick_fields = [
        "ev_top_action",
        "gus_top_action",
        "detector_endorsed",
        "detector_anti_endorsed",
        "dist_lens_top_action",
        "labels",
    ]
    sequence_index, sequence_header = load_action_index(args.sequence.resolve(), sequence_fields)
    seat_index, seat_header = load_action_index(args.seat.resolve(), seat_fields)
    handshape_index, handshape_header = load_action_index(args.handshape.resolve(), handshape_fields)
    source_picks_index, source_picks_header = load_action_index(args.source_picks.resolve(), source_pick_fields)

    sample = DeterministicSample(args.sample_rows)
    counters: Counter[str] = Counter()
    decisions: set[str] = set()
    dist_lens_confounded_decisions: set[str] = set()
    bid_values: set[int] = set()
    seen_action_ids: set[tuple[str, str]] = set()
    text_handle, raw_handle = open_deterministic_gzip_text(full_path)
    try:
        writer = csv.DictWriter(text_handle, fieldnames=ATLAS_FIELDS, extrasaction="raise", lineterminator="\n")
        writer.writeheader()
        for group in read_joined_groups(args.joined.resolve(), args.max_rows):
            summary = decision_summary(group)
            decision_key = group[0]["key"]
            decisions.add(decision_key)
            for base in group:
                identity = (base["key"], base["candidate_domino"])
                if identity in seen_action_ids:
                    raise ValueError(f"joined table has duplicate action identity {identity}")
                seen_action_ids.add(identity)
                bid = finite_float(base.get("bid_value"))
                if bid is not None:
                    bid_values.add(int(bid))
                if identity not in sequence_index:
                    raise ValueError(f"sequence table missing action identity {identity}")
                if identity not in seat_index:
                    raise ValueError(f"seat table missing action identity {identity}")
                if identity not in handshape_index:
                    raise ValueError(f"handshape table missing action identity {identity}")
                if identity not in source_picks_index:
                    raise ValueError(f"source-picks table missing action identity {identity}")
                row = enrich_action(
                    base,
                    sequence_index[identity],
                    seat_index[identity],
                    handshape_index[identity],
                    source_picks_index[identity],
                    summary,
                )
                writer.writerow(row)
                sample.add(row)
                counters["action_rows"] += 1
                if row["partner_coordination_proxy_labels"]:
                    counters["partner_proxy_action_rows"] += 1
                if row["bidding_risk_labels"]:
                    counters["bidding_risk_action_rows"] += 1
                if row["plan_proxy_labels"]:
                    counters["plan_proxy_action_rows"] += 1
                if row["eval_only_labels"]:
                    counters["eval_only_label_action_rows"] += 1
                if row["is_actual_action"] == "true":
                    counters["actual_action_rows"] += 1
                if row["book_detector_endorsed"] == "true":
                    counters["book_detector_endorsed_action_rows"] += 1
                if row["book_detector_anti_endorsed"] == "true":
                    counters["book_detector_anti_endorsed_action_rows"] += 1
                if row["threshold_utility_top_action_proxy"] == "true":
                    counters["threshold_utility_top_action_proxy_rows"] += 1
                if row["is_safest_tail"] == "true":
                    counters["safest_tail_action_rows"] += 1
                if row["dist_lens_top_action_confounded"]:
                    counters["dist_lens_confounded_action_rows"] += 1
                    dist_lens_confounded_decisions.add(row["decision_key"])
    finally:
        text_handle.close()
        raw_handle.close()

    if not args.max_rows:
        extra_sequence = set(sequence_index) - seen_action_ids
        extra_seat = set(seat_index) - seen_action_ids
        extra_handshape = set(handshape_index) - seen_action_ids
        extra_source_picks = set(source_picks_index) - seen_action_ids
        if extra_sequence or extra_seat or extra_handshape or extra_source_picks:
            raise ValueError(
                "full build requires identity equality; "
                f"extra sequence={len(extra_sequence)} extra seat={len(extra_seat)} "
                f"extra handshape={len(extra_handshape)} extra source-picks={len(extra_source_picks)}"
            )

    sample_rows = sample.rows()
    write_csv(sample_path, sample_rows, ATLAS_FIELDS)
    inventory_rows = build_evidence_inventory(
        args.joined.resolve(),
        args.sequence.resolve(),
        args.seat.resolve(),
        args.handshape.resolve(),
        args.source_picks.resolve(),
    )
    write_csv(inventory_path, inventory_rows, SOURCE_INVENTORY_FIELDS)

    summary = {
        "schema_version": SCHEMA_VERSION,
        "coverage": {
            "action_rows": counters["action_rows"],
            "decision_rows": len(decisions),
            "sample_action_rows": len(sample_rows),
            "actual_action_rows": counters["actual_action_rows"],
            "exact_sequence_joins": counters["action_rows"],
            "exact_seat_position_joins": counters["action_rows"],
            "exact_handshape_joins": counters["action_rows"],
            "exact_cross_ai_source_picks_joins": counters["action_rows"],
            "partner_proxy_action_rows": counters["partner_proxy_action_rows"],
            "bidding_risk_action_rows": counters["bidding_risk_action_rows"],
            "plan_proxy_action_rows": counters["plan_proxy_action_rows"],
            "eval_only_label_action_rows": counters["eval_only_label_action_rows"],
            "book_detector_endorsed_action_rows": counters["book_detector_endorsed_action_rows"],
            "book_detector_anti_endorsed_action_rows": counters["book_detector_anti_endorsed_action_rows"],
            "threshold_utility_top_action_proxy_rows": counters["threshold_utility_top_action_proxy_rows"],
            "safest_tail_action_rows": counters["safest_tail_action_rows"],
            "dist_lens_confounded_action_rows": counters["dist_lens_confounded_action_rows"],
            "dist_lens_confounded_decisions": len(dist_lens_confounded_decisions),
            "full_world_q_action_rows": 0,
            "gus_drama_joined_action_rows": 0,
            "action_likelihood_rows": 0,
            "source_trajectory_policy_rows": 0,
            "champion_action_rows": 0,
            "persistent_plan_state_rows": 0,
            "partner_assignment_rows": 0,
            "match_score_rows": 0,
            "auction_history_rows": 0,
            "bid_value_cardinality": len(bid_values),
        },
        "bidding_surface": {
            "bid_values": sorted(bid_values),
            "bid_value_cardinality": len(bid_values),
            "status": "fixed-bid30:cannot-estimate-bidding-or-bid-level-effects",
        },
        "join_result": (
            "The five retained W42 action tables join exactly by decision key plus candidate domino. "
            "Gus, arena, and champion evidence remain inventoried but unjoined."
        ),
        "scientific_boundary": {
            "what_this_is": (
                "An action-level measurement spine preserving available oracle distribution summaries, "
                "public role/order, and public/action-local partnership proxies."
            ),
            "what_this_is_not": (
                "It is not a champion trajectory corpus, partnership-effect estimate, action-likelihood model, "
                "persistent-plan trace, or match-score-conditioned dataset."
            ),
            "leakage": (
                "Oracle values and eval-only labels are retained as offline outcomes only; they are not promoted "
                "to runtime-observable features."
            ),
        },
        "inventory_scope": {
            "champion": {
                "included_classes": [
                    "A/B summary JSON",
                    "training metric JSON",
                    "artifact manifest JSON",
                    "registered prediction and build report Markdown",
                    "Jud demonstration manifests",
                    "run26 self-play A/B and convergence text",
                ],
                "omitted_classes": [
                    "model weights",
                    "PNG figures",
                    "NPZ demonstrations",
                    "executable source code",
                ],
                "claim": "bounded retained evidence inventory, not every file under champion",
            }
        },
        "structural_limits": [
            (
                "The q_per_world source artifact referenced by older manifests is not retained in this worktree; "
                "only mean, regret, threshold mass, and lower-tail mass survive in the compatible CSVs."
            ),
            "Gus drama rows use a different corpus/state namespace, so integer seed overlap cannot establish identity.",
            (
                "The source-picks field named gus_top_action is is_best_threshold, not actual Gus output; the "
                "atlas renames it threshold_utility_top_action_proxy."
            ),
            (
                "Nonblank dist_lens_top_action values came from a cross-corpus positional join on game index, "
                "decision index, and candidate domino; they remain confounded."
            ),
            (
                "Arena and champion evidence is at hand, match, or experiment granularity with no W42 "
                "action-state fingerprint."
            ),
            (
                "The retained actual-action indicator is not policy-fingerprinted, so its regret cannot be "
                "attributed to the current champion."
            ),
            (
                "Partner labels are public/action-local proxies; partner identity and fixed-versus-shuffled "
                "assignment are absent."
            ),
            (
                "Actual-action flags survive, but actor-policy likelihoods and action-conditioned posterior "
                "changes do not."
            ),
            "The bidder-lead-plan label is a detector firing, not a plan ID or persistent state across decisions.",
            (
                "Every compatible action row has bid_value=30. Declaration and partial risk proxies survive, "
                "but this corpus cannot estimate bidding or bid-level effects."
            ),
            "Complete auction history and pre-hand match score do not survive.",
        ],
    }
    write_json(summary_path, summary)

    artifact_paths = [full_path, sample_path, inventory_path, summary_path]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "runner": "w42/partnership_failure_atlas/run_atlas.py",
        "reproduce": "python w42/partnership_failure_atlas/run_atlas.py",
        "validator": "python w42/partnership_failure_atlas/validate_outputs.py",
        "inputs": {
            "joined_claim_actions": {
                "path": args.joined.resolve().relative_to(ROOT).as_posix(),
                "sha256": sha256_path(args.joined.resolve()),
            },
            "sequence_seat_actions": {
                "path": args.sequence.resolve().relative_to(ROOT).as_posix(),
                "sha256": sha256_path(args.sequence.resolve()),
                "columns": sequence_header,
            },
            "seat_position_actions": {
                "path": args.seat.resolve().relative_to(ROOT).as_posix(),
                "sha256": sha256_path(args.seat.resolve()),
                "columns": seat_header,
            },
            "sequence_handshape_actions": {
                "path": args.handshape.resolve().relative_to(ROOT).as_posix(),
                "sha256": sha256_path(args.handshape.resolve()),
                "columns": handshape_header,
            },
            "cross_ai_source_picks": {
                "path": args.source_picks.resolve().relative_to(ROOT).as_posix(),
                "sha256": sha256_path(args.source_picks.resolve()),
                "columns": source_picks_header,
            },
            "threshold_semantics_reference": {
                "path": "w42/branch_atlas_v1/build_branch_atlas.py",
                "sha256": sha256_path(ROOT / "w42/branch_atlas_v1/build_branch_atlas.py"),
                "role": "threshold_q_for_player semantics",
            },
            "lower_tail_semantics_reference": {
                "path": "w42/distribution_aware_ev_report/build_distribution_aware_ev_report.py",
                "sha256": sha256_path(
                    ROOT / "w42/distribution_aware_ev_report/build_distribution_aware_ev_report.py"
                ),
                "role": "lower-tail Q<=-18 semantics",
            },
        },
        "identity": {
            "fields": ["decision_key", "candidate_domino"],
            "uniqueness": "exact across all five compatible W42 tables",
        },
        "status_vocabulary": {
            "available": "Directly present or exactly joined at the action identity.",
            "partial": "Some evidence survives, but the full instrument needed for the construct does not.",
            "proxy-only": "A public/action-local detector exists, but the causal construct is not measured.",
            "unavailable": "The retained compatible inputs do not contain the field.",
            "unjoinable": "A source contains relevant evidence under a different granularity or state namespace.",
            "confounded": "A value is retained for audit but cannot be interpreted as clean cross-source evidence.",
        },
        "distribution_semantics": {
            "threshold_mass": "P(Q>=threshold_q)",
            "threshold_q_offense": "2*contract_points-42",
            "threshold_q_defense": "43-2*contract_points",
            "contract_points": "42 when bid_value=84, otherwise bid_value",
            "lower_tail_mass": "P(Q<=-18)",
            "sampled_world_count": "unavailable",
            "sampler_and_version": "unavailable",
            "world_weighting": "unavailable",
            "calibration": "unavailable",
        },
        "dimensions": {
            "uncertainty": (
                "partial Q-outcome summaries; worlds, count, sampler/version, weighting, and calibration unavailable"
            ),
            "role_order": "public role, seat, trick order, phase, and detector labels available",
            "partner_coordination": "action-local proxy labels only; no partner assignment or interaction estimate",
            "action_derived_inference": "actual-action indicator only; no likelihood or posterior delta",
            "plan_persistence": "bidder-lead-plan detector only; no cross-decision plan state",
            "distributional_utility": (
                "fixed threshold/tail ranking proxies survive; no utility transform, consumer, or context"
            ),
            "bidding": "fixed bid=30 plus declaration/risk proxies; no auction choices, history, or bid-level effect",
            "match_score": "unavailable at W42 action-row granularity",
        },
        "inventory_scope": summary["inventory_scope"],
        "atlas_fields": ATLAS_FIELDS,
        "bidding_surface": {
            "bid_values": sorted(bid_values),
            "bid_value_cardinality": len(bid_values),
            "status": "fixed-bid30:cannot-estimate-bidding-or-bid-level-effects",
        },
        "artifacts": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256_path(path)}
            for path in artifact_paths
        },
        "build_options": {"max_rows": args.max_rows, "sample_rows": args.sample_rows},
    }
    write_json(manifest_path, manifest)
    return summary


def main() -> None:
    args = parse_args()
    if args.max_rows < 0 or args.sample_rows < 0:
        raise SystemExit("--max-rows and --sample-rows must be non-negative")
    summary = build(args)
    print(json.dumps(summary["coverage"], sort_keys=True))


if __name__ == "__main__":
    main()
