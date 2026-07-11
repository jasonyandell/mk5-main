#!/usr/bin/env python3
"""CPU-only tests for the partnership failure atlas."""

from __future__ import annotations

import csv
import gzip
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

from run_atlas import SCHEMA_VERSION, decision_summary, split_labels, stable_row_id, threshold_q
from validate_outputs import validate


class UnitTests(unittest.TestCase):
    def test_stable_row_id_changes_with_action(self) -> None:
        self.assertEqual(stable_row_id("decision", "6-6"), stable_row_id("decision", "6-6"))
        self.assertNotEqual(stable_row_id("decision", "6-6"), stable_row_id("decision", "6-5"))

    def test_split_labels_deduplicates_and_sorts(self) -> None:
        self.assertEqual(split_labels("z|a|z||"), ("a", "z"))

    def test_threshold_q_is_role_and_bid_dependent(self) -> None:
        self.assertEqual(threshold_q("offense", 30), 18.0)
        self.assertEqual(threshold_q("defense", 30), -17.0)
        self.assertEqual(threshold_q("offense", 35), 28.0)
        self.assertEqual(threshold_q("defense", 35), -27.0)
        self.assertEqual(threshold_q("offense", 84), 42.0)

    def test_decision_summary_preserves_actual_and_value_spread(self) -> None:
        rows = [
            {
                "candidate_domino": "6-6",
                "mean": "10",
                "mean_regret": "2",
                "is_actual_action": "true",
                "is_best_mean": "false",
            },
            {
                "candidate_domino": "5-5",
                "mean": "12",
                "mean_regret": "0",
                "is_actual_action": "false",
                "is_best_mean": "true",
            },
            {
                "candidate_domino": "4-4",
                "mean": "7",
                "mean_regret": "5",
                "is_actual_action": "false",
                "is_best_mean": "false",
            },
        ]
        summary = decision_summary(rows)
        self.assertEqual(summary["legal_action_count"], "3")
        self.assertEqual(summary["decision_mean_q_span"], "5")
        self.assertEqual(summary["decision_best_second_gap"], "2")
        self.assertEqual(summary["actual_action_candidate"], "6-6")


class IntegrationTests(unittest.TestCase):
    def test_smoke_build_and_validate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            command = [
                sys.executable,
                str(HERE / "run_atlas.py"),
                "--output-dir",
                str(output),
                "--max-rows",
                "1000",
                "--sample-rows",
                "32",
            ]
            subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
            result = validate(output)
            self.assertEqual(result["action_rows"], 1000)
            self.assertEqual(result["sample_rows"], 32)

            with gzip.open(output / "atlas_full.csv.gz", "rt", newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertTrue(all(row["atlas_schema_version"] == SCHEMA_VERSION for row in rows))
            self.assertTrue(all(row["match_score_our"] == "" for row in rows))
            self.assertTrue(all(row["match_score_status"].startswith("unavailable:") for row in rows))
            self.assertTrue(all(row["source_trajectory_policy"] == "" for row in rows))
            self.assertTrue(all(row["champion_action_candidate"] == "" for row in rows))
            self.assertTrue(all(row["join_handshape_status"] == "available:exact-action-key" for row in rows))
            self.assertTrue(
                all(row["join_cross_ai_source_picks_status"] == "available:exact-action-key" for row in rows)
            )
            self.assertTrue(
                all(row["threshold_utility_top_action_proxy"] == row["is_best_threshold"] for row in rows)
            )
            self.assertTrue(all(row["uncertainty_status"].startswith("partial:") for row in rows))
            self.assertTrue(all(row["distributional_utility_status"].startswith("proxy-only:") for row in rows))
            self.assertTrue(all(row["auction_contract_status"].startswith("fixed-bid30:") for row in rows))

            summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["coverage"]["gus_drama_joined_action_rows"], 0)
            self.assertEqual(summary["coverage"]["persistent_plan_state_rows"], 0)
            self.assertEqual(summary["coverage"]["champion_action_rows"], 0)
            self.assertEqual(summary["coverage"]["exact_handshape_joins"], 1000)
            self.assertEqual(summary["coverage"]["exact_cross_ai_source_picks_joins"], 1000)
            self.assertEqual(summary["bidding_surface"]["bid_values"], [30])

            with (output / "evidence_inventory.csv").open(newline="", encoding="utf-8") as handle:
                inventory = {row["path"]: row for row in csv.DictReader(handle)}
            for parquet_path in [
                "gus/analysis/drama_atlas.parquet",
                "gus/analysis/drama_atlas_v2.parquet",
            ]:
                self.assertEqual(inventory[parquet_path]["row_count"], "280560")
                self.assertTrue(inventory[parquet_path]["atlas_join_status"].startswith("unjoinable:"))
            source_picks = inventory[
                "w42/book_validation_v1/wave1/t42-m2i7_cross_ai_agreement/per_action_source_picks.csv"
            ]
            self.assertIn("confounded", source_picks["scientific_status"])
            self.assertIn("cross-corpus", source_picks["confound_status"])


if __name__ == "__main__":
    unittest.main()
