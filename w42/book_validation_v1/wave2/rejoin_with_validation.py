#!/usr/bin/env python3
"""Re-join all per-batch .pt files including seed 9430 validation files.

This fixes the validation step by including the 9430 .pt files in the joined CSV.
"""

from __future__ import annotations
import sys
from pathlib import Path
import json
import math

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

sys.path.insert(0, str(Path(__file__).parent))
from run_bid_aware_atlas import (
    ACTION_COLUMNS,
    compute_mark_ev_divergence,
    process_pt_file,
    sha256_file,
    validate_bid30_against_atlas,
    write_csv,
    write_json,
    round4,
)

OUTPUT_DIR = ROOT / "w42/book_validation_v1/wave2/bid_aware_atlas"
BID_VALUES = [30, 32, 35, 36, 39, 42, 84]

# All batch .pt files to include
BATCH_RANGES = [
    (9000, 9009),
    (9010, 9019),
    (9020, 9029),
    (9030, 9039),
    (9040, 9049),
    (9430, 9430),  # validation seed
]

all_action_rows = []
join_log = []

print("Re-joining all .pt files including seed 9430 validation files...")
for bv in BID_VALUES:
    for (bstart, bend) in BATCH_RANGES:
        pt_path = OUTPUT_DIR / f"eq_pdf_seeds{bstart}-{bend}_bid{bv}_v2.pt"
        if not pt_path.exists():
            print(f"  MISSING: {pt_path.name}")
            join_log.append({"file": pt_path.name, "status": "missing"})
            continue
        try:
            rows = process_pt_file(pt_path, bv)
            all_action_rows.extend(rows)
            join_log.append({
                "file": pt_path.name, "status": "ok", "rows": len(rows),
                "sha256": sha256_file(pt_path),
            })
            print(f"  {pt_path.name}: {len(rows)} rows")
        except Exception as exc:
            print(f"  ERROR {pt_path.name}: {exc}")
            join_log.append({"file": pt_path.name, "status": "error", "error": str(exc)})

# Write joined CSV (only for 9000-9049, not 9430, for main corpus)
# But for validation we need 9430 rows in the rows_bid30 subset.
print(f"\nTotal rows: {len(all_action_rows)}")

# Separate: main corpus (seeds 9000-9049) for the joined CSV
main_rows = [r for r in all_action_rows if 9000 <= int(r["seed"]) <= 9049]
print(f"Main corpus rows (seeds 9000-9049): {len(main_rows)}")

# Write main CSV
joined_csv = OUTPUT_DIR / "bid_aware_actions.csv"
write_csv(joined_csv, main_rows, fieldnames=ACTION_COLUMNS)
print(f"Wrote {len(main_rows)} rows to {joined_csv}")

# Validation uses all rows (including 9430)
print("\nRunning validation at bid=30 (all rows including 9430)...")
atlas_pt = ROOT / "w42/branch_atlas_scaled_v0/eq_pdf_s9430_d10_bid30_joint_1000s_v2.pt"
rows_bid30 = [r for r in all_action_rows if int(r["bid_value"]) == 30]
val_csv = OUTPUT_DIR / "validation_check.csv"
val_result = validate_bid30_against_atlas(rows_bid30, atlas_pt, val_csv)
print(f"Validation: {val_result}")

# Divergence for main corpus
from collections import Counter
bid_row_counts = Counter(int(r["bid_value"]) for r in main_rows)
print("\nBid row counts (main corpus 9000-9049):")
for bv in BID_VALUES:
    print(f"  bid={bv}: {bid_row_counts.get(bv, 0)}")

divergence = compute_mark_ev_divergence(main_rows)

# Load existing manifest and update it
manifest_path = OUTPUT_DIR / "manifest.json"
with open(manifest_path) as f:
    manifest = json.load(f)

manifest["join_with_validation"] = join_log
manifest["validation_bid30"] = val_result
manifest["mark_ev_divergence"] = divergence
manifest["totals"]["n_action_rows"] = len(main_rows)
manifest["totals"]["bid_row_counts"] = {str(k): v for k, v in sorted(bid_row_counts.items())}
manifest["rejoin_note"] = (
    "Re-joined to include seed 9430 validation .pt files for validation step. "
    "bid_aware_actions.csv contains only seeds 9000-9049 (main corpus). "
    "Validation was re-run on rows from seeds 9000-9049 + 9430."
)

write_json(manifest_path, manifest)
print(f"\nUpdated manifest: {manifest_path}")
print(f"Validation pass: {val_result.get('pass')} ({val_result.get('n_within_noise')}/{val_result.get('n_decl_ids_compared')})")
