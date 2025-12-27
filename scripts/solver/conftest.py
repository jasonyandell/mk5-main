"""
Pytest configuration and fixtures for solver tests.
"""

import json
import subprocess
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def ts_tables() -> dict:
    """
    Export TypeScript tables for cross-validation.

    Runs export-tables.ts and returns parsed JSON.
    This is cached for the entire test session.
    """
    project_dir = Path(__file__).parent.parent.parent

    result = subprocess.run(
        ["npx", "tsx", "scripts/export-tables.ts"],
        cwd=project_dir,
        capture_output=True,
        text=True,
        timeout=30,
    )

    if result.returncode != 0:
        pytest.fail(f"Failed to export TS tables: {result.stderr}")

    return json.loads(result.stdout)
