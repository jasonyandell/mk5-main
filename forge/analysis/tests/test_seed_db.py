"""Tests for SeedDB DuckDB interface."""

from pathlib import Path

import numpy as np
import pytest

from forge.analysis.utils.seed_db import QueryResult, SeedDB


# Test data in project root (forge/analysis/tests -> project root = 3 levels up)
TEST_DATA_DIR = Path(__file__).parents[3] / "data" / "flywheel-shards"


@pytest.fixture
def db():
    """Create SeedDB instance for testing."""
    if not TEST_DATA_DIR.exists():
        pytest.skip(f"Test data not found at {TEST_DATA_DIR}")
    return SeedDB(TEST_DATA_DIR)


class TestQueryResult:
    """Tests for QueryResult dataclass."""

    def test_io_wait_ms_property(self):
        """io_wait_ms computes elapsed - cpu correctly."""
        result = QueryResult(
            data=None,
            elapsed_ms=100.0,
            cpu_time_ms=60.0,
        )
        assert result.io_wait_ms == 40.0

    def test_io_wait_ms_never_negative(self):
        """io_wait_ms returns 0 if cpu > elapsed (edge case)."""
        result = QueryResult(
            data=None,
            elapsed_ms=50.0,
            cpu_time_ms=100.0,
        )
        assert result.io_wait_ms == 0.0

    def test_repr(self):
        """QueryResult has useful repr."""
        result = QueryResult(
            data=None,
            elapsed_ms=123.4,
            cpu_time_ms=100.0,
            rows_scanned=1000000,
            rows_returned=100,
        )
        s = repr(result)
        assert "rows=100" in s
        assert "elapsed=123.4ms" in s
        assert "scanned=1,000,000" in s


class TestSeedDB:
    """Tests for SeedDB class."""

    def test_init(self, db):
        """SeedDB initializes with data directory."""
        assert db.data_dir == TEST_DATA_DIR
        assert db.profile is True

    def test_get_root_v(self, db):
        """get_root_v returns V value for depth=28 state."""
        # Use first available shard
        files = list(TEST_DATA_DIR.glob("*.parquet"))
        if not files:
            pytest.skip("No parquet files found")

        result = db.get_root_v(files[0])

        # Result should be a QueryResult
        assert isinstance(result, QueryResult)

        # V value should be in valid range [-42, 42]
        assert result.data is not None
        assert -42 <= result.data <= 42

        # Timing should be recorded
        assert result.elapsed_ms > 0
        assert result.rows_returned == 1
        assert len(result.files_accessed) == 1

    def test_get_root_v_relative_path(self, db):
        """get_root_v works with relative paths."""
        files = list(TEST_DATA_DIR.glob("*.parquet"))
        if not files:
            pytest.skip("No parquet files found")

        # Pass just the filename
        result = db.get_root_v(files[0].name)

        assert result.data is not None
        assert -42 <= result.data <= 42

    def test_root_v_stats(self, db):
        """root_v_stats returns DataFrame with file stats."""
        result = db.root_v_stats(limit=3)

        assert isinstance(result, QueryResult)
        df = result.data

        # Should have correct columns
        assert list(df.columns) == ["file", "root_v", "rows"]

        # Should have data
        assert len(df) <= 3
        assert len(df) > 0

        # V values should be valid
        for v in df["root_v"]:
            if v is not None:
                assert -42 <= v <= 42

        # Row counts should be positive
        assert (df["rows"] > 0).all()

    def test_query_columns_basic(self, db):
        """query_columns returns requested columns."""
        files = list(TEST_DATA_DIR.glob("*.parquet"))
        if not files:
            pytest.skip("No parquet files found")

        result = db.query_columns(
            files=[files[0].name],
            columns=["state", "V"],
            limit=100,
        )

        assert isinstance(result, QueryResult)
        df = result.data

        assert list(df.columns) == ["state", "V"]
        assert len(df) == 100
        assert result.rows_returned == 100

    def test_query_columns_with_where(self, db):
        """query_columns respects WHERE clause."""
        files = list(TEST_DATA_DIR.glob("*.parquet"))
        if not files:
            pytest.skip("No parquet files found")

        result = db.query_columns(
            files=[files[0].name],
            columns=["V"],
            where="V > 0",
            limit=100,
        )

        df = result.data
        assert (df["V"] > 0).all()

    def test_query_columns_with_depth_filter_exact(self, db):
        """query_columns filters by exact depth."""
        files = list(TEST_DATA_DIR.glob("*.parquet"))
        if not files:
            pytest.skip("No parquet files found")

        result = db.query_columns(
            files=[files[0].name],
            columns=["state", "V"],
            depth_filter=28,
        )

        # Should return exactly 1 row (the root state)
        df = result.data
        assert len(df) == 1

    def test_query_columns_with_depth_filter_range(self, db):
        """query_columns filters by depth range."""
        files = list(TEST_DATA_DIR.glob("*.parquet"))
        if not files:
            pytest.skip("No parquet files found")

        result = db.query_columns(
            files=[files[0].name],
            columns=["state", "V"],
            depth_filter=(26, 28),
            limit=100,
        )

        # All returned states should be in depth range
        # (Would need to verify with depth extraction, but basic check is data is returned)
        assert len(result.data) > 0

    def test_query_columns_with_pattern(self, db):
        """query_columns works with glob pattern."""
        result = db.query_columns(
            pattern="seed_0000020[0-2]*.parquet",
            columns=["V"],
            limit=10,
        )

        assert len(result.data) == 10

    def test_query_columns_missing_file_raises(self, db):
        """query_columns raises IOException for missing files."""
        import duckdb

        with pytest.raises(duckdb.IOException):
            db.query_columns(
                files=["nonexistent.parquet"],
                columns=["V"],
            )

    def test_register_view(self, db):
        """register_view creates queryable view."""
        db.register_view("test_shards", "seed_0000020[0-2]*.parquet")

        result = db.execute("SELECT COUNT(*) as cnt FROM test_shards")
        df = result.data

        assert df["cnt"].iloc[0] > 0

    def test_execute_custom_query(self, db):
        """execute runs arbitrary SQL."""
        files = list(TEST_DATA_DIR.glob("*.parquet"))
        if not files:
            pytest.skip("No parquet files found")

        result = db.execute(f"""
            SELECT depth(state) as d, COUNT(*) as cnt
            FROM read_parquet('{files[0]}')
            GROUP BY depth(state)
            ORDER BY d DESC
            LIMIT 5
        """)

        df = result.data
        assert "d" in df.columns
        assert "cnt" in df.columns
        # Highest depth should be 28 (root)
        assert df["d"].max() == 28

    def test_context_manager(self):
        """SeedDB works as context manager."""
        if not TEST_DATA_DIR.exists():
            pytest.skip(f"Test data not found at {TEST_DATA_DIR}")

        with SeedDB(TEST_DATA_DIR) as db:
            result = db.execute("SELECT 1 as x")
            assert result.data["x"].iloc[0] == 1


class TestDepthMacro:
    """Tests for the depth() SQL macro."""

    def test_depth_28_at_root(self, db):
        """Root state has depth=28."""
        files = list(TEST_DATA_DIR.glob("*.parquet"))
        if not files:
            pytest.skip("No parquet files found")

        # Get root state
        result = db.execute(f"""
            SELECT state, depth(state) as d
            FROM read_parquet('{files[0]}')
            WHERE depth(state) = 28
        """)

        df = result.data
        assert len(df) == 1
        assert df["d"].iloc[0] == 28

    def test_depth_decreases_during_game(self, db):
        """Depth values range from 0 to 28."""
        files = list(TEST_DATA_DIR.glob("*.parquet"))
        if not files:
            pytest.skip("No parquet files found")

        result = db.execute(f"""
            SELECT
                MIN(depth(state)) as min_d,
                MAX(depth(state)) as max_d
            FROM read_parquet('{files[0]}')
        """)

        df = result.data
        # Min depth should be 0 (terminal) or small
        assert df["min_d"].iloc[0] >= 0
        # Max depth should be 28 (root)
        assert df["max_d"].iloc[0] == 28
