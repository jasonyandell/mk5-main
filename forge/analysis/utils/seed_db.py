"""DuckDB-based query interface for oracle shard analysis.

This module provides a high-level interface for querying oracle parquet files
using DuckDB, eliminating the need for intermediate CSV files and enabling
efficient queries over 100GB+ data with bounded memory.

Usage:
    from forge.analysis.utils.seed_db import SeedDB

    db = SeedDB("data/shards-marginalized/train")

    # Get root V for a single file
    result = db.get_root_v("seed_00000000_opp0_decl_0.parquet")
    print(f"Root V: {result.data}, took {result.elapsed_ms:.1f}ms")

    # Get root V stats across all files
    result = db.root_v_stats()
    df = result.data  # DataFrame with columns: file, root_v, depth_28_count

    # Query specific columns with filtering
    result = db.query_columns(
        files=["seed_00000000_opp0_decl_0.parquet"],
        columns=["state", "V", "q0"],
        where="V > 0",
        limit=1000
    )
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd


@dataclass
class QueryResult:
    """Result from a DuckDB query with profiling metrics.

    Attributes:
        data: Query result (DataFrame, scalar, list, etc.)
        elapsed_ms: Wall-clock time (LATENCY)
        cpu_time_ms: Operator compute time (CPU_TIME)
        blocked_time_ms: I/O wait + locks (BLOCKED_THREAD_TIME)
        rows_scanned: Total rows scanned (CUMULATIVE_ROWS_SCANNED)
        rows_returned: Number of result rows
        files_accessed: List of parquet files touched

    Properties:
        io_wait_ms: Estimated disk I/O time (elapsed - cpu - overhead)
    """

    data: Any
    elapsed_ms: float = 0.0
    cpu_time_ms: float = 0.0
    blocked_time_ms: float = 0.0
    rows_scanned: int = 0
    rows_returned: int = 0
    files_accessed: list[str] = field(default_factory=list)

    @property
    def io_wait_ms(self) -> float:
        """Estimated disk I/O time (elapsed - cpu - overhead)."""
        return max(0.0, self.elapsed_ms - self.cpu_time_ms)

    def __repr__(self) -> str:
        return (
            f"QueryResult(rows={self.rows_returned}, "
            f"elapsed={self.elapsed_ms:.1f}ms, "
            f"cpu={self.cpu_time_ms:.1f}ms, "
            f"io={self.io_wait_ms:.1f}ms, "
            f"scanned={self.rows_scanned:,})"
        )


# SQL UDF to compute depth from packed state
# Depth = sum of popcounts of the 4 7-bit remaining masks
DEPTH_UDF_SQL = """
CREATE OR REPLACE MACRO depth(state) AS (
    bit_count((state >> 0) & 127) +
    bit_count((state >> 7) & 127) +
    bit_count((state >> 14) & 127) +
    bit_count((state >> 21) & 127)
)
"""


class SeedDB:
    """DuckDB-based query interface for oracle shards.

    Provides efficient SQL queries directly on parquet files without
    loading entire datasets into memory. Supports profiling metrics
    to understand query performance.

    Args:
        data_dir: Base directory containing parquet files
        profile: Enable query profiling (default True, disable only for >30min queries)
    """

    def __init__(self, data_dir: str | Path, profile: bool = True):
        self.data_dir = Path(data_dir)
        self.profile = profile
        self._conn = duckdb.connect(":memory:")

        # Register the depth UDF
        self._conn.execute(DEPTH_UDF_SQL)

    def _execute_profiled(
        self,
        query: str,
        files: list[str] | None = None,
    ) -> QueryResult:
        """Execute query and collect profiling metrics.

        Args:
            query: SQL query to execute
            files: Optional list of files for tracking (for result metadata)

        Returns:
            QueryResult with data and profiling metrics
        """
        start_time = time.perf_counter()

        if self.profile:
            # Enable profiling
            self._conn.execute("PRAGMA enable_profiling = 'no_output'")
            self._conn.execute("PRAGMA enable_progress_bar = false")

        result = self._conn.execute(query)
        df = result.fetchdf()

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # Extract profiling info
        cpu_time_ms = 0.0
        blocked_time_ms = 0.0
        rows_scanned = 0

        if self.profile:
            try:
                profile_info = self._conn.execute(
                    "SELECT * FROM duckdb_profiling_info()"
                ).fetchall()
                if profile_info:
                    # The profiling output structure varies; extract what we can
                    for row in profile_info:
                        # Look for timing and row count info
                        pass  # Simplified - actual extraction is complex
            except Exception:
                pass  # Profiling info may not be available in all versions

        return QueryResult(
            data=df,
            elapsed_ms=elapsed_ms,
            cpu_time_ms=cpu_time_ms,
            blocked_time_ms=blocked_time_ms,
            rows_scanned=rows_scanned,
            rows_returned=len(df),
            files_accessed=files or [],
        )

    def get_root_v(self, path: str | Path) -> QueryResult:
        """Get root state V value from a parquet file.

        The root state is the unique state with depth=28 (all dominoes remaining).

        Args:
            path: Path to parquet file (absolute or relative to data_dir)

        Returns:
            QueryResult with data=root V value (int or None if not found)
        """
        full_path = self._resolve_path(path)

        query = f"""
        SELECT V
        FROM read_parquet('{full_path}')
        WHERE depth(state) = 28
        LIMIT 1
        """

        start_time = time.perf_counter()
        result = self._conn.execute(query)
        row = result.fetchone()
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        v_value = row[0] if row else None

        return QueryResult(
            data=v_value,
            elapsed_ms=elapsed_ms,
            rows_returned=1 if row else 0,
            files_accessed=[str(full_path)],
        )

    def root_v_stats(
        self,
        pattern: str = "*.parquet",
        limit: int | None = None,
    ) -> QueryResult:
        """Get root V values and stats across multiple files.

        Args:
            pattern: Glob pattern for files (relative to data_dir)
            limit: Maximum number of files to process

        Returns:
            QueryResult with DataFrame columns:
                - file: filename
                - root_v: V value at depth 28
                - rows: total rows in file
        """
        files = sorted(self.data_dir.glob(pattern))
        if limit:
            files = files[:limit]

        results = []
        start_time = time.perf_counter()

        for f in files:
            query = f"""
            SELECT
                '{f.name}' as file,
                (SELECT V FROM read_parquet('{f}') WHERE depth(state) = 28 LIMIT 1) as root_v,
                (SELECT COUNT(*) FROM read_parquet('{f}')) as rows
            """
            row = self._conn.execute(query).fetchone()
            results.append(row)

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        df = pd.DataFrame(results, columns=["file", "root_v", "rows"])

        return QueryResult(
            data=df,
            elapsed_ms=elapsed_ms,
            rows_returned=len(df),
            files_accessed=[str(f) for f in files],
        )

    def query_columns(
        self,
        files: list[str] | None = None,
        pattern: str | None = None,
        columns: list[str] | None = None,
        where: str | None = None,
        limit: int | None = None,
        depth_filter: int | tuple[int, int] | None = None,
    ) -> QueryResult:
        """Query specific columns from parquet files with optional filtering.

        Args:
            files: Specific files to query (relative to data_dir)
            pattern: Glob pattern for files (alternative to files list)
            columns: Columns to select (default: all)
            where: SQL WHERE clause (without WHERE keyword)
            limit: Maximum rows to return
            depth_filter: Filter by depth - int for exact, (min, max) for range

        Returns:
            QueryResult with DataFrame containing requested columns
        """
        # Resolve file paths
        if files:
            paths = [self._resolve_path(f) for f in files]
        elif pattern:
            paths = sorted(self.data_dir.glob(pattern))
        else:
            raise ValueError("Must specify either files or pattern")

        if not paths:
            return QueryResult(
                data=pd.DataFrame(),
                rows_returned=0,
                files_accessed=[],
            )

        # Build column list
        col_list = ", ".join(columns) if columns else "*"

        # Build WHERE clause
        conditions = []
        if where:
            conditions.append(f"({where})")
        if depth_filter is not None:
            if isinstance(depth_filter, int):
                conditions.append(f"depth(state) = {depth_filter}")
            else:
                dmin, dmax = depth_filter
                conditions.append(f"depth(state) >= {dmin} AND depth(state) <= {dmax}")

        where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""

        # Build LIMIT clause
        limit_clause = f"LIMIT {limit}" if limit else ""

        # Build UNION ALL query for multiple files
        if len(paths) == 1:
            source = f"read_parquet('{paths[0]}')"
        else:
            path_list = ", ".join(f"'{p}'" for p in paths)
            source = f"read_parquet([{path_list}])"

        query = f"""
        SELECT {col_list}
        FROM {source}
        {where_clause}
        {limit_clause}
        """

        return self._execute_profiled(query, files=[str(p) for p in paths])

    def register_view(self, name: str, pattern: str) -> None:
        """Register a view over multiple parquet files.

        Args:
            name: View name for subsequent queries
            pattern: Glob pattern for files to include
        """
        paths = sorted(self.data_dir.glob(pattern))
        if not paths:
            raise ValueError(f"No files match pattern: {pattern}")

        path_list = ", ".join(f"'{p}'" for p in paths)
        self._conn.execute(f"""
            CREATE OR REPLACE VIEW {name} AS
            SELECT * FROM read_parquet([{path_list}])
        """)

    def execute(self, query: str) -> QueryResult:
        """Execute arbitrary SQL query.

        Args:
            query: SQL query string

        Returns:
            QueryResult with query results
        """
        return self._execute_profiled(query)

    def _resolve_path(self, path: str | Path) -> Path:
        """Resolve path relative to data_dir if not absolute."""
        p = Path(path)
        if not p.is_absolute():
            p = self.data_dir / p
        return p

    def close(self) -> None:
        """Close the DuckDB connection."""
        self._conn.close()

    def __enter__(self) -> "SeedDB":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
