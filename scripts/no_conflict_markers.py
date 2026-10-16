#!/usr/bin/env python3
# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0

"""no_conflict_markers.py: fail if any tracked text file holds a merge-conflict marker.

Checks every file `git ls-files` lists. A line starting with `<<<<<<< ` or
`>>>>>>> ` is a marker. A line that is exactly `=======` is a marker only
between an open `<<<<<<<` and its `>>>>>>>`, so Markdown setext headings
pass. Binary files and files over 5 MB are skipped. Prints file:line for each
marker found and exits 1; exits 0 when there are none.
"""

import subprocess
import sys
from pathlib import Path

OPEN, MID, CLOSE = "<<<<<<< ", "=======", ">>>>>>> "
MAX_BYTES = 5 * 1024 * 1024
ROOT = Path(__file__).resolve().parent.parent


def tracked_files(root):
    out = subprocess.run(["git", "ls-files", "-z"], cwd=root, capture_output=True, check=True).stdout
    return [root / p for p in out.decode("utf-8", "surrogateescape").split("\0") if p]


def markers_in(path):
    """(line number, text) for every conflict marker in one file; [] if skipped."""
    try:
        if path.stat().st_size > MAX_BYTES:
            return []
        data = path.read_bytes()
    except OSError:  # tracked but deleted from the working tree
        return []
    if b"\0" in data[:8000]:
        return []
    found, open_region = [], False
    for n, line in enumerate(data.decode("utf-8", "replace").splitlines(), 1):
        if line.startswith(OPEN) or line == OPEN.rstrip():
            open_region = True
            found.append((n, line))
        elif line.startswith(CLOSE) or line == CLOSE.rstrip():
            open_region = False
            found.append((n, line))
        elif line == MID and open_region:
            found.append((n, line))
    return found


def scan(root):
    """Map each offending tracked file (relative path) to its (line, text) markers."""
    report = {}
    for path in tracked_files(root):
        hits = markers_in(path)
        if hits:
            report[str(path.relative_to(root))] = hits
    return report


def main(root=ROOT):
    report = scan(root)
    if not report:
        print("no conflict markers in tracked files")
        return 0
    for rel, hits in report.items():
        for n, text in hits:
            print(f"{rel}:{n}: {text}")
    print(f"conflict markers found in {len(report)} file(s); resolve them before committing", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
