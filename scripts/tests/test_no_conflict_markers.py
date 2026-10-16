# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0

"""scripts/no_conflict_markers.py: flags merge-conflict markers in tracked files, passes
Markdown headings and untracked files. Every test builds its own temporary git repo."""

import io
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import no_conflict_markers as ncm  # noqa: E402

# Built by concatenation so that no line of this file starts with a marker.
LT, GT = "<" * 7, ">" * 7
MID = "=" * 7


def conflict(ours="ours", theirs="theirs"):
    return f"before\n{LT} HEAD\n{ours}\n{MID}\n{theirs}\n{GT} branch\nafter\n"


class MarkerScanTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        subprocess.run(["git", "init", "-q"], cwd=self.root, check=True)

    def track(self, name, data):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data if isinstance(data, bytes) else data.encode())
        subprocess.run(["git", "add", "--", name], cwd=self.root, check=True)

    def test_flags_conflict_block_with_line_numbers(self):
        self.track("index.md", conflict())
        report = ncm.scan(self.root)
        self.assertEqual(sorted(report), ["index.md"])
        self.assertEqual(report["index.md"], [
            (2, f"{LT} HEAD"), (4, MID), (6, f"{GT} branch"),
        ])

    def test_flags_stray_end_marker(self):
        self.track("a.txt", f"x\n{GT} branch\n")
        self.assertEqual(ncm.scan(self.root)["a.txt"], [(2, f"{GT} branch")])

    def test_passes_markdown_setext_headings(self):
        self.track("README.md", "Title\n=======\n\nBody\n\nSection\n=======\n")
        self.assertEqual(ncm.scan(self.root), {})

    def test_skips_binary_files(self):
        self.track("blob.bin", b"\0\x01" + conflict().encode())
        self.assertEqual(ncm.scan(self.root), {})

    def test_skips_files_over_five_megabytes(self):
        self.track("big.txt", conflict() + "x" * (ncm.MAX_BYTES + 1))
        self.assertEqual(ncm.scan(self.root), {})

    def test_ignores_untracked_files(self):
        path = self.root / "scratch.md"
        path.write_text(conflict())
        self.assertEqual(ncm.scan(self.root), {})

    def test_main_exit_status(self):
        self.track("ok.md", "# Fine\n")
        with redirect_stdout(io.StringIO()):
            self.assertEqual(ncm.main(self.root), 0)
        self.track("bad.md", conflict())
        out = io.StringIO()
        with redirect_stdout(out):
            self.assertEqual(ncm.main(self.root), 1)
        self.assertIn("bad.md:2: ", out.getvalue())


if __name__ == "__main__":
    unittest.main()
