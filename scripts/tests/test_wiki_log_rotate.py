# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0

"""scripts/wiki_log_rotate.py: stitching wiki/log.d/ fragments into log.md, and the
lint's view of pending fragments. Every test builds its own temporary wiki."""

import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import wiki_log_rotate as wlr  # noqa: E402

PREAMBLE = "# Log\n\n## Digest (2026-04-09 → {newest}, {n} archived entries)\n\nmap\n\n---\n"


def entry(date, title, desk="desk"):
    return f"## [{date} | {desk} | {title}]\n\n**Added:** {title}"


class StitchTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.wiki = Path(self.tmp.name)
        (self.wiki / "log.d").mkdir()
        self.archive = [entry("2026-04-09", "old")]
        self.write_log([])

    def write_log(self, entries):
        text = PREAMBLE.format(newest="2026-04-09", n=len(self.archive))
        (self.wiki / "log.md").write_text(text + "\n" + "\n\n".join(entries) + "\n")
        (self.wiki / "log-archive.md").write_text("# Archive\n\n" + "\n\n".join(self.archive) + "\n")

    def frag(self, name, text):
        (self.wiki / "log.d" / name).write_text(text + "\n")

    def run_rotate(self, dry=False):
        with redirect_stdout(io.StringIO()):
            wlr.rotate(str(self.wiki), dry)

    def headers(self):
        _, es = wlr.split_log((self.wiki / "log.md").read_text())
        return [wlr.header_of(e) for e in es]

    def left(self):
        return sorted(p.name for p in (self.wiki / "log.d").glob("*.md"))

    def test_ordering_newest_first(self):
        self.write_log([entry("2026-10-10", "existing")])
        self.frag("20261011T090000Z-a-early.md", entry("2026-10-11", "early"))
        self.frag("20261011T100000Z-b-late.md", entry("2026-10-11", "late"))
        self.frag("20261012T080000Z-c-newest.md", entry("2026-10-12", "newest"))
        self.run_rotate()
        self.assertEqual([h.split("| ")[-1] for h in self.headers()],
                         ["newest]", "late]", "early]", "existing]"])
        self.assertEqual(self.left(), [])

    def test_idempotent(self):
        self.write_log([entry("2026-10-10", "existing")])
        self.frag("20261011T090000Z-a-x.md", entry("2026-10-11", "x"))
        self.run_rotate()
        first = ((self.wiki / "log.md").read_text(), (self.wiki / "log-archive.md").read_text())
        self.run_rotate()
        self.assertEqual(first, ((self.wiki / "log.md").read_text(), (self.wiki / "log-archive.md").read_text()))

    def test_already_stitched_fragment_not_duplicated(self):
        self.write_log([entry("2026-10-10", "existing")])
        self.frag("20261010T090000Z-a-existing.md", entry("2026-10-10", "existing"))
        self.run_rotate()
        self.assertEqual(len(self.headers()), 1)
        self.assertEqual(self.left(), [])

    def test_empty_directory_is_a_no_op(self):
        self.write_log([entry("2026-10-10", "existing")])
        self.run_rotate()
        self.assertEqual(len(self.headers()), 1)
        (self.wiki / "log.d").rmdir()  # a missing directory is the same as an empty one
        self.run_rotate()
        self.assertEqual(len(self.headers()), 1)

    def test_malformed_fragment_aborts_and_changes_nothing(self):
        self.write_log([entry("2026-10-10", "existing")])
        before = (self.wiki / "log.md").read_text()
        self.frag("20261011T090000Z-a-good.md", entry("2026-10-11", "good"))
        for name, text in [("20261011T100000Z-b-none.md", "just prose"),
                           ("20261011T110000Z-c-two.md", entry("2026-10-11", "p") + "\n\n" + entry("2026-10-11", "q"))]:
            self.frag(name, text)
            with self.assertRaises(SystemExit):
                self.run_rotate()
            self.assertEqual((self.wiki / "log.md").read_text(), before)
            self.assertIn("20261011T090000Z-a-good.md", self.left())
            (self.wiki / "log.d" / name).unlink()

    def test_rotation_after_stitching(self):
        live = [entry(f"2026-10-{d:02d}", f"e{d}") for d in range(15, 1, -1)]  # 14 entries
        self.write_log(live)
        self.frag("20261016T090000Z-a-n1.md", entry("2026-10-16", "n1"))
        self.frag("20261017T090000Z-a-n2.md", entry("2026-10-17", "n2"))  # 16 total: rotation due
        self.run_rotate()
        hs = self.headers()
        self.assertEqual(len(hs), wlr.KEEP)
        self.assertTrue(hs[0].endswith("n2]") and hs[1].endswith("n1]"))
        log = (self.wiki / "log.md").read_text()
        self.assertIn("(2026-04-09 → 2026-10-07, 7 archived entries)", log)
        arch = (self.wiki / "log-archive.md").read_text()
        self.assertEqual(arch.count("## ["), 7)
        self.assertEqual(self.left(), [])
        before = (log, arch)
        self.run_rotate()
        self.assertEqual(before, ((self.wiki / "log.md").read_text(), (self.wiki / "log-archive.md").read_text()))

    def test_dry_run_leaves_fragments(self):
        self.frag("20261011T090000Z-a-x.md", entry("2026-10-11", "x"))
        self.run_rotate(dry=True)
        self.assertEqual(len(self.left()), 1)
        self.assertEqual(self.headers(), [])


if __name__ == "__main__":
    unittest.main()
