#!/usr/bin/env python
# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0

"""wiki_log_rotate.py — stitch wiki/log.d/ fragments into log.md, then roll the
oldest entries into log-archive.md.

Run on main, after merging, never on a branch. A session adds its entry as a
new file wiki/log.d/<UTC timestamp>-<desk>-<slug>.md holding exactly one entry
(`## [date | desk | title]` plus body), so no two sessions edit the same file.
This script is the single writer: it puts the fragments into log.md newest
first (a stable date sort; fragments sit above same-day entries already in
the log, the latest filename first), deletes them, then rotates. A fragment
whose header log.md or the archive already holds is dropped, not duplicated.
A malformed fragment (not exactly one entry) aborts before anything changes.
A fragment added concurrently is simply picked up on the next run.

Rotation, idempotent: with 15 or fewer entries it changes nothing. Past 15 it keeps the
newest 10 (by date; ties keep file order), appends the rest to the archive
oldest first, skips any entry whose header line the archive already holds,
and sets the digest's archived-entry count. It also re-spaces log.md, which a
union merge can leave with a missing blank line.

Folding the moved entries into the digest (one line per phase, links only)
is judgment, so the script prints the moved headers and leaves that to you.

Usage:
    python -u scripts/wiki_log_rotate.py            # rotate if due
    python -u scripts/wiki_log_rotate.py --dry-run  # say what would move
"""

import argparse
import glob
import os
import re
import sys

WIKI = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "wiki")
LOG = os.path.join(WIKI, "log.md")
ARCHIVE = os.path.join(WIKI, "log-archive.md")
FRAGMENTS = os.path.join(WIKI, "log.d")
ROTATE_ABOVE = 15
KEEP = 10

HEADER_RE = re.compile(r"^## \[(\d{4}-\d{2}-\d{2})[^|\n]*\|", re.M)
DIGEST_RE = re.compile(r"^(## Digest \(\S+ → )(\S+)(, )(\d+)( archived entries\))$", re.M)


def split_log(text):
    """(preamble through the '---' after the digest, [entries]) — each entry is
    its header line and body, stripped."""
    first = HEADER_RE.search(text)
    if first is None:
        return text.rstrip() + "\n", []
    preamble = text[: first.start()]
    starts = [m.start() for m in HEADER_RE.finditer(text)]
    entries = [text[a:b].strip() for a, b in zip(starts, starts[1:] + [len(text)])]
    return preamble.rstrip() + "\n", entries


def date_of(entry):
    return HEADER_RE.match(entry).group(1)


def header_of(entry):
    return entry.split("\n", 1)[0]


def read_fragments(frag_dir):
    """([(path, entry)] newest filename first, [error strings]). A fragment is
    well-formed when it holds exactly one entry that starts with its header."""
    good, errors = [], []
    for path in sorted(glob.glob(os.path.join(frag_dir, "*.md")), reverse=True):
        name = os.path.basename(path)
        with open(path) as fh:
            text = fh.read().strip()
        n = len(HEADER_RE.findall(text))
        if not HEADER_RE.match(text):
            errors.append(f"{name}: does not start with '## [YYYY-MM-DD | desk | title]'")
        elif n != 1:
            errors.append(f"{name}: holds {n} entries; one per file")
        elif not re.match(r"^## \[.*\]\s*$", text.split("\n", 1)[0]):
            errors.append(f"{name}: header line is not closed with ']'")
        else:
            good.append((path, text))
    return good, errors


def rotate(wiki, dry_run=False):
    log, archive_path = os.path.join(wiki, "log.md"), os.path.join(wiki, "log-archive.md")
    frags, errors = read_fragments(os.path.join(wiki, "log.d"))
    if errors:
        sys.exit("wiki_log_rotate: malformed fragment(s), nothing changed:\n  " + "\n  ".join(errors))

    with open(log) as fh:
        preamble, entries = split_log(fh.read())
    with open(archive_path) as fh:
        archive = fh.read()
    archived = {header_of(e) for e in split_log(archive)[1]}

    held = {header_of(e) for e in entries} | archived
    stitched = []
    for _, e in frags:
        if header_of(e) not in held:
            held.add(header_of(e))
            stitched.append(e)
    print(f"log.d: {len(frags)} fragment(s); stitching {len(stitched)}")
    entries = stitched + entries

    # Newest first; sorted() is stable, so same-day entries keep order.
    entries = sorted(entries, key=date_of, reverse=True)
    if len(entries) > ROTATE_ABOVE:
        keep, moved = entries[:KEEP], entries[KEEP:][::-1]  # moved: oldest first
    else:
        keep, moved = entries, []
    fresh = [e for e in moved if header_of(e) not in archived]

    total = len(archived) + len(fresh)
    newest_archived = max([date_of(e) for e in split_log(archive)[1]] + [date_of(e) for e in fresh], default=None)
    m = DIGEST_RE.search(preamble)
    if m is None:
        sys.exit("wiki_log_rotate: log.md has no '## Digest (… → …, N archived entries)' line")
    if newest_archived is not None:
        preamble = DIGEST_RE.sub(lambda g: f"{g[1]}{newest_archived}{g[3]}{total}{g[5]}", preamble, count=1)

    print(f"log.md: {len(entries)} entries; keeping {len(keep)}; "
          f"moving {len(moved)} ({len(moved) - len(fresh)} already archived); archive now {total}")
    for e in fresh:
        print(f"  -> {header_of(e)[:110]}")
    if dry_run:
        return

    with open(log, "w") as fh:
        fh.write(preamble + "\n" + "\n\n".join(keep) + "\n")
    if fresh:
        with open(archive_path, "w") as fh:
            fh.write(archive.rstrip() + "\n\n" + "\n\n".join(fresh) + "\n")
        print("Fold the moved entries into log.md's digest (one line per phase, links only).")
    for path, _ in frags:
        os.remove(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    rotate(WIKI, args.dry_run)


if __name__ == "__main__":
    main()
