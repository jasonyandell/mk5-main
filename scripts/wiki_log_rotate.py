#!/usr/bin/env python
# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0

"""wiki_log_rotate.py — roll the oldest wiki/log.md entries into log-archive.md.

Run on main, after merging, never on a branch. Branches only add their entry
to the top of log.md; git merges those additions with the union driver
(.gitattributes), so parallel branches never conflict over the log. Rotation
is the one step that moves entries between files, so it has a single writer.

Idempotent: with 15 or fewer entries it changes nothing. Past 15 it keeps the
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
import os
import re
import sys

WIKI = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "wiki")
LOG = os.path.join(WIKI, "log.md")
ARCHIVE = os.path.join(WIKI, "log-archive.md")
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    with open(LOG) as fh:
        preamble, entries = split_log(fh.read())
    with open(ARCHIVE) as fh:
        archive = fh.read()
    archived = {header_of(e) for e in split_log(archive)[1]}

    # Newest first; sorted() is stable, so same-day entries keep file order.
    entries = sorted(entries, key=date_of, reverse=True)
    if len(entries) > ROTATE_ABOVE:
        keep, moved = entries[:KEEP], entries[KEEP:][::-1]  # moved: oldest first
    else:
        keep, moved = entries, []
    fresh = [e for e in moved if header_of(e) not in archived]

    total = len(archived) + len(fresh)
    newest_archived = max([date_of(e) for e in split_log(archive)[1]] + [date_of(e) for e in fresh])
    m = DIGEST_RE.search(preamble)
    if m is None:
        sys.exit("wiki_log_rotate: log.md has no '## Digest (… → …, N archived entries)' line")
    preamble = DIGEST_RE.sub(lambda g: f"{g[1]}{newest_archived}{g[3]}{total}{g[5]}", preamble, count=1)

    print(f"log.md: {len(entries)} entries; keeping {len(keep)}; "
          f"moving {len(moved)} ({len(moved) - len(fresh)} already archived); archive now {total}")
    for e in fresh:
        print(f"  -> {header_of(e)[:110]}")
    if args.dry_run:
        return

    with open(LOG, "w") as fh:
        fh.write(preamble + "\n" + "\n\n".join(keep) + "\n")
    if fresh:
        with open(ARCHIVE, "w") as fh:
            fh.write(archive.rstrip() + "\n\n" + "\n\n".join(fresh) + "\n")
        print("Fold the moved entries into log.md's digest (one line per phase, links only).")


if __name__ == "__main__":
    main()
