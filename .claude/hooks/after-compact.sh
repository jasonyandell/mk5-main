#!/bin/sh
# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0
#
# SessionStart (compact): put Lem's framing back in context after every compaction,
# from the wiki's own card, so a summary cannot paraphrase it away (#193, #202).
page="$CLAUDE_PROJECT_DIR/wiki/topics/clean-neurons-42.md"
[ -f "$page" ] || exit 0
echo "THE FRAME, reloaded after compaction (Cammarata, \"Neural Networks Are Already Clean\", https://already-clean.goodfire.com/; Lem's core thesis, THE ALREADY-CLEAN TEST #193). From $page:"
echo
awk '/^## Re-read /{on=1} /^## Why it fits 42/{on=0} on' "$page"
