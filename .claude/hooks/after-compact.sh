#!/bin/sh
# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0
#
# SessionStart (compact): put Lem's framing back in context after every compaction (#193, #202).
#
# A compaction summary keeps the essay's slogans and loses its moves, and no model has the essay
# in its weights, so whatever the summary drops is refilled from the prior. This hook reloads the
# essay's own words (method sections from the local verbatim copy) and the card's positive section
# ("The work"). The copy has no licence: it is read into this local session only, never committed.
#
# Read the session's own checkout (a worktree may be ahead of the main one), else the project's.
cwd=$(sed -n 's/.*"cwd" *: *"\([^"]*\)".*/\1/p')

first_file() {
  for f in "$@"; do [ -f "$f" ] && { echo "$f"; return; }; done
}

rel_essay=scratch/sources/already-clean/essay.md
essay=$(first_file "$cwd/$rel_essay" "$CLAUDE_PROJECT_DIR/$rel_essay" "$HOME/code/mk5-main/$rel_essay")
page=$(first_file "$cwd/wiki/topics/clean-neurons-42.md" "$CLAUDE_PROJECT_DIR/wiki/topics/clean-neurons-42.md")

# One section of a markdown file by its exact heading text, up to the next heading at its level
# or above. String compares only, so BSD and GNU awk agree.
section() {
  awk -v h="$2" '
    function lvl(s,  n) { n = 0; while (substr(s, n + 1, 1) == "#") n++; return (substr(s, n + 1, 1) == " ") ? n : 0 }
    { l = lvl($0) }
    on && l > 0 && l <= want { exit }
    !on && l > 0 && substr($0, l + 2) == h { on = 1; want = l }
    on
  ' "$1"
}

# Standing desks run in tmux sessions named desk-<beat>. Only the Lem desk gets the frame; the
# other desks get one line, so they neither pay for the essay nor take its work as theirs. A
# session outside tmux (a Lem worktree, a lab agent) can't be told apart, so it gets the frame.
desk=$( [ -n "$TMUX_PANE" ] && tmux display-message -p -t "$TMUX_PANE" '#S' 2>/dev/null )
case "$desk" in
  desk-*) [ "$desk" = desk-lem ] || {
    echo "Lem's frame (THE ALREADY-CLEAN TEST, #193) is not reloaded for $desk. If you take up Lem work, read wiki/topics/clean-neurons-42.md and the essay whole first."
    exit 0; } ;;
esac

echo "THE FRAME, reloaded after compaction: Lem's core thesis, THE ALREADY-CLEAN TEST (#193)."
echo "Below are Cammarata's own words, then our card's description of the work. Do the work they"
echo "describe: label a unit, follow its circuit, find the motif, watch it grow, steer it."
echo

if [ -n "$essay" ]; then
  src="$(dirname "$essay")/SOURCE.txt"
  want=$( [ -f "$src" ] && grep -o '[0-9a-f]\{64\}' "$src" | head -n 1 )
  have=$( (sha256sum "$essay" 2>/dev/null || shasum -a 256 "$essay") | cut -c1-64 )
  if [ -n "$want" ] && [ "$want" != "$have" ]; then
    echo "NOTE: $essay does not match the sha256 in SOURCE.txt; check the copy before relying on it."
    echo
  fi
  echo "=== From \"Neural Networks Are Already Clean\" (Nick Cammarata, Goodfire; https://already-clean.goodfire.com/), local copy $essay; not for quoting at length or committing ==="
  for h in "Cars and Animals" "Recursive Generalization" "How I Think About Neurons" \
           "What I Mean by Clean" "Motifs Over Monosemanticity" "What a Solution Might Look Like"; do
    echo
    section "$essay" "$h"
  done
  echo
  echo "=== End of the essay excerpts. Before planning or interpreting a Lem test, read the whole copy (Read with allow_large: true, or page to the end; never WebFetch). ==="
else
  echo "MISSING: the essay's local copy ($rel_essay) is not in this checkout or ~/code/mk5-main."
  echo "Before any Lem work, restore the copy (curl -sL https://already-clean.goodfire.com/essay.md into ~/code/mk5-main/$rel_essay; never WebFetch, it returns a summary) and read it whole,"
  echo "and tell Jason the copy is missing. The card below is our summary, not the source."
fi

if [ -n "$page" ]; then
  echo
  echo "=== From $page ==="
  echo
  section "$page" "The work: what a day of Lem looks like"
fi

echo
echo "To read or audit a unit, use the clean-neuron skill (.claude/skills/clean-neuron). To read the book and the net together, use the book-interp skill (.claude/skills/book-interp)."
