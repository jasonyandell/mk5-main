#!/usr/bin/env bash
# Publish a finished parlor session: raw seat .jsonl logs + curated
# transcript(s) + PROTOCOL.md to the HF evidence dataset, pinned by tag.
# The raw logs are tier-3 run data (wiki/decisions/run-artifacts-policy.md):
# they live here on disk and on HF, never in git. The transcript is the
# curated keepsake: committed to git AND mirrored so the tag is
# self-contained. See .claude/skills/parlor/workflow.md for the full loop.
#
# Usage (from the repo root):  parlor/bin/publish-session.sh parlor-session-<n>
# Requires: hf CLI authenticated (docs/SECRETS.md).
set -euo pipefail
TAG="${1:?usage: publish-session.sh <tag> (e.g. parlor-session-2)}"
REPO=jasonyandell/mk5-run-evidence
[[ -f parlor/PROTOCOL.md ]] || { echo "error: run from the repo root" >&2; exit 1; }
# dirs mirror their repo-relative path; bin/ is code and stays git-only
hf upload "$REPO" parlor parlor --repo-type dataset --exclude "bin/*"
hf repo tag create "$REPO" "$TAG" --repo-type dataset
echo "pinned: https://huggingface.co/datasets/$REPO/tree/$TAG/parlor"
