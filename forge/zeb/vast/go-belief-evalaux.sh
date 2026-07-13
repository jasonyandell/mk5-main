#!/usr/bin/env bash
# go-belief-evalaux.sh — Compatibility wrapper.
# Usage: ./go-belief-evalaux.sh [N_WORKERS] [--dry-run]
#
# Prefer ./go-experiment.sh for experiment-scoped launches.
exec "$(dirname "$0")/go-experiment.sh" "$@"
