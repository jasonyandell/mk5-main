---
name: analyze-hand
description: Analyze a Texas 42 hand with gus and publish to https://gus-hands.pages.dev. Activates on /analyze-hand. Takes 7 compact tile tokens like "66 55 44 33 22 11 00" or "63 54 43 51 31 21 11" — each 2-char token is (high-pip, low-pip) regardless of order (e.g., "35" and "53" both mean 5-3). Runs `scratch/fam_apr23/add_hand.py` which invokes gus, writes charts + bidding.json, regenerates the landing grid, and deploys via wrangler.
---

# /analyze-hand

Invoked as `/analyze-hand TOKEN TOKEN TOKEN TOKEN TOKEN TOKEN TOKEN [--title "label"]`.

Each TOKEN is two digits 0-6; interpret the pair as an unordered domino and canonicalize to `high-low`. Examples:
- `66` → `6-6`
- `53` → `5-3`
- `35` → `5-3`  (same tile, different arg order)
- `00` → `0-0`

## Steps

1. **Parse the args**. Split the skill args on whitespace into exactly 7 tokens. For each token:
   - It must be 2 characters, both digits 0-6. Reject `7`, `8`, `9` with a clear message.
   - Parse `(a, b) = (int(tok[0]), int(tok[1]))`.
   - Canonicalize: `hi, lo = max(a, b), min(a, b)`. Emit `f"{hi}-{lo}"`.
2. **Validate**: exactly 7 canonicalized tiles, no duplicates. If dupes, tell the user which one and stop.
3. **Build the hand spec**: `",".join(canonical_tiles)`.
4. **Optional title**: if the user passed `--title "..."` take that; otherwise generate `hand-NNN` using the next slot (the script picks its own slot, so a safe default is just the raw token string joined with spaces).
5. **Run the pipeline**:
   ```bash
   /Users/jason/code/mk5-main/.venv/bin/python \
     /Users/jason/code/mk5-main/scratch/fam_apr23/add_hand.py \
     "<HAND_SPEC>" --title "<TITLE>" --deploy
   ```
   Run from the repo root `/Users/jason/code/mk5-main`. Stream the output.
6. **When it finishes**, report back concisely:
   - Best declaration (from stdout: `gus's declaration pick`)
   - Best bid (from stdout: `gus's best bid`)
   - Hand slot number (infer from the script's `✓ added hand NNN` line)
   - Link: `https://gus-hands.pages.dev/hands/NNN/` and landing `https://gus-hands.pages.dev/`

## Response style

Keep the user-facing message tight — one sentence of acknowledgement, then let the pipeline stream, then a compact result block at the end. No preamble, no task lists. This is a family-demo skill; speed and the final URL are what matters.

## Failure modes

- **Bad tokens** (letters, length≠2, digit>6): refuse up front, show the user which tokens were bad, don't invoke the script.
- **Duplicate tile**: refuse, name the dupe.
- **Script failure**: surface stderr and the exit code, don't pretend it worked.
- **Deploy failure but analysis succeeded**: tell the user the analysis URL isn't live yet and suggest rerunning the deploy step alone (`add_hand.py --deploy`).

## Reference

Pipeline files:
- `scratch/fam_apr23/add_hand.py` — orchestrator
- `scratch/fam_apr23/analyze_hand.py` — gus inference + bidding + charts
- `gus/bidding/` — self-play P(make) module
- `gus/adapters/v2_voids_3000g_big.pt` — best adapter (hardcoded)

Already-invoked deploy goes to the `main` branch on the `gus-hands` Cloudflare Pages project.
