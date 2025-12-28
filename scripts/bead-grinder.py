#!/usr/bin/env python3
"""
Bead Grinder - Autonomous agent for completing beads.

USAGE:
    source .venv/bin/activate
    python scripts/bead-grinder.py t42-abc
    python scripts/bead-grinder.py t42-abc t42-def t42-ghi  # Sequential processing

FEATURES:
    - Full Claude Code capabilities (all tools, MCP, CLAUDE.md)
    - Detailed logging to scratch/
    - Graceful interrupt with Ctrl+C

NOTE: For parallel/worktree workflows, use grinder beads instead:
    "Grind t42-abc and t42-def together" -> creates a grinder bead
    See CLAUDE.md or .claude/skills/grinder/SKILL.md for details.
"""

import asyncio
import signal
import sys
from datetime import datetime
from pathlib import Path

from claude_agent_sdk import (
    query,
    ClaudeAgentOptions,
    AssistantMessage,
    ResultMessage,
    TextBlock,
    ToolUseBlock,
    ToolResultBlock,
)

# ============================================================================
# Configuration
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.absolute()
SCRATCH_DIR = PROJECT_ROOT / "scratch"

# Default beads to process when no arguments provided
DEFAULT_BEADS = [
    # Codebase review epic: t42-21ze
    "t42-qsg6",  # Fix trick completion assumptions in kernel/view-projection
    "t42-zl13",  # Fix hints layer capability + requiredCapabilities semantics
    "t42-9an8",  # Make actionToId/actionToLabel exhaustive (URL 'unknown' events)
    "t42-umsi",  # Update URL tooling scripts to match current url-compression
    "t42-6hv5",  # Unify action equality/matching logic
    "t42-8s1f",  # Consolidate scoring helpers (avoid duplicate isGameComplete/getWinningTeam)
    "t42-wutc",  # Deduplicate capability builders and tighten playerIndex typing
    "t42-nw1n",  # Deduplicate 'which player executes this action' logic
    "t42-bxxp",  # Remove ad-hoc minimal GameState constructors with magic defaults
    "t42-u5oc",  # Clean up stores (await void, internal client access, get(store))
    "t42-43w4",  # Retire markdown checklist planning workflow (use bd instead)
]

SYSTEM_PROMPT = """You are an autonomous agent completing a task from the beads issue tracker.

## Project: Texas 42 Dominoes
A web implementation with pure functional architecture:
- Event sourcing with immutable state transitions
- Unified Layer system for game rules
- AI strategies (PIMC, Minimax, heuristics)

## Key Locations
- src/game/ai/ - AI strategies, Monte Carlo, hand sampling
- src/game/core/ - Core game logic, dominoes, state
- src/game/layers/ - Rule layers (base, nello, sevens, plunge, etc.)
- src/tests/ - Vitest unit tests and Playwright E2E

## Your Workflow
1. Run `bd show <bead-id>` to understand the task
2. Use TodoWrite to plan and track progress
3. Implement the solution (minimal, focused changes)
4. Run `npm run test:all` to verify
5. Run `bd close <bead-id>` when complete

## Quality Standards
- All tests must pass before closing
- Keep changes minimal - don't over-engineer
- No backwards compatibility code - this is greenfield
- Commit your work before closing the bead

## Important
- You are running autonomously - make decisions confidently
- If truly stuck, close the bead with a failure reason rather than spinning
"""

# ============================================================================
# Agent Runner
# ============================================================================

async def run_bead(bead_id: str) -> tuple[bool, int, float]:
    """Run agent on a single bead. Returns (success, turns, cost_usd)."""

    SCRATCH_DIR.mkdir(exist_ok=True)
    log_path = SCRATCH_DIR / f"bead-{bead_id}-{datetime.now():%Y%m%d-%H%M%S}.log"

    options = ClaudeAgentOptions(
        tools={"type": "preset", "preset": "claude_code"},
        setting_sources=["project"],
        system_prompt=SYSTEM_PROMPT,
        cwd=PROJECT_ROOT,
        permission_mode="bypassPermissions",
        max_turns=100,
    )

    prompt = f"""Complete bead {bead_id}.

Start by running: bd show {bead_id}

Then implement the solution, run tests, and close the bead with bd close."""

    success = False
    turns = 0
    cost = 0.0

    try:
        with open(log_path, "w") as log:
            log.write(f"=== Bead: {bead_id} ===\n")
            log.write(f"Started: {datetime.now().isoformat()}\n\n")

            async for message in query(prompt=prompt, options=options):
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            log.write(f"\n{block.text}\n")
                            # Show progress
                            preview = block.text[:60].replace('\n', ' ')
                            print(f"\r  {preview}...", end="", flush=True)
                        elif isinstance(block, ToolUseBlock):
                            log.write(f"\n→ {block.name}: {_summarize_input(block)}\n")
                            print(f"\r  → {block.name}...{' ' * 40}", end="", flush=True)
                        elif isinstance(block, ToolResultBlock):
                            if block.is_error:
                                log.write(f"  ❌ {str(block.content)[:200]}\n")
                            else:
                                log.write(f"  ✓\n")
                    log.flush()

                elif isinstance(message, ResultMessage):
                    turns = message.num_turns
                    cost = message.total_cost_usd
                    success = not message.is_error
                    log.write(f"\n=== Result ===\n")
                    log.write(f"Success: {success}\n")
                    log.write(f"Turns: {turns}\n")
                    log.write(f"Cost: ${cost:.4f}\n")

    except Exception as e:
        print(f"\n  ❌ Error: {e}")
        success = False

    print()  # Clear progress line
    return success, turns, cost


def _summarize_input(block: ToolUseBlock) -> str:
    """Create a short summary of tool input."""
    name = block.name
    inp = block.input

    if name == "Read":
        return Path(inp.get("file_path", "?")).name
    elif name in ("Edit", "Write"):
        return Path(inp.get("file_path", "?")).name
    elif name == "Bash":
        return inp.get("description", inp.get("command", "?")[:40])
    elif name == "Glob":
        return inp.get("pattern", "?")
    elif name == "Grep":
        return inp.get("pattern", "?")[:30]
    else:
        return ""


# ============================================================================
# Main
# ============================================================================

async def main():
    bead_ids = [a for a in sys.argv[1:] if not a.startswith("-")]

    # Use default beads if none provided
    if not bead_ids:
        bead_ids = DEFAULT_BEADS

    SCRATCH_DIR.mkdir(exist_ok=True)
    grinder_log = SCRATCH_DIR / f"grinder_log_{datetime.now():%Y%m%d-%H%M%S}.txt"

    def log(msg: str):
        """Write to both console and grinder log."""
        print(msg)
        with open(grinder_log, "a") as f:
            f.write(msg + "\n")

    log(f"\n🚀 Bead Grinder")
    log(f"   Started: {datetime.now().isoformat()}")
    log(f"   Beads: {len(bead_ids)} total")
    for i, bead_id in enumerate(bead_ids, 1):
        log(f"     {i}. {bead_id}")
    log("")

    results = []
    for i, bead_id in enumerate(bead_ids, 1):
        log(f"📋 [{i}/{len(bead_ids)}] Processing {bead_id}...")
        success, turns, cost = await run_bead(bead_id)
        results.append((bead_id, success, turns, cost))
        status = "✅" if success else "❌"
        log(f"   {status} {turns} turns, ${cost:.2f}")

    # Summary
    log("")
    log("=" * 50)
    completed = sum(1 for _, s, _, _ in results if s)
    failed = sum(1 for _, s, _, _ in results if not s)
    total_cost = sum(c for _, _, _, c in results)

    if failed:
        log(f"⚠️  {completed} completed, {failed} failed")
        for bead_id, success, _, _ in results:
            if not success:
                log(f"   ❌ {bead_id}")
    else:
        log(f"🎉 All {completed} beads complete!")

    log(f"   Total cost: ${total_cost:.2f}")
    log(f"   Finished: {datetime.now().isoformat()}")
    log("=" * 50)
    log(f"\n📄 Log saved to: {grinder_log}")


if __name__ == "__main__":
    signal.signal(signal.SIGINT, lambda *_: None)
    asyncio.run(main())
