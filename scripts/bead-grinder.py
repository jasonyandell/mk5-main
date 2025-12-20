#!/usr/bin/env python3
"""
Simple bead grinder - processes beads one by one with live logging.
No worktrees, just runs in the current directory.

USAGE (run from project root):

    # Activate the virtual environment first:
    source .venv/bin/activate

    # Then run the script:
    python scripts/bead-grinder.py

    # When done, deactivate (optional):
    deactivate

ONE-LINER:
    source .venv/bin/activate && python scripts/bead-grinder.py
"""

import asyncio
from claude_agent_sdk import query, ClaudeAgentOptions

BEADS = [
    "mk5-tailwind-f8l",
    "mk5-tailwind-xb4",
    "mk5-tailwind-stg",
    "mk5-tailwind-s6u",
]

async def grind_bead(bead_id: str):
    print(f"\n{'='*60}")
    print(f"🔮 Starting bead: {bead_id}")
    print(f"{'='*60}\n")

    options = ClaudeAgentOptions(
        allowed_tools=["Bash", "Read", "Edit", "Write"],
        permission_mode="acceptEdits",
    )

    async for message in query(
        prompt=f"Complete bead {bead_id}. Run 'bd show {bead_id}' first for context.",
        options=options
    ):
        # Print whatever comes back
        print(message)

    print(f"\n✅ Finished bead: {bead_id}\n")

async def main():
    for bead_id in BEADS:
        await grind_bead(bead_id)
    print("🎉 All beads complete!")

if __name__ == "__main__":
    asyncio.run(main())
