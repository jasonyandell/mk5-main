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
from claude_agent_sdk import (
    query,
    ClaudeAgentOptions,
    AssistantMessage,
    ResultMessage,
    TextBlock,
    ToolUseBlock,
)

BEADS = [
    "t42-f26",   # Delete Perfects feature
    "t42-lq0",   # Rules-base + layer rewiring
    "t42-mtq",   # Core cleanup: rule logic to GameRules
    "t42-3xp7",  # Server-owned projection; dumb client
    "t42-20ue",  # Remove suitAnalysis cache from state
    "t42-r3ia",  # Guardrails and contract tests
    "t42-qnwe",  # Follow-through: cleanup and assessment
]

def print_message(message):
    """Print messages in a readable format."""
    if isinstance(message, AssistantMessage):
        for block in message.content:
            if isinstance(block, TextBlock):
                print(block.text)
            elif isinstance(block, ToolUseBlock):
                print(f"  → {block.name}")
    elif isinstance(message, ResultMessage):
        status = "✅" if not message.is_error else "❌"
        print(f"\n{status} Done in {message.num_turns} turns (${message.total_cost_usd:.4f})")

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
        print_message(message)

async def main():
    for bead_id in BEADS:
        await grind_bead(bead_id)
    print("\n🎉 All beads complete!")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⏹️  Stopped.")
