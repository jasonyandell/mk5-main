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

LOGS:
    scratch/logs/bead-{id}.log     - per-bead transcript
    scratch/logs/grind-summary.json - run summary
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path
from claude_agent_sdk import (
    query,
    ClaudeAgentOptions,
    AssistantMessage,
    ResultMessage,
    TextBlock,
    ToolUseBlock,
)

BEADS = [
    # Crystal Forge: Lightning-First Architecture (t42-4cp6)
    "t42-4cp6.1",  # Lightning Bead 1: Oracle Lift
    "t42-4cp6.2",  # Lightning Bead 2: LightningModule + DataModule
    "t42-4cp6.3",  # Lightning Bead 3: Tokenization Pipeline
    "t42-4cp6.4",  # Lightning Bead 4: Training CLI + Wandb
    "t42-4cp6.5",  # Lightning Bead 5: Golden Path + Cleanup
]

LOG_DIR = Path("scratch/logs")


class TeeWriter:
    """Write to both stdout and a file."""

    def __init__(self, log_file: Path):
        self.log_file = log_file
        self.file = open(log_file, "w")

    def write(self, text: str):
        print(text, end="")
        self.file.write(text)
        self.file.flush()

    def writeln(self, text: str = ""):
        self.write(text + "\n")

    def close(self):
        self.file.close()


def format_message(message) -> str:
    """Format a message as a string."""
    lines = []
    if isinstance(message, AssistantMessage):
        for block in message.content:
            if isinstance(block, TextBlock):
                lines.append(block.text)
            elif isinstance(block, ToolUseBlock):
                lines.append(f"  → {block.name}")
    elif isinstance(message, ResultMessage):
        status = "✅" if not message.is_error else "❌"
        lines.append(f"\n{status} Done in {message.num_turns} turns (${message.total_cost_usd:.4f})")
    return "\n".join(lines)


async def grind_bead(bead_id: str, summary: dict) -> dict:
    """Process a single bead, return result info."""
    log_file = LOG_DIR / f"bead-{bead_id.replace('.', '-')}.log"
    tee = TeeWriter(log_file)

    start_time = datetime.now()
    result = {
        "bead_id": bead_id,
        "log_file": str(log_file),
        "started_at": start_time.isoformat(),
        "status": "running",
    }

    tee.writeln(f"\n{'='*60}")
    tee.writeln(f"🔮 Starting bead: {bead_id}")
    tee.writeln(f"   Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    tee.writeln(f"   Log: {log_file}")
    tee.writeln(f"{'='*60}\n")

    options = ClaudeAgentOptions(
        allowed_tools=["Bash", "Read", "Edit", "Write", "Glob", "Grep"],
        permission_mode="acceptEdits",
    )

    final_message = None
    try:
        async for message in query(
            prompt=f"Complete bead {bead_id}. Run 'bd show {bead_id}' first for context. When done, run 'bd close {bead_id}' with a brief reason.",
            options=options
        ):
            text = format_message(message)
            if text:
                tee.writeln(text)
            if isinstance(message, ResultMessage):
                final_message = message

        if final_message:
            result["status"] = "success" if not final_message.is_error else "error"
            result["turns"] = final_message.num_turns
            result["cost_usd"] = final_message.total_cost_usd
        else:
            result["status"] = "unknown"

    except Exception as e:
        result["status"] = "exception"
        result["error"] = str(e)
        tee.writeln(f"\n❌ Exception: {e}")

    end_time = datetime.now()
    result["ended_at"] = end_time.isoformat()
    result["duration_seconds"] = (end_time - start_time).total_seconds()

    tee.close()
    return result


def print_summary(summary: dict):
    """Print a nice summary table."""
    print("\n" + "="*70)
    print("📊 GRIND SUMMARY")
    print("="*70)
    print(f"Started:  {summary['started_at']}")
    print(f"Finished: {summary['ended_at']}")
    print(f"Duration: {summary['total_duration_seconds']:.1f}s")
    print()

    # Table header
    print(f"{'Bead':<20} {'Status':<10} {'Turns':<8} {'Cost':<10} {'Duration':<10}")
    print("-"*70)

    total_cost = 0.0
    total_turns = 0
    success_count = 0

    for r in summary["beads"]:
        status_icon = "✅" if r["status"] == "success" else "❌"
        turns = r.get("turns", "-")
        cost = f"${r.get('cost_usd', 0):.4f}"
        duration = f"{r.get('duration_seconds', 0):.1f}s"

        print(f"{r['bead_id']:<20} {status_icon} {r['status']:<7} {str(turns):<8} {cost:<10} {duration:<10}")

        if r["status"] == "success":
            success_count += 1
        total_cost += r.get("cost_usd", 0)
        total_turns += r.get("turns", 0)

    print("-"*70)
    print(f"{'TOTAL':<20} {success_count}/{len(summary['beads'])} ok    {total_turns:<8} ${total_cost:.4f}")
    print()

    # List any failures
    failures = [r for r in summary["beads"] if r["status"] != "success"]
    if failures:
        print("❌ FAILED BEADS:")
        for r in failures:
            print(f"   - {r['bead_id']}: {r['status']}")
            if "error" in r:
                print(f"     Error: {r['error']}")
            print(f"     Log: {r['log_file']}")
        print()
    else:
        print("✅ All beads completed successfully!\n")


async def main():
    # Ensure log directory exists
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    summary = {
        "started_at": datetime.now().isoformat(),
        "beads": [],
    }

    for bead_id in BEADS:
        result = await grind_bead(bead_id, summary)
        summary["beads"].append(result)

        # Save summary after each bead (in case of crash)
        summary["ended_at"] = datetime.now().isoformat()
        summary["total_duration_seconds"] = (
            datetime.fromisoformat(summary["ended_at"]) -
            datetime.fromisoformat(summary["started_at"])
        ).total_seconds()

        summary_file = LOG_DIR / "grind-summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

    # Print final summary
    print_summary(summary)

    # List all log files for easy copy/paste
    print("📁 Log files:")
    for r in summary["beads"]:
        print(f"   {r['log_file']}")
    print(f"   {LOG_DIR}/grind-summary.json")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⏹️  Stopped. Check scratch/logs/grind-summary.json for progress.")
