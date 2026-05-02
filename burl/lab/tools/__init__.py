"""First-class ToolSpec values for burl.lab.

Tools are values, not functions: each ToolSpec carries name, description,
params (JSON Schema), an example call, a protocol role, the literal protocol
phrase the system prompt renders, and a callable impl. The active set is
composed at run time and rendered into the prompt by core.render.

Subpackages:
  * ``base`` — the three always-present tools (belief_trajectory,
    explore_game, commit_play).
"""

from burl.lab.tools.base import (
    BELIEF_TRAJECTORY,
    COMMIT_PLAY,
    EXPLORE_GAME,
)

__all__ = ["BELIEF_TRAJECTORY", "COMMIT_PLAY", "EXPLORE_GAME"]
