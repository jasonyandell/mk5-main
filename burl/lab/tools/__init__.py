"""First-class ToolSpec values for burl.lab.

Tools are values, not functions: each ToolSpec carries name, description,
params (JSON Schema), an example call, a protocol role, the literal protocol
phrase the system prompt renders, and a callable impl. The active set is
composed at run time and rendered into the prompt by core.render.

Subpackages:
  * ``base`` — the three always-present tools (belief_trajectory,
    explore_game, commit_play).
  * ``chat_mined`` — first-class ToolSpecs wrapping useful improvised tools
    from the burl-chat spike.
"""

from burl.lab.tools.base import (
    BELIEF_TRAJECTORY,
    COMMIT_PLAY,
    EXPLORE_GAME,
)
from burl.lab.tools.chat_mined import (
    BOARD_SNAPSHOT,
    CHAT_MINED_TOOLS,
    LEGAL_PLAYS,
    PLAY_BRIEF,
    STATE_BRIEF,
)
from burl.lab.tools.hand_hypothesis import SIMULATE_HAND_IMPACT

__all__ = [
    "BELIEF_TRAJECTORY",
    "BOARD_SNAPSHOT",
    "CHAT_MINED_TOOLS",
    "COMMIT_PLAY",
    "EXPLORE_GAME",
    "LEGAL_PLAYS",
    "PLAY_BRIEF",
    "SIMULATE_HAND_IMPACT",
    "STATE_BRIEF",
]
