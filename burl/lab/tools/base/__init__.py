"""Base tools — the three ToolSpecs that are always in the active set.

These wrap reference implementations from ``burl.wax_museum.tools`` (read-only)
and ``burl.tools.belief_trajectory``. The wax_museum callables are not edited;
each impl here is a thin adapter that produces a ``ToolResult`` carrying both
prose (for the model) and structured evidence (for trace analysis).
"""

from burl.lab.tools.base.belief_trajectory import BELIEF_TRAJECTORY
from burl.lab.tools.base.commit_play import COMMIT_PLAY
from burl.lab.tools.base.explore_game import EXPLORE_GAME

__all__ = ["BELIEF_TRAJECTORY", "COMMIT_PLAY", "EXPLORE_GAME"]
