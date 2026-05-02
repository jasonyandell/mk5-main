"""Phase package — registers concrete phases on import.

The Phase Protocol lives in ``burl.lab.core.phase``; this package contains
the concrete phase implementations and registers them with the canonical
``PHASES`` dict on import.
"""

from __future__ import annotations

from burl.lab.core.phase import PHASES, register

from .in_run import IN_RUN
from .post_turn import POST_TURN
from .pre_game import PRE_GAME

register(PRE_GAME)
register(IN_RUN)
register(POST_TURN)


__all__ = ["PHASES", "PRE_GAME", "IN_RUN", "POST_TURN"]
