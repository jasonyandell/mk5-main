DESCRIPTION = 'One-read state synthesis: my hand, trump hierarchy (high→low, mine/played/unseen), current trick + led suit, bid math, count dominoes still loose, trick history. Pure rule-based, sub-ms. Use to orient before belief_trajectory() / explore_game().'

from burl.wax_museum.snapshot import (
    render_full_board_snapshot,
    render_full_board_structured,
)


def tool(ctx, **kwargs):
    prose = render_full_board_snapshot(ctx.game_state, ctx.me_abs)
    structured = render_full_board_structured(ctx.game_state, ctx.me_abs)
    return {"prose": prose, "structured": structured}
