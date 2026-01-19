"""Game outcome helpers for E[Q] dataset generation."""

from __future__ import annotations

from forge.eq.game import GameState
from forge.eq.types import DecisionRecord
from forge.oracle.tables import resolve_trick


def _fill_actual_outcomes(
    decisions: list[DecisionRecord],
    final_game: GameState,
    decl_id: int,
) -> None:
    """Compute actual_outcome for each decision via backward pass.

    After game completes, walk through play history and compute the actual
    margin (my team - opponent team) from each decision point to end of game.
    """
    play_history = final_game.play_history
    if len(play_history) != 28:
        return

    team_scores_at_decision: list[tuple[int, int]] = []
    team_scores = [0, 0]

    current_trick_dominoes: list[int] = []
    current_trick_players: list[int] = []

    for player, domino_id, lead_domino_id in play_history:
        team_scores_at_decision.append(tuple(team_scores))

        current_trick_dominoes.append(domino_id)
        current_trick_players.append(player)

        if len(current_trick_dominoes) == 4:
            outcome = resolve_trick(
                lead_domino_id, tuple(current_trick_dominoes), decl_id
            )
            trick_leader = current_trick_players[0]
            winner = (trick_leader + outcome.winner_offset) % 4
            winner_team = winner % 2
            team_scores[winner_team] += outcome.points

            current_trick_dominoes = []
            current_trick_players = []

    final_scores = tuple(team_scores)

    for i, decision in enumerate(decisions):
        player = decision.player
        team = player % 2
        opp_team = 1 - team

        my_before = team_scores_at_decision[i][team]
        opp_before = team_scores_at_decision[i][opp_team]
        my_final = final_scores[team]
        opp_final = final_scores[opp_team]

        actual_margin = (my_final - opp_final) - (my_before - opp_before)
        decision.actual_outcome = float(actual_margin)

