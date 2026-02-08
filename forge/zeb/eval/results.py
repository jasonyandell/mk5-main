"""Match result data types and formatting."""

from __future__ import annotations

import json
from dataclasses import dataclass


@dataclass(frozen=True)
class HalfResult:
    """Results from one team assignment (e.g., A as team 0)."""
    n_games: int
    wins: int
    win_rate: float
    avg_margin: float


@dataclass(frozen=True)
class MatchResult:
    """Full match result with both team assignments."""
    team_a_name: str
    team_b_name: str
    n_games: int
    team_a_wins: int
    team_b_wins: int
    team_a_win_rate: float
    avg_margin: float
    elapsed_s: float
    a_as_team0: HalfResult
    a_as_team1: HalfResult

    @property
    def games_per_sec(self) -> float:
        return self.n_games / self.elapsed_s if self.elapsed_s > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            'team_a': self.team_a_name,
            'team_b': self.team_b_name,
            'n_games': self.n_games,
            'team_a_wins': self.team_a_wins,
            'team_b_wins': self.team_b_wins,
            'team_a_win_rate': self.team_a_win_rate,
            'avg_margin': self.avg_margin,
            'elapsed_s': round(self.elapsed_s, 2),
            'games_per_sec': round(self.games_per_sec, 1),
            'a_as_team0': {
                'n_games': self.a_as_team0.n_games,
                'wins': self.a_as_team0.wins,
                'win_rate': self.a_as_team0.win_rate,
                'avg_margin': self.a_as_team0.avg_margin,
            },
            'a_as_team1': {
                'n_games': self.a_as_team1.n_games,
                'wins': self.a_as_team1.wins,
                'win_rate': self.a_as_team1.win_rate,
                'avg_margin': self.a_as_team1.avg_margin,
            },
        }


def format_result(result: MatchResult, *, json_mode: bool = False) -> str:
    """Format a match result for display."""
    if json_mode:
        return json.dumps(result.to_dict(), indent=2)

    lines = [
        f'{result.team_a_name} vs {result.team_b_name}  '
        f'({result.n_games} games, {result.elapsed_s:.1f}s, '
        f'{result.games_per_sec:.1f} games/s)',
        '',
        f'  {result.team_a_name} as Team 0:  '
        f'{result.a_as_team0.wins}/{result.a_as_team0.n_games} '
        f'({result.a_as_team0.win_rate:.1%})  '
        f'margin {result.a_as_team0.avg_margin:+.1f}',
        f'  {result.team_a_name} as Team 1:  '
        f'{result.a_as_team1.wins}/{result.a_as_team1.n_games} '
        f'({result.a_as_team1.win_rate:.1%})  '
        f'margin {result.a_as_team1.avg_margin:+.1f}',
        '',
        f'  Overall: {result.team_a_name} {result.team_a_win_rate:.1%} '
        f'({result.team_a_wins}/{result.n_games})',
    ]
    return '\n'.join(lines)


def format_matrix(results: list[list[MatchResult | None]], names: list[str],
                  *, json_mode: bool = False) -> str:
    """Format an NxN matrix of results.

    results[i][j] is the match where names[i] played as team A vs names[j].
    Diagonal entries are None.
    """
    if json_mode:
        matrix = {}
        for i, a in enumerate(names):
            for j, b in enumerate(names):
                if i != j and results[i][j] is not None:
                    matrix[f'{a} vs {b}'] = results[i][j].to_dict()
        return json.dumps(matrix, indent=2)

    n = len(names)
    # Find max name width
    name_w = max(len(name) for name in names)
    col_w = 8  # width per column

    # Header
    header = ' ' * (name_w + 2) + ''.join(f'{name:>{col_w}}' for name in names)
    lines = [header]

    for i, name in enumerate(names):
        row = f'{name:>{name_w}}  '
        for j in range(n):
            if i == j:
                row += f'{"---":>{col_w}}'
            elif results[i][j] is not None:
                wr = results[i][j].team_a_win_rate
                row += f'{wr:>{col_w}.1%}'
            else:
                row += f'{"":>{col_w}}'
        lines.append(row)

    return '\n'.join(lines)
