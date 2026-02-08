"""Tests for the unified eval CLI — all CPU-only, no GPU needed."""

from __future__ import annotations

import json

import pytest

from forge.zeb.eval.players import PlayerSpec, parse_player_spec, KNOWN_KINDS
from forge.zeb.eval.engine import get_dispatch_key
from forge.zeb.eval.results import HalfResult, MatchResult, format_result, format_matrix


# --- TestParsePlayerSpec ---

class TestParsePlayerSpec:
    def test_simple_kinds(self):
        for kind in ('random', 'heuristic'):
            spec = parse_player_spec(kind)
            assert spec.kind == kind
            assert spec.params == {}

    def test_eq_defaults(self):
        spec = parse_player_spec('eq')
        assert spec.kind == 'eq'
        assert spec.params == {'n': '100'}

    def test_eq_custom_n(self):
        spec = parse_player_spec('eq:n=50')
        assert spec.kind == 'eq'
        assert spec.params == {'n': '50'}

    def test_zeb_defaults(self):
        spec = parse_player_spec('zeb')
        assert spec.kind == 'zeb'
        assert spec.params == {'source': 'hf'}

    def test_zeb_custom_source(self):
        spec = parse_player_spec('zeb:source=path/to/model.pt')
        assert spec.kind == 'zeb'
        assert spec.params['source'] == 'path/to/model.pt'

    def test_zeb_weights_name(self):
        spec = parse_player_spec('zeb:source=hf,weights_name=large')
        assert spec.params == {'source': 'hf', 'weights_name': 'large'}

    def test_case_insensitive(self):
        spec = parse_player_spec('Random')
        assert spec.kind == 'random'

    def test_whitespace_tolerance(self):
        spec = parse_player_spec('  eq : n = 50  ')
        assert spec.kind == 'eq'
        assert spec.params == {'n': '50'}

    def test_unknown_kind_raises(self):
        with pytest.raises(ValueError, match="Unknown player kind"):
            parse_player_spec('unknown_player')

    def test_bad_param_format_raises(self):
        with pytest.raises(ValueError, match="Invalid param format"):
            parse_player_spec('eq:no_equals')

    def test_display_name_simple(self):
        assert parse_player_spec('random').display_name == 'random'

    def test_display_name_with_params(self):
        spec = parse_player_spec('eq:n=50')
        assert 'eq' in spec.display_name
        assert 'n=50' in spec.display_name

    def test_all_known_kinds_parseable(self):
        for kind in KNOWN_KINDS:
            spec = parse_player_spec(kind)
            assert spec.kind == kind


# --- TestDispatchSelection ---

class TestDispatchSelection:
    def test_eq_vs_random(self):
        assert get_dispatch_key('eq', 'random') == 'eq_vs_random'

    def test_random_vs_eq_symmetric(self):
        assert get_dispatch_key('random', 'eq') == 'eq_vs_random'

    def test_eq_vs_zeb(self):
        assert get_dispatch_key('eq', 'zeb') == 'eq_vs_zeb'

    def test_zeb_vs_eq_symmetric(self):
        assert get_dispatch_key('zeb', 'eq') == 'eq_vs_zeb'

    def test_zeb_vs_random(self):
        assert get_dispatch_key('zeb', 'random') == 'zeb_vs_random'

    def test_random_vs_zeb_symmetric(self):
        assert get_dispatch_key('random', 'zeb') == 'zeb_vs_random'

    def test_eq_vs_eq(self):
        assert get_dispatch_key('eq', 'eq') == 'eq_vs_eq'

    def test_generic_fallback(self):
        assert get_dispatch_key('random', 'heuristic') == 'generic'
        assert get_dispatch_key('heuristic', 'random') == 'generic'
        assert get_dispatch_key('heuristic', 'heuristic') == 'generic'

    def test_all_pairs_have_dispatch(self):
        """Every pair of known kinds returns a non-empty dispatch key."""
        for a in KNOWN_KINDS:
            for b in KNOWN_KINDS:
                key = get_dispatch_key(a, b)
                assert isinstance(key, str) and len(key) > 0


# --- TestResultFormatting ---

class TestResultFormatting:
    @pytest.fixture
    def sample_result(self):
        return MatchResult(
            team_a_name='random',
            team_b_name='heuristic',
            n_games=100,
            team_a_wins=40,
            team_b_wins=60,
            team_a_win_rate=0.4,
            avg_margin=-2.5,
            elapsed_s=5.0,
            a_as_team0=HalfResult(n_games=50, wins=22, win_rate=0.44, avg_margin=-1.5),
            a_as_team1=HalfResult(n_games=50, wins=18, win_rate=0.36, avg_margin=-3.5),
        )

    def test_human_readable_contains_names(self, sample_result):
        text = format_result(sample_result)
        assert 'random' in text
        assert 'heuristic' in text

    def test_human_readable_contains_stats(self, sample_result):
        text = format_result(sample_result)
        assert '100 games' in text
        assert '40.0%' in text
        assert '44.0%' in text
        assert '36.0%' in text

    def test_human_readable_contains_speed(self, sample_result):
        text = format_result(sample_result)
        assert '20.0 games/s' in text

    def test_json_mode_valid(self, sample_result):
        text = format_result(sample_result, json_mode=True)
        data = json.loads(text)
        assert data['team_a'] == 'random'
        assert data['team_b'] == 'heuristic'
        assert data['n_games'] == 100
        assert data['team_a_wins'] == 40
        assert data['team_a_win_rate'] == 0.4

    def test_json_contains_halves(self, sample_result):
        data = json.loads(format_result(sample_result, json_mode=True))
        assert data['a_as_team0']['wins'] == 22
        assert data['a_as_team1']['wins'] == 18

    def test_matrix_format(self):
        h = HalfResult(n_games=50, wins=25, win_rate=0.5, avg_margin=0.0)
        r_ab = MatchResult(
            team_a_name='A', team_b_name='B',
            n_games=100, team_a_wins=60, team_b_wins=40,
            team_a_win_rate=0.6, avg_margin=1.0, elapsed_s=1.0,
            a_as_team0=h, a_as_team1=h,
        )
        r_ba = MatchResult(
            team_a_name='B', team_b_name='A',
            n_games=100, team_a_wins=40, team_b_wins=60,
            team_a_win_rate=0.4, avg_margin=-1.0, elapsed_s=1.0,
            a_as_team0=h, a_as_team1=h,
        )
        results = [[None, r_ab], [r_ba, None]]
        text = format_matrix(results, ['A', 'B'])
        assert '60.0%' in text
        assert '40.0%' in text
        assert '---' in text

    def test_matrix_json(self):
        h = HalfResult(n_games=50, wins=25, win_rate=0.5, avg_margin=0.0)
        r = MatchResult(
            team_a_name='X', team_b_name='Y',
            n_games=100, team_a_wins=70, team_b_wins=30,
            team_a_win_rate=0.7, avg_margin=3.0, elapsed_s=2.0,
            a_as_team0=h, a_as_team1=h,
        )
        results = [[None, r], [None, None]]
        data = json.loads(format_matrix(results, ['X', 'Y'], json_mode=True))
        assert 'X vs Y' in data
        assert data['X vs Y']['team_a_win_rate'] == 0.7


# --- TestMatchResultNormalization ---

class TestMatchResultNormalization:
    def test_half_result_frozen(self):
        h = HalfResult(n_games=50, wins=25, win_rate=0.5, avg_margin=1.0)
        with pytest.raises(AttributeError):
            h.wins = 30  # type: ignore[misc]

    def test_match_result_frozen(self):
        h = HalfResult(n_games=50, wins=25, win_rate=0.5, avg_margin=0.0)
        r = MatchResult(
            team_a_name='A', team_b_name='B',
            n_games=100, team_a_wins=50, team_b_wins=50,
            team_a_win_rate=0.5, avg_margin=0.0, elapsed_s=1.0,
            a_as_team0=h, a_as_team1=h,
        )
        with pytest.raises(AttributeError):
            r.n_games = 200  # type: ignore[misc]

    def test_games_per_sec(self):
        h = HalfResult(n_games=50, wins=25, win_rate=0.5, avg_margin=0.0)
        r = MatchResult(
            team_a_name='A', team_b_name='B',
            n_games=100, team_a_wins=50, team_b_wins=50,
            team_a_win_rate=0.5, avg_margin=0.0, elapsed_s=4.0,
            a_as_team0=h, a_as_team1=h,
        )
        assert r.games_per_sec == 25.0

    def test_games_per_sec_zero_elapsed(self):
        h = HalfResult(n_games=50, wins=25, win_rate=0.5, avg_margin=0.0)
        r = MatchResult(
            team_a_name='A', team_b_name='B',
            n_games=100, team_a_wins=50, team_b_wins=50,
            team_a_win_rate=0.5, avg_margin=0.0, elapsed_s=0.0,
            a_as_team0=h, a_as_team1=h,
        )
        assert r.games_per_sec == 0.0

    def test_halves_sum_to_total(self):
        h0 = HalfResult(n_games=50, wins=30, win_rate=0.6, avg_margin=2.0)
        h1 = HalfResult(n_games=50, wins=20, win_rate=0.4, avg_margin=-1.0)

        total_wins = h0.wins + h1.wins
        total_games = h0.n_games + h1.n_games
        weighted_margin = (h0.avg_margin * h0.n_games + h1.avg_margin * h1.n_games) / total_games

        r = MatchResult(
            team_a_name='A', team_b_name='B',
            n_games=total_games, team_a_wins=total_wins,
            team_b_wins=total_games - total_wins,
            team_a_win_rate=total_wins / total_games,
            avg_margin=weighted_margin, elapsed_s=1.0,
            a_as_team0=h0, a_as_team1=h1,
        )

        assert r.team_a_wins == h0.wins + h1.wins
        assert r.team_b_wins == (h0.n_games - h0.wins) + (h1.n_games - h1.wins)
        assert r.n_games == h0.n_games + h1.n_games
        assert abs(r.avg_margin - 0.5) < 1e-9

    def test_to_dict_roundtrip(self):
        h = HalfResult(n_games=50, wins=25, win_rate=0.5, avg_margin=1.5)
        r = MatchResult(
            team_a_name='alpha', team_b_name='beta',
            n_games=100, team_a_wins=55, team_b_wins=45,
            team_a_win_rate=0.55, avg_margin=1.5, elapsed_s=3.0,
            a_as_team0=h, a_as_team1=h,
        )
        d = r.to_dict()
        assert d['team_a'] == 'alpha'
        assert d['team_b'] == 'beta'
        assert d['n_games'] == 100
        assert d['a_as_team0']['wins'] == 25
        assert d['a_as_team1']['avg_margin'] == 1.5
        # Verify JSON-serializable
        json.dumps(d)
