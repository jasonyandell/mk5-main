"""Core evaluation engine with dispatch to optimal eval paths."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from .players import PlayerSpec, build_player
from .results import HalfResult, MatchResult


@dataclass
class MatchConfig:
    """Configuration for a match evaluation."""
    spec_a: PlayerSpec
    spec_b: PlayerSpec
    n_games: int = 1000
    device: str = 'cuda'
    seed: int = 0
    batch_size: int = 0
    quiet: bool = False

    # Model cache shared across matrix matchups: keyed by (kind, params_tuple)
    model_cache: dict = field(default_factory=dict, repr=False)


def run_match(config: MatchConfig) -> MatchResult:
    """Run a match between two player specs.

    Always runs both team assignments (n/2 each) for team-symmetric evaluation.
    Dispatches to the optimal batched path when available.
    """
    half = config.n_games // 2
    t0 = time.time()

    a, b = config.spec_a, config.spec_b
    pair = (a.kind, b.kind)
    verbose = not config.quiet

    # Dispatch to optimal path
    if pair == ('eq', 'random') or pair == ('random', 'eq'):
        r0, r1 = _run_eq_vs_simple(config, half, opponent='random')
    elif pair == ('eq', 'zeb') or pair == ('zeb', 'eq'):
        r0, r1 = _run_eq_vs_zeb(config, half)
    elif pair == ('eq', 'eq'):
        r0, r1 = _run_eq_vs_eq(config, half)
    elif pair == ('zeb', 'random') or pair == ('random', 'zeb'):
        r0, r1 = _run_zeb_vs_random(config, half)
    else:
        r0, r1 = _run_generic(config, half)

    elapsed = time.time() - t0
    total_a_wins = r0.wins + r1.wins
    total_games = r0.n_games + r1.n_games
    total_margin = (r0.avg_margin * r0.n_games + r1.avg_margin * r1.n_games) / total_games

    return MatchResult(
        team_a_name=a.display_name,
        team_b_name=b.display_name,
        n_games=total_games,
        team_a_wins=total_a_wins,
        team_b_wins=total_games - total_a_wins,
        team_a_win_rate=total_a_wins / total_games,
        avg_margin=total_margin,
        elapsed_s=elapsed,
        a_as_team0=r0,
        a_as_team1=r1,
    )


def get_dispatch_key(kind_a: str, kind_b: str) -> str:
    """Return the dispatch path name for a player pair. Useful for testing."""
    pair = (kind_a, kind_b)
    if pair == ('eq', 'random') or pair == ('random', 'eq'):
        return 'eq_vs_random'
    elif pair == ('eq', 'zeb') or pair == ('zeb', 'eq'):
        return 'eq_vs_zeb'
    elif pair == ('eq', 'eq'):
        return 'eq_vs_eq'
    elif pair == ('zeb', 'random') or pair == ('random', 'zeb'):
        return 'zeb_vs_random'
    else:
        return 'generic'


def _log(config: MatchConfig, msg: str):
    if not config.quiet:
        print(msg)


def _get_cached_model(config: MatchConfig, spec: PlayerSpec, loader, *args, **kwargs):
    """Load or retrieve a cached model."""
    key = (spec.kind, tuple(sorted(spec.params.items())))
    if key not in config.model_cache:
        config.model_cache[key] = loader(*args, **kwargs)
    return config.model_cache[key]


# --- Dispatch implementations ---

def _run_eq_vs_simple(config: MatchConfig, half: int, opponent: str) -> tuple[HalfResult, HalfResult]:
    """E[Q] vs random using the optimized batched path."""
    from ..eq_player import evaluate_eq_vs_random
    from .loading import load_oracle, DEFAULT_ORACLE

    a, b = config.spec_a, config.spec_b
    eq_spec = a if a.kind == 'eq' else b

    checkpoint = eq_spec.params.get('checkpoint', DEFAULT_ORACLE)
    n_samples = int(eq_spec.params.get('n', '100'))

    _log(config, f'  Loading oracle: {checkpoint}')
    oracle = _get_cached_model(config, eq_spec, load_oracle, checkpoint, config.device)

    _log(config, f'  E[Q](n={n_samples}) vs random ({config.n_games} games)...')
    results = evaluate_eq_vs_random(
        oracle,
        n_games=config.n_games,
        n_samples=n_samples,
        device=config.device,
        verbose=not config.quiet,
    )

    # The eq_player returns results from E[Q]'s perspective.
    # If eq is spec_a, these map directly. If eq is spec_b, we flip.
    eq_is_a = (a.kind == 'eq')
    return _halfs_from_eq_results(results, eq_is_a=eq_is_a)


def _run_eq_vs_zeb(config: MatchConfig, half: int) -> tuple[HalfResult, HalfResult]:
    """E[Q] vs Zeb using the optimized batched path."""
    from ..eq_player import evaluate_eq_vs_zeb
    from .loading import load_oracle, load_zeb, DEFAULT_ORACLE

    a, b = config.spec_a, config.spec_b
    eq_spec = a if a.kind == 'eq' else b
    zeb_spec = a if a.kind == 'zeb' else b

    checkpoint = eq_spec.params.get('checkpoint', DEFAULT_ORACLE)
    n_samples = int(eq_spec.params.get('n', '100'))
    zeb_source = zeb_spec.params.get('source', 'hf')

    _log(config, f'  Loading oracle: {checkpoint}')
    oracle = _get_cached_model(config, eq_spec, load_oracle, checkpoint, config.device)

    _log(config, f'  Loading Zeb: {zeb_source}')
    zeb_kwargs = {}
    if 'weights_name' in zeb_spec.params:
        zeb_kwargs['weights_name'] = zeb_spec.params['weights_name']
    zeb_model = _get_cached_model(
        config, zeb_spec, load_zeb, zeb_source, config.device, **zeb_kwargs,
    )

    _log(config, f'  E[Q](n={n_samples}) vs Zeb ({config.n_games} games)...')
    results = evaluate_eq_vs_zeb(
        oracle, zeb_model,
        n_games=config.n_games,
        n_samples=n_samples,
        device=config.device,
        batch_size=config.batch_size,
        verbose=not config.quiet,
    )

    eq_is_a = (a.kind == 'eq')
    return _halfs_from_eq_results(results, eq_is_a=eq_is_a)


def _run_eq_vs_eq(config: MatchConfig, half: int) -> tuple[HalfResult, HalfResult]:
    """E[Q] vs E[Q] using the batched path (both sides share the oracle)."""
    from ..eq_player import evaluate_eq_vs_eq
    from .loading import load_oracle, DEFAULT_ORACLE

    a, b = config.spec_a, config.spec_b
    n_samples_a = int(a.params.get('n', '100'))
    n_samples_b = int(b.params.get('n', '100'))

    # Both sides use the same oracle — cache on spec_a (arbitrary, same model)
    checkpoint = a.params.get('checkpoint', DEFAULT_ORACLE)
    _log(config, f'  Loading oracle: {checkpoint}')
    oracle = _get_cached_model(config, a, load_oracle, checkpoint, config.device)

    _log(config, f'  E[Q](n={n_samples_a}) vs E[Q](n={n_samples_b}) ({config.n_games} games)...')
    results = evaluate_eq_vs_eq(
        oracle,
        n_games=config.n_games,
        n_samples_a=n_samples_a,
        n_samples_b=n_samples_b,
        device=config.device,
        batch_size=config.batch_size,
        verbose=not config.quiet,
    )

    # evaluate_eq_vs_eq returns results from A's perspective
    return _halfs_from_eq_results(results, eq_is_a=True)


def _run_zeb_vs_random(config: MatchConfig, half: int) -> tuple[HalfResult, HalfResult]:
    """Zeb vs random using the batched neural eval path."""
    from ..evaluate import evaluate_vs_random_batched
    from .loading import load_zeb

    a, b = config.spec_a, config.spec_b
    zeb_spec = a if a.kind == 'zeb' else b
    zeb_source = zeb_spec.params.get('source', 'hf')

    _log(config, f'  Loading Zeb: {zeb_source}')
    zeb_kwargs = {}
    if 'weights_name' in zeb_spec.params:
        zeb_kwargs['weights_name'] = zeb_spec.params['weights_name']
    model = _get_cached_model(
        config, zeb_spec, load_zeb, zeb_source, config.device, **zeb_kwargs,
    )

    zeb_is_a = (a.kind == 'zeb')

    # Run with zeb as team 0
    _log(config, f'  {zeb_spec.display_name} as Team 0 ({half} games)...')
    r0_raw = evaluate_vs_random_batched(model, n_games=half, device=config.device, neural_team=0)

    # Run with zeb as team 1
    _log(config, f'  {zeb_spec.display_name} as Team 1 ({half} games)...')
    r1_raw = evaluate_vs_random_batched(model, n_games=half, device=config.device, neural_team=1)

    # r0_raw: zeb=team0, team0_wins = zeb wins
    # r1_raw: zeb=team1, team1_wins = zeb wins
    zeb_wins_as_0 = r0_raw['team0_wins']
    zeb_margin_as_0 = r0_raw['avg_margin']  # team0 - team1
    zeb_wins_as_1 = r1_raw['team1_wins']
    zeb_margin_as_1 = -r1_raw['avg_margin']  # flip: zeb(team1) - random(team0)

    if zeb_is_a:
        # A=zeb: "a_as_team0" means zeb played as team 0
        h0 = HalfResult(n_games=half, wins=zeb_wins_as_0,
                         win_rate=zeb_wins_as_0 / half, avg_margin=zeb_margin_as_0)
        h1 = HalfResult(n_games=half, wins=zeb_wins_as_1,
                         win_rate=zeb_wins_as_1 / half, avg_margin=zeb_margin_as_1)
    else:
        # A=random: "a_as_team0" means random played as team 0
        # When random is team 0, zeb is team 1: use r1_raw
        random_wins_as_0 = r1_raw['team0_wins']
        random_margin_as_0 = r1_raw['avg_margin']
        # When random is team 1, zeb is team 0: use r0_raw
        random_wins_as_1 = r0_raw['team1_wins']
        random_margin_as_1 = -r0_raw['avg_margin']

        h0 = HalfResult(n_games=half, wins=random_wins_as_0,
                         win_rate=random_wins_as_0 / half, avg_margin=random_margin_as_0)
        h1 = HalfResult(n_games=half, wins=random_wins_as_1,
                         win_rate=random_wins_as_1 / half, avg_margin=random_margin_as_1)

    return h0, h1


def _run_generic(config: MatchConfig, half: int) -> tuple[HalfResult, HalfResult]:
    """Generic fallback using play_match with Player protocol objects."""
    from ..evaluate import play_match

    player_a = build_player(config.spec_a, config.device)
    player_b = build_player(config.spec_b, config.device)

    # A as team 0 (seats 0,2), B as team 1 (seats 1,3)
    _log(config, f'  {config.spec_a.display_name} as Team 0 ({half} games)...')
    r0_raw = play_match(
        players=(player_a, player_b, player_a, player_b),
        n_games=half,
        base_seed=config.seed,
    )

    # A as team 1 (seats 1,3), B as team 0 (seats 0,2)
    _log(config, f'  {config.spec_a.display_name} as Team 1 ({half} games)...')
    r1_raw = play_match(
        players=(player_b, player_a, player_b, player_a),
        n_games=half,
        base_seed=config.seed + half,
    )

    # r0_raw: A=team0, team0_wins = A's wins
    h0 = HalfResult(
        n_games=half, wins=r0_raw['team0_wins'],
        win_rate=r0_raw['team0_win_rate'], avg_margin=r0_raw['avg_margin'],
    )
    # r1_raw: A=team1, team1_wins = A's wins
    h1 = HalfResult(
        n_games=half, wins=r1_raw['team1_wins'],
        win_rate=r1_raw['team1_wins'] / half, avg_margin=-r1_raw['avg_margin'],
    )

    return h0, h1


def _halfs_from_eq_results(results: dict, *, eq_is_a: bool) -> tuple[HalfResult, HalfResult]:
    """Convert eq_player result dicts to HalfResult pairs.

    The eq_player functions return results with 'as_team0' and 'as_team1',
    where the E[Q] player's wins are always from E[Q]'s perspective.
    """
    r0 = results['as_team0']  # E[Q] as team 0
    r1 = results['as_team1']  # E[Q] as team 1

    if eq_is_a:
        # A = E[Q], so a_as_team0 = E[Q] as team 0
        h0 = HalfResult(n_games=r0['n_games'], wins=r0['eq_wins'],
                         win_rate=r0['eq_win_rate'], avg_margin=r0['avg_margin'])
        h1 = HalfResult(n_games=r1['n_games'], wins=r1['eq_wins'],
                         win_rate=r1['eq_win_rate'], avg_margin=r1['avg_margin'])
    else:
        # A = opponent, B = E[Q]
        # a_as_team0 means A played as team 0, so E[Q] was team 1
        opp_wins_as_0 = r1['n_games'] - r1['eq_wins']  # E[Q] as team 1, opp wins
        opp_wins_as_1 = r0['n_games'] - r0['eq_wins']  # E[Q] as team 0, opp wins
        h0 = HalfResult(n_games=r1['n_games'], wins=opp_wins_as_0,
                         win_rate=opp_wins_as_0 / r1['n_games'],
                         avg_margin=-r1['avg_margin'])
        h1 = HalfResult(n_games=r0['n_games'], wins=opp_wins_as_1,
                         win_rate=opp_wins_as_1 / r0['n_games'],
                         avg_margin=-r0['avg_margin'])

    return h0, h1
