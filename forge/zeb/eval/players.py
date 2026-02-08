"""Player specification parsing and construction."""

from __future__ import annotations

from dataclasses import dataclass, field


KNOWN_KINDS = ('random', 'heuristic', 'zeb', 'eq')

# Default params per kind
_DEFAULTS: dict[str, dict[str, str]] = {
    'eq': {'n': '100'},
    'zeb': {'source': 'hf'},
}


@dataclass(frozen=True)
class PlayerSpec:
    """Parsed player specification."""
    kind: str
    params: dict[str, str] = field(default_factory=dict)

    @property
    def display_name(self) -> str:
        if not self.params:
            return self.kind
        parts = [f'{k}={v}' for k, v in self.params.items()]
        return f'{self.kind}({", ".join(parts)})'


def parse_player_spec(s: str) -> PlayerSpec:
    """Parse a player spec string like 'eq:n=100' or 'random'.

    Format: KIND[:KEY=VAL[,KEY=VAL,...]]

    Examples:
        'random'         -> PlayerSpec(kind='random')
        'heuristic'      -> PlayerSpec(kind='heuristic')
        'eq:n=50'        -> PlayerSpec(kind='eq', params={'n': '50'})
        'zeb:source=hf'  -> PlayerSpec(kind='zeb', params={'source': 'hf'})
        'eq'             -> PlayerSpec(kind='eq', params={'n': '100'})  # defaults

    Raises:
        ValueError: If kind is unknown.
    """
    s = s.strip()
    if ':' in s:
        kind, param_str = s.split(':', 1)
        params = {}
        for pair in param_str.split(','):
            if '=' not in pair:
                raise ValueError(f"Invalid param format: {pair!r} (expected KEY=VAL)")
            k, v = pair.split('=', 1)
            params[k.strip()] = v.strip()
    else:
        kind = s
        params = {}

    kind = kind.strip().lower()
    if kind not in KNOWN_KINDS:
        raise ValueError(f"Unknown player kind: {kind!r}. Known: {', '.join(KNOWN_KINDS)}")

    # Apply defaults for missing params
    defaults = _DEFAULTS.get(kind, {})
    merged = {**defaults, **params}

    return PlayerSpec(kind=kind, params=merged)


def build_player(spec: PlayerSpec, device: str = 'cuda'):
    """Construct a Player protocol object from a spec.

    Used for the generic fallback path (play_match with individual Player objects).
    """
    from ..evaluate import RandomPlayer, RuleBasedPlayer, NeuralPlayer

    if spec.kind == 'random':
        return RandomPlayer()
    elif spec.kind == 'heuristic':
        return RuleBasedPlayer()
    elif spec.kind == 'zeb':
        from .loading import load_zeb
        source = spec.params.get('source', 'hf')
        weights_name = spec.params.get('weights_name')
        kwargs = {}
        if weights_name:
            kwargs['weights_name'] = weights_name
        model = load_zeb(source, device=device, **kwargs)
        return NeuralPlayer(model, device=device)
    elif spec.kind == 'eq':
        from ..eq_player import EQPlayer
        from .loading import load_oracle, DEFAULT_ORACLE
        checkpoint = spec.params.get('checkpoint', DEFAULT_ORACLE)
        n_samples = int(spec.params.get('n', '100'))
        model = load_oracle(checkpoint, device)
        return EQPlayer(model, n_samples=n_samples, device=device)
    else:
        raise ValueError(f"Cannot build player for kind: {spec.kind!r}")
