# STRATEGY-FUSION FENCE, sanctioned as a baseline. This seats E[Q] n=10 (LensPlay,
# the oracle era's champion: per-world clairvoyant Q averaged) in the tournament as
# a measuring stick, the use wiki/topics/strategy-fusion-fence.md § Sanctioned uses
# names ("Baselines"). Its games rate it against the others; nothing reads its
# values, and nothing in Zeb or Walt may import this module.
# The ruling: wiki/topics/strategy-fusion-fence.md.
"""The fenced E[Q] n=10 seat: the arena's `heuristic+lens:ev`, pinned by checkpoint hash."""
from __future__ import annotations

from .participants import Participant
from .seats import ArenaSeat

_MODELS: dict[tuple[str, str], object] = {}


def _device(requested: str | None) -> str:
    import torch

    if requested:
        return requested
    return "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"


def eq_seat(p: Participant, bidder_spec: str, play_spec: str, device: str | None) -> ArenaSeat:
    from arena.cli import parse_bidder
    from arena.lens_play import LensPlay
    from forge.zeb.eval.loading import load_oracle

    from .artifacts import ROOT

    dev = _device(device)
    key = (p.pin["checkpoint"], dev)
    if key not in _MODELS:
        _MODELS[key] = load_oracle(str(ROOT / p.pin["checkpoint"]), dev)
    utility = play_spec.partition(":")[2] or "ev"
    play = LensPlay(_MODELS[key], utility=utility, n_samples=int(p.pin["n_samples"]), device=dev)
    return ArenaSeat(parse_bidder(bidder_spec, device=dev, gus_adapter=None), play)
