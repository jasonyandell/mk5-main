"""Belief-weighted world-sampling weights (rung #25), model-free core math."""
import torch

from champion.belief import (
    belief_weights_for_worlds,
    effective_sample_size,
    weights_from_logP,
)


def _two_world_scene():
    """1 game, 2 worlds, 3 unseen dominoes (0,1,2). The belief strongly prefers
    dom0->rel0, dom1->rel1, dom2->rel2. World 0 matches; world 1 swaps 0<->1."""
    logP = torch.full((1, 28, 3), -10.0)
    logP[0, 0, 0] = 0.0
    logP[0, 1, 1] = 0.0
    logP[0, 2, 2] = 0.0
    worlds = torch.full((1, 2, 3, 7), -1, dtype=torch.long)
    worlds[0, 0, 0, 0] = 0  # world 0 matches belief
    worlds[0, 0, 1, 0] = 1
    worlds[0, 0, 2, 0] = 2
    worlds[0, 1, 0, 0] = 1  # world 1 swaps dom0<->dom1
    worlds[0, 1, 1, 0] = 0
    worlds[0, 1, 2, 0] = 2
    return logP, worlds


def test_weights_favor_high_belief_world():
    logP, worlds = _two_world_scene()
    w = weights_from_logP(logP, worlds, uniform_mix=0.0)
    assert w[0, 0] > w[0, 1]
    assert torch.allclose(w.sum(dim=1), torch.ones(1))


def test_weights_are_a_distribution():
    logP, worlds = _two_world_scene()
    w = weights_from_logP(logP, worlds, uniform_mix=0.1)
    assert torch.all(w >= 0)
    assert torch.allclose(w.sum(dim=1), torch.ones(1))


def test_uniform_mix_one_is_uniform():
    logP, worlds = _two_world_scene()
    w = weights_from_logP(logP, worlds, uniform_mix=1.0)
    assert torch.allclose(w, torch.full_like(w, 0.5))


def test_none_model_gives_uniform():
    _, worlds = _two_world_scene()
    w = belief_weights_for_worlds(None, False, [], worlds, "cpu")
    assert torch.allclose(w, torch.full_like(w, 0.5))


def test_ess_bounds():
    logP, worlds = _two_world_scene()
    w = weights_from_logP(logP, worlds, uniform_mix=0.1)
    ess = effective_sample_size(w)
    assert 1.0 <= ess.item() <= 2.0  # M = 2 worlds


def test_padded_slots_ignored():
    """A -1 pad must not index into logP or add spurious mass: with a flat
    belief, two worlds differing only in their (valid) single tile stay equal."""
    logP = torch.zeros(1, 28, 3)  # flat belief
    worlds = torch.full((1, 2, 3, 7), -1, dtype=torch.long)
    worlds[0, 0, 0, 0] = 5
    worlds[0, 1, 1, 0] = 6
    w = weights_from_logP(logP, worlds, uniform_mix=0.0)
    assert torch.allclose(w, torch.full_like(w, 0.5))
