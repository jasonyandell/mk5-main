# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0
"""The instrument must cut the intended paths and leave ordinary inference intact."""
import copy

import pytest
import torch

from lem.observer.data import SEAT, encode
from lem.observer.history_probe import attenuate_keys, completed_tile_masks
from lem.observer.model import Config, Observer


def fixture():
    torch.manual_seed(71)
    model = Observer(Config(width=16, layers=3, heads=2)).eval()
    # Instrument fixture, not game evidence: two complete tricks, one extra play.
    tokens = torch.tensor([*encode(127, 0, 0, 3,
                         [[i % 4, i + 7] for i in range(9)]), SEAT])
    return model, tokens[None], completed_tile_masks(tokens)


def test_masks_and_validation():
    model, x, masks = fixture()
    assert masks.sum(1).tolist() == [2] * 4
    assert masks[2].nonzero().flatten().tolist() == [18, 26]
    assert not masks[:, :14].any() and not masks[:, 30:].any()
    bad = x[0].clone(); bad[13] = SEAT + 1
    with pytest.raises(ValueError):
        completed_tile_masks(bad)
    with pytest.raises(ValueError):
        completed_tile_masks(x)
    with pytest.raises(ValueError):
        with attenuate_keys(model, torch.ones_like(x, dtype=torch.bool)):
            pass


def test_sham_cleanup_and_exact_state():
    model, x, masks = fixture(); state = copy.deepcopy(model.state_dict())
    with torch.inference_mode():
        baseline = model(x)
        for mask, strength in [(masks[2:3], 0), (torch.zeros_like(x, dtype=torch.bool), 1)]:
            with attenuate_keys(model, mask, strength):
                assert torch.equal(model(x), baseline)
        with pytest.raises(RuntimeError):
            with attenuate_keys(model, masks[2:3]):
                raise RuntimeError('interrupted instrument')
        assert torch.equal(model(x), baseline)
    assert all(torch.equal(v, state[k]) for k, v in model.state_dict().items())
    assert all(not b.attention._forward_hooks for b in model.blocks)


@pytest.mark.parametrize('strength', [.5, 1.])
def test_weights_match_independent_probability_renormalization(strength):
    model, x, masks = fixture(); mask = masks[2:3]
    # Only the last layer changes, so its inputs are identical to baseline.
    with torch.inference_mode():
        _, original = model(x, inspect=True)
        with attenuate_keys(model, mask, strength, last_only=True):
            _, changed = model(x, inspect=True)
    expected = original[-1]['attention'] * (1 - mask[:, None, None, :].float() * strength)
    expected /= expected.sum(-1, keepdim=True)
    torch.testing.assert_close(changed[-1]['attention'], expected, rtol=1e-5, atol=1e-7)
    assert torch.count_nonzero(torch.triu(changed[-1]['attention'], 1)) == 0
    torch.testing.assert_close(changed[-1]['attention'].sum(-1), torch.ones_like(expected[..., 0]))


def test_all_layers_cut_tile_identity_paths_but_last_layer_can_retain_them():
    model, x, masks = fixture(); mask = masks[2:3]
    altered = x.clone(); altered[mask] = (altered[mask] + 11) % 28
    with torch.inference_mode():
        with attenuate_keys(model, mask):
            assert torch.equal(model(x)[:, -1], model(altered)[:, -1])
        with attenuate_keys(model, mask, last_only=True):
            assert not torch.equal(model(x)[:, -1], model(altered)[:, -1])
        batch = x.repeat(2, 1); batch_mask = mask.repeat(2, 1)
        with attenuate_keys(model, batch_mask):
            before = model(batch)[0].clone()
            batch[1] = 0
            assert torch.equal(model(batch)[0], before)
