"""JudAuxPlay picks the legal slot the aux head values highest (acting-seat, no flip)."""
from __future__ import annotations

import pytest
import torch

from arena.jud_play import JudAuxPlay
from champion.jud_net import JudNet
from forge.zeb.game import current_player, legal_actions, new_game


def _playing_state():
    # skip_bidding=True yields a PLAYING state with a fixed contract.
    return new_game(seed=123, skip_bidding=True)


def test_rejects_aux_free_checkpoint():
    with pytest.raises(ValueError):
        JudAuxPlay(JudNet(aux_per_action=False))


def test_argmax_over_legal_aux_slots():
    torch.manual_seed(0)
    net = JudNet(aux_per_action=True)
    s = _playing_state()
    play = JudAuxPlay(net)
    (choice,) = play.choose([s], [30])
    legal = legal_actions(s)
    assert choice in legal
    # Recompute expected argmax by hand through the same forward.
    from champion.jud_net import featurize_state
    x = featurize_state(s, seat=current_player(s)).unsqueeze(0)
    with torch.no_grad():
        _, aux = net.forward_aux(x)
    expected = max(legal, key=lambda a: float(aux[0][a]))
    assert choice == expected
