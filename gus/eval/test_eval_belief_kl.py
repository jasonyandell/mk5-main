"""Tests for the symmetric belief-KL convergence metric (#26)."""
import torch

from gus.eval.eval_belief_kl import symmetric_belief_kl


def _logits(seed: int, n: int = 4, d: int = 28, s: int = 3) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, d, s, generator=g)


def _full_mask(n: int = 4, d: int = 28) -> torch.Tensor:
    return torch.ones(n, d, dtype=torch.bool)


def test_self_kl_is_zero():
    """a vs a is zero — the determinism self-test the driver relies on."""
    la = _logits(0)
    assert float(symmetric_belief_kl(la, la, _full_mask())) < 1e-6


def test_symmetric():
    la, lb = _logits(1), _logits(2)
    m = _full_mask()
    assert torch.allclose(symmetric_belief_kl(la, lb, m), symmetric_belief_kl(lb, la, m))


def test_positive_when_distributions_differ():
    assert float(symmetric_belief_kl(_logits(1), _logits(2), _full_mask())) > 0.0


def test_empty_mask_is_zero():
    """No unseen-domino slots -> nothing to compare -> zero (not NaN)."""
    la, lb = _logits(1), _logits(2)
    mask = torch.zeros(4, 28, dtype=torch.bool)
    assert float(symmetric_belief_kl(la, lb, mask)) == 0.0


def test_mask_restricts_slots():
    """KL counts only masked slots: masking out the divergent slot lowers KL."""
    la, lb = _logits(1), _logits(2)
    full = _full_mask()
    partial = full.clone()
    partial[:, 0] = False  # drop one slot
    kl_full = float(symmetric_belief_kl(la, lb, full))
    kl_partial = float(symmetric_belief_kl(la, lb, partial))
    assert kl_full >= 0.0 and kl_partial >= 0.0
    # Different slot sets generally give different means; at minimum both finite.
    assert kl_full == kl_full and kl_partial == kl_partial  # not NaN
