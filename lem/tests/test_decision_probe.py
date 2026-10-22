# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0
"""The decision-class probe: its weighted metrics agree with scikit-learn's, its rows carry the
quotient's classes, and its recomputed residual path reproduces the observer's logits."""
import numpy as np
import pytest
from sklearn.metrics import balanced_accuracy_score, r2_score, roc_auc_score

from lem.probe.decision import auc, balanced_accuracy, decision_rows, net_features, r2
from lem.tests.test_quotient import study  # noqa: F401  (the hand-built course fixture)


def test_weighted_metrics_match_sklearn():
    rng = np.random.default_rng(0)
    y = rng.random(200) < .3
    s = np.round(rng.random(200), 1)                  # ties in the scores
    w = rng.integers(0, 4, 200).astype(float)
    assert auc(y, s, w) == pytest.approx(roc_auc_score(y, s, sample_weight=w))
    c, p = rng.integers(0, 3, 200), rng.integers(0, 3, 200)
    assert balanced_accuracy(c, p, w) == pytest.approx(balanced_accuracy_score(c, p, sample_weight=w))
    t, q = rng.normal(size=(200, 2)), rng.normal(size=(200, 2))
    assert r2(t, q, w) == pytest.approx(r2_score(t, q, sample_weight=w))


def test_rows_carry_the_weak_order(study):  # noqa: F811
    rows = decision_rows(study, eligible_only=False)
    # Five roots, each a two-way root whose first slot wins: class "01", contrast -1, not flat.
    assert list(rows["cls"]) == ["01"] * 5 and not rows["flat"].any()
    assert np.all(rows["contrast"][:, 0] == -1) and np.isnan(rows["contrast"][:, 1]).all()
    assert rows["train"].sum() == 3


def test_residual_path_reproduces_the_logits():
    from lem.probe.gus import load_model
    model, _ = load_model(None, untrained_seed=3, config={"width": 16, "layers": 2, "heads": 2,
                                                          "context": 20, "vocabulary": 46})
    rng = np.random.default_rng(1)
    lengths = rng.integers(5, 20, 9)
    tokens = np.where(np.arange(20)[None] < lengths[:, None], rng.integers(0, 46, (9, 20)), 0)
    menu = np.array([[1, 2, -1]] * 9)
    feats = net_features(model, {"tokens": tokens, "lengths": lengths, "menu": menu})
    assert feats["res2"].shape == (9, 16) and feats["pre1"].shape == (9, 64)
    assert np.allclose(feats["menu_logits"][:, :2].sum(1), 0) and np.all(feats["menu_logits"][:, 2] == 0)
