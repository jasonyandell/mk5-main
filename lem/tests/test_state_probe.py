# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0
import numpy as np
import pytest

from lem.observer.data import encode, SEAT
from lem.observer.state_probe import public_score, ridge_fit, ridge_predict
from lem.tests.test_observer import recording
from scenario.record import empty_record
from scenario.rules import rules_for


def test_public_score_relative_seats_and_future_independence():
    episode = recording(); rec = empty_record(0); rules = rules_for(3)
    for i, (viewer, tile) in enumerate(episode['plays']):
        tokens = np.append(encode(episode['hands'][viewer], viewer, 0, 3,
                                  episode['plays'][:i]), SEAT)
        row = public_score(tokens)
        expected = [rec.banked[viewer % 2], rec.banked[1 - viewer % 2]]
        assert row['banked'] == expected
        assert row['margin'] == expected[0] - expected[1]
        assert row['completed'] == i // 4
        assert row['metadata'].shape == (26,) and row['bag'].shape == (166,)
        # Metadata has five one-hot fields; neither baseline gets the score.
        assert row['metadata'].sum() == 5 and row['bag'][54:].sum() == i
        rec = rec.advance(tile, rules)


def test_invalid_public_prefix_rejected():
    episode = recording(); actor = episode['plays'][0][0]
    x = np.append(encode(episode['hands'][actor], actor, 0, 3, []), SEAT)
    with pytest.raises(ValueError): public_score(x[:-1])
    bad = x.copy(); bad[3] = 41
    with pytest.raises(ValueError): public_score(bad)
    bad = x.copy(); bad[6] = bad[5]
    with pytest.raises(ValueError): public_score(bad)


def test_ridge_matches_augmented_least_squares_and_fit_only_scaling():
    rng = np.random.default_rng(91)
    x = rng.normal(size=(53, 6)); x[:, 5] = 4
    y = rng.normal(size=53); penalty = .001
    fitted = ridge_fit(x, y, penalty)
    z = np.column_stack([np.ones(len(x)), (x-x.mean(0))/np.maximum(x.std(0), 1e-8)])
    regularizer = np.diag(np.sqrt([0.] + [len(x)*penalty]*6))
    weights = np.linalg.lstsq(np.vstack([z, regularizer]), np.r_[y, np.zeros(7)], rcond=None)[0]
    np.testing.assert_allclose(weights, fitted['weights'], atol=1e-11, rtol=1e-10)
    before = {k:v.copy() for k,v in fitted.items()}
    ridge_predict(fitted, np.full((7, 6), 1000.))
    assert all(np.array_equal(before[k], fitted[k]) for k in fitted)
    np.testing.assert_allclose(ridge_predict(fitted, x), z@weights, atol=1e-11)
    with pytest.raises(ValueError): ridge_fit(x, y, float('nan'))
