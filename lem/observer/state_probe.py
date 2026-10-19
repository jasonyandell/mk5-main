# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0
"""Public-state labels and small linear readouts; never actor inputs or rewards."""
import numpy as np

from scenario.record import empty_record
from scenario.rules import rules_for
from .data import BOS, BID30, HAND, PLAY, PREFIX, SEAT, DECL


def public_score(tokens):
    """Reconstruct the banked partnership margin from an acting-viewer prefix.

    No hidden deal, future play, outcome, or legal mask is accepted. Baselines
    receive metadata and an orderless actor/tile bag, never the computed score.
    This diagnostic's metadata vocabulary deliberately covers pip trump only.
    """
    x = np.asarray(tokens)
    if x.ndim != 1 or len(x) < PREFIX + 1 or (len(x) - PREFIX - 1) % 2:
        raise ValueError('expected one acting-viewer token prefix')
    if list(x[[0, 1, 4, 12, -1]]) != [BOS, BID30, HAND, PLAY, SEAT]:
        raise ValueError('unsupported prefix markers')
    bidder, decl = int(x[2]) - SEAT, int(x[3]) - DECL
    if not 0 <= bidder < 4 or not 0 <= decl < 7:
        raise ValueError('bid30 and pip trump required')
    own = x[5:12]
    if len(set(map(int, own))) != 7 or np.any((own < 0) | (own >= 28)):
        raise ValueError('invalid original hand')
    record, rules = empty_record(bidder), rules_for(decl)
    bag = np.zeros((4, 28), np.float64)
    for pos in range(PREFIX, len(x) - 1, 2):
        seat, tile = int(x[pos]) - SEAT, int(x[pos + 1])
        if seat != record.seat or not 0 <= tile < 28 or record.played & (1 << tile):
            raise ValueError('invalid public play order or repeated tile')
        bag[seat, tile] = 1
        record = record.advance(tile, rules)
    if record.seat != 0:
        raise ValueError('query is not the next actor')
    completed, slot = divmod(record.plays_so_far(), 4)
    if completed > 6:
        raise ValueError('no action after completed hand')
    metadata = np.concatenate([np.eye(n)[i] for n, i in
                               [(4, bidder), (7, decl), (7, completed),
                                (4, slot), (4, record.leader)]])
    own_bits = np.zeros(28); own_bits[own.astype(int)] = 1
    return {'banked': list(record.banked),
            'margin': record.banked[0] - record.banked[1],
            'metadata': metadata, 'bag': np.concatenate([metadata, own_bits, bag.ravel()]),
            'completed': completed, 'slot': slot, 'bidder_team': bidder % 2}


def ridge_fit(x, y, penalty=.001):
    """Mean squared error plus ridge on fit-standardized nonintercept weights."""
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    if x.ndim != 2 or y.shape != (len(x),) or not len(x) or not np.isfinite(penalty) or penalty <= 0:
        raise ValueError('finite positive penalty and aligned nonempty rows required')
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('nonfinite fit data')
    mean, scale = x.mean(0), np.maximum(x.std(0), 1e-8)
    z = np.column_stack([np.ones(len(x)), (x - mean) / scale])
    regularizer = np.diag([0.] + [penalty] * x.shape[1])
    weights = np.linalg.solve(z.T @ z / len(z) + regularizer, z.T @ y / len(z))
    residual = z.T @ (z @ weights - y) / len(z) + regularizer @ weights
    if np.max(np.abs(residual)) > 1e-9:
        raise ValueError('ridge solution failed stationarity check')
    return {'mean': mean, 'scale': scale, 'weights': weights}


def ridge_predict(fit, x):
    return fit['weights'][0] + ((np.asarray(x) - fit['mean']) / fit['scale']) @ fit['weights'][1:]
