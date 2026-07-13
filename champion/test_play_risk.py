"""Score-conditioned play-risk dispatch + the upside_10 lens (rung #27 v2)."""
import torch

from champion.play_risk import select_by_score
from w42.lens_v1.lens import utility_scores


def _pdf(points_mass, n_bins=85):
    """An [85] pdf with given {Q_value: mass}; bin i represents Q = i - 42."""
    p = torch.zeros(n_bins)
    for q, m in points_mass.items():
        p[q + 42] = m
    return p


def _scene():
    """Two games, two legal actions each, built so `ev` and `upside_10` disagree
    on game 1: action 0 has the higher mean but a thin tail, action 1 a lower
    mean but a fat 10% upper tail at Q=20."""
    n = 2
    e_q = torch.full((n, 7), -100.0)
    e_q_pdf = torch.zeros(n, 7, 85)
    legal = torch.zeros(n, 7, dtype=torch.bool)
    legal[:, 0] = True
    legal[:, 1] = True

    e_q[0, 0], e_q[0, 1] = 5.0, 3.0
    e_q_pdf[0, 0] = _pdf({5: 1.0})
    e_q_pdf[0, 1] = _pdf({3: 1.0})

    e_q[1, 0], e_q[1, 1] = 5.0, 2.9
    e_q_pdf[1, 0] = _pdf({5: 1.0})
    e_q_pdf[1, 1] = _pdf({1: 0.9, 20: 0.1})

    bidder = torch.zeros(n, dtype=torch.long)
    current = torch.zeros(n, dtype=torch.long)
    bids = [30, 30]
    return e_q, e_q_pdf, bidder, current, bids, legal


def test_upside_10_isolates_the_fat_tail():
    """The boundary case: a 0.9-mass low bin must NOT pollute the top-10% tail.
    upside_10 of {Q=1:0.9, Q=20:0.1} is ~20 (only the 10% tail), not ~2.9."""
    e_q, e_q_pdf, bidder, current, bids, legal = _scene()
    scores = utility_scores("upside_10", e_q, e_q_pdf, bidder, current, bids)
    assert scores[1, 1].item() == 20.0          # pure upper tail
    assert scores[1, 0].item() == 5.0           # delta at 5
    assert scores[0, 0].item() == 5.0 and scores[0, 1].item() == 3.0


def test_select_by_score_ev_picks_highest_mean():
    e_q, e_q_pdf, bidder, current, bids, legal = _scene()
    actions = select_by_score(["ev", "ev"], e_q, e_q_pdf, bidder, current, bids, legal)
    assert actions.tolist() == [0, 0]


def test_select_by_score_dispatches_per_game():
    """Same tensors, different chosen lens per game -> different game-1 action.
    This is the core dispatch guarantee."""
    e_q, e_q_pdf, bidder, current, bids, legal = _scene()
    assert select_by_score(["ev", "ev"], e_q, e_q_pdf, bidder, current, bids, legal).tolist() == [0, 0]
    assert select_by_score(["ev", "upside_10"], e_q, e_q_pdf, bidder, current, bids, legal).tolist() == [0, 1]


def test_select_by_score_respects_legality():
    e_q, e_q_pdf, bidder, current, bids, legal = _scene()
    legal[1, 1] = False  # forbid game 1's fat-tail action
    actions = select_by_score(["ev", "upside_10"], e_q, e_q_pdf, bidder, current, bids, legal)
    assert actions.tolist() == [0, 0]
