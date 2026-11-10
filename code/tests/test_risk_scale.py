import numpy as np

from xai_ews.evaluation import risk_scale as rs


def test_grade_edges_and_alert_class():
    assert rs.grade(np.array([0.0001]))[0] == 1
    assert rs.grade(np.array([0.0099]))[0] == rs.ALERT_GRADE - 1
    assert rs.grade(np.array([0.01]))[0] == rs.ALERT_GRADE        # the alert threshold opens class 7
    assert rs.grade(np.array([0.2]))[0] == 10
    assert len(rs.grade_labels()) == 10


def test_relative_risk_and_percentile_by_group():
    p = np.array([0.001, 0.002, 0.003, 0.01, 0.02, 0.03])
    g = np.array([0, 0, 0, 1, 1, 1])
    rr = rs.relative_risk(p, g)
    assert np.allclose(rr, [0.5, 1.0, 1.5, 0.5, 1.0, 1.5])
    assert np.allclose(rs.percentile(p, g), [0, 1 / 3, 2 / 3, 0, 1 / 3, 2 / 3])


def test_points_are_monotone_and_halve_odds_every_pdo():
    p = np.array([0.001, 0.01, 0.1])
    pts = rs.score_points(p)
    assert np.all(np.diff(pts) < 0)                                  # higher PD -> fewer points
    odds = lambda q: q / (1 - q)
    q2 = 1 / (1 + 1 / (2 * odds(0.01)))                              # PD with doubled odds
    assert np.isclose(rs.score_points(np.array([0.01]))[0] - rs.score_points(np.array([q2]))[0], 20.0)
