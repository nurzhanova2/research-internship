"""Model metrics of methodology section 7 (stage 3).

Discrimination  AUC, Gini = 2 AUC - 1, KS (max distance between the score CDFs of events and
                non-events), PR-AUC (average precision; matters at event rates of 0.2-2 %).
Calibration     Brier (mean squared error of PD), log-loss, ECE (|mean PD - event rate| averaged
                over equal-count PD bins, weighted by bin size) and ECE / event rate,
                calibration slope and intercept: logistic regression of the outcome on logit(PD).
                slope 1 / intercept 0 = calibrated; slope < 1 = PD too extreme. The intercept is
                the calibration-in-the-large version: offset logit(PD) with slope fixed at 1.

All metrics take optional row weights (1/r for subsampled train negatives, omega for weighted
log-loss); evaluation samples are unweighted.
"""

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

EPS = 1e-7


def logit(p) -> np.ndarray:
    p = np.clip(np.asarray(p, np.float64), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sigmoid(z) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(z, np.float64)))


def auc(y, score, w=None) -> float:
    return float(roc_auc_score(y, score, sample_weight=w))


def ks(y, score, w=None) -> float:
    y = np.asarray(y)
    w = np.ones(len(y)) if w is None else np.asarray(w, np.float64)
    order = np.argsort(score, kind="stable")
    s, yy, ww = np.asarray(score)[order], y[order], w[order]
    cp = np.cumsum(ww * (yy == 1)) / np.sum(ww * (yy == 1))
    cn = np.cumsum(ww * (yy == 0)) / np.sum(ww * (yy == 0))
    last = np.r_[s[1:] != s[:-1], True]          # evaluate only at the end of tie groups
    return float(np.max(np.abs(cp[last] - cn[last])))


def pr_auc(y, score, w=None) -> float:
    return float(average_precision_score(y, score, sample_weight=w))


def log_loss(y, p, w=None) -> float:
    y = np.asarray(y, np.float64)
    p = np.clip(np.asarray(p, np.float64), EPS, 1 - EPS)
    w = np.ones(len(y)) if w is None else np.asarray(w, np.float64)
    return float(-np.sum(w * (y * np.log(p) + (1 - y) * np.log(1 - p))) / np.sum(w))


def brier(y, p, w=None) -> float:
    return float(np.average((np.asarray(p, np.float64) - np.asarray(y)) ** 2, weights=w))


def ece(y, p, bins: int = 10, w=None) -> float:
    """Expected calibration error over equal-count bins of PD."""
    y, p = np.asarray(y, np.float64), np.asarray(p, np.float64)
    w = np.ones(len(y)) if w is None else np.asarray(w, np.float64)
    order = np.argsort(p, kind="stable")
    cw = np.cumsum(w[order])
    edges = np.searchsorted(cw, np.linspace(0, cw[-1], bins + 1)[1:-1])
    out = 0.0
    for idx in np.split(order, edges):
        if len(idx):
            ww = w[idx]
            out += ww.sum() * abs(np.average(p[idx], weights=ww) - np.average(y[idx], weights=ww))
    return float(out / w.sum())


def _logistic_newton(y: np.ndarray, z: np.ndarray, with_slope: bool, iters: int = 100,
                     tol: float = 1e-10) -> np.ndarray:
    """Maximum-likelihood logistic regression of y on [1, z] (with_slope) or on 1 with offset z,
    by damped Newton steps on sums: O(n) memory (statsmodels GLM needs ~1 GB per million rows).
    Starts from the event rate (slope 0) and halves a step that does not raise the likelihood."""

    def eta_of(b):
        return b[0] + b[1] * z if with_slope else b[0] + z

    def loglik(b):
        e = eta_of(b)
        return float(np.sum(y * e - np.logaddexp(0.0, e)))

    rate = np.clip(y.mean(), 1e-9, 1 - 1e-9)
    beta = np.array([np.log(rate / (1 - rate)), 0.0]) if with_slope else np.array([np.log(rate / (1 - rate)) - z.mean()])
    ll = loglik(beta)
    for _ in range(iters):
        mu = sigmoid(eta_of(beta))
        r, w = y - mu, mu * (1.0 - mu)
        if with_slope:
            g = np.array([r.sum(), (r * z).sum()])
            h = np.array([[w.sum(), (w * z).sum()], [(w * z).sum(), (w * z * z).sum()]])
            step = np.linalg.solve(h, g)
        else:
            step = np.array([r.sum() / w.sum()])
        t = 1.0
        while t > 1e-6:
            new = beta + t * step
            new_ll = loglik(new)
            if new_ll >= ll - 1e-12:
                break
            t /= 2
        beta, ll = new, new_ll
        if np.max(np.abs(t * step)) < tol:
            break
    return beta


def calibration_slope_intercept(y, p) -> tuple[float, float]:
    """Slope: logit(P(y)) = a + b logit(PD). Intercept: a with b fixed at 1 (offset)."""
    z = logit(p)
    y = np.asarray(y, np.float64)
    slope = _logistic_newton(y, z, with_slope=True)[1]
    intercept = _logistic_newton(y, z, with_slope=False)[0]
    return float(slope), float(intercept)


def calibration_curve(y, p, bins: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """Mean PD and event rate per equal-count bin."""
    order = np.argsort(p, kind="stable")
    parts = np.array_split(order, bins)
    return (np.array([np.mean(np.asarray(p)[i]) for i in parts]),
            np.array([np.mean(np.asarray(y)[i]) for i in parts]))


def all_metrics(y, p) -> dict:
    """Every scalar metric of section 7 for calibrated PDs `p` on an unweighted sample."""
    y = np.asarray(y)
    a = auc(y, p)
    slope, intercept = calibration_slope_intercept(y, p)
    e = ece(y, p)
    return {"n": len(y), "events": int(y.sum()), "event_rate": float(y.mean()), "mean_pd": float(np.mean(p)),
            "auc": a, "gini": 2 * a - 1, "ks": ks(y, p), "pr_auc": pr_auc(y, p),
            "brier": brier(y, p), "log_loss": log_loss(y, p), "ece": e, "ece_norm": e / max(y.mean(), EPS),
            "cal_slope": slope, "cal_intercept": intercept}
