"""Statistical tests of stage 3 (methodology section 8).

    cluster bootstrap  loans are resampled with replacement and every snapshot of a drawn loan
                       comes along (row weight = how many times its loan was drawn); B = 1000.
                       Gives 95 % percentile CIs of any metric and of differences between models.
    DeLong             paired test of two AUCs on ONE random snapshot per loan (independent rows);
                       reused from stage 2 (xai_ews.features.evaluation).
    LR test            likelihood-ratio test of nested logistic regressions: 2 (ll_full - ll_restricted)
                       ~ chi2 with df = extra parameters (H2: feature x segment interactions).
    Holm               step-down correction of p-values for several model comparisons.
"""

from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from xai_ews.features.evaluation import (  # noqa: F401
    _SortedAUC,
    delong_test,
    one_snapshot_per_loan,
)


def bootstrap_weights(clusters, b: int, seed: int = 0):
    """Yields b row-weight vectors of a cluster bootstrap (multiplicity of the row's loan)."""
    rng = np.random.default_rng(seed)
    codes, uniq = pd.factorize(pd.Series(np.asarray(clusters)))
    k = len(uniq)
    for _ in range(b):
        yield np.bincount(rng.integers(0, k, k), minlength=k)[codes].astype(np.float64)


def cluster_bootstrap(metrics: dict[str, Callable[[np.ndarray], float]], clusters, b: int = 1000,
                      seed: int = 0, diffs: list[tuple[str, str]] | None = None) -> pd.DataFrame:
    """Point value and 95 % CI of every metric(w) (w = row weights, ones for the point value);
    `diffs` = pairs (a, b) for which the CI of metric_a - metric_b is also returned."""
    n = len(np.asarray(clusters))
    point = {k: f(np.ones(n)) for k, f in metrics.items()}
    draws = {k: np.empty(b) for k in metrics}
    for i, w in enumerate(bootstrap_weights(clusters, b, seed)):
        for k, f in metrics.items():
            draws[k][i] = f(w)
    rows = [{"metric": k, "value": point[k], "lo": np.percentile(draws[k], 2.5),
             "hi": np.percentile(draws[k], 97.5)} for k in metrics]
    for a, c in diffs or []:
        d = draws[a] - draws[c]
        rows.append({"metric": f"{a} - {c}", "value": point[a] - point[c], "lo": np.percentile(d, 2.5),
                     "hi": np.percentile(d, 97.5), "share_le_0": float(np.mean(d <= 0))})
    return pd.DataFrame(rows)


def weighted_auc_fn(y, score) -> Callable[[np.ndarray], float]:
    """AUC as a function of row weights, O(n) per call (sort order fixed once)."""
    return _SortedAUC(np.asarray(y), np.asarray(score, np.float64))


def lr_test(ll_full: float, ll_restricted: float, df: int) -> dict:
    stat = 2.0 * (ll_full - ll_restricted)
    return {"lr_stat": float(stat), "df": int(df), "p": float(stats.chi2.sf(stat, df))}


def holm(pvalues) -> np.ndarray:
    return multipletests(np.asarray(pvalues, float), method="holm")[1]


class _SortedKS:
    """KS with arbitrary row weights in O(n): sort order and tie groups fixed once."""

    def __init__(self, y, score):
        self.order = np.argsort(score, kind="stable")
        s = np.asarray(score)[self.order]
        self.ends = np.flatnonzero(np.r_[s[1:] != s[:-1], True])
        self.pos = (np.asarray(y) == 1)[self.order]

    def __call__(self, w: np.ndarray) -> float:
        ws = w[self.order]
        cp = np.cumsum(np.where(self.pos, ws, 0.0))
        cn = np.cumsum(np.where(self.pos, 0.0, ws))
        return float(np.max(np.abs(cp[self.ends] / cp[-1] - cn[self.ends] / cn[-1])))


def weighted_ks_fn(y, score) -> Callable[[np.ndarray], float]:
    return _SortedKS(np.asarray(y), np.asarray(score, np.float64))


def score_psi(train_score, test_score, train_weight=None, bins: int = 10) -> float:
    """PSI of a score between train (weighted quantile bins) and test: sum (q - p) ln(q / p)."""
    from xai_ews.features.woe import weighted_quantiles

    w = np.ones(len(train_score)) if train_weight is None else np.asarray(train_weight, np.float64)
    edges = np.unique(weighted_quantiles(np.asarray(train_score, np.float64), w, np.linspace(0, 1, bins + 1)[1:-1]))
    p = np.bincount(np.searchsorted(edges, train_score, side="right"), weights=w, minlength=len(edges) + 1)
    q = np.bincount(np.searchsorted(edges, test_score, side="right"), minlength=len(edges) + 1).astype(float)
    p, q = np.clip(p / p.sum(), 1e-6, None), np.clip(q / q.sum(), 1e-6, None)
    return float(np.sum((q - p) * np.log(q / p)))
