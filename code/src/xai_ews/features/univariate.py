"""Univariate strength of a feature (stage 2, steps 2-3): IV, KS, AUC, chi2, Cramér's V, BH.

Subsampling of negatives (weight 1/r) does not change the class-conditional distributions, so
IV, KS and AUC estimate the population values; weights are used anyway for the binning
quantiles. The chi2 test of independence is run on the UNWEIGHTED counts: under sampling that
depends on y only, x and y are independent in the sample iff they are independent in the
population. Cramér's V is taken from the weighted table (population effect size).
Observations of one loan are dependent, so chi2 p-values are too small; decisions use effect
sizes (IV), p-values are reported for completeness.
"""

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from xai_ews.features.woe import WoEBinner, fit_woe, woe_iv


def _tie_groups(score: np.ndarray):
    order = np.argsort(score, kind="stable")
    s = score[order]
    starts = np.flatnonzero(np.r_[True, s[1:] != s[:-1]])
    return order, starts


def weighted_auc(y, score, w=None) -> float:
    """Mann-Whitney AUC with weights and ties counted as 1/2 (equals sklearn roc_auc_score)."""
    y = np.asarray(y)
    score = np.asarray(score, dtype=np.float64)
    w = np.ones(len(y)) if w is None else np.asarray(w, dtype=np.float64)
    order, starts = _tie_groups(score)
    wp = np.add.reduceat((w * (y == 1))[order], starts)
    wn = np.add.reduceat((w * (y == 0))[order], starts)
    below = np.cumsum(wn) - wn
    P, N = wp.sum(), wn.sum()
    if P == 0 or N == 0:
        return float("nan")
    return float(np.sum(wp * (below + 0.5 * wn)) / (P * N))


def ks_statistic(y, score, w=None) -> float:
    """Kolmogorov-Smirnov: max gap between the weighted CDFs of the score for bads and goods."""
    y = np.asarray(y)
    score = np.asarray(score, dtype=np.float64)
    w = np.ones(len(y)) if w is None else np.asarray(w, dtype=np.float64)
    order, starts = _tie_groups(score)
    wp = np.add.reduceat((w * (y == 1))[order], starts)
    wn = np.add.reduceat((w * (y == 0))[order], starts)
    if wp.sum() == 0 or wn.sum() == 0:
        return float("nan")
    return float(np.max(np.abs(np.cumsum(wp) / wp.sum() - np.cumsum(wn) / wn.sum())))


def chi2_test(bin_idx, y) -> tuple[float, float, int]:
    """chi2 test of independence on the unweighted bins x {0, 1} table: (statistic, p, dof)."""
    table = pd.crosstab(np.asarray(bin_idx), np.asarray(y)).to_numpy()
    table = table[table.sum(axis=1) > 0]
    if table.shape[0] < 2 or table.shape[1] < 2:
        return 0.0, 1.0, 0
    chi2, p, dof, _ = stats.chi2_contingency(table, correction=False)
    return float(chi2), float(p), int(dof)


def cramers_v(bin_idx, y, w=None) -> float:
    """Cramér's V of the (weighted) contingency table bins x classes."""
    bin_idx = np.asarray(bin_idx)
    y = np.asarray(y)
    w = np.ones(len(y)) if w is None else np.asarray(w, dtype=np.float64)
    _, b = np.unique(bin_idx, return_inverse=True)
    _, c = np.unique(y, return_inverse=True)
    t = np.zeros((b.max() + 1, c.max() + 1))
    np.add.at(t, (b, c), w)
    if min(t.shape) < 2:
        return 0.0
    n = t.sum()
    exp = np.outer(t.sum(1), t.sum(0)) / n
    chi2 = np.sum((t - exp) ** 2 / np.where(exp > 0, exp, np.inf))
    return float(np.sqrt(chi2 / (n * (min(t.shape) - 1))))


def bh_adjust(p) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (false discovery rate)."""
    p = np.asarray(p, dtype=np.float64)
    out = np.full(len(p), np.nan)
    ok = ~np.isnan(p)
    if ok.any():
        out[ok] = multipletests(p[ok], method="fdr_bh")[1]
    return out


def direction(y, x, w=None) -> float:
    """AUC - 0.5 of the raw numeric value on non-missing rows: > 0 means higher value = riskier."""
    x = np.asarray(x, dtype=np.float64)
    ok = ~np.isnan(x)
    if ok.sum() == 0 or np.unique(x[ok]).size < 2:
        return float("nan")
    return weighted_auc(np.asarray(y)[ok], x[ok], None if w is None else np.asarray(w)[ok]) - 0.5


def level_rates(x, y, w, levels: list[str]) -> dict[str, float]:
    """Weighted event rate of given categorical levels (nan if a level is absent)."""
    x = np.asarray(x, dtype=object)
    y = np.asarray(y)
    w = np.asarray(w, dtype=np.float64)
    out = {}
    for lev in levels:
        m = x == lev
        out[lev] = float((w[m] * y[m]).sum() / w[m].sum()) if m.any() else float("nan")
    return out


def univariate_train(df, label: str, features: list[str], categorical: set[str],
                     cfg: dict) -> tuple[dict[str, WoEBinner], pd.DataFrame]:
    """Fit WoE bins on train and compute train metrics for every feature."""
    y = df[label].to_numpy()
    w = df["sample_weight"].to_numpy().astype(np.float64)
    binners, rows = {}, []
    for f in features:
        x = df[f].to_numpy()
        is_cat = f in categorical
        b = fit_woe(x, y, w, f, is_cat, cfg)
        binners[f] = b
        idx = b.bin_index(x)
        risk = -b.transform(x)
        chi2, p, dof = chi2_test(idx, y)
        rows.append({"feature": f, "n_bins": b.n_bins, "iv_train": b.iv,
                     "ks_train": ks_statistic(y, risk, w), "auc_woe_train": weighted_auc(y, risk, w),
                     "cramers_v_train": cramers_v(idx, y, w), "chi2_train": chi2, "chi2_dof": dof,
                     "chi2_p": p, "direction_train": np.nan if is_cat else direction(y, x, w),
                     "n_unique_train": int(pd.Series(x).nunique(dropna=True))})
    return binners, pd.DataFrame(rows)


def univariate_apply(df, label: str, binners: dict[str, WoEBinner], suffix: str,
                     smoothing: float = 0.5) -> pd.DataFrame:
    """Out-of-sample IV / KS / AUC with the train bins and WoE (e.g. on validation)."""
    y = df[label].to_numpy()
    w = df["sample_weight"].to_numpy().astype(np.float64)
    rows = []
    for f, b in binners.items():
        x = df[f].to_numpy()
        idx = b.bin_index(x)
        k = b.n_bins + 1
        idx_all = np.where(idx < 0, k - 1, idx)
        good = np.bincount(idx_all, weights=w * (y == 0), minlength=k)
        bad = np.bincount(idx_all, weights=w * (y == 1), minlength=k)
        seen = (good + bad) > 0
        _, iv = woe_iv(good[seen], bad[seen], smoothing)   # IV of the train bins on this sample
        risk = -b.transform(x)
        rows.append({"feature": f,
                     f"iv_{suffix}": iv,
                     f"ks_{suffix}": ks_statistic(y, risk, w),
                     f"auc_woe_{suffix}": weighted_auc(y, risk, w),
                     f"direction_{suffix}": np.nan if b.kind == "categorical" else direction(y, x, w)})
    return pd.DataFrame(rows)
