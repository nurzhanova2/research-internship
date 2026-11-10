"""Comparing two models' AUC (stage 2, step 9; methodology section 8).

    DeLong      paired test for two correlated AUCs on the same rows (DeLong et al. 1988, fast
                algorithm of Sun & Xu 2014). Assumes independent rows, so it is run on ONE
                random snapshot per loan.
    bootstrap   cluster bootstrap by loan on all snapshots: loans are resampled with
                replacement, every snapshot of a drawn loan comes along. Gives the CI of ΔAUC
                that respects the dependence of a loan's snapshots.
"""

import numpy as np
import pandas as pd
from scipy import stats


def one_snapshot_per_loan(loan_ids, seed: int = 0) -> np.ndarray:
    """Row indices of one randomly chosen snapshot for every loan."""
    rng = np.random.default_rng(seed)
    codes, _ = pd.factorize(pd.Series(np.asarray(loan_ids)))
    order = np.lexsort((rng.random(len(codes)), codes))
    first = np.r_[True, codes[order][1:] != codes[order][:-1]]
    return np.sort(order[first])


def _midranks(x: np.ndarray) -> np.ndarray:
    return stats.rankdata(x, method="average")


def delong_test(y, s1, s2) -> dict:
    """Paired DeLong test of AUC(s1) vs AUC(s2): AUCs, difference, its SE, z and two-sided p."""
    y = np.asarray(y)
    pos, neg = y == 1, y == 0
    m, n = pos.sum(), neg.sum()
    aucs, v10, v01 = [], [], []
    for s in (np.asarray(s1, float), np.asarray(s2, float)):
        tx, ty, tz = _midranks(s[pos]), _midranks(s[neg]), _midranks(np.r_[s[pos], s[neg]])
        aucs.append((tz[:m].sum() - m * (m + 1) / 2) / (m * n))
        v10.append((tz[:m] - tx) / n)                  # structural components of positives
        v01.append(1.0 - (tz[m:] - ty) / m)            # ... and of negatives
    s10, s01 = np.cov(np.vstack(v10)), np.cov(np.vstack(v01))
    cov = s10 / m + s01 / n
    diff = aucs[0] - aucs[1]
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    se = float(np.sqrt(max(var, 0.0)))
    z = diff / se if se > 0 else 0.0
    return {"auc_1": float(aucs[0]), "auc_2": float(aucs[1]), "delta": float(diff), "se": se,
            "z": float(z), "p": float(2 * stats.norm.sf(abs(z))), "n": int(m + n), "events": int(m)}


class _SortedAUC:
    """AUC with arbitrary row weights in O(n): the sort order and tie groups are fixed once."""

    def __init__(self, y, score):
        self.order = np.argsort(score, kind="stable")
        s = np.asarray(score)[self.order]
        self.starts = np.flatnonzero(np.r_[True, s[1:] != s[:-1]])
        self.pos = (np.asarray(y) == 1)[self.order]

    def __call__(self, w: np.ndarray) -> float:
        ws = w[self.order]
        wp = np.add.reduceat(np.where(self.pos, ws, 0.0), self.starts)
        wn = np.add.reduceat(np.where(self.pos, 0.0, ws), self.starts)
        below = np.cumsum(wn) - wn
        return float(np.sum(wp * (below + 0.5 * wn)) / (wp.sum() * wn.sum()))


def cluster_bootstrap_auc(y, scores: dict[str, np.ndarray], clusters, b: int = 1000, seed: int = 0,
                          reference: str | None = None) -> pd.DataFrame:
    """Point AUC and 95 % percentile CI per model; with `reference`, also ΔAUC = model − reference
    with its CI and the bootstrap share of Δ <= 0. Resampling unit = loan."""
    rng = np.random.default_rng(seed)
    codes, uniq = pd.factorize(pd.Series(np.asarray(clusters)))
    k = len(uniq)
    aucs = {name: _SortedAUC(y, s) for name, s in scores.items()}
    ones = np.ones(len(codes))
    point = {name: f(ones) for name, f in aucs.items()}
    draws = {name: np.empty(b) for name in scores}
    for i in range(b):
        w = np.bincount(rng.integers(0, k, k), minlength=k)[codes].astype(np.float64)
        for name, f in aucs.items():
            draws[name][i] = f(w)
    rows = []
    for name in scores:
        row = {"model": name, "auc": point[name],
               "auc_lo": float(np.percentile(draws[name], 2.5)),
               "auc_hi": float(np.percentile(draws[name], 97.5))}
        if reference is not None and name != reference:
            d = draws[name] - draws[reference]
            row |= {"delta_auc": point[name] - point[reference],
                    "delta_lo": float(np.percentile(d, 2.5)), "delta_hi": float(np.percentile(d, 97.5)),
                    "share_delta_le_0": float(np.mean(d <= 0))}
        rows.append(row)
    return pd.DataFrame(rows)


def compare_models(preds: dict, label: str, reference: str, b: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Validation comparison of several models against `reference`.

    `preds` maps model name -> polars frame from importance.fit_predict_validation (same rows for
    every model). Main sample = without the hurricane rows: AUC with a cluster
    bootstrap CI by loan and ΔAUC vs the reference; DeLong on one random snapshot per loan.
    Sensitivity: point AUC with the hurricane rows.
    """
    import polars as pl

    from xai_ews.features.univariate import weighted_auc

    keys = ["loan_identifier", "m"]
    base = preds[reference].select(*keys, label, "val_hurricane_2017")
    for name, p in preds.items():
        base = base.join(p.select(*keys, pl.col("logit").alias(name)), on=keys, how="left")
    main = base.filter(~pl.col("val_hurricane_2017"))
    y = main[label].to_numpy()
    scores = {n: main[n].to_numpy().astype(np.float64) for n in preds}
    out = cluster_bootstrap_auc(y, scores, main["loan_identifier"].to_numpy(), b=b, seed=seed,
                                reference=reference)
    one = one_snapshot_per_loan(main["loan_identifier"].to_numpy(), seed=seed)
    dl = []
    for n in preds:
        r = delong_test(y[one], scores[n][one], scores[reference][one]) if n != reference else {}
        dl.append({"model": n, "delong_delta_one_snapshot": r.get("delta", np.nan),
                   "delong_p": r.get("p", np.nan),
                   "auc_with_hurricanes": weighted_auc(base[label].to_numpy(), base[n].to_numpy())})
    out = out.merge(pd.DataFrame(dl), on="model")
    out.attrs["counts"] = {"rows": main.height, "loans": int(main["loan_identifier"].n_unique()),
                           "events_obs": int(y.sum()),
                           "event_loans": int(main.filter(pl.col(label) == 1)["loan_identifier"].n_unique()),
                           "one_snapshot_rows": len(one), "one_snapshot_events": int(y[one].sum())}
    return out



def confusion_at_threshold(pred, label: str, threshold: float = 0.5, score: str = "logit") -> dict:
    """Confusion matrix with an alert when the predicted PD (sigmoid of the logit) >= threshold.

    Returns counts tp / fp / fn / tn, accuracy = (tp + tn) / n, balanced accuracy = mean of the
    recall of each class (cannot be inflated by the majority class), precision = tp / (tp + fp),
    recall = tp / (tp + fn) and F1 = harmonic mean of precision and recall.
    """
    y = pred[label].to_numpy().astype(bool)
    a = 1.0 / (1.0 + np.exp(-pred[score].to_numpy().astype(np.float64))) >= threshold
    tp, fp = int((y & a).sum()), int((~y & a).sum())
    fn, tn = int((y & ~a).sum()), int((~y & ~a).sum())
    precision, recall = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"threshold": threshold, "rows": len(y), "events": tp + fn, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "accuracy": (tp + tn) / max(len(y), 1), "balanced_accuracy": (recall + tn / max(tn + fp, 1)) / 2,
            "precision": precision, "recall": recall, "f1": f1}
