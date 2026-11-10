"""Model comparison tables of stage 3 (test protocol, section 6), used for test and stress samples.

    metrics_table   every metric of methodology 7 per model on one sample; 95 % cluster-bootstrap
                    CIs (by loan) for AUC, KS and the captured EL share on the alert list
                    (calibrated PD >= threshold), point values for the rest
    pairwise        ΔAUC for pairs of models: cluster-bootstrap CI on all snapshots and DeLong on one
                    random snapshot per loan, Holm correction over the pairs
"""

import numpy as np
import pandas as pd

from xai_ews.evaluation.el import alert_mask
from xai_ews.evaluation.metrics import all_metrics, auc
from xai_ews.evaluation.stats import (
    bootstrap_weights,
    delong_test,
    holm,
    one_snapshot_per_loan,
    weighted_auc_fn,
    weighted_ks_fn,
)


def metrics_table(y, pd_by_model: dict[str, np.ndarray], loans, lgd=None, ead=None,
                  threshold: float | None = None, b: int = 1000, seed: int = 0) -> pd.DataFrame:
    """One row per model. With lgd / ead also the UPB-weighted AUC, the alert list (calibrated
    PD >= threshold: share of rows, precision, recall) and the captured EL share on it."""
    y = np.asarray(y)
    fns, rows = {}, {}
    loss = None if lgd is None else y * np.asarray(lgd, np.float64) * np.asarray(ead, np.float64)
    for name, p in pd_by_model.items():
        rows[name] = {"model": name, **all_metrics(y, p)}
        fns[(name, "auc")] = weighted_auc_fn(y, p)
        fns[(name, "ks")] = weighted_ks_fn(y, p)
        if lgd is not None:
            al = alert_mask(p, threshold)                     # the list is fixed, loans re-weighted
            rows[name]["auc_upb_weighted"] = auc(y, p, ead)
            rows[name]["alert_share"] = float(al.mean())
            rows[name]["alert_precision"] = float(y[al].mean()) if al.any() else np.nan
            rows[name]["alert_recall"] = float(y[al].sum() / y.sum())
            rows[name]["el_share_alert"] = float(loss[al].sum() / loss.sum())
            fns[(name, "el_share_alert")] = (lambda w, al=al: float((w * loss)[al].sum() / (w * loss).sum()))
    draws = {k: np.empty(b) for k in fns}
    for i, w in enumerate(bootstrap_weights(loans, b, seed)):
        for k, f in fns.items():
            draws[k][i] = f(w)
    for (name, metric), d in draws.items():
        rows[name][f"{metric}_lo"] = float(np.percentile(d, 2.5))
        rows[name][f"{metric}_hi"] = float(np.percentile(d, 97.5))
    out = pd.DataFrame(rows.values()).set_index("model")
    return out


def pairwise(y, score_by_model: dict[str, np.ndarray], loans, pairs: list[tuple[str, str]],
             b: int = 1000, seed: int = 0) -> pd.DataFrame:
    """ΔAUC = AUC(a) − AUC(b) per pair: bootstrap CI (all snapshots) and DeLong p on one random
    snapshot per loan, Holm-adjusted over `pairs`."""
    y = np.asarray(y)
    fns = {m: weighted_auc_fn(y, s) for m, s in score_by_model.items() if any(m in pr for pr in pairs)}
    draws = {m: np.empty(b) for m in fns}
    for i, w in enumerate(bootstrap_weights(loans, b, seed)):
        for m, f in fns.items():
            draws[m][i] = f(w)
    one = one_snapshot_per_loan(loans, seed=seed)
    rows = []
    for a, c in pairs:
        d = draws[a] - draws[c]
        dl = delong_test(y[one], score_by_model[a][one], score_by_model[c][one])
        rows.append({"pair": f"{a} − {c}", "delta_auc": fns[a](np.ones(len(y))) - fns[c](np.ones(len(y))),
                     "lo": float(np.percentile(d, 2.5)), "hi": float(np.percentile(d, 97.5)),
                     "share_le_0": float(np.mean(d <= 0)), "delong_delta_one_snapshot": dl["delta"],
                     "delong_p": dl["p"], "one_snapshot_rows": dl["n"], "one_snapshot_events": dl["events"]})
    out = pd.DataFrame(rows)
    out["delong_p_holm"] = holm(out["delong_p"])
    return out


def h4_check(d, label: str, pd_with: str, pd_without: str, threshold: float | None = None,
             lgd_col: str = "lgd", b: int = 1000, seed: int = 0) -> dict:
    """H4 decision rule on a polars frame with columns loan_identifier, upb, `lgd_col`, the label
    and the two calibrated PD columns (with / without the exposure weight).

    (a) Δ captured EL share on the alert list (PD >= threshold), 95 % cluster-bootstrap CI above 0;
    (b) |Δ calibration slope| <= 0.1 and |Δ calibration intercept| <= 0.1."""
    from xai_ews.evaluation.metrics import calibration_slope_intercept

    y, lgd, ead = d[label].to_numpy(), d[lgd_col].to_numpy(), d["upb"].to_numpy()
    pa, pb = d[pd_with].to_numpy().astype(np.float64), d[pd_without].to_numpy().astype(np.float64)
    loss = y * lgd * ead
    ta, tb = alert_mask(pa, threshold), alert_mask(pb, threshold)

    def share(top, w):
        return float((w * loss)[top].sum() / (w * loss).sum())
    draws = np.array([share(ta, w) - share(tb, w) for w in bootstrap_weights(d["loan_identifier"].to_numpy(), b, seed)])
    sa, ia = calibration_slope_intercept(y, pa)
    sb, ib = calibration_slope_intercept(y, pb)
    out = {"el_share_with": share(ta, 1.0), "el_share_without": share(tb, 1.0),
           "delta": share(ta, 1.0) - share(tb, 1.0), "delta_lo": float(np.percentile(draws, 2.5)),
           "delta_hi": float(np.percentile(draws, 97.5)), "slope_with": sa, "slope_without": sb, "d_slope": sa - sb,
           "intercept_with": ia, "intercept_without": ib, "d_intercept": ia - ib,
           "auc_with": auc(y, pa), "auc_without": auc(y, pb)}
    out["el_part"] = bool(out["delta_lo"] > 0)
    out["calibration_part"] = bool(abs(out["d_slope"]) <= 0.1 and abs(out["d_intercept"]) <= 0.1)
    out["verdict"] = "confirmed" if out["el_part"] and out["calibration_part"] else (
        "partial" if out["el_part"] or out["calibration_part"] else "rejected")
    return out
