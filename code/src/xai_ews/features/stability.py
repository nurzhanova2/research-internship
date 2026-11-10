"""Stability over time (stage 2, steps 6-7).

PSI (population stability index) compares the share of the population in each train WoE bin
between train and a later sample: PSI = sum (new - ref) ln(new / ref). Rule of thumb:
< 0.1 stable, 0.1-0.25 moderate shift, > 0.25 large shift.

Macro features change with calendar time by construction, so for them the rule asks for a
stable association instead: same direction and enough strength (IV) in every observation year.
"""

import numpy as np
import pandas as pd

from xai_ews.data.audit import psi_from_shares
from xai_ews.features.univariate import direction, weighted_auc
from xai_ews.features.woe import WoEBinner, woe_iv


def bin_shares(binner: WoEBinner, x, w=None) -> np.ndarray:
    """Weighted share per train bin; the last slot holds values unseen on train."""
    idx = binner.bin_index(x)
    k = binner.n_bins + 1
    idx = np.where(idx < 0, k - 1, idx)
    w = np.ones(len(idx)) if w is None else np.asarray(w, dtype=np.float64)
    c = np.bincount(idx, weights=w, minlength=k)
    return c / max(c.sum(), 1e-12)


def psi_binned(binner: WoEBinner, x_ref, w_ref, x_new, w_new=None) -> float:
    return psi_from_shares(bin_shares(binner, x_ref, w_ref), bin_shares(binner, x_new, w_new))


def strength_by_period(binner: WoEBinner, x, y, w, period, smoothing: float = 0.5) -> pd.DataFrame:
    """IV (train bins), WoE-AUC and raw direction (AUC - 0.5, numeric only) within each period."""
    x = np.asarray(x)
    y = np.asarray(y)
    w = np.asarray(w, dtype=np.float64)
    period = np.asarray(period)
    rows = []
    for p in np.unique(period):
        m = period == p
        idx = binner.bin_index(x[m])
        k = binner.n_bins + 1
        idx = np.where(idx < 0, k - 1, idx)
        good = np.bincount(idx, weights=w[m] * (y[m] == 0), minlength=k)
        bad = np.bincount(idx, weights=w[m] * (y[m] == 1), minlength=k)
        seen = (good + bad) > 0
        _, iv = woe_iv(good[seen], bad[seen], smoothing)
        rows.append({"feature": binner.feature, "period": p, "rows": int(m.sum()),
                     "events": int((y[m] == 1).sum()), "iv": iv,
                     "auc_woe": weighted_auc(y[m], -binner.transform(x[m]), w[m]),
                     "direction": np.nan if binner.kind == "categorical" else direction(y[m], x[m], w[m])})
    return pd.DataFrame(rows)


def macro_stable(by_period: pd.DataFrame, iv_min: float, min_effect: float) -> dict:
    """Stability rule for one macro feature: IV >= iv_min in every period and the same
    direction in every period (periods with |direction| < min_effect count as a sign break)."""
    d = by_period["direction"].to_numpy()
    signs = np.sign(np.where(np.abs(d) >= min_effect, d, 0.0))
    same_sign = bool(np.all(signs == signs[0]) and signs[0] != 0)
    strong = bool((by_period["iv"] >= iv_min).all())
    return {"feature": by_period["feature"].iloc[0], "iv_min_period": float(by_period["iv"].min()),
            "iv_max_period": float(by_period["iv"].max()), "sign_consistent": same_sign,
            "strength_ok": strong, "stable": same_sign and strong}


def sign_check(expected: str, observed: float, min_effect: float) -> str:
    """'match' / 'mismatch' / 'weak' (|effect| < min_effect) / 'no prior'."""
    if expected not in {"+", "-"}:
        return "no prior"
    if np.isnan(observed) or abs(observed) < min_effect:
        return "weak"
    return "match" if (observed > 0) == (expected == "+") else "mismatch"


def categorical_sign_check(rates: dict[str, float], risky: str, safe: str) -> str:
    r, s = rates.get(risky, np.nan), rates.get(safe, np.nan)
    if np.isnan(r) or np.isnan(s):
        return "level absent"
    return "match" if r > s else "mismatch"


def category_shares(x, w, categories: list[str]) -> np.ndarray:
    """Weighted shares of the given categories plus one slot for everything else."""
    s = pd.Series(np.asarray(x, dtype=object))
    w = np.ones(len(s)) if w is None else np.asarray(w, dtype=np.float64)
    pos = {c: i for i, c in enumerate(categories)}
    idx = s.map(pos).fillna(len(categories)).to_numpy().astype(int)
    c = np.bincount(idx, weights=w, minlength=len(categories) + 1)
    return c / max(c.sum(), 1e-12)


def psi_top_categories(x_ref, w_ref, x_new, w_new=None, top: int = 20) -> float:
    """PSI over the `top` most frequent (weighted) train categories plus "other"."""
    w_ref = np.ones(len(x_ref)) if w_ref is None else np.asarray(w_ref, dtype=np.float64)
    tops = pd.Series(w_ref).groupby(np.asarray(x_ref, dtype=object)).sum().nlargest(top).index.tolist()
    return psi_from_shares(category_shares(x_ref, w_ref, tops), category_shares(x_new, w_new, tops))
