"""H2: is the importance of features heterogeneous across borrowers? (methodology 8)

Restricted model  logit P(y) = b0 + Σ_k b_k WoE_k + Σ_s g_s [segment = s]
Full model        + Σ_k Σ_s d_ks WoE_k [segment = s]   (feature x segment interactions)
LR test           2 (ll_full − ll_restricted) ~ chi2(df = K x (S − 1)).

Both are unpenalised logistic regressions on the r = 0.2 train WITHOUT weights: negatives are
sampled independently of the features, so all slopes (and the interactions) are estimated
without bias and only the intercept shifts by ln r (case-control property); a weighted
likelihood would not give a valid LR statistic. Snapshots of one loan are dependent, so the p-value
is optimistic; the decision also needs the effect size Δlog-loss on validation (decision rule of
H2), and one snapshot per loan is the sensitivity check.

Segments are fixed before the calculation (test protocol): FICO at origination < 680 / 680–739
/ >= 740 (missing -> < 680); sensitivity: current MTM LTV <= 80 / 80–95 / > 95 (missing -> <= 80).

auc_within_segments: AUC inside each segment, i.e. how well a model orders loans with the same
FICO band; the part of the overall AUC that comes only from separating the bands drops out.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from xai_ews.evaluation.stats import lr_test

SEGMENTS = {
    "fico": ("borrower_credit_score_at_origination", [680, 740], ["<680", "680-739", ">=740"]),
    "mtm_ltv": ("mtm_ltv", [80, 95], ["<=80", "80-95", ">95"]),
}


def segment_codes(values, kind: str) -> np.ndarray:
    """0 .. S-1 per row; missing values go to segment 0."""
    _, edges, _ = SEGMENTS[kind]
    v = np.asarray(values, np.float64)
    right = kind == "mtm_ltv"                         # LTV bands are (a, b], FICO bands [a, b)
    codes = np.searchsorted(np.asarray(edges, float), v, side="left" if right else "right")
    codes[np.isnan(v)] = 0
    return codes.astype(np.int8)


def auc_within_segments(y, score, values, kind: str = "fico", w=None) -> pd.DataFrame:
    """One row per segment: rows, events, event rate and the (weighted) AUC of `score` inside it."""
    from xai_ews.evaluation.metrics import auc

    y, score = np.asarray(y), np.asarray(score, np.float64)
    w = np.ones(len(y)) if w is None else np.asarray(w, np.float64)
    seg = segment_codes(values, kind)
    rows = []
    for s, label in enumerate(SEGMENTS[kind][2]):
        m = seg == s
        rows.append({"segment": label, "rows": int(m.sum()), "events": int(y[m].sum()),
                     "event_rate": float(y[m].mean()) if m.any() else np.nan,
                     "auc": auc(y[m], score[m], w[m]) if 0 < y[m].sum() < m.sum() else np.nan})
    return pd.DataFrame(rows)


def design(x: np.ndarray, seg: np.ndarray, n_seg: int, interactions: bool) -> np.ndarray:
    """[WoE, segment dummies (S-1)] (+ WoE x dummies)."""
    dummies = np.column_stack([(seg == s) for s in range(1, n_seg)]).astype(np.float32)
    parts = [x, dummies]
    if interactions:
        parts += [x * dummies[:, [j]] for j in range(n_seg - 1)]
    return np.hstack(parts).astype(np.float32)


def fit_unpenalised(X: np.ndarray, y) -> tuple[LogisticRegression, float]:
    """Model and its log-likelihood on the fitting rows."""
    m = LogisticRegression(penalty=None, max_iter=2000, tol=1e-8).fit(X, np.asarray(y))
    z = m.decision_function(X)
    y = np.asarray(y, np.float64)
    ll = float(np.sum(y * z - np.logaddexp(0.0, z)))
    return m, ll


def lr_interaction_test(x: np.ndarray, seg: np.ndarray, y, n_seg: int) -> dict:
    m0, ll0 = fit_unpenalised(design(x, seg, n_seg, False), y)
    m1, ll1 = fit_unpenalised(design(x, seg, n_seg, True), y)
    return {"restricted": m0, "full": m1, "ll_restricted": ll0, "ll_full": ll1,
            **lr_test(ll1, ll0, x.shape[1] * (n_seg - 1))}
