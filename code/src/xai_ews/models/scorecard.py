"""B0 scorecard: logistic regression on WoE values.

Every feature is replaced by the WoE of its stage-2 bin (data/interim/stage2/woe_binners_<task>.json,
fitted on train), then an L2-regularised logistic regression is fitted with row weights
omega x 1/r. Only the regularisation strength C is tuned: about 10 values, chosen by
the omega-weighted log-loss on validation half A.

WoE is positive for safer-than-average bins, so every coefficient is expected to be NEGATIVE
(higher WoE -> lower log-odds of the event); a positive coefficient flags a feature whose effect
reverses once the other features are in the model.
"""

import numpy as np
import pandas as pd
import polars as pl
from sklearn.linear_model import LogisticRegression

from xai_ews.features.woe import load_binners
from xai_ews.paths import STAGE2_INTERIM_DIR

C_GRID = np.logspace(-4, 2, 10)


def binners(task: str) -> dict:
    return load_binners(STAGE2_INTERIM_DIR / f"woe_binners_{task}.json")


def woe_matrix(df: pl.DataFrame, features: list[str], task: str) -> np.ndarray:
    b = binners(task)
    return np.column_stack([b[f].transform(df[f].to_numpy()) for f in features]).astype(np.float32)


def fit(x: np.ndarray, y, w, c: float) -> LogisticRegression:
    m = LogisticRegression(C=c, max_iter=500, tol=1e-6)
    return m.fit(x, np.asarray(y), sample_weight=np.asarray(w) / np.mean(w))


def logit(model: LogisticRegression, x: np.ndarray) -> np.ndarray:
    return model.decision_function(x)


def coefficients(model: LogisticRegression, features: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"feature": features, "coef": model.coef_[0],
                         "sign_ok": model.coef_[0] < 0}).sort_values("coef", ignore_index=True)


def points_table(model: LogisticRegression, features: list[str], task: str, base: float = 600.0,
                 base_odds: float = 50.0, pdo: float = 20.0) -> pd.DataFrame:
    """Scorecard points per bin: score = offset - factor x logit(PD), factor = pdo / ln 2, so the
    score rises by `pdo` when the good:bad odds double and equals `base` at odds `base_odds`.
    The intercept is spread evenly over the features; higher points = safer."""
    factor = pdo / np.log(2)
    offset = base - factor * np.log(base_odds)
    b = binners(task)
    n = len(features)
    rows = []
    for f, c in zip(features, model.coef_[0], strict=True):
        for label, woe in zip(b[f].labels, b[f].woe, strict=True):
            rows.append({"feature": f, "bin": label, "woe": woe,
                         "points": offset / n - factor * (c * woe + model.intercept_[0] / n)})
    return pd.DataFrame(rows)


def load(task: str) -> dict:
    """{'model', 'platt', 'features'} saved by notebook 03_1."""
    import pickle

    from xai_ews.paths import MODELS_DIR
    return pickle.loads((MODELS_DIR / f"b0_{task}.pkl").read_bytes())


def predict(task: str, df: pl.DataFrame) -> np.ndarray:
    b0 = load(task)
    return logit(b0["model"], woe_matrix(df, b0["features"], task))
