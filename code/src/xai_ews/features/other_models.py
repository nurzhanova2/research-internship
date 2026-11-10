"""Steps 8-9 of the feature selection repeated with other learners (stage 2).

The selection itself uses CatBoost; this module checks that its decisions do not depend on that
choice. Same train (r = 0.2, weights 1/r), same validation, same rules as for CatBoost:
    step 8  drop a feature only if Boruta rejects it AND permutation importance <= 0
    step 9  drop a group if the model without it is significantly better (CI of ΔAUC > 0, DeLong)
Learners and their input:
    logistic_woe   logistic regression on the train WoE values of every feature (scorecard style)
    xgboost        gradient boosting; categoricals as pandas category of the train codes
    lightgbm       gradient boosting with leaf-wise trees; categoricals as pandas category
    random_forest  bagged deep trees; categoricals as their train WoE values (integer codes would
                   impose a fake order)
No CatBoost import here: keep XGBoost / LightGBM and CatBoost / PyTorch in separate processes (OpenMP).
"""

import os

import numpy as np
import pandas as pd
from scipy import stats

from xai_ews.features.univariate import weighted_auc

KINDS = ("logistic_woe", "xgboost", "lightgbm", "random_forest")
WOE_INPUT = {"logistic_woe": "all", "random_forest": "categorical"}   # which columns become WoE


def model_frame(df, features: list[str], encoder, kind: str, binners: dict | None = None) -> pd.DataFrame:
    """Model input of `kind` (see the module docstring)."""
    from xai_ews.features.data import to_model_frame

    X = to_model_frame(df, features, encoder)
    woe = WOE_INPUT.get(kind)
    for c in features:
        if woe == "all" or (c in encoder.maps and woe == "categorical"):
            X[c] = binners[c].transform(df[c].to_numpy()).astype(np.float32)
        elif c in encoder.maps:
            X[c] = pd.Categorical(X[c], categories=sorted(encoder.maps[c].values()))
    return X


def fit_model(kind: str, X: pd.DataFrame, y, w, params: dict, seed: int = 0):
    p = params[kind]
    if kind == "logistic_woe":
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(C=p["C"], max_iter=p["max_iter"])
    elif kind == "xgboost":
        import xgboost as xgb
        model = xgb.XGBClassifier(n_estimators=p["iterations"], max_depth=p["depth"],
                                  learning_rate=p["learning_rate"], reg_lambda=p.get("l2", 1.0),
                                  min_child_weight=p.get("min_leaf", 1), subsample=p.get("subsample", 1.0),
                                  tree_method="hist", enable_categorical=True, max_cat_to_onehot=1, n_jobs=-1,
                                  device=os.environ.get("XAI_XGB_DEVICE", "cpu"),
                                  random_state=seed, verbosity=0)
    elif kind == "lightgbm":
        import lightgbm as lgb
        model = lgb.LGBMClassifier(n_estimators=p["iterations"], num_leaves=p["num_leaves"],
                                   max_depth=p.get("depth", -1), learning_rate=p["learning_rate"],
                                   reg_lambda=p.get("l2", 0.0), min_child_samples=p.get("min_leaf", 20),
                                   subsample=p.get("subsample", 1.0), subsample_freq=1 if "subsample" in p else 0,
                                   n_jobs=-1, random_state=seed, verbose=-1)
    elif kind == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=p["trees"], min_samples_leaf=p["min_samples_leaf"],
                                       max_features=p["max_features"], n_jobs=-1, random_state=seed)
    else:
        raise ValueError(kind)
    model.fit(X, np.asarray(y), sample_weight=np.asarray(w))
    if kind == "logistic_woe":
        model.input_sd_ = X.std().to_numpy()          # for the standardised-coefficient importance
    return model


def predict_logit(model, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "decision_function") and not hasattr(model, "estimators_"):
        return model.decision_function(X)
    p = np.clip(model.predict_proba(X)[:, 1], 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


def native_importance(model) -> np.ndarray:
    """XGBoost: total gain; LightGBM: total gain; random forest: impurity decrease;
    logistic regression: |coefficient| x SD of the input (standardised coefficient)."""
    if hasattr(model, "get_booster"):
        gain = model.get_booster().get_score(importance_type="total_gain")
        return np.array([gain.get(f, 0.0) for f in model.get_booster().feature_names])
    if hasattr(model, "booster_"):
        return model.booster_.feature_importance(importance_type="gain")
    if hasattr(model, "coef_"):
        return np.abs(model.coef_[0]) * model.input_sd_
    return model.feature_importances_


def permutation_importance(model, X: pd.DataFrame, y, w=None, repeats: int = 3, seed: int = 0) -> pd.DataFrame:
    """AUC drop per feature (mean and sd over repeats); > 0 means the model uses the feature."""
    rng = np.random.default_rng(seed)
    base = weighted_auc(y, predict_logit(model, X), w)
    rows = []
    for f in X.columns:
        orig = X[f].copy()
        drops = []
        for _ in range(repeats):
            X[f] = orig.to_numpy()[rng.permutation(len(orig))]
            if isinstance(orig.dtype, pd.CategoricalDtype):
                X[f] = pd.Categorical(X[f], categories=orig.cat.categories)
            drops.append(base - weighted_auc(y, predict_logit(model, X), w))
        X[f] = orig
        rows.append({"feature": f, "perm_auc_drop": float(np.mean(drops)),
                     "perm_auc_drop_sd": float(np.std(drops, ddof=1)) if repeats > 1 else np.nan})
    return pd.DataFrame(rows).sort_values("perm_auc_drop", ascending=False, ignore_index=True)


def boruta(kind: str, X: pd.DataFrame, y, w, params: dict, n_iter: int = 20, alpha: float = 0.05,
           seed: int = 0) -> pd.DataFrame:
    """Boruta with the model's native importance; same decision rule as the CatBoost version."""
    rng = np.random.default_rng(seed)
    feats = list(X.columns)
    hits = np.zeros(len(feats), dtype=int)
    imp_sum = np.zeros(len(feats))
    for it in range(n_iter):
        shadows = {}
        for f in feats:
            v = X[f].to_numpy()[rng.permutation(len(X))]
            shadows[f"shadow__{f}"] = (pd.Categorical(v, categories=X[f].cat.categories)
                                       if isinstance(X[f].dtype, pd.CategoricalDtype) else v)
        XX = pd.concat([X, pd.DataFrame(shadows, index=X.index)], axis=1)
        imp = native_importance(fit_model(kind, XX, y, w, params, seed=seed + it))
        real, shad = imp[: len(feats)], imp[len(feats):]
        hits += real > shad.max()
        imp_sum += real
        del XX, shadows
    p_hi = stats.binom.sf(hits - 1, n_iter, 0.5)
    p_lo = stats.binom.cdf(hits, n_iter, 0.5)
    a = alpha / len(feats)
    decision = np.where(p_hi < a, "confirmed", np.where(p_lo < a, "rejected", "tentative"))
    return pd.DataFrame({"feature": feats, "boruta_hits": hits, "boruta_iterations": n_iter,
                         "boruta_mean_importance": imp_sum / n_iter, "boruta_decision": decision}
                        ).sort_values("boruta_mean_importance", ascending=False, ignore_index=True)


def cache_paths(kind: str, task: str, features: list[str], params: dict, name: str, seed: int = 0,
                train_rows: int | None = None):
    """(predictions .parquet, model .pkl) of fit_predict_validation; reading only the parquet
    needs no model library (no OpenMP clash when several learners are compared in one process)."""
    import hashlib
    import json

    from xai_ews.features.data import population
    from xai_ews.paths import STAGE2_INTERIM_DIR

    key = hashlib.sha1(json.dumps([kind, task, population(task), features, params[kind], seed, train_rows])
                       .encode()).hexdigest()[:12]
    stem = STAGE2_INTERIM_DIR / "models" / f"{name}_{task}_{key}"
    return stem.with_suffix(".parquet"), stem.with_suffix(".pkl")


def fit_predict_validation(kind: str, task: str, features: list[str], params: dict, name: str,
                           seed: int = 0, train_rows: int | None = None):
    """Fit on train (optionally a subsample with all events) -> logits on all validation rows.

    Cached in data/interim/stage2/models/ under the model kind, features, params and rows.
    Returns (prediction frame like importance.fit_predict_validation, model, encoder, binners).
    """
    import pickle

    import polars as pl

    from xai_ews.config import feature_config
    from xai_ews.data.features import categorical_features
    from xai_ews.features.data import CategoryEncoder, load_train, load_validation, subsample
    from xai_ews.features.woe import load_binners
    from xai_ews.paths import STAGE2_INTERIM_DIR

    f_pred, f_model = cache_paths(kind, task, features, params, name, seed, train_rows)
    f_pred.parent.mkdir(parents=True, exist_ok=True)
    cats = [c for c in categorical_features() if c in features]
    binners = load_binners(STAGE2_INTERIM_DIR / f"woe_binners_{task}.json") if kind in WOE_INPUT else None
    if f_pred.exists() and f_model.exists():
        model, enc = pickle.loads(f_model.read_bytes())
        return pl.read_parquet(f_pred), model, enc, binners
    tr = load_train(task, features)
    if train_rows:
        tr = subsample(tr, task, train_rows, seed)
    enc = CategoryEncoder(feature_config()["rules"]["stability"]["high_cardinality_top_n"]).fit(tr, cats)
    X = model_frame(tr, features, enc, kind, binners)
    y, w = tr[task].to_numpy(), tr["sample_weight"].to_numpy()
    del tr
    model = fit_model(kind, X, y, w, params, seed=seed)
    del X
    va = load_validation(task, features, hurricanes=True)
    logit = predict_logit(model, model_frame(va, features, enc, kind, binners))
    pred = va.select("loan_identifier", "m", task, "val_hurricane_2017").with_columns(
        pl.Series("logit", logit.astype(np.float32)))
    pred.write_parquet(f_pred)
    f_model.write_bytes(pickle.dumps((model, enc)))
    return pred, model, enc, binners
