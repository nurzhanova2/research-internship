"""Model-based importance on CatBoost (stage 2, step 8): SHAP, permutation, Boruta.

    SHAP          mean |SHAP value| per feature (TreeSHAP, raw logit scale), share of the total
    permutation   drop of validation AUC when one feature column is shuffled (model fixed)
    Boruta        each round adds a shuffled copy ("shadow") of every feature and refits; a feature
                  scores a hit when its importance beats the best shadow. After n rounds a
                  two-sided binomial test (p = 0.5, Bonferroni over features) marks it confirmed
                  (more hits than chance), rejected (fewer) or tentative (Kursa & Rudnicki 2010).
"""

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from scipy import stats

from xai_ews.features.univariate import weighted_auc


def make_pool(X: pd.DataFrame, y=None, w=None, cat_features: list[str] | None = None) -> Pool:
    cats = [c for c in (cat_features or []) if c in X.columns]
    return Pool(X, label=None if y is None else np.asarray(y), weight=None if w is None else np.asarray(w),
                cat_features=cats)


def fit_catboost(X: pd.DataFrame, y, w, cat_features: list[str], params: dict, seed: int = 0,
                 threads: int = -1) -> CatBoostClassifier:
    model = CatBoostClassifier(**params, loss_function="Logloss", random_seed=seed, verbose=0,
                               thread_count=threads, allow_writing_files=False)
    model.fit(make_pool(X, y, w, cat_features))
    return model


def predict_logit(model: CatBoostClassifier, X: pd.DataFrame, cat_features: list[str]) -> np.ndarray:
    return model.predict(make_pool(X, cat_features=cat_features), prediction_type="RawFormulaVal")


def shap_importance(model: CatBoostClassifier, X: pd.DataFrame, cat_features: list[str]) -> pd.DataFrame:
    sv = model.get_feature_importance(make_pool(X, cat_features=cat_features), type="ShapValues")
    mean_abs = np.abs(sv[:, :-1]).mean(axis=0)            # last column = expected value
    return pd.DataFrame({"feature": list(X.columns), "shap_mean_abs": mean_abs,
                         "shap_share": mean_abs / mean_abs.sum()}).sort_values(
        "shap_mean_abs", ascending=False, ignore_index=True)


def permutation_importance(model: CatBoostClassifier, X: pd.DataFrame, y, cat_features: list[str],
                           repeats: int = 3, seed: int = 0, w=None) -> pd.DataFrame:
    """AUC drop per feature (mean and sd over repeats); > 0 means the model uses the feature."""
    rng = np.random.default_rng(seed)
    base = weighted_auc(y, predict_logit(model, X, cat_features), w)
    rows = []
    for f in X.columns:
        drops = []
        orig = X[f].to_numpy().copy()
        for _ in range(repeats):
            X[f] = orig[rng.permutation(len(orig))]
            drops.append(base - weighted_auc(y, predict_logit(model, X, cat_features), w))
        X[f] = orig
        rows.append({"feature": f, "perm_auc_drop": float(np.mean(drops)),
                     "perm_auc_drop_sd": float(np.std(drops, ddof=1)) if repeats > 1 else np.nan})
    return pd.DataFrame(rows).sort_values("perm_auc_drop", ascending=False, ignore_index=True)


def boruta(X: pd.DataFrame, y, w, cat_features: list[str], params: dict, n_iter: int = 20,
           alpha: float = 0.05, seed: int = 0, progress=None) -> pd.DataFrame:
    """Boruta with CatBoost PredictionValuesChange importance; returns hits and decision."""
    rng = np.random.default_rng(seed)
    feats = list(X.columns)
    shadow_names = [f"shadow__{f}" for f in feats]
    hits = np.zeros(len(feats), dtype=int)
    imp_sum = np.zeros(len(feats))
    shadow_max = []
    for it in range(n_iter):
        shadows = pd.DataFrame({s: X[f].to_numpy()[rng.permutation(len(X))]
                                for s, f in zip(shadow_names, feats)}, index=X.index)
        XX = pd.concat([X, shadows], axis=1)
        cats = [*cat_features, *[f"shadow__{c}" for c in cat_features if c in feats]]
        model = fit_catboost(XX, y, w, cats, params, seed=seed + it)
        imp = model.get_feature_importance()
        real, shad = imp[: len(feats)], imp[len(feats):]
        hits += real > shad.max()
        imp_sum += real
        shadow_max.append(shad.max())
        del XX, shadows, model
        if progress:
            progress(it)
    p_hi = stats.binom.sf(hits - 1, n_iter, 0.5)           # P(hits >= observed)
    p_lo = stats.binom.cdf(hits, n_iter, 0.5)              # P(hits <= observed)
    a = alpha / len(feats)
    decision = np.where(p_hi < a, "confirmed", np.where(p_lo < a, "rejected", "tentative"))
    return pd.DataFrame({"feature": feats, "boruta_hits": hits, "boruta_iterations": n_iter,
                         "boruta_mean_importance": imp_sum / n_iter,
                         "boruta_shadow_max_mean": float(np.mean(shadow_max)),
                         "boruta_decision": decision}).sort_values(
        "boruta_mean_importance", ascending=False, ignore_index=True)


def cache_paths(task: str, features: list[str], params: dict, name: str, seed: int = 0):
    """(predictions .parquet, model .pkl) of fit_predict_validation in data/interim/stage2/models/;
    the key is the task, features, params and seed, so a changed set never reuses a stale model."""
    import hashlib
    import json

    from xai_ews.features.data import population
    from xai_ews.paths import STAGE2_INTERIM_DIR

    key = hashlib.sha1(json.dumps([task, population(task), features, params, seed]).encode()).hexdigest()[:12]
    stem = STAGE2_INTERIM_DIR / "models" / f"{name}_{task}_{key}"
    return stem.with_suffix(".parquet"), stem.with_suffix(".pkl")


def fit_predict_validation(task: str, features: list[str], params: dict, name: str,
                           seed: int = 0, recompute: bool = False):
    """CatBoost on the full train (r = 0.2, weights 1/r) -> raw logits on all validation rows,
    hurricane rows flagged. Model and predictions are cached in
    data/interim/stage2/ under `name`; the cache is reused only for the same features and params.

    Returns (predictions frame: loan_identifier, m, label, val_hurricane_2017, logit; model;
    fitted CategoryEncoder).
    """
    import pickle

    import polars as pl

    from xai_ews.config import feature_config
    from xai_ews.data.features import categorical_features
    from xai_ews.features.data import CategoryEncoder, load_train, load_validation, to_model_frame

    f_pred, f_model = cache_paths(task, features, params, name, seed)
    f_pred.parent.mkdir(parents=True, exist_ok=True)
    cats = [c for c in categorical_features() if c in features]
    if f_pred.exists() and f_model.exists() and not recompute:
        model, enc = pickle.loads(f_model.read_bytes())
        return pl.read_parquet(f_pred), model, enc
    top_n = feature_config()["rules"]["stability"]["high_cardinality_top_n"]
    tr = load_train(task, features)
    enc = CategoryEncoder(top_n).fit(tr, cats)
    X = to_model_frame(tr, features, enc)
    y, w = tr[task].to_numpy(), tr["sample_weight"].to_numpy()
    del tr
    model = fit_catboost(X, y, w, cats, params, seed=seed)
    del X
    va = load_validation(task, features, hurricanes=True)
    logit = predict_logit(model, to_model_frame(va, features, enc), cats)
    pred = va.select("loan_identifier", "m", task, "val_hurricane_2017").with_columns(
        pl.Series("logit", logit.astype(np.float32)))
    pred.write_parquet(f_pred)
    f_model.write_bytes(pickle.dumps((model, enc)))
    return pred, model, enc


def cv_oof_logits(task: str, feature_sets: dict[str, list[str]], params: dict, n_folds: int = 3,
                  seed: int = 0, name: str = "cv"):
    """Out-of-fold CatBoost logits on TRAIN only (no validation): GroupKFold by loan, so all
    snapshots of a loan fall into the same fold. One model per (feature set, fold), weights 1/r.
    Returns {set name: polars frame loan_identifier, m, label, val_hurricane_2017 (=False), logit};
    each set is cached in data/interim/stage2/models/ under its features, params and folds.
    """
    import hashlib
    import json

    import polars as pl
    from sklearn.model_selection import GroupKFold

    from xai_ews.config import feature_config
    from xai_ews.data.features import categorical_features
    from xai_ews.features.data import CategoryEncoder, load_train, to_model_frame
    from xai_ews.paths import STAGE2_INTERIM_DIR

    d = STAGE2_INTERIM_DIR / "models"
    d.mkdir(parents=True, exist_ok=True)
    paths = {s: d / (f"{name}_{s}_{task}_" + hashlib.sha1(json.dumps(
        [task, f, params, n_folds, seed]).encode()).hexdigest()[:12] + ".parquet") for s, f in feature_sets.items()}
    out = {s: pl.read_parquet(p) for s, p in paths.items() if p.exists()}
    todo = [s for s in feature_sets if s not in out]
    if not todo:
        return out
    union = sorted({f for s in todo for f in feature_sets[s]})
    tr = load_train(task, union)
    y, w = tr[task].to_numpy(), tr["sample_weight"].to_numpy()
    groups = pd.factorize(tr["loan_identifier"].to_numpy())[0]
    folds = list(GroupKFold(n_splits=n_folds).split(np.zeros(len(y)), y, groups))
    top_n = feature_config()["rules"]["stability"]["high_cardinality_top_n"]
    for s in todo:
        feats = feature_sets[s]
        cats = [c for c in categorical_features() if c in feats]
        logit = np.empty(len(y), dtype=np.float32)
        for k, (i_tr, i_te) in enumerate(folds):
            part = tr[i_tr]
            enc = CategoryEncoder(top_n).fit(part, cats)
            model = fit_catboost(to_model_frame(part, feats, enc), y[i_tr], w[i_tr], cats, params, seed=seed + k)
            del part
            logit[i_te] = predict_logit(model, to_model_frame(tr[i_te], feats, enc), cats)
            del model
        out[s] = tr.select("loan_identifier", "m", task).with_columns(
            pl.lit(False).alias("val_hurricane_2017"), pl.Series("logit", logit))
        out[s].write_parquet(paths[s])
    return out
