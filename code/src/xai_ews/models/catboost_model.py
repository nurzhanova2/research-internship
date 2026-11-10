"""M1 CatBoost: tuning, final fold models, out-of-fold predictions.

Device: CPU locally, GPU on Colab (environment variable XAI_CATBOOST_DEVICE=GPU). All CatBoost
models of stage 3 are tuned and trained on the same device (test protocol); GPU-trained models are
applied on CPU.

Fixed settings (not tuned, written in the test protocol):
    loss Logloss with row weight omega x 1/r; grow_policy Depthwise (so that min_data_in_leaf
    acts; the default symmetric trees ignore it); bootstrap Bernoulli (so that subsample acts);
    max_ctr_complexity 1 (no combinations of categorical features: 2x faster, 1.4 GB instead of
    2.2 GB on the 2.3 M-row train); at most MAX_ITER trees, early stopping after
    EARLY_STOP rounds without improvement of the omega-weighted log-loss on validation half A.
Tuned (Optuna TPE, 40 trials): depth 4-10, learning_rate, l2_leaf_reg, min_data_in_leaf,
subsample.

Final model = average of the logits of K = 5 GroupKFold models (folds by loan ID), each trained
on 4/5 of train with the tuned parameters and the tuned number of trees. Out-of-fold logits on
train and fold-average logits on the other samples are the input of the hybrid (stage 4).
"""

import os

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import GroupKFold

MAX_ITER = 3000
EARLY_STOP = 100
FIXED = {"loss_function": "Logloss", "eval_metric": "Logloss", "grow_policy": "Depthwise",
         "bootstrap_type": "Bernoulli", "max_ctr_complexity": 1, "allow_writing_files": False,
         "thread_count": -1, "verbose": 0}
DEVICE = os.environ.get("XAI_CATBOOST_DEVICE", "CPU").upper()
if DEVICE == "GPU":
    FIXED |= {"task_type": "GPU", "devices": "0"}
K_FOLDS = 5


def suggest(trial) -> dict:
    return {"depth": trial.suggest_int("depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 30.0, log=True),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 2000, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0)}


def pool(X: pd.DataFrame, y=None, w=None, cat_features=None) -> Pool:
    return Pool(X, label=None if y is None else np.asarray(y), weight=None if w is None else np.asarray(w),
                cat_features=[c for c in (cat_features or []) if c in X.columns])


def fit(train: Pool, params: dict, eval_pool: Pool | None = None, iterations: int | None = None,
        seed: int = 0) -> CatBoostClassifier:
    """With `eval_pool`: early stopping on it (best iteration kept). Without: `iterations` trees."""
    model = CatBoostClassifier(**FIXED, **params, random_seed=seed,
                               iterations=iterations or MAX_ITER,
                               early_stopping_rounds=EARLY_STOP if eval_pool is not None else None,
                               use_best_model=eval_pool is not None)
    model.fit(train, eval_set=eval_pool)
    return model


def logit(model: CatBoostClassifier, data: Pool) -> np.ndarray:
    return model.predict(data, prediction_type="RawFormulaVal")


def objective(train: Pool, eval_pool: Pool, y_eval, w_eval, seed: int = 0):
    """Optuna objective: omega-weighted log-loss on half A; AUC and trees stored as attributes."""
    from xai_ews.evaluation.metrics import auc, log_loss, sigmoid

    def run(trial) -> float:
        params = suggest(trial)
        model = fit(train, params, eval_pool, seed=seed)
        z = logit(model, eval_pool)
        trial.set_user_attr("trees", int(model.get_best_iteration()) + 1)
        trial.set_user_attr("auc_A", auc(y_eval, z))
        return log_loss(y_eval, sigmoid(z), w_eval)
    return run


def group_folds(groups, k: int = K_FOLDS) -> np.ndarray:
    """Fold id per row; all rows of a loan share a fold (GroupKFold)."""
    fold = np.empty(len(groups), np.int8)
    for i, (_, idx) in enumerate(GroupKFold(n_splits=k).split(np.zeros(len(groups)), groups=groups)):
        fold[idx] = i
    return fold


def fit_folds(X: pd.DataFrame, y, w, cat_features, params: dict, trees: int, folds: np.ndarray,
              seed: int = 0, progress=None) -> tuple[list[CatBoostClassifier], np.ndarray]:
    """K fold models and the out-of-fold logits on train."""
    oof = np.empty(len(X), np.float64)
    models = []
    y, w = np.asarray(y), np.asarray(w)
    for k in range(folds.max() + 1):
        tr, te = folds != k, folds == k
        m = fit(pool(X[tr], y[tr], w[tr], cat_features), params, iterations=trees, seed=seed + k)
        oof[te] = logit(m, pool(X[te], cat_features=cat_features))
        models.append(m)
        if progress:
            progress(k)
    return models, oof


def fold_average_logit(models: list[CatBoostClassifier], X: pd.DataFrame, cat_features) -> np.ndarray:
    p = pool(X, cat_features=cat_features)
    return np.mean([logit(m, p) for m in models], axis=0)
