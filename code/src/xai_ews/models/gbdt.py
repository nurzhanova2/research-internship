"""XGBoost and LightGBM: additional gradient-boosting baselines for paper 1 (post-hoc, after
test was opened). Same protocol as M1 CatBoost: row weight
omega x 1/r, 40 Optuna trials on the omega-weighted log-loss of validation half A, early stopping
after 100 rounds (at most 3000 trees), final model = mean logit of 5 GroupKFold fold models.

The search space mirrors CatBoost's five tuned parameters:
    depth 4-10, learning rate 0.02-0.3 (log), L2 1-30 (log), minimum leaf 1-2000 (log),
    row subsample 0.5-1.
XGBoost grows depth-wise trees with the histogram method; LightGBM grows leaf-wise trees capped at
2^depth leaves and the same maximum depth. Categorical features are passed as integer codes of
pandas categories and split natively by both libraries.
"""

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb

MAX_ITER = 3000
EARLY_STOP = 100
LIBS = ("xgboost", "lightgbm")


def suggest(trial) -> dict:
    return {"depth": trial.suggest_int("depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
            "l2": trial.suggest_float("l2", 1.0, 30.0, log=True),
            "min_leaf": trial.suggest_int("min_leaf", 1, 2000, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0)}


def as_frame(X: pd.DataFrame, categorical: list[str], n_categories: dict[str, int]) -> pd.DataFrame:
    """Integer category codes (CategoryEncoder) -> pandas category dtype with the encoder's full
    category set, so train, validation and test share the same categories."""
    X = X.copy()
    for c in categorical:
        X[c] = pd.Categorical(X[c].astype("int32"), categories=range(n_categories[c]))
    return X


def _native(lib: str, p: dict, seed: int) -> dict:
    if lib == "xgboost":
        return {"objective": "binary:logistic", "eval_metric": "logloss", "tree_method": "hist",
                "max_depth": p["depth"], "eta": p["learning_rate"], "lambda": p["l2"],
                "min_child_weight": p["min_leaf"], "subsample": p["subsample"], "max_cat_to_onehot": 1,
                "seed": seed, "nthread": -1, "verbosity": 0}
    return {"objective": "binary", "metric": "binary_logloss", "max_depth": p["depth"],
            "num_leaves": min(2 ** p["depth"], 1023), "learning_rate": p["learning_rate"], "lambda_l2": p["l2"],
            "min_data_in_leaf": p["min_leaf"], "bagging_fraction": p["subsample"], "bagging_freq": 1,
            "seed": seed, "num_threads": 0, "verbose": -1}


class Model:
    """A fitted booster of either library with a common `logit` method."""

    def __init__(self, lib: str, booster, n_trees: int):
        self.lib, self.booster, self.n_trees = lib, booster, n_trees

    def logit(self, X: pd.DataFrame) -> np.ndarray:
        if self.lib == "xgboost":
            d = xgb.DMatrix(X, enable_categorical=True)
            return self.booster.predict(d, output_margin=True, iteration_range=(0, self.n_trees))
        return self.booster.predict(X, raw_score=True, num_iteration=self.n_trees)

    def save(self, path) -> None:
        if self.lib == "xgboost":
            self.booster.save_model(str(path))
        else:
            self.booster.save_model(str(path), num_iteration=self.n_trees)

    @classmethod
    def load(cls, lib: str, path, n_trees: int) -> "Model":
        if lib == "xgboost":
            b = xgb.Booster()
            b.load_model(str(path))
            return cls(lib, b, n_trees)
        return cls(lib, lgb.Booster(model_file=str(path)), n_trees)


def fit(lib: str, X: pd.DataFrame, y, w, params: dict, eval_set=None, iterations: int | None = None,
        seed: int = 0) -> Model:
    """eval_set = (X, y, w): early stopping on its weighted log-loss, best iteration kept.
    Without eval_set: `iterations` trees."""
    p = _native(lib, params, seed)
    n = iterations or MAX_ITER
    if lib == "xgboost":
        dtrain = xgb.DMatrix(X, label=np.asarray(y), weight=np.asarray(w), enable_categorical=True)
        evals, es = [], None
        if eval_set is not None:
            evals = [(xgb.DMatrix(eval_set[0], label=np.asarray(eval_set[1]), weight=np.asarray(eval_set[2]),
                                  enable_categorical=True), "A")]
            es = EARLY_STOP
        b = xgb.train(p, dtrain, num_boost_round=n, evals=evals, early_stopping_rounds=es, verbose_eval=False)
        trees = b.best_iteration + 1 if eval_set is not None else n
        return Model(lib, b, trees)
    dtrain = lgb.Dataset(X, label=np.asarray(y), weight=np.asarray(w), free_raw_data=False)
    if eval_set is not None:
        dval = lgb.Dataset(eval_set[0], label=np.asarray(eval_set[1]), weight=np.asarray(eval_set[2]), reference=dtrain)
        b = lgb.train(p, dtrain, num_boost_round=n, valid_sets=[dval],
                      callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)])
        return Model(lib, b, b.best_iteration)
    b = lgb.train(p, dtrain, num_boost_round=n)
    return Model(lib, b, n)


def objective(lib: str, X: pd.DataFrame, y, w, Xa: pd.DataFrame, ya, wa, seed: int = 0):
    """Optuna objective: omega-weighted log-loss on half A; AUC and trees stored as attributes."""
    from xai_ews.evaluation.metrics import auc, log_loss, sigmoid

    def run(trial) -> float:
        m = fit(lib, X, y, w, suggest(trial), eval_set=(Xa, ya, wa), seed=seed)
        z = m.logit(Xa)
        trial.set_user_attr("trees", int(m.n_trees))
        trial.set_user_attr("auc_A", auc(ya, z))
        return log_loss(ya, sigmoid(z), wa)
    return run
