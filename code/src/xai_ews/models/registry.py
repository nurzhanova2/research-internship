"""Fitted stage-3 models on disk: data/processed/models/<name>/.

    CatBoost variant  5 GroupKFold models (fold_<k>.cbm), the CategoryEncoder, meta.json (features,
                      task, label, omega, r, tuned parameters, trees) and oof.parquet (out-of-fold
                      logits on train, the stage-4 input). Prediction = mean logit of the 5 models.
    MLP variant       5 seeds of the best configuration (seed_<k>.pt), encoder + preprocessor,
                      meta.json. Prediction = mean logit of the 5 seeds; per-seed logits too.

Hyperparameters come from the Optuna studies of scripts/tune.py; sensitivity
variants reuse those of the main model of their task (no new tuning).
"""

import json
import pickle

import numpy as np
import optuna
import polars as pl

from xai_ews.models import catboost_model as cb
from xai_ews.models.data import KEYS, arrays, fit_encoder, load_eval, load_train, to_model_frame
from xai_ews.paths import MODELS_DIR, OPTUNA_DB

STORAGE = f"sqlite:///{OPTUNA_DB}"
SEED = 20261025
N_SEEDS = 5


def study(name: str) -> optuna.Study:
    return optuna.load_study(study_name=name, storage=STORAGE)


def best(name: str) -> tuple[dict, dict]:
    """(params, user attributes) of the best completed trial."""
    t = study(name).best_trial
    return dict(t.params), dict(t.user_attrs)


def trials_frame(name: str):
    return study(name).trials_dataframe(attrs=("number", "value", "params", "user_attrs", "duration", "state"))


def _dir(name: str):
    d = MODELS_DIR / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def exists(name: str) -> bool:
    return (MODELS_DIR / name / "meta.json").exists()


def meta(name: str) -> dict:
    return json.loads((MODELS_DIR / name / "meta.json").read_text())


def fit_catboost_variant(name: str, task: str, features: list[str], categorical: list[str], study_name: str,
                         label: str | None = None, omega: str | None = "log", r: float = 0.2,
                         drop_prepaid: bool = False, train_filter: pl.Expr | None = None,
                         log=print) -> dict:
    """Train (or reuse) the 5 fold models of a CatBoost variant; `label` defaults to `task`."""
    if exists(name):
        return meta(name)
    label = label or task
    params, attrs = best(study_name)
    tr = load_train(label, features, categorical, r=r, omega=omega, drop_prepaid=drop_prepaid)
    if train_filter is not None:
        tr = tr.filter(train_filter)
    enc = fit_encoder(tr, categorical)
    X = to_model_frame(tr, features, enc)
    folds = cb.group_folds(tr["loan_identifier"].to_numpy())
    models, oof = cb.fit_folds(X, tr[label].to_numpy(), tr["weight"].to_numpy(), categorical, params,
                               attrs["trees"], folds, seed=SEED, progress=lambda k: log(f"{name}: fold {k} done"))
    d = _dir(name)
    for k, m in enumerate(models):
        m.save_model(str(d / f"fold_{k}.cbm"))
    (d / "encoder.pkl").write_bytes(pickle.dumps(enc))
    tr.select(*KEYS, label, "sample_weight", "weight").with_columns(pl.Series("fold", folds), pl.Series("logit", oof)) \
        .write_parquet(d / "oof.parquet")
    m = {"kind": "catboost", "task": task, "label": label, "features": features, "categorical": categorical,
         "omega": omega, "r": r, "drop_prepaid": drop_prepaid, "study": study_name, "params": params,
         "trees": attrs["trees"], "device": cb.DEVICE, "train_rows": tr.height, "train_events": int(tr[label].sum())}
    (d / "meta.json").write_text(json.dumps(m, indent=1))
    return m


def predict_catboost(name: str, df: pl.DataFrame) -> np.ndarray:
    from catboost import CatBoostClassifier

    m, d = meta(name), MODELS_DIR / name
    enc = pickle.loads((d / "encoder.pkl").read_bytes())
    models = []
    for k in range(cb.K_FOLDS):
        mk = CatBoostClassifier()
        mk.load_model(str(d / f"fold_{k}.cbm"))
        models.append(mk)
    return cb.fold_average_logit(models, to_model_frame(df, m["features"], enc), m["categorical"])


def fit_mlp_variant(name: str, task: str, features: list[str], categorical: list[str], study_name: str,
                    log=print) -> dict:
    """Train (or reuse) the best MLP configuration with N_SEEDS seeds; early stopping on half A."""
    import torch

    from xai_ews.models import mlp

    if exists(name):
        return meta(name)
    params, _ = best(study_name)
    cfg = mlp.Config(**params)
    tr = load_train(task, features, categorical)
    va = load_eval("validation", task, features, categorical).filter(pl.col("half") == "A")
    enc = fit_encoder(tr, categorical)
    xn, xc = arrays(tr, features, categorical, enc)
    pre = mlp.Preprocessor(seed=SEED).fit(xn)
    xn = pre.transform(xn)
    an, ac = arrays(va, features, categorical, enc)
    ev = (pre.transform(an), ac, va[task].to_numpy(), va["omega"].to_numpy())
    card = [len(enc.maps[c]) for c in categorical]
    d = _dir(name)
    epochs = []
    for s in range(N_SEEDS):
        model, hist = mlp.train(xn, xc, tr[task].to_numpy(), tr["weight"].to_numpy(), card, cfg, eval_data=ev,
                                seed=SEED + s)
        torch.save(model.state_dict(), d / f"seed_{s}.pt")
        epochs.append(hist["best_epoch"] + 1)
        log(f"{name}: seed {s} best epoch {hist['best_epoch'] + 1}, loss {hist['best_loss']:.6f}")
    (d / "encoder.pkl").write_bytes(pickle.dumps((enc, pre)))
    m = {"kind": "mlp", "task": task, "label": task, "features": features, "categorical": categorical,
         "study": study_name, "params": params, "cardinalities": card, "epochs": epochs,
         "n_num": int(xn.shape[1]), "train_rows": tr.height}
    (d / "meta.json").write_text(json.dumps(m, indent=1))
    return m


def predict_mlp(name: str, df: pl.DataFrame, per_seed: bool = False):
    import torch

    from xai_ews.models import mlp

    m, d = meta(name), MODELS_DIR / name
    enc, pre = pickle.loads((d / "encoder.pkl").read_bytes())
    xn, xc = arrays(df, m["features"], m["categorical"], enc)
    xn = pre.transform(xn)
    out = []
    for s in range(N_SEEDS):
        model = mlp.MLP(m["n_num"], m["cardinalities"], mlp.Config(**m["params"])).to(mlp.device())
        model.load_state_dict(torch.load(d / f"seed_{s}.pt", map_location=mlp.device()))
        out.append(mlp.predict_logit(model, xn, xc))
        del model
    if torch.backends.mps.is_available():     # the MPS cache keeps ~0.7 GB per call otherwise
        torch.mps.empty_cache()
    out = np.vstack(out)
    return (out.mean(axis=0), out) if per_seed else out.mean(axis=0)


def fit_gbdt_variant(name: str, lib: str, task: str, features: list[str], categorical: list[str],
                     study_name: str, log=print) -> dict:
    """XGBoost / LightGBM baseline: 5 GroupKFold models with the tuned parameters and
    number of trees, saved like the CatBoost variants (fold_<k>.model, encoder.pkl, meta.json, oof.parquet)."""
    from xai_ews.models import gbdt

    if exists(name):
        return meta(name)
    params, attrs = best(study_name)
    tr = load_train(task, features, categorical)
    enc = fit_encoder(tr, categorical)
    nc = {c: len(enc.maps[c]) for c in categorical}
    X = gbdt.as_frame(to_model_frame(tr, features, enc), categorical, nc)
    y, w = tr[task].to_numpy(), tr["weight"].to_numpy()
    folds = cb.group_folds(tr["loan_identifier"].to_numpy())
    oof = np.empty(len(X))
    d = _dir(name)
    for k in range(cb.K_FOLDS):
        trn, te = folds != k, folds == k
        m = gbdt.fit(lib, X[trn], y[trn], w[trn], params, iterations=attrs["trees"], seed=SEED + k)
        oof[te] = m.logit(X[te])
        m.save(d / f"fold_{k}.model")
        log(f"{name}: fold {k} done")
    (d / "encoder.pkl").write_bytes(pickle.dumps(enc))
    tr.select(*KEYS, task, "sample_weight", "weight").with_columns(pl.Series("fold", folds), pl.Series("logit", oof)) \
        .write_parquet(d / "oof.parquet")
    m = {"kind": lib, "task": task, "label": task, "features": features, "categorical": categorical, "omega": "log",
         "r": 0.2, "drop_prepaid": False, "study": study_name, "params": params, "trees": attrs["trees"],
         "n_categories": nc, "train_rows": tr.height, "train_events": int(tr[task].sum())}
    (d / "meta.json").write_text(json.dumps(m, indent=1))
    return m


def predict_gbdt(name: str, df: pl.DataFrame) -> np.ndarray:
    from xai_ews.models import gbdt

    m, d = meta(name), MODELS_DIR / name
    enc = pickle.loads((d / "encoder.pkl").read_bytes())
    X = gbdt.as_frame(to_model_frame(df, m["features"], enc), m["categorical"], m["n_categories"])
    return np.mean([gbdt.Model.load(m["kind"], d / f"fold_{k}.model", m["trees"]).logit(X)
                    for k in range(cb.K_FOLDS)], axis=0)


def predict(name: str, df: pl.DataFrame) -> np.ndarray:
    kind = meta(name)["kind"]
    if kind == "catboost":
        return predict_catboost(name, df)
    if kind in ("xgboost", "lightgbm"):
        return predict_gbdt(name, df)
    return predict_mlp(name, df)


def predict_any(name: str, df: pl.DataFrame) -> np.ndarray:
    """Raw logit of B0 (name 'b0_<task>'), a single stress model (<name>.pkl), a CatBoost or an
    MLP variant."""
    if name.startswith("b0_"):
        from xai_ews.models import scorecard
        return scorecard.predict(name.removeprefix("b0_"), df)
    if (MODELS_DIR / f"{name}.pkl").exists():
        model, enc, features = pickle.loads((MODELS_DIR / f"{name}.pkl").read_bytes())
        cats = [c for c in features if c in enc.maps]
        return cb.logit(model, cb.pool(to_model_frame(df, features, enc), cat_features=cats))
    return predict(name, df)


def calibrator(name: str, kind: str = "platt"):
    """Calibrator fitted on validation half B from the saved validation logits of the model
    (data/processed/predictions/<name>_validation.parquet); B0's Platt comes from its pickle."""
    from xai_ews.models.calibration import Isotonic, Platt
    from xai_ews.paths import PREDICTIONS_DIR

    if name.startswith("b0_") and kind == "platt":
        from xai_ews.models import scorecard
        return scorecard.load(name.removeprefix("b0_"))["platt"]
    p = pl.read_parquet(PREDICTIONS_DIR / f"{name}_validation.parquet").filter(pl.col("half") == "B")
    if name.startswith("b0_"):
        label = name.removeprefix("b0_")
        from xai_ews.models.data import load_eval
        lab = load_eval("validation", label, [], []).select("loan_identifier", "m", label)
        p = p.join(lab, on=["loan_identifier", "m"])
    else:
        label = meta(name)["label"]
    return (Platt if kind == "platt" else Isotonic)().fit(p["logit"].to_numpy(), p[label].to_numpy())


def fit_single_catboost(name: str, load, features: list[str], categorical: list[str], label: str,
                        study_name: str):
    """One CatBoost on CPU (no folds) with the tuned parameters and trees of `study_name`, for the
    stress-test robustness models (protocol 11); `load()` returns the training frame with a
    `weight` column. Saved as data/processed/models/<name>.pkl = (model, encoder, features)."""
    f = MODELS_DIR / f"{name}.pkl"
    if f.exists():
        return pickle.loads(f.read_bytes())
    params, attrs = best(study_name)
    tr = load()
    enc = fit_encoder(tr, categorical)
    model = cb.fit(cb.pool(to_model_frame(tr, features, enc), tr[label].to_numpy(), tr["weight"].to_numpy(),
                           categorical), params, iterations=attrs["trees"], seed=SEED)
    out = (model, enc, features)
    f.write_bytes(pickle.dumps(out))
    return out


def score_by_month(split: str, label: str, models: list[str], features: list[str], categorical: list[str],
                   keep: list[str], extra: list[str] | None = None, calibrate: bool = True,
                   exclusion: str = "main") -> pl.DataFrame:
    """Raw logits (`z_<model>`, float32) and validation-calibrated PDs (`pd_<model>`) of several
    models on a large split, one observation month at a time; only `keep` columns of the data are
    kept, so a 13 M-row stress sample needs about 1 GB instead of 6. `exclusion` as in load_eval."""
    from xai_ews.models.data import load_eval, split_months

    cals = {n: calibrator(n) for n in models} if calibrate else {}
    parts = []
    for m in split_months(split):
        d = load_eval(split, label, features, categorical, exclusion=exclusion, extra=extra, where=pl.col("m") == m)
        if not d.height:
            continue
        cols = []
        for n in models:
            z = predict_any(n, d)
            cols.append(pl.Series(f"z_{n}", z.astype(np.float32)))
            if calibrate:
                cols.append(pl.Series(f"pd_{n}", cals[n].predict(z).astype(np.float32)))
        parts.append(d.select(keep).with_columns(cols))
        del d
    return pl.concat(parts)


def predict_members(name: str, df: pl.DataFrame) -> np.ndarray:
    """Logits of every member of a variant (5 CatBoost / XGBoost / LightGBM folds or 5 MLP seeds),
    shape (members, rows): the single-model results behind the averaged prediction."""
    from catboost import CatBoostClassifier

    m, d = meta(name), MODELS_DIR / name
    if m["kind"] == "mlp":
        return predict_mlp(name, df, per_seed=True)[1]
    if m["kind"] == "catboost":
        enc = pickle.loads((d / "encoder.pkl").read_bytes())
        p = cb.pool(to_model_frame(df, m["features"], enc), cat_features=m["categorical"])
        out = []
        for k in range(cb.K_FOLDS):
            mk = CatBoostClassifier()
            mk.load_model(str(d / f"fold_{k}.cbm"))
            out.append(cb.logit(mk, p))
        return np.vstack(out)
    from xai_ews.models import gbdt
    enc = pickle.loads((d / "encoder.pkl").read_bytes())
    X = gbdt.as_frame(to_model_frame(df, m["features"], enc), m["categorical"], m["n_categories"])
    return np.vstack([gbdt.Model.load(m["kind"], d / f"fold_{k}.model", m["trees"]).logit(X) for k in range(cb.K_FOLDS)])
