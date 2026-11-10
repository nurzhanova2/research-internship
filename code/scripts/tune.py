"""Hyperparameter tuning of stage 3, run in the background:

    caffeinate -i uv run python scripts/tune.py catboost y1 main
    caffeinate -i uv run python scripts/tune.py mlp y1 main

model    catboost | mlp | xgboost | lightgbm (the last two: post-hoc baselines)
task     y1
set      main (selected features) | h3_static | h3_full (H3)

Optuna TPE (seed fixed), N_TRIALS trials in total per (model, task, set); the study lives in
data/processed/models/optuna.db and is resumed if the run stops. Metric = omega-weighted log-loss
on validation half A (rows kept by the disaster rule); AUC on half A is stored for control.
Train = full train, r = 0.2, row weight omega x 1/r.
"""

import gc
import os
import shutil
import sys
import time

import numpy as np
import optuna

from xai_ews.config import feature_config
from xai_ews.models.data import arrays, fit_encoder, load_eval, load_train, to_model_frame
from xai_ews.paths import MODELS_DIR, OPTUNA_DB

N_TRIALS = 40
SEED = 20261025
STORAGE = f"sqlite:///{OPTUNA_DB}"


def feature_set(name: str) -> tuple[list[str], list[str]]:
    s = feature_config()["selected"]
    if name == "main":
        return s["features"], s["categorical"]
    feats = s["static_h3" if name == "h3_static" else "h3_full"]
    return feats, [c for c in s["categorical_h3"] if c in feats]


def study_name(model: str, task: str, fset: str) -> str:
    return f"{model}_{task}_{fset}"


def main(model: str, task: str, fset: str) -> int:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    feats, cats = feature_set(fset)
    tr = load_train(task, feats, cats)
    va = load_eval("validation", task, feats, cats)
    va = va.filter(va["half"] == "A")
    enc = fit_encoder(tr, cats)
    y, w = tr[task].to_numpy(), tr["weight"].to_numpy()
    ya, wa = va[task].to_numpy(), va["omega"].to_numpy()
    print(f"{model} {task} {fset}: train {tr.height:,} rows, half A {va.height:,} rows, "
          f"{int(ya.sum()):,} events", flush=True)
    study = optuna.create_study(study_name=study_name(model, task, fset), storage=STORAGE,
                                direction="minimize", load_if_exists=True,
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.set_user_attr("features", feats)
    if model == "catboost":
        from xai_ews.models import catboost_model as cb
        study.set_user_attr("device", cb.DEVICE)
        train_pool = cb.pool(to_model_frame(tr, feats, enc), y, w, cats)
        eval_pool = cb.pool(to_model_frame(va, feats, enc), ya, wa, cats)
        del tr, va
        gc.collect()
        objective = cb.objective(train_pool, eval_pool, ya, wa, seed=SEED)
    elif model in ("xgboost", "lightgbm"):          # post-hoc baselines
        from xai_ews.models import gbdt
        nc = {c: len(enc.maps[c]) for c in cats}
        X = gbdt.as_frame(to_model_frame(tr, feats, enc), cats, nc)
        Xa = gbdt.as_frame(to_model_frame(va, feats, enc), cats, nc)
        del tr, va
        gc.collect()
        objective = gbdt.objective(model, X, y, w, Xa, ya, wa, seed=SEED)
    else:
        from xai_ews.models import mlp
        num = [f for f in feats if f not in cats]
        xn, xc = arrays(tr, feats, cats, enc)
        pre = mlp.Preprocessor(seed=SEED).fit(xn)
        xn = pre.transform(xn)
        an, ac = arrays(va, feats, cats, enc)
        an = pre.transform(an)
        card = [len(enc.maps[c]) for c in cats]
        del tr, va
        gc.collect()
        study.set_user_attr("numeric", num)

        def objective(trial):
            from xai_ews.evaluation.metrics import auc
            cfg = mlp.suggest(trial)
            m, hist = mlp.train(xn, xc, y, w, card, cfg, eval_data=(an, ac, ya, wa), seed=SEED)
            trial.set_user_attr("epochs", hist["best_epoch"] + 1)
            trial.set_user_attr("auc_A", auc(ya, mlp.predict_logit(m, an, ac)))
            return hist["best_loss"]

    done = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    for _ in range(N_TRIALS - done):
        t0 = time.time()
        study.optimize(objective, n_trials=1, gc_after_trial=True)
        t = study.trials[-1]
        if os.environ.get("XAI_BACKUP_DIR"):        # Colab: copy the journal to Google Drive
            shutil.copy(OPTUNA_DB, os.environ["XAI_BACKUP_DIR"])
        print(f"trial {t.number} {t.state.name}: loss {t.value or np.nan:.6f} auc_A {t.user_attrs.get('auc_A', np.nan):.4f} "
              f"{t.params} {time.time() - t0:.0f}s", flush=True)
    print("best", study.best_value, study.best_params, study.best_trial.user_attrs, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:4]))
