"""Final stage-3 models (test protocol, section 3 and 10), trained after tuning:

    caffeinate -i uv run python scripts/fit_models.py            # every variant, in order
    caffeinate -i uv run python scripts/fit_models.py m1_y1      # one variant

Each variant waits until its Optuna study has N_TRIALS complete trials, then trains and saves
5 GroupKFold CatBoost models (or 5 MLP seeds) into data/processed/models/<name>/ via
xai_ews.models.registry. Finished variants are skipped, so the script can be restarted.
"""

import sys
import time

import optuna

from xai_ews.config import feature_config
from xai_ews.models import registry

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from tune import N_TRIALS, feature_set

MAIN, CATS = feature_config()["selected"]["features"], feature_config()["selected"]["categorical"]
H3S, H3S_C = feature_set("h3_static")
H3F, H3F_C = feature_set("h3_full")

# name: (kind, study, kwargs)
VARIANTS = {
    "m2_y1": ("mlp", "mlp_y1_main", {"task": "y1", "features": MAIN, "categorical": CATS}),
    "m1_y1": ("catboost", "catboost_y1_main", {"task": "y1", "features": MAIN, "categorical": CATS}),
    "m1_y1_noomega": ("catboost", "catboost_y1_main", {"task": "y1", "features": MAIN, "categorical": CATS, "omega": None}),
    "h3_static_y1": ("catboost", "catboost_y1_h3_static", {"task": "y1", "features": H3S, "categorical": H3S_C}),
    "h3_full_y1": ("catboost", "catboost_y1_h3_full", {"task": "y1", "features": H3F, "categorical": H3F_C}),
    # sensitivity checks: hyperparameters of the main model, no new tuning (protocol 10)
    "m1_y1_linear": ("catboost", "catboost_y1_main", {"task": "y1", "features": MAIN, "categorical": CATS, "omega": "linear"}),
    "m1_y1_r01": ("catboost", "catboost_y1_main", {"task": "y1", "features": MAIN, "categorical": CATS, "r": 0.1}),
    "m1_y1_noprepaid": ("catboost", "catboost_y1_main", {"task": "y1", "features": MAIN, "categorical": CATS, "drop_prepaid": True}),
    # other gradient-boosting learners (XGBoost was as good as CatBoost in stage 2), own tuning
    "xgb_y1": ("xgboost", "xgboost_y1_main", {"task": "y1", "features": MAIN, "categorical": CATS}),
    "lgb_y1": ("lightgbm", "lightgbm_y1_main", {"task": "y1", "features": MAIN, "categorical": CATS}),
}


def complete(study_name: str) -> bool:
    try:
        s = registry.study(study_name)
    except KeyError:
        return False
    return sum(t.state == optuna.trial.TrialState.COMPLETE for t in s.trials) >= N_TRIALS


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def main(names: list[str]) -> int:
    todo = [n for n in (names or VARIANTS) if not registry.exists(n)]
    while todo:
        ready = [n for n in todo if complete(VARIANTS[n][1])]
        if not ready:
            time.sleep(300)
            continue
        name = ready[0]
        kind, study_name, kw = VARIANTS[name]
        log(f"fit {name} ({kind}, study {study_name})")
        t0 = time.time()
        if kind == "catboost":
            registry.fit_catboost_variant(name, study_name=study_name, log=log, **kw)
        elif kind in ("xgboost", "lightgbm"):
            registry.fit_gbdt_variant(name, kind, study_name=study_name, log=log, **kw)
        else:
            registry.fit_mlp_variant(name, study_name=study_name, log=log, **kw)
        log(f"done {name} in {(time.time() - t0) / 60:.0f} min")
        todo.remove(name)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
