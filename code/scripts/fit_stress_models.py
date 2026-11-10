"""Robustness models of the stress tests (test protocol, section 11), one CatBoost on CPU each:

    caffeinate -i uv run python scripts/fit_stress_models.py

vintage_<task>        vintage split: vintages 2014-2017, dates <= 2019-02, r = 0.2
unemp_<task>_<n>      full train r = 0.2 with the n selected features / n + 2 with two unemployment
                      features; trained the same way so the difference is the features
"""

import sys
import time

from xai_ews.config import feature_config
from xai_ews.models import registry
from xai_ews.models.data import load_train, load_vintage_train

SEL = feature_config()["selected"]
FEATS, CATS = SEL["features"], SEL["categorical"]
UNEMP = ["unemployment_rel_us", "unemployment_chg_12m_rel_us"]


def main() -> int:
    for task in ("y1",):
        study = f"catboost_{task}_main"
        jobs = [(f"vintage_{task}", lambda t=task: load_vintage_train(t, FEATS, CATS), FEATS),
                (f"unemp_{task}_{len(FEATS)}", lambda t=task: load_train(t, FEATS, CATS), FEATS),
                (f"unemp_{task}_{len(FEATS) + 2}", lambda t=task: load_train(t, FEATS + UNEMP, CATS), FEATS + UNEMP)]
        for name, load, feats in jobs:
            t0 = time.time()
            registry.fit_single_catboost(name, load, feats, CATS, task, study)
            print(time.strftime("%H:%M:%S"), name, f"{(time.time() - t0) / 60:.0f} min", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
