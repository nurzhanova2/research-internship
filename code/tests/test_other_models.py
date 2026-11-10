"""Steps 8-9 with other learners (xai_ews.features.other_models).

XGBoost / LightGBM and PyTorch / CatBoost deadlock or crash in one process on macOS (OpenMP), and
pytest imports all test modules into one process, so each check runs in a fresh interpreter.
"""

import subprocess
import sys

import pytest

CHECK = """
import numpy as np, pandas as pd
from xai_ews.features.other_models import boruta, fit_model, permutation_importance
kind = {kind!r}
params = {{"logistic_woe": {{"C": 1.0, "max_iter": 200}},
          "xgboost": {{"iterations": 50, "depth": 3, "learning_rate": 0.1}},
          "lightgbm": {{"iterations": 50, "num_leaves": 7, "learning_rate": 0.1}},
          "random_forest": {{"trees": 50, "min_samples_leaf": 20, "max_features": "sqrt"}}}}
rng = np.random.default_rng(0)
n = 4000
X = pd.DataFrame({{"signal": rng.normal(size=n), "noise": rng.normal(size=n)}})
y = (rng.random(n) < 1 / (1 + np.exp(-(-2 + 2 * X["signal"])))).astype(int)
w = np.ones(n)
model = fit_model(kind, X, y, w, params)
perm = permutation_importance(model, X.copy(), y, w, repeats=2).set_index("feature")
assert perm.loc["signal", "perm_auc_drop"] > 0.1 > abs(perm.loc["noise", "perm_auc_drop"])
bor = boruta(kind, X, y, w, params, n_iter=8).set_index("feature")["boruta_decision"]
assert bor["signal"] == "confirmed" and bor["noise"] != "confirmed"
"""


@pytest.mark.parametrize("kind", ["logistic_woe", "xgboost", "lightgbm", "random_forest"])
def test_other_models_keep_signal_and_reject_noise(kind):
    r = subprocess.run([sys.executable, "-c", CHECK.format(kind=kind)], capture_output=True, text=True,
                       timeout=300, check=False)
    assert r.returncode == 0, r.stderr[-2000:]
