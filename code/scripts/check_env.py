"""Smoke-test the research environment: imports, versions, compute backend, a tiny fit per library.

    uv run python scripts/check_env.py
"""

import platform
import sys

import numpy as np


def main() -> int:
    import catboost
    import pandas
    import polars
    import pyarrow
    import scipy
    import shap
    import sklearn
    import statsmodels
    import torch

    print(f"python      {sys.version.split()[0]} ({platform.platform()})")
    for mod in (np, pandas, polars, pyarrow, scipy, sklearn, statsmodels, catboost, torch, shap):
        print(f"{mod.__name__:<11} {mod.__version__}")

    mps = torch.backends.mps.is_available()
    cuda = torch.cuda.is_available()
    print(f"torch device: {'cuda' if cuda else 'mps' if mps else 'cpu'}")

    rng = np.random.default_rng(0)
    X = rng.normal(size=(2000, 5))
    y = (X[:, 0] + 0.5 * X[:, 1] + rng.normal(size=2000) > 0).astype(int)

    booster = catboost.CatBoostClassifier(iterations=20, verbose=0).fit(X, y)
    sv = shap.TreeExplainer(booster).shap_values(X[:100])
    print(f"catboost+shap ok, shap shape {np.shape(sv)}")

    device = "mps" if mps else "cpu"
    model = torch.nn.Linear(5, 1).to(device)
    xt = torch.tensor(X, dtype=torch.float32, device=device)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        model(xt).squeeze(1), torch.tensor(y, dtype=torch.float32, device=device)
    )
    loss.backward()
    print(f"torch ok on {device}, loss {loss.item():.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
