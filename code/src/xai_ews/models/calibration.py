"""Calibration of PD on validation half B.

Training uses weights omega x 1/r, so the raw model PD is not a calibrated probability of the
unweighted population. Platt scaling fits PD = sigmoid(a + b z) on the raw logit z by an
unweighted logistic regression on half B (main: linear in logits, keeps the additivity of SHAP
contributions). Isotonic regression (a monotone step function of z) is the sensitivity variant.
"""

from dataclasses import dataclass

import numpy as np
import statsmodels.api as sm
from sklearn.isotonic import IsotonicRegression

from xai_ews.evaluation.metrics import sigmoid


@dataclass
class Platt:
    a: float = 0.0
    b: float = 1.0

    def fit(self, z, y) -> "Platt":
        res = sm.GLM(np.asarray(y, np.float64), sm.add_constant(np.asarray(z, np.float64)),
                     family=sm.families.Binomial()).fit()
        self.a, self.b = map(float, res.params)
        return self

    def logit(self, z) -> np.ndarray:
        return self.a + self.b * np.asarray(z, np.float64)

    def predict(self, z) -> np.ndarray:
        return sigmoid(self.logit(z))


class Isotonic:
    def __init__(self):
        self.iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6)

    def fit(self, z, y) -> "Isotonic":
        self.iso.fit(np.asarray(z, np.float64), np.asarray(y, np.float64))
        return self

    def predict(self, z) -> np.ndarray:
        return self.iso.predict(np.asarray(z, np.float64))
