"""How a loan's risk is shown to people.

The calibrated PD stays the number for expected loss and provisions. Because Y1 is rare (mean PD
about 0.2 %), the PD of even a risky loan is a small number, so three readable scales are shown
next to it; all are monotone in PD, so ranking metrics and the alert list do not change:

    relative risk   PD / mean PD of the portfolio on the same date ("7 times the average")
    percentile      share of loans of the same date with a lower PD (0 … 1)
    risk class      1 … 10 on fixed PD edges; the alert threshold PD >= 1 % is the lower edge of class 7
    score points    the scorecard convention: higher = safer, 20 points halve the odds of default
"""

import numpy as np
import pandas as pd

# Lower PD edges of classes 2 … 10 (class 1 is below 0.05 %); fixed before looking at test.
GRADE_EDGES = (0.0005, 0.001, 0.002, 0.0035, 0.005, 0.01, 0.02, 0.035, 0.05)
ALERT_GRADE = 7


def relative_risk(pd_: np.ndarray, groups: np.ndarray | None = None) -> np.ndarray:
    """PD divided by the mean PD of its group (the portfolio on the same date); one group if None."""
    s = pd.Series(np.asarray(pd_, float))
    mean = s.groupby(np.asarray(groups)).transform("mean") if groups is not None else s.mean()
    return (s / mean).to_numpy()


def percentile(pd_: np.ndarray, groups: np.ndarray | None = None) -> np.ndarray:
    """Share of loans of the same group with a strictly lower PD, in [0, 1)."""
    s = pd.Series(np.asarray(pd_, float))
    g = s.groupby(np.asarray(groups)) if groups is not None else s.groupby(np.zeros(len(s)))
    below = g.rank(method="min") - 1
    return (below / g.transform("size")).to_numpy()


def grade(pd_: np.ndarray) -> np.ndarray:
    """Risk class 1 … 10 on GRADE_EDGES (a PD equal to an edge falls into the higher class)."""
    return np.searchsorted(GRADE_EDGES, np.asarray(pd_, float), side="right") + 1


def grade_labels() -> list[str]:
    """PD range of every class, e.g. '1 % – 2 %'."""
    fmt = lambda x: f"{100 * x:g} %"
    edges = [0.0, *GRADE_EDGES]
    return [f"< {fmt(edges[1])}"] + [f"{fmt(a)} – {fmt(b)}" for a, b in zip(edges[1:-1], edges[2:])] + [f"≥ {fmt(edges[-1])}"]


def score_points(pd_: np.ndarray, base: float = 600.0, base_odds: float = 50.0, pdo: float = 20.0) -> np.ndarray:
    """Points = offset − factor · logit(PD), factor = pdo / ln 2; equals `base` at good:bad odds
    `base_odds` (same convention as the scorecard's points table)."""
    p = np.clip(np.asarray(pd_, float), 1e-9, 1 - 1e-9)
    factor = pdo / np.log(2)
    offset = base - factor * np.log(base_odds)
    return offset - factor * np.log(p / (1 - p))


def present(pd_: np.ndarray, groups: np.ndarray | None = None) -> pd.DataFrame:
    """All scales side by side for a vector of calibrated PDs."""
    return pd.DataFrame({"pd": np.asarray(pd_, float), "relative_risk": relative_risk(pd_, groups),
                         "percentile": percentile(pd_, groups), "grade": grade(pd_), "points": score_points(pd_)})
