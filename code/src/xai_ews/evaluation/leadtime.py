"""Lead time of the alert (methodology 7; hypotheses H1, H3).

For a loan with an event in month E (first 90+ DPD or default exit):

    alert date   the first quarterly snapshot t in [E - 12, E - 1] at which the loan belongs to the
                 Y1 population and its calibrated PD >= the alert threshold
    lead time    E - t (months); a loan never alerted in that window is "missed"

Snapshots are scored at every quarterly date of the 12 months before the event, including dates
outside the evaluation split (embargo gaps): only features at t are used, never labels, so the
event's full 12-month history is visible. Population membership at t (pop_y1: current at t) is
required, so the alert must come while the loan still looks performing.

Summary: median lead time over alerted loans (primary, H3), share of event loans alerted, share
alerted >= 6 months ahead, and the median with missed loans counted as 0.
"""

import numpy as np
import pandas as pd
import polars as pl

from xai_ews.evaluation.el import alert_mask


def event_months(df: pl.DataFrame, label: str, month_col: str) -> pl.DataFrame:
    """loan_identifier -> E = m + offset of the first event, from the rows with label 1.

    `month_col` is the panel's offset column (y1_month: months from t to the event)."""
    return (df.filter(pl.col(label) == 1)
            .group_by("loan_identifier")
            .agg((pl.col("m") + pl.col(month_col).cast(pl.Int32)).min().alias("E")))


def alert_flags(scored: pl.DataFrame, threshold: float | None = None) -> pl.Series:
    """Alert flag per row: calibrated `pd` >= threshold."""
    return pl.Series("alert", alert_mask(scored["pd"].to_numpy(), threshold))


def lead_times(events: pl.DataFrame, scored: pl.DataFrame, threshold: float | None = None) -> pl.DataFrame:
    """Per event loan: E, alert month (null if missed), lead time in months.

    `scored`: loan_identifier, m, pd (calibrated) for every loan of the Y1 population at every
    snapshot date that can fall into [E - 12, E - 1] of some event."""
    al = scored.with_columns(alert_flags(scored, threshold)).filter(pl.col("alert"))
    j = events.join(al.select("loan_identifier", "m"), on="loan_identifier", how="left")
    j = j.filter(pl.col("m").is_between(pl.col("E") - 12, pl.col("E") - 1))
    first = j.group_by("loan_identifier").agg(pl.col("m").min().alias("m_alert"))
    return (events.join(first, on="loan_identifier", how="left")
            .with_columns((pl.col("E") - pl.col("m_alert")).alias("lead")))


def summarise(lt: pl.DataFrame) -> dict:
    lead = lt["lead"].to_numpy().astype(float)
    hit = ~np.isnan(lead)
    return {"event_loans": len(lead), "alerted": int(hit.sum()), "alert_rate": float(hit.mean()),
            "median_lead": float(np.median(lead[hit])) if hit.any() else np.nan,
            "mean_lead": float(np.mean(lead[hit])) if hit.any() else np.nan,
            "share_ge6": float(np.mean(lead[hit] >= 6)) if hit.any() else np.nan,
            "median_lead_missed0": float(np.median(np.where(hit, lead, 0.0)))}


def bootstrap_median(lt: pl.DataFrame, b: int = 1000, seed: int = 0) -> pd.Series:
    """Median lead time over alerted loans and its 95 % CI; the resampling unit is the event
    loan (one lead time per loan, so the cluster bootstrap is a plain bootstrap over loans)."""
    lead = lt["lead"].drop_nulls().to_numpy().astype(float)
    rng = np.random.default_rng(seed)
    draws = np.array([np.median(lead[rng.integers(0, len(lead), len(lead))]) for _ in range(b)])
    return pd.Series({"median_lead": float(np.median(lead)), "lo": float(np.percentile(draws, 2.5)),
                      "hi": float(np.percentile(draws, 97.5)), "alerted": len(lead)})


def paired_bootstrap(lt_a: pl.DataFrame, lt_b: pl.DataFrame, b: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Two models on the same event loans: Δ (a − b) of the alert rate and of the median lead time
    over alerted loans, with 95 % CIs from a bootstrap over event loans (descriptive)."""
    j = lt_a.select("loan_identifier", pl.col("lead").alias("a")).join(
        lt_b.select("loan_identifier", pl.col("lead").alias("b")), on="loan_identifier")
    a, c = j["a"].to_numpy().astype(float), j["b"].to_numpy().astype(float)

    def stats_(ia):
        x, y = a[ia], c[ia]
        return (np.mean(~np.isnan(x)) - np.mean(~np.isnan(y)),
                np.nanmedian(x) - np.nanmedian(y))
    rng = np.random.default_rng(seed)
    point = stats_(np.arange(len(a)))
    draws = np.array([stats_(rng.integers(0, len(a), len(a))) for _ in range(b)])
    return pd.DataFrame({"delta": point, "lo": np.percentile(draws, 2.5, axis=0),
                         "hi": np.percentile(draws, 97.5, axis=0)},
                        index=["alert_rate", "median_lead"])
