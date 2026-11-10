"""Expected loss and the alert list (methodology 2.3, 7; hypothesis H4).

    LGD      segment mean (reports/data_audit/lgd_variants.csv): MTM LTV bucket x
             judicial state; for live loans the bucket is taken from the CURRENT MTM LTV at t.
    EL       EL_i(t) = PD_i(t) x LGD_i(t) x EAD_i(t), EAD = UPB at t.
    alert    at every observation date the top q % of the portfolio by the signal (EL or PD) is
             on the monitoring list (q = 5 %, 10 %).
    captured EL share  share of the realised expected loss caught by the list:
             sum over the list of y x LGD x EAD / sum over all rows of y x LGD x EAD, summed over
             dates. y = observed event, so a model scores only by putting large losses that do
             happen on the list (a share of predicted EL would reward any inflated PD).
"""

import numpy as np
import pandas as pd
import polars as pl

from xai_ews.paths import DATA_AUDIT_DIR

LGD_FILE = DATA_AUDIT_DIR / "lgd_variants.csv"


def lgd_table(variant: str = "base") -> pd.DataFrame:
    t = pd.read_csv(LGD_FILE)
    return t[t["variant"] == variant].reset_index(drop=True)


def ltv_bucket_expr(edges: list[float], col: str = "mtm_ltv") -> pl.Expr:
    import itertools
    labels = [f"<{edges[0]}"] + [f"{a}-{b}" for a, b in itertools.pairwise(edges)] + [f"{edges[-1]}+"]
    return pl.col(col).cut(edges, labels=labels, left_closed=True).cast(pl.String)


def assign_lgd(df: pl.DataFrame, cfg: dict, variant: str = "base") -> pl.Series:
    """LGD per row from current MTM LTV and property_state; missing LTV -> the judicial group's
    portfolio mean; constant variants -> one value."""
    t = lgd_table(variant)
    if len(t) == 1:
        return pl.Series("lgd", np.full(df.height, t["lgd_mean"].iloc[0], np.float32))
    seg = cfg["lgd_segments"]
    rows = []
    for r in t.itertuples():
        for b in r.mtm_ltv_buckets.split(","):
            rows.append({"judicial": bool(r.judicial), "bucket": b, "lgd": float(r.lgd_mean)})
    m = pl.DataFrame(rows)
    fallback = (t.assign(w=t["lgd_mean"] * t["liquidations"]).groupby("judicial")[["w", "liquidations"]].sum())
    fb = {bool(k): v.w / v.liquidations for k, v in fallback.iterrows()}
    key = df.select(pl.col("property_state").is_in(seg["judicial_states"]).alias("judicial"),
                    ltv_bucket_expr(seg["mtm_ltv_buckets"]).alias("bucket"))
    out = key.join(m, on=["judicial", "bucket"], how="left", maintain_order="left")
    return out.select(pl.when(pl.col("lgd").is_null())
                      .then(pl.col("judicial").replace_strict(fb, return_dtype=pl.Float64))
                      .otherwise(pl.col("lgd")).cast(pl.Float32).alias("lgd")).to_series()


def alert_threshold() -> float:
    """Calibrated-PD threshold of the early-warning list (configs/data.yaml alert.pd_threshold)."""
    from xai_ews.config import data_config

    return float(data_config()["alert"]["pd_threshold"])


def alert_mask(pd_, threshold: float | None = None) -> np.ndarray:
    """A row is on the alert list if its calibrated PD >= threshold."""
    return np.asarray(pd_, np.float64) >= (alert_threshold() if threshold is None else threshold)


def captured_el_share(y, alert, lgd, ead, w=None) -> float:
    """Share of realised EL (y x LGD x EAD) on the alert list; `w` are bootstrap multiplicities
    (rows of a drawn loan count w times)."""
    loss = np.asarray(y, np.float64) * np.asarray(lgd, np.float64) * np.asarray(ead, np.float64)
    w = np.ones(len(loss)) if w is None else np.asarray(w, np.float64)
    alert = np.asarray(alert, bool)
    return float(np.sum((w * loss)[alert]) / np.sum(w * loss))


def captured_event_share(y, alert) -> float:
    """Recall of the list: share of event rows on the alert list."""
    y = np.asarray(y)
    return float(y[np.asarray(alert, bool)].sum() / y.sum())
