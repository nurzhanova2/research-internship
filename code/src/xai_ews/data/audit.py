"""Descriptive statistics for the data audit: PSI, event counts, LGD segments."""

import itertools

import numpy as np
import polars as pl

from xai_ews.data.panel import Macro, month_index

EPS = 1e-4


def psi_numeric(expected: pl.Series, actual: pl.Series, bins: int = 10) -> float:
    """PSI with quantile bins fitted on `expected` (train); nulls form their own bin."""
    e = expected.cast(pl.Float64)
    a = actual.cast(pl.Float64)
    edges = np.unique(np.nanquantile(e.drop_nulls().drop_nans().to_numpy(),
                                     np.linspace(0, 1, bins + 1)[1:-1]))

    def shares(s: pl.Series) -> np.ndarray:
        v = s.to_numpy()
        isnull = np.isnan(v)
        idx = np.searchsorted(edges, v[~isnull], side="right")
        counts = np.bincount(idx, minlength=len(edges) + 1).astype(float)
        counts = np.append(counts, isnull.sum())
        return counts / max(len(v), 1)

    return psi_from_shares(shares(e.fill_null(np.nan)), shares(a.fill_null(np.nan)))


def psi_categorical(expected: pl.Series, actual: pl.Series, top: int = 30) -> float:
    """PSI over the `top` most frequent train categories plus 'other' and null."""
    cats = expected.drop_nulls().value_counts(sort=True).head(top)[expected.name].to_list()

    def shares(s: pl.Series) -> np.ndarray:
        mapped = s.cast(pl.String).replace_strict(
            {c: c for c in map(str, cats)}, default="__other__").fill_null("__null__")
        vc = dict(mapped.value_counts().iter_rows())
        keys = [*map(str, cats), "__other__", "__null__"]
        return np.array([vc.get(k, 0) for k in keys], dtype=float) / max(s.len(), 1)

    return psi_from_shares(shares(expected), shares(actual))


def psi_from_shares(e: np.ndarray, a: np.ndarray) -> float:
    """PSI = sum (a - e) ln(a / e) over buckets; shares clipped at EPS to avoid log(0)."""
    e = np.clip(e, EPS, None)
    a = np.clip(a, EPS, None)
    return float(np.sum((a - e) * np.log(a / e)))


def event_table(lf: pl.LazyFrame, by: list[str]) -> pl.DataFrame:
    """Observations, event rates and counts for Y1 / Y2 (per observation and per unique loan),
    prepayment and forbearance shares."""
    return lf.group_by(by).agg(
        pl.len().alias("n_obs"),
        pl.col("loan_identifier").n_unique().alias("n_loans"),
        pl.col("y1").sum().alias("y1_events"),
        pl.col("loan_identifier").filter(pl.col("y1") == 1).n_unique().alias("y1_event_loans"),
        pl.col("y1").mean().alias("y1_rate"),
        pl.col("pop_y2").sum().alias("n_obs_y2_pop"),
        pl.col("y2").sum().alias("y2_events"),
        pl.col("loan_identifier").filter(pl.col("y2") == 1).n_unique().alias("y2_event_loans"),
        pl.col("y2").mean().alias("y2_rate"),
        pl.col("y2_strict").sum().alias("y2_strict_events"),
        pl.col("loan_identifier").filter(pl.col("y2_strict") == 1).n_unique()
        .alias("y2_strict_event_loans"),
        pl.col("prepaid_in_window").mean().alias("prepaid_in_window_share"),
        pl.col("forbearance_now").mean().alias("forbearance_share"),
        (pl.col("n_assistance_12m") > 0).mean().alias("assistance_12m_share"),
        pl.col("upb_masked").mean().alias("upb_masked_share"),
        pl.col("upb").sum().alias("upb_total"),
    ).sort(by).collect()


LOSS_COSTS = ["foreclosure_costs", "property_preservation_and_repair_costs", "asset_recovery_costs",
              "miscellaneous_holding_expenses_and_credits", "associated_taxes_for_holding_property"]
# Recoveries other than mortgage insurance; MI (credit_enhancement_proceeds) is added only in the
# after-insurance sensitivity (LGD gross of MI).
LOSS_RECOVERIES = ["net_sales_proceeds", "repurchase_make_whole_proceeds", "other_foreclosure_proceeds"]
MI_PROCEEDS = "credit_enhancement_proceeds"


def _sum(cols: list[str]) -> pl.Expr:
    return pl.sum_horizontal([pl.col(c).fill_null(0) for c in cols])


def _hpi_at(df: pl.DataFrame, q_expr: pl.Expr, name: str, macro: Macro) -> pl.DataFrame:
    z = macro.hpi_zip3.select("zip3", (pl.col("year").cast(pl.Int32) * 4 + pl.col("quarter") - 1)
                              .alias("q"), "hpi")
    s = macro.hpi_state.select("state", (pl.col("year").cast(pl.Int32) * 4 + pl.col("quarter") - 1)
                               .alias("q"), "hpi")
    df = df.with_columns(q_expr.alias("_q"))
    return (df.join(z.rename({"zip3": "zip_code_short", "q": "_q", "hpi": "_hz"}),
                    on=["zip_code_short", "_q"], how="left")
              .join(s.rename({"state": "property_state", "q": "_q", "hpi": "_hs"}),
                    on=["property_state", "_q"], how="left")
              .with_columns(pl.coalesce("_hz", "_hs").alias(name)).drop("_hz", "_hs", "_q"))


def liquidations_with_segments(events: pl.DataFrame, macro: Macro, cfg: dict) -> pl.DataFrame:
    """One row per liquidated loan with realised LGD variants and LGD segments.

    EAD = UPB at removal (the balance stops amortising once the loan is delinquent). Costs C and
    recoveries R (sale, make-whole, other; MI only after insurance) are taken at the disposition
    date and discounted at the note rate over the months from the last paid instalment (default)
    to disposition: DF = (1 + rate/1200)^-months.
        lgd           gross of MI, discounted          (base)
        lgd_net_mi    after MI, discounted             (sensitivity)
        lgd_undisc    gross of MI, undiscounted        (sensitivity)
    Delinquent accrued interest is not published, so every variant understates economic loss.
    The rate is the note rate at origination: the termination record carries no current rate,
    so the rate of modified loans is approximate. Negative workout times (disposition before
    the last paid instalment, data errors) count as 0 months.

    Segment for estimation: MTM LTV at default = original LTV x UPB / original UPB x HPI at
    origination / HPI at the quarter of the last paid instalment (same release lag as the panel).
    """
    seg = cfg["lgd_segments"]
    lag = cfg["external"]["hpi_lag_months"]
    upb = pl.col("upb_at_the_time_of_removal")
    months = (month_index("disposition_date") - month_index("last_paid_installment_date"))
    rate = pl.coalesce("current_interest_rate", "original_interest_rate") / 1200.0
    df = events.with_columns(
        month_index("monthly_reporting_period").alias("m"),
        pl.col("net_sales_proceeds").is_not_null().alias("has_loss_data"),
        months.alias("workout_months_raw"),
        months.clip(0, None).alias("workout_months"),
        _sum(LOSS_COSTS).alias("costs"),
        _sum(LOSS_RECOVERIES).alias("recoveries"),
        pl.col(MI_PROCEEDS).fill_null(0).alias("mi_proceeds"),
    ).with_columns(
        (1.0 + rate).pow(-pl.col("workout_months").cast(pl.Float64)).alias("discount_factor"),
    )
    ok = pl.col("has_loss_data") & (upb > 0)
    net = pl.col("recoveries") - pl.col("costs")
    df = df.with_columns(
        pl.when(ok).then((upb - pl.col("discount_factor") * net) / upb).alias("lgd"),
        pl.when(ok).then((upb - pl.col("discount_factor") * (net + pl.col("mi_proceeds"))) / upb)
        .alias("lgd_net_mi"),
        pl.when(ok).then((upb - net) / upb).alias("lgd_undisc"),
    )
    df = _hpi_at(df, (month_index("last_paid_installment_date") - lag) // 3, "hpi_default", macro)
    df = _hpi_at(df, (pl.col("m") - lag) // 3, "hpi_liq", macro)
    df = _hpi_at(df, month_index("origination_date") // 3, "hpi_orig", macro)
    edges = seg["mtm_ltv_buckets"]
    labels = [f"<{edges[0]}"] + [f"{a}-{b}" for a, b in itertools.pairwise(edges)] + [f"{edges[-1]}+"]
    ltv0 = pl.col("original_loan_to_value_ratio_ltv") * upb / pl.col("original_upb") * pl.col("hpi_orig")
    df = df.with_columns(
        (ltv0 / pl.col("hpi_default")).alias("mtm_ltv_at_default"),
        (ltv0 / pl.col("hpi_liq")).alias("mtm_ltv_at_liquidation"),
        pl.col("property_state").is_in(seg["judicial_states"]).alias("judicial"),
    )
    return df.with_columns(
        pl.col("mtm_ltv_at_default").cut(edges, labels=labels, left_closed=True)
        .cast(pl.String).fill_null("unknown").alias("mtm_ltv_bucket"),
    )


def truncated_histories(perf: pl.LazyFrame, data_end: int) -> pl.DataFrame:
    """Per loan: does the history stop before the data end without a zero-balance code?
    With delinquency profile to compare truncated loans with the rest."""
    dpd = pl.col("current_loan_delinquency_status")
    return (
        perf.with_columns(month_index("monthly_reporting_period").alias("m"))
        .group_by("loan_identifier")
        .agg(
            pl.col("m").max().alias("last_m"),
            pl.col("zero_balance_code").drop_nulls().last().alias("exit_code"),
            dpd.drop_nulls().last().alias("last_dpd"),
            dpd.max().alias("max_dpd"),
            pl.len().alias("months"),
        )
        .with_columns(
            (pl.col("exit_code").is_null() & (pl.col("last_m") < data_end)).alias("truncated"))
        .collect()
    )


def truncation_summary(loans: pl.DataFrame) -> pl.DataFrame:
    return loans.group_by("truncated").agg(
        pl.len().alias("loans"),
        pl.col("months").mean().alias("mean_months_observed"),
        (pl.col("last_dpd") >= 1).mean().alias("share_30p_at_last_record"),
        (pl.col("max_dpd") >= 1).mean().alias("share_ever_30p"),
        (pl.col("max_dpd") >= 2).mean().alias("share_ever_60p"),
        (pl.col("max_dpd") >= 3).mean().alias("share_ever_90p"),
    ).with_columns((pl.col("loans") / pl.col("loans").sum()).alias("share_of_loans")).sort("truncated")
