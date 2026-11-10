"""Loan x observation-date panel: behavioural features, macro features, Y1/Y2 labels.

Pipeline per vintage (all functions are pure and work on polars frames, so the label logic can
be tested on synthetic loan histories):

    monthly   = prepare_monthly(performance, loans)    dense months per loan, UPB reconstruction
    monthly   = add_history(monthly, cfg)              trailing 6/12-month behaviour, month t incl.
    monthly   = add_labels(monthly, cfg)               forward windows (t, t+12]
    panel     = select_snapshots(monthly, cfg)         quarter-end months, loan_age >= 6, active
    panel     = add_macro(panel, macro, cfg)           HPI / unemployment / PMMS with release lags

Windows use row shifts within a loan. prepare_monthly makes every loan's months contiguous
(missing months are inserted as empty rows), so shift k is exactly k months. Raw data have no
gaps so far (ingest_stats.json: gap_rows_month_skipped), the densifying is a safeguard.

Month index m = 12 * year + month - 1 throughout.
"""

from dataclasses import dataclass

import polars as pl

from xai_ews.paths import PANEL_DIR, TRAIN_FULL_DIR

ID = "loan_identifier"
M = "m"
DPD = "dpd"
ZBC = "zero_balance_code"

PREPAY_CODE = "01"


def _y1_population(lf: pl.LazyFrame) -> pl.LazyFrame:
    """pop_y1 = DPD at t <= labels.early_default.population_max_dpd and, with exclude_after_default,
    no 90+ DPD before t (scripts/build_default_history.py). The panel files were built with DPD 0-2;
    the population is narrowed here without a rebuild."""
    from xai_ews.config import data_config
    from xai_ews.paths import DEFAULT_HISTORY

    y1 = data_config()["labels"]["early_default"]
    pop = pl.col("pop_y1") & (pl.col(DPD) <= y1["population_max_dpd"])
    if y1["exclude_after_default"]:
        lf = lf.join(pl.scan_parquet(DEFAULT_HISTORY), on=ID, how="left")
        pop = pop & (pl.col("first_default_m").is_null() | (pl.col(M) < pl.col("first_default_m")))
    return lf.with_columns(pop.alias("pop_y1")).drop("first_default_m", strict=False)


def scan_panel() -> pl.LazyFrame:
    """All built vintages; `vintage` comes from the hive partition key."""
    return _y1_population(pl.scan_parquet(PANEL_DIR / "vintage=*" / "*.parquet", hive_partitioning=True))


def scan_train_full() -> pl.LazyFrame:
    """Train observations from the train vintages at 100 %, with `u_neg`."""
    return _y1_population(pl.scan_parquet(TRAIN_FULL_DIR / "vintage=*" / "*.parquet", hive_partitioning=True))


def month_index(col: str | pl.Expr) -> pl.Expr:
    e = pl.col(col) if isinstance(col, str) else col
    return e.dt.year().cast(pl.Int32) * 12 + e.dt.month().cast(pl.Int32) - 1


def month_from_index(col: str) -> pl.Expr:
    return pl.date(pl.col(col) // 12, pl.col(col) % 12 + 1, 1)


def ym_to_index(ym: str) -> int:
    """'2026-03' -> month index."""
    year, month = ym.split("-")
    return int(year) * 12 + int(month) - 1


def lag(col: str | pl.Expr, k: int) -> pl.Expr:
    """Value k months earlier (k > 0) or |k| months later (k < 0) for the same loan, else null."""
    e = pl.col(col) if isinstance(col, str) else col
    if k == 0:
        return e
    return pl.when(pl.col(ID).shift(k) == pl.col(ID)).then(e.shift(k))


def _annuity_balance(principal: pl.Expr, annual_rate: pl.Expr, term: pl.Expr, paid: pl.Expr) -> pl.Expr:
    """Scheduled balance after `paid` level payments of a fully amortising loan."""
    r = annual_rate / 1200.0
    growth_n = (1.0 + r).pow(term)
    growth_k = (1.0 + r).pow(paid.clip(0, term))
    amortising = principal * (growth_n - growth_k) / (growth_n - 1.0)
    linear = principal * (1.0 - paid.clip(0, term) / term)  # zero-rate edge case
    return pl.when(r > 0).then(amortising).otherwise(linear)


# --------------------------------------------------------------------------------------------
# 1. Monthly series


LOAN_COLUMNS = [ID, "original_upb", "original_interest_rate", "original_loan_term",
                "first_payment_date", "origination_date"]


def prepare_monthly(perf: pl.DataFrame, loans: pl.DataFrame) -> pl.DataFrame:
    """Dense monthly rows per loan with reconstructed UPB and per-month payment quantities.

    upb_masked: current_actual_upb reported as 0 (or missing) on a live month — Fannie Mae masks
    the balance of young loans. Those months take the contractual schedule balance instead.
    """
    df = perf.with_columns(month_index("monthly_reporting_period").alias(M)).sort(ID, M)

    # Insert missing months so that row shifts equal month offsets.
    span = df.group_by(ID).agg(pl.col(M).min().alias("m0"), pl.col(M).max().alias("m1"))
    grid = span.select(ID, pl.int_ranges("m0", pl.col("m1") + 1).alias(M)).explode(M)
    if grid.height != df.height:
        cols = df.columns
        df = grid.with_columns(pl.col(M).cast(pl.Int32)).join(df, on=[ID, M], how="left")
        df = df.select(cols).with_columns(
            pl.col("monthly_reporting_period").is_null().alias("inserted_month"))
    else:
        df = df.with_columns(pl.lit(False).alias("inserted_month"))

    df = df.join(loans.select(LOAN_COLUMNS), on=ID, how="left").sort(ID, M)

    # Fannie Mae: loan age counts months from (first payment date - 1 month).
    age_from_dates = pl.col(M) - (month_index("first_payment_date") - 1)
    df = df.with_columns(
        pl.coalesce(pl.col("loan_age").cast(pl.Int32), age_from_dates).alias("age"),
        pl.col("current_loan_delinquency_status").alias(DPD),
    )
    sched = _annuity_balance(pl.col("original_upb"), pl.col("original_interest_rate"),
                             pl.col("original_loan_term"), pl.col("age").cast(pl.Float64))
    live = pl.col(ZBC).is_null() & ~pl.col("inserted_month")
    masked = live & (pl.col("current_actual_upb").fill_null(0) <= 0)
    df = df.with_columns(sched.alias("sched_upb"), masked.alias("upb_masked"))
    df = df.with_columns(
        pl.when(pl.col("upb_masked")).then(pl.col("sched_upb"))
        .when(live).then(pl.col("current_actual_upb"))
        .alias("upb")
    )

    # Principal paid this month and the scheduled principal on the previous balance.
    prev_upb = lag("upb", 1)
    rate = pl.col("current_interest_rate").fill_null(pl.col("original_interest_rate")) / 1200.0
    n_left = (pl.col("remaining_months_to_maturity").cast(pl.Float64) + 1.0).clip(1.0, None)
    sched_prin = pl.when(rate > 0).then(prev_upb * rate / ((1.0 + rate).pow(n_left) - 1.0)) \
        .otherwise(prev_upb / n_left)
    both_reported = ~pl.col("upb_masked") & ~lag("upb_masked", 1).fill_null(True) & live
    df = df.with_columns(
        pl.when(both_reported).then(prev_upb - pl.col("upb")).alias("principal_paid"),
        pl.when(both_reported).then(sched_prin).alias("sched_principal"),
    )
    return df


# --------------------------------------------------------------------------------------------
# 2. Trailing history (month t included)


def _window(col: str | pl.Expr, w: int) -> list[pl.Expr]:
    return [lag(col, k) for k in range(w)]


def add_history(df: pl.DataFrame, cfg: dict) -> pl.DataFrame:
    windows = cfg["features"]["windows_months"]
    min_usd = cfg["features"]["curtailment_min_usd"]
    min_share = cfg["features"]["curtailment_min_share"]

    excess = pl.col("principal_paid") - pl.col("sched_principal")
    df = df.with_columns(
        (pl.col(DPD) >= 1).cast(pl.Int8).alias("_d30"),
        (pl.col(DPD) >= 2).cast(pl.Int8).alias("_d60"),
        ((pl.col(DPD) == 0) & (lag(DPD, 1) >= 1)).cast(pl.Int8).alias("_cure"),
        pl.col(DPD).is_null().cast(pl.Int8).alias("_unknown"),
        (excess > pl.max_horizontal(pl.lit(min_usd), min_share * pl.col("sched_principal")))
        .cast(pl.Int8).alias("_curtail"),
        pl.when(excess > 0).then(excess).otherwise(0.0).alias("_excess"),
        (pl.col("principal_paid") < -1.0).cast(pl.Int8).alias("_upb_up"),
        (pl.col("modification_flag") == "Y").cast(pl.Int8).alias("_mod"),
        (pl.col("borrower_assistance_plan").is_in(["F", "R", "T"])).cast(pl.Int8).alias("_assist"),
        pl.when(pl.col(DPD) >= 1).then(pl.col(M)).alias("_last30_m"),
    )

    exprs = []
    for w in windows:
        exprs += [
            pl.max_horizontal(_window(DPD, w)).alias(f"dpd_max_{w}m"),
            pl.sum_horizontal(_window("_d30", w)).alias(f"n_30p_{w}m"),
            pl.sum_horizontal(_window("_curtail", w)).alias(f"n_curtail_{w}m"),
            (pl.col("upb") / lag("upb", w) - 1.0).alias(f"upb_chg_{w}m"),
        ]
    w = max(windows)
    exprs += [
        pl.mean_horizontal(_window(DPD, w)).alias(f"dpd_mean_{w}m"),
        pl.sum_horizontal(_window("_d60", w)).alias(f"n_60p_{w}m"),
        pl.sum_horizontal(_window("_cure", w)).alias(f"n_cure_{w}m"),
        pl.sum_horizontal(_window("_unknown", w)).alias(f"n_dpd_unknown_{w}m"),
        pl.sum_horizontal(_window("_upb_up", w)).alias(f"n_upb_increase_{w}m"),
        (pl.sum_horizontal(_window("_excess", w)) / pl.col("original_upb"))
        .alias(f"excess_principal_{w}m"),
        pl.sum_horizontal(_window("_assist", w)).alias(f"n_assistance_{w}m"),
        # Population filter for Y2: no 30+ in months t-12..t-1 (unknown months count as clean).
        pl.max_horizontal([lag(DPD, k) for k in range(1, w + 1)]).alias("dpd_max_prev12"),
        pl.sum_horizontal([lag((~pl.col("inserted_month")).cast(pl.Int8), k) for k in range(w)])
        .alias(f"months_observed_{w}m"),
    ]
    df = df.with_columns(exprs)

    df = df.with_columns(
        pl.col("_last30_m").forward_fill().over(ID).alias("_last30_m"),
        pl.col("_mod").cum_max().over(ID).alias("ever_modified"),
    )
    return df.with_columns(
        (pl.col(M) - pl.col("_last30_m")).alias("months_since_30p"),
        pl.col("_last30_m").is_not_null().cast(pl.Int8).alias("ever_30p"),
        pl.col("_mod").alias("modified_now"),
        (pl.col("_assist") == 1).alias("assistance_now"),
        (pl.col("borrower_assistance_plan") == "F").alias("forbearance_now"),
        (pl.col("upb") / pl.col("original_upb")).alias("upb_to_orig"),
        # null past the contractual term (schedule balance 0) instead of inf
        pl.when(pl.col("sched_upb") > 1.0).then(pl.col("upb") / pl.col("sched_upb") - 1.0)
        .alias("upb_vs_schedule"),
        (pl.col("current_interest_rate") - pl.col("original_interest_rate")).alias("rate_change"),
    ).drop("_d30", "_d60", "_cure", "_unknown", "_curtail", "_excess", "_upb_up", "_mod",
           "_assist", "_last30_m")


# --------------------------------------------------------------------------------------------
# 3. Labels over (t, t+H]


@dataclass(frozen=True)
class LabelSpec:
    horizon: int
    y1_pop_max_dpd: int
    y1_event_dpd: int
    default_codes: tuple[str, ...]
    y2_pop_dpd: int
    y2_clean_months: int
    y2_event_dpd: int
    y2_strict: tuple[int, int]
    data_end: int  # month index of the last reporting period

    @classmethod
    def from_config(cls, cfg: dict) -> "LabelSpec":
        y1 = cfg["labels"]["early_default"]
        y2 = cfg["labels"]["hidden_deterioration"]
        if y1["horizon_months"] != y2["horizon_months"]:
            raise ValueError("Y1 and Y2 horizons differ; the panel assumes one horizon")
        if y2["clean_history_months"] != 12:
            raise ValueError("add_history computes the clean-history window for 12 months only")
        return cls(
            horizon=y1["horizon_months"],
            y1_pop_max_dpd=y1["population_max_dpd"],
            y1_event_dpd=y1["event_min_dpd"],
            default_codes=tuple(y1["default_zero_balance_codes"]),
            y2_pop_dpd=y2["population_dpd"],
            y2_clean_months=y2["clean_history_months"],
            y2_event_dpd=y2["event_min_dpd"],
            y2_strict=tuple(y2["strict_window"]),
            data_end=ym_to_index(cfg["raw"]["data_end"]),
        )


def _first_k(cond_at_k, ks) -> pl.Expr:
    """Smallest k in ks for which cond_at_k(k) holds, else null."""
    return pl.coalesce([pl.when(cond_at_k(k)).then(pl.lit(k, pl.Int8)) for k in ks])


def add_labels(df: pl.DataFrame, spec: LabelSpec) -> pl.DataFrame:
    """Y1, Y2, strict Y2 and outcome bookkeeping for every month t.

    For each t looks at months t+1..t+H of the same loan:
      y1 = 1 if DPD >= 3 or a default zero-balance code (02/03/09/15) occurs in the window;
      y2 = 1 if DPD >= 2 occurs in the window;
      y2_strict = 1 if the first DPD >= 2 falls in [t+6, t+12], 0 if none in the window,
                  null if it falls in (t, t+5] (neither positive nor a clean negative);
      prepaid_in_window: code 01 in the window (label stays 0 unless an event came first).
    A label is defined when the event is observed or the outcome window is fully observed: the
    loan is reported through t+H or leaves with a zero-balance code inside the window. Windows
    ending after the data end are undefined (dropped later).
    """
    H = spec.horizon
    ks = range(1, H + 1)
    is_default_exit = pl.col(ZBC).is_in(list(spec.default_codes))

    y1_k = _first_k(lambda k: (lag(DPD, -k) >= spec.y1_event_dpd) | lag(is_default_exit, -k), ks)
    y2_k = _first_k(lambda k: lag(DPD, -k) >= spec.y2_event_dpd, ks)
    exit_k = _first_k(lambda k: lag(pl.col(ZBC).is_not_null(), -k), ks)
    prepay_k = _first_k(lambda k: lag(pl.col(ZBC) == PREPAY_CODE, -k), ks)
    observed_to_end = lag(pl.col(ID).is_not_null(), -H).fill_null(False)

    df = df.with_columns(
        y1_k.alias("y1_month"),
        y2_k.alias("y2_month"),
        exit_k.alias("exit_month"),
        prepay_k.is_not_null().alias("prepaid_in_window"),
        observed_to_end.alias("_observed_to_end"),
    )

    exit_code = pl.coalesce([pl.when(pl.col("exit_month") == k).then(lag(ZBC, -k)) for k in ks])
    complete = pl.col("_observed_to_end") | pl.col("exit_month").is_not_null()
    in_data = pl.col(M) + H <= spec.data_end
    lo, hi = spec.y2_strict
    return df.with_columns(
        exit_code.alias("exit_code_in_window"),
        in_data.alias("window_in_data"),
        pl.when(in_data & (complete | pl.col("y1_month").is_not_null()))
        .then(pl.col("y1_month").is_not_null().cast(pl.Int8)).alias("y1"),
        pl.when(in_data & (complete | pl.col("y2_month").is_not_null()))
        .then(pl.col("y2_month").is_not_null().cast(pl.Int8)).alias("y2"),
    ).with_columns(
        pl.when(pl.col("y2") == 0).then(pl.lit(0, pl.Int8))
        .when((pl.col("y2") == 1) & (pl.col("y2_month") >= lo) & (pl.col("y2_month") <= hi))
        .then(pl.lit(1, pl.Int8))
        .alias("y2_strict"),
    ).drop("_observed_to_end")


# --------------------------------------------------------------------------------------------
# 4. Snapshots


def select_snapshots(df: pl.DataFrame, cfg: dict, spec: LabelSpec) -> pl.DataFrame:
    """Quarter-end months; loan_age >= 6; active (no zero-balance code, UPB > 0); DPD known and
    below the Y1 default threshold. Population flags pop_y1 / pop_y2 and labels outside a
    population are nulled. Rows with an undefined label window (beyond the data end or censored
    without a reason) are kept here with null labels and dropped by build_vintage_panel."""
    p = cfg["panel"]
    snap = df.filter(
        (pl.col(M) % 12 + 1).is_in(p["snapshot_months"])
        & ~pl.col("inserted_month")
        & (pl.col("age") >= p["min_loan_age"])
        & pl.col(ZBC).is_null()
        & (pl.col("upb") > 0)
        & pl.col(DPD).is_not_null()
        & (pl.col(DPD) <= spec.y1_pop_max_dpd)
    )
    pop_y2 = (pl.col(DPD) == spec.y2_pop_dpd) & (pl.col("dpd_max_prev12").fill_null(0) < 1)
    return snap.with_columns(
        pl.lit(True).alias("pop_y1"),
        pop_y2.alias("pop_y2"),
    ).with_columns(
        pl.when(pl.col("pop_y2")).then(pl.col("y2")).alias("y2"),
        pl.when(pl.col("pop_y2")).then(pl.col("y2_strict")).alias("y2_strict"),
    )


# --------------------------------------------------------------------------------------------
# 5. Macro features


@dataclass
class Macro:
    hpi_zip3: pl.DataFrame      # zip3, year, quarter, hpi
    hpi_state: pl.DataFrame     # state, year, quarter, hpi
    unemployment: pl.DataFrame  # state, month (Date), unemployment
    pmms: pl.DataFrame          # month (Date), pmms30
    hpi_us: pl.DataFrame | None = None  # year, quarter, hpi (USA); enables *_rel_us features


def load_macro() -> Macro:
    from xai_ews.data import external

    return Macro(hpi_zip3=external.load_hpi_zip3(), hpi_state=external.load_hpi_state(),
                 unemployment=external.load_unemployment(), pmms=external.load_pmms(),
                 hpi_us=external.load_hpi_us())


def _quarter_index(year: pl.Expr, quarter: pl.Expr) -> pl.Expr:
    return year.cast(pl.Int32) * 4 + quarter.cast(pl.Int32) - 1


def _hpi_long(macro: Macro) -> tuple[pl.DataFrame, pl.DataFrame]:
    z = macro.hpi_zip3.select("zip3", _quarter_index(pl.col("year"), pl.col("quarter")).alias("q"),
                              "hpi")
    s = macro.hpi_state.select("state", _quarter_index(pl.col("year"), pl.col("quarter")).alias("q"),
                               "hpi")
    return z, s


def _hpi_at(df: pl.DataFrame, q_col: str, out: str, z: pl.DataFrame, s: pl.DataFrame) -> pl.DataFrame:
    """HPI at quarter index q_col: 3-digit ZIP index, state index where the ZIP has none."""
    df = df.join(z.rename({"zip3": "zip_code_short", "q": q_col, "hpi": "_hz"}),
                 on=["zip_code_short", q_col], how="left")
    df = df.join(s.rename({"state": "property_state", "q": q_col, "hpi": "_hs"}),
                 on=["property_state", q_col], how="left")
    return df.with_columns(
        pl.coalesce("_hz", "_hs").alias(out),
        pl.col("_hz").is_null().alias(f"{out}_state_fallback"),
    ).drop("_hz", "_hs")


def add_macro(df: pl.DataFrame, macro: Macro, cfg: dict) -> pl.DataFrame:
    """Needs the loan's static fields (zip3, state, original LTV, rate, origination date)."""
    ext = cfg["external"]
    # HPI: quarter of month (t - lag); origination quarter for the denominator.
    z, s = _hpi_long(macro)
    m_avail = pl.col(M) - ext["hpi_lag_months"]
    df = df.with_columns(
        (m_avail // 3).alias("_q_t"),
        ((m_avail // 3) - 4).alias("_q_t4"),
        (month_index("origination_date") // 3).alias("_q_orig"),
    )
    df = _hpi_at(df, "_q_t", "hpi_t", z, s)
    df = _hpi_at(df, "_q_t4", "_hpi_t4", z, s).drop("_hpi_t4_state_fallback")
    df = _hpi_at(df, "_q_orig", "_hpi_orig", z, s).drop("_hpi_orig_state_fallback")
    df = df.with_columns(
        (pl.col("original_loan_to_value_ratio_ltv") * pl.col("upb_to_orig")
         * pl.col("_hpi_orig") / pl.col("hpi_t")).alias("mtm_ltv"),
        (pl.col("hpi_t") / pl.col("_hpi_orig") - 1.0).alias("hpi_chg_since_orig"),
        (pl.col("hpi_t") / pl.col("_hpi_t4") - 1.0).alias("hpi_chg_12m"),
    )
    if macro.hpi_us is not None:
        # Local price change relative to the national one: less calendar drift.
        us = macro.hpi_us.select(_quarter_index(pl.col("year"), pl.col("quarter")).alias("_q"), "hpi")
        for q, name in [("_q_t", "_us_t"), ("_q_t4", "_us_t4"), ("_q_orig", "_us_orig")]:
            df = df.join(us.rename({"_q": q, "hpi": name}), on=q, how="left")
        df = df.with_columns(
            (pl.col("hpi_chg_12m") - (pl.col("_us_t") / pl.col("_us_t4") - 1.0))
            .alias("hpi_chg_12m_rel_us"),
            (pl.col("hpi_chg_since_orig") - (pl.col("_us_t") / pl.col("_us_orig") - 1.0))
            .alias("hpi_chg_since_orig_rel_us"),
        ).drop("_us_t", "_us_t4", "_us_orig")
    df = df.drop("_q_t", "_q_t4", "_q_orig", "_hpi_t4", "_hpi_orig")

    # Unemployment: state, month t - lag; as-of join fills months BLS did not publish.
    u = macro.unemployment.select(
        pl.col("state").alias("property_state"),
        month_index("month").alias("_um"),
        "unemployment",
    ).sort("_um")
    lag_u = ext["unemployment_lag_months"]
    for k, name in [(0, "unemployment"), (6, "_u6"), (12, "_u12")]:
        df = df.with_columns((pl.col(M) - lag_u - k).alias("_um")).sort("_um")
        df = df.join_asof(u.rename({"unemployment": name}), on="_um", by="property_state",
                          strategy="backward", tolerance=3)
    df = df.with_columns(
        (pl.col("unemployment") - pl.col("_u6")).alias("unemployment_chg_6m"),
        (pl.col("unemployment") - pl.col("_u12")).alias("unemployment_chg_12m"),
    )
    # State minus national rate; 'US' rows come from BLS series LNS14000000.
    us = u.filter(pl.col("property_state") == "US").drop("property_state")
    if us.height:
        us_wide = None
        for k, name in [(0, "_us0"), (12, "_us12")]:
            part = us.select((pl.col("_um") + k).alias("_um"), pl.col("unemployment").alias(name))
            us_wide = part if us_wide is None else us_wide.join(part, on="_um", how="full",
                                                                coalesce=True)
        df = df.drop("_um").with_columns((pl.col(M) - lag_u).alias("_um")).sort("_um")
        df = df.join_asof(us_wide.sort("_um"), on="_um", strategy="backward", tolerance=3)
        df = df.with_columns(
            (pl.col("unemployment") - pl.col("_us0")).alias("unemployment_rel_us"),
            (pl.col("unemployment_chg_12m") - (pl.col("_us0") - pl.col("_us12")))
            .alias("unemployment_chg_12m_rel_us"),
        ).drop("_us0", "_us12")
    df = df.drop("_um", "_u6", "_u12")

    # PMMS: market 30y rate at t and at origination.
    p = macro.pmms.select(month_index("month").alias("_pm"), "pmms30")
    df = df.with_columns((pl.col(M) - ext["pmms_lag_months"]).alias("_pm"))
    df = df.join(p, on="_pm", how="left").rename({"pmms30": "pmms30_t"})
    df = df.with_columns(month_index("origination_date").alias("_pm")).join(p, on="_pm", how="left")
    df = df.with_columns(
        (pl.col("current_interest_rate") - pl.col("pmms30_t")).alias("rate_spread_pmms"),
        (pl.col("original_interest_rate") - pl.col("pmms30")).alias("sato"),
    ).drop("_pm", "pmms30")
    return df.sort(ID, M)


# --------------------------------------------------------------------------------------------
# 6. Whole vintage


PANEL_DROP = [
    # raw termination / loss fields: label and LGD only, never features
    "zero_balance_effective_date", "upb_at_the_time_of_removal", "last_paid_installment_date",
    "foreclosure_date", "disposition_date", "foreclosure_costs",
    "property_preservation_and_repair_costs", "asset_recovery_costs",
    "miscellaneous_holding_expenses_and_credits", "associated_taxes_for_holding_property",
    "net_sales_proceeds", "credit_enhancement_proceeds", "repurchase_make_whole_proceeds",
    "other_foreclosure_proceeds", "modification_related_non_interest_bearing_upb",
    "principal_forgiveness_amount", "foreclosure_principal_write_off_amount",
    # workout fields (exclude_initial_model); forbearance/assistance kept as report-only flags
    "servicing_activity_indicator", "alternative_delinquency_resolution",
    "alternative_delinquency_resolution_count", "total_deferral_amount",
    "payment_deferral_modification_event_indicator", "borrower_assistance_plan",
    # helpers
    "inserted_month", "monthly_reporting_period", "current_loan_delinquency_status",
    "current_actual_upb",  # raw, masked as 0 for young loans: use `upb`
    "maturity_date", "window_in_data",
    "loan_age", "modification_flag", ZBC, "principal_paid", "sched_principal",
    # static fields come back from the loans table in one join
    *LOAN_COLUMNS[1:],
]


def build_vintage_panel(perf: pl.DataFrame, loans: pl.DataFrame, macro: Macro | None,
                        cfg: dict) -> tuple[pl.DataFrame, dict]:
    spec = LabelSpec.from_config(cfg)
    monthly = prepare_monthly(perf, loans)
    monthly = add_history(monthly, cfg)
    monthly = add_labels(monthly, spec)
    snaps = select_snapshots(monthly, cfg, spec)
    stats = {
        "monthly_rows": monthly.height,
        "inserted_months": int(monthly["inserted_month"].sum()),
        "snapshots_active": snaps.height,
        "snapshots_beyond_data_end": int((~snaps["window_in_data"]).sum()),
        "snapshots_censored": int((snaps["window_in_data"] & snaps["y1"].is_null()).sum()),
        "upb_masked_at_snapshot": int(snaps["upb_masked"].sum()),
    }
    del monthly
    panel = snaps.filter(pl.col("y1").is_not_null())
    panel = panel.drop([c for c in PANEL_DROP if c in panel.columns])
    panel = panel.join(loans.drop("first_period", "u_sample"), on=ID, how="left")
    if macro is not None:
        panel = add_macro(panel, macro, cfg)
    panel = panel.with_columns(month_from_index(M).alias("obs_date"))
    stats["panel_rows"] = panel.height
    return panel, stats
