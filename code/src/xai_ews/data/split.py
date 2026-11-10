"""Sample assignment for panel rows, from configs/data.yaml (split section).

Columns added (functions of the observation month, the vintage and the loan's holdout draw):

    split_main     train / validation / test by observation date (12-month embargo, checked);
                   loans in the 20 % holdout never enter train or validation, only test
    holdout_loan   loan-disjoint check: u_holdout < loan_holdout_share
    new_cohort     vintage in new_cohorts_in_test (absent from train)
    split_vintage  robustness split by origination year, pre-COVID dates only
    val_hurricane_2017  validation rows in hurricane states at the affected dates:
                   excluded from calibration and tuning, kept for the sensitivity check
    stress         covid / gfc / rates: evaluation only, never training or recalibration

Rows outside every window keep nulls (e.g. dates inside an embargo gap).
"""

import polars as pl

from xai_ews.data.panel import M, ym_to_index


def vintage_index(q: str) -> int:
    """'2014Q3' -> 4 * 2014 + 2."""
    return int(q[:4]) * 4 + int(q[5]) - 1


def _vintage_idx_expr() -> pl.Expr:
    return pl.col("vintage").str.slice(0, 4).cast(pl.Int32) * 4 + \
        pl.col("vintage").str.slice(5, 1).cast(pl.Int32) - 1


def _in_window(w: dict) -> pl.Expr:
    t = pl.col(M).is_between(ym_to_index(w["start"]), ym_to_index(w["end"]))
    v = _vintage_idx_expr().is_between(vintage_index(w["vintages"]["from"]),
                                       vintage_index(w["vintages"]["to"]))
    return t & v


def check_embargo(cfg: dict) -> None:
    s = cfg["split"]
    horizon = cfg["labels"]["early_default"]["horizon_months"]
    for a, b in [("train", "validation"), ("validation", "test")]:
        gap = ym_to_index(s[b]["start"]) - ym_to_index(s[a]["end"])
        if gap < horizon:
            raise ValueError(f"embargo {a}->{b} is {gap} months, below the {horizon}-month horizon")


def assign_splits(df: pl.DataFrame, cfg: dict) -> pl.DataFrame:
    check_embargo(cfg)
    s = cfg["split"]
    holdout = pl.col("u_holdout") < s["loan_holdout_share"]
    main = (
        pl.when(_in_window(s["train"]) & ~holdout).then(pl.lit("train"))
        .when(_in_window(s["validation"]) & ~holdout).then(pl.lit("validation"))
        .when(_in_window(s["test"])).then(pl.lit("test"))
    )
    vr = s["vintage_robustness"]
    year = pl.col("vintage").str.slice(0, 4).cast(pl.Int32)
    pre_covid = pl.col(M) <= ym_to_index(vr["max_obs_date"])
    by_vintage = (
        pl.when(pre_covid & year.is_in(vr["train_vintages"])).then(pl.lit("train"))
        .when(pre_covid & year.is_in(vr["validation_vintages"])).then(pl.lit("validation"))
        .when(pre_covid & year.is_in(vr["test_vintages"])).then(pl.lit("test"))
    )
    hur = s["validation_exclusions"]["hurricanes_2017"]
    hurricane = (main == "validation").fill_null(False) \
        & pl.col("property_state").is_in(hur["states"]) \
        & pl.col(M).is_in([ym_to_index(d) for d in hur["dates"]])
    stress = pl.lit(None, pl.String)
    for name, w in reversed(list(s["stress"].items())):
        stress = pl.when(_in_window(w)).then(pl.lit(name)).otherwise(stress)
    nc = s["new_cohorts_in_test"]
    return df.with_columns(
        main.alias("split_main"),
        holdout.alias("holdout_loan"),
        _vintage_idx_expr().is_between(vintage_index(nc["from"]), vintage_index(nc["to"]))
        .alias("new_cohort"),
        by_vintage.alias("split_vintage"),
        hurricane.alias("val_hurricane_2017"),
        stress.alias("stress"),
    )


def add_negative_draw(df: pl.DataFrame, cfg: dict) -> pl.DataFrame:
    """u_neg ~ U[0, 1) per (loan, month), deterministic: the same r always keeps the same rows,
    and the r = 0.1 sample is a subset of the r = 0.2 sample."""
    from xai_ews.data.ingest import uniform_draw

    keys = (df["loan_identifier"] + ":" + df[M].cast(pl.String)).to_list()
    return df.with_columns(
        pl.Series("u_neg", uniform_draw(keys, cfg["panel"]["sample_seed"], "negatives")))


def train_sample(lf: pl.LazyFrame, label: str, r: float, columns: list[str]) -> pl.LazyFrame:
    """Full train population: all positives + negatives with u_neg < r, weight 1/r on negatives.

    `lf` is scan_train_full(); the population is the label's (pop_y1) with a defined
    label. The model's loss weight is omega x sample_weight; recalibrate on unweighted validation.
    """
    pop = {"y1": "pop_y1"}[label]
    return (
        lf.filter(pl.col(pop) & pl.col(label).is_not_null()
                  & ((pl.col(label) == 1) | (pl.col("u_neg") < r)))
        .with_columns(pl.when(pl.col(label) == 1).then(1.0).otherwise(1.0 / r)
                      .alias("sample_weight"))
        .select(*columns, label, "sample_weight")
    )
