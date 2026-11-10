"""Data for the stage-3 models, memory-lean (8 GB machine).

    train       full train population: all positives + negatives with u_neg < r,
                sample_weight = 1/r on negatives; omega = exposure weight; weight = omega x 1/r.
                `weighted=False` (H4 comparison model) keeps only 1/r.
    evaluation  validation / test / stress rows of the 15 % hash sample, unweighted, with the
                natural-disaster flags joined by (ZIP3, month) and the exclusion of the
                chosen variant applied; validation rows carry their half A / B.
    omega       ln(1 + UPB) / mean on the full train (primary), or UPB capped at the train p99 /
                mean (comparison); the norm is data/processed/panel/exposure_norm.json.

Numeric features are Float32, categoricals String with nulls -> "NA" (as in stage 2), then integer
codes from a CategoryEncoder fitted on train.
"""

import json

import numpy as np
import polars as pl

from xai_ews.data.disasters import exclusion_expr
from xai_ews.data.ingest import uniform_draw
from xai_ews.data.panel import scan_panel, scan_train_full
from xai_ews.data.split import train_sample
from xai_ews.features.data import NA, CategoryEncoder, to_model_frame  # noqa: F401
from xai_ews.paths import DISASTER_FLAGS, PANEL_DIR, VALIDATION_HALVES

ID, M = "loan_identifier", "m"
KEYS = [ID, M]
POP = {"y1": "pop_y1"}
EVENT_MONTH = {"y1": "y1_month"}
HALF_SEED = 20261025


def exposure_norm() -> dict:
    return json.loads((PANEL_DIR / "exposure_norm.json").read_text())


def omega_expr(kind: str = "log") -> pl.Expr:
    """Exposure weight from current UPB: 'log' (primary), 'linear' (capped at the train p99)."""
    n = exposure_norm()
    if kind == "log":
        return (pl.col("upb").log1p() / n["mean_log1p_upb"]).cast(pl.Float32)
    if kind == "linear":
        return (pl.col("upb").clip(None, n["upb_p99"]) / n["mean_upb_capped_p99"]).cast(pl.Float32)
    raise ValueError(kind)


def _cast(features: list[str], categorical: list[str]) -> list[pl.Expr]:
    cats = set(categorical)
    return [pl.col(f).cast(pl.String).fill_null(NA) if f in cats else pl.col(f).cast(pl.Float32)
            for f in features]


def load_train(label: str, features: list[str], categorical: list[str], r: float = 0.2,
               omega: str | None = "log", extra: list[str] | None = None,
               drop_prepaid: bool = False) -> pl.DataFrame:
    """keys, features, label, sample_weight (1/r), omega, weight (= omega x 1/r; 1/r if omega=None)."""
    extra = [c for c in (extra or []) if c not in features]
    cols = [*KEYS, *features, *extra, "upb", "prepaid_in_window"]
    lf = train_sample(scan_train_full(), label, r, list(dict.fromkeys(cols)))
    if drop_prepaid:
        lf = lf.filter(~pl.col("prepaid_in_window").fill_null(False))
    w = omega_expr(omega) if omega else pl.lit(1.0, pl.Float32)
    return lf.with_columns(*_cast(features, categorical), pl.col(label).cast(pl.Int8),
                           pl.col("sample_weight").cast(pl.Float32), w.alias("omega")) \
        .with_columns((pl.col("omega") * pl.col("sample_weight")).alias("weight")).collect()


def split_filter(split: str) -> pl.Expr:
    """'validation' / 'test' (split_main), a stress name, or 'vintage_<train|validation>'."""
    if split in ("train", "validation", "test"):
        return pl.col("split_main") == split
    if split.startswith("vintage_"):
        return pl.col("split_vintage") == split.removeprefix("vintage_")
    return pl.col("stress") == split


def with_disaster_flags(lf: pl.LazyFrame) -> pl.LazyFrame:
    flags = pl.scan_parquet(DISASTER_FLAGS)
    return lf.join(flags, left_on=["zip_code_short", M], right_on=["zip3", M], how="left")


def load_eval(split: str, label: str, features: list[str], categorical: list[str],
              exclusion: str = "main", extra: list[str] | None = None,
              omega: str = "log", where: pl.Expr | None = None) -> pl.DataFrame:
    """Evaluation rows of a split for the label's population: keys, features, label, upb, omega,
    exclusion flags of every variant, holdout_loan, new_cohort, prepaid_in_window, half (validation).
    Rows hit by the disaster rule of `exclusion` are dropped ('all' keeps everything); `where` is an
    extra row filter on panel columns (e.g. one observation month, for samples that do not fit)."""
    extra = [c for c in (extra or []) if c not in features]
    base = [*KEYS, *features, *extra, label, EVENT_MONTH[label], "upb", "mtm_ltv", "property_state",
            "zip_code_short", "stress", "holdout_loan", "new_cohort", "prepaid_in_window",
            "forbearance_now", "vintage"]
    lf = scan_panel().filter(split_filter(split) & pl.col(POP[label]) & pl.col(label).is_not_null())
    if where is not None:
        lf = lf.filter(where)
    lf = with_disaster_flags(lf.select(list(dict.fromkeys(base))))
    lf = lf.with_columns(*[exclusion_expr(v).alias(f"excl_{v}") for v in ("main", "hurricane", "any_overlap")])
    lf = lf.filter(~exclusion_expr(exclusion))
    if split == "validation" and VALIDATION_HALVES.exists():
        lf = lf.join(pl.scan_parquet(VALIDATION_HALVES), on=ID, how="left")
    return lf.with_columns(*_cast(features, categorical), pl.col(label).cast(pl.Int8),
                           omega_expr(omega).alias("omega")).collect()


def make_validation_halves(seed: int = HALF_SEED) -> pl.DataFrame:
    """Validation loans of the Y1 population (rows kept by the main disaster rule) split into
    halves A / B, stratified by whether the loan has a Y1 event; within each stratum loans are
    ordered by a hash draw and assigned alternately."""
    lf = with_disaster_flags(scan_panel().filter(pl.col("split_main") == "validation")
                             .select(ID, M, "zip_code_short", "stress", "y1", "pop_y1"))
    loans = (lf.filter(~exclusion_expr("main") & pl.col("pop_y1"))
             .group_by(ID).agg(pl.col("y1").fill_null(0).max().alias("ev1"))
             .collect().sort(ID))
    u = uniform_draw(loans[ID].to_list(), seed, "validation_half")
    loans = loans.with_columns(pl.Series("u", u)).sort("ev1", "u").with_columns(
        pl.int_range(pl.len()).over("ev1").alias("k"))
    return loans.select(ID, pl.when(pl.col("k") % 2 == 0).then(pl.lit("A")).otherwise(pl.lit("B"))
                        .alias("half"), "ev1")


def fit_encoder(train: pl.DataFrame, categorical: list[str], top_n: dict | None = None) -> CategoryEncoder:
    return CategoryEncoder(top_n if top_n is not None else {"seller_name": 20}).fit(train, categorical)


def arrays(df: pl.DataFrame, features: list[str], categorical: list[str], enc: CategoryEncoder):
    """(numeric float32 matrix, categorical int64 codes) for the network."""
    num = [f for f in features if f not in categorical]
    x_num = df.select(num).to_numpy().astype(np.float32)
    x_cat = enc.transform(df.select(categorical)).to_numpy().astype(np.int64)
    return x_num, x_cat


def exclusion_flags(split: str = "validation") -> pl.DataFrame:
    """keys + excl_<variant> for every row of a split (all populations), e.g. to re-filter
    predictions cached in stage 2 by the new disaster rule."""
    lf = with_disaster_flags(scan_panel().filter(split_filter(split))
                             .select(*KEYS, "zip_code_short", "stress", "val_hurricane_2017"))
    return lf.select(*KEYS, "val_hurricane_2017",
                     *[exclusion_expr(v).alias(f"excl_{v}") for v in ("main", "hurricane", "any_overlap")]).collect()


def load_snapshots(first: str, last: str, label: str, features: list[str], categorical: list[str],
                   holdout: bool | None = None, extra: list[str] | None = None) -> pl.DataFrame:
    """Every quarterly snapshot of the 15 % sample with t in [first, last] and the label's
    population, rows hit by the main disaster rule removed; labels are NOT required (dates in
    embargo gaps are scored for the lead time only). holdout=False drops the 20 % holdout loans
    (as in validation), None keeps all loans (as in test)."""
    from xai_ews.data.panel import ym_to_index

    extra = [c for c in (extra or []) if c not in features]
    lf = scan_panel().filter(pl.col(M).is_between(ym_to_index(first), ym_to_index(last)) & pl.col(POP[label]))
    if holdout is False:
        lf = lf.filter(~pl.col("holdout_loan"))
    lf = with_disaster_flags(lf.select(list(dict.fromkeys([*KEYS, *features, *extra, "upb", "zip_code_short",
                                                             "stress", "holdout_loan"]))))
    return lf.filter(~exclusion_expr("main")).with_columns(*_cast(features, categorical)).collect()


def load_vintage_train(label: str, features: list[str], categorical: list[str], r: float = 0.2,
                       extra: list[str] | None = None) -> pl.DataFrame:
    """Train of the vintage split: split_vintage = train rows of the 15 % sample,
    all positives + negatives with the stage-1 hash draw u_neg < r; weight = omega x 1/r. The
    disaster rule is not applied to training rows."""
    df = load_eval("vintage_train", label, features, categorical, exclusion="all", extra=extra)
    keys = (df[ID] + ":" + df[M].cast(pl.String)).to_list()
    from xai_ews.config import data_config
    u = uniform_draw(keys, data_config()["panel"]["sample_seed"], "negatives")
    df = df.with_columns(pl.Series("u_neg", u)).filter((pl.col(label) == 1) | (pl.col("u_neg") < r))
    return df.with_columns(pl.when(pl.col(label) == 1).then(1.0).otherwise(1.0 / r).cast(pl.Float32)
                           .alias("sample_weight")).with_columns(
        (pl.col("omega") * pl.col("sample_weight")).alias("weight"))


def split_months(split: str) -> list[int]:
    """Observation months (m) present in a split, sorted."""
    return sorted(scan_panel().filter(split_filter(split)).select(pl.col(M).unique()).collect()[M].to_list())
