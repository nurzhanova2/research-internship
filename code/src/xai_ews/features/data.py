"""Train / validation frames for stage 2, memory-lean (8 GB machine).

Train = full train population: all positives + negatives with u_neg < r, weight
1/r. Validation = 15 % hash sample, unweighted, without the `val_hurricane_2017` rows by default.
The test split is never read here.

Numeric features come as Float32, categoricals as String with nulls replaced by "NA".
"""

import pandas as pd
import polars as pl

from xai_ews.config import data_config, feature_config
from xai_ews.data.features import candidate_features, categorical_features
from xai_ews.data.panel import scan_panel, scan_train_full
from xai_ews.data.split import train_sample

# Stage 2 selects features for early default only (task -> population flag)
TASKS = {"y1": "pop_y1"}
ID = "loan_identifier"
KEYS = [ID, "m"]
NA = "NA"


def population(task: str) -> dict:
    """The population rule of a task; part of the model cache keys, so a changed population
    never reuses predictions made on the old one."""
    y1 = data_config()["labels"]["early_default"]
    return {"flag": TASKS[task], "max_dpd": y1["population_max_dpd"], "after_default": y1["exclude_after_default"]}


def _cast(features: list[str]) -> list[pl.Expr]:
    cats = set(categorical_features())
    return [pl.col(f).cast(pl.String).fill_null(NA) if f in cats else pl.col(f).cast(pl.Float32)
            for f in features]


def load_train(label: str, features: list[str] | None = None, r: float | None = None) -> pl.DataFrame:
    """loan_identifier, m, features, label, sample_weight (1 for positives, 1/r for negatives)."""
    features = candidate_features() if features is None else features
    r = feature_config()["data"]["negatives_rate"] if r is None else r
    lf = train_sample(scan_train_full(), label, r, [*KEYS, *features])
    return lf.with_columns(*_cast(features), pl.col(label).cast(pl.Int8),
                           pl.col("sample_weight").cast(pl.Float32)).collect()


def load_validation(label: str, features: list[str] | None = None,
                    hurricanes: bool | None = None) -> pl.DataFrame:
    """Validation population of the label with weight 1; hurricanes=True keeps the
    `val_hurricane_2017` rows (sensitivity) and adds the flag column."""
    features = candidate_features() if features is None else features
    if hurricanes is None:
        hurricanes = not feature_config()["data"]["validation_exclude_hurricanes"]
    lf = scan_panel().filter((pl.col("split_main") == "validation") & pl.col(TASKS[label])
                             & pl.col(label).is_not_null())
    extra = ["val_hurricane_2017"] if hurricanes else []
    if not hurricanes:
        lf = lf.filter(~pl.col("val_hurricane_2017"))
    return lf.select(*KEYS, *features, label, *extra).with_columns(
        *_cast(features), pl.col(label).cast(pl.Int8),
        pl.lit(1.0, pl.Float32).alias("sample_weight")).collect()


def event_counts(df: pl.DataFrame, label: str) -> dict:
    """Events per observation and per unique loan; weighted rows = population size."""
    pos = df.filter(pl.col(label) == 1)
    return {"rows": df.height, "loans": df[ID].n_unique(), "weighted_rows": float(df["sample_weight"].sum()),
            "events_obs": pos.height, "event_loans": pos[ID].n_unique(),
            "event_rate_weighted": float((df[label] * df["sample_weight"]).sum() / df["sample_weight"].sum())}


def subsample(df: pl.DataFrame, label: str, n: int, seed: int) -> pl.DataFrame:
    """All positives (if they fit) plus random negatives up to n rows; negative weights are scaled
    so the weighted population size is unchanged."""
    if df.height <= n:
        return df
    pos = df.filter(pl.col(label) == 1)
    neg = df.filter(pl.col(label) == 0)
    k = max(n - pos.height, 0)
    neg_s = neg.sample(n=min(k, neg.height), seed=seed)
    scale = neg["sample_weight"].sum() / max(neg_s["sample_weight"].sum(), 1e-9)
    return pl.concat([pos, neg_s.with_columns(pl.col("sample_weight") * scale)])


class CategoryEncoder:
    """Integer codes per categorical column, fitted on train.

    Codes follow train frequency (0 = most frequent). For columns in `top_n` only the top-N
    categories keep their own code, the rest -> "other". Unseen categories map to
    "other" (or to the last code if a column has no "other").
    """

    OTHER = "__other__"

    def __init__(self, top_n: dict[str, int] | None = None):
        self.top_n = top_n or {}
        self.maps: dict[str, dict[str, int]] = {}

    def fit(self, df: pl.DataFrame, columns: list[str]) -> "CategoryEncoder":
        for c in columns:
            vc = df.group_by(c).agg(pl.col("sample_weight").sum().alias("w")).sort("w", descending=True)
            cats = vc[c].to_list()
            if c in self.top_n:
                cats = cats[: self.top_n[c]]
            self.maps[c] = {k: i for i, k in enumerate([*cats, self.OTHER])}
        return self

    def group(self, df: pl.DataFrame) -> pl.DataFrame:
        """Same frame with rare categories of the top-N columns replaced by "__other__"."""
        return df.with_columns(
            pl.col(c).replace_strict({k: k for k in m}, default=self.OTHER) for c, m in self.maps.items()
            if c in self.top_n and c in df.columns)

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(
            pl.col(c).replace_strict(m, default=m[self.OTHER], return_dtype=pl.Int32)
            for c, m in self.maps.items() if c in df.columns)


def to_model_frame(df: pl.DataFrame, features: list[str], encoder: CategoryEncoder) -> pd.DataFrame:
    """pandas frame for CatBoost: int32 codes for categoricals, float32 numerics."""
    return encoder.transform(df.select(features)).to_pandas()
