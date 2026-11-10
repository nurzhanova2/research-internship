"""Quick leakage check: plain CatBoost train -> validation on Y1 with all candidate
features. Train = full train population: all positives + negatives at the
configured rate r with weight 1/r; validation = 15 % hash sample, unweighted. Not stage-3
modelling (no tuning, no exposure weight). AUC >= audit.leakage_check.auc_stop means stop and
look for leakage. Also reports the single-feature AUC of every feature: a single feature close
to the model's AUC points at the leak.

    uv run python scripts/leakage_check.py
"""

import json
import sys

import numpy as np
import polars as pl
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score

from xai_ews.config import data_config
from xai_ews.data.features import candidate_features, categorical_features
from xai_ews.data.panel import scan_panel, scan_train_full
from xai_ews.data.split import train_sample
from xai_ews.paths import DATA_AUDIT_DIR

TASKS = {"y1": "pop_y1"}


def load_validation(label: str, pop: str, feats: list[str]) -> pl.DataFrame:
    lf = scan_panel().filter((pl.col("split_main") == "validation") & pl.col(pop)
                             & pl.col(label).is_not_null())
    return lf.select(*feats, label).collect()


def to_pool(df: pl.DataFrame, feats: list[str], cats: list[str], label: str) -> Pool:
    x = df.select(
        *[pl.col(c).cast(pl.String).fill_null("NA") for c in cats],
        *[pl.col(c).cast(pl.Float32) for c in feats if c not in cats],
    ).to_pandas()
    w = df["sample_weight"].to_numpy() if "sample_weight" in df.columns else None
    return Pool(x, df[label].to_numpy(), cat_features=cats, weight=w)


def single_feature_auc(x: pl.Series, y: np.ndarray) -> float:
    v = x.cast(pl.Float64, strict=False).to_numpy()
    if np.all(np.isnan(v)):
        return float("nan")
    v = np.where(np.isnan(v), np.nanmedian(v), v)
    auc = roc_auc_score(y, v)
    return max(auc, 1 - auc)


def main() -> int:
    full_cfg = data_config()
    cfg = full_cfg["audit"]["leakage_check"]
    r = full_cfg["panel"]["y2_negative_subsampling"]["negatives_rate"]
    feats = candidate_features()
    cats = [c for c in categorical_features() if c in feats]
    results = {}
    for label, pop in TASKS.items():
        tr = train_sample(scan_train_full(), label, r, feats).collect()
        va = load_validation(label, pop, feats)
        model = CatBoostClassifier(iterations=cfg["iterations"], depth=cfg["depth"],
                                   learning_rate=0.1, eval_metric="AUC", verbose=0,
                                   random_seed=0, thread_count=-1, allow_writing_files=False)
        model.fit(to_pool(tr, feats, cats, label))
        p = model.predict_proba(to_pool(va, feats, cats, label))[:, 1]
        y = va[label].to_numpy()
        auc = roc_auc_score(y, p)
        imp = sorted(zip(model.feature_names_, model.get_feature_importance()), key=lambda t: -t[1])
        single = {f: single_feature_auc(va[f], y) for f in feats if f not in cats}
        top_single = sorted(single.items(), key=lambda t: -np.nan_to_num(t[1]))[:10]
        results[label] = {
            "train_rows": tr.height, "train_events": int(tr[label].sum()), "negatives_rate": r,
            "validation_rows": va.height, "validation_events": int(y.sum()),
            "validation_auc": round(auc, 4),
            "stop": auc >= cfg["auc_stop"],
            "top_importance": [(f, round(v, 2)) for f, v in imp[:15]],
            "top_single_feature_auc": [(f, round(v, 4)) for f, v in top_single],
        }
        print(label, json.dumps(results[label], indent=1), flush=True)
    (DATA_AUDIT_DIR / "leakage_check.json").write_text(json.dumps(results, indent=2))
    return 2 if any(r["stop"] for r in results.values()) else 0


if __name__ == "__main__":
    sys.exit(main())
