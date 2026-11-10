"""Bundle for training the stage-3 CatBoost models on Colab (GPU):

    uv run python scripts/make_colab_bundle.py      # -> data/colab_bundle.zip

The bundle keeps the repository layout, so xai_ews resolves its paths unchanged:
src/xai_ews, configs, scripts/{tune,fit_models}.py, notebooks/colab_stage3_catboost.ipynb, and
data/processed with the full train (only the columns the models need) and the validation rows of
the panel, the exposure norm, the disaster flags, the default history (Y1 population) and the
validation halves. Test and stress rows
are NOT in the bundle: test stays closed until the protocol is frozen.
"""

import shutil
import sys
import zipfile

import polars as pl

from xai_ews.config import feature_config
from xai_ews.paths import (
    DATA_DIR,
    DEFAULT_HISTORY,
    DISASTER_FLAGS,
    NOTEBOOKS_DIR,
    PANEL_DIR,
    PROJECT_ROOT,
    TRAIN_FULL_DIR,
    VALIDATION_HALVES,
)

OUT = DATA_DIR / "colab_bundle"


def main() -> int:
    sel = feature_config()["selected"]
    feats = list(dict.fromkeys([*sel["features"], *sel["h3_full"]]))
    keys = ["loan_identifier", "m"]
    labels = ["y1", "y1_month", "pop_y1", "dpd"]   # dpd: the Y1 population is narrowed at read time
    if OUT.exists():
        shutil.rmtree(OUT)
    for d in ("src", "configs"):
        shutil.copytree(PROJECT_ROOT / d, OUT / d, ignore=shutil.ignore_patterns("__pycache__"))
    (OUT / "scripts").mkdir(parents=True)
    for f in ("tune.py", "fit_models.py"):
        shutil.copy(PROJECT_ROOT / "scripts" / f, OUT / "scripts" / f)
    (OUT / "notebooks").mkdir()
    shutil.copy(NOTEBOOKS_DIR / "colab_stage3_catboost.ipynb", OUT / "notebooks")
    processed = OUT / "data" / "processed"
    tr_cols = [*keys, *feats, *labels, "upb", "prepaid_in_window", "u_neg"]
    for part in sorted(TRAIN_FULL_DIR.glob("vintage=*")):
        df = pl.read_parquet(part / "*.parquet", columns=tr_cols)
        (processed / "train_full" / part.name).mkdir(parents=True)
        df.write_parquet(processed / "train_full" / part.name / "part.parquet", compression="zstd")
    va_cols = [*keys, *feats, *labels, "upb", "mtm_ltv", "property_state", "zip_code_short", "stress",
               "split_main", "holdout_loan", "new_cohort", "prepaid_in_window", "forbearance_now",
               "val_hurricane_2017"]
    va_cols = list(dict.fromkeys(va_cols))
    for part in sorted(PANEL_DIR.glob("vintage=*")):
        df = pl.read_parquet(part / "*.parquet", columns=va_cols).filter(pl.col("split_main") == "validation")
        if df.height:
            (processed / "panel" / part.name).mkdir(parents=True)
            df.write_parquet(processed / "panel" / part.name / "part.parquet", compression="zstd")
    shutil.copy(PANEL_DIR / "exposure_norm.json", processed / "panel")
    shutil.copy(DISASTER_FLAGS, processed)
    shutil.copy(DEFAULT_HISTORY, processed)
    shutil.copy(VALIDATION_HALVES, processed)
    zpath = DATA_DIR / "colab_bundle.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_STORED) as z:
        for f in sorted(OUT.rglob("*")):
            if f.is_file():
                z.write(f, f.relative_to(OUT))
    print(f"{zpath}: {zpath.stat().st_size / 1e6:,.0f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
